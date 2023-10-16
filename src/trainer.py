import os
import sys
from distutils.dir_util import copy_tree

import torch
import wandb
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import get_loader
from evaluation import eval_nlp

WANDB = True


class Trainer:
    def __init__(
        self, model, train_dtset, eval_dtset, outdir, args, is_distributed=False
    ):
        super().__init__()
        self.is_distributed = is_distributed
        self.train_dtset = train_dtset
        self.eval_dtset = eval_dtset
        self.outdir = outdir
        self.args = args
        self.max_steps = self.args.max_steps
        self.eval_steps = self.args.eval_steps
        self.warmup_steps = self.args.warmup_steps
        self.bs = self.args.bs
        self.lr = self.args.lr
        self.min_lr = self.args.min_lr
        self.decay = self.args.decay
        self.fp16 = self.args.fp16
        self.num_workers = self.args.num_workers

        if self.is_distributed:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        self.model = model.to(self.gpu_id)

        self.opt = self.get_optimizer()
        self.train_loader = get_loader(
            self.train_dtset,
            self.num_workers,
            tokenizer=self.model.tokenizer,
            nr_learnable_tokens=self.model.nr_learnable_tokens,
            is_distributed=self.is_distributed,
            bs=self.bs,
            is_train=True,
        )
        self.eval_loader = get_loader(
            self.eval_dtset,
            self.num_workers,
            tokenizer=self.model.tokenizer,
            nr_learnable_tokens=self.model.nr_learnable_tokens,
            is_distributed=self.is_distributed,
            bs=self.bs,
            is_train=False,
        )
        self.scheduler = self.get_scheduler(
            self.opt, self.warmup_steps, self.max_steps, self.min_lr
        )

        if self.fp16:
            self.scaler = amp.GradScaler(enabled=True)

        if WANDB and self.gpu_id == 0:
            wandb.init(
                config=self.args,
                project="devil",
                name=self.outdir.split("/")[-1],
                dir=self.outdir,
            )
            self.best_score = 0.0
            self.best_loss = sys.maxsize

        self.step = 0
        if args.resume:
            assert (
                args.model_ckpt is not None
            ), "Please provide a pretrained model checkpoint to resume training."
            loc = f"cuda:{self.gpu_id}"
            model_snapshot = torch.load(
                os.path.join(args.model_ckpt, "model.pt"), map_location=loc
            )
            self.model.translation.load_state_dict(model_snapshot["MODEL_STATE"])
            self.opt.load_state_dict(model_snapshot["OPTIMIZER_STATE"])
            self.scheduler.load_state_dict(model_snapshot["SCHEDULER_STATE"])
            self.step = model_snapshot["STEPS_RUN"]
            self.best_score = model_snapshot["SCORE"]
            self.best_loss = model_snapshot["LOSS"]

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.forbidden_keys = ["id", "raw"]

    def get_parameter_names(self, model, forbidden_layer_types):
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def get_optimizer(self):
        decay_parameters = self.get_parameter_names(
            self.model.translation, [torch.nn.LayerNorm]
        )
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.translation.named_parameters()
                    if n in decay_parameters
                ],
                "weight_decay": self.decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.translation.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.lr)

    def get_scheduler(self, optimizer, num_warmup_steps, num_training_steps, min_lr):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # min_lr / self.lr (aka initial lr) because lambda is multiplied by initial lr (can be thought of as a %)
            return max(
                min_lr / self.lr,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(optimizer, lr_lambda, -1)

    def train(self):
        self.model.train()

        if self.gpu_id == 0:
            pbar = tqdm(self.train_loader, file=sys.stdout)
            pbar.set_description("training")
            data_iter = iter(pbar)
        else:
            data_iter = iter(self.train_loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                if self.gpu_id == 0:
                    pbar = tqdm(self.train_loader, file=sys.stdout)
                    pbar.set_description("training")
                    data_iter = iter(pbar)
                else:
                    data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = {
                k: v.to(self.gpu_id)
                for k, v in batch.items()
                if k not in self.forbidden_keys
            }

            with amp.autocast(enabled=self.fp16):
                loss = self.model(**batch)

            self.opt.zero_grad()
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                self.opt.step()

            if self.scheduler:
                self.scheduler.step()

            self.step += 1

            if self.gpu_id == 0:
                if WANDB:
                    logdict = {"train/loss": loss.item(), "train_step": self.step}
                    if self.scheduler:
                        logdict.update(
                            {
                                "lr": self.scheduler.get_last_lr()[0],
                                "lr_step": self.step,
                            }
                        )

                if self.step % self.eval_steps == 0:
                    val_loss, metrics, plot_data = eval_nlp(
                        self.model,
                        self.eval_loader,
                        self.outdir,
                        gpu_id=self.gpu_id,
                        is_distributed=self.is_distributed,
                        outname=f"step_{self.step:08d}",
                        milan=self.args.milan,
                    )
                    assert self.model.training is True

                    if WANDB:
                        logdict.update(
                            {
                                "val_step": self.step,
                                "val/loss": val_loss,
                            }
                        )
                        for k, v in metrics.items():
                            logdict["val/" + k] = v

                        for image_title, grid, cpt, _ in plot_data:
                            logdict.update(
                                {image_title: wandb.Image(grid, caption=cpt)}
                            )

                    # save model with best bertscore
                    bertscore = metrics["BERTScore"]
                    is_best_score = bertscore > self.best_score
                    self.best_score = max(self.best_score, bertscore)

                    # save model with best val loss
                    is_best_loss = val_loss < self.best_loss
                    self.best_loss = min(val_loss, self.best_loss)

                    # save latest checkpoint
                    self.save_checkpoint(
                        self.step, bertscore, val_loss, is_best_score, is_best_loss
                    )

                if WANDB:
                    wandb.log(logdict)

            if self.step == self.max_steps:
                break

    def save_checkpoint(
        self, curr_step, score, loss, is_best_score=False, is_best_loss=False
    ):
        if self.is_distributed:
            model = self.model.module
        else:
            model = self.model

        snapshot = {
            "MODEL_STATE": model.translation.state_dict(),
            "OPTIMIZER_STATE": self.opt.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "STEPS_RUN": curr_step,
            "SCORE": score,
            "LOSS": loss,
        }

        save_path = os.path.join(self.outdir, "latest_checkpoint")
        os.makedirs(save_path, exist_ok=True)

        torch.save(snapshot, os.path.join(save_path, "model.pt"))

        model.lm_model.config.to_json_file(
            os.path.join(save_path, "lm_model_config.json")
        )
        model.translation.config.to_json_file(
            os.path.join(save_path, "translation_model_config.json")
        )

        if hasattr(self, "tokenizer"):
            self.model.tokenizer.save_pretrained(save_path)

        if is_best_score:
            copy_tree(save_path, os.path.join(self.outdir, "best_checkpoint_score"))
        if is_best_loss:
            copy_tree(save_path, os.path.join(self.outdir, "best_checkpoint_loss"))
