import argparse
import datetime
import os
import random
from argparse import Namespace  # needed for reading saved argparse parameters

import torch
import yaml
from torch.distributed import destroy_process_group, init_process_group
from transformers import AutoConfig

from dataset import (
    CustomDataCollator,
    MILANDataset,
    get_loader,
    get_milan_transform,
    get_wds_dataset,
)
from evaluation import eval_nlp, eval_qualitative
from milan_keys import KEYS
from model import Features2WordsModel, TranslationTransformerConfig
from trainer import Trainer

random.seed(0)
torch.manual_seed(0)

KEYS = list(KEYS.keys())

DATASETS = ["cc3m"]
DATASETS.extend(KEYS)


def ddp_setup():
    init_process_group(backend="nccl")


def _create_folder(args):
    folder = args.logdir

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vision_backbone = args.vision_backbone.replace("/", "_")
    language_model = args.language_model.split("/")[-1]
    vision_feat_layers = (
        str(args.vision_feat_layers).replace("[", "").replace("]", "").replace(", ", "")
    )

    fname = ""
    fname += f"{vision_backbone}_{language_model}_layers_{vision_feat_layers}"
    if args.token_dropout > 0:
        fname += f"_dropout_{args.token_dropout}"
    if args.feature_dropout > 0:
        fname += f"_featdropout_{args.feature_dropout}"
    fname += f"_tokens_{args.nr_learnable_tokens}_{args.dataset}_{timestamp}"
    results_path = os.path.join(folder, fname)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to run the script.")

    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Use 16-bit floating-point precision.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Do not use 16-bit floating-point precision.",
    )
    parser.set_defaults(fp16=False)

    # Directories and paths
    parser.add_argument(
        "--logdir",
        type=str,
        default="./results",
        help="Directory where logs and models are to be stored.",
    )

    # Actions
    parser.add_argument(
        "--do_eval_nlp",
        action="store_true",
        help="Evaluate model given by model_ckpt on nlp metrics.",
    )
    parser.add_argument(
        "--do_eval_qualitative",
        action="store_true",
        help="Evaluate model given by model_ckpt qualitatively.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cc3m",
        choices=DATASETS,
        help="Dataset.",
    )
    parser.add_argument(
        "--by_unit",
        action="store_true",
        help="Use MILAN datasets by neuron unit (instead of by image).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory of the dataset.",
    )

    # Model
    parser.add_argument(
        "--nr_layers",
        type=int,
        default=12,
        help="Number of hidden layers in transformer translation model.",
    )
    parser.add_argument(
        "--nr_heads",
        type=int,
        default=12,
        help="Number of heads per attention layer.",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="Translation transformer intermediate size.",
    )
    parser.add_argument(
        "--nr_learnable_tokens",
        type=int,
        default=10,
        help="Number of learnable tokens in transformer translation model.",
    )
    parser.add_argument(
        "--language_model",
        type=str,
        default="facebook/opt-125m",
        choices=["gpt2", "facebook/opt-125m"],
        help="LM used to produce words.",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="timm_resnet50",
        help="Vision backbone from which image features are produced.",
    )
    parser.add_argument(
        "--vision_feat_func",
        type=str,
        default="avg_pooling",
        choices=["none", "avg_pooling"],
        help="Function applied to vision features.",
    )
    parser.add_argument(
        "--vision_feat_layers",
        nargs="+",
        type=int,
        default=[-1],
        help=(
            "List of feature layers (indexed by order) of the vision model "
            "to pass to the transformer translation model."
        ),
    )
    parser.add_argument(
        "--token_dropout",
        type=float,
        default=0.5,
        help="Dropout probability for vision tokens in the translation model.",
    )
    parser.add_argument(
        "--feature_dropout",
        type=float,
        default=0.5,
        help="Dropout probability for vision features in the backbone model.",
    )
    parser.add_argument(
        "--add_vit_embed_token",
        action="store_true",
        help="Extracts ViT embedding token in addition to spatial tokens.",
    )
    parser.add_argument(
        "--only_vit_embed_token",
        action="store_true",
        help="Extract only ViT embedding token. (Only supported for Clip ViT atm).",
    )

    # Training
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Pretrained model.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume model training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000000,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10000,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5000,
        help="Number of warmup steps for optimizer.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-6,
        help="Learning rate decay.",
    )

    # NLP eval
    parser.add_argument(
        "--layer",
        type=int,
        help="Layer id (eg. -1) for NLP eval."
        "If left None will compute metrics with features for all layers."
        "Only applicable to models trained with features for more than one layer.",
    )

    # Qualitative eval
    parser.add_argument(
        "--loc_ids",
        type=yaml.safe_load,
        help="Location ids per layer for qualitative eval."
        "Example: {-1: [[0, 0], [1, 6]], -3: []} (given between str quotes) will generate:"
        "- description for full image with features from all layers"
        "- description for full image with features from layers -1 and -3"
        "- description for image locations (0,0) and (1,6) for layer -1"
        "if, for example, -1: [[-1, -1]] is given it will generate descriptions for all locations in layer -1",
    )
    parser.add_argument(
        "--pool_locs",
        type=str,
        default="None",
        choices=["reduce_dims", "keep_dims"],
        help="Pool lower layer locations (available for eval_qualitative).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for pooling lower layer locations.",
    )

    # Generation
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Max length of generated sequence.",
    )

    # Logging
    parser.add_argument(
        "--wandb_online",
        action="store_true",
        help="Start WANDB in online sync mode.",
    )

    args = parser.parse_args()

    args.milan = args.dataset in KEYS

    vargs = vars(args)
    n_do_evals = sum([vargs[arg] for arg in vargs.keys() if arg.startswith("do_eval")])
    do_train = n_do_evals == 0

    if n_do_evals > 0:
        assert (
            args.model_ckpt is not None
        ), "Please provide a checkpoint to perform evaluation."

    if args.do_eval_qualitative:
        assert args.dataset == "cc3m", "Invalid dataset for do_eval_qualitative."

    if args.pool_locs != "None":
        assert (
            args.layer is not None and args.layer != "-1"
        ), "pool_locs is only available for layers other than -1."

    if args.only_vit_embed_token:
        assert (
            args.vision_backbone == "clip_ViT-B/32"
        ), "only_vit_embed_token is only supported for Clip ViT models."

    torch.set_num_threads(max(1, args.num_workers))
    torch.set_num_interop_threads(max(1, args.num_workers))
    if "LOCAL_RANK" in os.environ and do_train:
        ddp_setup()
        is_distributed = True
    else:
        is_distributed = False

    if not args.wandb_online:
        os.environ["WANDB_MODE"] = "offline"

    # ensure that vision_feat_layers are in decreasing order [-1, -2, -3]
    if len(args.vision_feat_layers) > 1:
        args.vision_feat_layers = sorted(args.vision_feat_layers)[::-1]

    if args.only_vit_embed_token:
        args.add_vit_embed_token = True

    if do_train:
        path = _create_folder(args)

        with open(os.path.join(path, "train_params.txt"), "w") as f:
            f.write(str(args))

        translation_config = TranslationTransformerConfig(
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.nr_layers,
            num_attention_heads=args.nr_heads,
            nr_learnable_tokens=args.nr_learnable_tokens,
            token_dropout=args.token_dropout,
            add_vit_embed_token=args.add_vit_embed_token,
        )

    else:
        translation_config = TranslationTransformerConfig.from_json_file(
            args.model_ckpt + "translation_model_config.json"
        )
        lm_model_config = AutoConfig.from_pretrained(
            args.model_ckpt + "lm_model_config.json"
        )
        args.max_length = lm_model_config.max_length

        with open(os.path.join(args.model_ckpt, "../train_params.txt"), "r") as f:
            namespace_str = f.read()
        train_args = eval(namespace_str)
        args.vision_backbone = train_args.vision_backbone
        args.vision_feat_func = train_args.vision_feat_func
        args.vision_feat_layers = train_args.vision_feat_layers
        args.language_model = train_args.language_model
        if "feature_dropout" in train_args:
            args.feature_dropout = train_args.feature_dropout
        if "by_unit" in train_args:
            args.by_unit = train_args.by_unit
        if "only_vit_embed_token" in train_args:
            args.only_vit_embed_token = train_args.only_vit_embed_token

        if args.do_eval_nlp and len(args.vision_feat_layers) == 1:
            assert (
                args.layer is None
            ), "A layer different from None can only be given when the model has been trained on features from more than one layer."

    model = Features2WordsModel(
        translation_config=translation_config,
        cnn_name=args.vision_backbone,
        vision_feat_func=args.vision_feat_func,
        vision_feat_layers=args.vision_feat_layers,
        lm_model_name=args.language_model,
        max_length=args.max_length,
        feature_dropout=args.feature_dropout,
        only_vit_embed_token=args.only_vit_embed_token,
    )

    test_dtset = None
    if args.milan:
        transform = get_milan_transform(model.transform)
        if do_train:
            tr_dtset = MILANDataset(
                args.dataset,
                os.path.join(args.data_root, "milan", "data"),
                split="train",
                transform=transform,
                by_unit=args.by_unit,
            )
        val_dtset = MILANDataset(
            args.dataset,
            os.path.join(args.data_root, "milan", "data"),
            split="test",
            transform=transform,
            by_unit=args.by_unit,
        )
    else:
        collator = CustomDataCollator(model.tokenizer, model.nr_learnable_tokens)

        train_shard = [f"{i:05d}" for i in range(332)]
        val_shard = [f"{i:05d}" for i in range(2)]
        train_path = "cc3m/cc3m/train"
        val_path = "cc3m/cc3m/valid"

        if do_train:
            tr_dtset = get_wds_dataset(
                os.path.join(args.data_root, train_path),
                train_shard,
                model.transform,
                args.bs,
                collator=collator,
            )
        val_dtset = get_wds_dataset(
            os.path.join(args.data_root, val_path),
            val_shard,
            model.transform,
            args.bs,
            collator=collator,
            val=True,
        )

    if do_train:
        # Trainer
        trainer = Trainer(model, tr_dtset, val_dtset, path, args, is_distributed)

        trainer.train()
    else:
        model_checkpoint = torch.load(os.path.join(args.model_ckpt, "model.pt"))
        model.translation.load_state_dict(model_checkpoint["MODEL_STATE"])
        model.eval()
        model.to(0)
        print(f'Loaded model from {os.path.join(args.model_ckpt, "model.pt")}')

        val_loader = get_loader(
            val_dtset,
            args.num_workers,
            tokenizer=model.tokenizer,
            nr_learnable_tokens=model.nr_learnable_tokens,
            is_distributed=False,
            bs=args.bs,
            is_train=False,
        )
        if test_dtset is not None:
            test_loader = get_loader(
                test_dtset,
                args.num_workers,
                tokenizer=model.tokenizer,
                nr_learnable_tokens=model.nr_learnable_tokens,
                is_distributed=False,
                bs=args.bs,
                is_train=False,
            )

        if args.pool_locs == "None":
            args.pool_locs = None

        if args.do_eval_nlp:
            outname = args.dataset
            if outname in KEYS:
                if args.by_unit:
                    outname += "_by_unit"
                else:
                    outname += "_by_imgs"
            eval_nlp(
                model,
                val_loader,
                args.model_ckpt,
                outname=outname,
                add_spice=True,
                layer=args.layer,
                viz=False,
                milan=args.milan,
            )
            if test_dtset is not None:
                outname += "_TEST"
                eval_nlp(
                    model,
                    test_loader,
                    args.model_ckpt,
                    outname=outname,
                    add_spice=True,
                    layer=args.layer,
                    viz=False,
                    milan=args.milan,
                )
        if args.do_eval_qualitative:
            if args.loc_ids is not None:
                for k, v in args.loc_ids.items():
                    args.loc_ids[k] = [tuple(l) for l in v]
            foldername = f"{args.dataset}_feat_select_results"
            if args.pool_locs is not None:
                foldername += f"_{args.pool_locs}_ks{args.kernel_size}"
            eval_qualitative(
                model,
                val_loader,
                os.path.join(args.model_ckpt, foldername),
                loc_ids=args.loc_ids,
                pool_locs=args.pool_locs,
                kernel_size=args.kernel_size,
            )

    if is_distributed:
        destroy_process_group()
