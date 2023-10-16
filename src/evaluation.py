import json
import os
import sys
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import load
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm


class RunningMean:
    def __init__(self):
        self.value = 0
        self.cnt = 0

    def update(self, val, n):
        if self.cnt == 0:
            self.value = val
            self.cnt = n
        else:
            self.cnt += n
            ratio = n / self.cnt
            self.value += (val - self.value) * ratio

    def get(self):
        return self.value


def grid(n, dim):
    return sorted(list(set(product(range(n), repeat=dim))))


def compute_bert_score(res, gts, milan=False, return_individual=False):
    scorer = load("bertscore")  # reset metric
    assert len(res) == len(gts)

    hyps = []
    refs = []
    for imgid in res.keys():
        hyps.append(res[imgid].lower())
        # support for multiple references per image
        refs.append([r.lower() for r in gts[imgid]])

    if milan:
        results = scorer.compute(
            predictions=hyps,
            references=refs,
            lang="en",
            idf=True,
            rescale_with_baseline=True,
            use_fast_tokenizer=True,
        )
    else:
        results = scorer.compute(predictions=hyps, references=refs, lang="en")

    if return_individual:
        return sum(results["f1"]) / len(results["f1"]), results["f1"]
    return sum(results["f1"]) / len(results["f1"])


def compute_coco_metrics(res, gts, add_spice=False, return_individual=False):
    def setImgToEvalImgs(imgToEval, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in imgToEval:
                imgToEval[imgId] = {}
                imgToEval[imgId]["image_id"] = imgId
            imgToEval[imgId][method] = score

    def setEvalImgs(imgToEval):
        return [eval for imgId, eval in imgToEval.items()]

    # =================================================
    # Set up scorers
    # =================================================
    print("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print("setting up scorers...")
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if add_spice:
        scorers.append((Spice(), "SPICE"))

    # =================================================
    # Compute scores
    # =================================================
    eval_dict = {}
    if return_individual:
        imgToEval = {}
    for scorer, method in scorers:
        print("computing %s score..." % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval_dict[m] = sc
                if return_individual:
                    setImgToEvalImgs(imgToEval, scs, gts.keys(), m)
                print("%s: %0.3f" % (m, sc))
        else:
            eval_dict[method] = score
            if return_individual:
                setImgToEvalImgs(imgToEval, scores, gts.keys(), method)
            print("%s: %0.3f" % (method, score))

    if return_individual:
        return eval_dict, setEvalImgs(imgToEval)
    return eval_dict


@torch.no_grad()
def eval_nlp(
    model,
    eval_loader,
    save_dir,
    gpu_id=0,
    is_distributed=False,
    outname=None,
    add_spice=False,
    layer=None,
    viz=True,
    milan=False,
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    avg_loss = RunningMean()
    gt_captions_coco = {}
    pred_captions_coco = {}
    gt_captions = {}
    pred_captions = {}

    pbar = tqdm(eval_loader, file=sys.stdout)
    pbar.set_description("validating")

    plot_data = []
    for step, batch in enumerate(pbar):
        ids = batch["id"]
        gt_strs = batch["raw"]

        forbidden_keys = ["id", "raw"]
        batch = {k: v.to(gpu_id) for k, v in batch.items() if k not in forbidden_keys}

        loss = model(**batch)
        token_cnt = (batch["gt_captions"][:, 1:] != -100).sum()

        avg_loss.update(loss.item(), token_cnt.item())

        # for NLG metrics computation
        mask = batch["mask"].to(gpu_id) if milan else None
        if is_distributed:
            gen_sents = model.module.generate(
                batch["pixel_values"], mask=mask, layer=layer
            )
            pred_strs = model.module.tokenizer.batch_decode(
                gen_sents, skip_special_tokens=True
            )
        else:
            gen_sents = model.generate(batch["pixel_values"], mask=mask, layer=layer)
            pred_strs = model.tokenizer.batch_decode(
                gen_sents, skip_special_tokens=True
            )

        for out_idx in range(len(ids)):
            gen = pred_strs[out_idx]
            gt = gt_strs[out_idx]
            img_id = ids[out_idx]

            gt_captions_coco[img_id] = [{"image_id": img_id, "caption": g} for g in gt]
            gt_captions[img_id] = gt

            pred_captions_coco[img_id] = [{"image_id": img_id, "caption": gen}]
            pred_captions[img_id] = gen

            if (
                viz and (step == 0 or step == 5) and out_idx < 10
            ):  # log first 10 images of first/6th batch
                og_img = batch["pixel_values"][out_idx].cpu()
                grid = make_grid([og_img], padding=5, pad_value=1, nrow=2)
                image_title = f"image_{img_id}"
                cpt = "GT: " + gt[0] + "\nGen: " + gen
                plot_data.append((image_title, grid, cpt, step))

    if outname is not None:
        save_dir = os.path.join(save_dir, outname + "-")
    else:
        if not save_dir.endswith("/"):
            save_dir += "/"

    if layer is None:
        save_dir += "all_features_"
    else:
        save_dir += f"layer{layer}_features_"

    eval_results = {}
    eval_results = compute_coco_metrics(
        pred_captions_coco, gt_captions_coco, add_spice=add_spice
    )

    to_save = pred_captions
    if milan:
        new_pred_captions = {}
        for imgid, caption in pred_captions.items():
            imginfo = eval_loader.dataset.idx_to_info[int(imgid)]
            dataset, layer, unit = imginfo["dataset"], imginfo["layer"], imginfo["unit"]
            new_pred_captions[imgid] = {
                "dataset": dataset,
                "layer": str(layer),
                "neuron": str(unit),
                "caption": caption,
            }
        to_save = new_pred_captions

    with open(save_dir + "val-captions.json", "w") as f:
        json.dump(to_save, f)

    bertscore = compute_bert_score(pred_captions, gt_captions, milan=milan)
    print(f"BERTScore: {bertscore}")

    eval_results["BERTScore"] = bertscore
    json.dump(eval_results, open(save_dir + "val-metrics-overall.json", "w"))
    model.train()

    return avg_loss.get(), eval_results, plot_data


@torch.no_grad()
def eval_qualitative(
    model, eval_loader, save_dir, gpu_id=0, loc_ids=None, pool_locs=None, kernel_size=3
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    convert_to_pil = ToPILImage()
    grid_sizes = model.grid_sizes[::-1]

    stride = 1
    if loc_ids is None:
        layers = model.vision_feat_layers
        loc_ids = {}
        for i in range(len(layers)):
            l = layers[i]
            if i > 0 and pool_locs == "reduce_dims":
                loc_ids[l] = loc_ids[layers[0]].copy()
                stride = int(gs[l] / gs[-1])
            else:
                gs = grid_sizes[l]
                loc_ids[l] = grid(gs, 2)
    else:
        layers = list(loc_ids.keys())
        for i in range(len(layers)):
            l = layers[i]
            if l not in model.vision_feat_layers:
                raise ValueError(
                    f"Layer {l} not in layers that model trained with ({model.vision_feat_layers})"
                )

            if len(loc_ids[l]) > 0:
                if loc_ids[l][0] == (-1, -1):
                    if i > 0 and pool_locs == "reduce_dims":
                        loc_ids[l] = loc_ids[layers[0]].copy()
                        stride = int(gs[l] / gs[-1])
                    else:
                        gs = grid_sizes[l]
                        loc_ids[l] = grid(gs, 2)

        for l in layers:
            if len(loc_ids[l]) > 0:
                gs = grid_sizes[l]
                max_dim0 = max(loc_ids[l], key=lambda item: item[0])
                max_dim1 = max(loc_ids[l], key=lambda item: item[1])
                if max(max_dim0) >= gs:
                    raise ValueError(
                        f"Location {max_dim0} is out of bounds for grid size {gs}"
                    )
                if max(max_dim1) >= gs:
                    raise ValueError(
                        f"Location {max_dim1} is out of bounds for grid size {gs}"
                    )

    for step, batch in tqdm(enumerate(eval_loader)):
        pixel_values = batch["pixel_values"].to(gpu_id)
        bs = pixel_values.shape[0]

        # TODO: adapt for MILAN

        # get full image caption with features from all layers
        print("Generating full image description with features from all layers")
        gen = model.generate(pixel_values=pixel_values)
        all_layers_full_gen = model.tokenizer.batch_decode(
            gen, skip_special_tokens=True
        )

        # model was trained on features from more than one layer
        # get full image caption with layer-wise features
        layer_full_gens = {}
        for layer in layers:
            print(f"Generating full image description with features from layer {layer}")
            gen = model.generate(pixel_values=pixel_values, layer=layer)
            layer_full_gens[layer] = model.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )

        # spatial captions
        layer_partial_gens = {}
        for l in layers:
            partial_gens = {}
            print(
                f"Generating {len(loc_ids[l])} partial image descriptions with features from layer {l}"
            )
            for t in loc_ids[l]:
                gen = model.generate(
                    pixel_values=pixel_values,
                    layer=l,
                    feat_index=t,
                    pool_locs=pool_locs,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                gen_str = model.tokenizer.batch_decode(gen, skip_special_tokens=True)
                partial_gens[t] = gen_str
            layer_partial_gens[l] = partial_gens

        # save results for each image in batch
        for bid in range(bs):
            img_id = batch["id"][bid]
            single_pixel_values = pixel_values[bid]
            fi_cpt_all_feat = all_layers_full_gen[bid].replace("\n", " ")
            if "raw" in batch:
                captions = batch["raw"][bid]

            # just resize and crop (do not normalize) to save image
            img_to_save = model.unnormalize(single_pixel_values)
            img_to_save = convert_to_pil(img_to_save)
            img_to_save.save(os.path.join(save_dir, str(img_id) + ".png"))

            text_file = open(os.path.join(save_dir, str(img_id) + ".txt"), "w")

            if "raw" in batch:
                print("GT Captions:\n", file=text_file)
                for c in captions:
                    print(f"\t{c}\n", file=text_file)
                print("\n", file=text_file)

            # save full image caption with all features
            print(
                f"Full gen caption (with all features): {fi_cpt_all_feat}\n",
                file=text_file,
            )

            # save full image caption with layer-wise features
            for k, v in layer_full_gens.items():
                v_bid = v[bid].replace("\n", "")
                print(f"Full gen caption (layer {k}): {v_bid}\n", file=text_file)

            # save partial captions per layer
            print("Partial captions:\n", file=text_file)
            for k, v in layer_partial_gens.items():
                if len(loc_ids[k]) > 0:
                    print(f"\tLayer {k}:\n", file=text_file)
                for t in loc_ids[k]:
                    pt_cpt = v[t][bid].replace("\n", " ")
                    print(f"\t\t({t[0]}, {t[1]}): {pt_cpt}\n", file=text_file)

            text_file.close()

        if (step + 1) * bs > 100:
            return


def prepare_input_ids(texts, tokenizer, add_prompt=False):
    tokens = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=100,
        return_tensors="pt",
    )

    input_ids = tokens.input_ids
    att_masks = tokens.attention_mask

    if add_prompt:
        prompt_id = tokenizer("the").input_ids
        assert len(prompt_id) == 2
        prompt_id = prompt_id[1]

        input_ids = torch.cat([input_ids[:, :1], input_ids], dim=1)
        input_ids[:, 1] = prompt_id
        att_masks = torch.cat([att_masks[:, :1], att_masks], dim=1)
        att_masks[:, 1] = 0

    gts_cap = input_ids.clone()
    gts_cap = torch.where(
        att_masks == 0, -100, gts_cap
    )  # ignore padding tokens in loss

    return input_ids, att_masks, gts_cap


@torch.no_grad()
def get_saliency_map(
    model,
    loc_ids,
    pixel_values,
    labels,
    layer_id,
    gpu_id=0,
    add_prompt=False,
    hres=False,
    pool_locs=None,
    kernel_size=3,
    stride=1,
):
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    input_ids, attn_masks, gt_caps = prepare_input_ids(
        labels, model.tokenizer, add_prompt=add_prompt
    )

    pixel_values = pixel_values.to(gpu_id)
    bs = pixel_values.shape[0]

    # compute image features
    # TODO: adapt for MILAN
    img_features_cnn, forced_token_mask = model.forward_vision_model(
        pixel_values, just_cnn_features=True
    )

    dim_lowest_layer = img_features_cnn[-1].shape[
        2:
    ]  # get shape of lowest layer (usually the third)
    if layer_id is None:
        # interpolate higher layers to dim of lowest layer
        for i in range(len(img_features_cnn[:-1])):
            img_features_cnn[i] = F.interpolate(
                img_features_cnn[i], tuple(dim_lowest_layer), mode="bilinear"
            )
    else:
        img_features_layer = img_features_cnn[layer_id]
        if pool_locs is not None:
            img_features_layer = F.avg_pool2d(
                img_features_layer,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
                stride=stride,
                count_include_pad=False,
            )
        elif hres:
            img_features_layer = F.interpolate(
                img_features_layer, tuple(dim_lowest_layer), mode="bilinear"
            )

    loss_per_loc = []
    logits_per_loc = []
    for loc in tqdm(loc_ids, desc="Location"):
        if layer_id is None:
            img_features = []
            for i in range(len(img_features_cnn)):
                feats = img_features_cnn[i][:, :, loc[0], loc[1]]
                feats = feats[:, :, None, None]
                img_features.append(feats.to(gpu_id))
            token_mask = None
        else:
            feats = img_features_layer[:, :, loc[0], loc[1]]
            feats = feats[:, :, None, None]
            img_features = [
                torch.zeros((bs, f.shape[1], 1, 1)).to(gpu_id) for f in img_features_cnn
            ]
            img_features[layer_id] = feats

            token_mask = torch.ones((bs, model.feat_seq_len)).to(
                gpu_id
            )  # mask all layer features
            token_mask[:, layer_id] = torch.zeros((bs)).to(
                gpu_id
            )  # unmask corresponding layer features

            token_mask = torch.cat(
                (token_mask, torch.zeros((bs, model.nr_learnable_tokens)).to(gpu_id)),
                dim=1,
            )
            token_mask = token_mask.bool()

        # batch-wise feature encoding
        encoded_features = model.translation(
            img_features, token_mask=token_mask, forced_token_mask=forced_token_mask
        )
        encoded_features = encoded_features[:, -model.nr_learnable_tokens :, :]

        input_ids = input_ids.to(gpu_id)
        inputs_embeds = model.embedding_weights(input_ids)
        inputs_embeds = torch.cat(
            (inputs_embeds[:, :1, :], encoded_features, inputs_embeds[:, 1:, :]), dim=1
        )

        attention_mask = torch.cat(
            (
                attn_masks[:, :1],
                torch.ones(
                    (encoded_features.shape[0], model.nr_learnable_tokens),
                    dtype=torch.long,
                ),
                attn_masks[:, 1:],
            ),
            dim=1,
        )

        gt_captions = torch.cat(
            (
                gt_caps[:, :1],
                torch.ones(
                    (encoded_features.shape[0], model.nr_learnable_tokens),
                    dtype=torch.long,
                )
                * (-100),
                gt_caps[:, 1:],
            ),
            dim=1,
        )

        # loss calculation begins
        out = model.lm_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask.to(gpu_id)
        )
        logits = out.logits[:, model.nr_learnable_tokens + 1 :, :]
        input_ids_wo_bos = input_ids[:, 1:].unsqueeze(2)
        logits = torch.gather(logits, 2, input_ids_wo_bos).squeeze(2)

        # shift so that tokens < n predict n
        shift_logits = out.logits[..., :-1, :].permute(0, 2, 1).contiguous()
        shift_labels = gt_captions[..., 1:].contiguous().to(gpu_id)
        # flatten the tokens
        loss = loss_fct(shift_logits, shift_labels)
        loss_per_loc.append(loss.sum(1))
        logits_per_loc.append(logits.sum(1))

    loss_per_loc = torch.stack(loss_per_loc, dim=1)
    logits_per_loc = torch.stack(logits_per_loc, dim=1)
    return loss_per_loc, logits_per_loc
