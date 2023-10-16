import os
import random
from random import choice
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Normalize, ToTensor

from milan_keys import DATASET_GROUPINGS, KEYS, TRAIN_TEST_PAIRS, WITHIN_NETWORK

random.seed(0)


class CustomDataCollator:
    def __init__(self, tokenizer, nr_learnable_tokens) -> None:
        self.tokenizer = tokenizer
        self.nr_learnable_tokens = nr_learnable_tokens

    def __call__(self, batch) -> Dict[str, Any]:
        out_batch = {}

        if "id" in batch[0]:
            ids = [i["id"] for i in batch]
        elif "__key__" in batch[0]:
            ids = [i["__key__"] for i in batch]
        else:
            raise ValueError

        imgs = [i["image"] for i in batch]
        imgs = torch.stack(imgs)

        texts = [i["text"] + self.tokenizer.eos_token for i in batch]
        texts = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        )

        bs = len(ids)

        gt_captions = texts.input_ids.clone()
        gt_captions = torch.where(
            texts.attention_mask == 0, -100, gt_captions
        )  # ignore padding tokens in loss
        gt_captions = torch.cat(
            (
                gt_captions[:, :1],
                torch.ones((bs, self.nr_learnable_tokens), dtype=torch.long) * (-100),
                gt_captions[:, 1:],
            ),
            dim=1,
        )

        attention_mask = texts.attention_mask
        attention_mask = torch.cat(
            (
                attention_mask[:, :1],
                torch.ones((bs, self.nr_learnable_tokens), dtype=torch.long),
                attention_mask[:, 1:],
            ),
            dim=1,
        )

        out_batch = {
            "id": ids,
            "pixel_values": imgs,
            "input_ids": texts["input_ids"],
            "attention_mask": attention_mask,
            "gt_captions": gt_captions,
        }

        if "label" in batch[0]:
            out_batch["label"] = torch.tensor([i["label"] for i in batch])

        # raw text for NLP metrics
        if "raw" in batch[0]:
            out_batch["raw"] = [x["raw"] for x in batch]
        else:
            out_batch["raw"] = [
                [x["text"]] for x in batch
            ]  # single caption (webdatasets)

        if "mask" in batch[0]:
            masks = [i["mask"] for i in batch]
            masks = torch.stack(masks)
            out_batch["mask"] = masks

        return out_batch


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = "png" in sample or "jpg" in sample or "jpeg" in sample
    return has_caption and has_image


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def get_wds_dataset(data_root, shard, transform, batch_size, collator=None, val=False):
    """
    return a dataset that returns an image, and text
    """
    if val == False:
        if "LOCAL_RANK" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            world_size = 1
            local_rank = 0

        shard_split = chunkIt(shard, world_size)
        shard = shard_split[local_rank]

    total_shards = len(shard)
    shard = "{" + f"{shard[0]}..{shard[-1]}" + "}.tar"

    input_shards = os.path.join(data_root, shard)

    pipeline = [
        wds.SimpleShardList(input_shards),
        # at this point we have an iterator over all the shards
    ]

    if val == False:
        pipeline.extend(
            [
                wds.shuffle(bufsize=total_shards, initial=total_shards),
            ]
        )

    pipeline.extend(
        [
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb"),
            wds.rename(image="jpg;png;jpeg", text="txt"),
            wds.map_dict(image=transform, text=lambda text: text),
        ]
    )

    if val == False:
        pipeline.extend([wds.shuffle(100 * batch_size)])

    pipeline.extend([wds.batched(batch_size, partial=False, collation_fn=collator)])

    dataset = wds.DataPipeline(*pipeline)
    return dataset


def get_loader(
    dtset,
    num_workers,
    tokenizer=None,
    nr_learnable_tokens=None,
    is_distributed=None,
    bs=None,
    is_train=None,
):
    if isinstance(dtset, MILANDataset):
        collator = CustomDataCollator(tokenizer, nr_learnable_tokens)

        return DataLoader(
            dtset,
            batch_size=bs,
            num_workers=num_workers,
            drop_last=is_train,
            collate_fn=collator,
            pin_memory=True,
            shuffle=(is_train) and not is_distributed,
            sampler=DistributedSampler(dtset)
            if (is_train) and is_distributed
            else None,
        )
    else:
        return wds.WebLoader(
            dtset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )


def get_milan_transform(transform):
    tfs = [ToTensor()]
    for t in transform.transforms:
        if isinstance(t, Normalize):
            tfs.append(t)
            break

    target_transform = Compose(tfs)
    return target_transform


class MILANDataset(Dataset):
    def __init__(self, dtset, root, split="train", transform=None, by_unit=False):
        super().__init__()
        self.dtset = KEYS[dtset]
        self.root = root
        self.split = split
        self.transform = transform
        self.by_unit = by_unit
        self.nr_imgs_per_unit = 15

        self.idx_to_info = {}
        self.global_id = 0
        if self.dtset in DATASET_GROUPINGS:
            if self.split == "train":
                dtset_list = DATASET_GROUPINGS[self.dtset]
            else:
                dtset_list = TRAIN_TEST_PAIRS[self.dtset]
                if dtset_list in DATASET_GROUPINGS:
                    dtset_list = DATASET_GROUPINGS[dtset_list]
                else:
                    dtset_list = [dtset_list]
            for name in dtset_list:
                self.get_single_dataset(name)
        elif self.dtset in WITHIN_NETWORK:
            self.get_single_dataset(self.dtset, self.split)
        else:
            raise NotImplementedError

    def get_single_dataset(self, name, split=None):
        anns_csv = os.path.join(os.path.join(self.root, name), "annotations.csv")
        anns = pd.read_csv(anns_csv)
        anns["summary"] = anns["summary"].astype(str)
        anns["layer"] = anns["layer"].astype(str)
        layers = sorted(list(anns["layer"].unique()))

        if split is not None:
            # process split file
            split_units = {}
            split_idxs = torch.load(
                os.path.join(self.root, name, f'{name.replace("/", "_")}-splits.pth')
            )[split]
            for item in split_idxs:
                if item["layer"] not in split_units:
                    split_units[item["layer"]] = []
                split_units[item["layer"]].append(item["unit"])

        for l in layers:
            units = anns[anns["layer"] == l]["unit"].unique()
            for u in units:
                # ignore units that are not in this split
                if (split is not None) and (u not in split_units[l]):
                    continue

                cpts = anns.loc[(anns["layer"] == l) & (anns["unit"] == u)]
                cpts = list(cpts["summary"].values)

                if self.by_unit:
                    dct = {"dataset": name, "layer": l, "unit": u, "captions": cpts}
                    self.idx_to_info[self.global_id] = dct
                    self.global_id += 1
                else:
                    # repeat information for all images of unit
                    for i in range(self.nr_imgs_per_unit):
                        dct = {
                            "dataset": name,
                            "layer": l,
                            "unit": u,
                            "imgid": i,
                            "captions": cpts,
                        }
                        self.idx_to_info[self.global_id] = dct
                        self.global_id += 1

    def __len__(self):
        return len(self.idx_to_info)

    def __getitem__(self, index):
        info = self.idx_to_info[index]
        dataset, layer, unit = info["dataset"], info["layer"], info["unit"]

        unit_imgs = np.load(
            os.path.join(self.root, dataset, layer, f"images_{unit}.npy")
        )
        unit_masks = np.load(
            os.path.join(self.root, dataset, layer, f"masks_{unit}.npy")
        )

        if self.by_unit == False:
            imgid = info["imgid"]
            unit_imgs = unit_imgs[imgid]
            unit_imgs = np.swapaxes(unit_imgs, 0, 1)
            unit_imgs = np.swapaxes(unit_imgs, 1, 2)
            if self.transform:
                unit_imgs = self.transform(unit_imgs)
            unit_masks = unit_masks[imgid]
        else:
            if self.transform:
                transformed = []
                for img in unit_imgs:
                    img = np.swapaxes(img, 0, 1)
                    img = np.swapaxes(img, 1, 2)
                    transformed.append(self.transform(img))
                unit_imgs = torch.stack(transformed)

        mask = torch.from_numpy(unit_masks).float()

        captions = info["captions"]
        captions = choice(captions)

        if self.split == "train":
            return {
                "image": unit_imgs,
                "mask": mask,
                "text": captions,
                "id": str(index),
            }
        return {
            "image": unit_imgs,
            "mask": mask,
            "text": captions,
            "raw": info["captions"],
            "id": str(index),
        }
