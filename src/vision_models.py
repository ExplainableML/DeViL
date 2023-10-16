import clip
import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torchvision.transforms._presets import ImageClassification


class WrapOutputInList(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [x]


class IndexOutput(nn.Module):
    def __init__(
        self, vision_feat_level, add_vit_embed_token=False, only_vit_embed_token=False
    ):
        super().__init__()
        self.vision_feat_level = vision_feat_level
        self.add_vit_embed_token = add_vit_embed_token
        self.only_vit_embed_token = only_vit_embed_token

    def select_indices(self, x):
        return [x[vfl] for vfl in self.vision_feat_level]

    def forward(self, x):
        if self.add_vit_embed_token and not self.only_vit_embed_token:
            # join lists
            return self.select_indices(x[0]) + self.select_indices(x[1])
        else:
            return self.select_indices(x)


def get_vision_model(
    vision_model_name,
    vision_feat_level=[-1],
    add_vit_embed_token=False,
    only_vit_embed_token=False,
):
    transform = None
    if vision_model_name.startswith("timm"):
        assert not add_vit_embed_token, "ViT embed token unsupported for timm models"
        timm_model_name = vision_model_name[5:]
        vision_model = timm.create_model(
            timm_model_name, features_only=True, pretrained=True
        )

        # Get model dims
        vision_dim = []
        grid_size = []
        if vision_model.default_cfg["pool_size"] is not None:
            assert (
                vision_model.default_cfg["pool_size"][0]
                == vision_model.default_cfg["pool_size"][1]
            )
            total_pool_size = vision_model.default_cfg["pool_size"][0]
        else:
            total_pool_size = 224 // vision_model.feature_info[-1]["reduction"]

        for vfl in vision_feat_level:
            vision_dim.append(vision_model.feature_info[vfl]["num_chs"])
            if timm_model_name.startswith(("resnet", "efficientformerv2")):
                level_factor = (
                    vision_model.feature_info[-1]["reduction"]
                    // vision_model.feature_info[vfl]["reduction"]
                )
            elif timm_model_name.startswith("davit"):
                assert vfl < 0
                level_factor = prod(
                    [vision_model.feature_info[l]["reduction"] for l in range(vfl, -1)]
                )
            else:
                raise NotImplementedError
            grid_size.append(total_pool_size * level_factor)

        config = resolve_data_config({}, model=vision_model)
        transform = create_transform(**config)

        vision_model = nn.Sequential(vision_model, IndexOutput(vision_feat_level))

    elif vision_model_name.startswith("clip"):
        clip_model_name = vision_model_name[5:]
        vision_model, transform = clip.load(clip_model_name)
        vision_model = vision_model.visual
        vision_model.float()
        if clip_model_name.startswith("RN"):
            assert (
                not add_vit_embed_token
            ), "ViT embed token unsupported for clip_RN models"
            new_forward = clip_resnet_forward
            vision_dim, grid_size = get_resnet_dims(vision_feat_level)
        elif clip_model_name.startswith("ViT"):
            vision_model.add_vit_embed_token = add_vit_embed_token
            vision_model.only_vit_embed_token = only_vit_embed_token
            new_forward = clip_vit_forward
            vision_dim = []
            grid_size = []
            if not only_vit_embed_token:
                vision_dim = [768] * len(vision_feat_level)
                grid_size = [7] * len(vision_feat_level)
            if add_vit_embed_token:
                vision_dim += [768] * len(vision_feat_level)
                grid_size += [1] * len(vision_feat_level)
        else:
            raise ValueError

        # add new_forward function to the model instance as a class method
        bound_method = new_forward.__get__(vision_model, vision_model.__class__)
        setattr(vision_model, "forward", bound_method)

        vision_model = nn.Sequential(
            vision_model,
            IndexOutput(vision_feat_level, add_vit_embed_token, only_vit_embed_token),
        )
    else:
        raise NotImplementedError

    assert transform is not None
    unnormalize = adjust_resize_and_get_unnormalize(transform)

    return vision_model, vision_dim, grid_size, transform, unnormalize


def get_resnet_dims(vision_feat_level):
    vision_dim = [256, 512, 1024, 2048]
    grid_size = [56, 28, 14, 7]
    vision_dim = [vision_dim[vfl] for vfl in vision_feat_level]
    grid_size = [grid_size[vfl] for vfl in vision_feat_level]
    return vision_dim, grid_size


def clip_resnet_forward(self, x):
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    # x = self.attnpool(x)

    return [x1, x2, x3, x4]


def clip_vit_forward(self, x):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    outputs = []
    embed_outputs = []
    for mod in self.transformer.resblocks:
        x = mod(x)
        if not self.only_vit_embed_token:
            outputs.append(x[1:].permute(1, 2, 0).reshape(x.shape[1], x.shape[2], 7, 7))
        if self.add_vit_embed_token:
            embed_outputs.append(
                x[:1].permute(1, 2, 0).reshape(x.shape[1], x.shape[2], 1, 1)
            )

    """
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        x = x @ self.proj
    """
    if self.only_vit_embed_token:
        return embed_outputs
    elif self.add_vit_embed_token:
        return (outputs, embed_outputs)
    else:
        return outputs


def adjust_resize_and_get_unnormalize(transform):
    if isinstance(transform, ImageClassification):
        mean = transform.mean
        std = transform.std
        transform.resize_size = (
            transform.crop_size
        )  # directly resize to final crop size (224x224)
    elif isinstance(transform, torchvision.transforms.Compose):
        has_normalize = False
        crop_size = None
        for t in transform.transforms:
            if isinstance(t, torchvision.transforms.Normalize):
                mean = t.mean
                std = t.std
                has_normalize = True
            if isinstance(t, torchvision.transforms.CenterCrop):
                crop_size = t.size if isinstance(t.size, int) else t.size[0]
        if crop_size is None:
            print("Crop size not found, leaving resizing transform unchanged.")
        else:
            for t in transform.transforms:
                if isinstance(t, torchvision.transforms.Resize):
                    t.size = crop_size
        if not has_normalize:
            return nn.Identity()
    else:
        raise NotImplementedError

    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
    unnormalize = torchvision.transforms.Normalize(mean=-mean / std, std=1 / std)
    return unnormalize
