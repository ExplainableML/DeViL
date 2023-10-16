from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LogitsProcessorList, PretrainedConfig, ViTPreTrainedModel
from transformers.utils import logging

from language_model import get_language_model
from vision_models import get_vision_model

logger = logging.get_logger(__name__)


class TranslationTransformerConfig(PretrainedConfig):
    model_type = "translationtransformer"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        initializer_range=0.02,
        dropout_prob=0.0,
        layer_norm_eps=1e-12,
        nr_learnable_tokens=10,
        feat_seq_len=49,
        token_dropout=0.5,
        add_vit_embed_token=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps

        self.nr_learnable_tokens = nr_learnable_tokens
        self.feat_seq_len = feat_seq_len
        self.token_dropout = token_dropout
        self.add_vit_embed_token = add_vit_embed_token


class TranslationTransformer(ViTPreTrainedModel):
    config_class = TranslationTransformerConfig

    def __init__(self, config):
        super().__init__(config)

        self.add_vit_embed_token = config.add_vit_embed_token
        self.learnable_inputs = nn.Parameter(
            torch.randn((1, config.nr_learnable_tokens, config.hidden_size)) * 0.1
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.feat_seq_len, config.hidden_size) * 0.1
        )

        bernoulli_prob = torch.cat(
            [
                torch.ones((1, config.feat_seq_len)) * config.token_dropout,
                torch.zeros(1, config.nr_learnable_tokens),
            ],
            dim=1,
        )
        self.register_buffer("bernoulli_prob", bernoulli_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout_prob,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.projections = nn.ModuleList()
        for i in range(len(config.img_embed_dims)):
            self.projections.append(
                nn.Linear(config.img_embed_dims[i], config.hidden_size)
            )
        self.out_proj = nn.Linear(config.hidden_size, config.lm_embed_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def _apply_token_mask_mode(self, token_mask, mode):
        n_tokens = len(self.projections)
        # mode == 0: mask all vit embed tokens, only allow spatial embeds
        m = mode == 0
        token_mask[m, n_tokens // 2 : n_tokens] = 1.0
        # mode == 1: mask all spatial embeds, only allow vit embed tokens
        m = mode == 1
        token_mask[m, : n_tokens // 2] = 1.0
        # mode == 2: mask per layer, keep masking of vit embed token and spatial same for each layer
        m = mode == 2
        token_mask[m, :n_tokens:2] = token_mask[m, 1:n_tokens:2]

        return token_mask

    def forward(
        self,
        img_feats,
        token_mask=None,
        forced_token_mask=None,
    ):
        if hasattr(self, "projections"):
            for i in range(len(img_feats)):
                feats = img_feats[i]
                dims = feats.shape
                feats = torch.reshape(feats, (dims[0], dims[1], dims[2] * dims[3]))
                feats = torch.permute(feats, (0, 2, 1))
                img_feats[i] = self.projections[i](feats)

            img_feats = torch.cat(img_feats, dim=1)

        img_feats = img_feats + self.position_embeddings

        bs = img_feats.shape[0]
        expanded_learnable_inputs = self.learnable_inputs.expand(bs, -1, -1)
        inputs = torch.cat((img_feats, expanded_learnable_inputs), dim=1)

        assert token_mask is None or forced_token_mask is None
        if forced_token_mask is not None and (
            (self.training and token_mask is None and self.bernoulli_prob.sum() == 0.0)
            or (not self.training)
        ):
            assert not forced_token_mask.all(dim=1).any()
            token_mask = torch.cat(
                [
                    forced_token_mask,
                    torch.zeros(
                        (forced_token_mask.shape[0], self.config.nr_learnable_tokens),
                        device=img_feats.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        elif self.training and token_mask is None and self.bernoulli_prob.sum() > 0.0:
            # sample token mask
            bernoulli_prob = self.bernoulli_prob.expand(bs, -1)
            token_mask = torch.bernoulli(bernoulli_prob)
            if self.add_vit_embed_token:
                mask_mode = torch.randint(3, size=(bs,))
                token_mask = self._apply_token_mask_mode(token_mask, mask_mode)

            if forced_token_mask is not None:
                token_mask[:, : forced_token_mask.shape[1]] += forced_token_mask.float()
                token_mask = torch.clamp(token_mask, max=1.0)

            # make sure at least one token is not masked
            all_masked = token_mask.sum(dim=1) == img_feats.shape[1]
            while all_masked.sum() > 0:
                all_masked = all_masked.unsqueeze(1)
                new_token_mask = torch.bernoulli(bernoulli_prob)
                if self.add_vit_embed_token:
                    token_mask = self._apply_token_mask_mode(token_mask, mask_mode)
                token_mask = ~all_masked * token_mask + all_masked * new_token_mask
                if forced_token_mask is not None:
                    token_mask[
                        :, : forced_token_mask.shape[1]
                    ] += forced_token_mask.float()
                    token_mask = torch.clamp(token_mask, max=1.0)
                all_masked = token_mask.sum(dim=1) == img_feats.shape[1]

            token_mask = token_mask.bool()

        sequence_output = self.encoder(inputs, src_key_padding_mask=token_mask)

        sequence_output = self.layernorm(sequence_output)
        sequence_output = self.out_proj(sequence_output)

        return sequence_output


class LMBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def greedy(
        self,
        input_ids: torch.LongTensor,
        visual_features: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs,
    ):
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        max_length = (
            max_length if max_length is not None else self.lm_model.config.max_length
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.lm_model.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.lm_model.config.eos_token_id
        )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        while True:
            # prepare model inputs
            model_inputs = self.lm_model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            if "attention_mask" in model_inputs:
                if model_inputs["attention_mask"] is not None:
                    model_inputs["attention_mask"] = torch.ones(
                        (
                            input_ids.shape[0],
                            input_ids.shape[1] + self.nr_learnable_tokens,
                        ),
                        device=input_ids.device,
                        dtype=torch.long,
                    )

            if visual_features is not None:
                inputs_embeds = self.embedding_weights(model_inputs["input_ids"])
                inputs_embeds = torch.cat(
                    (inputs_embeds[:, :1, :], visual_features, inputs_embeds[:, 1:, :]),
                    dim=1,
                )

                model_inputs.pop("input_ids")
                model_inputs["inputs_embeds"] = inputs_embeds

            # forward pass to get next token
            outputs = self.lm_model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )

                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens.detach()[:, None]], dim=1)
            if "past_key_values" in outputs:
                model_kwargs["past_key_values"] = outputs.past_key_values
                visual_features = None  # are only needed the first time; afterwards we use past attentions and just the latest input id

            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat(
                    [
                        token_type_ids,
                        torch.ones(
                            (token_type_ids.shape[0], 1),
                            dtype=torch.long,
                            device=token_type_ids.device,
                        ),
                    ],
                    dim=-1,
                )

            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break

        return input_ids


class Features2WordsModel(LMBase):
    def __init__(
        self,
        translation_config,
        cnn_name="resnet50",
        vision_feat_func="none",
        vision_feat_layers=[-1],
        lm_model_name="gpt2",
        max_length=None,
        feature_dropout=0.0,
        only_vit_embed_token=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_feat_func = vision_feat_func
        self.vision_feat_layers = vision_feat_layers
        self.only_vit_embed_token = only_vit_embed_token

        self.register_buffer(
            "feature_dropout", torch.tensor([feature_dropout]), persistent=False
        )

        # VISION MODEL
        (
            self.cnn,
            img_embed_dims,
            self.grid_sizes,
            self.transform,
            self.unnormalize,
        ) = get_vision_model(
            cnn_name,
            vision_feat_layers,
            add_vit_embed_token=translation_config.add_vit_embed_token,
            only_vit_embed_token=self.only_vit_embed_token,
        )
        if self.vision_feat_func == "avg_pooling":
            self.feat_seq_len = len(self.grid_sizes)
        else:
            self.feat_seq_len = sum([gs**2 for gs in self.grid_size])
            raise NotImplementedError

        # freeze cnn
        for p in self.cnn.parameters():
            p.requires_grad = False

        # LANGUAGE MODEL
        (
            self.lm_model,
            self.tokenizer,
            self.embedding_weights,
            lm_embed_dim,
        ) = get_language_model(lm_model_name)

        self.lm_model.config.pad_token_id = self.lm_model.config.eos_token_id
        if max_length is not None:
            self.lm_model.config.max_length = max_length

        # freeze language model
        for p in self.lm_model.parameters():
            p.requires_grad = False

        # TRANSLATION MODEL
        self.nr_learnable_tokens = translation_config.nr_learnable_tokens
        translation_config.feat_seq_len = self.feat_seq_len
        translation_config.img_embed_dims = img_embed_dims
        translation_config.lm_embed_dim = lm_embed_dim
        self.translation = TranslationTransformer(translation_config)

        self.train()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.cnn.eval()

    def forward_vision_model(
        self,
        pixel_values: torch.FloatTensor,
        mask: torch.LongTensor = None,
        just_cnn_features: bool = False,
    ):
        dims = None
        if pixel_values.dim() > 4:  # MILAN dataset by neuron (bs, 15, C, H, W)
            dims = pixel_values.shape
            pixel_values = pixel_values.view(
                dims[0] * dims[1], dims[2], dims[3], dims[4]
            )

            if mask is not None:
                mdims = mask.shape
                mask = mask.view(mdims[0] * mdims[1], mdims[2], mdims[3], mdims[4])

        img_features = self.cnn(pixel_values)

        forced_token_mask = []
        if just_cnn_features == False:
            if self.vision_feat_func == "avg_pooling":
                if (
                    self.feature_dropout.item() > 0.0 and self.training
                ) or mask is not None:
                    pooled_img_features = []
                    for imgf in img_features:
                        imgf_b4_flatten = imgf
                        imgf = imgf.flatten(-2)
                        if imgf.shape[-1] == 1 and mask is None:
                            # don't apply feature dropout on vit embed token
                            pooled_img_features.append(imgf.unsqueeze(-1))
                            continue
                        # 1 means keep, 0 means drop
                        if mask is not None:
                            # 1 means keep, 0 means drop
                            dropout_mask = F.interpolate(
                                mask,
                                imgf_b4_flatten.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            # Normalize the masks so they look more like attention. If any
                            # of them are all zeros, we'll end up with divide-by-zero errors.
                            zeros = torch.zeros_like(dropout_mask)
                            valid = (
                                ~dropout_mask.isclose(zeros)
                                .all(dim=-1)
                                .all(dim=-1)
                                .view(-1)
                            )
                            indices = valid.nonzero().squeeze()
                            dropout_mask[indices] /= dropout_mask[indices].sum(
                                dim=(-1, -2), keepdim=True
                            )

                            pooled_imgf = imgf_b4_flatten.mul(dropout_mask).sum(
                                dim=(-1, -2), keepdim=True
                            )

                            if dims is not None:
                                feat_dims = pooled_imgf.shape
                                pooled_imgf = pooled_imgf.view(
                                    dims[0],
                                    dims[1],
                                    feat_dims[1],
                                    feat_dims[2],
                                    feat_dims[3],
                                )
                                mdims = dropout_mask.shape
                                dropout_mask = dropout_mask.view(
                                    dims[0], dims[1], mdims[1], mdims[2], mdims[3]
                                )
                                dropout_mask = dropout_mask.sum(dim=1).sum(
                                    dim=(-1, -2), keepdim=True
                                )
                                indices = dropout_mask.view(dims[0], -1).sum(1) != 0.0
                                dropout_mask[indices] = 1 / dropout_mask[indices]
                                pooled_imgf = pooled_imgf.sum(dim=1) * dropout_mask
                                forced_token_mask.append(~indices)
                            pooled_img_features.append(pooled_imgf)
                        else:
                            dropout_mask = torch.bernoulli(
                                (1.0 - self.feature_dropout).expand(
                                    imgf.shape[0], imgf.shape[2]
                                )
                            )
                            while (dropout_mask.sum(1) == 0.0).any():
                                dropout_mask = torch.bernoulli(
                                    (1.0 - self.feature_dropout).expand(
                                        imgf.shape[0], imgf.shape[2]
                                    )
                                )
                            pooled_imgf = []
                            for img, dm in zip(imgf, dropout_mask):
                                idx = dm.nonzero().squeeze(1)
                                pimg = img.index_select(1, idx).mean(dim=1)
                                pooled_imgf.append(pimg[..., None, None])
                            pooled_img_features.append(torch.stack(pooled_imgf))
                    img_features = pooled_img_features
                else:
                    img_features = [
                        F.adaptive_avg_pool2d(feats, (1, 1)) for feats in img_features
                    ]
        else:
            if dims is not None:  # MILAN by_unit - average over the 15 image features
                new_dims = img_features.shape
                img_features = img_features.view(
                    dims[0], dims[1], new_dims[-2], new_dims[-1]
                )
                img_features = img_features.mean(dim=1)

        if len(forced_token_mask) > 0:
            forced_token_mask = torch.stack(forced_token_mask, dim=1)
        else:
            forced_token_mask = None

        return img_features, forced_token_mask

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        gt_captions: torch.LongTensor,
        mask: torch.LongTensor = None,
    ):
        img_features, forced_token_mask = self.forward_vision_model(
            pixel_values, mask=mask, just_cnn_features=False
        )
        encoded_features = self.translation(
            img_features, forced_token_mask=forced_token_mask
        )
        encoded_features = encoded_features[:, -self.nr_learnable_tokens :, :]

        inputs_embeds = self.embedding_weights(input_ids)

        inputs_embeds = torch.cat(
            (inputs_embeds[:, :1, :], encoded_features, inputs_embeds[:, 1:, :]), dim=1
        )

        out = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=gt_captions,
        )
        return out["loss"]

    @torch.no_grad()
    def generate(
        self,
        pixel_values=None,
        img_features=None,
        mask=None,
        forced_token_mask=None,
        max_length=None,
        layer=None,
        feat_index=None,
        pool_locs=None,
        kernel_size=3,
        stride=1,
    ):
        if pixel_values is None and img_features is None:
            raise ValueError("Please provide either pixel_values or img_features.")
        if self.vision_feat_func != "avg_pooling" and feat_index is not None:
            raise ValueError(
                "Index selection is only possible when vision_feat_func is avg_pooling"
            )
        if layer is None and feat_index is not None:
            raise ValueError(
                "Index selection is only possible when a layer is specified"
            )
        if mask is not None and feat_index is not None:
            raise NotImplementedError
        if pool_locs is not None and feat_index is None:
            raise ValueError(
                "pool_locs is only valid for a specific location, so feat_index cannot be None"
            )

        if img_features is None:
            just_cnn_features = False
            if feat_index is not None:
                just_cnn_features = True
            img_features, forced_token_mask = self.forward_vision_model(
                pixel_values, just_cnn_features=just_cnn_features, mask=mask
            )

        if isinstance(img_features, list):
            bs = img_features[0].shape[0]
        else:
            bs = img_features.shape[0]
        device = img_features[0].device

        if layer is not None:
            layer_id = -layer - 1

        if feat_index is not None:
            feats = img_features[layer_id]
            if pool_locs is not None:
                feats = F.avg_pool2d(
                    feats,
                    kernel_size=kernel_size,
                    padding=int((kernel_size - 1) / 2),
                    stride=stride,
                    count_include_pad=False,
                )
            feats = feats[:, :, feat_index[0], feat_index[1]]
            feats = feats[:, :, None, None]
            img_features = [
                torch.zeros((bs, f.shape[1], 1, 1), device=device) for f in img_features
            ]
            img_features[layer_id] = feats

        token_mask = None
        if layer is not None:
            token_mask = torch.ones(
                (bs, self.feat_seq_len), device=device
            )  # mask all layer features
            token_mask[:, layer_id] = torch.zeros(
                (bs), device=device
            )  # unmask corresponding layer features

            token_mask = torch.cat(
                (
                    token_mask,
                    torch.zeros((bs, self.nr_learnable_tokens), device=device),
                ),
                dim=1,
            )
            token_mask = token_mask.bool()

        encoded_features = self.translation(
            img_features, token_mask=token_mask, forced_token_mask=forced_token_mask
        )
        encoded_features = encoded_features[:, -self.nr_learnable_tokens :, :]

        input_ids = (
            torch.ones((bs, 1), dtype=torch.long, device=device)
            * self.lm_model.config.bos_token_id
        )

        generated = self.greedy(
            input_ids, visual_features=encoded_features, max_length=max_length
        )
        return generated
