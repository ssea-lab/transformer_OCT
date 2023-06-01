# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
# from visualizer import get_local
import torch
import torch.nn as nn
import math
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_no_pooling_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class VisionTransformerEnsemblePostCrossAttention(nn.Module):
    """
    the module is the ensemble model for mae vit fine tune
    """

    def __init__(self, image_size_list, **kwargs):
        super().__init__()
        self.vision_transformer1 = VisionTransformer(img_size=image_size_list[0], **kwargs)
        self.vision_transformer2 = VisionTransformer(img_size=image_size_list[1], **kwargs)
        self.vision_transformer3 = VisionTransformer(img_size=image_size_list[2], **kwargs)
        num_feature = self.vision_transformer1.num_features
        self.wq1 = nn.Linear(num_feature, num_feature, bias=False)
        self.wq2 = nn.Linear(num_feature, num_feature, bias=False)
        self.wq3 = nn.Linear(num_feature, num_feature, bias=False)
        self.wq2_1 = nn.Linear(num_feature, num_feature, bias=False)
        self.wq2_2 = nn.Linear(num_feature, num_feature, bias=False)
        self.wq2_3 = nn.Linear(num_feature, num_feature, bias=False)
        self.attn_drop = nn.Dropout(p=0.1)
        self.loss_weight = nn.Parameter(torch.ones(3))

    def forward_cross_attention(self, v1_feature, v2_feature, v3_feature):
        B, N1, C = v1_feature.shape
        N2 = v2_feature.shape[1]
        N3 = v3_feature.shape[1]
        num_heads = 8
        head_dim = C // num_heads
        scale = head_dim ** -0.5
        v1_feature = v1_feature.reshape(B, N1, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N1, head_dim]
        v2_feature = v2_feature.reshape(B, N2, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N2, head_dim]
        v3_feature = v3_feature.reshape(B, N3, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N3, head_dim]
        # v1, v1 self attention
        v1_v1_attn = (v1_feature @ v1_feature.transpose(-2, -1)) * scale
        v1_v1_attn = v1_v1_attn.softmax(dim=-1)
        v1_v1_attn = self.attn_drop(v1_v1_attn)  # [B, num_head, N1, N2]
        v1_v1_attn_visualize = v1_v1_attn
        v1_v1_attn_feature = (v1_v1_attn @ v1_feature).transpose(1, 2).reshape(B, N1,
                                                                               C)  # v1, v2 cross attention
        # v1, v2 cross attention
        v1_v2_attn = (v1_feature @ v2_feature[:, :, 1:, :].transpose(-2, -1)) * scale
        v1_v2_attn = v1_v2_attn.softmax(dim=-1)
        v1_v2_attn = self.attn_drop(v1_v2_attn)
        v1_v2_attn_visualize = v1_v2_attn
        v1_v2_attn_feature = (v1_v2_attn @ v2_feature[:, :, 1:, :]).transpose(1, 2).reshape(B, N1,
                                                                                            C)  # v1, v2 cross attention
        # v1, v3 cross attention
        v1_v3_attn = (v1_feature @ v3_feature[:, :, 1:, :].transpose(-2, -1)) * scale
        v1_v3_attn = v1_v3_attn.softmax(dim=-1)
        v1_v3_attn = self.attn_drop(v1_v3_attn)
        v1_v3_attn_visualize = v1_v3_attn
        v1_v3_attn_feature = (v1_v3_attn @ v3_feature[:, :, 1:, :]).transpose(1, 2).reshape(B, N1,
                                                                                            C)  # v1, v2 cross attention
        fused_feature = v1_v1_attn_feature + v1_v2_attn_feature + v1_v3_attn_feature
        return fused_feature

    def forward(self, x):
        # the input x is list of tensor
        v1_feature = self.vision_transformer1.forward_no_pooling_features(x[0])  # [batch_size, dim]
        v2_feature = self.vision_transformer2.forward_no_pooling_features(x[1])
        v3_feature = self.vision_transformer3.forward_no_pooling_features(x[2])
        v1_feature = self.wq1(v1_feature)
        v2_feature = self.wq2(v2_feature)
        v3_feature = self.wq3(v3_feature)
        v1_fused_feature = self.forward_cross_attention(v1_feature, v2_feature, v3_feature)
        v2_fused_feature = self.forward_cross_attention(v2_feature, v1_feature, v3_feature)
        v3_fused_feature = self.forward_cross_attention(v3_feature, v1_feature, v2_feature)
        v2_1_fused_feature = self.wq2_1(v1_fused_feature)
        v2_2_fused_feature = self.wq2_2(v2_fused_feature)
        v2_3_fused_feature = self.wq2_3(v3_fused_feature)
        v1_fused_feature = self.forward_cross_attention(v2_1_fused_feature, v2_2_fused_feature, v2_3_fused_feature)
        v2_fused_feature = self.forward_cross_attention(v2_2_fused_feature, v2_1_fused_feature, v2_3_fused_feature)
        v3_fused_feature = self.forward_cross_attention(v2_3_fused_feature, v2_1_fused_feature, v2_2_fused_feature)

        v1_fused_cls_feature = v1_fused_feature[:, 0]
        v2_fused_cls_feature = v2_fused_feature[:, 0]
        v3_fused_cls_feature = v3_fused_feature[:, 0]
        v1_prob = self.vision_transformer1.head(v1_fused_cls_feature)
        v2_prob = self.vision_transformer2.head(v2_fused_cls_feature)
        v3_prob = self.vision_transformer3.head(v3_fused_cls_feature)
        final_class_prob = [v1_prob, v2_prob, v3_prob]

        return final_class_prob


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def vit_base_patch16_ensemble_post_cross_attention(**kwargs):
    model = VisionTransformerEnsemblePostCrossAttention(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                                        mlp_ratio=4, qkv_bias=True,
                                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
