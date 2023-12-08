# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numba
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchaudio

from torchlibrosa.stft import STFT, LogmelFilterBank
from timm.models.layers import to_2tuple, trunc_normal_

from vision_transformer import VisionTransformer as _VisionTransformer

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num):
        super().__init__()
        params = torch.ones(num, requires_grad=False)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

class VisionTransformer(_VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, cls_num=1, mask_2d=True, use_custom_patch=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        img_size = (1024, 128) # 1024, 128
        in_chans = 2
        emb_dim = 768

        self.doa_tokens = nn.Parameter(torch.zeros(1, 2, emb_dim))
        torch.nn.init.normal_(self.doa_tokens, std=.02)

        del self.cls_token
        self.cls_num = cls_num
        self.cls_tokens = nn.Parameter(torch.zeros(1, cls_num, emb_dim))
        torch.nn.init.normal_(self.cls_tokens, std=.02)

        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=in_chans, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=True
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, 
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )
        
        self.conv_proj = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 2), stride=(1, 4), padding=(0, 0))
        
        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.mel_norm = nn.BatchNorm2d(in_chans, affine=False)
        del self.norm  # remove the original norm

        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch
        self.target_frame = 1024

        self.dis_norm = kwargs['norm_layer'](emb_dim)
        self.doa_norm = kwargs['norm_layer'](emb_dim)
        self.fc_norm = kwargs['norm_layer'](emb_dim)

        self.distance_head = nn.Linear(emb_dim, 11)
        self.azimuth_head = nn.Linear(emb_dim, 360)
        self.elevation_head = nn.Linear(emb_dim, 180)

        trunc_normal_(self.head.weight, std=2e-5)
        trunc_normal_(self.distance_head.weight, std=2e-5)
        trunc_normal_(self.azimuth_head.weight, std=2e-5)
        trunc_normal_(self.elevation_head.weight, std=2e-5)


    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        N, L, D = x.shape  # batch, length, dim
        T, F = 64, 8
        
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #4,1,1024,128

        x = x + self.pos_embed[:, 1:, :]

        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)

        cls_tokens = self.cls_tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        doa_tokens = self.doa_tokens.expand(B, -1, -1)
        x = torch.cat((doa_tokens, cls_tokens, x), dim=1)   # bsz, 512 + 2 + 10, 768 
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        return x

    # overwrite original timm
    def forward(self, waveforms, reverbs, mask_t_prob=0.0, mask_f_prob=0.0):
        waveforms = torchaudio.functional.fftconvolve(waveforms, reverbs, mode='full')[..., :waveforms.shape[-1]]
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        # bsz* channels, 1024, 513
        real, imag = self.spectrogram_extractor(waveforms) 
        real = real.reshape(B, C, real.shape[-2], real.shape[-1])
        imag = imag.reshape(B, C, imag.shape[-2], imag.shape[-1])

        log_magnitude = torch.log10(torch.sqrt(real**2 + imag**2) + 1e-8)
        log_magnitude = self.mel_norm(log_magnitude) # TODO
        # log_mel = self.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)
        
        complex_x = torch.view_as_complex(torch.stack([real, imag], dim=-1))
        ILD = 20 * torch.log10(torch.abs(complex_x[:, 1, :, :]) / torch.abs(complex_x[:, 0, :, :]) + 1e-8)
        IPD = torch.atan2(
            torch.imag(complex_x[:, 1, :, :]) * torch.real(complex_x[:, 0, :, :]) - 
            torch.real(complex_x[:, 1, :, :]) * torch.imag(complex_x[:, 0, :, :]), 
            torch.real(complex_x[:, 1, :, :]) * torch.real(complex_x[:, 0, :, :]) + 
            torch.imag(complex_x[:, 1, :, :]) * torch.imag(complex_x[:, 0, :, :])
        )
        phase_feats = torch.stack([ILD, IPD], dim=1)

        x = torch.cat([log_magnitude, phase_feats], dim=1)

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True)
        
        if self.training:
            x = x.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        x = self.conv_proj(x)
        x = self.patch_embed(x)
        x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)

        dis_token = x[:, 0]
        doa_token = x[:, 1]
        cls_tokens = x[:, 2:2+self.cls_num].mean(dim=1)

        dis_token = self.dis_norm(dis_token)
        doa_token = self.doa_norm(doa_token)
        cls_tokens = self.fc_norm(cls_tokens)

        classifier = self.head(cls_tokens)
        distance = self.distance_head(dis_token)
        azimuth = self.azimuth_head(doa_token)
        elevation = self.elevation_head(doa_token)

        return classifier, distance, azimuth, elevation


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

