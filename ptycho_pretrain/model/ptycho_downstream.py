import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from easydict import EasyDict as AttrDict
except Exception:
    class AttrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

from .ptycho_mae import (
    PtychoMAEEncoder,
    _cfg_get,
    _latest_checkpoint,
    _model_int,
    _strip_module_prefix,
    _torch_load_cpu,
    get_model_cfg,
    init_weights,
    load_encoder_weights,
)


class PtychoPPDecoder(nn.Module):
    """Decode the center-token latent into a projected potential patch.

    latent[B, embed_dim] -> Linear -> [B, base_ch, 4, 4]
    -> ConvTranspose2d stack (x2 each) -> Conv2d -> [B, 1, pp_patch_size, pp_patch_size]
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = get_model_cfg(config)
        self.embed_dim = _model_int(config, "embed_dim", 768)
        self.pp_patch_size = _model_int(config, "pp_patch_size", 16)
        self.base_grid = int(_cfg_get(model_cfg, "dec_base_grid", 4))
        channels = _cfg_get(model_cfg, "dec_channels", [512, 256, 128])
        channels = [int(c) for c in channels]
        if not channels:
            raise ValueError("model.dec_channels must contain at least one channel")

        if self.pp_patch_size % self.base_grid != 0:
            raise ValueError(
                f"pp_patch_size={self.pp_patch_size} must be divisible by dec_base_grid={self.base_grid}"
            )
        num_upsample = 0
        size = self.base_grid
        while size < self.pp_patch_size:
            size *= 2
            num_upsample += 1
        if size != self.pp_patch_size:
            raise ValueError(
                f"pp_patch_size={self.pp_patch_size} must be base_grid({self.base_grid}) scaled by powers of 2"
            )
        if len(channels) < num_upsample:
            raise ValueError(
                f"model.dec_channels needs >= {num_upsample} entries for "
                f"{self.base_grid}->{self.pp_patch_size} upsampling, got {len(channels)}"
            )

        self.base_channels = channels[0]
        self.proj = nn.Linear(self.embed_dim, self.base_channels * self.base_grid * self.base_grid, bias=True)

        layers: List[nn.Module] = []
        in_ch = channels[0]
        for idx in range(num_upsample):
            out_ch = channels[idx + 1] if idx + 1 < len(channels) else channels[-1]
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.upsample = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=True)
        self.apply(init_weights)

    def forward(self, latent_center: torch.Tensor) -> torch.Tensor:
        bsz = latent_center.shape[0]
        x = self.proj(latent_center)
        x = x.reshape(bsz, self.base_channels, self.base_grid, self.base_grid)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x.reshape(bsz, 1, self.pp_patch_size, self.pp_patch_size)


class PtychoProjectedPotentialModel(nn.Module):
    """Downstream projected-potential prediction from the full 11-token sequence.

    Uses the pretrained PtychoMAEEncoder (optionally frozen). The center
    diffraction token (idx=4) latent is decoded into the local projected
    potential patch.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = get_model_cfg(config)
        self.encoder = PtychoMAEEncoder(config)
        self.center_token_id = _model_int(config, "center_token_id", self.encoder.num_dp_tokens // 2)

        enc_ckpt = str(_cfg_get(model_cfg, "encoder_ckpt", "") or "").strip()
        if enc_ckpt:
            path, status = load_encoder_weights(self.encoder, enc_ckpt)
            print(f"Loaded Ptycho MAE encoder from {path}; status={status}")

        self.freeze_encoder = bool(_cfg_get(model_cfg, "freeze_encoder", True))
        if self.freeze_encoder:
            self.set_encoder_trainable(False)

        self.decoder = PtychoPPDecoder(config)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def set_encoder_trainable(self, trainable: bool):
        for param in self.encoder.parameters():
            param.requires_grad = bool(trainable)
        if not trainable:
            self.encoder.eval()

    def forward(self, batch, **kwargs):
        del kwargs
        inputs = batch["inputs"]
        target = batch["target"]

        if self.freeze_encoder:
            with torch.no_grad():
                latent = self.encoder.encode_full(inputs)
        else:
            latent = self.encoder.encode_full(inputs)

        latent_center = latent[:, self.center_token_id]
        pred = self.decoder(latent_center)
        loss = F.mse_loss(pred, target)
        return AttrDict(
            input=AttrDict(batch),
            latent=latent,
            pred_pp=pred,
            loss_metrics=AttrDict(loss=loss, pp_mse=loss),
        )

    @torch.no_grad()
    def load_ckpt(self, load_path: str):
        ckpt_path = _latest_checkpoint(load_path, prefix="ckpt_")
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found under {load_path}")
        checkpoint = _torch_load_cpu(ckpt_path)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        status = self.load_state_dict(_strip_module_prefix(state), strict=False)
        if self.freeze_encoder:
            self.set_encoder_trainable(False)
        return ckpt_path, status
