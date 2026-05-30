import math
import os
import traceback
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
except Exception:
    xops = None

try:
    from easydict import EasyDict as AttrDict
except Exception:
    class AttrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

from .latent_to_voxel_decoder import LatentToVoxelDecoder
from .odt_mae import ODTMAEEncoder, _latest_checkpoint, _strip_module_prefix, load_encoder_weights, model_dim


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _shape_tuple(value, default):
    if value is None:
        value = default
    return tuple(int(x) for x in value)


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D sin-cos dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=1)


class VoxelPosTokenizer(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(self.in_channels * (self.patch_size**3), int(d_model), bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> [B, (D/P)*(H/P)*(W/P), C*P^3]
        b, c, d, h, w = x.shape
        p = self.patch_size
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} voxel pos channels, got {c}")
        if d % p != 0 or h % p != 0 or w % p != 0:
            raise ValueError(f"voxel position grid {(d, h, w)} must be divisible by patch_size={p}")
        x = x.reshape(b, c, d // p, p, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return self.proj(x.reshape(b, (d // p) * (h // p) * (w // p), c * (p**3)))


class CrossAttention3DGridDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        self.enc_dim = int(_cfg_get(model_cfg, "encoder_dim", model_dim(config)))
        self.dec_dim = int(_cfg_get(model_cfg, "dec_emb_dim", _cfg_get(model_cfg, "dec_embed_dim", self.enc_dim)))
        self.grid_shape = _shape_tuple(_cfg_get(model_cfg, "voxel_grid", None), (16, 32, 32))
        self.voxel_size = _shape_tuple(_cfg_get(model_cfg, "voxel_size", None), (128, 256, 256))
        self.num_queries = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.num_heads = int(_cfg_get(model_cfg, "cross_attn_heads", _cfg_get(model_cfg, "dec_nhead", 8)))
        if self.dec_dim % self.num_heads != 0:
            raise ValueError(f"dec_embed_dim={self.dec_dim} must be divisible by cross_attn_heads={self.num_heads}")
        self.head_dim = self.dec_dim // self.num_heads
        if self.head_dim % 8 != 0:
            raise ValueError(
                f"FlashAttention requires cross-attn head_dim % 8 == 0, got dec_embed_dim={self.dec_dim}, "
                f"cross_attn_heads={self.num_heads}, head_dim={self.head_dim}. Use cross_attn_heads=16 "
                "when dec_embed_dim=512, or change dec_embed_dim."
            )
        self.query_chunk_size = int(_cfg_get(model_cfg, "cross_attn_query_chunk_size", 256))
        self.attn_dropout = float(_cfg_get(model_cfg, "cross_attn_dropout", 0.0))

        self.memory_proj = nn.Linear(self.enc_dim, self.dec_dim) if self.enc_dim != self.dec_dim else nn.Identity()
        self.memory_norm = nn.LayerNorm(self.dec_dim)
        voxel_pos_cfg = _cfg_get(model_cfg, "voxel_pos_tokenizer")
        self.voxel_pos_patch_size = int(_cfg_get(voxel_pos_cfg, "patch_size", _cfg_get(model_cfg, "voxel_pos_patch_size", 8)))
        self.voxel_pos_in_channels = int(
            _cfg_get(voxel_pos_cfg, "in_channels", _cfg_get(model_cfg, "voxel_pos_in_channels", 27))
        )
        token_grid = tuple(size // self.voxel_pos_patch_size for size in self.voxel_size)
        if any(size % self.voxel_pos_patch_size != 0 for size in self.voxel_size):
            raise ValueError(
                f"voxel_size={self.voxel_size} must be divisible by "
                f"voxel_pos_tokenizer.patch_size={self.voxel_pos_patch_size}"
            )
        if token_grid != self.grid_shape:
            raise ValueError(
                f"voxel_grid={self.grid_shape} must match voxel_size/voxel_pos_tokenizer.patch_size={token_grid}"
            )
        self.voxel_pos_tokenizer = VoxelPosTokenizer(
            in_channels=self.voxel_pos_in_channels,
            patch_size=self.voxel_pos_patch_size,
            d_model=self.dec_dim,
        )
        self.query_norm = nn.LayerNorm(self.dec_dim)

        self.q_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.k_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.v_proj = nn.Linear(self.dec_dim, self.dec_dim)
        self.out_proj = nn.Linear(self.dec_dim, self.dec_dim)

        channels = _cfg_get(model_cfg, "voxel_decoder_channels", [384, 192, 48])
        self.voxel_decoder = LatentToVoxelDecoder(
            latent_dim=self.dec_dim,
            latent_grid_shape=self.grid_shape,
            output_shape=self.voxel_size,
            channels=channels,
            head_channels=int(_cfg_get(model_cfg, "voxel_decoder_head_channels", 12)),
            output_activation=_cfg_get(model_cfg, "voxel_output_activation", "sigmoid"),
        )

    def nerf_positional_encoding(self, pos: torch.Tensor, num_freqs: int = 4, include_input: bool = True) -> torch.Tensor:
        # pos: [B, 3, D, H, W]; output channels = 3 + 3*2*num_freqs when include_input=True.
        if pos.dim() != 5 or pos.shape[1] != 3:
            raise ValueError(f"pos should be [B,3,D,H,W], got {pos.shape}")
        b, c, d, h, w = pos.shape
        freq_bands = (2.0 ** torch.arange(num_freqs, device=pos.device, dtype=pos.dtype)) * math.pi
        angles = pos.unsqueeze(2) * freq_bands.view(1, 1, num_freqs, 1, 1, 1)
        sin_enc = torch.sin(angles).reshape(b, c * num_freqs, d, h, w)
        cos_enc = torch.cos(angles).reshape(b, c * num_freqs, d, h, w)
        pe = torch.cat([sin_enc, cos_enc], dim=1)
        if include_input:
            pe = torch.cat([pos, pe], dim=1)
        return pe

    def voxel_position_encoding(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        d, h, w = self.voxel_size
        coords_w = torch.linspace(-0.5, 0.5, steps=w, device=device, dtype=dtype)
        coords_h = torch.linspace(-0.5, 0.5, steps=h, device=device, dtype=dtype)
        coords_d = torch.linspace(-0.5, 0.5, steps=d, device=device, dtype=dtype)
        grid_z, grid_y, grid_x = torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        pos = torch.stack([grid_x, grid_y, grid_z], dim=0)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        pos_pe = self.nerf_positional_encoding(pos, num_freqs=4, include_input=True)
        if pos_pe.shape[1] != self.voxel_pos_in_channels:
            raise ValueError(
                f"voxel_position_encoding produced {pos_pe.shape[1]} channels, "
                f"expected {self.voxel_pos_in_channels}"
            )
        return pos_pe

    def _attention_chunk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q/k/v: [B, L, H, Dh]
        if xops is not None and q.is_cuda:
            return xops.memory_efficient_attention(
                q,
                k,
                v,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        # if hasattr(F, "scaled_dot_product_attention"):
        #     q2 = q.transpose(1, 2)
        #     k2 = k.transpose(1, 2)
        #     v2 = v.transpose(1, 2)
        #     out = F.scaled_dot_product_attention(
        #         q2,
        #         k2,
        #         v2,
        #         dropout_p=self.attn_dropout if self.training else 0.0,
        #     )
        #     return out.transpose(1, 2)

        scale = self.head_dim ** -0.5
        q2 = q.permute(0, 2, 1, 3)
        k2 = k.permute(0, 2, 3, 1)
        v2 = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q2, k2) * scale
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v2)
        return out.permute(0, 2, 1, 3)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        b = memory.shape[0]
        memory = self.memory_norm(self.memory_proj(memory))
        voxel_pos_cond = self.voxel_position_encoding(b, memory.device, memory.dtype)
        queries = self.query_norm(self.voxel_pos_tokenizer(voxel_pos_cond))

        k = self.k_proj(memory).view(b, -1, self.num_heads, self.head_dim)
        v = self.v_proj(memory).view(b, -1, self.num_heads, self.head_dim)
        chunks = []
        for start in range(0, self.num_queries, self.query_chunk_size):
            q_chunk = queries[:, start : start + self.query_chunk_size, :]
            q = self.q_proj(q_chunk).view(b, -1, self.num_heads, self.head_dim)
            out = self._attention_chunk(q, k, v)
            chunks.append(out.reshape(b, -1, self.dec_dim))
        grid_tokens = self.out_proj(torch.cat(chunks, dim=1))
        voxel = self.voxel_decoder(grid_tokens).squeeze(1)
        return voxel


class VoxelLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        training = config.training
        self.l2_weight = float(_cfg_get(training, "l2_loss_weight", 1.0))
        self.l1_weight = float(_cfg_get(training, "l1_loss_weight", 0.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l2 = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target) if self.l1_weight > 0 else torch.zeros((), device=pred.device, dtype=pred.dtype)
        loss = self.l2_weight * l2 + self.l1_weight * l1
        return AttrDict(loss=loss, l2_loss=l2, l1_loss=l1)


class ODTDirectInversion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ODTMAEEncoder(config)
        model_cfg = config.model
        enc_ckpt = str(_cfg_get(model_cfg, "encoder_ckpt", "") or "").strip()
        if enc_ckpt:
            path, status = load_encoder_weights(self.encoder, enc_ckpt)
            print(f"Loaded ODT MAE encoder from {path}; status={status}")

        self.freeze_encoder = bool(_cfg_get(model_cfg, "freeze_encoder", True))
        self.memory_mode = str(_cfg_get(model_cfg, "memory_mode", "all_frames")).lower()
        if self.memory_mode not in ("all_frames", "cls"):
            raise ValueError(f"model.memory_mode must be 'all_frames' or 'cls', got {self.memory_mode!r}")

        self.patch_size = int(_cfg_get(model_cfg, "patch_size", 16))
        self.image_size = int(_cfg_get(model_cfg, "image_size", 256))
        self.spatial_grid = self.image_size // self.patch_size
        self.num_spatial_patches = self.spatial_grid * self.spatial_grid
        self.num_cls_tokens = int(getattr(self.encoder, "num_cls_tokens", 1))
        pos2d = _build_2d_sincos_pos_embed(self.spatial_grid, self.spatial_grid, self.encoder.embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial_patches, self.encoder.embed_dim))
        self.spatial_pos_alpha = nn.Parameter(torch.ones(()))
        self.register_buffer("spatial_sincos_pos_embed", pos2d.unsqueeze(0), persistent=False)

        if self.freeze_encoder:
            self.set_encoder_trainable(False)
        self.decoder = CrossAttention3DGridDecoder(config)
        self.loss_computer = VoxelLoss(config)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def set_encoder_trainable(self, trainable: bool):
        for param in self.encoder.parameters():
            param.requires_grad = bool(trainable)
        if not trainable:
            self.encoder.eval()

    def _patchify_frames(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        p = self.patch_size
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(f"Expected full frames {self.image_size}x{self.image_size}, got {(h, w)}")
        g = self.spatial_grid
        x = x.reshape(b, t, c, g, p, g, p)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        return x.reshape(b * g * g, t, c, p, p)

    def _encode_memory(self, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        img_patches = self._patchify_frames(image)
        bg_patches = self._patchify_frames(background)

        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.encoder.encode_full(img_patches, bg_patches)
        else:
            encoded = self.encoder.encode_full(img_patches, bg_patches)

        encoded = encoded.reshape(b, self.num_spatial_patches, encoded.shape[1], encoded.shape[2])
        spatial_pos = self.spatial_pos_embed.to(dtype=encoded.dtype, device=encoded.device)
        spatial_sincos = self.spatial_sincos_pos_embed.to(dtype=encoded.dtype, device=encoded.device)
        spatial_alpha = self.spatial_pos_alpha.to(dtype=encoded.dtype, device=encoded.device)
        spatial_pos = spatial_pos + spatial_alpha * spatial_sincos
        if self.memory_mode == "cls":
            cls_tokens = encoded[:, :, : self.num_cls_tokens, :] + spatial_pos.unsqueeze(2)
            return cls_tokens.reshape(
                b,
                self.num_spatial_patches * self.num_cls_tokens,
                encoded.shape[-1],
            )

        frame_tokens = encoded[:, :, self.num_cls_tokens :, :] + spatial_pos.unsqueeze(2)
        return frame_tokens.reshape(
            b,
            self.num_spatial_patches * (encoded.shape[2] - self.num_cls_tokens),
            encoded.shape[-1],
        )

    def forward(self, data_batch, **kwargs):
        del kwargs
        image = data_batch["image"]
        background = data_batch["background"]
        target = data_batch["voxel"]
        memory = self._encode_memory(image, background)
        pred = self.decoder(memory)
        loss_metrics = self.loss_computer(pred, target)
        return AttrDict(
            input=AttrDict(data_batch),
            memory=memory,
            predicted_voxel=pred,
            loss_metrics=loss_metrics,
        )

    @torch.no_grad()
    def load_ckpt(self, load_path: str):
        ckpt_path = _latest_checkpoint(load_path)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found under {load_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            traceback.print_exc()
            raise
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        status = self.load_state_dict(_strip_module_prefix(state), strict=False)
        if self.freeze_encoder:
            self.set_encoder_trainable(False)
        return ckpt_path, status
