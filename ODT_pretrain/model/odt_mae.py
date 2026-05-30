import math
import os
import traceback
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from easydict import EasyDict as AttrDict
except Exception:
    class AttrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

try:
    from .transformer import QK_Norm_TransformerBlock
except Exception:
    class QK_Norm_TransformerBlock(nn.Module):
        def __init__(self, dim, head_dim, use_qk_norm=False):
            super().__init__()
            del use_qk_norm
            num_heads = max(dim // head_dim, 1)
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )

        def forward(self, x):
            y = self.norm1(x)
            y, _ = self.attn(y, y, y, need_weights=False)
            x = x + y
            return x + self.mlp(self.norm2(x))


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_model_cfg(config):
    return _cfg_get(config, "model", config)


def model_dim(config) -> int:
    model_cfg = get_model_cfg(config)
    transformer = _cfg_get(model_cfg, "transformer")
    return int(_cfg_get(model_cfg, "enc_emb_dim", _cfg_get(model_cfg, "embed_dim", _cfg_get(transformer, "d", 768))))


def _head_dim_from_nhead(dim: int, nhead):
    if nhead is None:
        return None
    nhead = int(nhead)
    if dim % nhead != 0:
        raise ValueError(f"embedding dim {dim} must be divisible by nhead {nhead}")
    head_dim = dim // nhead
    if head_dim % 8 != 0:
        raise ValueError(
            f"FlashAttention requires head_dim % 8 == 0, got dim={dim}, "
            f"nhead={nhead}, head_dim={head_dim}. Use a compatible nhead, "
            "for example nhead=16 when dim=512, or change the embedding dim."
        )
    return head_dim


def enc_head_dim(config) -> int:
    model_cfg = get_model_cfg(config)
    transformer = _cfg_get(model_cfg, "transformer")
    dim = model_dim(config)
    from_heads = _head_dim_from_nhead(dim, _cfg_get(model_cfg, "enc_nhead"))
    if from_heads is not None:
        return from_heads
    return int(_cfg_get(model_cfg, "d_head", _cfg_get(transformer, "d_head", 64)))


def dec_head_dim(config) -> int:
    model_cfg = get_model_cfg(config)
    transformer = _cfg_get(model_cfg, "transformer")
    dim = int(_cfg_get(model_cfg, "dec_emb_dim", _cfg_get(model_cfg, "dec_embed_dim", model_dim(config))))
    from_heads = _head_dim_from_nhead(dim, _cfg_get(model_cfg, "dec_nhead"))
    if from_heads is not None:
        return from_heads
    return int(_cfg_get(model_cfg, "dec_d_head", _cfg_get(model_cfg, "d_head", _cfg_get(transformer, "d_head", 64))))


def enc_depth(config) -> int:
    model_cfg = get_model_cfg(config)
    transformer = _cfg_get(model_cfg, "transformer")
    return int(_cfg_get(model_cfg, "enc_nlayer", _cfg_get(model_cfg, "enc_depth", _cfg_get(transformer, "enc_depth", _cfg_get(transformer, "n_layer", 12)))))


def dec_depth(config) -> int:
    model_cfg = get_model_cfg(config)
    transformer = _cfg_get(model_cfg, "transformer")
    return int(_cfg_get(model_cfg, "dec_nlayer", _cfg_get(model_cfg, "dec_depth", _cfg_get(transformer, "dec_depth", 8))))


def frame_count(config) -> int:
    return int(_cfg_get(get_model_cfg(config), "frame_count", 240))


def patch_size(config) -> int:
    model_cfg = get_model_cfg(config)
    tokenizer = _cfg_get(model_cfg, "image_tokenizer")
    return int(_cfg_get(model_cfg, "patch_size", _cfg_get(tokenizer, "patch_size", 16)))


def mask_ratio(config) -> float:
    return float(_cfg_get(get_model_cfg(config), "mask_ratio", 0.75))


def cls_token_count(config) -> int:
    return int(_cfg_get(get_model_cfg(config), "num_cls_tokens", 16))


def get_1d_sincos_pos_embed(length: int, dim: int, cls_token: Union[bool, int] = False) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    channel = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    emb = pos / (10000 ** (2 * torch.div(channel, 2, rounding_mode="floor") / dim))
    emb[:, 0::2] = torch.sin(emb[:, 0::2])
    emb[:, 1::2] = torch.cos(emb[:, 1::2])
    if cls_token:
        num_cls = 1 if isinstance(cls_token, bool) else int(cls_token)
        emb = torch.cat([torch.zeros(num_cls, dim), emb], dim=0)
    return emb


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)


def init_conv_as_linear(conv: nn.Conv2d):
    weight = conv.weight.data
    nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _match_parameter_grad_layout(param: nn.Parameter):
    def _hook(grad: torch.Tensor):
        if grad is None or grad.stride() == param.stride():
            return grad
        out = torch.empty_strided(param.size(), param.stride(), device=grad.device, dtype=grad.dtype)
        out.copy_(grad)
        return out

    return _hook


def random_masking(x: torch.Tensor, ratio: float):
    b, length, dim = x.shape
    len_keep = int(length * (1.0 - ratio))
    noise = torch.rand(b, length, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    mask = torch.ones([b, length], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class ODTMAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frame_count = frame_count(config)
        self.patch_size = patch_size(config)
        self.embed_dim = model_dim(config)
        self.in_dim = 4 * self.patch_size * self.patch_size
        self.mask_ratio = mask_ratio(config)
        self.num_cls_tokens = cls_token_count(config)

        self.patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=(1, self.in_dim), bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls_tokens, self.embed_dim))
        pos = get_1d_sincos_pos_embed(self.frame_count, self.embed_dim, cls_token=self.num_cls_tokens)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [
                QK_Norm_TransformerBlock(self.embed_dim, enc_head_dim(config), use_qk_norm=True)
                for _ in range(enc_depth(config))
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        init_conv_as_linear(self.patch_embed)
        self.apply(init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        self.cls_token.register_hook(_match_parameter_grad_layout(self.cls_token))

    def _embed(self, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = image.shape
        if t != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} frames, got {t}")
        if (c, h, w) != (2, self.patch_size, self.patch_size):
            raise ValueError(f"Expected image [B,T,2,{self.patch_size},{self.patch_size}], got {image.shape}")
        x = torch.cat([image, background], dim=2).reshape(b, t, -1)
        return self.patch_embed(x[:, None]).flatten(2).transpose(1, 2)

    def forward(self, image: torch.Tensor, background: torch.Tensor, ratio: Optional[float] = None):
        ratio = self.mask_ratio if ratio is None else float(ratio)
        x = self._embed(image, background)
        x = x + self.pos_embed[:, self.num_cls_tokens :, :].to(dtype=x.dtype, device=x.device)
        x, mask, ids_restore = random_masking(x, ratio)

        cls = self.cls_token.to(dtype=x.dtype, device=x.device) + self.pos_embed[:, : self.num_cls_tokens, :].to(
            dtype=x.dtype, device=x.device
        )
        cls = cls.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore

    def encode_full(self, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        x = self._embed(image, background)
        x = x + self.pos_embed[:, self.num_cls_tokens :, :].to(dtype=x.dtype, device=x.device)
        cls = self.cls_token.to(dtype=x.dtype, device=x.device) + self.pos_embed[:, : self.num_cls_tokens, :].to(
            dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls.repeat(x.shape[0], 1, 1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ODTMAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frame_count = frame_count(config)
        self.patch_size = patch_size(config)
        self.enc_dim = model_dim(config)
        model_cfg = get_model_cfg(config)
        self.dec_dim = int(_cfg_get(model_cfg, "dec_emb_dim", _cfg_get(model_cfg, "dec_embed_dim", self.enc_dim)))
        self.num_cls_tokens = cls_token_count(config)
        self.enc_to_dec = nn.Conv2d(1, self.dec_dim, kernel_size=(1, self.enc_dim), bias=True)
        self.background_embed = nn.Conv2d(
            1,
            self.dec_dim,
            kernel_size=(1, 2 * self.patch_size * self.patch_size),
            bias=True,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        pos = get_1d_sincos_pos_embed(self.frame_count, self.dec_dim, cls_token=self.num_cls_tokens)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [
                QK_Norm_TransformerBlock(self.dec_dim, dec_head_dim(config), use_qk_norm=True)
                for _ in range(dec_depth(config))
            ]
        )
        self.norm = nn.LayerNorm(self.dec_dim)
        self.pred = nn.Linear(self.dec_dim, 2 * self.patch_size * self.patch_size, bias=True)
        init_conv_as_linear(self.enc_to_dec)
        init_conv_as_linear(self.background_embed)
        self.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)
        self.mask_token.register_hook(_match_parameter_grad_layout(self.mask_token))

    def forward(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        background: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.enc_to_dec(latent[:, None]).flatten(2).transpose(1, 2)
        bg_flat = background.reshape(background.shape[0], background.shape[1], -1)
        bg_tokens = self.background_embed(bg_flat[:, None]).flatten(2).transpose(1, 2)

        num_mask = ids_restore.shape[1] + self.num_cls_tokens - x.shape[1]
        mask_tokens = self.mask_token.to(dtype=x.dtype).repeat(x.shape[0], num_mask, 1)
        x_ = torch.cat([x[:, self.num_cls_tokens :, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x_ = x_ + bg_tokens.to(dtype=x_.dtype) * mask.unsqueeze(-1).to(dtype=x_.dtype)
        x = torch.cat([x[:, : self.num_cls_tokens, :], x_], dim=1)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pred(x[:, self.num_cls_tokens :, :])
        return x.reshape(x.shape[0], self.frame_count, 2, self.patch_size, self.patch_size)


class ODTMaskedAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ODTMAEEncoder(config)
        self.decoder = ODTMAEDecoder(config)
        self.default_mask_ratio = mask_ratio(config)

    def forward(self, batch, mask_ratio_override: Optional[float] = None, **kwargs):
        del kwargs
        image = batch["image"]
        background = batch["background"]
        latent, mask, ids_restore = self.encoder(image, background, ratio=mask_ratio_override)
        pred = self.decoder(latent, ids_restore, background, mask)
        masked_mse, full_mse = self.forward_loss(image, pred, mask)
        return AttrDict(
            input=AttrDict(batch),
            loss_metrics=AttrDict(
                loss=full_mse,
                mae_loss=full_mse,
                mae_full_mse=full_mse,
                mae_masked_mse=masked_mse,
            ),
            pred_image=pred,
            mask=mask,
            ids_restore=ids_restore,
        )

    @staticmethod
    def forward_loss(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_mse = ((pred - target) ** 2).flatten(2).mean(dim=-1)
        denom = mask.sum().clamp_min(1.0)
        masked_mse = (token_mse * mask).sum() / denom
        full_mse = token_mse.mean()
        return masked_mse, full_mse

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
        return ckpt_path, status


def _strip_module_prefix(state_dict):
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _latest_checkpoint(load_path: str) -> Optional[str]:
    if os.path.isdir(load_path):
        names = sorted(
            name for name in os.listdir(load_path) if name.endswith(".pt") or name.endswith(".pth")
        )
        if not names:
            return None
        return os.path.join(load_path, names[-1])
    if os.path.isfile(load_path):
        return load_path
    return None


def extract_encoder_state(checkpoint):
    if isinstance(checkpoint, dict) and "encoder" in checkpoint and isinstance(checkpoint["encoder"], dict):
        return _strip_module_prefix(checkpoint["encoder"])
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    out = {}
    for key, value in _strip_module_prefix(state).items():
        if key.startswith("encoder."):
            out[key[len("encoder.") :]] = value
    if out:
        return out
    return _strip_module_prefix(state)


def load_encoder_weights(encoder: nn.Module, load_path: str):
    ckpt_path = _latest_checkpoint(load_path)
    if ckpt_path is None:
        raise FileNotFoundError(f"No encoder checkpoint found at {load_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = extract_encoder_state(checkpoint)
    status = encoder.load_state_dict(state, strict=False)
    return ckpt_path, status
