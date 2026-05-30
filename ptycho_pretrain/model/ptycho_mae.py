import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from easydict import EasyDict as AttrDict
except Exception:
    class AttrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_model_cfg(config):
    return _cfg_get(config, "model", config)


def _model_int(config, key: str, default: int) -> int:
    return int(_cfg_get(get_model_cfg(config), key, default))


def _model_float(config, key: str, default: float) -> float:
    return float(_cfg_get(get_model_cfg(config), key, default))


def get_1d_sincos_pos_embed(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    half = max(dim // 2, 1)
    omega = torch.arange(half, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / half))
    out = pos * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def init_weights(module, std: float = 0.02):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.weight is not None:
            nn.init.ones_(module.weight)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = float(attn_dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.proj_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            head_dim=head_dim,
            attn_dropout=dropout,
            proj_dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PtychoMAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = get_model_cfg(config)
        self.dp_size = _model_int(config, "dp_size", 64)
        self.scan_patch_size = _model_int(config, "scan_patch_size", 3)
        self.num_dp_tokens = self.scan_patch_size * self.scan_patch_size
        self.use_probe_tokens = bool(_cfg_get(model_cfg, "use_probe_tokens", True))
        self.num_tokens = self.num_dp_tokens + (2 if self.use_probe_tokens else 0)
        self.center_token_id = _model_int(config, "center_token_id", self.num_dp_tokens // 2)
        self.embed_dim = _model_int(config, "embed_dim", 768)
        self.head_dim = _model_int(config, "d_head", 64)
        self.depth = _model_int(config, "enc_depth", 12)
        self.mlp_ratio = _model_float(config, "mlp_ratio", 4.0)
        self.dropout = _model_float(config, "dropout", 0.0)
        self.use_qk_norm = bool(_cfg_get(model_cfg, "use_qk_norm", True))
        self.in_dim = self.dp_size * self.dp_size

        self.token_embed = nn.Linear(self.in_dim, self.embed_dim, bias=True)
        pos = get_1d_sincos_pos_embed(self.num_tokens, self.embed_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.embed_dim,
                    self.head_dim,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    use_qk_norm=self.use_qk_norm,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(init_weights)

    def _embed_images(self, images: torch.Tensor) -> torch.Tensor:
        bsz, tokens, height, width = images.shape
        if (height, width) != (self.dp_size, self.dp_size):
            raise ValueError(f"Expected image tokens {self.dp_size}x{self.dp_size}, got {(height, width)}")
        return self.token_embed(images.reshape(bsz, tokens, -1))

    def _pos_for_ids(self, token_ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos = self.pos_embed.to(dtype=dtype, device=device)
        if token_ids.ndim == 1:
            return pos[:, token_ids.to(device), :]
        return pos.squeeze(0)[token_ids.to(device)]

    def forward(self, visible_images: torch.Tensor, visible_token_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed_images(visible_images)
        x = x + self._pos_for_ids(visible_token_ids, x.dtype, x.device)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def encode_full(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected full token images [B,{self.num_tokens},H,W], got {inputs.shape}")
        if inputs.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {inputs.shape[1]}")
        x = self._embed_images(inputs)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class PtychoMAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = get_model_cfg(config)
        self.dp_size = _model_int(config, "dp_size", 64)
        self.scan_patch_size = _model_int(config, "scan_patch_size", 3)
        self.num_dp_tokens = self.scan_patch_size * self.scan_patch_size
        self.use_probe_tokens = bool(_cfg_get(model_cfg, "use_probe_tokens", True))
        self.num_tokens = self.num_dp_tokens + (2 if self.use_probe_tokens else 0)
        self.center_token_id = _model_int(config, "center_token_id", self.num_dp_tokens // 2)
        self.enc_dim = _model_int(config, "embed_dim", 768)
        self.dec_dim = _model_int(config, "dec_embed_dim", 512)
        self.head_dim = _model_int(config, "d_head", 64)
        self.depth = _model_int(config, "dec_depth", 8)
        self.mlp_ratio = _model_float(config, "mlp_ratio", 4.0)
        self.dropout = _model_float(config, "dropout", 0.0)
        self.use_qk_norm = bool(_cfg_get(model_cfg, "use_qk_norm", True))

        self.enc_to_dec = nn.Linear(self.enc_dim, self.dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        pos = get_1d_sincos_pos_embed(self.num_tokens, self.dec_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.dec_dim,
                    self.head_dim,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    use_qk_norm=self.use_qk_norm,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.dec_dim)
        self.pred = nn.Linear(self.dec_dim, self.dp_size * self.dp_size, bias=True)
        self.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        latent_visible: torch.Tensor,
        visible_token_ids: torch.Tensor,
        target_token_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = latent_visible.shape[0]
        x_visible = self.enc_to_dec(latent_visible)
        if visible_token_ids.ndim == 1:
            visible_token_ids = visible_token_ids.unsqueeze(0).expand(bsz, -1)
        visible_token_ids = visible_token_ids.to(device=x_visible.device)

        x = self.mask_token.to(dtype=x_visible.dtype).expand(bsz, self.num_tokens, -1).clone()
        scatter_index = visible_token_ids.unsqueeze(-1).expand(-1, -1, self.dec_dim)
        x.scatter_(dim=1, index=scatter_index, src=x_visible)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if target_token_id is None:
            target_token_id = torch.full((bsz,), self.center_token_id, device=x.device, dtype=torch.long)
        else:
            target_token_id = target_token_id.to(device=x.device, dtype=torch.long).reshape(bsz)
        gather_index = target_token_id.reshape(bsz, 1, 1).expand(-1, 1, self.dec_dim)
        target_tokens = torch.gather(x, dim=1, index=gather_index).squeeze(1)
        pred = self.pred(target_tokens)
        return pred.reshape(bsz, 1, self.dp_size, self.dp_size)


class PtychoCenterMaskedAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = PtychoMAEEncoder(config)
        self.decoder = PtychoMAEDecoder(config)

    def forward(self, batch, **kwargs):
        del kwargs
        visible_images = batch["visible_images"]
        visible_token_ids = batch["visible_token_ids"]
        target = batch["target_image"]
        target_token_id = batch.get("target_token_id")
        latent = self.encoder(visible_images, visible_token_ids)
        pred = self.decoder(latent, visible_token_ids, target_token_id)
        loss = F.mse_loss(pred, target)
        return AttrDict(
            input=AttrDict(batch),
            latent_visible=latent,
            pred_center=pred,
            mask=batch.get("mask"),
            loss_metrics=AttrDict(loss=loss, mae_center_mse=loss),
        )

    @torch.no_grad()
    def load_ckpt(self, load_path: str):
        ckpt_path = _latest_checkpoint(load_path, prefix="ckpt_")
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found under {load_path}")
        checkpoint = _torch_load_cpu(ckpt_path)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        status = self.load_state_dict(_strip_module_prefix(state), strict=False)
        return ckpt_path, status


def _strip_module_prefix(state_dict):
    return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _latest_checkpoint(load_path: str, prefix: Optional[str] = None) -> Optional[str]:
    if os.path.isdir(load_path):
        names = [name for name in os.listdir(load_path) if name.endswith((".pt", ".pth"))]
        if prefix is not None:
            prefixed = [name for name in names if name.startswith(prefix)]
            names = prefixed or names
        if not names:
            return None
        return os.path.join(load_path, sorted(names)[-1])
    if os.path.isfile(load_path):
        return load_path
    return None


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_encoder_state(checkpoint):
    if isinstance(checkpoint, dict) and "encoder" in checkpoint and isinstance(checkpoint["encoder"], dict):
        return _strip_module_prefix(checkpoint["encoder"])
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = _strip_module_prefix(state)
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder.") :]] = value
    return encoder_state or state


def load_encoder_weights(encoder: nn.Module, load_path: str):
    ckpt_path = _latest_checkpoint(load_path, prefix="encoder_")
    if ckpt_path is None:
        raise FileNotFoundError(f"No encoder checkpoint found at {load_path}")
    checkpoint = _torch_load_cpu(ckpt_path)
    state = extract_encoder_state(checkpoint)
    status = encoder.load_state_dict(state, strict=False)
    return ckpt_path, status
