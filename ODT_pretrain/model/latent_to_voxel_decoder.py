import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


def _zero_init_conv(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class PixelShuffle3D(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = int(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        r = self.scale
        out_c = c // (r**3)
        if out_c * (r**3) != c:
            raise ValueError(f"channels={c} is not divisible by scale^3={r**3}")
        x = x.reshape(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.reshape(b, out_c, d * r, h * r, w * r)


class ChannelLayerNorm3D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class ConvNeXtRefineBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = ChannelLayerNorm3D(channels)
        self.fc1 = nn.Conv3d(channels, 2 * channels, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = _zero_init_conv(nn.Conv3d(2 * channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_norm1 = ChannelLayerNorm3D(in_channels)
        self.res_act1 = nn.SiLU()
        self.res_conv1 = nn.Conv3d(in_channels, 8 * out_channels, kernel_size=3, padding=1)
        self.res_shuffle = PixelShuffle3D(scale=2)
        self.res_norm2 = ChannelLayerNorm3D(out_channels)
        self.res_act2 = nn.SiLU()
        self.res_conv2 = _zero_init_conv(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))

        self.skip_conv = nn.Conv3d(in_channels, 8 * out_channels, kernel_size=1)
        self.skip_shuffle = PixelShuffle3D(scale=2)
        self.refine = ConvNeXtRefineBlock3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_norm1(x)
        residual = self.res_act1(residual)
        residual = self.res_conv1(residual)
        residual = self.res_shuffle(residual)
        residual = self.res_norm2(residual)
        residual = self.res_act2(residual)
        residual = self.res_conv2(residual)

        shortcut = self.skip_shuffle(self.skip_conv(x))
        return self.refine(shortcut + residual)


class StemRefineBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = ChannelLayerNorm3D(channels)
        self.fc1 = nn.Conv3d(channels, 2 * channels, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = _zero_init_conv(nn.Conv3d(2 * channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


def _upsample_steps(grid_shape: Sequence[int], output_shape: Sequence[int]) -> int:
    ratios = []
    for g, o in zip(grid_shape, output_shape):
        if o % g != 0:
            raise ValueError(f"output_shape={output_shape} must be divisible by grid_shape={grid_shape}")
        ratios.append(o // g)
    if len(set(ratios)) != 1:
        raise ValueError(f"Only isotropic 2x upsampling is supported, got ratios={ratios}")
    ratio = ratios[0]
    steps = int(math.log2(ratio))
    if 2**steps != ratio:
        raise ValueError(f"upsample ratio must be power of 2, got {ratio}")
    return steps


class LatentToVoxelDecoder(nn.Module):
    """3D PixelShuffle decoder from latent grid tokens to a single-channel voxel."""

    def __init__(
        self,
        latent_dim: int = 768,
        latent_grid_shape: Sequence[int] = (8, 16, 16),
        output_shape: Sequence[int] = (128, 256, 256),
        channels: Iterable[int] = (384, 192, 48, 12),
        head_channels: int = 12,
        out_channels: int = 1,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_grid_shape = tuple(int(x) for x in latent_grid_shape)
        self.output_shape = tuple(int(x) for x in output_shape)
        self.head_channels = int(head_channels)
        steps = _upsample_steps(self.latent_grid_shape, self.output_shape)

        channels = [int(x) for x in channels]
        if len(channels) != steps:
            if len(channels) > steps:
                channels = channels[:steps]
            else:
                last = channels[-1] if channels else max(self.latent_dim // 2, 16)
                channels.extend([max(last // (2 ** (i + 1)), 8) for i in range(steps - len(channels))])

        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.stem = StemRefineBlock3D(self.latent_dim)
        blocks = []
        in_ch = self.latent_dim
        for out_ch in channels:
            blocks.append(UpBlock3D(in_ch, out_ch))
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(blocks)

        self.head_norm = ChannelLayerNorm3D(in_ch)
        self.head_act1 = nn.SiLU()
        self.head_conv1 = nn.Conv3d(in_ch, self.head_channels, kernel_size=3, padding=1)
        self.head_act2 = nn.SiLU()
        self.head_conv2 = nn.Conv3d(self.head_channels, int(out_channels), kernel_size=1)
        if output_activation == "sigmoid":
            self.head_out = nn.Sigmoid()
        elif output_activation in ("identity", "none", ""):
            self.head_out = nn.Identity()
        else:
            raise ValueError(f"Unknown output_activation={output_activation!r}")

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.latent_grid_shape[0] * self.latent_grid_shape[1] * self.latent_grid_shape[2]
        if latent_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for grid {self.latent_grid_shape}, "
                f"got {latent_tokens.shape[1]}"
            )
        x = self.input_norm(latent_tokens)
        x = x.reshape(x.shape[0], *self.latent_grid_shape, self.latent_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.stem(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.head_norm(x)
        x = self.head_act1(x)
        x = self.head_conv1(x)
        x = self.head_act2(x)
        x = self.head_conv2(x)
        return self.head_out(x)
