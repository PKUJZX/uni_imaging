import torch
import torch.nn as nn


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    norm_type = (norm_type or "group").lower()
    if norm_type in ["group", "gn"]:
        # 8 groups is a common stable default for small channel counts
        num_groups = 8
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if norm_type in ["batch", "bn"]:
        return nn.BatchNorm2d(num_channels)
    if norm_type in ["instance", "in"]:
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type in ["none", "identity", ""]:
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "group", act: str = "silu"):
        super().__init__()
        if act.lower() == "relu":
            activation = nn.ReLU(inplace=True)
        elif act.lower() in ["silu", "swish"]:
            activation = nn.SiLU(inplace=True)
        elif act.lower() in ["gelu"]:
            activation = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(norm_type, out_ch),
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    Classic-style U-Net for 2D feature/image refinement.

    - Input expected in [0,1] by default (backgrounds in this project).
    - Output constrained to [0,1] via Sigmoid.
    - Input block: two 3×3 convs at full resolution (InputBlock-style), then 4× MaxPool
      encoder, bottleneck, 4× ConvTranspose2d upsampling with skip concatenation.

    Expects H and W divisible by 16 (e.g. 256×256) so skip tensors match upsampled maps.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        num_down: int = 4,
        norm_type: str = "group",
        act: str = "silu",
        input_in_01: bool = True,
    ):
        super().__init__()
        if num_down != 4:
            raise ValueError("UNet2D classic layout requires num_down == 4 (got %s)" % num_down)
        self.input_in_01 = input_in_01
        b = base_channels
        # Encoder channel schedule: b, 2b, 4b, 8b, 16b
        self.in_block = ConvBlock(in_channels, b, norm_type=norm_type, act=act)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(b, 2 * b, norm_type=norm_type, act=act)
        self.enc2 = ConvBlock(2 * b, 4 * b, norm_type=norm_type, act=act)
        self.enc3 = ConvBlock(4 * b, 8 * b, norm_type=norm_type, act=act)
        self.enc4 = ConvBlock(8 * b, 16 * b, norm_type=norm_type, act=act)

        self.bottleneck = ConvBlock(16 * b, 16 * b, norm_type=norm_type, act=act)

        self.up4 = nn.ConvTranspose2d(16 * b, 8 * b, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(16 * b, 8 * b, norm_type=norm_type, act=act)
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(8 * b, 4 * b, norm_type=norm_type, act=act)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(4 * b, 2 * b, norm_type=norm_type, act=act)
        self.up1 = nn.ConvTranspose2d(2 * b, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(2 * b, b, norm_type=norm_type, act=act)

        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1, bias=True)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            y: [B, out_channels, H, W] in [0,1]
        """
        if self.input_in_01:
            # Map [0,1] -> [-1,1] for more symmetric internal activations
            x = x * 2.0 - 1.0

        s0 = self.in_block(x)

        x = self.pool(s0)
        s1 = self.enc1(x)

        x = self.pool(s1)
        s2 = self.enc2(x)

        x = self.pool(s2)
        s3 = self.enc3(x)

        x = self.pool(s3)
        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.up4(x)
        x = self.dec4(torch.cat([x, s3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, s2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, s1], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s0], dim=1))

        y = self.out_act(self.out_conv(x))
        return y
