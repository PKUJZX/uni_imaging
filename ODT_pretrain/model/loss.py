# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path
import math



# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()
        
    def _build_vgg(self):
        """Create VGG model with 2-channel input and average pooling."""
        model = vgg19()
        
        # 修改第一层卷积为2通道输入
        original_conv1 = model.features[0]  # Conv2d(3, 64, kernel_size=3, padding=1)
        model.features[0] = nn.Conv2d(
            in_channels=2,  # 改为2通道输入
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        
        return model.to(self.device).eval()
    
    def _load_weights(self):
        """Load pre-trained VGG weights and adapt first layer for 2-channel input."""
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        
        if torch.distributed.get_rank() == 0:
            # Download weights if needed
            if not weight_file.exists():
                os.system(f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')
        torch.distributed.barrier()
        
        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        
        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                if layer_idx == 0:  # 特殊处理第一层
                    # 原始权重: [64, 3, 3, 3] -> [3, 3, 3, 64] -> [64, 3, 3, 3]
                    original_weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                    # original_weights shape: [64, 3, 3, 3]
                    
                    rgb_avg = original_weights.mean(dim=1, keepdim=True)  # [64, 1, 3, 3]
                    adapted_weights = rgb_avg.repeat(1, 2, 1, 1)      
                    
                    self.vgg.features[layer_idx].weight = nn.Parameter(adapted_weights, requires_grad=False)
                else:
                    # 其他层保持原样
                    weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                    self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                
                # Set biases (所有层都一样)
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        
        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())
        
        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
    
    def _preprocess_images(self, images):
        """Convert 2-channel images to VGG input format."""
        # images in [0,1]，用0.5作为全局中心更合理（对应127.5）
        mean = torch.tensor([127.5, 127.5], dtype=images.dtype).reshape(1, 2, 1, 1).to(images.device)
        return images * 255.0 - mean

    
    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    
    def forward(self, pred_img, target_img):
        """Compute perceptual loss between 2-channel prediction and target."""
        # 现在直接处理2通道图像
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)
        
        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)
        
        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)
        
        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5
        
        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
        
        return total_loss

class FrequencyLoss(nn.Module):
    """
    Frequency domain L1 loss for 2-channel (real, imag) NCHW tensors.
    Computes Rytov field log(|z|)+j*unwrap(∠z), FFT2, L1 of |F_pred-F_gt| in NA mask.
    Only computes loss within k_bound_pixel*1.5 radius (NA boundary region).
    """
    def __init__(self):
        super().__init__()
        # 硬编码的物理参数（来自 commercial_ODT.yaml）
        self.wavelength = 0.532  # μm
        self.NA = 1.32
        self.camera_pixel_size = 4.5  # μm
        self.magnification = 100
    
    def _compute_k_bound_pixel(self, image_size):
        """
        计算 k_bound_pixel，考虑图像尺寸的变化。
        
        Args:
            image_size: 当前图像尺寸（H 或 W，假设是正方形）
        
        Returns:
            k_bound_pixel: NA 边界对应的像素半径（已乘以 1.5）
        """
        import math
        pixelsize = self.camera_pixel_size / self.magnification  # μm
        
        # 计算当前图像尺寸对应的 spec_pixel_size
        # 原始：spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
        # 对于 image_size，spec_pixel_size_image = 2 * np.pi / (pixelsize * image_size * 2)
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        
        # 计算 k_bound_pixel
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        
        # 返回 k_bound_pixel * 1.5（转换为 int）
        return int(k_bound_pixel * 1.5)
    
    def _create_circular_mask(self, h, w, radius, device):
        """
        创建圆形 mask，中心在图像中心，半径为 radius。
        
        Args:
            h, w: 图像高度和宽度
            radius: 圆形半径（像素）
            device: 设备
        
        Returns:
            mask: [H, W] bool tensor，True 表示在圆形区域内
        """
        center_y = h // 2
        center_x = w // 2
        
        # 创建坐标网格（使用 float32 类型，因为 torch.arange 不支持 bool）
        y_coords = torch.arange(h, device=device, dtype=torch.float32)
        x_coords = torch.arange(w, device=device, dtype=torch.float32)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算每个点到中心的距离
        distances = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
        
        # 创建 mask：距离 <= radius 的位置为 True（返回 bool 类型）
        mask = distances <= radius
        
        return mask
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算频域 L1 loss（仅在 k_bound_pixel*1.5 范围内）。
        
        Args:
            x, y: [N, C, H, W] tensors，其中 C=2（实部通道0，虚部通道1）
        
        Returns:
            frequency_l1_loss: scalar tensor，mean(|F_pred - F_gt|) within NA boundary
        """
        assert x.shape == y.shape, f"FrequencyLoss: shape mismatch: {x.shape} vs {y.shape}"
        n, c, h, w = x.shape
        assert c == 2, f"FrequencyLoss: expected 2 channels (real, imag), got {c}"
        
        # 提取实部和虚部，并转换为 float32（torch.complex 不支持 BFloat16）
        x_real = x[:, 0:1, :, :].float()  # [N, 1, H, W]
        x_imag = x[:, 1:2, :, :].float()
        y_real = y[:, 0:1, :, :].float()
        y_imag = y[:, 1:2, :, :].float()
        
        # 构建复数：z = real + imag*j
        x_complex = torch.complex(x_real, x_imag)  # [N, 1, H, W] complex
        y_complex = torch.complex(y_real, y_imag)
        
        x_rytov = x_complex.squeeze(1)  # [N,H,W] complex
        y_rytov = y_complex.squeeze(1)  # [N,H,W] complex
        
        # FFT2：对每个样本分别做 FFT（在最后两个维度上）
        F_x = torch.fft.fft2(x_rytov, norm="ortho")  # [N, H, W] complex
        F_y = torch.fft.fft2(y_rytov, norm="ortho")
        
        # 对 FFT 结果做 fftshift，将零频率移到图像中心（与 mask 对齐）
        F_x = torch.fft.fftshift(F_x, dim=(-2, -1))  # [N, H, W] complex
        F_y = torch.fft.fftshift(F_y, dim=(-2, -1))
        
        # 频域差的模长：|F_pred - F_gt|
        diff = F_y - F_x  # [N, H, W] complex
        mag = torch.abs(diff)  # [N, H, W] real
        
        # 计算 k_bound_pixel（使用图像尺寸 h，假设是正方形）
        k_bound_radius = self._compute_k_bound_pixel(h)
        
        # 创建圆形 mask（只保留 k_bound_pixel*1.5 范围内的频率）
        # mask 以 (h/2, w/2) 为中心，与 fftshift 后的 FFT 结果对齐
        mask = self._create_circular_mask(h, w, k_bound_radius, device=mag.device)
        # mask: [H, W] -> [1, H, W] -> [N, H, W]
        mask = mask.unsqueeze(0).expand(n, -1, -1)  # [N, H, W]
        
        # 只在 mask 区域内计算 loss
        masked_mag = mag * mask.float()  # [N, H, W]
        
        # 计算平均 L1 loss（只对 mask 内的位置取平均）
        # 先计算 mask 内的总和，再除以 mask 内的元素数量
        mask_count = mask.sum().float()  # 总的有效像素数
        if mask_count > 0:
            frequency_l1_loss = masked_mag.sum() / mask_count
        else:
            # 如果 mask 为空，返回 0
            frequency_l1_loss = torch.tensor(0.0, device=mag.device)
        
        return frequency_l1_loss

class WeightedFrequencyLoss(nn.Module):
    """
    Frequency domain weighted L1 loss with pose-based mask shift.

    For each sample with scan direction (ky, kx) (normalized, from pose[:2]):
      1. Convert to pixel coords: k_scan_pix = pose_xy * km / spec_pixel_size_image
         where km = 2*pi/wavelength * n_medium
      2. Build circular mask centered at (center_y + ky_pix, center_x + kx_pix)
         with radius = k_bound_pixel (no *1.5).
      3. Apply spatially varying weight inside the mask:
         weight(y, x) = (dist_from_image_center(y,x) / (|k_scan_pix| + k_bound_pixel))^2
      4. Compute mean weighted L1 loss within the mask.
    """

    def __init__(self):
        super().__init__()
        self.wavelength = 0.532       # μm
        self.NA = 1.32
        self.camera_pixel_size = 4.5  # μm
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        """Return (spec_pixel_size_image, k_bound_pixel [int], km)."""
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [N, 2, H, W] prediction  (real, imag channels)
            y:    [N, 2, H, W] target
            pose: [N, d] — pose[:, 0]=ky, pose[:, 1]=kx (normalised scan positions)

        Returns:
            scalar weighted freq-L1 loss
        """
        assert x.shape == y.shape, f"WeightedFrequencyLoss: shape mismatch {x.shape} vs {y.shape}"
        n, c, h, w = x.shape
        assert c == 2, f"WeightedFrequencyLoss: expected 2 channels, got {c}"

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        # ---- Rytov transform + FFT (same pipeline as FrequencyLoss) ----
        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())   # [N, H, W]
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        mag = torch.abs(F_y - F_x)  # [N, H, W]

        # ---- k-scan pixel coordinates ----
        pose_xy = pose[:, :2].float()                         # [N, 2]
        k_scan_pix = pose_xy * km / spec_pixel_size_image     # [N, 2]: [ky_pix, kx_pix]
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)              # [N, 1, 1]
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        # ---- coordinate grids ----
        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)  # [1, H, W]
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        # ---- circular mask centred at (center_y + ky_pix, center_x + kx_pix) ----
        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )  # [N, H, W]
        mask = (dist_from_mask_center <= k_bound_pixel).float()  # [N, H, W]

        # ---- spatially varying weight based on distance from image centre ----
        dist_from_center = torch.sqrt(
            (Y - center_y) ** 2 + (X - center_x) ** 2
        )  # [1, H, W]
        k_scan_mag = torch.sqrt(kx_pix ** 2 + ky_pix ** 2)   # [N, 1, 1]
        weight = (dist_from_center / (k_scan_mag + k_bound_pixel)) ** 2  # [N, H, W]

        # ---- weighted masked L1 loss ----
        weighted_mag = mag * mask * weight                                    # [N, H, W]
        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)                   # [N]
        per_sample_loss = weighted_mag.sum(dim=(-2, -1)) / mask_count        # [N]

        return per_sample_loss.mean()


class NACircleFrequencyLoss(nn.Module):
    """
    Frequency domain L1 loss with pose-based mask shift, no weight.

    Same as WeightedFrequencyLoss but without the distance-based weight:
      - Mask: circle centred at (center_y + ky_pix, center_x + kx_pix)
      - Radius: k_bound_pixel (no *1.5)
      - Loss: mean(|F_pred - F_gt|) within mask
    """

    def __init__(self):
        super().__init__()
        self.wavelength = 0.532
        self.NA = 1.32
        self.camera_pixel_size = 4.5
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [N, 2, H, W] prediction
            y:    [N, 2, H, W] target
            pose: [N, d] — pose[:, 0]=ky, pose[:, 1]=kx
        """
        assert x.shape == y.shape
        n, c, h, w = x.shape
        assert c == 2

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        mag = torch.abs(F_y - F_x)  # [N, H, W]

        pose_xy = pose[:, :2].float()
        k_scan_pix = pose_xy * km / spec_pixel_size_image
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )
        mask = (dist_from_mask_center <= k_bound_pixel).float()  # [N, H, W]

        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)       # [N]
        per_sample_loss = (mag * mask).sum(dim=(-2, -1)) / mask_count

        return per_sample_loss.mean()


class NACircleFrequencyampandphsLoss(nn.Module):
    """
    NA-circle frequency loss that computes a combined amplitude/phase objective.

    The scalar loss is controlled by two external weights:
      - amp_weight: scales the amplitude mismatch term
      - phase_weight: scales the phase mismatch term

    The loss is averaged inside the pose-shifted NA circle mask.
    """

    def __init__(self, amp_weight: float, phase_weight: float):
        super().__init__()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        self.wavelength = 0.532
        self.NA = 1.32
        self.camera_pixel_size = 4.5
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor):
        assert x.shape == y.shape
        n, c, h, w = x.shape
        assert c == 2

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        amp_x = torch.abs(F_x)
        amp_y = torch.abs(F_y)
        phase_x = torch.angle(F_x)
        phase_y = torch.angle(F_y)

        pose_xy = pose[:, :2].float()
        k_scan_pix = pose_xy * km / spec_pixel_size_image
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )
        mask = (dist_from_mask_center <= k_bound_pixel).float()

        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)

        amp_term = self.amp_weight * (amp_y - amp_x) ** 2
        phs_term = self.phase_weight * (amp_y * amp_x) * (1.0 - torch.cos(phase_y - phase_x))
        combined = torch.sqrt(amp_term + phs_term)

        per_sample_loss = (combined * mask).sum(dim=(-2, -1)) / mask_count
        return per_sample_loss.mean()


class NACircleFrequencyampl1andphsl1Loss(nn.Module):
    """
    NA-circle frequency amplitude L1 / phase L1 loss.

    Returns amplitude and phase losses separately. Final weighting is handled
    by LossComputer using the yaml weights.
    """

    def __init__(self):
        super().__init__()
        self.wavelength = 0.532
        self.NA = 1.32
        self.camera_pixel_size = 4.5
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor):
        assert x.shape == y.shape
        n, c, h, w = x.shape
        assert c == 2

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        amp_x = torch.abs(F_x)
        amp_y = torch.abs(F_y)
        phase_x = torch.angle(F_x)
        phase_y = torch.angle(F_y)

        pose_xy = pose[:, :2].float()
        k_scan_pix = pose_xy * km / spec_pixel_size_image
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )
        mask = (dist_from_mask_center <= k_bound_pixel).float()

        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)

        amp_loss = (torch.abs(amp_y - amp_x) * mask).sum(dim=(-2, -1)) / mask_count
        phs_arg = torch.clamp(2.0 * amp_y * amp_x * (1.0 - torch.cos(phase_y - phase_x)), min=0.0)
        phs_l1 = torch.sqrt(phs_arg + 1e-12) - 1e-6
        phs_loss = (
            (phs_l1 * mask).sum(dim=(-2, -1))
            / mask_count
        )

        return amp_loss.mean(), phs_loss.mean()


class LogampFreqinNACircleampandphsl2Loss(nn.Module):
    """
    NA-circle frequency log-amplitude/phase L2 loss.

    Returns log1p(amplitude) and phase losses separately. Final weighting is
    handled by LossComputer using the yaml weights.
    """

    def __init__(self):
        super().__init__()
        self.wavelength = 0.532
        self.NA = 1.32
        self.camera_pixel_size = 4.5
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor):
        assert x.shape == y.shape
        n, c, h, w = x.shape
        assert c == 2

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        amp_x = torch.abs(F_x)
        amp_y = torch.abs(F_y)
        log_amp_x = torch.log1p(amp_x)
        log_amp_y = torch.log1p(amp_y)
        phase_x = torch.angle(F_x)
        phase_y = torch.angle(F_y)

        pose_xy = pose[:, :2].float()
        k_scan_pix = pose_xy * km / spec_pixel_size_image
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )
        mask = (dist_from_mask_center <= k_bound_pixel).float()

        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)

        amp_loss = (((log_amp_y - log_amp_x) ** 2) * mask).sum(dim=(-2, -1)) / mask_count
        phs_loss = (
            (log_amp_y**2 * (1.0 - torch.cos(phase_y - phase_x)) * mask).sum(dim=(-2, -1))
            / mask_count
        )

        return amp_loss.mean(), phs_loss.mean()


class LogampFreqinNACircleLoss(nn.Module):
    """
    NA circle L1 on log(1+|F|) with phase unchanged after FFT.

    Steps (aligned with NACircleFrequencyLoss for mask / pose):
      1) z = real + j*imag from [0,1] two channels
      2) F = fftshift(fft2(z), ortho)  — same as NACircleFrequencyLoss
      3) F' = log(1 + |F|) * exp(j * angle(F))  — log1p on amplitude, phase unchanged
      4) mean(|F'_y - F'_x|) within pose-shifted NA circle mask
    """

    def __init__(self):
        super().__init__()
        self.wavelength = 0.532
        self.NA = 1.32
        self.camera_pixel_size = 4.5
        self.magnification = 100
        self.n_medium = 1.33

    def _compute_params(self, image_size):
        pixelsize = self.camera_pixel_size / self.magnification
        spec_pixel_size_image = 2 * math.pi / (pixelsize * image_size * 2)
        k_bound_pixel = math.ceil(self.NA * 2 * math.pi / self.wavelength / spec_pixel_size_image)
        km = 2 * math.pi / self.wavelength * self.n_medium
        return spec_pixel_size_image, k_bound_pixel, km

    def forward(self, x: torch.Tensor, y: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [N, 2, H, W] prediction
            y:    [N, 2, H, W] target
            pose: [N, d] — pose[:, 0]=ky, pose[:, 1]=kx
        """
        assert x.shape == y.shape
        n, c, h, w = x.shape
        assert c == 2

        spec_pixel_size_image, k_bound_pixel, km = self._compute_params(h)

        center_y = h // 2
        center_x = w // 2
        device = x.device

        x_complex = torch.complex(x[:, 0].float(), x[:, 1].float())
        y_complex = torch.complex(y[:, 0].float(), y[:, 1].float())

        F_x = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
        F_y = torch.fft.fftshift(torch.fft.fft2(y_complex, norm="ortho"), dim=(-2, -1))

        amp_x = torch.abs(F_x)
        amp_y = torch.abs(F_y)
        phase_x = torch.angle(F_x)
        phase_y = torch.angle(F_y)
        F_x_logamp = torch.log1p(amp_x) * torch.exp(1j * phase_x)
        F_y_logamp = torch.log1p(amp_y) * torch.exp(1j * phase_y)

        mag = torch.abs(F_y_logamp - F_x_logamp)  # [N, H, W]

        pose_xy = pose[:, :2].float()
        k_scan_pix = pose_xy * km / spec_pixel_size_image
        ky_pix = k_scan_pix[:, 0].view(n, 1, 1)
        kx_pix = k_scan_pix[:, 1].view(n, 1, 1)

        Y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        X = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)

        dist_from_mask_center = torch.sqrt(
            (Y - (center_y + ky_pix)) ** 2 + (X - (center_x + kx_pix)) ** 2
        )
        mask = (dist_from_mask_center <= k_bound_pixel).float()  # [N, H, W]

        mask_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)       # [N]
        per_sample_loss = (mag * mask).sum(dim=(-2, -1)) / mask_count

        return per_sample_loss.mean()


class LossComputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.NA_circle_frequency_phs_l1_loss_weight = self.config.training.get(
            "NA_circle_frequency_phs_l1_loss_weight",
            self.config.training.get("NA_circle_frequency_phs_l2_loss_weight", 0.0),
        )

        if self.config.training.perceptual_loss_weight > 0.0:
            # 现在PerceptualLoss直接支持2通道
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())
        if self.config.training.frequency_loss_weight > 0.0:
            # Frequency loss for 2-channel images
            self.frequency_loss_module = FrequencyLoss()
        if self.config.training.weighted_frequency_loss_weight > 0.0:
            self.weighted_frequency_loss_module = WeightedFrequencyLoss()
        if self.config.training.NA_circle_frequency_loss_weight > 0.0:
            self.NA_circle_frequency_loss_module = NACircleFrequencyLoss()
        if self.config.training.logamp_freq_in_NA_circle_loss_weight > 0.0:
            self.logamp_freq_in_NA_circle_loss_module = LogampFreqinNACircleLoss()
        if (
            self.config.training.NA_circle_frequency_loss_amp_weight > 0.0
            and self.config.training.NA_circle_frequency_loss_phs_weight > 0.0
        ):
            self.NA_circle_frequency_amp_and_phs_loss_module = NACircleFrequencyampandphsLoss(
                amp_weight=self.config.training.NA_circle_frequency_loss_amp_weight,
                phase_weight=self.config.training.NA_circle_frequency_loss_phs_weight,
            )

        if (
            self.config.training.NA_circle_frequency_amp_l1_loss_weight > 0.0
            or self.NA_circle_frequency_phs_l1_loss_weight > 0.0
        ):
            self.NA_circle_frequency_amp_l1_and_phs_l1_loss_module = NACircleFrequencyampl1andphsl1Loss()

        if (
            self.config.training.logamp_freq_in_NA_circle_amp_l2_loss_weight > 0.0
            or self.config.training.logamp_freq_in_NA_circle_phs_l2_loss_weight > 0.0
        ):
            self.logamp_freq_in_NA_circle_amp_and_phs_l2_loss_module = LogampFreqinNACircleampandphsl2Loss()

    def compute_amp_phase_loss(self, rendering: torch.Tensor, target: torch.Tensor):
        """Compute amplitude and phase losses for complex 2-channel images."""
        eps = 1e-8
        pred_real = rendering[:, 0:1, :, :]
        pred_imag = rendering[:, 1:2, :, :]
        gt_real = target[:, 0:1, :, :]
        gt_imag = target[:, 1:2, :, :]

        amp_pred = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + eps)
        amp_gt = torch.sqrt(gt_real ** 2 + gt_imag ** 2 + eps)
        amp_loss = F.mse_loss(amp_pred, amp_gt)

        phase_pred = torch.atan2(pred_imag, pred_real)
        phase_gt = torch.atan2(gt_imag, gt_real)
        phase_loss = torch.mean(1.0 - torch.cos(phase_pred - phase_gt))

        return amp_loss, phase_loss

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    @staticmethod
    def _assert_range_11(x: torch.Tensor, name: str, eps: float = 0.0):
        # 允许极小的数值误差
        x_min = float(x.min().detach().cpu())
        x_max = float(x.max().detach().cpu())
        assert x_min >= -1.0 - eps, f"{name} min={x_min} out of range [-1,1]"
        assert x_max <= 1.0 + eps, f"{name} max={x_max} out of range [-1,1]"
    
    def forward(self, rendering, target, target_pose=None):
        """
        Calculate various losses between rendering and target images.
        
        Args:
            rendering: [b, v, 2, h, w], value range [-1, 1] (2 channels: real, imag)
            target: [b, v, 2, h, w], value range [-1, 1] (2 channels: real, imag)
            target_pose: [b, v, d] 或 None，格式为 [..., y, x, ...]（可选，用于频域 loss 的频移）
        
        Returns:
            Dictionary of loss metrics
        """
        b, v, c, h, w = rendering.size()
        assert c == 2, f"Expected 2 channels (real, imag), got {c}"

        self._assert_range_11(rendering, "rendering")
        self._assert_range_11(target, "target")

        
        rendering = rendering.reshape(b * v, c, h, w)
        target = target.reshape(b * v, c, h, w)
        target_pose_flat = None
        if target_pose is not None:
            target_pose_flat = target_pose.reshape(b * v, -1)
        
        # print("rendering", rendering.min(), rendering.max(), rendering.mean())
        # print("target", target.min(), target.max(), target.mean())

        amp_loss = torch.tensor(0.0, device=rendering.device)
        phase_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.amp_loss_weight > 0.0 and self.config.training.phase_loss_weight > 0.0:
            amp_loss, phase_loss = self.compute_amp_phase_loss(rendering, target)

        # L2 loss - 直接在2通道上计算（保持 [-1,1] 域）
        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss = F.mse_loss(rendering, target)

        # PSNR 按 [0,1] 域计算，更符合常规图像指标定义
        rendering_01 = torch.clamp((rendering + 1.0) / 2.0, 0.0, 1.0)
        target_01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        l2_loss_for_pnsr = F.mse_loss(rendering_01, target_01)
        psnr = -10.0 * torch.log10(l2_loss_for_pnsr)

        l1_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.l1_loss_weight > 0.0:
            l1_loss = F.l1_loss(rendering, target)

        # Perceptual loss - 现在直接支持2通道
        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        # ----- Frequency Loss -----
        frequency_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.frequency_loss_weight > 0.0:
            frequency_loss = self.frequency_loss_module(rendering, target)

        # ----- Weighted Frequency Loss -----
        weighted_frequency_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.weighted_frequency_loss_weight > 0.0:
            if target_pose_flat is not None:
                weighted_frequency_loss = self.weighted_frequency_loss_module(rendering, target, target_pose_flat)

        # ----- NA Circle Frequency Loss -----
        NA_circle_frequency_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.NA_circle_frequency_loss_weight > 0.0:
            if target_pose_flat is not None:
                NA_circle_frequency_loss = self.NA_circle_frequency_loss_module(rendering, target, target_pose_flat)

        logamp_freq_in_NA_circle_loss = torch.tensor(0.0, device=rendering.device)
        if self.config.training.logamp_freq_in_NA_circle_loss_weight > 0.0:
            if target_pose_flat is not None:
                logamp_freq_in_NA_circle_loss = self.logamp_freq_in_NA_circle_loss_module(
                    rendering, target, target_pose_flat
                )

        NA_circle_frequency_amp_and_phs_loss = torch.tensor(0.0, device=rendering.device)
        if (
            self.config.training.NA_circle_frequency_loss_amp_weight > 0.0
            and self.config.training.NA_circle_frequency_loss_phs_weight > 0.0
        ):
            if target_pose_flat is not None:
                NA_circle_frequency_amp_and_phs_loss = self.NA_circle_frequency_amp_and_phs_loss_module(
                    rendering, target, target_pose_flat
                )

        NA_circle_frequency_amp_l1_loss = torch.tensor(0.0, device=rendering.device)
        NA_circle_frequency_phs_l1_loss = torch.tensor(0.0, device=rendering.device)
        if (
            self.config.training.NA_circle_frequency_amp_l1_loss_weight > 0.0
            or self.NA_circle_frequency_phs_l1_loss_weight > 0.0
        ):
            if target_pose_flat is not None:
                NA_circle_frequency_amp_l1_loss, NA_circle_frequency_phs_l1_loss = self.NA_circle_frequency_amp_l1_and_phs_l1_loss_module(
                    rendering, target, target_pose_flat
                )

        logamp_freq_in_NA_circle_amp_l2_loss = torch.tensor(0.0, device=rendering.device)
        logamp_freq_in_NA_circle_phs_l2_loss = torch.tensor(0.0, device=rendering.device)
        if (
            self.config.training.logamp_freq_in_NA_circle_amp_l2_loss_weight > 0.0
            or self.config.training.logamp_freq_in_NA_circle_phs_l2_loss_weight > 0.0
        ):
            if target_pose_flat is not None:
                logamp_freq_in_NA_circle_amp_l2_loss, logamp_freq_in_NA_circle_phs_l2_loss = self.logamp_freq_in_NA_circle_amp_and_phs_l2_loss_module(
                    rendering, target, target_pose_flat
                )

        loss = (
            self.config.training.amp_loss_weight * amp_loss
            + self.config.training.phase_loss_weight * phase_loss
            + self.config.training.l2_loss_weight * l2_loss 
            + self.config.training.l1_loss_weight * l1_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
            + self.config.training.frequency_loss_weight * frequency_loss
            + self.config.training.weighted_frequency_loss_weight * weighted_frequency_loss
            + self.config.training.NA_circle_frequency_loss_weight * NA_circle_frequency_loss
            + self.config.training.logamp_freq_in_NA_circle_loss_weight * logamp_freq_in_NA_circle_loss
            + NA_circle_frequency_amp_and_phs_loss
            + self.config.training.NA_circle_frequency_amp_l1_loss_weight * NA_circle_frequency_amp_l1_loss
            + self.NA_circle_frequency_phs_l1_loss_weight * NA_circle_frequency_phs_l1_loss
            + self.config.training.logamp_freq_in_NA_circle_amp_l2_loss_weight * logamp_freq_in_NA_circle_amp_l2_loss
            + self.config.training.logamp_freq_in_NA_circle_phs_l2_loss_weight * logamp_freq_in_NA_circle_phs_l2_loss
        )

        loss_metrics = edict(
            loss=loss,
            amp_loss=amp_loss,
            phase_loss=phase_loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            frequency_loss=frequency_loss,
            weighted_frequency_loss=weighted_frequency_loss,
            NA_circle_frequency_loss=NA_circle_frequency_loss,
            logamp_freq_in_NA_circle_loss=logamp_freq_in_NA_circle_loss,
            NA_circle_frequency_amp_and_phs_loss=NA_circle_frequency_amp_and_phs_loss,
            NA_circle_frequency_amp_l1_loss=NA_circle_frequency_amp_l1_loss,
            NA_circle_frequency_phs_l1_loss=NA_circle_frequency_phs_l1_loss,
            logamp_freq_in_NA_circle_amp_l2_loss=logamp_freq_in_NA_circle_amp_l2_loss,
            logamp_freq_in_NA_circle_phs_l2_loss=logamp_freq_in_NA_circle_phs_l2_loss,
            psnr=psnr,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss,
        )
        return loss_metrics
