import torch
from torch import Tensor
from einops import rearrange
import os
from PIL import Image
from utils import data_utils
import numpy as np
import tifffile as tiff
from easydict import EasyDict as edict
import json
from rich import print
from skimage.metrics import structural_similarity

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")


@torch.no_grad()
def compute_voxel_mse_per_sample(
    ground_truth: Tensor,
    predicted: Tensor,
) -> Tensor:
    """
    体素逐样本 MSE（与训练 L2 一致：全体素均值）。
    ground_truth / predicted: [batch, ...] 任意尾部维度。
    Returns:
        [batch] 每个样本的 MSE。
    """
    gt = ground_truth.float()
    pr = predicted.float()
    return ((gt - pr) ** 2).flatten(1).mean(dim=1)


@torch.no_grad()
def compute_voxel_psnr_per_sample(
    ground_truth: Tensor,
    predicted: Tensor,
    data_range: float = 1.0,
) -> Tensor:
    mse = compute_voxel_mse_per_sample(ground_truth, predicted)
    data_range_t = torch.tensor(float(data_range), device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10((data_range_t * data_range_t) / mse)


@torch.no_grad()
def compute_voxel_ssim_per_sample(
    ground_truth: Tensor,
    predicted: Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
) -> Tensor:
    """
    3D SSIM for voxel tensors.
    ground_truth / predicted: [B, D, H, W]. Per-sample input to skimage is [D, H, W],
    so there is no channel axis.
    """
    gt = ground_truth.float()
    pr = predicted.float()
    if gt.shape != pr.shape:
        raise ValueError(f"Voxel shape mismatch: {gt.shape} vs {pr.shape}")
    if gt.ndim != 4:
        raise ValueError(f"Voxel SSIM expects [B, D, H, W], got {gt.shape}")

    ssim_values = []
    for gt_sample, pred_sample in zip(gt, pr):
        gt_np = gt_sample.detach().cpu().numpy()
        pred_np = pred_sample.detach().cpu().numpy()
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=win_size,
            gaussian_weights=True,
            data_range=data_range,
        )
        ssim_values.append(ssim)

    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)


def _normalize_image_tensor_for_uint8(x: torch.Tensor) -> torch.Tensor:
    """Map image tensors to [0,1] for visualization; supports already-[0,1] and signed inputs."""
    x = x.detach().float()
    if x.numel() == 0:
        return x
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin >= 0.0 and xmax <= 1.0:
        return x.clamp(0.0, 1.0)
    if xmin >= -1.05 and xmax <= 1.05:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    denom = xmax - xmin
    if denom <= 1e-12:
        return torch.zeros_like(x)
    return ((x - xmin) / denom).clamp(0.0, 1.0)


@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False
):
    """
    子目录按样本 uid（index[..., -1]）命名；保存 input 实部/虚部、体素 npy、可选 metrics.json。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    input_data = result.input
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save input 实部/虚部图，以及体素 npy（若有预测）
        _save_images(result, batch_idx, sample_dir)
        
        if compute_metrics and getattr(result, "predicted_voxel", None) is not None:
            _save_metrics(
                input_data.voxel[batch_idx],
                result.predicted_voxel[batch_idx],
                sample_dir,
                scene_name,
            )
        
        # # Save video if available
        # if hasattr(result, "video_rendering"):
        #     _save_video(result.video_rendering[batch_idx], sample_dir)

def visualize_intermediate_results(out_dir, result, save_arrays: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    input = result.input

    input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
    input_uid_based_filename = f"{input_uids[0]:08d}_{input_uids[-1]:08d}"
    
    # 创建输入图像的实部和虚部
    b, v, c, h, w = input.image.size()
    max_viz_views = 24
    if v > max_viz_views:
        view_idx = torch.linspace(0, v - 1, steps=max_viz_views, device=input.image.device).round().long()
        image_for_viz = input.image.index_select(1, view_idx)
        v = max_viz_views
    else:
        image_for_viz = input.image

    # 输入实部
    input_real = image_for_viz[:, :, 0:1, :, :].reshape(b * v, 1, h, w).detach().float().cpu()
    input_real = _normalize_image_tensor_for_uint8(input_real)
    input_real_grid = rearrange(input_real, "(b v) c h w -> (b h) (v w) c", v=v)
    input_real_grid = (input_real_grid.numpy() * 255.0).astype(np.uint8).squeeze(-1)

    # 输入虚部
    input_imag = image_for_viz[:, :, 1:2, :, :].reshape(b * v, 1, h, w).detach().float().cpu()
    input_imag = _normalize_image_tensor_for_uint8(input_imag)
    input_imag_grid = rearrange(input_imag, "(b v) c h w -> (b h) (v w) c", v=v)
    input_imag_grid = (input_imag_grid.numpy() * 255.0).astype(np.uint8).squeeze(-1)

    # 保存输入图像
    Image.fromarray(input_real_grid, mode='L').save(
        os.path.join(out_dir, f"input_real_{input_uid_based_filename}.jpg")
    )
    Image.fromarray(input_imag_grid, mode='L').save(
        os.path.join(out_dir, f"input_imag_{input_uid_based_filename}.jpg")
    )

    # 处理体素
    if result.predicted_voxel is not None:
        target_voxel = input.voxel  # 输入的体素
        predicted_voxel = result.predicted_voxel  # 预测的体素
        b, d, h, w = predicted_voxel.size()
        
        for batch_idx in range(b):
            scene_dir = os.path.join(out_dir, str(input.scene_name[batch_idx]))
            os.makedirs(scene_dir, exist_ok=True)
            
            # 获取当前场景的体素
            current_target = target_voxel[batch_idx].detach().float().cpu().numpy()  # [d, h, w]
            current_pred = predicted_voxel[batch_idx].detach().float().cpu().numpy()  # [d, h, w]
            
            if save_arrays:
                np.save(os.path.join(scene_dir, "gt_voxel.npy"), current_target)
                np.save(os.path.join(scene_dir, "predicted_voxel.npy"), current_pred)
            
            # ==================== XY切片（深度方向）====================
            # 以 d//2 为中心，向上（索引减小）30 张、向下（索引增大）30 张
            c = d // 2
            lower = list(range(max(0, c - 30), c))
            upper = list(range(c, min(d, c + 30)))
            slice_indices = lower + upper
            num_slices = len(slice_indices)
            
            cols = 12
            rows = (num_slices + cols - 1) // cols if num_slices > 0 else 0
            
            gt_grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
            pred_grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
            
            for i, slice_idx in enumerate(slice_indices):
                # 获取输入和预测的XY切片
                target_xy = current_target[slice_idx, :, :]  # [h, w]
                pred_xy = current_pred[slice_idx, :, :]  # [h, w]
                
                # 确保值在0-1范围内
                target_xy = np.clip(target_xy, 0, 1)
                pred_xy = np.clip(pred_xy, 0, 1)
                
                # 转换为灰度图
                target_xy_gray = (target_xy * 255).astype(np.uint8)
                pred_xy_gray = (pred_xy * 255).astype(np.uint8)
                
                # 计算在网格中的位置
                row = i // cols
                col = i % cols
                
                # 将切片放到网格中
                gt_grid[row*h:(row+1)*h, col*w:(col+1)*w] = target_xy_gray
                pred_grid[row*h:(row+1)*h, col*w:(col+1)*w] = pred_xy_gray
            
            if num_slices > 0:
                combined_grid = np.vstack([gt_grid, pred_grid])
                Image.fromarray(combined_grid, mode='L').save(
                    os.path.join(scene_dir, "xy_slices_grid.jpg")
                )
            
            # ==================== YZ切片（固定x = w/2）====================
            x_slice = w // 2
            target_yz = current_target[:, :, x_slice]  # [d, h]
            pred_yz = current_pred[:, :, x_slice]  # [d, h]
            
            # 确保值在0-1范围内
            target_yz = np.clip(target_yz, 0, 1)
            pred_yz = np.clip(pred_yz, 0, 1)
            
            # 转换为灰度图
            target_yz_gray = (target_yz * 255).astype(np.uint8)
            pred_yz_gray = (pred_yz * 255).astype(np.uint8)
            
            # 将输入和预测并排放置
            combined_yz = np.hstack([target_yz_gray, pred_yz_gray])
            
            # 保存图片
            Image.fromarray(combined_yz, mode='L').save(
                os.path.join(scene_dir, f"yz_slice_x{x_slice}_combined.jpg")
            )
            
            # ==================== XZ切片（固定y = h/2）====================
            y_slice = h // 2
            target_xz = current_target[:, y_slice, :]  # [d, w]
            pred_xz = current_pred[:, y_slice, :]  # [d, w]
            
            # 确保值在0-1范围内
            target_xz = np.clip(target_xz, 0, 1)
            pred_xz = np.clip(pred_xz, 0, 1)
            
            # 转换为灰度图
            target_xz_gray = (target_xz * 255).astype(np.uint8)
            pred_xz_gray = (pred_xz * 255).astype(np.uint8)
            
            # 将输入和预测并排放置
            combined_xz = np.hstack([target_xz_gray, pred_xz_gray])
            
            # 保存图片
            Image.fromarray(combined_xz, mode='L').save(
                os.path.join(scene_dir, f"xz_slice_y{y_slice}_combined.jpg")
            )
            
            # 创建一个简单的文本文件记录体素信息
            info_file = os.path.join(scene_dir, "voxel_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"scene_name: {input.scene_name[batch_idx]}\n")
                f.write(f"Voxel shape: {d} x {h} x {w}\n")
                f.write(
                    f"XY slices around d//2={c}: total {num_slices} "
                    f"(up {len(lower)} + down {len(upper)}), indices={slice_indices}\n"
                )
                f.write(f"YZ slice at x = {x_slice}\n")
                f.write(f"XZ slice at y = {y_slice}\n")
                f.write(f"XY grid layout: {rows} rows x {cols} columns\n")
        
        with open(os.path.join(out_dir, "uids.txt"), "w") as f:
            f.write("_".join(f"{u:08d}" for u in input_uids))

@torch.no_grad()
def export_metrics(
    result: edict,
):  
    """
    eval during training：体素 L2/PSNR/SSIM，逐样本计算后再 batch 求和。
    """
    target = result.input.voxel
    prediction = result.predicted_voxel

    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    
    count = target.size(0)
    l2_values = compute_voxel_mse_per_sample(target, prediction)
    psnr_values = compute_voxel_psnr_per_sample(target, prediction)
    ssim_values = compute_voxel_ssim_per_sample(target, prediction)

    return {
        "l2": float(l2_values.sum()),
        "psnr": float(psnr_values.sum()),
        "ssim": float(ssim_values.sum()),
        "count": count,
    }
def _save_images(result, batch_idx, out_dir):
    """保存 input 实部/虚部拼图；若有预测体素则保存 gt/predicted 的 npy 与同内容的 tif。"""
    def _to_uint8_single_channel(x: torch.Tensor) -> np.ndarray:
        """
        x: [v, 1, h, w]
        返回: [H, W] 的 uint8 图像 (灰度)
        """
        if x.size(0) > 24:
            idx = torch.linspace(0, x.size(0) - 1, steps=24, device=x.device).round().long()
            x = x.index_select(0, idx)
        x = _normalize_image_tensor_for_uint8(x.detach().float().cpu())
        x = rearrange(x, "v c h w -> h (v w) c")      # [H, W, 1]
        x = (x.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        x = x.squeeze(-1)                             # -> [H, W]
        return x

    # ---------- 保存 input 的实部 / 虚部 ----------
    # result.input.image[batch_idx]: [v, 2, h, w]
    input_img = result.input.image[batch_idx]

    # 假定 channel 0 是实部, channel 1 是虚部
    input_real = input_img[:, 0:1, ...]  # [v, 1, h, w]
    input_imag = input_img[:, 1:1+1, ...]  # [v, 1, h, w]

    input_real_img = _to_uint8_single_channel(input_real)
    input_imag_img = _to_uint8_single_channel(input_imag)

    Image.fromarray(input_real_img).save(os.path.join(out_dir, "input_real.png"))
    Image.fromarray(input_imag_img).save(os.path.join(out_dir, "input_imag.png"))

    pv = getattr(result, "predicted_voxel", None)
    if pv is not None:
        gt_v = result.input.voxel[batch_idx].detach().float().cpu().numpy()
        pr_v = pv[batch_idx].detach().float().cpu().numpy()
        np.save(os.path.join(out_dir, "gt_voxel.npy"), gt_v)
        np.save(os.path.join(out_dir, "predicted_voxel.npy"), pr_v)
        gt_v32 = np.ascontiguousarray(gt_v.astype(np.float32, copy=False))
        pr_v32 = np.ascontiguousarray(pr_v.astype(np.float32, copy=False))
        tiff.imwrite(os.path.join(out_dir, "gt_voxel.tif"), gt_v32)
        tiff.imwrite(os.path.join(out_dir, "predicted_voxel.tif"), pr_v32)


def _save_metrics(target_voxel, pred_voxel, out_dir, scene_name):
    """体素指标：单样本输入为 [D, H, W]，这里补 batch 维为 [1, D, H, W]。"""
    t = target_voxel.to(torch.float32).unsqueeze(0)
    p = pred_voxel.to(torch.float32).unsqueeze(0)
    mse = compute_voxel_mse_per_sample(t, p)[0].item()
    psnr = compute_voxel_psnr_per_sample(t, p)[0].item()
    ssim = compute_voxel_ssim_per_sample(t, p)[0].item()

    metrics = {
        "summary": {
            "scene_name": scene_name,
            # 单场景标量：全体素 MSE 均值（与 loss 中体素项同一归约方式）
            "l2": float(mse),
            "psnr": float(psnr),
            "ssim": float(ssim),
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: os.path.basename(x),
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
