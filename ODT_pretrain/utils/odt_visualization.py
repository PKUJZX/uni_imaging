import json
import os
from typing import Optional

import numpy as np
import torch


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _scale_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    vmin = float(np.percentile(x[finite], 0.1))
    vmax = float(np.percentile(x[finite], 99.9))
    if vmax <= vmin:
        vmax = float(x[finite].max())
        vmin = float(x[finite].min())
    if vmax <= vmin:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.nan_to_num(x, nan=vmin)
    return ((np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)) * 255.0).astype(np.uint8)


def _frame_patch_strip(frames: np.ndarray, channel: int) -> np.ndarray:
    # frames: [T, 2, P, P] -> [T, P*P]
    return frames[:, channel].reshape(frames.shape[0], -1)


def save_mae_reconstruction(
    result,
    out_dir: str,
    prefix: str = "mae",
    max_items: Optional[int] = 2,
    save_arrays: bool = True,
    save_summary: bool = True,
    combine_batch: bool = False,
):
    """Save MAE GT / masked / reconstruction strips for real and imag channels."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    image = _to_numpy(result.input.image)
    pred = _to_numpy(result.pred_image)
    mask = _to_numpy(result.mask)
    crop_xy = _to_numpy(result.input.get("crop_xy", np.zeros((image.shape[0], 2), dtype=np.int64)))
    scene_names = result.input.get("scene_name", [str(i) for i in range(image.shape[0])])

    b = image.shape[0] if max_items is None or int(max_items) <= 0 else min(int(max_items), image.shape[0])
    summaries = []
    if combine_batch:
        summaries = [
            {"index": bi, "scene_name": scene_names[bi], "crop_yx": crop_xy[bi].tolist()}
            for bi in range(b)
        ]
        for ch, name in [(0, "real"), (1, "imag")]:
            scene_strips = []
            for bi in range(b):
                gt = _frame_patch_strip(image[bi], ch)
                rec = _frame_patch_strip(pred[bi], ch)
                masked = gt.copy()
                masked[mask[bi] > 0.5] = np.nan
                scene_strips.append(np.concatenate([gt, masked, rec, np.abs(rec - gt)], axis=1))
            if scene_strips:
                batch_grid = np.concatenate(scene_strips, axis=0)
                Image.fromarray(_scale_to_uint8(batch_grid)).save(
                    os.path.join(out_dir, f"{prefix}_{name}_gt_mask_pred_diff_all.png")
                )
        if save_arrays:
            np.save(os.path.join(out_dir, "gt_image.npy"), image[:b])
            np.save(os.path.join(out_dir, "pred_image.npy"), pred[:b])
            np.save(os.path.join(out_dir, "mask.npy"), mask[:b])
        if save_summary:
            with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w") as f:
                json.dump(summaries, f, indent=2)
        return

    for bi in range(b):
        scene_dir = os.path.join(out_dir, f"{prefix}_{bi:02d}")
        os.makedirs(scene_dir, exist_ok=True)
        summaries.append({"index": bi, "scene_name": scene_names[bi], "crop_yx": crop_xy[bi].tolist()})

        for ch, name in [(0, "real"), (1, "imag")]:
            gt = _frame_patch_strip(image[bi], ch)
            rec = _frame_patch_strip(pred[bi], ch)
            masked = gt.copy()
            masked[mask[bi] > 0.5] = np.nan
            combined = np.concatenate([gt, masked, rec, np.abs(rec - gt)], axis=1)
            Image.fromarray(_scale_to_uint8(combined)).save(os.path.join(scene_dir, f"{name}_gt_mask_pred_diff.png"))

        if save_arrays:
            np.save(os.path.join(scene_dir, "gt_image.npy"), image[bi])
            np.save(os.path.join(scene_dir, "pred_image.npy"), pred[bi])
            np.save(os.path.join(scene_dir, "mask.npy"), mask[bi])

    if save_summary:
        with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)


@torch.no_grad()
def mae_metrics(result):
    image = result.input.image.float()
    pred = result.pred_image.float()
    mask = result.mask.float()
    token_mse = ((pred - image) ** 2).flatten(2).mean(dim=-1)
    masked_denom = mask.sum(dim=1).clamp_min(1.0)
    masked_mse_per_sample = (token_mse * mask).sum(dim=1) / masked_denom
    full_mse_per_sample = token_mse.mean(dim=1)
    return {
        "mae_masked_mse": float(masked_mse_per_sample.sum().detach().cpu()),
        "mae_full_mse": float(full_mse_per_sample.sum().detach().cpu()),
        "count": int(image.shape[0]),
    }
