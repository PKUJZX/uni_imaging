import json
import os

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
    vmin = float(np.percentile(x[finite], 1))
    vmax = float(np.percentile(x[finite], 99))
    if vmax <= vmin:
        vmin = float(x[finite].min())
        vmax = float(x[finite].max())
    if vmax <= vmin:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.nan_to_num(x, nan=vmin)
    x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _dp_context_grid(visible_images: np.ndarray, visible_ids: np.ndarray, dp_size: int) -> np.ndarray:
    grid = np.full((3 * dp_size, 3 * dp_size), np.nan, dtype=np.float32)
    for image, token_id in zip(visible_images, visible_ids):
        token_id = int(token_id)
        if token_id < 0 or token_id > 8:
            continue
        row = token_id // 3
        col = token_id % 3
        grid[row * dp_size : (row + 1) * dp_size, col * dp_size : (col + 1) * dp_size] = image
    return grid


def _save_png(path: str, array: np.ndarray):
    from PIL import Image

    Image.fromarray(_scale_to_uint8(array)).save(path)


def save_ptycho_mae_reconstruction(result, out_dir: str, prefix: str = "eval", max_items: int = 2):
    """Save context, target, prediction, and error PNGs for center-token MAE."""
    os.makedirs(out_dir, exist_ok=True)
    visible_images = _to_numpy(result.input.visible_images)
    visible_ids = _to_numpy(result.input.visible_token_ids)
    target = _to_numpy(result.input.target_image)
    pred = _to_numpy(result.pred_center)
    scan_xy = _to_numpy(result.input.get("scan_xy", np.zeros((visible_images.shape[0], 2), dtype=np.int64)))
    sample_names = result.input.get("sample_name", [str(i) for i in range(visible_images.shape[0])])

    max_items = min(int(max_items), visible_images.shape[0])
    summaries = []
    for item_idx in range(max_items):
        item_dir = os.path.join(out_dir, f"{prefix}_{item_idx:02d}")
        os.makedirs(item_dir, exist_ok=True)

        dp_size = int(target.shape[-1])
        context_grid = _dp_context_grid(visible_images[item_idx], visible_ids[item_idx], dp_size)
        gt = target[item_idx, 0]
        rec = pred[item_idx, 0]
        err = np.abs(rec - gt)
        strip = np.concatenate([gt, rec, err], axis=1)

        _save_png(os.path.join(item_dir, "visible_context_grid.png"), context_grid)
        _save_png(os.path.join(item_dir, "gt_pred_abs_error.png"), strip)
        if visible_images.shape[1] >= 10:
            probe_strip = np.concatenate([visible_images[item_idx, -2], visible_images[item_idx, -1]], axis=1)
            _save_png(os.path.join(item_dir, "probe_intensity_phase.png"), probe_strip)

        np.save(os.path.join(item_dir, "gt_center.npy"), gt)
        np.save(os.path.join(item_dir, "pred_center.npy"), rec)
        np.save(os.path.join(item_dir, "visible_images.npy"), visible_images[item_idx])
        summaries.append(
            {
                "index": item_idx,
                "sample_name": sample_names[item_idx] if item_idx < len(sample_names) else str(item_idx),
                "scan_yx": scan_xy[item_idx].astype(int).tolist(),
                "mse": float(np.mean((rec.astype(np.float64) - gt.astype(np.float64)) ** 2)),
            }
        )

    with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


@torch.no_grad()
def ptycho_mae_metric_sums(result):
    target = result.input.target_image.float()
    pred = result.pred_center.float()
    per_item = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return {
        "mae_center_mse_sum": float(per_item.sum().detach().cpu()),
        "count": int(per_item.numel()),
    }


def save_ptycho_pp_prediction(result, out_dir: str, prefix: str = "eval", max_items: int = 2):
    """Save GT / prediction / abs-error PNGs and arrays for projected potential."""
    os.makedirs(out_dir, exist_ok=True)
    target = _to_numpy(result.input.target)
    pred = _to_numpy(result.pred_pp)
    visible_images = _to_numpy(result.input.get("visible_images", np.zeros((target.shape[0], 0, 1, 1))))
    scan_xy = _to_numpy(result.input.get("scan_xy", np.zeros((target.shape[0], 2), dtype=np.int64)))
    sample_names = result.input.get("sample_name", [str(i) for i in range(target.shape[0])])

    max_items = min(int(max_items), target.shape[0])
    summaries = []
    for item_idx in range(max_items):
        item_dir = os.path.join(out_dir, f"{prefix}_{item_idx:02d}")
        os.makedirs(item_dir, exist_ok=True)

        gt = target[item_idx, 0]
        rec = pred[item_idx, 0]
        err = np.abs(rec - gt)
        strip = np.concatenate([gt, rec, err], axis=1)

        _save_png(os.path.join(item_dir, "gt_pred_abs_error.png"), strip)
        if visible_images.ndim == 4 and visible_images.shape[1] >= 2:
            dp_size = int(visible_images.shape[-1])
            visible_ids_obj = result.input.get("visible_token_ids", None)
            if visible_ids_obj is not None:
                visible_ids = _to_numpy(visible_ids_obj)[item_idx]
            else:
                visible_ids = np.arange(visible_images.shape[1])
            context_grid = _dp_context_grid(visible_images[item_idx], visible_ids, dp_size)
            _save_png(os.path.join(item_dir, "visible_context_grid.png"), context_grid)

        np.save(os.path.join(item_dir, "gt_pp.npy"), gt)
        np.save(os.path.join(item_dir, "pred_pp.npy"), rec)
        summaries.append(
            {
                "index": item_idx,
                "sample_name": sample_names[item_idx] if item_idx < len(sample_names) else str(item_idx),
                "scan_yx": scan_xy[item_idx].astype(int).tolist(),
                "mse": float(np.mean((rec.astype(np.float64) - gt.astype(np.float64)) ** 2)),
                "mae": float(np.mean(np.abs(rec.astype(np.float64) - gt.astype(np.float64)))),
            }
        )

    with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


@torch.no_grad()
def ptycho_pp_metric_sums(result):
    target = result.input.target.float()
    pred = result.pred_pp.float()
    se = ((pred - target) ** 2).flatten(1).mean(dim=1)
    ae = (pred - target).abs().flatten(1).mean(dim=1)
    return {
        "pp_mse_sum": float(se.sum().detach().cpu()),
        "pp_mae_sum": float(ae.sum().detach().cpu()),
        "count": int(se.numel()),
    }
