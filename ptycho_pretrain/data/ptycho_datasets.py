import glob
import os
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_path_list(path_or_paths, extensions: Sequence[str]) -> List[str]:
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, str):
        candidates: Iterable[str] = path_or_paths.split(",") if "," in path_or_paths else [path_or_paths]
    elif isinstance(path_or_paths, (list, tuple)):
        candidates = path_or_paths
    else:
        raise TypeError(f"dataset path must be str/list/tuple, got {type(path_or_paths)}")

    out = []
    for item in candidates:
        path = os.path.expandvars(os.path.expanduser(str(item).strip()))
        if not path:
            continue
        if os.path.isdir(path):
            for ext in extensions:
                out.extend(sorted(glob.glob(os.path.join(path, f"*{ext}"))))
        elif any(ch in path for ch in "*?[]"):
            out.extend(sorted(glob.glob(path)))
        else:
            out.append(path)

    seen = set()
    paths = []
    ext_lut = tuple(ext.lower() for ext in extensions)
    for path in out:
        if os.path.isfile(path) and path.lower().endswith(ext_lut) and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _sample_group_names(h5_file: h5py.File, dp_key: str) -> List[str]:
    if dp_key in h5_file:
        return [""]

    groups = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and dp_key in obj:
            groups.append(name)

    h5_file.visititems(visitor)
    return sorted(groups)


def _center_crop_2d(x: np.ndarray, size: int, name: str) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")
    h, w = x.shape
    if (h, w) == (size, size):
        return x
    if h < size or w < size:
        raise ValueError(f"{name} shape {x.shape} is smaller than requested {size}x{size}")
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return x[y0 : y0 + size, x0 : x0 + size]


def _zscore(x: np.ndarray, eps: float) -> Tuple[np.ndarray, float, float]:
    mean = float(x.mean())
    std = float(x.std())
    if not np.isfinite(std) or std < eps:
        std = 1.0
    return ((x - np.float32(mean)) / np.float32(std)).astype(np.float32, copy=False), mean, std


def _normalize_array(x: np.ndarray, kind: str, eps: float, scale: float = 1.0):
    kind = str(kind or "none").lower()
    x = x.astype(np.float32, copy=False)
    if kind in ("none", "raw", "identity"):
        return x, {"kind": kind, "mean": 0.0, "std": 1.0, "scale": 1.0}
    if kind in ("scale", "mul", "multiply"):
        out = (x * np.float32(scale)).astype(np.float32, copy=False)
        return out, {"kind": kind, "mean": 0.0, "std": 1.0, "scale": float(scale)}
    if kind in ("zscore", "meanstd"):
        out, mean, std = _zscore(x, eps)
        return out, {"kind": kind, "mean": mean, "std": std, "scale": 1.0}
    if kind in ("log1p_zscore", "log_zscore", "log1p"):
        logged = np.log1p(np.clip(x, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
        out, mean, std = _zscore(logged, eps)
        return out, {"kind": kind, "mean": mean, "std": std, "scale": 1.0}
    raise ValueError(f"Unsupported normalization kind: {kind!r}")


class PtychoCenterMAEDataset(Dataset):
    """Center-token MAE dataset for single-parameter 4D-STEM HDF5 samples.

    Each item is a 3x3 scan neighborhood. The center diffraction pattern
    (token 4) is the reconstruction target; the surrounding 8 diffraction
    patterns plus optional probe intensity/phase are visible context tokens.
    """

    def __init__(self, config, mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode = "eval" if mode in ("eval", "val", "valid", "test") else "train"
        self.data_cfg = _cfg_get(config, "data", {})
        self.model_cfg = _cfg_get(config, "model", {})
        self.training_cfg = _cfg_get(config, "training", {})

        self.dp_key = str(_cfg_get(self.data_cfg, "dp_key", "diffraction_patterns"))
        self.probe_intensity_key = str(_cfg_get(self.data_cfg, "probe_intensity_key", "probe_intensity"))
        self.probe_phase_key = str(_cfg_get(self.data_cfg, "probe_phase_key", "probe_phase"))
        self.projected_potential_key = str(_cfg_get(self.data_cfg, "projected_potential_key", "projected_potential"))

        self.dp_size = int(_cfg_get(self.model_cfg, "dp_size", _cfg_get(self.data_cfg, "dp_size", 64)))
        self.scan_patch_size = int(_cfg_get(self.model_cfg, "scan_patch_size", 3))
        if self.scan_patch_size != 3:
            raise ValueError("Ptycho center MAE currently expects scan_patch_size=3")
        self.num_dp_tokens = self.scan_patch_size * self.scan_patch_size
        self.center_token_id = int(_cfg_get(self.model_cfg, "center_token_id", self.num_dp_tokens // 2))
        if self.center_token_id != self.num_dp_tokens // 2:
            raise ValueError("Ptycho center MAE expects center_token_id=4 for a 3x3 scan patch")
        self.use_probe_tokens = bool(_cfg_get(self.model_cfg, "use_probe_tokens", True))
        self.num_tokens = self.num_dp_tokens + (2 if self.use_probe_tokens else 0)
        self.center_offset = self.scan_patch_size // 2

        norm_cfg = _cfg_get(self.data_cfg, "normalization", {})
        self.dp_norm = str(_cfg_get(norm_cfg, "dp", "log1p_zscore"))
        self.probe_norm = str(_cfg_get(norm_cfg, "probe", "zscore"))
        self.dp_scale = float(_cfg_get(norm_cfg, "dp_scale", 1.0))
        self.probe_scale = float(_cfg_get(norm_cfg, "probe_scale", 1.0))
        self.norm_eps = float(_cfg_get(norm_cfg, "eps", 1e-6))

        self.require_pp_bounds = self._require_pp_bounds()
        self.pp_patch_size = int(_cfg_get(self.data_cfg, "projected_potential_patch_size", 16))
        self.pp_scan_upsample = int(_cfg_get(self.data_cfg, "projected_potential_scan_upsample", 10))

        field = "dataset_path" if self.mode == "train" else "eval_dataset_path"
        paths = _normalize_path_list(_cfg_get(self.training_cfg, field), (".h5", ".hdf5"))
        if not paths:
            raise FileNotFoundError(f"No valid HDF5 files resolved from training.{field}")

        self.index = []
        for path in paths:
            self._append_file_index(path)
        if not self.index:
            raise ValueError(f"No valid 3x3 Ptycho MAE windows found in {paths}")

        self._h5_cache = {}

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_h5_cache", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def _require_pp_bounds(self) -> bool:
        return bool(_cfg_get(self.data_cfg, "require_projected_potential_bounds", False))

    def _append_file_index(self, path: str):
        with h5py.File(path, "r") as h5_file:
            group_names = _sample_group_names(h5_file, self.dp_key)
            if not group_names:
                raise KeyError(f"No groups containing {self.dp_key!r} found in {path}")
            for group_name in group_names:
                group = h5_file if group_name == "" else h5_file[group_name]
                if self.dp_key not in group:
                    continue
                dp_shape = tuple(group[self.dp_key].shape)
                if len(dp_shape) != 4:
                    raise ValueError(f"{path}:{group_name or '/'} {self.dp_key} must be [scan_y,scan_x,H,W], got {dp_shape}")
                scan_y, scan_x, height, width = dp_shape
                if (height, width) != (self.dp_size, self.dp_size):
                    raise ValueError(
                        f"{path}:{group_name or '/'} diffraction pattern size must be "
                        f"{self.dp_size}x{self.dp_size}, got {(height, width)}"
                    )
                if scan_y < self.scan_patch_size or scan_x < self.scan_patch_size:
                    continue
                for y0 in range(scan_y - self.scan_patch_size + 1):
                    for x0 in range(scan_x - self.scan_patch_size + 1):
                        if self._projected_potential_window_is_valid(group, y0, x0):
                            self.index.append((path, group_name, y0, x0))

    def _projected_potential_window_is_valid(self, group, y0: int, x0: int) -> bool:
        if not self.require_pp_bounds:
            return True
        if self.projected_potential_key not in group:
            return False
        pp_shape = tuple(group[self.projected_potential_key].shape)
        if len(pp_shape) < 2:
            return False
        pp_h, pp_w = pp_shape[-2], pp_shape[-1]
        center_y = (y0 + self.center_offset) * self.pp_scan_upsample
        center_x = (x0 + self.center_offset) * self.pp_scan_upsample
        half = self.pp_patch_size // 2
        return (
            center_y - half >= 0
            and center_x - half >= 0
            and center_y - half + self.pp_patch_size <= pp_h
            and center_x - half + self.pp_patch_size <= pp_w
        )

    def _get_group(self, path: str, group_name: str):
        handle = self._h5_cache.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._h5_cache[path] = handle
        return handle if group_name == "" else handle[group_name]

    def __getitem__(self, idx: int):
        path, group_name, y0, x0 = self.index[idx]
        group = self._get_group(path, group_name)

        dp_patch = np.asarray(
            group[self.dp_key][y0 : y0 + self.scan_patch_size, x0 : x0 + self.scan_patch_size],
            dtype=np.float32,
        )
        dp_tokens = dp_patch.reshape(self.num_dp_tokens, self.dp_size, self.dp_size)
        dp_tokens, dp_stats = _normalize_array(dp_tokens, self.dp_norm, self.norm_eps, self.dp_scale)

        visible_dp_ids = [token_id for token_id in range(self.num_dp_tokens) if token_id != self.center_token_id]
        visible_images = [dp_tokens[token_id] for token_id in visible_dp_ids]
        visible_token_ids = list(visible_dp_ids)

        all_images = [dp_tokens[token_id] for token_id in range(self.num_dp_tokens)]
        probe_stats = []
        if self.use_probe_tokens:
            probe_intensity = _center_crop_2d(
                np.asarray(group[self.probe_intensity_key], dtype=np.float32),
                self.dp_size,
                self.probe_intensity_key,
            )
            probe_phase = _center_crop_2d(
                np.asarray(group[self.probe_phase_key], dtype=np.float32),
                self.dp_size,
                self.probe_phase_key,
            )
            probe_intensity, intensity_stats = _normalize_array(
                probe_intensity, self.probe_norm, self.norm_eps, self.probe_scale
            )
            probe_phase, phase_stats = _normalize_array(probe_phase, self.probe_norm, self.norm_eps, self.probe_scale)
            visible_images.extend([probe_intensity, probe_phase])
            visible_token_ids.extend([self.num_dp_tokens, self.num_dp_tokens + 1])
            all_images.extend([probe_intensity, probe_phase])
            probe_stats = [intensity_stats, phase_stats]

        mask = np.zeros(self.num_tokens, dtype=np.float32)
        mask[self.center_token_id] = 1.0
        sample_name = group_name or os.path.splitext(os.path.basename(path))[0]
        center_y = y0 + self.center_offset
        center_x = x0 + self.center_offset

        return {
            "visible_images": torch.from_numpy(np.stack(visible_images, axis=0).astype(np.float32, copy=False)),
            "visible_token_ids": torch.tensor(visible_token_ids, dtype=torch.long),
            "target_image": torch.from_numpy(dp_tokens[self.center_token_id : self.center_token_id + 1].copy()),
            "target_token_id": torch.tensor(self.center_token_id, dtype=torch.long),
            "all_images": torch.from_numpy(np.stack(all_images, axis=0).astype(np.float32, copy=False)),
            "mask": torch.from_numpy(mask),
            "scan_xy": torch.tensor([center_y, center_x], dtype=torch.long),
            "window_xy": torch.tensor([y0, x0], dtype=torch.long),
            "sample_name": sample_name,
            "h5_path": path,
            "dp_norm_mean": torch.tensor(float(dp_stats["mean"]), dtype=torch.float32),
            "dp_norm_std": torch.tensor(float(dp_stats["std"]), dtype=torch.float32),
            "probe_norm_mean": torch.tensor(
                [float(item["mean"]) for item in probe_stats] if probe_stats else [], dtype=torch.float32
            ),
            "probe_norm_std": torch.tensor(
                [float(item["std"]) for item in probe_stats] if probe_stats else [], dtype=torch.float32
            ),
        }


class PtychoProjectedPotentialDataset(PtychoCenterMAEDataset):
    """Downstream projected-potential dataset for single-parameter 4D-STEM HDF5 samples.

    Each item provides the full 11-token sequence (9 diffraction patterns +
    probe intensity/phase) as ``inputs`` and the local projected potential
    patch around the center scan position as ``target``. Projected potential
    bounds are always required so every window has a valid supervision patch.
    """

    def __init__(self, config, mode: str = "train"):
        norm_cfg = _cfg_get(_cfg_get(config, "data", {}), "normalization", {})
        self.pp_norm = str(_cfg_get(norm_cfg, "pp", "zscore"))
        self.pp_scale = float(_cfg_get(norm_cfg, "pp_scale", 1.0))
        super().__init__(config, mode=mode)
        if self.projected_potential_key is None:
            raise ValueError("projected_potential_key must be set for the downstream dataset")

    def _require_pp_bounds(self) -> bool:
        return True

    def _read_pp_patch(self, group, y0: int, x0: int) -> np.ndarray:
        if self.projected_potential_key not in group:
            raise KeyError(f"{self.projected_potential_key!r} missing in HDF5 group for downstream supervision")
        pp = np.asarray(group[self.projected_potential_key], dtype=np.float32)
        if pp.ndim != 2:
            raise ValueError(f"projected_potential must be 2D, got shape {pp.shape}")
        center_y = (y0 + self.center_offset) * self.pp_scan_upsample
        center_x = (x0 + self.center_offset) * self.pp_scan_upsample
        half = self.pp_patch_size // 2
        y_start = center_y - half
        x_start = center_x - half
        patch = pp[y_start : y_start + self.pp_patch_size, x_start : x_start + self.pp_patch_size]
        if patch.shape != (self.pp_patch_size, self.pp_patch_size):
            raise ValueError(
                f"projected_potential patch at ({center_y},{center_x}) is out of bounds; got {patch.shape}"
            )
        return patch

    def __getitem__(self, idx: int):
        path, group_name, y0, x0 = self.index[idx]
        base = super().__getitem__(idx)
        group = self._get_group(path, group_name)

        pp_patch = self._read_pp_patch(group, y0, x0)
        pp_patch, pp_stats = _normalize_array(pp_patch, self.pp_norm, self.norm_eps, self.pp_scale)

        return {
            "inputs": base["all_images"],
            "target": torch.from_numpy(pp_patch[None].astype(np.float32, copy=False)),
            "visible_images": base["visible_images"],
            "visible_token_ids": base["visible_token_ids"],
            "scan_xy": base["scan_xy"],
            "window_xy": base["window_xy"],
            "sample_name": base["sample_name"],
            "h5_path": base["h5_path"],
            "pp_norm_mean": torch.tensor(float(pp_stats["mean"]), dtype=torch.float32),
            "pp_norm_std": torch.tensor(float(pp_stats["std"]), dtype=torch.float32),
        }
