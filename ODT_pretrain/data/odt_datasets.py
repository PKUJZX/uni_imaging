import glob
import json
import os
import random
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_path_list(path_or_paths) -> List[str]:
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
        p = os.path.expandvars(os.path.expanduser(str(item).strip()))
        if not p:
            continue
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.txt"))))
        elif any(ch in p for ch in "*?[]"):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)

    seen = set()
    paths = []
    for p in out:
        if os.path.isfile(p) and p.lower().endswith(".txt") and p not in seen:
            paths.append(p)
            seen.add(p)
    return paths


def _load_scene_paths(config, mode: str) -> List[str]:
    training = config.training
    field = "dataset_path" if mode == "train" else "eval_dataset_path"
    list_files = _normalize_path_list(_cfg_get(training, field))
    if not list_files:
        raise FileNotFoundError(f"No valid .txt files resolved from training.{field}")

    scene_paths = []
    for list_file in list_files:
        with open(list_file, "r") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    scene_paths.append(line)

    seen = set()
    unique_paths = []
    for p in scene_paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)
    return unique_paths


def _frame_limit(config) -> int:
    model_cfg = config.model
    return int(_cfg_get(model_cfg, "frame_count", 240))


def _image_size(config) -> int:
    model_cfg = config.model
    tokenizer = _cfg_get(model_cfg, "image_tokenizer")
    if tokenizer is not None:
        return int(_cfg_get(tokenizer, "image_size", _cfg_get(model_cfg, "image_size", 256)))
    return int(_cfg_get(model_cfg, "image_size", 256))


def _patch_size(config) -> int:
    model_cfg = config.model
    tokenizer = _cfg_get(model_cfg, "image_tokenizer")
    if tokenizer is not None:
        return int(_cfg_get(tokenizer, "patch_size", _cfg_get(model_cfg, "patch_size", 16)))
    return int(_cfg_get(model_cfg, "patch_size", 16))


def _safe_scale(value, fallback: float = 1.0) -> float:
    if value is None:
        return fallback
    scale = float(value)
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return fallback
    return scale


def _check_normalized_complex_range(name: str, array: np.ndarray, path: str, tol: float = 1e-6):
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values after normalization at {path}")

    amin = float(array.min())
    amax = float(array.max())
    amp_max = float(np.sqrt(np.sum(array.astype(np.float32, copy=False) ** 2, axis=-1)).max())
    limit = 1.0 + float(tol)
    if amin < -limit or amax > limit or amp_max > limit:
        raise ValueError(
            f"{name} expected normalized real/imag in [-1, 1] and amplitude <= 1; "
            f"got component range [{amin:.6g}, {amax:.6g}], max amplitude {amp_max:.6g} at {path}. "
            "Check global_max_amp/global_max_bg_amp in scene metadata."
        )


def _load_normalized_frame(frame, image_scale: float, bg_scale: float):
    image = np.load(frame["image_path"]).astype(np.float32, copy=False)
    background = np.load(frame["background_path"]).astype(np.float32, copy=False)
    if image.ndim != 3 or image.shape[-1] != 2:
        raise ValueError(f"image must be [H,W,2], got {image.shape} at {frame['image_path']}")
    if background.ndim != 3 or background.shape[-1] != 2:
        raise ValueError(
            f"background must be [H,W,2], got {background.shape} at {frame['background_path']}"
        )
    image = image / np.float32(image_scale)
    background = background / np.float32(bg_scale)
    _check_normalized_complex_range("image", image, frame["image_path"])
    _check_normalized_complex_range("background", background, frame["background_path"])
    image = np.transpose(image, (2, 0, 1))
    background = np.transpose(background, (2, 0, 1))
    return image.astype(np.float32, copy=False), background.astype(np.float32, copy=False)


class _ODTSceneBase(Dataset):
    def __init__(self, config, mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode = "eval" if mode in ("eval", "val", "valid", "test") else "train"
        self.scene_paths = _load_scene_paths(config, self.mode)
        self.frame_count = _frame_limit(config)
        self.image_size = _image_size(config)
        self.patch_size = _patch_size(config)
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}"
            )

    def __len__(self):
        return len(self.scene_paths)

    def _load_metadata(self, idx: int):
        scene_path = self.scene_paths[idx]
        with open(scene_path, "r") as f:
            meta = json.load(f)
        frames = meta.get("frames") or []
        if len(frames) < self.frame_count:
            raise ValueError(
                f"{scene_path} has {len(frames)} frames, expected at least {self.frame_count}"
            )
        frames = frames[: self.frame_count]
        image_scale = _safe_scale(meta.get("global_max_amp"))
        bg_scale = _safe_scale(meta.get("global_max_bg_amp"))
        return scene_path, meta, frames, image_scale, bg_scale

    def _index_tensor(self, idx: int):
        frame_indices = torch.arange(self.frame_count, dtype=torch.long).unsqueeze(-1)
        scene_indices = torch.full_like(frame_indices, idx)
        return torch.cat([frame_indices, scene_indices], dim=-1)


class ODTMAEPackedPatchDataset(_ODTSceneBase):
    """ODT MAE pretraining data: one packed 240-frame 16x16 patch per sample."""

    def __init__(self, config, mode: str = "train"):
        super().__init__(config, mode=mode)
        self.spatial_grid = self.image_size // self.patch_size
        self.expected_patch_count = self.spatial_grid * self.spatial_grid

    def _load_patch_metadata(self, idx: int):
        scene_path = self.scene_paths[idx]
        with open(scene_path, "r") as f:
            meta = json.load(f)
        patches = meta.get("patches") or []
        if not patches:
            raise ValueError(f"{scene_path} has no packed patches")
        image_scale = _safe_scale(meta.get("global_max_amp"))
        bg_scale = _safe_scale(meta.get("global_max_bg_amp"))
        return scene_path, meta, patches, image_scale, bg_scale

    def _choose_patch(self, patches):
        if self.mode == "train" or bool(_cfg_get(_cfg_get(self.config, "inference"), "random_eval_crop", False)):
            return patches[random.randrange(len(patches))]
        inf_cfg = _cfg_get(self.config, "inference")
        patch_id = _cfg_get(inf_cfg, "patch_id")
        if patch_id is None:
            patch_id = (self.spatial_grid // 2) * self.spatial_grid + (self.spatial_grid // 2)
        patch_id = int(patch_id)
        for patch in patches:
            if int(patch.get("patch_id", -1)) == patch_id:
                return patch
        raise ValueError(f"patch_id={patch_id} not found; available patches={len(patches)}")

    def _load_patch_array(self, path: str, scale: float, name: str) -> np.ndarray:
        raw = np.load(path, mmap_mode="r")
        expected = (self.frame_count, self.patch_size, self.patch_size, 2)
        if tuple(raw.shape) != expected:
            raise ValueError(f"{name} patch must be {expected}, got {raw.shape} at {path}")
        patch = np.asarray(raw, dtype=np.float32) / np.float32(scale)
        _check_normalized_complex_range(name, patch, path)
        patch = np.transpose(patch, (0, 3, 1, 2))
        return patch.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        scene_path, meta, patches, image_scale, bg_scale = self._load_patch_metadata(idx)
        patch = self._choose_patch(patches)
        patch_id = int(patch.get("patch_id", -1))
        y0 = int(patch.get("patch_y", (patch_id // self.spatial_grid) * self.patch_size))
        x0 = int(patch.get("patch_x", (patch_id % self.spatial_grid) * self.patch_size))

        image = self._load_patch_array(patch["image_path"], image_scale, "image patch")
        background = self._load_patch_array(patch["background_path"], bg_scale, "background patch")

        # Single-sample shapes; DataLoader collation prepends batch dim B.
        # image/background: [T, 2, P, P], T=frame_count, P=patch_size, 2=real/imag
        # index: [T, 2] with columns [frame_idx, scene_idx]
        # crop_xy: [2] as [y0, x0]; global_max_*: scalar tensors; scene_name: str
        return {
            "image": torch.from_numpy(image),
            "background": torch.from_numpy(background),
            "index": self._index_tensor(idx),
            "scene_name": meta.get("scene_name", os.path.splitext(os.path.basename(scene_path))[0]),
            "crop_xy": torch.tensor([y0, x0], dtype=torch.long),
            "global_max_amp": torch.tensor(float(image_scale), dtype=torch.float32),
            "global_max_bg_amp": torch.tensor(float(bg_scale), dtype=torch.float32),
        }


class ODTFullPatchVoxelDataset(_ODTSceneBase):
    """ODT direct inversion data: all 240 full frames plus the 3D voxel target."""

    def __getitem__(self, idx: int):
        scene_path, meta, frames, image_scale, bg_scale = self._load_metadata(idx)

        images = []
        backgrounds = []
        for frame in frames:
            image, background = _load_normalized_frame(frame, image_scale, bg_scale)
            if image.shape[-2:] != (self.image_size, self.image_size):
                raise ValueError(f"Expected image {self.image_size}x{self.image_size}, got {image.shape}")
            images.append(image)
            backgrounds.append(background)

        voxel = np.load(meta["voxel_path"]).astype(np.float32, copy=False)
        voxel_size = tuple(int(x) for x in _cfg_get(self.config.model, "voxel_size", [128, 256, 256]))
        if voxel.shape != voxel_size:
            raise ValueError(f"Expected voxel shape {voxel_size}, got {voxel.shape} at {meta['voxel_path']}")

        # Single-sample shapes; DataLoader collation prepends batch dim B.
        # image/background: [T, 2, H, W], T=frame_count, H=W=image_size, 2=real/imag
        # voxel: [D, Hv, Wv] from config.model.voxel_size
        # index: [T, 2] with columns [frame_idx, scene_idx]
        # global_max_*: scalar tensors; scene_name: str
        return {
            "image": torch.from_numpy(np.stack(images, axis=0)),
            "background": torch.from_numpy(np.stack(backgrounds, axis=0)),
            "voxel": torch.from_numpy(voxel),
            "index": self._index_tensor(idx),
            "scene_name": meta.get("scene_name", os.path.splitext(os.path.basename(scene_path))[0]),
            "global_max_amp": torch.tensor(float(image_scale), dtype=torch.float32),
            "global_max_bg_amp": torch.tensor(float(bg_scale), dtype=torch.float32),
        }


class ODTFullFramePackedVoxelDataset(_ODTSceneBase):
    """ODT direct inversion data packed as one full-frame image/background/voxel npy per scene."""

    def _load_packed_metadata(self, idx: int):
        scene_path = self.scene_paths[idx]
        with open(scene_path, "r") as f:
            meta = json.load(f)
        image_scale = _safe_scale(meta.get("global_max_amp"))
        bg_scale = _safe_scale(meta.get("global_max_bg_amp"))
        return scene_path, meta, image_scale, bg_scale

    def _load_full_frame_array(self, path: str, scale: float, name: str) -> np.ndarray:
        raw = np.load(path, mmap_mode="r")
        expected = (self.frame_count, self.image_size, self.image_size, 2)
        if tuple(raw.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {raw.shape} at {path}")
        x = np.asarray(raw, dtype=np.float32) / np.float32(scale)
        _check_normalized_complex_range(name, x, path)
        x = np.transpose(x, (0, 3, 1, 2))
        return np.ascontiguousarray(x, dtype=np.float32)

    def __getitem__(self, idx: int):
        scene_path, meta, image_scale, bg_scale = self._load_packed_metadata(idx)

        image = self._load_full_frame_array(meta["image_path"], image_scale, "packed image")
        background = self._load_full_frame_array(meta["background_path"], bg_scale, "packed background")

        voxel = np.load(meta["voxel_path"], mmap_mode="r")
        voxel_size = tuple(int(x) for x in _cfg_get(self.config.model, "voxel_size", [128, 256, 256]))
        if tuple(voxel.shape) != voxel_size:
            raise ValueError(f"Expected voxel shape {voxel_size}, got {voxel.shape} at {meta['voxel_path']}")
        voxel = np.array(voxel, dtype=np.float32, copy=True)

        # Single-sample shapes; DataLoader collation prepends batch dim B.
        # image/background: [T, 2, H, W], T=frame_count, H=W=image_size, 2=real/imag
        # voxel: [D, Hv, Wv] from config.model.voxel_size
        # index: [T, 2] with columns [frame_idx, scene_idx]
        # global_max_*: scalar tensors; scene_name: str
        return {
            "image": torch.from_numpy(image),
            "background": torch.from_numpy(background),
            "voxel": torch.from_numpy(np.ascontiguousarray(voxel)),
            "index": self._index_tensor(idx),
            "scene_name": meta.get("scene_name", os.path.splitext(os.path.basename(scene_path))[0]),
            "global_max_amp": torch.tensor(float(image_scale), dtype=torch.float32),
            "global_max_bg_amp": torch.tensor(float(bg_scale), dtype=torch.float32),
        }
