#!/usr/bin/env python3
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_INPUT_ROOTS = [
    "/gpfs/share/home/2401112587/lxy/odt_processdata_train_3d",
    "/gpfs/share/home/2401112587/lxy/odt_processdata_train_h1k293",
]


def _default_output_root(input_root: Path) -> Path:
    return input_root.with_name(f"{input_root.name}_fullframe")


def _read_full_list(path: Path):
    with path.open("r") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip() and not line.startswith("#")]


def _load_frame_stack(frames, key: str, frame_count: int, image_size: int) -> np.ndarray:
    stack = np.empty((frame_count, image_size, image_size, 2), dtype=np.float32)
    expected = (image_size, image_size, 2)
    for frame_idx, frame in enumerate(frames[:frame_count]):
        path = frame[key]
        arr = np.load(path, mmap_mode="r")
        if tuple(arr.shape) != expected:
            raise ValueError(f"{key} expected shape {expected}, got {arr.shape} at {path}")
        stack[frame_idx] = np.asarray(arr, dtype=np.float32)
    return stack


def _scene_complete(meta_path: Path, image_path: Path, background_path: Path, voxel_path: Path) -> bool:
    return meta_path.is_file() and image_path.is_file() and background_path.is_file() and voxel_path.is_file()


def _pack_scene(
    source_meta_path: Path,
    split_out: Path,
    frame_count: int,
    image_size: int,
    voxel_size,
    overwrite: bool,
):
    with source_meta_path.open("r") as f:
        meta = json.load(f)

    frames = meta.get("frames") or []
    if len(frames) < frame_count:
        raise ValueError(f"{source_meta_path} has {len(frames)} frames, expected at least {frame_count}")

    scene_name = meta.get("scene_name") or source_meta_path.stem
    scene_dir = split_out / "packed" / scene_name
    out_meta_path = split_out / "metadata" / f"{scene_name}.json"
    image_path = scene_dir / "image.npy"
    background_path = scene_dir / "background.npy"
    voxel_path = scene_dir / "voxel.npy"

    if not overwrite and _scene_complete(out_meta_path, image_path, background_path, voxel_path):
        return out_meta_path, "skipped"

    scene_dir.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    image_stack = _load_frame_stack(frames, "image_path", frame_count, image_size)
    background_stack = _load_frame_stack(frames, "background_path", frame_count, image_size)
    voxel = np.load(meta["voxel_path"], mmap_mode="r")
    if tuple(voxel.shape) != tuple(voxel_size):
        raise ValueError(f"voxel expected shape {tuple(voxel_size)}, got {voxel.shape} at {meta['voxel_path']}")

    np.save(image_path, np.ascontiguousarray(image_stack))
    np.save(background_path, np.ascontiguousarray(background_stack))
    np.save(voxel_path, np.asarray(voxel, dtype=np.float32))

    out_meta = {
        "scene_name": scene_name,
        "global_max_amp": meta.get("global_max_amp"),
        "global_max_bg_amp": meta.get("global_max_bg_amp"),
        "source_metadata_path": str(source_meta_path.resolve()),
        "source_voxel_path": meta.get("voxel_path"),
        "frame_count": frame_count,
        "image_size": image_size,
        "voxel_size": list(voxel_size),
        "image_path": str(image_path.resolve()),
        "background_path": str(background_path.resolve()),
        "voxel_path": str(voxel_path.resolve()),
    }
    with out_meta_path.open("w") as f:
        json.dump(out_meta, f, indent=2)
    return out_meta_path, "packed"


def _pack_scene_task(task):
    (
        scene_idx,
        source_meta_path,
        split_out,
        frame_count,
        image_size,
        voxel_size,
        overwrite,
    ) = task
    try:
        out_meta_path, status = _pack_scene(
            source_meta_path,
            split_out,
            frame_count=frame_count,
            image_size=image_size,
            voxel_size=voxel_size,
            overwrite=overwrite,
        )
        return {
            "scene_idx": scene_idx,
            "out_meta_path": str(out_meta_path.resolve()),
            "status": status,
            "error": None,
        }
    except Exception as exc:
        return {
            "scene_idx": scene_idx,
            "out_meta_path": None,
            "status": "failed",
            "error": f"{source_meta_path}: {exc}",
        }


def _pack_scene_chunk(tasks):
    return [_pack_scene_task(task) for task in tasks]


def _chunked(items, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _pack_split(input_root: Path, output_root: Path, split: str, args):
    full_list = input_root / split / "full_list.txt"
    if not full_list.is_file():
        print(f"[skip] {full_list} not found")
        return

    scene_paths = [Path(p) for p in _read_full_list(full_list)]
    if args.max_scenes is not None:
        scene_paths = scene_paths[: args.max_scenes]

    split_out = output_root / split
    split_out.mkdir(parents=True, exist_ok=True)
    out_full_list = split_out / "full_list.txt"
    voxel_size = tuple(int(x) for x in args.voxel_size)

    tasks = [
        (
            scene_idx,
            source_meta_path,
            split_out,
            args.frame_count,
            args.image_size,
            voxel_size,
            args.overwrite,
        )
        for scene_idx, source_meta_path in enumerate(scene_paths)
    ]

    out_paths_by_idx = {}
    packed = 0
    skipped = 0
    failed = 0

    def record(result):
        nonlocal packed, skipped, failed
        status = result["status"]
        packed += int(status == "packed")
        skipped += int(status == "skipped")
        failed += int(status == "failed")
        if result["out_meta_path"] is not None:
            out_paths_by_idx[int(result["scene_idx"])] = result["out_meta_path"]
        if result["error"] is not None:
            tqdm.write(f"[failed] {result['error']}")

    desc = f"{input_root.name}/{split}"
    if args.num_processes > 1:
        chunks = _chunked(tasks, args.chunk_size)
        with Pool(processes=args.num_processes) as pool:
            with tqdm(total=len(tasks), desc=desc, unit="scene") as pbar:
                for results in pool.imap_unordered(_pack_scene_chunk, chunks):
                    for result in results:
                        record(result)
                    pbar.update(len(results))
                    pbar.set_postfix(packed=packed, skipped=skipped, failed=failed)
    else:
        with tqdm(total=len(tasks), desc=desc, unit="scene") as pbar:
            for task in tasks:
                result = _pack_scene_task(task)
                record(result)
                pbar.update(1)
                pbar.set_postfix(packed=packed, skipped=skipped, failed=failed)

    out_paths = [out_paths_by_idx[idx] for idx in sorted(out_paths_by_idx)]
    with out_full_list.open("w") as f:
        f.write("\n".join(out_paths))
        if out_paths:
            f.write("\n")
    print(f"[done] wrote {out_full_list} scenes={len(out_paths)} packed={packed} skipped={skipped} failed={failed}")
    if failed:
        raise RuntimeError(f"{desc} failed for {failed} scene(s); see messages above.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack ODT full-frame image/background/voxel npy files per scene.")
    parser.add_argument("--input-root", action="append", default=None, help="Original ODT root. Repeatable.")
    parser.add_argument("--output-root", action="append", default=None, help="Output root. Repeatable; must match input roots.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to pack.")
    parser.add_argument("--frame-count", type=int, default=240)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--voxel-size", nargs=3, type=int, default=[128, 256, 256])
    parser.add_argument("--max-scenes", type=int, default=None, help="Pack only first N scenes per split.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing packed scenes.")
    parser.add_argument("--num-processes", type=int, default=16, help="Worker processes. Start small; each scene is large.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Scenes per multiprocessing task chunk.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_roots = [Path(p).expanduser().resolve() for p in (args.input_root or DEFAULT_INPUT_ROOTS)]
    if args.output_root is None:
        output_roots = [_default_output_root(root) for root in input_roots]
    else:
        output_roots = [Path(p).expanduser().resolve() for p in args.output_root]
        if len(output_roots) != len(input_roots):
            raise ValueError("--output-root count must match --input-root count")

    for input_root, output_root in zip(input_roots, output_roots):
        print(f"[root] {input_root} -> {output_root}")
        for split in args.splits:
            _pack_split(input_root, output_root, split, args)


if __name__ == "__main__":
    main()
