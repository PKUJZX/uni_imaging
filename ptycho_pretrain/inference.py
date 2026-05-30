import importlib
import json
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from pretrain_mae import (
    _autocast_context,
    _batch_to_device,
    _cfg_get,
    _make_loader,
    _reduce_metric_sums,
    init_config,
    init_distributed,
    print_rank0,
)
from utils.ptycho_visualization import ptycho_pp_metric_sums, save_ptycho_pp_prediction


def _resolve_output_dir(config):
    inference_cfg = _cfg_get(config, "inference", {})
    return (
        _cfg_get(inference_cfg, "output_dir", None)
        or os.path.join(config.training.checkpoint_dir, "inference")
    )


def _resolve_checkpoint(config):
    inference_cfg = _cfg_get(config, "inference", {})
    return (
        _cfg_get(inference_cfg, "checkpoint_dir", None)
        or config.training.get("resume_ckpt", "")
        or config.training.checkpoint_dir
    )


def _load_model_checkpoint(model, checkpoint_path):
    module = model.module if isinstance(model, DDP) else model
    if hasattr(module, "load_ckpt"):
        return module.load_ckpt(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return checkpoint_path, module.load_state_dict(state, strict=False)


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    ddp_info = init_distributed(seed=int(config.training.get("seed", 777)))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.training.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.training.use_tf32)

    inference_cfg = _cfg_get(config, "inference", {})
    out_dir = _resolve_output_dir(config)
    ckpt = _resolve_checkpoint(config)

    dataset_name = _cfg_get(inference_cfg, "dataset_name", None) or config.training.dataset_name
    dataset_module, dataset_class = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(dataset_module).__dict__[dataset_class]
    dataset = Dataset(config, mode="eval")
    sampler = DistributedSampler(dataset, shuffle=False) if ddp_info.distributed else None
    loader = _make_loader(dataset, config, sampler, shuffle=False, drop_last=False, device=ddp_info.device)

    model_module, model_class = config.model.class_name.rsplit(".", 1)
    Model = importlib.import_module(model_module).__dict__[model_class]
    model = Model(config).to(ddp_info.device)
    if ddp_info.distributed:
        device_ids = [ddp_info.local_rank] if ddp_info.device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)
    loaded_path, status = _load_model_checkpoint(model, ckpt)
    print_rank0(f"Loaded checkpoint from {loaded_path}; status={status}")

    if ddp_info.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        print_rank0(f"Running Ptycho projected-potential inference; results -> {out_dir}")
    if dist.is_initialized():
        dist.barrier()

    if sampler is not None:
        sampler.set_epoch(0)
    model.eval()
    metric_sums = {"pp_mse_sum": 0.0, "pp_mae_sum": 0.0, "count": 0.0}
    max_items = int(_cfg_get(inference_cfg, "max_items_per_batch", 2))
    with torch.no_grad(), _autocast_context(config, ddp_info.device):
        for step, batch in enumerate(loader):
            batch = _batch_to_device(batch, ddp_info.device)
            result = model(batch)
            batch_sums = ptycho_pp_metric_sums(result)
            metric_sums["pp_mse_sum"] += batch_sums["pp_mse_sum"]
            metric_sums["pp_mae_sum"] += batch_sums["pp_mae_sum"]
            metric_sums["count"] += batch_sums["count"]
            save_ptycho_pp_prediction(
                result,
                os.path.join(out_dir, f"rank_{ddp_info.global_rank:03d}_batch_{step:06d}"),
                prefix="pp",
                max_items=max_items,
            )

    metric_sums = _reduce_metric_sums(metric_sums, ddp_info.device)
    if ddp_info.is_main_process:
        summary = {}
        if metric_sums["count"] > 0:
            mse = metric_sums["pp_mse_sum"] / metric_sums["count"]
            mae = metric_sums["pp_mae_sum"] / metric_sums["count"]
            summary = {
                "pp_mse": mse,
                "pp_mae": mae,
                "pp_psnr": float(-10.0 * math.log10(mse)) if mse > 0 else float("inf"),
                "count": int(metric_sums["count"]),
            }
        with open(os.path.join(out_dir, "pp_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print_rank0(f"Ptycho PP metrics: {summary}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
