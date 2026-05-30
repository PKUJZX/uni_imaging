import importlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
from utils.odt_visualization import mae_metrics, save_mae_reconstruction
from utils.training_utils import print_rank0


def _amp_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[name]


def _get(config, dotted, default=None):
    cur = config
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _make_loader(dataset, config, sampler):
    kwargs = dict(
        dataset=dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=False,
        drop_last=False,
        sampler=sampler,
    )
    if config.training.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.training.prefetch_factor
    return DataLoader(**kwargs)


def _batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _load_model_checkpoint(model, checkpoint_path):
    module = model.module if isinstance(model, DDP) else model
    if hasattr(module, "load_ckpt"):
        return module.load_ckpt(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return checkpoint_path, module.load_state_dict(state, strict=False)


def _resolve_output_dir(config):
    return (
        _get(config, "inference.output_dir")
        or _get(config, "inference_out_dir")
        or os.path.join(config.training.checkpoint_dir, "inference")
    )


def _resolve_checkpoint(config):
    return _get(config, "inference.checkpoint_dir") or config.training.get("resume_ckpt", "") or config.training.checkpoint_dir


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    ddp_info = init_distributed(seed=int(config.training.get("seed", 777)))
    dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    task = str(_get(config, "inference.task", "voxel_main")).lower()
    out_dir = _resolve_output_dir(config)
    ckpt = _resolve_checkpoint(config)

    dataset_name = _get(config, "inference.dataset_name") or config.training.dataset_name
    dataset_module, dataset_class = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(dataset_module).__dict__[dataset_class]
    dataset = Dataset(config, mode="eval")
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = _make_loader(dataset, config, sampler)

    model_module, model_class = config.model.class_name.rsplit(".", 1)
    Model = importlib.import_module(model_module).__dict__[model_class]
    model = Model(config).to(ddp_info.device)
    model = DDP(model, device_ids=[ddp_info.local_rank])
    loaded_path, status = _load_model_checkpoint(model, ckpt)
    print_rank0(f"Loaded checkpoint from {loaded_path}; status={status}")

    if ddp_info.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        print_rank0(f"Running {task} inference; save results to: {out_dir}")
    dist.barrier()

    sampler.set_epoch(0)
    model.eval()
    mae_metric_sum = {"mae_masked_mse": 0.0, "mae_full_mse": 0.0, "count": 0}
    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=_amp_dtype(config.training.amp_dtype),
    ):
        for step, batch in enumerate(loader):
            batch = _batch_to_device(batch, ddp_info.device)
            result = model(batch)
            if task in ("mae_pretrain", "pretrain", "mae"):
                metrics = mae_metrics(result)
                for key, value in metrics.items():
                    mae_metric_sum[key] += value
                save_mae_reconstruction(
                    result,
                    os.path.join(out_dir, f"rank_{ddp_info.global_rank:03d}_batch_{step:06d}"),
                    prefix="mae",
                    max_items=int(_get(config, "inference.max_items_per_batch", 2)),
                )
            elif task in ("voxel_main", "voxel", "direct"):
                export_results(result, out_dir, compute_metrics=bool(_get(config, "inference.compute_metrics", True)))
            else:
                raise ValueError(f"Unknown inference.task={task!r}")
            torch.cuda.empty_cache()

    if task in ("mae_pretrain", "pretrain", "mae"):
        for key, value in mae_metric_sum.items():
            tensor = torch.tensor(value, device=ddp_info.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            mae_metric_sum[key] = tensor.item()
        if mae_metric_sum["count"] > 0:
            mae_metric_sum["mae_masked_mse"] /= mae_metric_sum["count"]
            mae_metric_sum["mae_full_mse"] /= mae_metric_sum["count"]
        if ddp_info.is_main_process:
            import json

            with open(os.path.join(out_dir, "mae_metrics.json"), "w") as f:
                json.dump(mae_metric_sum, f, indent=2)
            print_rank0(f"MAE metrics: {mae_metric_sum}")
    elif ddp_info.is_main_process and bool(_get(config, "inference.compute_metrics", True)):
        summarize_evaluation(out_dir)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
