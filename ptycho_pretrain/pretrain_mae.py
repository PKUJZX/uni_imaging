import argparse
import contextlib
import datetime
import importlib
import math
import os
import random
import re
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from utils.ptycho_visualization import ptycho_mae_metric_sums, save_ptycho_mae_reconstruction


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class AttrDict(dict):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        mapping = {} if mapping is None else dict(mapping)
        mapping.update(kwargs)
        for key, value in mapping.items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return AttrDict(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)


def _to_plain(obj: Any):
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(item) for item in obj]
    return obj


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _process_overrides(overrides):
    combined = " ".join(overrides)
    fixed = re.sub(r"(\S+)\s*=\s*(\S+)", r"\1=\2", combined)
    return re.findall(r"[^\s=]+=\S+|\S+", fixed)


def _set_by_dotted_key(config: dict, dotted_key: str, value):
    cur = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def init_config():
    parser = argparse.ArgumentParser(description="Ptycho center-token MAE pretraining")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Optional dotted overrides, e.g. training.train_steps=20")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for item in _process_overrides(args.overrides):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        _set_by_dotted_key(config, key, yaml.safe_load(raw_value))
    return AttrDict(config)


def _distributed_requested() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed(seed: int):
    distributed = _distributed_requested()
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=3600))
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        global_rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    process_seed = int(seed) + global_rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.benchmark = True

    return AttrDict(
        distributed=distributed,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
        is_main_process=global_rank == 0,
        seed=process_seed,
    )


def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def _format_number(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def print_model_parameters(model, title="Model Parameters"):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\n{title}:")
    print(f"  Total parameters: {_format_number(total)}")
    print(f"  Trainable parameters: {_format_number(trainable)}")
    print(f"  Frozen parameters: {_format_number(total - trainable)}\n")


def _amp_dtype(name: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported amp dtype: {name!r}")
    return mapping[key]


def create_optimizer(model, config):
    params = [param for param in model.parameters() if param.requires_grad]
    try:
        return torch.optim.AdamW(
            params,
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
            fused=torch.cuda.is_available(),
        )
    except TypeError:
        return torch.optim.AdamW(
            params,
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )


def create_lr_scheduler(optimizer, total_steps: int, warmup_steps: int, scheduler_type: str = "cosine"):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    scheduler_type = str(scheduler_type).lower()

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if scheduler_type == "constant":
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_loader(dataset, config, sampler, shuffle: bool, drop_last: bool, device):
    num_workers = int(config.training.num_workers)
    kwargs = dict(
        dataset=dataset,
        batch_size=int(config.training.batch_size_per_gpu),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(config.training.get("pin_memory", True)) and device.type == "cuda",
        drop_last=drop_last,
        sampler=sampler,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.training.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.training.get("prefetch_factor", 4))
    return DataLoader(**kwargs)


def _batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _log_to_tensorboard(writer, metrics, step, prefix="train"):
    if writer is None:
        return
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.item()
        writer.add_scalar(f"{prefix}/{key}", value, step)


def _save_checkpoint(model, optimizer, lr_scheduler, config, step, update_step):
    module = model.module if isinstance(model, DDP) else model
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": module.state_dict(),
        "encoder": module.encoder.state_dict(),
        "decoder": module.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": int(step),
        "param_update_step": int(update_step),
        "config": _to_plain(config),
    }
    ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{step:016}.pt")
    enc_path = os.path.join(config.training.checkpoint_dir, f"encoder_{step:016}.pt")
    dec_path = os.path.join(config.training.checkpoint_dir, f"decoder_{step:016}.pt")
    torch.save(checkpoint, ckpt_path)
    torch.save(module.encoder.state_dict(), enc_path)
    torch.save(module.decoder.state_dict(), dec_path)
    print_rank0(f"Saved Ptycho MAE checkpoint to {os.path.abspath(ckpt_path)}")
    print_rank0(f"Saved Ptycho MAE encoder to {os.path.abspath(enc_path)}")


def _reduce_metric_sums(metrics, device):
    if not dist.is_initialized():
        return metrics
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(float(value), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item()
    return reduced


def _autocast_context(config, device):
    enabled = bool(config.training.use_amp) and device.type == "cuda"
    return torch.autocast(
        device_type=device.type,
        enabled=enabled,
        dtype=_amp_dtype(config.training.amp_dtype),
    )


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    ddp_info = init_distributed(seed=int(config.training.get("seed", 777)))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.training.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.training.use_tf32)
        try:
            torch.set_float32_matmul_precision("high" if config.training.use_tf32 else "highest")
        except Exception:
            pass

    writer = None
    if ddp_info.is_main_process and SummaryWriter is not None:
        log_dir = os.path.join(config.training.checkpoint_dir, "tensorboard_logs", config.training.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir, flush_secs=int(config.training.get("tensorboard_flush_secs", 120)))
        print_rank0(f"TensorBoard logs will be saved to: {log_dir}")
    elif ddp_info.is_main_process:
        print_rank0("TensorBoard is not available; scalar logging will only go to stdout.")

    module_name, class_name = config.training.dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module_name).__dict__[class_name]
    train_dataset = Dataset(config, mode="train")
    eval_dataset = Dataset(config, mode="eval")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp_info.distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if ddp_info.distributed else None
    train_loader = _make_loader(
        train_dataset,
        config,
        train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        device=ddp_info.device,
    )
    eval_loader = _make_loader(
        eval_dataset,
        config,
        eval_sampler,
        shuffle=False,
        drop_last=False,
        device=ddp_info.device,
    )
    train_iter = iter(train_loader)

    model_module, model_class = config.model.class_name.rsplit(".", 1)
    Model = importlib.import_module(model_module).__dict__[model_class]
    model = Model(config).to(ddp_info.device)
    if ddp_info.distributed:
        device_ids = [ddp_info.local_rank] if ddp_info.device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False)
    print_model_parameters(model, "Ptycho MAE Model Parameters")

    optimizer = create_optimizer(model, config)
    total_update_steps = int(config.training.train_steps)
    grad_accum = int(config.training.grad_accum_steps)
    total_forward_steps = total_update_steps * grad_accum
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_update_steps,
        int(config.training.warmup),
        scheduler_type=config.training.get("scheduler_type", "cosine"),
    )
    use_scaler = (
        bool(config.training.use_amp)
        and str(config.training.amp_dtype).lower() == "fp16"
        and ddp_info.device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    step = 0
    update_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step < total_forward_steps:
        tic = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(update_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _batch_to_device(batch, ddp_info.device)
        should_update = (step + 1) % grad_accum == 0 or (step + 1) == total_forward_steps
        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not should_update:
            sync_context = model.no_sync()

        with sync_context:
            with _autocast_context(config, ddp_info.device):
                result = model(batch)
                loss = result.loss_metrics.loss / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        step += 1
        grad_norm = None
        if should_update:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad and param.grad is not None],
                float(config.training.grad_clip_norm),
            )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            update_step += 1

        if ddp_info.is_main_process:
            metrics = {key: float(value.detach().cpu()) for key, value in result.loss_metrics.items()}
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            metrics["iteration_time"] = time.time() - tic
            if grad_norm is not None:
                metrics["grad_norm"] = float(grad_norm)
            _log_to_tensorboard(writer, metrics, step, prefix="train")
            if step % int(config.training.print_every) == 0 or step <= 10:
                msg = f"[Ptycho MAE] step={step} update={update_step} lr={optimizer.param_groups[0]['lr']:.6g}"
                for key, value in result.loss_metrics.items():
                    msg += f" {key}={float(value.detach().cpu()):.6f}"
                print_rank0(msg)

        if step > 0 and step % int(config.training.eval_every) == 0:
            model.eval()
            metric_sums = {"mae_center_mse_sum": 0.0, "count": 0.0}
            last_eval = None
            if eval_sampler is not None:
                eval_sampler.set_epoch(step)
            with torch.no_grad():
                for eval_batch in tqdm(eval_loader, desc="Ptycho MAE eval", leave=False, disable=not ddp_info.is_main_process):
                    eval_batch = _batch_to_device(eval_batch, ddp_info.device)
                    with _autocast_context(config, ddp_info.device):
                        last_eval = model(eval_batch)
                    batch_sums = ptycho_mae_metric_sums(last_eval)
                    metric_sums["mae_center_mse_sum"] += batch_sums["mae_center_mse_sum"]
                    metric_sums["count"] += batch_sums["count"]
            metric_sums = _reduce_metric_sums(metric_sums, ddp_info.device)
            eval_metrics = {}
            if metric_sums["count"] > 0:
                eval_metrics["mae_center_mse"] = metric_sums["mae_center_mse_sum"] / metric_sums["count"]
            if ddp_info.is_main_process:
                _log_to_tensorboard(writer, eval_metrics, step, prefix="eval")
                print_rank0(f"Ptycho MAE eval metrics: {eval_metrics}")
                if last_eval is not None and step % int(config.training.vis_every) == 0:
                    vis_dir = os.path.join(config.training.checkpoint_dir, f"iter_{step:08d}")
                    save_ptycho_mae_reconstruction(
                        last_eval,
                        vis_dir,
                        prefix="eval",
                        max_items=int(config.training.get("max_vis_items", 2)),
                    )
            model.train()
            if dist.is_initialized():
                dist.barrier()

        if ddp_info.is_main_process and (
            step % int(config.training.checkpoint_every) == 0 or step == total_forward_steps
        ):
            _save_checkpoint(model, optimizer, lr_scheduler, config, step, update_step)

    if ddp_info.is_main_process and writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
