import importlib
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from setup import init_config, init_distributed
from utils.odt_visualization import mae_metrics, save_mae_reconstruction
from utils.training_utils import auto_resume_job, create_lr_scheduler, print_model_parameters, print_rank0


def _amp_dtype(name: str):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }
    return mapping[name]


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _make_worker_init_fn(base_seed: int, global_rank: int):
    def _init(worker_id: int):
        seed = int(base_seed) + int(global_rank) * 100000 + int(worker_id)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    return _init


def _make_loader(dataset, config, sampler, drop_last: bool, generator=None, worker_init_fn=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=bool(config.training.get("pin_memory", True)),
        drop_last=drop_last,
        sampler=sampler,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    if config.training.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.training.prefetch_factor
    return DataLoader(**kwargs)


def _batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


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
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": step,
        "param_update_step": update_step,
    }
    ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{step:016}.pt")
    enc_path = os.path.join(config.training.checkpoint_dir, f"encoder_{step:016}.pt")
    torch.save(checkpoint, ckpt_path)
    torch.save(module.encoder.state_dict(), enc_path)
    print_rank0(f"Saved MAE checkpoint to {os.path.abspath(ckpt_path)}")
    print_rank0(f"Saved MAE encoder to {os.path.abspath(enc_path)}")


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    ddp_info = init_distributed(seed=int(config.training.get("seed", 777)))
    dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    writer = None
    if ddp_info.is_main_process:
        log_dir = os.path.join(config.training.checkpoint_dir, "tensorboard_logs", config.training.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print_rank0(f"TensorBoard logs will be saved to: {log_dir}")
    dist.barrier()

    module_name, class_name = config.training.dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module_name).__dict__[class_name]
    train_dataset = Dataset(config, mode="train")
    eval_dataset = Dataset(config, mode="eval")
    base_seed = int(config.training.get("seed", 777))
    train_sampler = DistributedSampler(train_dataset, seed=base_seed)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False, seed=base_seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(base_seed + ddp_info.global_rank * 100000)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(base_seed + ddp_info.global_rank * 100000 + 9999)
    worker_init_fn = _make_worker_init_fn(base_seed, ddp_info.global_rank)
    train_loader = _make_loader(
        train_dataset,
        config,
        train_sampler,
        drop_last=True,
        generator=train_generator,
        worker_init_fn=worker_init_fn,
    )
    eval_loader = _make_loader(
        eval_dataset,
        config,
        eval_sampler,
        drop_last=False,
        generator=eval_generator,
        worker_init_fn=worker_init_fn,
    )
    train_iter = iter(train_loader)

    model_module, model_class = config.model.class_name.rsplit(".", 1)
    Model = importlib.import_module(model_module).__dict__[model_class]
    model = Model(config).to(ddp_info.device)
    model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=False)
    print_model_parameters(model, "MAE Model Parameters")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    total_update_steps = int(config.training.train_steps)
    grad_accum = int(config.training.grad_accum_steps)
    total_forward_steps = total_update_steps * grad_accum
    lr_scheduler = create_lr_scheduler(
        optimizer,
        total_update_steps,
        config.training.warmup,
        scheduler_type=config.training.get("scheduler_type", "cosine"),
    )
    enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=enable_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)

    reset_training_state = _as_bool(config.training.get("reset_training_state", False))
    step = 0
    update_step = 0
    if reset_training_state:
        print_rank0(
            "reset_training_state=True; starting MAE training from step 0 without loading an existing checkpoint."
        )
    else:
        ckpt_load_path = config.training.get("resume_ckpt", "") or config.training.checkpoint_dir
        optimizer, lr_scheduler, step, update_step = auto_resume_job(
            ckpt_load_path,
            model,
            optimizer,
            lr_scheduler,
            reset_training_state=False,
        )

    model.train()
    while step < total_forward_steps:
        tic = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch = int(update_step * config.training.batch_size_per_gpu * ddp_info.world_size / len(train_dataset))
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _batch_to_device(batch, ddp_info.device)
        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=_amp_dtype(config.training.amp_dtype),
        ):
            result = model(batch)
            loss = result.loss_metrics.loss / grad_accum

        should_update = (step + 1) % grad_accum == 0 or step == total_forward_steps
        if should_update:
            scaler.scale(loss).backward()
        else:
            with model.no_sync():
                scaler.scale(loss).backward()
        step += 1

        if should_update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                config.training.grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            update_step += 1
        else:
            grad_norm = None

        if ddp_info.is_main_process:
            metrics = {k: float(v.detach().cpu()) for k, v in result.loss_metrics.items()}
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            iter_time = time.time() - tic
            metrics["iteration_time"] = iter_time
            if grad_norm is not None:
                metrics["grad_norm"] = float(grad_norm)
            _log_to_tensorboard(writer, metrics, step, prefix="train")
            if step % config.training.print_every == 0 or step < 100:
                cur_epoch = int(
                    update_step * config.training.batch_size_per_gpu * ddp_info.world_size / len(train_dataset)
                )
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                msg = (
                    f"[{now}] [MAE] | Epoch {cur_epoch:>3d} | "
                    f"Forward step: {step:>6d} (Param update step: {update_step:>6d})"
                    f" | Iter Time: {iter_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6g}"
                )
                if grad_norm is not None:
                    msg += f" | Grad Norm: {float(grad_norm):.4f}"
                for key, value in result.loss_metrics.items():
                    msg += f" | {key}: {float(value.detach().cpu()):.6f}"
                msg += " |"
                print_rank0(msg)

        vis_every = int(config.training.get("vis_every", 0))
        if ddp_info.is_main_process and vis_every > 0 and step % vis_every == 0:
            vis_dir = os.path.join(config.training.checkpoint_dir, f"iter_{step:08d}")
            save_mae_reconstruction(
                result,
                vis_dir,
                prefix="train",
                max_items=4,
                save_arrays=False,
                save_summary=False,
                combine_batch=True,
            )

        eval_every = int(config.training.get("eval_every", 0))
        if eval_every > 0 and step > 0 and step % eval_every == 0:
            model.eval()
            metrics_sum = {"mae_masked_mse": 0.0, "mae_full_mse": 0.0, "count": 0}
            with torch.no_grad(), torch.autocast(
                enabled=config.training.use_amp,
                device_type="cuda",
                dtype=_amp_dtype(config.training.amp_dtype),
            ):
                for eval_batch in tqdm(eval_loader, desc="MAE eval", leave=False):
                    eval_batch = _batch_to_device(eval_batch, ddp_info.device)
                    eval_result = model(eval_batch)
                    batch_metrics = mae_metrics(eval_result)
                    for key, value in batch_metrics.items():
                        metrics_sum[key] += value
            for key, value in metrics_sum.items():
                tensor = torch.tensor(value, device=ddp_info.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics_sum[key] = tensor.item()
            if metrics_sum["count"] > 0:
                for key in ("mae_masked_mse", "mae_full_mse"):
                    metrics_sum[key] /= metrics_sum["count"]
            if ddp_info.is_main_process:
                _log_to_tensorboard(writer, {k: v for k, v in metrics_sum.items() if k != "count"}, step, prefix="eval")
                print_rank0(f"MAE eval metrics: {metrics_sum}")
            model.train()
            dist.barrier()

        if ddp_info.is_main_process and (
            step % config.training.checkpoint_every == 0 or step == total_forward_steps
        ):
            _save_checkpoint(model, optimizer, lr_scheduler, config, step, update_step)

    if ddp_info.is_main_process and writer is not None:
        writer.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
