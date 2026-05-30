import contextlib
import importlib
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from pretrain_mae import (
    SummaryWriter,
    _autocast_context,
    _batch_to_device,
    _log_to_tensorboard,
    _make_loader,
    _reduce_metric_sums,
    _to_plain,
    create_lr_scheduler,
    create_optimizer,
    init_config,
    init_distributed,
    print_model_parameters,
    print_rank0,
)
from torch.utils.data import DistributedSampler

from utils.ptycho_visualization import ptycho_pp_metric_sums, save_ptycho_pp_prediction


def _save_checkpoint(model, optimizer, lr_scheduler, config, step, update_step):
    module = model.module if isinstance(model, DDP) else model
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": int(step),
        "param_update_step": int(update_step),
        "config": _to_plain(config),
    }
    ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{step:016}.pt")
    torch.save(checkpoint, ckpt_path)
    print_rank0(f"Saved Ptycho downstream checkpoint to {os.path.abspath(ckpt_path)}")


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
    find_unused = bool(config.model.get("freeze_encoder", True))
    if ddp_info.distributed:
        device_ids = [ddp_info.local_rank] if ddp_info.device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=find_unused)
    print_model_parameters(model, "Ptycho Downstream Model Parameters")

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
                msg = f"[Ptycho PP] step={step} update={update_step} lr={optimizer.param_groups[0]['lr']:.6g}"
                for key, value in result.loss_metrics.items():
                    msg += f" {key}={float(value.detach().cpu()):.6f}"
                print_rank0(msg)

        if step > 0 and step % int(config.training.eval_every) == 0:
            model.eval()
            metric_sums = {"pp_mse_sum": 0.0, "pp_mae_sum": 0.0, "count": 0.0}
            last_eval = None
            if eval_sampler is not None:
                eval_sampler.set_epoch(step)
            with torch.no_grad():
                for eval_batch in tqdm(eval_loader, desc="Ptycho PP eval", leave=False, disable=not ddp_info.is_main_process):
                    eval_batch = _batch_to_device(eval_batch, ddp_info.device)
                    with _autocast_context(config, ddp_info.device):
                        last_eval = model(eval_batch)
                    batch_sums = ptycho_pp_metric_sums(last_eval)
                    metric_sums["pp_mse_sum"] += batch_sums["pp_mse_sum"]
                    metric_sums["pp_mae_sum"] += batch_sums["pp_mae_sum"]
                    metric_sums["count"] += batch_sums["count"]
            metric_sums = _reduce_metric_sums(metric_sums, ddp_info.device)
            eval_metrics = {}
            if metric_sums["count"] > 0:
                eval_metrics["pp_mse"] = metric_sums["pp_mse_sum"] / metric_sums["count"]
                eval_metrics["pp_mae"] = metric_sums["pp_mae_sum"] / metric_sums["count"]
            if ddp_info.is_main_process:
                _log_to_tensorboard(writer, eval_metrics, step, prefix="eval")
                print_rank0(f"Ptycho PP eval metrics: {eval_metrics}")
                if last_eval is not None and step % int(config.training.vis_every) == 0:
                    vis_dir = os.path.join(config.training.checkpoint_dir, f"iter_{step:08d}")
                    save_ptycho_pp_prediction(
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
