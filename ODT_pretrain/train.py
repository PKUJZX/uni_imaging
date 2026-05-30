
import importlib
import os
import time
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
# 添加TensorBoard导入
from torch.utils.tensorboard import SummaryWriter
from setup import init_config, init_distributed
from utils.metric_utils import visualize_intermediate_results,export_metrics
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0, print_model_parameters
from tqdm import tqdm


# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# 初始化TensorBoard (只在主进程中)
tensorboard_writer = None
if ddp_info.is_main_process:
    tensorboard_log_dir = os.path.join(config.training.checkpoint_dir, "tensorboard_logs",config.training.exp_name)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print_rank0(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

dist.barrier()

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=bool(config.training.get("pin_memory", False)),
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)
eval_dataset = Dataset(config, mode='eval')
# print(f"evaluation dataset length {len(eval_dataset)}")
eval_datasampler = DistributedSampler(eval_dataset, shuffle=False)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=bool(config.training.get("pin_memory", False)),
    drop_last=False,
    prefetch_factor=config.training.prefetch_factor,
    sampler=eval_datasampler,
)
eval_dataloader_iter = iter(eval_dataloader)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
# Frozen encoder parameters have requires_grad=False before DDP, so unused-parameter
# graph traversal is not needed unless a future model branch explicitly requires it.
find_unused = bool(config.training.get("find_unused_parameters", False))
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=find_unused)

optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
freeze_pretrained = config.training.get("freeze_pretrained", False)
# Exclude old decoder keys when loading pretrained weights (for infini decoder structure)
# Default to None, can be set in yaml config
exclude_keys = config.training.get("exclude_pretrained_keys", None)
print_rank0(f"exclude_pretrained_keys: {exclude_keys}")
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    freeze_pretrained=freeze_pretrained,
    exclude_keys=exclude_keys,
)

# Print final parameter information after all loading and freezing is done
# This ensures we show the correct trainable/frozen parameter status
if not freeze_pretrained:
    # Only print if we didn't already print in auto_resume_job
    print_model_parameters(model, "Model Parameters (final)")

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
else:
    scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()

# # 计算 target_has_input 切换点（前 1/5 步数）
# target_has_input_switch_step = total_train_steps // 5

# 添加TensorBoard日志记录函数
def log_to_tensorboard(writer, metrics, step, prefix="train"):
    """记录指标到TensorBoard"""
    if writer is None:
        return
    
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.item()
        writer.add_scalar(f"{prefix}/{key}", value, step)

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    total_grad_norm = None  # 初始化梯度范数，避免在非更新步骤时未定义
    try:
        data = next(dataloader_iter)##逐批次获得数据
    except StopIteration:##数据集遍历完开一个新的epoch
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)##每个epoch都重新打乱数据，且每次都不同
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}
    # eval during training
    eval_every = int(config.training.get("eval_every", 0))
    if eval_every > 0 and cur_train_step % eval_every == 0 and cur_train_step != 0:
        model.eval()
        
        metrics_sum = {"count": 0.0}
                    
        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            for eval_batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                eval_batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in eval_batch.items()}
                result = model(eval_batch, target_has_input=False)
                if result is None:
                    continue
                batch_metrics = export_metrics(result)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if k not in metrics_sum:
                        metrics_sum[k] = 0.0
                    metrics_sum[k] += v

            dist.barrier()
            # all reduce metrics
            for k, v in metrics_sum.items():
                tensor = torch.tensor(v, device=ddp_info.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics_sum[k] = tensor.item()

            # Compute averages across all batches
            for k in metrics_sum.keys():
                if k != "count" and metrics_sum["count"] > 0:
                    metrics_sum[k] /= metrics_sum["count"]
            
            # 不再在 key 里手动加 "eval/" 前缀，交给 log_to_tensorboard 的 prefix 参数来做
            metrics_to_log = {k: v for k, v in metrics_sum.items() if k != "count"}
            # if metrics_sum["count"] != total_batch_size:
            #     print(f"Warning: Evaluated {metrics_sum['count']} batches instead of {total_batch_size}.")

            # 只在主进程写 TensorBoard
            if ddp_info.is_main_process:
                # prefix="eval" -> 会写成 eval/psnr, eval/l2
                log_to_tensorboard(tensorboard_writer, metrics_to_log, cur_train_step, prefix="eval")
                print(f"Evaluation metrics: {metrics_to_log}")
        
        model.train()

    # # 动态设置 target_has_input：前 1/5 步为 True，之后为 False
    # current_target_has_input = (cur_train_step < target_has_input_switch_step)
    
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(batch)

    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        # with model.no_sync(): # no sync grads for efficiency
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()##累积梯度
    else:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    vis_every = int(config.training.get("vis_every", 0))
    export_inter_results = ((cur_train_step - 1) == start_train_step) or (
        vis_every > 0 and cur_train_step % vis_every == 0
    )

    if update_grads:##更新参数
        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if not skip_optimizer_step:
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        
            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                if total_grad_norm > config.training.grad_clip_norm * 2.0:
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

            # since skip flag may be updated because of grad norm, we check it again
            if not skip_optimizer_step:##更新一下参数
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # log and save checkpoint (添加TensorBoard日志记录)
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        
        # 记录到TensorBoard
        if tensorboard_writer is not None:
            # 基本训练指标
            tensorboard_metrics = {
                "learning_rate": optimizer.param_groups[0]['lr'],
                "iteration_time": time.time() - tic,
                "epoch": cur_epoch,
            }
            
            # 添加损失指标
            tensorboard_metrics.update(loss_dict)
            
            # 添加梯度范数（如果计算了的话）
            if total_grad_norm is not None:
                tensorboard_metrics["grad_norm"] = total_grad_norm
            
            # 添加缩放器状态
            if enable_grad_scaler:
                tensorboard_metrics["grad_scaler_scale"] = scaler.get_scale()
            
            # 记录所有指标（横轴使用 cur_train_step，对应每一次 forward/backward）
            log_to_tensorboard(tensorboard_writer, tensorboard_metrics, cur_train_step)
            
            # 每隔一定步数刷新写入
            if cur_train_step % config.training.get("tensorboard_flush_every", 100) == 0:
                tensorboard_writer.flush()
        
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print_str = f"[{now}] [Epoch {int(cur_epoch):>3d}] | Forward step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
            if total_grad_norm is not None:
                print_str += f" | Grad Norm: {total_grad_norm:.4f} | "
            print_str += " "
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict, save_arrays=False)
            torch.cuda.empty_cache()

    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()

# 关闭TensorBoard writer
if ddp_info.is_main_process and tensorboard_writer is not None:
    tensorboard_writer.close()
    print_rank0("TensorBoard writer closed")

dist.barrier()
dist.destroy_process_group()
