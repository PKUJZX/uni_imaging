# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
try:
    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
except Exception:
    from torch.optim.lr_scheduler import LambdaLR

    def _warmup_lambda(current_step, warmup_steps):
        if warmup_steps <= 0:
            return 1.0
        return min(float(current_step) / float(max(1, warmup_steps)), 1.0)

    def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
        return LambdaLR(optimizer, lambda step: _warmup_lambda(step, num_warmup_steps))

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(step):
            warm = _warmup_lambda(step, num_warmup_steps)
            if step < num_warmup_steps:
                return warm
            remain = max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
            return remain
        return LambdaLR(optimizer, lr_lambda)

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        import math

        def lr_lambda(step):
            if step < num_warmup_steps:
                return _warmup_lambda(step, num_warmup_steps)
            progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def print_model_parameters(model, title="Model Parameters"):
    """
    Print model parameter information including trainable and frozen parameters.
    This should be called after any parameter freezing to show accurate information.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]

        all_param_dict = {name: param for name, param in model.named_parameters()}
        trainable_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trainable_param_dict.values())
        frozen_params = total_params - trainable_params

        trainable_module_names = sorted(set(get_module_name(name) for name in trainable_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(trainable_param_dict.keys())))

        print(f'\n{title}:')
        print(f'  Total parameters: {format_number(total_params)}')
        print(f'  Trainable parameters: {format_number(trainable_params)}')
        print(f'  Frozen parameters: {format_number(frozen_params)}')
        print(f'  Trainable modules: {trainable_module_names}')
        print(f'  Frozen modules: {frozen_module_names}')
        print()

def create_optimizer(model, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    decay_params, nodecay_params = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    
    # Print Model Information
    if dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]  # Fallback to first part if no 'module.' prefix
        print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}')
        # Number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
        
        print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')
        
    return optimizer, optimized_param_dict, all_param_dict

def create_optimizer_all_params(model, weight_decay, learning_rate, betas):
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = all_param_dict

    decay_params, nodecay_params = [], []
    for _, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    if not dist.is_initialized() or dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values() if p.requires_grad)
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name, param in all_param_dict.items() if not param.requires_grad))
        print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}')
        print(f'Total parameters in optimizer: {format_number(total_params)}, Currently trainable parameters: {format_number(trainable_params)}')
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')

    return optimizer, optimized_param_dict, all_param_dict

def get_param_group_name(name):
    if name.startswith("module."):
        name = name[len("module."):]

    if (
        name.startswith("voxel_pos_tokenizer.")
        or name.startswith("voxel_token_decoder.")
        or name.startswith("latent_to_voxel_decoder.")
    ):
        return "voxel_decoder"
    if name.startswith("transformer_input_layernorm."):
        return "transformer_input_layernorm"
    if name.startswith("image_tokenizer."):
        return "image_tokenizer"
    if name.startswith("transformer_blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_idx = int(parts[1])
            if 18 <= layer_idx <= 23:
                return "transformer_blocks_18_23"
            if 12 <= layer_idx <= 17:
                return "transformer_blocks_12_17"
            if 0 <= layer_idx <= 11:
                return "transformer_blocks_0_11"
    return "other"


def create_optimizer_with_lr_multipliers(
    model,
    weight_decay,
    base_lr,
    betas,
    lr_multipliers=None,
):
    if lr_multipliers is None:
        lr_multipliers = {
            "voxel_decoder": 1.0,
            "transformer_input_layernorm": 0.3,
            "transformer_blocks_18_23": 0.2,
            "transformer_blocks_12_17": 0.1,
            "transformer_blocks_0_11": 0.05,
            "image_tokenizer": 0.1,
            "other": 1.0,
        }

    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = all_param_dict
    grouped_params = {}

    for name, param in optimized_param_dict.items():
        group_name = get_param_group_name(name)
        wd_name = "nodecay" if param.dim() == 1 or getattr(param, "_no_weight_decay", False) else "decay"
        grouped_params.setdefault((group_name, wd_name), []).append(param)

    optim_groups = []
    for (group_name, wd_name), params in grouped_params.items():
        lr_multiplier = lr_multipliers[group_name]
        optim_groups.append(
            {
                "params": params,
                "weight_decay": 0.0 if wd_name == "nodecay" else weight_decay,
                "lr": base_lr * lr_multiplier,
                "group_name": group_name,
                "decay_type": wd_name,
                "lr_multiplier": lr_multiplier,
            }
        )

    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=base_lr, betas=betas, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(optim_groups, lr=base_lr, betas=betas)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Optimizer: AdamW, base learning rate: {base_lr}, weight decay: {weight_decay}, betas: {betas}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values() if p.requires_grad)
        print(
            f"Total parameters in optimizer: {format_number(total_params)}, "
            f"Currently trainable parameters: {format_number(trainable_params)}"
        )
        for group in optim_groups:
            group_params = sum(p.numel() for p in group["params"])
            trainable_group_params = sum(p.numel() for p in group["params"] if p.requires_grad)
            print(
                f"  group={group['group_name']}/{group['decay_type']} "
                f"params={format_number(group_params)} "
                f"trainable={format_number(trainable_group_params)} "
                f"lr={group['lr']}"
            )

    return optimizer, optimized_param_dict, all_param_dict

def maybe_update_train_stage(model, cur_param_update_step, total_param_update_steps, config):
    del config
    if cur_param_update_step < 3000:
        stage_name = "stage1"
    elif cur_param_update_step < int(0.3 * total_param_update_steps):
        stage_name = "stage2"
    elif cur_param_update_step < int(0.6 * total_param_update_steps):
        stage_name = "stage3"
    else:
        stage_name = "stage4"

    model_module = model.module if isinstance(model, DDP) else model
    if not hasattr(model_module, "set_train_stage"):
        return None

    stage_changed = model_module.set_train_stage(stage_name)
    if stage_changed:
        print_rank0(
            f"Switch train stage to {stage_name} at param_update_step={cur_param_update_step} "
            f"(total_param_update_steps={total_param_update_steps})"
        )
        print_model_parameters(model, title=f"Model Parameters ({stage_name})")
    return stage_name

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [
            file_name
            for file_name in os.listdir(load_path)
            if file_name.startswith("ckpt_") and file_name.endswith(".pt")
        ]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    exclude_keys=None,
    freeze_pretrained=False
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state
        exclude_keys: list of key prefixes to exclude when loading (e.g., ['target_pose_tokenizer'])
        freeze_pretrained: whether to freeze pretrained weights after loading

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]##最新的权重
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise KeyError(f"{ckpt_path} is not a full training checkpoint with a 'model' state")
    except Exception:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    model_state_dict = checkpoint["model"]
    if exclude_keys is not None:
        model_state_dict = {
            k: v for k, v in model_state_dict.items()
            if not any(k.startswith(excluded_key) for excluded_key in exclude_keys)
        }
        print_rank0(f"Loading pretrained weights from {os.path.abspath(ckpt_path)} (excluding {exclude_keys})")

    if isinstance(model, DDP):
        status = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        status = model.load_state_dict(model_state_dict, strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    # Freeze pretrained weights if requested
    if freeze_pretrained:
        model_module = model.module if isinstance(model, DDP) else model
        if hasattr(model_module, 'freeze_pretrained_weights'):
            model_module.freeze_pretrained_weights()
            print_model_parameters(model, "Model Parameters (after freezing)")
        else:
            print_rank0("Warning: freeze_pretrained=True but model does not have freeze_pretrained_weights method")

    # resume training state
    if not reset_training_state:###是否重置训练状态，不重置则继续
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step
