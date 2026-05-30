import json
import os
import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _normalize_path_list(path_or_paths):
    """
    Normalize different path specifications into a flat list of valid .txt files.
    Supported input formats:
    - Single string: 'a.txt'
      - Comma separated: 'a.txt, b.txt'
      - Directory: '/path/to/dir'  -> expands to dir/*.txt
      - Glob pattern: './datasets/*/train/full_list.txt'
    - List or tuple: ['a.txt', 'b.txt', '/some/dir', './glob/*.txt']
    """
    if path_or_paths is None:
        return []

    def expand_one(p):
        p = os.path.expandvars(os.path.expanduser(str(p).strip()))
        if not p:
            return []
        # Directory -> all .txt files in it (non-recursive; for recursive use **/*.txt with recursive=True)
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, '*.txt')))
        # Glob pattern
        if any(ch in p for ch in '*?[]'):
            return sorted(glob.glob(p))
        # Normal file path
        return [p]

    paths = []
    if isinstance(path_or_paths, (list, tuple)):
        candidates = path_or_paths
    elif isinstance(path_or_paths, str):
        # Support comma separated string
        if ',' in path_or_paths:
            candidates = [s for s in path_or_paths.split(',')]
        else:
            candidates = [path_or_paths]
    else:
        raise TypeError(f"dataset path must be str or list, got {type(path_or_paths)}")

    for c in candidates:
        paths.extend(expand_one(c))

    # Keep only existing .txt files
    paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith('.txt')]
    # Deduplicate while preserving order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    return paths


class Dataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode

        # Select field name
        path_field = 'dataset_path' if mode == 'train' else 'eval_dataset_path'

        try:
            raw_paths = getattr(self.config.training, path_field)
        except AttributeError:
            raise AttributeError(f"config.training.{path_field} is missing")

        # Normalize into list of .txt files
        list_files = _normalize_path_list(raw_paths)
        if not list_files:
            raise FileNotFoundError(
                f"No valid .txt files resolved from config.training.{path_field}={raw_paths!r}"
            )

        # Read all list files and merge
        all_scene_paths = []
        for lf in list_files:
            try:
                with open(lf, 'r') as f:
                    lines = [ln.strip() for ln in f.read().splitlines()]
                    lines = [ln for ln in lines if ln and not ln.lstrip().startswith('#')]
                    all_scene_paths.extend(lines)
            except Exception as e:
                raise RuntimeError(f"Error reading list file: {lf}") from e

        # Deduplicate while preserving order
        seen = set()
        self.all_scene_paths = [p for p in all_scene_paths if not (p in seen or seen.add(p))]
        
        # Print resolved list files and scene count
        if torch.distributed.get_rank() == 0:
            print(f"Resolved list files ({path_field}):")
            for lf in list_files:
                print(f"  - {lf}")
            print(f"Number of scenes: {len(self.all_scene_paths)}")

    def __len__(self):
        return len(self.all_scene_paths)

    def preprocess_frames(self, frames_chosen, global_min=None, global_max=None):
        """
        预处理npy格式存储的二通道图像，归一化到0-1范围
        """
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        for cur_frame in frames_chosen:
            # 从npy文件加载二通道图像
            image_npy_path = cur_frame["image_path"]
            image = np.load(image_npy_path)  # shape: [H, W, 2] - 二通道图像
            
            original_image_h, original_image_w = image.shape[:2]
            
            # 计算resize后的宽度，确保能被patch_size整除
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            # 转换为tensor并调整维度: [H, W, 2] -> [2, H, W]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [2, H, W]
            
            # 使用torch进行resize
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),  # [1, 2, H, W]
                size=(resize_h, resize_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [2, H, W]

            # 如果需要方形裁剪
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image_tensor = image_tensor[:, start_h:start_h + min_size, start_w:start_w + min_size]

            # =============== 按通道全局归一化 ===============
            if global_min is not None and global_max is not None:
                # 转成 tensor，保证 dtype/device 一致
                gmin = torch.as_tensor(global_min, dtype=image_tensor.dtype, device=image_tensor.device)  # [2]
                gmax = torch.as_tensor(global_max, dtype=image_tensor.dtype, device=image_tensor.device)  # [2]
                denom = gmax - gmin  # [2]

                # 避免除零：对每个通道分别判断
                for c in range(image_tensor.shape[0]):  # 通常是 2
                    if denom[c] > 0:
                        image_tensor[c] = (image_tensor[c] - gmin[c]) / denom[c]

            
            if image_tensor.min() < 0 or image_tensor.max() > 1:
                print(
                    f"Image values out of [0, 1] range after normalization: "
                    f"min={image_tensor.min().item():.6f}, max={image_tensor.max().item():.6f}. "
                    f"Check normalization parameters or original data."
                )
            
            images.append(image_tensor)

        images = torch.stack(images, dim=0)  # [V, 2, H, W]
        return images

    def preprocess_backgrounds(self, frames_chosen, global_min_bg=None, global_max_bg=None):
        """
        预处理背景图像（两通道）
        
        Args:
            frames_chosen: 选择的帧列表
            global_min_bg: 背景全局最小值（长度为2的数组 [min_ch0, min_ch1]）
            global_max_bg: 背景全局最大值（长度为2的数组 [max_ch0, max_ch1]）
            
        Returns:
            torch.Tensor: 预处理后的背景图像 [V, 2, H, W]
        """
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        backgrounds = []
        for cur_frame in frames_chosen:
            # 从npy文件加载背景图像（两通道）
            bg_npy_path = cur_frame["background_path"]
            bg_image = np.load(bg_npy_path)  # shape: [H, W, 2] - 两通道图像
            
            original_image_h, original_image_w = bg_image.shape[:2]
            
            # 计算resize后的宽度，确保能被patch_size整除
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            # 转换为tensor并调整维度: [H, W, 2] -> [2, H, W]
            bg_tensor = torch.from_numpy(bg_image).permute(2, 0, 1).float()  # [2, H, W]
            
            # 使用torch进行resize
            bg_tensor = F.interpolate(
                bg_tensor.unsqueeze(0),  # [1, 2, H, W]
                size=(resize_h, resize_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [2, H, W]

            # 如果需要方形裁剪
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                bg_tensor = bg_tensor[:, start_h:start_h + min_size, start_w:start_w + min_size]

            # =============== 按通道全局归一化 ===============
            if global_min_bg is not None and global_max_bg is not None:
                # 转成 tensor，保证 dtype/device 一致
                gmin = torch.as_tensor(global_min_bg, dtype=bg_tensor.dtype, device=bg_tensor.device)  # [2]
                gmax = torch.as_tensor(global_max_bg, dtype=bg_tensor.dtype, device=bg_tensor.device)  # [2]
                denom = gmax - gmin  # [2]

                # 避免除零：对每个通道分别判断
                for c in range(bg_tensor.shape[0]):  # 通常是 2
                    if denom[c] > 0:
                        bg_tensor[c] = (bg_tensor[c] - gmin[c]) / denom[c]
            
            if bg_tensor.min() < 0 or bg_tensor.max() > 1:
                print(
                    f"Background values out of [0, 1] range after normalization: "
                    f"min={bg_tensor.min().item():.6f}, max={bg_tensor.max().item():.6f}. "
                    f"Check normalization parameters or original data."
                )   
            backgrounds.append(bg_tensor)

        backgrounds = torch.stack(backgrounds, dim=0)  # [V, 2, H, W]
        return backgrounds


    def preprocess_poses(self, poses_xy):
        """
        从球坐标的x,y计算极角的cos值
        """
        x, y = poses_xy[:, 0], poses_xy[:, 1]  # [num_views]
        
        # 从 x = sin(θ)cos(φ), y = sin(θ)sin(φ) 计算 cos(θ)
        sin_theta_squared = x**2 + y**2  # sin²(θ)
        sin_theta_squared = torch.clamp(sin_theta_squared, max=1.0)  # 数值稳定性
        
        # cos(θ) = sqrt(1 - sin²(θ))
        cos_theta = torch.sqrt(1 - sin_theta_squared)  # [num_views]
        
        # 返回 [x, y, cos(θ)]
        processed_poses = torch.stack([x, y, cos_theta], dim=1)  # [num_views, 3]
        
        return processed_poses
    

    def circular_even_sample_with_random_start(self, start: int, end: int, k: int):
        """
        在 [start, end] 闭区间（同一圈的索引序列）做"环形均匀采样"，
        但起点（相位）是随机的。
        """
        n = end - start + 1
        if k <= 0:
            return []
        if k >= n:
            return list(range(start, end + 1))

        offset = random.randrange(n)
        step = n / k

        chosen = []
        used = set()

        for i in range(k):
            pos = offset + i * step
            idx0 = int(round(pos)) % n
            idx = start + idx0

            if idx not in used:
                used.add(idx)
                chosen.append(idx)
                continue

            # 若冲突：在圈内就近找一个未用位置（双向扩张）
            found = False
            for d in range(1, n):
                cand1 = start + ((idx0 + d) % n)
                if cand1 not in used:
                    used.add(cand1)
                    chosen.append(cand1)
                    found = True
                    break
                cand2 = start + ((idx0 - d) % n)
                if cand2 not in used:
                    used.add(cand2)
                    chosen.append(cand2)
                    found = True
                    break
            if not found:
                pass

        # 如果由于极端冲突导致数量不足，随机补齐
        if len(chosen) < k:
            remain = [j for j in range(start, end + 1) if j not in used]
            need = k - len(chosen)
            chosen += random.sample(remain, need)

        return chosen


    def apply_jitter_in_ring(self, indices, start, end, prob=0.05, max_step=1):
        """
        小概率 jitter：每个 index 以 prob 的概率做 ±step（step<=max_step）。
        - 保证仍在 [start,end]
        - 尽量避免重复：冲突时回退到原 idx 或找圈内最近可用
        """
        n = end - start + 1
        used = set()
        out = []

        for idx in indices:
            new_idx = idx

            if random.random() < prob:
                step = random.randint(1, max_step)
                if random.random() < 0.5:
                    new_idx = idx + step
                else:
                    new_idx = idx - step
                new_idx = max(start, min(end, new_idx))

            if new_idx in used:
                # 优先尝试原 idx
                if idx not in used:
                    new_idx = idx
                else:
                    # 仍冲突：在圈内找最近可用
                    idx0 = new_idx - start
                    found = False
                    for d in range(1, n):
                        cand1 = start + ((idx0 + d) % n)
                        if cand1 not in used:
                            new_idx = cand1
                            found = True
                            break
                        cand2 = start + ((idx0 - d) % n)
                        if cand2 not in used:
                            new_idx = cand2
                            found = True
                            break
                    if not found:
                        continue

            used.add(new_idx)
            out.append(new_idx)

        # 如果 jitter 导致少了（很极端），补齐
        if len(out) < len(indices):
            remain = [j for j in range(start, end + 1) if j not in used]
            need = len(indices) - len(out)
            if need > 0 and len(remain) >= need:
                out += random.sample(remain, need)

        return out


    def view_selector(self, frames):
        """
        与 LVSM/data/dataset_scene 相同的四环角度采样 + jitter，但仅返回 input 视角索引
       （不追加 target 帧；batch 视角数 = num_input_views）。
        """
        total_frames = len(frames)
        num_input_views = self.config.training.num_input_views
        
        if total_frames < num_input_views:
            return None
        
        # 从config读取ring配置
        vs = getattr(self.config.training, "view_sampling", None)
        if vs is None or not hasattr(vs, "rings"):
            raise ValueError("需要在config.training.view_sampling.rings配置四圈的index range")
        
        rings = list(vs.rings)
        jitter_prob = float(getattr(vs, "jitter_prob", 0.05))
        jitter_max_step = int(getattr(vs, "jitter_max_step", 1))
        
        # 安全：裁剪ring到合法范围
        norm_rings = []
        for a, b in rings:
            a = max(0, min(int(a), total_frames - 1))
            b = max(0, min(int(b), total_frames - 1))
            if a > b:
                a, b = b, a
            norm_rings.append((a, b))
        
        # 1. 确定四个环的大小：[0-23], [24-71], [72-143], [144-239]
        ring_sizes = []
        for ring_start, ring_end in norm_rings:
            ring_size = ring_end - ring_start + 1
            ring_sizes.append(ring_size)
        
        # 2. 计算每个环的角度间隔（每个帧对应的角度）
        # 环1: 360/24 = 15度
        # 环2: 360/48 = 7.5度
        # 环3: 360/72 = 5度
        # 环4: 360/96 = 3.75度
        ring_angle_steps = [360.0 / size for size in ring_sizes]
        
        # 3. 计算输入视角的角度间隔
        angle_step = 360.0 / num_input_views
        
        # 4. 随机选择第一个视角
        # 先随机选择一个环，然后在该环随机选择一个帧
        first_ring_idx = random.randrange(len(norm_rings))
        ring_start, ring_end = norm_rings[first_ring_idx]
        first_idx = random.randint(ring_start, ring_end)
        
        # 计算第一个帧在环内的位置和对应的角度
        first_ring_pos = first_idx - ring_start
        first_angle = first_ring_pos * ring_angle_steps[first_ring_idx]
        
        input_indices = [first_idx]
        
        # 5. 为后续的每个角度位置选择帧
        for i in range(1, num_input_views):
            current_angle = (first_angle + i * angle_step) % 360
            
            # 对于这个角度位置，随机选择一个环（可以重复选择同一个环）
            # 但需要确保该环在当前角度位置有可用的帧
            available_rings = []
            for ring_idx in range(len(norm_rings)):
                ring_start, ring_end = norm_rings[ring_idx]
                ring_size = ring_sizes[ring_idx]
                
                # 计算在这个环上，当前角度对应的帧位置
                ring_pos = int(round(current_angle / ring_angle_steps[ring_idx])) % ring_size
                frame_idx = ring_start + ring_pos
                
                # 检查该帧是否已经被使用
                if frame_idx not in input_indices:
                    available_rings.append(ring_idx)
            
            # 如果有可用的环，随机选择一个
            if available_rings:
                chosen_ring_idx = random.choice(available_rings)
            else:
                # 如果没有可用的环，选择任意环，然后在该环内寻找可用帧
                chosen_ring_idx = random.randrange(len(norm_rings))
            
            # 获取选择的环的信息
            ring_start, ring_end = norm_rings[chosen_ring_idx]
            ring_size = ring_sizes[chosen_ring_idx]
            
            # 计算在该环上当前角度对应的帧位置
            ring_pos = int(round(current_angle / ring_angle_steps[chosen_ring_idx])) % ring_size
            frame_idx = ring_start + ring_pos
            
            # 如果帧已被使用，在该环内寻找最近可用帧
            if frame_idx in input_indices:
                found = False
                for d in range(1, ring_size):
                    # 向右搜索
                    cand1 = ring_start + ((ring_pos + d) % ring_size)
                    if cand1 not in input_indices:
                        frame_idx = cand1
                        found = True
                        break
                    # 向左搜索
                    cand2 = ring_start + ((ring_pos - d) % ring_size)
                    if cand2 not in input_indices:
                        frame_idx = cand2
                        found = True
                        break
                if not found:
                    # 如果这个环已满，尝试其他环
                    for other_ring_idx in range(len(norm_rings)):
                        if other_ring_idx == chosen_ring_idx:
                            continue
                        
                        other_start, other_end = norm_rings[other_ring_idx]
                        other_size = ring_sizes[other_ring_idx]
                        
                        # 在其他环的相同角度位置
                        other_pos = int(round(current_angle / ring_angle_steps[other_ring_idx])) % other_size
                        other_frame = other_start + other_pos
                        
                        if other_frame not in input_indices:
                            frame_idx = other_frame
                            chosen_ring_idx = other_ring_idx
                            found = True
                            break
            
            input_indices.append(frame_idx)
        
        # 6. 对每个环内的帧进行jitter
        # 由于帧可能来自不同环，需要分别对每个环内的帧进行jitter
        jittered_indices = []
        ring_frames_dict = {i: [] for i in range(len(norm_rings))}
        
        # 将帧按环分组
        for frame_idx in input_indices:
            for ring_idx, (ring_start, ring_end) in enumerate(norm_rings):
                if ring_start <= frame_idx <= ring_end:
                    ring_frames_dict[ring_idx].append(frame_idx)
                    break
        
        # 对每个环内的帧分别进行jitter
        for ring_idx in range(len(norm_rings)):
            if ring_frames_dict[ring_idx]:
                ring_start, ring_end = norm_rings[ring_idx]
                ring_frames = ring_frames_dict[ring_idx]
                
                # 对该环内的帧进行jitter
                jittered_frames = self.apply_jitter_in_ring(
                    ring_frames, ring_start, ring_end,
                    prob=jitter_prob, max_step=jitter_max_step
                )
                jittered_indices.extend(jittered_frames)
        
        # 确保jitter后没有重复（理论上不应该有）
        jittered_indices = list(dict.fromkeys(jittered_indices))
        
        # 如果jitter导致数量减少，补充随机帧
        if len(jittered_indices) < num_input_views:
            used = set(jittered_indices)
            # 收集所有未使用的帧（从所有环中）
            all_frames = []
            for ring_start, ring_end in norm_rings:
                for frame in range(ring_start, ring_end + 1):
                    if frame not in used:
                        all_frames.append(frame)
            
            need = num_input_views - len(jittered_indices)
            if len(all_frames) >= need:
                jittered_indices.extend(random.sample(all_frames, need))
            else:
                # 如果还是不够，重新生成所有输入视角
                jittered_indices = random.sample(range(total_frames), num_input_views)
        
        return jittered_indices


    def __getitem__(self, idx):
        try:
            scene_path = self.all_scene_paths[idx].strip()
            data_json = json.load(open(scene_path, 'r'))
            frames = data_json["frames"]
            scene_name = data_json["scene_name"]
            voxel_path = data_json["voxel_path"]
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # 计算全局的min/max值（用于归一化）
            global_min, global_max = data_json["global_min"], data_json["global_max"]
            global_min_bg, global_max_bg = data_json["global_min_bg"], data_json["global_max_bg"]

            # 根据选择的索引获取对应的frames
            frames_chosen = [frames[ic] for ic in image_indices]
            
            # 预处理图像和背景
            input_images = self.preprocess_frames(frames_chosen, global_min, global_max)
            backgrounds = self.preprocess_backgrounds(frames_chosen, global_min_bg, global_max_bg)
            
            # 从frames中提取pose数据
            poses_xy = []
            for frame in frames_chosen:
                poses_xy.append(frame["pose"])  # 每个pose是[x, y]的列表
            
            poses_xy = torch.tensor(poses_xy).float()  # [V, 2]
            
            # 预处理poses
            processed_poses = self.preprocess_poses(poses_xy)
            voxel = np.load(voxel_path).astype(np.float32)
            voxel_size_config = self.config.model.voxel_pos_tokenizer.voxel_size
            if voxel.shape != tuple(voxel_size_config):
                raise ValueError(f"Voxel shape incorrect in scene {scene_name}, got {voxel.shape}")
            voxel = torch.from_numpy(voxel).float() ##[192, 256, 256]
            if voxel.min() < 0 or voxel.max() > 1:
                print(
                    f"Voxel values out of [0, 1] range after normalization: "
                    f"min={voxel.min().item():.6f}, max={voxel.max().item():.6f}. "
                    f"Check normalization parameters or original data."
                )  

            # 创建索引信息
            image_indices_tensor = torch.tensor(image_indices).long().unsqueeze(-1)  # [V, 1]
            scene_indices = torch.full_like(image_indices_tensor, idx)  # [V, 1]
            indices = torch.cat([image_indices_tensor, scene_indices], dim=-1)  # [V, 2]

            return {
                "image": input_images,  # [V, 2, H, W]，V = num_input_views
                "background": backgrounds,  # [V, 2, H, W]
                "pose": processed_poses,  # [V, 3] - [x, y, cos_polar]
                "index": indices,  # [V, 2]
                "scene_name": scene_name,
                "voxel": voxel,  # [D, H, W]
                "global_min": torch.tensor(global_min).float(),  # [2]
                "global_max": torch.tensor(global_max).float(),  # [2]
                "global_min_bg": torch.tensor(global_min_bg).float(),
                "global_max_bg": torch.tensor(global_max_bg).float(),
            }
            
        except Exception as e:
            print(f"Error loading data for idx {idx}: {e}")
            print(f"Scene path: {self.all_scene_paths[idx] if idx < len(self.all_scene_paths) else 'Invalid index'}")
            # 返回一个随机的其他样本
            return self.__getitem__(random.randint(0, len(self) - 1))
