# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio



def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)



class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, pose, h, w, device="cuda", plane_offset=0.0):
        """
        正交相机射线：同一 view 内所有射线方向相同，起点固定在世界坐标 z=0 平面上。
        x,y 坐标用 linspace 映射到 [-1, 1]。

        Args:
            pose: [b, v, 3] forward方向（可非单位，内部会normalize）
            h,w: 图像尺寸
            plane_offset: 相机平面沿 forward 的平移（默认0：平面过原点）

        Returns:
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
        """
        b, v = pose.shape[:2]
        forward = pose  # [b,v,3]

        # # 选 up_ref，避免与 forward 平行导致叉乘退化
        # up_world = torch.tensor([0.0, 1.0, 0.0], device=device).view(1,1,3).expand(b,v,3)
        # alt_up   = torch.tensor([1.0, 0.0, 0.0], device=device).view(1,1,3).expand(b,v,3)
        # parallel = (torch.abs((forward * up_world).sum(-1)) > 0.99).unsqueeze(-1)  # [b,v,1]
        # up_ref = torch.where(parallel, alt_up, up_world)                            # [b,v,3]

        # x,y ∈ [-1,1]
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [h,w]

        # broadcast 形状对齐
        xx = xx.view(1, 1, h, w, 1)
        yy = yy.view(1, 1, h, w, 1)

        forward_e = forward[:, :, None, None, :]  # [b,v,1,1,3]

        # 射线起点固定在世界坐标 z=0 平面，不随 view 旋转
        ray_o_x = xx.expand(b, v, h, w, 1)
        ray_o_y = yy.expand(b, v, h, w, 1)
        ray_o_z = torch.zeros_like(ray_o_x)
        ray_o = torch.cat([ray_o_x, ray_o_y, ray_o_z], dim=-1)  # [b,v,h,w,3]
        if plane_offset != 0.0:
            # 可选：沿视线方向做平移，保持与旧接口兼容
            ray_o = ray_o + plane_offset * forward_e
        # 正交：同 view 内方向一致
        ray_d = forward_e.expand(b, v, h, w, 3)                        # [b,v,h,w,3]

        # 输出维度 [b,v,3,h,w]
        ray_o = ray_o.permute(0, 1, 4, 2, 3).contiguous()
        ray_d = ray_d.permute(0, 1, 4, 2, 3).contiguous()
        return ray_o, ray_d
    
    @torch.no_grad()
    def forward(self, data_batch, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w] (c=2)
                - 'pose' (torch.Tensor): Shape [b, v, 3] - [x, y, cos_polar]
                - 'voxel' (torch.Tensor): Shape [b, D, H, W]
                - 'index' (torch.Tensor): Shape [b, v, 2]
                - 'scene_name' (list): Length [b]
                - 'global_min' (torch.Tensor): Shape [b, 2]
                - 'global_max' (torch.Tensor): Shape [b, 2]
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            EasyDict: Contains processed data:
                - 'image' (torch.Tensor): Shape [b, v, 2, h, w]
                - 'pose' (torch.Tensor): Shape [b, v, 3]
                - 'voxel' (torch.Tensor): Shape [b, D, H, W]
                - 'ray_o' (torch.Tensor, optional): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor, optional): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
                - 'global_min' (torch.Tensor): Shape [b, 2]
                - 'global_max' (torch.Tensor): Shape [b, 2]
                - 'index' (torch.Tensor): Shape [b, v, 2]
                - 'scene_name' (list): Length [b]
        """
        output = edict()

        # 检查数据是否存在并获取维度信息
        if "image" not in data_batch:
            raise KeyError("Missing 'image' key in data_batch")
            
        # 直接从 data_batch 中提取数据
        b, v, _, h, w = data_batch["image"].shape
        output["image_h_w"] = (h, w)
        
        # 复制基本字段
        for key in [
            "image",
            "pose",
            "index",
            "voxel",
            "scene_name",
            "global_min",
            "global_max",
            "global_min_bg",
            "global_max_bg",
        ]:
            if key in data_batch:
                output[key] = data_batch[key]
        
        # 如果 Dataset 返回了 background，也添加进去
        if "background" in data_batch:
            output["background"] = data_batch["background"]

        # 计算射线
        if compute_rays:
            # 使用正交投影计算射线，直接传入 pose
            ray_o, ray_d = self.compute_rays(
                output["pose"],  # [b, v, 3]
                h,
                w,
                device=output["image"].device,
            )
            output["ray_o"] = ray_o
            output["ray_d"] = ray_d

        return output


      
      