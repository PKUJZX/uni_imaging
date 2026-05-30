# ODT_pretrain 模型与训练流程说明

本文档严格对齐当前代码实现，覆盖两条训练主线：

1. **预训练流程**：`pretrain_mae.py` + `model/odt_mae.py`
2. **主训练流程**：`train.py` + `model/odt_direct_inversion.py`

两个流程共享同一套数据组织方式，但任务、模型结构、张量维度、损失函数与训练范式不同。

---

## 1. 数据组织总览

### 1.1 主训练数据
主训练通过 `config.training.dataset_name` 动态选择 `data.odt_datasets` 中的具体数据集类，核心样本结构为：

- `image`: `[B, T, 2, H, W]`
- `background`: `[B, T, 2, H, W]`
- `voxel`: `[B, D, H_v, W_v]`
- `index`: `[B, T, 2]`

其中：

- `T = frame_count`，代码默认是 `240`
- `2` 表示复数的实部/虚部两个通道
- `H = W = image_size`，主训练中通常是 `256`
- `voxel_size` 在配置中显式给出，当前实现会严格检查 voxel 形状

主训练的数据集实现支持两种数据组织：

- `ODTFullPatchVoxelDataset`
- `ODTFullFramePackedVoxelDataset`

### 1.2 预训练数据
预训练使用 `ODTMAEPackedPatchDataset`，每个样本是单个空间 patch 对应的 240 帧序列：

- `image`: `[B, T, 2, P, P]`
- `background`: `[B, T, 2, P, P]`
- `index`: `[B, T, 2]`
- `crop_xy`: `[B, 2]`
- `global_max_amp`, `global_max_bg_amp`: 标量

其中：

- `T = 240`
- `P = patch_size = 16`
- 每个样本只覆盖整幅图像的一个 patch，但时间维仍是全帧。

---

## 2. 预训练流程：`pretrain_mae.py` + `ODTMaskedAutoencoder`

## 2.1 训练目标
预训练流程是标准的 masked autoencoder 思路，但和常见 ViT-MAE 不同的是：

- token 不是图像 patch embedding
- 而是 **时序 frame patch**
- 输入是一个空间 patch 上的 240 帧复数图像/背景序列
- 任务是重建被 mask 的 frame-token 对应内容

### 预训练样本含义
每个样本对应：

- 一个空间 patch
- 该 patch 上的所有 240 个时间帧
- 输入内容为 image + background 两路复数通道

所以预训练不是“整幅图像多 patch”，而是“单 patch 的长时序建模”。

---

## 2.2 `ODTMAEEncoder` 结构

### 输入形状
进入 encoder 的张量形状为：

- `image`: `[B, T, 2, P, P]`
- `background`: `[B, T, 2, P, P]`

其中：

- `T = frame_count = 240`
- `P = patch_size = 16`

### patch embedding
编码器先把 image/background 拼起来：

- 拼接后每帧通道数从 `2 + 2 = 4`
- 每帧展开后长度为：

\[
4 \times P \times P = 4 \times 16 \times 16 = 1024
\]

### `_embed()`
`_embed()` 的处理流程：

1. `torch.cat([image, background], dim=2)`
2. reshape 为 `[B, T, 4 * P * P]`
3. 送入：
   - `patch_embed = nn.Conv2d(1, embed_dim, kernel_size=(1, 1024))`
4. 最终输出：
   - `[B, T, E]`

其中 `E = embed_dim = enc_emb_dim`

### 位置编码与 cls token
encoder 使用：

- `num_cls_tokens = 16`
- `pos_embed` 是 1D sin-cos 位置编码

位置编码长度为：

- `frame_count + num_cls_tokens = 256`

因此：

- `pos_embed`: `[1, 256, E]`

### 随机 masking
`random_masking(x, ratio)` 会：

- 随机打乱 240 个 frame token
- 保留 `len_keep = int(T * (1 - mask_ratio))`

如果 `mask_ratio = 0.75`，则：

- 保留 `240 * 0.25 = 60` 个 frame token
- mask 掉 `180` 个 frame token

### encoder 输出
encoder forward 输出：

- `latent`: `[B, num_cls_tokens + len_keep, E]`
- `mask`: `[B, T]`
- `ids_restore`: `[B, T]`

若默认配置：

- `latent = [B, 16 + 60, 512] = [B, 76, 512]`

### `encode_full()`
预训练之外，主训练会用 `encode_full()`，即不做随机 masking，输出：

- `[B, 16 + 240, E] = [B, 256, 512]`

---

## 2.3 `ODTMAEDecoder` 结构

### 输入
decoder 输入：

- `latent`: `[B, 16 + 60, E]`
- `ids_restore`: `[B, 240]`
- `background`: `[B, 240, 2, 16, 16]`
- `mask`: `[B, 240]`

### `enc_to_dec`
先把 encoder latent 映射到 decoder dim：

- `enc_to_dec = nn.Conv2d(1, dec_dim, kernel_size=(1, enc_dim))`

输出变为：

- `[B, N_latent, dec_dim]`

### background embedding
背景分支用：

- `background_embed = nn.Conv2d(1, dec_dim, kernel_size=(1, 2 * P * P))`

输入背景每帧展开长度为：

- `2 * 16 * 16 = 512`

所以背景 token 形状为：

- `[B, T, dec_dim]`

### mask token 与恢复顺序
decoder 会：

1. 先保留 encoder 传入的 `cls token`
2. 仅对 **非 cls 的 frame token** 做恢复：把可见 token 与 `mask_token` 拼接成完整长度
3. 按 `ids_restore` 恢复原始 frame 顺序
4. 将背景 token 加到 masked 位置上：
   - `x_ = x_ + bg_tokens * mask`

这里的 `mask_token` 是**可学习参数**：

- `self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))`
- 初始化为正态分布 `std=0.02`

这里 `mask` 为 1 的位置对应原本被 mask 的 frame。

### decoder transformer
恢复后的序列再加上：

- decoder 的 1D sin-cos `pos_embed`

然后经过若干个 `QK_Norm_TransformerBlock`。

### 输出
最后：

- `pred = self.pred(x[:, num_cls_tokens:, :])`
- 每个 frame 预测 `2 * P * P = 512` 维
- reshape 后输出：

\[
[B, T, 2, P, P]
\]

即：

- `[B, 240, 2, 16, 16]`

---

## 2.4 预训练损失函数
`ODTMaskedAutoencoder.forward_loss()` 计算两种 MSE：

### token-level MSE
先对每个 frame token 的像素维做平均：

- `token_mse = ((pred - target) ** 2).flatten(2).mean(dim=-1)`

所以：

- `token_mse`: `[B, T]`

### masked MSE
只统计被 mask 的 frame：

\[
L_{masked} = \frac{\sum token\_mse \cdot mask}{\sum mask}
\]

### full MSE
所有 frame 上求平均：

\[
L_{full} = mean(token\_mse)
\]

### 实际训练损失
训练脚本中反向传播用的是：

- `result.loss_metrics.loss = full_mse`

也就是说，**当前代码实现中优化的是 full MSE，而不是 masked MSE**。

返回的指标包括：

- `loss`
- `mae_loss`
- `mae_full_mse`
- `mae_masked_mse`

其中：

- `loss == mae_loss == mae_full_mse`

---

## 2.5 预训练范式：`pretrain_mae.py`

### 训练循环特点
和主训练类似，也使用：

- `DDP`
- `DistributedSampler`
- `AMP`
- `grad_accum_steps`
- `auto_resume_job`
- checkpoint / tensorboard / eval / vis

但它的目标是 MAE 预训练，不是 voxel 回归。

### 训练中用到的 batch
batch 包含：

- `image`
- `background`
- `index`
- `crop_xy`
- `global_max_amp`
- `global_max_bg_amp`

### 前向与反向
每个 step：

1. `result = model(batch)`
2. `loss = result.loss_metrics.loss / grad_accum`
3. backward / step / clip

### checkpoint 保存
预训练 checkpoint 会同时保存：

- `ckpt_{step}.pt`
- `encoder_{step}.pt`

也就是说预训练阶段还单独导出 encoder 权重，方便后续主训练加载。

### eval
eval 时使用：

- `mae_metrics(eval_result)`

记录：

- `mae_masked_mse`
- `mae_full_mse`

### 可视化
通过：

- `save_mae_reconstruction(...)`

保存重建结果。

---

## 3. 主训练流程：`train.py` + `ODTDirectInversion`

## 3.1 训练目标
主训练的目标是：

- 输入：全帧的多视角 `image` 与 `background`
- 输出：三维 `voxel`
- 采用直接反演范式：
  1. 先用 `ODTMAEEncoder` 将二维时空输入编码为 memory token
  2. 再用 3D cross-attention decoder 将 memory 映射到体素网格
  3. 用 voxel 回归损失监督

这条流程不是 masked reconstruction，而是 **image/background -> voxel** 的监督回归。

---

## 3.2 `ODTDirectInversion` 总体结构
代码文件：`model/odt_direct_inversion.py`

主模块由三部分组成：

1. `encoder = ODTMAEEncoder(config)`
2. `decoder = CrossAttention3DGridDecoder(config)`
3. `loss_computer = VoxelLoss(config)`

### 编码器部分
编码器来自 `model/odt_mae.py` 中的 `ODTMAEEncoder`，但在主训练中调用的是：

- `encode_full(image, background)`

而不是 MAE 预训练中的随机 masking forward。

### 解码器部分
解码器是一个 3D grid cross-attention decoder：

- 先构造 voxel 网格上的 query token
- 再用 encoder memory 作为 key/value
- 最后通过 `LatentToVoxelDecoder` 还原成三维 voxel

### 损失
主训练只对 voxel 做回归，默认是：

- `L2 loss` 为主
- `L1 loss` 可选

当前实现中 `VoxelLoss` 返回：

- `loss`
- `l2_loss`
- `l1_loss`

---

## 3.3 编码器 `ODTMAEEncoder` 在主训练中的作用

### 输入张量
输入是：

- `image`: `[B, T, 2, H, W]`
- `background`: `[B, T, 2, H, W]`

代码要求：

- `T == frame_count`
- `H == W == patch_size` 才能进入 `ODTMAEEncoder._embed`

但在主训练中，`ODTDirectInversion` 先将 full frame 切成空间 patch，因此进入 encoder 的张量实际是：

- `image_patches`: `[B * G^2, T, 2, P, P]`
- `background_patches`: `[B * G^2, T, 2, P, P]`

其中：

- `G = image_size / patch_size`
- 当前默认 `image_size=256`, `patch_size=16`
- 所以 `G = 16`
- `G^2 = 256`

### `_patchify_frames`
主训练里先调用：

- `_patchify_frames(x)`

输入：`[B, T, C, H, W]`

输出：`[B * G^2, T, C, P, P]`

也就是每个空间位置一个 patch 序列。

### `encode_full`
`ODTMAEEncoder.encode_full(image, background)` 的输出形状：

- `[B', N_tokens, E]`

其中：

- `B' = B * G^2`
- `N_tokens = num_cls_tokens + T`
- `E = embed_dim = enc_emb_dim`

当前配置里常见的是：

- `enc_emb_dim = 512`
- `num_cls_tokens = 16`
- `T = 240`
- 所以每个 patch 的输出 token 数是 `256`

即：

- `encoded`: `[B * 256, 256, 512]`

### 主训练中的 memory 构造
`ODTDirectInversion._encode_memory()` 会把 patch 级别结果重新整理成：

- `encoded.reshape(B, G^2, N_tokens, E)`

然后根据 `memory_mode` 分两种情况：

#### `memory_mode = "all_frames"`
使用所有 frame token，不含 cls token：

- `frame_tokens = encoded[:, :, num_cls_tokens:, :]`
- reshape 后得到：
- `memory: [B, G^2 * T, E]`

即当前默认情况下：

- `memory: [B, 256 * 240, 512] = [B, 61440, 512]`

#### `memory_mode = "cls"`
只使用 cls token：

- `cls_tokens = encoded[:, :, :num_cls_tokens, :]`
- reshape 后得到：
- `memory: [B, G^2 * num_cls_tokens, E]`

即：

- `memory: [B, 256 * 16, 512] = [B, 4096, 512]`

### 空间位置编码
主训练还会给每个空间 patch 加一个二维位置编码：

- `spatial_pos_embed`: 可学习参数 `[1, G^2, E]`
- `spatial_sincos_pos_embed`: 固定 buffer `[1, G^2, E]`
- 实际使用：
  - `spatial_pos = spatial_pos_embed + spatial_pos_alpha * spatial_sincos_pos_embed`

然后把它加到每个空间 patch 的 token 上。

---

## 3.4 `CrossAttention3DGridDecoder` 结构与维度

### 基本参数
解码器根据配置读取：

- `enc_dim`: 编码器输出维度，默认等于 `model_dim(config)`
- `dec_dim`: decoder 维度，默认 `dec_emb_dim`
- `grid_shape`: 默认 `(16, 32, 32)`
- `voxel_size`: 默认 `(128, 256, 256)`
- `num_heads`: `cross_attn_heads` 或 `dec_nhead`

代码里强约束：

- `voxel_size` 必须能被 `voxel_pos_tokenizer.patch_size` 整除
- `voxel_grid == voxel_size / voxel_pos_patch_size`

默认情况下：

- `voxel_pos_patch_size = 8`
- `voxel_size = (128, 256, 256)`
- 所以 `voxel_grid = (16, 32, 32)`
- `num_queries = 16 * 32 * 32 = 16384`

### 3D 查询 token 的构造
decoder 先为整个体素网格构造位置条件：

1. `voxel_position_encoding()` 生成 `[B, C_pos, D, H, W]`
2. 其中 `C_pos = 27`
3. 这个 27 维来自 `nerf_positional_encoding`：
   - 输入 3 个坐标通道 `(x, y, z)`
   - 每个通道做 4 个频率的 sin/cos
   - 输出通道数：
     - `3 + 3 * 2 * 4 = 27`

然后：

- `voxel_pos_tokenizer` 把 `[B, 27, 128, 256, 256]` 切成 patch token
- 输出：`[B, 16384, dec_dim]`

如果 `dec_dim = 256`，则 query token 维度是：

- `[B, 16384, 256]`

### cross-attention
decoder 的核心是：

- query = voxel grid token
- key/value = encoder memory

维度关系：

- `q`: `[B, Lq, H, Dh]`
- `k/v`: `[B, Lm, H, Dh]`

其中：

- `Lq = 16384`
- `Lm = 61440`（all_frames）或 `4096`（cls）
- `H = num_heads`
- `Dh = dec_dim / num_heads`

代码要求：

- `dec_dim % num_heads == 0`
- `Dh % 8 == 0`

这是为了兼容 FlashAttention / memory efficient attention。

### chunked attention
因为 `Lq` 很大，decoder 采用 query chunk 方式：

- `query_chunk_size` 默认 `256`
- 按 chunk 逐段做 cross-attention
- 每段输出再拼接

最后：

- `grid_tokens: [B, 16384, dec_dim]`
- 再送入 `LatentToVoxelDecoder`
- 输出体素：`[B, D, H_v, W_v]`

默认即：

- `[B, 128, 256, 256]`

---

## 3.5 主训练损失函数
`VoxelLoss` 定义在 `model/odt_direct_inversion.py`。

### 公式
设：

- 预测 voxel 为 `pred`
- GT voxel 为 `target`

则：

- `l2 = MSE(pred, target)`
- `l1 = L1(pred, target)`（如果 `l1_loss_weight > 0` 才启用）
- 总损失：

\[
L = \lambda_{l2} \cdot L2 + \lambda_{l1} \cdot L1
\]

默认配置下通常是：

- `l2_loss_weight = 1.0`
- `l1_loss_weight = 0.0`

因此大多数情况下等价于纯 MSE 回归。

### 返回值
`forward()` 返回：

- `loss`
- `l2_loss`
- `l1_loss`

训练脚本中实际优化的是：

- `ret_dict.loss_metrics.loss`

---

## 3.6 主训练范式：`train.py`

主训练脚本的核心特点：

1. 使用 `DDP`
2. 支持梯度累积
3. 支持 eval
4. 支持中间可视化
5. 支持自动恢复 checkpoint

### 训练循环
每次迭代：

1. 取一个 batch
2. 前向：`ret_dict = model(batch)`
3. 反向：对 `ret_dict.loss_metrics.loss` 做梯度回传
4. 每 `grad_accum_steps` 次做一次 optimizer step
5. 定期 eval / checkpoint / visualize

### mixed precision
支持 AMP：

- `fp16`
- `bf16`
- `fp32`
- `tf32`

### 训练中记录的指标
训练脚本记录：

- `loss`
- `l2_loss`
- `l1_loss`
- `learning_rate`
- `iteration_time`
- `grad_norm`
- 以及 eval 结果

---

## 4. 两阶段训练关系

当前代码体现的是一个明确的两阶段范式：

### 阶段 1：MAE 预训练
文件：

- `pretrain_mae.py`
- `model/odt_mae.py`

目标：

- 学到对 `image + background` 的时序表征
- 通过随机 frame masking 进行重建
- 最终输出可复用的 encoder checkpoint

### 阶段 2：主训练 / 直接反演
文件：

- `train.py`
- `model/odt_direct_inversion.py`

目标：

- 固定或微调 MAE encoder
- 用 encoder 表征作为 memory
- 通过 3D cross-attention decoder 回归 voxel

### 两阶段连接方式
主训练中的 `ODTDirectInversion` 会：

- 创建 `ODTMAEEncoder(config)`
- 如配置了 `encoder_ckpt`，会通过 `load_encoder_weights()` 加载 MAE 预训练权重
- 如果 `freeze_encoder=True`，则 encoder 不参与训练

因此，预训练 encoder 是主训练的初始化来源。

---

## 5. 关键配置与实现约束

### MAE 预训练常见配置
- `frame_count = 240`
- `image_size = 256`
- `patch_size = 16`
- `num_cls_tokens = 16`
- `mask_ratio = 0.75`
- `enc_emb_dim = 512`
- `dec_emb_dim = 256`

### 主训练常见配置
- `frame_count = 240`
- `image_size = 256`
- `patch_size = 16`
- `voxel_size = (128, 256, 256)`
- `voxel_pos_patch_size = 8`
- `voxel_grid = (16, 32, 32)`
- `enc_emb_dim = 512`
- `dec_emb_dim` 取决于配置，常见为 `256`

### 代码中的硬约束
以下检查在代码里是显式存在的：

1. `image_size % patch_size == 0`
2. `voxel_size % voxel_pos_patch_size == 0`
3. `voxel_grid == voxel_size / voxel_pos_patch_size`
4. `dec_dim % num_heads == 0`
5. `head_dim % 8 == 0`
6. `frame_count`、`patch_size`、输入数据形状必须匹配
7. voxel shape 必须和配置完全一致

---

## 6. 小结

### 预训练
- 输入：单个空间 patch 的 240 帧序列
- 输出：重建后的 patch 序列
- 模型：`ODTMAEEncoder` + `ODTMAEDecoder`
- 监督：`full MSE` 为训练目标，同时记录 `masked MSE`
- 范式：**MAE 时序重建预训练**

### 主训练
- 输入：全帧 `image/background`
- 输出：`voxel`
- 模型：`ODTMAEEncoder` + `CrossAttention3DGridDecoder`
- 监督：`VoxelLoss`
- 范式：**监督式 voxel 回归 / direct inversion**

两者共享 encoder 设计，但解码器、损失与数据组织方式完全不同。
