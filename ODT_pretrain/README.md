# ODT_pretrain

`ODT_pretrain` 是一个用于光学衍射断层成像（ODT）重建的两阶段训练仓库。

- **阶段 1**：使用 `pretrain_mae.py` + `model/odt_mae.py` 进行 MAE 预训练
- **阶段 2**：使用 `train.py` + `model/odt_direct_inversion.py` 进行主训练 / 直接反演
- **推理**：使用 `inference.py` 进行统一推理

仓库支持分布式训练，配置文件位于 `configs/` 目录下。

---

## 仓库结构

```text
ODT_pretrain/
├── configs/                # 训练与推理配置
├── data/                   # 数据集实现与数据列表
├── model/                  # MAE 编码器/解码器与直接反演模型
├── scripts/                # 数据准备与辅助脚本
├── utils/                  # 训练、可视化、指标辅助函数
├── pretrain_mae.py         # MAE 预训练入口
├── train.py                # 主直接反演训练入口
├── inference.py            # 统一推理入口
├── setup.py                # 配置解析与分布式初始化
└── model.md                # 模型与训练细节文档
```

---

## 两阶段训练概览

### 1）MAE 预训练
相关文件：
- `pretrain_mae.py`
- `model/odt_mae.py`
- `configs/pretrain_mae.yaml`

目标：
- 从 `image + background` 中学习更强的时序编码器
- 采用 masked autoencoder 方式重建被遮挡的 frame token
- 保存完整 checkpoint 和 encoder-only 权重，供下游流程复用

### 2）主训练 / 直接反演
相关文件：
- `train.py`
- `model/odt_direct_inversion.py`
- `configs/train.yaml`

目标：
- 加载预训练 MAE 编码器
- 将编码后的多视角特征映射为三维 voxel
- 使用 voxel 监督进行训练

---

## 模型输入与输出

### MAE 预训练模型（`ODTMaskedAutoencoder`）

输入 batch：
- `image`: `[B, T, 2, P, P]`
- `background`: `[B, T, 2, P, P]`
- `index`: 元信息，模型前向不使用

默认形状：
- `T = 240`
- `P = 16`

输出：
- `pred_image`: `[B, T, 2, P, P]`
- `mask`: `[B, T]`
- `ids_restore`: `[B, T]`
- `loss_metrics`：包含 `loss`、`mae_full_mse`、`mae_masked_mse`

### 直接反演模型（`ODTDirectInversion`）

输入 batch：
- `image`: `[B, T, 2, H, W]`
- `background`: `[B, T, 2, H, W]`
- `voxel`: `[B, D, H_v, W_v]`

默认形状：
- `T = 240`
- `H = W = 256`
- `voxel_size` 在 `configs/train.yaml` 中配置

输出：
- `predicted_voxel`: `[B, D, H_v, W_v]`
- `loss_metrics`：包含 `loss`、`l2_loss`、`l1_loss`

---

## 配置文件

### `configs/pretrain_mae.yaml`
常见字段：
- `model.class_name`: `model.odt_mae.ODTMaskedAutoencoder`
- `model.frame_count`: 240
- `model.image_size`: 256
- `model.patch_size`: 16
- `model.num_cls_tokens`: 16
- `model.mask_ratio`: 0.75
- `model.enc_emb_dim`: 512
- `model.dec_emb_dim`: 256
- `training.dataset_name`: `data.odt_datasets.ODTMAEPackedPatchDataset`

### `configs/train_encoder_voxel.yaml`
这是当前仓库里用于主训练的配置文件。常见字段：
- `model.class_name`: `model.odt_direct_inversion.ODTDirectInversion`
- `model.encoder_ckpt`: 预训练 MAE encoder 的路径或 checkpoint 目录
- `model.freeze_encoder`: 是否冻结 encoder
- `model.memory_mode`: `all_frames` 或 `cls`
- `model.frame_count`: 240
- `model.image_size`: 256
- `model.patch_size`: 16
- `model.num_cls_tokens`: 16
- `model.voxel_size`: `[128, 256, 256]`
- `model.voxel_grid`: 体素网格尺寸
- `model.voxel_pos_tokenizer.patch_size`: 16
- `model.voxel_pos_tokenizer.in_channels`: 27
- `model.enc_emb_dim`: 512
- `model.dec_emb_dim`: 512
- `model.cross_attn_heads`: 16
- `model.cross_attn_query_chunk_size`: 256
- `model.voxel_decoder_channels`: `[384, 192, 48, 12]`
- `model.voxel_output_activation`: `sigmoid`
- `training.dataset_name`: `data.odt_datasets.ODTFullFramePackedVoxelDataset`
- `training.dataset_path`: 训练集 scene 列表文件
- `training.eval_dataset_path`: 验证集 scene 列表文件
- `training.l2_loss_weight`: 1.0
- `training.l1_loss_weight`: 0.0

### `configs/inference_pretrain.yaml`
由 `inference.py` 使用，面向 MAE 预训练流程。
常见字段：
- `inference.task`: `mae_pretrain`
- `inference.checkpoint_dir`: MAE checkpoint 路径或目录
- `inference.output_dir`: 输出目录
- `inference.random_eval_crop`: 是否随机选择 patch

### `configs/inference_voxel.yaml`
由 `inference.py` 使用，面向主训练 / 直接反演流程。
常见字段：
- `inference.task`: `voxel_main`
- `inference.checkpoint_dir`: 直接反演 checkpoint 路径或目录
- `inference.output_dir`: 输出目录
- `inference.compute_metrics`: 是否汇总评估指标

### `configs/inference.yaml`
统一推理配置文件。
常见字段：
- `inference.task`: `mae_pretrain` / `voxel_main`
- `inference.dataset_name`: 推理使用的数据集类
- `inference.checkpoint_dir`: checkpoint 路径或目录
- `inference.output_dir`: 输出目录
- `inference.compute_metrics`: 是否汇总评估指标

---

## 数据格式

数据集读取的是 scene 元数据 JSON 文件，这些 JSON 路径通常由 `data/` 下的文本列表文件或用户自定义列表文件提供。

### scene 元数据中常见字段
根据数据集类型不同，JSON 元数据中可能包含：
- `frames`
- `patches`
- `voxel_path`
- `image_path`
- `background_path`
- `scene_name`
- `global_max_amp`
- `global_max_bg_amp`

代码会使用保存的全局缩放因子对复数图像/背景进行归一化。

---

## 训练示例

### 1）MAE 预训练

单机示例：

```bash
python pretrain_mae.py --config configs/pretrain_mae.yaml
```

分布式示例：

```bash
torchrun --nproc_per_node=4 pretrain_mae.py --config configs/pretrain_mae.yaml
```

### 2）主直接反演训练

单机示例：

```bash
python train.py --config configs/train_encoder_voxel.yaml
```

分布式示例：

```bash
torchrun --nproc_per_node=4 train.py --config configs/train_encoder_voxel.yaml
```

### 3）从 checkpoint 目录恢复主训练

在配置中设置：

```yaml
training:
  resume_ckpt: /path/to/checkpoint_dir
```

然后运行：

```bash
torchrun --nproc_per_node=4 train.py --config configs/train_encoder_voxel.yaml
```

---

## 推理示例

`inference.py` 支持两类任务：

- `mae_pretrain` / `pretrain` / `mae`
- `voxel_main` / `voxel` / `direct`

### 1）MAE 推理 / 重建可视化

直接使用 MAE 推理配置：

```bash
python inference.py --config configs/inference_pretrain.yaml
```

如果你想单独指定统一推理配置，也可以使用：

```bash
python inference.py --config configs/inference.yaml
```

对应配置写法示例：

```yaml
inference:
  task: mae_pretrain
  checkpoint_dir: /path/to/mae_checkpoint_dir
  output_dir: ./inference_mae
  max_items_per_batch: 2
```

MAE 推理会对每个 batch 保存重建结果，并最终输出 `mae_metrics.json`。

### 2）直接反演推理

直接使用 voxel 推理配置：

```bash
python inference.py --config configs/inference_voxel.yaml
```

如果你想单独指定统一推理配置，也可以使用：

```bash
python inference.py --config configs/inference.yaml
```

对应配置写法示例：

```yaml
inference:
  task: voxel_main
  checkpoint_dir: /path/to/direct_inversion_checkpoint_dir
  output_dir: ./inference_voxel
  compute_metrics: true
```

直接反演推理时，脚本会导出 voxel 预测结果，并在 `compute_metrics=true` 时汇总评估指标。

---

## 关于 checkpoint

### MAE 预训练 checkpoint
`pretrain_mae.py` 会保存：
- `ckpt_XXXXXXXXXXXXXX.pt`
- `encoder_XXXXXXXXXXXXXX.pt`

### 主训练 checkpoint
`train.py` 会保存：
- `ckpt_XXXXXXXXXXXXXX.pt`

主训练模型可以通过 `model.encoder_ckpt` 加载预训练 encoder 权重。

---

## 重要实现约束

- `image_size % patch_size == 0`
- MAE encoder 输入的帧数必须与 `frame_count` 一致
- voxel 形状必须与配置中的 `voxel_size` 一致
- decoder 的 attention 维度必须与 head 数兼容
- 当前 MAE 预训练损失优化的是 **full MSE**，同时会记录 masked MSE

---

## 相关文件

- `model.md`：模型与训练的详细说明
- `data/odt_datasets.py`：数据集实现
- `model/odt_mae.py`：MAE 编码器/解码器
- `model/odt_direct_inversion.py`：直接反演模型
- `utils/`：loss / metric / visualization 辅助函数

---

## 版权 / 署名

Copyright (c) 2025 Haian Jin. Created for the LVSM project.
