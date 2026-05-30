# Ptycho MAE 预训练

本目录实现 Ptycho / 4D-STEM 的中心遮挡 MAE 预训练。预训练任务与
`Ptycho_dataflow.md` 中的数据流保持一致：从局部 `3 x 3` scan 邻域中取周围
8 个 diffraction pattern，加上 probe intensity / phase 两个 token，重建中心
位置的 diffraction pattern。

```text
3 x 3 DP patch:

0 1 2
3 4 5
6 7 8

visible: 0,1,2,3,5,6,7,8 + probe_intensity + probe_phase
target:  4
```

默认按 A100 训练环境配置：`bf16` AMP、TF32、DDP/NCCL、多 worker dataloader。
当前本地无 GPU 时，不建议在本机做性能验证；把数据路径改成真实 HDF5 后，迁移到
A100 服务器运行即可。

## 文件结构

```text
Ptycho_pretrain/
  configs/pretrain_mae.yaml        # A100 默认 MAE 预训练配置
  configs/train.yaml               # projected potential 下游任务配置
  data/ptycho_datasets.py          # HDF5 数据集与归一化（MAE + 下游）
  model/ptycho_mae.py              # encoder / decoder / MAE 模型
  model/ptycho_downstream.py       # 下游 projected potential 预测模型
  utils/ptycho_visualization.py    # MAE / 下游可视化与评估指标
  pretrain_mae.py                  # MAE 预训练入口（单卡 / torchrun 多卡）
  train.py                         # 下游训练入口
  inference.py                     # 下游推理入口
  Ptycho_dataflow.md               # 原始 Ptycho 数据流说明
```

## 数据格式

每个 HDF5 sample 可以直接位于 root，也可以位于 root 下的 sample group。每个
sample 需要包含：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `diffraction_patterns` | `(scan_y, scan_x, 64, 64)` | scan grid 上每个位置的 DP |
| `probe_intensity` | `(64, 64)` | probe intensity |
| `probe_phase` | `(64, 64)` | probe phase |

`projected_potential` 不是 MAE 预训练必需字段。配置中默认
`data.require_projected_potential_bounds: false`，所以数据集只依赖 DP 和 probe。

`training.dataset_path` 和 `training.eval_dataset_path` 支持：

- 单个 `.h5` / `.hdf5` 文件
- 包含多个 HDF5 文件的目录
- glob，例如 `/data/ptycho/train/*.h5`
- YAML list

## 模型输入输出

数据集返回的关键张量：

| key | shape | 说明 |
| --- | --- | --- |
| `visible_images` | `[10, 64, 64]` | 8 个可见 DP + 2 个 probe token |
| `visible_token_ids` | `[10]` | 可见 token 在 11-token 序列中的原始位置 |
| `target_image` | `[1, 64, 64]` | 中心 DP token，即 token `4` |
| `all_images` | `[11, 64, 64]` | 完整 token 序列，仅用于可视化 / 下游检查 |
| `mask` | `[11]` | 只有中心 token 为 `1` |

模型 token 顺序固定为：

| token id | 来源 |
| --- | --- |
| `0..8` | `3 x 3` scan patch 展平后的 9 个 DP |
| `9` | probe intensity |
| `10` | probe phase |

训练时 encoder 只接收 visible tokens；decoder 在 token `4` 位置插入 mask token，
恢复完整 11-token 序列后只预测中心 DP。

## 配置

默认配置在：

```text
configs/pretrain_mae.yaml
```

首跑前至少需要修改：

```yaml
training:
  dataset_path:
    - /path/to/ptycho/train
  eval_dataset_path:
    - /path/to/ptycho/val
  checkpoint_dir: ./ptycho_experiments/checkpoints/ptycho_mae_pretrain
```

关键模型配置：

```yaml
model:
  dp_size: 64
  scan_patch_size: 3
  center_token_id: 4
  use_probe_tokens: true
  embed_dim: 768
  dec_embed_dim: 512
  enc_depth: 12
  dec_depth: 8
  d_head: 64
```

默认归一化：

```yaml
data:
  normalization:
    dp: log1p_zscore
    probe: zscore
```

DP 会先做 `log1p`，再在当前 `3 x 3` patch 内做 z-score；target 和 prediction 都在
归一化域计算 MSE。

## 运行

进入目录：

```bash
cd Ptycho_pretrain
```

单进程运行：

```bash
python pretrain_mae.py --config configs/pretrain_mae.yaml
```

A100 多卡运行：

```bash
torchrun --nproc_per_node=8 pretrain_mae.py --config configs/pretrain_mae.yaml
```

建议 A100 首次只跑少量 step，检查 loss、PNG 和 checkpoint 是否正常：

```bash
torchrun --nproc_per_node=8 pretrain_mae.py \
  --config configs/pretrain_mae.yaml \
  training.train_steps=20 \
  training.eval_every=10 \
  training.vis_every=10 \
  training.checkpoint_every=20
```

也可以覆盖 batch size 或数据路径：

```bash
torchrun --nproc_per_node=8 pretrain_mae.py \
  --config configs/pretrain_mae.yaml \
  training.dataset_path=/data/ptycho/train \
  training.eval_dataset_path=/data/ptycho/val \
  training.batch_size_per_gpu=64
```

## 输出

训练输出位于 `training.checkpoint_dir`：

```text
ckpt_0000000000000020.pt       # full checkpoint
encoder_0000000000000020.pt    # standalone encoder checkpoint
decoder_0000000000000020.pt    # standalone decoder checkpoint
iter_00000010/                 # 可视化结果
tensorboard_logs/              # TensorBoard scalar logs
```

checkpoint 中包含：

- `model`: 完整 MAE 权重
- `encoder`: `PtychoMAEEncoder` 权重
- `decoder`: decoder 权重
- `optimizer` / `lr_scheduler`
- `fwdbwd_pass_step` / `param_update_step`
- `config`

可视化目录中会保存：

- `visible_context_grid.png`: 3x3 DP context，中心为空
- `gt_pred_abs_error.png`: GT center / prediction / absolute error
- `probe_intensity_phase.png`: probe intensity / phase
- 对应 `.npy` 文件

## 下游迁移

预训练 encoder 类为：

```python
from model.ptycho_mae import PtychoMAEEncoder, load_encoder_weights

encoder = PtychoMAEEncoder(config)
ckpt_path, status = load_encoder_weights(encoder, "/path/to/checkpoint_dir")
```

下游 projected-potential 预测任务可以直接使用：

```python
latent = encoder.encode_full(inputs11)
```

其中 `inputs11` 形状为 `[B, 11, 64, 64]`，token 顺序必须和预训练保持一致：
9 个 DP token + probe intensity + probe phase。

## 下游任务：projected potential 预测

`model/ptycho_downstream.py` 已实现完整下游任务：复用预训练 encoder，取中心
token（idx `4`）的 latent，经 ConvTranspose 解码出中心 scan 位置对应的局部
projected potential patch。

```text
inputs [B, 11, 64, 64]
  → PtychoMAEEncoder.encode_full         # 默认冻结 encoder
  → latent [B, 11, 768]
  → latent[:, 4]  [B, 768]
  → Linear → [B, 512, 4, 4]
  → ConvTranspose2d × 2 (4→8→16)
  → Conv2d → [B, 1, 16, 16]
loss: MSE(pred, projected_potential_patch)
```

### 数据集

`data.ptycho_datasets.PtychoProjectedPotentialDataset` 在 `PtychoCenterMAEDataset`
的 `3 x 3` 窗口索引基础上，强制要求 projected potential patch 完整落在范围内，并
返回：

| key | shape | 说明 |
| --- | --- | --- |
| `inputs` | `[11, 64, 64]` | 9 个 DP + probe intensity + probe phase |
| `target` | `[1, 16, 16]` | 中心 scan 位置的 projected potential patch |

patch 中心映射：`center = (window_xy + scan_patch_size // 2) * projected_potential_scan_upsample`，
默认上采样倍率 `10`、patch 边长 `16`。projected potential 默认按 patch 内 z-score
归一化（`data.normalization.pp`）。

### 训练

首跑前编辑 `configs/train.yaml` 的数据路径，并填写预训练 encoder：

```yaml
model:
  encoder_ckpt: /path/to/ptycho_mae_pretrain   # encoder_*.pt 所在目录或文件
  freeze_encoder: true                          # 改成 false 可端到端微调
training:
  dataset_path:
    - /data/ptycho/train
  eval_dataset_path:
    - /data/ptycho/val
```

单进程：

```bash
python train.py --config configs/train.yaml
```

A100 多卡：

```bash
torchrun --nproc_per_node=8 train.py --config configs/train.yaml
```

下游 checkpoint 保存为完整模型（`ckpt_*.pt` 中的 `model` 包含 encoder + decoder），
可视化目录 `iter_********/` 下保存 GT / 预测 / 绝对误差拼图。

### 推理

```bash
python inference.py --config configs/train.yaml \
  inference.checkpoint_dir=/path/to/ptycho_pp_downstream
```

推理会在 `inference.output_dir` 写出每个样本的可视化，并把整体 MSE / MAE / PSNR
汇总到 `pp_metrics.json`。

## 注意事项

- 当前实现没有在无 GPU 本地环境跑训练测试；代码只做了静态语法检查。
- A100 上推荐使用默认 `amp_dtype: bf16`，不要改成 `fp16`，除非明确需要。
- 如果真实 HDF5 的 probe 大于 `64 x 64`，数据集会中心裁剪到 `64 x 64`。
- 如果真实 DP 不是 `64 x 64`，请先在数据生成阶段对齐，或修改 `model.dp_size` 并确保模型和数据一致。
