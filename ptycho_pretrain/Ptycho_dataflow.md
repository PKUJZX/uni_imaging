# end2end_single 数据流与维度说明

本文档只说明 `end2end_single` 中从 HDF5 数据到 Dataset、ViT token、预测输出的张量流动与维度变化。数据格式与 `simulate/single.py` 生成的单参数 4D-STEM HDF5 数据类似。

默认记号：

| 符号 | 含义 | 当前默认值 |
|------|------|------------|
| `B` | batch size | 由 dataloader 决定 |
| `P_dp` | 扫描位置 patch 边长 | `3` |
| `P_pp` | 投影势标签 patch 边长 | `16` |
| `S` | projected potential 相对 scan grid 的上采样倍率 | `10` |
| `H_dp, W_dp` | 单张衍射图样输入尺寸 | `64, 64` |
| `N` | 输入 token 数 | `11` |
| `D` | transformer token 维度 | `768` |

---

## 1. 磁盘 HDF5 数据

每个 HDF5 sample group 对应一个结构样本，包含：

| 字段 | 内容 | 典型形状 |
|------|------|----------|
| `diffraction_patterns` | scan grid 上每个位置的衍射图样 | `(scan_y, scan_x, 64, 64)` |
| `probe_intensity` | probe intensity | `(64, 64)` |
| `probe_phase` | probe phase | `(64, 64)` |
| `projected_potential` | 投影势 ground truth | `(pp_y, pp_x)` |

其中 `simulate/single.py` 中的 `diffraction_patterns` 会先 resize 到 `256 x 256`，再中心裁剪为 `64 x 64`；`probe_intensity` 和 `probe_phase` 也保存中心 `64 x 64` 区域。

---

## 2. Dataset Patch 索引

`H5_4DSTEM_Dataset` 在每个 sample 内枚举 `3 x 3` 的 scan-grid 局部窗口：

```text
diffraction_patterns:
  (scan_y, scan_x, 64, 64)
    -> 取局部 scan patch
  (3, 3, 64, 64)
```

局部窗口中心位置映射到 projected potential：

```text
center_y = (i + P_dp // 2) * S
center_x = (j + P_dp // 2) * S

以 (center_y, center_x) 为中心，
裁剪 projected_potential 的 16 x 16 patch
```

只有当 `16 x 16` projected potential patch 完全落在图像范围内时，该 scan patch 才会作为有效样本。

---

## 3. Dataset 输出

单个 `__getitem__` 返回：

```text
inputs, target
```

### 3.1 衍射图样通道

```text
dp_patch: (3, 3, 64, 64)
  -> reshape
dp_tensor: [9, 64, 64]
```

这 9 个通道对应 `3 x 3` scan patch 中的 9 个衍射图样。

### 3.2 Probe 通道

```text
probe_intensity: [1, 64, 64]
probe_phase:     [1, 64, 64]
```

### 3.3 输入拼接

```text
inputs = cat([dp_tensor, probe_intensity, probe_phase], dim=0)

[9, 64, 64] + [1, 64, 64] + [1, 64, 64]
  -> [11, 64, 64]
```

### 3.4 标签

```text
target = projected_potential local patch
  -> [1, 16, 16]
```

---

## 4. Batch 后维度

经过 dataloader collate 后：

| 张量 | 形状 | 含义 |
|------|------|------|
| `inputs` | `[B, 11, 64, 64]` | 9 个局部衍射图样 + probe intensity + probe phase |
| `targets` | `[B, 1, 16, 16]` | 局部 projected potential |

---

## 5. ViT Token 流

模型入口为 `ViTEncoderDecoder`：

```text
inputs: [B, 11, 64, 64]
```

### 5.1 Patch embedding

模型把 11 个输入通道视为 11 个 token，每个 token 是一张 `64 x 64` 图：

```text
[B, 11, 64, 64]
  -> view
[B, 11, 4096]
  -> Linear(4096 -> 768)
[B, 11, 768]
```

随后加上可学习位置编码：

```text
token + pos_embedding
  -> [B, 11, 768]
```

### 5.2 Transformer encoder

```text
[B, 11, 768]
  -> QK-Norm Transformer blocks
[B, 11, 768]
```

token 序列长度始终为 `11`：

| token 范围 | 来源 |
|------------|------|
| `0..8` | `3 x 3` scan patch 的 9 个衍射图样 |
| `9` | probe intensity |
| `10` | probe phase |

### 5.3 取中心 token

模型只取中心衍射图样对应的 token 进入 decoder：

```text
CENTER_PATCH_IDX = 4

[B, 11, 768]
  -> x[:, 4]
[B, 768]
```

`idx=4` 对应 `3 x 3` scan patch 展平后的中心位置：

```text
0 1 2
3 4 5
6 7 8
```

### 5.4 图像 decoder

```text
[B, 768]
  -> Linear
[B, 512 * 4 * 4]
  -> reshape
[B, 512, 4, 4]
  -> ConvTranspose2d upsampling
[B, 1, 16, 16]
```

最终输出与 Dataset 的 target 对齐：

```text
prediction: [B, 1, 16, 16]
target:     [B, 1, 16, 16]
```

---

## 6. 全流程对照图

```text
HDF5 sample group
  diffraction_patterns: (scan_y, scan_x, 64, 64)
  probe_intensity:      (64, 64)
  probe_phase:          (64, 64)
  projected_potential:  (pp_y, pp_x)
          |
          v
Dataset 取局部 patch
  DP:    (3, 3, 64, 64) -> [9, 64, 64]
  Probe: [1, 64, 64] + [1, 64, 64]
  GT:    [1, 16, 16]
          |
          v
Batch
  inputs:  [B, 11, 64, 64]
  targets: [B, 1, 16, 16]
          |
          v
ViT tokenization
  [B, 11, 64, 64] -> [B, 11, 4096] -> [B, 11, 768]
          |
          v
Transformer encoder
  [B, 11, 768]
          |
          v
取中心 token
  [B, 768]
          |
          v
Decoder
  prediction: [B, 1, 16, 16]
```

---

## 7. 维度速查表

| 阶段 | 张量 | 形状 |
|------|------|------|
| 磁盘 DP | `diffraction_patterns` | `(scan_y, scan_x, 64, 64)` |
| 局部 DP patch | `dp_patch` | `(3, 3, 64, 64)` |
| DP 通道化 | `dp_tensor` | `[9, 64, 64]` |
| Probe intensity | `intensity_tensor` | `[1, 64, 64]` |
| Probe phase | `phase_tensor` | `[1, 64, 64]` |
| Dataset 输入 | `inputs` | `[11, 64, 64]` |
| Dataset 标签 | `target` | `[1, 16, 16]` |
| Batch 输入 | `inputs` | `[B, 11, 64, 64]` |
| Token 展平 | `x.view` | `[B, 11, 4096]` |
| Token embedding | `patch_embedding` | `[B, 11, 768]` |
| Transformer 输出 | `encoder(x)` | `[B, 11, 768]` |
| 中心 token | `x[:, 4]` | `[B, 768]` |
| 模型输出 | `prediction` | `[B, 1, 16, 16]` |

---

## 8. 未来可实现：MAE 预训练

当前代码直接用 `3 x 3` scan patch 预测中心位置对应的局部 projected potential。未来可以参考 `uni_imaging` 文档中“先预训练再下游任务”的思路，先做 4D-STEM 的自监督 MAE 预训练。

一个直接的预训练任务是：使用 `3 x 3` scan 邻域中周围 8 个衍射图样，预测中心 1 个衍射图样。

```text
3 x 3 DP patch:

0 1 2
3 4 5
6 7 8

输入:  [0, 1, 2, 3, 5, 6, 7, 8]
目标:  [4]
```

维度示意：

```text
MAE input:  [B, 8, 64, 64]
MAE target: [B, 1, 64, 64]
```

预训练完成后，可将学到的衍射图样局部上下文表征迁移到当前 projected potential 预测任务。这个 MAE 路径只是未来可实现方向，并不是当前 `end2end_single` 代码已经实现的流程。
