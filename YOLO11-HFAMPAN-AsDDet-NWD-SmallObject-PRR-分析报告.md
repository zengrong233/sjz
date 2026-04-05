# YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR 模型分析报告

## 📋 目录
- [模型概述](#模型概述)
- [PRR 版 vs 原版对比](#prr-版-vs-原版对比)
- [PRR 结构主创新详解](#prr-结构主创新详解)
- [A+B 训练辅创新详解](#ab-训练辅创新详解)
- [SSDS 监督补强详解](#ssds-监督补强详解)
- [架构优化分析](#架构优化分析)
- [完整数据流](#完整数据流)
- [消融实验设计](#消融实验设计)
- [训练命令参考](#训练命令参考)

---

## 🎯 模型概述

**YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR** 是在原始小目标检测模型基础上的"三位一体"增强版本：

| 创新维度 | 名称 | 核心思路 |
|---------|------|---------|
| **结构主创新** | PRR (P2/P3-guided Refocus Resampling) | 候选区域热力图 + 软重聚焦增强 |
| **训练辅创新** | A+B (Scale-Routed Optimizer + Noise-Aware Batch Curriculum) | 模块级差异化学习率 + 动态梯度累积 |
| **监督补强** | SSDS (Scale-Specific Dual-Branch Supervision) | P2 聚焦 tiny，P3 聚焦 small |

---

## 📊 PRR 版 vs 原版对比

### 模型规格（scale=n）

| 指标 | 原版 | PRR 版 | 变化 |
|------|------|--------|------|
| 层数 | 706 | 702 | -4（P4/P5 repeat 优化） |
| 参数量 | 8,207,790 | 7,292,282 | **-11.2%** |
| GFLOPs | 24.6 | 23.5 | **-4.5%** |
| CPU 推理延迟 | 543 ms | 503 ms | **-7.4%** |
| P5 分支占比 | 49.6% | 46.8% | 参数重新分配 |
| P2/P3 分支占比 | 8.0% | 9.0% | 小目标分支相对增强 |
| PRR 模块参数 | 0 | 8,780 | 仅 +0.12% |

### 架构优化要点

```
原版问题：
P5 RepBlock (1024ch, repeat=3) 单层占模型 40% 参数
→ 对小目标检测贡献最小，资源严重错配

PRR 版优化：
1. P5 RepBlock repeat 3→2（减 ~1.1M）
2. P4 RepBlock repeat 3→2（减 ~270K）
3. FPN P5/P4 侧 C2f repeat 3→2（加速融合）
4. P2/P3 小目标分支完整保留（精度核心）
5. 新增 PRR 模块（仅 ~8.8K 参数）
```

---

## 🔬 PRR 结构主创新详解

### 设计动机

```
传统 P2/P3 分支的局限：
┌──────────────────────────────────────────────┐
│ P2 特征 (160×160) → 全图均匀处理 → 检测头    │
│                                               │
│ 问题：小目标只占图像极小区域，但 P2 分支      │
│ 对所有位置投入相同计算量 → 效率低、噪声大     │
└──────────────────────────────────────────────┘

PRR 解决方案：
┌──────────────────────────────────────────────┐
│ P2 特征 → [候选热力图] → 高响应区域加权       │
│                ↓                               │
│         [软重聚焦增强] → 局部特征增强          │
│                ↓                               │
│         [残差回注] → 增强后的 P2 → 检测头      │
│                                               │
│ 效果：让网络"先决定去哪儿看，再看得更清楚"    │
└──────────────────────────────────────────────┘
```

### 模块组成

#### 1. CandidateHeatmapHead（候选区域热力图生成）

```python
# 轻量卷积头：depthwise + pointwise + 1x1 输出
输入: P2/P3 特征 (B, C, H, W)
  ↓ Depthwise Conv 3×3
  ↓ BatchNorm + SiLU
  ↓ Pointwise Conv 1×1 (C → C//4)
  ↓ BatchNorm + SiLU
  ↓ Conv 1×1 (C//4 → 1)
  ↓ Sigmoid
输出: 候选热力图 (B, 1, H, W)，值域 [0, 1]
```

**关键设计**：
- 输出 bias 初始化为 -2.0（sigmoid 后约 0.12），实现稀疏先验
- 参数量极小：P2 分支仅 ~1K 参数，P3 分支仅 ~3K 参数

#### 2. SoftRefocusEnhancer（软重聚焦增强，默认实现）

```
输入特征 x (B, C, H, W) + 热力图 hm (B, 1, H, W)
  ↓
weighted = x * hm          # 位置加权：高响应区域被放大
  ↓
enhanced = DW-Conv + PW-Conv(weighted)  # 局部增强
  ↓
output = x + γ * enhanced  # 残差回注（γ 可学习，初始 0.1）
```

**核心思想**：不改变特征图尺寸，只通过空间加权让网络聚焦小目标区域。

#### 3. GridSampleRefiner（可选，基于 grid_sample 的重采样）

```
输入特征 x + 热力图 hm
  ↓
offset = OffsetPredictor(cat(x, hm))  # 预测采样偏移
  ↓
offset = offset * hm * 0.1            # 热力图加权，限制偏移幅度
  ↓
grid = base_grid + offset             # 构建采样网格
  ↓
resampled = F.grid_sample(x, grid)    # 可微重采样
  ↓
output = x + γ * Conv(resampled)      # 残差回注
```

**适用场景**：当 softmask 不够时，grid_sample 提供更强的空间变换能力。

### YAML 中的位置

```yaml
# 层 37-38：PRR 插入在 RepBlock 之后、MFFF 之前
- [26, 1, RefocusSingle, [softmask]]  # 37 P2 重聚焦
- [30, 1, RefocusSingle, [softmask]]  # 38 P3 重聚焦

# 数据流：
# RepBlock(P2) → PRR(P2) → MFFF(P2) → FreqDown → Detect
# RepBlock(P3) → PRR(P3) → MFFF(P3) → FreqDown → Detect
```

**为什么插在 RepBlock 和 MFFF 之间**：
- RepBlock 已完成基础特征增强，PRR 在此基础上做空间聚焦
- MFFF 做频率域增强，PRR 先聚焦再增强，避免对噪声区域做无效频率处理

---

## ⚡ A+B 训练辅创新详解

### A 策略：Scale-Routed Optimizer

```
模型参数按顶层模块自动分为三组：

┌─────────────────────────────────────────────────────┐
│ backbone_group (model.0~9, 10~22)                    │
│   Conv, C3k2, SPPF, C2f, Upsample, Concat           │
│   lr = base_lr × 0.5 | betas = (0.9, 0.999)         │
│   → 稳定训练，避免破坏预训练特征                      │
├─────────────────────────────────────────────────────┤
│ small_object_group (model.23~43)                     │
│   TopBasicLayer, RepBlock, MFFF, RefocusSingle,      │
│   FrequencyFocusedDownSampling, SemanticAlign...     │
│   lr = base_lr × 1.25 | betas = (0.9, 0.9995)       │
│   weight_decay = 0.5 × base_wd                       │
│   → 更快学习小目标特征，更强动量保持方向              │
├─────────────────────────────────────────────────────┤
│ det_head_group (model.44 Detect)                     │
│   lr = base_lr × 1.5 | betas = (0.9, 0.999)         │
│   → 检测头最快响应，快速适配任务                      │
└─────────────────────────────────────────────────────┘
```

**路由机制**：基于顶层模块索引分类 + 参数路径前缀继承，不依赖层号硬编码。

### B 策略：Noise-Aware Batch Curriculum

```
每个 batch 实时计算难度：

tiny_ratio = 小目标数 / 总目标数    （面积 < 16×16）
density_ratio = 平均每图目标数 / 归一化阈值

difficulty = 0.7 × tiny_ratio + 0.3 × density_ratio
diff_ema = 0.9 × diff_ema + 0.1 × difficulty

动态梯度累积：
  diff_ema < 0.3  → accum = base_accum      (easy)
  diff_ema < 0.6  → accum = base_accum × 2  (medium)
  diff_ema ≥ 0.6  → accum = base_accum × 3  (hard)

效果：小目标密集的 batch 自动获得更大有效 batch size，
      降低梯度噪声，稳定训练。
```

---

## 🎯 SSDS 监督补强详解

### 核心机制

```
TAL 分配完成后，对 target_scores 做尺度感知 soft reweighting：

GT box 面积 < 256 (16×16)  → tiny  → P2 层权重 × 1.5
GT box 面积 < 1024 (32×32) → small → P3 层权重 × 1.3
其余                        → 保持不变

效果：
- P2 分支更聚焦极小目标的学习
- P3 分支更聚焦小目标的学习
- P4/P5 不受影响
```

### 两种模式

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| **soft**（默认） | 乘法加权，不改变分配 | 稳定，推荐首选 |
| **hard** | 非对应层权重降为 0.1 | 激进，可能伤召回 |

---

## 🏗️ 架构优化分析

### 参数分布对比

```
原版参数分布（8.2M）：
backbone  ████████████████████  1.45M (17.6%)
P2/P3     ████                  0.66M (8.0%)
P4        ██████████            1.18M (14.4%)
P5        ████████████████████████████████████████  4.07M (49.6%)
Detect    ██████                0.54M (6.6%)
其他      ███                   0.30M (3.8%)

PRR 版参数分布（7.3M）：
backbone  ████████████████████  1.45M (19.9%)
P2/P3     ████                  0.66M (9.0%)
PRR       ▏                     0.01M (0.1%)
P4        ████████              0.96M (13.1%)
P5        ██████████████████████████████████  3.42M (46.8%)
Detect    ██████                0.54M (7.4%)
其他      ███                   0.30M (4.1%)
```

### 优化逻辑

| 优化点 | 原版 | PRR 版 | 理由 |
|--------|------|--------|------|
| P5 RepBlock repeat | 3 | 2 | P5 负责大目标，repeat=2 足够 |
| P4 RepBlock repeat | 3 | 2 | P4 负责中目标，适度减负 |
| FPN C2f repeat (P4/P5) | 3 | 2 | FPN 融合不需要太深 |
| P2/P3 RepBlock repeat | 3 | **3（保持）** | 小目标核心，不能减 |
| P2/P3 C2f repeat | 3 | **3（保持）** | 高分辨率特征需要充分处理 |
| PRR 模块 | 无 | +RefocusSingle ×2 | 仅 8.8K 参数的结构创新 |

---

## 🔄 完整数据流

```
输入图像 (640×640×3)
        │
        ▼
┌─────────────── Backbone ───────────────┐
│ Conv→Conv→C3k2→Conv→C3k2→Conv→C3k2    │
│ →Conv→C3k2→SPPF                        │
│ 输出: P2(160²), P3(80²), P4(40²), P5(20²) │
└────────────────────────────────────────┘
        │
        ▼
┌─────────────── FPN Top-Down ───────────┐
│ P5→Conv→Up→Cat(P4)→C2f(repeat=2)      │
│ →Conv→Up→Cat(P3)→C2f(repeat=2)        │
│ →Conv→Up→Cat(P2)→C2f(repeat=3)        │
└────────────────────────────────────────┘
        │
        ▼
┌─────────────── HFAMPAN ────────────────┐
│ PyramidPoolAgg [P2,P3,P4,P5]           │
│        │                                │
│ P2分支: TopBasicLayer→LAF_h→Injection  │
│         →RepBlock(×3)                   │
│         →★ RefocusSingle(PRR) ★        │
│         →MFFF→FreqDown                  │
│                                         │
│ P3分支: TopBasicLayer→LAF_h→Injection  │
│         →RepBlock(×3)                   │
│         →★ RefocusSingle(PRR) ★        │
│         →MFFF→FreqDown                  │
│                                         │
│ P4分支: LAF_h→Injection→RepBlock(×2)   │
│ P5分支: LAF_h→Injection→RepBlock(×2)   │
│                                         │
│ SemanticAlignmenCalibration(P2+P3)     │
└────────────────────────────────────────┘
        │
        ▼
┌─────────────── Detect ─────────────────┐
│ 四尺度检测头 [P2, P3, P4, P5]          │
│ + NWD Loss                              │
│ + SSDS 尺度感知重加权                   │
└────────────────────────────────────────┘
```

---

## 🧪 消融实验设计

### 主消融矩阵

| # | 配置 | YAML | trainer_mode | 验证目标 |
|---|------|------|-------------|---------|
| 1 | baseline | 原版 | baseline | 基准 |
| 2 | +PRR | PRR 版 | baseline | PRR 结构增益 |
| 3 | +A+B | 原版 | ab | 训练策略增益 |
| 4 | +PRR+A+B | PRR 版 | ab | 结构+训练联合 |
| 5 | +PRR+SSDS | PRR 版 | prr_ssds | 结构+监督联合 |
| 6 | **full** | PRR 版 | full | **三位一体** |

### 细粒度消融

| # | 配置 | 验证目标 |
|---|------|---------|
| 7 | only A | Scale-Routed Optimizer 单独贡献 |
| 8 | only B | Batch Curriculum 单独贡献 |
| 9 | SSDS-soft vs SSDS-hard | 监督模式对比 |
| 10 | PRR-softmask vs PRR-gridsample | 重聚焦实现对比 |

### 关键评估指标

| 指标 | 说明 | 重点关注 |
|------|------|---------|
| mAP50 | 整体精度 | ✅ |
| mAP50:95 | 严格精度 | ✅ |
| AP_tiny (area<16²) | 极小目标 | ⭐ 核心 |
| AP_small (area<32²) | 小目标 | ⭐ 核心 |
| Recall | 召回率 | ✅ |
| 参数量 / GFLOPs | 效率 | ✅ |
| FPS | 推理速度 | ✅ |

---

## 🚀 训练命令参考

### 超算提交

```bash
# 三位一体（推荐）
sbatch train_slurm_ab.sh full

# 消融实验
sbatch train_slurm_ab.sh baseline       # 原始基线
sbatch train_slurm_ab.sh baseline       # 仅 PRR-v3 结构
sbatch train_slurm_ab.sh ab             # PRR-v3 + A+B
sbatch train_slurm_ab.sh full           # PRR-v3 + A+B + SSDS
```

### 本地调试

```bash
# 快速验证模型构建
python "train_yolo111 copy.py" \
    --trainer_mode baseline \
    --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml \
    --data data_VD.yaml --batch 2 --epochs 1

# 验证 A+B + SSDS 训练流程
python "train_yolo111 copy.py" \
    --trainer_mode full \
    --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml \
    --data data_VD.yaml --batch 2 --epochs 3 \
    --enable_ssds --debug_routing
```

---

## 📁 文件清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `ultralytics/nn/core11/refocus_resample.py` | PRR 模块（CandidateHeatmapHead + SoftRefocusEnhancer + GridSampleRefiner） |
| `ultralytics/engine/scale_supervision.py` | SSDS 尺度感知重加权器 |
| `ultralytics/utils/NewLoss/ssds_loss.py` | SSDS 增强检测损失 |
| `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml` | 当前默认 PRR-v3 模型配置 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `ultralytics/nn/tasks.py` | 注册 RefocusSingle / P2P3RefocusResample |
| `ultralytics/engine/optimizer_router.py` | PRR 模块加入小目标分组 |
| `ultralytics/engine/small_object_trainer.py` | 集成 SSDS init_criterion + 日志呼应 |
| `ultralytics/engine/trainer.py` | 修复 L1 正则 None 保护 + 删除错误的 model=weights |
| `ultralytics/nn/core11/GDM.py` | Spatial Reduction Attention 修复 P2 OOM |
| `train_yolo111 copy.py` | 完整消融开关 + SSDS 参数 |
| `train_slurm_ab.sh` | 超算脚本支持 baseline / ab / full |

---

**文档版本**: v2.0 (PRR 版)
**最后更新**: 2026-03-25
