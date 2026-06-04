# AGENTS.md

本文件用于协同会话中指导 Codex 和 Kiro / Agent 处理 `ultralyticsPro--YOLO11` 项目。所有回答、分析和交接记录默认使用简体中文。

## 项目定位

这是一个基于 Ultralytics YOLO11 的遥感小目标检测改进项目，当前研究重点不是通用 YOLO11，而是以下主结构及其轻量化消融：

```text
YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f
```

当前论文主线候选为：

```text
P1-NoSAC-FullC2f-LiteD1R2
```

核心任务通常围绕 RS-STOD / USOD 小目标检测训练、消融、自检和超算脚本展开。

## 必须优先遵守

1. 使用 `superpowers` / `Aegis` 本地 skills 做分析、自检、调试和验证。
2. 遇到 2 个以上互不依赖的只读检查，优先并行读取。
3. 不要未经确认删除、重置、覆盖已有实验结果。
4. 不要把不同代码状态下的结果混为同一 baseline。
5. 没有 fresh verification evidence 时，不要声称"已通过""已完成""可用"。
6. 本地 Windows 环境的 torch 可能有 DLL 问题，模型实例化和训练结论以超算为准。

## 当前代码状态

已知重要状态：

- `ultralytics/utils/loss.py` 中 NWD / GWD / WIoU 被普通 IoU 覆盖的问题已修。
- `loss: NWD` 当前应走 NWD 分支，不应再被后续普通 IoU 分支重置。
- `TopBasicLayer depths` 已参数化，可以通过 YAML 写 `[128, 1]` 或 `[256, 1]` 控制。
- `Attention.scale` 目前仍未修；如果以后修成 `attn * self.scale`，必须重新建立 baseline。
- `ioulossone.py` 中 NWD size 项 `/4` 暂不作为 bug 处理，不要直接改成 `/12`。

## 主结构说明

主 YAML：

```text
YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml
```

关键结构：

- Backbone：YOLO11 C3k2 + SSA / Sobel / SAConv。
- Neck：HFAMPAN + PyramidPoolAgg。
- PPA 输入：`[21, 17, 13, 9]`，表示全 C2f + SPPF 输入。
- P2/P3：`TopBasicLayer -> LAF_h -> Injection -> RepBlock`。
- P4：`LAF_h -> Injection -> RepBlock`。
- PRR：`RefocusSingle(gridsample)` 作用在 P2/P3。
- MFFF：频域增强作用在 P2/P3。
- Head：`AsDDet [P2-MFFF, P3-MFFF, P4-RepBlock]`。
- Loss：`NWD`。

## C3k2 + SSA / Sobel / SAConv 设计判断

当前 C3k2 增强不是替代 YOLO11 C3k2 主分支，而是在主分支外增加辅助分支：

```text
C3k2/C2f 主分支
+ Sobel 边缘分支
+ LightSAConv 上下文分支
-> concat
-> 1x1 Conv 融合
```

四层递进：

- Layer 2 / P2：Sobel only，强化小目标边界。
- Layer 4 / P3：Sobel + SAConv(d=2)，补局部上下文。
- Layer 6 / P4：C3k + Sobel + SAConv(d=3)，补语义和大感受野。
- Layer 8 / P5：C3k + Sobel + SAConv(d=3)，补深层语义。

已知风险：

- SAConv 有 small conv + dilation conv 两条标准卷积分支，P4/P5 成本较高。
- Sobel 是固定边缘算子，可能放大道路、建筑、纹理背景。
- LightSAConv 的 switch 是单通道空间门控，不是通道级自适应。

这些不是当前主线 bug。若优化，应新建 `SSALite` 分支，不要覆盖 E1 主结果。

## 实验路线

### E1: LiteD1R2

当前最推荐的轻量化主线。

改动：

- `TopBasicLayer depth: 2 -> 1`
- `P2/P3 RepBlock repeat: 3 -> 2`
- `P4 RepBlock repeat` 保持 2

在 n scale 下：

- P2/P3 TopBasicLayer：实际 2 block -> 1 block
- P2/P3 RepBlock：实际 2 -> 1
- P4 RepBlock：实际 1，不变

结论：主结果候选。

### E2: LiteD1Only

只改：

- `TopBasicLayer depth: 2 -> 1`

不改 RepBlock。用途：隔离 attention depth 的独立影响。只作为消融，不替代 E1。

### LiteR2Only

只改：

- `P2/P3 RepBlock repeat: 3 -> 2`

不改 TopBasicLayer。用途：隔离 RepBlock 减法影响。已有结果显示 RS-STOD 明显掉点，不适合作主线。

### NoTopBasic

激进消融：

```text
C2f -> Injection -> RepBlock
```

删除 P2/P3 的 `TopBasicLayer + LAF_h`。用途：证明 TopBasicLayer / LAF_h 的价值。不要优先作为主线。

### SSALite

建议中的新分支，不是当前主结果。

目标：

- SAConv 改为 depthwise separable。
- Sobel 分支增加可学习 gate。
- switch 从单通道空间门控改成 group-wise 或 channel-aware gate。
- P5 可考虑关闭 SAConv，只保留 C3k + Sobel。

必须新建 YAML / 模块分支，不要直接污染 E1。

## 已有主结果口径

### RS-STOD E1

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod\rsstod_lited1r2_smoke\rsstod_lited1r2_b6_e100_85036
```

虽然目录名含 `e100`，但实际为 300 epoch 结果。

关键指标：

- best epoch: 261
- mAP50: 0.7512
- mAP50-95: 0.45767
- Recall: 0.72438
- params: 5.697M
- GFLOPs: 26.8

采用 `weights/best.pt`。

### USOD E1

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\USOD\usod_lited1r2_b4_e300_from85139_85210
```

关键指标：

- best epoch: 299
- mAP50: 0.8293
- mAP50-95: 0.33549
- Recall: 0.75639
- params: 5.6969M
- GFLOPs: 26.8

采用 `weights/best.pt`。

### NWPU-VHR10 E1 (clean, 87416)

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod_usod_nwpu\nwpu_clean_yolo11n_b6_e300_87416
```

关键指标：

- best epoch: 261 / 300 ep / patience=100
- mAP50: 0.92279
- mAP50-95: 0.55702
- Recall: 0.88140
- Precision: 0.85375
- params: 5.697M
- GFLOPs: 26.8
- FPS @ A30: 49.2

采用 `weights/best.pt`。**替代旧 `86913nwpu_best.pt`** (refit chain 来源不干净，mAP50-95=0.53562)。

## 主结构实验卡 (RS-STOD / USOD / NWPU)

参考实验卡来源：`C:/Users/86155/Desktop/协同/memory/experiments/`。本节为各数据集主结构 (LiteD1R2 + NWD) 已完成实验的整合视图，便于跨实验对比与口径锁定。

### RS-STOD 主结构实验卡

- 实验范围：LiteD1R2 + NWD 在 RS-STOD 5 类 (Small/Large Vehicle, Ship, Airplane, Storage Tank) 上的 anchor + 消融 + 类内增强对拍
- 当前主线：`85036` (E1 LiteD1R2 300 ep clean)
- 数据集 PREP 锚定：`val-only Δ=-0.00026` (`exp-20260527-val-only-lited1r2-3ds-prep`)
- 短板：Large Vehicle 长期低 (210 train inst / val mAP50-95 ≈ 0.155)，是数据集结构性瓶颈

| 实验 | 起点 / 关键改动 | best ep | P | R | mAP50 | mAP50-95 | params | GFLOPs | 裁定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **E1 anchor 85036** | yolo11n.pt clean 300 ep | 261 | 0.78297 | 0.72438 | 0.75120 | **0.45767** | 5.697M | 26.8 | 主线 |
| `87109` lr-refit (high-lr) | from 85036 lr0=0.002 warmup=0 | 24 | — | — | — | 0.45352 | 5.697M | 26.8 | 失败，归档 |
| `87259` lr-refit (low-lr) | from 85036 lr0=0.0005 warmup=1.0 | 10 | 0.78751 | 0.72076 | 0.75207 | 0.45880 | 5.697M | 26.8 | 持平 (+0.00113 < 6× 噪声 0.003)，不替代 |
| `87414` LVBoost x2 | from 85036 + Large Vehicle ×2 oversampling | 54 | 0.79949 | 0.71529 | 0.75067 | 0.45816 | 5.697M | 26.8 | LV Recall 0.319→0.438 显著，整体不替代 |
| `85750` NoTopBasic 消融 | yolo11n.pt + 删 P2/P3 TopBasicLayer+LAF_h | 249 | 0.75254 | 0.71542 | 0.74021 | 0.45514 | 5.646M | 25.7 | 消融证据 (-1.1pt mAP50, +0.05M params 减少不值) |
| SSALite (ref) | yolo11n.pt + DW+PW SAConv + group switch + Sobel gate | — | — | — | — | — | — | — | 单独消融见 `exp-20260512-rsstod-lited1r2-ssalite-e100` 等 |

实验卡产物：

- `exp-20260512-rsstod-notopbasic-e300.md`
- `exp-20260527-rsstod-lited1r2-refit-lvboost-e100.md`
- `exp-20260527-rsstod-lited1r2-core-ablation-e300.md`
- `exp-20260527-rsstod-lited1r2-ssalite-e300.md`
- `exp-20260527-val-only-lited1r2-3ds-prep.md`

RS-STOD 关键结论：

- **lr-refit 不能稳定超过 anchor**：87109/87259 均未通过 ≥anchor+0.003 门槛
- **LVBoost x2 是类别专项工具**：可写 dataset-specific 增强章节，但不替代 E1 主线
- **NoTopBasic 是有效消融证据**：删 TopBasicLayer+LAF_h 算力收益 < 性能代价，归档
- **下一步建议**：保留 85036 主线，论文消融表加入 NoTopBasic / SSALite / LVBoost x2 三条横切

### USOD 主结构实验卡

- 实验范围：LiteD1R2 + NWD 在 USOD 单类 (vehicle) 极小目标上的 anchor + loss 切换 + clean augmentation 对拍
- 当前主线：`85210` (二阶段，`from 85139 best.pt`)
- 数据集 PREP 锚定：`val-only Δ=-0.00025` (`exp-20260527-val-only-lited1r2-3ds-prep`)
- 数据特征：train 中位 13×11 px，94.6% < 16×16 px，34470 train inst / 11613 val inst / 单类，已饱和无长尾问题
- 推理工具状态：`Soft-NMS` 已在 `ultralytics/utils/ops.py:_soft_nms_gaussian` 实现，可直接通过 `args.soft_nms / soft_nms_sigma` 调用

| 实验 | 起点 / 关键改动 | best ep | P | R | mAP50 | mAP50-95 | params | GFLOPs | 裁定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **E1 anchor 85210** | from 85139 best.pt 二阶段 | 299 | 0.84397 | 0.75639 | 0.82930 | **0.33549** | 5.697M | 26.8 | 主线 |
| `87110` lr-refit (high-lr) | from 85210 lr0=0.002 warmup=0 | 37 | — | — | — | 0.32891 | 5.697M | 26.8 | 失败，归档 |
| `87415` lr-refit (low-lr) | from 85210 lr0=0.0005 warmup=1.0 | 100 | 0.82893 | 0.76113 | 0.82647 | 0.33048 | 5.697M | 26.8 | 失败 (Precision -0.015)，归档 |
| `86369` CIoU | from yolo11n.pt clean，loss=CIoU | 300 | 0.80951 | 0.74796 | 0.81185 | 0.31953 | 5.697M | 26.8 | NWD 在 USOD 上仍优于 CIoU，不替换 |
| `87663` U0 clean (mosaic=1) | from yolo11n.pt clean 300 ep | 293 | 0.81576 | 0.77422 | 0.82386 | 0.33024 | 5.697M | 26.8 | 比 anchor -0.00525，单阶段难以复现二阶段 |
| `87798` U1a clean (mosaic=0) | from yolo11n.pt + mosaic=0 | 299 | 0.82531 | 0.76440 | 0.82667 | 0.33042 | 5.697M | 26.8 | vs U0 +0.00018 < 噪声，单变量验证无效 |

实验卡产物：

- `exp-20260527-usod-lited1r2-loss-augmentation.md`
- `exp-20260527-val-only-lited1r2-3ds-prep.md`

USOD 关键结论：

- **lr-refit 在 USOD 上不奏效**：与 RS-STOD 不同，87415 即使低 lr+warmup 也无法接近 anchor
- **NWD 仍是 USOD 跨数据集主线 loss**：CIoU 86369 比 anchor 低 0.016 mAP50-95
- **mosaic / scale / translate 网格无效**：U0 vs U1a +0.00018，已宣告无证据需求
- **85210 二阶段路径无法被单阶段 clean run 复现**：`87663` clean 比 anchor 低 0.005，是数据集 + 训练协议特性
- **下一步建议**：
  - P0: `Soft-NMS val-only` (V0/V1/V2/V3 sigma 网格，5 分钟实验)
  - P1: 若有原始未压缩 USOD 大图 → tile 训练 + 滑窗推理；否则上 `imgsz=800` clean 单变量
  - 不再调 mosaic / scale / translate / max_det

### NWPU-VHR10 主结构实验卡

- 实验范围：LiteD1R2 + NWD 在 NWPU-VHR10 10 类上的 refit anchor → clean upgrade → BridgeBoost 对拍
- 当前主线候选：`87416` (clean from yolo11n.pt 300 ep) — **替代旧 86913 refit anchor**
- 数据集 PREP 锚定：`val-only Δ=+0.00044` (`exp-20260527-val-only-lited1r2-3ds-prep`)
- 数据特征：86 val img / 480 inst，bridge 8 inst、ship 5 inst、storage tank 3 张图，**类级 AP 方差 ±0.10**
- 已知数据风险：Roboflow 导出 Resize 400×400 Stretch，可能损害 bridge 形状定位

| 实验 | 起点 / 关键改动 | best ep | P | R | mAP50 | mAP50-95 | params | GFLOPs | 裁定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 旧 anchor 86913 (refit) | refit chain (历史多阶段) | 11 | 0.78443 | 0.87529 | 0.89063 | 0.53562 | 5.697M | 26.8 | 已被 87416 替代 |
| `87111` lr-refit (high-lr) | from 86913 lr0=0.001 warmup=0 | 59 | — | — | — | 0.53526 | 5.697M | 26.8 | 失败 (last 衰退到 0.494)，归档 |
| `87260` lr-refit (low-lr) | from 86913 lr0=0.0005 warmup=1.0 | 61 | 0.78934 | 0.88385 | 0.88706 | 0.52540 | 5.697M | 26.8 | 失败 (-0.01 vs 86913)，归档 |
| **`87416` clean (主候选)** | yolo11n.pt clean 300 ep | 261 | 0.85375 | 0.88140 | 0.92279 | **0.55702** | 5.697M | 26.8 | **替代 86913，新主线** |
| `87656` BridgeBoost x2 | from yolo11n.pt + bridge ×2 oversampling | 199 | 0.56940 | 0.79129 | 0.67258 | 0.39652 | 5.697M | 26.8 | 失败 (val img 数变 86→133，口径不一致) |

实验卡产物：

- `exp-20260527-nwpu-lited1r2-clean-bridgeboost.md`
- `exp-20260527-val-only-lited1r2-3ds-prep.md`

NWPU 关键结论：

- **clean run 比 refit 稳定且更高**：87416 mAP50-95=0.557 比 86913 anchor +0.0214，是 NWPU 上最关键的提升
- **lr-refit 全部失败**：87111/87260 在 NWPU 上即使低 lr+warmup 也无法超过原 anchor
- **类级 AP 不作为消融判据**：bridge val 8 inst, test 5 inst，单 inst 翻转 = ±12% Recall
- **BridgeBoost 失败但归因不干净**：87656 val img 数变 (86→133)，口径破坏，需固定 split 重跑
- **下一步建议**：
  - P0: AGENTS.md "已有主结果口径" 把 NWPU anchor 从 86913 升级到 87416 (本次更新已落实)
  - P1: 87416 best.pt 在 test 集 val-only，给 bridge 加一个 test 锚锁定方差
  - P2: 如能拿到原始非 Roboflow 大图 → 重建数据 + bridge ×2；否则继续在 87416 上做 ablation 即可
  - 不为 8 inst 的 bridge 单独调超参

### 三数据集横切对比 (LiteD1R2 + NWD anchor)

| 维度 | RS-STOD (85036) | USOD (85210) | NWPU (87416) |
|---|---:|---:|---:|
| 类数 | 5 | 1 | 10 |
| val img / inst | 231 / 4732 | 700 / 11613 | 86 / 480 |
| mAP50 | 0.751 | 0.829 | **0.923** |
| mAP50-95 | 0.458 | 0.335 | **0.557** |
| Recall | 0.724 | 0.756 | **0.881** |
| Precision | 0.783 | 0.844 | 0.854 |
| best epoch / total | 261 / 300 | 299 / 300 | 261 / 300 |
| params | 5.697M | 5.697M | 5.697M |
| GFLOPs | 26.8 | 26.8 | 26.8 |
| 起点路径 | clean 单阶段 | 二阶段 (85139→85210) | clean 单阶段 |
| FPS @ A30 | 29.5 | 58.7 | 49.2 |

跨数据集观察：

- **同一 LiteD1R2 在三套不同的数据特征上都站得住**：单类 / 多类 / 长尾 / 极小目标 / 中等目标场景全覆盖
- **lr-refit 在三个数据集上都不是普适修复手段**：RS-STOD 持平、USOD 失败、NWPU 失败
- **clean run 是更稳的论文主表选择**：对应 NWPU 的 +0.021 / RS-STOD 持平 (anchor 本就 clean)
- **类级 AP / 单类问题 (LV / bridge / vehicle) 主要是数据问题**，不应通过改 LiteD1R2 结构修复

## val-only 数据口径锚定 (87xxx, 5p13 PREP)

使用 87xxx 系列 best.pt 对 `data_*_PREP_5p13.yaml` 做纯 val 验证（不训练），确认 PREP 口径未改变旧主结果指标。所有 best.pt 可作为后续 refit 的锚使用。

| 数据集 | best.pt | 旧 mAP50-95 | val-only mAP50-95 | Δ |
|---|---|---:|---:|---:|
| RS-STOD | `85036rsstod_best.pt` | 0.45767 | 0.45741 | -0.00026 |
| USOD    | `85210usod_best.pt`   | 0.33549 | 0.33524 | -0.00025 |
| NWPU    | `86913nwpu_best.pt`   | 0.53562 | 0.53606 | +0.00044 |

口径补充：

- 三组 Δ 均在 ±0.0005 (FP32 评估噪声) 内 → **H1a 通过**：RS-STOD/USOD/NWPU 87109/87110/87111 复训掉点不是数据预处理问题。
- 模型规模 (fuse 后): RS-STOD 5,697,288 / USOD 5,696,892 / NWPU 5,697,783 params, 统一报告口径 `5.697M / 26.8 GFLOPs` (head params 随 nc 单调变化)。
- val 日志会同时打印未 fuse (589 layers / 27.0 GFLOPs) 和 fuse (515 layers / 26.8 GFLOPs) 两个 summary，论文统一报 fuse 后口径。
- NWPU 类级 AP 方差 ±0.10 (bridge/ship/storage tank 实例数 5-110 不等)，类级指标不作为消融判据。

## refit 训练注意事项 (基于 87109/87110/87111 失败经验 + val-only 锚定)

从 best.pt 续训时，避免从头训练的默认超参，否则起点扰动过大、早期触顶后回落：

| 超参 | 从头训练默认 | refit 推荐 | 原因 |
|---|---|---|---|
| `lr0` | 0.01 | **≤ best 训练时 lr0 / 10** (典型 0.0005) | best 已在低 loss 邻域，大 lr 会跳出邻域 |
| `warmup_epochs` | 3.0 | **≥ 1.0** (禁止 0) | 续训直接全速会扰动 best 权重 |
| `warmup_bias_lr` | 0.1 | **0.0** | 同上 |
| `patience` | 100 | 80 | 早停防尾段过拟合 |
| `close_mosaic` | 10 | 10 | 训练 ≥ 100 ep 时仍有效；< 100 ep 时不会触发，无需关心 |

**禁忌**：

- ❌ refit 用 `lr0=0.001~0.01`（87109 ÷ 5、87111 ÷ 10 都翻车）
- ❌ `warmup_epochs=0`（87109/87110/87111 均触发漂移）
- ❌ 直接拿 87109/87110/87111 的 best 当主结果（已确认 mAP50-95 低于原 85036/85210/86913 主线）

## 不采用的结果

不要把以下结果作为最终主结果：

- `usod_lited1r2_b4_e300_from_usodbest_85355`
  - mAP50-95 低于 85210，属于二阶段微调无效提升。
- `rsstod_lited1r2_e300/rsstod_lited1r2_b6_e300_from84937best_85357`
  - mAP50-95 低于 85036。
- `LiteR2Only`
  - RS-STOD 明显掉点，只保留为消融证据。
- `rsstod_lited1r2_b6_e300_from85036_87109` (refit, lr0=0.002 warmup=0)
  - best ep=24 mAP50-95=0.45352 (低于 85036 的 0.45767)，归档为 refit 训练策略反例
- `usod_lited1r2_b4_e300_from85210_87110` (refit, lr0=0.002 warmup=0)
  - best ep=37 mAP50-95=0.32891 (低于 85210 的 0.33549)，归档为 refit 训练策略反例
- `nwpu_lited1r2_b6_e300_from86913_87111` (refit, lr0=0.001 warmup=0)
  - best ep=59 mAP50-95=0.53526 (与 86913 0.53562 持平但末段衰退到 0.494)，归档为 refit 训练策略反例
- `rsstod_notopbasic_b6_e300_85750`
  - mAP50=0.740 / mAP50-95=0.456 / Recall=0.714 / params=5.646M / GFLOPs=25.7
  - 相对 E1 LiteD1R2：mAP50 -0.011 / Recall -0.010，但算力仅减 1.1 GFLOPs (-4%)
  - 结论：去掉 TopBasicLayer + LAF_h 的算力收益 < 性能代价，归档为消融证据
  - 价值：证明 TopBasicLayer + LAF_h 对 RS-STOD 的 Recall / mAP50 有 ~1pt 贡献

## 超算路径

常用项目路径：

```bash
/share/home/u2415363072/5.3/ultralyticsPro--YOLO11
/share/home/u2415363072/5.9/ultralyticsPro--YOLO11
```

常用权重：

```bash
/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/best_pt/rsstod_best.pt
/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/best_pt/usod_best.pt
/share/home/u2415363072/5.3/ultralyticsPro--YOLO11/best_pt/yolo11n.pt
```

常用数据集：

```bash
datasets/RS-STOD
datasets/USOD
```

若当前项目目录没有数据，可回退检查：

```bash
/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets
```

## 训练约束

建议默认：

- RS-STOD：`batch=4` 或 `batch=6`，若 OOM 降到 4。
- USOD：`batch=4`。
- `amp=false` 更稳。
- `workers=8`。
- 使用 `--trainer_mode full --enable_ssds --ssds_mode soft`。

不要默认使用：

- USOD `batch=32`
- 未验证的新 loss 改动
- 修 `Attention.scale` 后继续复用旧 baseline 结论

## 常用验证

超算上每次新 YAML / 新模块至少运行：

```bash
python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('模型.yaml', task='detect').info()"
```

本地 Windows 若 torch DLL 报错，不要据此否定结构；以超算 dry-run 为准。

## 文档与沟通风格

- 所有分析使用简体中文。
- 先给结论，再给证据。
- 对训练结果要区分：
  - best 指标
  - final 指标
  - 是否早停
  - 是否从预训练 / best.pt 续训
  - 是否和旧代码状态混用
- 对 review 请求，优先列问题和风险，再给总结。
- 对超算脚本，注意检查引号闭合、路径存在、`logs/` 和 `runs/` 目录创建。

## 关键禁忌

1. 不要把 `Attention.scale` 修复前后的结果混成一组。
2. 不要直接把 `NWD /4` 改成 `/12`。
3. 不要直接覆盖 E1 主 YAML 做 SSALite。
4. 不要把 NoTopBasic 当第一主线。
5. 不要忽略 batch OOM 风险。
6. 不要只看 mAP50，小目标实验必须同时看 mAP50-95 和 Recall。

## Workflow Rules

- **工作流路由与 superpowers 使用规范**：参见 [`rules/workflow-superpowers.md`](./rules/workflow-superpowers.md)
  - 任务分流：Fast Path / Standard Path / Heavy Path
  - superpowers 使用原则：默认采用满足质量要求的最短路径
  - 流程升级 / 降级条件
  - 并行与子代理的边界约束
  - 交付门禁：无验证证据不声称完成
