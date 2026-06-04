# 2026-05-31 主实验卡与论文结果口径

本文件用于下个线程继续更新 `main.tex`。当前论文主线仍为：

```text
YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2
```

简称：

```text
LiteD1R2 + NWD
```

## 1. 当前主线结论

`LiteD1R2 + NWD` 是当前最稳的论文主结构。

核心理由：

- 在 RS-STOD、USOD、NWPU-VHR10 三个数据集上均有可用主结果。
- 参数和算力稳定在约 `5.697M / 26.8 GFLOPs`。
- 相比原 NoSAC-FullC2f，LiteD1R2 主要降低 FLOPs，精度基本保持。
- `AIMLAux / fd_aux` 当前不稳定，不能替代主线。
- `Attention.scale` 仍未修，不要把未来修复后的结果与当前 baseline 混用。
- `NWD /4` 不作为 bug，不要改成 `/12`。

## 2. 三数据集主结果

### 2.1 RS-STOD

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod\rsstod_lited1r2_smoke\rsstod_lited1r2_b6_e100_85036
```

说明：目录名含 `e100`，但实际是 300 epoch 主结果。

关键指标：

```text
Experiment: 85036
Model: LiteD1R2 + NWD
best epoch: 261
Precision: 0.78297
Recall: 0.72438
mAP50: 0.75120
mAP50-95: 0.45767
params: 5.697M
GFLOPs: 26.8
```

论文裁定：

```text
RS-STOD 主结果采用 85036 best.pt。
```

主要问题：

- `Large Vehicle` 类长期偏弱，属于数据集结构性瓶颈。
- LVBoost x2 能提升 Large Vehicle Recall，但整体 mAP50-95 不能显著超过 anchor。

### 2.2 USOD

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\USOD\usod_lited1r2_b4_e300_from85139_85210
```

关键指标：

```text
Experiment: 85210
Model: LiteD1R2 + NWD
best epoch: 299
Precision: 0.84397
Recall: 0.75639
mAP50: 0.82930
mAP50-95: 0.33549
params: 5.6969M
GFLOPs: 26.8
```

论文裁定：

```text
USOD 主结果采用 85210 best.pt。
```

重要补充：

```text
85210 + Soft-NMS sigma=0.3:
Precision: 0.83455
Recall: 0.81270
mAP50: 0.84757
mAP50-95: 0.34087
FPS: 约 9.9
```

Soft-NMS 可作为推理增强上界或补充实验，不建议作为部署主结果，因为后处理耗时明显增加。

### 2.3 NWPU-VHR10

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod_usod_nwpu\nwpu_clean_yolo11n_b6_e300_87416
```

关键指标：

```text
Experiment: 87416
Model: LiteD1R2 + NWD
Training: yolo11n.pt clean 300 epoch
best epoch: 261
Precision: 0.85375
Recall: 0.88140
mAP50: 0.92279
mAP50-95: 0.55702
params: 5.697M
GFLOPs: 26.8
FPS @ A30: 49.2
```

论文裁定：

```text
NWPU 主结果采用 87416 clean。
87416 替代旧的 86913 refit-chain 结果。
```

注意：

- NWPU Val86 验证集较小，类级 AP 方差大。
- `bridge` 只有 8 个 val instances，不能只为 bridge 调主线。

## 3. 主要消融与负结果

### 3.1 LiteD1R2 主线

设计：

```text
TopBasicLayer depth: 2 -> 1
P2/P3 RepBlock repeat: 3 -> 2
P4 RepBlock repeat: 保持 2
```

在 n scale 下实际效果：

```text
P2/P3 TopBasicLayer: 2 blocks -> 1 block
P2/P3 RepBlock: 2 blocks -> 1 block
P4 RepBlock: 1 block 不变
```

裁定：

```text
LiteD1R2 是当前轻量化主线。
```

### 3.2 NoTopBasic

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod\rsstod_notopbasic_b6_e300_85750
```

指标：

```text
mAP50: 0.74021
mAP50-95: 0.45514
Recall: 0.71542
params: 5.646M
GFLOPs: 25.7
```

相对 RS-STOD 85036：

```text
mAP50: -0.011
mAP50-95: -0.0025
Recall: -0.009
GFLOPs: -1.1
```

裁定：

```text
删除 TopBasicLayer + LAF_h 的算力收益小于性能代价。
NoTopBasic 仅作为消融证据，不作为主线。
```

### 3.3 LiteR2Only

用途：

```text
只改 P2/P3 RepBlock repeat 3 -> 2，
不改 TopBasicLayer depth。
```

裁定：

```text
RS-STOD 明显掉点，不适合作为主线。
```

### 3.4 CIoU 替换 NWD

USOD 结果：

```text
Experiment: 86369
loss: CIoU
mAP50: 0.81185
mAP50-95: 0.31953
```

相对 USOD 85210：

```text
mAP50-95: -0.01596
```

裁定：

```text
NWD 在 USOD 上明显优于 CIoU。
论文主线继续使用 NWD。
```

### 3.5 USOD imgsz=800

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod_usod_nwpu\usod_img800_from85210_b2_e100_88292
```

关键指标：

```text
best epoch: 7
Precision: 0.83608
Recall: 0.76025
mAP50: 0.81952
mAP50-95: 0.32198
FPS: 约 29.8
```

相对 USOD 85210：

```text
mAP50-95: -0.01351
```

裁定：

```text
USOD imgsz=800 失败，不继续 300 epoch。
```

## 4. AIMLAux / fd_aux 实验卡

### 4.1 AIML 与当前 fd_aux 的关系

AIML 论文强调：

```text
同一样本两个增强视图拉近: l_s
不同表征维度正交防塌缩: l_r
```

当前项目中的 `fd_aux` 实际是：

```text
P2/P3 positive anchor feature
vs
GT-center feature distribution
```

因此准确表述应为：

```text
AIML-inspired feature distribution auxiliary loss
```

不要写成：

```text
We implement AIML.
```

### 4.2 USOD AIMLAux

结果：

```text
Experiment: usod_nwd_aimlaux_w0003_b4_e100_88162
fd_aux_weight: 0.003
mAP50: 0.78853
mAP50-95: 0.30882
```

裁定：

```text
USOD 上 AIMLAux 明显失败。
不进入主线。
```

### 4.3 NWPU AIMLAux 单次

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod_usod_nwpu\nwpu_nwd_aimlaux_w0003_val86_yolo11n_b6_e300_88163
```

指标：

```text
Val86:
Precision: 0.88323
Recall: 0.84252
mAP50: 0.91917
mAP50-95: 0.54494
bridge mAP50-95: 0.224

Test split:
overall mAP50-95: 0.549
bridge mAP50-95: 0.516
```

裁定：

```text
对 bridge/test 有局部潜力，但整体低于 87416。
只能作为讨论项，不能作为主线。
```

### 4.4 NWPU AIMLAux 三 seed 稳定性

结果目录：

```text
C:\Users\86155\Desktop\水经注\数据\rsstod_usod_nwpu\nwpu_lited1r2_nwd_aimlaux_seed_stability
```

三 seed：

```text
seed0 / 88313:
best epoch = 263
mAP50 = 0.89707
mAP50-95 = 0.54630

seed1 / 88314:
best epoch = 159
mAP50 = 0.86249
mAP50-95 = 0.53168

seed2 / 88315:
best epoch = 203
mAP50 = 0.87917
mAP50-95 = 0.53782
```

统计：

```text
mean mAP50-95 = 0.53860
std = 0.00599
min / max = 0.53168 / 0.54630
```

相对 NWPU 87416：

```text
87416 mAP50-95 = 0.55702
三 seed 全部低于 87416
```

裁定：

```text
AIMLAux w=0.003 在 NWPU 上不稳定。
不建议继续作为主线增强。
```

### 4.5 AIMLAux 初始化再 NWD-only 微调

计划：

```text
88163 best.pt
-> 普通 LiteD1R2 + NWD
-> fd_aux 关闭
-> lr0=0.0002
-> 100 epoch
```

目的：

```text
验证 AIMLAux 是否只适合作为表征初始化，而不适合全程参与检测 loss。
```

当前状态：

```text
已给超算 5.13 sbatch 命令，等待结果。
```

## 5. LEVIR-Ship / HRSID 状态

### 5.1 LEVIR-Ship

本地数据：

```text
C:\Users\86155\Desktop\53\ultralyticsPro--YOLO11\datasets\LEVIR-Ship.v3i.yolov11
```

问题：

```text
原数据 nc=4，但 valid/test 只有 class 0。
需要合并为 1 类 ship。
```

已发现无效结果：

```text
C:\Users\86155\Desktop\水经注\数据\levirship\levirship_nwd_aimlaux_w0003_1cls_b6_e300_88316
```

不能采用原因：

- results.csv 只有 28 行，不是完整 300 epoch。
- 日志实际扫描原始 `LEVIR-Ship.v3i.yolov11` 路径，不是干净 `1cls-clean`。
- validation 出现大量 duplicate label warning。

下一步：

```text
5.30 重新构建 LEVIR-Ship-1cls-clean。
同时跑：
B0: LiteD1R2 + NWD
B1: LiteD1R2 + NWD + AIMLAux
```

判据：

```text
B1 mAP50-95 >= B0 + 0.003
```

### 5.2 HRSID

本地数据：

```text
C:\Users\86155\Desktop\53\ultralyticsPro--YOLO11\datasets\HRSID.v1i.yolov11
```

关键发现：

```text
HRSID 标签是 segmentation polygon 格式，不是 detect 5 列格式。
```

本地 split：

```text
train images = 3922
valid images = 1119
test images = 558
```

原始标签：

```text
class x1 y1 x2 y2 x3 y3 ...
```

下一步：

```text
在 5.30 将 polygon 外接矩形转为 HRSID-DET。
先跑 LiteD1R2 + NWD + AIMLAux。
后续必须补 LiteD1R2 + NWD baseline，否则无法判断 AIMLAux 是否有效。
```

## 6. TeX 写作建议

### 6.1 主结果表

建议主表只放：

```text
RS-STOD 85036
USOD 85210
NWPU 87416
```

表字段：

```text
Dataset
Model
Precision
Recall
mAP50
mAP50-95
Params
GFLOPs
```

### 6.2 消融表

建议消融表包含：

```text
LiteD1R2
NoTopBasic
LiteR2Only
CIoU
USOD imgsz800
AIMLAux w=0.003
Soft-NMS sigma=0.3
```

注意：

- Soft-NMS 是推理增强，不是结构改进。
- AIMLAux 是负结果 / 讨论项，不是主线。
- NoTopBasic 证明 TopBasicLayer + LAF_h 有价值。

### 6.3 方法章节措辞

推荐措辞：

```text
We use an NWD-based detection loss to better handle tiny object localization.
```

```text
Inspired by augmentation-invariant manifold learning, we further investigate
a train-only feature distribution auxiliary constraint on P2/P3 features.
However, its gains are not stable across seeds and datasets, so it is reported
as an auxiliary analysis rather than incorporated into the final architecture.
```

不要写：

```text
We implement AIML in YOLO.
```

## 7. 当前最终裁定

```text
最终主结构:
LiteD1R2 + NWD

最终主数据集:
RS-STOD / USOD / NWPU-VHR10

待补充数据集:
LEVIR-Ship / HRSID

AIMLAux:
不进入主线，只作为负结果或讨论项。
```
