# 2026-05-31 改进路线与下个 TeX 线程交接

本文件用于下个线程继续更新 `main.tex`，重点记录当前还可推进、应停止和需要补实验的路线。

## 1. 当前论文主线

主线保持：

```text
LiteD1R2 + NWD
```

不要改成：

```text
LiteD1R2 + NWD + AIMLAux
```

原因：

- `AIMLAux / fd_aux w=0.003` 在 NWPU 三 seed 中不稳定。
- USOD AIMLAux 明显掉点。
- AIMLAux 只能作为 train-only 辅助约束或讨论项。

## 2. 当前可写进论文的核心贡献

### 2.1 结构主线

可以写：

```text
YOLO11-based remote sensing small object detector
with HFAMPAN, AsDDet, NWD, PRR, MFFF, and LiteD1R2.
```

核心模块：

- `C3k2 + SSA/Sobel/SAConv`：增强边缘和上下文。
- `HFAMPAN + PyramidPoolAgg`：多尺度聚合。
- `TopBasicLayer + LAF_h`：P2/P3 空间建模与局部融合。
- `InjectionMultiSum_Auto_pool`：全局信息注入。
- `PRR / RefocusSingle`：P2/P3 重聚焦。
- `MFFF`：P2/P3 频域增强。
- `AsDDet`：检测头。
- `NWD`：小目标定位损失。
- `LiteD1R2`：轻量化主结构。

### 2.2 轻量化贡献

`LiteD1R2` 的论文表述：

```text
TopBasicLayer depth is reduced from 2 to 1 on P2/P3,
and P2/P3 RepBlock repeats are reduced from 3 to 2.
Under the nano scale multiplier, RepBlock depth changes from 2 to 1.
```

强调：

```text
主要收益在 GFLOPs，而不是参数量。
```

### 2.3 消融贡献

可写进消融：

```text
NoTopBasic:
Removing TopBasicLayer + LAF_h reduces GFLOPs but degrades mAP50 and Recall.
This confirms the value of the P2/P3 attention-fusion path.
```

```text
CIoU:
Replacing NWD with CIoU on USOD degrades mAP50-95,
showing NWD is better suited for dense tiny object localization.
```

```text
AIMLAux:
Although inspired by augmentation-invariant manifold learning,
the current fd_aux implementation is unstable and is not included in the final model.
```

## 3. 应停止的方向

### 3.1 USOD imgsz=800

结果：

```text
usod_img800_from85210_b2_e100_88292
mAP50-95 = 0.32198
```

相对主结果：

```text
85210 mAP50-95 = 0.33549
```

裁定：

```text
停止，不跑 300 epoch。
```

### 3.2 USOD AIMLAux

结果：

```text
usod_nwd_aimlaux_w0003_b4_e100_88162
mAP50-95 = 0.30882
```

裁定：

```text
停止，不做 USOD AIMLAux。
```

### 3.3 NWPU AIMLAux w=0.003 全程训练

三 seed：

```text
mean mAP50-95 = 0.53860
std = 0.00599
```

对比：

```text
87416 clean = 0.55702
```

裁定：

```text
停止，不进入主线。
```

### 3.4 NWPU from 88163 + AIMLAux low-lr refit

结果：

```text
nwpu_aimlaux_from88163_lowlr_b6_e300_88290
mAP50-95 = 0.53763
```

裁定：

```text
停止。
```

### 3.5 NoTopBasic 作为主线

裁定：

```text
只作为消融，不作为主线。
```

## 4. 仍可推进的实验

### 4.1 P0: AIMLAux 初始化再 NWD-only 微调

目的：

```text
验证 AIMLAux 是否只适合作为初始化，而不适合作为全程检测 loss。
```

实验：

```text
88163 best.pt
-> LiteD1R2 + NWD
-> fd_aux 关闭
-> lr0=0.0002
-> 100 epoch
```

判据：

```text
硬通过: mAP50-95 >= 0.55702
软通过: mAP50-95 >= 0.552 且 bridge / ship / vehicle 至少两个类别改善
失败: mAP50-95 < 0.552
```

状态：

```text
已给 5.13 sbatch 命令，等待结果。
```

如果失败：

```text
AIMLAux 当前路线整体归档。
```

### 4.2 P1: LEVIR-Ship 1cls-clean 对照

目的：

```text
验证 LiteD1R2 + NWD 在 tiny ship 检测上的泛化。
同时判断 AIMLAux 是否对 ship detection 有帮助。
```

必须先修数据：

```text
datasets/LEVIR-Ship-1cls-clean
```

要求：

- 不用 symlink。
- images 用 hardlink 或真实 copy。
- labels 全部重写为 class 0。
- 删除所有 `.cache`。
- 路径必须固定在 `LEVIR-Ship-1cls-clean`。

实验：

```text
B0: LiteD1R2 + NWD
B1: LiteD1R2 + NWD + AIMLAux
```

判据：

```text
B1 >= B0 + 0.003 才说明 AIMLAux 有效。
```

状态：

```text
已给 5.30 重建数据 + B0/B1 sbatch 命令，等待结果。
```

### 4.3 P1: HRSID polygon-to-detect

HRSID 本地状态：

```text
HRSID.v1i.yolov11 是 polygon segmentation label。
不是 detect 5 列格式。
```

处理方式：

```text
polygon -> bbox detect
输出 datasets/HRSID-DET
```

实验：

```text
先跑 LiteD1R2 + NWD + AIMLAux
后续必须补 LiteD1R2 + NWD baseline
```

注意：

```text
如果只有 AIMLAux，没有 NWD baseline，不能判断 AIMLAux 是否有效。
```

状态：

```text
已给 5.30 解压、转换、验证、训练命令，等待结果。
```

### 4.4 P2: USOD 外部同域预训练

USOD 当前训练侧很难继续提升。

已失败路线：

- low-lr refit
- imgsz=800
- AIMLAux
- CIoU
- mosaic off

可考虑：

```text
SIMD / VEDAI / UAVDT / VisDrone vehicle pretraining
-> USOD fine-tune
-> Soft-NMS sigma=0.3 val-only
```

优先数据：

```text
SIMD
VEDAI
UAVDT
VisDrone
TinyPerson 仅作为 tiny-object 泛化，不作为遥感主线
```

不建议：

```text
WiderPerson
VOC_MASK
SHWD
```

原因：

```text
非遥感或与船/车遥感检测关系弱。
```

## 5. 论文 TeX 更新建议

### 5.1 Abstract

强调：

```text
lightweight remote sensing small object detection
NWD-based localization
multi-scale P2/P3 enhancement
validated on RS-STOD, USOD, NWPU-VHR10
```

不要强调：

```text
AIMLAux
```

除非作为 discussion。

### 5.2 Method

建议结构：

```text
3.1 Overall Architecture
3.2 Enhanced Backbone with SSA/Sobel/SAConv
3.3 HFAMPAN Neck with P2/P3 Injection
3.4 PRR and MFFF for Small Object Feature Refinement
3.5 AsDDet Head and NWD Loss
3.6 LiteD1R2 Lightweight Variant
```

AIMLAux 不放主方法核心，可以放：

```text
3.7 Optional Training-only Auxiliary Constraint
```

或者放到 ablation/discussion。

### 5.3 Experiments

主表：

```text
RS-STOD 85036
USOD 85210
NWPU 87416
```

消融表：

```text
LiteD1R2 vs NoTopBasic
NWD vs CIoU
USOD Soft-NMS
AIMLAux seed stability
USOD imgsz800
```

新增数据集表，如果结果回来：

```text
LEVIR-Ship B0/B1
HRSID B0/B1
```

### 5.4 Discussion

建议写：

```text
Although augmentation-invariant manifold learning motivates feature consistency,
the current train-only fd_aux branch does not provide stable gains across seeds.
Therefore, it is not incorporated into the final detector.
```

这句话能合理解释 AIML 负结果。

## 6. 下个线程优先级

1. 检查 5.13 `nwpu_from88163_nwdonly_lowlr_b6_e100` 结果。
2. 检查 5.30 `LEVIR-Ship-1cls-clean` B0/B1 结果。
3. 检查 5.30 `HRSID-DET` 结果。
4. 若 LEVIR/HRSID 只有 AIMLAux，补 NWD-only baseline。
5. 更新 `main.tex`：
   - 主结果表
   - 消融表
   - AIMLAux discussion
   - 数据集说明

## 7. 最终写作裁定

```text
主线:
LiteD1R2 + NWD

主结果:
RS-STOD 85036
USOD 85210
NWPU 87416

AIMLAux:
当前为负结果/讨论项，不进入最终结构。

新增数据:
LEVIR-Ship / HRSID 等待 clean baseline 对照。
```
