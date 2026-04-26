# YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3 分析报告

## 1. 模型定位

`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml` 是当前项目的小目标检测主配置。

这条链路不是简单叠加若干模块，而是围绕“小目标优先”的目标，对结构、优化器、batch curriculum 和监督方式同时做了重分配。当前默认训练入口是 `train_yolo111 copy.py`，推荐模式是 `trainer_mode=full`。

当前主线可以概括为三层：

1. 结构主线：`PRR-v3`
2. 优化主线：`A+B`
3. 监督主线：`SSDS`

其中：

- `PRR-v3` 负责重新分配检测尺度与特征增强路径。
- `A+B` 负责让不同模块、不同 batch 难度得到不同训练强度。
- `SSDS` 负责把 tiny/small 目标更明确地推向 P2/P3。

## 2. 当前主配置的核心思路

### 2.1 尺度职责重分配

当前 `PRR-v3` 的核心决策不是“继续加层”，而是先重分配检测资源。

- `P2 (stride=4)`：主攻 tiny 目标
- `P3 (stride=8)`：主攻 small 目标
- `P4 (stride=16)`：承接 medium 及更大目标
- `P5`：只保留语义参与聚合，不再单独做检测

这意味着当前模型不是四尺度检测，而是三尺度检测。最终检测头输入来自 `[36, 40, 33]`，即 `P2 / P3 / P4`。

### 2.2 为什么去掉 P5 检测层

项目当前判断是：对于以小目标为核心的任务，P5 检测支路的边际收益低于其参数和算力消耗。

因此 `PRR-v3` 做了两件事：

- 保留 backbone 的 P5 深层语义，用于 SPPF 与 HFAMPAN 聚合。
- 删除 P5 独立检测职责，把预算回收到更接近小目标的高分辨率分支。

这样做的收益不是“完全不要深层语义”，而是“深层语义继续参与融合，但不再占用独立检测头预算”。

## 3. PRR-v3 的结构组成

## 3.1 Backbone 与 FPN 主干

主干仍然沿用 YOLO11 风格的 backbone + top-down FPN，只是在用途上做了重新约束：

- Backbone 负责提取多尺度基础语义
- FPN 负责把高层语义逐步传回到高分辨率层
- HFAMPAN 负责多尺度聚合与分支注入

`PRR-v3` 不是推翻 backbone，而是在 head 及 small-object 分支上重构重点。

## 3.2 HFAMPAN 聚合层

HFAMPAN 聚合仍然显式接收四路特征：

- 当前 P2 路
- 当前 P3 路
- 当前 P4 路
- layer 9 的 P5 深层语义

也就是说，P5 在当前版本里没有被删除，只是从“检测层”降级为“语义聚合输入”。

## 3.3 P2/P3/P4 三个输出分支

三条分支的职责明显不同：

- P2 分支是 tiny 目标主战场，保留了更深的高分辨率处理。
- P3 分支承担 small 目标，是 P2 与 P4 的中间桥梁。
- P4 分支不再追求极深堆叠，而是作为 higher-level detection 分支保留必要表征。

当前设计的重点是：高分辨率分支宁可复杂一些，也不把预算继续压到已经被证明对 tiny 目标不敏感的 P5 检测头上。

## 3.4 PRR：RefocusSingle with grid_sample

当前 `PRR-v3` 已不再使用旧版 `softmask` 作为默认实现，而是采用：

- `RefocusSingle(gridsample)`

即通过显式可微采样，对局部区域做重聚焦。

这一步的目标不是新增一个泛化很强的大模块，而是完成一个明确动作：

- 先从 P2/P3 上定位更值得关注的局部区域
- 再通过局部重采样强化这些区域的特征表达

插入位置也很明确：

- `RepBlock -> RefocusSingle -> MFFF`

这意味着 PRR 是在基础局部表征已经形成之后，再去做空间重聚焦，而不是直接对原始特征做粗暴采样。

## 3.5 小目标增强链路

当前 `PRR-v3` 的小目标增强链路是连续的，而不是单模块试验：

1. `RefocusSingle`
2. `MFFF`
3. `FrequencyFocusedDownSampling`
4. `SemanticAlignmenCalibration`

各模块职责如下：

- `RefocusSingle`：显式空间重聚焦
- `MFFF`：多频率特征融合
- `FrequencyFocusedDownSampling`：对下采样过程做频率敏感处理
- `SemanticAlignmenCalibration`：统一 P2/P3 的语义对齐

当前版本里，这条链路是 `PRR-v3` 的关键结构资产，不属于可以随意删掉的装饰模块。

## 3.6 AsDDet 检测头

最终检测头使用的是 `AsDDet`，不是普通 `Detect`。

当前工程里，这一层已经接入模型解析链，因此最后一层的真实类型就是 `AsDDet`。这点非常关键，因为很多文档错误都来自“YAML 写了 AsDDet，但训练时被回退成普通 Detect”。

当前 `PRR-v3` 已修复这个问题。

## 4. 训练入口与模式解释

当前默认训练入口：

- `train_yolo111 copy.py`

它会根据 `trainer_mode` 决定是否启用结构外的训练策略。

### 4.1 baseline

- 只使用 `PRR-v3 + AsDDet` 结构
- 不启用 A
- 不启用 B
- 不启用 SSDS

这个模式的用途是：验证纯结构增益。

### 4.2 ab

- 启用 `Scale-Routed Optimizer`
- 启用 `Noise-Aware Batch Curriculum`
- 不启用 SSDS

这个模式的用途是：验证训练策略增益。

### 4.3 full

- 启用 A
- 启用 B
- 启用 SSDS

这是当前项目的默认推荐模式，也是当前文档主线默认对应的训练链路。

## 5. A 策略：Scale-Routed Optimizer

`A` 的目的不是单纯改优化器种类，而是改“不同模块应该用同一套学习率吗”这个前提。

当前实现里，模型参数会被分成三组：

- `backbone`
- `small_object`
- `det_head`

其逻辑是：

- backbone 学习率更保守，避免破坏已有基础特征
- small_object 模块学习率更积极，以便快速学习小目标增强链路
- det_head 学习率最高，使任务适配更快

这对应当前代码中的默认缩放关系：

- `backbone_lr_scale = 0.5`
- `smallobj_lr_scale = 1.25`
- `head_lr_scale = 1.5`

同时，小目标组还会：

- 使用更高的 `beta2`
- 使用更小的 `weight_decay`
- 在 `optimizer_step` 里做专属梯度裁剪

这说明 A 不是单一超参，而是一整套“按模块路由训练强度”的策略。

## 6. B 策略：Noise-Aware Batch Curriculum

`B` 的目的不是修改样本本身，而是让“难 batch”获得更大的有效 batch size。

当前实现中，训练器会基于每个 batch 的：

- tiny 目标占比
- 目标密度

估计难度，并动态调整 `accumulate`。

关键点有三个：

1. `base_accum` 可显式指定，也可沿用 Ultralytics 的自动推导结果。
2. 难度不是瞬时值，而是 `EMA` 平滑后的 `diff_ema`。
3. `accumulate` 变化时，`weight_decay` 也会同步缩放，避免“有效 batch 变了但正则强度没变”的隐性不一致。

当前默认逻辑相对温和：

- `medium_accum_scale = 1.25`
- `hard_accum_scale = 1.5`
- `max_accum = 24`

这比旧方案更稳，目的就是避免一开始就把训练推到过大的累积步数。

## 7. SSDS：尺度感知监督补强

`SSDS` 的目标是把 tiny/small 目标更明确地压向最适合的检测层。

当前实现不是重写分配器，而是在现有分配结果上做 soft reweighting。

默认阈值：

- `tiny_area_thr = 256`，约等于 `16x16`
- `small_area_thr = 1024`，约等于 `32x32`

默认放大：

- tiny 在 P2 上放大 `1.5`
- small 在 P3 上放大 `1.3`

当前推荐模式是 `soft`，因为它保留原分配结构，只做乘法加权，更稳。

## 8. 为什么当前主线是 PRR-v3 + A+B + SSDS

这三者分工不同：

- `PRR-v3` 解决“结构资源如何分配”
- `A+B` 解决“训练强度如何分配”
- `SSDS` 解决“监督信号如何分配”

如果只看其中一层，容易误以为模型提升只来自某个模块。但当前工程主线实际追求的是：

- 结构层把 tiny/small 目标送到更合适的特征分支
- 优化层让这些分支学得更快、更稳
- 监督层进一步强化 tiny/small 的目标归属

这就是当前主线选择 `full` 的原因。

## 9. 当前推荐训练方式

本地或服务器直接训练时，当前推荐链路是：

```bash
python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml \
  --weights yolo11n.pt \
  --data <your_data_yaml> \
  --device 0 \
  --batch 4 \
  --epochs 300 \
  --imgsz 640 \
  --workers 8 \
  --trainer_mode full \
  --enable_ssds \
  --debug_routing
```

如果是继续训练，也可以把 `--weights` 替换成已有最佳权重路径。

## 10. 当前版本应如何理解

当前 `PRR-v3` 不是“最终论文版定稿”，但它已经是项目里的主训练配置，且具有明确边界：

- 默认主模型：`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml`
- 默认训练入口：`train_yolo111 copy.py`
- 默认推荐模式：`full`

如果后续再做轻量化、消融或结构变体，应该以 `PRR-v3` 为主基线展开，而不是反过来让变体代替主线。

## 11. 文档使用建议

这份文档适合作为：

- 项目内部结构说明
- 实验报告中的方法解释底稿
- 后续论文 Method 部分的技术分解参考

若用于论文正文，建议进一步提炼成：

- 一页结构图说明
- 一页训练策略说明
- 一页消融逻辑说明

当前这份文档的目标是“和代码一致”，优先保证可追溯和可复核。
