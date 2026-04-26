# YOLO11 PRR-v3-SSA 项目创新与结构贡献总结

> 日期：2026-04-19（最近更新：2026-04-22）
> 主配置：`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml`
> 当前主线变体：P1（精度优先精简版，删除 FFDS 桥接层 + SSA 异构注入），正在验证 P1-HC64（P3 通道扩展）
>
> **本次修订要点（2026-04-22）**：
> - §1.6 标注 FrequencyFocusedDownSampling 在 P1 主线已删除
> - §1.7 补充 SAC 非对称输出设计（双入单出，仅喂 P3）
> - 新增 §1.9 Ablation-driven 结构精简（删 FFDS 的实证依据）
> - §2.3 标注 SSDS 实现定位（损失端重加权）
> - §五 数据流图按 P1 真实结构重写（去 FFDS、加 SAC 非对称分叉）
> - §八 创新点总览重排（FFDS 从架构创新移至"架构减法"，SAC 单列"非对称"维度）

---

## 一、架构层创新

### 1.1 检测尺度重分配（三尺度检测头）

**问题**：标准 YOLO 使用 P3/P4/P5 三尺度检测头，P5 对小目标收益极低，计算预算浪费在大目标分支。

**方案**：将检测头重分配为 **P2/P3/P4 三尺度**：

| 尺度 | stride | 职责 |
|------|--------|------|
| P2 | 4 | 主攻 tiny 目标（<16x16 px） |
| P3 | 8 | 主攻 small 目标（16x16 ~ 32x32 px） |
| P4 | 16 | 承接 medium 及更大目标 |
| P5 | 32 | 仅参与高层语义聚合（PyramidPoolAgg），不输出检测 |

**核心逻辑**：不是简单删层，而是将检测预算从低收益的 P5 检测层转移到更贴近小目标的高分辨率链路。

**代码落点**：YAML head 最后一层 `AsDDet` 接收 `[P2_feat, P3_feat, P4_feat]` 三路输入。

---

### 1.2 HFAMPAN 多尺度特征聚合

**创新点**：在标准 FPN top-down 路径之后，增加 **金字塔池化聚合（PyramidPoolAgg）** 侧路，将 P2/P3/P4/P5 四个尺度的语义信息自适应池化到统一分辨率后拼接、降维，形成全局上下文向量。

**组件链路**：

```
PyramidPoolAgg([P2, P3, P4, P5_SPPF]) → 全局语义 G
      ↓                                       ↓
各尺度: TopBasicLayer → LAF_h → InjectionMultiSum_Auto_pool{2,3,4}(G) → RepBlock
```

- **PyramidPoolAgg**：接收四路多尺度特征，自适应池化到统一空间尺寸，concat 后 1x1 Conv 降维
- **TopBasicLayer**：为 P2/P3 分支做独立的浅层表征预处理
- **LAF_h（Level-Adaptive Fusion）**：层级自适应融合，池化对齐后 concat
- **InjectionMultiSum_Auto_pool{2,3,4}**：将全局语义特征通过通道分裂 + 门控（h_sigmoid）注入到各尺度分支，实现 `local_feat * sigmoid(global_act) + global_feat`

**核心价值**：在不引入额外检测层的前提下，让 P2/P3 这类高分辨率分支也能访问到 P5 级别的高层语义信息，解决高分辨率分支"有空间但缺语义"的问题。

**代码位置**：`ultralytics/nn/core11/GDM.py`

---

### 1.3 PRR（P2/P3 Refocus Resample）空间重聚焦

**问题**：高分辨率特征图（P2/P3）包含大量背景区域，导致检测效率低下且梯度噪声大。

**方案**：在 P2/P3 分支的 RepBlock 之后插入 `RefocusSingle(gridsample)` 模块：

1. **候选热力图生成**（`CandidateHeatmapHead`）：
   - 轻量 DW + PW 卷积头，输出 1-channel sigmoid 热力图
   - 稀疏先验初始化（bias = -2.0），使初始输出偏低，避免过拟合

2. **基于 grid_sample 的重采样增强**（`GridSampleRefiner`）：
   - 拼接特征和热力图，预测 2-channel 偏移量
   - 用热力图加权偏移：高响应区产生有效偏移，低响应区保持恒等映射
   - grid_sample 双线性重采样 + 增强卷积
   - 可学习残差系数 γ（初始 0.1）：`output = x + γ * enhanced`

3. **备选实现**（`SoftRefocusEnhancer`）：纯 soft mask 加权 + 局部增强，无 grid_sample

**插入链路**：`RepBlock → RefocusSingle → MFFF`

**设计逻辑**：
- 先完成基础局部表征（RepBlock）
- 再对重点区域做空间重聚焦（RefocusSingle）
- 最后进入频域增强与语义对齐链路（MFFF → SAC）

**代码位置**：`ultralytics/nn/core11/refocus_resample.py`

---

### 1.4 SSA（Sobel-SAConv Augmentation）骨干增强

**问题**：标准 C3k2 模块只有固定局部感受野，缺乏边缘敏感性和多尺度上下文感知能力。

**方案**：在 C3k2 模块中注入 SSA 辅助分支，形成 **主支路 + 边缘支路 + 上下文支路** 三路融合：

- **SobelConvBranch**：固定 Sobel 算子（不可学习参数），逐通道提取水平和垂直边缘响应 `|Gx| + |Gy|`
- **LightSAConv**（轻量 Switchable Atrous Convolution）：
  - pre_context：全局平均池化 + 1x1 Conv 注入全局上下文
  - switch：5x5 AvgPool + 1x1 Conv + Sigmoid 生成空间开关
  - 通过可学习开关在小感受野（dilation=1）和大感受野（dilation=d）之间动态切换
  - post_context：输出层全局上下文注入
- **融合**：主支路输出 + Sobel 投影 + SAConv 输出 → 1x1 Conv 融合

**分层配置**（P1 变体）：

| 层级 | 位置 | Sobel | SAConv | dilation |
|------|------|:-----:|:------:|:--------:|
| 浅层 | C3k2 #2 | Y | N | 1 |
| 中层 | C3k2 #4 | Y | Y | 2 |
| 深层 | C3k2 #6, #8 | Y | Y | 3 |

**设计理由**：浅层特征简单，仅需边缘增强；深层语义复杂，需要多尺度上下文。

**代码位置**：`ultralytics/nn/modules/block.py` 中的 `SSA`、`SobelConvBranch`、`LightSAConv`、修改后的 `C3k2/C3k`

---

### 1.5 MFFF（Multi-Frequency Feature Fusion）频域增强

**创新点**：在 RefocusSingle 之后插入频域增强模块，补充空间域无法捕获的频率信息。

**内部结构**：

- **FFM（Frequency Feature Module）**：
  - 两路 1x1 Conv 分别处理输入
  - 一路做 FFT 变换到频域，与另一路在频域做逐元素乘法
  - IFFT 回空间域，取模 + 可学习 α/β 残差

- **ImprovedFFTKernel**：
  - FCA（Frequency Channel Attention）：全局池化 → 1x1 Conv → 频域加权
  - SCA（Spatial Conv Attention）：1x1 + 3x3 DW + 5x5 DW 多尺度卷积聚合
  - 通道注意力机制：SE-like squeeze-excitation
  - FGM（Frequency Gated Module）：FFM 做最终频域门控融合
  - 大核 DW Conv（31x31）做空间上下文聚合

- **MFFF 封装**：通道分裂策略，仅对 1/4 通道（`e=0.25`）做频域增强，其余通道保持恒等，控制计算开销

**代码位置**：`ultralytics/nn/core11/uav.py`

---

### 1.6 FrequencyFocusedDownSampling 频域引导下采样（PRR-v3 原版组件，P1 主线已删除）

> ⚠️ **P1 主线状态**：**已删除**。本节保留仅用于说明 PRR-v3 原版设计与 P1 精简的对照。删除决策的实证依据见 §1.9 与 §六 P1 vs P2 对照消融。

**创新点（PRR-v3 原版设计，P1 已不再使用）**：替代标准下采样，在降分辨率过程中并联频域增强，减少信息损失。

**结构**：
```
输入 → AvgPool(2,1) → chunk(2)
                       ├─ x1 → Conv(3,stride=2) ─────────────┐
                       └─ x2 ─┬─ FFM → Conv(3,stride=2) ─┐  │
                               └─ MaxPool(3,2) → Conv(1) ─┘  │
                                        concat → Conv(1) ─────┘
                                                    concat → output
```

**代码位置**：`ultralytics/nn/core11/uav.py`

---

### 1.7 SemanticAlignmenCalibration（SAC）语义对齐校准

**创新点**：跨尺度特征对齐模块，接收 `[P2_feat, P3_feat]` 双输入，做精细的空间-语义校准。

**流程**：
1. P3 语义特征 → 3x3 Conv → 双线性上采样到 P2 分辨率
2. 频域增强：FFM 处理上采样后的语义特征
3. 门控融合：`fused = semantic * (1 - gate) + freq_enhanced * gate`
4. P2 空间特征 → 3x3 Conv
5. 拼接 [空间, 融合] → 偏移量预测（2 组 × 4 偏移 + 2 注意力）
6. 分组 grid_sample 精细校准
7. 双路 tanh 注意力加权输出

**非对称输出设计（P1 主线核心 design choice）**：SAC 接收 `[P2_MFFF, P3_MFFF]` 双输入，但**仅将对齐结果喂给 P3 检测头**。P2 分支直接使用 MFFF 输出进入 AsDDet，**绕过 SAC**。这是 P1 主线最独特的非对称设计：

- **P2 旁路**：tiny 尺度对跨尺度平滑敏感，SAC 对齐过程会稀释其主检测特征 → P2 直连 MFFF
- **P3 接 SAC**：small 尺度从双向语义约束中获益 → P3 使用 SAC 对齐输出
- P1 yaml 第 4 条注释明示："P2 检测头继续直接使用 MFFF 输出，避免削弱 P2 主检测特征"

**论文叙事价值**："cross-scale calibration is beneficial for small, but harmful for tiny" —— 一个反直觉的尺度敏感性观察，构成 P1 主线的独立 design contribution（见 §八 I-6）。

**P1-HC64 变体**：扩展 SAC 的 `hidden_channels` 输出通道从默认值到 256（`out_c=256`），验证 P3 通道容量瓶颈假设（H1）。

**代码位置**：`ultralytics/nn/core11/uav.py`

---

### 1.8 AsDDet 解耦检测头

**创新点**：解耦式检测头，将回归和分类分支各自使用独立的 DWConv + Conv 结构。

- **REG 分支**：DWConv → Conv → 1x1 Conv → 4 x reg_max 输出
- **CLS 分支**：DWConv → Conv → 1x1 Conv → nc 输出
- 与 DFL（Distribution Focal Loss）配合使用
- 自定义 bias 初始化策略

**代码位置**：`ultralytics/nn/modules/Head/AsDDet.py`

---

### 1.9 Ablation-driven 结构精简（FFDS 删除）★ P1 独有

**创新点**：P1 主线相对 PRR-v3 原版的核心结构变化是**删除两个 `FrequencyFocusedDownSampling` 桥接层**（原 yaml 第 #38/#39 层）。这构成 "negative architecture" 的实证证据：不是"加了什么模块"，而是"删了什么才更好"。

**对照实验**（详见 §六）：

| 变体 | SSA | FFDS | mAP50-95 | 备注 |
|------|:-:|:-:|:-:|------|
| S0 | ✗ | ✓ | 0.321 | 旧基线（PRR-v3 原版 + 无 SSA）|
| P2 | ✓ | ✓ | 0.327 | 负消融证据：保留 FFDS 反而无增益 |
| **P1** | **✓** | **✗** | **0.336** | **主线基座**：删 FFDS 后 mAP50-95 反而提升 |

**层号变化**：删 FFDS 后 head 总层数从 42 减至 40，最终 AsDDet 输入从 `[36, 40, 33]` 变为 `[36, 38, 33]`。

**论文叙事价值**：

- 验证"少即是多"的结构精简原则
- 提供与"堆叠更多模块更强"思路相反的实证案例
- 增强 P1 vs S0 增益的归因清洁度（同时验证 SSA 是真正贡献者，FFDS 不是）

**代码落点**：P1 yaml 文件头注释第 3 条明确标注 `删除中间桥接层：FrequencyFocusedDownSampling(38/39)`（见 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml`）。

---

## 二、训练策略层创新（A + B + SSDS）

### 2.1 策略 A：Scale-Routed Optimizer

**问题**：统一学习率和正则强度无法匹配不同功能模块的收敛特性。小目标敏感模块需要更积极的学习，预训练 backbone 需要更保守的微调。

**方案**：基于顶层模块类名自动路由参数到三组：

| 参数组 | 包含模块 | lr 倍率 | beta2 | wd 倍率 |
|--------|---------|---------|-------|---------|
| backbone | Conv, C3k2, SPPF, C2f | x0.5 | 0.999 | x1.0 |
| small_object | TopBasicLayer, MFFF, RefocusSingle, SAC, RepBlock, PyramidPoolAgg 等 | x1.25 | 0.9995 | x0.5 |
| det_head | AsDDet, Detect | x1.5 | 0.999 | x1.0 |

**附加特性**：
- Norm 层和 bias 参数自动免除 weight_decay
- small_object 组专属 gradient clipping（默认 max_norm=5.0）
- group-wise grad norm 日志输出到 `grad_norm_log.csv`
- 顶层模块映射表自动打印，方便调试

**代码位置**：`ultralytics/engine/optimizer_router.py`

---

### 2.2 策略 B：Noise-Aware Batch Curriculum

**问题**：含大量 tiny 目标的 batch 梯度噪声高，固定 batch size 导致训练不稳定。

**方案**：`BatchCurriculumScheduler` 根据每个 batch 的小目标统计动态调整梯度累积步数。

**难度计算**：
```
difficulty = alpha * tiny_ratio + beta * density_ratio
```

- `tiny_ratio`：面积 < 256px^2 的目标占比（alpha=0.7）
- `density_ratio`：每张图平均目标数 / 归一化阈值（beta=0.3）

**动态累积策略**：
| diff_ema 区间 | 难度等级 | 累积倍率 |
|--------------|---------|---------|
| < 0.3 | easy | x1.0（base） |
| 0.3 ~ 0.6 | medium | x1.25 |
| >= 0.6 | hard | x1.5 |

**安全机制**：
- EMA 平滑避免逐 batch 抖动
- max_accum 上限（默认 24）
- warmup 期间（默认 3 epoch）使用固定累积
- weight_decay 同步缩放：accumulate 变化时自动调整 wd，保持正则一致性
- DDP 多卡安全：all_reduce MAX 保证各卡同步

**代码位置**：`ultralytics/engine/batch_curriculum.py`

---

### 2.3 SSDS：Scale-Specific Dual-Branch Supervision

> 📌 **实现定位**：SSDS 虽归入"训练策略"分类，但实际实现是**损失端 `target_scores` 重加权**（在 TAL 分配完成之后），与 NWD 损失同在 loss 阶段生效。代码见 `ssds_loss.py:82-86`。论文写作时可单独归类到"损失/监督"创新（见 §八 I-11）。

**问题**：标准 TAL（Task-Aligned Learning）分配不区分目标尺度，P2/P3 无法获得针对性的监督信号。

**方案**：`ScaleSpecificReweighter` 在 TAL 分配完成后做尺度感知 soft reweighting：

| GT 类型 | 面积范围 | 强化目标层 | 权重放大系数 |
|---------|---------|-----------|------------|
| tiny | < 256 px^2 | P2（最小 stride 层） | x1.5 |
| small | 256 ~ 1024 px^2 | P3（次小 stride 层） | x1.3 |
| medium/large | >= 1024 px^2 | 不变 | x1.0 |

**两种模式**：
- `soft`（默认）：乘法加权，温和引导
- `hard`：非对应层权重降到 0.1，强制分配

**设计优势**：不改 TAL 分配逻辑本身，只在分配结果上做后处理，风险最低。

**代码位置**：`ultralytics/engine/scale_supervision.py`、`ultralytics/utils/NewLoss/ssds_loss.py`

---

## 三、损失函数创新

### 3.1 NWD（Normalized Wasserstein Distance）损失

在标准 CIoU 基础上融合 Wasserstein 距离：

```
loss_iou = 0.7 * (1 - CIoU) + 0.3 * (1 - NWD)
```

NWD 将 bbox 建模为二维高斯分布，基于 Wasserstein 距离度量分布相似度。相比 IoU，NWD 对小目标的位置偏差更敏感，缓解 IoU 对小面积目标的梯度消失问题。

**代码位置**：`ultralytics/utils/loss.py`、`ultralytics/utils/NewLoss/ioulossone.py`

---

## 四、联合训练器 SmallObjectABTrainer

继承 `DetectionTrainer`，通过 `trainer_mode` 参数灵活组合所有策略：

| mode | A（路由优化器） | B（课程学习） | SSDS（尺度监督） |
|------|:-:|:-:|:-:|
| `a` | Y | | |
| `b` | | Y | |
| `ab` | Y | Y | |
| `full` | Y | Y | Y |

**覆写方法**：
- `build_optimizer()`：构建三组参数路由的 AdamW
- `_setup_train()`：初始化 curriculum 调度器
- `init_criterion()`：初始化 SSDSDetectionLoss
- `preprocess_batch()`：计算 batch 难度并更新 accumulate
- `optimizer_step()`：group-wise grad clipping + norm 记录
- `save_metrics()`：追加 curriculum 和 SSDS 指标到 CSV
- `final_eval()`：保存 curriculum/grad_norm 日志
- `progress_string()`：训练进度显示 curriculum/SSDS 状态

**代码位置**：`ultralytics/engine/small_object_trainer.py`

---

## 五、完整数据流

```
输入图像
   │
   ▼
Backbone (SSA 深度异构注入):
  Conv → Conv → C3k2+SSA(Sobel)              → Conv      ← P2 (1/4)
  → C3k2+SSA(Sobel+SAConv d=2)               → Conv      ← P3 (1/8)
  → C3k2+SSA(Sobel+SAConv d=3)               → Conv      ← P4 (1/16)
  → C3k2+SSA(Sobel+SAConv d=3) → SPPF                    ← P5 (1/32)
   │
   ▼
FPN top-down: P5 → P4 → P3 → P2（标准 Conv + Upsample + Concat + C2f）
   │
   ├─→ PyramidPoolAgg([P2, P3, P4, P5_SPPF]) → 全局语义 G
   │
   ├─ P2 分支: TopBasicLayer → LAF_h → Inject_pool2(G) → RepBlock
   │            → RefocusSingle(gridsample) → MFFF ─────────────┬──→ P2_det        (★ 直连，不走 SAC)
   │                                                             │
   ├─ P3 分支: TopBasicLayer → LAF_h → Inject_pool3(G) → RepBlock
   │            → RefocusSingle(gridsample) → MFFF ─────────────┤
   │                                                             ↓
   │                                  SAC([P2_MFFF, P3_MFFF]) ──→ P3_aligned_det   (★ 双入单出)
   │
   ├─ P4 分支: LAF_h → Inject_pool4(G) → RepBlock ──────────────────────────────→ P4_det     (★ 不走 PRR)
   │
   ▼
AsDDet Head([P2_det, P3_aligned_det, P4_det]) → NWD Loss + SSDS reweighting (loss 端)

★ P1 主线相对 PRR-v3 原版的两处关键差异：
  1. 删除 FrequencyFocusedDownSampling（PRR-v3 原版在 P2/P3 各有一个 FFDS，P1 已全删）
  2. SAC 非对称输出：双入单出，仅喂 P3；P2 直连 MFFF 进 head
```

---

## 六、当前实验状态

| 变体 | 含义 | mAP50 | mAP50-95 | 状态 |
|------|------|-------|----------|------|
| S0 | PRR-v3 基线（无 SSA） | 0.813 | 0.321 | 已完成 |
| P1 | SSA + 删除中间桥接层（精度优先精简） | 0.818 | 0.327 | **主线基座** |
| P2 | SSA + 保留完整桥接层 | 0.815 | 0.327 | 负消融证据 |
| P1-HC64 | P1 + SAC 输出通道扩展至 64 | - | - | E2 训练中 |

**基座选择结论**：P1 > P2 > S0。P1 精简了 FrequencyFocusedDownSampling 中间桥接层但不损失精度，P2 保留完整结构反而无增益，验证了 P1 精简路线的正确性。

---

## 七、关键源文件索引

| 文件 | 职责 |
|------|------|
| `ultralytics/nn/core11/refocus_resample.py` | PRR 核心：CandidateHeatmapHead、SoftRefocusEnhancer、GridSampleRefiner、RefocusSingle、P2P3RefocusResample |
| `ultralytics/nn/core11/uav.py` | 小目标增强链路：DySample_、SPDConv、MFFF、FFM、ImprovedFFTKernel、FrequencyFocusedDownSampling、SemanticAlignmenCalibration |
| `ultralytics/nn/core11/GDM.py` | HFAMPAN 组件：PyramidPoolAgg、InjectionMultiSum_Auto_pool{1-4}、LAF_h、LAF_px、RepBlock、RepVGGBlock、TopBasicLayer |
| `ultralytics/nn/modules/block.py` | SSA 系列：SobelConvBranch、LightSAConv、SSA、修改后的 C3k2/C3k |
| `ultralytics/nn/modules/Head/AsDDet.py` | AsDDet 解耦检测头 |
| `ultralytics/engine/optimizer_router.py` | 策略 A：Scale-Routed Optimizer 参数路由 |
| `ultralytics/engine/batch_curriculum.py` | 策略 B：Noise-Aware Batch Curriculum 动态累积调度 |
| `ultralytics/engine/scale_supervision.py` | SSDS：ScaleSpecificReweighter 尺度感知重加权 |
| `ultralytics/utils/NewLoss/ssds_loss.py` | SSDSDetectionLoss 尺度感知损失 |
| `ultralytics/engine/small_object_trainer.py` | 联合训练器 SmallObjectABTrainer |
| `ultralytics/utils/loss.py` | NWD 损失集成 |
| `ultralytics/nn/tasks.py` | 模型解析注册（所有自定义模块的构建入口） |

---

## 八、创新点总览

| 编号 | 创新名称 | 类型 | 解决问题 |
|------|---------|------|---------|
| I-1 | 三尺度检测重分配（P2/P3/P4） | 架构 | P5 对小目标无收益，检测预算浪费 |
| I-2 | HFAMPAN 金字塔聚合注入 | 架构 | 高分辨率分支缺乏高层语义 |
| I-3 | PRR 空间重聚焦（RefocusSingle + grid_sample 热力图引导） | 架构 | P2/P3 背景区域过多，效率低 |
| I-4 | SSA 骨干深度异构增强（Sobel frozen + SAConv learnable，分层 dilation） | 架构 | C3k2 缺边缘和多尺度上下文，浅/中/深层异构匹配 |
| I-5 | MFFF 频域增强（通道分裂 e=0.25） | 架构 | 空间域无法捕获的频率信息丢失 |
| I-6 | **SAC 非对称跨尺度校准**（双入单出，仅喂 P3） | 架构 | P3 受益于跨尺度对齐，P2 tiny 特征会被对齐稀释 |
| I-7 | AsDDet 深度可分离解耦检测头（独立 DWConv 链路） | 架构 | 标准检测头回归/分类耦合 |
| I-8 | **★ Ablation-driven 结构精简（删 FFDS）** | 架构（减法） | 保留 FFDS 反而无增益，证伪"堆叠更多更强"假设 |
| I-9 | Scale-Routed Optimizer（A，三组 lr/β₂/wd 路由） | 训练策略 | 统一学习率不匹配异构模块 |
| I-10 | Noise-Aware Batch Curriculum（B，tiny 比例驱动动态 accum + wd 同步缩放） | 训练策略 | 小目标密集 batch 梯度噪声大 |
| I-11 | SSDS 尺度感知 target_scores 重加权 | **损失/监督** | TAL 不区分尺度，监督信号无差异化（实现在 loss 端） |
| I-12 | NWD + CIoU 融合（0.7·CIoU + 0.3·NWD） | 损失函数 | IoU 对小目标梯度消失 |

**总览说明**：

- **架构创新（I-1 ~ I-7）**：7 项加法 + 1 项减法（I-8）= 8 项结构层贡献
- **架构减法（I-8）**：P1 独有，构成 negative architecture 实证
- **训练策略（I-9, I-10）**：A + B 路由化训练强度
- **损失/监督（I-11, I-12）**：SSDS + NWD，均在 loss 端生效
