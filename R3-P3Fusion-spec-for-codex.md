# R3 · P3Fusion 实施规范（H2 验证专用）

> 发布日期：2026-04-20
> 前置规范：`R2-implementation-spec-for-codex.md` Section 13（H1 已证伪，R2 规范已终结）
> 面向对象：codex 线程
> 文档归属：ultralyticsPro--YOLO11 线程

本规范面向 `PRR-v3-SSA-hidden-risks-fix-corrected.md` 中 **H2（P3 检测头缺少直通 P3 语义）** 的实施。H1 已在 USOD 上被 P1-HC64 证伪，R2' 路线废止；本规范仅承载 **P3Fusion（H2）** 的独立验证，不附带任何 SAC 通道扩展。

---

## Section 1 · 背景与目标

### 1.1 H1 证伪后的路线位置

- S0 → P1 → P2（负向消融）→ P1-HC64（H1 证伪）→ **P1-P3Fusion（本规范）**
- 本变体与 P1 的**唯一差异**是在 SAC 输出端并联一条直通 P3 的融合节点，保持 SAC hidden 仍为 `inc[0]=32`，不重复 HC64 的失败实验。

### 1.2 验证目标

- 假设：P3 检测头同时接收"直通 P3 语义（MFFF 输出）"和"跨尺度校准结果（SAC 输出）"后，能带来 mAP50-95 增益。
- 预期：
  - 正向（H2 成立）：mAP50-95 ≥ P1（0.336）+ 0.5pp，且参数 / GFLOPs 增量 ≤ 5%
  - 持平：回头考虑退出 H2，转入 Phase C 跨数据集验证
  - 负向：H2 证伪，P1 为最终主线

### 1.3 设计原则（强约束）

1. **不动 SAC 通道**（规避 HC64 过拟合模式）
2. **不删 P1 既有节点**（不改建桥层和 SSA 骨干）
3. **只新增一个最小 fusion 节点**
4. **参数归 `small_object` 组**（避免错误路由到 backbone）

---

## Section 2 · 源码改动清单（4 处）

### 2.1 新增 `P3Fusion` 类（写入 `ultralytics/nn/core11/uav.py`）

**插入位置**：文件末尾，`SemanticAlignmenCalibration` 之后即可。

**类实现**：

```python
class P3Fusion(nn.Module):
    """
    P3 检测头双路融合模块（H2 验证专用）。

    输入：
        x: list of 2 tensors, e.g. [mfff_p3_feat, sac_p3_feat]
           两路通道可以不同，但空间分辨率必须一致。
    输出：
        fused: 单 tensor，通道数 = out_c

    设计：最简 Concat + 1x1 Conv 降维；不引入注意力 / 频域 / 重采样，
    纯粹验证"P3 检测头是否需要直通 P3 语义"这一命题。
    """

    def __init__(self, inc, out_c):
        super().__init__()
        if not isinstance(inc, (list, tuple)):
            inc = [inc]
        self.fuse = Conv(sum(inc), out_c, 1, 1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        return self.fuse(torch.cat(x, 1))
```

**依赖**：`Conv` 已在 `uav.py` 顶部导入（文件内使用），`torch` 同理。不需要新增 import。

**验证点**：新类可被 `from ultralytics.nn.core11.uav import P3Fusion` 正确导入。

---

### 2.2 修改 `ultralytics/nn/tasks.py`

#### 2.2.1 Import 扩展（L74 附近）

**当前**：
```python
from ultralytics.nn.core11.uav import DySample_,SPDConv,MFFF,FrequencyFocusedDownSampling,SemanticAlignmenCalibration, BottleNeck, BasicBlock_
```

**改为**：
```python
from ultralytics.nn.core11.uav import DySample_,SPDConv,MFFF,FrequencyFocusedDownSampling,SemanticAlignmenCalibration, BottleNeck, BasicBlock_, P3Fusion
```

（仅在末尾追加 `, P3Fusion`）

#### 2.2.2 parse 分支扩展（L1484–L1488 之后插入）

**当前 SAC 分支**（L1484–L1488）保持不变：
```python
elif m is SemanticAlignmenCalibration:
    c1 = [ch[x] for x in f]
    out_c = args[0] if args else None
    c2 = c1[0] if out_c is None else make_divisible(min(out_c, max_channels) * width, 8)
    args = [c1, c2]
```

**紧接其后插入**：
```python
elif m is P3Fusion:
    c1 = [ch[x] for x in f]
    out_c_raw = args[0] if len(args) else max(c1)
    c2 = make_divisible(min(out_c_raw, max_channels) * width, 8)
    args = [c1, c2]
```

**关键点**：
- `out_c_raw` 没传时默认为 `max(c1)`（即两路输入通道的较大者，确保不降采样）
- `make_divisible(min(out_c_raw, max_channels) * width, 8)` 与仓库内其它模块风格一致，在 `n` 尺度（width=0.25）下 `256 → 64`、`128 → 32`
- 不修改 `args`、`f` 的类型约束（`f` 必须是 list of int，由 YAML `[[i, j], ...]` 保证）

#### 2.2.3 验证点

- `python -c "from ultralytics.nn.tasks import parse_model; print('ok')"` 无 ImportError
- 静态 `py_compile` 通过

---

### 2.3 修改 `ultralytics/engine/optimizer_router.py`

**当前**（L10–L27）：
```python
SMALL_OBJECT_MODULE_CLASSES = {
    "TopBasicLayer",
    ...
    "GridSampleRefiner",
}
```

**改为**（在集合末尾追加一行）：
```python
SMALL_OBJECT_MODULE_CLASSES = {
    "TopBasicLayer",
    ...
    "GridSampleRefiner",
    "P3Fusion",
}
```

**作用**：确保 `model.39 P3Fusion(...)` 的所有参数（1x1 Conv 的 weight + BN + bias）被路由到 `small_object` 优化组，走 `lr_scale=1.25, wd_scale=0.5`。

---

### 2.4 新建 YAML：`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml`

**位置**：仓库根目录（与 P1.yaml、P1-HC64.yaml 同级）

**内容**（仅展示 head 尾部差异区域，前面与 P1.yaml 完全相同）：

```yaml
# ... 与 P1.yaml 的 L1–L75 完全一致 ...

  # ===== 小目标增强 =====
  - [34, 1, MFFF, [128, 0.25]]                         # 36
  - [35, 1, MFFF, [256, 0.25]]                         # 37

  # ===== 语义对齐校准（P1 原值：直接吃 MFFF 输出，不扩通道） =====
  - [[36, 37], 1, SemanticAlignmenCalibration, []]     # 38  out_c=inc[0]=32

  # ===== H2: P3 直通融合（MFFF(P3) 与 SAC 双路融合） =====
  - [[37, 38], 1, P3Fusion, [256]]                     # 39  n尺度下 -> 64ch

  # ===== 检测头 - 三尺度 [P2, P3_fused, P4] =====
  - [[36, 39, 33], 1, AsDDet, [nc]]                    # 40
```

**关键结构事实**（给 codex 自查）：

| 层号 | 模块 | from | 输出通道（n 尺度下） |
|---|---|---|---|
| 36 | MFFF | 34 | 32 |
| 37 | MFFF | 35 | 64 |
| 38 | SemanticAlignmenCalibration | [36, 37] | **32**（保持 P1 原值，不扩） |
| **39** | **P3Fusion** | **[37, 38]** | **64**（`make_divisible(min(256, 1024) * 0.25, 8) = 64`） |
| 40 | AsDDet | [36, 39, 33] | — |

**AsDDet 最终输入通道**：`[32, 64, 128]`
（注意：P3 通道从 P1 的 32 升到 64，但这次是由 P3Fusion 带来的，**不是** SAC 内部扩张）

**顶层配置**：完全继承 P1.yaml 的 `nc / scales / loss / backbone / FPN top-down / HFAMPAN`，不做任何其他修改。

---

## Section 3 · Smoke Test（E3 提交前强制）

### 3.1 命令（超算 CPU 节点 ln01，1 epoch, batch=1, CPU）

```bash
python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml \
  --weights yolo11n.pt \
  --data data_USOD_server2.yaml \
  --device cpu \
  --batch 1 \
  --epochs 1 \
  --imgsz 640 \
  --workers 0 \
  --trainer_mode baseline
```

### 3.2 首屏 parse_model 必须命中

```
 36   [34]      1  ...  ultralytics.nn.core11.uav.MFFF                              [32]
 37   [35]      1  ...  ultralytics.nn.core11.uav.MFFF                              [64]
 38   [36, 37]  1  ...  ultralytics.nn.core11.uav.SemanticAlignmenCalibration       [[32, 64], 32]
 39   [37, 38]  1  ...  ultralytics.nn.core11.uav.P3Fusion                          [[64, 32], 64]
 40   [36, 39, 33] 1 ... ultralytics.nn.modules.Head.AsDDet.AsDDet                  [1, [32, 64, 128]]
```

**关键验证点**：
- `model.38 SAC[[32, 64], 32]`：SAC 仍走 P1 原值 32（不是 HC64 的 64）
- `model.39 P3Fusion[[64, 32], 64]`：P3Fusion 建图通，输出 64 通道
- `model.40 AsDDet [1, [32, 64, 128]]`：P3 检测头通道由 P3Fusion 带到 64

**不允许出现**：
- `P3Fusion` 出现在 backbone 参数组统计里
- `SAC[[...], 64]`（说明误走 HC64 路线）
- 任何 shape mismatch 报错

### 3.3 参数组分布核验（smoke 训练日志必输出）

期望输出结构：
```
[Scale-Routed Optimizer] 顶层模块分类完成:
  backbone     : 参数数 ≈ <P1 原值>
  small_object : 参数数 ≈ <P1 原值> + <P3Fusion 增量 ≈ 6-10K>
  det_head     : 参数数 ≈ <P1 原值>（AsDDet 输入通道变，参数小幅增加）
```

如果 backbone 增量超过 1K，说明 `P3Fusion` 未被 `SMALL_OBJECT_MODULE_CLASSES` 捕获，必须回查 Section 2.3。

### 3.4 模型规模参考

- 参数量：与 P1（~5,845K）比，预期 +10K~+20K（P3Fusion + AsDDet 输入通道变）
- GFLOPs：与 P1（36.0）基本一致（P3Fusion 的 1x1 Conv 在 80x80 特征图上 FLOPs 可忽略）

### 3.5 Loss 验证

- 第一个 batch 正常产生 loss（`box_loss`, `cls_loss`, `dfl_loss` 均为有限正数）
- Curriculum / SSDS 相关字段正常初始化，不报错

---

## Section 4 · E3 正式训练

### 4.1 Slurm 脚本（新建）

**路径**：`train_usod_prrv3_ssa_p1_p3fusion_slurm.sh`（仓库根目录）

**关键字段**（基于 `train_usod_prrv3_ssa_p1_hc64_slurm.sh` 改）：

```bash
#!/bin/bash
#SBATCH --job-name=usod-p1-p3fusion
#SBATCH --partition=<与 P1 同池>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out

# ... 环境激活部分与现有脚本一致 ...

python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml \
  --weights yolo11n.pt \
  --data data_USOD_server2.yaml \
  --device 0 \
  --batch 4 \
  --epochs 300 \
  --imgsz 640 \
  --workers 8 \
  --trainer_mode full \
  --name usod-p1-p3fusion-${SLURM_JOB_ID} \
  --project runs/usod
```

### 4.2 强制约束（对比 P1-HC64 吸取的教训）

1. **`name=usod-p1-p3fusion-${SLURM_JOB_ID}`**：避免继续污染 `runs/detect/train/results.csv`
2. **`project=runs/usod`**：数据集级隔离
3. **超参与 P1 / P1-HC64 完全一致**：seed=0、batch=4、epochs=300、optimizer=auto、patience=100、trainer_mode=full
4. **不改任何训练策略**：A+B+SSDS 全开，与 P1 口径一致

### 4.3 判定 E3 是否"H2 成立"的标准

| 场景 | mAP50-95 vs P1 (0.336) | 判定 | 后续动作 |
|---|---|---|---|
| 显著正向 | ≥ +1.0pp（≥ 0.346） | H2 成立 | 写入论文主线结构，继续跨数据集泛化 |
| 边际正向 | +0.3 ~ +1.0pp（0.339–0.346） | H2 弱成立 | 进入 Phase C 确认（至少 2 个数据集持正） |
| 持平 | ±0.3pp（0.333–0.339） | H2 不能下定论 | 宣告 H2 无显著收益，锁 P1 为主线 |
| **负向** | **< −0.3pp（< 0.333）** | **H2 证伪** | 锁 P1 为主线；P3Fusion 作为负向消融证据 |

**注意**：无论判定结果，**都不再尝试 P3Fusion + HC 组合**（会复现 HC64 的失败模式）。

---

## Section 5 · 训完回写要求

codex 训练完成后向 ultralyticsPro--YOLO11 线程回报以下字段（用于沉淀 `exp-YYYYMMDD-usod-p1-p3fusion.md`）：

1. **指标口径**：
   - slurm `.out` 末行的 val 汇总（`all 600 10584 ...`）→ 这是 best.pt 指标，**论文直接引用**
   - `results.csv` 切分后的末行（含 epoch 号与切分起始行号）
   - 训练实际 epoch 数（是否触发 early stop）
   - peak mAP50-95 所在 epoch（诊断过拟合用）

2. **吞吐**：`Speed: X ms preprocess, X ms inference, X ms postprocess / image` + `FPS: X.X`

3. **模型规模**：
   - `nc=80` 初始口径：`layers / parameters / GFLOPs`
   - `nc=1` 最终口径：`layers / parameters / GFLOPs`
   - best.pt fuse 后口径

4. **Scale-Routed Optimizer 三组参数量**（日志首屏应有输出）：
   - `backbone / small_object / det_head` 参数数
   - 相对 P1 的变化

5. **路径归档**：
   - `slurm_<job_id>.out` 路径
   - `runs/usod/usod-p1-p3fusion-<job_id>/detect/train/` 下：
     - `args.yaml`
     - `results.csv`
     - `weights/best.pt`, `weights/last.pt`
     - `backup.tar.gz`（若有）
     - `curriculum_log.csv`, `grad_norm_log.csv`

6. **结构事实复查**（训练开始前/后截图或日志）：
   - AsDDet 输入通道是否仍为 `[32, 64, 128]`
   - P3Fusion 参数是否在 `small_object` 组（小于 1% 的偏差允许）

---

## Section 6 · 风险与回退

### 6.1 Smoke 失败的回退路径

| 症状 | 可能原因 | 回退动作 |
|---|---|---|
| `ImportError: P3Fusion` | Section 2.2.1 未追加 | 补 import |
| `TypeError: P3Fusion.__init__() takes 2 positional arguments but 3 were given` | YAML 参数格式不对 | 检查 YAML L39 是否 `P3Fusion, [256]` |
| `RuntimeError: Sizes of tensors must match` | 两路输入空间分辨率不一致 | 检查 YAML `from=[37, 38]` 的空间尺寸（都应为 80x80） |
| P3Fusion 出现在 `backbone` 组 | Section 2.3 未追加 | 补 `"P3Fusion"` 到集合 |
| AsDDet 输入通道不是 `[32, 64, 128]` | parse 分支有问题 | 回查 Section 2.2.2 的 `c2 = make_divisible(...)` 是否生效 |

### 6.2 E3 失败的回退路径（按判定结果）

- **H2 证伪**：
  - 本规范不启动 P3Fusion + HC64 组合（避免复现 HC64 过拟合）
  - 转入 Phase C，锁 P1 为主线
  - 本实验卡定位为"H2 负向消融"
- **H2 持平**：
  - 判定为 H2 无显著收益
  - Phase C 以 P1 为主线跨数据集补证
  - 本实验卡作为结构消融记录保留
- **H2 成立**：
  - 更新 `YOLO11-PRR-v3-SSA-创新总结.md`，将 P3Fusion 加入创新矩阵
  - Phase C 以 P1-P3Fusion 为主线跨数据集补证
  - 考虑补 R1' 完整版（P3Fusion + SAC 扩通道）—— **但注意 HC 已被证伪，组合版审慎**

### 6.3 纯 YAML POC 的替代路径（如 P3Fusion 类实现受阻）

若 codex 因环境原因无法 import/编译 `P3Fusion`，可临时降级为纯 YAML 版（非推荐，仅作阻塞时应急）：

```yaml
  - [[36, 37], 1, SemanticAlignmenCalibration, []]     # 38
  - [[37, 38], 1, Concat, [1]]                         # 39  直接拼接
  - [-1, 1, Conv, [256, 1, 1]]                         # 40  1x1 降维
  - [[36, 40, 33], 1, AsDDet, [nc]]                    # 41
```

**阻塞代价**：`Concat + Conv` 会被错误归入 `backbone` 组，优化路由偏差；**仅用于 smoke / 快速验证**，不作为 E3 正式训练。

---

## Section 7 · 规范生效范围

- 本规范仅覆盖 **H2 P3Fusion 的 USOD 单数据集验证**。
- 规范不涉及：
  - H3 PPA-C2f 消融（优先级最低，暂不动）
  - AITOD / VEDAI 跨数据集实验（Phase C 单独启动）
  - RS-STOD 清洁重跑 P1（Phase C 并行，codex 直接提交，不走本规范）
- 规范生命周期：E3 训练完成 + 实验卡沉淀后即终止；后续 H2 相关的跨数据集实验在各自实验卡中记录，不回写本规范。

---

## Section 8 · Cross Reference

- 前置规范：`R2-implementation-spec-for-codex.md` Section 13（H1 证伪与 R2 规范终结）
- 原命题文档：`PRR-v3-SSA-hidden-risks-fix-corrected.md`（第 4 章 H2 方案 2A'）
- 前置实验卡：
  - `C:\Users\86155\Desktop\协同\memory\experiments\exp-20260416-usod-p1.md`（基座）
  - `C:\Users\86155\Desktop\协同\memory\experiments\exp-20260419-usod-p2.md`（负向消融 A）
  - `C:\Users\86155\Desktop\协同\memory\experiments\exp-20260420-usod-p1-hc64.md`（负向消融 B，H1 证伪）
- Handoff：`C:\Users\86155\Desktop\协同\memory\handoffs\ultralyticsPro--YOLO11.md`（2026-04-20 章节）
- 决策日志：`C:\Users\86155\Desktop\协同\memory\decisions.md`（2026-04-20 条目，同日追加）
- 计划文件：`C:\Users\86155\.windsurf\plans\p1-hc64-postmortem-next-10df2f.md`
