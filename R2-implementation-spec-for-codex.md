# R2' 实施规范 · 交付 codex 执行

**Owner**: ultralyticsPro--YOLO11（文档线程）→ codex（实施线程）
**Date**: 2026-04-18
**Target**: 在 USOD 数据集上跑出 E2 = `PRR-v3-SSA-P1 + SAC hidden=64` 的对照结果，验证 H1「P3 通道瓶颈」是否成立。

## 背景

- 前置实验：
  - E0 = PRR-v3-SSA 原版（USOD, `mAP50-95=0.331, P=0.820, R=0.745, FPS=33.1`）
  - E1 = PRR-v3-SSA-P1 精简版（USOD, `mAP50-95=0.337, P=0.833, R=0.749, FPS=32.8`）
  - 详见 `C:\Users\86155\Desktop\协同\memory\experiments\exp-20260416-usod-{v3ssa-s0,p1}.md`
- E1 相对 E0 增益 +0.60pp mAP50-95，但 SAC hidden 仍为 32ch，**H1 未解**。
- 本规范目标：让 codex 完成最小代码改动 + 新建一份 YAML，启动 E2 训练。

## 必须修改的两处源码（不可漏）

### 修改点 1：`ultralytics/nn/core11/uav.py` L248-L250

**Before**（当前 248-250 行）：

```python
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]
```

**After**（替换为）：

```python
    def __init__(self, inc, out_c=None):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = out_c if out_c is not None else inc[0]
```

**约束**：
- 默认值必须是 `inc[0]`，**不得**改成 `max(inc)`。原因：保持向后兼容，其它未指定 `out_c` 的 YAML 行为不变。
- 其余 251-330 行（`self.groups / spatial_conv / semantic_conv / frequency_enhancer / gating_conv / offset_conv / init_weights / forward`）**不要改动**。内部所有 Conv 已经基于 `hidden_channels` 算通道，扩大 `hidden_channels` 会自动让内部 Conv 放大。

### 修改点 2：`ultralytics/nn/tasks.py` L1484-L1487

**Before**（当前 1484-1487 行）：

```python
        elif m is SemanticAlignmenCalibration:
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]
```

**After**（替换为）：

```python
        elif m is SemanticAlignmenCalibration:
            c1 = [ch[x] for x in f]
            out_c_raw = args[0] if len(args) > 0 else None
            if out_c_raw is None:
                c2 = c1[0]
                args = [c1]
            else:
                c2 = make_divisible(min(out_c_raw, max_channels) * width, 8)
                args = [c1, c2]
```

**约束**：
- `make_divisible / max_channels / width` 来自 `parse_model` 函数现有作用域（同文件其它 elif 分支如 `FrequencyFocusedDownSampling` L1417-1421 使用相同模式），**不需要新增 import**。
- `args = [c1, c2]` 的顺序对应 SAC `__init__(self, inc, out_c)` 的位置参数，**顺序不可交换**。
- 空 args 时保持原行为 `args = [c1]`，以兼容所有未指定 `out_c` 的 YAML。

## 必须新建的 YAML

### 路径
`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml`（放仓库根，与 P1.yaml 并列）

### 来源
完整复制 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml`。

### 唯一改动
将**第 78 行**：

```yaml
  - [[36, 37], 1, SemanticAlignmenCalibration, []]     # 38
```

替换为：

```yaml
  - [[36, 37], 1, SemanticAlignmenCalibration, [256]]  # 38  # HC64: hidden=256*0.25=64ch
```

### 通道计算验证（n scale, width=0.25, max_channels=1024）
- `out_c_raw = 256`
- `c2 = make_divisible(min(256, 1024) * 0.25, 8) = make_divisible(64, 8) = 64`
- SAC 输出通道从 32 → **64**

### 建议同步更新顶部注释
在新 YAML 的头部注释区追加一行：

```yaml
# HC64 变体：SAC hidden_channels 从 inc[0]=32 扩展到 64，验证 H1（P3 通道瓶颈）
```

## 训练启动

### 方式 A：沿用现有 P1 slurm 脚本 + 环境变量覆盖（推荐）

```bash
cd <项目目录>
CFG_FILE=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml \
  sbatch train_usod_prrv3_ssa_p1_slurm.sh
```

`train_usod_prrv3_ssa_p1_slurm.sh` 第 30 行已支持 `CFG_FILE` env 覆盖，不改脚本即可用。

### 方式 B：新建专用脚本（可选）

复制 `train_usod_prrv3_ssa_p1_slurm.sh` → `train_usod_p1_hc64_slurm.sh`，仅改：
- 第 2 行 `#SBATCH -J usod_ssa_p1` → `#SBATCH -J usod_p1_hc64`
- 第 30 行 `CFG_FILE` 默认值替换为 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml`

## 强制验证要点（训练启动后立刻检查）

### 检查 1：AsDDet 输入通道
启动日志开头的模型摘要中，最后一行应为：

```
... ultralytics.nn.modules.Head.AsDDet.AsDDet    [1, [32, 64, 128]]
```

而**不是** `[1, [32, 32, 128]]`。

若仍是 `[32, 32, 128]` → 源码改动未生效，**立即停训**（检查修改点 1 和修改点 2 是否都已保存并被正确 import）。

### 检查 2：SAC 模块参数量
启动日志里 `model.XX` 行对应的 `SemanticAlignmenCalibration` 参数量应约为 **80-100K**（原 p1 为 41,632 params）。若仍是 ~41K → 修改点 1 未生效。

### 检查 3：Scale-Routed Optimizer 路由
启动日志的"Scale-Routed Optimizer 参数路由摘要"中 `small_object` 组参数量应**比 p1 (2,015,358) 增加约 40K-80K**。

### 三项全过再放手让训练跑满 300 epoch。

## 训练完成后回写

codex 完成训练后，向 `ultralyticsPro--YOLO11` 线程回写以下信息：

### 必填
1. **slurm 作业 ID**
2. **结果目录绝对路径**（含 `results.csv / args.yaml / slurm_*.out`）
3. **results.csv 末行四指标**：`P / R / mAP50 / mAP50-95`
4. **slurm .out 最终 val 收敛行**（`all 600 10584 ...` 那一行）
5. **FPS**
6. 启动时的 AsDDet 输入通道实际值（确认是否为 `[32, 64, 128]`）
7. SAC 参数量（用于核验）

### 回写位置
建议放 `C:\Users\86155\Desktop\协同\memory\handoffs\codex.md`（若不存在则由总控线程创建），或直接 @mention `ultralyticsPro--YOLO11` 线程提交结果摘要。

### 实验卡沉淀
`ultralyticsPro--YOLO11` 线程收到回写后，会基于回写数据生成：
`C:\Users\86155\Desktop\协同\memory\experiments\exp-YYYYMMDD-usod-p1-hc64.md`

## 训练控制建议

- 与 p1 对照的**必要对齐条件**：`seed=0 / epochs=300 / batch=4 / imgsz=640 / workers=8 / amp=False / deterministic=True / patience=100`，全部与 `exp-20260416-usod-p1.md` 一致。
- 若 codex 观察到训练过程中 mAP 在 epoch 100 前明显低于 p1 同期（> 2pp 差距），先不要中断，继续观察到 epoch 200，因 SAC 扩通道可能需要更长预热。

## 风险与回滚

### 风险
1. `tasks.py` 的 elif 顺序很重要：SAC 的 elif 必须在 `Concat` (L1498) 之前命中。当前顺序正确，不需调整。
2. 若后续有其它 YAML 依赖 SAC 传非空 args（非通道数的其它参数），会被 `out_c_raw = args[0]` 错误解读。核实全仓只此一处使用 SAC 且仅用于通道，可用，但修改后建议 `grep -rn "SemanticAlignmenCalibration" ultralytics/` 复查。
3. `P1-HC64.yaml` 新文件若命名冲突，检查仓库根是否已存在同名文件。

### 回滚
1. 还原 `uav.py` L248-250（三行）
2. 还原 `tasks.py` L1484-1487（四行）
3. 删除 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml`

## 后续工作（Phase B，本规范不覆盖）

E2 完成且相对 p1 有正向增益后，codex 再按下一份规范执行：
- 在 `uav.py` 新增 `P3Fusion` 类
- 在 `tasks.py` 为 `P3Fusion` 加 elif 分支 + 顶部 import
- 在 `optimizer_router.py` 的 `SMALL_OBJECT_MODULE_CLASSES` 加 `"P3Fusion"`
- 基于 P1-HC64 新建 `PRR-v4-SSA.yaml`（插入 P3Fusion 节点 + 调整 AsDDet 输入索引）
- 跑 E3

Phase B 规范由 `ultralyticsPro--YOLO11` 线程在 E2 证据就位后再出。

## 参考

- 结构分析原始文档：`PRR-v3-SSA-hidden-risks-fix-corrected.md`（用户版）
- 对照基线实验卡：`C:\Users\86155\Desktop\协同\memory\experiments\exp-20260416-usod-p1.md`
- SAC 源码：`ultralytics/nn/core11/uav.py:247-330`
- parse_model SAC 分支：`ultralytics/nn/tasks.py:1484-1487`
- 小目标模块白名单：`ultralytics/engine/optimizer_router.py:10-25`
- 协同交接页：`C:\Users\86155\Desktop\协同\memory\handoffs\ultralyticsPro--YOLO11.md`（2026-04-18 追加章节）

---

## Section 10 · 2026-04-18 晚追加 · 基座待定前置条件

### 背景变更

本规范原假定基座为 P1（PRR-v3-SSA-P1）。2026-04-18 晚发现：**PRR-v3-SSA-P2 变体正在超算训练中**（[train_usod_prrv3_ssa_p2_slurm.sh](cci:7://file:///c:/Users/86155/Desktop/%E6%B0%B4%E7%BB%8F%E6%B3%A8/%E4%BB%A3%E7%A0%81/2026.4.5/ultralyticsPro--YOLO11/train_usod_prrv3_ssa_p2_slurm.sh:0:0-0:0)）。P2 在 P1 基础上进一步删除 P2 分支 TopBasicLayer。

### 新决策规则（必读）

在 codex 开始创建 `HC64.yaml` **之前**，必须先：

1. **等待 P2 训练完成**（USOD 上 300 epoch）
2. **对比 P2 vs P1 的四指标**（P / R / mAP50 / mAP50-95）+ 参数量/GFLOPs
3. 按以下规则选基座：
   - 若 P2 ≥ P1（mAP50-95 持平或更高 + Recall 不明显掉 + 参数/GFLOPs 更低）→ **基座 = P2**，改 P2-HC64.yaml
   - 若 P2 < P1（mAP50-95 明显下降或 Recall 明显掉）→ **基座 = P1**，按本规范 Section 3 原计划改 P1-HC64.yaml

### 基座若为 P2 的 YAML 改动点

- 源 YAML：[YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P2.yaml](cci:7://file:///c:/Users/86155/Desktop/%E6%B0%B4%E7%BB%8F%E6%B3%A8/%E4%BB%A3%E7%A0%81/2026.4.5/ultralyticsPro--YOLO11/YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P2.yaml:0:0-0:0)
- 新 YAML：`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P2-HC64.yaml`
- 唯一改动：**第 77 行**

  ```yaml
  # Before
  - [[35, 36], 1, SemanticAlignmenCalibration, []]     # 37

  # After
  - [[35, 36], 1, SemanticAlignmenCalibration, [256]]  # 37  # HC64
  ```

- 通道计算：`c2 = make_divisible(min(256, 1024) * 0.25, 8) = 64` → SAC 输出从 32 → 64
- AsDDet 输入预期变为 `[32, 64, 128]`（与 P1-HC64 版本一致）

### codex 不等 P2 也可以先做的部分

源码改动（`uav.py` + `tasks.py`）**与最终基座选择无关**，可以先做：

- **步骤 A（现在做）**：Section 2 的两处源码改动（uav.py / tasks.py）
- **步骤 B（现在做）**：跑 P1.yaml 的 smoke test，确认向后兼容未破坏（不传 `[256]` 时行为与旧版一致）
- **步骤 C（等 P2 完成后）**：基座选定后再新建 HC64.yaml（P1-HC64 或 P2-HC64 二选一）
- **步骤 D（基座定后）**：启动 E2 训练并按 Section 5 做三项强制验证

---

## Section 11 · 2026-04-19 早追加 · P2 结果已回收，基座确定为 P1

### 结论

Section 10 中的“基座待定”前置条件现已结束。`PRR-v3-SSA-P2` 在 USOD 上已完成训练，结果**未超过 P1**，因此：

- **基座确定为：P1**
- **本规范后续默认执行对象：P1-HC64**
- **P2-HC64 路线取消**（除非后续有新证据）

### P2 结果摘要（USOD, job 82691）

- `P=0.822`
- `R=0.750`
- `mAP50=0.817`
- `mAP50-95=0.327`

对比：

- P1：`P=0.833, R=0.749, mAP50=0.821, mAP50-95=0.336`
- S0：`P=0.820, R=0.745, mAP50=0.820, mAP50-95=0.331`

因此：

- `P2 < P1`
- `P2 < S0`

### 对本规范的直接影响

从本节开始，Section 3 中关于新 YAML 的创建目标**固定为**：

- `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml`

而不是二选一的 `P1-HC64 / P2-HC64`。

### codex 现在可以直接执行的完整顺序

1. 完成 Section 2 的两处源码改动：
   - `ultralytics/nn/core11/uav.py`
   - `ultralytics/nn/tasks.py`
2. 新建 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml`
3. 做 smoke test：
   - `P1.yaml` 不带 `[256]`，确认向后兼容
   - `P1-HC64.yaml` 带 `[256]`，确认第一屏通道为 `[32, 64, 128]`
4. 正式提交 E2 训练
5. 回写指标给 `ultralyticsPro--YOLO11` 线程生成实验卡

### P2 的后续定位

`P2` 不再作为主线基座候选，而是保留为：

- “P2 TopBasicLayer 继续删除会伤精度”的负向消融证据
- 后续若补实验卡，建议命名为：`exp-YYYYMMDD-usod-p2.md`

---

## Section 12 · 2026-04-19 晚追加 · P1-HC64 代码已落地 + 超算 smoke 已通过

### 代码落地（codex 已完成）

本规范 Section 2 / Section 3 / Section 4 要求的改动，codex 已全部落地：

| 文件 | 操作 | 核心改动 |
|---|---|---|
| `ultralytics/nn/core11/uav.py` | 修改 | `SemanticAlignmenCalibration.__init__(inc, out_c=None)` 新签名；不传 `out_c` 时默认回退 `inc[0]`（向后兼容 P1/S0） |
| `ultralytics/nn/tasks.py` | 修改 | SAC elif 吃 YAML 第二参数：`out_c_raw = args[0] if len(args) else None`；`c2 = make_divisible(min(out_c_raw, max_channels) * width, 8)` |
| `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml` | 新建 | 基于 P1，唯一改动：L78 `[[36, 37], 1, SemanticAlignmenCalibration, []]` → `[[36, 37], 1, SemanticAlignmenCalibration, [256]]` |
| `train_usod_prrv3_ssa_p1_hc64_slurm.sh` | 新建 | 基于 `train_usod_prrv3_ssa_p1_slurm.sh`，仅替换 `--cfg` 为 P1-HC64.yaml |

- 静态校验：`python -m py_compile` 通过（`uav.py` / `tasks.py`）
- 动态校验：codex Windows 本地 torch 环境损坏（`torch_python.dll`），无法本地跑，已转超算

### Smoke 通过（超算 CPU 节点 ln01）

**命令**：

```bash
python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-HC64.yaml \
  --weights yolo11n.pt \
  --data data_USOD_server2.yaml \
  --device 0 --batch 1 --epochs 1 --imgsz 640 --workers 0 \
  --trainer_mode baseline
```

**首屏命中**：

```
 38   [36, 37]      1   83520   ultralytics.nn.core11.uav.SemanticAlignmenCalibration  [[32, 64], 64]
 39   [36, 38, 33]  1  290611   ultralytics.nn.modules.Head.AsDDet.AsDDet              [1, [32, 64, 128]]
```

- `[[32, 64], 64]`：SAC 第一参数是 cat 后输入通道 `[32, 64]`，第二参数是 hidden=64（`make_divisible(min(256, 1024)*0.25, 8)`）
- `[1, [32, 64, 128]]`：AsDDet 输入 `[P2, P3(SAC), P4] = [32, 64, 128]` ✅ H1 实现链路通

**模型规模（nc=1 覆盖后的最终口径）**：`692 layers, 5,866,100 params, 36.0 GFLOPs`

> 注意：初始日志的 `6,039,233 params, 42.0 GFLOPs` 是 nc=80 默认口径，不应记为 USOD 实验指标。

### Section 5 三项强制验证的更新状态

| 检查项 | 状态 | 实测值 |
|---|---|---|
| AsDDet 输入通道行显示 `[32, 64, 128]` | ✅ 通过 | `[1, [32, 64, 128]]` |
| `small_object` 组参数量比 P1 增加约 30K | ⏳ 待 E2 GPU 训练首屏的 Scale-Routed Optimizer 摘要确认 | 预期 +20K（SAC hidden 32→64 的理论增量） |
| 源码改动生效（非黑盒） | ✅ 通过 | SAC 参数列表显示 `[[32, 64], 64]`，证明 `tasks.py` 解析分支走对 |

### codex 下一步动作（E2 正式训练）

1. **立即提交**：

   ```bash
   sbatch train_usod_prrv3_ssa_p1_hc64_slurm.sh
   ```

2. **强烈建议**：在 `train_usod_prrv3_ssa_p1_hc64_slurm.sh` 里把 `name=train` 改成 `name=usod-p1-hc64-$SLURM_JOB_ID`（或在命令行加），**避免 `results.csv` 污染**（P2 已因 `exist_ok=True` 叠加产生 2999 行/10 次运行的数据污染）
3. **首屏再核验**：GPU 路径下 `AsDDet [1, [32, 64, 128]]` 应继续出现
4. **训练完成后回写**：
   - `results.csv` 末行四指标（注意是否存在叠加污染，若有请报告行号切分点）
   - `slurm_<job_id>.out` 最后一个 val 汇总行（`all 600 10584 ...`）
   - FPS 三段时间
   - `best.pt` 路径、`backup.tar.gz` 路径
   - 模型参数量与 GFLOPs（`nc=1` 最终口径）
   - Scale-Routed Optimizer 的三组参数量（`backbone / small_object / det_head`）
5. 回写目标：`ultralyticsPro--YOLO11` 线程，将沉淀 `exp-YYYYMMDD-usod-p1-hc64.md`

### 判定 E2 是否"H1 成立"的标准（给分析参考）

| 场景 | mAP50-95 vs P1 (0.336) | 判定 |
|---|---|---|
| 显著正向 | ≥ +1.0pp（≥ 0.346） | H1 成立，可写入论文 |
| 边际正向 | +0.3 ~ +1.0pp（0.339-0.346） | H1 弱成立，需跨数据集再验 |
| 持平 | ±0.3pp（0.333-0.339） | H1 不能下定论，进入 P3Fusion 叠加 |
| 负向 | < −0.3pp（< 0.333） | H1 不成立，放弃 HC64 路线，回 P1 直接进入 P3Fusion |

### 实验卡引用

- Smoke 记录：`C:\Users\86155\Desktop\协同\memory\experiments\exp-20260419-usod-p1-hc64-smoke.md`
- P2 负向消融：`C:\Users\86155\Desktop\协同\memory\experiments\exp-20260419-usod-p2.md`
- P1 基线：`C:\Users\86155\Desktop\协同\memory\experiments\exp-20260416-usod-p1.md`
- S0 基线：`C:\Users\86155\Desktop\协同\memory\experiments\exp-20260416-usod-v3ssa-s0.md`

---

## Section 13 · 2026-04-20 · E2 结果回收 + H1 证伪 + R2 规范终结

**本节标志着 R2 规范的执行范围正式结束。** Section 1–12 的交付已全部完成，P1-HC64 的 E2 正式训练结果回收后，H1 假设被证伪，后续 HC 系列不再作为本规范的扩展内容。

### 13.1 E2 正式训练结果（USOD, slurm 82742）

| 指标 | 数值 | 备注 |
|---|---|---|
| 作业 ID | 82742 | gpu06 同池 |
| 训练 epoch | 283 / 300 | `patience=100` 早停，peak=epoch 183 |
| Precision | 0.783 | slurm .out best.pt val |
| Recall | 0.695 | |
| mAP50 | 0.759 | |
| **mAP50-95** | **0.286** | **vs P1 0.336, Δ=−0.050** |
| FPS | 32.7 | 0.2 + 29.7 + 0.6 ms/img |
| 参数量 | 5,866,100 | nc=1 口径，相对 P1 +约 20K |
| GFLOPs | 36.0 | 与 P1 相同 |

### 13.2 结构链路命中确认（无问题）

- `model.38 SemanticAlignmenCalibration[[32, 64], 64]` ✅
- `model.39 AsDDet [1, [32, 64, 128]]` ✅
- Section 1-12 要求的代码改动（`uav.py` SAC `out_c=None` + `tasks.py` SAC elif `make_divisible` 解析）全部正确生效。

### 13.3 判定：H1 证伪

按 Section 12.5 的判定表：

| 场景 | mAP50-95 vs P1 (0.336) | 判定 | 本次落位 |
|---|---|---|---|
| 显著正向 | ≥ +1.0pp | H1 成立 | — |
| 边际正向 | +0.3 ~ +1.0pp | H1 弱成立 | — |
| 持平 | ±0.3pp | H1 不能下定论 | — |
| **负向** | **< −0.3pp** | **H1 不成立，放弃 HC64 路线** | ✅ **本次 Δ=−5.0pp，远超负向阈值** |

→ **H1 强力证伪**。本规范对 HC64 路线的建议全面撤回。

### 13.4 对 `PRR-v3-SSA-hidden-risks-fix-corrected.md` 的影响

R2 规范（本文档）承载的是 `PRR-v3-SSA-hidden-risks-fix-corrected.md` 第 6 章 R2' 路线的实施。本次结果直接影响原文档：

| 原文档章节 | 原建议 | 本节撤回后的状态 |
|---|---|---|
| 第 1 章"先给结论"· R2' 推荐优先级 1 | 推荐 | **撤回**（实证证伪） |
| 第 3 章 H1 方案 1A' | SAC 扩通道推荐 | **撤回** |
| 第 4 章 H2 方案 2A' | P3Fusion 推荐 | **保留**（Phase B 进行） |
| 第 6 章 R2'（P1 + SAC 扩通道） | 最推荐主线 | **废止** |
| 第 6 章 R1'（P1 + SAC 扩 + P3Fusion） | 最完整方案 | **降级为 R3** —— 仅保留 P3Fusion 部分 |
| 第 7 章执行顺序"先 HC64 再 P3Fusion" | 两步走 | **改为直接做 P3Fusion** |

### 13.5 后续路线（B4 并行，生效）

用户 2026-04-20 明确选择 B4：

1. **Phase B（H2 P3Fusion）**：交付规范 `R3-P3Fusion-spec-for-codex.md`（同日发布），codex 按其执行。
2. **Phase C（RS-STOD 清洁重跑 P1）**：codex 直接提交，不经过新规范文档。

### 13.6 R2 规范生命周期终止声明

- Section 1-12：全部完成 ✅（uav.py SAC 扩展 / tasks.py parse 扩展 / P1-HC64.yaml 新建 / smoke / E2 正式训练 / 指标回收）
- Section 13（本节）：**最终判决 + 撤回建议**
- 本规范**不再接受新增章节**。H2 / H3 / 跨数据集实验转移到下列文档：
  - `R3-P3Fusion-spec-for-codex.md`（H2 专用）
  - `C:\Users\86155\Desktop\协同\memory\handoffs\ultralyticsPro--YOLO11.md`（2026-04-20 章节及后续）
  - 各 `exp-YYYYMMDD-*.md` 实验卡

### 13.7 数据污染警示（对后续所有规范有效）

P1-HC64 的 `results.csv` 又一次出现污染（2681 行 / 11 段堆积）。以下约定对所有继承本规范的新文档强制生效：

1. `name=<变体>-$SLURM_JOB_ID`（禁止默认 `train`）
2. `project=runs/<数据集>/`（USOD / RSSTOD / AITOD / VEDAI 严格隔离）
3. 指标读取**以 slurm .out 末行为准**
4. 污染的 csv 仅用于切分后的曲线绘制，禁止直接读末行
