# PRR-v3-SSA 三大隐患修正方案（可执行版）

本方案面向 `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA.yaml`，针对以下三项隐患给出修正后的可执行方案：

1. `H1`：P3 头通道容量瓶颈
2. `H2`：P3 检测头缺少直通 P3 语义
3. `H3`：PPA 聚合索引是否取错层

与上一版方案相比，这一版只保留**当前代码能真正落地**的路线，并补齐两个关键实现闭环：

- `SemanticAlignmenCalibration` 的 YAML 传参闭环
- 新增 fusion 节点的 `Scale-Routed Optimizer` 路由闭环

---

## 1. 先给结论

### 推荐优先级

1. **R2'：P1 + SAC 扩通道**
2. **R1'：v3 原版 + SAC 扩通道 + P3 直通融合**
3. **R3'：纯 YAML POC 版，仅做快速验证**

### 为什么这样排

- `P1` 已有对话内训练结果支撑，当前证据最强，但**尚未沉淀为正式实验卡**。
- `H3` 目前更像 HFAMPAN 设计约定，不像明确 bug，不建议先动。
- `R1` 原思路没错，但如果不同时改 `tasks.py` 和 `optimizer_router.py`，实际不会按预期生效。

---

## 2. 当前代码下必须先修正的两个实现缺口

### 问题 A：SAC 现在无法从 YAML 接收输出通道参数

当前 `tasks.py` 对 `SemanticAlignmenCalibration` 的解析是：

```python
elif m is SemanticAlignmenCalibration:
    c1 = [ch[x] for x in f]
    c2 = c1[0]
    args = [c1]
```

这意味着：

- YAML 里传的第二个参数会被丢弃
- 输出通道始终固定为 `inc[0]`

所以像下面这种写法：

```yaml
- [[38, 39], 1, SemanticAlignmenCalibration, [64]]
```

**当前不会生效**。

### 修正方式

同时改两处：

#### `uav.py`

```python
class SemanticAlignmenCalibration(nn.Module):
    def __init__(self, inc, out_c=None):
        super().__init__()
        hidden_channels = out_c if out_c is not None else inc[0]
```

这里默认值必须是 `inc[0]`，不能改成 `max(inc)`。

原因很直接：

- 当前行为就是 `inc[0]`
- 默认改成 `max(inc)` 会静默改变所有旧 YAML
- 那不叫“向后兼容”

#### `tasks.py`

应改成类似：

```python
elif m is SemanticAlignmenCalibration:
    c1 = [ch[x] for x in f]
    out_c = args[0] if len(args) else None
    c2 = c1[0] if out_c is None else make_divisible(min(out_c, max_channels) * width, 8)
    args = [c1, c2]
```

这里的关键点是：

- `args` 不能再被覆盖成只有 `[c1]`
- `c2` 必须跟实际输出通道一致
- `make_divisible / max_channels / width` 这三个名字直接沿用 `parse_model` 现有作用域，不需要额外 import

另外，这里最好沿用仓库里其它“显式输出通道模块”的解析风格，避免把 `SemanticAlignmenCalibration` 做成一个过于特殊的分支。

---

### 问题 B：新增 `Concat + Conv` fusion 节点不会自动进入 `small_object` 优化组

当前 `optimizer_router.py` 的规则是：

- `TopBasicLayer / MFFF / SAC / RefocusSingle / PyramidPoolAgg` 等被划到 `small_object`
- `Conv / Concat / Upsample` 默认都归到 `backbone`

所以如果按上一版方案直接插入：

```yaml
- [[37, 40], 1, Concat, [1]]
- [-1, 1, Conv, [256, 1, 1]]
```

这两层的参数会被归到 `backbone`，而不是 `small_object`。

这会导致：

- 结构能跑
- 但学习率路由不符合设计预期

### 修正方式

二选一：

#### 方案 B1：推荐

把 fusion 包成单独模块，例如：

- `P3Fusion`
- 内部实现 `Concat + Conv1x1`

这样顶层模块类名可直接加入 `SMALL_OBJECT_MODULE_CLASSES`。

#### 方案 B2：次优

继续用 `Concat + Conv`，但改 `optimizer_router.py`，按顶层索引把这些节点手动归入 `small_object`。

不推荐作为长期方案，因为可维护性差。

---

## 3. H1：P3 通道瓶颈的修正版方案

当前 `SAC` 输出通道由 `inc[0]` 决定，所以现状是：

- P2 head：`32`
- P3 head：`32`
- P4 head：`128`

这就是 `H1` 的根因。

### 方案 1A'【推荐默认】

扩展 `SemanticAlignmenCalibration(inc, out_c=None)`，并在 YAML 显式指定输出通道。

示例：

```yaml
- [[38, 39], 1, SemanticAlignmenCalibration, [256]]
```

在 `n` 尺度下，`256 -> 64ch`。

### 优点

- 直接扩大 `SAC` 内部融合容量
- `P3` 从 `32 -> 64`
- 与标准 FPN 的 `[32, 64, 128]` 更接近

### 代价

- 需要同时改 `uav.py` 和 `tasks.py`
- 参数会增加，但幅度可控

---

### 方案 1B'【仅 POC】

不改 `SAC`，只在 `SAC` 后外挂一个 `Conv1x1` 升通道：

```yaml
- [[38, 39], 1, SemanticAlignmenCalibration, []]
- [-1, 1, Conv, [256, 1, 1]]
- [[36, 41, 33], 1, AsDDet, [nc]]
```

### 结论

- 可以快速验证
- 但它**只改头输入，不改 SAC 内部容量**
- 只能做 POC，不建议做最终主线

---

## 4. H2：P3 直通缺失的修正版方案

当前 `P3` 检测头使用的是：

- `40 = SAC(...)`

并没有直接看到未经过对齐校准的 `MFFF(P3)`。

---

### 方案 2A'【推荐】

给 `P3` 检测头增加一条显式直通：

- 直通项：`37 = MFFF(P3)`
- 精修项：`40 = SAC(...)`

建议不要直接堆裸 `Concat + Conv` 顶层节点，而是新增一个 `P3Fusion` 模块：

```yaml
- [[38, 39], 1, SemanticAlignmenCalibration, [256]]   # 40 -> 64ch
- [[37, 40], 1, P3Fusion, [256]]                      # 41 -> 64ch
- [[36, 41, 33], 1, AsDDet, [nc]]                     # 42
```

其中 `P3Fusion` 内部逻辑可以很简单：

```python
class P3Fusion(nn.Module):
    def __init__(self, inc, out_c):
        super().__init__()
        self.fuse = Conv(sum(inc), out_c, 1, 1)

    def forward(self, x):
        return self.fuse(torch.cat(x, 1))
```

### 优点

- P3 头同时拿到“直通 P3”和“跨尺度校准 P3”
- `optimizer_router` 也更容易正确路由

### 代价

- 需要新增一个轻量模块
- 不再是纯 YAML 改法
- 还必须在 `tasks.py` 中给 `P3Fusion` 注册 parse 分支，并补 import

---

### 方案 2B'【当前最稳】

直接基于已经验证有效的 `P1`：

```yaml
- [[36, 37], 1, SemanticAlignmenCalibration, []]
- [[36, 38, 33], 1, AsDDet, [nc]]
```

再叠加 `1A'` 扩通道。

也就是：

- 先删 `FFDS`
- 再把 `SAC` 输出扩到 `64`

### 结论

这是当前最稳的主线，因为：

- `P1` 已经有正式实验支撑
- 结构更简洁
- 只差把 `SAC` 容量补起来

---

### 方案 2C'【可做补充消融】

保留 `FFDS`，但把 `SAC` 改成同级自校准：

```yaml
- [[37, 39], 1, SemanticAlignmenCalibration, []]
- [[36, 40, 33], 1, AsDDet, [nc]]
```

### 结论

- 逻辑上自洽
- 但证据链不如 `P1`
- 可以作为补充消融，不建议先做主线

---

## 5. H3：PPA 索引的修正版判断

### 结论

当前建议维持现状，不作为主修项。

原因：

- 当前 `[-1, -4, -8, 9]` 落到 reduce 后节点，这与 HFAMPAN 的上游约定一致
- 它更像设计选择，而不是明确 bug

### 仅在什么情况下再动

只有当你后续实验发现：

- 全局语义明显偏弱
- 且 `PPA` 成为主要怀疑点

才建议加一个 A/B 版：

```yaml
- [[21, 17, 13, 9], 1, PyramidPoolAgg, [512, 2, 'torch']]
```

作为纯消融分支，不直接替代主线。

---

## 6. 修正后的组合方案

### R1'【完整增强版】

组成：

- `1A'`：SAC 扩通道
- `2A'`：P3 直通融合
- `3A`：PPA 保持现状

### 特点

- 效果潜力最大
- 但要改：
  - `uav.py`：新增 `P3Fusion`
  - `uav.py`：扩展 `SemanticAlignmenCalibration(inc, out_c=None)`
  - `tasks.py`：修 `SemanticAlignmenCalibration` 的 parse 分支
  - `tasks.py`：新增 `P3Fusion` 的 parse 分支，并补 import
  - `optimizer_router.py`：把 `P3Fusion` 纳入 `SMALL_OBJECT_MODULE_CLASSES`

如果仓库还有额外的模块导出入口，也要同步确认 `P3Fusion` 的导出路径是否可被 `tasks.py` 正常引用。

### 适用

- 你准备把它作为 `v4` 正式主线时

---

### R2'【最推荐】

组成：

- 复用 `P1`
- 再加 `1A'`

也就是：

- `Refocus + MFFF + SAC`
- 删 `FFDS`
- `SAC` 输出从 `32 -> 64`

### 特点

- 结构最干净
- 已有 `P1` 对话日志结果支撑，但仍建议先补实验卡再作为论文证据引用
- 改动最集中
- 风险最低

### 当前建议

这条应作为下一阶段主线。

---

### R3'【快速 POC】

组成：

- `1B'`
- `2A'` 的简化版
- 不改 `SAC` 内部

### 特点

- 适合快速试方向
- 不适合作为最终结论

---

## 7. 建议的实施顺序

### 第一阶段：先做最稳主线

1. 基于 `P1` 新建 `P1-HC64` 版本
2. 修改 `SemanticAlignmenCalibration`
3. 修改 `tasks.py` 让 `out_c` 真正生效
4. 跑 `P1 vs P1-HC64`

### 第二阶段：再加 P3 直通

1. 新增 `P3Fusion`
2. 补 `optimizer_router.py`
3. 跑 `P1-HC64 vs R1'`

### 第三阶段：最后才碰 H3

1. 新建 `PPA-C2f` 版
2. 只做补充消融

---

## 8. 推荐实验矩阵

### 前置条件

在跑 `E2 / E3 / E4` 之前，必须先确认：

1. `uav.py` 与 `tasks.py` 的代码修改已经合入
2. `SemanticAlignmenCalibration` 的 `out_c` 已在建图日志中真正生效

否则：

- `E2` 会退化成和 `E1` 同一结构
- 最终对照会失真

建议在每张实验卡中都显式记录：

- `SemanticAlignmenCalibration` 的实际 `hidden_channels`
- `AsDDet` 三输入通道
- 是否启用了 `P3Fusion`

| 编号 | 配置 | 目的 |
|---|---|---|
| E0 | `v3-SSA` | 原始基线 |
| E1 | `P1` | 验证删 FFDS |
| E2 | `P1 + SAC64` | 验证 H1 是否成立 |
| E3 | `P1 + SAC64 + P3Fusion` | 验证 H2 是否成立 |
| E4 | `P1 + SAC64 + PPA-C2f` | 仅补充验证 H3 |

建议固定：

- 相同 seed
- 相同 batch / imgsz / optimizer 策略
- 至少对比 `mAP50-95`、`Recall`、`Precision`、`FPS`

如果数据侧允许，额外关注：

- `AP_tiny`
- `AP_small`
- 背景图误检

---

## 9. 最终建议

如果只给一句话：

> 当前不建议直接跳到“完整大修版 R1”，而应先走 `R2' = P1 + SAC 扩通道`，把最关键的 P3 容量问题先用最小代价验证清楚；只有这一步成立后，再引入 P3 直通融合。

这条路线最稳，也最符合你现在已经拿到的实验证据。

---

## 10. 证据状态说明

当前 `R2'` 推荐依据中的 `S0 / P1` 结果，已集中整理在：

- `C:\Users\86155\Desktop\水经注\数据\USOD\s0`
- `C:\Users\86155\Desktop\水经注\数据\USOD\p1`

其中包含：

- `slurm_*.out`
- `detect/train/args.yaml`
- `detect/train/results.csv`
- `detect/train/weights/best.pt`

`P2` 当前仍在超算训练中，尚未并入口径统一的结果集。

因此：

- 可以作为工程决策依据
- 但仍建议补一页统一格式的实验卡摘要，便于后续论文引用与横向对照

建议下一步补一张最小实验卡，至少记录：

- 配置文件
- 训练脚本
- 数据集
- 关键指标
- 与 `v3-SSA` 的差值
