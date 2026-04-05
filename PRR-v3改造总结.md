# PRR-v3 改造总结

## 当前可执行主链路

- 主配置：`YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml`
- 默认训练脚本：`train_yolo111 copy.py`
- 推荐模式：`full`

## 本轮落地的关键修复

1. `AsDDet` 已正式接入 `ultralytics/nn/tasks.py` 的模型解析链，不再回退为普通 `Detect`。
2. `PRR-v3` 检测头输入已修正为真正的三尺度 `P2/P3/P4`，对应节点 `[36, 40, 33]`，stride 为 `(4, 8, 16)`。
3. `SSDS` 已修复面积计算错误，tiny/small 目标现在会正确对 `P2/P3` 做权重强化。
4. 训练入口已统一到 `PRR-v3`，旧根目录 `SmallObject.yaml` 与 `PRR.yaml` 已移除，避免误训到旧链路。
5. B 策略已改为温和累积倍率：默认 medium=1.25x、hard=1.5x，并增加 `max_accum` 上限，避免早期训练直接顶到过大累积步数。
6. 动态 `accumulate` 变化时会同步缩放优化器 `weight_decay`，避免有效 batch size 改变而正则强度停留在旧值。

## 训练模式说明

- `baseline`：仅使用 `PRR-v3 + AsDDet` 结构，不启用 A+B / SSDS。
- `ab`：在 `baseline` 基础上启用 `Scale-Routed Optimizer + Noise-Aware Batch Curriculum`。
- `full`：在 `ab` 基础上再启用 `SSDS`，这是当前推荐模式。`base_accum=0` 表示自动沿用 Ultralytics 的基础累积步数推导。

兼容别名：

- `baseline_prr` 会自动归一化为 `baseline`
- `prr_ab` 会自动归一化为 `ab`
- `prr_ssds` 会自动归一化为 `full`

## 训练策略补充

- `--base_accum 0`：自动沿用 `nbs / batch` 推导出的基础累积步数。
- `--base_accum N`：显式覆盖基础累积步数。
- `--medium_accum_scale` / `--hard_accum_scale`：控制中高难度 batch 的累积倍率。
- `--max_accum`：限制动态累积上限，默认 `24`。

## 推荐命令

### 本地

```bash
python "train_yolo111 copy.py" --trainer_mode full --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml --data data_VD.yaml --batch 4 --enable_ssds --debug_routing
```

### SLURM

```bash
sbatch train_slurm_ab.sh full
```

## 已验证项

- `PRR-v3` 最后一层真实类型为 `AsDDet`
- 三尺度 stride 为 `(4, 8, 16)`
- `SSDS` 在 tiny/small 样例上加权结果为 `[1.5, 1.3, 1.0]`
- Curriculum 在 `base_accum=11` 下会输出 `14/16` 的温和累积步数（含上限约束）
- 动态 `weight_decay` 会随 `accumulate` 从 `8 -> 12` 同步缩放到 `0.015`
- 回归测试：`tests/test_small_object_prr_stack.py`
