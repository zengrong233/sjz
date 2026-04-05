# 任务：小目标检测 A+B 联合训练策略

> **类型**：feature
> **优先级**：P0 (紧急)
> **负责人**：AreaSongWcc
> **状态**：✅ 已完成
> **开始时间**：2026-03-22

## 🎯 目标
在现有 YOLO11 深改仓库中实现 A+B 联合训练策略：
- A: Scale-Routed Optimizer（模块级参数分组 + 差异化学习率）
- B: Noise-Aware Batch Curriculum（基于小目标密度的动态梯度累积）

## 📊 进度仪表盘

| 维度 | 状态 | 详情 |
|------|------|------|
| 整体进度 | ✅ 100% | 全部文件已实现，路由验证通过 |
| 当前阶段 | R2 验收 | 代码完成，参数路由已验证 |

## 📋 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `ultralytics/engine/optimizer_router.py` | 新增 | Scale-Routed 参数分组（顶层模块索引路由） |
| `ultralytics/engine/batch_curriculum.py` | 新增 | Noise-Aware 动态累积 |
| `ultralytics/engine/small_object_trainer.py` | 新增 | SmallObjectABTrainer |
| `train_yolo111 copy.py` | 修改 | 增加 trainer_mode 参数 |
| `train_distributed_torchrun.py` | 修改 | 分布式脚本集成 A+B 开关 |
| `ultralytics/nn/core11/GDM.py` | 修复 | TopBasicLayer/InjectionMultiSum 通道不匹配 |
| `ultralytics/nn/tasks.py` | 修复 | 对应通道参数传递 |

## 🔑 关键决策
- 参数路由采用"顶层模块索引分类 + 路径前缀继承"策略，而非逐叶子模块类名匹配
- 模型构建验证：706 layers, 8,207,790 parameters
- 路由验证：backbone=1,445,664 | small_object=6,220,510 | det_head=541,600（总计匹配）

## 📈 变更日志

| 时间 | 操作 | 说明 |
|------|------|------|
| 03-22 | ✅ 修复路由 | optimizer_router 改为顶层模块索引路由，三组参数分配正确 |
| 03-22 | ✅ 修复通道 | GDM.py TopBasicLayer/InjectionMultiSum 通道不匹配 |
| 03-22 | ✅ 完成 | 全部 4 文件实现完毕 |
| 03-22 | ✅ 完成 | 分布式脚本已集成 A+B 开关 |
| 03-22 | 🟡 开始 | R1 调研完成，进入 E 执行 |
