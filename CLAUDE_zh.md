# CLAUDE_zh.md

本文件为Claude Code (claude.ai/code) 在处理此仓库代码时提供指导。

## 项目概述

这是一个**增强版Ultralytics YOLO11仓库**，包含官方YOLO11实现以及大量修改和改进。该仓库包括：

- **核心YOLO11**：用于目标检测、分割、姿态估计和分类的官方Ultralytics实现
- **增强配置**：在`ultralytics/cfg_yolo11/`中包含100+个自定义YAML配置，具有各种架构改进
- **自定义模块**：在`ultralytics/nn/core11/`和`ultralytics/nn/modules/`中的增强神经网络组件
- **自定义损失函数**：在`ultralytics/utils/NewLoss/`中的高级损失函数
- **训练数据集**：在`datasets/`中的3类自定义检测数据集样本

## 核心架构

### 核心结构
- `ultralytics/`：包含模型、训练、推理和工具的主包
- `ultralytics/models/yolo/`：YOLO模型实现
- `ultralytics/nn/`：神经网络模块、层和架构
- `ultralytics/engine/`：训练、验证、预测和导出引擎
- `ultralytics/utils/`：数据加载、指标、可视化等工具

### 增强组件
- `ultralytics/cfg_yolo11/`：按改进类型组织的自定义YOLO11配置：
  - 注意力机制（CBAM、EMA、GAM等）
  - 骨干网络改进（ACMix、UAV等）
  - 卷积修改（DCNv3、DCNv4等）
  - 检测头改进（AsDDet、DynamicHead等）
  - 损失函数（SIoU、WIoU、NWD等）
  - 颈部/FPN改进（AFPN、HFAMPAN等）
- `ultralytics/nn/core11/`：增强的神经网络构建块
- `ultralytics/utils/NewLoss/`：自定义损失函数实现

## 常用开发命令

### 安装
```bash
pip install ultralytics
# 或者用于开发：
pip install -e .
```

### 训练
```bash
# 基础训练
yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640

# 自定义数据集训练（使用包含的3类数据集）
yolo train model=yolo11n.pt data=data_3c.yaml epochs=100 imgsz=640

# 使用自定义配置的增强模型训练
yolo train model=ultralytics/cfg_yolo11/YOLO11-多个创新点组合改进/YOLO11-HFAMPAN-AsDDet-NWD.yaml data=data_3c.yaml epochs=100 imgsz=640
```

### 验证和测试
```bash
# 验证
yolo val model=yolo11n.pt data=coco8.yaml

# 使用自定义模型测试
yolo val model=best.pt data=data_3c.yaml
```

### 预测/推理
```bash
# 单张图像预测
yolo predict model=yolo11n.pt source=path/to/image.jpg

# 批量预测
yolo predict model=best.pt source=path/to/images/
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行特定测试类别
pytest tests/test_engine.py
pytest tests/test_python.py

# 运行包含慢速测试的测试
pytest tests/ --slow
```

### 导出模型
```bash
# 导出为ONNX
yolo export model=yolo11n.pt format=onnx

# 导出为其他格式
yolo export model=yolo11n.pt format=torchscript
```

## 重要文件位置

### 配置文件
- `ultralytics/cfg/default.yaml`：默认训练/推理参数
- `data_3c.yaml`：自定义3类数据集配置（根目录级别）
- `YOLO11-HFAMPAN-AsDDet-NWD.yaml`：增强模型配置示例（根目录级别）

### 模型权重
- `yolo11n.pt`：预训练的YOLO11 nano模型（根目录级别）
- `3c_best.pt`：在3类数据集上训练的模型（根目录级别）
- `weights/`：存储模型权重的目录

### 训练脚本
- `train_yolo11.py`：自定义训练脚本
- `predict_11.py`：自定义预测脚本
- `predict_c.py`：3类模型的自定义预测

### 自定义模块
- `ultralytics/nn/core11/`：增强的构建块（注意力、FPN等）
- `ultralytics/utils/NewLoss/`：自定义损失实现
- `ultralytics/nn/modules/`：核心神经网络模块

## 开发说明

### 自定义模型配置
该仓库在`ultralytics/cfg_yolo11/`中包含100+个按改进类型组织的自定义YAML配置。这些配置可以直接用于训练增强的YOLO11模型。

### 数据集结构
包含的示例数据集遵循YOLO格式：
- `datasets/images/`：训练/验证/测试图像
- `datasets/labels/`：对应的标注文件
- 每个类别：0=第一类，1=第二类，2=第三类

### 增强功能
- 多种注意力机制（CBAM、EMA、GAM、SK等）
- 高级卷积类型（DCNv3、DCNv4、ODConv等）
- 改进的检测头（AsDDet、DynamicHead等）
- 增强的损失函数（SIoU、WIoU、NWD等）
- 高级FPN/颈部架构（AFPN、HFAMPAN等）

### 测试框架
- 使用pytest，在`pyproject.toml`中有自定义配置
- 慢速测试用`@pytest.mark.slow`标记
- 在`tests/conftest.py`中进行临时目录管理

### 文档
- 主要文档在`docs/`目录中，使用MkDocs
- API参考自动生成
- 支持多种语言（英语、中文等）