# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **enhanced Ultralytics YOLO11 repository** containing the official YOLO11 implementation plus extensive modifications and improvements. The repository includes:

- **Core YOLO11**: Official Ultralytics implementation for object detection, segmentation, pose estimation, and classification
- **Enhanced configurations**: Over 100+ custom YAML configurations in `ultralytics/cfg_yolo11/` with various architectural improvements
- **Custom modules**: Enhanced neural network components in `ultralytics/nn/core11/` and `ultralytics/nn/modules/`
- **Custom losses**: Advanced loss functions in `ultralytics/utils/NewLoss/`
- **Training datasets**: Sample 3-class custom detection dataset in `datasets/`

## Key Architecture

### Core Structure
- `ultralytics/`: Main package containing models, training, inference, and utilities
- `ultralytics/models/yolo/`: YOLO model implementations
- `ultralytics/nn/`: Neural network modules, layers, and architectures
- `ultralytics/engine/`: Training, validation, prediction, and export engines
- `ultralytics/utils/`: Utilities for data loading, metrics, visualization, etc.

### Enhanced Components
- `ultralytics/cfg_yolo11/`: Custom YOLO11 configurations organized by improvement type:
  - Attention mechanisms (CBAM, EMA, GAM, etc.)
  - Backbone improvements (ACMix, UAV, etc.)
  - Conv modifications (DCNv3, DCNv4, etc.)
  - Head improvements (AsDDet, DynamicHead, etc.)
  - Loss functions (SIoU, WIoU, NWD, etc.)
  - Neck/FPN improvements (AFPN, HFAMPAN, etc.)
- `ultralytics/nn/core11/`: Enhanced neural network building blocks
- `ultralytics/utils/NewLoss/`: Custom loss function implementations

## Common Development Commands

### Installation
```bash
pip install ultralytics
# OR for development:
pip install -e .
```

### Training
```bash
# Basic training
yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640

# Custom dataset training (using included 3-class dataset)
yolo train model=yolo11n.pt data=data_3c.yaml epochs=100 imgsz=640

# Enhanced model training with custom configs
yolo train model=ultralytics/cfg_yolo11/YOLO11-多个创新点组合改进/YOLO11-HFAMPAN-AsDDet-NWD.yaml data=data_3c.yaml epochs=100 imgsz=640
```

### Validation & Testing
```bash
# Validation
yolo val model=yolo11n.pt data=coco8.yaml

# Testing with custom models
yolo val model=best.pt data=data_3c.yaml
```

### Prediction/Inference
```bash
# Single image prediction
yolo predict model=yolo11n.pt source=path/to/image.jpg

# Batch prediction
yolo predict model=best.pt source=path/to/images/
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_engine.py
pytest tests/test_python.py

# Run tests with slow tests included
pytest tests/ --slow
```

### Export Models
```bash
# Export to ONNX
yolo export model=yolo11n.pt format=onnx

# Export to other formats
yolo export model=yolo11n.pt format=torchscript
```

## Important File Locations

### Configuration Files
- `ultralytics/cfg/default.yaml`: Default training/inference parameters
- `data_3c.yaml`: Custom 3-class dataset configuration (root level)
- `YOLO11-HFAMPAN-AsDDet-NWD.yaml`: Example enhanced model config (root level)

### Model Weights
- `yolo11n.pt`: Pre-trained YOLO11 nano model (root level)
- `3c_best.pt`: Trained model on 3-class dataset (root level)
- `weights/`: Directory for storing model weights

### Training Scripts
- `train_yolo11.py`: Custom training script
- `predict_11.py`: Custom prediction script
- `predict_c.py`: Custom prediction for 3-class model

### Custom Modules
- `ultralytics/nn/core11/`: Enhanced building blocks (attention, FPN, etc.)
- `ultralytics/utils/NewLoss/`: Custom loss implementations
- `ultralytics/nn/modules/`: Core neural network modules

## Development Notes

### Custom Model Configurations
The repository contains 100+ custom YAML configurations in `ultralytics/cfg_yolo11/` organized by improvement type. These configs can be used directly for training enhanced YOLO11 models.

### Dataset Structure
The included sample dataset follows YOLO format:
- `datasets/images/`: Training/validation/test images
- `datasets/labels/`: Corresponding annotation files
- Each class: 0=first_class, 1=second_class, 2=third_class

### Enhanced Features
- Multiple attention mechanisms (CBAM, EMA, GAM, SK, etc.)
- Advanced convolution types (DCNv3, DCNv4, ODConv, etc.)
- Improved detection heads (AsDDet, DynamicHead, etc.)
- Enhanced loss functions (SIoU, WIoU, NWD, etc.)
- Advanced FPN/neck architectures (AFPN, HFAMPAN, etc.)

### Testing Framework
- Uses pytest with custom configurations in `pyproject.toml`
- Slow tests marked with `@pytest.mark.slow`
- Temporary directory management in `tests/conftest.py`

### Documentation
- Main docs in `docs/` directory using MkDocs
- API reference automatically generated
- Multiple language support (English, Chinese, etc.)