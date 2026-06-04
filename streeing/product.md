# Product Overview

This is an enhanced fork of the Ultralytics YOLO11 repository for real-time object detection, segmentation, pose estimation, and classification.

On top of the official YOLO11 implementation, this repo adds:
- Main-line detector and ablation variants as root-level `YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-*.yaml` files (PRR/MFFF/GDM/AsDDet/NWD/AIML)
- Custom neural network modules in `ultralytics/nn/core11/` (DCNv3/v4, Mamba, Swin, EfficientNet variants, etc.)
- Custom loss functions in `ultralytics/utils/NewLoss/`
- A sample VisDrone 10-class detection dataset in `datasets/`
- Distributed and supercomputer training scripts (`train_slurm_ab.sh`, `train_distributed_torchrun.py`)
- A+B joint training strategy for small object detection:
  - A: Scale-Routed Optimizer (module-level parameter grouping with differentiated learning rates)
  - B: Noise-Aware Batch Curriculum (dynamic gradient accumulation based on small target density)
  - SmallObjectABTrainer in `ultralytics/engine/small_object_trainer.py`
- Spatial Reduction Attention in `GDM.py` TopBasicLayer to handle high-resolution P2 features
- Multiple training configurations for memory optimization and gradient accumulation

The project is licensed under AGPL-3.0. Current version: 8.3.13.
