# Project Structure

```
ultralytics/                  # Main package
├── __init__.py               # Version, public API exports
├── cfg/                      # Official configs
│   ├── default.yaml          # Default training hyperparameters
│   ├── datasets/             # Dataset configs (COCO, VOC, etc.)
│   ├── models/               # Official model architecture YAMLs
│   ├── solutions/            # Solution configs
│   └── trackers/             # Tracker configs (BotSORT, ByteTrack)
├── cfg_yolo11/               # Custom enhanced YOLO11 configs (100+)
│   ├── YOLO11-Attention/     # Attention mechanisms (CBAM, EMA, GAM, etc.)
│   ├── YOLO11-Backbone/      # Backbone improvements
│   ├── YOLO11-Conv改进/       # Conv modifications (DCNv3, DCNv4, ODConv)
│   ├── YOLO11-Head/          # Detection head improvements (AsDDet, DynamicHead)
│   ├── YOLO11-Loss/          # Loss function configs (SIoU, WIoU, NWD)
│   ├── YOLO11-Neck/          # Neck/FPN improvements (AFPN, HFAMPAN)
│   ├── YOLO11-OBB/           # Oriented bounding box configs
│   ├── YOLO11-Pose/          # Pose estimation configs
│   ├── YOLO11-Seg/           # Segmentation configs
│   └── YOLO11-多个创新点组合改进/  # Multi-improvement combinations
├── data/                     # Data loading, augmentation, dataset handling
├── engine/                   # Core engines
│   ├── trainer.py            # Training loop (with L1 sparse regularization)
│   ├── optimizer_router.py   # Scale-Routed Optimizer parameter grouping (A strategy)
│   ├── batch_curriculum.py   # Noise-Aware Batch Curriculum scheduler (B strategy)
│   ├── small_object_trainer.py # SmallObjectABTrainer (A+B joint trainer)
│   ├── validator.py          # Validation loop
│   ├── predictor.py          # Inference
│   ├── exporter.py           # Model export (ONNX, TF, etc.)
│   └── model.py              # High-level model API
├── models/                   # Model implementations
│   ├── yolo/                 # YOLO family (detect, segment, pose, classify, obb)
│   ├── sam/                  # Segment Anything Model
│   ├── rtdetr/               # RT-DETR
│   ├── fastsam/              # FastSAM
│   └── nas/                  # Neural Architecture Search
├── nn/                       # Neural network modules
│   ├── tasks.py              # Model building from YAML configs
│   ├── modules/              # Core NN building blocks (conv, attention, heads)
│   ├── core11/               # Enhanced custom modules (50+ files)
│   │   └── GDM.py            # TopBasicLayer with Spatial Reduction Attention
│   └── autobackend.py        # Multi-backend inference
├── solutions/                # Ready-to-use CV solutions (counting, heatmaps, etc.)
├── trackers/                 # Object tracking (BotSORT, ByteTrack)
└── utils/                    # Utilities
    ├── loss.py               # Standard loss functions
    ├── NewLoss/              # Custom loss implementations (NWD, etc.)
    ├── metrics.py            # Evaluation metrics (mAP, etc.)
    ├── tal.py                # Task-aligned label assignment
    ├── ops.py                # Tensor operations (NMS, etc.)
    └── callbacks/            # Training callbacks

# Root-level training scripts
train_yolo111 copy.py         # Main training entry (baseline/a/b/ab modes)
train_distributed_torchrun.py # Distributed training with torchrun
train_slurm_ab.sh             # SLURM supercomputer submission script

# Dataset configs
data_VD.yaml                  # VisDrone dataset (Windows paths)
data_VD_slurm.yaml            # VisDrone dataset (supercomputer paths)

# Model configs
YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml  # 当前 VisDrone 小目标主配置（PRR-v3 + AsDDet）

datasets/                     # Sample datasets (YOLO format)
├── images/{VD_train,VD_val,VD_test}/  # VisDrone images
└── labels/{VD_train,VD_val,VD_test}/  # YOLO-format label .txt files

tasks/                        # RIPER-5 task tracking
tests/                        # pytest test suite
docker/                       # Dockerfiles for various platforms
docs/                         # MkDocs documentation source
examples/                     # Inference examples (Python, C++, Rust, ONNX)
weights/                      # Model weight files
```

## Key Patterns
- Model architectures are defined as YAML configs and parsed by `ultralytics/nn/tasks.py`
- New modules are registered in `ultralytics/nn/tasks.py` to be available in YAML configs
- Dataset configs are YAML files specifying paths, class count, and class names
- The `yolo` CLI entry point is defined in `ultralytics/cfg/__init__.py`
- Training results go to `runs/detect/` by default
- A+B training: `--trainer_mode {baseline,a,b,ab}` in `train_yolo111 copy.py`
- Parameter routing: top-level module index classification in `optimizer_router.py`
