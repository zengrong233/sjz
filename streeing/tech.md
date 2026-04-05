# Tech Stack & Build System

## Language & Runtime
- Python >= 3.8

## Build System
- setuptools + wheel (configured in `pyproject.toml`)
- Editable install: `pip install -e .`
- Package install: `pip install ultralytics`

## Core Dependencies
- PyTorch >= 1.8.0, TorchVision >= 0.9.0
- NumPy, OpenCV, Pillow, Matplotlib, SciPy
- pandas, seaborn (visualization)
- PyYAML (config parsing)
- tqdm (progress bars)
- ultralytics-thop (FLOPs computation)

## Optional Dependencies
- `dev`: pytest, pytest-cov, coverage, mkdocs (docs)
- `export`: onnx, coremltools, openvino, tensorflow, tensorflowjs
- `logging`: comet, tensorboard, dvclive
- `extra`: albumentations, pycocotools, hub-sdk

## Code Style
- Line length: 120 characters (ruff, isort, yapf all configured to 120)
- Formatter: yapf (PEP8 based), ruff
- Import sorting: isort (single-line output)
- Docstrings: Google-style, wrapped at 120 chars (docformatter)
- Spell checking: codespell

## Common Commands

```bash
# Install for development
pip install -e .

# Train baseline (PRR-v3 structure only)
python "train_yolo111 copy.py" --trainer_mode baseline --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml --data data_VD.yaml --batch 4

# Train with A+B strategy
python "train_yolo111 copy.py" --trainer_mode ab --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml --data data_VD.yaml --batch 4 --debug_routing

# Train full stack (recommended)
python "train_yolo111 copy.py" --trainer_mode full --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml --data data_VD.yaml --batch 4 --enable_ssds --debug_routing

# Ablation: only A (Scale-Routed Optimizer)
python "train_yolo111 copy.py" --trainer_mode a --batch 4

# Ablation: only B (Noise-Aware Batch Curriculum)
python "train_yolo111 copy.py" --trainer_mode b --batch 4

# Supercomputer (SLURM)
sbatch train_slurm_ab.sh baseline
sbatch train_slurm_ab.sh ab
sbatch train_slurm_ab.sh full

# Validate
yolo val model=best.pt data=data_VD.yaml

# Predict
yolo predict model=yolo11n.pt source=path/to/image.jpg

# Export
yolo export model=yolo11n.pt format=onnx

# Run tests
pytest tests/
```

## Testing
- Framework: pytest
- Config in `pyproject.toml` under `[tool.pytest.ini_options]`
- Markers: `@pytest.mark.slow` for slow tests
- Coverage source: `ultralytics/`
- Doctest modules enabled by default (`--doctest-modules`)

## GPU Memory Notes
- Target model: YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3 (three-scale P2/P3/P4 + AsDDet)
- TopBasicLayer uses Spatial Reduction Attention (SR_RATIO=4) for P2 layer (160x160)
- Recommended: batch=4 on 24GB GPU, batch=8 on 40GB+ GPU
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to reduce fragmentation
