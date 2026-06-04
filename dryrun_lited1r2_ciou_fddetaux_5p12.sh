#!/bin/bash
set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.12/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-CIoU-FDDetAux.yaml"

echo "============================================"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "关键 YAML 开关:"
grep -nE '^(loss|fd_aux|fd_aux_weight|fd_aux_levels|fd_aux_area_thr):' "$MODEL_CFG"
echo "============================================"

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('$MODEL_CFG', task='detect').info()"
