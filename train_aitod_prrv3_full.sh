#!/bin/bash
#SBATCH -J aitod_prrv3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.out

PROJECT_DIR="/share/home/u2415363072/4.12/ultralyticsPro--YOLO11"

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml \
  --weights yolo11n.pt \
  --data data_AITOD_slurm1.yaml \
  --device 0 \
  --batch 4 \
  --epochs 300 \
  --imgsz 640 \
  --workers 8 \
  --trainer_mode full \
  --enable_ssds \
  --debug_routing
