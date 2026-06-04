#!/bin/bash
#SBATCH -J vedai_nosac_fullc2f
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR=/share/home/u2415363072/5.3/ultralyticsPro--YOLO11
cd "$PROJECT_DIR"

mkdir -p logs

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_NAME="vedai_nosac_fullc2f_640_b4_e300_${JOB_ID}"

python "train_yolo111 copy.py" \
  --cfg YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml \
  --data data_VEDAI_server2.yaml \
  --weights /share/home/u2415363072/5.3/ultralyticsPro--YOLO11/best_pt/yolo11n.pt \
  --device 0 \
  --epochs 300 \
  --patience 100 \
  --imgsz 640 \
  --batch 4 \
  --workers 8 \
  --project runs/vedai_p1_nosac_fullc2f \
  --name "$RUN_NAME" \
  --trainer_mode full \
  --enable_ssds \
  --base_accum 16 \
  --max_accum 24 \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 3.0 \
  --warmup_momentum 0.8 \
  --warmup_bias_lr 0.1 \
  --backbone_lr_scale 0.5 \
  --smallobj_lr_scale 1.25 \
  --head_lr_scale 1.5

echo "训练完成，退出码: $?"
