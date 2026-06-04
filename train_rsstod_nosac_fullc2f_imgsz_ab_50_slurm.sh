#!/bin/bash
#SBATCH --job-name=rsstod_imgsz_ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%A_%a.out
#SBATCH --error=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11
cd "$PROJECT_DIR"

mkdir -p logs

MODEL=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml
DATA=data_RS_STOD_server2.yaml
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  IMGSZ=800
  BATCH=4
  RUN_NAME=rsstod_nosac_fullc2f_img800_b4_e50_seed0_${SLURM_JOB_ID}
else
  IMGSZ=1024
  BATCH=2
  RUN_NAME=rsstod_nosac_fullc2f_img1024_b2_e50_seed0_${SLURM_JOB_ID}
fi

PROJECT_OUT=runs/rsstod_p1_nosac_fullc2f_imgsz_ab50

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL"
echo "数据 YAML: $DATA"
echo "初始权重: $WEIGHTS"
echo "imgsz/batch: $IMGSZ/$BATCH"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "============================================"

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到初始权重: $WEIGHTS"
  exit 1
fi

python "train_yolo111 copy.py" \
  --cfg "$MODEL" \
  --data "$DATA" \
  --weights "$WEIGHTS" \
  --epochs 50 \
  --patience 50 \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device 0 \
  --workers 8 \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
  --trainer_mode full \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 3.0 \
  --warmup_momentum 0.8 \
  --warmup_bias_lr 0.1 \
  --enable_ssds

echo "训练完成，退出码: $?"
