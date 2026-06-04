#!/bin/bash
#SBATCH -J usod_e1_e300
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="/share/home/u2415363072/5.3/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/usod_best.pt"

USOD_ROOT="$PROJECT_DIR/datasets/USOD"
if [ ! -d "$USOD_ROOT/images/train" ]; then
  USOD_ROOT="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/datasets/USOD"
fi

DATA_YAML="$PROJECT_DIR/data_USOD_5p3_runtime_${JOB_ID}.yaml"
cat > "$DATA_YAML" <<EOF
path: $USOD_ROOT
train: images/train
val: images/val
test: images/test
nc: 1
names: ['vehicle']
EOF

rm -f "$USOD_ROOT"/labels/*.cache "$USOD_ROOT"/labels/train.cache "$USOD_ROOT"/labels/val.cache "$USOD_ROOT"/labels/test.cache 2>/dev/null || true

python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 300 \
  --patience 100 \
  --imgsz 640 \
  --batch 4 \
  --workers 8 \
  --project runs/usod_lited1r2_e300 \
  --name "usod_lited1r2_b4_e300_from_usodbest_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.003 \
  --lrf 0.01 \
  --warmup_epochs 0 \
  --warmup_bias_lr 0.003 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
