#!/bin/bash
#SBATCH --job-name=rsstod_fix_p3fb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.out
#SBATCH --error=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.err

set -euo pipefail

PROJECT_DIR="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"

mkdir -p logs

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/rsstod_best.pt"
DATA_ROOT="$PROJECT_DIR/datasets/RS-STOD"

DATA_YAML="$PROJECT_DIR/data_RS_STOD_4p25_runtime_${SLURM_JOB_ID}.yaml"

PROJECT_OUT="runs/rsstod_p1_nosac_fullc2f_fix_ssds_warmup"
RUN_NAME="rsstod_p1_nosac_fullc2f_fix_warmup_p3fb_${SLURM_JOB_ID}"

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

export PYTHONUNBUFFERED=1
export ULTRALYTICS_SKIP_DOWNLOAD=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据根目录: $DATA_ROOT"
echo "数据 YAML: $DATA_YAML"
echo "初始权重: $WEIGHTS"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "修复项: warmup=0 + SSDS P3 fallback"
echo "============================================"

test -f "$MODEL_CFG" || { echo "错误: 未找到模型配置 $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "错误: 未找到初始权重 $WEIGHTS"; exit 1; }
test -d "$DATA_ROOT/images/train" || { echo "错误: 未找到 $DATA_ROOT/images/train"; exit 1; }
test -d "$DATA_ROOT/images/val" || { echo "错误: 未找到 $DATA_ROOT/images/val"; exit 1; }
test -d "$DATA_ROOT/labels/train" || { echo "错误: 未找到 $DATA_ROOT/labels/train"; exit 1; }
test -d "$DATA_ROOT/labels/val" || { echo "错误: 未找到 $DATA_ROOT/labels/val"; exit 1; }

cat > "$DATA_YAML" <<YAML
path: $DATA_ROOT
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: Small Vehicle
  1: Large Vehicle
  2: Ship
  3: Airplane
  4: Storage Tank
YAML

nvidia-smi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}")
PY

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/engine/scale_supervision.py \
  ultralytics/utils/NewLoss/ssds_loss.py \
  ultralytics/engine/small_object_trainer.py

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --trainer_mode full \
  --epochs 60 \
  --patience 30 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --device 0 \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
  --backbone_lr_scale 0.08 \
  --smallobj_lr_scale 0.25 \
  --head_lr_scale 0.35 \
  --base_accum 10 \
  --max_accum 20 \
  --warmup_curriculum_epochs 1 \
  --warmup_epochs 0 \
  --warmup_bias_lr 0.0 \
  --warmup_momentum 0.937 \
  --enable_ssds \
  --ssds_mode soft \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_topk 1 \
  --ssds_p3_fallback_score 0.2 \
  --ssds_p3_fallback_min_area 64 \
  --ssds_p3_fallback_max_area 1024

EXIT_CODE=$?
echo "训练完成，退出码: $EXIT_CODE"
exit $EXIT_CODE
