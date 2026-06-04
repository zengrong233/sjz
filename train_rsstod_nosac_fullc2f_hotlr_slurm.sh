#!/bin/bash
#SBATCH --job-name=rsstod_hotlr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.out
#SBATCH --error=/share/home/u2415363072/4.25/ultralyticsPro--YOLO11/logs/slurm_%j.err

set -euo pipefail

# =========================
# 基础路径
# =========================
PROJECT_DIR="/share/home/u2415363072/4.25/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"

mkdir -p logs

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/rsstod_best.pt"
DATA_ROOT="$PROJECT_DIR/datasets/RS-STOD"

PROJECT_OUT="runs/rsstod_p1_nosac_fullc2f_hotlr"
RUN_NAME="rsstod_p1_nosac_fullc2f_rsstodbest_lr008_025_035_${SLURM_JOB_ID}"

DATA_YAML="$PROJECT_DIR/data_RS_STOD_4p25_runtime_${SLURM_JOB_ID}.yaml"

# =========================
# 热启动低 LR 参数
# =========================
BACKBONE_LR_SCALE="0.08"
SMALLOBJ_LR_SCALE="0.25"
HEAD_LR_SCALE="0.35"

EPOCHS="100"
BATCH="6"
IMGSZ="640"
WORKERS="8"
DEVICE="0"

# batch=6 时 nbs=64 对应基础累积约 10~11，这里固定 10，避免动态过激
BASE_ACCUM="10"
MAX_ACCUM="20"

# =========================
# 环境
# =========================
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

export PYTHONUNBUFFERED=1
export ULTRALYTICS_SKIP_DOWNLOAD=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# =========================
# 强校验
# =========================
echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据根目录: $DATA_ROOT"
echo "初始权重: $WEIGHTS"
echo "输出目录: $PROJECT_OUT/$RUN_NAME"
echo "batch/imgsz/amp: $BATCH/$IMGSZ/0"
echo "LR scales: backbone=$BACKBONE_LR_SCALE small_object=$SMALLOBJ_LR_SCALE head=$HEAD_LR_SCALE"
echo "============================================"

test -f "$MODEL_CFG" || { echo "错误: 未找到模型配置 $MODEL_CFG"; exit 1; }
test -f "$WEIGHTS" || { echo "错误: 未找到初始权重 $WEIGHTS"; exit 1; }
test -d "$DATA_ROOT/images/train" || { echo "错误: 未找到 $DATA_ROOT/images/train"; exit 1; }
test -d "$DATA_ROOT/images/val" || { echo "错误: 未找到 $DATA_ROOT/images/val"; exit 1; }
test -d "$DATA_ROOT/labels/train" || { echo "错误: 未找到 $DATA_ROOT/labels/train"; exit 1; }
test -d "$DATA_ROOT/labels/val" || { echo "错误: 未找到 $DATA_ROOT/labels/val"; exit 1; }

cat > "$DATA_YAML" <<EOF
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
EOF

nvidia-smi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}")
PY

# =========================
# Dry-run：确认 YAML 可解析
# =========================
python - <<PY
from ultralytics import YOLO
m = YOLO("$MODEL_CFG", task="detect")
m.info(detailed=False)
PY

# =========================
# 正式训练
# =========================
python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --trainer_mode full \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --project "$PROJECT_OUT" \
  --name "$RUN_NAME" \
  --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --smallobj_lr_scale "$SMALLOBJ_LR_SCALE" \
  --head_lr_scale "$HEAD_LR_SCALE" \
  --base_accum "$BASE_ACCUM" \
  --max_accum "$MAX_ACCUM" \
  --warmup_curriculum_epochs 1 \
  --enable_ssds \
  --ssds_mode soft \
  --tiny_boost 1.5 \
  --small_boost 1.3

EXIT_CODE=$?
echo "训练完成，退出码: $EXIT_CODE"
exit $EXIT_CODE