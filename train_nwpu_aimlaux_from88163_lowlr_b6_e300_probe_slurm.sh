#!/bin/bash
#SBATCH -J nwpu_ai_refit300
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

PROJECT_DIR="/share/home/u2415363072/5.13/ultralyticsPro--YOLO11"
cd "$PROJECT_DIR"
mkdir -p logs runs/nwpu_aimlaux_bridge_refit_e300

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-NWD-AIMLAux.yaml"
DATA_YAML="$PROJECT_DIR/data_NWPU-VHR10_Val86_ZIP_5p13.yaml"
WEIGHTS="$PROJECT_DIR/runs/nwpu_lited1r2_nwd_aimlaux_e300/nwpu_nwd_aimlaux_w0003_val86_yolo11n_b6_e300_88163/weights/best.pt"

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 未找到数据 YAML: $DATA_YAML"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到 88163 best.pt: $WEIGHTS"
  exit 1
fi

if ! grep -qE '^loss:[[:space:]]*NWD' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 不是明确 loss: NWD"
  grep -nE '^loss:' "$MODEL_CFG" || true
  exit 1
fi

if ! grep -qE '^fd_aux:[[:space:]]*true' "$MODEL_CFG"; then
  echo "错误: 当前 YAML 未开启 fd_aux"
  grep -nE '^fd_aux' "$MODEL_CFG" || true
  exit 1
fi

echo "============================================"
echo "作业ID: $JOB_ID"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "数据 YAML: $DATA_YAML"
echo "初始权重: $WEIGHTS"
echo "输出目录: runs/nwpu_aimlaux_bridge_refit_e300/nwpu_aimlaux_from88163_lowlr_b6_e300_${JOB_ID}"
echo "loss/fd_aux_weight: NWD / 0.003"
echo "lr0/lrf/warmup: 0.0002 / 0.1 / 1.0"
echo "batch/imgsz/amp: 6/640/false"
echo "============================================"

grep -nE '^(loss:|fd_aux|fd_aux_weight|fd_aux_levels|fd_aux_area_thr)' "$MODEL_CFG"

nvidia-smi || true

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py \
  ultralytics/nn/tasks.py \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py

python - <<PY
from ultralytics import YOLO
m = YOLO("$MODEL_CFG", task="detect")
m.info(detailed=False)
PY

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 300 \
  --patience 100 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --project runs/nwpu_aimlaux_bridge_refit_e300 \
  --name "nwpu_aimlaux_from88163_lowlr_b6_e300_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.0002 \
  --lrf 0.1 \
  --warmup_epochs 1.0 \
  --warmup_bias_lr 0.0 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
