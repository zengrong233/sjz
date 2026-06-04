#!/bin/bash
#SBATCH -J hrsid_aiml_b4
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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.30/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs runs/hrsid_lited1r2_nwd_aimlaux

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2-NWD-AIMLAux.yaml"
DATA_YAML="data_HRSID-DET_5p30.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/yolo11n.pt"

RUN_NAME="hrsid_nwd_aimlaux_w0003_det_b4_e300_${JOB_ID}"

echo "============================================"
echo "作业ID: $JOB_ID"
echo "实验: HRSID-DET LiteD1R2 + NWD + AIMLAux"
echo "模型配置: $MODEL_CFG"
echo "数据 YAML: $DATA_YAML"
echo "初始权重: $WEIGHTS"
echo "输出目录: runs/hrsid_lited1r2_nwd_aimlaux/$RUN_NAME"
echo "batch/imgsz/amp: 4/640/false"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "============================================"

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 模型配置不存在: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 数据 YAML 不存在: $DATA_YAML"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 初始权重不存在: $WEIGHTS"
  exit 1
fi

echo "[CHECK] YAML key fields"
grep -nE '^(loss|fd_aux|fd_aux_weight|fd_aux_levels|fd_aux_area_thr):' "$MODEL_CFG"

echo "[CHECK] HRSID-DET counts"
python - <<'PY'
from pathlib import Path
import yaml

with open("data_HRSID-DET_5p30.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

root = Path(data["path"])
for split in ["train", "valid", "test"]:
    img_dir = root / split / "images"
    lab_dir = root / split / "labels"
    imgs = list(img_dir.glob("*"))
    labs = list(lab_dir.glob("*.txt"))
    boxes = 0
    bad = 0
    for p in labs:
        for line in p.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                bad += 1
            else:
                boxes += 1
    print(f"{split}: images={len(imgs)}, labels={len(labs)}, boxes={boxes}, bad={bad}")
PY

nvidia-smi || true

python -m py_compile \
  "train_yolo111 copy.py" \
  ultralytics/nn/core11/GDM.py \
  ultralytics/nn/modules/block.py \
  ultralytics/nn/tasks.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

python -c "from ultralytics import YOLO; YOLO('${MODEL_CFG}', task='detect').info(detailed=False)"

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
  --project runs/hrsid_lited1r2_nwd_aimlaux \
  --name "$RUN_NAME" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.01 \
  --lrf 0.01 \
  --warmup_epochs 3.0 \
  --warmup_bias_lr 0.1 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
