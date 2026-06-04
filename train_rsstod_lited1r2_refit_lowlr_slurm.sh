#!/bin/bash
# RS-STOD low-lr refit (P0-A): 验证 H1b
#
# 假设：85036rsstod_best.pt 续训之所以 ep24 触顶后回落，是因为 lr0=0.002 + warmup=0
# 起点扰动偏大。本次将 lr0 调到 0.0005 (相对 87109 ÷4) + 启用 warmup_epochs=1.0。
#
# 数据口径已经过 val-only 锚定（87xxx, Δ=-0.00026）：H1a 通过。
# 所以本次结果可直接与原主结果 mAP50-95=0.45767 对比，差异归因于训练策略。

#SBATCH -J rsstod_refit_lowlr
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11_py310

PROJECT_DIR="${PROJECT_DIR:-/share/home/u2415363072/5.13/ultralyticsPro--YOLO11}"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

MODEL_CFG="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f-LiteD1R2.yaml"
WEIGHTS="$PROJECT_DIR/best_pt/85036rsstod_best.pt"
DATA_YAML="$PROJECT_DIR/data_RS-STOD_PREP_5p13.yaml"

if [ ! -f "$MODEL_CFG" ]; then
  echo "错误: 未找到模型配置: $MODEL_CFG"
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "错误: 未找到 best.pt: $WEIGHTS"
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "错误: 未找到数据 YAML: $DATA_YAML"
  exit 1
fi

# 参考 RS-STOD val 集 cache，避免上一次 train cache 影响指标
RSSTOD_DIR="$(grep -oE 'path:\s*\S+' "$DATA_YAML" | awk '{print $2}')" || true
if [ -n "${RSSTOD_DIR:-}" ] && [ -d "${RSSTOD_DIR}/labels" ]; then
  rm -f "${RSSTOD_DIR}/labels/"*.cache
fi

echo "============================================"
echo "作业ID: $JOB_ID"
echo "节点: ${SLURMD_NODENAME:-unknown}"
echo "实验: P0-A RS-STOD low-lr refit (验证 H1b)"
echo "项目目录: $PROJECT_DIR"
echo "模型配置: $MODEL_CFG"
echo "起始权重: $WEIGHTS (anchor: mAP50-95=0.45767)"
echo "数据 YAML: $DATA_YAML (PREP 锚定 Δ=-0.00026)"
echo "输出目录: runs/rsstod_lited1r2_refit_lowlr/rsstod_refit_lowlr_b6_e100_${JOB_ID}"
echo "训练超参: lr0=0.0005 lrf=0.1 warmup=1.0 patience=80 epochs=100 batch=6 amp=false"
echo "============================================"

nvidia-smi || true

# 静态自检
python -m py_compile \
  ultralytics/nn/core11/GDM.py \
  ultralytics/utils/loss.py \
  ultralytics/utils/NewLoss/ioulossone.py

# 模型实例化自检（params/GFLOPs 必须与 5.697M / 26.8 GFLOPs 一致）
python -c "from ultralytics import YOLO; m = YOLO('$MODEL_CFG', task='detect'); m.info()"

python "train_yolo111 copy.py" \
  --cfg "$MODEL_CFG" \
  --weights "$WEIGHTS" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 100 \
  --patience 80 \
  --imgsz 640 \
  --batch 6 \
  --workers 8 \
  --seed 0 \
  --project runs/rsstod_lited1r2_refit_lowlr \
  --name "rsstod_refit_lowlr_b6_e100_${JOB_ID}" \
  --trainer_mode full \
  --enable_ssds \
  --ssds_mode soft \
  --small_area_thr 1024 \
  --tiny_boost 1.5 \
  --small_boost 1.3 \
  --ssds_p3_fallback \
  --ssds_p3_fallback_score 0.10 \
  --lr0 0.0005 \
  --lrf 0.1 \
  --warmup_epochs 1.0 \
  --warmup_bias_lr 0.0 \
  --close_mosaic 10

echo "训练完成，退出码: $?"
