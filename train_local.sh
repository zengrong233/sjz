#!/bin/bash
# Local runner converted from train_slurm_ab.sh
# Usage: ./train_local.sh [mode]
# mode: baseline | ab | full
# legacy aliases: baseline_prr -> baseline, prr_ab -> ab

set -euo pipefail

# --- project and env ---
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-yolov11}"
TRAINER_MODE="${1:-full}"
DEVICE="${DEVICE:-0}"
DATA_YAML="${DATA_YAML:-data_VD_slurm.yaml}"

# Try to source conda if available (common miniconda location in this workspace)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}" || echo "Warning: could not activate ${CONDA_ENV}"
else
  echo "Note: conda not found; ensure Python env meets requirements."
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/logs"

echo "============================================"
echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-$DEVICE}"
echo "Trainer mode: ${TRAINER_MODE}"
echo "Project dir: ${PROJECT_DIR}"
echo "============================================"
nvidia-smi || true
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')" || true
echo "============================================"

cd "${PROJECT_DIR}"

YAML_PRR_V3="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml"

case "${TRAINER_MODE}" in
    baseline)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device ${DEVICE} --batch 6 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode baseline
        ;;
    baseline_prr)
        echo "[Legacy Alias] baseline_prr -> baseline (PRR-v3 structure only)"
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device ${DEVICE} --batch 6 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode baseline
        ;;
    ab)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device ${DEVICE} --batch 6 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode ab --debug_routing
        ;;
    prr_ab)
        echo "[Legacy Alias] prr_ab -> ab (PRR-v3 + A+B)"
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device ${DEVICE} --batch 6 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode ab --debug_routing
        ;;
    full)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device ${DEVICE} --batch 6 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode full --enable_ssds --debug_routing
        ;;
    *)
        echo "未知模式: ${TRAINER_MODE}"
        echo "可用: baseline | ab | full"
        exit 1
        ;;
esac

echo "训练完成，退出码: $?"
