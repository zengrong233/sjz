#!/bin/bash
# Local runner
# Usage: ./train_local.sh [mode] [dataset]
# mode: baseline | ab | full
# dataset: visdrone | uavdt | tinyperson
# legacy aliases: baseline_prr -> baseline, prr_ab -> ab

set -euo pipefail

# --- project and env ---
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-yolov11}"
TRAINER_MODE="${1:-full}"
DATASET="${2:-${DATASET:-visdrone}}"
DEVICE="${DEVICE:-0}"

resolve_data_yaml() {
  local dataset="$1"
  local platform="windows"
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux*|Darwin*) platform="linux" ;;
  esac

  case "$dataset" in
    visdrone|vd)
      [ "$platform" = "linux" ] && echo "data_VD_slurm.yaml" || echo "data_VD.yaml"
      ;;
    uavdt)
      [ "$platform" = "linux" ] && echo "data_UAVDT_slurm.yaml" || echo "data_UAVDT.yaml"
      ;;
    tinyperson|tiny)
      [ "$platform" = "linux" ] && echo "data_TinyPerson_slurm.yaml" || echo "data_TinyPerson.yaml"
      ;;
    *) return 1 ;;
  esac
}

if [ -z "${DATA_YAML:-}" ]; then
  DATA_YAML="$(resolve_data_yaml "${DATASET}")" || {
    echo "未知数据集: ${DATASET}"
    echo "可用: visdrone | uavdt | tinyperson"
    exit 1
  }
fi

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
echo "Dataset: ${DATASET}"
echo "Data yaml: ${DATA_YAML}"
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
