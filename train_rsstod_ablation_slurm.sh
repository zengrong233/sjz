#!/bin/bash
#SBATCH -J rsstod_abl
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.out

# RS-STOD 最小消融入口
# 用法:
#   sbatch train_rsstod_ablation_slurm.sh baseline
#   sbatch train_rsstod_ablation_slurm.sh ab
#   sbatch train_rsstod_ablation_slurm.sh full

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-yolov11_py310}"
TRAINER_MODE="${1:-full}"
CFG_FILE="${CFG_FILE:-YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml}"
DATA_YAML="${DATA_YAML:-data_RS_STOD_server2.yaml}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/share/home/u2415363072/4.6/ultralyticsPro--YOLO11/best_pt/rsstod_best.pt}"

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/logs"

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "模式: ${TRAINER_MODE}"
echo "项目目录: ${PROJECT_DIR}"
echo "模型配置: ${CFG_FILE}"
echo "数据 YAML: ${DATA_YAML}"
echo "初始权重: ${WEIGHTS_PATH}"
echo "============================================"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')"
echo "============================================"

cd "${PROJECT_DIR}"

case "${TRAINER_MODE}" in
    baseline)
        python "train_yolo111 copy.py" \
            --cfg "${CFG_FILE}" \
            --weights "${WEIGHTS_PATH}" \
            --data "${DATA_YAML}" \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode baseline
        ;;
    ab)
        python "train_yolo111 copy.py" \
            --cfg "${CFG_FILE}" \
            --weights "${WEIGHTS_PATH}" \
            --data "${DATA_YAML}" \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode ab --debug_routing
        ;;
    full)
        python "train_yolo111 copy.py" \
            --cfg "${CFG_FILE}" \
            --weights "${WEIGHTS_PATH}" \
            --data "${DATA_YAML}" \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode full --enable_ssds --debug_routing
        ;;
    *)
        echo "未知模式: ${TRAINER_MODE}"
        echo "可用: baseline | ab | full"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "训练完成，退出码: ${EXIT_CODE}"
exit "${EXIT_CODE}"
