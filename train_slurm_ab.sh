#!/bin/bash
#SBATCH -J yolo11_prr       # 作业名称
#SBATCH -p gpu              # 队列: gpu
#SBATCH -N 1                # 节点数
#SBATCH -n 1                # 进程数
#SBATCH --gres=gpu:1        # GPU 卡数
#SBATCH -o logs/slurm_%j.out # 标准输出+错误合并到一个文件
#SBATCH -e logs/slurm_%j.out # stderr 也写入同一文件

# ============================================================
# 超算训练脚本 - YOLO11 小目标检测 PRR + A+B + SSDS
# 用法:
#   sbatch train_slurm_ab.sh [mode] [dataset]
#   mode: baseline | ab | full
#   dataset: visdrone | uavdt | tinyperson
# 兼容旧别名:
#   baseline_prr -> baseline
#   prr_ab       -> ab
# ============================================================

# ---------- 路径配置 ----------
PROJECT_DIR="/share/home/u2415363072/3.30/ultralyticsPro--YOLO11"
CONDA_ENV="yolov11"

# ---------- 训练模式 ----------
TRAINER_MODE="${1:-full}"
DATASET="${2:-${DATASET:-visdrone}}"

resolve_data_yaml() {
  case "$1" in
    visdrone|vd) echo "data_VD_slurm.yaml" ;;
    uavdt) echo "data_UAVDT_slurm.yaml" ;;
    tinyperson|tiny) echo "data_TinyPerson_slurm.yaml" ;;
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

# ---------- 环境初始化 ----------
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export https_proxy=https://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/logs"

# ---------- 环境信息 ----------
echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "训练模式: ${TRAINER_MODE}"
echo "数据集: ${DATASET}"
echo "数据 YAML: ${DATA_YAML}"
echo "============================================"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')"
echo "============================================"

cd "${PROJECT_DIR}"

# ---------- 根据模式选择 YAML 和参数 ----------
YAML_PRR_V3="YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3.yaml"

case "${TRAINER_MODE}" in
    baseline)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode baseline
        ;;
    baseline_prr)
        echo "[Legacy Alias] baseline_prr -> baseline (PRR-v3 structure only)"
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode baseline
        ;;
    ab)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode ab --debug_routing
        ;;
    prr_ab)
        echo "[Legacy Alias] prr_ab -> ab (PRR-v3 + A+B)"
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode ab --debug_routing
        ;;
    full)
        python "train_yolo111 copy.py" \
            --cfg ${YAML_PRR_V3} \
            --weights yolo11n.pt \
            --data ${DATA_YAML} \
            --device 0 --batch 4 --epochs 300 --imgsz 640 --workers 8 \
            --trainer_mode full --enable_ssds --debug_routing
        ;;
    *)
        echo "未知模式: ${TRAINER_MODE}"
        echo "可用: baseline | ab | full"
        exit 1
        ;;
esac

echo "训练完成，退出码: $?"
