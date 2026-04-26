#!/bin/bash
#SBATCH -J usod_ssa_p1_p3fusion
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.out

# USOD 训练脚本（Phase B · H2 验证 · P1-P3Fusion）
# 重要：本脚本强制 project/name 隔离，禁止复用 runs/detect/train
#
# 默认口径:
#   - 结构: YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml
#   - 模式: full
#   - 数据: data_USOD_server2.yaml
#   - 初始权重: yolo11n.pt
#
# 用法:
#   PROJECT_DIR=/share/home/u2415363072/4.17/ultralyticsPro--YOLO11 \
#   sbatch train_usod_prrv3_ssa_p1_p3fusion_slurm.sh
#
# 可选环境变量覆盖:
#   PROJECT_DIR=/share/home/u2415363072/4.17/ultralyticsPro--YOLO11
#   CONDA_ENV=yolov11_py310
#   CFG_FILE=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml
#   DATA_YAML=data_USOD_server2.yaml
#   WEIGHTS_PATH=yolo11n.pt
#   TRAINER_MODE=full
#   PROJECT_OUT=runs/usod_p1_p3fusion
#   RUN_NAME=usod_p1_p3fusion_${SLURM_JOB_ID}
#   CLEAR_CACHE=1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/u2415363072/4.17/ultralyticsPro--YOLO11}}"
CONDA_ENV="${CONDA_ENV:-yolov11_py310}"
CFG_FILE="${CFG_FILE:-YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-P3Fusion.yaml}"
DATA_YAML="${DATA_YAML:-data_USOD_server2.yaml}"
WEIGHTS_PATH="${WEIGHTS_PATH:-yolo11n.pt}"
TRAINER_MODE="${TRAINER_MODE:-full}"
PROJECT_OUT="${PROJECT_OUT:-runs/usod_p1_p3fusion}"
RUN_NAME="${RUN_NAME:-usod_p1_p3fusion_${SLURM_JOB_ID}}"
CLEAR_CACHE="${CLEAR_CACHE:-1}"

if [ ! -f "${PROJECT_DIR}/train_yolo111 copy.py" ]; then
  echo "错误: PROJECT_DIR 无效，未找到训练入口: ${PROJECT_DIR}/train_yolo111 copy.py"
  echo "请通过环境变量显式传入 PROJECT_DIR，例如："
  echo "PROJECT_DIR=/share/home/u2415363072/4.17/ultralyticsPro--YOLO11 sbatch train_usod_prrv3_ssa_p1_p3fusion_slurm.sh"
  exit 2
fi

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/logs"

if [ "${CLEAR_CACHE}" = "1" ]; then
  rm -f "${PROJECT_DIR}/datasets/USOD/labels/train.cache" \
        "${PROJECT_DIR}/datasets/USOD/labels/val.cache" \
        "${PROJECT_DIR}/datasets/USOD/labels/test.cache"
fi

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "项目目录: ${PROJECT_DIR}"
echo "训练模式: ${TRAINER_MODE}"
echo "模型配置: ${CFG_FILE}"
echo "数据 YAML: ${DATA_YAML}"
echo "初始权重: ${WEIGHTS_PATH}"
echo "输出目录: ${PROJECT_OUT}/${RUN_NAME}"
echo "============================================"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')"
echo "============================================"

cd "${PROJECT_DIR}"

python "train_yolo111 copy.py" \
  --cfg "${CFG_FILE}" \
  --weights "${WEIGHTS_PATH}" \
  --data "${DATA_YAML}" \
  --device 0 \
  --batch 4 \
  --epochs 300 \
  --imgsz 640 \
  --workers 8 \
  --project "${PROJECT_OUT}" \
  --name "${RUN_NAME}" \
  --trainer_mode "${TRAINER_MODE}" \
  --enable_ssds \
  --debug_routing

EXIT_CODE=$?
echo "训练完成，退出码: ${EXIT_CODE}"
exit "${EXIT_CODE}"
