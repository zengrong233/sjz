#!/bin/bash
#SBATCH -J aitod_p1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.out

# AI-TOD 训练脚本（P1 主线版，建议放在 4.20 使用）
# 默认口径:
#   - 结构: YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml
#   - 模式: full
#   - 数据: data_AI-TOD_server2.yaml
#   - 初始权重: yolo11n.pt
#
# 用法:
#   PROJECT_DIR=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11 sbatch train_aitod_prrv3_ssa_p1_slurm.sh
#
# 可选环境变量覆盖:
#   PROJECT_DIR=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11
#   CONDA_ENV=yolov11_py310
#   CFG_FILE=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml
#   DATA_YAML=data_AI-TOD_server2.yaml
#   WEIGHTS_PATH=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11/best_pt/yolo11n.pt
#   TRAINER_MODE=full
#   PROJECT_OUT=runs/aitod_p1
#   RUN_NAME=aitod_p1_${SLURM_JOB_ID}
#   CLEAR_CACHE=1
#   BATCH_SIZE=6
#   IMG_SIZE=640
#   EPOCHS=300
#   WORKERS=8
#   AMP=1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/u2415363072/4.20/ultralyticsPro--YOLO11}}"
CONDA_ENV="${CONDA_ENV:-yolov11_py310}"
CFG_FILE="${CFG_FILE:-YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1.yaml}"
DATA_YAML="${DATA_YAML:-data_AI-TOD_server2.yaml}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${PROJECT_DIR}/best_pt/yolo11n.pt}"
TRAINER_MODE="${TRAINER_MODE:-full}"
PROJECT_OUT="${PROJECT_OUT:-runs/aitod_p1}"
RUN_NAME="${RUN_NAME:-aitod_p1_${SLURM_JOB_ID}}"
CLEAR_CACHE="${CLEAR_CACHE:-1}"
BATCH_SIZE="${BATCH_SIZE:-6}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-300}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-1}"

if [ ! -f "${PROJECT_DIR}/train_yolo111 copy.py" ]; then
  echo "错误: PROJECT_DIR 无效，未找到训练入口: ${PROJECT_DIR}/train_yolo111 copy.py"
  exit 2
fi

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [ "${AMP}" = "1" ]; then
  AMP_ARGS=(--amp)
else
  AMP_ARGS=()
fi

mkdir -p "${PROJECT_DIR}/logs"

if [ "${CLEAR_CACHE}" = "1" ]; then
  rm -f "${PROJECT_DIR}/datasets/AI-TOD/labels/train.cache" \
        "${PROJECT_DIR}/datasets/AI-TOD/labels/val.cache" \
        "${PROJECT_DIR}/datasets/AI-TOD/labels/test.cache"
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
echo "batch/imgsz/amp: ${BATCH_SIZE}/${IMG_SIZE}/${AMP}"
echo "============================================"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')"
echo "============================================"

cd "${PROJECT_DIR}"

# AMP 自检内部会查找项目根目录下的 yolo11n.pt，不存在时会触发下载。
# 这里统一软链接到本地权重，避免作业依赖外网。
if [ -f "${WEIGHTS_PATH}" ] && [ ! -f "${PROJECT_DIR}/yolo11n.pt" ]; then
  ln -sf "${WEIGHTS_PATH}" "${PROJECT_DIR}/yolo11n.pt"
fi

python "train_yolo111 copy.py" \
  --cfg "${CFG_FILE}" \
  --weights "${WEIGHTS_PATH}" \
  --data "${DATA_YAML}" \
  --device 0 \
  --batch "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMG_SIZE}" \
  --workers "${WORKERS}" \
  --project "${PROJECT_OUT}" \
  --name "${RUN_NAME}" \
  --trainer_mode "${TRAINER_MODE}" \
  --enable_ssds \
  --debug_routing \
  "${AMP_ARGS[@]}"

EXIT_CODE=$?
echo "训练完成，退出码: ${EXIT_CODE}"
exit "${EXIT_CODE}"
