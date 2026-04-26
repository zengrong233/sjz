#!/bin/bash
#SBATCH -J usod_p1_nosac_c2f
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.out

# USOD 训练脚本（P1-NoSAC-FullC2f 变体，建议放在 4.20 使用）
# 默认口径:
#   - 结构: YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml
#   - 模式: full
#   - 数据: 运行时生成绝对路径 YAML，避免 Ultralytics settings.json 拼成 datasets/datasets
#   - 初始权重: best_pt/yolo11n.pt（新结构默认冷启动）
#
# 用法:
#   PROJECT_DIR=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11 \
#   sbatch train_usod_prrv3_ssa_p1_nosac_fullc2f_slurm.sh
#
# 可选环境变量覆盖:
#   PROJECT_DIR=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11
#   CONDA_ENV=yolov11_py310
#   CFG_FILE=YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml
#   DATA_ROOT=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11/datasets/USOD
#   DATA_YAML=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11/_runtime_data_USOD_420_abs.yaml
#   GENERATE_DATA_YAML=1
#   WEIGHTS_PATH=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11/best_pt/yolo11n.pt
#   TRAINER_MODE=full
#   PROJECT_OUT=runs/usod_p1_nosac_fullc2f
#   RUN_NAME=usod_p1_nosac_fullc2f_${SLURM_JOB_ID}
#   CLEAR_CACHE=1
#   BATCH_SIZE=4
#   IMG_SIZE=640
#   EPOCHS=300
#   WORKERS=8

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/u2415363072/4.20/ultralyticsPro--YOLO11}}"
CONDA_ENV="${CONDA_ENV:-yolov11_py310}"
CFG_FILE="${CFG_FILE:-YOLO11-HFAMPAN-AsDDet-NWD-SmallObject-PRR-v3-SSA-P1-NoSAC-FullC2f.yaml}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/datasets/USOD}"
DATA_YAML="${DATA_YAML:-${PROJECT_DIR}/_runtime_data_USOD_420_abs.yaml}"
GENERATE_DATA_YAML="${GENERATE_DATA_YAML:-1}"
WEIGHTS_PATH="${WEIGHTS_PATH:-${PROJECT_DIR}/best_pt/yolo11n.pt}"
TRAINER_MODE="${TRAINER_MODE:-full}"
PROJECT_OUT="${PROJECT_OUT:-runs/usod_p1_nosac_fullc2f}"
RUN_NAME="${RUN_NAME:-usod_p1_nosac_fullc2f_${SLURM_JOB_ID}}"
CLEAR_CACHE="${CLEAR_CACHE:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-300}"
WORKERS="${WORKERS:-8}"

if [ ! -f "${PROJECT_DIR}/train_yolo111 copy.py" ]; then
  echo "错误: PROJECT_DIR 无效，未找到训练入口: ${PROJECT_DIR}/train_yolo111 copy.py"
  echo "请通过环境变量显式传入 PROJECT_DIR，例如："
  echo "PROJECT_DIR=/share/home/u2415363072/4.20/ultralyticsPro--YOLO11 sbatch train_usod_prrv3_ssa_p1_nosac_fullc2f_slurm.sh"
  exit 2
fi

if [ ! -f "${PROJECT_DIR}/${CFG_FILE}" ] && [ ! -f "${CFG_FILE}" ]; then
  echo "错误: 未找到模型配置: ${CFG_FILE}"
  exit 2
fi

if [ ! -f "${WEIGHTS_PATH}" ] && [ ! -f "${PROJECT_DIR}/${WEIGHTS_PATH}" ]; then
  echo "错误: 未找到初始权重: ${WEIGHTS_PATH}"
  exit 2
fi

if [ ! -d "${DATA_ROOT}/images/train" ] || [ ! -d "${DATA_ROOT}/images/val" ]; then
  echo "错误: DATA_ROOT 无效，未找到 USOD 数据目录: ${DATA_ROOT}"
  exit 2
fi

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "${PROJECT_DIR}/logs"

if [ "${GENERATE_DATA_YAML}" = "1" ]; then
  cat > "${DATA_YAML}" <<EOF
# USOD dataset config - absolute path runtime yaml for 4.20

path: ${DATA_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: vehicle
EOF
fi

if [ ! -f "${DATA_YAML}" ]; then
  echo "错误: 未找到数据 YAML: ${DATA_YAML}"
  exit 2
fi

if [ "${CLEAR_CACHE}" = "1" ]; then
  rm -f "${DATA_ROOT}/labels/train.cache" \
        "${DATA_ROOT}/labels/val.cache" \
        "${DATA_ROOT}/labels/test.cache"
fi

echo "============================================"
echo "作业ID: ${SLURM_JOB_ID}"
echo "节点: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "项目目录: ${PROJECT_DIR}"
echo "训练模式: ${TRAINER_MODE}"
echo "模型配置: ${CFG_FILE}"
echo "数据根目录: ${DATA_ROOT}"
echo "数据 YAML: ${DATA_YAML}"
echo "初始权重: ${WEIGHTS_PATH}"
echo "输出目录: ${PROJECT_OUT}/${RUN_NAME}"
echo "batch/imgsz: ${BATCH_SIZE}/${IMG_SIZE}"
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
  --batch "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMG_SIZE}" \
  --workers "${WORKERS}" \
  --project "${PROJECT_OUT}" \
  --name "${RUN_NAME}" \
  --trainer_mode "${TRAINER_MODE}" \
  --enable_ssds \
  --debug_routing

EXIT_CODE=$?
echo "训练完成，退出码: ${EXIT_CODE}"
exit "${EXIT_CODE}"
