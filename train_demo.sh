#!/bin/bash
#SBATCH -J YOLO11_4GPU_distributed    # 作业名称
#SBATCH -p gpu                        # 分区名称
#SBATCH -N 1                          # 使用1个节点
#SBATCH --ntasks-per-node=4           # 每个节点的进程数设置为4
#SBATCH --gres=gpu:4                  # 使用4个GPU
#SBATCH --cpus-per-task=4             # 每个任务分配4个CPU核心
#SBATCH --mem=64G                     # 每个节点分配64GB内存
#SBATCH --time=24:00:00               # 最大运行时间24小时

# 加载环境（根据实际超算环境修改）
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate yolov11  # 使用正确的conda环境名称

# 显示环境信息
echo "作业ID: $SLURM_JOB_ID"
echo "节点列表: $SLURM_JOB_NODELIST"
echo "GPU数量: $SLURM_GPUS"
echo "开始时间: $(date)"
nvidia-smi

# 设置最小日志输出环境变量
export ULTRALYTICS_VERBOSE=False
export TQDM_DISABLE=1
export WANDB_DISABLED=true
export COMET_DISABLE=1
export PYTHONUNBUFFERED=1
export ULTRALYTICS_SKIP_DOWNLOAD=1

# 清理可能存在的旧进程
echo "清理旧进程..."
pkill -f "torchrun" || true
pkill -f "train_distributed_torchrun.py" || true
sleep 2

# 查找可用端口
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "使用端口: $MASTER_PORT"

# 设置分布式训练环境变量
export MASTER_ADDR="localhost"
export MASTER_PORT=$MASTER_PORT

# 设置NCCL环境变量以提高4GPU分布式训练稳定性
export NCCL_TIMEOUT=18000          # 增加超时时间到5小时
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN             # 减少日志输出
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo       # 强制使用本地回环接口
export NCCL_BUFFSIZE=16777216      # 增加缓冲区大小
export NCCL_CHECKS_DISABLE=1       # 禁用一些检查以提高性能

echo "=== NCCL环境变量 ==="
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "NCCL_BLOCKING_WAIT: $NCCL_BLOCKING_WAIT"
echo "NCCL_ASYNC_ERROR_HANDLING: $NCCL_ASYNC_ERROR_HANDLING"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_BUFFSIZE: $NCCL_BUFFSIZE"

# 预先清理GPU缓存
echo "清理GPU缓存..."
nvidia-smi --gpu-reset

# 使用torchrun启动4GPU分布式训练
echo "使用torchrun启动YOLO11 4GPU分布式训练..."
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    --max_restarts=0 \
    /share/home/u2415363072/qzj/ultralyticsPro--YOLO11/train_distributed_torchrun.py \
    --batch 1 \
    --epochs 300 \
    --imgsz 640 \
    --workers 2 \
    --fraction 0.3 \
    --cache '' \
    --close-mosaic 10 \
    --cfg /share/home/u2415363072/qzj/ultralyticsPro--YOLO11/YOLO11-P2-Simple.yaml \
    --weights /share/home/u2415363072/qzj/ultralyticsPro--YOLO11/yolo11n.pt \
    --data /share/home/u2415363072/qzj/ultralyticsPro--YOLO11/data_AITOD.yaml \
    --project runs/distributed_train \
    --name yolo11_4gpu_fp32_exp

echo "训练完成时间: $(date)"

# 显示最终GPU状态
nvidia-smi

# 可选：发送完成通知邮件（如果超算支持）
# echo "YOLO11分布式训练完成" | mail -s "训练任务完成" your_email@example.com