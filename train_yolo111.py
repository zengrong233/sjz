import sys
import argparse
import os
import torch
import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(opt):
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        device = 'cpu'
    else:
        device = opt.device
        print(f"使用设备: {device}")
        print(f"可用的GPU数量: {torch.cuda.device_count()}")
    
    yaml = opt.cfg 
    weights = opt.weights
    data = opt.data
    
    # 初始化模型
    model = YOLO(yaml, task='detect')  # 明确指定任务类型
    if weights and os.path.exists(weights):
        model = model.load(weights)
    
    model.info()

    # 训练模型
    results = model.train(
        data=data,
        epochs=300, 
        imgsz=640, 
        workers=8, 
        batch=32,
        device=device,  # 使用检测到的设备
        exist_ok=True,  # 允许覆盖已存在的实验目录
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                       type=str, 
                       default=os.path.join(PROJECT_DIR, 'YOLO11-P2-Simple.yaml'),
                       help='model yaml path')
    parser.add_argument('--weights', 
                       type=str, 
                       default=os.path.join(PROJECT_DIR, 'yolo11n.pt'),
                       help='initial weights path')
    parser.add_argument('--data',
                       type=str,
                       default=os.path.join(PROJECT_DIR, 'data_VD.yaml'),
                       help='dataset yaml path')
    parser.add_argument('--device', 
                       type=str,
                       default='0',  # 默认使用第一个GPU
                       help='cuda device (e.g. 0 or 0,1,2,3 or cpu)')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    # 设置环境变量
    os.environ['ULTRALYTICS_SKIP_DOWNLOAD'] = '1'  # 禁用自动下载
    
    opt = parse_opt()
    main(opt)
