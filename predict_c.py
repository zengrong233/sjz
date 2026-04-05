from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import sys
import os
import tkinter as tk
from tkinter import filedialog

def choose_input_source():
    print("请选择输入来源：")
    print("[1] 摄像头")
    print("[2] 视频文件")
    choice = input("请输入数字 (1 或 2): ").strip()

    if choice == "1":
        return 0, "摄像头"
    elif choice == "2":
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mkv;*.mov"), ("所有文件", "*.*")]
        )
        if not video_path:
            print("未选择视频文件，程序退出")
            sys.exit(0)
        return video_path, video_path
    else:
        print("无效的输入，程序退出")
        sys.exit(1)

def detect_media():
    # ======================= 配置区 =======================
    model_config = {
        'model_path': r'C:\Users\86155\Desktop\ultralyticsPro--YOLO11\3c_best.pt',  # 本地模型路径
    }

    predict_config = {
        'conf_thres': 0.25,  # 置信度阈值
        'iou_thres': 0.45,   # IoU阈值
        'imgsz': 640,        # 输入分辨率
        'line_width': 2,     # 检测框线宽
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    # ====================== 配置结束 ======================

    try:
        # 选择输入来源
        input_source, source_desc = choose_input_source()

        # 初始化视频源
        cap = cv2.VideoCapture(input_source)
        if isinstance(input_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise IOError(f"无法打开视频源 ({source_desc})，请检查设备连接或文件路径。")

        # 询问是否保存推理出的视频文件
        save_video = False
        video_writer = None
        output_path = None
        answer = input("是否保存推理出的视频文件？(y/n): ").strip().lower()
        if answer == "y":
            save_video = True
            save_dir = os.path.join(os.getcwd(), "predict")
            os.makedirs(save_dir, exist_ok=True)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 25
            output_path = os.path.join(save_dir, "output_inference.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"推理视频将保存至: {output_path}")

        # 检查模型文件是否存在
        if not Path(model_config['model_path']).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_config['model_path']}")

        # 加载模型
        model = YOLO(model_config['model_path']).to(predict_config['device'])
        print(f"✅ 模型加载成功 | 设备: {predict_config['device'].upper()}")
        print(f"输入来源: {source_desc}")

        # 实时检测循环
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或中断")
                break

            results = model.predict(
                source=frame,
                stream=True,
                verbose=False,
                conf=predict_config['conf_thres'],
                iou=predict_config['iou_thres'],
                imgsz=predict_config['imgsz'],
                device=predict_config['device']
            )

            for result in results:
                annotated_frame = result.plot(line_width=predict_config['line_width'])
                break

            if isinstance(input_source, int):
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('YOLO Real-time Detection', annotated_frame)

            if save_video and video_writer is not None:
                video_writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("✅ 检测结束")
        if save_video and output_path is not None:
            print(f"推理结果视频已保存至: {output_path}")

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("问题排查建议：")
        print("1. 检查视频源是否正确连接或文件路径是否正确")
        print("2. 确认模型文件路径正确")
        print("3. 检查CUDA是否可用（如需GPU加速）")
        print("4. 尝试降低分辨率设置")

if __name__ == "__main__":
    detect_media()
