import torch
from ultralytics import YOLO
import os

def predict_simple(model_path, test_images_path, save_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO(model_path)
    print("Model loaded.")

    if save_path is None:
        save_path = os.path.join(test_images_path, 'predictions')
    os.makedirs(save_path, exist_ok=True)

    image_files = [f for f in os.listdir(test_images_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for img_file in image_files:
        img_path = os.path.join(test_images_path, img_file)
        results = model.predict(source=img_path, conf=0.25, save=True, project=save_path, exist_ok=True)
        print(f"Predicted {img_file}")

    print(f"Predictions saved in {save_path}")

def main():
    weights_path = r'C:\Users\86155\Desktop\ultralyticsPro--YOLO11\3c_best_demo.pt'  
    test_images_path = r"C:\Users\86155\Desktop\ultralyticsPro--YOLO11\datasets\images\3c_test"
    predict_simple(weights_path, test_images_path)

if __name__ == '__main__':
    main()
