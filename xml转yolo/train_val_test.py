#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO格式数据集划分脚本
将images和labels按照7:2:1的比例划分为训练集、验证集和测试集
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse

def get_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    获取图像和标签文件的配对列表
    
    Args:
        images_dir: 图像文件夹路径
        labels_dir: 标签文件夹路径
    
    Returns:
        配对的图像和标签文件路径列表
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    # 创建图像和标签的配对
    pairs = []
    missing_labels = []
    
    for image_path in image_files:
        # 构造对应的标签文件路径
        label_name = image_path.stem + '.txt'
        label_path = labels_dir / label_name
        
        if label_path.exists():
            pairs.append((str(image_path), str(label_path)))
        else:
            missing_labels.append(str(image_path))
    
    print(f"✅ 找到 {len(pairs)} 对有效的图像-标签配对")
    if missing_labels:
        print(f"⚠️  发现 {len(missing_labels)} 个图像文件缺少对应的标签文件")
        print("前5个缺少标签的图像:")
        for img in missing_labels[:5]:
            print(f"   - {img}")
    
    return pairs

def create_directory_structure(output_dir: str) -> dict:
    """
    创建数据集目录结构
    
    Args:
        output_dir: 输出根目录
    
    Returns:
        包含各个子目录路径的字典
    """
    output_path = Path(output_dir)
    
    # 创建目录结构
    dirs = {
        'train_images': output_path / 'train' / 'images',
        'train_labels': output_path / 'train' / 'labels',
        'val_images': output_path / 'val' / 'images',
        'val_labels': output_path / 'val' / 'labels',
        'test_images': output_path / 'test' / 'images',
        'test_labels': output_path / 'test' / 'labels'
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    return dirs

def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, test_ratio: float = 0.1) -> dict:
    """
    按比例划分数据集
    
    Args:
        pairs: 图像-标签配对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        包含各个数据集的字典
    """
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1.0，当前为{total_ratio}")
    
    # 随机打乱数据
    random.shuffle(pairs)
    
    total_count = len(pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # 剩余的都给测试集
    
    # 划分数据集
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]
    
    print(f"📊 数据集划分结果:")
    print(f"   训练集: {len(train_pairs)} 对 ({len(train_pairs)/total_count*100:.1f}%)")
    print(f"   验证集: {len(val_pairs)} 对 ({len(val_pairs)/total_count*100:.1f}%)")
    print(f"   测试集: {len(test_pairs)} 对 ({len(test_pairs)/total_count*100:.1f}%)")
    print(f"   总计: {total_count} 对")
    
    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

def copy_files(dataset_splits: dict, dirs: dict, copy_mode: str = 'copy'):
    """
    复制或移动文件到对应目录
    
    Args:
        dataset_splits: 数据集划分结果
        dirs: 目录路径字典
        copy_mode: 'copy' 或 'move'
    """
    operation = shutil.copy2 if copy_mode == 'copy' else shutil.move
    operation_name = "复制" if copy_mode == 'copy' else "移动"
    
    for split_name, pairs in dataset_splits.items():
        print(f"\n🚀 开始{operation_name}{split_name}集文件...")
        
        images_dir = dirs[f'{split_name}_images']
        labels_dir = dirs[f'{split_name}_labels']
        
        success_count = 0
        error_count = 0
        
        for i, (image_path, label_path) in enumerate(pairs):
            try:
                # 复制/移动图像文件
                image_dest = images_dir / Path(image_path).name
                operation(image_path, image_dest)
                
                # 复制/移动标签文件
                label_dest = labels_dir / Path(label_path).name
                operation(label_path, label_dest)
                
                success_count += 1
                
                # 显示进度
                if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                    print(f"   进度: {i + 1}/{len(pairs)} ({(i + 1)/len(pairs)*100:.1f}%)")
                    
            except Exception as e:
                print(f"   ❌ 处理文件失败: {Path(image_path).name} - {str(e)}")
                error_count += 1
        
        print(f"   ✅ {split_name}集处理完成: 成功{success_count}对，失败{error_count}对")

def create_yaml_config(output_dir: str, class_names: List[str] = None):
    """
    创建YOLO训练配置文件
    
    Args:
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    output_path = Path(output_dir)
    
    # 默认类别名称（DIOR数据集）
    if class_names is None:
        class_names = [
            'airplane', 'airport', 'baseball_field', 'basketball_court', 'bridge',
            'chimney', 'dam', 'expressway_service_area', 'expressway_toll_station',
            'golf_course', 'ground_track_field', 'harbor', 'overpass', 'ship',
            'stadium', 'storage_tank', 'tennis_court', 'train_station', 'vehicle', 'windmill'
        ]
    
    # 创建YAML配置
    yaml_content = f"""# DIOR数据集配置文件
# 数据集路径配置
path: {output_path.absolute()}  # 数据集根目录
train: train/images  # 训练集图像路径（相对于path）
val: val/images      # 验证集图像路径（相对于path）
test: test/images    # 测试集图像路径（相对于path）

# 类别配置
nc: {len(class_names)}  # 类别数量
names: {class_names}  # 类别名称列表

# 数据集信息
dataset_info:
  name: "DIOR"
  description: "DIOR (Dataset for Object Detection in Aerial Images) 遥感目标检测数据集"
  total_images: "待统计"
  train_images: "待统计"
  val_images: "待统计"
  test_images: "待统计"
  
# 训练参数建议
training_params:
  epochs: 300
  batch_size: 16
  imgsz: 640
  lr0: 0.01
  weight_decay: 0.0005
  mosaic: 1.0
  mixup: 0.1
"""
    
    yaml_path = output_path / 'dataset_config.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"📄 创建配置文件: {yaml_path}")

def analyze_dataset(dirs: dict):
    """
    分析数据集统计信息
    
    Args:
        dirs: 目录路径字典
    """
    print("\n📈 数据集统计分析:")
    print("-" * 50)
    
    for split in ['train', 'val', 'test']:
        images_dir = dirs[f'{split}_images']
        labels_dir = dirs[f'{split}_labels']
        
        image_count = len(list(images_dir.glob('*')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"{split.upper()}集:")
        print(f"  图像数量: {image_count}")
        print(f"  标签数量: {label_count}")
        
        # 分析标签文件中的目标数量
        total_objects = 0
        if label_count > 0:
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        total_objects += len([line.strip() for line in lines if line.strip()])
                except:
                    continue
        
        print(f"  目标总数: {total_objects}")
        if image_count > 0:
            print(f"  平均每张图像目标数: {total_objects/image_count:.2f}")
        print()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO格式数据集划分工具')
    parser.add_argument('--images_dir', type=str, 
                       default=r'D:\BaiduNetdiskDownload\kdqs\DIOR\images',
                       help='图像文件夹路径')
    parser.add_argument('--labels_dir', type=str,
                       default=r'D:\BaiduNetdiskDownload\kdqs\DIOR\labels', 
                       help='标签文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\BaiduNetdiskDownload\kdqs\DIOR\dataset_split',
                       help='输出目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例 (默认: 0.1)')
    parser.add_argument('--copy_mode', type=str, choices=['copy', 'move'], 
                       default='copy', help='文件操作模式: copy(复制) 或 move(移动)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("🚀 YOLO格式数据集划分工具")
    print("=" * 60)
    print(f"图像目录: {args.images_dir}")
    print(f"标签目录: {args.labels_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例: 训练集{args.train_ratio} : 验证集{args.val_ratio} : 测试集{args.test_ratio}")
    print(f"操作模式: {args.copy_mode}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.images_dir):
        print(f"❌ 图像目录不存在: {args.images_dir}")
        return
    
    if not os.path.exists(args.labels_dir):
        print(f"❌ 标签目录不存在: {args.labels_dir}")
        return
    
    try:
        # 1. 获取图像-标签配对
        print("\n📂 扫描图像和标签文件...")
        pairs = get_image_label_pairs(args.images_dir, args.labels_dir)
        
        if len(pairs) == 0:
            print("❌ 没有找到有效的图像-标签配对，请检查文件路径和格式")
            return
        
        # 2. 创建目录结构
        print("\n📁 创建输出目录结构...")
        dirs = create_directory_structure(args.output_dir)
        
        # 3. 划分数据集
        print("\n🎯 划分数据集...")
        dataset_splits = split_dataset(pairs, args.train_ratio, args.val_ratio, args.test_ratio)
        
        # 4. 复制文件
        print(f"\n📋 开始{args.copy_mode}文件...")
        copy_files(dataset_splits, dirs, args.copy_mode)
        
        # 5. 创建配置文件
        print("\n📄 创建YAML配置文件...")
        create_yaml_config(args.output_dir)
        
        # 6. 分析数据集
        analyze_dataset(dirs)
        
        print("\n🎉 数据集划分完成!")
        print(f"📁 输出目录: {args.output_dir}")
        print("💡 使用建议:")
        print("   1. 检查生成的 dataset_config.yaml 文件")
        print("   2. 根据需要调整类别名称和训练参数")
        print("   3. 使用该配置文件进行YOLO模型训练")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()