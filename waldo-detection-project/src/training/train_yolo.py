"""
YOLOv8 Training Script
======================
Train YOLOv8 model for Waldo detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_waldo_detector(
    data_yaml: str,
    model_size: str = "s",
    epochs: int = 40,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "waldo_detector",
    pretrained: bool = True,
    augment: bool = True
):
    """
    Train YOLOv8 model for Waldo detection.
    
    Args:
        data_yaml: Path to data.yaml configuration
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to use ('0', '1', 'cpu', etc.)
        project: Project directory for runs
        name: Name of this training run
        pretrained: Use pretrained weights
        augment: Enable data augmentation
    """
    # Select model
    if pretrained:
        model_name = f"yolov8{model_size}.pt"
        print(f"📦 Using pretrained model: {model_name}")
    else:
        model_name = f"yolov8{model_size}.yaml"
        print(f"🔧 Training from scratch with: {model_name}")
    
    model = YOLO(model_name)
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print(f"   Device: {device}")
    print(f"   Augmentation: {augment}\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        augment=augment,
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n✅ Training complete!")
    print(f"📊 Results saved to: {project}/{name}")
    print(f"🎯 Best model: {project}/{name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Waldo detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to data.yaml (default: data.yaml)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="s",
        choices=['n', 's', 'm', 'l', 'x'],
        help="Model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: s)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs (default: 40)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: GPU id (0, 1, ...) or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory (default: runs/train)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="waldo_detector",
        help="Run name (default: waldo_detector)"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (no pretrained weights)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    
    args = parser.parse_args()
    
    # Train
    train_waldo_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
