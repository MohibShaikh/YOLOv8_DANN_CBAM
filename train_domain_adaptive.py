from ultralytics import YOLO
from model import DomainAdaptiveYOLOv8
import torch
import argparse
from pathlib import Path
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-data', type=str, required=True, help='Path to source domain dataset yaml')
    parser.add_argument('--target-data', type=str, required=True, help='Path to target domain dataset yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt', help='Pretrained YOLOv8n weights')
    return parser.parse_args()

def main(args):
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base YOLOv8n model
    model = YOLO(args.pretrained)
    
    # Replace with our domain-adaptive model
    model.model = DomainAdaptiveYOLOv8(
        cfg=model.model.cfg,
        ch=3,
        nc=model.model.nc,
        task='detect'
    )
    
    # Load pretrained weights into our model
    model.model.load_state_dict(torch.load(args.pretrained)['model'].state_dict(), strict=False)
    
    # Training configuration
    train_args = {
        'data': args.source_data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': args.device,
        'workers': args.workers,
        'project': str(save_dir),
        'name': 'domain_adaptive',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'close_mosaic': 10,
        'target_data': args.target_data,  # Custom argument for domain adaptation
    }
    
    # Train the model
    results = model.train(**train_args)
    
    # Save the final model
    model.save(save_dir / 'final.pt')

if __name__ == '__main__':
    args = parse_args()
    main(args) 