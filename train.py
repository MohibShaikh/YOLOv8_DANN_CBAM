import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.ops import non_max_suppression
from model import DomainAdaptiveYOLOv8
import yaml
import argparse
from pathlib import Path
import torch.distributed as dist
import os
import gc
import numpy as np
from tqdm import tqdm
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils import LOGGER, TQDM, colorstr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-data', type=str, required=True, help='Path to source domain dataset yaml')
    parser.add_argument('--target-data', type=str, required=True, help='Path to target domain dataset yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Number of images to process in memory at once')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--val-interval', type=int, default=5, help='Validation interval in epochs')
    return parser.parse_args()

def setup_distributed(args):
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        args.world_size = 1
        args.rank = 0

def validate(model, val_loader, args, device):
    model.eval()
    stats = []
    metrics = DetMetrics()
    
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader), bar_format=TQDM.bar_format)
    
    with torch.no_grad():
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(imgs)
            
            # NMS
            preds = non_max_suppression(preds, args.conf_thres, args.iou_thres)
            
            # Update metrics
            for i, pred in enumerate(preds):
                if pred is not None and len(pred):
                    # Convert predictions to [x1, y1, x2, y2, conf, cls] format
                    pred_boxes = pred[:, :4]
                    pred_scores = pred[:, 4]
                    pred_cls = pred[:, 5]
                    
                    # Get ground truth
                    gt_boxes = targets[targets[:, 0] == i, 2:6]
                    gt_cls = targets[targets[:, 0] == i, 1]
                    
                    # Calculate IoU
                    iou = box_iou(pred_boxes, gt_boxes)
                    
                    # Update metrics
                    metrics.update(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls)
    
    # Calculate final metrics
    stats = metrics.get_stats()
    
    # Print metrics
    if args.rank == 0:
        LOGGER.info('\nValidation Results:')
        LOGGER.info(f'mAP@0.5: {stats[0]:.4f}')
        LOGGER.info(f'mAP@0.5:0.95: {stats[1]:.4f}')
        LOGGER.info(f'Precision: {stats[2]:.4f}')
        LOGGER.info(f'Recall: {stats[3]:.4f}')
    
    return stats

def train(args):
    # Setup distributed training
    setup_distributed(args)
    
    # Initialize device
    device = select_device(args.device)
    
    # Initialize mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type != 'cpu' else None
    
    # Load dataset configurations
    with open(args.source_data) as f:
        source_cfg = yaml.safe_load(f)
    with open(args.target_data) as f:
        target_cfg = yaml.safe_load(f)
    
    # Build datasets
    source_dataset = build_yolo_dataset(source_cfg, args.img_size, args.batch_size, args.workers)
    target_dataset = build_yolo_dataset(target_cfg, args.img_size, args.batch_size, args.workers)
    val_dataset = build_yolo_dataset(source_cfg, args.img_size, args.batch_size, args.workers, split='val')
    
    if args.local_rank != -1:
        source_sampler = DistributedSampler(source_dataset)
        target_sampler = DistributedSampler(target_dataset)
        val_sampler = DistributedSampler(val_dataset)
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, sampler=source_sampler)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, sampler=target_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    else:
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = DomainAdaptiveYOLOv8(cfg='yolov8n.yaml', ch=3, nc=source_dataset.nc)
    model = model.to(device)
    
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    domain_optimizer = optim.Adam(model.domain_classifier.parameters(), lr=0.001)
    
    # Loss functions
    detection_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_map = 0.0
    for epoch in range(args.epochs):
        if args.local_rank != -1:
            source_sampler.set_epoch(epoch)
            target_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0
        total_domain_loss = 0
        
        # Calculate alpha for gradient reversal layer
        alpha = 2.0 / (1.0 + torch.exp(-10 * epoch / args.epochs)) - 1
        
        optimizer.zero_grad()
        domain_optimizer.zero_grad()
        
        pbar = enumerate(zip(source_loader, target_loader))
        pbar = tqdm(pbar, total=len(source_loader), bar_format=TQDM.bar_format)
        
        for batch_idx, ((source_imgs, source_targets), (target_imgs, _)) in pbar:
            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)
            source_targets = source_targets.to(device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast() if args.mixed_precision and device.type != 'cpu' else torch.no_grad():
                # Source domain forward pass
                source_detections, source_domain_pred = model(source_imgs, alpha, return_domain=True)
                
                # Target domain forward pass
                _, target_domain_pred = model(target_imgs, alpha, return_domain=True)
                
                # Detection loss
                detection_loss = detection_criterion(source_detections, source_targets)
                
                # Domain classification loss
                source_domain_labels = torch.ones(source_domain_pred.size(0), 1).to(device)
                target_domain_labels = torch.zeros(target_domain_pred.size(0), 1).to(device)
                
                domain_loss = (domain_criterion(source_domain_pred, source_domain_labels) +
                             domain_criterion(target_domain_pred, target_domain_labels)) / 2
                
                # Total loss
                loss = (detection_loss + domain_loss) / args.gradient_accumulation_steps
            
            # Mixed precision backward pass
            if args.mixed_precision and device.type != 'cpu':
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.step(domain_optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    domain_optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    domain_optimizer.step()
                    optimizer.zero_grad()
                    domain_optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_domain_loss += domain_loss.item()
            
            # Update progress bar
            pbar.set_description(f'Epoch {epoch+1}/{args.epochs} '
                               f'loss: {loss.item():.4f} '
                               f'det: {detection_loss.item():.4f} '
                               f'dom: {domain_loss.item():.4f}')
            
            # Clear memory
            if batch_idx % args.chunk_size == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Print epoch statistics
        if args.rank == 0:
            avg_loss = total_loss / len(source_loader)
            avg_domain_loss = total_domain_loss / len(source_loader)
            LOGGER.info(f'Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Domain Loss: {avg_domain_loss:.4f}')
            
            # Validation
            if (epoch + 1) % args.val_interval == 0:
                stats = validate(model, val_loader, args, device)
                current_map = stats[1]  # mAP@0.5:0.95
                
                # Save best model
                if current_map > best_map:
                    best_map = current_map
                    save_path = Path(args.save_dir) / 'best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if args.local_rank != -1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'domain_optimizer_state_dict': domain_optimizer.state_dict(),
                        'mAP': current_map,
                    }, save_path)
                    LOGGER.info(f'New best model saved with mAP: {current_map:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = Path(args.save_dir) / f'epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.local_rank != -1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'domain_optimizer_state_dict': domain_optimizer.state_dict(),
                    'loss': avg_loss,
                }, save_path)

if __name__ == '__main__':
    args = parse_args()
    train(args) 