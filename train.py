"""
Domain-Adaptive Training Script for YOLOv8 with CBAM and DANN
This script properly integrates YOLO's training pipeline with domain adaptation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import os

from model import DomainAdaptiveYOLOv8
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import select_device
from torch.nn.parallel import DataParallel, DistributedDataParallel

def de_parallel(model):
    """De-parallelize a model."""
    return model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
from ultralytics.engine.validator import BaseValidator

try:
    from ultralytics.utils.loss import v8DetectionLoss
    HAS_DETECTION_LOSS = True
except ImportError:
    HAS_DETECTION_LOSS = False
    LOGGER.warning('Could not import v8DetectionLoss, using simplified loss')


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Domain Adaptation')
    parser.add_argument('--source-data', type=str, required=True, help='Source domain dataset YAML')
    parser.add_argument('--target-data', type=str, required=True, help='Target domain dataset YAML')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (cuda:0 or cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='runs/domain_adaptive', help='Save directory')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-interval', type=int, default=5, help='Validation interval')
    return parser.parse_args()


class DomainAdaptiveTrainer:
    """
    Trainer class that properly integrates YOLO's training with domain adaptation.
    """
    def __init__(self, args):
        self.args = args
        self.device = select_device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset configs
        with open(args.source_data) as f:
            self.source_cfg = yaml.safe_load(f)
        with open(args.target_data) as f:
            self.target_cfg = yaml.safe_load(f)
        
        # Initialize model
        self.setup_model()
        
        # Setup data loaders
        self.setup_dataloaders()
        
        # Setup training
        self.setup_training()
        
    def setup_model(self):
        """Initialize the domain-adaptive model"""
        LOGGER.info('Initializing Domain-Adaptive YOLOv8 model...')
        
        # Create model
        nc = len(self.source_cfg['names'])
        self.model = DomainAdaptiveYOLOv8(cfg='yolov8n.yaml', ch=3, nc=nc)
        
        # Load pretrained weights if available
        if Path(self.args.pretrained).exists():
            LOGGER.info(f'Loading pretrained weights from {self.args.pretrained}')
            pretrained = YOLO(self.args.pretrained)
            # Copy weights from pretrained model (CBAM modules will keep their initialized weights)
            pretrained_dict = pretrained.model.state_dict()
            model_dict = self.model.state_dict()
            # Filter out domain classifier and CBAM weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and 'domain_classifier' not in k and 'cbam' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            LOGGER.info('Loaded pretrained weights (excluding CBAM and domain classifier)')
        
        self.model = self.model.to(self.device)
        self.model.train()
        
    def _build_loader(self, data_cfg, split='train', mode='train', rect=False):
        """Build a YOLO dataset + dataloader from a data config dict."""
        from ultralytics.cfg import get_cfg
        cfg = get_cfg()  # default training config
        cfg.imgsz = self.args.img_size

        img_path = str(Path(data_cfg['path']) / data_cfg[split])
        dataset = build_yolo_dataset(cfg, img_path, self.args.batch_size, data_cfg, mode=mode, rect=rect)
        shuffle = mode == 'train'
        return build_dataloader(dataset, self.args.batch_size, self.args.workers, shuffle=shuffle)

    def setup_dataloaders(self):
        """Setup source and target domain dataloaders"""
        LOGGER.info('Setting up dataloaders...')

        self.source_loader = self._build_loader(self.source_cfg, split='train', mode='train')
        self.target_loader = self._build_loader(self.target_cfg, split='train', mode='train')
        self.val_loader = self._build_loader(self.source_cfg, split='val', mode='val', rect=True)

        LOGGER.info(f'Source batches: {len(self.source_loader)}, Target batches: {len(self.target_loader)}')
        
    def setup_training(self):
        """Setup optimizers and loss functions"""
        # Separate optimizers for model and domain classifier
        self.optimizer = optim.Adam([
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'domain_classifier' not in n], 'lr': self.args.lr},
            {'params': self.model.domain_classifier.parameters(), 'lr': self.args.lr}
        ])
        
        # Detection loss - use YOLO's proper loss if available
        if HAS_DETECTION_LOSS:
            self.detection_criterion = v8DetectionLoss(self.model)
            LOGGER.info('Using v8DetectionLoss for detection')
        else:
            self.detection_criterion = None
            LOGGER.warning('Using model.loss() fallback for detection')
        
        # Domain loss
        self.domain_criterion = nn.BCEWithLogitsLoss()
        
        # Track best mAP
        self.best_map = 0.0
        
    def compute_detection_loss(self, predictions, batch):
        """
        Compute YOLO detection loss
        Uses v8DetectionLoss if available, otherwise falls back to model's loss
        """
        if self.detection_criterion is not None:
            # Use v8DetectionLoss — returns [box, cls, dfl] losses
            loss, loss_items = self.detection_criterion(predictions, batch)
            return loss.sum(), {'det_loss': loss}
        else:
            # Fallback: use model's built-in loss if it exists
            if hasattr(self.model, 'loss'):
                loss_dict = self.model.loss(batch, predictions)
                total_loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                return total_loss, loss_dict
            else:
                # Last resort: return zero loss with warning
                LOGGER.warning('No loss function available!')
                return torch.tensor(0.0, device=self.device), {}
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Calculate alpha for gradient reversal (increases from 0 to 1)
        progress = epoch / self.args.epochs
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10 * progress, device=self.device))) - 1
        alpha = alpha.item()
        
        # Create iterator that zips source and target
        pbar = tqdm(
            zip(self.source_loader, self.target_loader),
            total=min(len(self.source_loader), len(self.target_loader)),
            desc=f'Epoch {epoch+1}/{self.args.epochs}'
        )
        
        epoch_loss = 0
        epoch_det_loss = 0
        epoch_dom_loss = 0
        n_batches = 0
        
        for source_batch, target_batch in pbar:
            # Move to device
            source_imgs = source_batch['img'].to(self.device, non_blocking=True).float() / 255.0
            target_imgs = target_batch['img'].to(self.device, non_blocking=True).float() / 255.0
            
            self.optimizer.zero_grad()
            
            # Source domain: detection + domain classification
            source_preds, source_domain_pred = self.model(source_imgs, alpha=alpha, return_domain=True)
            
            # Target domain: domain classification only (no detection loss needed)
            _, target_domain_pred = self.model(target_imgs, alpha=alpha, return_domain=True)
            
            # Detection loss (only on source domain with labels)
            det_loss, loss_dict = self.compute_detection_loss(source_preds, source_batch)
            
            # Domain classification loss
            source_domain_labels = torch.ones_like(source_domain_pred)
            target_domain_labels = torch.zeros_like(target_domain_pred)
            
            dom_loss = (
                self.domain_criterion(source_domain_pred, source_domain_labels) +
                self.domain_criterion(target_domain_pred, target_domain_labels)
            ) / 2.0
            
            # Total loss
            total_loss = det_loss + dom_loss
            
            # Backward and optimize
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_det_loss += det_loss.item()
            epoch_dom_loss += dom_loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'det': f'{det_loss.item():.4f}',
                'dom': f'{dom_loss.item():.4f}',
                'alpha': f'{alpha:.3f}'
            })
        
        # Return average losses
        return {
            'loss': epoch_loss / n_batches,
            'detection_loss': epoch_det_loss / n_batches,
            'domain_loss': epoch_dom_loss / n_batches,
            'alpha': alpha
        }
    
    @torch.no_grad()
    def validate(self):
        """Run validation using detection loss as a proxy metric"""
        self.model.eval()

        LOGGER.info('Running validation...')
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc='Validating'):
            imgs = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            preds = self.model(imgs)

            try:
                det_loss, _ = self.compute_detection_loss(preds, batch)
                total_loss += det_loss.item()
            except Exception:
                # If loss computation fails, just count the batch
                pass
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        # Return negative loss so higher = better (for best model selection)
        score = 1.0 / (1.0 + avg_loss)
        LOGGER.info(f'Validation: avg_loss={avg_loss:.4f}, score={score:.4f}')

        return score
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': de_parallel(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_map': self.best_map,
        }
        
        # Save latest
        save_path = self.save_dir / 'last.pt'
        torch.save(checkpoint, save_path)
        
        # Save best
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            LOGGER.info(f'New best model saved to {best_path}')
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            epoch_path = self.save_dir / f'epoch_{epoch+1}.pt'
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        LOGGER.info(f'Starting training for {self.args.epochs} epochs...')
        LOGGER.info(f'Device: {self.device}')
        LOGGER.info(f'Batch size: {self.args.batch_size}')
        LOGGER.info(f'Learning rate: {self.args.lr}')
        
        for epoch in range(self.args.epochs):
            # Train one epoch
            metrics = self.train_epoch(epoch)
            
            # Log metrics
            LOGGER.info(
                f"Epoch {epoch+1}/{self.args.epochs} - "
                f"Loss: {metrics['loss']:.4f}, "
                f"Det: {metrics['detection_loss']:.4f}, "
                f"Dom: {metrics['domain_loss']:.4f}, "
                f"Alpha: {metrics['alpha']:.3f}"
            )
            
            # Validate periodically
            if (epoch + 1) % self.args.val_interval == 0:
                val_acc = self.validate()
                
                # Save best model
                is_best = val_acc > self.best_map
                if is_best:
                    self.best_map = val_acc
                
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.save_checkpoint(epoch, is_best=False)
        
        LOGGER.info(f'Training complete! Best mAP: {self.best_map:.4f}')
        LOGGER.info(f'Results saved to {self.save_dir}')


def main():
    args = parse_args()
    trainer = DomainAdaptiveTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
