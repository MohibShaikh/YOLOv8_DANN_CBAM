from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import RANK, TQDM
from ultralytics.utils.torch_utils import torch_distributed_zero_first
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm

class DomainAdaptiveTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_data = kwargs.get('target_data')
        self.domain_criterion = nn.BCEWithLogitsLoss()
        
    def get_dataset(self, data, mode='train'):
        """Get dataset for source or target domain"""
        if mode == 'train':
            return super().get_dataset(data, mode)
        else:
            return super().get_dataset(self.target_data, mode)
    
    def get_dataloader(self, dataset, batch_size, mode='train'):
        """Get dataloader for source or target domain"""
        if mode == 'train':
            return super().get_dataloader(dataset, batch_size, mode)
        else:
            return super().get_dataloader(dataset, batch_size, 'val')
    
    def train_step(self, batch, nb):
        """Training step with domain adaptation"""
        # Get source and target batches
        source_batch = batch
        target_batch = next(iter(self.train_loader_target))
        
        # Forward pass for source domain
        source_imgs = source_batch['img'].to(self.device)
        source_targets = source_batch['bboxes'].to(self.device)
        source_detections, source_domain_pred = self.model(source_imgs, self.alpha, return_domain=True)
        
        # Forward pass for target domain
        target_imgs = target_batch['img'].to(self.device)
        _, target_domain_pred = self.model(target_imgs, self.alpha, return_domain=True)
        
        # Detection loss
        detection_loss = self.criterion(source_detections, source_targets)
        
        # Domain classification loss
        source_domain_labels = torch.ones(source_domain_pred.size(0), 1).to(self.device)
        target_domain_labels = torch.zeros(target_domain_pred.size(0), 1).to(self.device)
        
        domain_loss = (self.domain_criterion(source_domain_pred, source_domain_labels) +
                      self.domain_criterion(target_domain_pred, target_domain_labels)) / 2
        
        # Total loss
        loss = detection_loss + domain_loss
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Optimize
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), detection_loss.item(), domain_loss.item()
    
    def train(self):
        """Training loop with domain adaptation"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Initialize target dataloader
        self.train_loader_target = self.get_dataloader(
            self.get_dataset(self.target_data, 'train'),
            self.args.batch,
            'train'
        )
        
        # Training loop
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # Calculate alpha for gradient reversal layer
            self.alpha = 2.0 / (1.0 + torch.exp(-10 * epoch / self.epochs)) - 1
            
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=self.nb, bar_format=TQDM.bar_format)
            
            self.tloss = None
            for i, batch in pbar:
                # Run training step
                loss, det_loss, dom_loss = self.train_step(batch, i)
                
                # Update progress bar
                if self.tloss is None:
                    self.tloss = (loss, det_loss, dom_loss)
                else:
                    self.tloss = (0.9 * self.tloss[0] + 0.1 * loss,
                                0.9 * self.tloss[1] + 0.1 * det_loss,
                                0.9 * self.tloss[2] + 0.1 * dom_loss)
                
                pbar.set_description(f'Epoch {epoch}/{self.epochs - 1} '
                                   f'loss: {self.tloss[0]:.4f} '
                                   f'det: {self.tloss[1]:.4f} '
                                   f'dom: {self.tloss[2]:.4f}')
            
            # Validation
            if self.epoch % self.args.val_interval == 0:
                self.val()
            
            # Save checkpoint
            if self.epoch % self.args.save_interval == 0:
                self.save_model()
    
    def val(self):
        """Validation on source domain"""
        self.model.eval()
        stats = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                # Forward pass
                preds = self.model(batch['img'].to(self.device))
                
                # Calculate metrics
                stats.append(self.criterion(preds, batch['bboxes'].to(self.device)))
        
        # Calculate average metrics
        stats = torch.stack(stats).mean()
        
        # Print metrics
        if RANK in (-1, 0):
            print(f'\nValidation Results:')
            print(f'mAP@0.5: {stats[0]:.4f}')
            print(f'mAP@0.5:0.95: {stats[1]:.4f}')
            print(f'Precision: {stats[2]:.4f}')
            print(f'Recall: {stats[3]:.4f}')
        
        return stats 