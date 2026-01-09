import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.cfg import get_cfg
import yaml

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        
    def forward(self, x):
        # Process in chunks to save memory
        batch_size = x.size(0)
        chunk_size = min(batch_size, 32)  # Process 32 images at a time
        
        outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = x[i:i + chunk_size]
            avg_out = self.fc(self.avg_pool(chunk).view(chunk.size(0), -1))
            max_out = self.fc(self.max_pool(chunk).view(chunk.size(0), -1))
            out = avg_out + max_out
            outputs.append(torch.sigmoid(out).view(chunk.size(0), -1, 1, 1))
            
        return torch.cat(outputs, dim=0)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # Process in chunks to save memory
        batch_size = x.size(0)
        chunk_size = min(batch_size, 32)  # Process 32 images at a time
        
        outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = x[i:i + chunk_size]
            avg_out = torch.mean(chunk, dim=1, keepdim=True)
            max_out, _ = torch.max(chunk, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(x_cat)
            outputs.append(torch.sigmoid(out))
            
        return torch.cat(outputs, dim=0)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply attention in sequence to save memory
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size=1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, alpha=1.0):
        # Process in chunks to save memory
        batch_size = x.size(0)
        chunk_size = min(batch_size, 32)  # Process 32 images at a time
        
        outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = x[i:i + chunk_size]
            chunk = GradientReversalLayer.apply(chunk, alpha)
            chunk = chunk.view(chunk.size(0), -1)
            outputs.append(self.classifier(chunk))
            
        return torch.cat(outputs, dim=0)

class CBAMC2f(C2f):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(args[0])
        
    def forward(self, x):
        x = super().forward(x)
        return self.cbam(x)

class DomainAdaptiveYOLOv8(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, task='detect'):
        # Load YOLOv8 configuration
        if isinstance(cfg, str):
            cfg = get_cfg(cfg)
        
        # Initialize base model
        super().__init__(cfg, ch, nc, task)
        
        # Replace C2f blocks with CBAMC2f in backbone
        for i, m in enumerate(self.model):
            if isinstance(m, C2f):
                self.model[i] = CBAMC2f(m.c, m.c, m.n, m.shortcut, m.g, m.e)
        
        # Add domain classifier
        last_feature_map_in_channels = self.model[-1].cv2[-1][0].conv.in_channels
        self.domain_classifier = DomainClassifier(last_feature_map_in_channels)
        
        # Initialize weights
        initialize_weights(self)
        
    def forward(self, x, alpha=1.0, return_domain=False):
        """
        Forward pass through the model. It works similar to `ultralytics.nn.tasks.BaseModel._forward_once`.
        We manually iterate through the model layers to extract features for the domain classifier.
        """
        y = []
        features_for_domain_classifier = None
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if i == len(self.model) - 1:  # Just before the DetectionHead
                # x is a list of feature maps. We'll use the last one for domain classification.
                features_for_domain_classifier = x[-1]

            x = m(x)  # run layer
            y.append(x if m.i in self.save else None)  # save output

        detections = x

        if return_domain:
            domain_pred = self.domain_classifier(features_for_domain_classifier, alpha)
            return detections, domain_pred

        return detections
    
    def predict(self, source, target=None, alpha=1.0, **kwargs):
        """
        Perform prediction on source and optionally target domains
        """
        # Source domain prediction
        source_pred = super().predict(source, **kwargs)
        
        if target is not None:
            # Target domain prediction
            target_pred = super().predict(target, **kwargs)
            return source_pred, target_pred
        
        return source_pred
    
    def export(self, **kwargs):
        """
        Export model to ONNX format
        """
        # Remove domain classifier for export
        domain_classifier = self.domain_classifier
        self.domain_classifier = None
        
        # Export base model
        exported_model = super().export(**kwargs)
        
        # Restore domain classifier
        self.domain_classifier = domain_classifier
        
        return exported_model 