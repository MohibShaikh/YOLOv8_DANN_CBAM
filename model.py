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
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return torch.sigmoid(avg_out + max_out).view(x.size(0), -1, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

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
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size=1024):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
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
        x = GradientReversalLayer.apply(x, alpha)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

class CBAMC2f(C2f):
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__(c1, c2, *args, **kwargs)
        # CBAM should use output channels (c2), not input channels (c1)
        self.cbam = CBAM(c2)
        
    def forward(self, x):
        x = super().forward(x)
        return self.cbam(x)

class DomainAdaptiveYOLOv8(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, task='detect'):
        # Initialize base model — DetectionModel resolves the YAML internally
        super().__init__(cfg, ch, nc)

        # Set args needed by v8DetectionLoss
        if not hasattr(self, 'args'):
            from ultralytics.cfg import get_cfg
            self.args = get_cfg()

        # Replace C2f blocks with CBAMC2f in backbone
        # This includes both direct children and nested modules
        self._replace_c2f_with_cbam()
        
        # Add domain classifier - calculate feature dimensions dynamically
        # Get the channels from the last backbone layer before detection head
        # YOLOv8 structure: backbone layers output feature maps that go to detection head
        # We'll use a dummy forward pass to determine feature dimensions
        self.domain_classifier = None  # Initialize later after determining size
        self._initialize_domain_classifier()
        
        # Register forward hook on the second-to-last layer to capture features
        self._cached_features = None
        self._feature_hook = self.model[-2].register_forward_hook(self._hook_features)

        # Initialize weights
        initialize_weights(self)

    def _hook_features(self, module, input, output):
        """Forward hook that caches features from the layer before the detection head"""
        if isinstance(output, (list, tuple)):
            self._cached_features = output[-1]
        else:
            self._cached_features = output

    def _replace_c2f_with_cbam(self):
        """Recursively replace all C2f modules with CBAMC2f"""
        replaced_count = 0
        for i, m in enumerate(self.model):
            if isinstance(m, C2f) and not isinstance(m, CBAMC2f):
                c1 = m.cv1.conv.in_channels
                c2 = m.cv2.conv.out_channels
                n = len(m.m)  # number of Bottleneck blocks
                shortcut = m.m[0].add if n > 0 else False
                g = m.m[0].cv2.conv.groups if n > 0 else 1
                e = m.c / c2 if c2 > 0 else 0.5  # m.c is hidden channels = c2 * e
                new_module = CBAMC2f(c1, c2, n, shortcut, g, e)
                new_module.load_state_dict(m.state_dict(), strict=False)
                # Preserve model graph attributes set by DetectionModel
                for attr in ('i', 'f', 'type', 'np'):
                    if hasattr(m, attr):
                        setattr(new_module, attr, getattr(m, attr))
                self.model[i] = new_module
                replaced_count += 1
        
        print(f"Replaced {replaced_count} C2f modules with CBAMC2f (with CBAM attention)")
        return replaced_count
    
    def _initialize_domain_classifier(self):
        """Initialize domain classifier with proper feature dimensions"""
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            features = self._extract_features(dummy_input)
            if features is not None:
                # Only need channel count -- AdaptiveAvgPool2d(1) in DomainClassifier handles spatial dims
                self.domain_classifier = DomainClassifier(features.shape[1])
            else:
                self.domain_classifier = DomainClassifier(256)
    
    def _extract_features(self, x):
        """Extract features from backbone before detection head"""
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)  # run layer
            y.append(x if m.i in self.save else None)
            
            # The detection head is the last module, extract features before it
            if i == len(self.model) - 2:
                # At this point, x contains the feature pyramid
                # For domain classification, use the smallest feature map (highest semantic level)
                if isinstance(x, (list, tuple)):
                    return x[-1]  # Use the deepest feature map
                else:
                    return x
        
        return None
        
    def forward(self, x, alpha=1.0, return_domain=False):
        """
        Forward pass through the model.
        Args:
            x: Input tensor
            alpha: Alpha value for gradient reversal (0 to 1)
            return_domain: Whether to return domain predictions
        """
        # Single forward pass — the hook captures features automatically
        detections = super().forward(x)

        if return_domain and self.domain_classifier is not None:
            features = self._cached_features
            if features is not None:
                domain_pred = self.domain_classifier(features, alpha)
                return detections, domain_pred
            else:
                batch_size = x.shape[0]
                domain_pred = torch.zeros(batch_size, 1, device=x.device)
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
        """Export model to ONNX format"""
        # Remove domain classifier and hook for export
        domain_classifier = self.domain_classifier
        self.domain_classifier = None
        self._feature_hook.remove()

        exported_model = super().export(**kwargs)

        # Restore domain classifier and hook
        self.domain_classifier = domain_classifier
        self._feature_hook = self.model[-2].register_forward_hook(self._hook_features)

        return exported_model