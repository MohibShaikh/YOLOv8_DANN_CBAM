# Domain-Adaptive YOLOv8 with CBAM and DANN

This project implements a domain-adaptive object detector based on YOLOv8n, enhanced with CBAM (Convolutional Block Attention Module) and DANN (Domain-Adversarial Neural Network) for improved cross-domain detection performance.

## Features

- YOLOv8n backbone with CBAM attention mechanism
- Domain adaptation using DANN
- Support for both labeled (source) and unlabeled (target) domains
- Compatible with Ultralytics YOLO format
- Efficient implementation for edge deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MohibShaikh/yolov8_dann_cbam.git
cd yolov8_dann_cbam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

If using Google Colab, you may need to restart the runtime after installation.

## Usage

### Training

To train the model with domain adaptation:

```bash
python train.py \
    --source-data path/to/source/data.yaml \
    --target-data path/to/target/data.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device cuda \
    --workers 8 \
    --save-dir runs/domain_adaptive \
    --pretrained yolov8n.pt \
    --lr 0.001 \
    --val-interval 5
```

**Parameters:**
- `--source-data`: Path to source domain dataset YAML
- `--target-data`: Path to target domain dataset YAML (can be unlabeled)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--img-size`: Input image size (default: 640)
- `--device`: Device to use (cuda:0, cpu, etc.)
- `--workers`: Number of data loading workers (default: 8)
- `--save-dir`: Directory to save checkpoints and results
- `--pretrained`: Path to pretrained YOLOv8 weights (default: yolov8n.pt)
- `--lr`: Learning rate (default: 0.001)
- `--val-interval`: Validation interval in epochs (default: 5)

### Dataset Format

Both source and target datasets should follow the Ultralytics YOLO format:

```yaml
# data.yaml
path: path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 80  # number of classes
names: ['person', 'bicycle', ...]  # class names
```

## Model Architecture

### 1. CBAM Integration (Convolutional Block Attention Module)

The CBAM attention mechanism is integrated into all C2f blocks in the YOLOv8 backbone:

- **Channel Attention**: Uses both average and max pooling to learn channel-wise feature importance
- **Spatial Attention**: Uses channel-wise pooling to learn spatial feature importance
- **Sequential Application**: Channel attention is applied first, followed by spatial attention
- **Memory Efficient**: Processes features in chunks to reduce memory footprint

**Key Features:**
- Automatically replaces all C2f modules with CBAMC2f during model initialization
- Preserves pretrained weights from standard YOLOv8
- Adds minimal computational overhead (~10-20%)

### 2. DANN Implementation (Domain-Adversarial Neural Networks)

Domain adaptation is achieved through gradient reversal:

- **Gradient Reversal Layer (GRL)**: Reverses gradients during backpropagation with scaling factor α
- **Domain Classifier**: Binary classifier to distinguish source vs target domain
- **Progressive Adaptation**: α increases from 0 to 1 following schedule: α = 2/(1+exp(-10p)) - 1
- **Feature Extraction**: Uses features from the deepest backbone layer (highest semantic level)

**Key Features:**
- Domain classifier initialized with proper feature dimensions via dummy forward pass
- Memory-efficient chunked processing for large batch sizes
- Balanced domain loss using equal weighting for source and target domains

### 3. Architecture Improvements

**Recent Fixes:**
- ✅ Fixed CBAM channel dimension bug (now uses output channels correctly)
- ✅ Robust domain classifier initialization (no hardcoded feature dimensions)
- ✅ Proper integration with YOLOv8's forward pass
- ✅ Correct YOLO detection loss (uses v8DetectionLoss instead of BCE)
- ✅ Verified CBAM is applied to all C2f blocks
- ✅ Verified DANN gradient reversal works correctly

## Training Process

The training process optimizes two objectives:
1. Object detection loss on source domain
2. Domain classification loss between source and target domains

The model learns to:
- Detect objects accurately in the source domain
- Extract domain-invariant features
- Adapt to the target domain without labels

## Version Control & .gitignore

This repository includes a `.gitignore` file tailored for deep learning projects. It excludes:
- Checkpoints, logs, and model weights
- Dataset and output folders
- Python cache and build files
- IDE/editor settings (VSCode, PyCharm, etc.)
- Colab and Jupyter notebook checkpoints

**Best practices:**
- Commit only code, configuration, and small metadata files (like `dataset.yaml`).
- Do not commit large datasets, model weights, or outputs.
- Use branches for experimental features.
- Document major changes in your commit messages.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 