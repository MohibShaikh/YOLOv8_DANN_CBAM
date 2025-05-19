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
    --save-dir runs/train
```

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

1. **CBAM Integration**:
   - Channel Attention: Learns channel-wise feature importance
   - Spatial Attention: Learns spatial feature importance
   - Integrated into YOLOv8n backbone and neck

2. **DANN Implementation**:
   - Gradient Reversal Layer (GRL)
   - Domain Classifier
   - Progressive domain adaptation with increasing alpha

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