# Domain-Adaptive YOLOv8 with CBAM and DANN

A domain-adaptive object detector based on YOLOv8n, enhanced with CBAM (Convolutional Block Attention Module) and DANN (Domain-Adversarial Neural Network) for cross-domain steel defect detection.

## Features

- YOLOv8n backbone with CBAM attention on all C2f blocks
- Domain adaptation using DANN with gradient reversal
- Source/target domain split with synthetic augmentations to simulate camera/lighting shifts
- Google Colab notebook for easy training on GPU
- NEU-GC10 steel defect dataset (15 classes) via Roboflow

## Dataset

This project uses the **NEU-GC10** steel surface defect dataset with 15 classes:

`crazing`, `crease`, `crescent_gap`, `inclusion`, `oil_spot`, `patches`, `pitted_surface`, `punching_hole`, `rolled-in_scale`, `rolled_pit`, `scratches`, `silk_spot`, `waist_folding`, `water_spot`, `welding_line`

The dataset is auto-downloaded from Roboflow when using the Colab notebook.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MohibShaikh/YOLOv8_DANN_CBAM.git
cd YOLOv8_DANN_CBAM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Google Colab (Recommended)

Open `YOLOv8_DANN_CBAM_Colab.ipynb` in Google Colab. It handles dataset download, domain experiment setup, and training automatically.

### Local Training

#### 1. Prepare the dataset

Download the NEU-GC10 dataset and place it under `neugc10/`.

#### 2. Set up the domain experiment

Split the dataset into source and target domains with synthetic augmentations:

```bash
python setup_domain_experiment.py
```

This creates `datasets/neu_domain_experiment/` with `source.yaml` and `target.yaml` configs. Target domain images are augmented with brightness reduction, color temperature shift, Gaussian noise, and blur to simulate different inspection conditions.

#### 3. Train

```bash
python train.py \
    --source-data datasets/neu_domain_experiment/source.yaml \
    --target-data datasets/neu_domain_experiment/target.yaml \
    --epochs 50 \
    --batch-size 8 \
    --img-size 640 \
    --device cuda
```

**Parameters:**
- `--source-data`: Source domain dataset YAML
- `--target-data`: Target domain dataset YAML
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--device`: Device to use (`cuda`, `cpu`)
- `--workers`: Data loading workers (default: 8)
- `--save-dir`: Checkpoint directory (default: `runs/domain_adaptive`)
- `--pretrained`: Pretrained YOLOv8 weights (default: `yolov8n.pt`)
- `--lr`: Learning rate (default: 0.001)
- `--val-interval`: Validation interval in epochs (default: 5)

## Model Architecture

### CBAM (Convolutional Block Attention Module)

Integrated into all C2f blocks in the YOLOv8 backbone:

- **Channel Attention**: Average + max pooling to learn channel-wise feature importance
- **Spatial Attention**: Channel-wise pooling to learn spatial feature importance
- Sequential application (channel first, then spatial)
- Preserves pretrained YOLOv8 weights; adds ~10-20% overhead

### DANN (Domain-Adversarial Neural Network)

Domain adaptation through gradient reversal:

- **Gradient Reversal Layer (GRL)**: Reverses gradients during backpropagation with scaling factor alpha
- **Domain Classifier**: Binary classifier (source vs target domain)
- **Progressive Adaptation**: alpha increases from 0 to 1 following: `alpha = 2/(1+exp(-10p)) - 1`
- Features extracted from the deepest backbone layer

### Training Objectives

1. **Detection loss** on source domain (labeled) using v8DetectionLoss
2. **Domain classification loss** between source and target domains

The model learns domain-invariant features that transfer across inspection conditions without target domain labels.

## Project Structure

```
├── model.py                     # CBAM, DANN, DomainAdaptiveYOLOv8
├── train.py                     # Training script
├── setup_domain_experiment.py   # Source/target domain split + augmentation
├── prepare_dataset.py           # Dataset verification utility
├── dataset.yaml                 # Dataset config
├── requirements.txt             # Dependencies
├── YOLOv8_DANN_CBAM_Colab.ipynb # Colab notebook
└── neugc10/                     # NEU-GC10 dataset (not committed)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
