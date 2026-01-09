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

This project integrates two key techniques into the YOLOv8n architecture: the **Convolutional Block Attention Module (CBAM)** and **Domain-Adversarial Neural Networks (DANN)**.

### 1. CBAM Integration

CBAM is an attention mechanism that helps the model focus on important features and suppress irrelevant ones. It consists of two sequential sub-modules:

-   **Channel Attention Module**: This module learns which channels in a feature map are most important. It computes a channel attention map by applying both average-pooling and max-pooling operations to the input feature map and feeding the results through a shared multi-layer perceptron (MLP).
-   **Spatial Attention Module**: This module learns which spatial locations in a feature map are most informative. It generates a 2D spatial attention map by applying pooling operations along the channel axis and feeding the result through a convolutional layer.

In this implementation, the standard `C2f` blocks in the YOLOv8n backbone and neck are replaced with `CBAMC2f` blocks, which apply CBAM to the output of the `C2f` module. This allows the model to learn more discriminative features for object detection.

### 2. DANN Implementation

DANN is a domain adaptation technique that encourages the model to learn features that are both discriminative for the main task (object detection) and invariant to the domain (e.g., synthetic vs. real images). It consists of three components:

-   **Feature Extractor**: This is the main YOLOv8 backbone and neck, which extracts features from the input images.
-   **Label Predictor**: This is the standard YOLOv8 detection head, which predicts bounding boxes and class labels from the extracted features.
-   **Domain Classifier**: This is a separate head that is trained to distinguish between features from the source and target domains.

The key component of DANN is the **Gradient Reversal Layer (GRL)**, which is placed between the feature extractor and the domain classifier.

## Training Process

The training process is designed to optimize two competing objectives simultaneously:

1.  **Minimize the detection loss** on the labeled source domain.
2.  **Maximize the domain classification loss** between the source and target domains.

### Gradient Reversal Layer (GRL)

During the forward pass, the GRL acts as an identity function, passing the features from the feature extractor to the domain classifier without modification. However, during the backward pass, the GRL reverses the gradient by multiplying it by a negative constant (`-alpha`).

This gradient reversal has the following effect:

-   The **domain classifier** is trained to minimize the domain classification error, learning to distinguish between source and target features.
-   The **feature extractor** is trained to *maximize* the domain classification error, learning to produce features that are indistinguishable to the domain classifier.

### Combined Loss Function

The total loss function is a combination of the detection loss and the domain classification loss:

`Total Loss = Detection Loss + Domain Loss`

-   The **Detection Loss** is calculated only on the labeled source domain data and is backpropagated through the entire network (feature extractor and label predictor).
-   The **Domain Loss** is a binary cross-entropy loss that measures how well the domain classifier can distinguish between the source and target domains. It is backpropagated through the domain classifier and, via the GRL, to the feature extractor.

### Progressive Domain Adaptation

To stabilize the training process, the influence of the domain classifier is gradually increased over time. This is achieved by using a variable `alpha` for the GRL, which is calculated as follows:

`alpha = (2.0 / (1.0 + exp(-10 * epoch / total_epochs))) - 1`

This formula causes `alpha` to increase from 0 to 1 as the training progresses. In the early epochs, the model focuses on learning the primary task of object detection. As `alpha` increases, the model is increasingly penalized for learning domain-specific features, forcing it to learn more domain-invariant representations.

This process encourages the model to learn features that are robust and generalizable, allowing it to adapt from the labeled source domain to the unlabeled target domain.

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