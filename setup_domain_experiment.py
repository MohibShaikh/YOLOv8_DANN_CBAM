"""
Setup a domain adaptation experiment from the NEU-GC10 steel defect dataset.

Strategy: Split into source (original) and target (augmented to simulate different
camera/lighting conditions). This mimics a real industrial scenario where you have
labels from one inspection station and need to deploy on another with different
lighting, camera angle, or surface finish.

Target domain augmentations:
- Brightness reduction (simulates darker lighting)
- Contrast shift
- Gaussian noise (simulates different sensor)
- Color temperature shift
"""

import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def augment_target_domain(img):
    """Apply domain-shift augmentations to simulate a different camera/environment."""
    # Darken + shift color temperature (blueish tint, like fluorescent lighting)
    img = img.astype(np.float32)

    # Reduce brightness by 30-50%
    img *= random.uniform(0.5, 0.7)

    # Add blue tint (simulate different lighting)
    img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(1.1, 1.3), 0, 255)  # B
    img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(0.8, 0.9), 0, 255)  # R

    # Add Gaussian noise (different sensor)
    noise = np.random.normal(0, random.uniform(8, 15), img.shape)
    img = np.clip(img + noise, 0, 255)

    # Slight blur (different lens)
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0).astype(np.float32)

    return img.astype(np.uint8)


def setup_experiment(dataset_root, output_root, target_ratio=0.4):
    """
    Split dataset into source and target domains.

    Args:
        dataset_root: Path to neugc10 dataset
        output_root: Where to create source/ and target/ directories
        target_ratio: Fraction of training data to use as target domain
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    # Paths
    source_dir = output_root / 'source'
    target_dir = output_root / 'target'

    # Collect all training images
    train_images = sorted((dataset_root / 'train' / 'images').glob('*.jpg'))
    random.seed(42)
    random.shuffle(train_images)

    # Split: source gets (1 - target_ratio), target gets target_ratio
    split_idx = int(len(train_images) * (1 - target_ratio))
    source_imgs = train_images[:split_idx]
    target_imgs = train_images[split_idx:]

    print(f'Total training images: {len(train_images)}')
    print(f'Source domain: {len(source_imgs)} images (original)')
    print(f'Target domain: {len(target_imgs)} images (augmented, simulating different camera)')

    # --- Source domain: original images with labels ---
    for split_name, img_list in [('train', source_imgs)]:
        img_out = source_dir / split_name / 'images'
        lbl_out = source_dir / split_name / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_list, desc=f'Source {split_name}'):
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    # Source val = original val set
    val_img_out = source_dir / 'val' / 'images'
    val_lbl_out = source_dir / 'val' / 'labels'
    val_img_out.mkdir(parents=True, exist_ok=True)
    val_lbl_out.mkdir(parents=True, exist_ok=True)

    val_images = sorted((dataset_root / 'valid' / 'images').glob('*.jpg'))
    for img_path in tqdm(val_images, desc='Source val'):
        shutil.copy2(img_path, val_img_out / img_path.name)
        lbl_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        if lbl_path.exists():
            shutil.copy2(lbl_path, val_lbl_out / lbl_path.name)

    # --- Target domain: augmented images (labels copied but only used for domain loss) ---
    tgt_img_out = target_dir / 'train' / 'images'
    tgt_lbl_out = target_dir / 'train' / 'labels'
    tgt_img_out.mkdir(parents=True, exist_ok=True)
    tgt_lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(target_imgs, desc='Target train (augmenting)'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aug_img = augment_target_domain(img)
        cv2.imwrite(str(tgt_img_out / img_path.name), aug_img)

        # Copy labels (needed by dataloader, but DANN only uses them for domain classification)
        lbl_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        if lbl_path.exists():
            shutil.copy2(lbl_path, tgt_lbl_out / lbl_path.name)

    # --- Write YAML configs ---
    # Read class names from dataset's own data.yaml if available
    data_yaml = dataset_root / 'data.yaml'
    if data_yaml.exists():
        with open(data_yaml) as f:
            ds_cfg = yaml.safe_load(f)
        class_names = ds_cfg.get('names', [])
        print(f'Read {len(class_names)} classes from {data_yaml}')
    else:
        # Fallback for NEU-GC10
        class_names = [
            'crazing', 'crease', 'crescent_gap', 'inclusion', 'oil_spot',
            'patches', 'pitted_surface', 'punching_hole', 'rolled-in_scale',
            'rolled_pit', 'scratches', 'silk_spot', 'waist_folding', 'water_spot',
            'welding_line'
        ]
        print(f'Using default {len(class_names)} GC10 class names (no data.yaml found)')

    source_yaml = {
        'path': str(source_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 15,
        'names': class_names,
    }

    target_yaml = {
        'path': str(target_dir.resolve()),
        'train': 'train/images',
        'val': 'train/images',  # reuse train for target val
        'nc': 15,
        'names': class_names,
    }

    with open(output_root / 'source.yaml', 'w') as f:
        yaml.dump(source_yaml, f, default_flow_style=False)

    with open(output_root / 'target.yaml', 'w') as f:
        yaml.dump(target_yaml, f, default_flow_style=False)

    print(f'\nDone! Configs written to:')
    print(f'  {output_root / "source.yaml"}')
    print(f'  {output_root / "target.yaml"}')
    print(f'\nRun training with:')
    print(f'  python3 train.py \\')
    print(f'    --source-data {output_root / "source.yaml"} \\')
    print(f'    --target-data {output_root / "target.yaml"} \\')
    print(f'    --epochs 50 --batch-size 8 --device 0')


if __name__ == '__main__':
    setup_experiment(
        dataset_root='neugc10',
        output_root='datasets/neu_domain_experiment',
    )
