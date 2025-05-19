import os
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

def verify_dataset_structure(dataset_path):
    """Verify and create dataset structure if needed"""
    dataset_path = Path(dataset_path)
    
    # Create main directories if they don't exist
    for split in ['train', 'valid', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Check if images and labels exist
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        
        # Count files
        n_images = len(list(img_dir.glob('*.*')))
        n_labels = len(list(label_dir.glob('*.txt')))
        
        print(f'\n{split.upper()} split:')
        print(f'Images: {n_images}')
        print(f'Labels: {n_labels}')
        
        if n_images == 0 or n_labels == 0:
            print(f'Warning: {split} split is empty!')
        
        # Verify image-label pairs
        if n_images > 0 and n_labels > 0:
            img_files = {f.stem for f in img_dir.glob('*.*')}
            label_files = {f.stem for f in label_dir.glob('*.txt')}
            
            missing_labels = img_files - label_files
            missing_images = label_files - img_files
            
            if missing_labels:
                print(f'Warning: {len(missing_labels)} images missing labels')
            if missing_images:
                print(f'Warning: {len(missing_images)} labels missing images')

def update_dataset_yaml(dataset_path, num_classes, class_names):
    """Update dataset.yaml with correct class information"""
    yaml_path = Path('dataset.yaml')
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {}
    
    # Update dataset information
    data['path'] = str(dataset_path)
    data['train'] = 'train/images'
    data['valid'] = 'valid/images'
    data['test'] = 'test/images'
    data['nc'] = num_classes
    data['names'] = class_names
    
    # Save updated yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f'\nUpdated {yaml_path} with:')
    print(f'Number of classes: {num_classes}')
    print(f'Class names: {class_names}')

def main():
    # Get dataset path
    dataset_path = input('Enter path to your dataset: ').strip()
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f'Error: Dataset path {dataset_path} does not exist!')
        return
    
    # Verify dataset structure
    print('\nVerifying dataset structure...')
    verify_dataset_structure(dataset_path)
    
    # Get class information
    num_classes = int(input('\nEnter number of classes: '))
    class_names = []
    print('\nEnter class names (one per line, press Enter twice to finish):')
    while True:
        name = input().strip()
        if not name:
            break
        class_names.append(name)
    
    if len(class_names) != num_classes:
        print(f'Warning: Number of class names ({len(class_names)}) does not match num_classes ({num_classes})')
    
    # Update dataset.yaml
    update_dataset_yaml(dataset_path, num_classes, class_names)
    
    print('\nDataset preparation complete!')
    print('You can now use this dataset with the domain adaptation training script.')

if __name__ == '__main__':
    main() 