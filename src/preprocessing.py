"""
Preprocessing Module for PatchCore Federated Learning

Dataset classes and transforms for the following structure:

Training: data/federated_data/client_X/category/train/good/image.png
Test: data/test_data_centralized/category/test/good|bad/image.png
Masks: data/test_data_centralized/category/ground_truth/defect_type/XXXX/0000.png
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


def get_transforms(img_size=256, augment=True):
    """
    Get image transforms for training and validation.
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        train_transform: Transforms for training
        val_transform: Transforms for validation
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class AutoVIDataset(Dataset):
    """
    Dataset for AutoVI training.
    
    Expected structure:
        client_X/category/train/good/image.png
        client_X/category/train/bad/image.png (optional, excluded when train=True)
    """
    
    def __init__(self, path=None, transform=None, train=True):
        """
        Initialize dataset.
        
        Args:
            path: Path or list of paths to client directories
            transform: Torchvision transforms
            train: If True, exclude images from 'bad' folder
        """
        if path is None:
            data_path = [f"data/federated_data/client_{i}" for i in range(5)]
        else:
            data_path = path
        
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.images = []
        self.categories = []
        
        for client_path in data_path:
            if not os.path.exists(client_path):
                print(f"WARNING: {client_path} does not exist, skipping")
                continue
            
            # Iterate through categories in client
            for category in os.listdir(client_path):
                category_path = os.path.join(client_path, category)
                if not os.path.isdir(category_path):
                    continue
                
                # Look in train/good
                good_path = os.path.join(category_path, 'train', 'good')
                if os.path.exists(good_path):
                    for img_name in os.listdir(good_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
                            self.images.append(os.path.join(good_path, img_name))
                            self.categories.append(category)
                
                # If not training, include bad images too
                if not train:
                    bad_path = os.path.join(category_path, 'train', 'bad')
                    if os.path.exists(bad_path):
                        for img_name in os.listdir(bad_path):
                            if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
                                self.images.append(os.path.join(bad_path, img_name))
                                self.categories.append(category)
        
        self.transform = transform
        print(f"Dataset loaded: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        category = self.categories[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Label: 1 if in 'bad' folder, 0 otherwise
        label = 1 if '/bad/' in img_path or '\\bad\\' in img_path else 0
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'category': category
        }


class AutoVIEvalDataset(Dataset):
    """
    Dataset for AutoVI evaluation.
    
    Expected structure:
        root_path/category/test/good/image.png
        root_path/category/test/bad/defectType_XXXX.png
        root_path/category/ground_truth/defectType/XXXX/0000.png
    """
    
    def __init__(self, root_path, transform=None):
        """
        Initialize evaluation dataset.
        
        Args:
            root_path: Path to test data root directory
            transform: Torchvision transforms
        """
        self.samples = []
        self.transform = transform
        
        print(f"Loading eval dataset from {root_path}...")
        
        # Iterate through categories
        for category in os.listdir(root_path):
            cat_dir = os.path.join(root_path, category)
            if not os.path.isdir(cat_dir):
                continue
            
            test_dir = os.path.join(cat_dir, 'test')
            gt_dir = os.path.join(cat_dir, 'ground_truth')
            
            if not os.path.exists(test_dir):
                continue
            
            # Good images
            good_dir = os.path.join(test_dir, 'good')
            if os.path.exists(good_dir):
                for img_name in os.listdir(good_dir):
                    if not img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
                        continue
                    self.samples.append({
                        'path': os.path.join(good_dir, img_name),
                        'mask_path': None,
                        'label': 0,
                        'category': category
                    })
            
            # Bad images
            bad_dir = os.path.join(test_dir, 'bad')
            if os.path.exists(bad_dir):
                for img_name in os.listdir(bad_dir):
                    if not img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
                        continue
                    
                    img_path = os.path.join(bad_dir, img_name)
                    
                    # Find mask
                    # Format: defectType_XXXX.png
                    mask_path = None
                    if '_' in img_name:
                        parts = img_name.rsplit('.', 1)[0].split('_')  # Remove extension, split by _
                        if len(parts) >= 2:
                            defect_type = parts[0]
                            img_num = parts[1]
                            
                            # Look for mask in ground_truth/defect_type/img_num/0000.png
                            potential_mask = os.path.join(gt_dir, defect_type, img_num, '0000.png')
                            if os.path.exists(potential_mask):
                                mask_path = potential_mask
                    
                    self.samples.append({
                        'path': img_path,
                        'mask_path': mask_path,
                        'label': 1,
                        'category': category
                    })
        
        print(f"Eval dataset: {len(self.samples)} images "
              f"({sum(1 for s in self.samples if s['label']==0)} good, "
              f"{sum(1 for s in self.samples if s['label']==1)} bad)")
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        # Handle mask
        if s['mask_path'] and os.path.exists(s['mask_path']):
            mask = Image.open(s['mask_path']).convert("L")
            mask = transforms.Resize((img.shape[1], img.shape[2]))(mask)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()
            has_mask = 1
        else:
            mask = torch.zeros((1, img.shape[1], img.shape[2]))
            has_mask = 0
        
        return {
            'image': img,
            'label': torch.tensor(s['label']),
            'mask': mask,
            'category': s['category'],
            'has_mask': torch.tensor(has_mask)
        }
    
    def __len__(self):
        return len(self.samples)


def get_client_dataloader(client_id, data_root='data/federated_data',
                          batch_size=32, img_size=256, augment=True, 
                          num_workers=0):
    """
    Get DataLoader for a specific client.
    
    Args:
        client_id: Client identifier
        data_root: Root directory for federated data
        batch_size: Batch size
        img_size: Image size
        augment: Whether to apply data augmentation
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for the specified client
    """
    client_path = os.path.join(data_root, f'client_{client_id}')
    
    train_transform, _ = get_transforms(img_size, augment)
    
    dataset = AutoVIDataset(
        path=client_path,
        transform=train_transform,
        train=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def get_centralized_dataloader(data_root='data/federated_data', num_clients=5,
                               batch_size=32, img_size=256, num_workers=0):
    """
    Get DataLoader with all data (centralized).
    
    Args:
        data_root: Root directory for federated data
        num_clients: Number of clients
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader containing all client data
    """
    client_paths = [
        os.path.join(data_root, f'client_{i}')
        for i in range(num_clients)
    ]
    
    train_transform, _ = get_transforms(img_size, augment=True)
    
    dataset = AutoVIDataset(
        path=client_paths,
        transform=train_transform,
        train=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def get_eval_dataloader(test_dir, batch_size=8, img_size=256, num_workers=0):
    """
    Get DataLoader for evaluation.
    
    Args:
        test_dir: Path to test data directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for evaluation
    """
    _, val_transform = get_transforms(img_size, augment=False)
    
    dataset = AutoVIEvalDataset(test_dir, transform=val_transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )