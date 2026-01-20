"""
Training with Optimized Hyperparameters for WSL
"""

import torch
import os
import argparse
from model import PatchCoreImproved, save_model
from preprocessing import get_client_dataloader, get_centralized_dataloader


def train_standalone_improved(
    client_id=None, 
    data_root='data/federated_data',
    num_clients=5, 
    save_dir='checkpoints/improved',
    device='cuda',
    # IMPROVED HYPERPARAMETERS
    img_size=320,  # Up from 256 for better details
    patches_per_category=30000,  # Up from 25000
    batch_size=16,  # Down from 32 for WSL safety
    L=0.3,  # Down from 0.5 for less strict fairness
    lam=0.5,  # Up from 0.3 for more fairness
    adaptive_sampling=True,
    layer_indices=[2, 3]  # layers 2 and 3 for best multi-scale
):
    """
    Training with improvements:
    - Adaptive sampling per category
    - Better hyperparameters
    - WSL-safe memory management
    
    Args:
        client_id: Client ID for federated learning (None for centralized)
        data_root: Root directory for federated data
        num_clients: Number of clients (for centralized)
        save_dir: Directory to save model
        device: Computation device
        img_size: Image size for training
        patches_per_category: Base number of patches per category
        batch_size: Batch size
        L: Lipschitz constant for fairness
        lam: Weight for fairness penalty
        adaptive_sampling: Whether to use adaptive sampling
        layer_indices: Backbone layer indices to extract features from
        
    Returns:
        model: Trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    if client_id is not None:
        print(f"IMPROVED TRAINING - CLIENT {client_id}")
    else:
        print("IMPROVED CENTRALIZED TRAINING")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Base patches/category: {patches_per_category}")
    print(f"Adaptive sampling: {adaptive_sampling}")
    print(f"Fairness: L={L}, lambda={lam}")
    print(f"Layers: {layer_indices}")
    print(f"{'=' * 70}\n")
    
    # Data
    if client_id is not None:
        from preprocessing import AutoVIDataset, get_transforms
        from torch.utils.data import DataLoader
        
        client_path = os.path.join(data_root, f'client_{client_id}')
        train_transform, _ = get_transforms(img_size, augment=True)
        
        dataset = AutoVIDataset(
            path=client_path,
            transform=train_transform,
            train=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 for WSL safety
            pin_memory=True
        )
        save_name = f'client_{client_id}_improved.joblib'
    else:
        from preprocessing import AutoVIDataset, get_transforms
        from torch.utils.data import DataLoader
        
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
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        save_name = 'centralized_improved.joblib'
    
    # Model
    model = PatchCoreImproved(backbone='wide_resnet50', layer_indices=layer_indices)
    
    # Training
    model.fit(
        loader,
        device,
        patches_per_category=patches_per_category,
        L=L,
        lam=lam,
        adaptive_sampling=adaptive_sampling
    )
    
    # Save
    save_path = os.path.join(save_dir, save_name)
    save_model(model, save_path)
    
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"Model saved: {save_path}")
    print(f"Memory bank: {model.memory_bank.shape}")
    if model.category_stats:
        print("\nCategory statistics:")
        for cat, stats in sorted(model.category_stats.items()):
            print(f"  {cat}: {stats['target_patches']} patches "
                  f"({stats['image_count']} images)")
    print(f"{'=' * 70}\n")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved PatchCore Training')
    parser.add_argument('--client_id', type=int, default=None)
    parser.add_argument('--data_root', type=str, default='data/federated_data')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints/improved')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--img_size', type=int, default=320)
    parser.add_argument('--patches_per_category', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--L', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--no_adaptive', action='store_true')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 3])
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_standalone_improved(
        client_id=args.client_id,
        data_root=args.data_root,
        num_clients=args.num_clients,
        save_dir=args.save_dir,
        device=device,
        img_size=args.img_size,
        patches_per_category=args.patches_per_category,
        batch_size=args.batch_size,
        L=args.L,
        lam=args.lam,
        adaptive_sampling=not args.no_adaptive,
        layer_indices=args.layers
    )