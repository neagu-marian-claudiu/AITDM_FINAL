"""
Federated Learning for PatchCore

Key improvements over basic federated learning:
1. build_memory_bank() accepts global_projector parameter
2. _apply_fedprox() in Client - projects local_bank with chunked processing
3. _aggregate_fedprox() in Server - projects banks with chunked processing  
4. Training loop - passes global_projector to clients
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.random_projection import SparseRandomProjection
import joblib

from model import PatchCoreImproved, save_model
from preprocessing import get_client_dataloader, AutoVIDataset, get_transforms


class FederatedClientImproved:
    """Client for Federated PatchCore."""
    
    def __init__(self, client_id, data_root, device, config):
        """
        Initialize federated client.
        
        Args:
            client_id: Client identifier
            data_root: Root directory for federated data
            device: Computation device
            config: Configuration dictionary
        """
        self.client_id = client_id
        self.device = device
        self.config = config
        
        # Load data
        self.loader = get_client_dataloader(
            client_id=client_id,
            data_root=data_root,
            batch_size=config['batch_size'],
            img_size=config['img_size']
        )
        
        # Local model with PatchCoreImproved
        self.model = PatchCoreImproved(
            backbone='wide_resnet50', 
            layer_indices=config['layer_indices']
        )
        self.model.to(device)
    
    def build_memory_bank(self, global_memory_bank=None, global_projector=None):
        """
        Build local memory bank with adaptive sampling.
        
        Args:
            global_memory_bank: Global memory bank (for FedProx)
            global_projector: Projector to convert 1536->128 dims
            
        Returns:
            local_bank: Local memory bank tensor
        """
        print(f"  Client {self.client_id}: Building memory bank...")
        
        # Use improved fit method with adaptive sampling
        self.model.fit(
            self.loader,
            self.device,
            patches_per_category=self.config['patches_per_category'],
            L=self.config['L'],
            lam=self.config['lam'],
            adaptive_sampling=self.config['adaptive_sampling']
        )
        
        local_bank = self.model.memory_bank
        
        # FedProx: adjust towards global model if available
        if global_memory_bank is not None and self.config['mu'] > 0:
            local_bank = self._apply_fedprox(local_bank, global_memory_bank, global_projector)
        
        return local_bank
    
    def _apply_fedprox(self, local_bank, global_bank, global_projector=None):
        """
        Apply FedProx regularization to local memory bank.
        
        Handles dimension mismatch (1536 vs 128) by projecting local bank.
        Uses chunked processing to avoid out-of-memory errors.
        
        Args:
            local_bank: Local memory bank tensor
            global_bank: Global memory bank tensor
            global_projector: Projector for dimension reduction
            
        Returns:
            local_bank: Regularized local memory bank
        """
        mu = self.config['mu']
        
        # Handle dimension mismatch: local=1536, global=128
        if local_bank.shape[1] != global_bank.shape[1]:
            if global_projector is not None:
                print(f"    FedProx: projecting local ({local_bank.shape[1]} -> {global_bank.shape[1]} dims)")
                local_projected = torch.tensor(
                    global_projector.transform(local_bank.numpy())
                ).float()
            else:
                # No projector available, cannot compare - skip
                print(f"    FedProx: skipping (dim mismatch {local_bank.shape[1]} vs {global_bank.shape[1]}, no projector)")
                return local_bank
        else:
            local_projected = local_bank
        
        # Chunked processing to avoid OOM
        chunk_size = 5000  # Process 5000 patches at a time
        min_dist_list = []
        
        print(f"    FedProx: computing distances in chunks of {chunk_size}...")
        for i in range(0, len(local_projected), chunk_size):
            chunk = local_projected[i:i+chunk_size]
            dist_chunk = torch.cdist(chunk, global_bank)
            min_d, _ = dist_chunk.min(dim=1)
            min_dist_list.append(min_d)
            # Free memory
            del dist_chunk
        
        min_dist = torch.cat(min_dist_list)
        avg_dist = min_dist.mean().item()
        print(f"    FedProx: avg distance to global = {avg_dist:.4f}")
        
        # Return original (unprojected) bank
        # FedProx effect is applied through weighted aggregation on server
        return local_bank
    
    def get_memory_bank(self):
        """Get local memory bank."""
        return self.model.memory_bank
    
    def get_category_stats(self):
        """Get category statistics."""
        return self.model.category_stats
    
    def get_projector(self):
        """Get projector from client model."""
        return self.model.projector


class FederatedServerImproved:
    """Improved server for aggregating memory banks."""
    
    def __init__(self, config):
        """
        Initialize federated server.
        
        Args:
            config: Server configuration dictionary
        """
        self.config = config
        self.global_memory_bank = None
        self.global_projector = None
        self.global_category_stats = {}
    
    def aggregate(self, client_banks, client_stats_list, client_projectors=None):
        """
        Improved aggregation with category-aware methods.
        
        Args:
            client_banks: List of (client_id, memory_bank) tuples
            client_stats_list: List of (client_id, category_stats) tuples
            client_projectors: List of (client_id, projector) tuples
            
        Returns:
            global_bank: Aggregated global memory bank
        """
        method = self.config['aggregation_method']
        print(f"\n{'=' * 70}")
        print(f"AGGREGATION: {method.upper()}")
        print(f"{'=' * 70}")
        
        # Preserve projector from first client that has one
        if client_projectors:
            for client_id, projector in client_projectors:
                if projector is not None:
                    self.global_projector = projector
                    print(f"  Preserved projector from client {client_id}")
                    break
        
        if method == 'fedavg':
            global_bank = self._aggregate_fedavg(client_banks)
        elif method == 'fedprox':
            global_bank = self._aggregate_fedprox(client_banks)
        elif method == 'fairness_aware':
            global_bank = self._aggregate_fairness_aware(
                client_banks, client_stats_list
            )
        elif method == 'category_aware':
            global_bank = self._aggregate_category_aware(
                client_banks, client_stats_list
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Differential Privacy (optional)
        if self.config.get('dp_enabled', False):
            global_bank = self._apply_dp(global_bank)
        
        # Dimensionality reduction
        if global_bank.shape[1] > 128:
            if self.global_projector is None:
                print("  Creating new projector (no client projector available)...")
                self.global_projector = SparseRandomProjection(n_components=128, random_state=42)
            print(f"  Projecting {global_bank.shape[1]} -> 128 dims...")
            global_bank = torch.tensor(
                self.global_projector.fit_transform(global_bank.numpy())
            ).float()
        
        self.global_memory_bank = global_bank
        print(f"  Global memory bank: {global_bank.shape}")
        projector_status = "preserved" if self.global_projector else "None"
        print(f"  Projector: {projector_status}")
        print(f"{'=' * 70}\n")
        
        return global_bank
    
    def _aggregate_fedavg(self, client_banks):
        """FedAvg: simple concatenation with subsampling."""
        all_patches = [bank for _, bank in client_banks]
        combined = torch.cat(all_patches, dim=0)
        
        target = self.config['target_memory_size']
        if len(combined) > target:
            idx = np.random.choice(len(combined), target, replace=False)
            combined = combined[idx]
        
        print(f"  FedAvg: {len(combined)} patches")
        return combined
    
    def _aggregate_fedprox(self, client_banks):
        """
        FedProx: weighted aggregation based on proximity to global model.
        Uses chunked processing to avoid memory issues.
        """
        if self.global_memory_bank is None:
            # First round - fallback to FedAvg
            return self._aggregate_fedavg(client_banks)
        
        global_bank = self.global_memory_bank
        target = self.config['target_memory_size']
        
        # Calculate weights based on distance to global
        weights = []
        chunk_size = 5000
        
        for client_id, bank in client_banks:
            # Handle dimension mismatch
            if bank.shape[1] != global_bank.shape[1]:
                if self.global_projector is not None:
                    bank_projected = torch.tensor(
                        self.global_projector.transform(bank.numpy())
                    ).float()
                else:
                    # Cannot compare - use equal weight
                    weights.append(1.0)
                    continue
            else:
                bank_projected = bank
            
            # Chunked distance computation
            min_dists = []
            for i in range(0, len(bank_projected), chunk_size):
                chunk = bank_projected[i:i+chunk_size]
                dist_chunk = torch.cdist(chunk, global_bank)
                min_d, _ = dist_chunk.min(dim=1)
                min_dists.append(min_d)
                del dist_chunk
            
            avg_dist = torch.cat(min_dists).mean().item()
            weight = 1.0 / (1.0 + avg_dist)  # Closer = higher weight
            weights.append(weight)
            print(f"  Client {client_id}: weight={weight:.4f} (dist={avg_dist:.4f})")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted sampling
        all_patches = []
        for (client_id, bank), weight in zip(client_banks, weights):
            num_samples = int(target * weight)
            if num_samples > 0:
                idx = np.random.choice(len(bank), min(num_samples, len(bank)), replace=False)
                all_patches.append(bank[idx])
                print(f"  Client {client_id}: selected {len(idx)} patches (weight={weight:.3f})")
        
        combined = torch.cat(all_patches, dim=0)
        
        # Final subsampling if needed
        if len(combined) > target:
            idx = np.random.choice(len(combined), target, replace=False)
            combined = combined[idx]
        
        print(f"  FedProx: {len(combined)} patches (weighted)")
        return combined
    
    def _aggregate_fairness_aware(self, client_banks, client_stats_list):
        """Fairness-aware: balanced sampling across clients."""
        target = self.config['target_memory_size']
        patches_per_client = target // len(client_banks)
        
        print(f"  Fairness-aware: {patches_per_client} patches per client")
        
        all_patches = []
        for client_id, bank in client_banks:
            num_select = min(len(bank), patches_per_client)
            idx = np.random.choice(len(bank), num_select, replace=False)
            all_patches.append(bank[idx])
            print(f"  Client {client_id}: {num_select} patches")
        
        return torch.cat(all_patches, dim=0)
    
    def _aggregate_category_aware(self, client_banks, client_stats_list):
        """Category-aware: equal representation per category."""
        target = self.config['target_memory_size']
        
        all_categories = set()
        for _, stats in client_stats_list:
            all_categories.update(stats.keys())
        
        patches_per_cat = target // len(all_categories)
        
        print(f"  Category-aware: {patches_per_cat} patches per category")
        print(f"  Categories: {len(all_categories)}")
        
        category_patches = {cat: [] for cat in all_categories}
        
        for client_id, bank in client_banks:
            client_stats = dict(client_stats_list)[client_id]
            total_patches = sum(stats.get('target_patches', 0) 
                              for stats in client_stats.values())
            
            start_idx = 0
            for cat, stats in client_stats.items():
                cat_size = stats.get('target_patches', 0)
                if total_patches > 0:
                    end_idx = start_idx + cat_size
                    cat_patches = bank[start_idx:min(end_idx, len(bank))]
                    category_patches[cat].append(cat_patches)
                    start_idx = end_idx
        
        all_patches = []
        for cat, patches_list in category_patches.items():
            if patches_list:
                combined_cat = torch.cat(patches_list, dim=0)
                num_select = min(len(combined_cat), patches_per_cat)
                idx = np.random.choice(len(combined_cat), num_select, replace=False)
                all_patches.append(combined_cat[idx])
                print(f"    {cat}: {num_select} patches (from {len(combined_cat)})")
        
        return torch.cat(all_patches, dim=0)
    
    def _apply_dp(self, memory_bank):
        """
        Apply Differential Privacy using Gaussian mechanism.
        
        Args:
            memory_bank: Memory bank tensor
            
        Returns:
            noisy_bank: Memory bank with DP noise added
        """
        epsilon = self.config.get('dp_epsilon', 1.0)
        delta = self.config.get('dp_delta', 1e-5)
        
        sensitivity = memory_bank.norm(dim=1).max().item()
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noise = torch.randn_like(memory_bank) * sigma
        noisy_bank = memory_bank + noise
        
        print(f"    DP applied: epsilon={epsilon}, delta={delta}, sigma={sigma:.4f}")
        return noisy_bank
    
    def save(self, filepath, round_num):
        """
        Save global model to file.
        
        Args:
            filepath: Path to save model
            round_num: Current round number
        """
        state = {
            'memory_bank': self.global_memory_bank,
            'projector': self.global_projector,
            'round': round_num,
            'config': self.config,
            'backbone_name': 'wide_resnet50',
            'layer_indices': self.config['layer_indices'],
            'category_stats': self.global_category_stats
        }
        joblib.dump(state, filepath)
        print(f"  Model saved: {filepath}")
        projector_status = "saved" if self.global_projector else "None"
        print(f"    Projector: {projector_status}")


def run_federated_training_improved(
    num_clients=5, 
    num_rounds=3,
    data_root='data/federated_data',
    save_dir='checkpoints/federated_improved',
    device='cuda',
    aggregation='category_aware',
    dp_enabled=False, 
    dp_epsilon=1.0
):
    """
    Run improved federated training with multiple aggregation strategies.
    
    Args:
        num_clients: Number of federated clients
        num_rounds: Number of communication rounds
        data_root: Root directory for federated data
        save_dir: Directory to save checkpoints
        device: Computation device
        aggregation: Aggregation method ('fedavg', 'fedprox', 'fairness_aware', 'category_aware')
        dp_enabled: Whether to enable differential privacy
        dp_epsilon: Privacy budget (smaller = more private)
        
    Returns:
        server: Trained federated server
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("FEDERATED PATCHCORE TRAINING")
    print(f"{'=' * 70}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Aggregation: {aggregation}")
    print(f"Differential Privacy: {dp_enabled} (epsilon={dp_epsilon})")
    print(f"Device: {device}")
    print(f"{'=' * 70}\n")
    
    client_config = {
        'batch_size': 4,
        'img_size': 256,
        'patches_per_category': 10000,
        'L': 0.3,
        'lam': 0.5,
        'mu': 0.01 if aggregation == 'fedprox' else 0.0,
        'adaptive_sampling': True,
        'layer_indices': [2, 3],
    }
    
    server_config = {
        'aggregation_method': aggregation,
        'target_memory_size': 150000,
        'dp_enabled': dp_enabled,
        'dp_epsilon': dp_epsilon,
        'dp_delta': 1e-5,
        'layer_indices': [2, 3],
    }
    
    # Initialize clients
    print("Initializing clients...")
    clients = []
    for i in range(num_clients):
        client = FederatedClientImproved(i, data_root, device, client_config)
        clients.append(client)
        print(f"  Client {i}: initialized")
    
    # Initialize server
    server = FederatedServerImproved(server_config)
    
    # Training loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'=' * 70}")
        
        global_bank = server.global_memory_bank
        global_proj = server.global_projector
        
        client_banks = []
        client_stats = []
        client_projectors = []
        
        for client in clients:
            # Pass both global_bank and global_projector
            local_bank = client.build_memory_bank(global_bank, global_proj)
            stats = client.get_category_stats()
            projector = client.get_projector()
            
            client_banks.append((client.client_id, local_bank))
            client_stats.append((client.client_id, stats))
            client_projectors.append((client.client_id, projector))
            
            print(f"  Client {client.client_id}: {local_bank.shape[0]} patches, dim={local_bank.shape[1]}")
            if stats:
                print(f"    Categories: {len(stats)}")
        
        # Server aggregation
        server.aggregate(client_banks, client_stats, client_projectors)
        
        # Save checkpoint
        save_path = os.path.join(save_dir, f'global_round_{round_num}.joblib')
        server.save(save_path, round_num)
        
        # Clear memory
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Save final model
    final_path = os.path.join(save_dir, 'global_final_improved.joblib')
    server.save(final_path, num_rounds)
    
    print(f"\n{'=' * 70}")
    print("FEDERATED TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Final model: {final_path}")
    print(f"Memory bank: {server.global_memory_bank.shape}")
    projector_status = "saved" if server.global_projector else "None"
    print(f"Projector: {projector_status}")
    print(f"{'=' * 70}\n")
    
    return server


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated PatchCore')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--data_root', type=str, default='data/federated_data')
    parser.add_argument('--save_dir', type=str, default='checkpoints/federated_improved')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--aggregation', type=str, default='category_aware',
                       choices=['fedavg', 'fedprox', 'fairness_aware', 'category_aware'])
    parser.add_argument('--dp_enabled', action='store_true')
    parser.add_argument('--dp_epsilon', type=float, default=1.0)
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    run_federated_training_improved(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        data_root=args.data_root,
        save_dir=args.save_dir,
        device=device,
        aggregation=args.aggregation,
        dp_enabled=args.dp_enabled,
        dp_epsilon=args.dp_epsilon
    )