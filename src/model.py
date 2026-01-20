import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm
import joblib


class PatchCoreImproved(nn.Module):
    """
    PatchCore with:
    - Optimized multi-scale features
    - Adaptive coreset per category
    - Memory-efficient processing
    """
    
    def __init__(self, backbone='wide_resnet50', layer_indices=[2, 3]):
        """
        Initialize PatchCore model.
        
        Args:
            backbone: Backbone network ('wide_resnet50' or 'resnet18')
            layer_indices: Indices of layers to extract features from
        """
        super().__init__()
        
        if backbone == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.layer_indices = layer_indices
        self.backbone_name = backbone
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.memory_bank = None
        self.projector = None
        
        # Store per-category statistics for adaptive sampling
        self.category_stats = {}
    
    def extract_features(self, x):
        """Extract multi-scale features from input images."""
        features = []
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 1 in self.layer_indices:
            features.append(x)
        
        x = self.backbone.layer2(x)
        if 2 in self.layer_indices:
            features.append(x)
        
        x = self.backbone.layer3(x)
        if 3 in self.layer_indices:
            features.append(x)
        
        x = self.backbone.layer4(x)
        if 4 in self.layer_indices:
            features.append(x)
        
        return features
    
    def aggregate_features(self, features):
        """Aggregate features from different scales."""
        # Use the largest feature map as base
        target_size = features[0].shape[2:]
        
        aggregated = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # L2 normalize features for better distance metrics
            feat = F.normalize(feat, p=2, dim=1)
            aggregated.append(feat)
        
        aggregated = torch.cat(aggregated, dim=1)
        B, C, H, W = aggregated.shape
        patches = aggregated.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        return patches, (H, W)
    
    def fit(self, train_loader, device, patches_per_category=25000, 
            L=0.5, lam=0.3, adaptive_sampling=True):
        """
        Train the model with improved coreset selection.
        
        Args:
            train_loader: DataLoader for training data
            device: Computation device
            patches_per_category: Base number of patches per category
            L: Lipschitz constant for fairness constraint
            lam: Weight for fairness penalty
            adaptive_sampling: Whether to use adaptive sampling per category
            
        Returns:
            self: Trained model
        """
        self.to(device)
        self.eval()
        
        cat_patches = {}
        cat_counts = {}
        
        # First pass: collect statistics
        print("Pass 1: Collecting category statistics...")
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Statistics", leave=False):
                categories = batch['category']
                for cat in categories:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        print(f"\nCategory distribution:")
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat}: {count} images")
        
        # Calculate adaptive patches per category
        if adaptive_sampling:
            total_images = sum(cat_counts.values())
            base_patches = patches_per_category
            
            for cat, count in cat_counts.items():
                # More patches for categories with fewer images
                ratio = total_images / (count * len(cat_counts))
                adaptive_patches = int(base_patches * min(ratio, 2.0))
                self.category_stats[cat] = {
                    'target_patches': adaptive_patches,
                    'image_count': count
                }
                print(f"  {cat}: target {adaptive_patches} patches (ratio: {ratio:.2f})")
        else:
            for cat in cat_counts.keys():
                self.category_stats[cat] = {
                    'target_patches': patches_per_category,
                    'image_count': cat_counts[cat]
                }
        
        # Second pass: extract features with memory management
        print("\nPass 2: Extracting features...")
        buffer_multiplier = 3  # Reduced from 5 for WSL safety
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extracting", leave=False):
                images = batch['image'].to(device)
                categories = batch['category']
                
                # Process in smaller chunks if batch is large
                if len(images) > 16:
                    chunks = torch.split(images, 16)
                    cat_chunks = [categories[i:i+16] for i in range(0, len(categories), 16)]
                else:
                    chunks = [images]
                    cat_chunks = [categories]
                
                for img_chunk, cat_chunk in zip(chunks, cat_chunks):
                    features = self.extract_features(img_chunk)
                    patches, _ = self.aggregate_features(features)
                    patches = patches.cpu()
                    
                    for i, cat in enumerate(cat_chunk):
                        if cat not in cat_patches:
                            cat_patches[cat] = []
                        cat_patches[cat].append(patches[i])
                        
                        # Memory protection
                        target = self.category_stats[cat]['target_patches']
                        current_len = sum(len(p) for p in cat_patches[cat])
                        buffer_limit = target * buffer_multiplier
                        
                        if current_len > buffer_limit:
                            merged = torch.cat(cat_patches[cat], dim=0)
                            keep_num = target * 2
                            idx = torch.randperm(len(merged))[:keep_num]
                            cat_patches[cat] = [merged[idx]]
                            torch.cuda.empty_cache() if device == 'cuda' else None
        
        # Final selection with improved coreset
        print(f"\nFinal coreset selection (L={L}, lambda={lam})...")
        final_bank = []
        
        for cat, patches_list in cat_patches.items():
            cat_all = torch.cat(patches_list, dim=0)
            target = self.category_stats[cat]['target_patches']
            
            print(f"\n  {cat}: {len(cat_all)} candidates -> {target} patches")
            
            # Pre-selection: random sample 2x target
            num_candidates = min(len(cat_all), target * 2)
            if len(cat_all) > num_candidates:
                idx_pre = torch.randperm(len(cat_all))[:num_candidates]
                candidates = cat_all[idx_pre]
            else:
                candidates = cat_all
            
            # Improved coreset selection
            selected = self._select_coreset_improved(
                candidates, target, L, lam
            )
            final_bank.append(selected)
            
            print(f"    Selected: {len(selected)} patches")
        
        self.memory_bank = torch.cat(final_bank, dim=0)
        print(f"\nTotal memory bank: {self.memory_bank.shape}")
        
        # Dimensionality reduction
        if self.memory_bank.shape[1] > 128:
            print("Applying dimensionality reduction...")
            self.projector = SparseRandomProjection(n_components=128, random_state=42)
            self.memory_bank = torch.tensor(
                self.projector.fit_transform(self.memory_bank.numpy())
            ).float()
            print(f"After projection: {self.memory_bank.shape}")
        
        return self
    
    def _select_coreset_improved(self, candidates, target, L, lam):
        """
        Improved coreset selection with:
        - Better diversity metric
        - Fairness constraints
        - Greedy approximation for large sets
        """
        n = len(candidates)
        
        if n <= target:
            return candidates
        
        # For very large candidate sets, use greedy k-center
        # WSL-safe: use greedy for anything over 5000 candidates
        if n > 5000:
            print(f"    Using greedy k-center for {n} candidates (WSL-safe)")
            return self._greedy_kcenter(candidates, target)
        
        # Distance matrix
        dist_matrix = torch.cdist(candidates, candidates)
        
        # Diversity score (distance to nearest neighbors)
        k = min(5, n - 1)
        dist_with_inf = dist_matrix + torch.eye(n) * 1e6
        knn_dist, _ = torch.topk(dist_with_inf, k=k, largest=False, dim=1)
        diversity_score = knn_dist.mean(dim=1)
        
        # Fairness constraint (Lipschitz)
        score_diff = torch.abs(diversity_score.unsqueeze(0) - diversity_score.unsqueeze(1))
        violations = torch.clamp(score_diff - L * dist_matrix, min=0)
        fairness_penalty = violations.sum(dim=1)
        
        # Coverage score (maximize distance to all others)
        coverage_score = dist_matrix.mean(dim=1)
        
        # Combined objective: diversity + coverage - fairness_penalty
        total_score = diversity_score + coverage_score - lam * fairness_penalty
        
        # Select top-k
        _, best_idx = torch.topk(total_score, target, largest=True)
        
        return candidates[best_idx]
    
    def _greedy_kcenter(self, candidates, k):
        """
        Ultra-fast greedy k-center for very large sets (WSL optimized).
        Uses batched distance computation and early stopping.
        """
        n = len(candidates)
        
        # For very large sets, use hybrid approach
        if n > 15000:
            print(f"      Ultra-large set ({n}), using fast sampling...")
            # Step 1: Random sample to 15000 first
            sample_idx = np.random.choice(n, 15000, replace=False)
            sampled = candidates[sample_idx]
            
            # Step 2: Greedy k-center on sample
            selected_from_sample = self._greedy_kcenter_fast(sampled, k)
            return selected_from_sample
        else:
            return self._greedy_kcenter_fast(candidates, k)
    
    def _greedy_kcenter_fast(self, candidates, k):
        """Fast greedy k-center with chunked processing."""
        n = len(candidates)
        selected_idx = [np.random.randint(n)]
        
        # Track minimum distances to selected set
        min_distances = torch.full((n,), float('inf'))
        
        # Chunk size for processing (WSL-safe)
        chunk_size = 5000
        
        for iteration in range(k - 1):
            # Update distances only for new selected point
            new_point = candidates[selected_idx[-1]].unsqueeze(0)
            
            # Process in chunks to avoid OOM
            for start_idx in range(0, n, chunk_size):
                end_idx = min(start_idx + chunk_size, n)
                chunk = candidates[start_idx:end_idx]
                
                # Distance to new point
                dist_to_new = torch.cdist(chunk, new_point).squeeze(1)
                
                # Update minimum distances
                min_distances[start_idx:end_idx] = torch.minimum(
                    min_distances[start_idx:end_idx],
                    dist_to_new
                )
            
            # Find farthest point
            farthest_idx = min_distances.argmax().item()
            selected_idx.append(farthest_idx)
            
            # Progress indicator for large sets
            if (iteration + 1) % 1000 == 0 or iteration == 0:
                print(f"      Progress: {iteration+1}/{k-1} points selected")
        
        return candidates[selected_idx]
    
    def compute_anomaly_score_batch(self, x, device):
        """Calculate anomaly scores (optimized for WSL)."""
        self.eval()
        
        with torch.no_grad():
            features = self.extract_features(x.to(device))
            patches, (H, W) = self.aggregate_features(features)
            
            B, N, C = patches.shape
            patches_flat = patches.reshape(-1, C).cpu()
            
            if self.projector is not None:
                patches_flat = torch.tensor(
                    self.projector.transform(patches_flat.numpy())
                ).float()
            
            # Process in chunks to avoid OOM
            chunk_size = 5000
            all_min_distances = []
            
            for i in range(0, len(patches_flat), chunk_size):
                chunk = patches_flat[i:i+chunk_size]
                distances = torch.cdist(chunk, self.memory_bank)
                min_distances, _ = distances.min(dim=1)
                all_min_distances.append(min_distances)
            
            min_distances = torch.cat(all_min_distances)
            
            anomaly_maps = min_distances.reshape(B, H, W)
            scores = anomaly_maps.reshape(B, -1).max(dim=1)[0].cpu().numpy()
            
            return scores, anomaly_maps.cpu().numpy()


def save_model(model, filepath):
    """Save model to file."""
    state = {
        'memory_bank': model.memory_bank,
        'projector': model.projector,
        'layer_indices': model.layer_indices,
        'backbone_name': model.backbone_name,
        'category_stats': model.category_stats
    }
    joblib.dump(state, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath, device):
    """Load model from file."""
    state = joblib.load(filepath)
    model = PatchCoreImproved(
        backbone=state['backbone_name'],
        layer_indices=state['layer_indices']
    ).to(device)
    model.memory_bank = state['memory_bank']
    model.projector = state['projector']
    model.category_stats = state.get('category_stats', {})
    return model