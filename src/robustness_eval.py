#!/usr/bin/env python3
"""
Robustness Evaluation Module

Evaluates model robustness to various perturbations.

Tests:
1. Gaussian Noise - random noise
2. Salt & Pepper Noise - corrupted pixels
3. Brightness variations - lighting changes
4. Blur - image blurring

Usage:
    python robustness_eval.py --checkpoint <model.joblib> --test_dir <path> --output_dir results/robustness
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from model import load_model
from preprocessing import AutoVIEvalDataset, get_transforms


class NoisyDataset(torch.utils.data.Dataset):
    """Dataset wrapper that adds perturbations."""
    
    def __init__(self, base_dataset, noise_type='none', noise_level=0.0):
        """
        Initialize noisy dataset.
        
        Args:
            base_dataset: Base evaluation dataset
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'brightness', 'blur')
            noise_level: Level of noise to apply
        """
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample['image']
        
        if self.noise_type == 'gaussian':
            image = self._add_gaussian_noise(image, self.noise_level)
        elif self.noise_type == 'salt_pepper':
            image = self._add_salt_pepper(image, self.noise_level)
        elif self.noise_type == 'brightness':
            image = self._adjust_brightness(image, self.noise_level)
        elif self.noise_type == 'blur':
            image = self._add_blur(image, self.noise_level)
        
        sample['image'] = image
        return sample
    
    def _add_gaussian_noise(self, image, std):
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * std
        noisy = image + noise
        return torch.clamp(noisy, -3, 3)  # Clip to reasonable range for normalized images
    
    def _add_salt_pepper(self, image, prob):
        """Add salt & pepper noise."""
        noisy = image.clone()
        mask = torch.rand_like(image) < prob
        salt = torch.rand_like(image) < 0.5
        
        noisy[mask & salt] = 3.0  # Salt (white, normalized)
        noisy[mask & ~salt] = -3.0  # Pepper (black, normalized)
        return noisy
    
    def _adjust_brightness(self, image, factor):
        """Adjust brightness. factor > 0 = brighter."""
        return image + factor
    
    def _add_blur(self, image, kernel_size):
        """Add blur. kernel_size = 1, 3, 5, etc."""
        if kernel_size < 2:
            return image
        
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        
        # Apply average pooling as blur
        padding = k // 2
        blurred = F.avg_pool2d(
            image.unsqueeze(0), 
            kernel_size=k, 
            stride=1, 
            padding=padding
        ).squeeze(0)
        return blurred


def evaluate_with_noise(model, base_dataset, device, noise_type, noise_level):
    """Evaluate model on data with perturbations."""
    noisy_dataset = NoisyDataset(base_dataset, noise_type, noise_level)
    loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=8, shuffle=False)
    
    all_scores = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            scores, _ = model.compute_anomaly_score_batch(batch['image'], device)
            all_scores.extend(scores)
            all_labels.extend(batch['label'].numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        return {'auroc': None, 'ap': None}
    
    return {
        'auroc': roc_auc_score(all_labels, all_scores),
        'ap': average_precision_score(all_labels, all_scores)
    }


def robustness_analysis(model, test_dir, device, output_dir):
    """Complete robustness analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'=' * 70}\n")
    
    # Load base dataset
    _, val_transform = get_transforms(256, augment=False)
    base_dataset = AutoVIEvalDataset(test_dir, transform=val_transform)
    
    # Define perturbations to test
    perturbations = {
        'gaussian': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        'salt_pepper': [0.0, 0.01, 0.02, 0.05, 0.1, 0.15],
        'brightness': [0.0, 0.2, 0.4, 0.6, -0.2, -0.4],
        'blur': [0, 3, 5, 7, 9, 11]
    }
    
    results = {}
    
    # Baseline (no noise)
    print("Evaluating baseline (no perturbation)...")
    baseline = evaluate_with_noise(model, base_dataset, device, 'none', 0)
    results['baseline'] = baseline
    print(f"  Baseline AUROC: {baseline['auroc']:.4f}")
    
    # Test each perturbation type
    for noise_type, levels in perturbations.items():
        print(f"\nTesting {noise_type} perturbation...")
        results[noise_type] = {}
        
        for level in tqdm(levels, desc=f"  {noise_type}"):
            metrics = evaluate_with_noise(model, base_dataset, device, noise_type, level)
            results[noise_type][str(level)] = metrics
            
            if metrics['auroc']:
                degradation = (baseline['auroc'] - metrics['auroc']) / baseline['auroc'] * 100
                print(f"    Level {level}: AUROC={metrics['auroc']:.4f} (degradation: {degradation:+.1f}%)")
    
    # Generate visualizations
    print("\nGenerating robustness plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gaussian Noise
    ax = axes[0, 0]
    levels = [float(l) for l in results['gaussian'].keys()]
    aurocs = [results['gaussian'][str(l)]['auroc'] for l in levels]
    ax.plot(levels, aurocs, 'o-', linewidth=2, markersize=8, color='blue')
    ax.axhline(y=baseline['auroc'], color='green', linestyle='--', label='Baseline')
    ax.set_xlabel('Gaussian Noise Std')
    ax.set_ylabel('AUROC')
    ax.set_title('Robustness to Gaussian Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Plot 2: Salt & Pepper
    ax = axes[0, 1]
    levels = [float(l) for l in results['salt_pepper'].keys()]
    aurocs = [results['salt_pepper'][str(l)]['auroc'] for l in levels]
    ax.plot(levels, aurocs, 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=baseline['auroc'], color='green', linestyle='--', label='Baseline')
    ax.set_xlabel('Salt & Pepper Probability')
    ax.set_ylabel('AUROC')
    ax.set_title('Robustness to Salt & Pepper Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Plot 3: Brightness
    ax = axes[1, 0]
    levels = [float(l) for l in results['brightness'].keys()]
    aurocs = [results['brightness'][str(l)]['auroc'] for l in levels]
    ax.plot(levels, aurocs, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=baseline['auroc'], color='green', linestyle='--', label='Baseline')
    ax.set_xlabel('Brightness Change')
    ax.set_ylabel('AUROC')
    ax.set_title('Robustness to Brightness Variations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Plot 4: Blur
    ax = axes[1, 1]
    levels = [float(l) for l in results['blur'].keys()]
    aurocs = [results['blur'][str(l)]['auroc'] for l in levels]
    ax.plot(levels, aurocs, 'o-', linewidth=2, markersize=8, color='red')
    ax.axhline(y=baseline['auroc'], color='green', linestyle='--', label='Baseline')
    ax.set_xlabel('Blur Kernel Size')
    ax.set_ylabel('AUROC')
    ax.set_title('Robustness to Blur')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_curves.png'), dpi=150)
    plt.close()
    
    # Summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate max degradation for each noise type
    degradations = {}
    for noise_type in ['gaussian', 'salt_pepper', 'brightness', 'blur']:
        max_deg = 0
        for level, metrics in results[noise_type].items():
            if metrics['auroc']:
                deg = (baseline['auroc'] - metrics['auroc']) / baseline['auroc'] * 100
                max_deg = max(max_deg, deg)
        degradations[noise_type] = max_deg
    
    bars = ax.bar(degradations.keys(), degradations.values(), color=['blue', 'orange', 'purple', 'red'], alpha=0.7)
    ax.set_ylabel('Max AUROC Degradation (%)')
    ax.set_title('Vulnerability to Different Perturbations')
    ax.axhline(y=10, color='green', linestyle='--', label='10% threshold')
    ax.axhline(y=20, color='orange', linestyle='--', label='20% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, degradations.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vulnerability_summary.png'), dpi=150)
    plt.close()
    
    # Save results
    with open(os.path.join(output_dir, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nBaseline AUROC: {baseline['auroc']:.4f}")
    print(f"\nMax Degradation by Perturbation Type:")
    for noise_type, deg in degradations.items():
        if deg < 10:
            status = "Robust"
        elif deg < 20:
            status = "Sensitive"
        else:
            status = "Vulnerable"
        print(f"  {noise_type:<15}: {deg:>5.1f}%  {status}")
    
    overall = np.mean(list(degradations.values()))
    print(f"\nOverall Robustness Score: {100 - overall:.1f}% (lower degradation = better)")
    
    if overall < 10:
        print("-> Model is HIGHLY ROBUST to perturbations")
    elif overall < 20:
        print("-> Model is MODERATELY ROBUST to perturbations")
    else:
        print("-> Model is SENSITIVE to perturbations")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - robustness_curves.png")
    print(f"  - vulnerability_summary.png")
    print(f"  - robustness_results.json")
    print(f"{'=' * 70}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robustness Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/robustness')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    robustness_analysis(model, args.test_dir, device, args.output_dir)
    