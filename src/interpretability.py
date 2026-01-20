#!/usr/bin/env python3
"""
Interpretability Module for PatchCore

Visualizes anomaly maps to understand model decisions.

PatchCore is inherently interpretable - anomaly maps show WHERE the model
detects anomalies based on patch distances from the normal memory bank.

Usage:
    python interpretability.py --checkpoint <model.joblib> --test_dir <path> --output_dir results/interpretability
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import argparse
import joblib
from tqdm import tqdm

from model import load_model
from preprocessing import get_eval_dataloader, get_transforms


def visualize_anomaly_map(image, anomaly_map, score, label, category, save_path=None):
    """
    Visualize original image with overlaid anomaly map.
    Similar to Grad-CAM, but specific for PatchCore.
    
    Args:
        image: Input image tensor
        anomaly_map: Anomaly heatmap
        score: Anomaly score
        label: Ground truth label
        category: Image category
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. Original image
    img_display = image.permute(1, 2, 0).numpy()
    img_display = (img_display * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    
    axes[0].imshow(img_display)
    axes[0].set_title(f'Original\n{category}')
    axes[0].axis('off')
    
    # 2. Anomaly map (raw)
    im = axes[1].imshow(anomaly_map, cmap='hot')
    axes[1].set_title(f'Anomaly Map\nScore: {score:.4f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # 3. Normalized anomaly map
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    axes[2].imshow(norm_map, cmap='jet')
    axes[2].set_title('Normalized Map')
    axes[2].axis('off')
    
    # 4. Overlay on image
    # Resize anomaly map to image size
    h, w = img_display.shape[:2]
    map_resized = F.interpolate(
        torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float(),
        size=(h, w), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    map_norm = (map_resized - map_resized.min()) / (map_resized.max() - map_resized.min() + 1e-8)
    heatmap = cm.jet(map_norm)[:, :, :3]
    
    overlay = 0.6 * img_display + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    axes[3].imshow(overlay)
    label_str = "ANOMALY" if label == 1 else "NORMAL"
    color = 'red' if label == 1 else 'green'
    axes[3].set_title(f'Overlay\nGT: {label_str}', color=color)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_interpretability_report(model, test_loader, device, output_dir, num_samples=20):
    """
    Generate complete interpretability report.
    
    Args:
        model: PatchCore model
        test_loader: DataLoader for test set
        device: Computation device
        output_dir: Directory to save outputs
        num_samples: Number of samples to visualize
        
    Returns:
        summary: Dictionary with summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("INTERPRETABILITY ANALYSIS")
    print(f"{'=' * 70}\n")
    
    model.eval()
    
    # Collect samples: anomalies and normals
    anomaly_samples = []
    normal_samples = []
    
    print("Collecting samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            images = batch['image']
            labels = batch['label'].numpy()
            categories = batch['category']
            
            scores, maps = model.compute_anomaly_score_batch(images, device)
            
            for i in range(len(images)):
                sample = {
                    'image': images[i],
                    'label': labels[i],
                    'category': categories[i],
                    'score': scores[i],
                    'map': maps[i]
                }
                
                if labels[i] == 1 and len(anomaly_samples) < num_samples:
                    anomaly_samples.append(sample)
                elif labels[i] == 0 and len(normal_samples) < num_samples:
                    normal_samples.append(sample)
                
                if len(anomaly_samples) >= num_samples and len(normal_samples) >= num_samples:
                    break
            
            if len(anomaly_samples) >= num_samples and len(normal_samples) >= num_samples:
                break
    
    # Generate visualizations
    print(f"\nGenerating {len(anomaly_samples)} anomaly visualizations...")
    for i, sample in enumerate(anomaly_samples):
        save_path = os.path.join(output_dir, 'visualizations', f'anomaly_{i:03d}_{sample["category"]}.png')
        visualize_anomaly_map(
            sample['image'], sample['map'], sample['score'],
            sample['label'], sample['category'], save_path
        )
    
    print(f"Generating {len(normal_samples)} normal visualizations...")
    for i, sample in enumerate(normal_samples):
        save_path = os.path.join(output_dir, 'visualizations', f'normal_{i:03d}_{sample["category"]}.png')
        visualize_anomaly_map(
            sample['image'], sample['map'], sample['score'],
            sample['label'], sample['category'], save_path
        )
    
    # Score distribution analysis
    print("\nAnalyzing score distributions...")
    anomaly_scores = [s['score'] for s in anomaly_samples]
    normal_scores = [s['score'] for s in normal_samples]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='green')
    axes[0].hist(anomaly_scores, bins=20, alpha=0.7, label='Anomaly', color='red')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Score Comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=150)
    plt.close()
    
    # Per-category analysis
    print("Analyzing per-category patterns...")
    category_scores = {}
    for sample in anomaly_samples + normal_samples:
        cat = sample['category']
        if cat not in category_scores:
            category_scores[cat] = {'normal': [], 'anomaly': []}
        
        if sample['label'] == 0:
            category_scores[cat]['normal'].append(sample['score'])
        else:
            category_scores[cat]['anomaly'].append(sample['score'])
    
    # Category plot
    fig, ax = plt.subplots(figsize=(12, 5))
    categories = sorted(category_scores.keys())
    x = np.arange(len(categories))
    width = 0.35
    
    normal_means = [np.mean(category_scores[c]['normal']) if category_scores[c]['normal'] else 0 for c in categories]
    anomaly_means = [np.mean(category_scores[c]['anomaly']) if category_scores[c]['anomaly'] else 0 for c in categories]
    
    ax.bar(x - width/2, normal_means, width, label='Normal', color='green', alpha=0.7)
    ax.bar(x + width/2, anomaly_means, width, label='Anomaly', color='red', alpha=0.7)
    ax.set_xlabel('Category')
    ax.set_ylabel('Mean Anomaly Score')
    ax.set_title('Per-Category Score Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_analysis.png'), dpi=150)
    plt.close()
    
    # Summary statistics
    summary = {
        'num_anomaly_samples': len(anomaly_samples),
        'num_normal_samples': len(normal_samples),
        'anomaly_score_mean': float(np.mean(anomaly_scores)),
        'anomaly_score_std': float(np.std(anomaly_scores)),
        'normal_score_mean': float(np.mean(normal_scores)),
        'normal_score_std': float(np.std(normal_scores)),
        'score_separation': float(np.mean(anomaly_scores) - np.mean(normal_scores)),
        'categories_analyzed': list(category_scores.keys())
    }
    
    import json
    with open(os.path.join(output_dir, 'interpretability_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("INTERPRETABILITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nScore Statistics:")
    print(f"  Normal samples:  {summary['normal_score_mean']:.4f} ± {summary['normal_score_std']:.4f}")
    print(f"  Anomaly samples: {summary['anomaly_score_mean']:.4f} ± {summary['anomaly_score_std']:.4f}")
    print(f"  Score separation: {summary['score_separation']:.4f}")
    print(f"\nInterpretation:")
    print(f"  - Higher anomaly scores indicate patches far from normal memory bank")
    print(f"  - Anomaly maps highlight regions with unusual patterns")
    print(f"  - Good separation ({summary['score_separation']:.4f}) = model distinguishes well")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - visualizations/  : {num_samples*2} sample visualizations")
    print(f"  - score_distribution.png")
    print(f"  - category_analysis.png")
    print(f"  - interpretability_summary.json")
    print(f"{'=' * 70}\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interpretability Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/interpretability')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    print(f"Loading test data from {args.test_dir}...")
    test_loader = get_eval_dataloader(args.test_dir, batch_size=8)
    
    generate_interpretability_report(
        model, test_loader, device,
        args.output_dir, args.num_samples
    )
