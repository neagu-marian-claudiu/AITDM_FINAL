import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
import os
import json
import argparse
import joblib

try:
    from model import PatchCoreImproved, load_model
except ImportError:
    from model import PatchCore as PatchCoreImproved, load_model

from preprocessing import get_eval_dataloader


def compute_tpr_at_tnr(y_true, y_score, target_tnr):
    """Calculate True Positive Rate at a given True Negative Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tnr = 1 - fpr
    idx = np.argmin(np.abs(tnr - target_tnr))
    return tpr[idx], thresholds[idx]


def evaluate_model(model, test_loader, device, img_size=256):
    """
    Complete evaluation with performance and fairness metrics.
    
    Args:
        model: PatchCore model
        test_loader: DataLoader for test set
        device: Computation device
        img_size: Image size (default: 256)
        
    Returns:
        results: Dictionary containing evaluation metrics
        img_scores: Array of image-level anomaly scores
        img_gt: Array of ground truth labels
        categories: Array of category labels
    """
    model.eval()
    
    img_scores, img_gt, categories = [], [], []
    pix_scores, pix_gt = [], []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch_scores, batch_maps = model.compute_anomaly_score_batch(
                batch['image'], device
            )
            
            labels = batch['label'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()
            has_mask = batch['has_mask'].cpu().numpy()
            batch_cats = batch['category']
            
            for i in range(len(batch_scores)):
                img_scores.append(batch_scores[i])
                img_gt.append(labels[i])
                categories.append(batch_cats[i])
                
                if has_mask[i] == 1:
                    m = torch.from_numpy(batch_maps[i]).unsqueeze(0).unsqueeze(0)
                    m = F.interpolate(m, size=(masks[i].shape[1], masks[i].shape[2]),
                                     mode='bilinear', align_corners=False)
                    pix_scores.extend(m.squeeze().numpy().flatten())
                    pix_gt.extend(masks[i].flatten())
    
    img_scores = np.array(img_scores)
    img_gt = np.array(img_gt)
    categories = np.array(categories)
    
    # Global metrics
    results = {
        'image_auroc': roc_auc_score(img_gt, img_scores),
        'image_ap': average_precision_score(img_gt, img_scores),
        'pixel_auroc': roc_auc_score(pix_gt, pix_scores) if pix_gt else None,
        'num_samples': len(img_gt),
        'num_anomalies': int(img_gt.sum()),
        'num_normal': len(img_gt) - int(img_gt.sum()),
    }
    
    # TPR at different TNR thresholds
    for tnr in [0.90, 0.95, 0.99]:
        tpr, thresh = compute_tpr_at_tnr(img_gt, img_scores, tnr)
        results[f'tpr_at_tnr_{int(tnr*100)}'] = tpr
    
    # Per-category metrics
    cat_results = []
    for cat in np.unique(categories):
        mask = categories == cat
        cat_labels = img_gt[mask]
        cat_scores = img_scores[mask]
        
        if len(np.unique(cat_labels)) >= 2:
            auroc = roc_auc_score(cat_labels, cat_scores)
            ap = average_precision_score(cat_labels, cat_scores)
            tpr90, _ = compute_tpr_at_tnr(cat_labels, cat_scores, 0.90)
            tpr95, _ = compute_tpr_at_tnr(cat_labels, cat_scores, 0.95)
            tpr99, _ = compute_tpr_at_tnr(cat_labels, cat_scores, 0.99)
        else:
            auroc = ap = tpr90 = tpr95 = tpr99 = None
        
        cat_results.append({
            'category': cat,
            'samples': int(mask.sum()),
            'anomalies': int(cat_labels.sum()),
            'auroc': auroc,
            'ap': ap,
            'tpr_90': tpr90,
            'tpr_95': tpr95,
            'tpr_99': tpr99,
        })
    
    results['per_category'] = cat_results
    
    # Fairness metrics
    aurocs = [r['auroc'] for r in cat_results if r['auroc'] is not None]
    if aurocs:
        results['auroc_mean'] = np.mean(aurocs)
        results['auroc_std'] = np.std(aurocs)
        results['auroc_min'] = np.min(aurocs)
        results['auroc_max'] = np.max(aurocs)
        results['auroc_range'] = np.max(aurocs) - np.min(aurocs)
        results['disparate_impact'] = np.min(aurocs) / np.max(aurocs)
    
    return results, img_scores, img_gt, categories


def print_results(results):
    """Display evaluation results in a formatted manner."""
    print(f"\n{'=' * 70}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\nGlobal Metrics:")
    print(f"  Image AUROC:     {results['image_auroc']:.4f}")
    print(f"  Image AP:        {results['image_ap']:.4f}")
    if results['pixel_auroc']:
        print(f"  Pixel AUROC:     {results['pixel_auroc']:.4f}")
    print(f"  TPR @ TNR=90%:   {results['tpr_at_tnr_90']*100:.1f}%")
    print(f"  TPR @ TNR=95%:   {results['tpr_at_tnr_95']*100:.1f}%")
    print(f"  TPR @ TNR=99%:   {results['tpr_at_tnr_99']*100:.1f}%")
    
    print(f"\nFairness Metrics:")
    if 'auroc_mean' in results:
        print(f"  AUROC Mean±Std:  {results['auroc_mean']:.4f} ± {results['auroc_std']:.4f}")
        print(f"  AUROC Range:     {results['auroc_range']:.4f}")
        print(f"  Disparate Impact:{results['disparate_impact']:.4f}")
        
        # Interpret fairness
        di = results['disparate_impact']
        if di >= 0.8:
            print(f"  -> EXCELLENT fairness (DI >= 0.8)")
        elif di >= 0.7:
            print(f"  -> GOOD fairness (DI >= 0.7)")
        elif di >= 0.6:
            print(f"  -> MODERATE fairness (DI >= 0.6)")
        else:
            print(f"  -> POOR fairness (DI < 0.6)")
    
    print(f"\nPer-Category Results:")
    print(f"{'Category':<20} {'Samples':>8} {'Anomalies':>10} {'AUROC':>8} {'TPR@95':>8}")
    print("-" * 60)
    for cat in results['per_category']:
        auroc_str = f"{cat['auroc']:.4f}" if cat['auroc'] else "N/A"
        tpr_str = f"{cat['tpr_95']*100:.1f}%" if cat['tpr_95'] else "N/A"
        print(f"{cat['category']:<20} {cat['samples']:>8} {cat['anomalies']:>10} {auroc_str:>8} {tpr_str:>8}")
    
    print(f"{'=' * 70}\n")


def evaluate_from_checkpoint(checkpoint_path, test_dir, device='cuda', 
                            output_dir='results', img_size=256):
    """
    Evaluate a model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dir: Path to test data directory
        device: Computation device
        output_dir: Directory to save results
        img_size: Image size for evaluation
        
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {os.path.basename(checkpoint_path)}")
    print(f"{'=' * 70}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    print(f"Memory bank: {model.memory_bank.shape}")
    if hasattr(model, 'category_stats') and model.category_stats:
        print("Category-adaptive sampling detected:")
        for cat, stats in sorted(model.category_stats.items()):
            print(f"  {cat}: {stats.get('target_patches', 'N/A')} patches")
    
    # Load test data
    test_loader = get_eval_dataloader(test_dir, batch_size=8, img_size=img_size)
    
    # Evaluate
    results, scores, labels, cats = evaluate_model(model, test_loader, device, img_size)
    
    # Print
    print_results(results)
    
    # Save
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    # JSON results
    json_results = {k: v for k, v in results.items() 
                   if not isinstance(v, (np.ndarray, list)) or k == 'per_category'}
    # Convert numpy values
    for k, v in json_results.items():
        if isinstance(v, (np.floating, np.integer)):
            json_results[k] = float(v)
        elif k == 'per_category':
            for cat in v:
                for ck, cv in cat.items():
                    if isinstance(cv, (np.floating, np.integer)):
                        cat[ck] = float(cv) if cv is not None else None
    
    results_path = os.path.join(output_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    # CSV per-category
    df = pd.DataFrame(results['per_category'])
    csv_path = os.path.join(output_dir, f'{model_name}_per_category.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PatchCore')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        device=device,
        output_dir=args.output_dir,
        img_size=args.img_size
    )