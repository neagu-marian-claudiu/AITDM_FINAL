#!/usr/bin/env python3
"""
Trade-offs Analysis Module

Analyzes trade-offs between:
- Accuracy vs Fairness
- Performance vs Privacy (DP)
- Centralized vs Federated

Generates report and visualizations.

Usage:
    python tradeoffs_analysis.py --results_dir results --output_dir results/tradeoffs
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import argparse


def load_all_results(results_dir):
    """Load all results from the specified directory."""
    results = {}
    
    # Search patterns
    patterns = [
        ('Centralized', 'results/improved/*_results.json'),
        ('Centralized', 'centralized_*_results.json'),
        ('Aggregated', 'results/aggregated/*_results.json'),
        ('FedAvg', 'results/federated/fedavg/*_results.json'),
        ('FedProx', 'results/federated/fedprox/*_results.json'),
        ('Category-Aware', 'results/federated/category_aware/*_results.json'),
        ('Fairness-Aware', 'results/federated/fairness/*_results.json'),
    ]
    
    for name, pattern in patterns:
        for filepath in glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                # Use filename to distinguish if multiple matches
                key = name
                if key in results:
                    key = f"{name} ({os.path.basename(filepath)})"
                results[key] = data
                print(f"Loaded: {key} from {filepath}")
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
    
    return results


def analyze_accuracy_vs_fairness(results, output_dir):
    """Analyze trade-off between accuracy and fairness."""
    print("\n" + "=" * 70)
    print("ACCURACY VS FAIRNESS TRADE-OFF ANALYSIS")
    print("=" * 70)
    
    data = []
    for name, res in results.items():
        if res.get('image_auroc') and res.get('disparate_impact'):
            data.append({
                'Model': name,
                'AUROC': res['image_auroc'],
                'AP': res.get('image_ap', 0),
                'Disparate Impact': res['disparate_impact'],
                'AUROC Range': res.get('auroc_range', 0),
                'TPR@95': res.get('tpr_at_tnr_95', 0) * 100
            })
    
    if not data:
        print("Error: No data available for analysis")
        return None
    
    df = pd.DataFrame(data)
    
    # Calculate trade-off score (higher = better balance)
    # Normalize both metrics to 0-1 range
    auroc_norm = (df['AUROC'] - df['AUROC'].min()) / (df['AUROC'].max() - df['AUROC'].min() + 1e-8)
    di_norm = (df['Disparate Impact'] - df['Disparate Impact'].min()) / (df['Disparate Impact'].max() - df['Disparate Impact'].min() + 1e-8)
    
    # Trade-off score: geometric mean of normalized metrics
    df['Trade-off Score'] = np.sqrt(auroc_norm * di_norm)
    
    # Sort by trade-off score
    df = df.sort_values('Trade-off Score', ascending=False)
    
    print("\n" + df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    scatter = ax.scatter(df['AUROC'], df['Disparate Impact'], c=colors, s=150, alpha=0.8)
    
    for i, row in df.iterrows():
        ax.annotate(row['Model'], (row['AUROC'], row['Disparate Impact']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('AUROC (Accuracy)', fontsize=11)
    ax.set_ylabel('Disparate Impact (Fairness)', fontsize=11)
    ax.set_title('Accuracy vs Fairness Trade-off', fontsize=12, fontweight='bold')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='DI=0.8 (good)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='DI=0.7 (fair)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Trade-off scores
    ax = axes[1]
    colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in df['Trade-off Score']]
    bars = ax.barh(df['Model'], df['Trade-off Score'], color=colors, alpha=0.7)
    ax.set_xlabel('Trade-off Score', fontsize=11)
    ax.set_title('Model Rankings\n(Balance of Accuracy & Fairness)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Parallel coordinates (normalized)
    ax = axes[2]
    metrics = ['AUROC', 'Disparate Impact', 'TPR@95']
    for i, row in df.iterrows():
        values = [
            (row['AUROC'] - df['AUROC'].min()) / (df['AUROC'].max() - df['AUROC'].min() + 1e-8),
            (row['Disparate Impact'] - df['Disparate Impact'].min()) / (df['Disparate Impact'].max() - df['Disparate Impact'].min() + 1e-8),
            (row['TPR@95'] - df['TPR@95'].min()) / (df['TPR@95'].max() - df['TPR@95'].min() + 1e-8)
        ]
        ax.plot(metrics, values, 'o-', label=row['Model'], linewidth=2, markersize=8)
    
    ax.set_ylabel('Normalized Score', fontsize=11)
    ax.set_title('Multi-metric Comparison', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_fairness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df


def generate_tradeoffs_report(results, output_dir):
    """Generate complete trade-offs report."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("TRADE-OFFS ANALYSIS REPORT")
    print(f"{'=' * 70}")
    print(f"Models found: {len(results)}")
    
    # Run analyses
    df_tradeoff = analyze_accuracy_vs_fairness(results, output_dir)
    
    # Generate summary report
    report = {
        'num_models': len(results),
        'models': list(results.keys()),
        'findings': []
    }
    
    if df_tradeoff is not None:
        best_tradeoff = df_tradeoff.iloc[0]
        report['best_tradeoff_model'] = best_tradeoff['Model']
        report['best_tradeoff_score'] = float(best_tradeoff['Trade-off Score'])
        report['findings'].append(
            f"Best accuracy-fairness balance: {best_tradeoff['Model']} "
            f"(score: {best_tradeoff['Trade-off Score']:.3f})"
        )
    
    # Key insights
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")
    
    for i, finding in enumerate(report['findings'], 1):
        print(f"{i}. {finding}")
    
    # Save report
    with open(os.path.join(output_dir, 'tradeoffs_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save summary table
    if df_tradeoff is not None:
        df_tradeoff.to_csv(os.path.join(output_dir, 'tradeoffs_summary.csv'), index=False)
    
    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  - {output_dir}/accuracy_vs_fairness.png")
    print(f"  - {output_dir}/tradeoffs_report.json")
    print(f"  - {output_dir}/tradeoffs_summary.csv")
    print(f"{'=' * 70}\n")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trade-offs Analysis')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='results/tradeoffs')
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_all_results(args.results_dir)
    
    if not results:
        print("Error: No results found! Run evaluation first.")
        exit(1)
    
    generate_tradeoffs_report(results, args.output_dir)
