#!/usr/bin/env python3
"""
Complete Analysis

Runs all analyses:
1. Fix models (if needed)
2. Interpretability
3. Robustness
4. Trade-offs

Usage:
    python run_analysis.py
"""

import subprocess
import sys
import os
from datetime import datetime


def run_command(cmd, desc):
    """
    Run command and display output.
    
    Args:
        cmd: Command to execute
        desc: Description of the task
        
    Returns:
        returncode: Exit code of the command
    """
    print(f"\n{'=' * 70}")
    print(f"Running: {desc}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Warning: {desc} completed with warnings/errors")
    else:
        print(f"Success: {desc} completed successfully")
    
    return result.returncode


def main():
    print("""
================================================================================
           TRUSTWORTHINESS ANALYSIS
================================================================================
  1. Interpretability Analysis
  2. Robustness Evaluation
  3. Trade-offs Analysis
================================================================================
""")
    
    # Check if required files exist
    checkpoint = "checkpoints/improved/centralized_improved.joblib"
    test_dir = "data/test_data_centralized"
    
    if not os.path.exists(checkpoint):
        # Try alternative
        checkpoint = "checkpoints/federated/fedavg/global_final_improved.joblib"
    
    if not os.path.exists(checkpoint):
        print("Error: No model checkpoint found!")
        print("Please ensure you have trained models in checkpoints/")
        return
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    print(f"Using checkpoint: {checkpoint}")
    print(f"Test directory: {test_dir}")
    
    input("\nPress Enter to start analysis or Ctrl+C to cancel...")
    
    start_time = datetime.now()
    
    # 1. Fix models first
    print("\n" + "=" * 70)
    print("STEP 0: Fixing models (if needed)")
    print("=" * 70)
    
    if os.path.exists("fix_all_models.py"):
        run_command([sys.executable, "fix_all_models.py"], "Fix models")
    
    # 2. Interpretability
    run_command(
        [sys.executable, "src/interpretability.py",
         "--checkpoint", checkpoint,
         "--test_dir", test_dir,
         "--output_dir", "results/interpretability",
         "--num_samples", "15",
         "--device", "auto"],
        "Interpretability Analysis"
    )
    
    # 3. Robustness
    run_command(
        [sys.executable, "src/robustness_eval.py",
         "--checkpoint", checkpoint,
         "--test_dir", test_dir,
         "--output_dir", "results/robustness",
         "--device", "auto"],
        "Robustness Evaluation"
    )
    
    # 4. Trade-offs
    run_command(
        [sys.executable, "src/tradeoffs_analysis.py",
         "--results_dir", "results",
         "--output_dir", "results/tradeoffs"],
        "Trade-offs Analysis"
    )
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'=' * 70}")
    print("STAGE 2 ANALYSIS COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nDuration: {duration/60:.1f} minutes")
    print(f"\nOutput directories:")
    print(f"  results/interpretability/  - Anomaly visualizations")
    print(f"  results/robustness/        - Robustness curves")
    print(f"  results/tradeoffs/         - Trade-off analysis")
    
    print(f"\nKey files for report:")
    print(f"  - results/interpretability/score_distribution.png")
    print(f"  - results/interpretability/visualizations/*.png")
    print(f"  - results/robustness/robustness_curves.png")
    print(f"  - results/robustness/vulnerability_summary.png")
    print(f"  - results/tradeoffs/accuracy_vs_fairness.png")
    print(f"  - results/tradeoffs/tradeoffs_report.json")
    
    print(f"\n{'=' * 70}")
    print("Stage 2 requirements covered:")
    print("  * Differential Privacy (implemented in federated_train.py)")
    print("  * Fairness-aware training (Lipschitz constraints, category-aware)")
    print("  * Fairness evaluation (Disparate Impact, per-category metrics)")
    print("  * Robustness evaluation (noise, blur, brightness)")
    print("  * Interpretability (anomaly maps visualization)")
    print("  * Trade-offs analysis (accuracy vs fairness)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
