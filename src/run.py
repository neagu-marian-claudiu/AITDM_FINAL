#!/usr/bin/env python3

import subprocess
import sys
import os
from datetime import datetime
import time
import shutil


def run_command(cmd, desc, phase=""):
    """
    Run command and handle errors.
    
    Args:
        cmd: Command to execute
        desc: Description of the task
        phase: Phase identifier (optional)
        
    Returns:
        duration: Execution time in seconds
    """
    if phase:
        print(f"\n{'=' * 70}")
        print(f"{phase}")
        print(f"{'=' * 70}\n")
    
    print(f"Running: {desc}...")
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    duration = time.time() - start
    
    if result.returncode != 0:
        print(f"\nError: {desc} failed!")
        print(f"Exit code: {result.returncode}")
        sys.exit(1)
    
    print(f"\nCompleted: {desc} ({duration/60:.1f} minutes)")
    return duration


def main():
    print("""
================================================================================
         OVERNIGHT TRAINING - ALL METHODS (WSL OPTIMIZED)
================================================================================

Configuration:
  - Batch size: 4
  - Patches per category: 10000
  - Image size: 256
  - Device: auto (CUDA if available)

This will train:
  1. Centralized Improved (baseline)
  2. 5 Individual Clients
  3. Aggregated (Category-Aware)
  4. Federated FedAvg
  5. Federated FedProx
  6. Federated Category-Aware

Estimated time: 4-6 hours
================================================================================
""")
    
    input("Press Enter to start or Ctrl+C to cancel...")
    
    start_time = datetime.now()
    print(f"\nStarted at: {start_time}")
    
    timings = {}
    
    # Phase 1: Centralized Improved
    duration = run_command(
        [sys.executable, "src/train_standalone.py",
         "--data_root", "data/federated_data",
         "--save_dir", "checkpoints/improved",
         "--batch_size", "4",
         "--patches_per_category", "10000",
         "--img_size", "256",
         "--device", "auto"],
        "Centralized Improved training",
        "PHASE 1/8: CENTRALIZED IMPROVED TRAINING"
    )
    timings['centralized_train'] = duration
    
    # Phase 2: Evaluate Centralized
    duration = run_command(
        [sys.executable, "src/evaluate.py",
         "--checkpoint", "checkpoints/improved/centralized_improved.joblib",
         "--test_dir", "data/test_data_centralized",
         "--output_dir", "results/improved",
         "--img_size", "256",
         "--device", "auto"],
        "Centralized model evaluation",
        "PHASE 2/8: EVALUATE CENTRALIZED MODEL"
    )
    timings['centralized_eval'] = duration
    
    # Phase 3: Train Individual Clients
    print(f"\n{'=' * 70}")
    print("PHASE 3/8: TRAIN INDIVIDUAL CLIENTS (5 clients)")
    print(f"{'=' * 70}\n")
    
    client_durations = []
    for i in range(5):
        print(f"\n{'-' * 70}")
        print(f"Training Client {i}...")
        print(f"{'-' * 70}\n")
        
        duration = run_command(
            [sys.executable, "src/train_standalone.py",
             "--client_id", str(i),
             "--batch_size", "4",
             "--patches_per_category", "10000",
             "--img_size", "256",
             "--data_root", "data/federated_data",
             "--save_dir", "checkpoints/clients",
             "--device", "auto"],
            f"Client {i} training",
            ""
        )
        client_durations.append(duration)
    
    timings['clients_train'] = sum(client_durations)
    print(f"\nAll 5 clients trained ({sum(client_durations)/60:.1f} minutes)")
    
    # Phase 4: Aggregate Clients
    os.makedirs("checkpoints/aggregated", exist_ok=True)
    
    duration = run_command(
        [sys.executable, "src/aggregate_models.py",
         "--checkpoints",
         "checkpoints/clients/client_0_improved.joblib",
         "checkpoints/clients/client_1_improved.joblib",
         "checkpoints/clients/client_2_improved.joblib",
         "checkpoints/clients/client_3_improved.joblib",
         "checkpoints/clients/client_4_improved.joblib",
         "--output", "checkpoints/aggregated/federated_category_aware.joblib",
         "--method", "category_aware",
         "--target_size", "50000"],
        "Category-Aware aggregation",
        "PHASE 4/8: AGGREGATE CLIENTS (Category-Aware)"
    )
    timings['aggregation'] = duration
    
    # Phase 5: Evaluate Aggregated
    duration = run_command(
        [sys.executable, "src/evaluate.py",
         "--checkpoint", "checkpoints/aggregated/federated_category_aware.joblib",
         "--test_dir", "data/test_data_centralized",
         "--output_dir", "results/aggregated",
         "--img_size", "256",
         "--device", "auto"],
        "Aggregated model evaluation",
        "PHASE 5/8: EVALUATE AGGREGATED MODEL"
    )
    timings['aggregated_eval'] = duration
    
    # Phase 6: Federated - FedAvg
    duration = run_command(
        [sys.executable, "src/federated_train.py",
         "--num_clients", "5",
         "--num_rounds", "3",
         "--data_root", "data/federated_data",
         "--save_dir", "checkpoints/federated/fedavg",
         "--device", "auto",
         "--aggregation", "fedavg"],
        "Federated FedAvg training",
        "PHASE 6/8: FEDERATED TRAINING - FedAvg"
    )
    timings['fedavg_train'] = duration
    
    duration = run_command(
        [sys.executable, "src/evaluate.py",
         "--checkpoint", "checkpoints/federated/fedavg/global_final_improved.joblib",
         "--test_dir", "data/test_data_centralized",
         "--output_dir", "results/federated/fedavg",
         "--img_size", "256",
         "--device", "auto"],
        "FedAvg evaluation",
        ""
    )
    timings['fedavg_eval'] = duration
    
    # Phase 7: Federated - FedProx
    duration = run_command(
        [sys.executable, "src/federated_train.py",
         "--num_clients", "5",
         "--num_rounds", "3",
         "--data_root", "data/federated_data",
         "--save_dir", "checkpoints/federated/fedprox",
         "--device", "auto",
         "--aggregation", "fedprox"],
        "Federated FedProx training",
        "PHASE 7/8: FEDERATED TRAINING - FedProx"
    )
    timings['fedprox_train'] = duration
    
    duration = run_command(
        [sys.executable, "src/evaluate.py",
         "--checkpoint", "checkpoints/federated/fedprox/global_final_improved.joblib",
         "--test_dir", "data/test_data_centralized",
         "--output_dir", "results/federated/fedprox",
         "--img_size", "256",
         "--device", "auto"],
        "FedProx evaluation",
        ""
    )
    timings['fedprox_eval'] = duration
    
    # Phase 8: Federated - Category-Aware
    duration = run_command(
        [sys.executable, "src/federated_train.py",
         "--num_clients", "5",
         "--num_rounds", "3",
         "--data_root", "data/federated_data",
         "--save_dir", "checkpoints/federated/category_aware",
         "--device", "auto",
         "--aggregation", "category_aware"],
        "Federated Category-Aware training",
        "PHASE 8/8: FEDERATED TRAINING - Category-Aware (BEST)"
    )
    timings['category_aware_train'] = duration
    
    duration = run_command(
        [sys.executable, "src/evaluate.py",
         "--checkpoint", "checkpoints/federated/category_aware/global_final_improved.joblib",
         "--test_dir", "data/test_data_centralized",
         "--output_dir", "results/federated/category_aware",
         "--img_size", "256",
         "--device", "auto"],
        "Category-Aware evaluation",
        ""
    )
    timings['category_aware_eval'] = duration
    
    # Comparison Report
    print(f"\n{'=' * 70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'=' * 70}\n")
    
    try:
        run_command(
            [sys.executable, "src/compare_models.py"],
            "Comparison report generation",
            ""
        )
    except:
        print("Warning: Comparison failed, but all models are trained")
    
    # Final Summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    
    print(f"\n{'=' * 70}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'=' * 70}\n")
    print(f"Started:  {start_time}")
    print(f"Finished: {end_time}")
    print(f"Duration: {hours}h {minutes}m")
    
    print("\nTime breakdown:")
    for key, duration in timings.items():
        print(f"  {key:<25} {duration/60:>6.1f} min")
    
    print("\nModels saved:")
    print("  * checkpoints/improved/centralized_improved.joblib")
    print("  * checkpoints/clients/client_*_improved.joblib (5 clients)")
    print("  * checkpoints/aggregated/federated_category_aware.joblib")
    print("  * checkpoints/federated/fedavg/global_final_improved.joblib")
    print("  * checkpoints/federated/fedprox/global_final_improved.joblib")
    print("  * checkpoints/federated/category_aware/global_final_improved.joblib")
    
    print("\nResults:")
    print("  * results/improved/")
    print("  * results/aggregated/")
    print("  * results/federated/fedavg/")
    print("  * results/federated/fedprox/")
    print("  * results/federated/category_aware/")
    
    print("\nComparison: model_comparison.csv & model_comparison.png")
    print("\nSUCCESS!\n")


if __name__ == "__main__":
    main()
