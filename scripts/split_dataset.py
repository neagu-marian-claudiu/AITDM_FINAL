#!/usr/bin/env python3

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class DatasetFormat(Enum):
    """Dataset format types in source data"""
    MVTEC = "mvtec"
    SIMPLIFIED = "simplified"


@dataclass
class CategoryConfig:
    """Configuration for a single category"""
    name: str
    format: DatasetFormat
    anomaly_rate: float
    has_train_anomalies: bool = False
    anomaly_types: List[str] = field(default_factory=list)


# Dataset category configurations
CATEGORY_CONFIGS = {
    'engine_wiring': CategoryConfig(
        name='engine_wiring',
        format=DatasetFormat.MVTEC,
        anomaly_rate=53.0,
        has_train_anomalies=False,
        anomaly_types=['blue_hoop', 'cardboard', 'fastening', 'multiple', 'obstruction']
    ),
    'pipe_clip': CategoryConfig(
        name='pipe_clip',
        format=DatasetFormat.MVTEC,
        anomaly_rate=42.1,
        has_train_anomalies=False,
        anomaly_types=['operator', 'unclipped']
    ),
    'pipe_staple': CategoryConfig(
        name='pipe_staple',
        format=DatasetFormat.MVTEC,
        anomaly_rate=38.4,
        has_train_anomalies=False,
        anomaly_types=['missing']
    ),
    'tank_screw': CategoryConfig(
        name='tank_screw',
        format=DatasetFormat.MVTEC,
        anomaly_rate=23.0,
        has_train_anomalies=False,
        anomaly_types=['missing']
    ),
    'underbody_pipes': CategoryConfig(
        name='underbody_pipes',
        format=DatasetFormat.MVTEC,
        anomaly_rate=53.3,
        has_train_anomalies=False,
        anomaly_types=['multiple', 'obstruction', 'operator']
    ),
    'underbody_screw': CategoryConfig(
        name='underbody_screw',
        format=DatasetFormat.MVTEC,
        anomaly_rate=4.6,
        has_train_anomalies=False,
        anomaly_types=['missing']
    ),
    'brake_disc': CategoryConfig(
        name='brake_disc',
        format=DatasetFormat.SIMPLIFIED,
        anomaly_rate=1.5,
        has_train_anomalies=True,
        anomaly_types=['KO']
    ),
    'oil_pump_connector': CategoryConfig(
        name='oil_pump_connector',
        format=DatasetFormat.SIMPLIFIED,
        anomaly_rate=15.0,
        has_train_anomalies=True,
        anomaly_types=['KO']
    ),
    'right_radiator': CategoryConfig(
        name='right_radiator',
        format=DatasetFormat.SIMPLIFIED,
        anomaly_rate=10.5,
        has_train_anomalies=True,
        anomaly_types=['KO']
    ),
}


def get_source_train_paths(dataset_path: str, category: str, config: CategoryConfig) -> Tuple[str, Optional[str]]:
    if config.format == DatasetFormat.MVTEC:
        good_path = os.path.join(dataset_path, category, category, 'train', 'good')
        return good_path, None
    else:
        good_path = os.path.join(dataset_path, category, 'training', 'OK')
        bad_path = os.path.join(dataset_path, category, 'training', 'KO')
        return good_path, bad_path


def get_source_test_path(dataset_path: str, category: str, config: CategoryConfig) -> str:
    if config.format == DatasetFormat.MVTEC:
        return os.path.join(dataset_path, category, category, 'test')
    else:
        return os.path.join(dataset_path, category, 'test')


def get_source_ground_truth_path(dataset_path: str, category: str, config: CategoryConfig) -> Optional[str]:
    if config.format == DatasetFormat.MVTEC:
        return os.path.join(dataset_path, category, category, 'ground_truth')
    return None


def get_source_validation_path(dataset_path: str, category: str, config: CategoryConfig) -> Optional[str]:
    if config.format == DatasetFormat.SIMPLIFIED:
        val_path = os.path.join(dataset_path, category, 'validation')
        if os.path.exists(val_path):
            return val_path
    return None


def count_images(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])


def get_images(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    return sorted([f for f in os.listdir(path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])


def copy_images(src_path: str, dst_path: str, images: List[str]):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    for img in images:
        src = os.path.join(src_path, img)
        dst = os.path.join(dst_path, img)
        shutil.copy2(src, dst)


def create_autovi_federated_splits_v4(
    dataset_path: str,
    output_path: str,
    seed: int = 42
):
    """
    Create BALANCED federated splits using ALL data.
    
    Strategy:
    =========
    - NO SUBSAMPLING: All images are used
    - LARGE CATEGORIES SPLIT: brake_disc is distributed across multiple clients
    - 5 CLIENTS with balanced sizes (~400-500 images each)
    - NON-IID: Each client specializes in certain anomaly rate ranges
    
    Distribution Logic:
    ===================
    - Small/medium categories (≤300 imgs): 70% to specialist, 30% to Client 4
    - Large categories (>300 imgs): Split across multiple clients for balance
    - brake_disc (1204 imgs): Split across Client 2, 3, and 4
    
    Client Assignments:
    ===================
    Client 0 - High Anomaly (robustness): engine_wiring, underbody_pipes
    Client 1 - Medium Anomaly (fairness): pipe_clip, pipe_staple, oil_pump_connector
    Client 2 - Mixed + brake_disc part 1 (interpretability): tank_screw, right_radiator, brake_disc
    Client 3 - Low Anomaly + brake_disc part 2 (privacy): underbody_screw, brake_disc
    Client 4 - Cross-domain (generalization): 30% of all + remaining brake_disc
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("AutoVI FEDERATED SPLIT v4 - ALL DATA, BALANCED")
    print("="*80)
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Seed:   {seed}")
    print("="*80)
    print("\n*** ALL IMAGES WILL BE USED - NO SUBSAMPLING ***")
    print("*** Large categories (brake_disc) split across multiple clients ***")
    print("\nOutput structure: category/train/good/ and category/train/bad/")
    print("="*80 + "\n")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    standard_categories = {
        'client_0': ['engine_wiring', 'underbody_pipes'],
        'client_1': ['pipe_clip', 'pipe_staple', 'oil_pump_connector'],
        'client_2': ['tank_screw', 'right_radiator'],
        'client_3': ['underbody_screw'],
    }
    
    # Special handling for brake_disc (split across 3 clients)
    brake_disc_distribution = {
        'client_2': 0.33, 
        'client_3': 0.33, 
        'client_4': 0.34,  
    }
    
    client_info = {
        'client_0': {
            'name': 'High Anomaly Expert',
            'description': 'High anomaly rates (~53%), complex defect patterns',
            'trust_focus': 'robustness',
        },
        'client_1': {
            'name': 'Medium Anomaly Expert',
            'description': 'Medium anomaly rates (15-42%), diverse defect types',
            'trust_focus': 'fairness',
        },
        'client_2': {
            'name': 'Mixed + Visual Expert',
            'description': 'Mix of anomaly rates, good for visual interpretation',
            'trust_focus': 'interpretability',
        },
        'client_3': {
            'name': 'Low Anomaly Expert',
            'description': 'Low anomaly rates (1.5-4.6%), safety-critical parts',
            'trust_focus': 'privacy',
        },
        'client_4': {
            'name': 'Cross-Domain Validator',
            'description': 'Samples from all categories for cross-evaluation',
            'trust_focus': 'generalization',
        },
    }
    
    # Create output directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    test_data_path = os.path.join(output_path, 'test_data_centralized')
    Path(test_data_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {f'client_{i}': {'train_good': {}, 'train_bad': {}, 'categories': []} for i in range(5)}
    
    # =========================================================================
    # PROCESS STANDARD CATEGORIES (70/30 split)
    # =========================================================================
    
    for client_key, categories in standard_categories.items():
        for category in categories:
            cat_config = CATEGORY_CONFIGS[category]
            
            print(f"\n{'='*80}")
            print(f"Processing: {category} -> {client_key} (70%) + client_4 (30%)")
            print(f"{'='*80}")
            
            good_path, bad_path = get_source_train_paths(dataset_path, category, cat_config)
            
            if not os.path.exists(good_path):
                print(f"  WARNING: {good_path} does not exist! Skipping...")
                continue
            
            # Get all images
            good_images = get_images(good_path)
            random.shuffle(good_images)
            
            bad_images = []
            if bad_path and os.path.exists(bad_path):
                bad_images = get_images(bad_path)
                random.shuffle(bad_images)
            
            print(f"  Total: {len(good_images)} good, {len(bad_images)} bad")
            
            # Split 70/30
            split_idx_good = int(len(good_images) * 0.7)
            split_idx_bad = int(len(bad_images) * 0.7)
            
            specialist_good = good_images[:split_idx_good]
            validator_good = good_images[split_idx_good:]
            
            specialist_bad = bad_images[:split_idx_bad]
            validator_bad = bad_images[split_idx_bad:]
            
            # Copy to specialist client
            dst_good = os.path.join(output_path, client_key, category, 'train', 'good')
            copy_images(good_path, dst_good, specialist_good)
            
            if specialist_bad:
                dst_bad = os.path.join(output_path, client_key, category, 'train', 'bad')
                copy_images(bad_path, dst_bad, specialist_bad)
            
            stats[client_key]['train_good'][category] = len(specialist_good)
            stats[client_key]['train_bad'][category] = len(specialist_bad)
            if category not in stats[client_key]['categories']:
                stats[client_key]['categories'].append(category)
            
            print(f"  {client_key}: {len(specialist_good)} good, {len(specialist_bad)} bad")
            
            # Copy to Client 4 (validator)
            dst_good = os.path.join(output_path, 'client_4', category, 'train', 'good')
            copy_images(good_path, dst_good, validator_good)
            
            if validator_bad:
                dst_bad = os.path.join(output_path, 'client_4', category, 'train', 'bad')
                copy_images(bad_path, dst_bad, validator_bad)
            
            stats['client_4']['train_good'][category] = len(validator_good)
            stats['client_4']['train_bad'][category] = len(validator_bad)
            if category not in stats['client_4']['categories']:
                stats['client_4']['categories'].append(category)
            
            print(f"  client_4: {len(validator_good)} good, {len(validator_bad)} bad")
    
    # =========================================================================
    # PROCESS BRAKE_DISC (split across 3 clients)
    # =========================================================================
    
    category = 'brake_disc'
    cat_config = CATEGORY_CONFIGS[category]
    
    print(f"\n{'='*80}")
    print(f"Processing: {category} -> SPLIT across client_2 (33%), client_3 (33%), client_4 (34%)")
    print(f"{'='*80}")
    
    good_path, bad_path = get_source_train_paths(dataset_path, category, cat_config)
    
    if os.path.exists(good_path):
        good_images = get_images(good_path)
        random.shuffle(good_images)
        
        bad_images = []
        if bad_path and os.path.exists(bad_path):
            bad_images = get_images(bad_path)
            random.shuffle(bad_images)
        
        print(f"  Total: {len(good_images)} good, {len(bad_images)} bad")
        
        # Calculate split indices
        total_good = len(good_images)
        total_bad = len(bad_images)
        
        idx1_good = int(total_good * 0.33)
        idx2_good = int(total_good * 0.66)
        
        idx1_bad = int(total_bad * 0.33)
        idx2_bad = int(total_bad * 0.66)
        
        # Client 2: first 33%
        c2_good = good_images[:idx1_good]
        c2_bad = bad_images[:idx1_bad]
        
        # Client 3: next 33%
        c3_good = good_images[idx1_good:idx2_good]
        c3_bad = bad_images[idx1_bad:idx2_bad]
        
        # Client 4: remaining 34%
        c4_good = good_images[idx2_good:]
        c4_bad = bad_images[idx2_bad:]
        
        # Copy to clients
        for client_key, (c_good, c_bad) in [('client_2', (c2_good, c2_bad)), 
                                              ('client_3', (c3_good, c3_bad)),
                                              ('client_4', (c4_good, c4_bad))]:
            dst_good = os.path.join(output_path, client_key, category, 'train', 'good')
            copy_images(good_path, dst_good, c_good)
            
            if c_bad:
                dst_bad = os.path.join(output_path, client_key, category, 'train', 'bad')
                copy_images(bad_path, dst_bad, c_bad)
            
            # Update stats (add to existing if category already there)
            existing_good = stats[client_key]['train_good'].get(category, 0)
            existing_bad = stats[client_key]['train_bad'].get(category, 0)
            stats[client_key]['train_good'][category] = existing_good + len(c_good)
            stats[client_key]['train_bad'][category] = existing_bad + len(c_bad)
            
            if category not in stats[client_key]['categories']:
                stats[client_key]['categories'].append(category)
            
            print(f"  {client_key}: {len(c_good)} good, {len(c_bad)} bad")
    
    # =========================================================================
    # Copy TEST DATA to centralized location
    # =========================================================================
    print(f"\n{'='*80}")
    print("Copying TEST DATA to centralized location...")
    print(f"{'='*80}")
    
    test_stats = {}
    actual_anomaly_rates = {}
    
    for category, cat_config in CATEGORY_CONFIGS.items():
        test_path = get_source_test_path(dataset_path, category, cat_config)
        
        if not os.path.exists(test_path):
            print(f"  WARNING: {category} test path does not exist, skipping...")
            continue
        
        central_test_base = os.path.join(test_data_path, category, 'test')
        test_ok_count = 0
        test_ko_count = 0
        
        if cat_config.format == DatasetFormat.MVTEC:
            # Copy good -> good
            src_good = os.path.join(test_path, 'good')
            if os.path.exists(src_good):
                dst_good = os.path.join(central_test_base, 'good')
                shutil.copytree(src_good, dst_good, dirs_exist_ok=True)
                test_ok_count = count_images(dst_good)
            
            # Copy all anomaly types -> bad (merged)
            dst_bad = os.path.join(central_test_base, 'bad')
            Path(dst_bad).mkdir(parents=True, exist_ok=True)
            
            for anomaly_type in cat_config.anomaly_types:
                src_anomaly = os.path.join(test_path, anomaly_type)
                if os.path.exists(src_anomaly):
                    for img in get_images(src_anomaly):
                        src = os.path.join(src_anomaly, img)
                        dst = os.path.join(dst_bad, f"{anomaly_type}_{img}")
                        shutil.copy2(src, dst)
            
            test_ko_count = count_images(dst_bad)
            
            # Copy ground truth
            gt_path = get_source_ground_truth_path(dataset_path, category, cat_config)
            if gt_path and os.path.exists(gt_path):
                central_gt_dir = os.path.join(test_data_path, category, 'ground_truth')
                shutil.copytree(gt_path, central_gt_dir, dirs_exist_ok=True)
                
        else:
            # Simplified format: OK -> good, KO -> bad
            src_ok = os.path.join(test_path, 'OK')
            src_ko = os.path.join(test_path, 'KO')
            
            if os.path.exists(src_ok):
                dst_good = os.path.join(central_test_base, 'good')
                shutil.copytree(src_ok, dst_good, dirs_exist_ok=True)
                test_ok_count = count_images(dst_good)
            
            if os.path.exists(src_ko):
                dst_bad = os.path.join(central_test_base, 'bad')
                shutil.copytree(src_ko, dst_bad, dirs_exist_ok=True)
                test_ko_count = count_images(dst_bad)
        
        # Copy validation if exists
        val_path = get_source_validation_path(dataset_path, category, cat_config)
        if val_path and os.path.exists(val_path):
            central_val_base = os.path.join(test_data_path, category, 'validation')
            src_val_ok = os.path.join(val_path, 'OK')
            src_val_ko = os.path.join(val_path, 'KO')
            
            if os.path.exists(src_val_ok):
                shutil.copytree(src_val_ok, os.path.join(central_val_base, 'good'), dirs_exist_ok=True)
            if os.path.exists(src_val_ko):
                shutil.copytree(src_val_ko, os.path.join(central_val_base, 'bad'), dirs_exist_ok=True)
        
        total_test = test_ok_count + test_ko_count
        actual_rate = (test_ko_count / total_test * 100) if total_test > 0 else 0
        actual_anomaly_rates[category] = actual_rate
        
        test_stats[category] = {
            'test_good': test_ok_count,
            'test_bad': test_ko_count,
            'total': total_test,
            'anomaly_rate': round(actual_rate, 2)
        }
        
        print(f"  {category:25s}: {test_ok_count:4d} good, {test_ko_count:4d} bad ({actual_rate:.1f}%)")
    
    # =========================================================================
    # Calculate final statistics
    # =========================================================================
    
    for client_id in range(5):
        client_key = f'client_{client_id}'
        
        total_good = sum(stats[client_key]['train_good'].values())
        total_bad = sum(stats[client_key]['train_bad'].values())
        total_train = total_good + total_bad
        
        # Calculate weighted average anomaly rate
        weighted_rate = 0
        total_weight = 0
        for cat in stats[client_key]['categories']:
            cat_total = stats[client_key]['train_good'].get(cat, 0) + stats[client_key]['train_bad'].get(cat, 0)
            if cat_total > 0:
                rate = actual_anomaly_rates.get(cat, CATEGORY_CONFIGS[cat].anomaly_rate)
                weighted_rate += cat_total * rate
                total_weight += cat_total
        
        avg_rate = weighted_rate / total_weight if total_weight > 0 else 0
        
        stats[client_key]['total_train_good'] = total_good
        stats[client_key]['total_train_bad'] = total_bad
        stats[client_key]['total_train'] = total_train
        stats[client_key]['avg_anomaly_rate'] = round(avg_rate, 2)
        stats[client_key]['num_categories'] = len(stats[client_key]['categories'])
        stats[client_key]['info'] = client_info[client_key]
    
    stats['test_data'] = test_stats
    stats['actual_anomaly_rates'] = {k: round(v, 2) for k, v in actual_anomaly_rates.items()}
    stats['metadata'] = {
        'version': 'v4_all_data_balanced',
        'subsampling': False,
        'output_structure': 'train/good and train/bad',
        'brake_disc_split': 'Distributed across client_2, client_3, client_4'
    }
    
    # Save configuration
    config = {
        'version': 'v4_all_data_balanced',
        'num_clients': 5,
        'strategy': 'all_data_balanced_distribution',
        'seed': seed,
        'subsampling': False,
        'output_structure': 'category/train/good and category/train/bad',
        'test_data_location': 'test_data_centralized/',
        'distribution': {
            'standard_categories': '70% to specialist, 30% to client_4',
            'brake_disc': '33% client_2, 33% client_3, 34% client_4'
        },
        'client_info': client_info,
    }
    
    config_path = os.path.join(output_path, 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save statistics
    stats_path = os.path.join(output_path, 'split_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print_statistics_v4(stats, client_info)
    
    print(f"\n{'='*80}")
    print("SPLIT COMPLETE - ALL DATA USED!")
    print(f"{'='*80}")
    print(f"Data saved in: {output_path}")
    print(f"Statistics: {stats_path}")
    print(f"Test data: {test_data_path}")
    print(f"{'='*80}\n")
    
    return stats


def print_statistics_v4(stats: Dict, client_info: Dict):
    """Print formatted statistics"""
    
    print(f"\n{'='*80}")
    print("FEDERATED SPLIT STATISTICS v4 - ALL DATA BALANCED")
    print(f"{'='*80}\n")
    
    for client_id in range(5):
        client_key = f'client_{client_id}'
        info = client_info[client_key]
        
        print(f"┌{'─'*78}┐")
        print(f"│ CLIENT {client_id}: {info['name']:<64} │")
        print(f"│ Focus: {info['trust_focus']:<68} │")
        print(f"├{'─'*78}┤")
        
        for cat in sorted(stats[client_key]['categories']):
            good = stats[client_key]['train_good'].get(cat, 0)
            bad = stats[client_key]['train_bad'].get(cat, 0)
            total = good + bad
            if total > 0:
                line = f"│   {cat:22s}: {good:4d} good + {bad:4d} bad = {total:4d}"
                print(f"{line:<79}│")
        
        total_good = stats[client_key]['total_train_good']
        total_bad = stats[client_key]['total_train_bad']
        total = stats[client_key]['total_train']
        avg_rate = stats[client_key]['avg_anomaly_rate']
        
        print(f"├{'─'*78}┤")
        print(f"│   TOTAL: {total_good:4d} good + {total_bad:4d} bad = {total:4d} images{' '*29}│")
        print(f"│   Avg anomaly rate: {avg_rate:.1f}%{' '*52}│")
        print(f"└{'─'*78}┘\n")
    
    # Summary
    print(f"\n{'='*80}")
    print("BALANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Client':<10} {'Name':<28} {'Images':<10} {'Categories':<12}")
    print(f"{'-'*60}")
    
    sizes = []
    for client_id in range(5):
        client_key = f'client_{client_id}'
        info = client_info[client_key]
        total = stats[client_key]['total_train']
        num_cats = stats[client_key]['num_categories']
        sizes.append(total)
        print(f"{client_key:<10} {info['name']:<28} {total:<10} {num_cats:<12}")
    
    print(f"{'-'*60}")
    print(f"{'TOTAL':<10} {'':<28} {sum(sizes):<10}")
    
    # Balance check
    specialist_sizes = sizes[:4]
    min_s, max_s = min(specialist_sizes), max(specialist_sizes)
    ratio = min_s / max_s if max_s > 0 else 0
    
    print(f"\nBalance ratio (specialists): {ratio:.2f} ", end="")
    print("✓ GOOD" if ratio >= 0.7 else "⚠ Check distribution")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoVI Federated Split v4 - All Data Balanced')
    parser.add_argument('--dataset', type=str, 
                        default="/mnt/c/Users/neagu/Downloads/AutoVI-FULL",
                        help='Path to AutoVI dataset')
    parser.add_argument('--output', type=str,
                        default=os.path.expanduser("~/AutoVI_federated_v4"),
                        help='Output path for federated splits')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--clean', action='store_true',
                        help='Delete existing output directory')
    
    args = parser.parse_args()
    
    if args.clean and os.path.exists(args.output):
        print(f"Deleting existing output: {args.output}")
        shutil.rmtree(args.output)
    
    stats = create_autovi_federated_splits_v4(
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed
    )
    
    print("\nDone! All images have been distributed across clients.")
