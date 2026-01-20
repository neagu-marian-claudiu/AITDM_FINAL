#!/usr/bin/env python3
"""
Model Aggregation Script

Patches an aggregated model missing the projector by loading it from a client model.

Usage:
    python aggregate_models.py

This will:
1. Load the aggregated model
2. Load the projector from client_0
3. Save the fixed model
"""

import os
import joblib


def fix_aggregated_model():
    """Fix aggregated model by copying missing components from client models."""
    # Paths
    aggregated_path = "checkpoints/aggregated/federated_category_aware.joblib"
    client_path = "checkpoints/clients/client_0_improved.joblib"
    
    print("\n" + "=" * 70)
    print("FIXING AGGREGATED MODEL")
    print("=" * 70 + "\n")
    
    # Check files exist
    if not os.path.exists(aggregated_path):
        print(f"Error: Aggregated model not found: {aggregated_path}")
        return False
    
    if not os.path.exists(client_path):
        print(f"Warning: Client model not found: {client_path}")
        print("Trying other clients...")
        for i in range(5):
            alt_path = f"checkpoints/clients/client_{i}_improved.joblib"
            if os.path.exists(alt_path):
                client_path = alt_path
                print(f"Found: {client_path}")
                break
        else:
            print("Error: No client models found!")
            return False
    
    # Load aggregated model
    print(f"Loading aggregated model: {aggregated_path}")
    agg_state = joblib.load(aggregated_path)
    
    print(f"  Memory bank: {agg_state['memory_bank'].shape}")
    print(f"  Projector: {agg_state.get('projector', 'MISSING')}")
    print(f"  Backbone: {agg_state.get('backbone_name', 'MISSING')}")
    print(f"  Layers: {agg_state.get('layer_indices', 'MISSING')}")
    
    # Load client model to get projector
    print(f"\nLoading client model: {client_path}")
    client_state = joblib.load(client_path)
    
    print(f"  Projector: {'present' if client_state.get('projector') else 'None'}")
    print(f"  Backbone: {client_state.get('backbone_name')}")
    print(f"  Layers: {client_state.get('layer_indices')}")
    
    # Check if fix is needed
    if agg_state.get('projector') is not None:
        print("\nWarning: Aggregated model already has a projector!")
        print("Checking if it needs other fixes...")
    
    # Apply fixes
    fixes_applied = []
    
    # Fix 1: Projector
    if agg_state.get('projector') is None and client_state.get('projector') is not None:
        agg_state['projector'] = client_state['projector']
        fixes_applied.append("projector")
        print("\nFixed: Added projector from client model")
    
    # Fix 2: Backbone name
    if agg_state.get('backbone_name') in [None, 'resnet18']:
        correct_backbone = client_state.get('backbone_name', 'wide_resnet50')
        agg_state['backbone_name'] = correct_backbone
        fixes_applied.append(f"backbone_name -> {correct_backbone}")
        print(f"Fixed: backbone_name -> {correct_backbone}")
    
    # Fix 3: Layer indices
    if agg_state.get('layer_indices') in [None, [1, 2]]:
        correct_layers = client_state.get('layer_indices', [2, 3])
        agg_state['layer_indices'] = correct_layers
        fixes_applied.append(f"layer_indices -> {correct_layers}")
        print(f"Fixed: layer_indices -> {correct_layers}")
    
    # Fix 4: Category stats
    if not agg_state.get('category_stats') and client_state.get('category_stats'):
        agg_state['category_stats'] = client_state['category_stats']
        fixes_applied.append("category_stats")
        print("Fixed: Added category_stats from client model")
    
    if not fixes_applied:
        print("\nNo fixes needed - model looks correct!")
        return True
    
    # Backup original
    backup_path = aggregated_path.replace('.joblib', '_backup.joblib')
    print(f"\nCreating backup: {backup_path}")
    joblib.dump(joblib.load(aggregated_path), backup_path)
    
    # Save fixed model
    print(f"Saving fixed model: {aggregated_path}")
    joblib.dump(agg_state, aggregated_path)
    
    # Verify
    print("\nVerifying fix...")
    verify_state = joblib.load(aggregated_path)
    print(f"  Memory bank: {verify_state['memory_bank'].shape}")
    projector_status = "present" if verify_state.get('projector') else "MISSING"
    print(f"  Projector: {projector_status}")
    print(f"  Backbone: {verify_state.get('backbone_name')}")
    print(f"  Layers: {verify_state.get('layer_indices')}")
    
    print("\n" + "=" * 70)
    print("FIX COMPLETE!")
    print("=" * 70)
    print(f"\nFixes applied: {', '.join(fixes_applied)}")
    print(f"Backup saved to: {backup_path}")
    print("\nYou can now re-run the evaluation.")
    print("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    fix_aggregated_model()