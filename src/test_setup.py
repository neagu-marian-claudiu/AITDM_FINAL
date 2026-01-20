"""
Setup Testing Script

Verifies that all components are working correctly
"""

import torch
from model import PatchCore, count_parameters

print("=" * 60)
print("TESTING SETUP")
print("=" * 60)

print("\n[1/3] Testing model...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = PatchCore(backbone='wide_resnet50', layer_indices=[2, 3])
    model.to(device)
    print(f"Model created successfully")
    print(f"Backbone parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    features = model.extract_features(x)
    patches, (H, W) = model.aggregate_features(features)
    print(f"Input: {x.shape}")
    print(f"Features extracted: {len(features)} levels")
    print(f"Patches: {patches.shape} (H={H}, W={W})")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n[2/3] Testing imports...")
try:
    from preprocessing import get_transforms, AutoVIDataset
    print("preprocessing.py - OK")
    
    from federated_train import FederatedClient, FederatedServer
    print("federated_train.py - OK")
    
    from evaluate import evaluate_model, compute_tpr_at_tnr
    print("evaluate.py - OK")
    
    from aggregate_models import aggregate_models
    print("aggregate_models.py - OK")
    
    from train_standalone import train_standalone
    print("train_standalone.py - OK")
    
except Exception as e:
    print(f"Import ERROR: {e}")

print("\n[3/3] Testing transforms...")
try:
    train_tf, val_tf = get_transforms(256, augment=True)
    print(f"Train transform: {train_tf}")
    print(f"Val transform: {val_tf}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("SETUP TEST COMPLETE!")
print("=" * 60)
print("\nTo train the model, follow the steps in README.md")
