"""
Simple test script to verify the stress testing framework works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from src.models import ModelFactory
from src.attacks import AdversarialAttacker
from src.uncertainty import UncertaintyQuantifier
from src.ood import OODDetector
from src.utils import set_seed, get_device


def test_basic_functionality():
    """Test basic functionality of the stress testing framework."""
    print("Testing AI System Stress Testing Framework...")
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Test model creation
    print("Testing model creation...")
    model = ModelFactory.create_model("simple_mlp", 10, 2)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test data
    print("Testing data generation...")
    X = torch.randn(100, 10).to(device)
    y = torch.randint(0, 2, (100,)).to(device)
    
    # Test adversarial attacks
    print("Testing adversarial attacks...")
    attacker = AdversarialAttacker(device, {})
    
    # FGSM attack
    X_adv = attacker.fgsm_attack(model, X, y, epsilon=0.1)
    print(f"FGSM attack completed, perturbation: {torch.norm(X_adv - X, p=2).mean():.4f}")
    
    # Test uncertainty quantification
    print("Testing uncertainty quantification...")
    uncertainty_quantifier = UncertaintyQuantifier(device, {})
    
    predictions, uncertainties = uncertainty_quantifier.monte_carlo_dropout(model, X, n_samples=10)
    print(f"Monte Carlo Dropout completed, mean uncertainty: {uncertainties.mean():.4f}")
    
    # Test OOD detection
    print("Testing OOD detection...")
    ood_detector = OODDetector(device, {})
    
    energy_scores = ood_detector.energy_based_detection(model, X)
    print(f"Energy-based OOD detection completed, mean score: {energy_scores.mean():.4f}")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_basic_functionality()
