#!/usr/bin/env python3
"""
Example script demonstrating the modernized AI System Stress Testing framework.

This script shows how to use the stress testing framework to evaluate
model robustness, uncertainty quantification, and out-of-distribution detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models import ModelFactory
from src.attacks import AdversarialAttacker
from src.uncertainty import UncertaintyQuantifier
from src.ood import OODDetector
from src.utils import set_seed, get_device, setup_logging


def main():
    """Main function demonstrating stress testing."""
    print("=" * 60)
    print("AI System Stress Testing - Example Script")
    print("=" * 60)
    
    # Set up
    set_seed(42)
    device = get_device()
    logger = setup_logging("INFO")
    
    print(f"Using device: {device}")
    print("DISCLAIMER: This is for research/educational purposes only.")
    print("Results may be unstable and should not be used for regulated decisions.\n")
    
    # Generate synthetic data
    print("1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_redundant=2,
        n_informative=18,
        noise=0.1,
        random_state=42
    )
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    print(f"   Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"   Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Create and train model
    print("\n2. Creating and training model...")
    model = ModelFactory.create_model("simple_mlp", X_train.shape[1], len(np.unique(y)))
    model = model.to(device)
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == y_train_tensor).float().mean().item()
                print(f"   Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")
    
    # Evaluate clean performance
    print("\n3. Evaluating clean performance...")
    model.eval()
    with torch.no_grad():
        clean_outputs = model(X_test_tensor)
        clean_preds = torch.argmax(clean_outputs, dim=1)
        clean_accuracy = (clean_preds == y_test_tensor).float().mean().item()
    
    print(f"   Clean accuracy: {clean_accuracy:.4f}")
    
    # Adversarial attacks
    print("\n4. Running adversarial attacks...")
    attacker = AdversarialAttacker(device, {})
    
    attack_methods = ["fgsm", "pgd"]
    epsilon_values = [0.01, 0.05, 0.1, 0.2]
    
    attack_results = {}
    for method in attack_methods:
        print(f"   Testing {method.upper()} attack...")
        method_results = {}
        
        for epsilon in epsilon_values:
            if method == "fgsm":
                X_adv = attacker.fgsm_attack(model, X_test_tensor, y_test_tensor, epsilon)
            elif method == "pgd":
                X_adv = attacker.pgd_attack(model, X_test_tensor, y_test_tensor, epsilon)
            
            with torch.no_grad():
                adv_outputs = model(X_adv)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                adv_accuracy = (adv_preds == y_test_tensor).float().mean().item()
                perturbation = torch.norm(X_adv - X_test_tensor, p=2, dim=1).mean().item()
            
            method_results[epsilon] = {
                'accuracy': adv_accuracy,
                'perturbation': perturbation
            }
            
            print(f"     ε={epsilon}: Accuracy={adv_accuracy:.4f}, Perturbation={perturbation:.4f}")
        
        attack_results[method] = method_results
    
    # Uncertainty quantification
    print("\n5. Running uncertainty quantification...")
    uncertainty_quantifier = UncertaintyQuantifier(device, {})
    
    # Monte Carlo Dropout
    print("   Testing Monte Carlo Dropout...")
    predictions, uncertainties = uncertainty_quantifier.monte_carlo_dropout(
        model, X_test_tensor, n_samples=50
    )
    
    mean_uncertainty = uncertainties.mean().item()
    std_uncertainty = uncertainties.std().item()
    print(f"     Mean uncertainty: {mean_uncertainty:.4f}")
    print(f"     Std uncertainty: {std_uncertainty:.4f}")
    
    # Temperature Scaling
    print("   Testing Temperature Scaling...")
    uncertainty_quantifier.temperature_scaling(model, X_test_tensor, y_test_tensor)
    calibrated_predictions = uncertainty_quantifier.calibrated_predictions(model, X_test_tensor)
    
    # Calculate calibration metrics
    max_probs = calibrated_predictions.max(dim=1)[0]
    predicted_classes = calibrated_predictions.argmax(dim=1)
    correct_predictions = (predicted_classes == y_test_tensor).float()
    
    # Simple calibration metric (ECE approximation)
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        if in_bin.sum() > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin.sum()
    
    ece = ece / len(y_test_tensor)
    print(f"     Expected Calibration Error: {ece:.4f}")
    
    # Out-of-distribution detection
    print("\n6. Running OOD detection...")
    ood_detector = OODDetector(device, {})
    
    # Create OOD data
    X_ood = X_test_tensor + torch.randn_like(X_test_tensor) * 2.0
    y_ood = y_test_tensor
    
    ood_methods = ["energy", "max_softmax", "entropy"]
    
    for method in ood_methods:
        print(f"   Testing {method} OOD detection...")
        
        if method == "energy":
            id_scores = ood_detector.energy_based_detection(model, X_test_tensor)
            ood_scores = ood_detector.energy_based_detection(model, X_ood)
        elif method == "max_softmax":
            id_scores = ood_detector.max_softmax_probability(model, X_test_tensor)
            ood_scores = ood_detector.max_softmax_probability(model, X_ood)
        elif method == "entropy":
            id_scores = ood_detector.entropy_based_detection(model, X_test_tensor)
            ood_scores = ood_detector.entropy_based_detection(model, X_ood)
        
        id_mean = id_scores.mean().item()
        ood_mean = ood_scores.mean().item()
        separation = ood_mean - id_mean
        
        print(f"     ID mean score: {id_mean:.4f}")
        print(f"     OOD mean score: {ood_mean:.4f}")
        print(f"     Score separation: {separation:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("STRESS TESTING SUMMARY")
    print("=" * 60)
    
    print(f"Clean Accuracy: {clean_accuracy:.4f}")
    
    print("\nAdversarial Robustness:")
    for method, method_results in attack_results.items():
        print(f"  {method.upper()}:")
        for epsilon, results in method_results.items():
            acc_drop = clean_accuracy - results['accuracy']
            print(f"    ε={epsilon}: {results['accuracy']:.4f} (drop: {acc_drop:.4f})")
    
    print(f"\nUncertainty Quantification:")
    print(f"  Monte Carlo Dropout: {mean_uncertainty:.4f} ± {std_uncertainty:.4f}")
    print(f"  Temperature Scaling ECE: {ece:.4f}")
    
    print(f"\nOOD Detection:")
    print(f"  Methods tested: {', '.join(ood_methods)}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze results and provide recommendations
    worst_attack_acc = min([min(method_results.values(), key=lambda x: x['accuracy'])['accuracy'] 
                           for method_results in attack_results.values()])
    
    if worst_attack_acc < 0.5:
        print("⚠️  HIGH VULNERABILITY: Model shows significant vulnerability to adversarial attacks")
        print("   Recommendation: Consider adversarial training or robust optimization")
    elif worst_attack_acc < 0.7:
        print("⚠️  MODERATE VULNERABILITY: Model shows moderate vulnerability to adversarial attacks")
        print("   Recommendation: Implement additional robustness measures")
    else:
        print("✅ GOOD ROBUSTNESS: Model shows good robustness to adversarial attacks")
    
    if ece > 0.1:
        print("⚠️  POOR CALIBRATION: Model shows poor calibration")
        print("   Recommendation: Implement temperature scaling or ensemble methods")
    else:
        print("✅ GOOD CALIBRATION: Model shows good calibration")
    
    if mean_uncertainty < 0.5:
        print("⚠️  LOW UNCERTAINTY: Model may be overconfident")
        print("   Recommendation: Implement uncertainty-aware training")
    else:
        print("✅ APPROPRIATE UNCERTAINTY: Model shows appropriate uncertainty levels")
    
    print("\n" + "=" * 60)
    print("LIMITATIONS")
    print("=" * 60)
    print("• Results are based on synthetic data and may not generalize")
    print("• Adversarial examples are artificially generated")
    print("• Uncertainty estimates are approximations")
    print("• Stress testing is not a substitute for comprehensive validation")
    print("• Always validate results with domain experts")
    
    print("\nExample script completed successfully!")


if __name__ == "__main__":
    main()
