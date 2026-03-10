"""
Adversarial attack implementations for AI System Stress Testing.

This module provides various adversarial attack methods including FGSM, PGD,
and C&W attacks for testing model robustness.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks import FGSM, PGD, CW, DeepFool, AutoAttack


class AdversarialAttacker:
    """Main class for performing adversarial attacks on models."""
    
    def __init__(self, device: torch.device, attack_config: Dict):
        """Initialize adversarial attacker.
        
        Args:
            device: PyTorch device
            attack_config: Configuration dictionary for attacks
        """
        self.device = device
        self.config = attack_config
        self.attack_methods = {}
        
    def register_attack(self, name: str, attack: nn.Module) -> None:
        """Register an attack method.
        
        Args:
            name: Name of the attack
            attack: Attack instance
        """
        self.attack_methods[name] = attack
        
    def fgsm_attack(self, 
                   model: nn.Module, 
                   X: torch.Tensor, 
                   y: torch.Tensor,
                   epsilon: float = 0.1) -> torch.Tensor:
        """Perform FGSM attack.
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            epsilon: Attack strength
            
        Returns:
            Adversarial examples
        """
        X.requires_grad_(True)
        
        # Forward pass
        outputs = model(X)
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        X_adv = X + epsilon * X.grad.sign()
        X_adv = torch.clamp(X_adv, 0, 1)  # Assuming normalized inputs
        
        return X_adv.detach()
    
    def pgd_attack(self, 
                  model: nn.Module, 
                  X: torch.Tensor, 
                  y: torch.Tensor,
                  epsilon: float = 0.1,
                  alpha: float = 0.01,
                  num_iter: int = 40) -> torch.Tensor:
        """Perform PGD attack.
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            epsilon: Attack strength
            alpha: Step size
            num_iter: Number of iterations
            
        Returns:
            Adversarial examples
        """
        X_adv = X.clone().detach()
        
        for _ in range(num_iter):
            X_adv.requires_grad_(True)
            
            # Forward pass
            outputs = model(X_adv)
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            X_adv = X_adv + alpha * X_adv.grad.sign()
            
            # Project to epsilon ball
            delta = X_adv - X
            delta = torch.clamp(delta, -epsilon, epsilon)
            X_adv = X + delta
            
            # Clamp to valid range
            X_adv = torch.clamp(X_adv, 0, 1)
            
        return X_adv.detach()
    
    def cw_attack(self, 
                 model: nn.Module, 
                 X: torch.Tensor, 
                 y: torch.Tensor,
                 c: float = 1.0,
                 kappa: float = 0.0,
                 max_iter: int = 1000) -> torch.Tensor:
        """Perform C&W attack (simplified version).
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            c: Confidence parameter
            kappa: Confidence threshold
            max_iter: Maximum iterations
            
        Returns:
            Adversarial examples
        """
        # This is a simplified C&W implementation
        # For production use, consider using torchattacks.CW
        
        X_adv = X.clone().detach()
        X_adv.requires_grad_(True)
        
        # Use Adam optimizer for C&W
        optimizer = torch.optim.Adam([X_adv], lr=0.01)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_adv)
            
            # C&W loss
            target_scores = outputs.gather(1, y.unsqueeze(1))
            max_other_scores = outputs.scatter(1, y.unsqueeze(1), -float('inf')).max(1)[0]
            
            loss1 = torch.clamp(max_other_scores - target_scores.squeeze() + kappa, min=0)
            loss2 = torch.norm(X_adv - X, p=2)
            
            loss = loss1.sum() + c * loss2.sum()
            
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            X_adv = torch.clamp(X_adv, 0, 1)
        
        return X_adv.detach()
    
    def evaluate_attack(self, 
                       model: nn.Module, 
                       X: torch.Tensor, 
                       y: torch.Tensor,
                       attack_name: str,
                       **attack_kwargs) -> Dict[str, float]:
        """Evaluate model performance under attack.
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            attack_name: Name of attack to use
            **attack_kwargs: Additional attack parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        # Get clean accuracy
        with torch.no_grad():
            clean_outputs = model(X)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_accuracy = (clean_preds == y).float().mean().item()
        
        # Generate adversarial examples
        if attack_name == "fgsm":
            X_adv = self.fgsm_attack(model, X, y, **attack_kwargs)
        elif attack_name == "pgd":
            X_adv = self.pgd_attack(model, X, y, **attack_kwargs)
        elif attack_name == "cw":
            X_adv = self.cw_attack(model, X, y, **attack_kwargs)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            adv_outputs = model(X_adv)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_accuracy = (adv_preds == y).float().mean().item()
        
        # Calculate perturbation magnitude
        perturbation = torch.norm(X_adv - X, p=2, dim=1).mean().item()
        
        return {
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "accuracy_drop": clean_accuracy - adv_accuracy,
            "perturbation_magnitude": perturbation,
            "attack_success_rate": 1 - adv_accuracy
        }
    
    def robustness_curve(self, 
                        model: nn.Module, 
                        X: torch.Tensor, 
                        y: torch.Tensor,
                        attack_name: str,
                        epsilon_range: List[float],
                        **attack_kwargs) -> Dict[str, List[float]]:
        """Generate robustness curve across different epsilon values.
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            attack_name: Name of attack to use
            epsilon_range: List of epsilon values to test
            **attack_kwargs: Additional attack parameters
            
        Returns:
            Dictionary with robustness metrics
        """
        accuracies = []
        perturbations = []
        
        for epsilon in epsilon_range:
            kwargs = attack_kwargs.copy()
            kwargs['epsilon'] = epsilon
            
            results = self.evaluate_attack(model, X, y, attack_name, **kwargs)
            accuracies.append(results['adversarial_accuracy'])
            perturbations.append(results['perturbation_magnitude'])
        
        return {
            "epsilon_values": epsilon_range,
            "accuracies": accuracies,
            "perturbations": perturbations
        }
    
    def batch_attack(self, 
                    model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    attack_name: str,
                    **attack_kwargs) -> Dict[str, float]:
        """Perform attack on entire dataset.
        
        Args:
            model: Target model
            dataloader: DataLoader with test data
            attack_name: Name of attack to use
            **attack_kwargs: Additional attack parameters
            
        Returns:
            Dictionary with overall evaluation metrics
        """
        model.eval()
        
        total_samples = 0
        total_correct_clean = 0
        total_correct_adv = 0
        total_perturbation = 0
        
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            batch_size = X.size(0)
            
            # Clean accuracy
            with torch.no_grad():
                clean_outputs = model(X)
                clean_preds = torch.argmax(clean_outputs, dim=1)
                total_correct_clean += (clean_preds == y).sum().item()
            
            # Adversarial accuracy
            if attack_name == "fgsm":
                X_adv = self.fgsm_attack(model, X, y, **attack_kwargs)
            elif attack_name == "pgd":
                X_adv = self.pgd_attack(model, X, y, **attack_kwargs)
            elif attack_name == "cw":
                X_adv = self.cw_attack(model, X, y, **attack_kwargs)
            else:
                raise ValueError(f"Unknown attack: {attack_name}")
            
            with torch.no_grad():
                adv_outputs = model(X_adv)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                total_correct_adv += (adv_preds == y).sum().item()
            
            # Perturbation magnitude
            perturbation = torch.norm(X_adv - X, p=2, dim=1).sum().item()
            total_perturbation += perturbation
            
            total_samples += batch_size
        
        return {
            "clean_accuracy": total_correct_clean / total_samples,
            "adversarial_accuracy": total_correct_adv / total_samples,
            "accuracy_drop": (total_correct_clean - total_correct_adv) / total_samples,
            "avg_perturbation": total_perturbation / total_samples,
            "attack_success_rate": 1 - (total_correct_adv / total_samples)
        }
