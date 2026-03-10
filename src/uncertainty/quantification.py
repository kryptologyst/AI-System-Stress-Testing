"""
Uncertainty quantification methods for AI System Stress Testing.

This module provides various uncertainty quantification techniques including
Monte Carlo Dropout, Deep Ensembles, and Temperature Scaling.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from netcal.metrics import ECE, MCE, ACE
from netcal.scaling import TemperatureScaling


class UncertaintyQuantifier:
    """Main class for uncertainty quantification in neural networks."""
    
    def __init__(self, device: torch.device, config: Dict):
        """Initialize uncertainty quantifier.
        
        Args:
            device: PyTorch device
            config: Configuration dictionary
        """
        self.device = device
        self.config = config
        self.temperature_scaler = None
        
    def monte_carlo_dropout(self, 
                           model: nn.Module, 
                           X: torch.Tensor, 
                           n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo Dropout for uncertainty estimation.
        
        Args:
            model: Neural network model
            X: Input data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = model(X)
                predictions.append(F.softmax(outputs, dim=1))
        
        predictions = torch.stack(predictions)
        mean_predictions = predictions.mean(dim=0)
        
        # Calculate epistemic uncertainty (variance across samples)
        epistemic_uncertainty = predictions.var(dim=0).sum(dim=1)
        
        # Calculate aleatoric uncertainty (entropy of mean predictions)
        aleatoric_uncertainty = -(mean_predictions * torch.log(mean_predictions + 1e-8)).sum(dim=1)
        
        return mean_predictions, epistemic_uncertainty
    
    def deep_ensemble(self, 
                     models: List[nn.Module], 
                     X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Deep Ensemble for uncertainty estimation.
        
        Args:
            models: List of trained models
            X: Input data
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                predictions.append(F.softmax(outputs, dim=1))
        
        predictions = torch.stack(predictions)
        mean_predictions = predictions.mean(dim=0)
        
        # Calculate epistemic uncertainty (variance across models)
        epistemic_uncertainty = predictions.var(dim=0).sum(dim=1)
        
        # Calculate aleatoric uncertainty (entropy of mean predictions)
        aleatoric_uncertainty = -(mean_predictions * torch.log(mean_predictions + 1e-8)).sum(dim=1)
        
        return mean_predictions, epistemic_uncertainty
    
    def temperature_scaling(self, 
                           model: nn.Module, 
                           X_val: torch.Tensor, 
                           y_val: torch.Tensor) -> None:
        """Calibrate model using temperature scaling.
        
        Args:
            model: Neural network model
            X_val: Validation data
            y_val: Validation labels
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(X_val)
        
        # Convert to numpy for netcal
        logits_np = logits.cpu().numpy()
        y_val_np = y_val.cpu().numpy()
        
        # Fit temperature scaling
        self.temperature_scaler = TemperatureScaling()
        self.temperature_scaler.fit(logits_np, y_val_np)
    
    def calibrated_predictions(self, 
                             model: nn.Module, 
                             X: torch.Tensor) -> torch.Tensor:
        """Get temperature-scaled predictions.
        
        Args:
            model: Neural network model
            X: Input data
            
        Returns:
            Calibrated predictions
        """
        if self.temperature_scaler is None:
            raise ValueError("Temperature scaler not fitted. Call temperature_scaling first.")
        
        model.eval()
        
        with torch.no_grad():
            logits = model(X)
        
        # Convert to numpy for netcal
        logits_np = logits.cpu().numpy()
        
        # Apply temperature scaling
        calibrated_probs = self.temperature_scaler.transform(logits_np)
        
        return torch.FloatTensor(calibrated_probs).to(self.device)
    
    def evaluate_calibration(self, 
                           model: nn.Module, 
                           dataloader: DataLoader,
                           method: str = "mc_dropout",
                           **kwargs) -> Dict[str, float]:
        """Evaluate model calibration using various metrics.
        
        Args:
            model: Neural network model
            dataloader: DataLoader with test data
            method: Uncertainty method ("mc_dropout", "deep_ensemble", "temperature_scaling")
            **kwargs: Additional parameters for uncertainty method
            
        Returns:
            Dictionary with calibration metrics
        """
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            if method == "mc_dropout":
                predictions, uncertainties = self.monte_carlo_dropout(
                    model, X, kwargs.get('n_samples', 100)
                )
            elif method == "deep_ensemble":
                models = kwargs.get('models', [model])
                predictions, uncertainties = self.deep_ensemble(models, X)
            elif method == "temperature_scaling":
                predictions = self.calibrated_predictions(model, X)
                uncertainties = -(predictions * torch.log(predictions + 1e-8)).sum(dim=1)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())
        
        # Concatenate all results
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        uncertainties = np.concatenate(all_uncertainties)
        
        # Calculate calibration metrics
        ece = ECE(bins=15)
        mce = MCE(bins=15)
        ace = ACE(bins=15)
        
        ece_score = ece.measure(predictions, labels)
        mce_score = mce.measure(predictions, labels)
        ace_score = ace.measure(predictions, labels)
        
        # Calculate Brier score
        brier_score = torchmetrics.functional.brier_score(
            torch.FloatTensor(predictions), 
            torch.LongTensor(labels)
        ).item()
        
        # Calculate negative log-likelihood
        nll = -np.mean(np.log(predictions[np.arange(len(labels)), labels] + 1e-8))
        
        return {
            "ece": ece_score,
            "mce": mce_score,
            "ace": ace_score,
            "brier_score": brier_score,
            "nll": nll,
            "mean_uncertainty": np.mean(uncertainties),
            "std_uncertainty": np.std(uncertainties)
        }
    
    def uncertainty_analysis(self, 
                           model: nn.Module, 
                           X: torch.Tensor, 
                           y: torch.Tensor,
                           method: str = "mc_dropout",
                           **kwargs) -> Dict[str, torch.Tensor]:
        """Perform detailed uncertainty analysis.
        
        Args:
            model: Neural network model
            X: Input data
            y: True labels
            method: Uncertainty method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with uncertainty analysis results
        """
        if method == "mc_dropout":
            predictions, uncertainties = self.monte_carlo_dropout(
                model, X, kwargs.get('n_samples', 100)
            )
        elif method == "deep_ensemble":
            models = kwargs.get('models', [model])
            predictions, uncertainties = self.deep_ensemble(models, X)
        elif method == "temperature_scaling":
            predictions = self.calibrated_predictions(model, X)
            uncertainties = -(predictions * torch.log(predictions + 1e-8)).sum(dim=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate prediction confidence
        max_probs = predictions.max(dim=1)[0]
        predicted_classes = predictions.argmax(dim=1)
        
        # Calculate accuracy vs uncertainty relationship
        correct_predictions = (predicted_classes == y).float()
        
        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "max_probs": max_probs,
            "predicted_classes": predicted_classes,
            "correct_predictions": correct_predictions,
            "accuracy": correct_predictions.mean().item()
        }
    
    def reliability_diagram(self, 
                           model: nn.Module, 
                           dataloader: DataLoader,
                           method: str = "mc_dropout",
                           n_bins: int = 10,
                           **kwargs) -> Dict[str, np.ndarray]:
        """Generate reliability diagram data.
        
        Args:
            model: Neural network model
            dataloader: DataLoader with test data
            method: Uncertainty method
            n_bins: Number of bins for reliability diagram
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with reliability diagram data
        """
        all_predictions = []
        all_labels = []
        
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            if method == "mc_dropout":
                predictions, _ = self.monte_carlo_dropout(
                    model, X, kwargs.get('n_samples', 100)
                )
            elif method == "deep_ensemble":
                models = kwargs.get('models', [model])
                predictions, _ = self.deep_ensemble(models, X)
            elif method == "temperature_scaling":
                predictions = self.calibrated_predictions(model, X)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        
        # Calculate reliability diagram
        max_probs = predictions.max(axis=1)
        predicted_classes = predictions.argmax(axis=1)
        correct_predictions = (predicted_classes == labels)
        
        # Bin the predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        return {
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
            "bin_boundaries": bin_boundaries
        }
