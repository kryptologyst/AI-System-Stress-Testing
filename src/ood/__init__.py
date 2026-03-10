"""
Out-of-distribution detection methods for AI System Stress Testing.

This module provides various OOD detection techniques including energy-based,
Mahalanobis distance, and statistical methods.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
import alibi_detect
from alibi_detect.od import Mahalanobis, LSVDD, OutlierVAE


class OODDetector:
    """Main class for out-of-distribution detection."""
    
    def __init__(self, device: torch.device, config: Dict):
        """Initialize OOD detector.
        
        Args:
            device: PyTorch device
            config: Configuration dictionary
        """
        self.device = device
        self.config = config
        self.feature_extractor = None
        self.mahalanobis_detector = None
        self.energy_threshold = None
        
    def energy_based_detection(self, 
                              model: nn.Module, 
                              X: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        """Perform energy-based OOD detection.
        
        Args:
            model: Neural network model
            X: Input data
            temperature: Temperature scaling for energy calculation
            
        Returns:
            Energy scores (lower = more likely OOD)
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(X)
            # Energy = -log(sum(exp(logits)))
            energy = -torch.logsumexp(logits / temperature, dim=1)
        
        return energy
    
    def mahalanobis_distance(self, 
                            model: nn.Module, 
                            X: torch.Tensor,
                            X_train: torch.Tensor,
                            y_train: torch.Tensor) -> torch.Tensor:
        """Calculate Mahalanobis distance for OOD detection.
        
        Args:
            model: Neural network model
            X: Input data
            X_train: Training data for fitting Mahalanobis detector
            y_train: Training labels
            
        Returns:
            Mahalanobis distances
        """
        model.eval()
        
        # Extract features from the last layer before classification
        with torch.no_grad():
            features = self._extract_features(model, X)
            train_features = self._extract_features(model, X_train)
        
        # Fit Mahalanobis detector on training features
        if self.mahalanobis_detector is None:
            self.mahalanobis_detector = Mahalanobis()
            self.mahalanobis_detector.fit(train_features.cpu().numpy())
        
        # Calculate Mahalanobis distances
        distances = self.mahalanobis_detector.score(features.cpu().numpy())
        
        return torch.FloatTensor(distances).to(self.device)
    
    def _extract_features(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        """Extract features from the model's penultimate layer.
        
        Args:
            model: Neural network model
            X: Input data
            
        Returns:
            Extracted features
        """
        # This is a simplified version - in practice, you'd need to modify
        # the model to return intermediate features
        model.eval()
        
        with torch.no_grad():
            # For now, use the logits as features
            # In practice, you'd extract from the penultimate layer
            features = model(X)
        
        return features
    
    def max_softmax_probability(self, 
                               model: nn.Module, 
                               X: torch.Tensor) -> torch.Tensor:
        """Calculate maximum softmax probability for OOD detection.
        
        Args:
            model: Neural network model
            X: Input data
            
        Returns:
            Maximum softmax probabilities
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
        
        return max_probs
    
    def entropy_based_detection(self, 
                               model: nn.Module, 
                               X: torch.Tensor) -> torch.Tensor:
        """Calculate entropy for OOD detection.
        
        Args:
            model: Neural network model
            X: Input data
            
        Returns:
            Entropy values (higher = more likely OOD)
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        return entropy
    
    def evaluate_ood_detection(self, 
                             model: nn.Module,
                             id_dataloader: DataLoader,
                             ood_dataloader: DataLoader,
                             method: str = "energy",
                             **kwargs) -> Dict[str, float]:
        """Evaluate OOD detection performance.
        
        Args:
            model: Neural network model
            id_dataloader: In-distribution data
            ood_dataloader: Out-of-distribution data
            method: OOD detection method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with OOD detection metrics
        """
        # Collect scores for ID data
        id_scores = []
        for X, _ in id_dataloader:
            X = X.to(self.device)
            
            if method == "energy":
                scores = self.energy_based_detection(model, X, **kwargs)
            elif method == "mahalanobis":
                scores = self.mahalanobis_distance(model, X, **kwargs)
            elif method == "max_softmax":
                scores = self.max_softmax_probability(model, X)
            elif method == "entropy":
                scores = self.entropy_based_detection(model, X)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            id_scores.append(scores.cpu().numpy())
        
        # Collect scores for OOD data
        ood_scores = []
        for X, _ in ood_dataloader:
            X = X.to(self.device)
            
            if method == "energy":
                scores = self.energy_based_detection(model, X, **kwargs)
            elif method == "mahalanobis":
                scores = self.mahalanobis_distance(model, X, **kwargs)
            elif method == "max_softmax":
                scores = self.max_softmax_probability(model, X)
            elif method == "entropy":
                scores = self.entropy_based_detection(model, X)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            ood_scores.append(scores.cpu().numpy())
        
        # Concatenate scores
        id_scores = np.concatenate(id_scores)
        ood_scores = np.concatenate(ood_scores)
        
        # Create labels (0 for ID, 1 for OOD)
        id_labels = np.zeros(len(id_scores))
        ood_labels = np.ones(len(ood_scores))
        
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([id_labels, ood_labels])
        
        # Calculate AUROC
        if method in ["energy", "mahalanobis", "entropy"]:
            # For these methods, higher scores = more likely OOD
            auroc = roc_auc_score(all_labels, all_scores)
        else:
            # For max_softmax, lower scores = more likely OOD
            auroc = roc_auc_score(all_labels, -all_scores)
        
        # Calculate AUPRC
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        # Calculate FPR at 95% TPR
        tpr_95_idx = np.where(tpr >= 0.95)[0]
        if len(tpr_95_idx) > 0:
            fpr_95 = fpr[tpr_95_idx[0]]
        else:
            fpr_95 = 1.0
        
        # Calculate detection accuracy at optimal threshold
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        predictions = (all_scores >= optimal_threshold).astype(int)
        detection_accuracy = (predictions == all_labels).mean()
        
        return {
            "auroc": auroc,
            "fpr_at_95_tpr": fpr_95,
            "detection_accuracy": detection_accuracy,
            "optimal_threshold": optimal_threshold,
            "id_mean_score": np.mean(id_scores),
            "id_std_score": np.std(id_scores),
            "ood_mean_score": np.mean(ood_scores),
            "ood_std_score": np.std(ood_scores)
        }
    
    def threshold_based_detection(self, 
                                 model: nn.Module,
                                 X: torch.Tensor,
                                 method: str = "energy",
                                 threshold: Optional[float] = None,
                                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform threshold-based OOD detection.
        
        Args:
            model: Neural network model
            X: Input data
            method: OOD detection method
            threshold: Detection threshold (if None, uses optimal threshold)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (scores, ood_predictions)
        """
        if method == "energy":
            scores = self.energy_based_detection(model, X, **kwargs)
        elif method == "mahalanobis":
            scores = self.mahalanobis_distance(model, X, **kwargs)
        elif method == "max_softmax":
            scores = self.max_softmax_probability(model, X)
        elif method == "entropy":
            scores = self.entropy_based_detection(model, X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if threshold is None:
            # Use a simple heuristic threshold
            if method in ["energy", "mahalanobis", "entropy"]:
                threshold = scores.mean() + 2 * scores.std()
            else:
                threshold = scores.mean() - 2 * scores.std()
        
        # Predict OOD samples
        if method in ["energy", "mahalanobis", "entropy"]:
            ood_predictions = scores > threshold
        else:
            ood_predictions = scores < threshold
        
        return scores, ood_predictions
    
    def calibration_analysis(self, 
                           model: nn.Module,
                           id_dataloader: DataLoader,
                           ood_dataloader: DataLoader,
                           method: str = "energy",
                           **kwargs) -> Dict[str, np.ndarray]:
        """Analyze calibration of OOD detection scores.
        
        Args:
            model: Neural network model
            id_dataloader: In-distribution data
            ood_dataloader: Out-of-distribution data
            method: OOD detection method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with calibration analysis results
        """
        # Get scores for both ID and OOD data
        id_scores = []
        for X, _ in id_dataloader:
            X = X.to(self.device)
            if method == "energy":
                scores = self.energy_based_detection(model, X, **kwargs)
            elif method == "mahalanobis":
                scores = self.mahalanobis_distance(model, X, **kwargs)
            elif method == "max_softmax":
                scores = self.max_softmax_probability(model, X)
            elif method == "entropy":
                scores = self.entropy_based_detection(model, X)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            id_scores.append(scores.cpu().numpy())
        
        ood_scores = []
        for X, _ in ood_dataloader:
            X = X.to(self.device)
            if method == "energy":
                scores = self.energy_based_detection(model, X, **kwargs)
            elif method == "mahalanobis":
                scores = self.mahalanobis_distance(model, X, **kwargs)
            elif method == "max_softmax":
                scores = self.max_softmax_probability(model, X)
            elif method == "entropy":
                scores = self.entropy_based_detection(model, X)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            ood_scores.append(scores.cpu().numpy())
        
        id_scores = np.concatenate(id_scores)
        ood_scores = np.concatenate(ood_scores)
        
        return {
            "id_scores": id_scores,
            "ood_scores": ood_scores,
            "id_mean": np.mean(id_scores),
            "id_std": np.std(id_scores),
            "ood_mean": np.mean(ood_scores),
            "ood_std": np.std(ood_scores),
            "score_separation": np.mean(ood_scores) - np.mean(id_scores)
        }
