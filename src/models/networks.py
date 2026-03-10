"""
Neural network models for AI System Stress Testing.

This module provides various model architectures for stress testing.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for stress testing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout_rate: float = 0.2):
        """Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)


class RobustMLP(nn.Module):
    """Robust MLP with additional regularization for stress testing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """Initialize robust MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(RobustMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(model_type: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
        """Create a model instance.
        
        Args:
            model_type: Type of model to create
            input_dim: Input dimension
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if model_type == "simple_mlp":
            hidden_dims = kwargs.get('hidden_dims', [128, 64])
            dropout_rate = kwargs.get('dropout_rate', 0.2)
            return SimpleMLP(input_dim, hidden_dims, num_classes, dropout_rate)
        
        elif model_type == "robust_mlp":
            hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            use_batch_norm = kwargs.get('use_batch_norm', True)
            return RobustMLP(input_dim, hidden_dims, num_classes, dropout_rate, use_batch_norm)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_config(model_type: str) -> Dict:
        """Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration dictionary
        """
        configs = {
            "simple_mlp": {
                "hidden_dims": [128, 64],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "epochs": 100
            },
            "robust_mlp": {
                "hidden_dims": [256, 128, 64],
                "dropout_rate": 0.3,
                "use_batch_norm": True,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "epochs": 100
            }
        }
        
        return configs.get(model_type, {})
