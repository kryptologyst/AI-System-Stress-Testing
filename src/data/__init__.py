"""
Data loading and preprocessing utilities for AI System Stress Testing.

This module provides data loading, preprocessing, and synthetic data generation
capabilities for stress testing various AI models.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class StressTestDataset(Dataset):
    """PyTorch Dataset wrapper for stress testing data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[callable] = None):
        """Initialize dataset.
        
        Args:
            X: Input features
            y: Target labels
            transform: Optional data transformation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y.dtype in [np.int32, np.int64] else torch.FloatTensor(y)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class DataManager:
    """Manages data loading, preprocessing, and augmentation for stress testing."""
    
    def __init__(self, config: Dict):
        """Initialize data manager.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        
    def load_synthetic_data(self, 
                          dataset_type: str = "classification",
                          n_samples: int = 1000,
                          n_features: int = 20,
                          n_classes: int = 2,
                          noise: float = 0.1,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate synthetic dataset for stress testing.
        
        Args:
            dataset_type: Type of dataset ("classification" or "regression")
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes (for classification)
            noise: Amount of noise to add
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, metadata)
        """
        if dataset_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=2,
                n_informative=n_features-2,
                n_clusters_per_class=1,
                noise=noise,
                random_state=random_state
            )
            self.target_names = [f"class_{i}" for i in range(n_classes)]
        elif dataset_type == "regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=random_state
            )
            self.target_names = ["target"]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        metadata = {
            "dataset_type": dataset_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes if dataset_type == "classification" else None,
            "noise": noise,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "sensitive_attributes": [],  # No sensitive attributes in synthetic data
            "monotonic_features": [],    # No monotonicity constraints
            "feature_ranges": {
                name: (X[:, i].min(), X[:, i].max()) 
                for i, name in enumerate(self.feature_names)
            }
        }
        
        return X, y, metadata
    
    def load_sklearn_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load a standard sklearn dataset.
        
        Args:
            dataset_name: Name of the dataset ("iris", "wine", "breast_cancer")
            
        Returns:
            Tuple of (X, y, metadata)
        """
        dataset_loaders = {
            "iris": load_iris,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer
        }
        
        if dataset_name not in dataset_loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        data = dataset_loaders[dataset_name]()
        X, y = data.data, data.target
        
        self.feature_names = data.feature_names if hasattr(data, 'feature_names') else None
        self.target_names = data.target_names if hasattr(data, 'target_names') else None
        
        metadata = {
            "dataset_name": dataset_name,
            "dataset_type": "classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "sensitive_attributes": [],  # Standard datasets don't have sensitive attributes
            "monotonic_features": [],
            "feature_ranges": {
                name: (X[:, i].min(), X[:, i].max()) 
                for i, name in enumerate(self.feature_names or [f"feature_{i}" for i in range(X.shape[1])])
            }
        }
        
        return X, y, metadata
    
    def preprocess_data(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data.
        
        Args:
            X: Input features
            y: Target labels
            fit_scaler: Whether to fit the scaler or use existing one
            
        Returns:
            Tuple of (X_scaled, y_encoded)
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # For classification, encode labels if they're strings
        if y.dtype == object or (hasattr(y, 'dtype') and 'str' in str(y.dtype)):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.copy()
        
        return X_scaled, y_encoded
    
    def create_train_test_split(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train-test split.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def create_imbalanced_data(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              imbalance_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create imbalanced dataset for stress testing.
        
        Args:
            X: Input features
            y: Target labels
            imbalance_ratio: Ratio of minority class to majority class
            
        Returns:
            Tuple of (X_imbalanced, y_imbalanced)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Find majority and minority classes
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Get indices for each class
        majority_indices = np.where(y == majority_class)[0]
        minority_indices = np.where(y == minority_class)[0]
        
        # Calculate how many minority samples to keep
        n_minority_samples = int(len(minority_indices) * imbalance_ratio)
        
        # Randomly sample minority class
        np.random.seed(42)
        selected_minority = np.random.choice(minority_indices, n_minority_samples, replace=False)
        
        # Combine majority class with selected minority samples
        selected_indices = np.concatenate([majority_indices, selected_minority])
        
        return X[selected_indices], y[selected_indices]
    
    def add_noise(self, X: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to the data.
        
        Args:
            X: Input features
            noise_level: Standard deviation of noise
            
        Returns:
            Noisy data
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    def create_out_of_distribution_data(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray,
                                       ood_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create out-of-distribution data for stress testing.
        
        Args:
            X: Input features
            y: Target labels
            ood_ratio: Fraction of data to make OOD
            
        Returns:
            Tuple of (X_id, y_id, X_ood, y_ood)
        """
        n_ood_samples = int(len(X) * ood_ratio)
        
        # Randomly select samples to make OOD
        np.random.seed(42)
        ood_indices = np.random.choice(len(X), n_ood_samples, replace=False)
        id_indices = np.setdiff1d(np.arange(len(X)), ood_indices)
        
        # Create OOD data by adding significant noise/shift
        X_ood = X[ood_indices] + np.random.normal(0, 2.0, (n_ood_samples, X.shape[1]))
        y_ood = y[ood_indices]
        
        X_id = X[id_indices]
        y_id = y[id_indices]
        
        return X_id, y_id, X_ood, y_ood
    
    def save_metadata(self, metadata: Dict, save_path: str) -> None:
        """Save dataset metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            save_path: Path to save metadata
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, metadata_path: str) -> Dict:
        """Load dataset metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Metadata dictionary
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def create_dataloaders(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_test: np.ndarray, 
                          y_test: np.ndarray,
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for DataLoader
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_dataset = StressTestDataset(X_train, y_train)
        test_dataset = StressTestDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
