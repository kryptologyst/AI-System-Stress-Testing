"""
Main stress testing framework for AI System Stress Testing.

This module provides the core stress testing functionality that integrates
adversarial attacks, uncertainty quantification, OOD detection, and calibration.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import set_seed, get_device, setup_logging, ExperimentTracker
from .data import DataManager
from .models import ModelFactory
from .attacks import AdversarialAttacker
from .uncertainty import UncertaintyQuantifier
from .ood import OODDetector


class StressTester:
    """Main class for AI system stress testing."""
    
    def __init__(self, config: Dict):
        """Initialize stress tester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device()
        self.logger = setup_logging(config.get('log_level', 'INFO'))
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Initialize components
        self.data_manager = DataManager(config.get('data', {}))
        self.attacker = AdversarialAttacker(self.device, config.get('attacks', {}))
        self.uncertainty_quantifier = UncertaintyQuantifier(self.device, config.get('uncertainty', {}))
        self.ood_detector = OODDetector(self.device, config.get('ood', {}))
        
        # Initialize experiment tracking
        self.experiment_dir = config.get('experiment_dir', 'experiments')
        self.tracker = ExperimentTracker(self.experiment_dir)
        
        # Model and data
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.ood_loader = None
        
    def load_data(self) -> None:
        """Load and preprocess data."""
        self.logger.info("Loading data...")
        
        data_config = self.config.get('data', {})
        dataset_type = data_config.get('dataset_type', 'synthetic')
        
        if dataset_type == 'synthetic':
            X, y, metadata = self.data_manager.load_synthetic_data(
                dataset_type=data_config.get('task_type', 'classification'),
                n_samples=data_config.get('n_samples', 1000),
                n_features=data_config.get('n_features', 20),
                n_classes=data_config.get('n_classes', 2),
                noise=data_config.get('noise', 0.1)
            )
        else:
            X, y, metadata = self.data_manager.load_sklearn_dataset(dataset_type)
        
        # Preprocess data
        X_scaled, y_encoded = self.data_manager.preprocess_data(X, y)
        
        # Create train-test split
        X_train, X_test, y_train, y_test = self.data_manager.create_train_test_split(
            X_scaled, y_encoded, 
            test_size=data_config.get('test_size', 0.2)
        )
        
        # Create OOD data
        X_id, y_id, X_ood, y_ood = self.data_manager.create_out_of_distribution_data(
            X_test, y_test, 
            ood_ratio=data_config.get('ood_ratio', 0.2)
        )
        
        # Create data loaders
        self.train_loader, self.test_loader = self.data_manager.create_dataloaders(
            X_train, y_train, X_id, y_id,
            batch_size=data_config.get('batch_size', 32)
        )
        
        # Create OOD data loader
        ood_dataset = self.data_manager.StressTestDataset(X_ood, y_ood)
        self.ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False)
        
        # Save metadata
        metadata_path = os.path.join(self.experiment_dir, 'data_metadata.json')
        self.data_manager.save_metadata(metadata, metadata_path)
        
        self.logger.info(f"Data loaded: {X_train.shape[0]} train, {X_id.shape[0]} test, {X_ood.shape[0]} OOD")
        
    def create_model(self) -> None:
        """Create and initialize model."""
        self.logger.info("Creating model...")
        
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'simple_mlp')
        
        # Get input dimension from data
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch[0].shape[1]
        
        # Get number of classes
        num_classes = len(torch.unique(sample_batch[1]))
        
        # Create model
        self.model = ModelFactory.create_model(
            model_type, input_dim, num_classes, **model_config
        ).to(self.device)
        
        self.logger.info(f"Model created: {model_type} with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_model(self) -> None:
        """Train the model."""
        self.logger.info("Training model...")
        
        model_config = self.config.get('model', {})
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config.get('learning_rate', 0.001),
            weight_decay=model_config.get('weight_decay', 1e-4)
        )
        
        criterion = nn.CrossEntropyLoss()
        epochs = model_config.get('epochs', 100)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(self.train_loader)
            
            self.tracker.log_metric('train_loss', avg_loss, epoch)
            self.tracker.log_metric('train_accuracy', accuracy, epoch)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        self.logger.info("Model training completed")
        
    def evaluate_clean_performance(self) -> Dict[str, float]:
        """Evaluate model performance on clean data."""
        self.logger.info("Evaluating clean performance...")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        self.tracker.log_metric('clean_accuracy', accuracy)
        
        self.logger.info(f"Clean accuracy: {accuracy:.4f}")
        return {'clean_accuracy': accuracy}
        
    def run_adversarial_tests(self) -> Dict[str, Dict[str, float]]:
        """Run adversarial attack tests."""
        self.logger.info("Running adversarial attack tests...")
        
        attack_config = self.config.get('attacks', {})
        attack_methods = attack_config.get('methods', ['fgsm', 'pgd'])
        epsilon_values = attack_config.get('epsilon_values', [0.01, 0.05, 0.1, 0.2])
        
        results = {}
        
        for method in attack_methods:
            self.logger.info(f"Testing {method} attack...")
            
            # Test with different epsilon values
            method_results = {}
            for epsilon in epsilon_values:
                result = self.attacker.batch_attack(
                    self.model, self.test_loader, method, epsilon=epsilon
                )
                method_results[f'epsilon_{epsilon}'] = result
                
                # Log metrics
                self.tracker.log_metric(f'{method}_accuracy_eps_{epsilon}', result['adversarial_accuracy'])
                self.tracker.log_metric(f'{method}_perturbation_eps_{epsilon}', result['avg_perturbation'])
            
            results[method] = method_results
            
            # Generate robustness curve
            robustness_curve = self.attacker.robustness_curve(
                self.model, 
                next(iter(self.test_loader))[0].to(self.device),
                next(iter(self.test_loader))[1].to(self.device),
                method, 
                epsilon_values
            )
            
            # Plot robustness curve
            self._plot_robustness_curve(method, robustness_curve)
        
        return results
        
    def run_uncertainty_tests(self) -> Dict[str, Dict[str, float]]:
        """Run uncertainty quantification tests."""
        self.logger.info("Running uncertainty quantification tests...")
        
        uncertainty_config = self.config.get('uncertainty', {})
        methods = uncertainty_config.get('methods', ['mc_dropout', 'temperature_scaling'])
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Testing {method} uncertainty quantification...")
            
            if method == 'temperature_scaling':
                # Fit temperature scaling on validation data
                val_data = next(iter(self.test_loader))
                X_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)
                self.uncertainty_quantifier.temperature_scaling(self.model, X_val, y_val)
            
            # Evaluate calibration
            calibration_results = self.uncertainty_quantifier.evaluate_calibration(
                self.model, self.test_loader, method
            )
            
            results[method] = calibration_results
            
            # Log metrics
            for metric, value in calibration_results.items():
                self.tracker.log_metric(f'{method}_{metric}', value)
            
            # Generate reliability diagram
            reliability_data = self.uncertainty_quantifier.reliability_diagram(
                self.model, self.test_loader, method
            )
            
            # Plot reliability diagram
            self._plot_reliability_diagram(method, reliability_data)
        
        return results
        
    def run_ood_tests(self) -> Dict[str, Dict[str, float]]:
        """Run out-of-distribution detection tests."""
        self.logger.info("Running OOD detection tests...")
        
        ood_config = self.config.get('ood', {})
        methods = ood_config.get('methods', ['energy', 'max_softmax', 'entropy'])
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Testing {method} OOD detection...")
            
            # Evaluate OOD detection
            ood_results = self.ood_detector.evaluate_ood_detection(
                self.model, self.test_loader, self.ood_loader, method
            )
            
            results[method] = ood_results
            
            # Log metrics
            for metric, value in ood_results.items():
                self.tracker.log_metric(f'{method}_{metric}', value)
            
            # Generate calibration analysis
            calibration_data = self.ood_detector.calibration_analysis(
                self.model, self.test_loader, self.ood_loader, method
            )
            
            # Plot OOD detection results
            self._plot_ood_detection(method, calibration_data)
        
        return results
        
    def run_stress_tests(self) -> Dict[str, Dict]:
        """Run all stress tests."""
        self.logger.info("Starting comprehensive stress testing...")
        
        # Load data and create model
        self.load_data()
        self.create_model()
        
        # Train model
        self.train_model()
        
        # Run all stress tests
        results = {
            'clean_performance': self.evaluate_clean_performance(),
            'adversarial_tests': self.run_adversarial_tests(),
            'uncertainty_tests': self.run_uncertainty_tests(),
            'ood_tests': self.run_ood_tests()
        }
        
        # Save results
        self.tracker.save_results()
        
        # Generate summary report
        self._generate_summary_report(results)
        
        self.logger.info("Stress testing completed!")
        return results
        
    def _plot_robustness_curve(self, method: str, curve_data: Dict[str, List[float]]) -> None:
        """Plot robustness curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(curve_data['epsilon_values'], curve_data['accuracies'], 'o-', label=f'{method.upper()} Attack')
        plt.xlabel('Epsilon (Attack Strength)')
        plt.ylabel('Adversarial Accuracy')
        plt.title(f'Robustness Curve - {method.upper()} Attack')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.experiment_dir, f'robustness_curve_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_reliability_diagram(self, method: str, reliability_data: Dict[str, np.ndarray]) -> None:
        """Plot reliability diagram."""
        plt.figure(figsize=(10, 8))
        
        bin_confidences = reliability_data['bin_confidences']
        bin_accuracies = reliability_data['bin_accuracies']
        bin_counts = reliability_data['bin_counts']
        
        # Plot reliability diagram
        plt.subplot(2, 1, 1)
        plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, label='Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram - {method.upper()}')
        plt.legend()
        plt.grid(True)
        
        # Plot bin counts
        plt.subplot(2, 1, 2)
        plt.bar(bin_confidences, bin_counts, width=0.1, alpha=0.7)
        plt.xlabel('Confidence')
        plt.ylabel('Sample Count')
        plt.title('Sample Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.experiment_dir, f'reliability_diagram_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_ood_detection(self, method: str, calibration_data: Dict[str, np.ndarray]) -> None:
        """Plot OOD detection results."""
        plt.figure(figsize=(12, 5))
        
        # Plot score distributions
        plt.subplot(1, 2, 1)
        plt.hist(calibration_data['id_scores'], bins=50, alpha=0.7, label='In-Distribution', density=True)
        plt.hist(calibration_data['ood_scores'], bins=50, alpha=0.7, label='Out-of-Distribution', density=True)
        plt.xlabel('Detection Score')
        plt.ylabel('Density')
        plt.title(f'OOD Detection Scores - {method.upper()}')
        plt.legend()
        plt.grid(True)
        
        # Plot score separation
        plt.subplot(1, 2, 2)
        separation = calibration_data['score_separation']
        plt.bar(['Score Separation'], [separation])
        plt.ylabel('Mean Score Difference')
        plt.title('ID vs OOD Score Separation')
        plt.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.experiment_dir, f'ood_detection_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_summary_report(self, results: Dict[str, Dict]) -> None:
        """Generate summary report."""
        report_path = os.path.join(self.experiment_dir, 'stress_test_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# AI System Stress Testing Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This report summarizes the results of comprehensive stress testing on the AI system.\n\n")
            
            # Clean performance
            f.write("## Clean Performance\n\n")
            clean_acc = results['clean_performance']['clean_accuracy']
            f.write(f"- **Clean Accuracy**: {clean_acc:.4f}\n\n")
            
            # Adversarial tests
            f.write("## Adversarial Robustness\n\n")
            for method, method_results in results['adversarial_tests'].items():
                f.write(f"### {method.upper()} Attack\n\n")
                for eps, eps_results in method_results.items():
                    f.write(f"- **{eps}**: Accuracy={eps_results['adversarial_accuracy']:.4f}, "
                           f"Perturbation={eps_results['avg_perturbation']:.4f}\n")
                f.write("\n")
            
            # Uncertainty tests
            f.write("## Uncertainty Quantification\n\n")
            for method, method_results in results['uncertainty_tests'].items():
                f.write(f"### {method.upper()}\n\n")
                f.write(f"- **ECE**: {method_results['ece']:.4f}\n")
                f.write(f"- **Brier Score**: {method_results['brier_score']:.4f}\n")
                f.write(f"- **NLL**: {method_results['nll']:.4f}\n\n")
            
            # OOD tests
            f.write("## Out-of-Distribution Detection\n\n")
            for method, method_results in results['ood_tests'].items():
                f.write(f"### {method.upper()}\n\n")
                f.write(f"- **AUROC**: {method_results['auroc']:.4f}\n")
                f.write(f"- **FPR@95%TPR**: {method_results['fpr_at_95_tpr']:.4f}\n")
                f.write(f"- **Detection Accuracy**: {method_results['detection_accuracy']:.4f}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Adversarial Robustness**: Consider adversarial training or robust optimization techniques.\n")
            f.write("2. **Uncertainty Quantification**: Implement temperature scaling or ensemble methods for better calibration.\n")
            f.write("3. **OOD Detection**: Use energy-based methods for better out-of-distribution detection.\n")
            f.write("4. **Monitoring**: Implement continuous monitoring of model performance and uncertainty.\n\n")
            
            f.write("## Limitations\n\n")
            f.write("- Results are based on synthetic data and may not generalize to real-world scenarios.\n")
            f.write("- Stress testing is not a substitute for comprehensive validation.\n")
            f.write("- Adversarial examples are artificially generated and may not represent real attacks.\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
