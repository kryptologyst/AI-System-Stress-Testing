#!/usr/bin/env python3
"""
Main script for running AI System Stress Testing.

This script provides a command-line interface for running comprehensive
stress tests on AI models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import yaml
from omegaconf import DictConfig, OmegaConf

from src.stress_tester import StressTester
from src.utils import create_experiment_dir, log_experiment_info


def main():
    """Main function for running stress tests."""
    parser = argparse.ArgumentParser(description="AI System Stress Testing")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="stress_test",
                       help="Name for the experiment")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with reduced parameters")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    if args.quick:
        # Quick test configuration
        config.data.n_samples = 200
        config.data.n_features = 10
        config.model.epochs = 20
        config.attacks.epsilon_values = [0.1, 0.2]
        config.uncertainty.n_samples = 20
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    config.experiment_dir = experiment_dir
    
    # Initialize stress tester
    stress_tester = StressTester(config)
    
    # Log experiment info
    log_experiment_info(stress_tester.logger, config, stress_tester.device)
    
    # Run stress tests
    try:
        results = stress_tester.run_stress_tests()
        
        print("\n" + "="*50)
        print("STRESS TESTING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Results saved to: {experiment_dir}")
        print(f"Summary report: {experiment_dir}/stress_test_report.md")
        
        # Print key results
        print("\nKey Results:")
        print(f"- Clean Accuracy: {results['clean_performance']['clean_accuracy']:.4f}")
        
        for method, method_results in results['adversarial_tests'].items():
            eps_01_acc = method_results.get('epsilon_0.1', {}).get('adversarial_accuracy', 0)
            print(f"- {method.upper()} Attack (ε=0.1): {eps_01_acc:.4f}")
        
        for method, method_results in results['uncertainty_tests'].items():
            ece = method_results.get('ece', 0)
            print(f"- {method.upper()} ECE: {ece:.4f}")
        
        for method, method_results in results['ood_tests'].items():
            auroc = method_results.get('auroc', 0)
            print(f"- {method.upper()} OOD AUROC: {auroc:.4f}")
        
    except Exception as e:
        print(f"Error during stress testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
