"""
Core utilities for AI System Stress Testing.

This module provides common utilities for device management, seeding, logging,
and other shared functionality across the stress testing framework.
"""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the stress testing framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("stress_testing")
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    OmegaConf.save(config, save_path)


def suppress_warnings() -> None:
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir


def log_experiment_info(logger: logging.Logger, config: DictConfig, device: torch.device) -> None:
    """Log experiment configuration and environment info.
    
    Args:
        logger: Logger instance
        config: Experiment configuration
        device: PyTorch device being used
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(config))
    logger.info("=" * 50)


class ExperimentTracker:
    """Simple experiment tracking for stress testing results."""
    
    def __init__(self, experiment_dir: str):
        """Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to save experiment results
        """
        self.experiment_dir = experiment_dir
        self.results = {}
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.results:
            self.results[name] = []
        
        self.results[name].append({
            'value': value,
            'step': step
        })
    
    def save_results(self, filename: str = "results.json") -> None:
        """Save results to JSON file.
        
        Args:
            filename: Name of the results file
        """
        import json
        
        results_path = os.path.join(self.experiment_dir, filename)
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_best_metric(self, metric_name: str, mode: str = "max") -> float:
        """Get the best value for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: "max" or "min" for optimization direction
            
        Returns:
            Best metric value
        """
        if metric_name not in self.results:
            raise ValueError(f"Metric {metric_name} not found")
        
        values = [entry['value'] for entry in self.results[metric_name]]
        return max(values) if mode == "max" else min(values)
