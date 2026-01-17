"""
Logging utilities for experiments
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path("experiments/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "experiment", log_file: str = None) -> logging.Logger:
    """
    Setup a logger for experiments
    
    Args:
        name: Logger name
        log_file: Optional log file name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"experiment_{timestamp}.log"
    
    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def log_experiment_start(logger: logging.Logger, config: dict):
    """Log experiment start with configuration"""
    logger.info("=" * 70)
    logger.info("EXPERIMENT STARTED")
    logger.info("=" * 70)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 70)


def log_experiment_end(logger: logging.Logger, results: dict):
    """Log experiment end with results"""
    logger.info("-" * 70)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("-" * 70)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 70)


def log_model_result(logger: logging.Logger, model_name: str, metrics: dict):
    """Log model evaluation results"""
    logger.info(f"Model: {model_name}")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
