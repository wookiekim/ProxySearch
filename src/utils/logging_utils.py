"""Logging utilities for experiment tracking."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    experiment_name: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """
    Set up logging for an experiment.

    Args:
        log_dir: Directory to save log files (optional).
        log_level: Logging level (default: INFO).
        experiment_name: Name of the experiment for log file naming.
        use_wandb: Whether to initialize Weights & Biases logging.
        wandb_project: W&B project name (required if use_wandb=True).
        wandb_config: Configuration dict to log to W&B.

    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("proxysearch")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or "experiment"
        log_file = log_dir / f"{exp_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Initialize W&B if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed, skipping W&B initialization")
        else:
            if wandb_project is None:
                wandb_project = "proxysearch"
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config,
            )
            logger.info(f"Initialized W&B project: {wandb_project}")

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
    logger: Optional[logging.Logger] = None,
    use_wandb: bool = False,
) -> None:
    """
    Log metrics to console and optionally to W&B.

    Args:
        metrics: Dictionary of metric names to values.
        step: Global step/iteration number (for W&B).
        prefix: Prefix to add to metric names.
        logger: Logger instance (uses default if None).
        use_wandb: Whether to log to W&B.
    """
    if logger is None:
        logger = logging.getLogger("proxysearch")

    # Add prefix to metric names
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    # Log to console
    metrics_str = ", ".join(f"{k}: {_format_value(v)}" for k, v in metrics.items())
    step_str = f"[Step {step}] " if step is not None else ""
    logger.info(f"{step_str}{metrics_str}")

    # Log to W&B
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def _format_value(value: Any) -> str:
    """Format a value for logging."""
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return f"{value.item():.4f}"
        return f"array(shape={value.shape})"
    return str(value)


def log_images(
    images: List[Any],
    prompts: List[str],
    scores: Optional[List[float]] = None,
    step: Optional[int] = None,
    prefix: str = "samples",
    use_wandb: bool = False,
) -> None:
    """
    Log images to W&B.

    Args:
        images: List of PIL images to log.
        prompts: Corresponding prompts for each image.
        scores: Optional scores for each image.
        step: Global step/iteration number.
        prefix: Prefix for the image log name.
        use_wandb: Whether to log to W&B.
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb_images = []
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        caption = prompt
        if scores is not None:
            caption = f"{prompt} (score: {scores[i]:.4f})"
        wandb_images.append(wandb.Image(img, caption=caption))

    wandb.log({prefix: wandb_images}, step=step)


def log_histogram(
    values: Union[List[float], np.ndarray],
    name: str,
    step: Optional[int] = None,
    use_wandb: bool = False,
) -> None:
    """
    Log a histogram to W&B.

    Args:
        values: Values to create histogram from.
        name: Name of the histogram.
        step: Global step/iteration number.
        use_wandb: Whether to log to W&B.
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb.log({name: wandb.Histogram(values)}, step=step)


def finish_logging(use_wandb: bool = False) -> None:
    """
    Finish logging session and cleanup.

    Args:
        use_wandb: Whether W&B was used.
    """
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
