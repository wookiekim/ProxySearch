"""Reproducibility utilities for seeding random number generators."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(
    seed: Optional[int] = None,
    device: str = "cpu",
) -> torch.Generator:
    """
    Create a PyTorch random number generator with optional seed.

    Args:
        seed: Random seed (optional). If None, uses random seed.
        device: Device for the generator ("cpu" or "cuda").

    Returns:
        Seeded torch.Generator instance.
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def get_prompt_seed(base_seed: int, prompt_id: str) -> int:
    """
    Generate a deterministic seed for a specific prompt.

    This ensures that the same prompt always gets the same noise vectors
    across different runs, while different prompts get different seeds.

    Args:
        base_seed: Base seed for the experiment.
        prompt_id: Unique identifier for the prompt.

    Returns:
        Deterministic seed for this prompt.
    """
    # Use hash of prompt_id combined with base_seed
    prompt_hash = hash(prompt_id) & 0xFFFFFFFF  # Ensure positive 32-bit integer
    return (base_seed + prompt_hash) % (2**31)


def get_candidate_seed(base_seed: int, prompt_id: str, candidate_idx: int) -> int:
    """
    Generate a deterministic seed for a specific candidate of a prompt.

    Args:
        base_seed: Base seed for the experiment.
        prompt_id: Unique identifier for the prompt.
        candidate_idx: Index of the candidate (0 to N-1).

    Returns:
        Deterministic seed for this specific candidate.
    """
    prompt_seed = get_prompt_seed(base_seed, prompt_id)
    return (prompt_seed + candidate_idx) % (2**31)


class SeededRandomState:
    """Context manager for temporarily setting a seed."""

    def __init__(self, seed: int):
        """
        Initialize with a seed.

        Args:
            seed: The seed to use within the context.
        """
        self.seed = seed
        self._random_state: Optional[tuple] = None
        self._np_state: Optional[dict] = None
        self._torch_state: Optional[torch.Tensor] = None
        self._cuda_state: Optional[torch.Tensor] = None

    def __enter__(self) -> "SeededRandomState":
        """Save current state and set new seed."""
        # Save current states
        self._random_state = random.getstate()
        self._np_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._cuda_state = torch.cuda.get_rng_state()

        # Set new seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore previous state."""
        if self._random_state is not None:
            random.setstate(self._random_state)
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        if self._torch_state is not None:
            torch.set_rng_state(self._torch_state)
        if self._cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(self._cuda_state)
