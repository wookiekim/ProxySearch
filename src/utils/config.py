"""Configuration loading and merging utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a YAML configuration file with support for defaults inheritance.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        DictConfig containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Handle defaults inheritance (similar to Hydra)
    if "defaults" in config:
        defaults = config.pop("defaults")
        base_configs = []

        for default in defaults:
            if isinstance(default, str):
                # Resolve relative path from current config's directory
                default_path = config_path.parent / f"{default}.yaml"
                if default_path.exists():
                    base_config = load_config(default_path)
                    base_configs.append(base_config)

        # Merge: base configs first, then current config on top
        if base_configs:
            merged = base_configs[0]
            for bc in base_configs[1:]:
                merged = OmegaConf.merge(merged, bc)
            config = OmegaConf.merge(merged, config)

    return config


def merge_configs(
    base_config: DictConfig,
    override_configs: Optional[List[DictConfig]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Merge multiple configurations with priority ordering.

    Priority (highest to lowest):
    1. CLI overrides
    2. Override configs (later configs override earlier)
    3. Base config

    Args:
        base_config: Base configuration to start from.
        override_configs: List of configs to merge on top of base.
        cli_overrides: Dictionary of CLI overrides (highest priority).

    Returns:
        Merged DictConfig.
    """
    configs_to_merge = [base_config]

    if override_configs:
        configs_to_merge.extend(override_configs)

    if cli_overrides:
        configs_to_merge.append(OmegaConf.create(cli_overrides))

    return OmegaConf.merge(*configs_to_merge)


def load_config_with_defaults(
    config_path: Union[str, Path],
    default_config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load a configuration file with optional defaults and CLI overrides.

    Args:
        config_path: Path to the main configuration file.
        default_config_path: Path to default configuration (optional).
        cli_overrides: Dictionary of CLI overrides.

    Returns:
        Merged DictConfig.
    """
    if default_config_path is not None:
        base_config = load_config(default_config_path)
        override_config = load_config(config_path)
        return merge_configs(base_config, [override_config], cli_overrides)
    else:
        config = load_config(config_path)
        if cli_overrides:
            return merge_configs(config, cli_overrides=cli_overrides)
        return config


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save a configuration to a YAML file.

    Args:
        config: Configuration to save.
        save_path: Path to save the configuration file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """
    Convert a DictConfig to a plain Python dictionary.

    Args:
        config: Configuration to convert.

    Returns:
        Plain Python dictionary.
    """
    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: DictConfig, required_keys: List[str]) -> None:
    """
    Validate that required keys are present in the configuration.

    Args:
        config: Configuration to validate.
        required_keys: List of required key paths (e.g., ["model.name", "search.num_candidates"]).

    Raises:
        KeyError: If a required key is missing.
    """
    for key in required_keys:
        parts = key.split(".")
        current = config
        for part in parts:
            if part not in current:
                raise KeyError(f"Required configuration key missing: {key}")
            current = current[part]
