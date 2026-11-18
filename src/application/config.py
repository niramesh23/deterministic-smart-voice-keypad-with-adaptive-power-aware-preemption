"""
src/application/config.py
-------------------------
Configuration loader and validator for Real-Time Voice Recognition.
Loads YAML configurations, validates required parameters, and provides
convenient access to runtime settings.
"""

import os
import pathlib
from typing import Any, Dict, Optional
import yaml


REQUIRED_SECTIONS = ["audio", "dsp", "rts", "ml", "system", "experiment"]

REQUIRED_KEYS = {
    "audio": ["sample_rate", "channels", "hop_size", "frame_size", "queue_capacity", "pre_roll_ms"],
    "dsp": ["rms_window_samples", "vad_energy_threshold_factor", "feature_type"],
    "rts": ["policy", "frame_deadline_ms", "inference_deadline_ms"],
    "ml": ["target_commands", "data_raw_dir", "data_manifests_dir"],
    "system": ["log_level"],
    "experiment": ["simulation_mode", "loop_period_ms", "max_jitter_ms"],
}


class ConfigError(ValueError):
    """Raised when configuration file is invalid or missing required keys."""
    pass


class Config:
    """Configuration wrapper offering both dict and attribute access."""

    def __init__(self, data: Dict[str, Any], config_path: Optional[pathlib.Path] = None):
        self._data = data
        self._config_path = config_path

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def config_path(self) -> Optional[pathlib.Path]:
        return self._config_path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return Config(val, self._config_path)
            return val
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Config({self._data})"


def get_project_root() -> pathlib.Path:
    """Returns the resolved root path of the project workspace."""
    # Assuming this file is located at <project_root>/src/application/config.py
    return pathlib.Path(__file__).resolve().parent.parent.parent


def validate_config(raw_cfg: Dict[str, Any]) -> None:
    """Validates that all essential sections and fields exist in the configuration."""
    if not isinstance(raw_cfg, dict):
        raise ConfigError("Configuration root must be a dictionary/mapping.")

    for section in REQUIRED_SECTIONS:
        if section not in raw_cfg:
            raise ConfigError(f"Missing required configuration section: '{section}'")

        expected_keys = REQUIRED_KEYS.get(section, [])
        section_data = raw_cfg[section]
        if not isinstance(section_data, dict):
            raise ConfigError(f"Configuration section '{section}' must be a dictionary.")

        for k in expected_keys:
            if k not in section_data:
                raise ConfigError(f"Missing required key '{k}' in section '{section}'.")

    # Specific semantic checks
    if raw_cfg["audio"]["sample_rate"] <= 0:
        raise ConfigError("audio.sample_rate must be a positive integer.")
    if raw_cfg["audio"]["channels"] != 1:
        raise ConfigError(f"Currently only mono audio (channels=1) is supported, got {raw_cfg['audio']['channels']}.")
    if raw_cfg["audio"]["hop_size"] <= 0 or raw_cfg["audio"]["frame_size"] <= 0:
        raise ConfigError("audio.hop_size and audio.frame_size must be positive integers.")
    if raw_cfg["audio"]["hop_size"] > raw_cfg["audio"]["frame_size"]:
        raise ConfigError("audio.hop_size cannot exceed audio.frame_size.")


def load_config(config_path: Optional[str | pathlib.Path] = None) -> Config:
    """
    Loads and validates YAML configuration file.

    Parameters:
        config_path: Path to YAML configuration file. Defaults to 'configs/default.yaml'
                     resolved relative to project root.

    Returns:
        Config object populated with validated settings.
    """
    project_root = get_project_root()

    if config_path is None:
        resolved_path = project_root / "configs" / "default.yaml"
    else:
        resolved_path = pathlib.Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file at {resolved_path}: {e}") from e

    validate_config(raw_data)
    return Config(raw_data, resolved_path)

