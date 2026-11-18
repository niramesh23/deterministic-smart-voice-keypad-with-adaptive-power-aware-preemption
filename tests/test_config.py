"""
tests/test_config.py
--------------------
Tests for configuration loading, schema validation, and parameter consistency.
"""

import pathlib
import pytest
from src.application.config import load_config, validate_config, Config, ConfigError


def test_load_default_config():
    """Verifies that the default YAML configuration loads cleanly and passes validation."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.config_path is not None
    assert cfg.config_path.is_file()


def test_audio_configuration_parameters():
    """Validates that initial audio configuration satisfies real-time project constraints."""
    cfg = load_config()

    # Target: 16,000 Hz sample rate
    assert cfg.audio.sample_rate == 16000
    # Target: Mono audio (1 channel)
    assert cfg.audio.channels == 1
    # Target: Hop size 320 samples (20 ms @ 16 kHz)
    assert cfg.audio.hop_size == 320
    # Target: Frame size 640 samples (40 ms @ 16 kHz)
    assert cfg.audio.frame_size == 640
    # Target: Pre-roll 200 ms (captures speech onset before trigger)
    assert cfg.audio.pre_roll_ms == 200
    # Target: Configurable queue capacity
    assert cfg.audio.queue_capacity > 0
    assert isinstance(cfg.audio.queue_capacity, int)


def test_rts_and_experiment_parameters():
    """Verifies scheduling policy, timing thresholds, and simulation parameters."""
    cfg = load_config()

    assert cfg.rts.policy in ["EDF", "RM", "FIFO"]
    assert cfg.rts.frame_deadline_ms > 0
    assert cfg.rts.inference_deadline_ms > 0

    assert isinstance(cfg.experiment.simulation_mode, bool)
    assert cfg.experiment.loop_period_ms > 0
    assert cfg.experiment.max_jitter_ms > 0


def test_ml_and_paths_no_hardcoded_user_paths():
    """Verifies dataset and checkpoint paths are relative and do not embed user-specific paths."""
    cfg = load_config()

    for path_key in ["data_raw_dir", "data_processed_dir", "data_manifests_dir", "checkpoints_dir"]:
        path_str = cfg.ml.get(path_key)
        assert path_str is not None
        p = pathlib.Path(path_str)
        # Must NOT be absolute (no hardcoded Windows drive letters or user directories)
        assert not p.is_absolute(), f"Path '{path_key}' must be relative, got: {path_str}"

    assert len(cfg.ml.target_commands) == 10
    assert "yes" in cfg.ml.target_commands
    assert "stop" in cfg.ml.target_commands


def test_missing_config_raises_file_not_found():
    """Verifies that attempting to load a non-existent configuration raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file_12345.yaml")


def test_invalid_config_validation():
    """Verifies that invalid configuration payloads trigger ConfigError."""
    # Test missing section
    with pytest.raises(ConfigError, match="Missing required configuration section"):
        validate_config({"other_section": {}})

    # Test missing key in section
    with pytest.raises(ConfigError, match="Missing required key 'sample_rate' in section 'audio'"):
        validate_config({"audio": {}})

    # Test invalid channels
    invalid_audio = {
        "audio": {
            "sample_rate": 16000,
            "channels": 2,  # Stereo is unsupported
            "hop_size": 320,
            "frame_size": 640,
            "queue_capacity": 50,
            "pre_roll_ms": 200,
        },
        "dsp": {"rms_window_samples": 320, "vad_energy_threshold_factor": 2.5, "feature_type": "mfcc"},
        "rts": {"policy": "EDF", "frame_deadline_ms": 20.0, "inference_deadline_ms": 100.0},
        "ml": {"target_commands": ["yes"], "data_raw_dir": "data/raw", "data_manifests_dir": "data/manifests"},
        "system": {"log_level": "INFO"},
        "experiment": {"simulation_mode": True, "loop_period_ms": 20.0, "max_jitter_ms": 5.0},
    }
    with pytest.raises(ConfigError, match="only mono audio"):
        validate_config(invalid_audio)
