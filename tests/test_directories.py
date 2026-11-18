"""
tests/test_directories.py
-------------------------
Tests for project directory structure integrity, module layout, and Stage 1 asset preservation.
"""

import pathlib


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def test_required_architecture_directories_exist():
    """Verifies that all required architecture directories exist in the project."""
    root = get_project_root()

    required_dirs = [
        "configs",
        "data/downloads",
        "data/raw",
        "data/processed",
        "data/manifests",
        "models/checkpoints",
        "src/audio",
        "src/dsp",
        "src/ml",
        "src/rts",
        "src/application",
        "src/monitoring",
        "tests",
        "scripts",
        "results",
    ]

    for d in required_dirs:
        dir_path = root / d
        assert dir_path.is_dir(), f"Required directory missing: '{d}' at {dir_path}"


def test_stage1_dataset_and_reports_preserved():
    """Verifies that the Google Speech Commands dataset and Stage 1 verification assets remain intact."""
    root = get_project_root()

    # Raw dataset check
    raw_dir = root / "data" / "raw"
    assert raw_dir.is_dir(), "Dataset directory 'data/raw' is missing!"
    for core_dir in ["yes", "no", "stop", "go", "_background_noise_"]:
        assert (raw_dir / core_dir).is_dir(), f"Core dataset class folder '{core_dir}' is missing!"

    # Stage 1 verification report check
    report_file = root / "data" / "manifests" / "dataset_report.txt"
    assert report_file.is_file(), "Stage 1 report 'data/manifests/dataset_report.txt' is missing!"
    assert report_file.stat().st_size > 0, "Stage 1 report is empty!"

    # Stage 1 verification script check
    script_file = root / "scripts" / "verify_dataset.py"
    assert script_file.is_file(), "Verification script 'scripts/verify_dataset.py' is missing!"
    assert script_file.stat().st_size > 0, "Verification script is empty!"


def test_src_submodules_contain_init():
    """Verifies that all src/ subpackages contain an __init__.py file."""
    root = get_project_root()
    submodules = ["audio", "dsp", "ml", "rts", "application", "monitoring"]

    for sub in submodules:
        init_file = root / "src" / sub / "__init__.py"
        assert init_file.is_file(), f"Submodule '{sub}' missing __init__.py at {init_file}"

