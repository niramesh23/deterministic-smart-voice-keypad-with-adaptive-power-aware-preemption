"""
scripts/precompute_21class_features.py
--------------------------------------
Precompute Log-Mel spectrogram features for 21-class dataset splits.
"""

import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset import precompute_dataset_features

def main():
    print("=" * 70)
    print("PRECOMPUTING FEATURES FOR 21-CLASS DATASET")
    print("=" * 70)

    splits = [
        ("data/manifests/val_manifest.json", "data/processed/val_features_21class.pt"),
        ("data/manifests/test_manifest.json", "data/processed/test_features_21class.pt"),
        ("data/manifests/train_manifest.json", "data/processed/train_features_21class.pt"),
    ]

    for manifest, output in splits:
        m_path = pathlib.Path(manifest)
        o_path = pathlib.Path(output)
        if o_path.is_file():
            print(f"[SKIP] Output already exists: {o_path} ({o_path.stat().st_size / (1024*1024):.1f} MB)")
            continue

        print(f"\nProcessing {m_path.name} -> {o_path.name}...")
        t0 = time.perf_counter()
        precompute_dataset_features(
            manifest_path=str(m_path),
            output_file=str(o_path),
            num_workers=8,
        )
        print(f"Done in {time.perf_counter() - t0:.1f}s")

if __name__ == "__main__":
    main()

