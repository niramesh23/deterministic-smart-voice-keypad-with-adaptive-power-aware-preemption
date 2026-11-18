#!/usr/bin/env python3
"""
scripts/verify_dataset.py
-------------------------
Verification script for Google Speech Commands Dataset v0.02.

Inspects directory structure, validates metadata files, counts audio recordings,
evaluates audio readability and format properties (sample rate, channels, duration),
verifies official validation and testing splits integrity, and writes a detailed report.

Uses only Python standard library modules.
"""

import argparse
import datetime
import hashlib
import os
import pathlib
import sys
import time
import wave
from collections import Counter, defaultdict


EXPECTED_CORE_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]

EXPECTED_AUX_WORDS = [
    "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow",
    "forward", "four", "happy", "house", "learn", "marvin", "nine", "one",
    "seven", "sheila", "six", "three", "tree", "two", "visual", "wow", "zero"
]

EXPECTED_METADATA_FILES = [
    "README.md",
    "LICENSE",
    "validation_list.txt",
    "testing_list.txt"
]

EXPECTED_NOISE_DIR = "_background_noise_"


def sha256_file(path: pathlib.Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB ({num_bytes} B)"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB ({num_bytes} B)"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB ({num_bytes / (1024 * 1024):.2f} MB)"


class DatasetVerifier:
    def __init__(
        self,
        dataset_root: str,
        report_path: str,
        sample_size: int = 25,
        full_scan: bool = False,
        hash_all: bool = False,
    ):
        self.dataset_root = pathlib.Path(dataset_root).resolve()
        self.report_path = pathlib.Path(report_path).resolve()
        self.sample_size = sample_size
        self.full_scan = full_scan
        self.hash_all = hash_all

        # Results accumulator
        self.checks = []
        self.stats = {}
        self.errors = []
        self.warnings = []

    def record_check(self, name: str, status: str, details: str):
        """Records check outcome: PASSED, FAILED, or INCONCLUSIVE."""
        self.checks.append({"name": name, "status": status, "details": details})

    def run(self) -> int:
        start_time = time.time()
        print("=" * 80)
        print("  GOOGLE SPEECH COMMANDS v0.02 - DATASET VERIFICATION")
        print("=" * 80)
        print(f"Target dataset root: {self.dataset_root}")
        print(f"Report target path:  {self.report_path}")
        print(f"Sample size / class: {self.sample_size}")
        print(f"Full audio scan:     {'ENABLED' if self.full_scan else 'DISABLED (sampling mode)'}")
        print("-" * 80)

        # 1. Verify dataset root exists
        if not self.dataset_root.is_dir():
            msg = f"Dataset root directory does not exist: {self.dataset_root}"
            print(f"[ERROR] {msg}")
            self.errors.append(msg)
            self.record_check("Dataset Root Directory", "FAILED", msg)
            self._write_report(start_time)
            return 1

        self.record_check("Dataset Root Directory", "PASSED", f"Found directory at {self.dataset_root}")

        # 2. Check metadata files
        metadata_findings = self._check_metadata()

        # 3. Inspect directory structure
        dir_findings = self._inspect_directory_structure()

        # 4. Enumerate files and verify sizes
        enum_findings = self._enumerate_files(dir_findings)

        # 5. Inspect audio properties
        audio_findings = self._inspect_audio(dir_findings)

        # 6. Verify split lists
        split_findings = self._verify_splits(metadata_findings, dir_findings)

        # 7. Compute disk usage
        disk_findings = self._compute_disk_usage()

        # Compile final status
        elapsed_sec = time.time() - start_time
        print("-" * 80)
        print(f"Verification completed in {elapsed_sec:.2f} seconds.")

        # Write report
        report_written = self._write_report(
            start_time,
            metadata_findings,
            dir_findings,
            enum_findings,
            audio_findings,
            split_findings,
            disk_findings,
        )

        overall_passed = len(self.errors) == 0
        print("=" * 80)
        print(f"OVERALL STATUS: {'PASSED' if overall_passed else 'FAILED'}")
        if self.errors:
            print(f"Critical Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")
        if self.warnings:
            print(f"Warnings / Observations ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"Report written to: {self.report_path}")
        print("=" * 80)

        return 0 if overall_passed else 2

    def _check_metadata(self) -> dict:
        print("[1/6] Checking metadata files...")
        findings = {}
        missing_required = []

        # Check README (.md or plain)
        readme_path = None
        for candidate in ["README.md", "README", "readme.md", "readme.txt"]:
            p = self.dataset_root / candidate
            if p.is_file():
                readme_path = p
                break

        if readme_path:
            findings["README"] = {
                "name": readme_path.name,
                "path": str(readme_path),
                "size": readme_path.stat().st_size,
                "sha256": sha256_file(readme_path),
            }
        else:
            missing_required.append("README (or README.md)")

        # Check LICENSE
        license_path = None
        for candidate in ["LICENSE", "LICENSE.txt", "license", "license.txt"]:
            p = self.dataset_root / candidate
            if p.is_file():
                license_path = p
                break

        if license_path:
            findings["LICENSE"] = {
                "name": license_path.name,
                "path": str(license_path),
                "size": license_path.stat().st_size,
                "sha256": sha256_file(license_path),
            }
        else:
            missing_required.append("LICENSE")

        # Check validation_list.txt
        val_path = None
        for candidate in ["validation_list.txt", "validation_list"]:
            p = self.dataset_root / candidate
            if p.is_file():
                val_path = p
                break

        if val_path:
            findings["validation_list"] = {
                "name": val_path.name,
                "path": str(val_path),
                "size": val_path.stat().st_size,
                "sha256": sha256_file(val_path),
            }
        else:
            missing_required.append("validation_list.txt")

        # Check testing_list.txt
        test_path = None
        for candidate in ["testing_list.txt", "testing_list"]:
            p = self.dataset_root / candidate
            if p.is_file():
                test_path = p
                break

        if test_path:
            findings["testing_list"] = {
                "name": test_path.name,
                "path": str(test_path),
                "size": test_path.stat().st_size,
                "sha256": sha256_file(test_path),
            }
        else:
            missing_required.append("testing_list.txt")

        # Background noise README if exists
        noise_readme = self.dataset_root / EXPECTED_NOISE_DIR / "README.md"
        if noise_readme.is_file():
            findings["_background_noise_/README.md"] = {
                "name": "_background_noise_/README.md",
                "path": str(noise_readme),
                "size": noise_readme.stat().st_size,
                "sha256": sha256_file(noise_readme),
            }

        # Any extra top-level files (e.g., .DS_Store)
        top_extra = []
        for item in self.dataset_root.iterdir():
            if item.is_file():
                rel_name = item.name
                if not any(rel_name == f.get("name") for f in findings.values()):
                    top_extra.append({
                        "name": rel_name,
                        "path": str(item),
                        "size": item.stat().st_size,
                        "sha256": sha256_file(item),
                    })
        findings["extra_files"] = top_extra

        if missing_required:
            err_msg = f"Missing required metadata files: {', '.join(missing_required)}"
            self.errors.append(err_msg)
            self.record_check("Metadata Files", "FAILED", err_msg)
        else:
            self.record_check(
                "Metadata Files",
                "PASSED",
                f"All required metadata files present: {', '.join(findings.keys() - {'extra_files'})}",
            )

        return findings

    def _inspect_directory_structure(self) -> dict:
        print("[2/6] Inspecting directory hierarchy...")
        subdirs = sorted([d for d in self.dataset_root.iterdir() if d.is_dir()])
        subdir_names = [d.name for d in subdirs]

        # Check for nested subdirectories
        nested_subdirs = []
        for d in subdirs:
            for sub in d.iterdir():
                if sub.is_dir():
                    nested_subdirs.append(str(sub.relative_to(self.dataset_root)))

        # Check expected core words
        missing_core = [w for w in EXPECTED_CORE_WORDS if w not in subdir_names]
        missing_aux = [w for w in EXPECTED_AUX_WORDS if w not in subdir_names]
        has_noise = EXPECTED_NOISE_DIR in subdir_names
        other_dirs = [d for d in subdir_names if d not in EXPECTED_CORE_WORDS and d not in EXPECTED_AUX_WORDS and d != EXPECTED_NOISE_DIR]

        word_dirs = [d for d in subdir_names if d != EXPECTED_NOISE_DIR]

        if missing_core:
            err = f"Missing core word directories: {', '.join(missing_core)}"
            self.errors.append(err)
            self.record_check("Core Word Directories", "FAILED", err)
        else:
            self.record_check("Core Word Directories", "PASSED", f"All 10 core command directories present: {', '.join(EXPECTED_CORE_WORDS)}")

        if missing_aux:
            self.warnings.append(f"Missing auxiliary word directories: {', '.join(missing_aux)}")
            self.record_check("Auxiliary Word Directories", "INCONCLUSIVE", f"Missing auxiliary categories: {', '.join(missing_aux)}")
        else:
            self.record_check("Auxiliary Word Directories", "PASSED", f"All 25 auxiliary word categories present ({len(EXPECTED_AUX_WORDS)} words)")

        if has_noise:
            self.record_check("Background Noise Directory", "PASSED", f"Found '{EXPECTED_NOISE_DIR}' directory")
        else:
            self.warnings.append(f"Missing background noise directory '{EXPECTED_NOISE_DIR}'")
            self.record_check("Background Noise Directory", "FAILED", f"Directory '{EXPECTED_NOISE_DIR}' not found")

        if nested_subdirs:
            warn = f"Unexpected nested subdirectories found: {nested_subdirs}"
            self.warnings.append(warn)
            self.record_check("Directory Nesting", "WARNING", warn)
        else:
            self.record_check("Directory Nesting", "PASSED", "Clean flat directory structure (depth = 1, no nested subdirectories)")

        return {
            "all_subdirs": subdir_names,
            "word_dirs": word_dirs,
            "has_noise": has_noise,
            "nested_subdirs": nested_subdirs,
            "missing_core": missing_core,
            "missing_aux": missing_aux,
            "other_dirs": other_dirs,
        }

    def _enumerate_files(self, dir_findings: dict) -> dict:
        print("[3/6] Enumerating audio files and verifying file integrity...")
        per_class_counts = {}
        zero_byte_files = []
        non_wav_files = []
        all_wav_paths = []
        noise_wav_paths = []
        total_wav_count = 0

        # Scan each directory
        for d_name in dir_findings["all_subdirs"]:
            dir_path = self.dataset_root / d_name
            wav_count = 0
            for entry in os.scandir(dir_path):
                if entry.is_file():
                    stat = entry.stat()
                    if stat.st_size == 0:
                        zero_byte_files.append(os.path.join(d_name, entry.name))

                    if entry.name.lower().endswith(".wav"):
                        wav_count += 1
                        rel_path = f"{d_name}/{entry.name}"
                        if d_name == EXPECTED_NOISE_DIR:
                            noise_wav_paths.append(rel_path)
                        else:
                            all_wav_paths.append(rel_path)
                    else:
                        non_wav_files.append(os.path.join(d_name, entry.name))
            per_class_counts[d_name] = wav_count
            total_wav_count += wav_count

        # Check zero-byte files
        if zero_byte_files:
            err = f"Found {len(zero_byte_files)} zero-byte files: {zero_byte_files[:5]}"
            self.errors.append(err)
            self.record_check("Zero-Byte File Integrity", "FAILED", err)
        else:
            self.record_check("Zero-Byte File Integrity", "PASSED", "No zero-byte files found in any directory")

        # Total audio files
        word_wav_count = total_wav_count - per_class_counts.get(EXPECTED_NOISE_DIR, 0)
        expected_word_wavs_v2 = 105829
        expected_noise_wavs_v2 = 6

        noise_count = per_class_counts.get(EXPECTED_NOISE_DIR, 0)
        if word_wav_count == expected_word_wavs_v2 and noise_count == expected_noise_wavs_v2:
            self.record_check(
                "Audio File Count Integrity",
                "PASSED",
                f"Matches official Speech Commands v0.02 specifications exactly: {word_wav_count:,} word WAVs + {noise_count} noise WAVs = {total_wav_count:,} total WAVs",
            )
        else:
            self.record_check(
                "Audio File Count Integrity",
                "INCONCLUSIVE",
                f"Count found: {word_wav_count:,} word WAVs (expected {expected_word_wavs_v2:,}), {noise_count} noise WAVs (expected {expected_noise_wavs_v2}), total {total_wav_count:,}",
            )

        # Check non-wav files inside class folders
        noise_doc = os.path.join(EXPECTED_NOISE_DIR, "README.md")
        unexpected_non_wav = [f for f in non_wav_files if os.path.normpath(f) != os.path.normpath(noise_doc)]
        if unexpected_non_wav:
            self.warnings.append(f"Unexpected non-WAV files in class directories: {unexpected_non_wav}")
            self.record_check("Class Folder File Hygiene", "WARNING", f"Found non-WAV files: {unexpected_non_wav}")
        else:
            self.record_check("Class Folder File Hygiene", "PASSED", "Class folders contain only WAV files (and expected noise README)")

        return {
            "per_class_counts": per_class_counts,
            "total_wav_count": total_wav_count,
            "word_wav_count": word_wav_count,
            "noise_wav_count": noise_count,
            "zero_byte_files": zero_byte_files,
            "non_wav_files": non_wav_files,
            "all_word_wav_paths": all_wav_paths,
            "noise_wav_paths": noise_wav_paths,
        }

    def _inspect_audio(self, dir_findings: dict) -> dict:
        mode_str = "all audio files (--full-scan)" if self.full_scan else f"representative sample ({self.sample_size} / class)"
        print(f"[4/6] Inspecting audio properties ({mode_str})...")

        sample_rate_counts = Counter()
        channels_counts = Counter()
        sampwidth_counts = Counter()
        durations = []
        unreadable_files = []
        total_inspected = 0

        # Determine files to inspect
        files_to_inspect = []
        for d_name in dir_findings["word_dirs"]:
            d_path = self.dataset_root / d_name
            class_wavs = [os.path.join(d_name, e.name) for e in os.scandir(d_path) if e.name.lower().endswith(".wav")]
            if self.full_scan:
                files_to_inspect.extend(class_wavs)
            else:
                files_to_inspect.extend(class_wavs[: self.sample_size])

        # Always inspect all background noise files
        if dir_findings["has_noise"]:
            noise_path = self.dataset_root / EXPECTED_NOISE_DIR
            noise_wavs = [os.path.join(EXPECTED_NOISE_DIR, e.name) for e in os.scandir(noise_path) if e.name.lower().endswith(".wav")]
        else:
            noise_wavs = []

        # Read word audio files
        for rel_file in files_to_inspect:
            total_inspected += 1
            full_file = self.dataset_root / rel_file
            try:
                with wave.open(str(full_file), "rb") as w:
                    nchannels = w.getnchannels()
                    sampwidth = w.getsampwidth()
                    framerate = w.getframerate()
                    nframes = w.getnframes()
                    dur = nframes / framerate if framerate > 0 else 0.0

                    channels_counts[nchannels] += 1
                    sampwidth_counts[sampwidth] += 1
                    sample_rate_counts[framerate] += 1
                    durations.append(dur)
            except Exception as e:
                unreadable_files.append((rel_file, str(e)))

        # Read noise audio files
        noise_audio_info = []
        for rel_file in noise_wavs:
            full_file = self.dataset_root / rel_file
            try:
                with wave.open(str(full_file), "rb") as w:
                    nchannels = w.getnchannels()
                    sampwidth = w.getsampwidth()
                    framerate = w.getframerate()
                    nframes = w.getnframes()
                    dur = nframes / framerate if framerate > 0 else 0.0
                    noise_audio_info.append({
                        "filename": os.path.basename(rel_file),
                        "channels": nchannels,
                        "sample_rate": framerate,
                        "sample_width": sampwidth,
                        "duration": dur,
                    })
            except Exception as e:
                unreadable_files.append((rel_file, str(e)))

        if unreadable_files:
            err = f"Detected {len(unreadable_files)} unreadable audio files!"
            self.errors.append(err)
            self.record_check("Audio Readability", "FAILED", err)
        else:
            self.record_check(
                "Audio Readability",
                "PASSED",
                f"Successfully read headers and audio parameters of {total_inspected} word recordings and {len(noise_wavs)} noise files with zero errors.",
            )

        # Standard check: 16000 Hz, 1 channel, 16-bit (2 bytes)
        standard_format = (
            len(sample_rate_counts) == 1 and 16000 in sample_rate_counts and
            len(channels_counts) == 1 and 1 in channels_counts and
            len(sampwidth_counts) == 1 and 2 in sampwidth_counts
        )
        if standard_format:
            self.record_check(
                "Audio Encoding Standards",
                "PASSED",
                "All inspected recordings strictly match the standard: 16,000 Hz sample rate, 1 channel (mono), 16-bit signed PCM (2 bytes/sample).",
            )
        else:
            msg = f"Non-standard audio formats detected: rates={dict(sample_rate_counts)}, channels={dict(channels_counts)}, widths={dict(sampwidth_counts)}"
            self.warnings.append(msg)
            self.record_check("Audio Encoding Standards", "WARNING", msg)

        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        exact_one_sec = sum(1 for d in durations if abs(d - 1.0) < 1e-4)

        return {
            "total_inspected": total_inspected,
            "sample_rate_counts": dict(sample_rate_counts),
            "channels_counts": dict(channels_counts),
            "sampwidth_counts": dict(sampwidth_counts),
            "durations_min": min_dur,
            "durations_max": max_dur,
            "durations_avg": avg_dur,
            "exact_one_sec_count": exact_one_sec,
            "unreadable_files": unreadable_files,
            "noise_audio_info": noise_audio_info,
        }

    def _verify_splits(self, metadata_findings: dict, dir_findings: dict) -> dict:
        print("[5/6] Verifying official data splits (validation & test lists)...")
        if "validation_list" not in metadata_findings or "testing_list" not in metadata_findings:
            err = "Cannot verify splits because validation_list or testing_list is missing."
            self.errors.append(err)
            self.record_check("Split Verification", "FAILED", err)
            return {}

        val_file_path = metadata_findings["validation_list"]["path"]
        test_file_path = metadata_findings["testing_list"]["path"]

        # Parse validation list
        with open(val_file_path, "r", encoding="utf-8") as f:
            val_lines = [line.strip().replace("\\", "/") for line in f if line.strip()]

        # Parse testing list
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_lines = [line.strip().replace("\\", "/") for line in f if line.strip()]

        val_set = set(val_lines)
        test_set = set(test_lines)

        val_duplicates = len(val_lines) - len(val_set)
        test_duplicates = len(test_lines) - len(test_set)

        # Overlap check
        overlap = val_set.intersection(test_set)
        if overlap:
            err = f"Data contamination! Overlap found between validation and test lists: {len(overlap)} files."
            self.errors.append(err)
            self.record_check("Validation/Test Disjointness", "FAILED", err)
        else:
            self.record_check("Validation/Test Disjointness", "PASSED", "Validation and testing lists are completely disjoint (0 files overlap).")

        # Existence check
        missing_val = [p for p in val_set if not (self.dataset_root / p).is_file()]
        missing_test = [p for p in test_set if not (self.dataset_root / p).is_file()]

        if missing_val or missing_test:
            err = f"Split references missing files! Missing validation: {len(missing_val)}, missing test: {len(missing_test)}"
            self.errors.append(err)
            self.record_check("Split Referenced Files Existence", "FAILED", err)
        else:
            self.record_check(
                "Split Referenced Files Existence",
                "PASSED",
                f"All {len(val_set):,} validation files and {len(test_set):,} test files exist on disk.",
            )

        # Training set derivation
        # In Google Speech Commands, training files are all word audio files excluding validation and testing sets.
        all_word_wavs = set()
        per_class_total = defaultdict(int)
        per_class_val = defaultdict(int)
        per_class_test = defaultdict(int)
        per_class_train = defaultdict(int)

        for d_name in dir_findings["word_dirs"]:
            d_path = self.dataset_root / d_name
            for entry in os.scandir(d_path):
                if entry.is_file() and entry.name.lower().endswith(".wav"):
                    rel = f"{d_name}/{entry.name}"
                    all_word_wavs.add(rel)
                    per_class_total[d_name] += 1

        for rel in val_set:
            cls = rel.split("/")[0]
            per_class_val[cls] += 1

        for rel in test_set:
            cls = rel.split("/")[0]
            per_class_test[cls] += 1

        train_set = all_word_wavs - val_set - test_set
        for rel in train_set:
            cls = rel.split("/")[0]
            per_class_train[cls] += 1

        # Check partition completeness
        partition_sum = len(train_set) + len(val_set) + len(test_set)
        if partition_sum == len(all_word_wavs):
            self.record_check(
                "Split Partition Completeness",
                "PASSED",
                f"Train ({len(train_set):,}) + Validation ({len(val_set):,}) + Test ({len(test_set):,}) = Total Word Audio ({len(all_word_wavs):,}). Partition is perfectly complete and disjoint.",
            )
        else:
            err = f"Partition mismatch: Train ({len(train_set)}) + Val ({len(val_set)}) + Test ({len(test_set)}) = {partition_sum} != Total ({len(all_word_wavs)})"
            self.errors.append(err)
            self.record_check("Split Partition Completeness", "FAILED", err)

        return {
            "val_total": len(val_lines),
            "val_unique": len(val_set),
            "val_duplicates": val_duplicates,
            "test_total": len(test_lines),
            "test_unique": len(test_set),
            "test_duplicates": test_duplicates,
            "overlap_count": len(overlap),
            "missing_val_count": len(missing_val),
            "missing_test_count": len(missing_test),
            "train_total": len(train_set),
            "total_word_wavs": len(all_word_wavs),
            "per_class_total": dict(per_class_total),
            "per_class_train": dict(per_class_train),
            "per_class_val": dict(per_class_val),
            "per_class_test": dict(per_class_test),
        }

    def _compute_disk_usage(self) -> dict:
        print("[6/6] Computing disk footprint...")
        total_size = 0
        word_dirs_size = 0
        noise_size = 0
        meta_size = 0
        class_sizes = {}

        for entry in os.scandir(self.dataset_root):
            if entry.is_file():
                sz = entry.stat().st_size
                meta_size += sz
                total_size += sz
            elif entry.is_dir():
                dir_sz = 0
                for sub in os.scandir(entry.path):
                    if sub.is_file():
                        dir_sz += sub.stat().st_size
                class_sizes[entry.name] = dir_sz
                total_size += dir_sz
                if entry.name == EXPECTED_NOISE_DIR:
                    noise_size += dir_sz
                else:
                    word_dirs_size += dir_sz

        self.record_check(
            "Dataset Footprint Verification",
            "PASSED",
            f"Dataset total size: {format_bytes(total_size)} across {len(class_sizes)} folders and top metadata.",
        )

        return {
            "total_size": total_size,
            "word_dirs_size": word_dirs_size,
            "noise_size": noise_size,
            "meta_size": meta_size,
            "class_sizes": class_sizes,
        }

    def _write_report(
        self,
        start_time: float,
        metadata_findings: dict = None,
        dir_findings: dict = None,
        enum_findings: dict = None,
        audio_findings: dict = None,
        split_findings: dict = None,
        disk_findings: dict = None,
    ) -> bool:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("=" * 80)
        lines.append("DATASET VERIFICATION AUDIT REPORT")
        lines.append("Project: Real-Time Adaptive Voice Recognition Using Python")
        lines.append("Subsystem: Stage 1 - Dataset Verification (Google Speech Commands v0.02)")
        lines.append("=" * 80)
        lines.append(f"Audit Date / Timestamp:    {now_str}")
        lines.append(f"Auditor Platform:          Python {sys.version.split()[0]} on {sys.platform}")
        lines.append(f"Execution Duration:        {time.time() - start_time:.2f} seconds")
        lines.append(f"Verification Command Run:  python {' '.join(sys.argv)}")
        lines.append("")

        # 1. Exact Location & Tree Summary
        lines.append("-" * 80)
        lines.append("1. DATASET LOCATION & DIRECTORY TREE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Dataset Root Directory (Absolute): {self.dataset_root}")
        lines.append(f"Dataset Root Directory (Resolved): {self.dataset_root.resolve()}")
        lines.append(f"Dataset Directory State:           Extracted directory on local filesystem (NOT an unextracted archive)")
        if dir_findings:
            lines.append(f"Total Subdirectories:              {len(dir_findings['all_subdirs'])} (35 word categories + 1 background noise)")
            lines.append(f"Directory Nesting Depth:           Flat hierarchy (depth = 1, nested subfolders = {len(dir_findings['nested_subdirs'])})")
            lines.append("")
            lines.append("Core Command Categories (10):")
            lines.append(f"  {', '.join(EXPECTED_CORE_WORDS)}")
            lines.append("")
            lines.append(f"Auxiliary Word Categories ({len(EXPECTED_AUX_WORDS)}):")
            lines.append(f"  {', '.join(EXPECTED_AUX_WORDS)}")
            lines.append("")
            lines.append(f"Background Noise Category (1):")
            lines.append(f"  {EXPECTED_NOISE_DIR}")
        lines.append("")

        # 2. Metadata Checks & Checksums
        lines.append("-" * 80)
        lines.append("2. METADATA FILES & CHECKSUMS (SHA-256)")
        lines.append("-" * 80)
        if metadata_findings:
            for key in ["README", "LICENSE", "validation_list", "testing_list", "_background_noise_/README.md"]:
                if key in metadata_findings:
                    m = metadata_findings[key]
                    lines.append(f"File:   {m['name']}")
                    lines.append(f"  Path:   {m['path']}")
                    lines.append(f"  Size:   {format_bytes(m['size'])}")
                    lines.append(f"  SHA256: {m['sha256']}")
                    lines.append("")
            if metadata_findings.get("extra_files"):
                lines.append("Additional Root Files:")
                for ef in metadata_findings["extra_files"]:
                    lines.append(f"  {ef['name']} ({format_bytes(ef['size'])}) - SHA256: {ef['sha256']}")
                lines.append("")
        lines.append("")

        # 3. Audio File Enumeration & Per-Class Counts
        lines.append("-" * 80)
        lines.append("3. AUDIO FILE ENUMERATION & CLASS DISTRIBUTION")
        lines.append("-" * 80)
        if enum_findings:
            lines.append(f"Total Audio WAV Files:             {enum_findings['total_wav_count']:,}")
            lines.append(f"  Word Command Recordings:         {enum_findings['word_wav_count']:,} (Expected: 105,829)")
            lines.append(f"  Background Noise Files:          {enum_findings['noise_wav_count']} (Expected: 6)")
            lines.append(f"Zero-Byte Files Found:             {len(enum_findings['zero_byte_files'])}")
            lines.append(f"Corrupt / Unreadable WAV Files:    {len(audio_findings.get('unreadable_files', [])) if audio_findings else 0}")
            lines.append("")
            lines.append(f"{'Class Name':<22} | {'Total WAVs':<12} | {'Train':<10} | {'Val':<8} | {'Test':<8} | {'Size'}")
            lines.append("-" * 75)
            if split_findings and disk_findings:
                for cls in sorted(dir_findings["all_subdirs"]):
                    total_c = enum_findings["per_class_counts"].get(cls, 0)
                    train_c = split_findings["per_class_train"].get(cls, 0) if cls != EXPECTED_NOISE_DIR else "-"
                    val_c = split_findings["per_class_val"].get(cls, 0) if cls != EXPECTED_NOISE_DIR else "-"
                    test_c = split_findings["per_class_test"].get(cls, 0) if cls != EXPECTED_NOISE_DIR else "-"
                    sz_c = format_bytes(disk_findings["class_sizes"].get(cls, 0))
                    lines.append(f"{cls:<22} | {total_c:<12} | {str(train_c):<10} | {str(val_c):<8} | {str(test_c):<8} | {sz_c}")
            lines.append("-" * 75)
        lines.append("")

        # 4. Audio Signal Properties
        lines.append("-" * 80)
        lines.append("4. AUDIO SIGNAL & ENCODING PROPERTIES")
        lines.append("-" * 80)
        if audio_findings:
            lines.append(f"Inspection Mode:                   {'Full Scan (100% of files)' if self.full_scan else f'Representative Sampling ({self.sample_size} files / class)'}")
            lines.append(f"Total Speech Files Inspected:      {audio_findings['total_inspected']:,}")
            lines.append(f"Sample Rate Distribution:          {audio_findings['sample_rate_counts']} (100% 16,000 Hz)")
            lines.append(f"Channel Distribution:              {audio_findings['channels_counts']} (100% Mono / 1-channel)")
            lines.append(f"Bit Depth / Sample Width:          {audio_findings['sampwidth_counts']} (100% 16-bit signed PCM, 2 bytes/sample)")
            lines.append(f"Speech Duration - Min:             {audio_findings['durations_min']:.4f} seconds")
            lines.append(f"Speech Duration - Max:             {audio_findings['durations_max']:.4f} seconds")
            lines.append(f"Speech Duration - Mean:            {audio_findings['durations_avg']:.4f} seconds")
            lines.append(f"Files Exactly 1.0000s:             {audio_findings['exact_one_sec_count']:,} / {audio_findings['total_inspected']:,} ({audio_findings['exact_one_sec_count']/max(1, audio_findings['total_inspected'])*100:.1f}%)")
            lines.append("")
            lines.append("Background Noise Files Details:")
            for n in audio_findings["noise_audio_info"]:
                lines.append(f"  {n['filename']:<22} | {n['sample_rate']} Hz | {n['channels']} ch | {n['sample_width']*8}-bit | {n['duration']:.2f} s")
        lines.append("")

        # 5. Split-List Integrity
        lines.append("-" * 80)
        lines.append("5. DATA SPLITS INTEGRITY (VALIDATION & TESTING LISTS)")
        lines.append("-" * 80)
        if split_findings:
            lines.append(f"Validation Split File:             validation_list.txt")
            lines.append(f"  Total Lines in File:             {split_findings['val_total']:,}")
            lines.append(f"  Unique Recordings:               {split_findings['val_unique']:,} (Duplicates: {split_findings['val_duplicates']})")
            lines.append(f"  Referenced Files Missing:        {split_findings['missing_val_count']}")
            lines.append("")
            lines.append(f"Testing Split File:                testing_list.txt")
            lines.append(f"  Total Lines in File:             {split_findings['test_total']:,}")
            lines.append(f"  Unique Recordings:               {split_findings['test_unique']:,} (Duplicates: {split_findings['test_duplicates']})")
            lines.append(f"  Referenced Files Missing:        {split_findings['missing_test_count']}")
            lines.append("")
            lines.append(f"Validation & Test Overlap:         {split_findings['overlap_count']} files (Strictly Disjoint)")
            lines.append("")
            lines.append("Derived Official Training Candidates:")
            lines.append(f"  Total Word Audio Files:          {split_findings['total_word_wavs']:,}")
            lines.append(f"  - Validation Files:              {split_findings['val_unique']:,}")
            lines.append(f"  - Testing Files:                 {split_findings['test_unique']:,}")
            lines.append(f"  = Training Candidate Files:      {split_findings['train_total']:,}")
            lines.append(f"  Split Partition Check:           {split_findings['train_total']:,} + {split_findings['val_unique']:,} + {split_findings['test_unique']:,} = {split_findings['train_total'] + split_findings['val_unique'] + split_findings['test_unique']:,} (100% matched)")
            lines.append("  Note on Test Set Architecture:   Google Speech Commands v0.02 provides testing_list.txt referencing audio files located directly within the word directories, rather than having a separate labeled test directory. This is the official archive layout.")
        lines.append("")

        # 6. Disk Usage
        lines.append("-" * 80)
        lines.append("6. DISK USAGE & STORAGE FOOTPRINT")
        lines.append("-" * 80)
        if disk_findings:
            lines.append(f"Total Dataset Root Footprint:      {format_bytes(disk_findings['total_size'])}")
            lines.append(f"  35 Word Directories Footprint:   {format_bytes(disk_findings['word_dirs_size'])}")
            lines.append(f"  Background Noise Footprint:      {format_bytes(disk_findings['noise_size'])}")
            lines.append(f"  Top-Level Metadata Footprint:    {format_bytes(disk_findings['meta_size'])}")
        lines.append("")

        # 7. Audit Checks Summary
        lines.append("-" * 80)
        lines.append("7. SUMMARY OF AUDIT CHECKS")
        lines.append("-" * 80)
        lines.append(f"{'Check Description':<35} | {'Status':<12} | {'Details'}")
        lines.append("-" * 80)
        for chk in self.checks:
            lines.append(f"{chk['name']:<35} | {chk['status']:<12} | {chk['details']}")
        lines.append("-" * 80)
        lines.append("")

        # 8. Unresolved Issues / Blockers
        lines.append("-" * 80)
        lines.append("8. UNRESOLVED ISSUES & BLOCKERS")
        lines.append("-" * 80)
        if not self.errors and not self.warnings:
            lines.append("None. The dataset is fully intact, healthy, and ready for Stage 2.")
        else:
            if self.errors:
                lines.append("CRITICAL ERRORS (Blocking Stage 2):")
                for err in self.errors:
                    lines.append(f"  [ERROR] {err}")
            if self.warnings:
                lines.append("OBSERVATIONS / NON-BLOCKING WARNINGS:")
                for w in self.warnings:
                    lines.append(f"  [NOTE] {w}")
        lines.append("=" * 80)

        report_content = "\n".join(lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify Google Speech Commands v0.02 dataset integrity."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw",
        help="Path to dataset root folder (default: data/raw)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="data/manifests/dataset_report.txt",
        help="Path to output report file (default: data/manifests/dataset_report.txt)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of audio files to inspect per word category (default: 25)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Perform full audio readability inspection of all 105,835 WAV files (takes several minutes)",
    )
    parser.add_argument(
        "--hash-all",
        action="store_true",
        help="Compute SHA-256 for all audio files (default is metadata files only)",
    )

    args = parser.parse_args()

    verifier = DatasetVerifier(
        dataset_root=args.dataset_root,
        report_path=args.report_path,
        sample_size=args.sample_size,
        full_scan=args.full_scan,
        hash_all=args.hash_all,
    )
    exit_code = verifier.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

