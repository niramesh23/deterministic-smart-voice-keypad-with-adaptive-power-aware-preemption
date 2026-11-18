"""
src/ml/classes.py
-----------------
Vocabulary definitions and class-label mappings for Keyword Spotting (KWS).

Classes:
- 10 Core Target Commands: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
- 9 Spoken-Number Commands: "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
- 2 Special Acoustic Classes:
  - "_silence_": Background acoustic noise and ambient room silence.
  - "_unknown_": Spoken words from auxiliary categories that are not target commands.
Total: 21 classes.
"""

from typing import Dict, List, Optional, Tuple

# 10 core command words
CORE_COMMANDS: List[str] = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]

# 9 spoken number command words
NUMBER_COMMANDS: List[str] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
]

# Mapping between spoken number words and numeric digits (for display)
NUMBER_TO_DIGIT: Dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
DIGIT_TO_NUMBER: Dict[str, str] = {v: k for k, v in NUMBER_TO_DIGIT.items()}

# 19 combined target commands
TARGET_COMMANDS: List[str] = CORE_COMMANDS + NUMBER_COMMANDS

# 16 auxiliary word categories used for "_unknown_" class (non-target words)
AUXILIARY_WORDS: List[str] = [
    "backward", "bed", "bird", "cat", "dog", "follow", "forward", "happy",
    "house", "learn", "marvin", "sheila", "tree", "visual", "wow", "zero"
]

# Legacy 12-class vocabulary (preserved for backwards-compatibility checks)
CLASS_NAMES_12: List[str] = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

# Canonical 21-class vocabulary with fixed numerical index mapping
CLASS_NAMES_21: List[str] = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

# Canonical active vocabulary
CLASS_NAMES: List[str] = CLASS_NAMES_21
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)


def map_word_to_class(word: str, allowed_targets: Optional[List[str]] = None) -> str:
    """
    Maps an audio folder name to the 21-class vocabulary (or custom allowed_targets).
    - Target commands map to their respective class names.
    - Background noise maps to '_silence_'.
    - All other words map to '_unknown_'.
    """
    targets = allowed_targets if allowed_targets is not None else TARGET_COMMANDS
    word_clean = word.lower().strip()
    if word_clean in targets:
        return word_clean
    elif word_clean in ("_background_noise_", "_silence_"):
        return "_silence_"
    return "_unknown_"


def get_display_command(command: Optional[str]) -> str:
    """
    Returns user-facing display label for a command.
    Spoken numbers optionally include their digit equivalent, e.g., 'seven (7)'.
    Internal class label is never altered.
    """
    if not command:
        return "---"
    if command in NUMBER_TO_DIGIT:
        return f"{command} ({NUMBER_TO_DIGIT[command]})"
    return command
