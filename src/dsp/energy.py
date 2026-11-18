"""
src/dsp/energy.py
-----------------
Root Mean Square (RMS) energy and power calculations for audio frames.

Formula:
    RMS = sqrt( (1 / N) * sum_{n=0}^{N-1} x[n]^2 )
"""

import math
from typing import Union
import numpy as np


def calculate_rms(samples: Union[np.ndarray, list]) -> float:
    """
    Computes the Root Mean Square (RMS) energy of an audio frame.

    Parameters:
        samples: 1D NumPy array or sequence of audio samples.
                 Expected normalized float32 in [-1.0, 1.0] or int16.

    Returns:
        float: RMS amplitude >= 0.0.
    """
    if len(samples) == 0:
        return 0.0

    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.flatten()

    # If samples are int16, normalize to [-1.0, 1.0] for consistent RMS scale
    if arr.dtype == np.int16:
        float_arr = arr.astype(np.float32) / 32768.0
    else:
        float_arr = arr.astype(np.float32)

    mean_square = float(np.mean(float_arr ** 2))
    return float(math.sqrt(max(0.0, mean_square)))


def calculate_frame_energy(samples: np.ndarray) -> float:
    """
    Calculates normalized short-time mean square energy (RMS^2).
    """
    if len(samples) == 0:
        return 0.0
    arr = np.asarray(samples)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
    return float(np.mean(arr ** 2))


def rms_to_db(rms: float, ref: float = 1.0, floor_db: float = -120.0) -> float:
    """
    Converts RMS amplitude to decibels relative to full scale (dBFS).
    """
    if rms <= 0.0:
        return floor_db
    db = 20.0 * math.log10(rms / ref)
    return max(floor_db, db)

