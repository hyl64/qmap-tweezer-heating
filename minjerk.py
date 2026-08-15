"""Minimum-jerk trajectory interpolation.

s(t) = 10u³ - 15u⁴ + 6u⁵, u = t/T ∈ [0,1]
ẋ0(t) = (x_end - x_start) * s'(t) / T
ddot x0(t) = (x_end - x_start) * s''(t) / T²
"""

import numpy as np
import torch
from typing import Tuple, List

def s_minjerk(u: float) -> float:
    """s(u) = 10u³ - 15u⁴ + 6u⁵, u ∈ [0,1]"""
    return 10*u**3 - 15*u**4 + 6*u**5

def s_prime(u: float) -> float:
    """ds/du"""
    return 30*u**2 - 60*u**3 + 30*u**4

def s_double_prime(u: float) -> float:
    """d²s/du²"""
    return 60*u - 180*u**2 + 120*u**3

def interpolate_traj(
    start: np.ndarray,   # (K,2)
    end: np.ndarray,     # (K,2)
    n_frames: int,
    kind: str = "minjerk"
) -> np.ndarray:      # (n_frames, K, 2)
    """Generate position frames along trajectory."""
    K = len(start)
    frames = np.zeros((n_frames, K, 2), dtype=np.float32)

    if kind == "minjerk":
        u = np.linspace(0, 1, n_frames)
        s = s_minjerk(u)
        ds = s_prime(u)
        dds = s_double_prime(u)

        for f in range(n_frames):
            x0 = start + (end - start) * s[f]
            frames[f] = x0
    else:  # linear
        for f in range(n_frames):
            s = f / (n_frames - 1)
            frames[f] = start + (end - start) * s

    return frames
