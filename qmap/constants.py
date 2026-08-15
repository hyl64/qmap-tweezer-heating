"""Physical constants for the quantum mapping (相位→n̄ 严格量子映射) simulator.

SI units throughout. Defaults follow the coherent-transport pipeline:
⁸⁷Rb atoms in a moving optical tweezer (atom_dynamics.py conventions).
"""
from __future__ import annotations

import numpy as np

HBAR = 1.054_571_817e-34          # J·s
M_RB87 = 1.443_160_60e-25         # kg (⁸⁷Rb)
OMEGA0_DEFAULT = 2.0 * np.pi * 80e3   # rad/s (radial trap, demo default)
PX_M_DEFAULT = 0.5e-6             # 0.5 µm/pixel (demo calibration)
LAMBDA_LIGHT = 0.80e-6            # m (trapping wavelength, demo default)

def a_ho(omega: float = OMEGA0_DEFAULT, mass: float = M_RB87) -> float:
    """Harmonic-oscillator length √(ħ/mω). ~38 nm @ 80 kHz for ⁸⁷Rb."""
    return float(np.sqrt(HBAR / (mass * omega)))

def omega_from_ho_len(a: float, mass: float = M_RB87) -> float:
    return float(HBAR / (mass * a * a))

def k_light(lam: float = LAMBDA_LIGHT) -> float:
    """Free-space wavevector 2π/λ."""
    return float(2.0 * np.pi / lam)

def omegaT_dimensionless(omega: float, T: float) -> float:
    """Adiabaticity figure of merit ωT (adiabatic ⇔ ωT ≫ 1)."""
    return omega * T
