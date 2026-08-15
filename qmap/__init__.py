"""qmap — 相位→n̄ 严格量子映射: simulator + numerical verification.

Modules
-------
constants   physical constants & conversions
models      trap potentials + pluggable phase→motion coupling models
solver      exact quantum engines: SOFT grid, moving-frame Fock, analytic
observables phonon statistics, eigenbases, Wigner functions
verify      the numerical verification suite (run_verify.py entry)
pipeline    P2WGS hologram integration (real focal fields → n̄ per line)
figures     paper figures

The physics-derivation interface (what a collaborator must supply) is
documented in docs/COUPLING-INTERFACE.md; models.PhaseCoupling is the hook.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .constants import (HBAR, M_RB87, OMEGA0_DEFAULT, PX_M_DEFAULT,
                        LAMBDA_LIGHT, a_ho, k_light)
from . import constants, models, solver, observables  # noqa: F401
