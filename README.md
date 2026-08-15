# qmap — strict quantum mapping of phase → phonon heating

**Staircase heating law for atoms transported by discrete-frame optical tweezers**

`qmap` is the numerical companion of the paper *"Staircase heating law for atoms
transported by discrete-frame optical tweezers"* (Y. Huang).  It provides:

- **Exact quantum engines** for an atom in a moving optical tweezer:
  split-operator grid (SOFT), comoving-frame Fock, instantaneous-basis Fock,
  and analytic solutions (driven oscillator, staircase coherent sum,
  parametric resonance, sudden displacement).
- **The staircase heating law**: for frame-constant SLM trajectories the exact
  phonon number is  n̄ = (1/2a₀²) |Σ_f δx_f e^{iωt_f}|²  — a coherent sum of
  frame displacements with the accumulated inter-frame phases.  At integer
  trap periods per frame (ω·δt_f = 2πm) all phases coincide and
  n̄ = D²/2a₀², the *global heating maximum* for monotone transport.
- **Pluggable phase-coupling models** (the interface for analytic derivations):
  Null (|E|²-only), InterferencePosition (δx = −δφ/2k_eff),
  ParametricIntensity (ω(t) ripple), PhaseGradientKick.
- **A self-contained verification suite** (22 automated checks) in which the
  four engines agree at 10⁻⁷–10⁻³ on sudden displacement, driven oscillator,
  resonant drive, staircase, parametric resonance, classical limit, and
  unitarity.
- **Pipeline integration** with real P2WGS hologram sequences (included engine):
  per-frame extraction of trap trajectory / frequency / focal phase, and the
  full quantum mapping for phase-independent vs warm-started vs phase-carrying
  hologram lines (multi-seed, 8-bit SLM quantization aware).

## Installation

```bash
pip install numpy scipy matplotlib     # core engines + verification suite
pip install torch                      # P2WGS pipeline (GPU recommended)
```

## Quick start

```bash
# 1) run the verification suite (22/22 checks, ~10-40 min depending on hardware)
python run_verify.py

# 2) optional: full pipeline on real P2WGS hologram sequences (GPU, ~8 min)
python instrument_run.py

# 3) multi-seed robustness sweep and extra exact-wavefunction cross-checks
python run_multi_seed.py
python verify_extra.py
```

## Repository layout

```
qmap/                  the package
  constants.py         SI constants (⁸⁷Rb, trap defaults)
  models.py            trap potentials + pluggable PhaseCoupling models
  solver.py            SOFT grid / Fock engines / analytic benchmarks
  observables.py       phonon statistics, eigenbases, Wigner function
  verify.py            the 22-check verification suite
  pipeline.py          P2WGS hologram integration (GPU)
  figures.py           paper figures
run_verify.py          verification entry point (exit 0 = all pass)
instrument_run.py      full pipeline driver (seed 0)
run_multi_seed.py      seeds 3/7/11 robustness sweep
verify_extra.py        resonance-structure + scaled-D cross-checks (E1/E2)
p2wgs engine/          P2WGS phase-aware WGS (propagator, wgs_torch, p2wgs,
                       p2wgs_metrics, minjerk, atom_dynamics,
                       coherent_pipeline, auction_pipeline)
data/                  sample outputs (pipeline npz, verification summary)
```

## The physics in one paragraph

An atom transported by a spatial-light-modulator (SLM) tweezer does not see the
commanded smooth path: the hologram is refreshed at a finite frame rate, so the
trap position is *piecewise constant*.  The exact phonon number of that staircase
is the coherent sum above.  At the common operating point of kHz SLM updates with
10–100 kHz traps (ω·δt_f an integer multiple of 2π), all jumps add in phase and
n̄ = D²/2a₀²: the atom behaves as under a single sudden displacement by the total
transport distance — the smooth-path adiabatic floor underpredicts the heating by
fourteen orders of magnitude.  The half-integer points approach complete
cancellation: a resonance design rule with two orders of magnitude contrast,
testable by resolved sideband spectroscopy (see the paper).

Measured hologram intensity flicker de-phases the coherent sum; the de-phased
value is controlled by the flicker *phase history*, not its amplitude, so no
systematic advantage of phase-continuous hologram lines exists in the intensity
channel (verified over four seeds) — a deliberately honest negative result.  A
small systematic coherent advantage appears in the phase-gradient kick channel,
and an exact phase-to-phonon transduction law,
n̄ = (δφ/2k_eff)²/(2a₀²) per frame jump, holds for interference-based tweezer
geometries.

## Coupling-model interface (for analytic derivations)

```python
from qmap import models as M

class MyModel(M.PhaseCoupling):
    """Your analytic derivation: phase → effective trap parameters."""
    def trap_params(self, t, phi, I_rel, x0_cmd, omega_ref, mass=M.M_RB87):
        ...   # return M.TrapParams(t=t, x0=..., omega=...)
        pass
```

The verification suite then checks the model against the exact engines, the
analytic benchmark laws, and the classical correspondence.

## Reproducing the paper numbers

- `data/pipeline_s0_K16_F14_f20.0.npz` — per-line quantum mapping on a real
  rearrangement path (16 atoms, 14 frames, 256² focal grid, 20 kHz, 1 ms):
  n̄_floor ~ 1e-13, n̄_stair ~ 850, de-phased n̄_stair^φ (seed 0: 355 indep / 232
  warm), n̄_A ~ 860, n̄_C ~ 0.02–0.04, n̄_para ≤ 4e-4.
- `data/verify_summary.json` — the 22/22 verification record.

## License

MIT (see LICENSE).  © 2026 Yuliang Huang.

## Citation

```bibtex
@misc{huang2026staircase,
  title  = {Staircase heating law for atoms transported by discrete-frame
            optical tweezers},
  author = {Huang, Yuliang},
  year   = {2026},
  note   = {arXiv:XXXX.XXXXX},
}
```
