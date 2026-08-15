import sys, time
sys.path.insert(0, ".")
import numpy as np
from qmap.constants import HBAR, M_RB87, a_ho
from qmap import solver as S
from qmap.pipeline import soft_crosscheck

omega = 2*np.pi*20e3
mass = M_RB87
a0 = a_ho(omega, mass)
w = omega
F = 12
dt = 1e-3
D = 6.0*a0
u = np.linspace(0, 1, F)
s = 10*u**3 - 15*u**4 + 6*u**5
x0 = D*s
print("E1: resonance structure vs SOFT (analytic coherent sum)")
for Np in (20.0, 20.5, 21.0):
    w_eff = 2*np.pi*Np/dt
    n_pred = S.nbar_staircase(x0, dt, w_eff, mass)
    t0 = time.time()
    sc = soft_crosscheck(x0, np.full(F, w_eff), w_eff, dt, mass=mass,
                         n_grid=2048, dt_w=0.05, scale_d=1.0)
    msg = "N_p=%s: analytic=%.3f SOFT=%.3f ratio=%.4f norm=%.6f (%.0fs)" % (
        Np, n_pred, sc["n"], sc["n"]/n_pred, sc["norm"], time.time()-t0)
    print("  " + msg, flush=True)

d = np.load("qmap_out/pipeline_s0_K16_F14_f20.0.npz")
x0r = d["carry5_x_x0_m"][:, 4]
omr = d["carry5_x_om_t"][:, 4]
print("E2: real trajectory scale 1/10 (nbar~15)")
t0 = time.time()
sc2 = soft_crosscheck(x0r, omr, omega, 1e-3, mass=mass, n_grid=4096,
                      dt_w=0.02, scale_d=1.0/10.0)
msg = "  scale1/10: SOFT=%.3f pred=%.3f ratio=%.4f norm=%.8f (%.0fs)" % (
    sc2["n"], sc2["n_pred"], sc2["n"]/sc2["n_pred"], sc2["norm"], time.time()-t0)
print(msg)
print("E DONE")