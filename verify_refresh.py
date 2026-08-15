import sys, time
sys.path.insert(0, ".")
import numpy as np
from qmap.constants import HBAR, M_RB87, a_ho
from qmap import solver as S
from qmap import models as M

omega = 2*np.pi*20e3
mass = M_RB87
a0 = a_ho(omega, mass)
w = omega

# single jump: trap at 0, ramps to dx over tau at t=tau0, holds
def traj_ramp(t, dx, tau0, tau):
    out = np.zeros_like(t)
    m = (t >= tau0) & (t < tau0 + tau)
    out[m] = dx * (t[m] - tau0) / tau
    out[t >= tau0 + tau] = dx
    return out

dx = 3.0*a0
Tend = 3.0/w
# analytic: nbar(tau) = (dx^2/2a0^2) * sinc^2(w tau / 2pi)   [sinc(x)=sin(pi x)/(pi x)]
def sinc2(x):
    x = np.where(x == 0, 1e-12, x)
    return (np.sin(np.pi*x)/(np.pi*x))**2

print("tau*w/2pi   analytic   SOFT   ratio")
for r in (0.0, 0.25, 0.5, 1.0, 1.5):
    tau = r / w
    n_an = dx**2/(2*a0**2) * sinc2(w*tau/(2*np.pi))
    # SOFT: sample the trajectory densely
    nq = 4000
    tq = np.linspace(0, Tend, nq)
    x0q = traj_ramp(tq, dx, 0.5/w, tau)
    tp = M.TrapParams(t=tq, x0=x0q, omega=np.full(nq, w))
    V = M.MovingHarmonic(tp, mass)
    x = np.linspace(-8*a0, 8*a0, 2048)
    t0 = time.time()
    psi, _ = S.soft_solve(V, x, S.harmonic_ground(x, 0.0, w, mass),
                          np.array([0.0, Tend]), 0.02/w, mass=mass)
    psif = psi[-1,0]
    # energy-moment nbar (final trap at dx, freq w)
    ddx = x[1]-x[0]
    kk = 2*np.pi*np.fft.fftfreq(len(x), d=ddx)
    dp = HBAR*kk*np.fft.fft(psif)
    Tk = np.trapezoid(np.abs(np.fft.ifft(dp))**2, x)/(2*mass)
    Vr = 0.5*mass*w**2*np.trapezoid(np.abs(psif)**2*(x-dx)**2, x)
    n_soft = (Tk+Vr)/(HBAR*w) - 0.5
    print(f"  {r:4.2f}      {n_an:8.4f}  {n_soft:8.4f}  {n_soft/n_an:6.3f}   ({time.time()-t0:.0f}s)", flush=True)
print("done")