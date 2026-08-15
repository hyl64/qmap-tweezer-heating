import sys, time
sys.path.insert(0, ".")
import numpy as np
import qmap.pipeline as QP

orig_hl = QP.run_lines_holograms
def hl(*a, **k):
    print("  >> run_lines_holograms start", flush=True)
    t0 = time.time()
    r = orig_hl(*a, **k)
    print(f"  >> run_lines_holograms done {time.time()-t0:.1f}s", flush=True)
    return r
QP.run_lines_holograms = hl
orig_ext = QP.extract_traps
def ext(*a, **k):
    print("  >> extract_traps start", flush=True)
    t0 = time.time()
    r = orig_ext(*a, **k)
    print(f"  >> extract_traps done {time.time()-t0:.1f}s", flush=True)
    return r
QP.extract_traps = ext
orig_ml = QP.map_line
def ml(*a, **k):
    print("  >> map_line start", flush=True)
    t0 = time.time()
    r = orig_ml(*a, **k)
    print(f"  >> map_line done {time.time()-t0:.1f}s", flush=True)
    return r
QP.map_line = ml

lines, table, sc = QP.run(seed=0, K=16, frames=14, Ngrid=256, f_khz=20.0,
                          dt_ms=1.0, px_um=0.5, iters=5, device="cuda",
                          do_soft_check=True)
print("RUN COMPLETE")
