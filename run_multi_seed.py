import sys
sys.path.insert(0, ".")
import numpy as np, json
from qmap.pipeline import run

results = {}
for seed in (3, 7, 11):
    print(f"===== seed {seed} =====", flush=True)
    lines, table, sc = run(seed=seed, K=16, frames=14, Ngrid=256, f_khz=20.0,
                           dt_ms=1.0, px_um=0.5, iters=5, device="cuda",
                           do_soft_check=False,
                           out_prefix="qmap_out/pipeline")
    results[seed] = {r["line"]: dict(n_stair=r["n_stair"], n_stair_ph=r.get("n_stair_ph", float("nan")),
                                     n_A=r["n_A"], n_C=r["n_C"], n_para=r["n_para"]) for r in table}
    print(json.dumps(results[seed], indent=1), flush=True)
print("=== SUMMARY ===")
for seed, res in results.items():
    ind, wm = res["indep5"], res["warm5"]
    print(f"seed {seed}: stair {ind["n_stair"]:.0f}/{wm["n_stair"]:.0f} | "
          f"stair_ph {ind["n_stair_ph"]:.0f}/{wm["n_stair_ph"]:.0f} | ratio {wm["n_stair_ph"]/ind["n_stair_ph"]:.3f} | "
          f"C {ind["n_C"]:.4f}/{wm["n_C"]:.4f} | para {ind["n_para"]:.2e}/{wm["n_para"]:.2e}")