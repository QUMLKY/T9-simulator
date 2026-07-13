"""10M anchored-point headline run: rho*=0.8, n=5 seeds, ONE FRESH PROCESS PER
SEED (subprocess; scale10m convention). Aggregates per-seed JSONs at the end.
"""
import sys, json, subprocess, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

PY = sys.executable
WORKER = str(Path(__file__).parent / "v10_10m_worker.py")
SEEDS = [90213, 90214, 90215, 90216, 90217]

for seed in SEEDS:
    t = time.time()
    print(f"=== 10M seed {seed} starting ===", flush=True)
    rc = subprocess.run([PY, WORKER, str(seed)]).returncode
    print(f"=== 10M seed {seed} rc={rc} ({(time.time()-t)/60:.0f} min) ===", flush=True)

# aggregate
rows = []
for seed in SEEDS:
    p = Path(f"docs/results/v10_10m_s{seed % 100}.json")
    if not p.exists():
        print(f"MISSING {p}"); continue
    out = json.load(open(p))
    C, k = out["conditions"], out.get("contrasts", {})
    g = lambda c, key: C[c].get(key)
    gb = lambda c, key: (C[c].get("clf_bidder") or {}).get(key)
    rows.append({"seed": seed,
                 "auc_clf": (k.get("ssp_C3_C1_clf") or {}).get("diff"),
                 "auc_aft": (k.get("ssp_C3_C1_aft") or {}).get("diff"),
                 "rmse_all_gain": 100 * (1 - g("C3", "price_rmse_all") / g("C1", "price_rmse_all")),
                 "crps_all_gain": 100 * (1 - g("C3", "price_crps_all") / g("C1", "price_crps_all")),
                 "surplus_up_clfb": (100 * (gb("C3", "surplus_per_won") / gb("C1", "surplus_per_won") - 1)
                                     if gb("C1", "surplus_per_won") else None),
                 "profit_up_clfb": (100 * (gb("C3", "profit") / gb("C1", "profit") - 1)
                                    if gb("C1", "profit") else None)})
def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2: return None
    from scipy import stats
    m, se = v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v)-1)*se
    return {"mean": round(float(m),4), "ci95": [round(float(m-h),4), round(float(m+h),4)],
            "n_pos": int((v>0).sum()), "n": len(v)}
summary = {kk: ci([r.get(kk) for r in rows])
           for kk in ("auc_clf","auc_aft","rmse_all_gain","crps_all_gain",
                      "surplus_up_clfb","profit_up_clfb")}
print("\n===== 10M rho*=0.8 n=5 summary ====="); print(json.dumps(summary, indent=1))
json.dump({"rows": rows, "summary": summary},
          open("docs/results/v10_10m_n5.json", "w"), indent=2, default=float)
print("Saved docs/results/v10_10m_n5.json")
