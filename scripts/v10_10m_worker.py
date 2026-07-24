"""One 10M v10 anchored-point run (rho*=0.8) in a FRESH process (one seed per
process, per the scale10m convention - avoids cross-seed memory accumulation).
Deletes the ~2.5GB parquet dir after eval (JSON kept; data reproducible by seed).
Usage: python v10_10m_worker.py <seed>
"""
import sys, json, shutil
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8')
import t9sim.simulate as sim
from t9sim.pipeline import run_profile
from t9sim.paths import OUTPUT_DIR

seed = int(sys.argv[1])
V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
tag = f"v10_10m_s{seed % 100}"

sim.run("scale10m", out_name=tag, seed=seed, rho=0.8, bn_edges=EDGES)
out = run_profile(tag, do_shap=False, hist_clearing=True)
json.dump(out, open(f"docs/results/{tag}.json", "w"), indent=2, default=float)
k = out.get("contrasts", {})
print(f"WORKER DONE seed={seed}: clf={(k.get('ssp_C3_C1_clf') or {}).get('diff')} "
      f"aft={(k.get('ssp_C3_C1_aft') or {}).get('diff')}")
shutil.rmtree(OUTPUT_DIR / tag, ignore_errors=True)   # reclaim ~2.5GB
