"""IPW weight-cap sensitivity sweep (hardens the C1 IPW null against the obvious
reviewer attack: 'your correction fails only because you capped the weights').

Same CROSS-FIT method as the authoritative 10M run (win/propensity model fit on
the VALIDATION split, applied to TRAIN rows), at 1M for speed, n=10 seeds. Holds
everything fixed and sweeps ONLY the weight cap = 1/p_min over
{10, 20, 50, 100, none}. Records IPW - reference on the two metrics where the
null is significant (install AUC, EV-Spearman) + the bias metric (ev_ratio) +
weight stats. If the null is cap-robust, IPW never overtakes the uncorrected
reference on ranking at any cap.

Run from repo root:  t9_sim/venv/Scripts/python.exe t9_sim/scripts/ipw_cap_sweep.py
"""
import sys, json, time, shutil
sys.path.insert(0, 't9_sim/src'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from scipy import stats

import t9sim.simulate as sim
from t9sim import censor, schema
from t9sim.paths import OUTPUT_DIR
from t9sim.pipeline import (PipelineConfig, temporal_split, train_tier1, predict_ev,
                            train_tier2_clf, attach_hist_clearing, t2_feats, _as_cat)
from method_bench import train_tier1_ipw, eval_funnel

V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
RHO = 0.8
SEEDS = [90213, 90214, 90215, 90216, 90217, 90218, 90219, 90220, 90221, 90222]
PMINS = [0.10, 0.05, 0.02, 0.01, 0.0]          # cap = 1/p_min: 10, 20(baseline), 50, 100, none
OUT = "docs/results/ipw_cap_sweep_1m_n10.json"
KEYS = ("auc_install_all", "ev_spearman_all", "ev_ratio_all")


def run_seed(seed):
    tag = f"capsw_s{seed % 100}"
    sim.run("test", out_name=tag, seed=seed, rho=RHO, bn_edges=EDGES)
    pc = PipelineConfig()
    path = OUTPUT_DIR / tag / "auctions.parquet"
    pf_cols = set(pq.ParquetFile(path).schema_arrow.names)
    needed = [c for c in (set(schema.NEEDED) |
              {"p_click", "p_install", "p_payer", "e_ltv"}) if c in pf_cols]
    df = pd.read_parquet(path, columns=needed)
    for c in pc.t1_cat:
        if c in df.columns:
            df[c] = df[c].astype("category")
    sp = temporal_split(df).values
    tr0, va0, te0 = df[sp == "train"], df[sp == "val"], df[sp == "test"]
    del df

    cond = "C1"
    base_tr = censor.view(tr0, cond)
    va = censor.view(va0, cond)
    te = te0.copy()
    attach_hist_clearing(base_tr, va, te, censor.CONDITIONS[cond]["clearing"])
    feats2 = t2_feats(cond, True, view_cols=base_tr.columns)
    clf_bst, clf_feats = train_tier2_clf(va, va, cond, feats=feats2)      # cross-fit win model
    p = clf_bst.predict(xgb.DMatrix(_as_cat(base_tr, clf_feats),
                                    enable_categorical=True))              # p_win on TRAIN rows

    ref_row = eval_funnel("reference", predict_ev(pc, train_tier1(pc, base_tr, va, cond), te), te)
    out = {"seed": seed, "reference": {k: ref_row.get(k) for k in KEYS}, "caps": []}
    for pmin in PMINS:
        lo = pmin if pmin > 0 else 1e-6
        w = 1.0 / np.clip(p, lo, 1.0)
        cap = (1.0 / pmin) if pmin > 0 else float("inf")
        frac_capped = (float((p <= pmin + 1e-12).mean()) if pmin > 0 else 0.0)
        tr = base_tr.assign(_ipw_w=w)
        row = eval_funnel("ipw", predict_ev(pc, train_tier1_ipw(pc, tr, va, cond), te), te)
        out["caps"].append({
            "pmin": pmin, "cap": (None if pmin == 0 else round(cap, 1)),
            "auc_install_all": row.get("auc_install_all"),
            "ev_spearman_all": row.get("ev_spearman_all"),
            "ev_ratio_all": row.get("ev_ratio_all"),
            "d_auc_install": round((row.get("auc_install_all") or 0) - (ref_row.get("auc_install_all") or 0), 4),
            "d_ev_spearman": round((row.get("ev_spearman_all") or 0) - (ref_row.get("ev_spearman_all") or 0), 4),
            "w_mean": round(float(w.mean()), 3), "w_max": round(float(w.max()), 2),
            "frac_capped": round(frac_capped, 4)})
    shutil.rmtree(OUTPUT_DIR / tag, ignore_errors=True)
    return out


def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 2:
        return None
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": round(float(m), 4), "ci95": [round(float(m - h), 4), round(float(m + h), 4)],
            "n_pos": int((v > 0).sum()), "n": len(v)}


def aggregate(seeds):
    per_cap = {}
    for i, pmin in enumerate(PMINS):
        d_auc = [s["caps"][i]["d_auc_install"] for s in seeds]
        d_sp = [s["caps"][i]["d_ev_spearman"] for s in seeds]
        evr = [s["caps"][i]["ev_ratio_all"] for s in seeds]
        fc = [s["caps"][i]["frac_capped"] for s in seeds]
        wm = [s["caps"][i]["w_mean"] for s in seeds]
        per_cap[str(seeds[0]["caps"][i]["cap"])] = {
            "pmin": pmin, "cap": seeds[0]["caps"][i]["cap"],
            "ipw_minus_ref_auc_install": ci(d_auc),
            "ipw_minus_ref_ev_spearman": ci(d_sp),
            "ipw_ev_ratio_all": ci(evr),
            "frac_capped_mean": round(float(np.mean(fc)), 4),
            "w_mean_mean": round(float(np.mean(wm)), 3)}
    return per_cap


def main():
    seeds = []
    t0 = time.time()
    for k, seed in enumerate(SEEDS):
        t = time.time()
        try:
            seeds.append(run_seed(seed))
            json.dump({"scale": "1M", "rho": RHO, "method": "cross-fit (val->train)",
                       "seeds": seeds, "aggregate": aggregate(seeds)},
                      open(OUT, "w"), indent=2, default=float)
            last = seeds[-1]
            print(f"[{k+1}/{len(SEEDS)}] seed {seed} done ({(time.time()-t)/60:.1f} min) | "
                  f"cap20 dAUC={last['caps'][1]['d_auc_install']:+.4f} "
                  f"nocap dAUC={last['caps'][-1]['d_auc_install']:+.4f}", flush=True)
        except Exception as e:                       # keep partial progress on any seed failure
            print(f"[{k+1}/{len(SEEDS)}] seed {seed} FAILED: {type(e).__name__}: {e}", flush=True)
    agg = aggregate(seeds)
    print(f"\n===== IPW cap sweep, 1M, n={len(seeds)} ({(time.time()-t0)/60:.0f} min) =====")
    print(f"{'cap':>6} {'frac_capped':>11} {'w_mean':>7} {'IPW-ref install AUC':>28} {'IPW-ref EV-Spearman':>28}")
    for key, s in agg.items():
        a, sp = s["ipw_minus_ref_auc_install"], s["ipw_minus_ref_ev_spearman"]
        print(f"{str(s['cap']):>6} {s['frac_capped_mean']:>11.3f} {s['w_mean_mean']:>7.2f} "
              f"{a['mean']:>+9.4f} {str(a['ci95']):>18} ({a['n_pos']}/{a['n']})  "
              f"{sp['mean']:>+9.4f} ({sp['n_pos']}/{sp['n']})")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
