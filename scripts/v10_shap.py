"""V10 SHAP attribution — BOTH tiers, the full attribution analysis on the
operative V10 schema (rival_pool, rho*=0.8). Supersedes the 19-Jun v7-BN SHAP.

Tier-1 (funnel / MMP 'why'): mean|SHAP| per head (click/install/payer/spend),
  C1 vs C2 -> where MMP moves the drawn signal (expect app_id/campaign_id rising
  in the deep funnel = the de-biasing signature).
Tier-2 (win model / SSP 'why'): mean|SHAP| on the won-classifier, C1 vs C3 ->
  which features carry C3's edge (expect the SSP-only market features
  hist_clearing_ssp + bid_density/H9 appearing only under C3).

Reuses the pipeline train path (censor.view -> train_tier1 / attach_hist_clearing
/ t2_feats / train_tier2_clf) so the SHAP'd models ARE the evaluated models.
Usage: python v10_shap.py <profile>   (test=1M, scale10m=10M)  [seed]
                          [--mode interventional] [--save-phi]
Writes docs/results/v10_shap_<profile>.json for the default run
(seed 90217, path-dependent); non-default seed appends _s<NN>, interventional
mode appends _interv (so the tracked default JSONs are never overwritten).
--save-phi stores the raw per-row SHAP matrices for the two decisive heads
(tier-1 C2 payer, tier-2 C3 win) to docs/results/<tag>_phi.npz for
beeswarm / dependence plots.
"""
import sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, shap
import t9sim.simulate as sim
from t9sim import censor, schema
from t9sim.pipeline import (PipelineConfig, train_tier1, train_tier2_clf,
                            attach_hist_clearing, t2_feats, _as_cat, temporal_split)
from t9sim.paths import OUTPUT_DIR

args = [a for a in sys.argv[1:] if not a.startswith("--")]
profile = args[0] if args else "test"
seed = int(args[1]) if len(args) > 1 else 90217
MODE = "interventional" if "--mode" in sys.argv and "interventional" in sys.argv else "path"
SAVE_PHI = "--save-phi" in sys.argv
V7 = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7, "rival_pool": True}
tag = f"v10_shap_{profile}"
if seed != 90217:
    tag += f"_s{seed % 100}"
if MODE == "interventional":
    tag += "_interv"
if SAVE_PHI:
    tag += "_phirun"    # never overwrite the tracked default JSONs

sim.run(profile, out_name=tag, seed=seed, rho=0.8, bn_edges=EDGES)
df = pd.read_parquet(OUTPUT_DIR / tag / "auctions.parquet")
pc = PipelineConfig()
for c in pc.t1_cat:
    if c in df.columns:
        df[c] = df[c].astype("category")
split = temporal_split(df)
tr_m, va_m, te_m = split == "train", split == "val", split == "test"
base_tr, base_va, base_te = df[tr_m], df[va_m], df[te_m]

PHI_STORE = {}      # populated when SAVE_PHI (raw sv + X sample per stored head)


def _shap_matrix(model, Xs, feats, dm):
    """Per-row SHAP matrix under the selected perturbation mode.

    path (default):  library default tree_path_dependent (tier-1 via
      TreeExplainer.shap_values, tier-2 via xgboost pred_contribs - both
      path-dependent TreeSHAP, mode held FIXED across conditions/heads).
    interventional:  TreeExplainer(model, data=background, feature_perturbation=
      "interventional") with a 200-row background from the same test sample,
      for BOTH tiers (the DeepRead robustness cross-check)."""
    Xc = _as_cat(Xs, feats)
    if MODE == "interventional":
        bg = Xc.sample(min(200, len(Xc)), random_state=1)
        expl = shap.TreeExplainer(model, data=bg,
                                  feature_perturbation="interventional")
        return np.asarray(expl.shap_values(Xc, check_additivity=False))
    if dm:
        sv = model.predict(xgb.DMatrix(Xc, enable_categorical=True),
                           pred_contribs=True)
        return sv[:, :-1]   # drop bias col
    return np.asarray(shap.TreeExplainer(model).shap_values(Xc))


def meanshap(model, X, feats, k=8, dm=False, store=None):
    Xs = X.sample(min(4000, len(X)), random_state=0)
    sv = _shap_matrix(model, Xs, feats, dm)
    if SAVE_PHI and store:
        PHI_STORE[store] = {"phi": sv.astype(np.float32),
                            "X": _as_cat(Xs, feats), "feats": list(feats)}
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    return {feats[i]: round(float(imp[i]), 5) for i in order}

import xgboost as xgb
out = {"profile": profile, "seed": seed, "schema": "v10 (rival_pool, rho=0.8)",
       "mode": ("tree_path_dependent" if MODE == "path" else "interventional"),
       "tier1": {}, "tier2_clf": {}}

# ---- Tier-1: 4 heads, C1 vs C2 (MMP why) ----
HEADS = ["click", "install", "payer", "spend"]
for cond in ("C1", "C2"):
    tr = censor.view(base_tr, cond); va = censor.view(base_va, cond)
    models = train_tier1(pc, tr, va, cond)
    out["tier1"][cond] = {h: meanshap(models[h], base_te, pc.t1_feats,
                                      store=(f"t1_{cond}_{h}" if cond == "C2" and h in ("payer", "spend") else None))
                          for h in HEADS}
    print(f"tier1 {cond} done", flush=True)

# ---- Tier-2: won-classifier, C1 vs C3 (SSP why) ----
for cond in ("C1", "C3"):
    tr = censor.view(base_tr, cond); va = censor.view(base_va, cond); te = base_te.copy()
    attach_hist_clearing(tr, va, te, censor.CONDITIONS[cond]["clearing"])
    feats = t2_feats(cond, True, view_cols=tr.columns)
    bst, cfeats = train_tier2_clf(tr, va, cond, feats=feats)
    out["tier2_clf"][cond] = meanshap(bst, te, cfeats, dm=True,
                                      store=(f"t2_{cond}_win" if cond == "C3" else None))
    print(f"tier2 {cond} done ({len(cfeats)} feats: {cfeats[-4:]})", flush=True)

json.dump(out, open(f"docs/results/{tag}.json", "w"), indent=2, default=float)
if SAVE_PHI and PHI_STORE:
    npz = {}
    for key, d in PHI_STORE.items():
        npz[f"{key}__phi"] = d["phi"]
        npz[f"{key}__feats"] = np.array(d["feats"])
        for f in d["feats"]:                      # store columns numerically
            col = d["X"][f]
            npz[f"{key}__col__{f}"] = (col.cat.codes.to_numpy() if str(col.dtype) == "category"
                                       else col.to_numpy())
    np.savez_compressed(f"docs/results/{tag}_phi.npz", **npz)
    print(f"phi matrices saved: docs/results/{tag}_phi.npz ({list(PHI_STORE)})")
import os, shutil
if os.environ.get("T9_DELETE_PARQUET") == "1":   # default: KEEP
    shutil.rmtree(OUTPUT_DIR / tag, ignore_errors=True)
print(f"\nSaved docs/results/{tag}.json")
print("TIER-1 payer C1:", out["tier1"]["C1"]["payer"])
print("TIER-1 payer C2:", out["tier1"]["C2"]["payer"])
print("TIER-2 win  C1:", out["tier2_clf"]["C1"])
print("TIER-2 win  C3:", out["tier2_clf"]["C3"])
