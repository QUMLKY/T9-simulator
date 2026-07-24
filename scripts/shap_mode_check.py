"""SHAP perturbation-mode robustness check (report BOTH modes).

shap's interventional TreeSHAP cannot handle XGBoost native-categorical models
(dtype('O') cast error in the C extension), so the production models support
only tree_path_dependent. This check therefore trains ORDINAL-ENCODED TWIN
models (same rows, same funnel populations, same hyperparameters; category
columns as integer codes) on which BOTH modes are computable, and asks whether
the paper's attribution claims are robust to the mode choice:

  - tier-1 payer + spend heads, C1 vs C2 (the app_id de-biasing signature)
  - tier-2 win classifier, C1 vs C3 (the SSP-exclusive features)

Reports, per (condition, head): top-8 under each mode, Spearman rank
correlation of the full |phi| rankings between modes, top-8 overlap, and the
claim-specific checks (app_id rank; bid_density / hist_clearing_ssp presence).
Also reports ordinal-twin vs native-model (path mode) rank agreement so the
twin is anchored to the production models.

Reuses the 1M seed-90217 parquet generated at output/v10_shap_test_interv.
Run from repo root:
  python scripts/shap_mode_check.py
Writes docs/results/shap_mode_check_1m.json.
"""
import json
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

from t9sim import censor
from t9sim.paths import OUTPUT_DIR
from t9sim.pipeline import (PipelineConfig, XGB_CLF, XGB_REG, attach_hist_clearing,
                            funnel_population, t2_feats, temporal_split)

SEED = 90217
SRC = OUTPUT_DIR / "v10_shap_test_interv" / "auctions.parquet"   # 1M, seed 90217
NATIVE_JSON = "docs/results/v10_shap_test.json"                   # native path-dep run

df = pd.read_parquet(SRC)
pc = PipelineConfig()
CATS = [c for c in pc.t1_cat if c in df.columns]
for c in CATS:
    df[c] = df[c].astype("category")
split = temporal_split(df)
base_tr, base_va, base_te = (df[split == s] for s in ("train", "val", "test"))


def ordinal(X, feats):
    """Category columns -> integer codes (global category objects => consistent)."""
    out = pd.DataFrame(index=X.index)
    for f in feats:
        col = X[f]
        out[f] = col.cat.codes.astype(np.int32) if str(col.dtype) == "category" \
            else col.to_numpy()
    return out


def train_ordinal(tr, va, target, feats, is_clf):
    params = {k: v for k, v in (XGB_CLF if is_clf else XGB_REG).items()
              if k != "enable_categorical"}
    m = (xgb.XGBClassifier if is_clf else xgb.XGBRegressor)(**params)
    m.fit(ordinal(tr, feats), tr[target],
          eval_set=[(ordinal(va, feats), va[target])], verbose=False)
    return m


def both_modes(model, X, feats):
    Xs = ordinal(X.sample(min(4000, len(X)), random_state=0), feats)
    path = np.asarray(shap.TreeExplainer(model).shap_values(Xs))
    bg = Xs.sample(100, random_state=1)
    interv = np.asarray(shap.TreeExplainer(model, data=bg,
                                           feature_perturbation="interventional")
                        .shap_values(Xs, check_additivity=False))
    res = {}
    for name, sv in (("path", path), ("interventional", interv)):
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1]
        res[name] = {feats[i]: round(float(imp[i]), 5) for i in order[:8]}
        res[f"_imp_{name}"] = imp
    rc = spearmanr(res["_imp_path"], res["_imp_interventional"]).correlation
    res["rank_corr_modes"] = round(float(rc), 4)
    res["top8_overlap"] = len(set(res["path"]) & set(res["interventional"]))
    del res["_imp_path"], res["_imp_interventional"]
    return res


out = {"seed": SEED, "scale": "1M", "encoding": "ordinal twin (integer codes)",
       "note": ("interventional TreeSHAP is not computable on the production "
                "native-categorical models (shap library limitation); this twin "
                "check certifies mode-robustness on identically trained "
                "ordinal-encoded models"),
       "tier1": {}, "tier2": {}}

HEADS = (("payer", "is_payer", True), ("spend", "ltv_value", False))
for cond in ("C1", "C2"):
    tr_v, va_v = censor.view(base_tr, cond), censor.view(base_va, cond)
    out["tier1"][cond] = {}
    for stage, target, is_clf in HEADS:
        tr = funnel_population(tr_v, cond, stage)
        va = funnel_population(va_v, cond, stage)
        m = train_ordinal(tr, va, target, pc.t1_feats, is_clf)
        out["tier1"][cond][stage] = both_modes(m, base_te, pc.t1_feats)
        print(f"tier1 {cond} {stage}: rank_corr={out['tier1'][cond][stage]['rank_corr_modes']} "
              f"overlap={out['tier1'][cond][stage]['top8_overlap']}/8", flush=True)

for cond in ("C1", "C3"):
    tr = censor.view(base_tr, cond); va = censor.view(base_va, cond); te = base_te.copy()
    attach_hist_clearing(tr, va, te, censor.CONDITIONS[cond]["clearing"])
    feats = t2_feats(cond, True, view_cols=tr.columns)
    feats = feats + [f for f in ("bid_price", "floor_price") if f not in feats]
    m = train_ordinal(tr, va, "won", feats, True)
    out["tier2"][cond] = both_modes(m, te, feats)
    print(f"tier2 {cond}: rank_corr={out['tier2'][cond]['rank_corr_modes']} "
          f"overlap={out['tier2'][cond]['top8_overlap']}/8", flush=True)

# claim-specific checks
naive = json.load(open(NATIVE_JSON))
c2p = out["tier1"]["C2"]["payer"]
out["claims"] = {
    "app_id_in_C2_payer_top8_path": "app_id" in c2p["path"],
    "app_id_in_C2_payer_top8_interventional": "app_id" in c2p["interventional"],
    "app_id_in_C1_payer_top8_either_mode": ("app_id" in out["tier1"]["C1"]["payer"]["path"]
                                            or "app_id" in out["tier1"]["C1"]["payer"]["interventional"]),
    "ssp_feats_in_C3_win_path": [f for f in ("bid_density", "hist_clearing_ssp")
                                 if f in out["tier2"]["C3"]["path"]],
    "ssp_feats_in_C3_win_interventional": [f for f in ("bid_density", "hist_clearing_ssp")
                                           if f in out["tier2"]["C3"]["interventional"]],
    "ssp_feats_in_C1_win_either_mode": [f for f in ("bid_density", "hist_clearing_ssp")
                                        if f in out["tier2"]["C1"]["path"]
                                        or f in out["tier2"]["C1"]["interventional"]],
}
# ordinal twin anchored to native production models (path mode, same seed/scale)
anchor = {}
for cond in ("C1", "C3"):
    nat = naive["tier2_clf"][cond]
    twin = out["tier2"][cond]["path"]
    common = [f for f in nat if f in twin]
    if len(common) >= 4:
        rc = spearmanr([list(nat).index(f) for f in common],
                       [list(twin).index(f) for f in common]).correlation
        anchor[f"tier2_{cond}_top8_rank_corr_native_vs_twin"] = round(float(rc), 3)
    anchor[f"tier2_{cond}_top8_overlap_native_vs_twin"] = len(set(nat) & set(twin))
out["native_anchor"] = anchor

json.dump(out, open("docs/results/shap_mode_check_1m.json", "w"), indent=2, default=float)
print("\nSaved docs/results/shap_mode_check_1m.json")
print("claims:", json.dumps(out["claims"], indent=1))
