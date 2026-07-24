"""V10 SHAP figure for the paper.

Reads the TRACKED SHAP outputs (docs/results/v10_shap_scale10m.json; 10M,
seed 90217, v10 rival market at rho*=0.8, tree_path_dependent TreeSHAP) and
renders the three-panel attribution figure:

  (a) Tier-1 P(payer|install), C1 vs C2 - the MMP de-biasing signature
      (app_id absent from C1's top-8 -> #1 under C2);
  (b) Tier-1 E(spend|payer), C1 vs C2 - the spend-side signature (app_id ~6x);
  (c) Tier-2 P(win|bid,ctx), C1 vs C3 - the SSP edge carried by the two
      SSP-exclusive features (bid_density, hist_clearing_ssp).

Replaces the superseded 19-Jun v7-BN figure (T9_SHAP_figure_10M.*).
Run from repo root:
  python scripts/make_v10_shap_figure.py
Writes Schema diagrams/T9_SHAP_figure_v10.svg + .png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
J = json.load(open(ROOT / "docs/results/v10_shap_scale10m.json"))

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9,
                     "axes.edgecolor": "#666666", "axes.linewidth": 0.8})
C_BASE, C_LIFT, C_SSP = "#9aa5b1", "#2f6db3", "#c25b1e"

SSP_ONLY = {"bid_density", "hist_clearing_ssp", "min_bid_to_win"}


def panel(ax, d_base, d_new, base_lbl, new_lbl, title, xlabel, new_color,
          highlight=(), note=None):
    feats = list(dict.fromkeys(list(d_new.keys()) + list(d_base.keys())))[:9]
    feats = feats[::-1]                       # top at the top
    y = np.arange(len(feats))
    vb = [d_base.get(f, 0.0) for f in feats]
    vn = [d_new.get(f, 0.0) for f in feats]
    ax.barh(y + 0.2, vb, height=0.38, color=C_BASE, label=base_lbl)
    bars = ax.barh(y - 0.2, vn, height=0.38, color=new_color, label=new_lbl)
    for i, f in enumerate(feats):
        if f in highlight:
            bars[i].set_color(C_SSP)
            bars[i].set_hatch("//")
    ax.set_yticks(y)
    ax.set_yticklabels([f + (" *" if f in highlight else "") for f in feats],
                       fontsize=8)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        ax.text(0.98, 0.98, note, transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top", style="italic", color="#444444")


fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
panel(axes[0], J["tier1"]["C1"]["payer"], J["tier1"]["C2"]["payer"],
      "C1 (DSP only)", "C2 (DSP+MMP)",
      "(a) P(payer | install)", "mean |SHAP| (log-odds)", C_LIFT,
      note="app_id: absent from C1 top-8 → #1 under C2")
panel(axes[1], J["tier1"]["C1"]["spend"], J["tier1"]["C2"]["spend"],
      "C1 (DSP only)", "C2 (DSP+MMP)",
      "(b) E(spend | payer)", "mean |SHAP| (USD)", C_LIFT,
      note="app_id: 1.06 → 6.10")
panel(axes[2], J["tier2_clf"]["C1"], J["tier2_clf"]["C3"],
      "C1 (DSP only)", "C3 (DSP+SSP)",
      "(c) P(win | bid, context)", "mean |SHAP| (log-odds)", C_LIFT,
      highlight=SSP_ONLY, note="* SSP-exclusive features")
fig.suptitle("T9 v10 SHAP attributions - 10M auctions, seed 90217, "
             "TreeSHAP (tree_path_dependent), 4,000-row test sample",
             fontsize=9, y=1.02)
fig.tight_layout()
out = ROOT / "Schema diagrams" / "T9_SHAP_figure_v10"
fig.savefig(f"{out}.svg", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
print(f"written: {out}.svg / .png")
