"""Beeswarm + dependence plots for the two decisive SHAP heads (v10, 10M).

Consumes the raw per-row SHAP matrices saved by
  v10_shap.py scale10m 90217 --save-phi
    -> docs/results/v10_shap_scale10m_phirun_phi.npz
containing t1_C2_payer, t1_C2_spend and t2_C3_win (4,000-row test sample).

Outputs (Schema diagrams/):
  T9_SHAP_beeswarm_v10.png/.svg    beeswarm: C2 payer head + C3 win model
  T9_SHAP_dependence_v10.png/.svg  dependence: app_id (payer head, per-app
                                   contribution spread) + bid_density (win model)

Run from repo root:
  python scripts/make_v10_shap_beeswarm.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "docs/results/v10_shap_scale10m_phirun_phi.npz"
z = np.load(NPZ, allow_pickle=False)

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9})


def load(key):
    phi = z[f"{key}__phi"]
    feats = [str(f) for f in z[f"{key}__feats"]]
    cols = {f: z[f"{key}__col__{f}"] for f in feats}
    return phi, feats, cols


def beeswarm(ax, key, title, k=8):
    phi, feats, cols = load(key)
    imp = np.abs(phi).mean(axis=0)
    top = np.argsort(imp)[::-1][:k][::-1]
    rng = np.random.default_rng(0)
    for row, i in enumerate(top):
        v = phi[:, i]
        c = np.asarray(cols[feats[i]], dtype=float)
        lo, hi = np.nanpercentile(c, [5, 95])
        cn = np.clip((c - lo) / max(hi - lo, 1e-9), 0, 1)
        y = row + (rng.random(len(v)) - 0.5) * 0.55
        ax.scatter(v, y, c=cn, cmap="coolwarm", s=4, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feats[i] for i in top], fontsize=8)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xlabel("SHAP value (log-odds)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
beeswarm(axes[0], "t1_C2_payer", "(a) P(payer | install) under C2 - beeswarm")
beeswarm(axes[1], "t2_C3_win", "(b) P(win | bid, ctx) under C3 - beeswarm")
fig.suptitle("T9 v10 SHAP beeswarm - 10M, seed 90217 (colour = feature value, "
             "5th-95th pct; categorical features shown by integer code)",
             fontsize=8.5, y=1.03)
fig.tight_layout()
out = ROOT / "Schema diagrams" / "T9_SHAP_beeswarm_v10"
fig.savefig(f"{out}.svg", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
print(f"written: {out}.svg/.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.6))
phi, feats, cols = load("t1_C2_payer")
i = feats.index("app_id")
axes2[0].scatter(cols["app_id"], phi[:, i], s=4, alpha=0.4, color="#2f6db3",
                 linewidths=0)
axes2[0].set_title("(a) app_id contribution to P(payer|install), C2",
                   fontsize=9.5, fontweight="bold", loc="left")
axes2[0].set_xlabel("app_id (integer code; 500 apps)", fontsize=8)
axes2[0].set_ylabel("SHAP value (log-odds)", fontsize=8)
phi, feats, cols = load("t2_C3_win")
i = feats.index("bid_density")
axes2[1].scatter(cols["bid_density"], phi[:, i], s=4, alpha=0.4,
                 color="#c25b1e", linewidths=0)
axes2[1].set_title("(b) bid_density contribution to P(win), C3",
                   fontsize=9.5, fontweight="bold", loc="left")
axes2[1].set_xlabel("bid_density (realized rival count)", fontsize=8)
axes2[1].set_ylabel("SHAP value (log-odds)", fontsize=8)
for a in axes2:
    a.axhline(0, color="#888888", lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
fig2.suptitle("T9 v10 SHAP dependence - 10M, seed 90217", fontsize=8.5, y=1.03)
fig2.tight_layout()
out2 = ROOT / "Schema diagrams" / "T9_SHAP_dependence_v10"
fig2.savefig(f"{out2}.svg", bbox_inches="tight")
fig2.savefig(f"{out2}.png", dpi=200, bbox_inches="tight")
print(f"written: {out2}.svg/.png")
