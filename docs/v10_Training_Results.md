# T9 — V10 Training/Eval Results (anchored point, 1M n=10 · 10M n=10)

**Schema:** V10 "Private-Rival Market" (BUILT + EVALUATED; branch `t9-v10-rival`, not merged; v7 stays
operative pending sign-off). **Generator:** v10 build, flag-gated — `rival_pool` OFF reproduces v7
byte-identical (gate G1, hash-checked). **Operating point:** the externally **anchored** privateness
ρ\* = 0.8 (value-of-lost-price-information band, Wang et al. 2023 GMM-vs-CGMM: 5–13% all-rows RMSE
gain at ~30% win rate; T9 ρ-grid mapping 24.8% @ρ=0 → 3.7% @ρ=1 → ρ\*≈0.8; `v10_Anchor_Bands.md`).

**V10 "ON" = v7 funnel BN edges (#1–#5) + `rival_pool`** (K=8 rival archetypes; LU7 = max over
participating rivals' structured bids; H9 = endogenous rival count, SSP-only) **+** pipeline
`hist_clearing=True` (leak-gated OOF/LOO rolling clearing features, C3/C4).

**Pipeline (extended 5 Jul 2026):** Tier-1 4-head hurdle (XGBoost) · Tier-2 = **AFT + first-class
binary win CLASSIFIER co-reported** (monotone sweep of bid as explicit input) · paired **DeLong**
tests with 95% CIs on AUC contrasts · **ECE/MCE** win-calibration · closed-form lognormal **CRPS** +
price **RMSE** on all/won/lost rows · **TWO bidders** — AFT-curve bidder and classifier-curve (CLF)
bidder — so every economic metric is **model-robust** · temporal split · `--no-shap`.

**Conditions:** **C1** DSP · **C2** DSP+MMP · **C3** DSP+SSP · **C4** all · **oracle** = same pipeline
handed the true latents (upper bound).

**Seeds / n:** 1M = **10 seeds** (90213–90222), no outlier removal; 10M = **10 seeds** (90213–90222),
**one fresh process per seed** (scale10m convention; parquet deleted after eval, reproducible by seed).
Cells are seed means; headline contrasts carry mean [95% t-CI] + sign counts.

**Raw per-seed JSON:** `docs/results/v10_anchor_s13.json` … `s22.json` (1M);
`docs/results/v10_10m_s13.json` … `s22.json` (10M); aggregates `v10_anchor_n10.json`, `v10_10m_n10.json`.
Runners: `t9_sim/scripts/v10_anchor_5seed.py`, `v10_anchor_n10.py`, `v10_10m_worker.py` + `v10_10m_driver.py`.
Earlier stages (100K ablation, ρ-grid, decisive ρ=1 test): `docs/results/v10_*.json`, write-up
`v10_Results_5Jul2026.md`.

---

## 1M — anchored point ρ\*=0.8, n=10 seeds (means)

| Metric | C1 (DSP) | C2 (+MMP) | C3 (+SSP) | C4 (all) | Oracle |
|---|---|---|---|---|---|
| ev_spearman | 0.445 | **0.552** | 0.445 | **0.552** | 1.000 |
| ev_ratio | 0.633 | **0.968** | 0.633 | **0.968** | 1.000 |
| auc_click | 0.6797 | **0.7100** | 0.6797 | **0.7100** | 0.7843 |
| auc_install | 0.6217 | **0.6725** | 0.6217 | **0.6725** | 0.8609 |
| auc_payer | 0.5563 | **0.6141** | 0.5563 | **0.6141** | 0.8357 |
| auc_win (AFT) | 0.7648 | 0.7648 | **0.7823** | **0.7823** | — |
| auc_win (CLF) | 0.8091 | 0.8091 | **0.8227** | **0.8227** | — |
| logloss_win (CLF) | 0.4525 | 0.4525 | **0.4389** | **0.4389** | — |
| ece_win | **0.0517** | **0.0517** | 0.0554 | 0.0554 | — |
| mce_win | 0.189 | 0.189 | **0.118** | **0.118** | — |
| price_rmse_all | 25.7 | 25.7 | **24.4** | **24.4** | — |
| price_rmse_lost | 29.7 | 29.7 | **28.3** | **28.3** | — |
| price_rmse_won | 7.17 | 7.17 | **4.71** | **4.71** | — |
| price_crps_all | 0.658 | 0.658 | **0.621** | **0.621** | — |
| price_crps_lost | 0.577 | 0.577 | **0.560** | **0.560** | — |
| overpay/won, AFT bidder ($) | 0.00229 | 0.00338 | **0.00213** | 0.00315 | 0.01187 |
| overpay/won, CLF bidder ($) | **0.00110** | 0.00122 | **0.00110** | 0.00123 | — |
| surplus/won, AFT bidder ($) | 0.00901 | 0.01456 | 0.00827 | **0.01646** | 0.18458 |
| surplus/won, CLF bidder ($) | 0.00682 | **0.01557** | 0.00721 | 0.01545 | — |
| profit, AFT bidder ($) | 653 | 932 | 585 | **1,059** | 2,427 |
| profit, CLF bidder ($) | 367 | 778 | 401 | **786** | — |
| ROAS, CLF bidder | 3.53 | **6.59** | 3.64 | 6.51 | — |
| n_won, CLF bidder | 49,833 | 51,209 | 50,622 | **52,041** | — |

## 10M — anchored point ρ\*=0.8, n=10 seeds (90213–90222) (means)

| Metric | C1 (DSP) | C2 (+MMP) | C3 (+SSP) | C4 (all) | Oracle |
|---|---|---|---|---|---|
| ev_spearman | 0.589 | **0.633** | 0.589 | **0.633** | 1.000 |
| ev_ratio | 0.524 | **0.889** | 0.524 | **0.889** | 1.000 |
| auc_click | 0.7090 | **0.7212** | 0.7090 | **0.7212** | 0.7842 |
| auc_install | 0.6719 | **0.6994** | 0.6719 | **0.6994** | 0.8595 |
| auc_payer | 0.5855 | **0.6835** | 0.5855 | **0.6835** | 0.8352 |
| auc_win (AFT) | 0.7772 | 0.7772 | **0.7899** | **0.7899** | — |
| auc_win (CLF) | 0.8142 | 0.8142 | **0.8280** | **0.8280** | — |
| logloss_win (CLF) | 0.4463 | 0.4463 | **0.4318** | **0.4318** | — |
| ece_win | **0.0521** | **0.0521** | 0.0540 | 0.0540 | — |
| mce_win | 0.119 | 0.119 | **0.095** | **0.095** | — |
| price_rmse_all | 24.59 | 24.59 | **24.10** | **24.10** | — |
| price_rmse_lost | 28.54 | 28.54 | **27.98** | **27.98** | — |
| price_rmse_won | 4.72 | 4.72 | **4.56** | **4.56** | — |
| price_crps_all | 0.634 | 0.634 | **0.614** | **0.614** | — |
| price_crps_lost | 0.558 | 0.558 | **0.551** | **0.551** | — |
| overpay/won, AFT bidder ($) | 0.00215 | 0.00377 | **0.00213** | 0.00379 | 0.01189 |
| overpay/won, CLF bidder ($) | **0.00118** | 0.00145 | 0.00119 | 0.00145 | — |
| surplus/won, AFT bidder ($) | 0.01554 | 0.02075 | 0.01593 | **0.02104** | 0.14812 |
| surplus/won, CLF bidder ($) | 0.01023 | **0.01382** | 0.01025 | 0.01290 | — |
| profit, AFT bidder ($) | 7,783 | 11,798 | 7,904 | **11,858** | 19,202 |
| profit, CLF bidder ($) | 4,746 | **6,575** | 4,817 | 6,236 | — |
| ROAS, CLF bidder | 4.65 | **5.31** | 4.62 | 5.01 | — |
| n_won, CLF bidder | 442,763 | 477,606 | 449,195 | **483,266** | — |

---

## Headline contrasts (mean [95% t-CI], sign count)

| Contrast | 1M, n=10 | 10M, n=10 | Verdict |
|---|---|---|---|
| **SSP prediction** — auc_win CLF, C3−C1 | +0.0137 [0.0053, 0.0220], 10/10 | +0.0137 [0.0055, 0.0220], 10/10 | **real, scale-stable** |
| SSP prediction — auc_win AFT, C3−C1 | +0.0175, 10/10 | +0.0128 [0.0051, 0.0204], 10/10 | real (both heads agree) |
| SSP prediction — price CRPS gain, all rows | +5.3% | +2.9% [1.2, 4.7], 9/10 | real |
| SSP prediction — price RMSE gain, all rows | +4.8% | +2.0% [1.2, 2.8], 10/10 | real, modest |
| **SSP economics** — overpay/won reduction | +6.5% (AFT bidder, 10/10) **but −0.6% [−1.7, +0.4] (CLF bidder, 2/10)** | −0.9% [−2.5, +0.7] (CLF bidder, 1/10) | **AFT artifact — not demonstrated** |
| SSP economics — CLF-bidder profit, C3 vs C1 | CI spans 0 | +1.6% [−3.3, +6.4], 4/10 | not demonstrated |
| **MMP economics** — CLF-bidder profit, C2 vs C1 | ≈ 2.1× (367 → 778) | +39% (4,746 → 6,575) | **robust** |
| MMP value-bias — ev_ratio C1 → C2 | 0.63 → 0.97 | 0.52 → 0.89 | durable (selection bias, uncorrectable by scale) |

**Context — the decisive test that precedes these numbers (ρ=1, 100K/1M):** the v10 SSP contrast
**survives the well-specified classifier** (+0.021 vs v8's +0.0002) — v8's null was DGP-conditional
(conditional-IPV market ⇒ single price is sufficient, Athey–Haile), not a property of SSP data.
The anchored point then reads off the honest niche-market estimate. Full arc: `v10_Results_5Jul2026.md`.

---

## Reading the tables

**Design identities (hold at every scale, every seed):** MMP touches only the funnel/value axis →
`C3 = C1`, `C4 = C2` on ev_spearman/ev_ratio/auc_click/auc_install/auc_payer. SSP touches only the
price/win axis → `C1 = C2`, `C3 = C4` on auc_win/logloss/ECE/RMSE/CRPS. So **MMP value = C2−C1 ≡
C4−C3**; **SSP value = C3−C1 ≡ C4−C2**. The repeated cells in the tables are these identities, not
errors.

**AFT vs CLF (why two of everything):** the v8-era shrink test showed the AFT head under-fits both
conditions equally; every v10 ranking claim is therefore quoted from BOTH heads (they agree), and
every economic claim from BOTH bidders. The two-bidder check is what exposed the overpayment
"reduction" as an AFT-curve artifact — under the classifier-curve bidder it vanishes. Quote SSP
economics only from the CLF bidder.

**Why SSP model-gains don't convert to profit:** the bidder is gated on the **value side** — EV
ranking (ev_spearman 0.44–0.64 vs oracle 1.0) determines which impressions are worth buying; a
better win-curve cannot rescue mispriced value. This is the same gate the 12-Jun oracle-features
diagnostic identified. MMP improves exactly that side (ev_ratio, payer-head AUC) and its profit
advantage is robust under both bidders at both scales.

**ROAS is seed-unstable — report profit, not ROAS.** The n=5 apparent inversion (CLF C2 ROAS 4.41 <
C1 4.65) did **not** survive n=10 (C2 5.31 > C1 4.65): revenue is whale-dominated, so ROAS swings widely
across seeds. Profit is the seed-stable economic metric — the MMP profit gap is robust at both scales
(now **+39%** CLF at 10M, n=10) — while ROAS is reported for completeness only.

**Oracle gap:** all conditions sit far below the oracle on the value axis by design — the latent
slice (LU/LA/LC + rival privateness) is the ablation's headroom, not a defect to be closed.

---

**Bottom line:** at the externally anchored operating point of a market whose competing bids now
arise from named rival mechanisms rather than iid noise, the ablation's SSP verdict is **axis-split
and scale-stable: SSP data improves the model (win-AUC ≈+0.014 at both scales, CRPS +3–5%, all seeds
positive at both scales) but does not demonstrably convert to money (overpay artifact; profit CI
spans zero), while MMP data makes the money (2.1× @1M → +39% @10M classifier-bidder profit; ev_ratio
0.63→0.97 / 0.52→0.89).** Reason 3 of the proof chain is resolved axis-specifically: falsified for
prediction, sustained for economics in this niche-market configuration. (V10 = BUILT + EVALUATED,
branch `t9-v10-rival` not merged; v7 stays operative pending sign-off.)
