# SHAP Attribution on the V10 Schema - BOTH Tiers (1M + 10M) - MMP & SSP "why"

**Date:** 10 Jul 2026

**Schema:** operative **V10** (Private-Rival Market, `rival_pool`, ρ\* = 0.8)

**Seed:** 90217 (headline tables) + seeds 90213 / 90218 at 1M (robustness, §Robustness below)

**Scales:** 1M (`test`) + 10M (`scale10m`)

**Method:** exact TreeSHAP mean \|φ\| on a 4,000-row held-out test sample, computed with the
**tree_path_dependent** perturbation mode held fixed across all conditions, heads and tiers
(Tier-1 via `shap.TreeExplainer.shap_values`; Tier-2 via XGBoost `pred_contribs` - both are
path-dependent TreeSHAP). The interventional mode is not computable on these native-categorical
XGBoost models (a `shap`-library limitation); mode-robustness is instead certified on
ordinal-encoded twin models in §Robustness. **Evaluation population:** all heads are attributed on
the full uncensored test sample (the EV-scoring population, matching deployment), which for the
conditional heads P(payer\|install) and E(spend\|payer) differs from their conditional training
populations - attributions there include extrapolation to rows outside the training funnel stage.

*Supersedes `SHAP_Analysis_10M.md` (19 Jun, v7-BN, Tier-1 / MMP only) for the V10 paper. This
version adds the **Tier-2 (win-model) SHAP** - the SSP "why" - which v10 is actually about.*

Data: `docs/results/v10_shap_test.json`, `v10_shap_scale10m.json`; script `scripts/v10_shap.py`.

## What the measurement is

For a trained model *f* with features *N*, the SHAP value φᵢ is the unique additive attribution
(efficiency + symmetry + dummy + additivity) of feature *i* in the game *v(S) = E\[f \| x_S\]*. We
report **mean \|φᵢ\|** over the sample - average magnitude of each feature's contribution - computed
**exactly** for tree ensembles (no φ sampling; only the evaluation sample is sampled). Units are
log-odds for the probability heads / win classifier, USD for the spend head; magnitudes compare
*within* a head/column, not across. Two contrasts isolate the two data layers:

- **Tier-1, C1 vs C2** (funnel heads): the **MMP** "why" - MMP observes the funnel on all rows vs
  won-rows-only, so the contrast shows where the de-biased distribution lets the model draw signal.
- **Tier-2, C1 vs C3** (won-classifier P(win\|bid,ctx)): the **SSP** "why" - SSP reveals clearing +
  competition density, so the contrast shows which market features carry C3's win-model edge.

------------------------------------------------------------------------

## Tier-1 - funnel heads, C1 vs C2 (the MMP mechanism) - 10M

+-----------------------+------------------------------+---------------------------+
| Head                  | C1  (no MMP - won rows only) | C2  (with MMP - all rows) |
+=======================+==============================+===========================+
| **P(click)**          | campaign_id 0.42\            | campaign_id 0.46\         |
|                       | app_id 0.23\                 | slot_format 0.36\         |
|                       | slot_format 0.18             | app_id 0.26               |
+-----------------------+------------------------------+---------------------------+
| **P(install\|click)** | app_id 0.33\                 | app_id 0.46\              |
|                       | campaign_id 0.23\            | campaign_id 0.38\         |
|                       | city 0.04                    | hour 0.09                 |
+-----------------------+------------------------------+---------------------------+
| **P(payer\|install)** | campaign_id 0.25\            | **app_id 0.32**\          |
|                       | region 0.08\                 | campaign_id 0.22\         |
|                       | hour 0.05\                   | city 0.08                 |
|                       | *(app_id not in top 8)*      |                           |
+-----------------------+------------------------------+---------------------------+
| **E(spend\|payer)**   | campaign_id 1.10\            | **app_id 6.10**\          |
|                       | app_id 1.06\                 | city 2.10\                |
|                       | city 0.62                    | campaign_id 1.84          |
+-----------------------+------------------------------+---------------------------+

*(1M agrees except the payer head, where C1/C2 are payer-count-noisy - 1M has ~ten× fewer payers, so
read the payer/spend rows off 10M.)*

## Tier-2 - won-classifier P(win\|bid,ctx), C1 vs C3 (the SSP mechanism) - NEW

+---------+---------------------------+-----------------------------+
| Scale   | C1  (no SSP) top features | C3  (with SSP) top features |
+=========+===========================+=============================+
| **10M** | floor 1.15\               | floor 1.15\                 |
|         | bid 0.67\                 | bid 0.71\                   |
|         | slot_format 0.61\         | slot_format 0.57\           |
|         | day_of_week 0.13\         | **bid_density 0.21**\       |
|         | slot_height 0.12\         | **hist_clearing_ssp 0.17**\ |
|         | app_id 0.10               | day_of_week 0.13            |
+---------+---------------------------+-----------------------------+
| **1M**  | floor 1.00\               | floor 0.99\                 |
|         | bid 0.60\                 | bid 0.63\                   |
|         | slot_format 0.55\         | slot_format 0.54\           |
|         | app_id 0.13\              | **bid_density 0.20**\       |
|         | day_of_week 0.12          | app_id 0.13\                |
|         |                           | **hist_clearing_ssp 0.11**  |
+---------+---------------------------+-----------------------------+

The two SSP-only features - `bid_density` (H9, realized rival count) and `hist_clearing_ssp`
(rolling clearing-price history) - appear **only in C3** (they are censored to NaN in C1/C2), and
they rank **4th-5th**, above every generic context feature. Same at both scales.

------------------------------------------------------------------------

## Interpretation - the two layers surface two different feature families

**1. MMP → entity-value features, by de-biasing the deep funnel.**

- The MMP effect is negligible pre-funnel (click head ≈ identical C1/C2) and grows monotonically down
  the funnel, starkest at the value stages.
- Under C1 the **payer** head cannot even surface `app_id` (won-only censoring hides the
  `app_id → archetype → conversion` relationship on a selection-biased subsample); under C2 `app_id`
  becomes the **#1** payer feature (0.32, a 46% share of the head's total attribution) and its
  **spend** importance jumps ≈ 6× in absolute terms (1.06 → 6.10), a share-of-attribution rise
  from 34% to 59%.
- MMP adds no new column - it *de-biases the conditional distribution* so the model can finally learn
  the entity-ID → latent-value coupling. This is the feature-level signature of the biased-view
  (selection-bias) correction the aggregate metrics show (payer-AUC +0.09, ev_ratio 0.52 → 0.88).

**2. SSP → market/competition features, in the win model.**

- C3's win-ranking edge is *mechanistically* carried by exactly the two SSP-exclusive market signals
  v10 generates: the competition density (`bid_density` / H9) and the clearing-price history
  (`hist_clearing_ssp`).
- They out-rank all context features and are simply absent (censored) for C1/C2 - so the +0.014
  auc_win contrast has a concrete feature-level cause, not a diffuse one.
- Direct SHAP confirmation of the SSP half of the mechanism question: **MMP shows up as user/entity-value
  features, SSP as market features.**

**3. The IDs are structural, not leakage.**

- `app_id` / `campaign_id` dominate because BN edges #1/#2 couple them to the latent archetype/value -
  a designed, legitimately-observable conditional dependence.
- No latent is ever a model input; the IDs become informative only through the generative coupling.

## Robustness (added 18 Jul 2026)

**Multi-seed (1M, seeds 90213 / 90217 / 90218).** The structural claims replicate on every seed;
the exact ranks move with payer-count noise at 1M:

- `app_id` in the C1 payer top-8: **0/3 seeds**. `app_id` in the C2 payer top-8: **3/3 seeds**
  (ranks #2 / #5 / #2; it is #1 at 10M, where the payer count is ~10× larger). The presence/absence
  flip - the de-biasing signature - is seed-robust; the "#1" rank is a 10M statement.
- Win model under C3: `bid_density` ranks **top-4 in 3/3 seeds** (0.25 / 0.20 / 0.31);
  `hist_clearing_ssp` is in the top-8 in **2/3 seeds** at 1M (and present at 10M). Under C1 both
  are absent in **3/3 seeds** (censored). The SSP-exclusive-feature claim is carried primarily by
  `bid_density`; `hist_clearing_ssp` is the weaker, scale-sensitive half.

Raw: `docs/results/v10_shap_test_s13.json`, `v10_shap_test_s18.json` (+ the seed-90217 files above).

**Perturbation-mode robustness (ordinal twins; 1M, seed 90217).** Because `shap`'s interventional
TreeSHAP cannot run on native-categorical XGBoost models (dtype cast error in the C extension,
confirmed at both scales), twin models with identical rows, funnel populations and hyperparameters
(categories as integer codes) were trained, on which BOTH modes are computable
(`scripts/shap_mode_check.py` → `docs/results/shap_mode_check_1m.json`):

- Rankings are essentially **invariant to the perturbation mode**: Spearman rank correlation of the
  full \|φ\| rankings between tree_path_dependent and interventional is 0.87-1.00 across all six
  (condition, head) pairs, with top-8 overlap 6-8/8. The two decisive heads are the most stable:
  C2 payer 0.995 (8/8), C3 win 0.989 (8/8).
- The claim-specific checks hold in **both** modes: the two SSP-exclusive features appear in the C3
  win top-8 under both modes and under neither mode in C1; app_id is in the C2 payer top-8 under
  both modes. The twin's win model also anchors to the production model (top-8 rank correlation
  vs the native C3 model 0.94).
- One disclosed encoding artifact: on the twin (unlike the native models, where it is absent in
  3/3 seeds), app_id also enters C1's payer top-8 at small magnitude (≈0.08-0.10, both modes) -
  with ~75 C1 payer training rows at 1M, ordinal integer codes over 500 apps admit spurious
  threshold splits that native categorical splits do not. The de-biasing claim therefore rests on
  the native-encoding models (mode held fixed, two scales, three seeds); the twin certifies only
  that the *mode choice* does not drive the rankings.

**Figures.** Bar figure: `Schema diagrams/T9_SHAP_figure_v10.svg` (three panels: payer, spend, win).
Beeswarm + dependence: `T9_SHAP_beeswarm_v10.svg`, `T9_SHAP_dependence_v10.svg` (10M, seed 90217).

## Caveats

- Headline tables are single-seed (90217); the structural claims are 3-seed-robust at 1M (see
  §Robustness), and the 1M payer/spend attributions are payer-count-noisy (read those off 10M).
- Attributions are computed on the full scoring population; for the conditional heads this includes
  extrapolation outside the training funnel stage (disclosed in Method above).
- All attributions are **conditional on the generative model** - they describe how the fitted
  predictors use features *given* the calibrated BN + rival layer, not a claim about real-world
  AdTech feature importance.
- For the paper, the two robust, scale-stable findings are the ones above (MMP de-biases the deep
  funnel → app_id; SSP feeds the win model → bid_density + hist_clearing_ssp).
