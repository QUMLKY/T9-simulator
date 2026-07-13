# T9 Simulator Output Schema (v10)

**v10 status (upd. 10 Jul 2026): BUILT + EVALUATED (100K · 1M · 10M); pending Dr Klimis sign-off; v7 remains operative in the repo.**
**KDD decision (10 Jul, `KDD_Schema_Decision_10Jul2026.md`):** v10 at the anchored ρ\*=0.8 is the **operative ablation schema the paper reports** (v8 = the contrasting iid-market null). The two **sensitivity arms B1 (`min_bid_to_win`, §Rival-Pool Layer) and B2 (`explore_traffic`, logging-policy)** are flag-gated additions reported *alongside* the headline, **NOT folded into this baseline v10 spec** — so this document describes v10 without them; their build lives on branch `t9-v11-mbw` (OFF = byte-identical v10). B3 removed (§Deferred value-side extension).
v10 = v7 + the **Private-Rival Market layer** (§Rival-Pool Layer): the competing bid LU7 becomes the maximum over K = 8 generated rival-bidder archetypes whose bids carry structured private components, replacing v7's reduced two-term form. Flag-gated (`rival_pool`); **all-OFF reproduces v7 byte-identical** (hash-checked). Branch **`t9-v10-rival`**.
*Version trail:* **v8** (price/SSP feature layer) = evaluated **structural null**, kept unmerged for the trail (`v8_1M_Results.md`, `v8_Training_Results.md`). **v9** (LA3 pub_tier + deal_type) = **abandoned unbuilt** — superseded by v10, which fixes the market structure rather than adding SSP-exclusive features. Rationale and proof chain: `SSP_Null_Validity_Discussion_5Jul2026.md`; design: `v10_Proposal_5Jul2026.md`; results: `v10_Results_5Jul2026.md`, `v10_Training_Results.md`.
**Formal BN definition** (variables · DAG · joint factorization · censoring operators · estimands · identification propositions): **`docs/BN_Formalisation.md`** — the source of truth for the paper's "Formal model" section and dissertation §5; this schema doc stays the operative build spec.

**Market framing.** Tier-1 **US** mobile gaming market, calibrated to US 2025 benchmarks, priced in **USD ($)**. iPinYou (2013, China) supplies empirical *distribution shapes* for a subset of features; price *levels* are rescaled to 2025 US mobile eCPM. Region/city codes are anonymised integers with a canonical `city → region` hierarchy (no real names).

**Scope.** Every auction is **mobile in-app** inventory in a **gaming app** (publisher = a game). One unified genre taxonomy `G = [casual, strategy, rpg, hypercasual]` (order per config `category_order`) spans user interest, host-app category and advertised genre. *Simplification:* real UA also runs on non-gaming inventory; relaxing this is future work. The **competition** is **not** gaming-only — in v10 this is modelled structurally (3 gaming + 5 non-gaming rival archetypes, §Rival-Pool Layer), no longer as an iid noise term.

**Auction format.** **All auctions are first-price** — consistent with 2025 mobile in-app mediation (AppLovin MAX, Unity LevelPlay, AdMob), which is predominantly first-price post-2019. iPinYou's second-price `paying_price` is reused as the empirical **shape** for first-price clearing, justified by the **Revenue Equivalence Theorem** (Vickrey 1961; Myerson 1981): expected clearing prices coincide across formats under standard assumptions. We abstract away format-specific bid-shading dynamics — a declared, theorem-backed modelling assumption.

**Attribution.** Single-touch; no cross-auction user history. A user's latents (LU1–LU6) and Section-A features are shared across all that user's auctions; only per-auction quantities vary.

------------------------------------------------------------------------

## Terminology — Observability Statuses

**Conditions vs campaign columns.** The four experimental conditions are **C1** = DSP only · **C2** = DSP + MMP · **C3** = DSP + SSP · **C4** = all layers — censored views of one ground truth (see §Censoring Map). Campaign *columns* `C1`–`C4` (`advertiser_id` … `ad_genre`) are always written as code; plain C1–C4 means the conditions.

Every column in the schema has exactly one **primary** status below (H4 additionally serves as a model *input* in the conditions where it is observed). The test for *latent* is a single question — **does the variable ever appear in the data the learner sees?** — not whether the generator knows its value (the generator knows everything).

| Status | Columns | Meaning |
|----|----|----|
| **Latent (ground-truth) parameters** | LU1–LU6 (per user) · LA1–LA2 (per app) · LC1–LC2 (per campaign) · **LR1–LR4 (per rival archetype, v10)** | Variables of the generative process: they parameterise the outcome draws but never appear as columns in **any** condition C1–C4, and no model attempts to infer them. |
| **Latent auction variable** | LU7 `competing_bid` | A realised hidden draw **per auction** (not a pool parameter). Never a column anywhere; on lost-but-sold rows (`LU7 ≥ H1`) in C3/C4 it is revealed *through* `H4 = LU7` — deliberately: that leak **is** the SSP signal. In v10, LU7 = max over generated rival bids (§Auction). |
| **Observed features** | A1–A5, B1–B6, `C1`–`C4`, D1–D3, H1 (floor), H2 (own bid) | Visible columns — model inputs in every condition. |
| **Outcome labels** | E1–E2, F1–F2, G1–G4, H3, H4, **H9 (v10)** | Realised random draws, generated on **all** master rows; each is **observed or censored per condition AND per row** (§Censoring Map): H3 everywhere; E/F/G on won rows in C1/C3 and on all rows in C2/C4; H4 **and H9** in C3/C4 only (where observed, H4/H9 also serve as Tier-2 inputs). |
| **Estimands — latent ground-truth probabilities** | P(click), P(install\|click), P(payer\|install), E\[ltv\], EV; P(win\|bid) for Tier 2 | The deterministic quantities the outcome formulas compute *before* each random draw. The labels are their observed noisy realisations; Tier-1 models estimate the funnel quantities and Tier 2 estimates P(win\|bid), all from visible features. |

**Notes.**

- *Latent* is used in its standard statistical sense (factor analysis, hidden Markov models): **hidden from the learner** — not "estimated by the model" and not "added later in development". A latent variable is simultaneously *ground truth* to the simulator and *latent* to every model; the two descriptions are not in tension — one describes its role in generation, the other its observability in the dataset.
- **Censored ≠ latent.** Censoring is the **experimental manipulation**: an *observable* label withheld from specific conditions/rows (e.g. G2 on lost rows in C1). Latency is a **design invariant**: hidden from *all* conditions, always (e.g. LU5). The C1–C4 comparison is built entirely on the former; the latents drive the data but are never part of any view.

------------------------------------------------------------------------

## Censoring Map — C1–C4 views (corrected 10 Jun 2026; approved 11 Jun 2026; H9 added by v10)

One master table; four views. **Identical rows and temporal split in every condition** (split days 1–20 / 21–24 / 25–28 BEFORE censoring); only label visibility differs. Views are masks, never copies.

| Condition | Layers | E/F/G (click, install, payer, LTV) | H4 clearing | H9 rival count (v10) | Always visible |
|----|----|----|----|----|----|
| **C1** | DSP only | **won rows only** (standard postbacks) | hidden | hidden | A, B, `C`, D, H1, H2, H3 |
| **C2** | DSP + MMP | **all rows** (cross-network attribution) | hidden | hidden | A, B, `C`, D, H1, H2, H3 |
| **C3** | DSP + SSP | **won rows only** | **all rows** (NaN only where unsold) | **all rows** | A, B, `C`, D, H1, H2, H3 |
| **C4** | DSP + MMP + SSP | **all rows** | **all rows** | **all rows** | A, B, `C`, D, H1, H2, H3 |

Latents (LU/LA/LC/LR) and the estimand columns are visible in **no** condition; `user_id` is master-lineage only (dropped from every view — v4 decision: user/device ids carry no signal under single-touch).

**Representation.** Censored cells are **NaN** in the view — distinct from `−1` (an *observed* no-event timestamp) and from `0` labels; this applies to all of E1–E2, F1–F2, G1–G4 on masked rows. "Hidden" H4/H9 in C1/C2 = all-NaN columns (kept, not dropped, so all views share one column schema).

**Why this is the real-world map.** Any DSP that serves an impression receives the MMP postback for it — won-row attribution is table stakes (C1/C3). What **MMP ownership** uniquely adds (AppLovin/Adjust) is attribution for auctions the DSP **lost**: the MMP's tracking links record clicks, and its SDK inside the advertised games records installs and spend, *regardless of which network served the winning impression* (C2/C4). Without it, a DSP literally cannot know what became of the users it bid on and lost — and because of the DSP's biased view (selection bias) those are systematically the **high**-value impressions. On the SSP side, an SSP sees every auction it runs — clearing prices on losses (H4) and the realized competition density (H9) are exactly what SSP ownership reveals and a lone DSP cannot observe.

**Counterfactual simplification (declared).** Lost-row E/F/G represent what the attribution chain would have recorded had *our* ad been served; we do not model the competitor's creative. The outcome formulas never depend on `won`, so this counterfactual is well-defined within the model.

**What each lift measures.**

- **C1→C2** — value of cross-network attribution: same rows, unbiased funnel labels; breaks the DSP's-biased-view selection bias.
- **C1→C3** — value of price visibility: lost-row clearing + competition density fingerprint hidden rival structure; Tier-2 win-curve precision.
- **C2→C4** — marginal SSP on top of full attribution.
- **C3→C4** — marginal MMP on top of price visibility.

**v7 BN dependencies (#1/#2) and the censoring map.** Unchanged from v7 (see the v7 schema §Censoring Map for the full note): #1 user↔app pairing acts symmetrically through the always-visible `app_id`; #2 value-exposure leaves no footprint in always-visible columns; both are marginal-preserving (IPF). The v10 rival-pool layer likewise leaves the funnel side untouched — its only view changes are H9 (new, SSP-only) and the *generative origin* of H4's lost-row values.

------------------------------------------------------------------------

## Generative Structure — four pools, then auctions (v10: rival pool added)

1.  **User pool** (N users) — latents **LU1–LU6** + Section-A features. A-features calibrated from **deduplicated** iPinYou users (user-population shape, not bid-weighted).
2.  **App pool** (apps) — visible `app_id` + `app_category`, plus **latent** `app_quality`, `app_audience_profile`.
3.  **Campaign pool** (campaigns) — visible `campaign_id` → `advertiser_id` → `advertiser_scale`, `ad_genre`, plus **latent** `creative_appeal`, `game_quality`.
4.  **Rival pool (v10)** — K = 8 persistent rival-bidder archetypes (3 gaming DSPs, 2 e-commerce retargeters, 1 brand buyer, 2 generalists), each with **latent** private LTV-model loadings, retargeting matrix, budget-pacing state and exchange-participation profile (LR1–LR4). Frozen at creation like the other pools; sampled per auction via participation draws.
5.  **Auctions** (M rows) — each samples one user + one app + one campaign, then draws Time (D), the **rival-market competing bid** (participating rivals' bids → LU7 = max, N → H9), bid/price labels (H), and funnel outcomes (E, F, G) on **every row** — lost-row outcomes are the declared counterfactual (§Censoring Map). Won/lost affects *visibility* (censoring), never *generation*.

**Pool semantics (confirmed v5; extends to the rival pool).** All pools follow the same generate-once-then-reference design: entities' features and latents are frozen at creation and identical across every auction row they appear in. Rival archetypes persist across the whole 28-day window — their retargeting rates are persistent, their pacing follows a day-level AR(1) process, and their flight schedules switch campaigns on and off — which is exactly what makes their footprints *statistically* recoverable from rolling SSP aggregates (§Rival-Pool Layer, footprint rule).

------------------------------------------------------------------------

## Section map

| v10 | Section | Cols | Notes |
|----|----|----|----|
| **A** | User | A1–A5 | A6 browser, A7 tags dropped |
| **B** | App context | B1–B6 | B6 visibility dropped → renumbered |
| **C** | Campaign | C1–C4 | scale not genre; targeting/creative dropped |
| **D** | Time | D1–D3 | Unix seconds |
| **E** | Click | E1–E2 | all rows; censored per §Censoring Map |
| **F** | Install | F1–F2 | all rows; censored per §Censoring Map |
| **G** | Conversion / LTV | G1–G4 | all rows; censored per §Censoring Map |
| **H** | Bid / floor / won / clearing / competition | H1–H4, **H9** | all rows; H4 rival-market clearing + H9 rival count, both C3/C4-only |
| **L** | Latents | LU1–LU7 · LA1–LA2 · LC1–LC2 · **LR1–LR4** | never a column in C1–C4 (see §Terminology) |

------------------------------------------------------------------------

## Visible Columns

*A/B/C/D/E/F/G columns are unchanged from v7 — see the v7 schema for the full table. H family below (v10 adds H9).*

| \# | Column | Type | Description | Source |
|----|----|----|----|----|
|  | **H — Bid / floor / won / clearing / competition** (all rows) |  |  |  |
| H1 | `floor_price` | float | Floor eCPM ($) | iPinYou shape, rescaled |
| H2 | `bid_price` | float | Our DSP bid eCPM ($) | value-estimate × exploration |
| H3 | `won` | int | 0/1 | `H2 ≥ max(LU7, floor)` |
| H4 | `clearing_price` | float | Cleared eCPM ($); NaN if unsold | first-price → `H2` if won; `LU7` if lost & sold |
| **H9** | `bid_density` | int | **Realized rival count N (v10)** — endogenous competition density; NaN in C1/C2 views | rival-pool participation draws (§Rival-Pool Layer); replaces v8's exogenous-Poisson H5 (measured ~0 signal) |

**Per-row availability (master table).** Every row carries every column: A, B, C, D, E, F, G, H + latents + estimands. The master is never NaN-censored — the only NaN is **H4 on unsold rows** (no clearing price exists); E2/F2 use −1 for "no event". All won/lost-conditional hiding happens in the **condition views** (§Censoring Map), never in generation.

------------------------------------------------------------------------

## Latent (Ground-Truth) Parameters — never a column in C1–C4

*User latents LU1–LU6, app latents LA1–LA2 and campaign latents LC1–LC2 are unchanged from v7 — see the v7 schema for the full tables. v10 adds the rival-pool family and redefines LU7's generator.*

### Per-auction latent

| \# | Column | Type | Description | Generated By |
|----|----|----|----|----|
| LU7 | `competing_bid` | float | Top competing bid (**v10: max over participating rivals' structured bids** — per auction; the "LU" prefix is historical) | §Auction (v10 form) |

### Rival-pool latents (per rival archetype, v10)

| \# | Latent | Type | Description | Drives |
|----|----|----|----|----|
| LR1 | `w_k` (private valuation loading) | scalar | Rival k's private loading on the latent impression-value score `z = z(log EV)` (built simplification of the proposal's φ(LU) feature map; non-gaming rivals = **smaller loadings on the same z**) | rival bid level (value-correlated, DSP-unreconstructable) |
| LR2 | `R[s,k]` (retargeting matrix) | matrix | Retargeting **level-shifters** (~N(0,1), scaled by β_R) per (observable user segment s, rival k); segments = **os × device_type**; persistent for the window | rival bid level (level-footprint in observable segment cells) |
| LR3 | `p_k(d)`, `F_k(d)` (pacing + flights) | series | Day-level AR(1) budget-pacing state (`p_k(d) = φ_p·p_k(d−1) + σ_p·η`) and campaign flight on/off windows | bid level + participation over time (footprint in time cells) |
| LR4 | `π_k(e)` (participation profile) | vector | Per-rival **per-exchange** participation propensity (time enters participation at day level only, via flights/pacing) | who shows up → composition of the participant set, N (H9) |

**Note — recoverability by design (the footprint rule).** No view ever sees LR1–LR4. What C3/C4 can recover is their *statistical footprint*: level components (retargeting, pacing, participation) leave cell-level traces in clearing prices (H4) and competition density (H9) over observable cells (time, exchange, segment); the latent-value interaction `w_k·z` is recoverable by **no one** (it prices a latent impression value that no side observes) — it supplies realism and DSP-unpredictability, not SSP-recoverable signal. This is the empirically confirmed footprint rule (pilot 2, 5 Jul 2026).

### Ground-truth / estimand columns (master table only — NEVER in any C1–C4 view)

Unchanged from v7: `p_click`, `p_install`, `p_payer`, `e_ltv`, `ev_truth` (see the v7 schema table). `LU7 competing_bid` remains master-only; it surfaces only as `H4` on lost-sold rows in C3/C4.

------------------------------------------------------------------------

## Matching Engine, Outcome Formulas, Feature–Latent Dependency (BN edges #1–#5)

**Unchanged from v7.** The funnel side (relevance, E/F/G formulas, the five marginal-preserving BN edges #1–#5 with their IPF constructions) is identical to the v7 spec — v10 touches only the auction/price layer. See the v7 schema §Matching Engine, §Outcome Formulas, §Feature–Latent Dependency.

------------------------------------------------------------------------

## Auction — Private-Rival Market (H, LU7, H9) — v10 form

So that SSP-layer data carries recoverable signal about market structure. The SSP-exclusive columns are **H4 and H9**: win/loss (H3) is visible in *all four* conditions (a DSP always observes its own auction outcome); what C3/C4 add is the *price* on losses (`H4 = LU7`) and the *competition density* (H9 = N).

**Impression expected value (ground truth)** — unchanged from v7:

    EV = P(click) · P(install|click) · P(payer|install) · E[ltv|payer]

**Competing bid — max over a generated rival market (v10; flag `rival_pool`):**

For auction *i* (user u, day d, hour h, exchange e):

    Participation:  Z_ik ~ Bernoulli( min(1, π_k(e) · F_k(d) · gate(p_k(d))) ),   A_i = {k : Z_ik = 1},   N_i = |A_i|
    Rival bid:      log b_ik = log base_e(i) + ρ·( w_k·z + β_R·R[s(u),k] + p_k(d) ) + σ_k·ε_ik,   ε iid N(0,1)
    Outcome:        LU7_i = max_{k ∈ A_i} b_ik        H9_i = N_i

- `z = z(log EV)` = the standardized log of the latent impression value (a function of user, app AND campaign latents); `base_e(i)` = iPinYou clearing **shape** (median-normalised) × the format's eCPM target — the shared, DSP-predictable core (unchanged from v7).
- **Archetype k=0 is forced always-on** (A_i ≠ ∅ ⇒ LU7 > 0 always). *Build divergence, flagged:* in the current build that slot is the first **gaming-DSP archetype** (strongly value-loaded), not the value-neutral generalist the proposal specified.
- **ρ (private-structure amplitude dial)** ∈ [0,1], multiplying the structured components (w, R, pacing). *Declared limitation:* the build does **not** rake log-bid moments to v7, so raising ρ *inflates* total price variance (this shapes the inverted anchor mapping below); the win-rate autocal + 6 validation gates are the calibration backstop. ρ = 0 ⇒ zero private structure — though the multi-rival mechanics alone (endogenous participation, order-statistic max) already give a small real SSP contrast there.
- **Anchored operating point: ρ\* ≈ 0.8** (band [0.6, 0.9]) — pinned externally by the value-of-lost-price-information (§Rival-Pool Layer, Anchoring). All v10 headline results are quoted at the anchored point.

**Our bid (H2), win rule (H3) and clearing (H4)** — unchanged from v7:

    H2 = value_estimate(DSP-visible features) · shade · LogNormal(0, σ_explore)
    H3 won = 1 if H2 ≥ max(LU7, floor) else 0
    H4 clearing = H2 if won | LU7 if lost & LU7 ≥ floor | NaN if unsold

**Adverse selection — the DSP's biased view (extended by v10).** In v7 the market priced hidden *impression value* (γ·z(log EV)); the DSP systematically wins low-value and loses high-value impressions, and every condition sees that pattern through H3. v10 adds a second, richer layer: the market now also prices **rival-private structure** (private valuations of user types, retargeting, pacing) that no DSP feature can reconstruct even in principle. C2/C4 (MMP) still see the *outcomes* of lost rows; C3/C4 (SSP) see the *prices* and the *density* of the competition — and in v10 those prices finally contain structure worth recovering.

------------------------------------------------------------------------

## Rival-Pool Layer — v10 build specification (BUILT + EVALUATED)

**Purpose (reason-1c fidelity fix).** v7/v8 folded everything private about competitors — retargeting lists, rivals' own LTV models, budget-pacing and flight schedules, exchange dynamics, competition density — into one iid noise term. That erasure made LU7 maximally DSP-predictable and removed exactly the signal SSP data exists to recover; identification theory (Athey–Haile 2002) shows that under this conditional-IPV structure a single clearing price adds no conditioning power — the structural ground of the v8 null. v10 exits IPV the way the theory prescribes: a generated rival-bid order statistic PLUS value-correlated components not reconstructable from DSP covariates. This is niche-market **fidelity, not widening** — real mobile-game inventory is genuinely contested by this rival mix.

**The five modelled mechanisms** (v7 faults → v10 remedies):

| # | Previously-missing mechanism | v10 remedy |
|---|----|----|
| a | Rivals' own retargeting lists / user-ID methods | `R[s,k]` — per (observable segment, rival) retargeting matrix (LR2) |
| b | Rivals' own LTV models | `w_k·z` — private loading on the latent impression value (LR1) |
| c | Budget-pacing state, campaign flight schedules | `p_k(d)` AR(1) pacing + `F_k(d)` flight masks (LR3) |
| d | AdEx dynamics | `π_k(e)` — per-rival exchange participation (LR4) |
| e | Competition density | endogenous N → **H9** (replaces v8's exogenous-Poisson H5) |

**Design principles (hard-learned from v8 + the pilots):**

1. **Flag-gated; all-OFF = byte-identical v7** (hash-checked, gate G1).
2. **The footprint rule** (stated at §Latent note above; empirically confirmed, pilot 2): recovery is earned only via level-footprints in SSP-observable cells — no `rival_seg` feature is handed to C3. Corollary: a v10 null at the anchor would have been a *stronger* reason-3 confirmation, not a stacked deck.
3. **Calibration backstop** — win-rate autocal + the 6 validation gates (no moment raking in the build; §Auction); gates stay green.
4. **RNG alignment** — OFF/ON differ only in LU7/H9 structure (consume-and-discard pattern).

**BN graph additions** (diagram: `Schema diagrams/Full_parameter_map_v10.svg` — v8→v10 delta):

| edge | from → to |
|---|---|
| E1 | z (latent impression value) → b_k (via private loading w_k) |
| E2 | D (day) → pacing p_k → b_k, participation |
| E3 | ad_exchange → participation π_k → composition of A_i |
| E4 | observable user segment → retargeting R → b_k |
| E5 | {b_k} → LU7; A_i → H9 |

**Derived Tier-2 features (pipeline, not generator — H6/H7 are NOT schema columns):** the v8 `hist_clearing_ssp/dsp` machinery (leak-gated OOF/LOO cell means of H4/H2) plus rolling per-cell clearing statistics — SSP-only, gated by the generator-OFF negative control. The rolling features are load-bearing: static components (w, R) are learnable from raw cell features + C3's exact loss-price labels, but the **time-varying** components (pacing, flights) can only be carried by rolling cell statistics.

**No H8 / bid-landscape column.** "Second-highest bid" was challenged and dropped (first-price environment; H4 already carries the price signal the 1c test needs; expanding the SSP information set mid-ablation would change what "SSP layer" means).

**Sensitivity arm B1 — `min_bid_to_win` (H10), BUILT 8 Jul, kept as an arm (not baseline).** The field real first-price exchanges report (Google RTB feedback): on wins it equals `max(LU7, floor)`, giving SSP owners the win-side price they otherwise lack (H4 = own bid on wins). Emitted only on WON rows (NaN elsewhere), C3/C4-only via the censoring map, and — being derivable from the rival vector v10 already generates — a pure observability change, no new generative structure and no re-anchoring; OFF = byte-identical v10. It is the **"SSP-full-feedback" upper-bound arm** (does maximal transparency reduce overpayment? — evaluated answer: no, under the model-robust bidder). Kept flag-gated on branch `t9-v11-mbw`, reported alongside the v10 headline per the KDD decision, **not folded into this baseline**; still needs Klimis sign-off as a named arm. Results: `docs/results/v11_mbw_100k_n10.json`, `v11_1m_s*.json`, `v11_10m_s*.json`; framing: `KDD_Schema_Decision_10Jul2026.md`.

**Sensitivity arm B2 — `explore_traffic` (logging policy), BUILT 8 Jul, kept as an arm.** A small share (~5%) of our own bids receives wide lognormal jitter — realistic exploration traffic (real DSPs run it). Not a schema column; a bid-policy flag. Lifts win-AUC for *all* conditions equally (+0.0046, scale-stable) and mildly *compresses* the SSP contrast (~3%, exploration partially substitutes for SSP data), while exposing an exploration tax (overpay ↑, volume ↑ more → net profit ↑). OFF = byte-identical v10; reported as the exploration robustness/mechanism arm, not baseline.

**Anchoring ρ (the operating point is external, not tuned).** The anchor is the **value of lost-price information**: the published censored-vs-full-information winning-price comparison (Wang et al. 2023, GMM vs CGMM) implies a 5–13% all-rows price-RMSE gain from observing lost prices at ~30% win rate (band centre ~8%). T9's ρ-grid maps this monotonically — **all-rows** price-RMSE gain 24.8% at ρ=0 → 3.7% at ρ=1 — giving **ρ\* ≈ 0.8**, band ρ ∈ [0.6, 0.9] (`v10_Anchor_Bands.md`). The pre-registered low-ρ expectation was falsified by this inverted mapping and the record kept. (The simulator's ~30% win rate is an instrumentation choice — enough won rows at feasible scale; real DSP win rates are typically lower, making the measured SSP value conservative.)

**Gates (all passed before headlines quoted):** G1 all-OFF byte-identity (hash) · G2 6/6 validation targets + 5/5 direction checks · G3 generator-OFF leak control for H9/rolling features · G4 multi-seed sign-consistency.

**Evaluation summary (full numbers: `v10_Training_Results.md`, `v10_Results_5Jul2026.md`):**

- **Decisive ceiling test (ρ=1):** the SSP contrast **survives the well-specified win classifier** (auc_win CLF C3−C1 ≈ +0.021 vs +0.0002 on v8 data) — v10's private structure is real signal no better model can recover without SSP data; the v8 null is **DGP-conditional** (reason 1c confirmed and repaired).
- **Anchored verdict (ρ\*=0.8, 1M n=10 + 10M n=5):** axis-split — **SSP data improves the model** (auc_win CLF +0.0137 [0.0053, 0.0220] 10/10 at 1M; +0.0182 [0.0009, 0.0355] 5/5 at 10M; price CRPS +4–5%), **but not the money** (overpay reduction is an AFT-bidder artifact — ≈0 under the model-robust classifier bidder; profit CI spans zero); **MMP makes the money** (classifier-bidder profit ≈2.1× at 1M, +19% at 10M; ev_ratio 0.63→0.97 / 0.52→0.88). Reason 3 is resolved as **axis-specific**: falsified for prediction, sustained for economics in this market configuration.

------------------------------------------------------------------------

## Deferred value-side extension — Track-C IAP bundle process (and the B3 decision)

**B3 "MMP transaction count" — built, tested, REMOVED (9 Jul 2026).** A standalone G5
`n_transactions` column (payers: `1 + Poisson(λ(LU5))`, funnel-censored like E/F/G, plus a per-app
OOF aggregate Tier-1 feature) was implemented on branch `t9-v11-mbw` and evaluated in the 4-arm
100K × 10-seed run (`docs/results/v11_b23_100k_n10.json`; price-side deltas identically zero —
perfect flag isolation). It was then **removed from the schema**, for an epistemic reason stronger
than its (scale-bound) 100K noise: unlike B1/B2, which add *no new causal structure* (B1 reveals an
already-generated quantity; B2 perturbs the logging policy), the standalone B3 required a **new
LU5 → count edge with a free coupling parameter and no external anchor** — any positive result
would have been a designed-in effect, the tuned-knob pattern the ρ-anchoring discipline exists to
prevent.

**The correct future home is the Track-C IAP bundle process** (`v10_Improvement_Proposals_7Jul2026.md`,
Track C / P2-E): replace the continuous payer-spend draw with a compound purchase process —
transaction count `n_tx` ~ a low-mean count distribution and per-transaction price tier ~
categorical over US app-store tiers ($0.99 … $99.99), `spend = Σ count_m × price_m` — raked so the
calibrated payer-spend moments (median, mean, whale share) stay in-band. In that design,
**transaction counts arise organically from the spend mechanics** (whales make repeat purchases
because that is how their spend is generated), so a G5 count column and its MMP-side aggregates
need no assumed coupling; the calibration targets come with citations (CALTV, TapTap: >95% of
payers make <5 transactions; starter-bundle spike). Guardrails recorded in the proposal doc:
flag-gated with OFF byte-identical; spend feeds `base_e`, which enters every rival bid — rake
spend moments or freeze EV inputs so the anchored price layer is not perturbed; per-tier counts
stay generator-internal unless an MMP-callback column is separately justified.

*(v11 status for the trail: branch `t9-v11-mbw` = v10 + **B1** `min_bid_to_win` (H10; kept as an SSP-full-feedback
arm) + **B2** `explore_traffic` (kept as an exploration arm); B3 removed as above. All three built + evaluated at
100K/1M/10M (8–10 Jul); per the 10 Jul KDD decision, B1/B2 are reported as sensitivity arms alongside the operative
v10 headline, not folded into the baseline — see the status block at the top of this doc and `KDD_Schema_Decision_10Jul2026.md`.)*

------------------------------------------------------------------------

## Tunable Parameters

| Group | Params |
|----|----|
| Funnel base rates (calibration knobs — *auto-calibrated*) | `base_ctr`, `base_ir`, `base_payer`, LTV `μ/σ` |
| Match strengths (effect-size knobs — *inferred*) | `w_click`, `w_install`, `w_pay` |
| Auction, v7 core (effect-size knobs) | `γ` (retained for the OFF path), `k_cpa`, `shade`, `σ_explore` |
| **Rival pool (v10)** | **K** (8) + archetype mix (3 gaming / 2 retargeter / 1 brand / 2 generalist); per-archetype `w_k`, `σ_k`; retargeting `β_R` + segment shifters `R[s,k]` (os×device); pacing `φ_p`, `σ_p` + flight windows `F_k`; participation `π_k(e)`; **ρ private-structure amplitude dial (anchored ρ\*≈0.8)**; calibration backstop = win-rate autocal + 6 gates (no moment raking) |
| Latent spreads (effect-size) | `σ_app`, `σ_cre`, `σ_game`, interest (LU6) & app-audience (LA2) Dirichlet concentrations |
| Categories / mixes | `app_categories` (B2), `ad_genre` mix (C4), `ad_exchanges` (B3), `slot_format` mix (B6), `slot_sizes` (B4×B5), `advertiser_scale` |
| BN dependency layer (v7, active) | unchanged from v7 (see v7 schema) |

------------------------------------------------------------------------

## Validation Targets

Unchanged from v7 (funnel metrics computed on the **master**; all six gates re-verified with `rival_pool` ON):

| Metric                     | Target                              |
|----------------------------|-------------------------------------|
| Population CTR (mean E1)   | 2–5%                                |
| Click→install (F1\|E1)     | 25–40%                              |
| Install→payer (G1\|F1)     | 2–5%                                |
| Whale concentration        | top 5% payers → ~55–65% of total G2 |
| Median payer spend         | ~$6                                 |
| Auction win rate (mean H3) | 20–40% (calibrated-pinned at 30%)   |

------------------------------------------------------------------------

*v10 spec assembled 7 Jul 2026 from `v10_Proposal_5Jul2026.md` (approved design) + the built implementation (`t9_sim/src/t9sim/rival_pool.py`, branch `t9-v10-rival`). Core v7 sections referenced rather than duplicated where unchanged. v8 layer: superseded (structural null, trail in the v8 schema doc). v9: abandoned unbuilt.*
