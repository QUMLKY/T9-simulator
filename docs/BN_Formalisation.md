# T9 — Bayesian-Network Formalisation of the Simulator (7 Jul 2026)

**Purpose.** The mathematical definition of the T9 generative model (Dr Klimis's formalisation ask):
variables, the DAG, the joint factorization, the four experimental conditions as censoring operators,
the estimands, and the identification scaffold. This document is the **source of truth**; the KDD
paper's "Formal model" section and dissertation §5 are condensations of it. Operative build detail
(flags, knobs, calibration) stays in the schema docs (`Simulator_Schema - June 10 (v7).md` core,
`(v10).md` rival layer); schema column codes are given in parentheses throughout.

------------------------------------------------------------------------

## 1. Variable sets and notation

**Index sets.** Users `u ∈ U`, apps `a ∈ 𝒜`, campaigns `c ∈ 𝒞`, rival archetypes `k ∈ 𝒦` (|𝒦| = 8),
auctions `i ∈ {1..M}`, days `d ∈ {1..28}`, hours `h ∈ {0..23}`. Genre taxonomy
`G = {casual, strategy, rpg, hypercasual}`.

**User pool.** Each user carries a latent vector (never observed in any view)

    θ_u = ( κ_u,  q_u^clk,  q_u^ins,  q_u^pay,  m_u,  ι_u )          (LU1–LU6)

— archetype `κ_u ∈ {whale, engaged, casual, time_filler, inactive} ~ Cat(π)`; propensities
`q^· ~ Beta(·|κ_u)`; LTV multiplier `m_u ~ LogNormal(·|κ_u)`; interest vector
`ι_u ~ Dirichlet(·|κ_u) ∈ Δ^{|G|}` — and observables `x_u` (A1–A5: region, city, os, os_version,
device), with `p(x_u | κ_u)` given by the marginal-preserving CPT tilts (the os tilt — part of edge
#3's wiring — and the device tilt; §6). The archetype→hour tilt (edge #5) acts on per-auction
context, not on `x_u` (§2).

**App pool.** Observables `(id_a, g_a)` (B1–B2); latents `λ_a ~ LogNormal(0, σ_app)` (LA1
app-quality) and audience profile `α_a ~ Dirichlet(k_aud · c[g_a]) ∈ Δ^5` (LA2).

**Campaign pool.** Observables `x_c` (C1–C4: advertiser, scale, campaign id, ad-genre `g_c`);
latents `η_c ~ LogNormal(0, σ_cre)` (LC1 creative appeal), `χ_c ~ LogNormal(0, σ_game)` (LC2 game
quality).

**Rival pool (v10; latent family LR1–LR4, observed by no view).** Each archetype `k` carries

    ψ_k = ( w_k,  R_{·,k},  {F_k(d)}_d,  π_k(·),  σ_k )        plus the pacing chain {p_k(d)}_d

— a private valuation **loading** `w_k` on the latent impression-value score `z_i` (§3). *Build
note:* the proposal specified a user-latent feature map `φ(LU) = (z, m_u, archetype)` with
non-gaming archetypes confined to a general-value slice; the built model implements the **scalar**
`z` with non-gaming archetypes as *smaller loadings on the same z* — a declared simplification.
Retargeting **level-shifters** `R_{s,k} ~ N(0,1)` (scaled by `β_R`) per **observable** user segment
`s(u) = (os × device_type)`, persistent over the window; flight indicators `F_k(d) ∈ {0,1}` and the
day-level budget-pacing state `p_k(d) = φ_p · p_k(d−1) + σ_p · η_{kd}` (AR(1); kept outside `ψ_k`
in the factorization, §2); **per-exchange** participation propensities `π_k(e)` — time enters
participation at day level only, via flights and pacing; bid dispersion `σ_k`. Archetype `k = 0`
is forced **always-on** (guaranteeing `A_i ≠ ∅`); in the current build that slot is the first
**gaming-DSP archetype** (strongly value-loaded), whereas the proposal specified a value-neutral
generalist — a declared build divergence, flagged for review.

**Per-auction variables.** Auction `i` samples `(u_i, a_i, c_i)` from the pools (§2, edges #1/#2)
and draws context `ξ_i = (e_i, slot_i, τ_i)` (B3–B6 exchange and slot; D1–D3 time,
`τ_i = (d_i, h_i)` with day-of-week `δ_i = dow(d_i)`, D3). Outcomes: clicks/installs
`(y_i^clk, y_i^ins)` (E1, F1) with timestamps (E2, F2), payer flag and spend `(y_i^pay, y_i^spd)`
(G1, G2; G3/G4 fixed fractions), floor `ℓ_i` (H1), own bid `b_i^own` (H2), win `y_i^win` (H3),
clearing `y_i^clr` (H4), rival count `N_i` (H9), and the top competing bid `V_i` (LU7, latent; §3).

------------------------------------------------------------------------

## 2. The joint factorization

Pools are drawn once; auction rows are conditionally independent given the pools **and the rival
day-state** — the only cross-auction dependence is the AR(1) pacing chain and the flight calendar
(plus rolling *pipeline* features, which are estimator-side, not part of the DGP). The joint law:

    p( pools, auctions ) =
        ∏_u p(θ_u) p(x_u | κ_u)                                   — user pool (CPT tilts)
      · ∏_a p(λ_a) p(α_a | g_a)                                   — app pool
      · ∏_c p(η_c) p(χ_c) p(x_c)                                  — campaign pool
      · ∏_k [ p(ψ_k) · ∏_d p(p_k(d) | p_k(d−1)) ]                 — rival pool + pacing chains (v10)
      · ∏_i [ p(u_i, a_i | Π^pair) · p(c_i | u_i; Π^exp) · p(ξ_i | κ_{u_i})
              · p(funnel_i | θ_{u_i}, λ_{a_i}, η_{c_i}, χ_{c_i}, a_i, c_i, ξ_i)          — §2.1
              · p(auction_i | z_i, x_{u_i}, ψ_·, p_·(d_i), ξ_i) ]                        — §3

**Assignment mechanisms (BN edges #1/#2, marginal-preserving).**
`Π^pair` is the archetype×app coupling: seed `M_{κ,a} = π_κ · pop_a · (1 + s·(α_a[κ]/π_κ − 1))`,
IPF-raked to row sums `π` and column sums `pop`; `(κ_i, a_i)` drawn jointly from `M` (edge #1).
`Π^exp` is value-optimised exposure: `p(c_i | u_i) ∝ budget_c · exp(β_vo · g_u · z(log χ_c))`,
with user value score `g_u = z(log(q_u^pay · m_u))`, IPF-raked to preserve `p(campaign)` and the
ad-genre mix (edge #2). `p(ξ_i | κ_u)` carries the archetype→hour von-Mises tilt (edge #5).

### 2.1 Funnel CPDs and estimands

Relevance `r_i = ι_{u_i}[g_{c_i}]`; stage multipliers `m_st(r) = (1 − w_st) + w_st · r`; slot
quality `v_i = fmt_wt(B6) · size_wt(B4,B5)`. The funnel is a chain of conditional Bernoullis and a
gated spend draw, **generated on every row** (lost-row outcomes are the declared counterfactual —
no funnel formula reads `y^win`):

    ν_i^clk = β_clk · q_{u_i}^clk · v_i · m_clk(r_i) · λ_{a_i} · η_{c_i}          y^clk ~ Bern(ν^clk)
    ν_i^ins = β_ins · q_{u_i}^ins · ease(g_{a_i}) · m_ins(r_i) · λ_{a_i}          y^ins | y^clk=1 ~ Bern(ν^ins)
    ν_i^pay = β_pay · q_{u_i}^pay · m_pay(r_i) · t_pay(h_i, δ_i)                  y^pay | y^ins=1 ~ Bern(ν^pay)   (edge #4)
    y_i^spd | y^pay=1  ~  LogNormal( μ'_{g_a} , σ_ltv ) · m_{u_i} · χ_{c_i} · plat(os_u)          (edge #3)

The **estimands** are the deterministic pre-draw quantities (master-only columns):

    ν_i = ( ν^clk, ν^ins, ν^pay, ē_i ),   ē_i = E[y^spd | pay] ,
    EV_i = ν^clk · ν^ins · ν^pay · ē_i                                   (ev_truth)

Tier-1's four heads estimate the four factors per condition; the zero-inflated gate×lognormal
structure of the spend branch matches the ZILN generative form in the CLTV literature (OptDist).

------------------------------------------------------------------------

## 3. The auction layer (v10 Private-Rival Market)

**Participation** (who shows up; `gate(·)` maps pacing state to participation propensity; the
product is clipped at 1):

    Z_ik ~ Bern( min(1, π_k(e_i) · F_k(d_i) · gate(p_k(d_i))) ),    A_i = {k : Z_ik = 1},    N_i = |A_i|   (H9)

**Rival bids** (shared value core + ρ-scaled private structure + idiosyncratic noise):

    log B_ik = log β_0(i) + ρ · ( w_k · z_i + β_R · R_{s(u_i),k} + p_k(d_i) ) + σ_k · ε_ik ,   ε iid N(0,1)

where `z_i = z(log EV_i)` is the standardized log of the **latent impression value** (§2.1 — a
function of user, app *and* campaign latents), and `β_0(i)` = format-level eCPM base (iPinYou
clearing shape, median-normalised, rescaled) — the shared, DSP-predictable core.

**Variance control (declared limitation of the build).** The built model does **not** rake log-bid
moments to v7: `ρ` multiplies the structured components directly, so raising `ρ` **inflates total
price variance** rather than reallocating it — a fact the anchor mapping depends on (P8; the
`rival_pool` module documents the win-rate autocal `k_global` + the six validation gates as the
explicit calibration backstop). `ρ ∈ [0,1]` is therefore the **private-structure amplitude dial**;
the proposal's variance-share reading is design intent, not an implemented invariant. The operating
point is externally anchored at **ρ* ≈ 0.8** (§7, P8).

**Top competing bid, win, clearing:**

    V_i = max_{k ∈ A_i} B_ik                                   (LU7; A_i ≠ ∅ by the always-on archetype k=0)
    b_i^own = π^bid( visible features of i ) · shade · exp(σ_expl · ζ_i)          (H2; policy — see below)
    y_i^win = 1[ b_i^own ≥ max(V_i, ℓ_i) ]                     (H3)
    y_i^clr = b_i^own · y_i^win + V_i · (1 − y_i^win) · 1[V_i ≥ ℓ_i]   ;   NaN if max(b^own, V) < ℓ    (H4, first-price)

**Policy measurability (formal property of H2).** `π^bid` is a function of *visible* columns only —
`σ(x_u, x_a, x_c, ξ_i)`-measurable, never of any latent. Hence `b^own ⊥ V | visibles`, which makes
the win-curve estimation a single-agent problem (V is exogenous to the bid) and makes the
exploration noise `ζ` the source of the bid variation Tier-2 needs.

**v7 special case.** With `rival_pool` OFF the layer reduces to v7's two-term form
`V = max(β_0·e^{γ·z(log EV)+ε_g}, β_0·e^{ε_x})` — a conditionally-IPV market (§7, P6). ρ = 0 keeps
the v10 mechanics (order statistic, endogenous participation) with zero private-structure share.

------------------------------------------------------------------------

## 4. The four conditions as censoring operators

Let the master row be `W_i = (x_i^vis, Y_i, y_i^win)` with visible features
`x_i^vis = (x_u, x_a, x_c, ξ_i, ℓ_i, b_i^own)` and label block
`Y_i = (E_i, F_i, G_i, y_i^clr, N_i)`. Latents `(θ, λ, α, η, χ, ψ, V, ν, EV)` appear in **no** view.
Each condition `C ∈ {C1, C2, C3, C4}` is an **observation map** `O_C` applied row-wise:

    O_C(W_i) = ( x_i^vis,  y_i^win,
                 E_i, F_i, G_i   if  MMP_C ∨ y_i^win = 1   else NaN,      — funnel mask
                 y_i^clr, N_i    if  SSP_C                 else NaN )      — price mask

with `MMP_C = 1` for C2/C4 and `SSP_C = 1` for C3/C4. All four views share identical rows and the
temporal split (applied before censoring); views are masks, never copies. Two formal remarks:

- **Censoring ≠ latency.** `O_C` withholds *observable* labels per condition/row (the experimental
  manipulation); latents are excluded from every view by construction (a design invariant).
- **The funnel mask is outcome-dependent (MNAR) — the "DSP's biased view".** Missingness of
  `E/F/G` in C1/C3 depends on `y^win`, and `y^win` depends on `V`, which is increasing in `EV`
  (through `γ` in v7; through the value-loaded rival bids `w_k · z_i` in v10). Hence selection is
  informative:

      **P4 (selection bias).**  If the market prices value (γ > 0, or ρ > 0 with value-loading w),
      then E[EV | y^win = 1] < E[EV] — the DSP systematically wins low-value and loses high-value
      impressions, so C1/C3's funnel training population is value-biased downward.

  MMP ownership (C2/C4) restores the complete-data likelihood on the funnel; this is what the
  C1→C2 lift measures (level face: `ev_ratio`; it is a selection effect, hence not correctable by
  more C1 data — the empirically durable ev_ratio gap).

### 4.1 Sensitivity arms (B1, B2) — formal modifications, not part of the operative model

The operative model above is **v10**; per the KDD decision (`KDD_Schema_Decision_10Jul2026.md`)
two flag-gated arms are reported *alongside* it, each a minimal, local modification (OFF = the model
above, byte-identical):

- **B1 (`min_bid_to_win`, "SSP-full-feedback").** Adds one observable `M_i = max(V_i, ℓ_i)` on **won**
  rows (`y^win = 1`), NaN elsewhere — a deterministic function of already-generated quantities, no new
  latent. It enters only the **observation operator** `O_C` (§4): the price mask gains the term
  `M_i · 1[SSP_C ∧ y^win = 1]`. Effect on the estimand: it **tightens the win-row censoring** of the
  Tier-2 target — a won auction with `M_i > ℓ_i` upgrades from the interval `V_i ∈ [0, b^own_i)` to the
  **point** `V_i = M_i` (§5's current-status observation becomes exact on those rows). It is the upper
  bound on SSP price transparency; it does not touch the generative law `p(pools, auctions)`.

- **B2 (`explore_traffic`).** Perturbs only the **bid policy** `π^bid` (§3): with probability ε a
  row's own bid is scaled by a wide multiplicative jitter, `b^own ← b^own · ξ`, `ξ ~ LogNormal(0, σ_w)`.
  It changes neither the competing bid `V`, the funnel, nor any latent — it enriches the conditional
  bid variation that identifies `W(·;x)` (P5), so it raises win-curve quality in **all** conditions
  (and, because it helps C1 too, marginally compresses the C3−C1 contrast). Formally it widens the
  support of the logging policy while preserving `b^own ⊥ V | visibles`.

Both are excluded from the core factorization and `O_C` above; this subsection is the formal
statement the paper's sensitivity section condenses.

------------------------------------------------------------------------

## 5. Estimands and estimators

**Tier-1.** For each condition, four XGBoost heads estimate the funnel factors from `O_C`-visible
data: `f_C^clk ≈ ν^clk`, `f_C^ins ≈ ν^ins`, `f_C^pay ≈ ν^pay`, `f_C^spd ≈ ē`, composed into
`EV̂_C = f^clk · f^ins · f^pay · f^spd`. Reported against `EV` (ev_truth):
`ev_ratio = E[EV̂]/E[EV]` (level), `ev_spearman = ρ_rank(EV̂, EV)` (ranking).

**Tier-2 estimand.** The structural win-curve

    W(b; x) = P( V < b | x )      ⇒      P(win | b, x, ℓ) = 1[b ≥ ℓ] · W(b; x)

**Observation structure (current-status).** In every condition, each row is a current-status
observation of `W` at the operating bid: `y^win = 1 ⇒ V < b^own` (left-censored),
`y^win = 0, b^own > ℓ ⇒ V > b^own` (right-censored); floor-caused losses (`b^own ≤ ℓ`) are
uninformative about `V`. C3/C4 additionally observe **point values** `V_i = y_i^clr` on sold losses.
Two heads estimate `W` per condition: the AFT (lognormal, on `V` with the per-view likelihood) and
the first-class binary classifier with `b` as a swept input (in C1/C2 the AFT likelihood
algebraically degenerates to the classifier's — they must agree there, a built-in check).

**Bidder (decision layer).** `b*_i = argmax_b (EV̂_i − b) · Ŵ(b; x_i)` over a bid grid — computed
with both heads' `Ŵ` (AFT-curve and CLF-curve bidders) so all economic metrics are model-robust.

**The ablation, formally.** For any **single-axis** metric `m` — one that reads only funnel/value
outputs, or only win/price outputs — the layer values are the paired contrasts
`Δ_MMP = m(C2) − m(C1) ≡ m(C4) − m(C3)` and `Δ_SSP = m(C3) − m(C1) ≡ m(C4) − m(C2)`. The design
identities hold because `O_C` factorizes into independent funnel and price masks: MMP changes only
funnel-label completeness (so C1 ≡ C3, C2 ≡ C4 on funnel/value metrics), SSP only price
observability (so C1 ≡ C2, C3 ≡ C4 on win/price metrics). For **mixed** metrics that compose both
axes (profit, ROAS) the two paired contrasts need not coincide and are reported separately.

------------------------------------------------------------------------

## 6. Marginal-preservation propositions (the BN edges never move a calibrated marginal)

**P1 (CPT tilt).** For a categorical observable with calibrated marginal `m_v` and archetype shares
`π_κ`: any tilt `P(v|κ) = m_v + δ_{κ,v}` with `Σ_v δ_{κ,v} = 0` and `Σ_κ π_κ δ_{κ,v} = 0` preserves
the marginal exactly; the IPF construction (seed `m_v·exp(t_{κ,v})`, rake to row sums 1 and
π-weighted column sums `m_v`) produces such a tilt. (Verified < 1e−12 in `bn_cpts.yaml`.)

**P2 (bi-marginal coupling).** The IPF-raked pairing matrix `M` (edge #1) lies in the
transportation polytope with marginals `π` (archetypes) and `pop` (app visits): both calibrated
marginals are preserved while `P(κ | a)` becomes non-uniform — `app_id` turns into an archetype
proxy without moving any marginal. Same construction family for edge #2 (preserving `p(campaign)`
and the genre mix).

**P3 (v10 variance control — corrected statement).** The *proposal* specified variance-preserving
raking (private components reallocate log-bid variance, never add it). The **built** model
implements no raking: `ρ` multiplies the structured components directly, so total log-bid variance
grows with `ρ`. Calibration invariance is enforced **empirically**, not by construction — the
win-rate autocal (`k_global`) re-solves the global bid level every run and the six validation
gates must pass. Formal statements must therefore treat `ρ` as an **amplitude** dial; the
variance-share interpretation is design intent (approximately valid only at small `ρ`), and the
variance inflation at high `ρ` is precisely what shapes the inverted anchor mapping (P8).

------------------------------------------------------------------------

## 7. Identification scaffold (why v8 was null and v10 is the correct exit)

**P5 (win-curve identification in C1/C2).** From current-status data `(b^own, y^win)` alone, `W(·;x)`
is identified (parametrically; nonparametrically on the bid support) wherever the logging policy
has conditional bid variation — supplied by the exploration noise `ζ`. The SSP's point
observations buy statistical *efficiency* and tail *extrapolation*, not new identifiability.
(Current-status/interval-censoring; the C1/C2 AFT⇄classifier degeneracy of §5.)

**P6 (conditional-IPV sufficiency — the structural ground of the v8 null).** v7/v8's competing bid
has the form `log V = g0(·) + ε`, where the shared mean `g0 = γ·z(log EV)` is a function of the
**latent** impression value whose variation is largely *reconstructable from DSP-visible
covariates through the BN couplings* (feature-Bayes 0.880 vs value-oracle 0.904 — a small,
declared gap), and `ε` is idiosyncratic noise. This is the structural **analogue** of
Athey–Haile's conditionally-independent-private-values form (`U = g0(W0) + A`; Thm 3/6,
Econometrica 2002), not a literal identity. Under that structure, a single transaction price is a
sufficient statistic: extra order statistics, rival counts, and price labels add no conditioning
power beyond what the covariates already carry. Hence the v8 feature layer *had* to be null — its
price multipliers were covariate-measurable (helping C1 as much as C3), and its residual was
unpredictable by anyone. The null is a property of the *market model*, not of SSP data.

**P7 (the v10 exit + the footprint rule).** v10 leaves IPV exactly as the theory prescribes:
(i) `V` becomes a max over a generated rival-bid vector, and (ii) bids carry components correlated
with the user but not `x`-measurable. The components split by recoverability (empirically
confirmed, pilot 2):

- **level components** (retargeting `R` over observable os×device segments; pacing over days;
  participation over exchanges, day-gated by flights/pacing) leave conditional-mean footprints in
  SSP-observable cells → this is the SSP-recoverable signal (earned via H4 point labels, H9, and
  rolling cell statistics);
- **the latent-value interaction** `w_k · z_i` is recoverable by *no* view (it prices a latent
  impression value nobody observes) → it supplies realism and DSP-unpredictability, not
  SSP-recoverable signal.

Corollary: a v10 null at the anchored point would have been a *stronger* reason-3 confirmation,
not a stacked deck.

**P8 (anchoring — ρ is externally pinned, not tuned).** ρ is identified against the published
**value of lost-price information**: censored-vs-full-information winning-price estimation (Wang
et al. 2023, GMM vs CGMM) implies a 5–13% all-rows price-RMSE gain at ~30% win rate; T9's ρ-grid
maps this gain monotonically (24.8% at ρ=0 → 3.7% at ρ=1), giving ρ* ≈ 0.8 (band [0.6, 0.9]).

**Empirical closure (for completeness; numbers in `v10_Training_Results.md`).** At ρ=1 the SSP
contrast survives the well-specified classifier (+0.021 vs v8's +0.0002) — P6 confirmed as the
null's mechanism, 1c repaired. At ρ*: SSP improves the model (win-AUC, CRPS; scale-stable 1M→10M)
but not the money (P4's value-side gate binds: profit is limited by EV ranking, where MMP
dominates) — the axis-split verdict.

------------------------------------------------------------------------

## 8. DAG (reading view)

Plate-notation DAG: **`Schema diagrams/T9_BN_Formal_DAG.svg`** (generated by
`t9_simulator/diagrams/make_bn_formal_dag.py`; project diagram style). Pools (user, app, campaign,
rival + pacing chain) → assignment mechanisms (#1/#2) → funnel chain with estimand `EV` → rival
bids `B_ik` → `V, N` → `(y^win, h)` — with censoring badges marking what each condition observes
(funnel labels: all rows C2/C4, won rows C1/C3; `h, N`: C3/C4 only). Parameter-level maps:
`Full_parameter_map_v10.svg` (v8→v10 delta), `BN_dependency_DAG.svg` (funnel edges #1–#5).

------------------------------------------------------------------------

*Assembled 7 Jul 2026 (§4.1 sensitivity arms added 10 Jul per the KDD decision) from the operative schema (v7 core + v10 rival layer), `v10_Proposal_5Jul2026.md`
§3, and the identification deep-reads (`DeepRead_WinningPrice_Top5.md`, `DeepRead_Auction5_WinningPrice.md`).
References: Athey & Haile (Econometrica 2002); Zhou et al. (KDD 2021); Pan et al. (AdKDD 2020);
Wang et al. (KBS 2023); Ma et al. ESMM (SIGIR 2018) and OptDist (CIKM 2024) for the funnel/value side.*
