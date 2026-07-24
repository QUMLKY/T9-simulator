# T9Sim RTB Data Simulator: Generating, Censoring and Benchmarking DSP/MMP/SSP Adtech Data Layers

T9 is a calibrated simulator of mobile-game real-time-bidding (RTB) auctions **with retained
ground truth**, and a benchmark built on it: **the data stay fixed, the training method varies,
and every method is scored against the truth** — including on the auctions the bidder *lost*,
whose outcomes no real log contains.

Every auction carries its full counterfactual: funnel outcomes (click → install → payer → 90-day
spend) are generated on **all** rows, won or lost, and the top competing bid is retained. Four
**censoring operators** then reduce this one master table to the views real platforms actually
hold:

| View | Sees | Real-world analogue |
|---|---|---|
| **C1** | own wins' funnel only, no clearing prices | a DSP alone (the "biased view") |
| **C2** | + funnel labels on *all* rows | DSP + MMP (attribution partner) |
| **C3** | + clearing prices & rival count | DSP + SSP (supply-side) |
| **C4** | all layers | fully integrated stack |

Because the truth is retained, T9 can issue verdicts that are **uncomputable on any real
dataset** — e.g. whether a censoring correction actually recovers the lost-row counterfactual,
or merely looks good on observable data.

## Three verdicts (10M auctions × 10 seeds; `docs/Method_Benchmark_10M_Results_13Jul2026.md`)

1. **Model quality** — a structured value model (two-tier XGBoost) beats a linear floor on
   ranking by a wide, tight margin: install AUC **+0.131 [0.128, 0.134]**, EV rank-correlation
   **+0.497 [0.452, 0.541]**, 10/10 seeds.
2. **Bias correction** — cross-fitted inverse-propensity weighting (the textbook fix for
   won-rows-only labels) **improves the aggregate bias it targets but degrades ranking**
   (install AUC −0.017, 0/10), closes *less* of the floor→oracle gap than the uncorrected
   model, and recovers no lost-row truth — because the selection operates on **latent** value
   that no observable propensity can capture. Robust to the weight cap (swept 10→uncapped).
3. **Price censoring** — a censoring-aware (AFT) price model recovers lost-row price
   distributions **+41.6% better (CRPS, 10/10)** than a naive fit that *looks superior on its
   own observable metric* (the metric illusion).

Headline ablation (schema V10, anchored ρ\*=0.8; n=10 seeds at 1M **and** 10M): **SSP data
improves the model** (win-AUC +0.014, CI excludes 0, 10/10 at both scales) **but does not
demonstrably convert to money; MMP data drives the economics** (classifier-bidder profit +39%
at 10M; EV-bias 0.52→0.89). Full tables: `docs/v10_Training_Results.md`.

## Install

```bash
python -m venv venv && venv/Scripts/activate    # or source venv/bin/activate
pip install -e .
pytest                                          # 35 tests, ~1 min
```

Python ≥ 3.11. Dependencies are pinned in `requirements.txt` (pandas, numpy, scipy, pyarrow,
xgboost, scikit-learn, pyyaml).

## Quickstart (~3 min)

```bash
python examples/quickstart_100k.py
```

generates a 100K master under the paper's operative configuration (schema **V10**, the
private-rival market, at the externally anchored privateness ρ\*=0.8), trains the reference
stack under C1–C4, and prints the headline metrics. The same loop in code:

```python
import t9sim.api as t9

t9.generate("test", seed=90213, rho=t9.RHO_STAR, bn_edges=t9.V10_EDGES)  # 1M master
results = t9.evaluate("test")                    # reference models, scored vs truth
```

To benchmark **your own method**: load the master parquet, take a censored view with
`t9.view(df, "C1")`, train on what the view exposes, and score your predictions against the
master's retained truth columns (`p_click`, `p_install`, `p_payer`, `e_ltv`, `ev_truth`,
`lu7_competing_bid`) on the all / won / **lost** row slices. The reference entrants
(linear floor, XGBoost, cross-fitted IPW, naive price, AFT, oracle) live in
`scripts/method_bench_worker.py`.

## Reproduce the paper numbers

| Result | Command |
|---|---|
| 1M ablation, n=10 seeds | `python scripts/v10_anchor_5seed.py` then `scripts/v10_anchor_n10.py` |
| 10M ablation, n=10 (one fresh process/seed, ~16 GB) | `python scripts/v10_10m_driver.py` |
| Method benchmark, 10M × n=10 | `python scripts/method_bench_driver.py` |
| IPW cap-robustness sweep | `python scripts/ipw_cap_sweep.py` |
| Leakage negative control | `python scripts/neg_control_generator_off.py` |
| Rebuild the deposited 10M datasets (~40 min, ~18 GB) | `python scripts/deposit_gen_driver.py` |

Per-seed result JSONs for the shipped write-ups are under `docs/results/`.

## Data & reproducibility

Generation is **seed-deterministic**: the same (profile, seed, ρ, edges) reproduces the dataset
byte-for-byte (verified via the content fingerprint in `src/t9sim/fingerprint.py`; the flag-OFF
configuration reproduces the v7 baseline fingerprint `0xdf0ac3e18624cf2b`). The paper uses seeds
**90213–90222**.

The paper's operative configuration is declared in `config/benchmarks.yaml` — the privateness
dial `rho`, the `bn.rival_pool` edge flag, and the rival-pool parameters (`K`, `n_gaming`,
`beta_R`, `pacing_ar`, `pacing_sigma`) — so the run is reproducible from configuration alone.
`CHECKSUMS.txt` pins the SHA-256 of every calibration target and config file that fixes the
output; regenerate it with `python scripts/make_checksums.py`.

Generated data therefore ships as *recipes* (profile + seed + config), not binaries; frozen
dataset archives with checksums and a DOI will accompany the camera-ready release (Zenodo).

The ten 10M masters behind the headline results (seeds 90213–90222, ~17 GB) can be rebuilt
locally rather than downloaded:

```bash
python scripts/deposit_gen_driver.py                              # ~40 min, resumable
python scripts/make_deposit_manifest.py output/v10_10m_s* --fingerprint
```

`DEPOSIT_MANIFEST.txt` records what was deposited: MD5 and SHA-256 per archive, row counts per
file, and a **content fingerprint** per dataset. Compare fingerprints. Regeneration on the same
platform and library versions does reproduce the files byte-for-byte — verified, SHA-256 matched
on all forty files — but parquet stores the writer version, so a different pyarrow release can
change the bytes while the data is identical. The fingerprint hashes column values and survives
that.

Calibration targets in `calibration/` are **aggregated distribution shapes** derived from the
public iPinYou dataset (Zhang et al., 2014), rescaled to a 2025 US mobile-gaming market - no
raw iPinYou records are included. The derivation is fully reproducible:
`scripts/calibrate_ipinyou.py` regenerates every CSV in `calibration/` from the raw iPinYou
season 2+3 logs (download separately; ~28 GB).

## Repository layout

```
src/t9sim/        the package: generator (auctions.py), censoring (censor.py),
                  training/eval pipeline (pipeline.py), validation, fingerprint, api
config/           all tunable parameters (YAML; every value provenance-tagged)
calibration/      iPinYou-derived distribution shapes (CSV)
docs/             the specification (V10) + implementation status, results write-ups
docs/results/     per-seed and aggregated result JSONs backing the write-ups
scripts/          paper runners: ablation (1M/10M), method benchmark, sensitivity sweeps
tests/            35 tests incl. leakage gates and schema contracts
diagrams/         schema maps (overview, generation, dependencies, rival prices),
                  the generator and dependency-graph figures, per-feature calculation SVGs
examples/         quickstart
```

## Formal model

The generative model is specified in `docs/T9Sim_Specification_v10.md` — the single source of
truth for the schema:

- **Part I — the data.** Every variable with its domain and parents, which layer observes it,
  and the C1–C4 censoring map (column- *and* row-conditional), reconciled against the emitted
  parquet.
- **Part II — the generator.** The four entity pools, the joint factorisation, the dependency
  graph over the anchored variables, each conditional law with its clips, and the rival-pool
  (private-price) layer.
- **Part III — identification.** Propositions **P1–P4**: what each data layer does and does not
  identify, and the two-dial result (value-awareness flat, privateness ρ responsive).

Its companion `docs/T9Sim_Implementation_Status.md` carries the build-side detail kept out of
the specification: edge wiring status, declared code-vs-spec divergences, tunables, validation
targets, and the audit trail.

A **datasheet** for the dataset (motivation, composition, collection, uses and their limits,
distribution, maintenance) is in `DATASHEET.md`.

## Status, license, citation

Private preview for collaborators, ahead of public release.

**Code** (the `t9sim` package, scripts, tests, examples, config) is **MIT** — see `LICENSE`.
**Data** (calibration tables, result JSONs, and the deposited datasets) is **CC BY 4.0** — see
`LICENSE-DATA`. Cite via `CITATION.cff` (paper reference and Zenodo DOI to follow).
