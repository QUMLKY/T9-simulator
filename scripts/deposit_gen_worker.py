"""Generate ONE 10M V10 anchored master (rho* = 0.8) as deposited under the DOI.

Generation only. Unlike v10_10m_worker.py this does not run the training
pipeline: the deposited artifact is the DATA, and the per-seed metrics are
already in docs/results/v10_10m_s*.json. Skipping the pipeline removes most of
the wall clock and all of its memory pressure.

Writes the four parquets that make up one dataset (auctions plus the three
entity pools) into output/v10_10m_s{seed % 100}.

Usage: python scripts/deposit_gen_worker.py <seed>       # seeds 90213-90222
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

import t9sim.simulate as sim          # noqa: E402

seed = int(sys.argv[1])

# the paper's operative configuration: the v7 BN edges plus the v10 rival market
V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
tag = f"v10_10m_s{seed % 100}"

sim.run("scale10m", out_name=tag, seed=seed, rho=0.8, bn_edges=EDGES)
print(f"GEN DONE seed={seed} -> {tag}", flush=True)
