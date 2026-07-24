"""Generate the 1M sample dataset for the deposit.

The ten 10M masters come from deposit_gen_driver.py. The sample is a separate
call because it uses the smaller "test" profile, matching how the 1M anchored
point was produced for the paper (v10_anchor_n10.py): same operative
configuration, rho* = 0.8 with the v10 edge set, one tenth the auctions.

Included in the deposit so a reader can inspect the schema and run a quick
experiment without downloading 17 GB.

Usage: python scripts/deposit_gen_sample.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

import t9sim.simulate as sim          # noqa: E402

V7_BN = {e: True for e in ("pairing", "os_spend", "payer_timing", "hour", "exposure")}
EDGES = {**V7_BN, "rival_pool": True}
SEED = 90213
TAG = f"v10_anchor_s{SEED % 100}"

sim.run("test", out_name=TAG, seed=SEED, rho=0.8, bn_edges=EDGES)
print(f"GEN DONE seed={SEED} -> {TAG}", flush=True)
