"""Regenerate the ten 10M V10 anchored masters (seeds 90213-90222) deposited under the DOI.

Use this to rebuild the archived datasets from source rather than downloading
them, or to verify that a download matches. Pair it with

    python scripts/make_deposit_manifest.py output/v10_10m_s* --fingerprint

and compare the content fingerprints against the deposit manifest. The
fingerprints, not the file hashes, are the meaningful comparison: parquet embeds
non-deterministic metadata, so an identical regeneration produces different
bytes.

Roughly 4 minutes and 1.8 GB per seed on a modern laptop, so about 40 minutes
and 18 GB for the full set.

One fresh process per seed (the scale10m convention: avoids cross-seed memory
accumulation). Resumable: a seed whose four parquets are present, readable and
full length is skipped, so an interrupted run can simply be restarted. Stops
early if free disk falls below a safety floor rather than filling the volume.

Usage: python scripts/deposit_gen_driver.py
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from t9sim.paths import OUTPUT_DIR          # noqa: E402

PY = sys.executable
WORKER = str(Path(__file__).resolve().parent / "deposit_gen_worker.py")
SEEDS = list(range(90213, 90223))
EXPECTED = ["auctions.parquet", "pool_users.parquet",
            "pool_apps.parquet", "pool_campaigns.parquet"]
N_AUCTIONS = 10_000_000      # scale10m profile; a short file means a killed run
MIN_FREE_GB = 8.0
EST_PER_SEED_GB = 1.8


def complete(tag: str) -> bool:
    """True only if every parquet is present, READABLE and full length.

    Existence is not enough. A run killed mid-write leaves a plausible-looking
    auctions.parquet of several hundred MB that pyarrow cannot open, and a
    resume that trusted existence would skip it and deposit corrupt data.
    Reading the footer metadata is cheap and catches truncation.
    """
    d = OUTPUT_DIR / tag
    for f in EXPECTED:
        p = d / f
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            rows = pq.ParquetFile(p).metadata.num_rows
        except Exception:
            return False                      # truncated / unreadable footer
        if f == "auctions.parquet" and rows != N_AUCTIONS:
            return False
    return True


def free_gb() -> float:
    return shutil.disk_usage(OUTPUT_DIR).free / 1e9


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = [], [], []
    t_all = time.time()
    print(f"free disk at start: {free_gb():.1f} GB", flush=True)

    for seed in SEEDS:
        tag = f"v10_10m_s{seed % 100}"
        if complete(tag):
            print(f"=== {tag}: already on disk, skipping ===", flush=True)
            skipped.append(tag)
            continue

        if free_gb() - EST_PER_SEED_GB < MIN_FREE_GB:
            print(f"!!! stopping: only {free_gb():.1f} GB free, "
                  f"need {EST_PER_SEED_GB:.1f} GB + {MIN_FREE_GB:.1f} GB floor", flush=True)
            break

        t0 = time.time()
        print(f"=== {tag} (seed {seed}) starting, {free_gb():.1f} GB free ===", flush=True)
        rc = subprocess.run([PY, WORKER, str(seed)]).returncode
        mins = (time.time() - t0) / 60
        if rc == 0 and complete(tag):
            print(f"=== {tag} OK in {mins:.0f} min ===", flush=True)
            done.append(tag)
        else:
            print(f"=== {tag} FAILED rc={rc} after {mins:.0f} min ===", flush=True)
            failed.append(tag)
            if not done:
                # first attempt failed: almost certainly systemic (config, memory,
                # import). Stop rather than repeat it nine more times.
                print("!!! stopping: first generation attempt failed", flush=True)
                break

    print(f"\ngenerated={len(done)} skipped={len(skipped)} failed={len(failed)} "
          f"in {(time.time() - t_all) / 60:.0f} min, {free_gb():.1f} GB free", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
    ready = sorted(t for t in set(done) | set(skipped) if complete(t))
    print(f"datasets ready for manifest ({len(ready)}/{len(SEEDS)}): "
          + " ".join(ready), flush=True)


if __name__ == "__main__":
    main()
