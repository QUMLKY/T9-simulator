"""Build the manifest for a Zenodo (or other archive) deposit of generated datasets.

Two independent guarantees, because they catch different failures:

  SHA-256      the uploaded bytes are intact. Catches truncated or corrupted transfer.
  fingerprint  the DATA is the data the paper used. Catches a dataset regenerated
               from the wrong seed, config or schema version. Parquet embeds
               non-deterministic metadata, so two byte-different files can hold
               identical data; a raw hash cannot tell you that, and this can.

The fingerprint pass reads each parquet fully into memory (about 1.5 GB per 10M
master), so it is opt-in via --fingerprint. Run it at least once before uploading.

Usage:
  python scripts/make_deposit_manifest.py output/v10_10m_s13 output/v10_10m_s14 ...
  python scripts/make_deposit_manifest.py output/v10_10m_s* --fingerprint
  python scripts/make_deposit_manifest.py output/v10_10m_s13 -o DEPOSIT_MANIFEST.txt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="dataset directories containing *.parquet")
    ap.add_argument("--fingerprint", action="store_true",
                    help="also compute the t9sim content fingerprint (slow, memory-heavy)")
    ap.add_argument("-o", "--out", default="DEPOSIT_MANIFEST.txt")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.datasets]
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        ap.error("not a directory: " + ", ".join(str(m) for m in missing))

    lines = [
        "# T9 dataset deposit manifest",
        "#",
        "# SHA-256 verifies the uploaded bytes are intact.",
        "# The content fingerprint verifies the DATA matches what the paper used,",
        "# independently of parquet's non-deterministic file metadata. Recompute with:",
        "#     python -m t9sim.fingerprint <dataset_dir>",
        "#",
    ]
    if not args.fingerprint:
        lines.append("# NOTE: run again with --fingerprint before uploading.")
        lines.append("#")

    total_bytes = 0
    for d in dirs:
        files = sorted(d.glob("*.parquet"))
        if not files:
            print(f"warning: no parquet files in {d}", file=sys.stderr)
            continue

        lines.append("")
        lines.append(f"## {d.name}")
        for f in files:
            size = f.stat().st_size
            total_bytes += size
            rows = pq.ParquetFile(f).metadata.num_rows
            lines.append(f"{sha256(f)}  {f.name}  rows={rows}  bytes={size}")
            print(f"  hashed {d.name}/{f.name} ({size / 1e9:.2f} GB)")

        if args.fingerprint:
            from t9sim.fingerprint import fingerprint    # noqa: E402
            print(f"  fingerprinting {d.name} (reads each parquet fully) ...")
            fp = fingerprint(str(d))
            lines.append(f"fingerprint = {fp:#018x}")

    lines.append("")
    lines.append(f"# {len(dirs)} datasets, {total_bytes / 1e9:.2f} GB total")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out} ({len(dirs)} datasets, {total_bytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
