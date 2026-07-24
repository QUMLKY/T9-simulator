"""Package the T9 datasets for the Zenodo deposit, and manifest what goes up.

Produces one zip per dataset, each a complete self-contained unit (auctions plus
the three entity pools). Per-seed archives rather than one 17 GB blob (which
would force an all-or-nothing download) or 40 loose files (which would leave the
record unreadable and give no clue which four files belong together).

Archives are STORED, not deflated: parquet is already compressed internally, so
deflate buys about 6% for real CPU on both ends. Zip still carries a CRC32 per
member, so integrity is retained.

The manifest records, for each archive:
  sha256 + md5   of the archive itself. Zenodo displays MD5 for every uploaded
                 file, so md5 is what lets a depositor tick off the transfer
                 without rehashing gigabytes locally; sha256 is the stronger
                 check for anyone verifying later.
  the members    each inner parquet with its own sha256 and row count, so a
                 user who unpacks can verify what came out.
  fingerprint    the t9sim content fingerprint of the dataset, which verifies
                 the DATA rather than the bytes. Parquet embeds
                 non-deterministic metadata, so a correct regeneration differs
                 byte-for-byte; only this survives that.

Usage:
  python scripts/make_deposit_package.py            # package + manifest
  python scripts/make_deposit_package.py --manifest-only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "output"
DEPOSIT = ROOT / "zenodo_deposit"
MEMBERS = ["auctions.parquet", "pool_users.parquet",
           "pool_apps.parquet", "pool_campaigns.parquet"]

# (source dataset dir, archive stem, human label)
PAYLOAD = [(f"v10_10m_s{s}", f"t9_v10_10m_seed902{s}", f"10M master, seed 902{s}")
           for s in range(13, 23)]
PAYLOAD.append(("v10_anchor_s13", "t9_v10_1m_sample_seed90213",
                "1M sample, seed 90213"))


def hashes(path: Path) -> tuple[str, str, int]:
    """sha256, md5 and size in one pass over the file."""
    h256, hmd5 = hashlib.sha256(), hashlib.md5()
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h256.update(chunk)
            hmd5.update(chunk)
            n += len(chunk)
    return h256.hexdigest(), hmd5.hexdigest(), n


def build(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(".zip.part")     # never leave a half-written .zip
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as z:
        for m in MEMBERS:
            z.write(src / m, arcname=f"{dest.stem}/{m}")
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true",
                    help="skip zipping; manifest the archives already present")
    args = ap.parse_args()

    DEPOSIT.mkdir(exist_ok=True)
    missing = [s for s, _, _ in PAYLOAD if not (OUT_DIR / s).is_dir()]
    if missing:
        sys.exit("missing source datasets: " + ", ".join(missing))

    lines = [
        "# T9 Zenodo deposit manifest",
        "#",
        "# One archive per dataset. Each is self-contained: the auction master",
        "# plus the three entity pools (users, apps, campaigns).",
        "#",
        "# md5     matches what Zenodo displays per uploaded file - use it to",
        "#         confirm the transfer arrived intact.",
        "# sha256  stronger check of the same archive.",
        "# members each inner parquet, verifiable after unpacking.",
        "# fingerprint  content hash of the DATA, independent of parquet's",
        "#         non-deterministic file metadata. Recompute with:",
        "#             python -m t9sim.fingerprint <unpacked_dir>",
        "#         This is what proves a dataset matches the published one; a",
        "#         correct regeneration will NOT match on bytes.",
        "",
    ]

    total = 0
    for src_name, stem, label in PAYLOAD:
        src, dest = OUT_DIR / src_name, DEPOSIT / f"{stem}.zip"

        if not args.manifest_only:
            print(f"  packaging {stem}.zip ...", flush=True)
            build(src, dest)
        elif not dest.exists():
            sys.exit(f"--manifest-only but {dest.name} is absent")

        sha, md5, size = hashes(dest)
        total += size

        lines.append(f"## {stem}.zip    ({label})")
        lines.append(f"bytes  = {size}")
        lines.append(f"md5    = {md5}")
        lines.append(f"sha256 = {sha}")
        lines.append("members:")
        for m in MEMBERS:
            f = src / m
            msha, _, msize = hashes(f)
            rows = pq.ParquetFile(f).metadata.num_rows
            lines.append(f"  {msha}  {m}  rows={rows}  bytes={msize}")

        from t9sim.fingerprint import fingerprint      # noqa: E402
        print(f"  fingerprinting {src_name} ...", flush=True)
        lines.append(f"fingerprint = {fingerprint(str(src)):#018x}")
        lines.append("")
        print(f"  {stem}.zip  {size / 1e9:.2f} GB  md5={md5[:12]}...", flush=True)

    lines.append(f"# {len(PAYLOAD)} archives, {total / 1e9:.2f} GB total")
    (DEPOSIT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {DEPOSIT / 'MANIFEST.txt'}")
    print(f"{len(PAYLOAD)} archives, {total / 1e9:.2f} GB in {DEPOSIT}")


if __name__ == "__main__":
    main()
