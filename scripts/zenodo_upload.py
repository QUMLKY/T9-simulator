"""Upload the packaged deposit to a Zenodo draft record, verifying every transfer.

Browser upload of eleven 1.7 GB files is where large deposits fail: no resume, no
integrity check, and a dropped connection late in a file wastes the whole thing.
This streams each file to Zenodo's bucket API and checks the MD5 Zenodo computes
against the one in MANIFEST.txt, so a silently corrupted transfer cannot pass.

Resumable: files already on the record with a matching MD5 are skipped, so
re-running after an interruption costs nothing.

Standard library only, so it runs in any environment that can run the simulator.

SETUP
  1. Create the draft record on the website first and reserve its DOI.
  2. Note the record id from the URL: zenodo.org/uploads/<ID>
  3. Create a personal access token with the "deposit:write" scope:
       zenodo.org/account/settings/applications/tokens/new
  4. Put the token in the environment (NOT on the command line, where it would
     be visible to anyone listing processes):
       $env:ZENODO_TOKEN = "..."          # PowerShell
       export ZENODO_TOKEN=...            # bash

USAGE
  python scripts/zenodo_upload.py <RECORD_ID>
  python scripts/zenodo_upload.py <RECORD_ID> --sandbox
  python scripts/zenodo_upload.py <RECORD_ID> --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPOSIT = ROOT / "zenodo_deposit"
CHUNK = 1 << 20


def api_base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def call(url: str, token: str, method: str = "GET", body=None, length: int | None = None):
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(length))
        req.data = body
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"\nZenodo API {e.code} on {method} {url.split('?')[0]}\n{detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"\nnetwork error contacting Zenodo: {e.reason}")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def expected_md5s() -> dict[str, str]:
    """Parse MANIFEST.txt -> {filename: md5}. Empty if it is absent."""
    man = DEPOSIT / "MANIFEST.txt"
    if not man.exists():
        return {}
    out, current = {}, None
    for line in man.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## "):
            current = s[3:].split()[0]
        elif s.startswith("md5") and "=" in s and current:
            out[current] = s.split("=", 1)[1].strip()
    return out


class Progress:
    """Wraps a file object so urllib's read loop reports progress as it streams."""

    def __init__(self, fh, total, label):
        self.fh, self.total, self.label, self.sent = fh, total, label, 0

    def read(self, n=-1):
        b = self.fh.read(n)
        self.sent += len(b)
        pct = self.sent / self.total * 100 if self.total else 100
        print(f"\r    {self.label}  {pct:5.1f}%  "
              f"{self.sent / 1e9:.2f}/{self.total / 1e9:.2f} GB", end="", flush=True)
        return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_id")
    ap.add_argument("--sandbox", action="store_true",
                    help="upload to sandbox.zenodo.org (a throwaway clone; test here first)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would upload, contacting Zenodo only to read the record")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_TOKEN in the environment (see this file's docstring)")

    files = sorted(p for p in DEPOSIT.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"nothing to upload in {DEPOSIT}")

    base = api_base(args.sandbox)
    dep = call(f"{base}/deposit/depositions/{args.record_id}", token)
    bucket = (dep.get("links") or {}).get("bucket")
    if not bucket:
        raise SystemExit("record has no bucket link; is it a published record rather "
                         "than a draft? Files can only be added to a draft.")

    on_record = {f["filename"]: f.get("checksum", "").replace("md5:", "")
                 for f in dep.get("files", [])}
    expect = expected_md5s()
    title = (dep.get("metadata") or {}).get("title", "(untitled draft)")
    print(f"record {args.record_id} on {'SANDBOX' if args.sandbox else 'zenodo.org'}: {title}")
    print(f"{len(files)} local files, {len(on_record)} already on the record\n")

    uploaded = skipped = 0
    for p in files:
        size = p.stat().st_size
        want = expect.get(p.name)

        if p.name in on_record:
            if want and on_record[p.name] == want:
                print(f"  skip  {p.name}  (already uploaded, MD5 matches)")
                skipped += 1
                continue
            print(f"  REUPLOAD {p.name}  (on record but MD5 differs)")

        if args.dry_run:
            print(f"  would upload  {p.name}  {size / 1e9:.2f} GB")
            continue

        with p.open("rb") as fh:
            r = call(f"{bucket}/{p.name}", token, "PUT",
                     body=Progress(fh, size, p.name), length=size)
        got = (r.get("checksum") or "").replace("md5:", "")
        print()

        if want and got != want:
            raise SystemExit(f"    MD5 MISMATCH for {p.name}\n"
                             f"      manifest: {want}\n      zenodo:   {got}\n"
                             f"    The transfer was corrupted. Re-run to retry.")
        print(f"    ok  md5={got[:12]}...{'  (verified against manifest)' if want else ''}")
        uploaded += 1

    print(f"\nuploaded={uploaded} skipped={skipped}")
    if not args.dry_run:
        print("\nNext: check the record on the website, then Publish.")
        print("Publishing is irreversible for files.")


if __name__ == "__main__":
    main()
