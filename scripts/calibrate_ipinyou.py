"""iPinYou calibration extraction — chunked, full-log capable (v6).

Reads impression + bid logs from one or more directories (Season 2 .tsv,
Season 3 .txt.bz2, or comma-separated *_Nrow.csv previews) in chunks, so
the FULL dataset streams within the 16 GB constraint.

Outputs (t9_simulator/calibration/, same names the generator reads):
  user-side, DEDUPED by ipinyou_id (user-population shapes, per schema):
    ipinyou_region_distribution.csv      A1
    ipinyou_city_distribution.csv        A2 | A1
  auction-side, impression-weighted:
    ipinyou_exchange_distribution.csv    B3 shape
    ipinyou_slot_distribution.csv        (diagnostic; v6 sizes are config)
    ipinyou_visibility_distribution.csv  (diagnostic; dropped in v4)
    ipinyou_hourdow_distribution.csv     D1 hour x day-of-week
    ipinyou_floor_distribution.csv       floor PMF, WON rows (biased; diag)
    ipinyou_paying_distribution.csv      clearing PMF -> base_eCPM shape
    ipinyou_paying_by_exchange.csv
  bid-log side (ALL auctions):
    ipinyou_floor_bid_distribution.csv   unbiased floor shape -> H1
    ipinyou_floor_by_exchange.csv
    ipinyou_winrate_summary.csv          imp rows / bid rows per day
  misc:
    ipinyou_user_activity.csv            imps-per-user PMF (deferred weighting)
    ipinyou_price_summary.csv            medians/quantiles for normalisation

UA-derived extractions (os/browser/device, user_tags) are SKIPPED by
default — v6 sources A3-A5 from config (StatCounter) and dropped tags;
pass --with-ua to recompute them (slow on full logs).

Usage:
  python generator/calibrate_ipinyou.py                  # FULL: S2 + S3
  python generator/calibrate_ipinyou.py <dir> [<dir>..]  # specific dirs
  python generator/calibrate_ipinyou.py --with-ua <dir>
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PKG_ROOT.parent
OUTPUT_DIR = PKG_ROOT / "calibration"
OUTPUT_DIR.mkdir(exist_ok=True)

# Raw iPinYou season 2+3 logs (NOT included in this repo - download from the
# iPinYou/UCL mirror and unpack here):
DEFAULT_DIRS = [PKG_ROOT / "Data" / "iPinYou" / "training2nd",
                PKG_ROOT / "Data" / "iPinYou" / "training3rd"]

CHUNK = 2_000_000

# imp/clk/conv logs: 24 columns. bid logs: 21.
IMP_COLS = ["bid_id", "timestamp", "log_type", "ipinyou_id", "user_agent",
            "ip", "region", "city", "ad_exchange", "domain", "url",
            "anon_url", "ad_slot_id", "width", "height", "visibility",
            "format", "floor_price", "creative_id", "bidding_price",
            "paying_price", "landing_url", "advertiser_id", "user_tags"]
BID_COLS = ["bid_id", "timestamp", "ipinyou_id", "user_agent", "ip",
            "region", "city", "ad_exchange", "domain", "url", "anon_url",
            "ad_slot_id", "width", "height", "visibility", "format",
            "floor_price", "creative_id", "bidding_price", "advertiser_id",
            "user_tags"]


def log_files(dirs, prefix):
    """(files, sep) for a log type across dirs; csv preview / tsv / bz2."""
    files = []
    for d in dirs:
        for pat in (f"{prefix}.*.csv", f"{prefix}.*.tsv",
                    f"{prefix}.*.txt.bz2"):
            files += glob.glob(os.path.join(str(d), pat))
    files = sorted(files)
    sep = "," if files and files[0].endswith(".csv") else "\t"
    return files, sep


def add(acc, key, series):
    acc[key] = series if key not in acc else acc[key].add(series,
                                                          fill_value=0)


def pmf_stats(counts):
    s = counts.sort_index()
    total = s.sum()
    cum = s.cumsum() / total
    q = {p: s.index[int(np.argmax(cum.values >= p))]
         for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
    mean = float((s.index * s).sum() / total)
    return {"n": int(total), "mean": round(mean, 2), "median": q[0.50],
            "p10": q[0.10], "p25": q[0.25], "p75": q[0.75],
            "p90": q[0.90], "p99": q[0.99]}


def save_pmf(counts, name, index_name):
    pmf = (counts / counts.sum()).sort_index()
    pmf.index.name = index_name
    pmf.rename("proportion").to_csv(OUTPUT_DIR / name)


def save_grouped(counts, name, names):
    df = counts.rename("count").reset_index()
    df.columns = [*names, "count"]
    df["proportion"] = df["count"] / df.groupby(names[0])["count"] \
        .transform("sum")
    df.to_csv(OUTPUT_DIR / name, index=False)


def parse_ua(chunk, acc):
    ua = chunk["user_agent"].astype(str).str.lower()
    osname = np.select(
        [ua.str.contains("windows"), ua.str.contains("android"),
         ua.str.contains("iphone|ipad"), ua.str.contains("linux"),
         ua.str.contains("mac")],
        ["Windows", "Android", "iOS", "Linux", "Mac"], "Other")
    add(acc, "os", pd.Series(osname).value_counts())
    device = np.select(
        [ua.str.contains("mobile|android|iphone"),
         ua.str.contains("ipad|tablet")],
        ["Mobile", "Tablet"], "Desktop")
    add(acc, "device", pd.Series(device).value_counts())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_ua = "--with-ua" in sys.argv
    dirs = args or DEFAULT_DIRS
    t0 = time.time()
    print(f"calibrating from: {[str(d) for d in dirs]}  "
          f"(chunk={CHUNK:,}, with_ua={with_ua})")

    # ── impression logs ─────────────────────────────────────────────
    imp_files, imp_sep = log_files(dirs, "imp")
    if not imp_files:
        raise FileNotFoundError(f"no imp.* logs in {dirs}")
    print(f"{len(imp_files)} impression logs")

    acc = {}
    seen = set()                      # hashed ipinyou_id (user dedup)
    activity = {}                     # hashed id -> imp count (Series adds)
    imp_rows_per_day = {}
    n_imp = 0

    # keep usecols SORTED: pandas assigns `names` to selected columns in
    # file order, not list order — an unsorted list silently misaligns
    usecols = sorted([1, 3, 6, 7, 8, 13, 14, 15, 17, 20]
                     + ([4] if with_ua else []))
    names = [IMP_COLS[i] for i in usecols]
    for f in imp_files:
        day = Path(f).name.split(".")[1].split("_")[0]
        # dtype=str: no inference -> no DtypeWarning (whose formatter crashes
        # under usecols+names on mixed-dtype chunks); numerics converted below
        for chunk in pd.read_csv(f, header=None, sep=imp_sep,
                                 usecols=usecols, names=names, dtype=str,
                                 chunksize=CHUNK, na_filter=False):
            n_imp += len(chunk)
            imp_rows_per_day[day] = imp_rows_per_day.get(day, 0) + len(chunk)

            # per-user dedup for A-feature shapes + activity counts
            h = chunk["ipinyou_id"].map(hash)
            hc = h.value_counts()
            add(activity, "a", hc)
            first = ~h.duplicated() & ~h.isin(seen)
            seen.update(h[first])
            uniq = chunk.loc[first, ["region", "city"]].apply(
                pd.to_numeric, errors="coerce").dropna().astype("int64")
            add(acc, "region", uniq["region"].value_counts())
            add(acc, "city", uniq.groupby(["region", "city"]).size())

            # impression-weighted auction-side shapes
            add(acc, "exchange", chunk["ad_exchange"].value_counts())
            add(acc, "slot", chunk.groupby(["width", "height"]).size())
            add(acc, "vis", chunk["visibility"].astype(str).value_counts())
            ts = chunk["timestamp"].astype(str)
            dt = pd.to_datetime(ts.str[:10], format="%Y%m%d%H",
                                errors="coerce")
            add(acc, "hourdow",
                pd.DataFrame({"dow": dt.dt.dayofweek, "hour": dt.dt.hour})
                .dropna().astype(int).groupby(["dow", "hour"]).size())
            for col, key in (("floor_price", "floor_imp"),
                             ("paying_price", "paying_imp")):
                v = pd.to_numeric(chunk[col], errors="coerce").dropna()
                v = v[v >= 0].astype(int)
                add(acc, key, v.value_counts())
            pe = chunk[["ad_exchange", "paying_price"]].copy()
            pe["paying_price"] = pd.to_numeric(pe["paying_price"],
                                               errors="coerce")
            pe = pe.dropna()
            pe = pe[pe["paying_price"] >= 0].astype(
                {"paying_price": "int64"})
            add(acc, "paying_exch",
                pe.groupby(["ad_exchange", "paying_price"]).size())
            if with_ua:
                parse_ua(chunk, acc)
        print(f"  imp {Path(f).name}: total {n_imp:,} rows, "
              f"{len(seen):,} users  [{time.time()-t0:.0f}s]")

    # ── bid logs (floors on ALL auctions + win rate) ────────────────
    bid_files, bid_sep = log_files(dirs, "bid")
    bid_rows_per_day = {}
    n_bid = 0
    if bid_files:
        usecols_b = [7, 16]
        for f in bid_files:
            day = Path(f).name.split(".")[1].split("_")[0]
            for chunk in pd.read_csv(f, header=None, sep=bid_sep,
                                     usecols=usecols_b,
                                     names=["ad_exchange", "floor_price"],
                                     dtype=str, chunksize=CHUNK,
                                     na_filter=False):
                n_bid += len(chunk)
                bid_rows_per_day[day] = bid_rows_per_day.get(day, 0) \
                    + len(chunk)
                v = pd.to_numeric(chunk["floor_price"],
                                  errors="coerce").dropna()
                v = v[v >= 0].astype(int)
                add(acc, "floor_bid", v.value_counts())
                fb = chunk.copy()
                fb["floor"] = pd.to_numeric(fb["floor_price"],
                                            errors="coerce")
                fb = fb.dropna(subset=["floor"])
                fb = fb[fb["floor"] >= 0].astype({"floor": "int64"})
                add(acc, "floor_exch",
                    fb.groupby(["ad_exchange", "floor"]).size())
            print(f"  bid {Path(f).name}: total {n_bid:,} rows  "
                  f"[{time.time()-t0:.0f}s]")
    else:
        print("WARNING: no bid logs found — floor_bid/win-rate skipped")

    # ── write outputs ───────────────────────────────────────────────
    save_pmf(acc["region"], "ipinyou_region_distribution.csv", "region")
    save_grouped(acc["city"], "ipinyou_city_distribution.csv",
                 ["region", "city"])
    save_pmf(acc["exchange"], "ipinyou_exchange_distribution.csv",
             "ad_exchange")
    save_grouped(acc["slot"], "ipinyou_slot_distribution.csv",
                 ["width", "height"])
    save_pmf(acc["vis"], "ipinyou_visibility_distribution.csv",
             "visibility")
    save_grouped(acc["hourdow"], "ipinyou_hourdow_distribution.csv",
                 ["dow", "hour"])
    save_pmf(acc["floor_imp"], "ipinyou_floor_distribution.csv",
             "floor_price")
    save_pmf(acc["paying_imp"], "ipinyou_paying_distribution.csv",
             "paying_price")
    save_grouped(acc["paying_exch"], "ipinyou_paying_by_exchange.csv",
                 ["ad_exchange", "paying_price"])
    if with_ua:
        save_pmf(acc["os"], "ipinyou_os_distribution.csv", "os")
        save_pmf(acc["device"], "ipinyou_device_distribution.csv",
                 "device_type")

    summary = []
    if bid_files:
        save_pmf(acc["floor_bid"], "ipinyou_floor_bid_distribution.csv",
                 "floor_price")
        save_grouped(acc["floor_exch"], "ipinyou_floor_by_exchange.csv",
                     ["ad_exchange", "floor_price"])
        wr = pd.DataFrame({"bid_rows": pd.Series(bid_rows_per_day),
                           "imp_rows": pd.Series(imp_rows_per_day)}) \
            .fillna(0).astype(int)
        wr["win_rate"] = (wr["imp_rows"]
                          / wr["bid_rows"].replace(0, np.nan)).round(4)
        wr.index.name = "date"
        wr.to_csv(OUTPUT_DIR / "ipinyou_winrate_summary.csv")
        summary.append({"series": "floor_bid", **pmf_stats(acc["floor_bid"])})
        overall = wr["imp_rows"].sum() / max(wr["bid_rows"].sum(), 1)
        print(f"win rate overall: {overall:.4f} "
              f"({wr['imp_rows'].sum():,} imps / {wr['bid_rows'].sum():,} bids)")

    for key in ("floor_imp", "paying_imp"):
        summary.append({"series": key, **pmf_stats(acc[key])})
    pd.DataFrame(summary).set_index("series").to_csv(
        OUTPUT_DIR / "ipinyou_price_summary.csv")

    act_counts = activity["a"].value_counts().sort_index()
    act_counts.index.name = "imps_per_user"
    act_counts.rename("n_users").to_csv(OUTPUT_DIR /
                                        "ipinyou_user_activity.csv")

    print(f"\n=== Summary ===")
    print(f"impressions: {n_imp:,}   bids: {n_bid:,}   "
          f"unique users: {len(seen):,}")
    print(f"paying median: {pmf_stats(acc['paying_imp'])['median']}   "
          f"floor_bid zero-share: "
          f"{acc['floor_bid'].get(0, 0) / acc['floor_bid'].sum():.3f}"
          if bid_files else "")
    print(f"done in {time.time()-t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
