"""Reduce completed GPU burst receipts to per-run metrics for summarize.py."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def measure(path):
    payload = path.read_bytes()
    run = json.loads(payload)
    if run.get("schema") != "kvladder-conf-arm-receipt-v2" or run.get("scenario") != "burst":
        return None
    phases = {"recency", "confirmatory_v2_attempt2", "matched", "e2_matched_admission",
              "length_ragged", "e3_prime_length_ragged"}
    if run.get("phase") not in phases:
        raise ValueError(f"unexpected burst campaign in {path}: {run.get('phase')}")
    if not run["passed"]:
        raise ValueError(f"failed burst: {path}")
    branches = [row for row in run["rows"] if row["role"] == "branch"]
    config, summary = run["configuration"], run["summary"]
    if len(branches) != config["branches"] or not all(row["completed"] for row in branches):
        raise ValueError(f"incomplete burst: {path}")
    geometry = run.get("published_base_geometry") or {}
    campaign = "recency"
    if config.get("compression_scorer") == "fastkvzip":
        campaign = "length_ragged"
    elif run.get("admission_mode") == "tight":
        campaign = "matched"
    times = sorted(row["branch_start_latency_ms"] for row in branches)
    return dict(campaign=campaign, arm=run["arm"], context=run["trace"]["context_tokens"],
                budget_gib=config["kv_cache_memory_bytes"] // 2**30, branches=len(branches), seed=config["seed"],
                first_round=sum(t - times[0] < 50 for t in times),
                max_concurrent=summary["maximum_concurrent_physically_admitted"],
                max_ttft_ms=max(row["ttft_ms"] for row in branches),
                throughput_rps=summary["throughput_requests_per_s"], peak_kv_bytes=summary["peak_live_hbm_bytes"],
                base_pages=geometry.get("total_pages"), distinct_lengths=geometry.get("distinct_retained_lengths"),
                selection_fingerprint=geometry.get("selection_fingerprint"), receipt_sha256=hashlib.sha256(payload).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+", help="files or result directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = set()
    for item in args.receipts:
        paths.update(item.rglob("*.json") if item.is_dir() else [item])
    rows = [row for path in sorted(paths) if (row := measure(path)) is not None]
    if not rows:
        parser.error("no burst receipts found")
    ids = {row["receipt_sha256"] for row in rows}
    if len(ids) != len(rows):
        parser.error("duplicate receipts in inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
