#!/usr/bin/env python3
"""Fail-closed paired analysis for full-session SCBench QA Native vs Shared receipts.

This script consumes immutable generation receipts and writes a descriptive paired
quality analysis.  It does not run a model, does not declare equivalence, and treats
any mismatch between the paired inputs as an error rather than silently intersecting
requests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "kvladder-conf-full-session-quality-analysis-v1"
INPUT_SCHEMA = "kvladder-conf-full-session-quality-v1"
LEGACY_QUALITY_SCHEMA = "kvladder-conf-quality-receipt-v1"
NATIVE_ARM = "native"
SHARED_ARM = "shared"
REQUEST_FIELD = "answer_f1"
QA_TASK = "scbench_qa_eng"
DEFAULT_EXPECT_SESSIONS = 20
DEFAULT_EXPECT_REQUESTS = 102
DEFAULT_BOOTSTRAP_DRAWS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260912

# Minimal keys that must be present, plus the baseline protocol surface.  The
# full config union is compared below; this list prevents tiny synthetic receipts
# from passing just because both sides omitted important provenance.
PRIMARY_CONFIG_KEYS = (
    "model",
    "tokenizer",
    "retained_ratio",
    "target_retained_tokens_per_chunk",
    "body_keep_ratio",
    "eviction_ratio",
    "compression_chunk_size",
    "compression_scorer",
    "compression_sink_tokens",
    "compression_window_size",
    "compression_floor_min",
    "page_group_size",
    "kv_cache_memory_bytes",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "seed",
    "max_new_tokens",
    "temperature",
    "add_special_tokens",
)

REQUIRED_CONFIG_KEYS = PRIMARY_CONFIG_KEYS + (
    "resolved_enable_prefix_caching",
    "kv_cache_dtype",
)

# The full-session runner builds both arms through the same ragged FP16 Native
# builder; only the arm identity, Shared publication boundary, and registry module
# hash are allowed to differ.
ALLOWED_CONFIG_DIFF_KEYS = {"arm", "published_prefix_tokens", "registry_module_sha256"}

CSV_FIELDS = (
    "request_id",
    "session_index",
    "turn_index",
    "prompt_tokens",
    "prefix_tokens",
    "question_tokens",
    "native_answer_f1",
    "shared_answer_f1",
    "delta_shared_minus_native",
    "native_generated_token_ids_sha256",
    "shared_generated_token_ids_sha256",
    "prompt_token_ids_sha256",
    "shared_cached_tokens",
    "shared_published_prefix_tokens",
    "shared_registry_hits_delta",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - exact JSON error text is not contract
        die(f"cannot read JSON receipt {path}: {exc}")
    if not isinstance(receipt, dict):
        die(f"receipt {path} is not a JSON object")
    if not isinstance(receipt.get("rows"), list):
        die(f"receipt {path} has no rows list")
    if not isinstance(receipt.get("summary"), dict):
        die(f"receipt {path} has no summary object")
    if not isinstance(receipt.get("dataset"), dict):
        die(f"receipt {path} has no dataset object")
    if not isinstance(receipt.get("configuration"), dict):
        die(f"receipt {path} has no configuration object")
    return receipt


def require_no_skips(name: str, receipt: dict[str, Any]) -> None:
    skipped = receipt.get("summary", {}).get("skipped", [])
    if skipped:
        die(f"{name} receipt has skipped requests: {skipped!r}")


def validate_task(name: str, receipt: dict[str, Any]) -> None:
    if receipt.get("task") != QA_TASK:
        die(f"{name} receipt task is {receipt.get('task')!r}, expected {QA_TASK!r}")
    metric = receipt.get("summary", {}).get("metric")
    if metric not in (None, "f1"):
        die(f"{name} receipt metric is {metric!r}, expected 'f1'")


def validate_input_schema_arm(name: str, receipt: dict[str, Any], expected_arm: str) -> None:
    schema = receipt.get("schema")
    if schema != INPUT_SCHEMA:
        die(f"{name} receipt schema is {schema!r}, expected {INPUT_SCHEMA!r}")
    arm = receipt.get("arm")
    if arm != expected_arm:
        die(f"{name} receipt arm is {arm!r}, expected {expected_arm!r}")
    cfg_arm = receipt.get("configuration", {}).get("arm")
    if cfg_arm != expected_arm:
        die(f"{name} receipt configuration.arm is {cfg_arm!r}, expected {expected_arm!r}")


def validate_historical_schema(name: str, receipt: dict[str, Any]) -> None:
    schema = receipt.get("schema")
    if schema not in (INPUT_SCHEMA, LEGACY_QUALITY_SCHEMA):
        die(f"{name} receipt schema is {schema!r}, expected {INPUT_SCHEMA!r} or {LEGACY_QUALITY_SCHEMA!r}")
    if not receipt.get("arm"):
        die(f"{name} receipt missing arm metadata")
    if not receipt.get("configuration", {}).get("arm"):
        die(f"{name} receipt missing configuration.arm metadata")


def rows_by_request(name: str, receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(receipt["rows"]):
        if not isinstance(row, dict):
            die(f"{name} row {i} is not an object")
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid:
            die(f"{name} row {i} has invalid request_id {rid!r}")
        if rid in seen:
            die(f"{name} has duplicate request_id {rid}")
        seen[rid] = row
        for field in ("session_index", "turn_index", "prompt_tokens", "prefix_tokens", "question_tokens", "gold", REQUEST_FIELD, "generated_token_ids"):
            if field not in row:
                die(f"{name} row {rid} missing {field}")
        f1 = row[REQUEST_FIELD]
        if not isinstance(f1, (int, float)) or not math.isfinite(float(f1)):
            die(f"{name} row {rid} has non-finite {REQUEST_FIELD}: {f1!r}")
    return seen


def validate_counts(name: str, rows: dict[str, dict[str, Any]], expected_sessions: int, expected_requests: int) -> None:
    sessions = {int(row["session_index"]) for row in rows.values()}
    if len(rows) != expected_requests:
        die(f"{name} has {len(rows)} requests, expected {expected_requests}")
    if len(sessions) != expected_sessions:
        die(f"{name} has {len(sessions)} sessions, expected {expected_sessions}")
    expected_session_ids = set(range(expected_sessions))
    if sessions != expected_session_ids:
        die(f"{name} session ids are {sorted(sessions)}, expected contiguous {sorted(expected_session_ids)}")
    by_session_turns: dict[int, list[int]] = {}
    for rid, row in rows.items():
        session = int(row["session_index"])
        turn = int(row["turn_index"])
        expected_prefix = f"s{session:02d}t{turn}"
        if rid != expected_prefix:
            die(f"{name} row id {rid!r} does not match session/turn {expected_prefix!r}")
        by_session_turns.setdefault(session, []).append(turn)
    for session, turns in sorted(by_session_turns.items()):
        expected_turns = list(range(len(turns)))
        if sorted(turns) != expected_turns:
            die(f"{name} session {session} turns are {sorted(turns)}, expected contiguous {expected_turns}")


def compare_dataset(native: dict[str, Any], other: dict[str, Any], other_name: str = "shared") -> dict[str, Any]:
    n_sha = native["dataset"].get("sha256")
    o_sha = other["dataset"].get("sha256")
    if not n_sha or not o_sha:
        die(f"dataset sha256 missing: native={n_sha!r} {other_name}={o_sha!r}")
    if n_sha != o_sha:
        die(f"dataset sha256 mismatch: native={n_sha!r} {other_name}={o_sha!r}")
    n_path = native["dataset"].get("path")
    o_path = other["dataset"].get("path")
    return {
        "sha256": n_sha,
        "native_path": n_path,
        f"{other_name}_path": o_path,
        "paths_match": n_path == o_path,
        "note": "dataset identity is checked by sha256; path strings are reported for provenance only",
    }


def compare_config(native: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    nc = native["configuration"]
    sc = shared["configuration"]
    for key in REQUIRED_CONFIG_KEYS:
        if key not in nc or key not in sc:
            die(f"required configuration key {key!r} missing on one side")
    checked: dict[str, Any] = {}
    for key in sorted(set(nc) | set(sc)):
        if key in ALLOWED_CONFIG_DIFF_KEYS:
            continue
        if key not in nc or key not in sc:
            die(f"configuration key {key!r} missing on one side")
        if nc[key] != sc[key]:
            die(f"configuration mismatch for {key}: native={nc[key]!r} shared={sc[key]!r}")
        checked[key] = nc[key]
    return checked


def compare_row_identity(rid: str, native: dict[str, Any], shared: dict[str, Any]) -> None:
    for field in ("session_index", "turn_index", "prompt_tokens", "prefix_tokens", "question_tokens", "gold"):
        if native.get(field) != shared.get(field):
            die(f"paired row {rid} mismatch for {field}: native={native.get(field)!r} shared={shared.get(field)!r}")
    nh = native.get("prompt_token_ids_sha256")
    sh = shared.get("prompt_token_ids_sha256")
    if nh is None or sh is None:
        die(f"paired row {rid} missing prompt_token_ids_sha256")
    if nh != sh:
        die(f"paired row {rid} prompt_token_ids_sha256 mismatch: native={nh!r} shared={sh!r}")


def shared_publications_by_session(receipt: dict[str, Any], expected_sessions: int) -> dict[int, dict[str, Any]]:
    publications = receipt.get("publications")
    if not isinstance(publications, list):
        die("shared receipt missing publications list")
    by: dict[int, dict[str, Any]] = {}
    for i, pub in enumerate(publications):
        if not isinstance(pub, dict):
            die(f"shared publication {i} is not an object")
        sid = pub.get("session_index")
        if not isinstance(sid, int):
            die(f"shared publication {i} has invalid session_index {sid!r}")
        if sid in by:
            die(f"shared has duplicate publication for session {sid}")
        if pub.get("published_prefix_tokens") != 32768:
            die(f"shared publication {sid} published_prefix_tokens is {pub.get('published_prefix_tokens')!r}, expected 32768")
        pages = pub.get("physical_page_ids")
        if not isinstance(pages, list) or not pages:
            die(f"shared publication {sid} has no physical_page_ids")
        if not all(isinstance(x, int) for x in pages):
            die(f"shared publication {sid} physical_page_ids must be integers")
        by[sid] = pub
    expected = set(range(expected_sessions))
    if set(by) != expected:
        die(f"shared publication sessions are {sorted(by)}, expected contiguous {sorted(expected)}")
    return by


def validate_shared_attach(rid: str, row: dict[str, Any], publication: dict[str, Any]) -> dict[str, Any]:
    attach = row.get("shared_attach")
    if not isinstance(attach, dict):
        die(f"shared row {rid} missing shared_attach object")
    if attach.get("attached") is not True:
        die(f"shared row {rid} was not attached: {attach!r}")
    if attach.get("same_resident_pages") is not True:
        die(f"shared row {rid} did not verify same resident pages: {attach!r}")
    published = attach.get("published_prefix_tokens")
    cached = attach.get("cached_tokens")
    hits = attach.get("registry_hits_delta")
    if not isinstance(published, int) or published <= 0:
        die(f"shared row {rid} has invalid published_prefix_tokens: {published!r}")
    if cached != published:
        die(f"shared row {rid} cached_tokens {cached!r} != published_prefix_tokens {published!r}")
    if row.get("num_cached_tokens") != cached:
        die(f"shared row {rid} num_cached_tokens {row.get('num_cached_tokens')!r} != shared_attach.cached_tokens {cached!r}")
    if hits != 1:
        die(f"shared row {rid} registry_hits_delta {hits!r} != 1")
    if published != publication.get("published_prefix_tokens"):
        die(f"shared row {rid} published_prefix_tokens {published!r} != publication boundary {publication.get('published_prefix_tokens')!r}")
    if published > int(row["prefix_tokens"]):
        die(f"shared row {rid} published_prefix_tokens exceeds prefix_tokens")
    event = attach.get("event")
    if not isinstance(event, dict):
        die(f"shared row {rid} missing attach event")
    if event.get("event") != "lookup_hit_attach":
        die(f"shared row {rid} attach event is {event.get('event')!r}, expected 'lookup_hit_attach'")
    if event.get("request_id") != rid:
        die(f"shared row {rid} attach event request_id is {event.get('request_id')!r}")
    if event.get("raw_page_ids") != publication.get("physical_page_ids"):
        die(f"shared row {rid} attach raw_page_ids do not match publication physical_page_ids")
    if attach.get("lookup_digest") != publication.get("lookup_digest"):
        die(f"shared row {rid} lookup_digest does not match publication")
    return attach


def sem(xs: list[float]) -> float | None:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        die("cannot compute percentile of empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_ci_session_deltas(session_deltas: list[float], draws: int, seed: int) -> dict[str, Any]:
    if not session_deltas:
        die("no session deltas for bootstrap")
    rng = random.Random(seed)
    n = len(session_deltas)
    means = []
    for _ in range(draws):
        means.append(st.fmean(session_deltas[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return {
        "method": "session-cluster percentile bootstrap over per-session deltas",
        "draws": draws,
        "seed": seed,
        "level": 0.95,
        "low": percentile(means, 0.025),
        "high": percentile(means, 0.975),
        "note": "descriptive interval only; no equivalence margin or gate is applied",
    }


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return st.fmean(float(row[field]) for row in rows)


def build_pairs(native_rows: dict[str, dict[str, Any]], shared_rows: dict[str, dict[str, Any]], publications: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[float]]]:
    n_ids = set(native_rows)
    s_ids = set(shared_rows)
    if n_ids != s_ids:
        missing_shared = sorted(n_ids - s_ids)
        extra_shared = sorted(s_ids - n_ids)
        die(f"request sets differ: missing_shared={missing_shared} extra_shared={extra_shared}")
    pairs: list[dict[str, Any]] = []
    by_session_delta: dict[int, list[float]] = {}
    for rid in sorted(n_ids, key=lambda x: (int(native_rows[x]["session_index"]), int(native_rows[x]["turn_index"]), x)):
        nr = native_rows[rid]
        sr = shared_rows[rid]
        compare_row_identity(rid, nr, sr)
        attach = validate_shared_attach(rid, sr, publications[int(sr["session_index"])])
        delta = float(sr[REQUEST_FIELD]) - float(nr[REQUEST_FIELD])
        session = int(nr["session_index"])
        by_session_delta.setdefault(session, []).append(delta)
        pairs.append({
            "request_id": rid,
            "session_index": session,
            "turn_index": int(nr["turn_index"]),
            "prompt_tokens": int(nr["prompt_tokens"]),
            "prefix_tokens": int(nr["prefix_tokens"]),
            "question_tokens": int(nr["question_tokens"]),
            "native_answer_f1": float(nr[REQUEST_FIELD]),
            "shared_answer_f1": float(sr[REQUEST_FIELD]),
            "delta_shared_minus_native": delta,
            "native_generated_token_ids_sha256": sha256_json(nr["generated_token_ids"]),
            "shared_generated_token_ids_sha256": sha256_json(sr["generated_token_ids"]),
            "prompt_token_ids_sha256": nr["prompt_token_ids_sha256"],
            "shared_cached_tokens": int(attach["cached_tokens"]),
            "shared_published_prefix_tokens": int(attach["published_prefix_tokens"]),
            "shared_registry_hits_delta": int(attach["registry_hits_delta"]),
        })
    return pairs, by_session_delta


def session_means(rows: dict[str, dict[str, Any]]) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for row in rows.values():
        by.setdefault(int(row["session_index"]), []).append(float(row[REQUEST_FIELD]))
    return {s: st.fmean(vals) for s, vals in sorted(by.items())}


def historical_config_comparison(native: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    nc = native["configuration"]
    hc = historical["configuration"]
    mismatches = []
    missing_on_historical = []
    missing_on_native = []
    for key in PRIMARY_CONFIG_KEYS:
        if key in nc and key in hc:
            if nc[key] != hc[key]:
                mismatches.append({"key": key, "native": nc[key], "historical": hc[key]})
        elif key in nc:
            missing_on_historical.append(key)
        elif key in hc:
            missing_on_native.append(key)
    return {
        "primary_keys_compared": [key for key in PRIMARY_CONFIG_KEYS if key in nc and key in hc],
        "primary_key_mismatches": mismatches,
        "primary_keys_missing_on_historical": missing_on_historical,
        "primary_keys_missing_on_native": missing_on_native,
    }


def historical_comparison(native: dict[str, Any], historical: dict[str, Any] | None) -> dict[str, Any] | None:
    if historical is None:
        return None
    validate_historical_schema("historical", historical)
    validate_task("historical", historical)
    require_no_skips("historical", historical)
    dataset_cmp = compare_dataset(native, historical, "historical")
    cfg_cmp = historical_config_comparison(native, historical)
    hrows = rows_by_request("historical", historical)
    nrows = rows_by_request("native", native)
    if set(hrows) != set(nrows):
        die(f"historical request set differs from native: missing={sorted(set(nrows)-set(hrows))} extra={sorted(set(hrows)-set(nrows))}")
    deltas = []
    generated_same = 0
    for rid in sorted(nrows):
        nr = nrows[rid]
        hr = hrows[rid]
        for field in ("session_index", "turn_index", "prompt_tokens", "prefix_tokens", "question_tokens", "gold"):
            if nr.get(field) != hr.get(field):
                die(f"historical row {rid} mismatch for {field}: native={nr.get(field)!r} historical={hr.get(field)!r}")
        deltas.append(float(nr[REQUEST_FIELD]) - float(hr[REQUEST_FIELD]))
        if nr.get("generated_token_ids") == hr.get("generated_token_ids"):
            generated_same += 1
    return {
        "historical_arm": historical.get("arm"),
        "historical_schema": historical.get("schema"),
        "requests": len(deltas),
        "native_minus_historical_request_mean_delta": st.fmean(deltas),
        "max_abs_request_delta": max(abs(x) for x in deltas) if deltas else None,
        "generated_token_ids_equal_requests": generated_same,
        "dataset_comparison": dataset_cmp,
        "configuration_comparison": cfg_cmp,
        "note": "descriptive comparison of this Native receipt with the optional historical Native receipt",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--shared", type=Path, required=True)
    ap.add_argument("--historical", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-sessions", type=int, default=DEFAULT_EXPECT_SESSIONS)
    ap.add_argument("--expected-requests", type=int, default=DEFAULT_EXPECT_REQUESTS)
    ap.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = ap.parse_args()

    if args.bootstrap_draws <= 0:
        die("--bootstrap-draws must be positive")
    native = load_receipt(args.native)
    shared = load_receipt(args.shared)
    historical = load_receipt(args.historical) if args.historical else None

    validate_input_schema_arm("native", native, NATIVE_ARM)
    validate_input_schema_arm("shared", shared, SHARED_ARM)
    validate_task("native", native)
    validate_task("shared", shared)
    require_no_skips("native", native)
    require_no_skips("shared", shared)
    dataset_comparison = compare_dataset(native, shared, "shared")
    checked_config = compare_config(native, shared)

    native_rows = rows_by_request("native", native)
    shared_rows = rows_by_request("shared", shared)
    validate_counts("native", native_rows, args.expected_sessions, args.expected_requests)
    validate_counts("shared", shared_rows, args.expected_sessions, args.expected_requests)

    publications = shared_publications_by_session(shared, args.expected_sessions)
    pairs, by_session_delta = build_pairs(native_rows, shared_rows, publications)
    if len(by_session_delta) != args.expected_sessions:
        die(f"paired data has {len(by_session_delta)} sessions, expected {args.expected_sessions}")
    session_delta_rows = [
        {
            "session_index": s,
            "requests": len(vals),
            "mean_delta_shared_minus_native": st.fmean(vals),
        }
        for s, vals in sorted(by_session_delta.items())
    ]
    session_deltas = [row["mean_delta_shared_minus_native"] for row in session_delta_rows]

    # Validate optional historical input before opening or writing any output
    # artifacts, so failed provenance never leaves a partial analysis directory.
    hist_cmp = historical_comparison(native, historical)

    native_session_means = session_means(native_rows)
    shared_session_means = session_means(shared_rows)
    native_f1s = [float(r[REQUEST_FIELD]) for r in native_rows.values()]
    shared_f1s = [float(r[REQUEST_FIELD]) for r in shared_rows.values()]
    deltas = [p["delta_shared_minus_native"] for p in pairs]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.output_dir / "all_pairs.csv"
    session_csv = args.output_dir / "session_deltas.csv"
    analysis_json = args.output_dir / "full_session_quality.json"
    manifest_json = args.output_dir / "artifact_manifest.json"
    for path in (pair_csv, session_csv, analysis_json, manifest_json):
        if path.exists():
            die(f"refusing to overwrite output {path}")

    write_csv(pair_csv, pairs)
    with session_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("session_index", "requests", "mean_delta_shared_minus_native"))
        writer.writeheader()
        writer.writerows(session_delta_rows)

    inputs = {
        "native": {"path": rel(args.native), "sha256": sha256_path(args.native)},
        "shared": {"path": rel(args.shared), "sha256": sha256_path(args.shared)},
    }
    if args.historical:
        inputs["historical"] = {"path": rel(args.historical), "sha256": sha256_path(args.historical)}

    out = {
        "schema": SCHEMA,
        "label": "descriptive paired full-session SCBench QA quality analysis; no equivalence gate",
        "task": QA_TASK,
        "metric": "answer_f1",
        "inputs": inputs,
        "dataset": dataset_comparison,
        "arms": {"native": native.get("arm"), "shared": shared.get("arm")},
        "configuration_checked_equal": checked_config,
        "counts": {"sessions": args.expected_sessions, "requests": args.expected_requests},
        "request_weighted": {
            "native_mean": st.fmean(native_f1s),
            "shared_mean": st.fmean(shared_f1s),
            "delta_shared_minus_native": st.fmean(deltas),
            "note": "request-weighted mean, matching the Fig. 4 style aggregation",
        },
        "equal_session": {
            "native_mean": st.fmean(native_session_means.values()),
            "shared_mean": st.fmean(shared_session_means.values()),
            "delta_shared_minus_native": st.fmean(session_deltas),
            "se_delta_over_sessions": sem(session_deltas),
            "bootstrap95_ci_delta": bootstrap_ci_session_deltas(session_deltas, args.bootstrap_draws, args.bootstrap_seed),
        },
        "per_question": {"wins_shared": wins, "losses_shared": losses, "ties": ties, "note": "descriptive counts over paired questions"},
        "historical_native_comparison": hist_cmp,
        "outputs": {"all_pairs_csv": "all_pairs.csv", "session_deltas_csv": "session_deltas.csv"},
    }
    analysis_json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    manifest = {
        "schema": "kvladder-conf-full-session-quality-manifest-v1",
        "producer": {"path": rel(Path(__file__).resolve()), "sha256": sha256_path(Path(__file__).resolve())},
        "immutable_inputs": inputs,
        "outputs": {
            "full_session_quality.json": sha256_path(analysis_json),
            "all_pairs.csv": sha256_path(pair_csv),
            "session_deltas.csv": sha256_path(session_csv),
        },
        "regenerate": "python {} --native {} --shared {}{} --output-dir {} --expected-sessions {} --expected-requests {}".format(
            rel(Path(__file__).resolve()),
            rel(args.native),
            rel(args.shared),
            f" --historical {rel(args.historical)}" if args.historical else "",
            rel(args.output_dir),
            args.expected_sessions,
            args.expected_requests,
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "requests": len(pairs),
        "sessions": len(session_delta_rows),
        "native_request_mean": out["request_weighted"]["native_mean"],
        "shared_request_mean": out["request_weighted"]["shared_mean"],
        "delta_session_mean": out["equal_session"]["delta_shared_minus_native"],
        "ci95": out["equal_session"]["bootstrap95_ci_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
