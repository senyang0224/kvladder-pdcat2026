#!/usr/bin/env python3
"""Run the KVLadder conference fallback audit on one live Tangram engine.

This is an additive audit harness.  It reuses ``run_conf_arm`` for LLM
construction, lookup construction, sampling defaults, engine access, and block
ownership summaries, but it does not modify that historical runner.  Importing
this module is CPU-only; ``main``/``run`` starts vLLM and must stay behind the
GPU-idle gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src" / "refusal") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "refusal"))

from kvladder_tangram_ragged_prefix import (  # noqa: E402
    TangramRaggedPrefixCache,
    TangramRaggedPrefixLookup,
    digest_token_ids,
)
from scripts.audit import run_arm as base  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

SCHEMA = "kvladder-conf-audit-fallback-receipt-v1"
ANALYSIS_SCHEMA = "kvladder-conf-audit-fallback-analysis-v1"
SHARED_ARM = "kvladder-fp16-shared"
EXPECTED_PROBES = (
    "token_mismatch",
    "short_prefix",
    "exact_prefix_no_private_token",
    "malformed_spec",
    "unsupported_retained_rule_origin",
    "valid_shared_positive_control",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def git_dirty(path: Path) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:
        return None


def _base_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        arm=SHARED_ARM,
        scenario="burst",
        phase=args.phase,
        trace=args.trace,
        branches=1,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        repetition=args.repetition,
        commit_on_success=False,
        microbench_ops=0,
        identity_capture=False,
        identity_payload_hash=False,
        identity_capture_strict=False,
        commit_private_tokens=64,
        commit_probe="none",
        expect_gpu_uuid=args.expect_gpu_uuid,
        max_wall_s=args.max_wall_s,
        admission_cap=0,
        output=args.output,
    )


def _spec_dict(lookup: TangramRaggedPrefixLookup) -> dict[str, object]:
    return asdict(lookup)


def _sampling_from_extra(
    trace: Mapping[str, Any], extra: Mapping[str, object] | None, max_tokens: int
) -> SamplingParams:
    gen = trace["generation"]
    return SamplingParams(
        temperature=float(gen["temperature"]),
        max_tokens=int(max_tokens),
        stop=list(gen["stop"]),
        ignore_eos=True,
        extra_args=None if extra is None else dict(extra),
    )


def _sampling_for_lookup(
    trace: Mapping[str, Any], lookup: TangramRaggedPrefixLookup, max_tokens: int
) -> SamplingParams:
    return base.sampling(trace, lookup=lookup, max_tokens=max_tokens, ignore_eos=True)


def _queue_ids(queue: Any) -> list[str]:
    try:
        return sorted(str(getattr(r, "request_id", r)) for r in list(queue))
    except Exception:
        return []


def _request_ids(scheduler: Any) -> dict[str, list[str]]:
    return {
        "requests": sorted(str(v) for v in getattr(scheduler, "requests", {}).keys()),
        "running": _queue_ids(getattr(scheduler, "running", [])),
        "waiting": _queue_ids(getattr(scheduler, "waiting", [])),
    }


def _registry_state(registry: TangramRaggedPrefixCache) -> dict[str, object]:
    tel = registry.telemetry()
    return {
        "entries": int(tel["entries"]),
        "live_requests": int(tel["live_requests"]),
        "private_fallbacks": int(tel.get("private_fallbacks", 0)),
        "publication_refusals": int(tel.get("publication_refusals", 0)),
        "attaches": int(tel.get("attaches", 0)),
    }


def _base_entry_ids(registry: TangramRaggedPrefixCache, lookup: TangramRaggedPrefixLookup) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in registry.entry(lookup).snapshot.physical_block_ids)
    except Exception:
        return ()


def _base_ref_summary(registry: TangramRaggedPrefixCache, lookup: TangramRaggedPrefixLookup) -> dict[str, object]:
    ids = _base_entry_ids(registry, lookup)
    if not ids:
        return {"pages": 0, "min": None, "max": None}
    vals = tuple(int(v) for v in registry.manager.ragged_prefix_refcounts(ids))
    return {"pages": len(ids), "min": min(vals), "max": max(vals)}


def _events_for(registry: TangramRaggedPrefixCache, request_id: str) -> list[dict[str, object]]:
    return [
        e
        for e in registry.telemetry().get("events", [])
        if str(e.get("request_id", "")) == request_id
    ]


def _drain_one(llm: Any, request_id: str, *, deadline: float) -> dict[str, object]:
    generated_ids: list[int] = []
    generated = ""
    num_cached_tokens = 0
    finish_reason = None
    first_token_at = None
    finished_at = None
    steps = 0
    while time.perf_counter() < deadline:
        outputs = llm.llm_engine.step()
        steps += 1
        for out in outputs:
            if str(out.request_id) != request_id or not out.outputs:
                continue
            cand = out.outputs[0]
            ids = [int(v) for v in cand.token_ids]
            if ids and first_token_at is None:
                first_token_at = time.perf_counter()
            generated_ids = ids
            generated = str(cand.text)
            num_cached_tokens = int(out.num_cached_tokens or 0)
            if out.finished:
                finished_at = time.perf_counter()
                finish_reason = str(cand.finish_reason)
                return {
                    "completed": True,
                    "steps": steps,
                    "first_token_at": first_token_at,
                    "finished_at": finished_at,
                    "generated": generated,
                    "generated_token_ids": generated_ids,
                    "output_tokens": len(generated_ids),
                    "num_cached_tokens": num_cached_tokens,
                    "finish_reason": finish_reason,
                }
        if not llm.llm_engine.has_unfinished_requests():
            break
    raise RuntimeError(f"request {request_id} did not finish")


def _run_probe(
    *,
    llm: Any,
    scheduler: Any,
    block_pool: Any,
    registry: TangramRaggedPrefixCache,
    base_lookup: TangramRaggedPrefixLookup,
    trace: Mapping[str, Any],
    name: str,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    deadline: float,
) -> dict[str, object]:
    before_events = len(registry.telemetry().get("events", []))
    before = {
        "free_blocks": int(block_pool.get_num_free_blocks()),
        "request_blocks": base.request_block_counts(scheduler),
        "scheduler": _request_ids(scheduler),
        "registry": _registry_state(registry),
        "base_refcounts": _base_ref_summary(registry, base_lookup),
    }
    added_at = time.perf_counter()
    add_error = None
    completion: dict[str, object] | None = None
    try:
        llm.llm_engine.add_request(
            request_id,
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params,
        )
        completion = _drain_one(llm, request_id, deadline=deadline)
    except Exception as exc:
        add_error = f"{type(exc).__name__}: {exc}"
    after_events_all = registry.telemetry().get("events", [])
    probe_events = list(after_events_all[before_events:])
    request_events = _events_for(registry, request_id)
    after = {
        "free_blocks": int(block_pool.get_num_free_blocks()),
        "request_blocks": base.request_block_counts(scheduler),
        "scheduler": _request_ids(scheduler),
        "registry": _registry_state(registry),
        "base_refcounts": _base_ref_summary(registry, base_lookup),
    }
    cached = int((completion or {}).get("num_cached_tokens", 0) or 0)
    shared_attached = any(e.get("event") == "lookup_hit_attach" for e in request_events)
    private_fallback = any(e.get("event") == "private_fallback" for e in request_events)
    publication_refused = any(e.get("event") == "publication_refused" for e in request_events)
    return {
        "name": name,
        "request_id": request_id,
        "add_error": add_error,
        "same_request_completed": bool(completion and completion.get("completed")),
        "completed": bool(completion and completion.get("completed")),
        "elapsed_ms": (time.perf_counter() - added_at) * 1000.0,
        "num_cached_tokens": cached,
        "cached_tokens_is_zero": cached == 0,
        "cached_tokens_is_context": cached == int(trace["context_tokens"]),
        "shared_attached": shared_attached,
        "private_fallback": private_fallback,
        "publication_refused": publication_refused,
        "before": before,
        "after": after,
        "registry_events_for_request": request_events,
        "registry_events_since_probe_start": probe_events,
        "completion": completion,
    }


def _int_or_none(value: object) -> int | None:
    return value if type(value) is int else None


def _ref_tuple(state: Mapping[str, object]) -> tuple[int, int, int] | None:
    refs = state.get("base_refcounts")
    if not isinstance(refs, Mapping):
        return None
    pages = _int_or_none(refs.get("pages"))
    min_ref = _int_or_none(refs.get("min"))
    max_ref = _int_or_none(refs.get("max"))
    if pages is None or pages <= 0 or min_ref is None or max_ref is None:
        return None
    return (pages, min_ref, max_ref)


def _request_blocks_empty(state: Mapping[str, object], request_id: str) -> bool | None:
    blocks = state.get("request_blocks")
    if not isinstance(blocks, Mapping):
        return None
    return request_id not in blocks


def analyze_receipt(receipt: Mapping[str, Any]) -> dict[str, object]:
    probes = {str(p.get("name")): p for p in receipt.get("probes", [])}
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    def common_probe_checks(name: str, p: Mapping[str, object]) -> None:
        request_id = str(p.get("request_id"))
        check(
            f"{name}.same_request_completed",
            bool(p.get("same_request_completed")),
            "request finished instead of being dropped",
        )
        check(f"{name}.no_add_error", p.get("add_error") is None, str(p.get("add_error")))
        before = p.get("before")
        after = p.get("after")
        check(f"{name}.before_state_present", isinstance(before, Mapping), str(before))
        check(f"{name}.after_state_present", isinstance(after, Mapping), str(after))
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            return
        before_free = _int_or_none(before.get("free_blocks"))
        after_free = _int_or_none(after.get("free_blocks"))
        check(
            f"{name}.free_blocks_observed",
            before_free is not None and after_free is not None,
            f"before={before_free}, after={after_free}",
        )
        before_refs = _ref_tuple(before)
        after_refs = _ref_tuple(after)
        check(
            f"{name}.base_refcounts_observed",
            before_refs is not None and after_refs is not None,
            f"before={before.get('base_refcounts')}, after={after.get('base_refcounts')}",
        )
        reg = after.get("registry")
        sched = after.get("scheduler")
        check(f"{name}.after_registry_present", isinstance(reg, Mapping), str(reg))
        check(f"{name}.after_scheduler_present", isinstance(sched, Mapping), str(sched))
        if isinstance(reg, Mapping):
            live = _int_or_none(reg.get("live_requests"))
            check(f"{name}.live_registry_observed", live is not None, str(reg))
            check(f"{name}.no_live_registry_request", live == 0, str(reg))
        if isinstance(sched, Mapping):
            requests = sched.get("requests")
            check(f"{name}.scheduler_requests_observed", isinstance(requests, list), str(sched))
            check(
                f"{name}.not_in_scheduler",
                isinstance(requests, list) and request_id not in set(requests),
                str(sched),
            )
        owned_empty = _request_blocks_empty(after, request_id)
        check(
            f"{name}.request_blocks_released",
            owned_empty is True,
            str(after.get("request_blocks")),
        )

    for name in EXPECTED_PROBES:
        p = probes.get(name)
        check(f"{name}.present", p is not None, "probe exists")
        if isinstance(p, Mapping):
            common_probe_checks(name, p)

    for name in EXPECTED_PROBES[:4]:
        p = probes.get(name) or {}
        check(
            f"{name}.private_fallback",
            bool(p.get("private_fallback")),
            "expected request-level private fallback event",
        )
        check(
            f"{name}.no_shared_attach",
            not bool(p.get("shared_attached")),
            "must not attach mismatched/invalid request",
        )
        check(
            f"{name}.zero_cached_tokens",
            bool(p.get("cached_tokens_is_zero")),
            str(p.get("num_cached_tokens")),
        )
        before = p.get("before") if isinstance(p, Mapping) else None
        after = p.get("after") if isinstance(p, Mapping) else None
        before_free = _int_or_none(before.get("free_blocks")) if isinstance(before, Mapping) else None
        after_free = _int_or_none(after.get("free_blocks")) if isinstance(after, Mapping) else None
        before_refs = _ref_tuple(before) if isinstance(before, Mapping) else None
        after_refs = _ref_tuple(after) if isinstance(after, Mapping) else None
        check(
            f"{name}.free_blocks_unchanged_by_refusal",
            before_free is not None and before_free == after_free,
            f"before={before_free}, after={after_free}",
        )
        check(
            f"{name}.base_refs_unchanged_by_refusal",
            before_refs is not None and before_refs == after_refs,
            f"before={before_refs}, after={after_refs}",
        )

    p = probes.get("unsupported_retained_rule_origin") or {}
    check(
        "unsupported_retained_rule_origin.publication_refused",
        bool(p.get("publication_refused")),
        "origin should finish but not publish unsupported retained rule",
    )
    check(
        "unsupported_retained_rule_origin.zero_cached_tokens",
        bool(p.get("cached_tokens_is_zero")),
        str(p.get("num_cached_tokens")),
    )

    p = probes.get("valid_shared_positive_control") or {}
    check(
        "valid_shared_positive_control.shared_attached",
        bool(p.get("shared_attached")),
        "positive control must hit published base",
    )
    check(
        "valid_shared_positive_control.context_cached_tokens",
        bool(p.get("cached_tokens_is_context")),
        str(p.get("num_cached_tokens")),
    )
    before = p.get("before") if isinstance(p, Mapping) else None
    after = p.get("after") if isinstance(p, Mapping) else None
    before_refs = _ref_tuple(before) if isinstance(before, Mapping) else None
    after_refs = _ref_tuple(after) if isinstance(after, Mapping) else None
    check(
        "valid_shared_positive_control.base_refs_returned_after_finish",
        before_refs is not None and before_refs == after_refs,
        f"before={before_refs}, after={after_refs}",
    )
    owned_empty = (
        _request_blocks_empty(after, str(p.get("request_id")))
        if isinstance(after, Mapping)
        else None
    )
    check(
        "valid_shared_positive_control.owner_blocks_released",
        owned_empty is True,
        str(after.get("request_blocks") if isinstance(after, Mapping) else None),
    )

    drain = receipt.get("drain") or {}
    initial = _int_or_none(drain.get("initial_free_blocks")) if isinstance(drain, Mapping) else None
    after_drain = _int_or_none(drain.get("free_blocks_after_drain")) if isinstance(drain, Mapping) else None
    check(
        "drain.free_blocks_observed",
        initial is not None and after_drain is not None,
        str(drain),
    )
    check(
        "drain.back_to_initial",
        initial is not None and after_drain is not None and after_drain == initial,
        str(drain),
    )
    passed = all(bool(c["pass"]) for c in checks)
    return {"schema": ANALYSIS_SCHEMA, "passed": passed, "checks": checks}


def run(args: argparse.Namespace) -> dict[str, object]:
    smi_before = base.nvidia_smi()
    trace = json.loads(args.trace.read_text())
    prefix_ids = [int(v) for v in trace["prefix_token_ids"]]
    if digest_token_ids(prefix_ids) != trace["token_prefix_digest"]:
        raise RuntimeError("trace token prefix digest mismatch")
    ns = _base_namespace(args)
    llm, cfg = base.build_llm(ns, trace)
    scheduler = base.engine_core(llm).scheduler
    manager = scheduler.kv_cache_manager
    block_pool = manager.block_pool
    block_size = int(manager.ragged_block_size)
    registry = TangramRaggedPrefixCache()
    scheduler.set_kvladder_ragged_prefix_cache(registry)
    lookup = base.make_lookup(trace, block_size, float(cfg["eviction_ratio"]))
    initial_free = int(block_pool.get_num_free_blocks())
    deadline = time.perf_counter() + float(args.max_wall_s)
    gpu = base.gpu_identity()
    if args.expect_gpu_uuid and gpu.get("uuid") and args.expect_gpu_uuid not in str(gpu["uuid"]):
        raise RuntimeError(f"GPU UUID {gpu['uuid']} does not match expected {args.expect_gpu_uuid}")

    q0 = [int(v) for v in trace["requests"][0]["query_token_ids"]]
    q1 = [int(v) for v in trace["requests"][1]["query_token_ids"]]
    max_tokens = int(args.max_tokens)

    # Publish one valid shared base first.  This is not a probe: it establishes
    # the resident object every expected-fallback request either refuses to use
    # or the positive control attaches.
    llm.llm_engine.add_request(
        "audit-fallback-publish-base",
        TokensPrompt(prompt_token_ids=prefix_ids),
        _sampling_for_lookup(trace, lookup, 1),
    )
    base_completion = _drain_one(llm, "audit-fallback-publish-base", deadline=deadline)
    if not registry.telemetry().get("entry_rows"):
        raise RuntimeError("valid base did not publish before fallback probes")
    base_after_publish = {
        "completion": base_completion,
        "free_blocks": int(block_pool.get_num_free_blocks()),
        "registry": _registry_state(registry),
        "base_refcounts": _base_ref_summary(registry, lookup),
    }

    unsupported_lookup = replace(lookup, retained_position_rule="unsupported-selector-v1")
    malformed_extra = {TangramRaggedPrefixCache.EXTRA_ARG: "not-a-mapping"}
    probes: list[dict[str, object]] = []
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="token_mismatch",
            request_id="audit-fallback-token-mismatch",
            prompt_token_ids=[prefix_ids[0] + 1] + prefix_ids[1:] + q0,
            sampling_params=_sampling_for_lookup(trace, lookup, max_tokens),
            deadline=deadline,
        )
    )
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="short_prefix",
            request_id="audit-fallback-short-prefix",
            prompt_token_ids=prefix_ids[:-1],
            sampling_params=_sampling_for_lookup(trace, lookup, max_tokens),
            deadline=deadline,
        )
    )
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="exact_prefix_no_private_token",
            request_id="audit-fallback-exact-prefix",
            prompt_token_ids=prefix_ids,
            sampling_params=_sampling_for_lookup(trace, lookup, max_tokens),
            deadline=deadline,
        )
    )
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="malformed_spec",
            request_id="audit-fallback-malformed-spec",
            prompt_token_ids=prefix_ids + q0,
            sampling_params=_sampling_from_extra(trace, malformed_extra, max_tokens),
            deadline=deadline,
        )
    )
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="unsupported_retained_rule_origin",
            request_id="audit-fallback-unsupported-origin",
            prompt_token_ids=prefix_ids,
            sampling_params=_sampling_for_lookup(trace, unsupported_lookup, max_tokens),
            deadline=deadline,
        )
    )
    probes.append(
        _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="valid_shared_positive_control",
            request_id="audit-fallback-valid-shared",
            prompt_token_ids=prefix_ids + q1,
            sampling_params=_sampling_for_lookup(trace, lookup, max_tokens),
            deadline=deadline,
        )
    )

    # Optional opportunistic unaligned-origin probe.  It is evidence only if the
    # actual runtime produces a partial-page publish boundary; the harness never
    # fabricates such a snapshot.
    unaligned: dict[str, object] | None = None
    if args.try_unaligned_origin:
        unaligned_lookup = replace(lookup, prefix_token_count=len(prefix_ids) - 1, token_prefix_digest=digest_token_ids(prefix_ids[:-1]))
        unaligned = _run_probe(
            llm=llm,
            scheduler=scheduler,
            block_pool=block_pool,
            registry=registry,
            base_lookup=lookup,
            trace=trace,
            name="unaligned_origin_opportunistic",
            request_id="audit-fallback-unaligned-origin",
            prompt_token_ids=prefix_ids[:-1],
            sampling_params=_sampling_for_lookup(trace, unaligned_lookup, max_tokens),
            deadline=deadline,
        )
        probes.append(unaligned)

    before_drain = int(block_pool.get_num_free_blocks())
    drain_error = None
    try:
        registry.drain()
    except Exception as exc:  # retain evidence rather than hiding a leak
        drain_error = f"{type(exc).__name__}: {exc}"
    scheduler.set_kvladder_ragged_prefix_cache(None)
    after_drain = int(block_pool.get_num_free_blocks())

    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "phase": args.phase,
        "seed": int(args.seed),
        "repetition": int(args.repetition),
        "trace": {
            "path": str(args.trace),
            "content_sha256": str(trace["content_sha256"]),
            "context_tokens": int(trace["context_tokens"]),
        },
        "configuration": {
            **cfg,
            "arm": SHARED_ARM,
            "kv_cache_memory_bytes": int(args.kv_cache_memory_bytes),
            "max_model_len": int(args.max_model_len),
            "max_num_seqs": int(args.max_num_seqs),
            "max_tokens_per_probe": max_tokens,
            "single_engine_serial_probes": True,
        },
        "base_publication": base_after_publish,
        "probes": probes,
        "drain": {
            "initial_free_blocks": initial_free,
            "free_blocks_before_drain": before_drain,
            "free_blocks_after_drain": after_drain,
            "drain_error": drain_error,
        },
        "registry_after_drain": registry.telemetry(),
        "environment": {
            "python": platform.python_version(),
            "repo_commit": git_head(REPO),
            "repo_dirty": git_dirty(REPO),
            "runner_wrapper_path": str(Path(__file__).resolve().relative_to(REPO)),
            "runner_wrapper_sha256": sha256(Path(__file__).resolve()),
            "base_runner_path": "scripts/audit/run_arm.py",
            "base_runner_sha256": sha256(REPO / "scripts/audit/run_arm.py"),
            "registry_module_path": str(Path(sys.modules["kvladder_tangram_ragged_prefix"].__file__).resolve()),
            "registry_module_sha256": sha256(Path(sys.modules["kvladder_tangram_ragged_prefix"].__file__).resolve()),
            "tangram_root": str(Path(base.vllm.__file__).resolve().parents[1]),
            "tangram_overlay_diff_sha256": base.diff_sha256(Path(base.vllm.__file__).resolve().parents[1]),
            "tangram_overlay_dirty": base.git_dirty(Path(base.vllm.__file__).resolve().parents[1]),
            "gpu": gpu,
            "nvidia_smi_before": smi_before,
            "nvidia_smi_after": base.nvidia_smi(),
            "cli": sys.argv,
            "hostname": platform.node(),
        },
        "claim_boundary": "Audit-only fallback harness: expected sharing refusals must complete the same request privately; unknown failures are not downgraded. Positive control must attach the already-published Shared base.",
    }
    analysis = analyze_receipt(receipt)
    receipt["analysis"] = analysis
    receipt["passed"] = bool(analysis["passed"] and drain_error is None)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, required=True, choices=(17, 18, 19))
    p.add_argument("--repetition", type=int, required=True, choices=(0, 1, 2))
    p.add_argument("--phase", default="audit-fallback")
    p.add_argument("--expect-gpu-uuid", default="")
    p.add_argument("--max-wall-s", type=float, default=1800.0)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--try-unaligned-origin", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(args.output),
                "passed": receipt.get("passed"),
                "analysis": receipt.get("analysis"),
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
    )
    if not receipt.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
