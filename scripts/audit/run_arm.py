#!/usr/bin/env python3
"""Run one representation-matched arm of the KVLadder conference campaign.

Arms
  full-fp16-apc        stock dense FP16 prefix cache (page_group_size=None, APC on)
  ragged-fp16-copy     Tangram ragged FP16, no registry: every branch owns its own base
  kvladder-fp16-shared Tangram ragged FP16 + KVLadder registry: one shared immutable base
  fp8-apc              stock uniform FP8 prefix cache (deployment reference)

Scenarios
  burst      B simultaneous branches of one trace (RQ2 memory, RQ3 serving, RQ5)
  lifecycle  2 live branches + committed branch + aborted branch + injected preemption
             + identity-mismatch rejection (RQ1 correctness; ragged arms only)

The measurement loop is adapted from the historical P4/P5 runners (same per-request
timing, occupancy sampling and drain checks) and adds: one KV budget knob for all
arms (kv_cache_memory_bytes), --branches, per-request block ownership samples for
the memory decomposition, GPU UUID + nvidia-smi before/after, torch peak memory,
an optional refcount/attach microbenchmark (RQ4) and fault injection for RQ1.
No mechanism code is changed by this script.
"""

from __future__ import annotations

import argparse
import faulthandler
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "recency"))
from dataclasses import replace as dc_replace  # noqa: E402

import numpy as np  # noqa: E402

from scripts.audit.identity_capture import (  # noqa: E402
    IdentityCapture,
    retained_positions_digest_from_base_tail,
)

import kvladder_tangram_ragged_prefix as kvladder_registry_module  # noqa: E402
from kvladder_tangram_ragged_prefix import (  # noqa: E402
    TangramRaggedPrefixCache,
    TangramRaggedPrefixEntry,
    TangramRaggedPrefixError,
    TangramRaggedPrefixLookup,
    TangramRaggedPrefixSnapshot,
)

SCHEMA = "kvladder-conf-arm-receipt-v2"
# V2 harness amendment 1 (2026-09-06): the frozen-copy control arm's admission gating must leave
# room for Tangram's admission reservation (scheduler_reserve_full_isl reserves the *uncompressed*
# sequence length in KVCacheManager._full_sequence_fits).  Attempt 1 gated only on the copy pages
# and livelocked under the constrained RQ3 budget (all copies published, no branch admissible).
HARNESS_AMENDMENT = "KVLADDER_CONF_V2_HARNESS_AMENDMENT_1_FROZEN_ADMISSION_RESERVATION"
RAGGED_ARMS = ("ragged-fp16-copy", "frozen-fp16-copy", "kvladder-fp16-shared")
REGISTRY_ARMS = ("frozen-fp16-copy", "kvladder-fp16-shared")
APC_ARMS = ("full-fp16-apc", "fp8-apc")
RETAINED_RATIO = 0.296875  # frozen historical ratio; block-aligned at 1K/32K/64K
SINK_TOKENS = 16
COMPRESSION_SCORER = "streamingllm"


# --------------------------------------------------------------------------- utils
def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def quality(generated: str, answers: list[str]) -> dict[str, object]:
    gen = normalize(generated)
    cands = [normalize(a) for a in answers]
    contains = any(c and c in gen for c in cands)
    exact = any(c == gen for c in cands)
    gen_tokens = gen.split()
    best = 0.0
    for c in cands:
        gold = c.split()
        common = sum((Counter(gen_tokens) & Counter(gold)).values())
        if common:
            p = common / len(gen_tokens)
            r = common / len(gold)
            best = max(best, 2 * p * r / (p + r))
    return {"answer_contains": contains, "answer_exact_match": exact, "answer_f1": best}


def digest_token_ids(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for t in token_ids:
        digest.update(int(t).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def tangram_ratio(context_tokens: int) -> tuple[int, float, float]:
    target = int(context_tokens * RETAINED_RATIO)
    body = target - SINK_TOKENS
    if body <= 0:
        raise RuntimeError("retained target must exceed protected sink tokens")
    keep = body / context_tokens
    return target, keep, 1.0 - keep


def nvidia_smi() -> dict[str, object]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        return {"t": time.time(), "gpus": out.split("\n"), "compute_apps": apps.split("\n") if apps else []}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"t": time.time(), "error": str(exc)}


def gpu_identity() -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = getattr(props, "uuid", None)
    return {
        "name": props.name,
        "uuid": str(uuid) if uuid is not None else None,
        "total_memory_bytes": int(props.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compute_capability": [props.major, props.minor],
    }


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        return bool(out.strip())
    except Exception:
        return None


def diff_sha256(path: Path) -> str | None:
    try:
        diff = subprocess.check_output(["git", "-C", str(path), "diff"], text=True)
        return hashlib.sha256(diff.encode()).hexdigest()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------- engine
def engine_core(llm: LLM) -> Any:
    return llm.llm_engine.engine_core.engine_core


def cache_bytes(scheduler: Any) -> int:
    config = scheduler.kv_cache_manager.kv_cache_config
    return sum(int(t.size) for t in config.kv_cache_tensors)


def make_lookup(trace: Mapping[str, Any], block_size: int, eviction_ratio: float) -> TangramRaggedPrefixLookup:
    model = str(trace["model"])
    return TangramRaggedPrefixLookup(
        token_prefix_digest=str(trace["token_prefix_digest"]),
        prefix_token_count=int(trace["context_tokens"]),
        model_id=model,
        model_config_digest="llama-3.1-8b-instruct-fp16",
        tokenizer_id=model,
        position_contract="absolute-causal-rope-v1",
        kv_dtype="float16",
        compressor_id=f"tangram-{COMPRESSION_SCORER}",
        compressor_revision="44ed97e8",
        compression_policy_digest=(
            f"evict_body={eviction_ratio},retain_total={RETAINED_RATIO},"
            f"chunk={trace['context_tokens']},window=32,sink={SINK_TOKENS}"
        ),
        layout_contract=f"tangram-3d-ragged-block{block_size}",
        retained_position_rule="streamingllm-sink-recent-v1",
        sink_tokens=SINK_TOKENS,
    )


def sampling(trace: Mapping[str, Any], *, lookup: TangramRaggedPrefixLookup | None, max_tokens: int | None = None, commit: bool = False, ignore_eos: bool = False) -> SamplingParams:
    gen = trace["generation"]
    extra = None
    if lookup is not None:
        extra = {TangramRaggedPrefixCache.EXTRA_ARG: {**asdict(lookup), "commit_on_success": bool(commit)}}
    return SamplingParams(
        temperature=float(gen["temperature"]),
        max_tokens=int(max_tokens or gen["max_new_tokens"]),
        stop=list(gen["stop"]),
        ignore_eos=bool(ignore_eos),
        extra_args=extra,
    )


def drain_one(llm: LLM, request_id: str, *, limit: int = 100000) -> None:
    for _ in range(limit):
        for out in llm.llm_engine.step():
            if str(out.request_id) == request_id and out.finished:
                return
        if not llm.llm_engine.has_unfinished_requests():
            break
    raise RuntimeError(f"request {request_id} did not finish")


def build_llm(args: argparse.Namespace, trace: Mapping[str, Any]) -> tuple[LLM, dict[str, object]]:
    ctx = int(trace["context_tokens"])
    common = dict(
        model=str(trace["model"]),
        tokenizer=str(trace["tokenizer"]),
        dtype="float16",
        kv_cache_memory_bytes=int(args.kv_cache_memory_bytes),
        max_model_len=int(args.max_model_len),
        enforce_eager=not bool(getattr(args, "cuda_graphs", False)),  # campaigns pass no attribute -> eager, unchanged
        max_num_seqs=int(args.max_num_seqs),
        max_num_batched_tokens=ctx,
        seed=int(args.seed),
    )
    cfg: dict[str, object] = {}
    if args.arm in RAGGED_ARMS:
        target, keep, evict = tangram_ratio(ctx)
        llm = LLM(
            **common,
            kv_cache_dtype="auto",
            enable_prefix_caching=True,  # Tangram resolves this to False in ragged mode (gap receipt)
            page_group_size=4,
            compression_ratio=evict,
            compression_scorer=COMPRESSION_SCORER,
            compression_budget_scope="layer",
            compression_chunk_size=ctx,
            compression_floor_min=64,
            compression_window_size=32,
            compression_n_sink_tokens=SINK_TOKENS,
            scheduler_reserve_full_isl=True,
        )
        cfg.update(
            retained_ratio=RETAINED_RATIO,
            target_retained_tokens=target,
            body_keep_ratio=keep,
            eviction_ratio=evict,
            compression_chunk_size=ctx,
            compression_scorer=COMPRESSION_SCORER,
            compression_sink_tokens=SINK_TOKENS,
            page_group_size=4,
            requested_enable_prefix_caching=True,
        )
    else:
        llm = LLM(
            **common,
            kv_cache_dtype="fp8" if args.arm == "fp8-apc" else "auto",
            enable_prefix_caching=not bool(getattr(args, "apc_off", False)),  # diagnostic control only; campaigns pass no attribute
            page_group_size=None,
        )
        cfg.update(page_group_size=None, requested_enable_prefix_caching=True)
    cfg["resolved_enable_prefix_caching"] = bool(llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)
    cfg["enforce_eager"] = bool(llm.llm_engine.vllm_config.model_config.enforce_eager)
    cfg["kv_cache_dtype"] = str(llm.llm_engine.vllm_config.cache_config.cache_dtype)
    cfg["admission_cap"] = int(getattr(args, "admission_cap", 0) or 0)  # harness-only admission gate (same-batch control); 0 = engine decides
    cfg["attention_backend"] = str(os.environ.get("VLLM_ATTENTION_BACKEND"))
    return llm, cfg


# -------------------------------------------------------------------- ownership
def frozen_admission_blocks(rmgr: Any, request_id: str, prompt_len: int, max_model_len: int, base_ids: "np.ndarray", watermark_blocks: int) -> dict[str, int]:
    """Free blocks the pool must hold before a frozen-copy branch's copy is published.

    = copy pages (one private physical copy of the published base) + the scheduler's admission
    reservation for the continuation + the admission watermark.  The reservation is computed with
    vLLM's own accounting (``get_num_blocks_to_allocate``) for a request that already owns the copy's
    pages, i.e. exactly the check ``_full_sequence_fits`` applies once the branch has attached.  Every
    ragged arm's request must pass that reservation; the frozen arm must additionally hold its copy,
    which is precisely the duplication cost this control arm exists to expose.
    """
    n_copy = int(base_ids.size)
    full = min(int(prompt_len), int(max_model_len))
    reservation = int(rmgr.get_num_blocks_to_allocate(request_id, full, base_ids))
    wm = int(watermark_blocks)
    return {"copy_blocks": n_copy, "reservation_blocks": reservation, "watermark_blocks": wm, "need_blocks": n_copy + reservation + wm}


def ragged_manager(manager: Any) -> Any | None:
    managers = [m for m in manager.coordinator.single_type_managers if getattr(m, "ragged", False)]
    return managers[0] if len(managers) == 1 else None


def request_block_counts(scheduler: Any) -> dict[str, int]:
    """Blocks currently owned by each live request (ragged: ndarray ids; regular: KVCacheBlocks)."""

    manager = scheduler.kv_cache_manager
    counts: dict[str, int] = {}
    single = manager.coordinator.single_type_managers
    for m in single:
        table = getattr(m, "req_to_block_ids", None) or getattr(m, "req_to_blocks", None)
        if table is None:
            continue
        for req_id, value in list(table.items()):
            try:
                n = int(len(value))
            except TypeError:
                n = 0
            counts[str(req_id)] = counts.get(str(req_id), 0) + n
    return counts


# ------------------------------------------------------------ identity capture
# Implemented in scripts.audit.identity_capture.


# --------------------------------------------------------- frozen-copy control
class FrozenCopyRegistry(TangramRaggedPrefixCache):
    """Causal control baseline (harness control path, no mechanism change).

    Publishes, for one branch, a *private physical copy* of an already published
    immutable base under a per-branch identity (adapter_id), so the branch runs
    exactly the shared arm's private-continuation path (attach, protected prefix,
    private compaction) against pages that no other branch owns. The copy is the
    published representation byte for byte (per-row page copy on the live KV
    tensors); it is released when the branch finishes.
    """

    def publish_copy(self, base: TangramRaggedPrefixEntry, lookup_copy: TangramRaggedPrefixLookup, block_pool: Any, kv_caches: list, request_id: str) -> tuple[TangramRaggedPrefixEntry | None, dict[str, Any]]:
        snap = base.snapshot
        ids = np.asarray(snap.physical_block_ids, dtype=np.int32)
        n = int(ids.size)
        if int(block_pool.get_num_free_blocks()) < n:
            return None, {"waited": True}
        t0 = time.perf_counter_ns()
        new_ids = block_pool.get_new_block_ids(n)  # allocation owner reference (1)
        mapping = dict(zip(ids.tolist(), new_ids.tolist()))
        rows = len(snap.block_counts)
        num_groups = rows // len(kv_caches)
        new_table = []
        for r, (row, count) in enumerate(zip(snap.block_table, snap.block_counts)):
            layer = r // num_groups
            kv = kv_caches[layer]
            src = torch.as_tensor([int(v) for v in row[:count]], device=kv.device, dtype=torch.long)
            dst_list = [mapping[int(v)] for v in row[:count]]
            dst = torch.as_tensor(dst_list, device=kv.device, dtype=torch.long)
            kv[:, dst] = kv[:, src]
            new_table.append(tuple(dst_list) + (0,) * (len(row) - count))
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        copy_snapshot = TangramRaggedPrefixSnapshot(tuple(new_table), tuple(snap.block_counts), tuple(snap.effective_seq_lens))
        entry = self._publish(lookup=lookup_copy, snapshot=copy_snapshot, retained_positions=base.retained_positions, event="frozen_copy_published", request_id=request_id)
        block_pool.release_block_ids_array(new_ids)  # hand ownership to the registry (2 -> 1)
        t2 = time.perf_counter_ns()
        return entry, {"waited": False, "pages": n, "bytes": n * int(self.manager.ragged_physical_block_bytes), "copy_ms": (t1 - t0) / 1e6, "publish_ms": (t2 - t1) / 1e6}

    def release_copy(self, lookup_copy: TangramRaggedPrefixLookup) -> bool:
        with self._lock:
            entry = self._entries.pop(lookup_copy.semantic_digest, None)
        if entry is None:
            return False
        self.manager.release_ragged_prefix_block_ids(entry.snapshot.physical_block_ids)
        with self._lock:
            self._events.append({"event": "frozen_copy_released", "lookup_digest": lookup_copy.semantic_digest, "pages": len(entry.snapshot.physical_block_ids)})
        return True


# ------------------------------------------------------------------- lifecycle
class PreemptInjector:
    """One-shot fault injection at vLLM's own preemption decision point.

    Wraps KVCacheManager.allocate_slots so that exactly one decode allocation for
    `target` returns None (no free slots), which makes the scheduler take its
    ordinary preemption path (free pages, registry on_request_preempted, status
    PREEMPTED, resume later). Nothing else is modified.
    """

    def __init__(self, manager: Any, target: str, ready: "callable") -> None:
        self.manager = manager
        self.target = target
        self.ready = ready
        self.fired = False
        self.original = manager.allocate_slots

    def install(self) -> None:
        injector = self

        def allocate_slots(request, *args, **kwargs):
            if (
                not injector.fired
                and str(getattr(request, "request_id", "")) == injector.target
                and injector.ready()
            ):
                injector.fired = True
                return None
            return injector.original(request, *args, **kwargs)

        self.manager.allocate_slots = allocate_slots

    def uninstall(self) -> None:
        self.manager.allocate_slots = self.original


# ------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> dict[str, object]:
    trace = json.loads(args.trace.read_text())
    prefix_ids = [int(v) for v in trace["prefix_token_ids"]]
    if digest_token_ids(prefix_ids) != trace["token_prefix_digest"]:
        raise RuntimeError("trace token prefix digest mismatch")
    if args.arm in APC_ARMS and args.scenario == "lifecycle":
        raise RuntimeError("lifecycle scenario applies to ragged arms only")
    smi_before = nvidia_smi()
    torch.cuda.reset_peak_memory_stats()
    t_build = time.perf_counter()
    llm, cfg = build_llm(args, trace)
    build_s = time.perf_counter() - t_build
    scheduler = engine_core(llm).scheduler
    manager = scheduler.kv_cache_manager
    block_pool = manager.block_pool
    ragged = bool(getattr(manager, "ragged", False))
    block_size = int(manager.ragged_block_size) if ragged else int(llm.llm_engine.vllm_config.cache_config.block_size)
    bytes_per_block = int(manager.ragged_physical_block_bytes) if ragged else int(cache_bytes(scheduler) // manager.kv_cache_config.num_blocks)
    initial_free = int(block_pool.get_num_free_blocks())
    committed = cache_bytes(scheduler)
    gpu = gpu_identity()
    if args.expect_gpu_uuid and gpu["uuid"] and args.expect_gpu_uuid not in str(gpu["uuid"]):
        raise RuntimeError(f"GPU UUID {gpu['uuid']} does not match lock {args.expect_gpu_uuid}")

    identity: IdentityCapture | None = None
    model_runner = engine_core(llm).model_executor.driver_worker.model_runner
    if ragged and args.identity_capture:
        exe = getattr(model_runner, "compression_executor", None)
        if exe is not None:
            identity = IdentityCapture(
                exe,
                int(trace["context_tokens"]),
                payload_hash=bool(args.identity_payload_hash),
                strict=bool(args.identity_capture_strict),
            )
            identity.install()
    registry: TangramRaggedPrefixCache | None = None
    lookup: TangramRaggedPrefixLookup | None = None
    if args.arm in REGISTRY_ARMS:
        registry = FrozenCopyRegistry() if args.arm == "frozen-fp16-copy" else TangramRaggedPrefixCache()
        scheduler.set_kvladder_ragged_prefix_cache(registry)
        lookup = make_lookup(trace, block_size, float(cfg["eviction_ratio"]))
    frozen_copies: dict[str, dict[str, Any]] = {}  # request_id -> {lookup, entry, stats}

    def frozen_lookup(request_id: str) -> TangramRaggedPrefixLookup:
        return dc_replace(lookup, adapter_id=f"frozen-copy:{request_id}")

    def frozen_try_publish(request_id: str) -> bool:
        """Publish the branch's private copy of the base if the pool has room; True when ready."""
        if request_id in frozen_copies and frozen_copies[request_id].get("entry") is not None:
            return True
        base = registry.entry(lookup)
        lk = frozen_lookup(request_id)
        entry, stats = registry.publish_copy(base, lk, block_pool, model_runner.kv_caches, request_id)  # type: ignore[attr-defined]
        rec = frozen_copies.setdefault(request_id, {"lookup": lk, "entry": None, "stats": {}, "wait_started": time.perf_counter()})
        if entry is None:
            return False
        rec["entry"] = entry
        rec["stats"] = stats
        rec["stats"]["wait_ms"] = (time.perf_counter() - rec["wait_started"]) * 1000.0
        return True

    def frozen_release(request_id: str) -> None:
        rec = frozen_copies.get(request_id)
        if rec and rec.get("entry") is not None and not rec.get("released"):
            registry.release_copy(rec["lookup"])  # type: ignore[attr-defined]
            rec["released"] = True

    # ---- warm / publish (outside measurement), mirrors P4/P5
    t_warm = time.perf_counter()
    q0 = [int(v) for v in trace["requests"][0]["query_token_ids"]]
    if args.arm == "kvladder-fp16-shared":
        llm.llm_engine.add_request("warm-shared-origin", TokensPrompt(prompt_token_ids=prefix_ids), sampling(trace, lookup=lookup, max_tokens=1))
        drain_one(llm, "warm-shared-origin")
        llm.llm_engine.add_request("warm-shared-hit", TokensPrompt(prompt_token_ids=prefix_ids + q0), sampling(trace, lookup=lookup, max_tokens=2))
        drain_one(llm, "warm-shared-hit")
    elif args.arm == "frozen-fp16-copy":
        llm.llm_engine.add_request("warm-shared-origin", TokensPrompt(prompt_token_ids=prefix_ids), sampling(trace, lookup=lookup, max_tokens=1))
        drain_one(llm, "warm-shared-origin")
        if not frozen_try_publish("warm-frozen-hit"):
            raise RuntimeError("frozen-copy warm publish did not fit the pool")
        llm.llm_engine.add_request("warm-frozen-hit", TokensPrompt(prompt_token_ids=prefix_ids + q0), sampling(trace, lookup=frozen_copies["warm-frozen-hit"]["lookup"], max_tokens=2))
        drain_one(llm, "warm-frozen-hit")
        frozen_release("warm-frozen-hit")
    else:
        llm.llm_engine.add_request("warm", TokensPrompt(prompt_token_ids=prefix_ids + q0), sampling(trace, lookup=None, max_tokens=2))
        drain_one(llm, "warm")
    warm_s = time.perf_counter() - t_warm
    free_after_warm = int(block_pool.get_num_free_blocks())
    registry_before = registry.telemetry() if registry is not None else None

    # ---- instrumentation
    admission: dict[str, float] = {}
    occupancy: list[dict[str, object]] = []
    private_meta: set[str] = set()
    private_boundary: set[str] = set()
    schedule_original = scheduler.schedule
    t0 = 0.0

    def sample(phase: str, extra: dict[str, object] | None = None) -> None:
        free = int(block_pool.get_num_free_blocks())
        row: dict[str, object] = {
            "t_s": (time.perf_counter() - t0) if t0 else 0.0,
            "phase": phase,
            "free_blocks": free,
            "live_blocks": initial_free - free,
            "live_hbm_bytes": (initial_free - free) * bytes_per_block,
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
            "request_blocks": request_block_counts(scheduler),
            "request_state": {
                str(r.request_id): {
                    "computed": int(r.num_computed_tokens),
                    "compression_done": bool(getattr(r, "compression_done", False)),
                    "status": str(getattr(r.status, "name", r.status)),
                }
                for r in list(scheduler.running)
            },
        }
        if ragged and args.identity_capture:
            try:
                ib = model_runner.input_batch
                eff = ib.effective_seq_lens_cpu
                if eff is not None:
                    row["worker_effective_seq_lens"] = {
                        str(rid): [int(eff[idx].min()), int(eff[idx].max())]
                        for rid, idx in list(ib.req_id_to_index.items()) if str(rid).startswith("conf-")
                    }
            except Exception:
                pass
        if extra:
            row.update(extra)
        occupancy.append(row)

    def schedule_instrumented() -> Any:
        out = schedule_original()
        for req_id, md in out.compression_metadata.items():
            if getattr(md, "private_continuation", False):
                private_meta.add(str(req_id))
                if bool(md.run_compression):
                    private_boundary.add(str(req_id))
        now = time.perf_counter()
        for r in out.scheduled_new_reqs:
            admission.setdefault(str(r.req_id), now)
        sample("post_schedule_pre_execute", {"scheduled_new": [str(r.req_id) for r in out.scheduled_new_reqs]})
        return out

    scheduler.schedule = schedule_instrumented

    free_original = getattr(scheduler, "_free_blocks", None)

    def retained_positions_for_finish(request: Any) -> dict[str, Any] | None:
        if registry is None:
            return None
        try:
            live = getattr(registry, "_requests", {}).get(str(request.request_id))
            parent = getattr(live, "parent", None)
            if parent is None:
                return None
            parent_tokens = int(parent.identity.lookup.prefix_token_count)
            computed = int(getattr(request, "num_computed_tokens", 0))
            return retained_positions_digest_from_base_tail(
                parent.retained_positions, parent_tokens, computed
            )
        except Exception as exc:
            return {
                "schema": "kvladder-retained-positions-digest-v1",
                "source": "registry_policy_base_positions_plus_derived_contiguous_tail",
                "base_position_lineage": "policy_derived_registry_parent",
                "private_tail_lineage": "derived_from_worker_computed_token_count_not_observed_selection",
                "capture_error": f"{type(exc).__name__}: {exc}",
            }

    if identity is not None and free_original is not None:
        def free_blocks_instrumented(request: Any) -> Any:
            if str(getattr(request, "request_id", "")).startswith("conf-"):
                identity.record_final_request(
                    req_id=str(request.request_id),
                    model_runner=model_runner,
                    request=request,
                    retained_positions=retained_positions_for_finish(request),
                )
            return free_original(request)

        scheduler._free_blocks = free_blocks_instrumented

    specs = list(trace["requests"])[: int(args.branches)]
    if len(specs) != int(args.branches):
        raise RuntimeError(f"trace has {len(specs)} requests < --branches {args.branches}")
    states: dict[str, dict[str, Any]] = {}

    def add(request_id: str, spec: Mapping[str, Any], *, commit: bool = False, lk: TangramRaggedPrefixLookup | None = None, max_tokens: int | None = None, ignore_eos: bool = False, arrival_at: float | None = None) -> None:
        prompt = prefix_ids + [int(v) for v in spec["query_token_ids"]]
        if args.arm == "frozen-fp16-copy" and lk is None:
            if not frozen_try_publish(request_id):
                raise RuntimeError(f"frozen copy for {request_id} does not fit the pool at add time")
            lk = frozen_copies[request_id]["lookup"]
        states[request_id] = {
            "spec": spec,
            "added_at": arrival_at if arrival_at is not None else time.perf_counter(),
            "first_token_at": None,
            "finished_at": None,
            "token_ids": [],
            "token_event_times": [],
            "generated": "",
            "num_cached_tokens": 0,
            "finish_reason": None,
            "role": "branch",
            "commit_on_success": bool(commit),
        }
        llm.llm_engine.add_request(request_id, TokensPrompt(prompt_token_ids=prompt), sampling(trace, lookup=lk if lk is not None else lookup, commit=commit, max_tokens=max_tokens, ignore_eos=ignore_eos))

    def ingest(outputs: list[Any], event_time: float) -> None:
        for out in outputs:
            rid = str(out.request_id)
            st = states.get(rid)
            if st is None or not out.outputs:
                continue
            cand = out.outputs[0]
            ids = [int(v) for v in cand.token_ids]
            prior = len(st["token_ids"])
            if len(ids) > prior:
                st["token_ids"] = ids
                st["token_event_times"].extend([event_time] * (len(ids) - prior))
                st["first_token_at"] = st["first_token_at"] or event_time
            st["generated"] = str(cand.text)
            st["num_cached_tokens"] = int(out.num_cached_tokens or 0)
            if out.finished:
                st["finished_at"] = event_time
                st["finish_reason"] = str(cand.finish_reason)
                if args.arm == "frozen-fp16-copy":
                    frozen_release(rid)

    def step() -> None:
        outs = llm.llm_engine.step()
        now = time.perf_counter()
        sample("post_execute")
        ingest(outs, now)

    def progress(msg: str) -> None:
        print(f"[conf-arm {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    def spin_until(pred, what: str, *, max_idle_steps: int = 20000) -> None:
        """Step until pred() holds; fail loudly instead of spinning forever."""
        idle = 0
        phantom_steps = 0
        while not pred():
            if time.perf_counter() > deadline:
                raise RuntimeError(f"lifecycle exceeded max wall time while waiting for {what}")
            if not llm.llm_engine.has_unfinished_requests():
                idle += 1
                if idle > max_idle_steps:
                    raise RuntimeError(f"engine idle with nothing to schedule while waiting for {what}")
            elif not scheduler.requests and not scheduler.waiting and not scheduler.running:
                phantom_steps += 1
                if phantom_steps > 50:
                    raise RuntimeError(f"client-side phantom request: engine reports unfinished work but the scheduler is empty while waiting for {what}")
            step()

    lifecycle: dict[str, object] = {}
    t0 = time.perf_counter()
    deadline = t0 + float(args.max_wall_s)

    if args.scenario == "burst":
        burst_start = time.perf_counter()
        if args.arm == "frozen-fp16-copy":
            # Every branch logically arrives at burst start; a branch is admitted when its private
            # copy of the base fits the pool, otherwise it waits (the duplication cost surfaces as
            # queueing, exactly like the native copy arm's queued prefills).
            pending = [spec for spec in specs]
            rmgr = ragged_manager(manager)
            base_ids = np.asarray(registry.entry(lookup).snapshot.physical_block_ids, dtype=np.int32)
            watermark_blocks = int(getattr(manager, "watermark_blocks", 0))

            def try_admit() -> None:
                nonlocal pending
                still = []
                for spec in pending:
                    rid = str(spec["request_id"])
                    need = frozen_admission_blocks(rmgr, rid, len(prefix_ids) + len(spec["query_token_ids"]), int(args.max_model_len), base_ids, watermark_blocks)
                    free = int(block_pool.get_num_free_blocks())
                    if free >= need["need_blocks"] and frozen_try_publish(rid):
                        frozen_copies[rid]["stats"].update(need)
                        frozen_copies[rid]["stats"]["free_blocks_before_publish"] = free
                        add(rid, spec, commit=False, arrival_at=burst_start)
                    else:
                        frozen_copies.setdefault(rid, {"lookup": frozen_lookup(rid), "entry": None, "stats": {}, "wait_started": time.perf_counter()})
                        still.append(spec)
                pending = still
            try_admit()
            idle = 0
            starved = 0
            while pending or llm.llm_engine.has_unfinished_requests():
                if time.perf_counter() > deadline:
                    raise RuntimeError(f"frozen-copy arm exceeded max wall time with {len(pending)} pending")
                if llm.llm_engine.has_unfinished_requests():
                    step()
                    idle = 0
                    if scheduler.waiting and not scheduler.running:
                        starved += 1
                        if starved > 200:
                            raise RuntimeError("frozen-copy arm: scheduler admits no waiting branch while nothing runs (admission reservation starvation)")
                    else:
                        starved = 0
                else:
                    idle += 1
                    if idle > 1000:
                        raise RuntimeError("frozen-copy arm: pool never admits the next copy (leak or budget too small)")
                try_admit()
        elif int(getattr(args, "admission_cap", 0) or 0) > 0:
            # Same-batch control (supplement lock S1): the harness admits at most `cap` branches at a
            # time, so the shared arm sees the frozen-copy arm's admission waves without any memory
            # pressure argument.  Every branch still logically arrives at burst start.
            cap = int(args.admission_cap)
            pending = list(specs)

            def active_count() -> int:
                return sum(1 for st in states.values() if st["finished_at"] is None)

            def try_admit_cap() -> None:
                nonlocal pending
                while pending and active_count() < cap:
                    spec = pending.pop(0)
                    add(str(spec["request_id"]), spec, commit=bool(args.commit_on_success), arrival_at=burst_start)

            try_admit_cap()
            while pending or llm.llm_engine.has_unfinished_requests():
                if time.perf_counter() > deadline:
                    pending_ids = [r for r, s in states.items() if s["finished_at"] is None]
                    llm.llm_engine.abort_request(pending_ids)
                    raise RuntimeError(f"arm exceeded max wall time with {len(pending_ids) + len(pending)} pending (admission cap {cap})")
                if llm.llm_engine.has_unfinished_requests():
                    step()
                try_admit_cap()
        else:
            for spec in specs:
                add(str(spec["request_id"]), spec, commit=bool(args.commit_on_success), arrival_at=burst_start)
            while llm.llm_engine.has_unfinished_requests():
                if time.perf_counter() > deadline:
                    pending_ids = [r for r, s in states.items() if s["finished_at"] is None]
                    llm.llm_engine.abort_request(pending_ids)
                    raise RuntimeError(f"arm exceeded max wall time with {len(pending_ids)} pending")
                step()
    else:
        # ---- RQ1 lifecycle: two live siblings (one commits), an aborted branch,
        # an injected preemption, and an identity-mismatch rejection.
        rc = lambda: [int(v) for v in manager.ragged_prefix_refcounts(registry_entry_ids())] if registry is not None else None  # noqa: E731

        def registry_entry_ids() -> tuple[int, ...]:
            tel = registry.telemetry()
            rows = tel["entry_rows"]
            return tuple(rows[0]["raw_page_ids"]) if rows else ()

        def refcount_summary() -> dict[str, object] | None:
            if registry is None:
                return None
            ids = registry_entry_ids()
            if not ids:
                return {"pages": 0}
            vals = manager.ragged_prefix_refcounts(ids)
            return {"pages": len(ids), "min": int(min(vals)), "max": int(max(vals))}

        lifecycle["refcount_after_warm"] = refcount_summary()
        a_id, b_id, c_id, d_id = (str(s["request_id"]) for s in specs[:4])
        q_len = len(specs[0]["query_token_ids"])
        gen_default = int(trace["generation"]["max_new_tokens"])
        # Commit publishes "frozen prefix + exact private tail".  The tail must be
        # block-aligned (validate_immutable_boundary) and must not have been
        # compacted by Tangram's private compression (exact-tail contract), i.e. it
        # must stay at or below compression_floor_min (64 tokens here).  vLLM's
        # num_computed_tokens at finish = prompt + (max_tokens - 1), so
        # max_tokens = commit_total - q_len + 1 (P2 used 1 + 16 - 1 = 16).
        commit_total = (int(args.commit_private_tokens) // block_size) * block_size
        if commit_total <= q_len:
            raise RuntimeError("commit private tail must exceed the private prompt")
        commit_max_tokens = commit_total - q_len + 1
        lifecycle["commit_alignment"] = {"private_prompt_tokens": q_len, "commit_max_tokens": commit_max_tokens, "private_computed_tokens_at_finish": commit_total, "block_size": block_size, "rule": "num_computed_tokens_at_finish = prompt + max_tokens - 1; tail <= compression_floor_min keeps it exact"}
        probe_errors: dict[str, str] = {}

        # --- probe 1: aborted branch (admit, >= 1 token, abort)
        progress("probe 1: abort")
        add(c_id, specs[2], commit=False)
        states[c_id]["role"] = "aborted"
        spin_until(lambda: states[c_id]["first_token_at"] is not None, "C first token")
        free_before_abort = int(block_pool.get_num_free_blocks())
        blocks_before_abort = request_block_counts(scheduler).get(c_id)
        llm.llm_engine.abort_request([c_id])
        step()
        lifecycle["abort"] = {
            "free_blocks_before_abort": free_before_abort,
            "free_blocks_after_abort_step": int(block_pool.get_num_free_blocks()),
            "aborted_request_blocks_before": blocks_before_abort,
            "refcount_after_abort": refcount_summary(),
            "aborted_tokens_before_abort": len(states[c_id]["token_ids"]),
            "aborted_request_in_scheduler": c_id in scheduler.requests,
        }
        # --- probe 2: injected preemption of branch D after its first token
        progress("probe 2: preemption")
        add(d_id, specs[3], commit=False)
        states[d_id]["role"] = "preempted"
        spin_until(lambda: states[d_id]["first_token_at"] is not None, "D first token")
        injector = PreemptInjector(manager, d_id, lambda: True)
        injector.install()
        preempt_error: str | None = None
        try:
            for _ in range(3):
                step()
        except Exception as exc:  # the prototype's preemption path may fail closed
            preempt_error = f"{type(exc).__name__}: {exc}"
        finally:
            injector.uninstall()
        if preempt_error is None:
            try:
                spin_until(lambda: not llm.llm_engine.has_unfinished_requests(), "D resume+finish", max_idle_steps=1)
            except Exception as exc:
                preempt_error = f"resume: {type(exc).__name__}: {exc}"
        lifecycle["preemption"] = {
            "injected": injector.fired,
            "engine_error": preempt_error,
            "registry_preemptions": (registry.telemetry()["preemptions"] if registry is not None else None),
            "refcount_after_preempt": refcount_summary() if preempt_error is None else None,
            "d_status_after": str(getattr(getattr(scheduler.requests.get(d_id), "status", None), "name", "gone")),
            "d_completed": states[d_id]["finished_at"] is not None,
            "d_output_tokens": len(states[d_id]["token_ids"]),
            "d_num_preemptions": int(getattr(scheduler.requests.get(d_id), "num_preemptions", -1)) if d_id in scheduler.requests else None,
        }
        if preempt_error is not None:
            probe_errors["preemption"] = preempt_error
        engine_alive = preempt_error is None
        # --- probe 3: identity-mismatch rejection at arrival (no leak)
        progress(f"probe 3: identity mismatch (engine_alive={engine_alive})")
        if registry is not None and engine_alive:
            bad = TangramRaggedPrefixLookup(**{**asdict(lookup), "token_prefix_digest": "0" * 64})
            if args.arm == "frozen-fp16-copy":
                bad = dc_replace(bad, adapter_id="frozen-copy:conf-mismatch")
            free_before = int(block_pool.get_num_free_blocks())
            rejected: str | None = None
            try:
                add("conf-mismatch", specs[4] if len(specs) > 4 else specs[0], lk=bad)
            except TangramRaggedPrefixError as exc:
                rejected = str(exc)
            except Exception as exc:  # engine wraps it
                rejected = f"{type(exc).__name__}: {exc}"
            states.pop("conf-mismatch", None)
            # The rejection happened inside the scheduler's fail-closed add_request;
            # the in-process LLMEngine's output processor may still track the request
            # it never got back (a client-side phantom that keeps has_unfinished_requests()
            # true).  Clear it on the client side and record whether it existed.
            phantom = bool(llm.llm_engine.has_unfinished_requests()) and not scheduler.requests
            if phantom:
                llm.llm_engine.abort_request(["conf-mismatch"])
            lifecycle["identity_mismatch"] = {
                "rejected": rejected is not None,
                "error": rejected,
                "free_blocks_before": free_before,
                "free_blocks_after": int(block_pool.get_num_free_blocks()),
                "live_requests_in_registry": registry.telemetry()["live_requests"],
                "request_in_scheduler": "conf-mismatch" in scheduler.requests,
                "client_side_phantom_cleared": phantom,
                "engine_unfinished_after_cleanup": bool(llm.llm_engine.has_unfinished_requests()),
            }
        # --- probe 4: two live siblings, A commits (block-aligned tail), B released
        progress("probe 4: two live siblings")
        if engine_alive:
            probe_commit = args.commit_probe == "aligned"
            add(a_id, specs[0], commit=probe_commit, max_tokens=commit_max_tokens if probe_commit else None, ignore_eos=probe_commit)
            add(b_id, specs[1], commit=False)
            for _ in range(3):
                step()
            lifecycle["refcount_two_live"] = refcount_summary()
            lifecycle["blocks_two_live"] = request_block_counts(scheduler)
            commit_error: str | None = None
            a_trace: list[dict[str, object]] = []

            def a_diag() -> None:
                live = registry._requests.get(a_id) if registry is not None else None  # diagnostic read of registry state
                req = scheduler.requests.get(a_id)
                if live is None or req is None:
                    return
                snap = live.latest_snapshot
                desc = live.descriptor
                a_trace.append({
                    "computed": int(req.num_computed_tokens),
                    "worker_eff_min_max": [min(snap.effective_seq_lens), max(snap.effective_seq_lens)] if snap else None,
                    "worker_counts_min_max": [min(snap.block_counts), max(snap.block_counts)] if snap else None,
                    "descriptor_eff_min_max": ([min(desc.base_effective_seq_lens) + desc.private_token_delta, max(desc.base_effective_seq_lens) + desc.private_token_delta] if desc else None),
                    "descriptor_counts_min_max": [min(desc.block_counts), max(desc.block_counts)] if desc else None,
                    "private_compression_state": str(getattr(req, "kvladder_private_compression_state", None)),
                    "compress_max_eff_seq_len": getattr(req, "compress_max_eff_seq_len", None),
                })

            try:
                spin_until(lambda: (a_diag() or not llm.llm_engine.has_unfinished_requests()), "A/B finish", max_idle_steps=1)
                progress("probe 4 done")
            except Exception as exc:
                commit_error = f"{type(exc).__name__}: {exc}"
                probe_errors["commit"] = commit_error
                engine_alive = False
            tel = registry.telemetry() if registry is not None else {}
            lifecycle["commit"] = {
                "probe": args.commit_probe,
                "engine_error": commit_error,
                "commits": tel.get("commits"),
                "entries_after_commit": tel.get("entries"),
                "entry_rows_after_commit": [{"logical_tokens": e["logical_tokens"], "pages": len(e["raw_page_ids"]), "refcount_min": min(e["refcounts"]), "refcount_max": max(e["refcounts"])} for e in tel.get("entry_rows", [])],
                "trim_events": [e for e in tel.get("events", []) if e.get("event") == "commit_trimmed_slack_pages"],
                "a_output_tokens": len(states[a_id]["token_ids"]),
                "a_finish_reason": states[a_id]["finish_reason"],
                "a_diagnostic_trace_tail": a_trace[-4:],
                "a_diagnostic_trace_head": a_trace[:3],
            }
        lifecycle["refcount_after_all_finished"] = refcount_summary()
        lifecycle["committed_entries"] = registry.telemetry()["entries"] if registry is not None else None
        lifecycle["free_blocks_after_all_finished"] = int(block_pool.get_num_free_blocks())
        # --- probe 5 (last): a commit with a misaligned private tail must be refused fail-closed
        progress(f"probe 5: unsupported commit (engine_alive={engine_alive})")
        if registry is not None and engine_alive and len(specs) > 4:
            e_id = str(specs[4]["request_id"])
            free_before_e = int(block_pool.get_num_free_blocks())
            refs_before_e = refcount_summary()
            add(e_id, specs[4], commit=True, max_tokens=gen_default, ignore_eos=True)
            states[e_id]["role"] = "misaligned_commit"
            misaligned_error: str | None = None
            try:
                spin_until(lambda: not llm.llm_engine.has_unfinished_requests(), "E finish", max_idle_steps=1)
            except Exception as exc:
                misaligned_error = f"{type(exc).__name__}: {exc}"
            progress(f"probe 5 done: {misaligned_error}")
            lifecycle["misaligned_commit"] = {
                "private_computed_tokens": len(specs[4]["query_token_ids"]) + gen_default - 1,
                "refused": misaligned_error is not None,
                "error": misaligned_error,
                "free_blocks_before": free_before_e,
                "free_blocks_after": int(block_pool.get_num_free_blocks()),
                "refcount_before": refs_before_e,
                "refcount_after": refcount_summary(),
                "request_still_in_scheduler": e_id in scheduler.requests,
                "registry_live_requests": registry.telemetry()["live_requests"],
                "registry_entries": registry.telemetry()["entries"],
            }
        lifecycle["probe_errors"] = probe_errors

    t_end = time.perf_counter()

    # ---- rows
    rows = []
    for rid, st in states.items():
        spec = st["spec"]
        added = float(st["added_at"])
        adm = admission.get(rid)
        ft = st["first_token_at"]
        fin = st["finished_at"]
        times = [float(v) for v in st["token_event_times"]]
        tbt = [(r - l) * 1000.0 for l, r in zip(times, times[1:])]
        row = {
            "request_id": rid,
            "role": st["role"],
            "commit_on_success": st["commit_on_success"],
            "arrival_sequence": int(spec["arrival_sequence"]),
            "turn_index": int(spec["turn_index"]),
            "prompt_tokens": int(spec["prompt_tokens"]),
            "gold_answers": list(spec["gold_answers"]),
            "generated": str(st["generated"]),
            "generated_token_ids": list(st["token_ids"]),
            "output_tokens": len(st["token_ids"]),
            "num_cached_tokens": int(st["num_cached_tokens"]),
            "finish_reason": st["finish_reason"],
            "completed": fin is not None,
            "branch_start_latency_ms": (adm - added) * 1000.0 if adm is not None else None,
            "ttft_ms": (ft - added) * 1000.0 if ft is not None else None,
            "e2e_ms": (fin - added) * 1000.0 if fin is not None else None,
            "token_tbt_ms": tbt,
            "request_p95_tbt_ms": percentile(tbt, 0.95),
            **quality(str(st["generated"]), list(spec["gold_answers"])),
        }
        rows.append(row)
    rows.sort(key=lambda r: int(r["arrival_sequence"]))
    completed = [r for r in rows if r["completed"] and r["role"] == "branch"]
    elapsed = t_end - t0
    peak = max(occupancy, key=lambda r: int(r["live_blocks"])) if occupancy else {"live_blocks": 0, "live_hbm_bytes": 0, "phase": None}
    summary = {
        "offered_requests": len([r for r in rows if r["role"] == "branch"]),
        "completed_requests": len(completed),
        "maximum_concurrent_physically_admitted": max((int(r["running"]) for r in occupancy), default=0),
        "quality_passes": sum(bool(r["answer_contains"]) for r in completed),
        "mean_answer_f1": statistics.fmean([float(r["answer_f1"]) for r in completed]) if completed else None,
        "p50_ttft_ms": percentile([float(r["ttft_ms"]) for r in completed if r["ttft_ms"] is not None], 0.50),
        "p95_ttft_ms": percentile([float(r["ttft_ms"]) for r in completed if r["ttft_ms"] is not None], 0.95),
        "p95_e2e_ms": percentile([float(r["e2e_ms"]) for r in completed if r["e2e_ms"] is not None], 0.95),
        "p95_tbt_ms": percentile([v for r in completed for v in r["token_tbt_ms"]], 0.95),
        "throughput_requests_per_s": (len(completed) / elapsed) if elapsed > 0 else None,
        "measurement_elapsed_s": elapsed,
        "engine_build_s": build_s,
        "warm_s": warm_s,
        "initial_free_blocks": initial_free,
        "free_blocks_after_warm": free_after_warm,
        "warm_live_blocks": initial_free - free_after_warm,
        "peak_live_blocks": int(peak["live_blocks"]),
        "peak_live_hbm_bytes": int(peak["live_hbm_bytes"]),
        "peak_live_hbm_fraction": (int(peak["live_blocks"]) / initial_free) if initial_free else None,
        "peak_occupancy_phase": peak["phase"],
        "committed_cache_hbm_bytes": committed,
        "bytes_per_physical_block": bytes_per_block,
        "block_size_tokens": block_size,
        "min_free_blocks": min((int(r["free_blocks"]) for r in occupancy), default=initial_free),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "minimum_cached_tokens": min((int(r["num_cached_tokens"]) for r in rows), default=0),
        "maximum_cached_tokens": max((int(r["num_cached_tokens"]) for r in rows), default=0),
    }

    # ---- RQ4 microbenchmark on the live allocator (shared arm only)
    micro: dict[str, object] | None = None
    if registry is not None and int(args.microbench_ops) > 0:
        n = int(args.microbench_ops)
        tel = registry.telemetry()
        rows_e = tel["entry_rows"]
        ids = tuple(rows_e[0]["raw_page_ids"]) if rows_e else ()
        single = ragged_manager(manager)

        def bench(fn, k: int) -> dict[str, float]:
            lat = []
            for _ in range(k):
                t = time.perf_counter_ns()
                fn()
                lat.append((time.perf_counter_ns() - t) / 1e3)
            return {"count": k, "median_us": percentile(lat, 0.5), "p95_us": percentile(lat, 0.95), "p99_us": percentile(lat, 0.99), "min_us": min(lat), "max_us": max(lat)}

        def retain_release() -> None:
            manager.retain_ragged_prefix_block_ids(ids)
            manager.release_ragged_prefix_block_ids(ids)

        counter = {"i": 0}

        def attach_detach() -> None:
            rid = f"mb-{counter['i']}"
            counter["i"] += 1
            manager.attach_ragged_prefix_block_ids(rid, ids)
            manager.detach_ragged_prefix_block_ids(rid)

        def refcount_read() -> None:
            manager.ragged_prefix_refcounts(ids)

        free_before_mb = int(block_pool.get_num_free_blocks())
        micro = {
            "base_pages": len(ids),
            "base_bytes": len(ids) * bytes_per_block,
            "refcount_retain_release_pair": bench(retain_release, n),
            "allocator_attach_detach_pair": bench(attach_detach, n),
            "refcount_read": bench(refcount_read, n),
            "registry_attach_latency_ns_hits": tel["attach_latency_ns"],
            "descriptor_counters": {k: tel[k] for k in ("descriptor_schedule_updates", "descriptor_block_delta_updates", "descriptor_delta_block_ids", "descriptor_full_snapshot_materializations")},
            "free_blocks_before": free_before_mb,
            "free_blocks_after": int(block_pool.get_num_free_blocks()),
        }

    registry_after = registry.telemetry() if registry is not None else None
    identity_summary = identity.summary() if identity is not None else None
    identity_capture_complete = None
    if identity_summary is not None:
        identity_capture_complete = identity_capture_is_complete(
            identity_summary, specs, require_payload=args.identity_payload_hash
        )
    if identity is not None:
        identity.uninstall()
    executor = getattr(engine_core(llm).model_executor.driver_worker.model_runner, "compression_executor", None)
    executor_telemetry = None if executor is None else {
        "immutable_prefix_write_bytes": int(getattr(executor, "immutable_prefix_write_bytes", 0)),
        "protected_prefix_skip_bytes": int(getattr(executor, "protected_prefix_skip_bytes", 0)),
        "private_write_bytes": int(getattr(executor, "private_write_bytes", 0)),
        "ordinary_write_bytes": int(getattr(executor, "ordinary_write_bytes", 0)),
    }
    before_drain = int(block_pool.get_num_free_blocks())
    drain_error: str | None = None
    if registry is not None:
        try:
            registry.drain()
        except Exception as exc:
            drain_error = f"{type(exc).__name__}: {exc}"
        scheduler.set_kvladder_ragged_prefix_cache(None)
    reset_ok = True
    if args.arm in APC_ARMS:
        reset_ok = bool(scheduler.reset_prefix_cache())
    after_drain = int(block_pool.get_num_free_blocks())
    smi_after = nvidia_smi()
    tangram_root = Path(vllm.__file__).resolve().parents[1]
    receipt = {
        "schema": SCHEMA,
        "arm": args.arm,
        "scenario": args.scenario,
        "phase": str(args.phase),
        "repetition": int(args.repetition),
        "passed": (
            (
                len(completed) == len(specs)
                and after_drain == initial_free
                and drain_error is None
                and reset_ok
                and (identity_capture_complete is not False)
            )
            if args.scenario == "burst"
            else (
                lifecycle.get("abort") is not None
                and lifecycle.get("preemption") is not None
                and lifecycle.get("free_blocks_after_all_finished") is not None
            )
        ),
        "trace": {"path": str(args.trace), "content_sha256": str(trace["content_sha256"]), "context_tokens": int(trace["context_tokens"]), "requests_used": len(specs)},
        "configuration": {
            **cfg,
            "branches": int(args.branches),
            "kv_cache_memory_bytes": int(args.kv_cache_memory_bytes),
            "max_model_len": int(args.max_model_len),
            "max_num_seqs": int(args.max_num_seqs),
            "max_num_batched_tokens": int(trace["context_tokens"]),
            "seed": int(args.seed),
            "commit_on_success": bool(args.commit_on_success),
            "ragged": ragged,
            "ragged_block_size": block_size,
        },
        "summary": summary,
        "rows": rows,
        "occupancy": occupancy,
        "lifecycle": lifecycle or None,
        "registry_before_measurement": registry_before,
        "registry_after_measurement": registry_after,
        "tangram_complete_lifecycle": {"private_metadata_requests": len(private_meta), "private_boundary_requests": len(private_boundary), "executor": executor_telemetry},
        "microbenchmark": micro,
        "compressed_state_identity": identity_summary,
        "identity_capture_complete": identity_capture_complete,
        "frozen_copies": ({rid: {"pages": r.get("stats", {}).get("pages"), "bytes": r.get("stats", {}).get("bytes"), "copy_ms": r.get("stats", {}).get("copy_ms"), "publish_ms": r.get("stats", {}).get("publish_ms"), "wait_ms": r.get("stats", {}).get("wait_ms"), "released": bool(r.get("released")),
                                 "copy_blocks": r.get("stats", {}).get("copy_blocks"), "reservation_blocks": r.get("stats", {}).get("reservation_blocks"), "watermark_blocks": r.get("stats", {}).get("watermark_blocks"),
                                 "need_blocks": r.get("stats", {}).get("need_blocks"), "free_blocks_before_publish": r.get("stats", {}).get("free_blocks_before_publish")} for rid, r in frozen_copies.items()} if args.arm == "frozen-fp16-copy" else None),
        "frozen_admission_rule": ("copy_blocks + scheduler reservation for the uncompressed continuation length (vLLM get_num_blocks_to_allocate with the copy attached) + watermark <= free blocks; " + HARNESS_AMENDMENT) if args.arm == "frozen-fp16-copy" else None,
        "drain": {"free_blocks_before_drain": before_drain, "free_blocks_after_drain": after_drain, "expected_baseline": initial_free, "drain_error": drain_error, "prefix_reset_succeeded": reset_ok},
        "environment": {
            "harness_amendment": HARNESS_AMENDMENT,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "vllm_path": vllm.__file__,
            "tangram_upstream_commit": git_head(tangram_root),
            "tangram_overlay_diff_sha256": diff_sha256(tangram_root),
            "tangram_overlay_dirty": git_dirty(tangram_root),
            "repo_commit": git_head(REPO),
            "repo_dirty": git_dirty(REPO),
            "registry_module_path": str(Path(getattr(kvladder_registry_module, "__file__", "")).resolve()) if getattr(kvladder_registry_module, "__file__", None) else None,
            "registry_module_sha256": file_sha256(Path(kvladder_registry_module.__file__).resolve()) if getattr(kvladder_registry_module, "__file__", None) else None,
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "gpu": gpu,
            "nvidia_smi_before": smi_before,
            "nvidia_smi_after": smi_after,
            "cli": sys.argv,
            "hostname": platform.node(),
        },
        "claim_boundary": "One conference arm; representation-matched FP16 ragged arms; APC arms are stock vLLM V1 prefix caching inside the same Tangram build; timing is claimable only with the lock's repetition and contention rules.",
    }
    return receipt


def identity_capture_is_complete(summary: Mapping[str, Any], specs: list[Mapping[str, Any]], *, require_payload: bool) -> bool:
    """Require one real final capture per offered trace request, never a vacuous match."""
    expected = [str(spec["request_id"]) for spec in specs]
    if not expected or len(set(expected)) != len(expected):
        return False
    if (summary.get("_capture_summary") or {}).get("failures"):
        return False
    for rid in expected:
        row = summary.get(rid) or {}
        if (row.get("capture_ok") is not True
                or row.get("final_capture_stage") != "scheduler_free_blocks_pre_free"
                or not isinstance(row.get("final_kept_max"), int)
                or row["final_kept_max"] <= 0):
            return False
        if require_payload and (row.get("final_payload_hashed") is not True or not row.get("final_payload_sha256")):
            return False
    return True


def main() -> None:
    faulthandler.dump_traceback_later(180.0, repeat=True, file=sys.stderr)  # periodic stack dump: hangs become diagnosable in the log
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=RAGGED_ARMS + APC_ARMS, required=True)
    p.add_argument("--scenario", choices=("burst", "lifecycle"), default="burst")
    p.add_argument("--phase", default="smoke")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--branches", type=int, required=True)
    p.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--repetition", type=int, default=0)
    p.add_argument("--commit-on-success", action="store_true")
    p.add_argument("--microbench-ops", type=int, default=0)
    p.add_argument("--identity-capture", action=argparse.BooleanOptionalAction, default=True, help="record per-(layer,group) kept-position hashes of every compression pass and worker effective lengths (Gate P0); harness-only, changes no mechanism")
    p.add_argument("--identity-payload-hash", action=argparse.BooleanOptionalAction, default=False, help="when identity capture is enabled, hash logical resident KV payload bytes as well as metadata; slower due to GPU->CPU readback")
    p.add_argument("--identity-capture-strict", action=argparse.BooleanOptionalAction, default=True, help="fail the run when identity capture cannot record a non-empty resident state for a compression pass")
    p.add_argument("--commit-private-tokens", type=int, default=64, help="lifecycle: committed branch's exact private tail (block-aligned, <= compression_floor_min)")
    p.add_argument("--commit-probe", choices=("none", "aligned"), default="none", help="lifecycle: 'aligned' attempts a child publication for branch A (refused on the complete overlay: private compaction breaks the exact-tail contract, see smoke v4-v7); 'none' keeps A ordinary and records only the last-probe refusal")
    p.add_argument("--expect-gpu-uuid", default="")
    p.add_argument("--max-wall-s", type=float, default=1800.0)
    p.add_argument("--admission-cap", type=int, default=0, help="burst: admit at most this many branches concurrently (harness gate; supplement lock S1 same-batch control); 0 = no cap")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    receipt = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"arm": receipt["arm"], "scenario": receipt["scenario"], "passed": receipt["passed"], "summary": receipt["summary"], "lifecycle": receipt["lifecycle"], "drain": receipt["drain"], "output": str(args.output)}, indent=1, default=str))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
