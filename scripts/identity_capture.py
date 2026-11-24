"""KVLadder conference compressed-state identity capture.

This module is harness-only instrumentation.  It records the final resident
ragged KV state produced by each ``CompressionExecutor.run_request`` call; it
never mutates model state.  The original v2 harness only recorded keep vectors
inside ``_gather_and_writeback_kept_kv``.  That misses valid compression passes
whose policy keeps the current private continuation without a gather/writeback
(e.g. an attached frozen base plus a short query tail), and those passes were
reported as the SHA-256 of the empty string.  Here a pass is valid only if we can
also hash the resident state from the live block table; request-level identity
evidence uses the finish-before-free scheduler hook rather than that earlier
post-compression snapshot.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

try:  # Torch is already required by the runner; keep tests importable when absent.
    import torch
except Exception:  # pragma: no cover - exercised only on non-runner hosts.
    torch = None  # type: ignore[assignment]


TokenMajorFn = Callable[[Any, Any], Any]


def _json_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    try:
        raw = cpu.view(dtype=torch.uint8).numpy().tobytes()
    except Exception:  # older torch builds spell dtype reinterpretation differently
        raw = cpu.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _logical_payload_sha256(
    *,
    kv_cache: Any,
    block_ids: Any,
    kept_length: int,
    page_group_size: int,
    block_size: int,
    token_major_fn: TokenMajorFn | None = None,
    logical_start: int = 0,
    logical_end: int | None = None,
) -> dict[str, Any]:
    """Hash valid KV payload bytes in logical token order, excluding padding.

    ``block_ids`` are used only to read the resident pages; their physical
    values are intentionally not part of the digest.  The digest includes only
    dtype/shape and the valid ``[0, kept_length)`` token-major payload for this
    layer/group.  Padding in the final physical page and unused pages are not
    read into the hash.
    """

    if torch is None:
        raise RuntimeError("torch is required for KV payload capture")
    if kept_length < 0:
        raise RuntimeError(f"negative kept_length {kept_length}")
    if logical_end is None:
        logical_end = kept_length
    logical_start = int(logical_start)
    logical_end = int(logical_end)
    if logical_start < 0 or logical_end < 0:
        raise RuntimeError(f"negative logical payload range [{logical_start}, {logical_end})")
    if logical_end < logical_start:
        raise RuntimeError(f"invalid logical payload range [{logical_start}, {logical_end})")
    if logical_end > int(kept_length):
        raise RuntimeError(
            f"logical payload range [{logical_start}, {logical_end}) exceeds kept_length {kept_length}"
        )
    if logical_end == logical_start:
        payload = kv_cache.new_empty((2, 0, page_group_size, kv_cache.shape[-1]))
    else:
        if token_major_fn is None:
            from vllm.v1.attention.backends.ragged_layout import (  # type: ignore
                cluster_pages_token_major,
            )

            token_major_fn = cluster_pages_token_major
        slab = token_major_fn(kv_cache, block_ids)
        pos = torch.arange(logical_start, logical_end, device=kv_cache.device, dtype=torch.long)
        blk = torch.div(pos, block_size, rounding_mode="floor")
        off = pos - blk * block_size
        # slab: [2, n_blocks, block_size, page_group_size, head_size]
        payload = slab[:, blk, off, :, :].contiguous()
    meta = {
        "dtype": str(payload.dtype),
        "shape": [int(v) for v in payload.shape],
        "kept_length": int(kept_length),
        "logical_start": int(logical_start),
        "logical_end": int(logical_end),
        "page_group_size": int(page_group_size),
        "block_size": int(block_size),
    }
    return {**meta, "payload_sha256": _tensor_sha256(payload)}


def retained_positions_digest_from_base_tail(base_positions: Any, tail_start: int, tail_end: int) -> dict[str, Any]:
    """Hash policy-derived base rows plus a derived contiguous private tail.

    This is a computed-range diagnostic, not retained private-position lineage:
    the base rows come from the registry's published policy output, while the
    tail range is derived from the request's computed token count.
    """

    digest = hashlib.sha256(b"kvladder-retained-positions-v1")
    rows = 0
    lengths: list[int] = []
    base_row_lengths: list[int] = []
    tail_start = int(tail_start)
    tail_end = int(tail_end)
    if tail_end < tail_start:
        raise RuntimeError(f"invalid retained-position tail [{tail_start}, {tail_end})")
    for row in base_positions:
        rows += 1
        base = [int(v) for v in row]
        base_len = len(base)
        total_len = base_len + (tail_end - tail_start)
        base_row_lengths.append(base_len)
        lengths.append(total_len)
        digest.update(total_len.to_bytes(8, "little", signed=False))
        for value in base:
            digest.update(value.to_bytes(8, "little", signed=True))
        for value in range(tail_start, tail_end):
            digest.update(int(value).to_bytes(8, "little", signed=True))
    return {
        "schema": "kvladder-retained-positions-digest-v1",
        "source": "registry_policy_base_positions_plus_derived_contiguous_tail",
        "base_position_lineage": "policy_derived_registry_parent",
        "private_tail_lineage": "derived_from_worker_computed_token_count_not_observed_selection",
        "private_lineage_observed": False,
        "row_count": rows,
        "base_row_length_min": min(base_row_lengths, default=None),
        "base_row_length_max": max(base_row_lengths, default=None),
        "base_row_lengths": base_row_lengths,
        "row_length_min": min(lengths, default=None),
        "row_length_max": max(lengths, default=None),
        "tail_start": tail_start,
        "tail_end": tail_end,
        "sha256": digest.hexdigest(),
    }


class IdentityCapture:
    """Record final compressed state for every compression pass.

    The capture has two independent channels:
    * ``selection``: keep-position hashes observed when Tangram gathers and
      rewrites KV during an eviction.
    * ``resident_state``: post-run resident lengths and, optionally, KV payload
      hashes read from the live block table.  This channel also covers keep-all
      / no-writeback compression passes, so empty gather buffers no longer count
      as successful identity evidence.
    """

    def __init__(
        self,
        executor: Any,
        context_tokens: int,
        *,
        payload_hash: bool = False,
        strict: bool = True,
    ) -> None:
        self.executor = executor
        self.ctx = int(context_tokens)
        self.payload_hash = bool(payload_hash)
        self.strict = bool(strict)
        self.current: str | None = None
        self.passes: dict[str, list[dict[str, Any]]] = {}
        self.finals: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []
        self._orig_run = executor.run_request
        self._orig_gather = executor._gather_and_writeback_kept_kv
        self._buf: list[dict[str, Any]] = []
        self._protected = 0

    def install(self) -> None:
        cap = self

        def run_request(
            layer_kv_caches,
            block_table,
            prev_seq_lens_static_cpu,
            compressor,
            compression_metadata,
        ):
            req_id = str(compression_metadata.req_id)
            cap.current = req_id
            cap._protected = int(getattr(compression_metadata, "immutable_prefix_tokens", 0))
            cap._buf = []
            started = time.perf_counter()
            result = None
            run_exc: BaseException | None = None
            try:
                result = cap._orig_run(
                    layer_kv_caches,
                    block_table,
                    prev_seq_lens_static_cpu,
                    compressor,
                    compression_metadata,
                )
                return result
            except BaseException as exc:  # record then preserve original behavior
                run_exc = exc
                raise
            finally:
                try:
                    if run_exc is None:
                        assert result is not None
                        cap._record_pass(
                            req_id=req_id,
                            elapsed_ms=(time.perf_counter() - started) * 1000.0,
                            layer_kv_caches=layer_kv_caches,
                            block_table=block_table,
                            compression_metadata=compression_metadata,
                            kept_lengths_all=result,
                        )
                    else:
                        cap._record_failure(req_id, "run_request_error", run_exc)
                finally:
                    cap.current = None
                    cap._buf = []

        def gather(**kwargs):
            keep = cap._orig_gather(**kwargs)
            cap._record_selection(keep, kwargs)
            return keep

        self.executor.run_request = run_request
        self.executor._gather_and_writeback_kept_kv = gather

    def uninstall(self) -> None:
        self.executor.run_request = self._orig_run
        self.executor._gather_and_writeback_kept_kv = self._orig_gather

    def _record_failure(self, req_id: str, reason: str, exc: BaseException) -> None:
        item = {"request_id": req_id, "reason": reason, "error": f"{type(exc).__name__}: {exc}"}
        self.failures.append(item)
        self.passes.setdefault(req_id, []).append({
            "pass": len(self.passes.get(req_id, [])),
            "capture_ok": False,
            "capture_error": item,
            "selection_groups": len(self._buf),
        })

    def _record_selection(self, keep: Any, kwargs: dict[str, Any]) -> None:
        try:
            pos = keep.detach().to("cpu").contiguous()
            arr = pos.numpy().astype("int64", copy=False)
            row0 = arr[0] if pos.dim() == 2 else arr
            kept_lo = int(kwargs.get("kept_lo", 0))
            self._buf.append({
                "kept": int(row0.shape[0]),
                "positions_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "private_positions": [int(v) for v in row0[row0 >= kept_lo].tolist()] if len(self._buf) == 0 else None,
                "kept_lo": kept_lo,
            })
        except Exception as exc:  # capture must not alter the run
            self._buf.append({
                "kept": None,
                "positions_sha256": None,
                "capture_error": f"{type(exc).__name__}: {exc}",
                "private_positions": None,
            })

    def _record_pass(
        self,
        *,
        req_id: str,
        elapsed_ms: float,
        layer_kv_caches: list[Any],
        block_table: Any,
        compression_metadata: Any,
        kept_lengths_all: Any,
    ) -> None:
        try:
            state = capture_resident_state(
                executor=self.executor,
                layer_kv_caches=layer_kv_caches,
                block_table=block_table,
                row_idx=int(compression_metadata.row_idx),
                kept_lengths_all=kept_lengths_all,
                include_payload=self.payload_hash,
            )
            selection = list(self._buf)
            selection_status = "captured" if selection else "not_invoked_keep_all_or_metadata_only"
            capture_ok = bool(state["groups"]) and all(
                g.get("kept_length") is not None for g in state["groups"]
            )
            rec = {
                "pass": len(self.passes.get(req_id, [])),
                "capture_ok": capture_ok,
                "elapsed_ms": float(elapsed_ms),
                "immutable_prefix_tokens": self._protected,
                "selection_status": selection_status,
                "selection_groups": len(selection),
                "selection_kept_min": min((g["kept"] for g in selection if g.get("kept") is not None), default=None),
                "selection_kept_max": max((g["kept"] for g in selection if g.get("kept") is not None), default=None),
                "selection_group0_private_positions": selection[0].get("private_positions") if selection else None,
                "selection_group_sha256": [g.get("positions_sha256") for g in selection],
                "resident_state": state,
            }
            self.passes.setdefault(req_id, []).append(rec)
            if self.strict and not capture_ok:
                raise RuntimeError(f"identity capture produced no resident groups for {req_id}")
        except Exception as exc:
            self._record_failure(req_id, "resident_state_capture_error", exc)
            if self.strict:
                raise

    def record_final_request(
        self,
        *,
        req_id: str,
        model_runner: Any,
        request: Any,
        retained_positions: dict[str, Any] | None = None,
        stage: str = "scheduler_free_blocks_pre_free",
    ) -> None:
        """Capture a finished request's authoritative resident state before free."""

        try:
            ib = model_runner.input_batch
            row_idx = ib.req_id_to_index[str(req_id)]
            eff = ib.effective_seq_lens_cpu
            if eff is None:
                raise RuntimeError("input_batch.effective_seq_lens_cpu is unavailable")
            kept_lengths = eff[row_idx]
            tables = ib.block_table.block_tables
            if not ib.block_table.ragged or len(tables) != 1:
                raise RuntimeError("final ragged capture requires the single native 3-D BlockTable")
            immutable = int(getattr(request, "kvladder_immutable_prefix_tokens", 0) or 0)
            state = capture_resident_state(
                executor=self.executor,
                layer_kv_caches=model_runner.kv_caches,
                block_table=tables[0],
                row_idx=int(row_idx),
                kept_lengths_all=kept_lengths,
                include_payload=self.payload_hash,
                segments=(
                    {
                        "base": (0, retained_positions.get("base_row_lengths")),
                        "private_tail": (retained_positions.get("base_row_lengths"), None),
                    }
                    if retained_positions and retained_positions.get("base_row_lengths")
                    else {"full": (0, None)}
                ),
            )
            lengths = [int(g["kept_length"]) for g in state["groups"]]
            rec = {
                "schema": "kvladder-final-request-capture-v2",
                "capture_ok": bool(state["groups"]),
                "final_capture_stage": stage,
                "computed_tokens": int(getattr(request, "num_computed_tokens", -1)),
                "num_prompt_tokens": int(getattr(request, "num_prompt_tokens", -1)),
                "status": str(getattr(getattr(request, "status", None), "name", getattr(request, "status", ""))),
                "immutable_prefix_tokens": immutable,
                "logical_shared_prefix_tokens": int(getattr(request, "kvladder_logical_shared_prefix_tokens", 0) or 0),
                "derived_position_diagnostic": retained_positions,
                "resident_state": state,
                "kept_min": min(lengths, default=None),
                "kept_max": max(lengths, default=None),
            }
            self.finals[str(req_id)] = rec
        except Exception as exc:
            item = {"request_id": str(req_id), "reason": "final_request_capture_error", "error": f"{type(exc).__name__}: {exc}"}
            self.failures.append(item)
            self.finals[str(req_id)] = {
                "schema": "kvladder-final-request-capture-v2",
                "capture_ok": False,
                "final_capture_stage": stage,
                "capture_error": item,
                "computed_tokens": int(getattr(request, "num_computed_tokens", -1)),
                "status": str(getattr(getattr(request, "status", None), "name", getattr(request, "status", ""))),
            }


    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        req_ids = sorted(set(self.passes) | set(self.finals))
        for rid in req_ids:
            passes = self.passes.get(rid, [])
            final = self.finals.get(rid)
            source = final or (passes[-1] if passes else {})
            state = source.get("resident_state") or {}
            groups = state.get("groups") or []
            lengths = [int(g["kept_length"]) for g in groups if g.get("kept_length") is not None]
            out[rid] = {
                "schema": "kvladder-identity-request-summary-v4",
                "passes": len(passes),
                "capture_ok": bool(source.get("capture_ok")),
                "capture_error": source.get("capture_error"),
                "final_capture_stage": source.get("final_capture_stage", "post_compression_pass"),
                "computed_tokens": source.get("computed_tokens"),
                "status": source.get("status"),
                "final_state_sha256": state.get("state_sha256"),
                "final_combined_sha256": state.get("state_sha256"),
                "final_metadata_sha256": state.get("metadata_sha256"),
                "final_payload_sha256": state.get("payload_sha256"),
                "final_segment_payload_sha256": state.get("segment_payload_sha256"),
                "final_payload_hashed": bool(state.get("payload_hashed")),
                "final_kept_min": min(lengths, default=None),
                "final_kept_max": max(lengths, default=None),
                "final_group0_private_positions": (passes[-1].get("selection_group0_private_positions") if passes else None),
                "final_immutable_prefix_tokens": source.get("immutable_prefix_tokens"),
                "final_derived_position_diagnostic": source.get("derived_position_diagnostic"),
                "selection_status": (passes[-1].get("selection_status") if passes else "no_compression_pass"),
                "selection_groups": (passes[-1].get("selection_groups") if passes else 0),
                "pass_state_shas": [(p.get("resident_state") or {}).get("state_sha256") for p in passes],
                "pass_shas": [(p.get("resident_state") or {}).get("state_sha256") for p in passes],
                "pass_kept_max": [
                    max(
                        (int(g["kept_length"]) for g in (p.get("resident_state") or {}).get("groups", []) if g.get("kept_length") is not None),
                        default=None,
                    )
                    for p in passes
                ],
                "passes_detail": passes,
                "final_detail": final,
            }
        out["_capture_summary"] = {
            "schema": "kvladder-identity-capture-v4",
            "strict": self.strict,
            "payload_hash": self.payload_hash,
            "requests": len([k for k in out if not k.startswith("_")]),
            "final_requests": len(self.finals),
            "failures": self.failures,
        }
        return out


def capture_resident_state(
    *,
    executor: Any,
    layer_kv_caches: list[Any],
    block_table: Any,
    row_idx: int,
    kept_lengths_all: Any,
    include_payload: bool,
    segments: dict[str, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Capture the post-run resident ragged state from the live block table."""

    raw_kept_np = np.asarray(kept_lengths_all, dtype=np.int64)
    num_groups = int(executor.num_head_groups_per_layer)
    page_group_size = int(executor.page_group_size)
    block_size = int(executor.block_size)
    compressed_layer_ids = list(getattr(executor, "compressed_layer_ids", range(raw_kept_np.shape[0] if raw_kept_np.ndim else 1)))
    num_layers = len(compressed_layer_ids)
    expected_lengths = num_layers * num_groups
    flat_kept_np = raw_kept_np.reshape(-1)
    if flat_kept_np.size != expected_lengths:
        raise RuntimeError(
            f"kept length shape {list(raw_kept_np.shape)} has {flat_kept_np.size} values; "
            f"expected {expected_lengths} (= {num_layers} compressed layers x {num_groups} groups)"
        )
    kept_np = flat_kept_np.reshape(num_layers, num_groups)
    block_table_gpu = block_table.block_table.gpu

    def _validate_segment_bound_list(name: str, bound: Any) -> None:
        if isinstance(bound, (list, tuple)) and len(bound) != expected_lengths:
            raise RuntimeError(
                f"segment {name} bound list has {len(bound)} rows; "
                f"expected {expected_lengths} (= {num_layers} compressed layers x {num_groups} groups)"
            )

    if segments:
        for seg_name, (lo, hi) in segments.items():
            _validate_segment_bound_list(f"{seg_name}.lo", lo)
            _validate_segment_bound_list(f"{seg_name}.hi", hi)

    def _segment_bound(bound: Any, group_ordinal: int, default: int | None) -> int | None:
        if bound is None:
            return default
        if isinstance(bound, (list, tuple)):
            return int(bound[group_ordinal])
        return int(bound)

    groups: list[dict[str, Any]] = []
    payload_group_shas: list[str] = []
    group_ordinal = 0
    for static_idx, layer_idx in enumerate(compressed_layer_ids):
        kv_cache = layer_kv_caches[int(layer_idx)]
        layer_first = int(layer_idx) * num_groups
        for group_idx in range(num_groups):
            kept_length = int(kept_np[static_idx, group_idx])
            n_blocks = (kept_length + block_size - 1) // block_size
            group: dict[str, Any] = {
                "static_layer": int(static_idx),
                "physical_layer": int(layer_idx),
                "group": int(group_idx),
                "kept_length": kept_length,
                "n_blocks": int(n_blocks),
            }
            if include_payload:
                block_ids = block_table_gpu[row_idx, layer_first + group_idx, :n_blocks].long()
                # Null/stale rows are not evidence. Physical addresses are used
                # for validation/readback only, never included in identity hashes.
                if block_ids.numel() != n_blocks:
                    raise RuntimeError(
                        f"live block table has {block_ids.numel()} pages, expected {n_blocks}"
                    )
                if n_blocks and bool(((block_ids <= 0) | (block_ids >= kv_cache.shape[1])).any().item()):
                    raise RuntimeError("live block table contains null, negative, or out-of-range physical page IDs")
                if n_blocks and torch.unique(block_ids).numel() != n_blocks:
                    raise RuntimeError("live block table repeats a physical page within one logical row")
                payload = _logical_payload_sha256(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    kept_length=kept_length,
                    page_group_size=page_group_size,
                    block_size=block_size,
                )
                group["payload"] = payload
                payload_group_shas.append(str(payload["payload_sha256"]))
                if segments:
                    group_segments = {}
                    for name, (lo, hi) in segments.items():
                        seg_start = _segment_bound(lo, group_ordinal, 0)
                        seg_end = _segment_bound(hi, group_ordinal, kept_length)
                        seg = _logical_payload_sha256(
                            kv_cache=kv_cache,
                            block_ids=block_ids,
                            kept_length=kept_length,
                            page_group_size=page_group_size,
                            block_size=block_size,
                            logical_start=0 if seg_start is None else int(seg_start),
                            logical_end=kept_length if seg_end is None else int(seg_end),
                        )
                        group_segments[name] = seg
                    group["segments"] = group_segments
            groups.append(group)
            group_ordinal += 1

    metadata_groups = []
    for g in groups:
        mg = {k: v for k, v in g.items() if k not in ("payload", "segments")}
        if "payload" in g:
            mg["payload_meta"] = {
                k: v for k, v in g["payload"].items()
                if k != "payload_sha256"
            }
        if "segments" in g:
            mg["segment_meta"] = {
                name: {k: v for k, v in seg.items() if k != "payload_sha256"}
                for name, seg in g["segments"].items()
            }
        metadata_groups.append(mg)
    metadata = {
        "num_groups": len(groups),
        "page_group_size": page_group_size,
        "block_size": block_size,
        "kept_lengths": [int(v) for v in kept_np.reshape(-1).tolist()],
        "groups": metadata_groups,
    }
    metadata_sha = _json_sha256(metadata)
    payload_sha = hashlib.sha256("".join(payload_group_shas).encode()).hexdigest() if include_payload else None
    segment_payload_sha = None
    if include_payload and segments:
        segment_payload_sha = {
            name: hashlib.sha256(
                "".join(str((g.get("segments") or {}).get(name, {}).get("payload_sha256")) for g in groups).encode()
            ).hexdigest()
            for name in segments
        }
    state_sha = _json_sha256({"metadata_sha256": metadata_sha, "payload_sha256": payload_sha, "segment_payload_sha256": segment_payload_sha})
    return {
        "schema": "kvladder-resident-state-v1",
        "payload_hashed": bool(include_payload),
        "metadata": metadata,
        "metadata_sha256": metadata_sha,
        "payload_sha256": payload_sha,
        "segment_payload_sha256": segment_payload_sha,
        "state_sha256": state_sha,
        "groups": groups,
    }
