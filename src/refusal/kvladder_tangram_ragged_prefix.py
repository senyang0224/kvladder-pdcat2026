"""Narrow representation-aware prefix registry for Tangram ragged KV.

This module deliberately contains no generic version graph.  It owns one
immutable ragged representation per semantic lookup key, attaches the raw
physical page ids to independent requests, and optionally publishes one
frozen-prefix-plus-exact-tail child when a request finishes successfully.

Tangram supplies the native ragged allocator, scheduler and attention kernel.
The small duck-typed manager contract used here is implemented by the pinned
upstream patch in
``experiments/patches/tangram_44ed97e8_kvladder_ragged_prefix_cache.patch``.
"""

from __future__ import annotations

from dataclasses import field, asdict, dataclass, replace
import hashlib

import numpy as np
import json
import threading
import time
from typing import Any, Mapping, Sequence


class TangramRaggedPrefixError(RuntimeError):
    """The representation or ownership contract cannot be satisfied safely."""


class TangramRaggedPrefixMiss(TangramRaggedPrefixError):
    """No immutable ragged representation matches the semantic lookup key."""


class TangramRaggedPrefixRefusal(TangramRaggedPrefixError):
    """A request cannot safely share the published ragged prefix.

    This is not a scheduler/system failure.  The caller may roll back any
    partial shared-prefix state and admit the same request on the ordinary
    private path.
    """


def _nonempty(value: object, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_token_ids(token_ids: Sequence[int]) -> str:
    """Hash logical token identity independently of physical placement.

    The byte stream is every token id as a signed 8-byte little-endian integer, exactly as the original per-token
    loop produced it (``int(t).to_bytes(8, "little", signed=True)``); it is built with numpy in one call so that a
    32K-token prefix costs about 2 ms instead of 8 ms on the attach path (diagnostic of 2026-09-10)."""

    arr = np.asarray(token_ids, dtype="<i8")
    if arr.ndim != 1:
        raise ValueError("token_ids must be a flat sequence of integers")
    return hashlib.sha256(arr.tobytes()).hexdigest()


RetainedPositions = tuple[tuple[int, ...], ...]


def retained_positions_fingerprint(positions: RetainedPositions) -> str:
    return _canonical_digest(positions)


@dataclass(frozen=True, slots=True)
class TangramRaggedPrefixLookup:
    """Semantic lookup contract known before executing a deterministic selector.

    The exact retained-position fingerprint is added to the published identity.
    Physical page ids are absent from both objects.
    """

    token_prefix_digest: str
    prefix_token_count: int
    model_id: str
    model_config_digest: str
    tokenizer_id: str
    position_contract: str
    kv_dtype: str
    compressor_id: str
    compressor_revision: str
    compression_policy_digest: str
    layout_contract: str
    retained_position_rule: str
    sink_tokens: int = 0
    adapter_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "token_prefix_digest",
            "model_id",
            "model_config_digest",
            "tokenizer_id",
            "position_contract",
            "kv_dtype",
            "compressor_id",
            "compressor_revision",
            "compression_policy_digest",
            "layout_contract",
            "retained_position_rule",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.prefix_token_count) is not int or self.prefix_token_count <= 0:
            raise ValueError("prefix_token_count must be a positive integer")
        if type(self.sink_tokens) is not int or self.sink_tokens < 0:
            raise ValueError("sink_tokens must be a nonnegative integer")
        if self.sink_tokens > self.prefix_token_count:
            raise ValueError("sink_tokens exceeds the logical prefix")
        if self.adapter_id is not None:
            object.__setattr__(
                self, "adapter_id", _nonempty(self.adapter_id, "adapter_id")
            )

    @property
    def semantic_digest(self) -> str:
        return _canonical_digest(asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TangramRaggedPrefixLookup":
        required = {
            "token_prefix_digest",
            "prefix_token_count",
            "model_id",
            "model_config_digest",
            "tokenizer_id",
            "position_contract",
            "kv_dtype",
            "compressor_id",
            "compressor_revision",
            "compression_policy_digest",
            "layout_contract",
            "retained_position_rule",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError("ragged prefix spec is missing: " + ", ".join(missing))
        return cls(
            token_prefix_digest=str(value["token_prefix_digest"]),
            prefix_token_count=int(value["prefix_token_count"]),
            model_id=str(value["model_id"]),
            model_config_digest=str(value["model_config_digest"]),
            tokenizer_id=str(value["tokenizer_id"]),
            position_contract=str(value["position_contract"]),
            kv_dtype=str(value["kv_dtype"]),
            compressor_id=str(value["compressor_id"]),
            compressor_revision=str(value["compressor_revision"]),
            compression_policy_digest=str(value["compression_policy_digest"]),
            layout_contract=str(value["layout_contract"]),
            retained_position_rule=str(value["retained_position_rule"]),
            sink_tokens=int(value.get("sink_tokens", 0)),
            adapter_id=(
                None if value.get("adapter_id") is None else str(value["adapter_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class TangramRaggedPrefixIdentity:
    lookup: TangramRaggedPrefixLookup
    selection_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_fingerprint",
            _nonempty(self.selection_fingerprint, "selection_fingerprint"),
        )

    @property
    def semantic_digest(self) -> str:
        return _canonical_digest(
            {
                "lookup": asdict(self.lookup),
                "selection_fingerprint": self.selection_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class TangramRaggedPrefixRequestSpec:
    lookup: TangramRaggedPrefixLookup
    commit_on_success: bool = False

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "TangramRaggedPrefixRequestSpec":
        return cls(
            lookup=TangramRaggedPrefixLookup.from_mapping(value),
            commit_on_success=bool(value.get("commit_on_success", False)),
        )


@dataclass(frozen=True, slots=True)
class TangramRaggedPrefixSnapshot:
    """Exact worker-side ragged table and per-row occupancy.

    The flat page-id tuple is computed once at construction (stage B of the 2026-09-10 diagnostic): an attach reads
    it five times, and regenerating 38,912 ids with a uniqueness check each time cost about 8 ms per attach. The two
    cache fields do not take part in equality or hashing."""

    block_table: tuple[tuple[int, ...], ...]
    block_counts: tuple[int, ...]
    effective_seq_lens: tuple[int, ...]
    _flat_ids: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _flat_ids_unique: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = tuple(
            page_id
            for row, count in zip(self.block_table, self.block_counts)
            for page_id in row[:count]
        )
        object.__setattr__(self, "_flat_ids", ids)
        object.__setattr__(self, "_flat_ids_unique", len(set(ids)) == len(ids))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TangramRaggedPrefixSnapshot":
        block_table = tuple(
            tuple(int(page_id) for page_id in row)
            for row in value["block_table"]  # type: ignore[index]
        )
        block_counts = tuple(int(count) for count in value["block_counts"])  # type: ignore[index]
        effective = tuple(
            int(length) for length in value["effective_seq_lens"]  # type: ignore[index]
        )
        snapshot = cls(block_table, block_counts, effective)
        snapshot.validate_shape()
        return snapshot

    def validate_shape(self) -> None:
        rows = len(self.block_table)
        if (
            rows == 0
            or len(self.block_counts) != rows
            or len(self.effective_seq_lens) != rows
        ):
            raise TangramRaggedPrefixError("ragged snapshot row counts disagree")
        widths = {len(row) for row in self.block_table}
        if len(widths) != 1:
            raise TangramRaggedPrefixError("ragged block-table rows have unequal width")
        width = next(iter(widths))
        for count, length in zip(self.block_counts, self.effective_seq_lens):
            if not 0 <= count <= width or length < 0:
                raise TangramRaggedPrefixError("invalid ragged row count or length")

    def validate_immutable_boundary(self, *, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        for row, count, length in zip(
            self.block_table, self.block_counts, self.effective_seq_lens
        ):
            if length % block_size:
                raise TangramRaggedPrefixRefusal(
                    "shared ragged boundary must be block aligned; partial-page COW "
                    "is outside the P2 integration; "
                    f"block_size={block_size}, effective_seq_lens="
                    f"{self.effective_seq_lens}"
                )
            expected = length // block_size
            if count != expected:
                raise TangramRaggedPrefixError(
                    "ragged row page count does not match effective length"
                )
            if any(page_id <= 0 for page_id in row[:count]):
                raise TangramRaggedPrefixError(
                    "live ragged page ids must be positive"
                )

    @property
    def physical_block_ids(self) -> tuple[int, ...]:
        if not self._flat_ids_unique:
            raise TangramRaggedPrefixError(
                "one physical ragged page appears in multiple live table cells"
            )
        return self._flat_ids

    def worker_restore(self) -> dict[str, object]:
        return {
            "block_table": [list(row) for row in self.block_table],
            "block_counts": list(self.block_counts),
            "effective_seq_lens": list(self.effective_seq_lens),
            "base_block_ids": list(self.physical_block_ids),
        }


def trim_snapshot_to_length(
    snapshot: TangramRaggedPrefixSnapshot, *, block_size: int
) -> tuple[TangramRaggedPrefixSnapshot, tuple[int, ...]]:
    """Drop trailing pages that hold no computed token before publishing.

    vLLM allocates the slot for the next token ahead of computing it, so a
    finished branch may own one more page per row than ``effective_seq_len //
    block_size``.  Those pages carry no KV of the published prefix; the request
    releases them on finish.  Publication must not retain them, otherwise the
    block-aligned boundary check refuses a well-formed commit.  Returns the
    trimmed snapshot and the number of pages dropped per row.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rows: list[tuple[int, ...]] = []
    counts: list[int] = []
    dropped: list[int] = []
    for row, count, length in zip(
        snapshot.block_table, snapshot.block_counts, snapshot.effective_seq_lens
    ):
        needed = length // block_size
        if length % block_size:
            needed += 1  # partial page still holds computed tokens; keep it
        if count < needed:
            raise TangramRaggedPrefixError(
                "ragged row owns fewer pages than its computed length needs"
            )
        rows.append(tuple(row[:needed]) + (0,) * (len(row) - needed))
        counts.append(needed)
        dropped.append(count - needed)
    trimmed = TangramRaggedPrefixSnapshot(
        tuple(rows), tuple(counts), tuple(snapshot.effective_seq_lens)
    )
    trimmed.validate_shape()
    return trimmed, tuple(dropped)


def _derive_retained_positions(
    lookup: TangramRaggedPrefixLookup,
    effective_seq_lens: Sequence[int],
) -> RetainedPositions:
    if lookup.retained_position_rule != "streamingllm-sink-recent-v1":
        raise TangramRaggedPrefixRefusal(
            "P2 requires an explicit deterministic retained-position rule"
        )
    retained: list[tuple[int, ...]] = []
    logical = lookup.prefix_token_count
    sink = lookup.sink_tokens
    for length in effective_seq_lens:
        if not sink <= length <= logical:
            raise TangramRaggedPrefixRefusal(
                "effective length is incompatible with sink/recent selection"
            )
        recent = length - sink
        recent_start = logical - recent
        if recent_start < sink:
            positions = tuple(range(logical))
        else:
            positions = tuple(range(sink)) + tuple(range(recent_start, logical))
        if len(positions) != length or tuple(sorted(set(positions))) != positions:
            raise TangramRaggedPrefixError("derived retained positions are invalid")
        retained.append(positions)
    return tuple(retained)


@dataclass(frozen=True, slots=True)
class TangramRaggedPrefixEntry:
    identity: TangramRaggedPrefixIdentity
    snapshot: TangramRaggedPrefixSnapshot
    retained_positions: RetainedPositions
    physical_block_bytes: int

    @property
    def physical_base_bytes(self) -> int:
        return len(self.snapshot.physical_block_ids) * self.physical_block_bytes

    @property
    def metadata_bytes(self) -> int:
        live_pages = len(self.snapshot.physical_block_ids)
        rows = len(self.snapshot.block_counts)
        retained = sum(len(row) for row in self.retained_positions)
        return live_pages * 4 + rows * 8 + retained * 8


@dataclass(slots=True)
class _PersistentRaggedDescriptor:
    """Scheduler-side mirror of one worker ragged row.

    The immutable base table is copied once at attach.  Subsequent decode steps
    only advance ``private_token_delta`` (O(1)); physical page-table changes
    append the allocator's already-produced block-id delta.  The full snapshot
    is materialized only for a lifecycle boundary or diagnostic.
    """

    descriptor_id: str
    block_table: list[list[int]]
    block_counts: list[int]
    base_effective_seq_lens: tuple[int, ...]
    flat_block_ids: list[int]
    flat_block_id_set: set[int]
    private_token_delta: int = 0
    schedule_updates: int = 0
    block_delta_updates: int = 0
    delta_block_ids: int = 0

    @classmethod
    def from_snapshot(
        cls,
        *,
        descriptor_id: str,
        snapshot: TangramRaggedPrefixSnapshot,
    ) -> "_PersistentRaggedDescriptor":
        return cls(
            descriptor_id=descriptor_id,
            block_table=[list(row) for row in snapshot.block_table],
            block_counts=list(snapshot.block_counts),
            base_effective_seq_lens=snapshot.effective_seq_lens,
            flat_block_ids=list(snapshot.physical_block_ids),
            flat_block_id_set=set(snapshot.physical_block_ids),
        )

    def _append_grouped(self, block_ids: Sequence[int]) -> None:
        ids = [int(value) for value in block_ids]
        if not ids:
            return
        if any(value <= 0 for value in ids) or len(set(ids)) != len(ids):
            raise TangramRaggedPrefixError(
                "descriptor block delta must be unique and non-null"
            )
        if self.flat_block_id_set.intersection(ids):
            raise TangramRaggedPrefixError(
                "descriptor block delta aliases an existing physical page"
            )
        num_groups = len(self.block_counts)
        total = len(ids)
        sum_starts = sum(self.block_counts)
        if num_groups == 0 or (sum_starts + total) % num_groups:
            raise TangramRaggedPrefixError(
                "descriptor block delta cannot form a uniform ragged target"
            )
        target = (sum_starts + total) // num_groups
        if target < max(self.block_counts):
            raise TangramRaggedPrefixError(
                "descriptor block delta would shrink a ragged row"
            )
        for row in self.block_table:
            if len(row) < target:
                row.extend([0] * (target - len(row)))
        position = 0
        for group, start in enumerate(self.block_counts):
            count = target - start
            if count:
                self.block_table[group][start:target] = ids[
                    position : position + count
                ]
                position += count
        if position != total:
            raise TangramRaggedPrefixError(
                "descriptor did not consume the full physical page delta"
            )
        self.block_counts[:] = [target] * num_groups
        self.flat_block_ids.extend(ids)
        self.flat_block_id_set.update(ids)
        self.block_delta_updates += 1
        self.delta_block_ids += total

    def apply_new_request_state(
        self,
        *,
        full_block_ids: Sequence[int],
        scheduled_tokens: int,
    ) -> None:
        full = [int(value) for value in full_block_ids]
        if full[: len(self.flat_block_ids)] != self.flat_block_ids:
            raise TangramRaggedPrefixError(
                "descriptor attach disagrees with scheduler raw page ownership"
            )
        self._append_grouped(full[len(self.flat_block_ids) :])
        self.advance(scheduled_tokens)

    def apply_cached_delta(
        self,
        *,
        new_block_ids: Sequence[int],
        scheduled_tokens: int,
    ) -> None:
        self._append_grouped(new_block_ids)
        self.advance(scheduled_tokens)

    def advance(self, scheduled_tokens: int) -> None:
        delta = int(scheduled_tokens)
        if delta < 0:
            raise TangramRaggedPrefixError("scheduled token delta must be nonnegative")
        self.private_token_delta += delta
        self.schedule_updates += 1

    def snapshot(self) -> TangramRaggedPrefixSnapshot:
        effective = tuple(
            length + self.private_token_delta
            for length in self.base_effective_seq_lens
        )
        snapshot = TangramRaggedPrefixSnapshot(
            tuple(tuple(row) for row in self.block_table),
            tuple(self.block_counts),
            effective,
        )
        snapshot.validate_shape()
        if set(snapshot.physical_block_ids) != set(self.flat_block_ids):
            raise TangramRaggedPrefixError(
                "descriptor table and allocator page ownership diverged"
            )
        return snapshot


@dataclass(slots=True)
class _LiveRequest:
    spec: TangramRaggedPrefixRequestSpec
    parent: TangramRaggedPrefixEntry | None
    attached: bool
    descriptor_id: str
    descriptor: _PersistentRaggedDescriptor | None = None
    latest_snapshot: TangramRaggedPrefixSnapshot | None = None


class TangramRaggedPrefixCache:
    """Scheduler-facing immutable ragged prefix registry."""

    EXTRA_ARG = "kvladder_ragged_prefix"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._manager: Any | None = None
        self._entries: dict[str, TangramRaggedPrefixEntry] = {}
        self._requests: dict[str, _LiveRequest] = {}
        self._events: list[dict[str, object]] = []
        self._counters: dict[str, int] = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "published": 0,
            "attaches": 0,
            "commits": 0,
            "aborts": 0,
            "preemptions": 0,
            "native_worker_snapshots": 0,
            "native_hit_worker_snapshots": 0,
            "descriptor_schedule_updates": 0,
            "descriptor_block_delta_updates": 0,
            "descriptor_delta_block_ids": 0,
            "descriptor_full_snapshot_materializations": 0,
            "cache_hit_tokens": 0,
            "payload_copy_bytes": 0,
            "base_copy_bytes": 0,
            "dense_materialization_bytes": 0,
            "immutable_rewrite_bytes": 0,
            "evictions": 0,
            "evicted_pages": 0,
            "refusals": 0,
            "private_fallbacks": 0,
            "publication_refusals": 0,
        }
        self._attach_latencies_ns: list[int] = []
        # v2 policy (K-axis v2 lock): last publish/attach time per entry, for whole-entry LRU eviction.
        self._last_used: dict[str, float] = {}

    def bind_kv_cache_manager(self, manager: Any) -> None:
        if not bool(getattr(manager, "ragged", False)):
            raise TangramRaggedPrefixError(
                "Tangram ragged prefix cache needs ragged KV"
            )
        if int(getattr(manager, "ragged_block_size", 0)) <= 0:
            raise TangramRaggedPrefixError("ragged manager did not expose block size")
        if int(getattr(manager, "ragged_physical_block_bytes", 0)) <= 0:
            raise TangramRaggedPrefixError("ragged manager did not expose block bytes")
        with self._lock:
            if self._manager is not None and self._manager is not manager:
                raise TangramRaggedPrefixError(
                    "cache is already bound to another manager"
                )
            self._manager = manager

    @property
    def manager(self) -> Any:
        if self._manager is None:
            raise TangramRaggedPrefixError("ragged prefix cache is not allocator-bound")
        return self._manager

    @staticmethod
    def _request_extra(request: Any) -> Mapping[str, object] | None:
        sampling = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling, "extra_args", None)
        if not isinstance(extra_args, Mapping):
            return None
        value = extra_args.get(TangramRaggedPrefixCache.EXTRA_ARG)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("ragged prefix extra arg must be a mapping")
        return value

    @staticmethod
    def _success(request: Any) -> bool:
        status = str(getattr(getattr(request, "status", None), "name", ""))
        return status in {"FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"}

    @staticmethod
    def _reset_request_for_private(request: Any) -> None:
        request.num_computed_tokens = 0
        request.num_cached_tokens = -1
        request.compress_max_eff_seq_len = None
        request.compression_done = False
        request.kvladder_shared_prefix_ready = False
        request.kvladder_private_compression_state = "not_applicable"
        request.kvladder_immutable_prefix_tokens = 0
        request.kvladder_logical_shared_prefix_tokens = 0
        request.kvladder_ragged_prefix_metadata = None

    def prepare_private_fallback(
        self,
        request: Any,
        exc: BaseException | None = None,
        *,
        reason: str | None = None,
    ) -> bool:
        """Roll a recoverable sharing refusal back to ordinary private execution.

        Returns ``True`` only for KVLadder refusal conditions.  Unknown
        allocator, scheduler, or programmer failures remain fatal to avoid
        hiding a corrupt ownership state.
        """

        if exc is not None and not isinstance(exc, TangramRaggedPrefixRefusal):
            return False
        request_id = str(getattr(request, "request_id", ""))
        with self._lock:
            live = self._requests.pop(request_id, None)
            self._counters["refusals"] += 1
            self._counters["private_fallbacks"] += 1
            self._events.append(
                {
                    "event": "private_fallback",
                    "request_id": request_id,
                    "reason": reason or (str(exc) if exc is not None else "refusal"),
                    "had_live_record": live is not None,
                    "had_attached_parent": bool(
                        live is not None and live.parent is not None
                    ),
                }
            )
        if live is not None and live.attached:
            self.manager.detach_ragged_prefix_block_ids(request_id)
        self._reset_request_for_private(request)
        return True

    def prepare_finish_refusal(
        self, request: Any, exc: BaseException, *, reason: str | None = None
    ) -> bool:
        """Convert a recoverable publication/commit refusal into no-publication.

        The request has already completed on its private continuation; only the
        optional publication is refused.
        """

        if not isinstance(exc, TangramRaggedPrefixRefusal):
            return False
        request_id = str(getattr(request, "request_id", ""))
        request.kvladder_ragged_prefix_metadata = None
        with self._lock:
            self._counters["refusals"] += 1
            self._counters["publication_refusals"] += 1
            self._events.append(
                {
                    "event": "publication_refused",
                    "request_id": request_id,
                    "reason": reason or str(exc),
                }
            )
        return True

    def on_request_arrival(self, request: Any) -> bool:
        """Lookup and attach before Tangram scheduler admission."""

        try:
            value = self._request_extra(request)
            if value is None:
                return False
            spec = TangramRaggedPrefixRequestSpec.from_mapping(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise TangramRaggedPrefixRefusal(
                "invalid ragged prefix request spec; falling back to private "
                "execution"
            ) from exc
        except TangramRaggedPrefixError:
            raise
        started = time.perf_counter_ns()
        request_id = _nonempty(request.request_id, "request_id")
        prompt = tuple(int(token) for token in request.prompt_token_ids)
        lookup = spec.lookup
        descriptor_id = _canonical_digest(
            {
                "request_id": request_id,
                "lookup_digest": lookup.semantic_digest,
                "descriptor_contract": "scheduler-delta-v1",
            }
        )
        if len(prompt) < lookup.prefix_token_count:
            raise TangramRaggedPrefixRefusal(
                "request is shorter than its prefix contract"
            )
        observed_digest = digest_token_ids(prompt[: lookup.prefix_token_count])
        if observed_digest != lookup.token_prefix_digest:
            raise TangramRaggedPrefixRefusal(
                "request token prefix does not match semantic cache identity"
            )
        with self._lock:
            if request_id in self._requests:
                raise TangramRaggedPrefixError("request is already registered")
            self._counters["lookups"] += 1
            entry = self._entries.get(lookup.semantic_digest)
            if entry is None:
                self._counters["misses"] += 1
                live = _LiveRequest(
                    spec=spec,
                    parent=None,
                    attached=False,
                    descriptor_id=descriptor_id,
                )
                self._requests[request_id] = live
                request.kvladder_ragged_prefix_metadata = {
                    "capture": True,
                    "capture_mode": "publish_once",
                    "descriptor_id": descriptor_id,
                }
                self._events.append(
                    {"event": "lookup_miss", "request_id": request_id}
                )
                return False
            if len(prompt) == lookup.prefix_token_count:
                raise TangramRaggedPrefixRefusal(
                    "a cache hit needs at least one private continuation token"
                )

        try:
            self.manager.attach_ragged_prefix_block_ids(
                request_id, entry.snapshot.physical_block_ids
            )
            request.num_computed_tokens = lookup.prefix_token_count
            request.num_cached_tokens = lookup.prefix_token_count
            request.compress_max_eff_seq_len = max(entry.snapshot.effective_seq_lens)
            # The immutable base is ready, but Tangram still owns the private
            # continuation lifecycle.  ``compression_done=True`` here used to
            # suppress every subsequent scheduler compression directive.
            # P10 deliberately splits those two states.
            immutable_lengths = set(entry.snapshot.effective_seq_lens)
            if len(immutable_lengths) != 1:
                raise TangramRaggedPrefixRefusal(
                    "Tangram-Complete P10 currently requires one uniform "
                    "block-aligned immutable-prefix length; exact per-head "
                    "continuation protection is outside this fixed scope"
                )
            immutable_prefix_tokens = next(iter(immutable_lengths))
            request.kvladder_shared_prefix_ready = True
            request.kvladder_private_compression_state = "pending"
            request.kvladder_immutable_prefix_tokens = immutable_prefix_tokens
            request.kvladder_logical_shared_prefix_tokens = (
                lookup.prefix_token_count
            )
            request.compression_done = False
            # A branch that may commit needs the worker's authoritative ragged
            # table at its finish (private compaction frees pages the O(1)
            # scheduler-side descriptor does not observe); ordinary hits keep the
            # snapshot-free stable-descriptor path (P9).
            request.kvladder_ragged_prefix_metadata = {
                "capture": bool(spec.commit_on_success),
                "capture_mode": (
                    "commit_capture" if spec.commit_on_success else "stable_descriptor"
                ),
                "descriptor_id": descriptor_id,
                "parent_semantic_digest": entry.identity.semantic_digest,
                "shared_prefix_ready": True,
                "private_compression_state": "pending",
                "immutable_prefix_tokens": immutable_prefix_tokens,
                "logical_shared_prefix_tokens": lookup.prefix_token_count,
                "restore": entry.snapshot.worker_restore(),
            }
        except BaseException:
            # Real Tangram attach_shared_block_ids validates before refcounting,
            # then installs req ownership immediately after retain.  Detach is
            # nevertheless safe/no-op if attach failed before the req map entry,
            # and it rolls back both expected post-attach refusals and unknown
            # failures that occur after ownership is visible.
            self.manager.detach_ragged_prefix_block_ids(request_id)
            raise
        latency = time.perf_counter_ns() - started
        with self._lock:
            self._requests[request_id] = _LiveRequest(
                spec=spec,
                parent=entry,
                attached=True,
                descriptor_id=descriptor_id,
                descriptor=_PersistentRaggedDescriptor.from_snapshot(
                    descriptor_id=descriptor_id,
                    snapshot=entry.snapshot,
                ),
            )
            self._counters["hits"] += 1
            self._counters["attaches"] += 1
            self._counters["cache_hit_tokens"] += lookup.prefix_token_count
            self._attach_latencies_ns.append(latency)
            self._last_used[lookup.semantic_digest] = time.monotonic()
            self._events.append(
                {
                    "event": "lookup_hit_attach",
                    "request_id": request_id,
                    "semantic_digest": entry.identity.semantic_digest,
                    "raw_page_ids": entry.snapshot.physical_block_ids,
                    "attach_latency_ns": latency,
                }
            )
        return True

    def on_schedule_output(self, scheduler_output: Any) -> None:
        """Apply O(1) token advances and allocator page-table deltas.

        This replaces the old per-step worker export of every retained length
        and every 3-D block-table cell.  The scheduler already transports exact
        new physical block ids to the worker, so mirroring that delta preserves
        the attention-visible layout without another full snapshot.
        """

        scheduled = scheduler_output.num_scheduled_tokens
        updates = 0
        block_updates = 0
        block_ids = 0
        with self._lock:
            for data in scheduler_output.scheduled_new_reqs:
                request_id = str(data.req_id)
                live = self._requests.get(request_id)
                if live is None or live.descriptor is None:
                    continue
                if len(data.block_ids) != 1:
                    raise TangramRaggedPrefixError(
                        "stable ragged descriptor requires one KV cache group"
                    )
                before_updates = live.descriptor.block_delta_updates
                before_ids = live.descriptor.delta_block_ids
                live.descriptor.apply_new_request_state(
                    full_block_ids=data.block_ids[0],
                    scheduled_tokens=int(scheduled[request_id]),
                )
                updates += 1
                block_updates += (
                    live.descriptor.block_delta_updates - before_updates
                )
                block_ids += live.descriptor.delta_block_ids - before_ids

            cached = scheduler_output.scheduled_cached_reqs
            for request_id, delta in zip(cached.req_ids, cached.new_block_ids):
                request_id = str(request_id)
                live = self._requests.get(request_id)
                if live is None or live.descriptor is None:
                    continue
                if request_id in cached.resumed_req_ids:
                    raise TangramRaggedPrefixError(
                        "a preempted ragged descriptor cannot be resumed"
                    )
                if delta is not None and len(delta) != 1:
                    raise TangramRaggedPrefixError(
                        "stable ragged descriptor requires one KV cache group"
                    )
                new_ids = () if delta is None else delta[0]
                before_updates = live.descriptor.block_delta_updates
                before_ids = live.descriptor.delta_block_ids
                live.descriptor.apply_cached_delta(
                    new_block_ids=new_ids,
                    scheduled_tokens=int(scheduled[request_id]),
                )
                updates += 1
                block_updates += (
                    live.descriptor.block_delta_updates - before_updates
                )
                block_ids += live.descriptor.delta_block_ids - before_ids

            self._counters["descriptor_schedule_updates"] += updates
            self._counters["descriptor_block_delta_updates"] += block_updates
            self._counters["descriptor_delta_block_ids"] += block_ids

    def _publish(
        self,
        *,
        lookup: TangramRaggedPrefixLookup,
        snapshot: TangramRaggedPrefixSnapshot,
        retained_positions: RetainedPositions,
        event: str,
        request_id: str,
    ) -> TangramRaggedPrefixEntry:
        snapshot.validate_immutable_boundary(
            block_size=int(self.manager.ragged_block_size)
        )
        if len(retained_positions) != len(snapshot.effective_seq_lens):
            raise TangramRaggedPrefixRefusal(
                "retained-position rows do not match table"
            )
        if tuple(map(len, retained_positions)) != snapshot.effective_seq_lens:
            raise TangramRaggedPrefixRefusal(
                "retained-position lengths do not match physical row lengths"
            )
        identity = TangramRaggedPrefixIdentity(
            lookup=lookup,
            selection_fingerprint=retained_positions_fingerprint(retained_positions),
        )
        lookup_digest = lookup.semantic_digest
        with self._lock:
            existing = self._entries.get(lookup_digest)
            if existing is not None:
                if existing.identity != identity:
                    raise TangramRaggedPrefixRefusal(
                        "deterministic lookup produced another retained representation"
                    )
                return existing
        self.manager.retain_ragged_prefix_block_ids(snapshot.physical_block_ids)
        entry = TangramRaggedPrefixEntry(
            identity=identity,
            snapshot=snapshot,
            retained_positions=retained_positions,
            physical_block_bytes=int(self.manager.ragged_physical_block_bytes),
        )
        with self._lock:
            if lookup_digest in self._entries:
                self.manager.release_ragged_prefix_block_ids(
                    snapshot.physical_block_ids
                )
                return self._entries[lookup_digest]
            self._entries[lookup_digest] = entry
            self._last_used[lookup_digest] = time.monotonic()
            self._counters["published"] += 1
            self._events.append(
                {
                    "event": event,
                    "request_id": request_id,
                    "lookup_digest": lookup_digest,
                    "semantic_digest": identity.semantic_digest,
                    "raw_page_ids": snapshot.physical_block_ids,
                    "physical_base_bytes": entry.physical_base_bytes,
                }
            )
        return entry

    def on_model_output(self, request: Any, value: Mapping[str, object]) -> None:
        """Ingest a one-shot publish/diagnostic worker ragged snapshot."""

        request_id = str(request.request_id)
        snapshot = TangramRaggedPrefixSnapshot.from_mapping(value)
        with self._lock:
            live = self._requests.get(request_id)
            if live is None:
                return
            live.latest_snapshot = snapshot
            # An explicit worker snapshot is authoritative. Normal prefix-hit
            # decode never sends one; tests/diagnostics and initial publish may.
            live.descriptor = _PersistentRaggedDescriptor.from_snapshot(
                descriptor_id=live.descriptor_id,
                snapshot=snapshot,
            )
            self._counters["native_worker_snapshots"] += 1
            parent_matches = None
            if live.parent is not None:
                self._counters["native_hit_worker_snapshots"] += 1
                parent_snapshot = live.parent.snapshot
                parent_matches = all(
                    child_count >= parent_count
                    and child_row[:parent_count] == parent_row[:parent_count]
                    for parent_row, parent_count, child_row, child_count in zip(
                        parent_snapshot.block_table,
                        parent_snapshot.block_counts,
                        snapshot.block_table,
                        snapshot.block_counts,
                    )
                )
            native_ids = snapshot.physical_block_ids
            self._events.append(
                {
                    "event": "native_worker_ragged_snapshot",
                    "request_id": request_id,
                    "native_raw_page_count": len(native_ids),
                    "native_raw_page_digest": _canonical_digest(native_ids),
                    "native_raw_page_id_sample": list(native_ids[:8]),
                    "parent_raw_page_ids_match": parent_matches,
                }
            )
            should_publish = (
                live.parent is None
                and int(request.num_computed_tokens)
                == live.spec.lookup.prefix_token_count
                and bool(request.compression_done)
            )
        if should_publish:
            try:
                retained = _derive_retained_positions(
                    live.spec.lookup, snapshot.effective_seq_lens
                )
                self._publish(
                    lookup=live.spec.lookup,
                    snapshot=snapshot,
                    retained_positions=retained,
                    event="compressed_prefix_published",
                    request_id=request_id,
                )
            except TangramRaggedPrefixRefusal as exc:
                self.prepare_finish_refusal(
                    request, exc, reason="publish:" + str(exc)
                )

    def prepare_request_finish(self, request: Any) -> TangramRaggedPrefixEntry | None:
        """Publish only bounded frozen-prefix + exact-tail state on success."""

        request_id = str(request.request_id)
        with self._lock:
            live = self._requests.get(request_id)
            if live is None or not live.spec.commit_on_success or live.parent is None:
                return None
            if not self._success(request):
                return None
            # Prefer the worker's authoritative snapshot (captured for commit
            # branches); the scheduler-side descriptor is a fallback only.
            if live.latest_snapshot is not None:
                snapshot = live.latest_snapshot
            elif live.descriptor is not None:
                snapshot = live.descriptor.snapshot()
                self._counters["descriptor_full_snapshot_materializations"] += 1
            else:
                snapshot = None
        if snapshot is None:
            raise TangramRaggedPrefixRefusal("commit has no worker ragged snapshot")
        snapshot, dropped_pages = trim_snapshot_to_length(
            snapshot, block_size=int(self.manager.ragged_block_size)
        )
        with self._lock:
            self._events.append(
                {
                    "event": "commit_trimmed_slack_pages",
                    "request_id": request_id,
                    "dropped_pages_per_row": list(dropped_pages),
                }
            )
        parent = live.parent
        computed = int(request.num_computed_tokens)
        parent_tokens = parent.identity.lookup.prefix_token_count
        if computed <= parent_tokens:
            raise TangramRaggedPrefixRefusal("commit has no private computed tail")
        tokens = tuple(int(token) for token in request.all_token_ids[:computed])
        child_lookup = replace(
            parent.identity.lookup,
            token_prefix_digest=digest_token_ids(tokens),
            prefix_token_count=computed,
            compression_policy_digest=(
                parent.identity.lookup.compression_policy_digest
                + ":frozen-exact-tail-v1"
            ),
        )
        exact_tail = tuple(range(parent_tokens, computed))
        retained = tuple(
            tuple(parent_row) + exact_tail for parent_row in parent.retained_positions
        )
        child = self._publish(
            lookup=child_lookup,
            snapshot=snapshot,
            retained_positions=retained,
            event="branch_committed",
            request_id=request_id,
        )
        with self._lock:
            self._counters["commits"] += 1
        return child

    def on_request_finished(self, request: Any, *, reason: str) -> None:
        request_id = str(request.request_id)
        with self._lock:
            live = self._requests.pop(request_id, None)
            if live is None:
                return
            if live.attached and not self._success(request):
                self._counters["aborts"] += 1
            self._events.append(
                {
                    "event": "request_finished",
                    "request_id": request_id,
                    "success": self._success(request),
                    "reason": reason,
                }
            )

    def on_request_preempted(self, request: Any) -> None:
        request_id = str(request.request_id)
        with self._lock:
            live = self._requests.pop(request_id, None)
            if live is not None:
                self._counters["preemptions"] += 1

    def entry(self, lookup: TangramRaggedPrefixLookup) -> TangramRaggedPrefixEntry:
        with self._lock:
            try:
                return self._entries[lookup.semantic_digest]
            except KeyError as exc:
                raise TangramRaggedPrefixMiss(lookup.semantic_digest) from exc

    def evictable_entries(self) -> list[str]:
        """Lookup digests of entries with no live reference, least recently used first.

        A live reference is any registered request whose parent is the entry: a running
        branch, or a branch that attached at arrival and is still waiting for admission.
        """
        with self._lock:
            referenced = {
                live.parent.identity.lookup.semantic_digest
                for live in self._requests.values()
                if live.parent is not None
            }
            cands = [d for d in self._entries if d not in referenced]
            cands.sort(key=lambda d: self._last_used.get(d, 0.0))
            return cands

    def evict_lru(
        self, pages_needed: int | None = None, *, reason: str = "blocked"
    ) -> dict[str, object]:
        """K-axis v2 policy: whole-entry LRU eviction of unreferenced entries.

        Evicts the least recently attached or published entry that no live request
        references (running or waiting).  With ``pages_needed`` it keeps evicting until
        that many pages were released or no candidate remains; without it, one entry.
        Pages go back to the allocator through the same release call as ``drain``.
        Intended to be called by the admission loop only when the head of the waiting
        queue is blocked and nothing runs (reactive), never proactively.
        """
        evicted: list[dict[str, object]] = []
        pages = 0
        while True:
            with self._lock:
                cands = self.evictable_entries()
                if not cands:
                    break
                if pages_needed is not None and pages >= int(pages_needed):
                    break
                victim = cands[0]
                entry = self._entries.pop(victim)
                self._last_used.pop(victim, None)
                n_pages = len(entry.snapshot.physical_block_ids)
                self._counters["evictions"] += 1
                self._counters["evicted_pages"] += n_pages
                self._events.append(
                    {
                        "event": "entry_evicted",
                        "lookup_digest": victim,
                        "semantic_digest": entry.identity.semantic_digest,
                        "pages": n_pages,
                        "reason": reason,
                    }
                )
            self.manager.release_ragged_prefix_block_ids(
                entry.snapshot.physical_block_ids
            )
            evicted.append({"lookup_digest": victim, "pages": n_pages})
            pages += n_pages
            if pages_needed is None:
                break
        return {"evicted": evicted, "pages": pages, "remaining_entries": len(self._entries)}

    def drain(self) -> None:
        with self._lock:
            if self._requests:
                raise TangramRaggedPrefixError(
                    "cannot drain with live requests: "
                    + ", ".join(sorted(self._requests))
                )
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self.manager.release_ragged_prefix_block_ids(
                entry.snapshot.physical_block_ids
            )

    def telemetry(self) -> dict[str, object]:
        with self._lock:
            entries = tuple(self._entries.values())
            unique_pages = {
                page_id
                for entry in entries
                for page_id in entry.snapshot.physical_block_ids
            }
            entry_rows = []
            for entry in entries:
                page_ids = entry.snapshot.physical_block_ids
                entry_rows.append(
                    {
                        "lookup_digest": entry.identity.lookup.semantic_digest,
                        "semantic_digest": entry.identity.semantic_digest,
                        "logical_tokens": entry.identity.lookup.prefix_token_count,
                        "selection_fingerprint": (entry.identity.selection_fingerprint),
                        "raw_page_ids": list(page_ids),
                        "refcounts": list(
                            self.manager.ragged_prefix_refcounts(page_ids)
                        ),
                        "physical_base_bytes": entry.physical_base_bytes,
                        "metadata_bytes": entry.metadata_bytes,
                    }
                )
            latencies = tuple(self._attach_latencies_ns)
            return {
                "schema": "kvladder-tangram-ragged-prefix-cache-v1",
                **self._counters,
                "entries": len(entries),
                "live_requests": len(self._requests),
                "physical_shared_bytes": (
                    len(unique_pages)
                    * int(getattr(self.manager, "ragged_physical_block_bytes", 0))
                ),
                "attach_latency_ns": {
                    "count": len(latencies),
                    "min": min(latencies) if latencies else 0,
                    "max": max(latencies) if latencies else 0,
                    "mean": (sum(latencies) / len(latencies) if latencies else 0.0),
                },
                "entry_rows": entry_rows,
                "events": list(self._events),
            }


__all__ = [
    "TangramRaggedPrefixCache",
    "TangramRaggedPrefixEntry",
    "TangramRaggedPrefixError",
    "TangramRaggedPrefixIdentity",
    "TangramRaggedPrefixLookup",
    "TangramRaggedPrefixMiss",
    "TangramRaggedPrefixRefusal",
    "TangramRaggedPrefixRequestSpec",
    "TangramRaggedPrefixSnapshot",
    "digest_token_ids",
    "retained_positions_fingerprint",
    "trim_snapshot_to_length",
]
