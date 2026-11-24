from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kvladder_tangram_ragged_prefix import (
    TangramRaggedPrefixCache,
    TangramRaggedPrefixError,
    TangramRaggedPrefixLookup,
    TangramRaggedPrefixMiss,
    TangramRaggedPrefixRefusal,
    TangramRaggedPrefixSnapshot,
    digest_token_ids,
)


class _FakeManager:
    ragged = True
    ragged_block_size = 4
    ragged_physical_block_bytes = 1024

    def __init__(self, page_ids=range(1, 65)) -> None:
        self.refs = {int(page_id): 0 for page_id in page_ids}
        self.requests: dict[str, tuple[int, ...]] = {}

    def own(self, request_id: str, page_ids) -> None:
        ids = tuple(map(int, page_ids))
        assert request_id not in self.requests
        self.requests[request_id] = ids
        self.retain_ragged_prefix_block_ids(ids)

    def retain_ragged_prefix_block_ids(self, page_ids) -> None:
        for page_id in page_ids:
            self.refs[int(page_id)] += 1

    def release_ragged_prefix_block_ids(self, page_ids) -> None:
        for page_id in page_ids:
            self.refs[int(page_id)] -= 1
            assert self.refs[int(page_id)] >= 0

    def attach_ragged_prefix_block_ids(self, request_id: str, page_ids) -> None:
        self.own(request_id, page_ids)

    def detach_ragged_prefix_block_ids(self, request_id: str) -> None:
        self.free_request(request_id)

    def free_request(self, request_id: str) -> None:
        self.release_ragged_prefix_block_ids(self.requests.pop(request_id, ()))

    def ragged_prefix_refcounts(self, page_ids):
        return tuple(self.refs[int(page_id)] for page_id in page_ids)


@dataclass
class _Status:
    name: str


def _lookup(tokens, *, policy="ratio-r030") -> TangramRaggedPrefixLookup:
    return TangramRaggedPrefixLookup(
        token_prefix_digest=digest_token_ids(tokens),
        prefix_token_count=len(tokens),
        model_id="test-model",
        model_config_digest="model-config",
        tokenizer_id="test-tokenizer",
        position_contract="absolute-rope-v1",
        kv_dtype="float16",
        compressor_id="tangram-streamingllm",
        compressor_revision="44ed97e8",
        compression_policy_digest=policy,
        layout_contract="tangram-ragged-page-group-v1",
        retained_position_rule="streamingllm-sink-recent-v1",
        sink_tokens=0,
    )


def _spec(lookup, *, commit=False):
    value = (
        {key: value for key, value in lookup.__dict__.items()}
        if hasattr(lookup, "__dict__")
        else {field: getattr(lookup, field) for field in lookup.__dataclass_fields__}
    )
    value["commit_on_success"] = commit
    return value


def _request(request_id, prompt, lookup, *, commit=False, status="RUNNING"):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=SimpleNamespace(
            extra_args={
                TangramRaggedPrefixCache.EXTRA_ARG: _spec(lookup, commit=commit)
            }
        ),
        num_computed_tokens=0,
        num_cached_tokens=-1,
        compress_max_eff_seq_len=None,
        compression_done=False,
        all_token_ids=list(prompt),
        status=_Status(status),
    )


def _snapshot(page_ids=(1, 2, 3, 4)):
    return {
        "block_table": [[page_ids[0], page_ids[1]], [page_ids[2], page_ids[3]]],
        "block_counts": [2, 2],
        "effective_seq_lens": [8, 8],
    }


def _publish(cache, manager, lookup, *, request_id="origin", page_ids=(1, 2, 3, 4)):
    origin = _request(request_id, range(8), lookup)
    assert cache.on_request_arrival(origin) is False
    manager.own(request_id, page_ids)
    origin.num_computed_tokens = 8
    origin.compression_done = True
    cache.on_model_output(origin, _snapshot(page_ids))
    return origin, cache.entry(lookup)


def test_semantic_identity_excludes_physical_relocation() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    first = TangramRaggedPrefixSnapshot.from_mapping(_snapshot((1, 2, 3, 4)))
    second = TangramRaggedPrefixSnapshot.from_mapping(_snapshot((9, 10, 11, 12)))
    assert first.physical_block_ids != second.physical_block_ids
    assert lookup.semantic_digest == _lookup(tokens).semantic_digest


def test_scheduler_lookup_aliases_raw_pages_without_copy_or_reconstruction() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, entry = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    branch = _request("branch", tokens + (99,), lookup)
    assert cache.on_request_arrival(branch) is True
    assert manager.requests["branch"] == entry.snapshot.physical_block_ids
    assert branch.num_cached_tokens == 8
    assert branch.num_computed_tokens == 8
    assert branch.compression_done is False
    assert branch.kvladder_shared_prefix_ready is True
    assert branch.kvladder_private_compression_state == "pending"
    assert branch.kvladder_immutable_prefix_tokens == 8
    assert branch.kvladder_logical_shared_prefix_tokens == 8
    assert branch.kvladder_ragged_prefix_metadata["shared_prefix_ready"] is True
    assert (
        branch.kvladder_ragged_prefix_metadata["private_compression_state"]
        == "pending"
    )
    assert manager.ragged_prefix_refcounts(entry.snapshot.physical_block_ids) == (
        2,
        2,
        2,
        2,
    )
    telemetry = cache.telemetry()
    assert telemetry["base_copy_bytes"] == 0
    assert telemetry["payload_copy_bytes"] == 0
    assert telemetry["dense_materialization_bytes"] == 0
    assert telemetry["immutable_rewrite_bytes"] == 0

    branch.status = _Status("FINISHED_ABORTED")
    manager.free_request("branch")
    cache.on_request_finished(branch, reason="cancel")
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_commit_pins_shared_parent_and_private_tail_while_abort_reclaims() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    committed = _request("commit", tokens + (90, 91, 92, 93), lookup, commit=True)
    assert cache.on_request_arrival(committed)
    manager.requests["commit"] += (5, 6)
    manager.retain_ragged_prefix_block_ids((5, 6))
    committed.num_computed_tokens = 12
    committed.all_token_ids = list(tokens + (90, 91, 92, 93))
    committed.status = _Status("FINISHED_LENGTH_CAPPED")
    child_snapshot = {
        "block_table": [[1, 2, 5], [3, 4, 6]],
        "block_counts": [3, 3],
        "effective_seq_lens": [12, 12],
    }
    cache.on_model_output(committed, child_snapshot)
    child = cache.prepare_request_finish(committed)
    assert child is not None
    assert child.snapshot.physical_block_ids[:2] == (1, 2)
    assert set(parent.snapshot.physical_block_ids).issubset(
        child.snapshot.physical_block_ids
    )
    manager.free_request("commit")
    cache.on_request_finished(committed, reason="finish")

    aborted = _request("abort", tokens + (77,), lookup, commit=True)
    assert cache.on_request_arrival(aborted)
    manager.requests["abort"] += (7, 8)
    manager.retain_ragged_prefix_block_ids((7, 8))
    aborted.status = _Status("FINISHED_ABORTED")
    assert cache.prepare_request_finish(aborted) is None
    manager.free_request("abort")
    cache.on_request_finished(aborted, reason="cancel")
    assert manager.refs[7] == manager.refs[8] == 0

    telemetry = cache.telemetry()
    assert telemetry["commits"] == 1
    assert telemetry["aborts"] == 1
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_stable_descriptor_replays_scheduler_delta_without_worker_snapshot() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    branch = _request("delta", tokens + (90, 91, 92, 93), lookup, commit=True)
    assert cache.on_request_arrival(branch)
    # 2026-09-05 (conference Phase D): a branch that may commit asks the worker
    # for its authoritative table (private compaction frees pages the O(1)
    # descriptor never sees; receipt smoke_lifecycle_shared_ctx32768_v4.json).
    # Ordinary hits keep the snapshot-free stable descriptor; the descriptor
    # remains the fallback below when no worker snapshot arrived.
    assert branch.kvladder_ragged_prefix_metadata["capture"] is True
    assert branch.kvladder_ragged_prefix_metadata["capture_mode"] == (
        "commit_capture"
    )
    plain = _request("plain-hit", tokens + (1,), lookup, commit=False)
    assert cache.on_request_arrival(plain)
    assert plain.kvladder_ragged_prefix_metadata["capture"] is False
    assert plain.kvladder_ragged_prefix_metadata["capture_mode"] == (
        "stable_descriptor"
    )
    manager.free_request("plain-hit")
    plain.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(plain, reason="finish")
    descriptor_id = branch.kvladder_ragged_prefix_metadata["descriptor_id"]
    assert isinstance(descriptor_id, str) and descriptor_id

    manager.requests["delta"] += (5, 6)
    manager.retain_ragged_prefix_block_ids((5, 6))
    cache.on_schedule_output(
        SimpleNamespace(
            num_scheduled_tokens={"delta": 4},
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="delta",
                    block_ids=([1, 2, 3, 4, 5, 6],),
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], resumed_req_ids=set()
            ),
        )
    )
    branch.num_computed_tokens = 12
    branch.all_token_ids = list(tokens + (90, 91, 92, 93))
    branch.status = _Status("FINISHED_LENGTH_CAPPED")

    child = cache.prepare_request_finish(branch)
    assert child is not None
    assert child.snapshot.block_counts == (3, 3)
    assert child.snapshot.effective_seq_lens == (12, 12)
    assert child.snapshot.physical_block_ids == (1, 2, 5, 3, 4, 6)
    assert set(parent.snapshot.physical_block_ids).issubset(
        child.snapshot.physical_block_ids
    )
    telemetry = cache.telemetry()
    assert telemetry["descriptor_schedule_updates"] == 1
    assert telemetry["descriptor_block_delta_updates"] == 1
    assert telemetry["descriptor_delta_block_ids"] == 2
    assert telemetry["descriptor_full_snapshot_materializations"] == 1
    assert telemetry["native_hit_worker_snapshots"] == 0

    manager.free_request("delta")
    cache.on_request_finished(branch, reason="finish")
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_representation_or_token_mismatch_cannot_hit() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, _ = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    wrong_policy = _lookup(tokens, policy="ratio-r050")
    miss = _request("miss", tokens + (9,), wrong_policy)
    assert cache.on_request_arrival(miss) is False

    wrong_tokens = _request("wrong", tuple(range(7)) + (99, 9), lookup)
    with pytest.raises(TangramRaggedPrefixError, match="token prefix"):
        cache.on_request_arrival(wrong_tokens)

    cache.on_request_finished(miss, reason="cancel")
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_partial_shared_boundary_is_rejected_fail_closed() -> None:
    snapshot = TangramRaggedPrefixSnapshot.from_mapping(
        {
            "block_table": [[1, 2]],
            "block_counts": [2],
            "effective_seq_lens": [7],
        }
    )
    with pytest.raises(TangramRaggedPrefixError, match="block aligned"):
        snapshot.validate_immutable_boundary(block_size=4)


def _scheduler_add_with_private_fallback(cache, manager, waiting, request) -> None:
    try:
        try:
            cache.on_request_arrival(request)
        except BaseException as exc:
            if not cache.prepare_private_fallback(
                request, exc, reason="arrival_refused"
            ):
                raise
        waiting.add_request(request)
    except BaseException:
        manager.free_request(request.request_id)
        cache.on_request_finished(request, reason="arrival_failed")
        raise


class _FakeWaiting(list):
    def add_request(self, request) -> None:
        self.append(request)


class _FailingWaiting(list):
    def add_request(self, request) -> None:
        raise RuntimeError("queue insert failed")


def test_refused_token_mismatch_is_admitted_as_private_not_failed() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    wrong = _request("wrong", tuple(range(7)) + (99, 9), lookup)
    waiting = _FakeWaiting()
    _scheduler_add_with_private_fallback(cache, manager, waiting, wrong)

    assert waiting == [wrong]
    assert "wrong" not in manager.requests
    assert wrong.num_computed_tokens == 0
    assert wrong.num_cached_tokens == -1
    assert wrong.compress_max_eff_seq_len is None
    assert wrong.compression_done is False
    assert wrong.kvladder_shared_prefix_ready is False
    assert wrong.kvladder_private_compression_state == "not_applicable"
    assert wrong.kvladder_ragged_prefix_metadata is None
    assert manager.ragged_prefix_refcounts(parent.snapshot.physical_block_ids) == (
        1,
        1,
        1,
        1,
    )
    tel = cache.telemetry()
    assert tel["private_fallbacks"] == 1
    assert any(e["event"] == "private_fallback" for e in tel["events"])
    wrong.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(wrong, reason="private_finish")
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_partial_attach_refusal_rolls_back_before_private_admission() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    # A refusal raised after the pages are retained must roll back the temporary raw-page owner before the
    # scheduler falls back to a normal private admission.  The trigger used to be a non-uniform published
    # base, which the worker now protects per cluster and the registry therefore attaches (see
    # tests/projects/kvladder_conf/test_ragged_immutable_prefix.py).  The guarantee under test is the
    # rollback, not that particular refusal, so the manager raises one at the same point instead.
    class _RefusingManager(type(manager)):
        def attach_ragged_prefix_block_ids(self, request_id, page_ids):
            super().attach_ragged_prefix_block_ids(request_id, page_ids)
            raise TangramRaggedPrefixRefusal("simulated post-retain attach refusal")

    manager.__class__ = _RefusingManager
    snapshot = TangramRaggedPrefixSnapshot.from_mapping(
        {
            "block_table": [[1, 2, 0], [3, 4, 5]],
            "block_counts": [2, 3],
            "effective_seq_lens": [8, 12],
        }
    )
    cache._publish(
        lookup=lookup,
        snapshot=snapshot,
        retained_positions=(tuple(range(8)), tuple(range(12))),
        event="test_publish_nonuniform",
        request_id="origin",
    )
    assert manager.ragged_prefix_refcounts(snapshot.physical_block_ids) == (
        1,
        1,
        1,
        1,
        1,
    )

    branch = _request("branch", tokens + (99,), lookup)
    waiting = _FakeWaiting()
    _scheduler_add_with_private_fallback(cache, manager, waiting, branch)

    assert waiting == [branch]
    assert "branch" not in manager.requests
    assert manager.ragged_prefix_refcounts(snapshot.physical_block_ids) == (
        1,
        1,
        1,
        1,
        1,
    )
    tel = cache.telemetry()
    fallback = [e for e in tel["events"] if e["event"] == "private_fallback"][-1]
    assert fallback["had_live_record"] is False
    assert branch.kvladder_ragged_prefix_metadata is None
    branch.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(branch, reason="private_finish")
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_unknown_attach_exception_is_not_downgraded_to_private_fallback() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, _ = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    def boom(_request_id, _page_ids):
        raise RuntimeError("allocator exploded")

    manager.attach_ragged_prefix_block_ids = boom
    branch = _request("branch", tokens + (99,), lookup)
    with pytest.raises(RuntimeError, match="allocator exploded"):
        _scheduler_add_with_private_fallback(cache, manager, _FakeWaiting(), branch)
    assert cache.telemetry()["private_fallbacks"] == 0


def test_commit_publication_refusal_does_not_abort_finished_request() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    branch = _request("no-tail", tokens + (99,), lookup, commit=True)
    assert cache.on_request_arrival(branch)
    branch.num_computed_tokens = parent.identity.lookup.prefix_token_count
    branch.all_token_ids = list(tokens)
    branch.status = _Status("FINISHED_LENGTH_CAPPED")
    try:
        cache.prepare_request_finish(branch)
    except BaseException as exc:
        handled = cache.prepare_finish_refusal(
            branch, exc, reason="finish:FINISHED_LENGTH_CAPPED"
        )
    else:  # pragma: no cover - this should refuse publication
        handled = False
    assert handled is True
    manager.free_request("no-tail")
    cache.on_request_finished(branch, reason="finish")
    tel = cache.telemetry()
    assert tel["publication_refusals"] == 1
    assert any(e["event"] == "publication_refused" for e in tel["events"])
    cache.drain()
    assert all(value == 0 for value in manager.refs.values())


def test_finish_unknown_exception_still_propagates() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    req = _request("plain", tokens + (99,), lookup, commit=True)
    assert cache.prepare_finish_refusal(req, RuntimeError("bug")) is False



def test_malformed_extra_arg_falls_back_to_private_admission() -> None:
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    request = SimpleNamespace(
        request_id="malformed",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(
            extra_args={TangramRaggedPrefixCache.EXTRA_ARG: "not-a-mapping"}
        ),
        num_computed_tokens=123,
        num_cached_tokens=123,
        compress_max_eff_seq_len=123,
        compression_done=True,
        all_token_ids=[1, 2, 3],
        status=_Status("RUNNING"),
    )
    waiting = _FakeWaiting()
    _scheduler_add_with_private_fallback(cache, manager, waiting, request)

    assert waiting == [request]
    assert request.num_computed_tokens == 0
    assert request.num_cached_tokens == -1
    assert request.kvladder_ragged_prefix_metadata is None
    assert cache.telemetry()["private_fallbacks"] == 1


def test_unknown_partial_arrival_failure_cleans_attached_owner_and_live_state() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    original_attach = manager.attach_ragged_prefix_block_ids

    def attach_then_boom(request_id, page_ids):
        original_attach(request_id, page_ids)
        raise RuntimeError("post-attach scheduler invariant failed")

    manager.attach_ragged_prefix_block_ids = attach_then_boom
    branch = _request("branch", tokens + (99,), lookup)
    with pytest.raises(RuntimeError, match="post-attach"):
        _scheduler_add_with_private_fallback(cache, manager, _FakeWaiting(), branch)

    assert "branch" not in manager.requests
    assert cache.telemetry()["live_requests"] == 0
    assert cache.telemetry()["private_fallbacks"] == 0
    assert manager.ragged_prefix_refcounts(parent.snapshot.physical_block_ids) == (
        1,
        1,
        1,
        1,
    )


def test_queue_add_failure_after_attach_uses_old_cleanup_and_raises() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, parent = _publish(cache, manager, lookup)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    branch = _request("branch", tokens + (99,), lookup)
    with pytest.raises(RuntimeError, match="queue insert failed"):
        _scheduler_add_with_private_fallback(cache, manager, _FailingWaiting(), branch)

    assert "branch" not in manager.requests
    assert cache.telemetry()["live_requests"] == 0
    assert manager.ragged_prefix_refcounts(parent.snapshot.physical_block_ids) == (
        1,
        1,
        1,
        1,
    )


def test_origin_unsupported_publication_rule_is_refused_but_request_can_finish() -> None:
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    lookup = TangramRaggedPrefixLookup(
        token_prefix_digest=lookup.token_prefix_digest,
        prefix_token_count=lookup.prefix_token_count,
        model_id=lookup.model_id,
        model_config_digest=lookup.model_config_digest,
        tokenizer_id=lookup.tokenizer_id,
        position_contract=lookup.position_contract,
        kv_dtype=lookup.kv_dtype,
        compressor_id=lookup.compressor_id,
        compressor_revision=lookup.compressor_revision,
        compression_policy_digest=lookup.compression_policy_digest,
        layout_contract=lookup.layout_contract,
        retained_position_rule="unsupported-selector-v1",
        sink_tokens=lookup.sink_tokens,
    )
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin = _request("origin", tokens, lookup)
    assert cache.on_request_arrival(origin) is False
    manager.own("origin", (1, 2, 3, 4))
    origin.num_computed_tokens = 8
    origin.compression_done = True
    cache.on_model_output(origin, _snapshot((1, 2, 3, 4)))

    with pytest.raises(TangramRaggedPrefixMiss):
        cache.entry(lookup)
    assert cache.telemetry()["publication_refusals"] == 1
    assert origin.kvladder_ragged_prefix_metadata is None
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    manager.free_request("origin")
    cache.on_request_finished(origin, reason="finish")
    assert cache.telemetry()["live_requests"] == 0
    assert all(value == 0 for value in manager.refs.values())


def _ragged_snapshot(page_ids=(1, 2, 3)):
    """Two clusters at different retained lengths: 8 tokens over two pages, 4 over one.

    ``_snapshot`` is uniform by construction, which is why every descriptor test before
    2026-09-16 exercised the arithmetic that a uniform base makes trivially satisfiable.
    """
    return {
        "block_table": [[page_ids[0], page_ids[1]], [page_ids[2], 0]],
        "block_counts": [2, 1],
        "effective_seq_lens": [8, 4],
    }


def _publish_ragged(cache, manager, lookup, *, request_id="ragged-origin", page_ids=(1, 2, 3)):
    origin = _request(request_id, range(8), lookup)
    assert cache.on_request_arrival(origin) is False
    manager.own(request_id, page_ids)
    origin.num_computed_tokens = 8
    origin.compression_done = True
    cache.on_model_output(origin, _ragged_snapshot(page_ids))
    return origin, cache.entry(lookup)


def test_ragged_base_branch_takes_the_authoritative_snapshot_path() -> None:
    """A length-ragged base's branch must not ride the O(1) stable descriptor.

    Measured 2026-09-16 on a 64-cluster, 38,922-page base: the descriptor mirrors the pages
    the scheduler allocates and never the ones the branch's own compaction frees, and on a
    ragged base that drift is not a multiple of the head-group count (38,922 mod 64 = 10),
    so the first post-compaction delta refuses with "cannot form a uniform ragged target".
    A uniform base hides the same drift, which is why this went unseen until a ragged base
    was grown past one token.
    """
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, entry = _publish_ragged(cache, manager, lookup)
    assert len(set(entry.snapshot.effective_seq_lens)) == 2, "fixture must publish a ragged base"
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    # An ordinary hit on a ragged base: capture is on, and the mode says why.
    plain = _request("ragged-hit", tokens + (11, 12), lookup, commit=False)
    assert cache.on_request_arrival(plain)
    meta = plain.kvladder_ragged_prefix_metadata
    assert meta["capture"] is True
    assert meta["capture_mode"] == "ragged_capture"
    assert meta["immutable_prefix_lengths"] == (8, 4)
    # ... and it still publishes nothing: publication is gated on commit_on_success alone.
    manager.free_request("ragged-hit")
    plain.status = _Status("FINISHED_STOPPED")
    assert cache.prepare_request_finish(plain) is None
    cache.on_request_finished(plain, reason="done")

    # A committing branch is unchanged.
    branch = _request("ragged-commit", tokens + (13, 14), lookup, commit=True)
    assert cache.on_request_arrival(branch)
    assert branch.kvladder_ragged_prefix_metadata["capture"] is True
    assert branch.kvladder_ragged_prefix_metadata["capture_mode"] == "commit_capture"
    manager.free_request("ragged-commit")
    branch.status = _Status("FINISHED_ABORTED")
    cache.on_request_finished(branch, reason="cleanup")
    cache.drain()


def test_uniform_base_branch_keeps_the_stable_descriptor() -> None:
    """The negative control for the change above: nothing moves for a uniform base."""
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, entry = _publish(cache, manager, lookup)
    assert len(set(entry.snapshot.effective_seq_lens)) == 1
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")

    plain = _request("uniform-hit", tokens + (11,), lookup, commit=False)
    assert cache.on_request_arrival(plain)
    assert plain.kvladder_ragged_prefix_metadata["capture"] is False
    assert plain.kvladder_ragged_prefix_metadata["capture_mode"] == "stable_descriptor"
    assert plain.kvladder_ragged_prefix_metadata["immutable_prefix_lengths"] is None
    manager.free_request("uniform-hit")
    plain.status = _Status("FINISHED_STOPPED")
    cache.on_request_finished(plain, reason="done")
    cache.drain()
