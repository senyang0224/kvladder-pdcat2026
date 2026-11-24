"""A published base may hold a different retained length in each head-group cluster.

Until now the registry refused such a base outright, which is what excluded every head-level budget policy
from the sharing contract. These tests pin the two halves of lifting that refusal: a ragged base attaches and
carries its per-cluster lengths to the worker, and a uniform base carries nothing new, which is what keeps
every already-measured path on exactly the code it was measured on.
"""
from __future__ import annotations


import pytest


from kvladder_tangram_ragged_prefix import (  # noqa: E402
    TangramRaggedPrefixCache,
    TangramRaggedPrefixRefusal,
    TangramRaggedPrefixSnapshot,
)
from test_kvladder_tangram_ragged_prefix import (  # noqa: E402
    _FakeManager,
    _lookup,
    _request,
    _Status,
)

RAGGED = {
    "block_table": [[1, 2], [3, 4]],
    "block_counts": [2, 1],          # block_size is 4 in the fake manager
    "effective_seq_lens": [8, 4],
}
UNIFORM = {
    "block_table": [[1, 2], [3, 4]],
    "block_counts": [2, 2],
    "effective_seq_lens": [8, 8],
}


def _publish(cache, manager, lookup, snapshot, *, request_id="origin", page_ids=(1, 2, 3, 4)):
    origin = _request(request_id, range(8), lookup)
    assert cache.on_request_arrival(origin) is False
    manager.own(request_id, page_ids)
    origin.num_computed_tokens = 8
    origin.compression_done = True
    cache.on_model_output(origin, snapshot)
    return origin, cache.entry(lookup)


def _attached(snapshot):
    tokens = tuple(range(8))
    lookup = _lookup(tokens)
    manager = _FakeManager()
    cache = TangramRaggedPrefixCache()
    cache.bind_kv_cache_manager(manager)
    origin, entry = _publish(cache, manager, lookup, snapshot)
    manager.free_request(origin.request_id)
    origin.status = _Status("FINISHED_LENGTH_CAPPED")
    cache.on_request_finished(origin, reason="warmup finish")
    branch = _request("branch", tokens + (11, 12), lookup)
    assert cache.on_request_arrival(branch) is True
    return cache, branch, entry


def test_a_ragged_base_attaches_instead_of_being_refused():
    _, branch, entry = _attached(RAGGED)
    assert entry.snapshot.effective_seq_lens == (8, 4)
    assert branch.kvladder_shared_prefix_ready is True
    assert branch.kvladder_immutable_prefix_lengths == (8, 4)


def test_the_scalar_stays_the_maximum():
    """The ratio arithmetic subtracts it from the prompt to size the private continuation, and that
    difference is the same for every cluster, so taking the max leaves the requested budget unchanged."""
    _, branch, _ = _attached(RAGGED)
    assert branch.kvladder_immutable_prefix_tokens == 8
    assert branch.kvladder_ragged_prefix_metadata["immutable_prefix_tokens"] == 8
    assert branch.kvladder_ragged_prefix_metadata["immutable_prefix_lengths"] == (8, 4)


def test_a_uniform_base_carries_no_per_cluster_lengths():
    """The no-op property every already-measured receipt depends on: nothing downstream reads a per-cluster
    value unless a ragged base put one there."""
    _, branch, _ = _attached(UNIFORM)
    assert branch.kvladder_immutable_prefix_tokens == 8
    assert branch.kvladder_immutable_prefix_lengths is None
    assert branch.kvladder_ragged_prefix_metadata["immutable_prefix_lengths"] is None


def test_a_misaligned_ragged_length_is_still_refused_at_publication():
    """Block alignment is validated once, where the snapshot is published; the attach does not re-check it
    against a second notion of the block size."""
    snapshot = TangramRaggedPrefixSnapshot.from_mapping(
        {"block_table": [[1, 2], [3, 4]], "block_counts": [2, 1], "effective_seq_lens": [8, 5]})
    with pytest.raises(TangramRaggedPrefixRefusal):
        snapshot.validate_immutable_boundary(block_size=4)


def test_a_row_count_inconsistent_with_its_length_is_an_error_not_a_refusal():
    snapshot = TangramRaggedPrefixSnapshot.from_mapping(
        {"block_table": [[1, 2], [3, 4]], "block_counts": [2, 2], "effective_seq_lens": [8, 4]})
    with pytest.raises(Exception) as excinfo:
        snapshot.validate_immutable_boundary(block_size=4)
    assert not isinstance(excinfo.value, TangramRaggedPrefixRefusal)
