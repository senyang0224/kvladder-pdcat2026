#!/usr/bin/env python3
"""Full-document QA, with a real single-chunk Shared base and private document tail.

This is NOT the 32K+1K PPL addendum. All natural questions and all original
context tokens are scored. The published prefix is one 32K chunk: the pinned
registry's sink/recent identity rule is not a multi-chunk lineage rule.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from scripts.audit import run_arm as base
from scripts.quality import run_quality_arm as original
DEFAULT_REGISTRY = Path(base.kvladder_registry_module.__file__)
from kvladder_tangram_ragged_prefix import TangramRaggedPrefixCache
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
import torch
import vllm


def generate(llm, request_id, tokens, params):
    llm.llm_engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens), params)
    for _ in range(100000):
        for out in llm.llm_engine.step():
            if out.request_id == request_id and out.finished:
                return out
    raise RuntimeError(f'request did not finish: {request_id}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('native', 'shared'), required=True)
    ap.add_argument('--task', default='scbench_qa_eng', choices=('scbench_qa_eng',))
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--phase', required=True)
    ap.add_argument('--kv-cache-memory-bytes', type=int, default=48*2**30)
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--chunk-size', type=int, default=32768)
    ap.add_argument('--max-num-seqs', type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--retained-ratio', type=float, default=0.296875)
    ap.add_argument('--limit-sessions', type=int, default=0)
    ap.add_argument('--limit-turns', type=int, default=0)
    ap.add_argument('--expect-gpu-uuid', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.chunk_size != 32768 or args.retained_ratio != 0.296875:
        raise ValueError('this addendum fixes one 32768-token published chunk at rho=0.296875')
    if (args.limit_sessions or args.limit_turns) and 'smoke' not in args.phase:
        raise ValueError('cohort limits are smoke-only')
    output = args.output
    partial = output.with_suffix('.requests.jsonl')
    if output.exists() or partial.exists():
        raise FileExistsError('refusing to overwrite receipt/checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    gpu = base.gpu_identity()
    if args.expect_gpu_uuid.removeprefix('GPU-') != str(gpu['uuid']).removeprefix('GPU-'):
        raise RuntimeError(f'wrong GPU: {gpu}')
    before = base.nvidia_smi()
    parquet = args.data_dir / f'{args.task}.parquet'
    sessions = original.load_sessions(parquet)
    available = len(sessions)
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Both arms use EXACTLY the Fig.4 native engine builder; only Shared installs
    # the pinned registry and supplies lookup metadata to requests.
    arm = args.arm
    args.arm = 'ragged-fp16-copy'
    started = time.perf_counter()
    llm, cfg = original.build(args)
    args.arm = arm
    build_s = time.perf_counter() - started
    cfg['arm'] = arm
    cfg.update(model=args.model, tokenizer=args.model)
    cfg.update(publication_boundary_rule='first-complete-32768-token-chunk',
               published_prefix_tokens=32768 if arm == 'shared' else None,
               full_context_preserved=True, sequential_requests=True,
               scoring='first-nonempty-line-token-F1-Q1', registry_module_sha256=base.file_sha256(DEFAULT_REGISTRY) if arm == 'shared' else None)
    core = base.engine_core(llm)
    registry = None
    if arm == 'shared':
        registry = TangramRaggedPrefixCache()
        core.scheduler.set_kvladder_ragged_prefix_cache(registry)
    rows, publications = [], []
    start = time.perf_counter()
    with partial.open('x') as progress:
        for sess in sessions:
            context = [int(v) for v in tok(sess['context'], add_special_tokens=False)['input_ids']]
            sid = sess['session_index']
            lookup = None
            if registry is not None:
                if len(context) <= args.chunk_size:
                    raise RuntimeError('expected full document longer than the published single chunk')
                prefix = context[:args.chunk_size]
                lookup = base.make_lookup({'model': args.model, 'token_prefix_digest': base.digest_token_ids(prefix),
                                           'context_tokens': args.chunk_size}, 16, cfg['eviction_ratio'])
                extra = {registry.EXTRA_ARG: {**asdict(lookup), 'commit_on_success': False}}
                warm_id = f'quality-origin-s{sid:02d}'
                generate(llm, warm_id, prefix, SamplingParams(temperature=0.0, max_tokens=1, extra_args=extra))
                entry = registry.entry(lookup)  # A missing publication is a failure, not a Native observation.
                telemetry = registry.telemetry()
                if telemetry['entries'] != 1 or telemetry['live_requests'] != 0:
                    raise RuntimeError('publication did not leave exactly one resident base and zero live requests')
                pages = list(entry.snapshot.physical_block_ids)
                if not pages:
                    raise RuntimeError('published base has no physical pages')
                publications.append({'session_index': sid, 'origin_request_id': warm_id, 'origin_finished': True,
                                     'document_tokens': len(context), 'published_prefix_tokens': len(prefix),
                                     'private_document_tokens': len(context)-len(prefix),
                                     'lookup_digest': lookup.semantic_digest, 'selection_fingerprint': entry.identity.selection_fingerprint,
                                     'effective_seq_lens': list(entry.snapshot.effective_seq_lens),
                                     'physical_page_ids': pages, 'telemetry_after_origin': telemetry})
                progress.write(json.dumps({'publication': publications[-1]})+'\n'); progress.flush()
            questions = sess['questions'][:args.limit_turns] if args.limit_turns else sess['questions']
            for ti, (question, gold) in enumerate(zip(questions, sess['answers'])):
                q_ids = [int(v) for v in tok(question, add_special_tokens=False)['input_ids']]
                prompt = context + q_ids
                if len(prompt)+args.max_new_tokens > args.max_model_len:
                    raise RuntimeError(f'no skipping permitted: session{sid}/turn{ti} exceeds max_model_len')
                rid = f's{sid:02d}t{ti}'
                extra = None
                prior = None
                if registry is not None:
                    prior = registry.telemetry()
                    extra = {registry.EXTRA_ARG: {**asdict(lookup), 'commit_on_success': False}}
                t0 = time.perf_counter()
                out = generate(llm, rid, prompt, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, extra_args=extra))
                cand = out.outputs[0]
                answer = original.first_line(cand.text)
                cached = int(getattr(out, 'num_cached_tokens', 0) or 0)
                row = {'request_id': rid, 'session_index': sid, 'turn_index': ti,
                       'prompt_tokens': len(prompt), 'prefix_tokens': len(context), 'question_tokens': len(q_ids),
                       'prompt_token_ids_sha256': base.digest_token_ids(prompt),
                       'num_cached_tokens': cached, 'generated': cand.text, 'answer': answer,
                       'generated_token_ids': [int(v) for v in cand.token_ids], 'finish_reason': str(cand.finish_reason),
                       'gold': gold, 'elapsed_s': time.perf_counter()-t0,
                       **base.quality(answer, [gold]), 'answer_f1_full_generation': base.quality(cand.text, [gold])['answer_f1']}
                if registry is not None:
                    after = registry.telemetry()
                    events = [e for e in after['events'][len(prior['events']):]
                              if e['event'] == 'lookup_hit_attach' and e['request_id'] == rid]
                    delta = int(after['hits']) - int(prior['hits'])
                    page_match = len(events) == 1 and events[0]['raw_page_ids'] == pages
                    attached = cached == args.chunk_size and delta == 1 and page_match
                    row['shared_attach'] = {'attached': attached, 'cached_tokens': cached, 'registry_hits_delta': delta,
                                            'published_prefix_tokens': args.chunk_size, 'same_resident_pages': page_match,
                                            'lookup_digest': lookup.semantic_digest, 'event': events[0] if len(events)==1 else None}
                    if not attached or after['live_requests'] != 0 or after['entries'] != 1:
                        progress.write(json.dumps({'failed_request': row, 'telemetry': after})+'\n'); progress.flush()
                        raise RuntimeError(f'{rid}: Shared path was not verified')
                rows.append(row)
                progress.write(json.dumps({'request': row})+'\n'); progress.flush()
                print(json.dumps({'arm': arm, 'request_id': rid, 'completed': len(rows), 'f1': row['answer_f1'],
                                  'prompt_tokens': len(prompt), 'cached_tokens': cached, 'elapsed_s': row['elapsed_s']}), flush=True)
            if registry is not None:
                publications[-1]['telemetry_before_drain'] = registry.telemetry()
                registry.drain()
                if registry.telemetry()['entries'] or registry.telemetry()['live_requests']:
                    raise RuntimeError('registry did not drain between documents')
    if not args.limit_sessions and not args.limit_turns and (len(sessions), len(rows)) != (20, 102):
        raise RuntimeError('full cohort is not exactly20sessions/102questions')
    runtime = Path(vllm.__file__).resolve().parents[1]
    receipt = {'schema': 'kvladder-conf-full-session-quality-v1', 'arm': arm, 'task': args.task, 'phase': args.phase,
               'retained_ratio': args.retained_ratio,
               'label': 'complete-document QA; Shared freezes first32K only, remaining document+question private; no equivalence claim',
               'dataset': {'path': str(parquet), 'sha256': base.file_sha256(parquet), 'sessions_available': available,
                           'sessions_used': len(sessions), 'limit_turns': args.limit_turns},
               'configuration': cfg, 'summary': {'requests': len(rows), 'sessions': len(sessions), 'skipped': [],
                    'metric': 'f1', 'value': sum(r['answer_f1'] for r in rows)/len(rows),
                    'exact_match_rate': sum(bool(r['answer_exact_match']) for r in rows)/len(rows),
                    'wall_s': time.perf_counter()-start, 'engine_build_s': build_s},
               'rows': rows, 'publications': publications,
               'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'vllm': vllm.__version__,
                    'vllm_path': vllm.__file__, 'tangram_upstream_commit': base.git_head(runtime),
                    'tangram_overlay_diff_sha256': base.diff_sha256(runtime), 'repo_commit': base.git_head(REPO),
                    'repo_dirty': base.git_dirty(REPO), 'runner_sha256': base.file_sha256(Path(__file__)),
                    'registry_module_sha256': base.file_sha256(DEFAULT_REGISTRY), 'gpu': gpu,
                    'nvidia_smi_before': before, 'nvidia_smi_after': base.nvidia_smi(), 'cli': sys.argv}}
    with output.open('x') as handle:
        json.dump(receipt, handle, indent=1); handle.write('\n')
    print(json.dumps(receipt['summary']), flush=True)


if __name__ == '__main__':
    main()
