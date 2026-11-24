#!/usr/bin/env python3
"""Quality characterization of three KV representations on SCBench (supplement lock Q1).

One engine per (arm, task); every session's shared context is the prefix and every natural turn is one
request (prefix + question), issued sequentially with greedy decoding.  Arms: full-fp16-apc (dense FP16
prefix cache), fp8-apc (uniform FP8 KV, prefix cache), ragged-fp16-copy (native Tangram per-head eviction,
retained ratio 0.297 per 32K chunk, no registry -- KVLadder does not participate).  Scoring, on the first non-empty line of the greedy continuation: token F1 for
scbench_qa_eng (same normalisation as the campaign harness), option accuracy for scbench_choice_eng.
No pass/fail threshold: this characterises the representation the paper shares; it gates no claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.recency.run_arm import (  # noqa: E402  (imports vllm; run inside the campaign environment)
    COMPRESSION_SCORER, RETAINED_RATIO, SINK_TOKENS, diff_sha256, file_sha256, git_dirty, git_head, gpu_identity, normalize, nvidia_smi, quality, tangram_ratio,
)
import torch  # noqa: E402
import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

SCHEMA = "kvladder-conf-quality-receipt-v1"
ARMS = ("full-fp16-apc", "fp8-apc", "ragged-fp16-copy")
TASKS = {"scbench_qa_eng": "f1", "scbench_choice_eng": "accuracy"}
OPTION_RE = re.compile(r"^([A-D])\.\s*(.+?)\s*$", re.M)


def parse_options(question: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in OPTION_RE.finditer(question)}


def first_line(text: str) -> str:
    """The answer is the first non-empty line of the greedy continuation (the prompt ends with 'Answer:' /
    'The the correct answer is'); later lines are free-running text and are not scored."""
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return str(text).strip()


def choice_correct(generated: str, gold: str, options: dict[str, str]) -> dict[str, object]:
    """Correct if the first standalone option letter in the answer names the gold option, or the gold text appears."""
    gold_n = normalize(gold)
    gen_n = normalize(generated)
    gold_letter = next((k for k, v in options.items() if normalize(v) == gold_n), None)
    m = re.search(r"\b([A-D])\b", generated)
    pred_letter = m.group(1) if m else None
    by_letter = pred_letter is not None and gold_letter is not None and pred_letter == gold_letter
    by_text = bool(gold_n) and gold_n in gen_n
    return {"gold_letter": gold_letter, "pred_letter": pred_letter, "correct": bool(by_letter or by_text), "by_letter": by_letter, "by_text": by_text}


def load_sessions(parquet: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(parquet)
    sessions = []
    for i, row in df.iterrows():
        prompts = [str(x) for x in list(row["prompts"])]
        gts = [str(x) for x in list(row["ground_truth"])]
        if len(prompts) != len(gts) + 1:
            raise RuntimeError(f"session {i}: {len(prompts)} prompts vs {len(gts)} answers")
        sessions.append({"session_index": int(i), "context": prompts[0], "questions": prompts[1:], "answers": gts})
    return sessions


def build(args: argparse.Namespace) -> tuple[LLM, dict]:
    chunk = int(args.chunk_size)
    common = dict(model=str(args.model), tokenizer=str(args.model), dtype="float16", kv_cache_memory_bytes=int(args.kv_cache_memory_bytes),
                  max_model_len=int(args.max_model_len), enforce_eager=True, max_num_seqs=int(args.max_num_seqs), max_num_batched_tokens=chunk, seed=int(args.seed))
    cfg: dict = {"arm": args.arm, "task": args.task}
    if args.arm == "ragged-fp16-copy":
        target, keep, evict = tangram_ratio(chunk)
        llm = LLM(**common, kv_cache_dtype="auto", enable_prefix_caching=True, page_group_size=4, compression_ratio=evict, compression_scorer=COMPRESSION_SCORER,
                  compression_budget_scope="layer", compression_chunk_size=chunk, compression_floor_min=64, compression_window_size=32, compression_n_sink_tokens=SINK_TOKENS,
                  scheduler_reserve_full_isl=True)
        cfg.update(retained_ratio=RETAINED_RATIO, target_retained_tokens_per_chunk=target, body_keep_ratio=keep, eviction_ratio=evict, compression_chunk_size=chunk,
                   compression_scorer=COMPRESSION_SCORER, compression_sink_tokens=SINK_TOKENS, compression_window_size=32, compression_floor_min=64, page_group_size=4)
    else:
        llm = LLM(**common, kv_cache_dtype="fp8" if args.arm == "fp8-apc" else "auto", enable_prefix_caching=True, page_group_size=None)
        cfg.update(page_group_size=None)
    cfg["resolved_enable_prefix_caching"] = bool(llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)
    cfg["kv_cache_dtype"] = str(llm.llm_engine.vllm_config.cache_config.cache_dtype)
    cfg.update(kv_cache_memory_bytes=int(args.kv_cache_memory_bytes), max_model_len=int(args.max_model_len), max_num_batched_tokens=chunk, max_num_seqs=int(args.max_num_seqs), seed=int(args.seed),
               max_new_tokens=int(args.max_new_tokens), temperature=0.0, add_special_tokens=False)
    return llm, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--task", choices=tuple(TASKS), required=True)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "SCBench")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", default="smoke")
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=48 * 2 ** 30)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--chunk-size", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit-sessions", type=int, default=0, help="smoke only; 0 = all sessions (the lock requires all)")
    ap.add_argument("--limit-turns", type=int, default=0, help="smoke only; 0 = all turns")
    ap.add_argument("--expect-gpu-uuid", default="")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    gpu = gpu_identity()
    if args.expect_gpu_uuid and args.expect_gpu_uuid not in str(gpu.get("uuid", "")):
        raise RuntimeError(f"unexpected GPU {gpu}")
    smi_before = nvidia_smi()
    parquet = args.data_dir / f"{args.task}.parquet"
    sessions = load_sessions(parquet)
    if args.limit_sessions:
        sessions = sessions[: int(args.limit_sessions)]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    t_build = time.perf_counter()
    llm, cfg = build(args)
    build_s = time.perf_counter() - t_build
    sp = SamplingParams(temperature=0.0, max_tokens=int(args.max_new_tokens))
    rows = []
    skipped = []
    t0 = time.perf_counter()
    for sess in sessions:
        prefix_ids = tok(sess["context"], add_special_tokens=False)["input_ids"]
        questions = sess["questions"][: int(args.limit_turns)] if args.limit_turns else sess["questions"]
        for ti, (q, gold) in enumerate(zip(questions, sess["answers"])):
            q_ids = tok(q, add_special_tokens=False)["input_ids"]
            prompt = [int(v) for v in prefix_ids] + [int(v) for v in q_ids]
            if len(prompt) + int(args.max_new_tokens) > int(args.max_model_len):
                skipped.append({"session_index": sess["session_index"], "turn_index": ti, "prompt_tokens": len(prompt), "reason": "exceeds max_model_len"})
                continue
            rid = f"s{sess['session_index']:02d}t{ti}"
            t1 = time.perf_counter()
            out = llm.generate([TokensPrompt(prompt_token_ids=prompt)], sp, use_tqdm=False)[0]
            elapsed = time.perf_counter() - t1
            cand = out.outputs[0]
            text = cand.text
            answer = first_line(text)
            row = {"request_id": rid, "session_index": sess["session_index"], "turn_index": ti, "prompt_tokens": len(prompt), "prefix_tokens": len(prefix_ids), "question_tokens": len(q_ids),
                   "num_cached_tokens": int(getattr(out, "num_cached_tokens", 0) or 0), "generated": text, "answer": answer, "generated_token_ids": [int(v) for v in cand.token_ids], "finish_reason": str(cand.finish_reason),
                   "gold": gold, "elapsed_s": elapsed}
            if TASKS[args.task] == "f1":
                row.update(quality(answer, [gold]))
                row["answer_f1_full_generation"] = quality(text, [gold])["answer_f1"]
            else:
                row.update(choice_correct(answer, gold, parse_options(q)))
            rows.append(row)
    wall = time.perf_counter() - t0
    if TASKS[args.task] == "f1":
        metric = {"metric": "f1", "value": (sum(float(r["answer_f1"]) for r in rows) / len(rows)) if rows else None, "exact_match_rate": (sum(bool(r["answer_exact_match"]) for r in rows) / len(rows)) if rows else None}
    else:
        metric = {"metric": "accuracy", "value": (sum(bool(r["correct"]) for r in rows) / len(rows)) if rows else None, "by_letter_rate": (sum(bool(r["by_letter"]) for r in rows) / len(rows)) if rows else None}
    smi_after = nvidia_smi()
    tangram_root = Path(vllm.__file__).resolve().parents[1]
    receipt = {
        "schema": SCHEMA, "arm": args.arm, "task": args.task, "phase": str(args.phase), "label": "descriptive characterization of the representation; gates no claim",
        "dataset": {"path": str(parquet.relative_to(REPO)) if parquet.is_relative_to(REPO) else str(parquet), "sha256": file_sha256(parquet), "sessions_available": len(load_sessions(parquet)), "sessions_used": len(sessions), "limit_turns": int(args.limit_turns)},
        "configuration": cfg,
        "summary": {"requests": len(rows), "skipped": skipped, **metric, "mean_prompt_tokens": (sum(r["prompt_tokens"] for r in rows) / len(rows)) if rows else None,
                    "mean_elapsed_s": (sum(r["elapsed_s"] for r in rows) / len(rows)) if rows else None, "wall_s": wall, "engine_build_s": build_s},
        "rows": rows,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "vllm": vllm.__version__, "vllm_path": vllm.__file__, "tangram_upstream_commit": git_head(tangram_root),
                        "tangram_overlay_diff_sha256": diff_sha256(tangram_root), "tangram_overlay_dirty": git_dirty(tangram_root), "repo_commit": git_head(REPO), "repo_dirty": git_dirty(REPO),
                        "runner_sha256": file_sha256(Path(__file__).resolve()), "gpu": gpu, "nvidia_smi_before": smi_before, "nvidia_smi_after": smi_after, "cli": sys.argv, "hostname": platform.node()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite receipt {args.output}")
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"arm": args.arm, "task": args.task, "requests": len(rows), **metric, "wall_s": round(wall, 1)}))


if __name__ == "__main__":
    main()
