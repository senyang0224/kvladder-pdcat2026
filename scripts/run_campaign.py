"""Launch each selected conference cell in a fresh process, or print its command."""
import argparse
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
CONTROLS = ("frozen-fp16-copy", "kvladder-fp16-shared")
ARMS = ("full-fp16-apc", "fp8-apc", "ragged-fp16-copy", *CONTROLS)
SEEDS = (17, 18, 19)


def matrix(campaign):
    """Yield (group, context, budget GiB, branches, arm, seed) from the three frozen matrices."""
    if campaign == "length_ragged":
        for budget, b, arm, seed in itertools.product((24, 8), (1, 4, 8), CONTROLS, SEEDS):
            yield ("memory" if budget == 24 else "serving", 32768, budget, b, arm, seed)
        return
    for ctx, large, small in ((32768, 24, 8), (65536, 40, 16)):
        if campaign == "recency":
            for b, arm in itertools.product((1, 2, 4, 8), ARMS):
                yield ("memory", ctx, large, b, arm, 17)
            yield ("microbench", ctx, large, 8, "kvladder-fp16-shared", 17)
        for b, arm, seed in itertools.product((4, 8), ARMS if campaign == "recency" else CONTROLS, SEEDS):
            yield ("serving", ctx, small, b, arm, seed)
    if campaign == "recency":
        for arm, seed in itertools.product(("ragged-fp16-copy", *CONTROLS), SEEDS):
            yield ("lifecycle", 32768, 16, 5, arm, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=("recency", "matched", "length_ragged"))
    parser.add_argument("--group", choices=("serving", "memory", "microbench", "lifecycle"))
    parser.add_argument("--context", type=int, choices=(32768, 65536))
    parser.add_argument("--branches", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--trace-dir", type=Path, default=Path("traces"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_ATTENTION_BACKEND="FLASH_ATTN",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cells = [row for row in matrix(args.campaign)
             if all(value is None or row[i] == value for i, value in
                    ((0, args.group), (1, args.context), (3, args.branches), (4, args.arm), (5, args.seed)))]
    if not cells:
        parser.error("no conference cells match these filters")
    for group, ctx, budget, b, arm, seed in cells:
        dest = args.output_dir / args.campaign / group / f"{arm}_{ctx}_b{b}_s{seed}.json"
        trace = args.trace_dir / f"qa-{ctx}.json"
        cmd = [sys.executable, str(SCRIPTS / args.campaign / "run_arm.py"), "--arm", arm,
               "--scenario", "lifecycle" if group == "lifecycle" else "burst", "--phase", args.campaign,
               "--trace", str(trace), "--branches", str(b), "--kv-cache-memory-bytes", str(budget << 30),
               "--max-model-len", str(ctx + 512), "--seed", str(seed), "--repetition", str(seed-17),
               "--max-wall-s", "2400", "--microbench-ops", "500" if group == "microbench" else "0",
               "--output", str(dest)]
        if args.campaign != "recency":
            cmd.extend(["--admission-mode", "tight"])
        if args.campaign == "length_ragged":
            cmd.extend(["--compression-scorer", "fastkvzip"])
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            if not trace.is_file():
                parser.error(f"missing trace {trace}; build the traces first")
            if dest.exists():
                parser.error(f"receipt already exists: {dest}; choose a fresh output directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
