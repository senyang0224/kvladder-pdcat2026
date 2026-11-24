# Reproducing the PDCAT 2026 experiments

Run the commands from the public repository root. The CPU workflow reads archived measurements. The GPU workflow creates fresh receipts from locally staged inputs.

## Campaigns and comparisons

| Campaign | Representation and admission | Cells | What to compare |
| --- | --- | --- | --- |
| `recency` | Sink-and-window, equal retained lengths. Native full-input admission. Dense FP16/FP8 APC, native private ragged, frozen private copy and shared arms. | 111 runs. 40 all-resident memory, 60 constrained serving, 2 additional Shared microbenchmarks and 9 lifecycle runs. | At 32K / 24 GiB / eight resident branches, Frozen uses 10.75 GiB and Shared uses 1.25 GiB (88.4% less live KV). |
| `matched` | Same recency representation. Frozen and Shared both reserve the uncomputed private prompt and output budget. Frozen also allocates one base copy per branch. | 2 contexts × 2 fan-outs (4, 8) × 2 arms × 3 seeds = 24 primary runs. Two preflights are excluded. | At 32K / 8 GiB / eight offered branches, first-round admission is 5 versus 8. Burst maximum TTFT is 1,517 versus 401 ms. |
| `length_ragged` | FastKVzip content gate, per-layer budget. Matched admission with accounting for outstanding private tails. | 32K × 2 budgets (24, 8 GiB) × 3 fan-outs (1, 4, 8) × 2 arms × 3 seeds = 36 replication runs, or 18 pairs. | At 24 GiB and eight resident branches the difference is 311,376 pages, exactly eight frozen-base copies. At 8 GiB and eight offered branches, first-round admission is 2 versus 5. |

All three campaigns used Llama-3.1-8B-Instruct in FP16 on one A100-SXM4-80GB in eager mode. Serving branches share the first 32,768 or 65,536 tokens of SCBench QA session 0. Eight prompts cycle over its five turns. The generator emits 32 greedy tokens in these bursts. Recency retains 9,728 or 19,456 positions per row, respectively (ratio 0.296875). Sink, window and floor are 16, 32 and 64 tokens. A block holds 16 tokens and a page group covers four KV heads.

The 32K/64K memory budgets were 24/40 GiB and the serving budgets were 8/16 GiB. Native recency memory fan-outs were 1, 2, 4 and 8. Each memory point has one run, plus a second seed-17 Shared observation at fan-out eight from the microbenchmark. Serving medians use three serial processes with seeds 17, 18 and 19. The native and matched sets were run separately. Pair timing ratios by seed **within** a campaign.

## Reading and rebuilding the measurements

`data/bursts.csv` contains 162 per-run rows extracted from the completed burst receipts (102 recency, 24 matched, 36 length-ragged). Each row includes the SHA-256 of the source receipt. The lifecycle probes have a different workload and are excluded from this CSV. The packaged table omits generated text, model inputs, device identifiers and raw allocator logs.

```sh
python scripts/summarize.py --campaign recency --context 32768 --branches 8
python scripts/summarize.py --campaign matched --context 65536 --branches 8
python scripts/summarize.py --campaign length_ragged --context 32768 --branches 4
```

- A branch's TTFT starts at burst arrival and ends at its first output token. The burst statistic is the **maximum** over branches. The archived receipt field named `p95_ttft_ms` is legacy terminology. The extraction script computes the maximum directly from branch rows.
- First-round admission counts scheduler admissions within 50 ms of the first admission. A queued branch may already hold its base reference. Peak concurrency counts running, unfinished requests.
- Live allocated KV is the allocator's peak in bytes divided by 2³⁰ for GiB. Total process HBM, reserved cache capacity and live KV have different denominators.
- A warm-up request publishes the frozen base or warms APC before timing. Attach and copy costs and scheduler waiting are included. The native private ragged baseline compresses each prompt.

## GPU setup

The recorded environment used Linux, Python 3.10.19, torch 2.9.0+cu128, CUDA 12.8, driver 580.173.02 and FlashAttention. The historical setup also had flashinfer-python 0.5.2 installed. The runtime upstream is Tangram commit `44ed97e8b7e5cf50ed0bf654b004edca27ef0609`. Each supplied patch is a complete diff against that commit. Use a separate checkout and Python environment for each overlay you run.

| Overlay | Experiments |
| --- | --- |
| `recency.patch` | Native recency, lifecycle, same-batch identity and full-session quality. |
| `matched.patch` | Matched recency and length-ragged replication. These share the engine and registry but have separate runners. |
| `refusal.patch` | The repaired fallback-on-arrival experiment, with its own registry snapshot. |

For example, in an activated CUDA-compatible environment, prepare the matched runtime. Choose `recency` or `refusal` for the other overlays. Set `CUDA_VISIBLE_DEVICES` to the single GPU used for the replay.

```sh
ROOT="$PWD"
OVERLAY=matched
git clone https://github.com/aiha-lab/tangram.git "tangram-$OVERLAY"
git -C "tangram-$OVERLAY" checkout --detach 44ed97e8b7e5cf50ed0bf654b004edca27ef0609
git -C "tangram-$OVERLAY" apply --check "$ROOT/src/overlays/$OVERLAY.patch"
git -C "tangram-$OVERLAY" apply "$ROOT/src/overlays/$OVERLAY.patch"
git -C "tangram-$OVERLAY" diff | sha256sum
```

Install the patched checkout in editable mode with CUDA extensions compatible with that checkout and your torch installation. Tangram documents an editable precompiled installation using `uv` (shown below). Record the resolved torch/vLLM versions before comparing timings. The historical binary environment is described above; a portable wheel or container image is outside this release. [Tangram installation at the pinned revision](https://github.com/aiha-lab/tangram/tree/44ed97e8b7e5cf50ed0bf654b004edca27ef0609#installation)

```sh
(cd "tangram-$OVERLAY" && VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto)
uv pip install pyarrow pandas
python -c 'import torch,vllm; print(torch.__version__, vllm.__file__)'
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

## Inputs and trace construction

Stage the model and tokenizer in a directory named `Llama-3.1-8B-Instruct`. Tangram uses that basename to resolve the content gate. The campaign used the **preprocessed** SCBench parquet with `prompts` and `ground_truth` columns. The trace builder checks the input file's SHA-256 and both token-prefix digests. Download the gate for the length-ragged campaign into the Hugging Face cache used by the runtime. The runner verifies the checkpoint actually selected by Tangram. [Model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Preprocessed SCBench](https://huggingface.co/datasets/Jang-Hyun/SCBench-preprocessed/tree/main) · [Gate checkpoint](https://huggingface.co/hmkim97/tangram-gate/tree/06a628fab229ff3075f69b61f43fa0ef6631c875/llama3.1-8b-instruct)

```sh
export MODEL="$PWD/models/Llama-3.1-8B-Instruct"
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir "$MODEL"
hf download Jang-Hyun/SCBench-preprocessed scbench_qa_eng.parquet scbench_choice_eng.parquet --repo-type dataset --local-dir data
hf download hmkim97/tangram-gate llama3.1-8b-instruct/q4_dim16_sink16.pt --revision 06a628fab229ff3075f69b61f43fa0ef6631c875
python scripts/build_trace.py --model "$MODEL" --parquet data/scbench_qa_eng.parquet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

| Input | SHA-256 |
| --- | --- |
| QA parquet | `6408de16ad0b6d1ffab9fd3f89180f44996f19fb2734bf9b840bb3249032c010` |
| Model config.json | `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e` |
| Tokenizer JSON | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |
| 32K token IDs (signed 64-bit little endian) | `e9a5a78325aab29e4f520a072882a5143b9c9ee1f80cc2ae7df29777791afa9a` |
| 64K token IDs (same encoding) | `9b1439171e827dba654cf92648a823c7af26b23a5ae97c93f9c1c2a62f46c223` |
| FastKVzip gate | `12749a3439be38455224720b056e469fb303dd07f15d2f73839389b41160ddff` |

## Running a campaign

Print the full planned matrix with `--dry-run`; add filters for a single operating point. The following commands execute the six primary matched runs at 32K / eight branches (two arms, three seeds) and summarize only the fresh receipts.

```sh
python scripts/run_campaign.py matched --dry-run
python scripts/run_campaign.py matched --context 32768 --branches 8 --output-dir results/replay
python scripts/extract_bursts.py results/replay/matched --output results/replay/matched.csv
python scripts/summarize.py --input results/replay/matched.csv
```

For the complete matrix, omit the context and branch filters. Select `recency` with its own overlay or `length_ragged` with the matched overlay. The launcher accepts `--group`, `--arm` and `--seed` filters. Each cell starts a fresh Python process and writes one JSON receipt. Choose an unused output directory for a repeat.

### Lifecycle, identity and quality

The recency lifecycle workload exercises two siblings, abort after one token, preemption, a mismatched attach and refused promotion of a compacted private tail. The payload identity audit uses the earlier registry and a separate instrumented runner. Add `--admission-cap 3` to the Shared constrained cells for the same-batch control. For the repaired refusal study, switch to the refusal overlay before running the third command below (repeat for seeds 17–19 with repetitions 0–2).

```sh
python scripts/run_campaign.py recency --group lifecycle
python scripts/audit/run_arm.py --arm kvladder-fp16-shared --trace traces/qa-32768.json --branches 8 --kv-cache-memory-bytes 8589934592 --max-model-len 33280 --seed 17 --admission-cap 3 --identity-payload-hash --output results/identity/shared.json
python scripts/audit/run_refusals.py --trace traces/qa-32768.json --seed 17 --repetition 0 --kv-cache-memory-bytes 17179869184 --max-model-len 33280 --output results/refusals/seed17.json
```

The measured 102 Frozen/Shared request pairs had identical published-base payloads. The full live state matched in all 30 all-resident pairs. Under the constrained budget, 72 of 72 full states differed with different batches and matched under the same-batch control. Twelve repaired refusal probes (four inputs × three seeds) completed privately.

The separate quality experiments used 48 GiB, complete input documents, greedy decoding capped at 64 tokens and sequential requests. Use the recency overlay. For Q1, run `scripts/quality/run_baseline.py` for FP16 APC, FP8 APC and native ragged over `scbench_qa_eng` (first-line token F1) and `scbench_choice_eng` (option accuracy). For the paired QA experiment use `run_full_session.py` as below. The device UUID is the UUID of the selected device on the replay machine.

```sh
GPU_UUID=$(python -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)')
python scripts/quality/run_baseline.py --model "$MODEL" --data-dir data --arm full-fp16-apc --task scbench_qa_eng --phase quality --expect-gpu-uuid "$GPU_UUID" --output results/quality/dense-qa.json
for arm in native shared; do
  python scripts/quality/run_full_session.py --model "$MODEL" --data-dir data --arm "$arm" --phase full_session_quality_v1 --expect-gpu-uuid "$GPU_UUID" --output "results/quality/$arm.json"
done
python scripts/quality/analyze.py --native results/quality/native.json --shared results/quality/shared.json --output-dir results/quality/analysis
```

The [paired scores](data/qa_pairs.csv) cover all 102 questions in 20 sessions. Request-weighted F1 is 0.21657 (Native) and 0.21814 (Shared). Taking the mean of within-session differences gives +0.00138, with a session-bootstrap 95% interval of [−0.00318, +0.00735] (20,000 draws, seed 20260912). Native compresses the entire input. Shared attaches its session's first 32K tokens and prefills the remainder privately. These processing differences limit attribution to freezing. The experiment specified no equivalence margin.

### Interpreting the length-ragged replication

Every base contained 64 layer/head-group rows with 36 distinct lengths from 64 to 19,408 tokens, totaling 622,752 retained positions on 38,922 pages. All 36 runs had the same selection fingerprint. At 24 GiB and eight branches, peaks were 20.20 and 10.70 GiB. The difference equals eight copies of the same base; private tails in both arms are padded to the longest retained row. At 8 GiB the arms admit different numbers of branches, so their two memory peaks also include different numbers of tails.

The [initial verdict](data/length_ragged_initial.json) failed two prespecified conditions. It counted running requests as owners although queued attaches already held references, and it compared peak memory at unequal concurrency. The [replication verdict](data/length_ragged_replication.json) uses one registry reference plus attached, unreleased owners and compares memory only at matched concurrency. The workload, runtime and gate were identical between these studies.

## Code provenance and limits

[Source fingerprints](source_hashes.csv) map published files to the code used for each campaign. Registry and overlay bytes match their recorded hashes. The distributed runners use portable imports and paths. The length-ragged runner also verifies the selected gate's hash. Fresh runner and whole-trace hashes therefore differ from the historical files. CPU validation covers the actual registry, trace reconstruction and statistics. Historical GPU timings are supplied through the per-run extract.

The prototype was evaluated with serialized publication callbacks on a single GPU and one model. The serving evidence covers a single main document at two prefix lengths and fan-out up to eight. Arbitrarily concurrent publishers, crash recovery, content-gated task quality, production tail percentiles and end-to-end cold-document latency were outside these experiments.
