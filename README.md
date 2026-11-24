# KVLadder · PDCAT 2026

**KVLadder: Immutable Sharing of Compressed Ragged KV Prefixes for LLM Serving**

Sen Yang, Jian Chen, Congming Gao and Jiwu Shu · PDCAT 2026

KVLadder lets several requests share a compressed KV prefix even when attention head groups retain different positions or lengths. A publisher freezes the retained layout and physical pages. Each branch references those pages, protects the retained part of each row and owns its continuation. The prototype extends Tangram's ragged cache in vLLM.

At a warm 32K-token prefix, eight simultaneous branches and an 8 GiB KV budget, the frozen-base private-copy control admitted five branches in the first round. KVLadder admitted eight. Under the same compression-aware admission rule, the median burst maximum time to first token (TTFT) fell from **1,517 ms to 401 ms** across three runs. Publication preceded the burst. Both attaching a shared branch and copying a private base were timed.

## Quick start

From the repository root, print the measurements behind the matched comparison. This uses Python 3.10+ and its standard library.

```sh
python scripts/summarize.py --campaign matched --context 32768 --branches 8
python scripts/summarize.py --campaign length_ragged --context 32768 --branches 8
```

To plot the matched TTFT medians and three-seed ranges, install Matplotlib and run the following command.

```sh
python -m pip install matplotlib
python scripts/summarize.py --campaign matched --plot plots/matched-ttft.pdf
```

The registry tests run on CPU. They cover physical-page aliasing, release and abort, refused attachments and protection of unequal row lengths.

```sh
python -m pip install numpy pytest
PYTHONPATH=src python -m pytest -q reproduction/checks
```

## Repository guide

| Path | Contents |
| --- | --- |
| [src/](src/) | Registry, campaign-specific snapshots and patches to Tangram's scheduler, cache manager and worker. |
| [scripts/](scripts/) | Trace builder, GPU campaign runners, quality and lifecycle checks, and CPU analysis and plotting. |
| [reproduction/REPRODUCTION.md](reproduction/REPRODUCTION.md) | GPU setup, commands, source fingerprints, data provenance and comparison boundaries. |
| [reproduction/data/](reproduction/data/) | Per-run burst measurements, paired QA scores and the length-ragged study's initial and replication verdicts. |
| [paper/](paper/) | [Download the PDCAT 2026 paper (PDF)](paper/KVLadder_PDCAT2026.pdf). |

The native-recency, matched-recency and content-gated experiments used distinct protocols. In the content-gated replication, **36 is both the number of runs and the number of distinct retained lengths**. Each base had **64 rows**. The 24 GiB runs compare memory at equal concurrency. The 8 GiB runs compare how many requests fit under the same budget. The quality study measures complete documents separately.

## License

The code is available under [Apache License 2.0](LICENSE). The runtime patches build on Tangram and vLLM. Their attribution is included in [the upstream notice](src/overlays/NOTICE). Model weights, gate weights, benchmark inputs and publication materials retain their respective terms.
