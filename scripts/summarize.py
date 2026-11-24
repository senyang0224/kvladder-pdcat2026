"""Print per-cell medians and optionally plot the matched-policy 32K burst."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

DEFAULT = Path(__file__).resolve().parents[1] / "reproduction/data/bursts.csv"
FROZEN, SHARED = "frozen-fp16-copy", "kvladder-fp16-shared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--campaign", choices=("recency", "matched", "length_ragged"))
    parser.add_argument("--context", type=int)
    parser.add_argument("--branches", type=int)
    parser.add_argument("--plot", type=Path, help="write the matched 32K / 8 GiB TTFT panel")
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.input.open(newline="") as fp:
        for row in csv.DictReader(fp):
            groups[row["campaign"], int(row["context"]), int(row["budget_gib"]),
                   int(row["branches"]), row["arm"]].append(row)
    if args.plot:
        plot_matched(groups, args.plot)
    selected = [(key, runs) for key, runs in sorted(groups.items()) if
                (args.campaign is None or args.campaign == key[0]) and
                (args.context is None or args.context == key[1]) and
                (args.branches is None or args.branches == key[3])]
    if not selected:
        parser.error("no matching measurements")
    print("campaign       tokens GiB  B arm                    n round1 conc.   TTFT ms   req/s   KV GiB")
    for (campaign, ctx, budget, b, arm), runs in selected:
        wave, conc, ttft, throughput, peak = (median(float(run[k]) for run in runs) for k in
                                               ("first_round", "max_concurrent", "max_ttft_ms", "throughput_rps", "peak_kv_bytes"))
        print(f"{campaign:13} {ctx:7} {budget:3} {b:2} {arm:21} {len(runs):2} "
              f"{wave:6g} {conc:5g} {ttft:9,.0f} {throughput:7.2f} {peak/2**30:8.2f}")


def plot_matched(groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.3), layout="constrained")
    for arm, label, color, marker in ((FROZEN, "Frozen-base private copy", "#884614", "o"),
                                      (SHARED, "KVLadder shared", "#206d85", "s")):
        fanouts = sorted(key[3] for key in groups if key[:3] == ("matched", 32768, 8) and key[4] == arm)
        if not fanouts:
            raise ValueError(f"plot requires matched 32K / 8 GiB runs for {arm}")
        values = [[float(run["max_ttft_ms"]) for run in groups["matched", 32768, 8, b, arm]] for b in fanouts]
        mids = [median(series) for series in values]
        ax.errorbar(fanouts, mids, yerr=[[m - min(v) for m, v in zip(mids, values)],
                                        [max(v) - m for m, v in zip(mids, values)]],
                    color=color, marker=marker, capsize=4, label=label)
    ax.set(xticks=sorted({key[3] for key in groups if key[:3] == ("matched", 32768, 8)}),
           xlabel="Branches per burst", ylabel="Maximum TTFT (ms)", title="Warm 32K prefix · 8 GiB KV budget")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
