"""Render v3 research results from completed, mutually bound JSON evidence.

This script performs no fitting, inference, selection or bootstrap resampling.
It refuses missing or inconsistent inputs and never creates placeholder plots.
Install the repository's optional requirements-plot.txt dependencies to render.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform


ARMS = (
    ("full13", "Full + history gate"),
    ("no_category", "Without explicit\ncategory columns"),
    ("no_recent", "Without last-history\ninteraction column"),
    ("no_gate", "Without history gate"),
)
COMPONENTS = (
    ("no_category", "Explicit category\noverlap columns"),
    ("no_recent", "Last-history\ninteraction column"),
    ("no_gate", "History-length gate"),
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def interval_data(interval, expected_delta, label):
    delta = number(interval["delta"], label + " delta")
    if not math.isclose(delta, expected_delta, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label} paired delta disagrees with arm metrics")
    ci = interval["ci95"]
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("bootstrap interval must contain two endpoints")
    lower, upper = [number(value, label + " CI") for value in ci]
    if lower > upper or min(lower, delta) < -1 or max(upper, delta) > 1:
        raise ValueError("recall difference or bootstrap interval is invalid")
    if (interval.get("resampling_unit") != "user" or interval.get("replicates") != 1000
            or interval.get("seed") != 2026):
        raise ValueError("expected the declared 1,000-repeat paired user bootstrap")
    return {"delta_pp": delta * 100, "ci95_pp": [lower * 100, upper * 100]}


def metric_recall(report, arm):
    metric = report["metrics"][arm]
    total = report["all_clicks"]
    if type(total) is not int or total <= 0:
        raise ValueError("positive integer logged-click denominator required")
    for k in (10, 100):
        hits, recall = metric[f"hits{k}"], number(metric[f"recall{k}"], f"{arm} R@{k}")
        if (type(hits) is not int or not 0 <= hits <= total or not 0 <= recall <= 1
                or not math.isclose(recall, hits / total, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("recall disagrees with the reported integer counts")
    if metric["hits10"] > metric["hits100"]:
        raise ValueError("top-10 hits exceed top-100 hits")
    return float(metric["recall10"])


def extract_plot_data(dev, confirm, benchmark, *, confirm_sha256):
    """Validate provenance, same-candidate invariants and every plotted delta."""
    if dev.get("phase") != "dev" or confirm.get("phase") != "confirm":
        raise ValueError("completed v3 dev and confirm reports are required")
    selection = dev.get("selection_sha256")
    if (not valid_sha(selection) or confirm.get("selection_sha256") != selection
            or dev.get("confirm_sha256") != confirm_sha256):
        raise ValueError("dev must bind this exact confirmation and the same selection")
    dev_bindings, confirm_bindings = dev["bindings"], confirm["bindings"]
    if (not dev_bindings.get("sources") or dev_bindings["sources"] != confirm_bindings.get("sources")
            or dev_bindings.get("fit_report_sha256") != confirm_bindings.get("fit_report_sha256")
            or not valid_sha(dev_bindings.get("fit_report_sha256"))
            or dev_bindings.get("selection_sha256") != selection
            or confirm_bindings.get("selection_sha256") != selection):
        raise ValueError("dev and confirmation must share frozen fit and source bindings")
    if (dev.get("delta") not in (0, 2, 5, 10) or dev["delta"] != confirm.get("delta")
            or isinstance(dev["delta"], bool)):
        raise ValueError("dev and confirmation must use the same declared delta")
    for report, split in ((dev, 2), (confirm, 1)):
        rows = report["bindings"]["rows"]
        if (type(rows.get("count")) is not int or rows["count"] <= 0
                or rows.get("split_values") != [split] or not valid_sha(rows.get("row_sha256"))):
            raise ValueError("invalid evaluation population binding")
        if report["invariants"].get("per_request_candidate_and_hit100_invariant") is not True:
            raise ValueError("same-candidate per-request Recall@100 proof is required")

    base = metric_recall(dev, "baseline")
    baseline_hits100 = dev["metrics"]["baseline"]["hits100"]
    arms = []
    for arm, label in ARMS:
        value = metric_recall(dev, arm)
        if dev["metrics"][arm]["hits100"] != baseline_hits100:
            raise ValueError("an ablation changed the fixed candidate set's hit100 total")
        record = interval_data(dev["metrics"][arm]["versus_baseline"], value - base, arm)
        arms.append({"arm": arm, "label": label, "recall10": value, **record})
    full = dev["metrics"]["full13"]["recall10"]
    components = []
    for arm, label in COMPONENTS:
        record = interval_data(dev["metrics"]["full13"]["versus_" + arm],
                               full - dev["metrics"][arm]["recall10"], "full_minus_" + arm)
        components.append({"ablated_arm": arm, "label": label, **record})

    confirm_base, confirm_full = (metric_recall(confirm, arm) for arm in ("baseline", "full13"))
    confirmation = interval_data(confirm["metrics"]["full13"]["versus_baseline"],
                                 confirm_full - confirm_base, "confirmation")
    passed = confirmation["delta_pp"] > 0 and confirmation["ci95_pp"][0] > 0
    if (confirm.get("deployment_gate_passed") is not passed or dev.get("deployment_gate_passed") is not passed
            or dev.get("deployed_recipe") != ("full13+gate" if passed else "baseline")):
        raise ValueError("deployment outcome differs from the prespecified confirmation rule")

    if benchmark.get("experiment") != "frozen_v2_vs_exact_v3_popular_only_retrieval":
        raise ValueError("unexpected retrieval performance experiment")
    equivalence = benchmark["equivalence"]
    if (equivalence.get("exact") is not True or equivalence.get("mismatch_field_occurrences") != 0
            or equivalence.get("numeric_tolerance") != 0
            or type(equivalence.get("candidate_batches_compared")) is not int
            or equivalence["candidate_batches_compared"] <= 0):
        raise ValueError("exact retrieval-equivalence proof is required")
    if (benchmark["protocol"].get("sequence") != ["v2", "v3_fast", "v3_fast", "v2"]
            or benchmark["protocol"].get("cycles_per_mode") != 3
            or benchmark["candidate_config"].get("pool_size") != 100
            or benchmark["candidate_config"].get("channels") != ["popular"]):
        raise ValueError("unexpected ABBA protocol or candidate budget")
    for report in (dev, confirm):
        if (report["runtime"].get("generator_bindings") != benchmark.get("input_bindings")
                or report["runtime"].get("generator_version") != benchmark["generator_versions"]["v3_fast"]):
            raise ValueError("quality and cost reports use different candidate generators or inputs")
    for current, historical in (("newsforge/v3/fast_retrieval.py", "fast_retrieval.py"),
                                ("newsforge/v2/retrieval.py", "v2/retrieval.py")):
        if dev_bindings["sources"].get(current) != benchmark["source_sha256"].get(historical):
            raise ValueError("quality and performance calculation sources differ")

    performance = []
    for mode in ("cold", "warm"):
        summary = benchmark["summary"][mode]
        seconds = [number(summary[implementation]["measured_query_wall_seconds"], mode + " seconds")
                   for implementation in ("v2", "v3_fast")]
        counts = [summary[implementation]["measured_queries"] for implementation in ("v2", "v3_fast")]
        ratio = number(summary["observed_wall_speedup_v2_over_v3"], mode + " ratio")
        if (any(value <= 0 for value in seconds) or ratio <= 0
                or any(type(value) is not int or value <= 0 for value in counts) or counts[0] != counts[1]
                or not math.isclose(ratio, seconds[0] / seconds[1], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("pooled observed ratio disagrees with equal-budget measured wall time")
        cycles = summary["paired_abba_cycles"]
        if len(cycles) != 3:
            raise ValueError("all three cold and warm ABBA cycles must be retained")
        cycle_ratios = [number(record["observed_wall_speedup_v2_over_v3"], mode + " cycle") for record in cycles]
        if any(value <= 0 for value in cycle_ratios):
            raise ValueError("ABBA cycle ratio must be positive")
        performance.append({"mode": mode, "observed_pooled_ratio": ratio,
                            "cycle_ratios": cycle_ratios, "queries_per_implementation": counts[0],
                            "v2_seconds": seconds[0], "fast_seconds": seconds[1],
                            "latency_ms": {name: summary[name]["latency_ms"] for name in ("v2", "v3_fast")}})
    return {"selection_sha256": selection, "delta": dev["delta"], "dev_requests": dev_bindings["rows"]["count"],
            "baseline_recall10": base, "all_arms_recall100": dev["metrics"]["baseline"]["recall100"],
            "arms_versus_baseline": arms, "full_minus_ablations": components,
            "confirmation": {**confirmation, "gate_passed": passed,
                             "deployed_recipe": dev["deployed_recipe"], "requests": confirm_bindings["rows"]["count"]},
            "performance": performance, "exact_batches_compared": equivalence["candidate_batches_compared"]}


def dependency_versions():
    result = {"python": platform.python_version()}
    for name in ("matplotlib", "numpy", "Pillow", "contourpy", "cycler", "fonttools", "kiwisolver",
                 "packaging", "pyparsing", "python-dateutil"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def draw_intervals(axis, records, color, title, subtitle, xlabel):
    from matplotlib.ticker import MaxNLocator

    bounds = [0.]
    for row, record in enumerate(records):
        lower, upper = record["ci95_pp"]
        bounds.extend((lower, upper, record["delta_pp"]))
        # Percentile intervals need not enclose their point estimate.
        axis.hlines(row, lower, upper, color=color, linewidth=2, zorder=3)
        axis.vlines([lower, upper], row - .07, row + .07, color=color, linewidth=1.2)
        axis.scatter([record["delta_pp"]], [row], color=color, s=47, edgecolor="white", linewidth=.6, zorder=4)
        axis.annotate(f"{record['delta_pp']:+.2f} pp", (record["delta_pp"], row),
                      xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color="#26333f")
    span = max(max(bounds) - min(bounds), .2)
    axis.set_xlim(min(bounds) - .25 * span, max(bounds) + .25 * span)
    axis.set_ylim(len(records) - .4, -.7)
    axis.set_yticks(range(len(records)), [record["label"] for record in records])
    axis.tick_params(axis="y", length=0, pad=8)
    axis.axvline(0, color="#8a96a0", linestyle=(0, (3, 3)), linewidth=1, zorder=1)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.grid(axis="x", color="#e7edf1", linewidth=.7, zorder=0)
    axis.spines["left"].set_visible(False)
    axis.set_xlabel(xlabel, labelpad=12)
    axis.set_title(title, loc="left", pad=35)
    axis.text(0, 1.045, subtitle, transform=axis.transAxes, fontsize=8.5, color="#52606d")


def render(dev_path, confirm_path, benchmark_path, output_dir, *, overwrite=False):
    paths = {"dev": Path(dev_path), "confirm": Path(confirm_path), "benchmark": Path(benchmark_path)}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    reports = {name: json.loads(value) for name, value in raw.items()}
    data = extract_plot_data(reports["dev"], reports["confirm"], reports["benchmark"],
                             confirm_sha256=sha256_bytes(raw["confirm"]))
    output_dir = Path(output_dir)
    outputs = {kind: output_dir / f"v3_results.{kind}" for kind in ("png", "svg")}
    metadata_path = output_dir / "v3_results.metadata.json"
    if not overwrite and any(path.exists() for path in [*outputs.values(), metadata_path]):
        raise FileExistsError("plot outputs exist; --overwrite regenerates only visualization files")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    style = {"font.family": "DejaVu Sans", "font.size": 9.2, "axes.labelsize": 9.2,
             "axes.titlesize": 11.5, "axes.titleweight": "semibold", "figure.facecolor": "white",
             "axes.facecolor": "white", "axes.edgecolor": "#c0c7cf", "axes.labelcolor": "#26333f",
             "xtick.color": "#52606d", "ytick.color": "#26333f", "axes.spines.top": False,
             "axes.spines.right": False, "svg.hashsalt": "newsforge-v3-results-v1", "svg.fonttype": "path",
             "savefig.facecolor": "white", "savefig.edgecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.2), gridspec_kw={"width_ratios": [1.1, 1.05, 1.]})
        fig.subplots_adjust(left=.145, right=.975, top=.735, bottom=.36, wspace=.95)
        fig.suptitle("NewsForge v3: bounded personalization and exact retrieval", x=.04, y=.96,
                     ha="left", fontsize=19, fontweight="bold", color="#1f3042")
        fig.text(.04, .901, "Fixed candidate sets; exploratory dev comparisons; cold and warm cost shown separately.",
                 fontsize=10.5, color="#52606d")
        draw_intervals(axes[0], data["arms_versus_baseline"], "#245b91", "A  Dev versus popularity",
                       "Paired user-cluster 95% CI; exploratory", "Recall@10 difference (pp)")
        draw_intervals(axes[1], data["full_minus_ablations"], "#247c79", "B  Full minus ablation",
                       "Same frozen delta; column-group effects", "Recall@10 difference (pp)")

        cost = axes[2]
        maximum = 1.
        for row, record in enumerate(data["performance"]):
            value, observations = record["observed_pooled_ratio"], record["cycle_ratios"]
            maximum = max(maximum, value, *observations)
            color = "#247c79" if value > 1 else "#906547"
            for offset, observation in zip((-.10, 0., .10), observations):
                cost.scatter(observation, row + offset, s=20, color="#a1adb7", zorder=2)
            cost.scatter(value, row, marker="D", s=65, color=color, edgecolor="white", linewidth=.7, zorder=4)
            cost.annotate(f"{value:.2f}x", (value, row), xytext=(0, 18), textcoords="offset points",
                          ha="center", fontsize=10, fontweight="semibold", color=color)
        cost.axvline(1, color="#8a96a0", linestyle=(0, (3, 3)), linewidth=1)
        cost.set_xlim(0, maximum * 1.22)
        cost.set_ylim(1.45, -.55)
        cost.set_yticks([0, 1], ["Cold derived\ncache", "Warm derived\ncache"])
        cost.tick_params(axis="y", length=0, pad=8)
        cost.xaxis.set_major_locator(MaxNLocator(nbins=4))
        cost.grid(axis="x", color="#e7edf1", linewidth=.7, zorder=0)
        cost.spines["left"].set_visible(False)
        cost.set_xlabel("Measured v2 time / fast time (x)", labelpad=12)
        cost.set_title("C  Retrieval cost", loc="left", pad=35)
        cost.text(0, 1.045, "Diamond: pooled ABBA; dots: 3 cycles\nObserved ratios, not confidence intervals",
                  transform=cost.transAxes, fontsize=8.5, color="#52606d", linespacing=1.3)

        confirmation = data["confirmation"]
        decision = "personalized recipe passed" if confirmation["gate_passed"] else "keep popularity service"
        ci = confirmation["ci95_pp"]
        notes = [
            f"A/B: {data['dev_requests']:,} previously observed dev requests; same <=100 candidates, per-request Recall@100 invariant "
            f"(aggregate {data['all_arms_recall100'] * 100:.2f}%).",
            f"Confirmation: delta={data['delta']}; change {confirmation['delta_pp']:+.2f} pp "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]; {decision}. Dev ablations do not change this decision.",
            "A/B intervals: 1,000 paired user-cluster bootstrap repeats; dev comparisons are exploratory and unadjusted for multiplicity.",
            "B deletes explicit columns; category and recent-history information remain in other features. pp = percentage points.",
            f"C: {data['exact_batches_compared']:,} exact CandidateBatch comparisons; cold resets derived state only, not OS caches. "
            "Ratios below 1 indicate a slowdown.",
            "C measures retrieval only on a shared CPU desktop; no HTTP/production throughput or online CTR claim.",
        ]
        for row, note in enumerate(notes):
            fig.text(.04, .228 - row * .033, note, fontsize=8.5, color="#52606d", ha="left")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(outputs["png"], dpi=220,
                        metadata={"Software": f"NewsForge v3; Matplotlib {matplotlib.__version__}"})
            fig.savefig(outputs["svg"], metadata={"Creator": "NewsForge v3 plot_v3_results.py", "Date": None})
        finally:
            plt.close(fig)

    metadata = {
        "schema_version": 1,
        "inputs": {name: {"file_name": paths[name].name, "sha256": sha256_bytes(value)} for name, value in raw.items()},
        "plot_source": {"path": "scripts/plot_v3_results.py", "sha256": sha256_bytes(Path(__file__).read_bytes())},
        "dependencies": dependency_versions(),
        "rendering": {"backend": "Agg", "size_inches": [16.8, 7.2], "png_dpi": 220,
                      "font": "DejaVu Sans", "svg_hashsalt": style["svg.hashsalt"], "svg_date": None,
                      "svg_fonttype": "path"},
        "plotted_data": data,
        "uncertainty": {"quality_panels": "paired user-cluster bootstrap 95% CI, 1000 repeats, seed 2026; exploratory dev comparisons",
                        "cost_panel": "pooled measured-time ratios and all three ABBA cycle ratios; no speedup confidence interval"},
        "outputs": {kind: {"file_name": path.name, "sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}
                    for kind, path in outputs.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="reports/v3/dev.json")
    parser.add_argument("--confirm", default="reports/v3/confirm.json")
    parser.add_argument("--benchmark", default="reports/v3/fast_retrieval_benchmark.json")
    parser.add_argument("--output", default="reports/v3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = render(args.dev, args.confirm, args.benchmark, args.output, overwrite=args.overwrite)
    print(json.dumps({"outputs": metadata["outputs"], "metadata": str(Path(args.output) / "v3_results.metadata.json")}, indent=2))


if __name__ == "__main__":
    main()
