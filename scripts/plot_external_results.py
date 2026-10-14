"""Plot completed external aggregate evidence; never read events or recompute inference.

This is a presentation-only consumer of evaluation.json. All existing outputs
are protected, and negative or inconclusive results are displayed unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform


METHODS = (
    ("baseline", "Visit-popularity baseline", "#64748b"),
    ("full13+gate", "Full + history gate", "#2563eb"),
    ("no_category+gate", "Without category columns", "#c2410c"),
    ("no_recent+gate", "Without recent-history column", "#7c3aed"),
    ("full13+no_gate", "Without history gate", "#0f766e"),
)
DAYS = ("2017-01-05", "2017-01-06", "2017-01-07")
COMPARISONS = (
    ("full13+gate_minus_baseline", "Full + gate minus baseline\nPrimary comparison", "full13+gate", "baseline"),
    ("no_category+gate_minus_full13+gate", "Without category minus full\nDescriptive ablation", "no_category+gate", "full13+gate"),
    ("no_recent+gate_minus_full13+gate", "Without recent-history minus full\nDescriptive ablation", "no_recent+gate", "full13+gate"),
    ("full13+no_gate_minus_full13+gate", "Without gate minus full\nDescriptive ablation", "full13+no_gate", "full13+gate"),
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("plot values must be finite numbers")
    return float(value)


def extract(report):
    if report.get("status") != "passed":
        raise ValueError("a completed external aggregate report is required")
    if (report.get("external_model_fitting") is not False
            or report.get("external_hyperparameter_selection") is not False
            or report.get("prospective_future_generalization") is not False
            or report.get("online_ab_test") is not False):
        raise ValueError("report does not match the frozen retrospective experiment")
    if (report.get("delta") != 10 or report.get("seeds") != [17, 29, 43]
            or report.get("audit", {}).get("status") != "passed"
            or report["audit"].get("queries") != 128):
        raise ValueError("expected frozen recipe and completed independent audit")
    periods = []
    for label, summary in [("Overall", report["overall"])] + [(day, report["by_day"][day]) for day in DAYS]:
        count = summary["queries"]
        if type(count) is not int or count < 0 or (label == "Overall" and count == 0):
            raise ValueError("invalid query denominator")
        values = {}
        for method, _, _ in METHODS:
            if not count:
                values[method] = None
                continue
            result = summary["methods"][method]
            hits, rate = result["hits10"], number(result["hit_rate10"])
            if (type(hits) is not int or not 0 <= hits <= count or result["denominator"] != count
                    or not math.isclose(rate, hits / count, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("HitRate@10 disagrees with integer counts")
            values[method] = rate * 100.
        periods.append({"label": label, "queries": count, "hit_rate10_percent": values})
    if sum(period["queries"] for period in periods[1:]) != periods[0]["queries"]:
        raise ValueError("daily denominators do not sum to the overall denominator")
    for method, _, _ in METHODS:
        day_hits = sum(report["by_day"][day]["methods"][method]["hits10"]
                       for day in DAYS if report["by_day"][day]["queries"])
        if day_hits != report["overall"]["methods"][method]["hits10"]:
            raise ValueError("daily hits do not sum to the overall hits")
    intervals = []
    for key, label, left, right in COMPARISONS:
        item = report["overall"]["comparisons"][key]
        point = number(item["difference"])
        endpoints = item["ci95"]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError("interval must contain two endpoints")
        lower, upper = map(number, endpoints)
        expected = (periods[0]["hit_rate10_percent"][left] - periods[0]["hit_rate10_percent"][right]) / 100.
        if (not math.isclose(point, expected, rel_tol=0, abs_tol=1e-12)
                or not -1 <= lower <= upper <= 1 or not -1 <= point <= 1):
            raise ValueError("paired difference or interval is inconsistent")
        if (item.get("bootstrap_repetitions") != 1000 or item.get("bootstrap_rng_seed") != 2026
                or item.get("valid_repetitions") != 1000):
            raise ValueError("unexpected stored bootstrap configuration")
        intervals.append({"comparison": key, "label": label, "left_method": left,
                          "right_method": right, "difference_pp": point * 100.,
                          "ci95_pp": [lower * 100., upper * 100.],
                          "interpretation": "primary" if left == "full13+gate" else "descriptive ablation"})
    return {"periods": periods, "intervals": intervals,
            "methods": [{"key": key, "label": label, "color": color} for key, label, color in METHODS]}


def render(report_path, output_dir=None):
    report_path = Path(report_path)
    output_dir = report_path.parent if output_dir is None else Path(output_dir)
    raw = report_path.read_bytes()
    report = json.loads(raw)
    plotted = extract(report)
    outputs = {kind: output_dir / filename for kind, filename in (
        ("png", "external_results.png"), ("svg", "external_results.svg"), ("metadata", "metadata.json"))}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("plot output exists; retain it and choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(17.5, 7.7), gridspec_kw={"width_ratios": [1.15, 1.]})
    left, right = axes
    x = np.arange(4)
    width = .145
    for index, (method, label, color) in enumerate(METHODS):
        values = [period["hit_rate10_percent"][method] for period in plotted["periods"]]
        left.bar(x + (index - 2) * width, [float("nan") if value is None else value for value in values],
                 width=width, color=color, label=label, zorder=3)
    left.set_xticks(x, [f"{period['label']}\nn = {period['queries']:,}" for period in plotted["periods"]])
    left.set_ylabel("HitRate@10 (%)")
    highest = max(value or 0. for period in plotted["periods"] for value in period["hit_rate10_percent"].values())
    left.set_ylim(0, max(1., highest * 1.2))
    for index, period in enumerate(plotted["periods"]):
        if not period["queries"]:
            left.text(index, .04, "No queries", transform=left.get_xaxis_transform(), ha="center", fontsize=9)
    left.set_title("A  |  Overall and daily hit rates", loc="left", pad=18, fontweight="bold")
    left.grid(axis="y", alpha=.2, zorder=0)
    left.legend(loc="upper left", bbox_to_anchor=(-.015, -.17), frameon=False, ncol=2, fontsize=9)

    colors = {method: color for method, _, color in METHODS}
    bounds = [0.]
    for index, interval in enumerate(plotted["intervals"]):
        lo, hi = interval["ci95_pp"]
        point = interval["difference_pp"]
        color = colors[interval["left_method"]]
        right.hlines(index, lo, hi, color=color, linewidth=2.3)
        right.plot([lo, hi], [index, index], linestyle="none", marker="|", color=color, markersize=10)
        right.plot(point, index, marker="o", color=color, markersize=7)
        right.annotate(f"{point:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]", (point, index),
                       xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9, color=color)
        bounds.extend((lo, hi, point))
    margin = max(.25, (max(bounds) - min(bounds)) * .18)
    right.set_xlim(min(bounds) - margin, max(bounds) + margin)
    right.set_ylim(3.6, -.65)
    right.set_yticks(range(4), [interval["label"] for interval in plotted["intervals"]])
    right.set_xlabel("HitRate@10 difference (percentage points)", labelpad=12)
    right.axvline(0, color="#334155", linestyle="--", linewidth=1, zorder=0)
    right.grid(axis="x", alpha=.2)
    right.set_title("B  |  Overall paired differences and 95% intervals", loc="left", pad=18, fontweight="bold")

    figure.suptitle("Adressa | Frozen MIND transfer", x=.06, y=.97, ha="left", fontsize=21, fontweight="bold")
    figure.text(.06, .91, "Retrospective page-visit recommendation; Norwegian titles through a frozen English representation.", fontsize=11, color="#475569")
    if report.get("synthetic_fixture") is True:
        figure.text(.06, .855, "SYNTHETIC TEST DATA — implementation check only", fontsize=10, color="#a16207", fontweight="bold")
    figure.text(.06, .065, "Intervals: paired user-cluster bootstrap (1,000 repeats; seed 2026), conditional on these three days. Ablations are descriptive.", fontsize=9, color="#475569")
    figure.text(.06, .035, "No external fitting or tuning. No exposure negatives. No online A/B or prospective generalization claim.", fontsize=9, color="#475569")
    figure.subplots_adjust(left=.06, right=.98, top=.79, bottom=.25, wspace=.82)
    try:
        for kind in ("png", "svg"):
            with outputs[kind].open("xb") as stream:
                figure.savefig(stream, format=kind, dpi=180, metadata={"Creator": "NewsForge external aggregate plot"})
    finally:
        plt.close(figure)
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": report_path.name, "source_sha256": digest(raw),
        "script_name": Path(__file__).name, "script_sha256": digest(Path(__file__).read_bytes()),
        "source_status": report["status"], "source_scope": report["scope"],
        "synthetic_fixture": report.get("synthetic_fixture") is True,
        "archive_sha256": report.get("archive_sha256"),
        "sample_identity_sha256": report.get("sample_identity_sha256"),
        "source_positive_transfer_supported": report.get("positive_transfer_supported_on_preregistered_sample"),
        "presentation_only": True, "raw_event_access": False, "statistical_recomputation": False,
        "displayed_units": {"panel_a": "percent", "panel_b": "percentage points (left method minus right method)"},
        "interval_note": "Stored 95% paired user-cluster intervals; three realized dates; ablations descriptive.",
        "plotted_data": plotted, "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "outputs": {kind: {"name": outputs[kind].name, "sha256": digest(outputs[kind].read_bytes())} for kind in ("png", "svg")},
    }
    with outputs["metadata"].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/external/adressa_v1/evaluation.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = render(args.report, args.output_dir)
    print(json.dumps({"source_sha256": result["source_sha256"], "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
