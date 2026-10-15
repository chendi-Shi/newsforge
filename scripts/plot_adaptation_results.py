"""Render completed Adressa adaptation aggregates without reading private rows.

Presentation only: use stored effects and intervals, never select a model or
recompute inference/bootstrap. The output directory must exist; outputs are
created exclusively and existing results are never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform


METHODS = (
    ("baseline", "Popularity baseline", "#64748b"),
    ("legacy_frozen_english_delta10", "Frozen English (legacy delta 10)", "#b45309"),
    ("matched_frozen_english", "Frozen English (matched delta)", "#a16207"),
    ("adapted_english", "Refit English", "#7c3aed"),
    ("adapted_local", "Refit local (full)", "#2563eb"),
    ("adapted_local_no_recent", "Local without recent column", "#0891b2"),
    ("adapted_local_no_gate", "Local without history gate", "#0f766e"),
    ("deployed", "Selected replay policy", "#334155"),
)
COMPARISONS = (
    ("adapted_local_minus_baseline", "Local full minus popularity", "adapted_local", "baseline"),
    ("deployed_minus_baseline", "Selected policy minus popularity", "deployed", "baseline"),
    ("adapted_english_minus_matched_frozen_english", "Refit English minus\nmatched frozen English", "adapted_english", "matched_frozen_english"),
    ("adapted_local_minus_adapted_english", "Local full minus refit English", "adapted_local", "adapted_english"),
    ("adapted_local_minus_no_recent", "Local full minus\nwithout recent column", "adapted_local", "adapted_local_no_recent"),
    ("adapted_local_minus_no_gate", "Local full minus\nwithout history gate", "adapted_local", "adapted_local_no_gate"),
    ("adapted_local_minus_legacy_frozen_english", "Local full minus\nlegacy frozen English", "adapted_local", "legacy_frozen_english_delta10"),
)
DAYS = ("2017-01-05", "2017-01-06", "2017-01-07")
OUTPUT_NAMES = {
    "png": "adaptation_results.png",
    "svg": "adaptation_results.svg",
    "metadata": "adaptation_results.metadata.json",
}
RENDER_CONFIG = {
    "figsize_inches": [18, 11.5], "dpi": 180, "font_family": "DejaVu Sans",
    "font_size": 10, "svg_fonttype": "none", "background": "white",
    "panel_a_rectangle": [.19, .50, .285, .285],
    "panel_b_rectangle": [.735, .23, .235, .555],
    "panel_c_rectangle": [.19, .20, .285, .19],
    "absolute_rate_axis_starts_at_zero": True,
    "effect_units": "percentage points; left method minus right method",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("plot values must be finite numbers")
    return float(value)


def integer(value, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError("invalid integer count")
    return value


def close(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def summary_rates(summary):
    count = integer(summary["requests"], positive=True)
    if set(summary["methods"]) != {key for key, _, _ in METHODS}:
        raise ValueError("all eight declared methods must be present, without extra methods")
    rates, reachable = {}, set()
    for key, _, _ in METHODS:
        item = summary["methods"][key]
        hits = integer(item["hits10"])
        h100 = integer(item["hits100"])
        rate, rate100 = number(item["hit_rate10"]), number(item["hit_rate100"])
        if (item["denominator"] != count or not 0 <= hits <= h100 <= count
                or not close(rate, hits / count) or not close(rate100, h100 / count)):
            raise ValueError("rates must agree with integer counts and the shared denominator")
        reachable.add(h100)
        rates[key] = {"hits10": hits, "denominator": count, "hit_rate10_percent": 100 * rate}
    if len(reachable) != 1:
        raise ValueError("the shared-candidate HitRate@100 invariant failed")
    return count, rates


def interval(item, expected):
    point = number(item["difference"])
    endpoints = item["ci95"]
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("a stored 95% interval needs exactly two endpoints")
    lower, upper = map(number, endpoints)
    if not -1 <= lower <= upper <= 1 or not -1 <= point <= 1 or not close(point, expected):
        raise ValueError("stored effect or interval disagrees with its paired hit rates")
    if (item.get("bootstrap_repetitions") != 1000 or item.get("bootstrap_rng_seed") != 2026
            or item.get("valid_repetitions") != 1000):
        raise ValueError("unexpected stored bootstrap configuration")
    return {"difference_pp": point * 100, "ci95_pp": [lower * 100, upper * 100]}


def extract(report):
    if (report.get("schema_version") != 1 or report.get("passed") is not True
            or report.get("new_unseen_external_test") is not False
            or report.get("target_domain_fitting") is not True
            or report.get("online_ab_test") is not False
            or report.get("audit", {}).get("passed") is not True):
        raise ValueError("a completed, audited, retrospective adaptation report is required")
    count, overall = summary_rates(report["overall"])
    known = {key for key, *_ in COMPARISONS}
    present = set(report["overall"]["comparisons"])
    if (not present <= known or len(present) < 6
            or not {"adapted_local_minus_baseline", "deployed_minus_baseline"} <= present):
        raise ValueError("expected six or seven declared comparisons, including the selected policy")
    comparisons = []
    for key, label, left, right in COMPARISONS:
        if key not in present:
            continue
        expected = (overall[left]["hits10"] - overall[right]["hits10"]) / count
        comparisons.append({"key": key, "label": label, "left_method": left,
                            "right_method": right,
                            **interval(report["overall"]["comparisons"][key], expected)})
    if set(report["by_day"]) != set(DAYS):
        raise ValueError("the figure requires the three fixed benchmark dates")
    daily, totals = [], {key: 0 for key, _, _ in METHODS}
    for day in DAYS:
        n, rates = summary_rates(report["by_day"][day])
        expected = (rates["adapted_local"]["hits10"] - rates["baseline"]["hits10"]) / n
        effect = interval(report["by_day"][day]["comparisons"]["adapted_local_minus_baseline"], expected)
        daily.append({"day": day, "requests": n, "comparison": "adapted_local_minus_baseline", **effect})
        for key in totals:
            totals[key] += rates[key]["hits10"]
    if sum(item["requests"] for item in daily) != count or any(totals[k] != overall[k]["hits10"] for k in totals):
        raise ValueError("daily counts must reconcile exactly with overall counts")

    decision = report["confirmation"]
    delta, passed = decision["delta"], decision["gate_passed"]
    if type(passed) is not bool or type(delta) not in (int, float) or delta not in (0, 2, 5, 10):
        raise ValueError("unexpected confirmation gate or delta")
    if report["selection"]["delta"] != delta:
        raise ValueError("selection and confirmation deltas disagree")
    n = integer(decision["requests"], positive=True)
    baseline_hits = integer(decision["baseline_hits10"])
    local_hits = integer(decision["adapted_local_hits10"])
    if max(baseline_hits, local_hits) > n:
        raise ValueError("confirmation hits exceed its denominator")
    effect = interval(decision["primary_comparison"], (local_hits - baseline_hits) / n)
    expected_gate = delta > 0 and effect["difference_pp"] > 0 and effect["ci95_pp"][0] > 0
    policy = "adapted_local" if passed else "baseline"
    if passed != expected_gate or decision["deployed_recipe"] != policy:
        raise ValueError("confirmation decision disagrees with the declared gate")
    for summary in (report["overall"], *(report["by_day"][day] for day in DAYS)):
        for metric in ("hits10", "hits100", "hit_rate10", "hit_rate100", "mrr10", "denominator"):
            if summary["methods"]["deployed"][metric] != summary["methods"][policy][metric]:
                raise ValueError("selected replay policy metrics do not match the confirmed recipe")
        policy_effect = summary["comparisons"]["deployed_minus_baseline"]
        expected_effect = (summary["comparisons"]["adapted_local_minus_baseline"] if passed
                           else {"difference": 0., "ci95": [0., 0.]})
        if any(policy_effect[key] != expected_effect[key] for key in ("difference", "ci95")):
            raise ValueError("selected policy effect and interval disagree with its confirmed recipe")
    return {
        "requests": count,
        "methods": [{"key": key, "label": label, "color": color, **overall[key]} for key, label, color in METHODS],
        "overall_comparisons": comparisons,
        "daily_local_minus_baseline": daily,
        "confirmation": {"gate_passed": passed, "delta": delta, "requests": n,
                         "selected_replay_policy": policy, **effect},
    }


def _forest(axis, rows, labels, colors, *, title):
    bounds = [0.]
    for row, (item, color) in enumerate(zip(rows, colors)):
        lo, hi = item["ci95_pp"]
        point = item["difference_pp"]
        axis.hlines(row, lo, hi, color=color, linewidth=2.2, zorder=3)
        axis.plot([lo, hi], [row, row], linestyle="none", marker="|", color=color, markersize=9)
        axis.plot(point, row, marker="o", color=color, markersize=6.5, zorder=4)
        axis.annotate(f"{point:+.2f}  [{lo:+.2f}, {hi:+.2f}]", (point, row),
                      xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9, color=color)
        bounds.extend((lo, hi, point))
    span = max(bounds) - min(bounds)
    margin = max(.25, span * .25)
    axis.set_xlim(min(bounds) - margin, max(bounds) + margin)
    axis.set_ylim(len(rows) - .45, -.65)
    axis.set_yticks(range(len(rows)), labels)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.axvline(0, color="#64748b", linewidth=1, linestyle="--", zorder=1)
    axis.grid(axis="x", alpha=.18, zorder=0)
    axis.set_xlabel("HitRate@10 difference (percentage points)", labelpad=11)
    axis.set_title(title, loc="left", pad=23, fontweight="bold", fontsize=12)


def render(report_path, output_dir):
    report_path, output_dir = Path(report_path), Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError("the output directory must already exist")
    outputs = {kind: output_dir / name for kind, name in OUTPUT_NAMES.items()}
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("plot output exists; retain it and choose a new output directory")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    plotted = extract(report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL

    rc = {"font.family": RENDER_CONFIG["font_family"], "font.size": RENDER_CONFIG["font_size"],
          "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
          "svg.fonttype": RENDER_CONFIG["svg_fonttype"], "svg.hashsalt": "newsforge-adaptation-v1",
          "savefig.facecolor": RENDER_CONFIG["background"]}
    with plt.rc_context(rc):
        figure = plt.figure(figsize=RENDER_CONFIG["figsize_inches"], facecolor="white")
        a = figure.add_axes(RENDER_CONFIG["panel_a_rectangle"])
        b = figure.add_axes(RENDER_CONFIG["panel_b_rectangle"])
        c = figure.add_axes(RENDER_CONFIG["panel_c_rectangle"])
        values = [item["hit_rate10_percent"] for item in plotted["methods"]]
        colors = {key: color for key, _, color in METHODS}
        a.barh(np.arange(len(METHODS)), values, color=[item["color"] for item in plotted["methods"]], height=.63, zorder=3)
        a.set_yticks(range(len(METHODS)), [item["label"] for item in plotted["methods"]])
        a.tick_params(axis="y", length=0, pad=9)
        a.set_ylim(len(METHODS) - .4, -.7)
        a.set_xlim(0, max(1., max(values) * 1.20))
        for row, value in enumerate(values):
            a.annotate(f"{value:.3f}%", (value, row), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        a.set_xlabel("HitRate@10 (%) — shared request denominator", labelpad=11)
        a.grid(axis="x", alpha=.18, zorder=0)
        a.set_title(f"A  |  Absolute hit rates · n = {plotted['requests']:,}", loc="left", pad=23, fontsize=12, fontweight="bold")
        comparisons = plotted["overall_comparisons"]
        _forest(b, comparisons, [item["label"] for item in comparisons],
                [colors[item["left_method"]] for item in comparisons],
                title="B  |  All paired effects and stored 95% intervals")
        daily = plotted["daily_local_minus_baseline"]
        _forest(c, daily, [f"{item['day']}\nn = {item['requests']:,}" for item in daily],
                [colors["adapted_local"]] * len(daily), title="C  |  Daily local full minus popularity")

        gate = plotted["confirmation"]
        decision = "PASSED" if gate["gate_passed"] else "FAILED"
        policy_label = "local refit" if gate["gate_passed"] else "popularity baseline"
        figure.text(.045, .955, "Adressa | Local representation and matched refits", fontsize=22, fontweight="bold")
        figure.text(.045, .913, "Previously observed benchmark; target-domain adaptation", fontsize=16, fontweight="bold", color="#334155")
        figure.text(.045, .877,
                    f"Jan 4 confirmation: {decision}  |  Selected delta = {gate['delta']:g}  |  Selected replay policy: {policy_label}",
                    fontsize=12, color="#475569")
        if report.get("synthetic_fixture") is True:
            figure.text(.045, .842, "SYNTHETIC TEST DATA — layout and validation check only", fontsize=11, fontweight="bold", color="#b45309")
        figure.text(.045, .095, "January 5–7 comparisons are descriptive; this figure does not select or promote a model.", fontsize=10, color="#475569")
        figure.text(.045, .069, "Intervals: stored paired user-cluster bootstrap, 1,000 repetitions, seed 2026; conditional on the realized dates and shared feedback stream.", fontsize=10, color="#475569")
        figure.text(.045, .043, "No exposure negatives or online A/B claim. Matched ablations share the full model's selected delta; the legacy arm retains delta 10.", fontsize=10, color="#475569")
        try:
            # Reserve every destination exclusively before writing any image.
            with ExitStack() as stack:
                streams = {kind: stack.enter_context(path.open("xb")) for kind, path in outputs.items()}
                for kind in ("png", "svg"):
                    figure.savefig(streams[kind], format=kind, dpi=RENDER_CONFIG["dpi"],
                                   metadata={"Creator": "NewsForge adaptation aggregate plot"})
                    streams[kind].flush()
                metadata = {
                    "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_name": report_path.name, "source_sha256": digest(raw), "source_bytes": len(raw),
                    "plot_source_name": Path(__file__).name,
                    "plot_source_sha256": digest(Path(__file__).read_bytes()),
                    "presentation_only": True, "raw_event_access": False, "statistical_recomputation": False,
                    "synthetic_fixture": report.get("synthetic_fixture") is True,
                    "new_unseen_external_test": False, "target_domain_fitting": True,
                    "displayed_units": {"panel_a": "percent", "panels_b_c": "percentage points"},
                    "render_config": RENDER_CONFIG, "plotted_data": plotted,
                    "dependencies": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                     "numpy": np.__version__, "pillow": PIL.__version__},
                    "outputs": {kind: {"name": outputs[kind].name,
                                       "sha256": digest(outputs[kind].read_bytes()),
                                       "bytes": outputs[kind].stat().st_size} for kind in ("png", "svg")},
                }
                streams["metadata"].write((json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        finally:
            plt.close(figure)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = render(args.report, args.output_dir)
    # The metadata file cannot contain its own hash; expose it alongside the
    # image hashes in this caller-visible receipt, without filesystem paths.
    outputs = dict(metadata["outputs"])
    data = (args.output_dir / OUTPUT_NAMES["metadata"]).read_bytes()
    outputs["metadata"] = {"name": OUTPUT_NAMES["metadata"], "sha256": digest(data), "bytes": len(data)}
    print(json.dumps({"source_sha256": metadata["source_sha256"], "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
