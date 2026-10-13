"""Plot component ablations from completed, mutually bound v2 evidence files.

No model inference or evaluation occurs here. Missing or inconsistent inputs
raise before any output is created. Optional plotting dependencies are isolated
in requirements-plot.txt; the historical v1 dependency lock is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import statistics

SEEDS = (17, 29, 43)
COMPONENTS = (
    ("no_popular", "Popularity route"),
    ("no_content", "Content similarity"),
    ("no_tower", "Learned tower"),
    ("no_freshness", "First-seen window"),
)
FEATURE_GROUPS = (
    ("content_only", "Content\n15 features"),
    ("context_only", "Temporal context\n7 features"),
    ("full", "Combined\n22 features"),
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def extract_plot_data(catalog, ranking):
    """Validate provenance and reconcile every plotted aggregate with its arms."""
    if catalog.get("phase") != "catalog" or ranking.get("phase") != "ranking":
        raise ValueError("Completed catalog and sampled-ranking phase reports are required")
    selection_sha = catalog.get("selection_sha256", "")
    if (len(selection_sha) != 64 or any(c not in "0123456789abcdef" for c in selection_sha)
            or ranking.get("selection_sha256") != selection_sha):
        raise ValueError("Catalog and ranking must reference the same frozen selection SHA")
    if not catalog.get("bindings") or catalog["bindings"] != ranking.get("bindings"):
        raise ValueError("Catalog and ranking evidence bindings differ or are missing")
    if (not catalog.get("selected_before_v2_dev")
            or catalog["selected_before_v2_dev"] != ranking.get("selected_before_v2_dev")):
        raise ValueError("Evidence must share the same pre-dev selection record")
    queries, impressions = catalog.get("queries"), ranking.get("impressions")
    if (type(queries) is not int or queries <= 0 or type(impressions) is not int or impressions <= 0
            or catalog.get("query_split_counts") != {"2": queries}):
        raise ValueError("Positive official-dev query/impression populations are required")

    left = []
    full_recall = finite_number(catalog["metrics"]["full"]["click_micro_recall100"], "full Recall@100")
    for arm, label in COMPONENTS:
        interval = catalog["paired_user_intervals"][f"full_minus_{arm}_recall100"]
        delta = finite_number(interval["delta"], arm + " delta")
        ci = interval["ci95"]
        if len(ci) != 2:
            raise ValueError("Each paired comparison requires two CI endpoints")
        lower, upper = [finite_number(v, arm + " CI endpoint") for v in ci]
        if lower > upper or min(lower, delta) < -1 or max(upper, delta) > 1:
            raise ValueError("Recall difference or confidence interval is invalid")
        if interval.get("resampling_unit") != "user" or interval.get("replicates", 0) < 1:
            raise ValueError("Left-panel intervals must be paired user-cluster bootstrap intervals")
        ablated_recall = finite_number(catalog["metrics"][arm]["click_micro_recall100"], arm + " Recall@100")
        if not (0 <= full_recall <= 1 and 0 <= ablated_recall <= 1):
            raise ValueError("Recall@100 must be between zero and one")
        if not math.isclose(delta, full_recall - ablated_recall, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Paired delta disagrees with full-minus-ablated Recall@100")
        left.append({"arm": arm, "label": label, "delta_pp": 100 * delta,
                     "ci95_pp": [100 * lower, 100 * upper],
                     "bootstrap_replicates": interval["replicates"]})

    right = []
    for variant, label in FEATURE_GROUPS:
        stats = ranking["seed_statistics"][variant]["ndcg10"]
        mean = finite_number(stats["mean"], variant + " mean")
        sd = finite_number(stats["sample_std"], variant + " sample standard deviation")
        if not 0 <= mean <= 1 or sd < 0:
            raise ValueError("Ranking means and standard deviations are invalid")
        seed_values = []
        for seed in SEEDS:
            metric = ranking["metrics"][f"{variant}-seed{seed}"]["ndcg10"]
            value = finite_number(metric["mean"], variant + " seed mean")
            if not 0 <= value <= 1 or metric["eligible_impressions"] != impressions:
                raise ValueError("Each ranking seed must cover the same complete impression population")
            seed_values.append(value)
        if (not math.isclose(mean, statistics.mean(seed_values), abs_tol=1e-12, rel_tol=0)
                or not math.isclose(sd, statistics.stdev(seed_values), abs_tol=1e-12, rel_tol=0)):
            raise ValueError("Ranking aggregate disagrees with its three per-seed metrics")
        right.append({"variant": variant, "label": label, "mean_ndcg10": mean,
                      "sample_std_ndcg10": sd, "seeds": list(SEEDS), "per_seed_ndcg10": seed_values})
    return {"catalog_queries": queries, "ranking_impressions": impressions,
            "selection_sha256": selection_sha, "catalog_components": left,
            "ranking_feature_groups": right}


def dependency_versions():
    result = {"python": platform.python_version()}
    for package in ("matplotlib", "numpy", "Pillow", "contourpy", "cycler", "fonttools",
                    "kiwisolver", "packaging", "pyparsing", "python-dateutil"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def render_ablation(catalog_path, ranking_path, output_dir, *, overwrite=False):
    catalog_path, ranking_path, output_dir = map(Path, (catalog_path, ranking_path, output_dir))
    # Read and validate all inputs before creating directories, loading plotting
    # libraries, or writing anything. No placeholder formal plots are produced.
    catalog_bytes, ranking_bytes = catalog_path.read_bytes(), ranking_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    ranking = json.loads(ranking_bytes)
    data = extract_plot_data(catalog, ranking)
    outputs = {kind: output_dir / f"ablation.{kind}" for kind in ("png", "svg")}
    metadata_path = output_dir / "ablation.metadata.json"
    if not overwrite and any(path.exists() for path in [*outputs.values(), metadata_path]):
        raise FileExistsError("Plot outputs already exist; use --overwrite only to regenerate the visualization")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c0c7cf", "axes.labelcolor": "#26333f",
        "xtick.color": "#52606d", "ytick.color": "#26333f",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "newsforge-v2-component-ablations-v1", "svg.fonttype": "path",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
    }
    blue, teal, dark = "#245b91", "#237b78", "#1f3042"
    with plt.rc_context(style):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.9),
                                        gridspec_kw={"width_ratios": [1.12, 1]})
        fig.subplots_adjust(left=.165, right=.97, top=.77, bottom=.28, wspace=.38)
        fig.suptitle("NewsForge v2: component ablations", x=.07, y=.955, ha="left",
                     fontsize=18, fontweight="bold", color=dark)
        fig.text(.07, .887, "Two offline tasks; the uncertainty bars answer different questions.",
                 fontsize=10.5, color="#52606d")

        # Draw intervals as independent lines: percentile CIs need not contain
        # their point estimate, so asymmetric-errorbar positivity is not assumed.
        y = list(range(len(data["catalog_components"])))
        bounds = [0.0]
        for index, record in enumerate(data["catalog_components"]):
            lower, upper = record["ci95_pp"]
            bounds.extend([lower, upper, record["delta_pp"]])
            left.hlines(index, lower, upper, color=blue, linewidth=2.1, zorder=3)
            left.vlines([lower, upper], index - .085, index + .085, color=blue, linewidth=1.3)
            left.scatter([record["delta_pp"]], [index], color=blue, s=47, zorder=4,
                         edgecolor="white", linewidth=.7)
            left.annotate(f"{record['delta_pp']:+.2f} pp", (record["delta_pp"], index),
                          xytext=(0, 12), textcoords="offset points", ha="center",
                          color=dark, fontsize=9)
        span = max(max(bounds) - min(bounds), .2)
        left.set_xlim(min(bounds) - .13 * span, max(bounds) + .13 * span)
        left.set_ylim(len(y) - .45, -.62)
        left.set_yticks(y, [item["label"] for item in data["catalog_components"]])
        left.tick_params(axis="y", length=0, pad=9)
        left.axvline(0, color="#8693a0", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
        left.xaxis.set_major_locator(MaxNLocator(nbins=5))
        left.grid(axis="x", color="#e9edf1", linewidth=.7, zorder=0)
        left.set_xlabel("Full minus ablated Recall@100 (percentage points)", labelpad=11)
        left.set_title("A  Candidate components", loc="left", pad=20)
        left.text(0, 1.035, "Paired user-cluster bootstrap 95% CI", transform=left.transAxes,
                  color="#52606d", fontsize=9)
        left.spines["left"].set_visible(False)

        means = [item["mean_ndcg10"] for item in data["ranking_feature_groups"]]
        deviations = [item["sample_std_ndcg10"] for item in data["ranking_feature_groups"]]
        positions = list(range(3))
        right.errorbar(positions, means, yerr=deviations, fmt="o", color=teal,
                       markerfacecolor=teal, markeredgecolor="white", markeredgewidth=.7,
                       markersize=7.5, elinewidth=1.9, capsize=5, capthick=1.5, zorder=3)
        for index, record in enumerate(data["ranking_feature_groups"]):
            right.annotate(f"{record['mean_ndcg10']:.4f}\n± {record['sample_std_ndcg10']:.4f}",
                           (index, record["mean_ndcg10"] + record["sample_std_ndcg10"]),
                           xytext=(0, 10), textcoords="offset points", ha="center",
                           color=dark, fontsize=9, linespacing=1.4)
        lower = min([0.0] + [mean - sd for mean, sd in zip(means, deviations)])
        upper = max(mean + sd for mean, sd in zip(means, deviations))
        right.set_ylim(lower, max(.1, math.ceil((upper + .08) * 10) / 10))
        right.set_xlim(-.55, 2.55)
        right.set_xticks(positions, [item["label"] for item in data["ranking_feature_groups"]])
        right.tick_params(axis="x", length=0, pad=9)
        right.set_ylabel("Impression-macro nDCG@10", labelpad=9)
        right.grid(axis="y", color="#e9edf1", linewidth=.7, zorder=0)
        right.set_title("B  Ranking features", loc="left", pad=20)
        right.text(0, 1.035, "Mean ± sample SD across seeds 17, 29, 43", transform=right.transAxes,
                   color="#52606d", fontsize=9)

        notes = [
            f"A: logged-click micro recall on {data['catalog_queries']:,} catalog queries.  "
            f"B: full-candidate ranking on {data['ranking_impressions']:,} impressions.",
            "Left intervals quantify query-sample uncertainty by user resampling; right bars quantify training-seed variation, not 95% CIs.",
            "These tasks and uncertainty measures are distinct. Delayed-feedback replay results do not establish online CTR gains.",
        ]
        for line, note in enumerate(notes):
            fig.text(.07, .14 - line * .038, note, ha="left", fontsize=8.6, color="#52606d")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(outputs["png"], dpi=220,
                        metadata={"Software": f"NewsForge v2; Matplotlib {matplotlib.__version__}"})
            fig.savefig(outputs["svg"], metadata={"Creator": "NewsForge v2 plot.py", "Date": None})
        finally:
            plt.close(fig)

    metadata = {
        "schema_version": 1,
        "inputs": {
            "catalog": {"file_name": catalog_path.name, "sha256": sha256_bytes(catalog_bytes)},
            "ranking": {"file_name": ranking_path.name, "sha256": sha256_bytes(ranking_bytes)},
        },
        "plot_source": {"path": "newsforge/v2/plot.py", "sha256": sha256_bytes(Path(__file__).read_bytes())},
        "dependencies": dependency_versions(),
        "rendering": {"backend": "Agg", "size_inches": [12.2, 5.9], "png_dpi": 220,
                      "font": "DejaVu Sans", "svg_hashsalt": style["svg.hashsalt"],
                      "svg_date": None, "svg_fonttype": "path"},
        "plotted_data": data,
        "uncertainty": {
            "left": "paired user-cluster bootstrap 95% CI for full-minus-ablated micro Recall@100; raw fractions multiplied by 100 to percentage points",
            "right": "arithmetic mean and sample standard deviation (ddof=1) across three training-seed nDCG@10 means; not a confidence interval",
        },
        "outputs": {kind: {"file_name": path.name, "sha256": sha256_bytes(path.read_bytes()),
                           "bytes": path.stat().st_size} for kind, path in outputs.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("reports/v2/catalog.json"))
    parser.add_argument("--ranking", type=Path, default=Path("reports/v2/ranking.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--overwrite", action="store_true", help="Regenerate visualization files only; input evidence is never changed")
    args = parser.parse_args()
    result = render_ablation(args.catalog, args.ranking, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "outputs": result["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
