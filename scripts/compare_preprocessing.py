"""Compare two trusted local processed MIND snapshots without fitting or inference.

Inputs and output are explicit. Article/category identities must match before
rowwise comparison. Reports contain aggregate numbers and hashes, never input
paths, article identities, vocabulary terms or titles. Current analysis package
versions describe this comparison process, not either historical training run.
Only deserialize joblib files from trusted local preprocessing artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np


FILES = {
    "feature_array": "news_features.npy", "article_index": "news_ids.json",
    "category_dictionary": "category_vocabulary.json", "manifest": "manifest.json",
    "tfidf_state": "tfidf.joblib", "svd_state": "svd.joblib",
}
OPTIONAL_FILES = {"training_visibility": "news_train_observed.npy", "article_first_seen": "news_first_seen.npy"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_difference(old, new, rows=4096):
    if old.shape != new.shape:
        return {"shape_equal": False, "old_shape": list(old.shape), "new_shape": list(new.shape)}
    total, maximum, changed, count = 0., 0., 0, 0
    for start in range(0, len(old), min(rows, 4096)):
        left = np.asarray(old[start:start + min(rows, 4096)], dtype=np.float64)
        right = np.asarray(new[start:start + min(rows, 4096)], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("non-finite numerical artifact")
        difference = np.abs(left - right)
        total += float(difference.sum())
        maximum = max(maximum, float(difference.max(initial=0)))
        changed += int(np.count_nonzero(difference))
        count += difference.size
    return {"shape_equal": True, "shape": list(old.shape), "old_dtype": str(old.dtype),
            "new_dtype": str(new.dtype), "exact_equal": changed == 0,
            "different_elements": changed, "elements": count,
            "max_absolute_difference": maximum, "mean_absolute_difference": total / count if count else 0.}


def row_geometry(old, new, text_dimensions):
    count, paired, changed, zero_old, zero_new = len(old) - 1, 0, 0, 0, 0
    sums = dict.fromkeys(("old_norm", "new_norm", "norm_absolute_difference", "cosine_distance", "l2_difference"), 0.)
    maxima = dict.fromkeys(("norm_absolute_difference", "cosine_distance", "l2_difference", "old_unit_norm_error", "new_unit_norm_error"), 0.)
    minimum_cosine = None
    for start in range(1, len(old), 4096):
        left = np.asarray(old[start:start + 4096, :text_dimensions], dtype=np.float64)
        right = np.asarray(new[start:start + 4096, :text_dimensions], dtype=np.float64)
        nl, nr = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
        mask = (nl > 0) & (nr > 0)
        paired += int(mask.sum())
        zero_old += int(np.count_nonzero(nl == 0)); zero_new += int(np.count_nonzero(nr == 0))
        changed += int(np.any(left != right, axis=1).sum())
        cosine = np.clip(np.sum(left[mask] * right[mask], axis=1) / (nl[mask] * nr[mask]), -1, 1)
        values = {"old_norm": nl, "new_norm": nr, "norm_absolute_difference": np.abs(nl - nr),
                  "cosine_distance": 1 - cosine, "l2_difference": np.linalg.norm(left - right, axis=1)}
        for name, value in values.items():
            sums[name] += float(value.sum())
        values.update(old_unit_norm_error=np.abs(nl[nl > 0] - 1), new_unit_norm_error=np.abs(nr[nr > 0] - 1))
        for name in maxima:
            maxima[name] = max(maxima[name], float(values[name].max(initial=0)))
        if len(cosine):
            current = float(cosine.min())
            minimum_cosine = current if minimum_cosine is None else min(minimum_cosine, current)
    means = {name: value / (paired if name == "cosine_distance" else count)
             if (paired if name == "cosine_distance" else count) else None for name, value in sums.items()}
    return {"nonpadding_rows": count, "nonzero_paired_rows": paired, "text_rows_with_any_difference": changed,
            "old_zero_text_rows": zero_old, "new_zero_text_rows": zero_new,
            "means": means, "maxima": maxima, "minimum_cosine_similarity": minimum_cosine}


def compare(old_processed, new_processed, output):
    old_processed, new_processed, output = Path(old_processed), Path(new_processed), Path(output)
    if output.exists():
        raise FileExistsError("comparison output exists; choose a new output")
    started = time.perf_counter()
    selected = dict(FILES)
    for name, filename in OPTIONAL_FILES.items():
        present = [(root / filename).exists() for root in (old_processed, new_processed)]
        if present[0] != present[1]:
            raise ValueError("processed snapshots have different optional artifacts")
        if present[0]:
            selected[name] = filename
    bindings = {name: {"old_sha256": sha256(old_processed / filename), "new_sha256": sha256(new_processed / filename)}
                for name, filename in selected.items()}
    mappings = {}
    for name in ("article_index", "category_dictionary"):
        left, right = [read_json(root / FILES[name]) for root in (old_processed, new_processed)]
        if not isinstance(left, list) or not isinstance(right, list) or left != right:
            raise ValueError("article or category mapping differs; rowwise comparison refused")
        mappings[name] = {"equal": True, "entries": len(left)}
    manifest_old, manifest_new = [read_json(root / FILES["manifest"]) for root in (old_processed, new_processed)]
    text_dimensions = manifest_old["features"]["text_dimensions"]
    left, right = [np.load(root / FILES["feature_array"], mmap_mode="r", allow_pickle=False) for root in (old_processed, new_processed)]
    if (left.ndim != 2 or left.shape != right.shape or len(left) != mappings["article_index"]["entries"]
            or type(text_dimensions) is not int or text_dimensions <= 0
            or text_dimensions != manifest_new["features"]["text_dimensions"]
            or left.shape[1] - text_dimensions != mappings["category_dictionary"]["entries"]):
        raise ValueError("feature layout disagrees with its aligned mappings")
    numerical = {"all_features": array_difference(left, right),
                 "text": array_difference(left[:, :text_dimensions], right[:, :text_dimensions]),
                 "category": array_difference(left[:, text_dimensions:], right[:, text_dimensions:]),
                 "row_geometry": row_geometry(left, right, text_dimensions),
                 "padding": {"old_all_zero": bool(np.all(left[0] == 0)), "new_all_zero": bool(np.all(right[0] == 0))}}
    del left, right
    vectorizer_old, vectorizer_new = [joblib.load(root / FILES["tfidf_state"]) for root in (old_processed, new_processed)]
    tfidf = {"parameters_equal": vectorizer_old.get_params() == vectorizer_new.get_params(),
             "vocabulary_equal": vectorizer_old.vocabulary_ == vectorizer_new.vocabulary_,
             "old_vocabulary_size": len(vectorizer_old.vocabulary_), "new_vocabulary_size": len(vectorizer_new.vocabulary_),
             "idf": array_difference(vectorizer_old.idf_, vectorizer_new.idf_)}
    del vectorizer_old, vectorizer_new
    svd_old, svd_new = [joblib.load(root / FILES["svd_state"]) for root in (old_processed, new_processed)]
    svd = {"parameters_equal": svd_old.get_params() == svd_new.get_params()}
    for name in ("components_", "singular_values_", "explained_variance_", "explained_variance_ratio_"):
        svd[name.rstrip("_")] = array_difference(getattr(svd_old, name), getattr(svd_new, name), rows=16)
    del svd_old, svd_new
    for name, filename in selected.items():
        if sha256(old_processed / filename) != bindings[name]["old_sha256"] or sha256(new_processed / filename) != bindings[name]["new_sha256"]:
            raise ValueError("input artifact changed during read-only comparison")
    report = {"schema_version": 1, "status": "comparison_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Numerical comparison only; no historical-environment inference or downstream causal attribution.",
              "script_sha256": sha256(Path(__file__)), "input_sha256": bindings, "mappings": mappings,
              "max_feature_rows_per_chunk": 4096, "svd_component_rows_per_chunk": 16,
              "statistics_dtype": "float64", "text_dimensions": text_dimensions,
              "numerical_comparison": numerical, "tfidf": tfidf, "svd": svd,
              "inputs_unchanged_after_comparison": True, "fitting_or_transform_performed": False,
              "analysis_environment_only": {"historical_run_environment": False, "python": platform.python_version(),
                                            "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}},
              "elapsed_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-processed", type=Path, required=True)
    parser.add_argument("--new-processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old_processed, args.new_processed, args.output)
    print(json.dumps({"status": result["status"], "text": result["numerical_comparison"]["text"],
                      "category": result["numerical_comparison"]["category"], "script_sha256": result["script_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
