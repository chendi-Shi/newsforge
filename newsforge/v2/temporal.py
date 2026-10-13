"""Bounded-memory hourly replay with a declared one-hour feedback delay.

Only NumPy and the standard library are used. Existing v1 arrays are read-only.
An event in [H,H+3600) is aggregated at H+3600; snapshot T has cutoff T-3600
and includes source timestamps strictly below that cutoff. This is an hourly
EMA approximation, not exact decay at individual event or click-arrival times.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
from numpy.lib.format import open_memmap


HOUR = 3600
MAX_CHUNK_CANDIDATES = 200_000
POLICY = {
    "version": "hour_end_ema_strict_delay_v1",
    "bucket_seconds": HOUR,
    "feedback_delay_seconds": HOUR,
    "half_life_seconds": 12 * HOUR,
    "source_rule": "source_timestamp < snapshot_start - feedback_delay_seconds",
    "catalogue_rule": "news_id > 0 and first_seen < cutoff",
    "ema_clock": "cutoff; aggregate each event hour at its end, then decay between hour ends",
    "ema_recurrence": "state(C) = 2**(-1/12) * state(C-3600) + histogram[C-3600,C)",
    "availability_semantics": "first observed in request history/candidates, not publication time",
    "feedback_semantics": "hypothetical fixed delay; actual click-arrival timestamps are unavailable",
    "pad_index": 0,
}
SOURCE_FILES = (
    "manifest.json", "all_timestamps.npy", "all_offsets.npy", "all_candidates.npy",
    "all_labels.npy", "news_first_seen.npy", "news_train_clicks.npy", "news_train_exposures.npy",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _close(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _memory_sample():
    """Read our process RSS and system availability without optional packages."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        status, counters = MemoryStatus(), Counters()
        status.length, counters.cb = ctypes.sizeof(status), ctypes.sizeof(counters)
        available = int(status.avail_phys) if kernel.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        rss = int(counters.rss) if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb) else None
        return rss, available
    if Path("/proc/self/statm").exists():
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE")
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return rss, int(memory["MemAvailable"].split()[0]) * 1024
    return None, None


class _MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            self.samples.append(_memory_sample())
        except (OSError, ValueError, KeyError):
            self.samples.append((None, None))

    def _run(self):
        while not self.stop.wait(.1):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self._sample()

    def report(self):
        rss = [value[0] for value in self.samples if value[0] is not None]
        available = [value[1] for value in self.samples if value[1] is not None]
        return {"sampling_interval_ms": 100, "samples": len(self.samples),
                "initial_rss_bytes": self.samples[0][0],
                "max_observed_rss_bytes": max(rss) if rss else None,
                "minimum_system_available_bytes": min(available) if available else None,
                "note": "RSS includes mapped pages; sampling can miss peaks; shared desktop load is uncontrolled."}


def _array_record(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {"sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
                "shape": list(array.shape), "dtype": str(array.dtype)}
    finally:
        _close(array)


def build_temporal(data="data/processed", output="data/temporal_v2", *, chunk_candidates=MAX_CHUNK_CANDIDATES):
    """Build a new immutable artifact directory; never overwrite prior evidence."""
    if not isinstance(chunk_candidates, int) or isinstance(chunk_candidates, bool) or not 1 <= chunk_candidates <= MAX_CHUNK_CANDIDATES:
        raise ValueError("chunk_candidates must be an integer between 1 and 200000")
    data, output = Path(data), Path(output)
    if data.resolve() == output.resolve() or output.exists():
        raise ValueError("output must be a new directory distinct from processed data")
    source_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    started = time.perf_counter()
    mapped = []
    source_stats = {name: (data / name).stat() for name in SOURCE_FILES}
    with _MemoryMonitor() as monitor:
        sources = {name: {"sha256": sha256_file(data / name), "bytes": source_stats[name].st_size}
                   for name in SOURCE_FILES}
        try:
            def read(name):
                value = np.load(data / name, mmap_mode="r", allow_pickle=False)
                mapped.append(value)
                return value

            times, offsets = read("all_timestamps.npy"), read("all_offsets.npy")
            candidates, labels = read("all_candidates.npy"), read("all_labels.npy")
            first_seen = read("news_first_seen.npy")
            static_clicks, static_exposures = read("news_train_clicks.npy"), read("news_train_exposures.npy")
            n_rows, n_news = len(times), len(first_seen)
            if (not n_rows or times.ndim != 1 or not np.issubdtype(times.dtype, np.integer)
                    or offsets.shape != (n_rows + 1,) or offsets[0] != 0
                    or not np.issubdtype(offsets.dtype, np.integer)
                    or candidates.ndim != 1 or labels.shape != candidates.shape
                    or not np.issubdtype(candidates.dtype, np.integer)
                    or offsets[-1] != len(candidates)
                    or static_clicks.shape != (n_news,) or static_exposures.shape != (n_news,)):
                raise ValueError("invalid source array contract")
            low, high = np.iinfo(np.int64).max, -1
            for start in range(0, n_rows, 65536):
                values = times[start:start + 65536]
                low, high = min(low, int(values.min())), max(high, int(values.max()))
                if np.any(np.diff(offsets[start:min(start + 65536, n_rows) + 1]) < 0):
                    raise ValueError("candidate offsets must be nondecreasing")
            if low < 0:
                raise ValueError("nonnegative timestamps are required")
            base = (low // HOUR) * HOUR
            count_hours = (high // HOUR) - (base // HOUR) + 1
            shape = (count_hours, n_news)
            hist_dtype = np.uint32 if len(candidates) <= np.iinfo(np.uint32).max else np.uint64
            histogram_files = ("hour_clicks.npy", "hour_exposures.npy")
            hc, he = [open_memmap(output / name, mode="w+", dtype=hist_dtype, shape=shape)
                      for name in histogram_files]
            hc[:] = 0; he[:] = 0
            positive_count = 0
            try:
                flat_c, flat_e = hc.reshape(-1), he.reshape(-1)
                for start in range(0, len(candidates), chunk_candidates):
                    end = min(start + chunk_candidates, len(candidates))
                    ids, ys = np.asarray(candidates[start:end]), np.asarray(labels[start:end])
                    if np.any((ids <= 0) | (ids >= n_news)) or np.any((ys != 0) & (ys != 1)):
                        raise ValueError("source candidates must be non-PAD catalog IDs with binary labels")
                    rows = np.searchsorted(offsets, np.arange(start, end, dtype=np.int64), side="right") - 1
                    hours = (times[rows] - base) // HOUR
                    keys = hours * n_news + ids
                    keys_e, counts_e = np.unique(keys, return_counts=True)
                    keys_c, counts_c = np.unique(keys[ys == 1], return_counts=True)
                    flat_e[keys_e] += counts_e.astype(hist_dtype)
                    flat_c[keys_c] += counts_c.astype(hist_dtype)
                    positive_count += int((ys == 1).sum())
                    if end == len(candidates) or start // 5_000_000 != end // 5_000_000:
                        print(f"Temporal histogram {end:,}/{len(candidates):,} candidate exposures", flush=True)
                hc.flush(); he.flush()
                exposure_total = sum(int(he[index].sum(dtype=np.uint64)) for index in range(count_hours))
                click_total = sum(int(hc[index].sum(dtype=np.uint64)) for index in range(count_hours))
                if (exposure_total != len(candidates) or click_total != positive_count
                        or np.any(hc[:, 0]) or np.any(he[:, 0])
                        or source_manifest["candidates"] != exposure_total
                        or source_manifest["positives"] != click_total
                        or source_manifest["rows"] != n_rows
                        or source_manifest["news_including_padding"] != n_news):
                    raise ValueError("histogram counts do not match the source manifest")
            finally:
                del flat_c, flat_e
                _close(hc); _close(he)

            snapshot_times = base + np.arange(count_hours, dtype=np.int64) * HOUR
            np.save(output / "snapshot_times.npy", snapshot_times, allow_pickle=False)
            hc, he = [np.load(output / name, mmap_mode="r", allow_pickle=False) for name in histogram_files]
            sc, se = [open_memmap(output / name, mode="w+", dtype=np.float32, shape=shape)
                      for name in ("snapshot_clicks.npy", "snapshot_exposures.npy")]
            state_c, state_e = np.zeros(n_news, dtype=np.float64), np.zeros(n_news, dtype=np.float64)
            decay = 2 ** (-1 / 12)
            try:
                for index in range(count_hours):
                    # At snapshot T, cutoff=T-1h: only event hours ending <=cutoff enter.
                    if index >= 2:
                        state_c *= decay; state_e *= decay
                        state_c += hc[index - 2]; state_e += he[index - 2]
                    if (not np.isfinite(state_c).all() or not np.isfinite(state_e).all()
                            or np.any(state_c < 0) or np.any(state_e < state_c)):
                        raise ValueError("invalid EMA state")
                    state_c[0] = state_e[0] = 0
                    sc[index], se[index] = state_c, state_e
                sc.flush(); se.flush()
            finally:
                _close(hc); _close(he); _close(sc); _close(se)
            if (np.any(static_clicks < 0) or np.any(static_exposures < static_clicks)
                    or static_clicks[0] != 0 or static_exposures[0] != 0
                    or int(static_clicks.sum(dtype=np.uint64)) != source_manifest["splits"]["0"]["positives"]
                    or int(static_exposures.sum(dtype=np.uint64)) != source_manifest["splits"]["0"]["candidates"]):
                raise ValueError("frozen training statistics do not match the source manifest")
            for filename, values in (("frozen_clicks.npy", static_clicks), ("frozen_exposures.npy", static_exposures)):
                np.save(output / filename, np.asarray(values, dtype=np.float32), allow_pickle=False)
            array_names = (*histogram_files, "snapshot_times.npy", "snapshot_clicks.npy", "snapshot_exposures.npy",
                           "frozen_clicks.npy", "frozen_exposures.npy")
            arrays = {name: _array_record(output / name) for name in array_names}
            for name, before in source_stats.items():
                after = (data / name).stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"source changed during temporal construction: {name}")
        finally:
            for value in mapped:
                _close(value)
    manifest = {
        "artifact_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY, "policy_hash": policy_digest(POLICY),
        "builder_source_sha256": sha256_file(__file__), "sources": sources, "arrays": arrays,
        "data_manifest_sha256": sources["manifest.json"]["sha256"],
        "n_news": n_news, "n_snapshots": count_hours,
        "min_as_of": low, "max_as_of": high,
        "snapshot_start_min": int(snapshot_times[0]), "snapshot_start_max": int(snapshot_times[-1]),
        "train_cutoff_timestamp": int(source_manifest["train_cutoff_timestamp"]),
        "internal_evaluation_min_timestamp": int(((int(source_manifest["train_cutoff_timestamp"]) + HOUR + HOUR - 1) // HOUR) * HOUR),
        "chunk_candidates": chunk_candidates, "counts": {
            "source_rows": n_rows, "source_candidate_exposures": exposure_total,
            "source_positive_clicks": click_total, "histogram_counts_match_source": True, "padding_zero": True},
        "validation": {"array_hashes_generated": True, "histogram_totals_recomputed": True,
                       "snapshot_state_checked_finite_nonnegative_clicks_le_exposures": True},
        "memory": monitor.report(), "seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": "v2 retrospective temporal replay; v1 dev was previously inspected; no model fit or metric selection",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class Snapshot:
    index: int
    as_of_start: int
    cutoff: int
    clicks: np.ndarray
    exposures: np.ndarray
    mode: str = "rolling"


class TemporalStore:
    def __init__(self, path="data/temporal_v2", processed_dir="data/processed", *, verify=True):
        self.path, self.processed_dir = Path(path), Path(processed_dir)
        manifest_path = self.path / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.version = sha256_file(manifest_path)
        self.policy_hash = policy_digest(self.manifest["policy"])
        self._mapped = []
        if self.manifest["policy"] != POLICY or self.manifest["policy_hash"] != self.policy_hash:
            raise ValueError("temporal policy mismatch")
        if verify:
            if self.manifest["builder_source_sha256"] != sha256_file(__file__):
                raise ValueError("temporal builder source SHA256 mismatch")
            for name, record in self.manifest["sources"].items():
                if sha256_file(self.processed_dir / name) != record["sha256"]:
                    raise ValueError(f"source SHA256 mismatch: {name}")
            for name, record in self.manifest["arrays"].items():
                if sha256_file(self.path / name) != record["sha256"]:
                    raise ValueError(f"snapshot SHA256 mismatch: {name}")
        try:
            def load(name, source=False):
                value = np.load((self.processed_dir if source else self.path) / name, mmap_mode="r", allow_pickle=False)
                self._mapped.append(value)
                if not source:
                    record = self.manifest["arrays"][name]
                    if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                        raise ValueError(f"snapshot shape/dtype mismatch: {name}")
                return value

            self.snapshot_times = load("snapshot_times.npy")
            self.clicks, self.exposures = load("snapshot_clicks.npy"), load("snapshot_exposures.npy")
            self.frozen_clicks, self.frozen_exposures = load("frozen_clicks.npy"), load("frozen_exposures.npy")
            self.first_seen = load("news_first_seen.npy", source=True)
            n, hours = self.manifest["n_news"], self.manifest["n_snapshots"]
            if (self.clicks.shape != (hours, n) or self.exposures.shape != (hours, n)
                    or self.clicks.dtype != np.float32 or self.exposures.dtype != np.float32
                    or self.first_seen.shape != (n,) or self.frozen_clicks.shape != (n,)
                    or self.frozen_exposures.shape != (n,) or self.snapshot_times.shape != (hours,)
                    or self.snapshot_times.dtype != np.int64
                    or np.any(np.diff(self.snapshot_times) != HOUR)
                    or self.snapshot_times[0] != self.manifest["snapshot_start_min"]
                    or self.snapshot_times[-1] != self.manifest["snapshot_start_max"]):
                raise ValueError("temporal artifact shape/time contract mismatch")
        except Exception:
            self.close()
            raise

    def snapshot(self, as_of, *, mode="rolling"):
        if isinstance(as_of, (bool, np.bool_)) or not isinstance(as_of, (int, np.integer)):
            raise ValueError("as_of must be an integer Unix timestamp")
        if not self.manifest["min_as_of"] <= as_of <= self.manifest["max_as_of"]:
            raise ValueError("as_of is outside the temporal snapshot time range")
        start = (int(as_of) // HOUR) * HOUR
        index = (start - int(self.snapshot_times[0])) // HOUR
        cutoff = start - HOUR
        if mode == "rolling":
            clicks, exposures = self.clicks[index], self.exposures[index]
        elif mode == "frozen":
            if cutoff <= self.manifest["train_cutoff_timestamp"]:
                raise ValueError("frozen full-training statistics are not yet available at this cutoff")
            clicks, exposures = self.frozen_clicks, self.frozen_exposures
        else:
            raise ValueError("mode must be rolling or frozen")
        return Snapshot(int(index), start, cutoff, clicks, exposures, mode)

    def frozen_snapshot(self, as_of):
        return self.snapshot(as_of, mode="frozen")

    def age_hours(self, ids, as_of):
        snapshot = self.snapshot(as_of)
        ids = np.asarray(ids)
        if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= len(self.first_seen))):
            raise ValueError("news IDs must be valid integer catalog indices")
        first = self.first_seen[ids]
        known = (ids > 0) & (first < snapshot.cutoff)
        # Mask unknown max-int timestamps before subtraction to avoid overflow.
        safe_first = np.where(known, first, snapshot.as_of_start)
        return np.maximum((snapshot.as_of_start - safe_first) / HOUR, 0).astype(np.float32)

    def close(self):
        for value in self._mapped:
            _close(value)
        self._mapped.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/temporal_v2"))
    parser.add_argument("--chunk-candidates", type=int, default=MAX_CHUNK_CANDIDATES)
    args = parser.parse_args()
    result = build_temporal(args.data, args.output, chunk_candidates=args.chunk_candidates)
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "n_snapshots": result["n_snapshots"], "memory": result["memory"]}), flush=True)


if __name__ == "__main__":
    main()
