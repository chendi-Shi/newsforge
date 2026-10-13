"""Show one real internal-validation news recommendation in the terminal.

Run ``python -m newsforge.demo --launch-server`` from the project directory.
The deterministic request reads no labels. News titles and history stay in
terminal output; an optional verification report contains only aggregates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from urllib.error import URLError
from urllib.parse import urlparse

import numpy as np

from newsforge.benchmark_service import http_json, launched_server, validate_response


def select_request(processed_dir):
    """Choose the first stored split-1 row with nonempty history, without labels."""
    processed_dir = Path(processed_dir)
    news_ids = json.loads((processed_dir / "news_ids.json").read_text(encoding="utf-8"))
    arrays = []
    try:
        for name in ("all_history", "all_timestamps", "all_split"):
            arrays.append(np.load(processed_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False))
        histories, timestamps, splits = arrays
        if histories.ndim != 2 or timestamps.shape != (len(histories),) or splits.shape != timestamps.shape:
            raise ValueError("history/timestamp/split arrays have incompatible shapes")
        for start in range(0, len(splits), 65536):
            for offset in np.flatnonzero(splits[start:start + 65536] == 1):
                row = start + int(offset)
                numbers = histories[row][histories[row] != 0].astype(np.int64)
                if not len(numbers):
                    continue
                if np.any(numbers < 1) or np.any(numbers >= len(news_ids)):
                    raise ValueError("selected history contains an invalid article index")
                timestamp = int(timestamps[row])
                if not 0 <= timestamp < np.iinfo(np.int64).max:
                    raise ValueError("selected request has an unknown timestamp")
                return {"history": [news_ids[int(item)] for item in numbers],
                        "as_of": timestamp, "top_k": 10, "candidate_pool_size": 100}, news_ids
        raise ValueError("no nonempty internal-validation history is available")
    finally:
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def article_metadata(raw_dir, wanted_ids):
    """Read requested titles/categories only; duplicate IDs prefer train, like data.py."""
    wanted = set(wanted_ids)
    metadata = {}
    for split in ("train", "dev"):
        path = Path(raw_dir) / split / "news.tsv"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t")
                if not fields or fields[0] not in wanted or fields[0] in metadata:
                    continue
                if len(fields) < 4:
                    raise ValueError("requested news record lacks title/category fields")
                metadata[fields[0]] = {"category": fields[1], "subcategory": fields[2], "title": fields[3]}
    if len(metadata) != len(wanted):
        raise ValueError(f"raw news.tsv files lack metadata for {len(wanted - metadata.keys())} requested articles")
    return metadata


def _display(value):
    # Do not let source text emit terminal control sequences.
    return " ".join("".join(character if character.isprintable() else " " for character in value).split())


def validate_local_url(base_url):
    parsed = urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("demo requires a plain localhost HTTP base URL")
    return base_url.rstrip("/")


def demonstrate(base_url, processed_dir, raw_dir):
    base_url = validate_local_url(base_url)
    request, news_ids = select_request(processed_dir)
    health = http_json(base_url + "/health")
    if health.get("status") != "ok":
        raise ValueError("local recommendation service is not healthy")
    cutoff = health.get("model_train_cutoff")
    if cutoff is None or request["as_of"] < cutoff:
        raise ValueError("request time predates the serving bundle's known training cutoff")
    response = http_json(base_url + "/recommend", request)
    first_seen = np.load(Path(processed_dir) / "news_first_seen.npy", allow_pickle=False)
    validate_response(response, request, health, {item: i for i, item in enumerate(news_ids)}, first_seen)
    wanted = set(request["history"]) | {item["item_id"] for item in response["items"]}
    metadata = article_metadata(raw_dir, wanted)

    timestamp = datetime.fromtimestamp(request["as_of"], timezone.utc).isoformat()
    print("NewsForge | 本地新闻推荐演示")
    print(f"固定内部验证请求 | 时间坐标 {timestamp} | 历史 {len(request['history'])} 条")
    print("时间按预处理的 UTC 坐标显示；原始 MIND 时间字段未注明时区。")
    print(f"CPU threads: torch={health.get('torch_threads')}, faiss={health.get('faiss_threads')}")
    print(f"Model: {response['model_version']}\nIndex: {response['index_version']}")
    print("\n输入历史（时间顺序；只在本终端展示）")
    for position, item_id in enumerate(request["history"], 1):
        article = metadata[item_id]
        print(f"  {position:02d}. [{_display(article['category'])}/{_display(article['subcategory'])}] "
              f"{_display(article['title'])}")
    print(f"\nTop 10 推荐 | 实获候选 {response['candidate_count']} | 策略 {response['strategy']}")
    for position, item in enumerate(response["items"], 1):
        article = metadata[item["item_id"]]
        rerank = "n/a" if item["rerank_score"] is None else f"{item['rerank_score']:.5f}"
        print(f"  {position:02d}. [{_display(article['category'])}/{_display(article['subcategory'])}] "
              f"{_display(article['title'])}")
        print(f"      retrieval={item['retrieval_score']:.5f}  rerank={rerank}")
    print(f"\n返回 {len(response['items'])}/10；不足 {response['shortfall']}。"
          "已核对版本、分数、时间可用性与已读过滤；分数用于排序，不是校准点击率。")
    # Never persist the selected request, article IDs, titles, or item scores.
    return {"history_count": len(request["history"]), "candidate_count": response["candidate_count"],
            "recommendation_count": len(response["items"]), "shortfall_count": response["shortfall"],
            "model_version": response["model_version"], "index_version": response["index_version"],
            "checks": {"internal_validation_only": True, "nonempty_history": True,
                       "labels_not_read": True, "model_cutoff_respected": True,
                       "response_versions_and_filters_valid": True, "all_titles_found": True}}


def main():
    # Windows pipes otherwise inherit GBK, corrupting Chinese titles in UTF-8
    # terminals and command capture. Only the CLI owns these output streams.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/index_hnsw"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--verification-report", type=Path,
                        help="optional aggregate-only JSON; excludes request/history/news text")
    args = parser.parse_args()
    log_path = None
    try:
        if args.launch_server:
            required = (args.artifact_dir / "retriever.pt", args.artifact_dir / "reranker.pt",
                        args.index_dir / "manifest.json")
            if any(not path.is_file() for path in required):
                raise FileNotFoundError("trained retriever/reranker and persisted index are required; complete training and evaluate_retrieval first")
            log_path = args.artifact_dir.resolve() / f".demo-server-{uuid.uuid4().hex}.log"
            with launched_server(args.processed_dir, args.artifact_dir, args.index_dir,
                                 threads=1, max_overfetch=8192, log_path=log_path) as (url, _):
                verification = demonstrate(url, args.processed_dir, args.raw_dir)
        else:
            verification = demonstrate(args.base_url, args.processed_dir, args.raw_dir)
        if args.verification_report:
            args.verification_report.parent.mkdir(parents=True, exist_ok=True)
            args.verification_report.write_text(json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8")
    except (OSError, ValueError, RuntimeError, URLError) as error:
        parser.exit(2, f"Demo failed: {error}\n")
    finally:
        # Only this invocation's known log file is removed; no directory delete.
        if log_path is not None:
            log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
