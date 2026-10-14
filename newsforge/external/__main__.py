"""Four explicit, append-only phases for the preregistered external experiment."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquisition", "acquisition-parallel", "prepare", "evaluate", "aggregate"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/external/adressa_v1"))
    parser.add_argument("--archive", type=Path, default=Path("data/external/adressa/raw/one_week.tar.gz"))
    parser.add_argument("--reports", type=Path, default=Path("reports/external/adressa_v1"))
    args = parser.parse_args()
    if args.phase == "acquisition":
        from .acquire import acquire
        acquire(args.repo, args.run_dir, args.archive)
    elif args.phase == "acquisition-parallel":
        from .acquire_parallel import acquire_parallel
        acquire_parallel(args.repo, args.run_dir, args.archive)
    elif args.phase == "prepare":
        from .prepare import prepare
        prepare(args.repo, args.run_dir, args.archive)
    elif args.phase == "evaluate":
        from .evaluate import evaluate
        evaluate(args.repo, args.run_dir)
    else:
        from .aggregate import aggregate
        aggregate(args.repo, args.run_dir, args.reports)


if __name__ == "__main__":
    main()
