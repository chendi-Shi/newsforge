"""Run each immutable target-adaptation stage explicitly."""
import argparse
from pathlib import Path
from . import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'fit', 'tune', 'confirm', 'evaluate'))
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reports', type=Path)
    args = parser.parse_args()
    if args.phase == 'evaluate':
        if args.reports is None:
            parser.error('--reports is required for evaluate')
        experiment.evaluate(args.repo, args.base, args.run, args.reports)
    else:
        getattr(experiment, args.phase)(args.repo, args.base, args.run)


if __name__ == '__main__':
    main()
