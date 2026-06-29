from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.analyze.analyze import DEFAULT_DATASET, print_summary, read_jobs
from src.analyze.graphs import build_graphs


DEFAULT_GRAPHS_DIR = Path("generated/graphs")


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def log_base(value: str) -> float:
    if value.lower() in {"e", "natural", "ln"}:
        return math.e
    result = positive_float(value)
    if result == 1:
        raise argparse.ArgumentTypeError("log base must not be equal to 1")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze filtered SCC accounting data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--graphs-dir", type=Path, default=DEFAULT_GRAPHS_DIR)
    parser.add_argument("--manifest", type=Path, default=Path("generated/slurm_jobs/manifest.csv"))
    parser.add_argument("--execution-log", type=Path, default=Path("generated/logs/generated-jobs.log"))
    parser.add_argument("--cluster-results", type=Path)
    parser.add_argument(
        "--log-base",
        type=log_base,
        default=math.e,
        help="Logarithm base for log-scale graphs. Use 10 for decimal logarithm or e for natural logarithm.",
    )
    parser.add_argument(
        "--max-group-size",
        type=positive_int,
        default=50,
        help="Maximum group size for smoothed cluster result graphs.",
    )
    parser.add_argument("--no-graphs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = read_jobs(args.dataset)
    if not jobs:
        raise SystemExit(f"no jobs read from {args.dataset}")

    print_summary(jobs)
    if args.no_graphs:
        return

    paths = build_graphs(
        jobs=jobs,
        output_dir=args.graphs_dir,
        manifest_path=args.manifest,
        log_path=args.execution_log,
        cluster_results_path=args.cluster_results,
        max_group_size=args.max_group_size,
        log_base=args.log_base,
    )
    for path in paths:
        print(f"graph={path}")


if __name__ == "__main__":
    main()
