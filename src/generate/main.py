from __future__ import annotations

import argparse
from pathlib import Path

from src.analyze.analyze import DEFAULT_DATASET, read_jobs
from src.generate.generate import DEFAULT_OUTPUT_DIR, DEFAULT_PARTITION, CLUSTER_MAX_NODES, GenerateConfig, generate_jobs


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM batch scripts from fitted task-runtime distribution."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--sleep-scale", type=positive_float, default=0.01)
    parser.add_argument("--time-scale", type=positive_float, default=0.01)
    parser.add_argument("--seed", type=int, default=50728)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--max-nodes", type=positive_int, default=CLUSTER_MAX_NODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = read_jobs(args.dataset)
    if not jobs:
        raise SystemExit(f"no jobs read from {args.dataset}")

    runtime_distribution, manifest_path = generate_jobs(
        source_jobs=jobs,
        config=GenerateConfig(
            output_dir=args.output_dir,
            count=args.count,
            sleep_scale=args.sleep_scale,
            time_scale=args.time_scale,
            seed=args.seed,
            partition=args.partition,
            max_nodes=args.max_nodes,
        ),
    )
    print(f"generated={args.count}")
    print(f"output_dir={args.output_dir}")
    print(f"manifest={manifest_path}")
    for index, component in enumerate(runtime_distribution.components, start=1):
        print(f"runtime_component_{index}_mu={component.mu:.6f}")
        print(f"runtime_component_{index}_sigma={component.sigma:.6f}")
        print(f"runtime_component_{index}_weight={runtime_distribution.weights[index - 1]:.6f}")


if __name__ == "__main__":
    main()
