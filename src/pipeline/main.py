from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analyze.analyze import print_summary, read_jobs
from src.analyze.graphs import build_graphs
from src.generate.generate import GenerateConfig, generate_jobs
from src.generate.run_cluster import run_cluster_jobs


DEFAULT_CONFIG = Path("config/pipeline.json")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def path_from_config(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise SystemExit(f"missing required config key: {key}")
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation, optional Slurm execution, and analysis synchronously."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Generate scripts and graphs without submitting jobs to Slurm.",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation after analysis.",
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        help="Override generate.count from the config for quick test runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = path_from_config(config, "dataset")
    generate_config = config.get("generate", {})
    cluster_config = config.get("cluster", {})
    analyze_config = config.get("analyze", {})

    count = (
        args.count
        if args.count is not None
        else int(generate_config.get("count", 20))
    )
    output_dir = Path(generate_config.get("output_dir", "generated/slurm_jobs"))
    graphs_dir = Path(analyze_config.get("graphs_dir", "generated/graphs"))

    jobs = read_jobs(dataset_path)
    if not jobs:
        raise SystemExit(f"no jobs read from {dataset_path}")

    print(f"stage=generate dataset={dataset_path} count={count}")
    runtime_distribution, manifest_path = generate_jobs(
        source_jobs=jobs,
        config=GenerateConfig(
            output_dir=output_dir,
            count=count,
            sleep_scale=float(generate_config.get("sleep_scale", 0.01)),
            time_scale=float(generate_config.get("time_scale", 0.01)),
            seed=int(generate_config.get("seed", 50728)),
            partition=str(generate_config.get("partition", "debug")),
            max_nodes=int(generate_config.get("max_nodes", 4)),
        ),
    )
    print(f"generated_manifest={manifest_path}")
    for index, component in enumerate(runtime_distribution.components, start=1):
        print(f"runtime_component_{index}_mu={component.mu:.6f}")
        print(f"runtime_component_{index}_sigma={component.sigma:.6f}")
        print(
            f"runtime_component_{index}_weight="
            f"{runtime_distribution.weights[index - 1]:.6f}"
        )

    cluster_results_path: Path | None = None
    should_run_cluster = (
        bool(cluster_config.get("enabled", False)) and not args.skip_cluster
    )
    if should_run_cluster:
        print(f"stage=cluster host={cluster_config.get('host', 'mgmt')}")
        cluster_results_path = run_cluster_jobs(
            host=str(cluster_config.get("host", "mgmt")),
            local_dir=output_dir,
            results_root=Path(
                cluster_config.get("results_root", "generated/cluster_runs")
            ),
            remote_root=Path(
                cluster_config.get("remote_root", "generated/cluster_runs")
            ),
            poll_interval=float(cluster_config.get("poll_interval", 1.0)),
        )
    else:
        print("stage=cluster skipped=true")

    print("stage=analyze")
    print_summary(jobs)
    if args.no_graphs:
        return

    graph_paths = build_graphs(
        jobs=jobs,
        output_dir=graphs_dir,
        manifest_path=manifest_path,
        log_path=None,
        cluster_results_path=cluster_results_path,
    )
    for path in graph_paths:
        print(f"graph={path}")


if __name__ == "__main__":
    main()
