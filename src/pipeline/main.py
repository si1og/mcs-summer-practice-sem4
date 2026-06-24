from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analyze.analyze import print_summary
from src.analyze.graphs import build_graphs
from src.core.jobs import read_jobs
from src.pipeline.cluster import (
    copy_existing_manifest,
    run_remote_generator,
    submit_remote_run,
)


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
        "--analyze-only",
        action="store_true",
        help="Build graphs from existing local manifest/results without touching Slurm.",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation after analysis.",
    )
    parser.add_argument("--count", type=positive_int, help="Override configured count.")
    parser.add_argument("--remote-run-dir", help="Use an existing generated run on VM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = path_from_config(config, "dataset")
    generator_config = config.get("remote_generator", {})
    cluster_config = config.get("cluster", {})
    analyze_config = config.get("analyze", {})

    count = (
        args.count
        if args.count is not None
        else int(generator_config.get("count", 20))
    )
    graphs_dir = Path(analyze_config.get("graphs_dir", "generated/graphs"))
    results_root = Path(cluster_config.get("results_root", "generated/cluster_runs"))

    jobs = read_jobs(dataset_path)
    if not jobs:
        raise SystemExit(f"no jobs read from {dataset_path}")

    manifest_path = Path("generated/slurm_jobs/manifest.csv")
    cluster_results_path: Path | None = None

    if not args.analyze_only and bool(cluster_config.get("enabled", False)):
        host = str(cluster_config.get("host", generator_config.get("host", "mgmt")))
        remote_run_dir = (
            args.remote_run_dir
            or str(cluster_config.get("remote_run_dir") or "").strip()
        )

        if remote_run_dir:
            print(f"stage=manifest host={host} remote_run_dir={remote_run_dir}")
            remote_run = copy_existing_manifest(
                host=host,
                remote_run_dir=remote_run_dir,
                results_root=results_root,
            )
        else:
            generator_host = str(generator_config.get("host", host))
            generator_dir = str(generator_config.get("generator_dir", "~/slurm-generator"))
            generator_command = (
                str(generator_config.get("command", "python3 -m generate.main"))
                + f" --count {count}"
                + f" --sleep-scale {float(generator_config.get('sleep_scale', 0.01))}"
                + f" --time-scale {float(generator_config.get('time_scale', 0.01))}"
                + f" --runtime-mu {float(generator_config.get('runtime_mu', 7.418158))}"
                + f" --runtime-sigma {float(generator_config.get('runtime_sigma', 0.283774))}"
                + f" --error-mu {float(generator_config.get('error_mu', 9.8981))}"
                + f" --error-sigma {float(generator_config.get('error_sigma', 0.0371))}"
                + f" --seed {int(generator_config.get('seed', 50728))}"
                + f" --partition {str(generator_config.get('partition', 'debug'))}"
                + f" --max-nodes {int(generator_config.get('max_nodes', 4))}"
            )
            print(
                f"stage=generate host={generator_host} "
                f"generator_dir={generator_dir} count={count}"
            )
            remote_run = run_remote_generator(
                host=generator_host,
                generator_dir=generator_dir,
                generator_command=generator_command,
                results_root=results_root,
            )

        manifest_path = remote_run.manifest_path
        print(f"generated_manifest={manifest_path}")
        print(f"remote_run_dir={remote_run.remote_run_dir}")

        print(f"stage=cluster host={host}")
        cluster_results_path = submit_remote_run(
            host=host,
            remote_run_dir=remote_run.remote_run_dir,
            local_run_dir=remote_run.local_run_dir,
            manifest_path=manifest_path,
            poll_interval=float(cluster_config.get("poll_interval", 1.0)),
        )
    else:
        print("stage=cluster skipped=true")
        latest_results = sorted(results_root.glob("run_*/results.csv"))
        if latest_results:
            cluster_results_path = latest_results[-1]
            manifest_path = cluster_results_path.parent / "manifest.csv"

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
