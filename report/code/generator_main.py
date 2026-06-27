from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path
from generate.template import render_batch_script



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


def parse_component(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("component must be center,sigma,weight")
    center = float(parts[0])
    sigma = float(parts[1])
    weight = float(parts[2])
    if sigma <= 0 or weight <= 0:
        raise argparse.ArgumentTypeError("sigma and weight must be positive")
    return center, sigma, weight


def choose_component(rng: random.Random, components: list[tuple[float, float, float]]) -> tuple[float, float]:
    total = sum(weight for _, _, weight in components)
    roll = rng.random() * total
    cumulative = 0.0
    for center, sigma, weight in components:
        cumulative += weight
        if roll <= cumulative:
            return center, sigma
    center, sigma, _ = components[-1]
    return center, sigma


def sample_lognormal_value(rng: random.Random, components: list[tuple[float, float, float]]) -> int:
    mu, sigma = choose_component(rng, components)
    return max(1, int(round(rng.lognormvariate(mu, sigma))))


def sample_normal_value(rng: random.Random, components: list[tuple[float, float, float]]) -> int:
    mean, sigma = choose_component(rng, components)
    return max(1, int(round(rng.normalvariate(mean, sigma))))


def sample_node_count(rng: random.Random) -> int:
    sampled = int(round(rng.normalvariate(2.0, 0.75)))
    return min(4, max(1, sampled))


def format_slurm_time(total_seconds: int) -> str:
    total_seconds = max(60, int(math.ceil(total_seconds / 60.0) * 60))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "slurm_jobs").mkdir(parents=True)


def generate_jobs(
    run_dir: Path,
    count: int,
    sleep_scale: float,
    time_scale: float,
    seed: int,
    partition: str,
    runtime_components: list[tuple[float, float, float]],
    error_components: list[tuple[float, float, float]],
) -> Path:
    clean_run_dir(run_dir)
    rng = random.Random(seed)
    slurm_jobs_dir = run_dir / "slurm_jobs"
    manifest_path = run_dir / "manifest.csv"
    log_file = Path("generated/logs/generated-jobs.log")

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "script",
                "source_job_id",
                "source_state",
                "source_elapsed_seconds",
                "sampled_elapsed_seconds",
                "sampled_error_seconds",
                "generated_forecast_seconds",
                "scaled_sleep_seconds",
                "scaled_forecast_seconds",
                "slurm_time",
                "nodes",
                "ntasks",
            ],
        )
        writer.writeheader()

        for index in range(1, count + 1):
            sampled_elapsed_seconds = sample_normal_value(
                rng, runtime_components
            )
            sampled_error_seconds = sample_lognormal_value(rng, error_components)
            generated_forecast_seconds = sampled_elapsed_seconds + sampled_error_seconds
            scaled_sleep_seconds = max(1, int(round(sampled_elapsed_seconds * sleep_scale)))
            scaled_forecast_seconds = max(
                60,
                int(round(generated_forecast_seconds * time_scale)),
            )
            nodes = sample_node_count(rng)
            ntasks = nodes
            slurm_time = format_slurm_time(scaled_forecast_seconds)
            source_job_id = str(507280000 + index)
            script_name = f"job_{index:04d}_{source_job_id}.slurm"
            script_path = slurm_jobs_dir / script_name

            script_path.write_text(
                render_batch_script(
                    index=index,
                    source_job_id=source_job_id,
                    partition=partition,
                    nodes=nodes,
                    ntasks=ntasks,
                    slurm_time=slurm_time,
                    sleep_seconds=scaled_sleep_seconds,
                    log_file=log_file,
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)

            writer.writerow(
                {
                    "script": script_name,
                    "source_job_id": source_job_id,
                    "source_state": "GENERATED",
                    "source_elapsed_seconds": sampled_elapsed_seconds,
                    "sampled_elapsed_seconds": sampled_elapsed_seconds,
                    "sampled_error_seconds": sampled_error_seconds,
                    "generated_forecast_seconds": generated_forecast_seconds,
                    "scaled_sleep_seconds": scaled_sleep_seconds,
                    "scaled_forecast_seconds": scaled_forecast_seconds,
                    "slurm_time": slurm_time,
                    "nodes": nodes,
                    "ntasks": ntasks,
                }
            )

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Slurm jobs on the mgmt VM.")
    parser.add_argument("--output-root", type=Path, default=Path("generated/cluster_runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=positive_int, default=200)
    parser.add_argument("--sleep-scale", type=positive_float, default=0.01)
    parser.add_argument("--time-scale", type=positive_float, default=0.01)
    parser.add_argument("--seed", type=int, default=50728)
    parser.add_argument("--partition", default="debug")
    parser.add_argument(
        "--runtime-normal-component",
        action="append",
        type=parse_component,
        default=[],
        help="Runtime normal component as mean,sigma,weight.",
    )
    parser.add_argument(
        "--error-component",
        action="append",
        type=parse_component,
        default=[],
        help="Forecast error lognormal component as mu,sigma,weight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (args.output_root / args.run_id).resolve()
    runtime_components = args.runtime_normal_component or [(1701.49, 435.21, 1.0)]
    error_components = args.error_component or [(9.8981, 0.0371, 1.0)]
    generate_jobs(
        run_dir=run_dir,
        count=args.count,
        sleep_scale=args.sleep_scale,
        time_scale=args.time_scale,
        seed=args.seed,
        partition=args.partition,
        runtime_components=runtime_components,
        error_components=error_components,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
