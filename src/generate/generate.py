from __future__ import annotations

import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.analyze.analyze import (
    LognormalMixture,
    SourceJob,
    fit_elapsed_mixture,
    fit_forecast_error_mixture,
)
from src.generate.template import render_batch_script


DEFAULT_OUTPUT_DIR = Path("generated/slurm_jobs")
DEFAULT_PARTITION = "debug"
CLUSTER_MAX_NODES = 4


@dataclass(frozen=True)
class GenerateConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    count: int = 20
    sleep_scale: float = 0.01
    time_scale: float = 0.01
    seed: int = 50728
    partition: str = DEFAULT_PARTITION
    max_nodes: int = CLUSTER_MAX_NODES


@dataclass(frozen=True)
class GeneratedJob:
    script: str
    source_job_id: str
    source_state: str
    source_elapsed_seconds: int
    sampled_elapsed_seconds: int
    sampled_error_seconds: int
    generated_forecast_seconds: int
    scaled_sleep_seconds: int
    slurm_time: str
    nodes: int
    ntasks: int

    def as_row(self) -> dict[str, str | int]:
        return {
            "script": self.script,
            "source_job_id": self.source_job_id,
            "source_state": self.source_state,
            "source_elapsed_seconds": self.source_elapsed_seconds,
            "sampled_elapsed_seconds": self.sampled_elapsed_seconds,
            "sampled_error_seconds": self.sampled_error_seconds,
            "generated_forecast_seconds": self.generated_forecast_seconds,
            "scaled_sleep_seconds": self.scaled_sleep_seconds,
            "slurm_time": self.slurm_time,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
        }


def format_slurm_time(total_seconds: int) -> str:
    total_seconds = max(60, int(math.ceil(total_seconds / 60.0) * 60))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sample_lognormal_value(rng: random.Random, distribution: LognormalMixture) -> int:
    roll = rng.random()
    cumulative = 0.0
    component = distribution.components[-1]
    for candidate, weight in zip(distribution.components, distribution.weights):
        cumulative += weight
        if roll <= cumulative:
            component = candidate
            break
    return max(1, int(round(rng.lognormvariate(component.mu, component.sigma))))


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def generate_jobs(source_jobs: list[SourceJob], config: GenerateConfig) -> tuple[LognormalMixture, Path]:
    rng = random.Random(config.seed)
    runtime_distribution = fit_elapsed_mixture(source_jobs)
    error_distribution = fit_forecast_error_mixture(source_jobs)

    clean_output_dir(config.output_dir)
    logs_dir = config.output_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.csv"
    log_file = logs_dir / "generated-jobs.log"

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
                "slurm_time",
                "nodes",
                "ntasks",
            ],
        )
        writer.writeheader()

        for index in range(1, config.count + 1):
            source = rng.choice(source_jobs)
            sampled_elapsed_seconds = sample_lognormal_value(rng, runtime_distribution)
            sampled_error_seconds = sample_lognormal_value(rng, error_distribution)
            generated_forecast_seconds = sampled_elapsed_seconds + sampled_error_seconds
            scaled_sleep_seconds = max(1, int(round(sampled_elapsed_seconds * config.sleep_scale)))
            scaled_forecast_seconds = max(60, int(round(generated_forecast_seconds * config.time_scale)))

            nodes = min(max(1, source.req_nodes), config.max_nodes)
            ntasks = nodes
            slurm_time = format_slurm_time(scaled_forecast_seconds)
            script_name = f"job_{index:04d}_{source.job_id}.slurm"
            script_path = config.output_dir / script_name

            script_path.write_text(
                render_batch_script(
                    index=index,
                    source=source,
                    partition=config.partition,
                    nodes=nodes,
                    ntasks=ntasks,
                    slurm_time=slurm_time,
                    sleep_seconds=scaled_sleep_seconds,
                    log_file=log_file,
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)

            generated_job = GeneratedJob(
                script=script_name,
                source_job_id=source.job_id,
                source_state=source.state,
                source_elapsed_seconds=source.elapsed_seconds,
                sampled_elapsed_seconds=sampled_elapsed_seconds,
                sampled_error_seconds=sampled_error_seconds,
                generated_forecast_seconds=generated_forecast_seconds,
                scaled_sleep_seconds=scaled_sleep_seconds,
                slurm_time=slurm_time,
                nodes=nodes,
                ntasks=ntasks,
            )
            writer.writerow(generated_job.as_row())

    return runtime_distribution, manifest_path
