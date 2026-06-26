from __future__ import annotations

from src.core.jobs import (
    DEFAULT_DATASET,
    SourceJob,
    fit_elapsed_normal_mixture,
    fit_forecast_error_lognormal_mixture,
    read_jobs,
    summarize_jobs,
)


def print_summary(jobs: list[SourceJob]) -> None:
    summary = summarize_jobs(jobs)
    elapsed_mixture = fit_elapsed_normal_mixture(jobs)
    error_mixture = fit_forecast_error_lognormal_mixture(jobs)

    print(f"jobs={summary.jobs}")
    print("task_runtime_distribution=ElapsedRaw")
    print("runtime_distribution=normal_mixture")
    for index, (component, weight) in enumerate(
        zip(elapsed_mixture.components, elapsed_mixture.weights), start=1
    ):
        print(f"runtime_component_{index}_mean={component.mean:.6f}")
        print(f"runtime_component_{index}_sigma={component.sigma:.6f}")
        print(f"runtime_component_{index}_weight={weight:.6f}")
    print("forecast_error_distribution=TimelimitRaw * 60 - ElapsedRaw")
    print("forecast_error_model=lognormal_mixture")
    for index, (component, weight) in enumerate(
        zip(error_mixture.components, error_mixture.weights), start=1
    ):
        print(f"error_component_{index}_mu={component.mu:.6f}")
        print(f"error_component_{index}_sigma={component.sigma:.6f}")
        print(f"error_component_{index}_weight={weight:.6f}")
    print(f"median_elapsed_seconds={summary.elapsed_median:.0f}")
    print(f"mean_elapsed_seconds={summary.elapsed_mean:.2f}")
    print(f"min_elapsed_seconds={summary.elapsed_min}")
    print(f"max_elapsed_seconds={summary.elapsed_max}")
    print(
        "states="
        + ", ".join(f"{key}:{value}" for key, value in summary.state_counts.items())
    )
