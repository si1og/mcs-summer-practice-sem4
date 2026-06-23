from __future__ import annotations

from src.core.jobs import (
    DEFAULT_DATASET,
    SourceJob,
    fit_elapsed_mixture,
    read_jobs,
    summarize_jobs,
)


def print_summary(jobs: list[SourceJob]) -> None:
    summary = summarize_jobs(jobs)
    elapsed_mixture = fit_elapsed_mixture(jobs)

    print(f"jobs={summary.jobs}")
    print("task_runtime_distribution=ElapsedRaw")
    for index, component in enumerate(elapsed_mixture.components, start=1):
        print(f"runtime_component_{index}_mu={component.mu:.6f}")
        print(f"runtime_component_{index}_sigma={component.sigma:.6f}")
        print(
            f"runtime_component_{index}_weight="
            f"{elapsed_mixture.weights[index - 1]:.6f}"
        )
    print(f"runtime_components={elapsed_mixture.component_count}")
    print(f"median_elapsed_seconds={summary.elapsed_median:.0f}")
    print(f"mean_elapsed_seconds={summary.elapsed_mean:.2f}")
    print(f"min_elapsed_seconds={summary.elapsed_min}")
    print(f"max_elapsed_seconds={summary.elapsed_max}")
    print(
        "states="
        + ", ".join(f"{key}:{value}" for key, value in summary.state_counts.items())
    )
