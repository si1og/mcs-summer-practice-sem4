from __future__ import annotations

from src.core.jobs import (
    DEFAULT_DATASET,
    SourceJob,
    fit_elapsed_lognormal,
    fit_forecast_error_lognormal,
    read_jobs,
    summarize_jobs,
)


def print_summary(jobs: list[SourceJob]) -> None:
    summary = summarize_jobs(jobs)
    elapsed_fit = fit_elapsed_lognormal(jobs)
    error_fit = fit_forecast_error_lognormal(jobs)

    print(f"jobs={summary.jobs}")
    print("task_runtime_distribution=ElapsedRaw")
    print("runtime_distribution=lognormal")
    print(f"runtime_mu={elapsed_fit.mu:.6f}")
    print(f"runtime_sigma={elapsed_fit.sigma:.6f}")
    print("forecast_error_distribution=TimelimitRaw * 60 - ElapsedRaw")
    print(f"error_mu={error_fit.mu:.6f}")
    print(f"error_sigma={error_fit.sigma:.6f}")
    print(f"median_elapsed_seconds={summary.elapsed_median:.0f}")
    print(f"mean_elapsed_seconds={summary.elapsed_mean:.2f}")
    print(f"min_elapsed_seconds={summary.elapsed_min}")
    print(f"max_elapsed_seconds={summary.elapsed_max}")
    print(
        "states="
        + ", ".join(f"{key}:{value}" for key, value in summary.state_counts.items())
    )
