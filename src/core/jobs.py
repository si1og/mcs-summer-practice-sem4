from __future__ import annotations

import csv
import math
import os
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET = Path("dataset/acct_0921-0923_uid50728_qe7")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


@dataclass(frozen=True)
class SourceJob:
    job_id: str
    uid: str
    job_name: str
    partition: str
    req_nodes: int
    req_cpus: int
    alloc_nodes: int
    alloc_cpus: int
    elapsed_seconds: int
    timelimit_minutes: int
    priority: int
    state: str

    @property
    def forecast_seconds(self) -> int:
        return self.timelimit_minutes * 60

    @property
    def forecast_error_seconds(self) -> int:
        return self.forecast_seconds - self.elapsed_seconds


@dataclass(frozen=True)
class LognormalFit:
    mu: float
    sigma: float
    count: int

    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        denominator = x * self.sigma * math.sqrt(2.0 * math.pi)
        exponent = -((math.log(x) - self.mu) ** 2) / (2.0 * self.sigma**2)
        return math.exp(exponent) / denominator


@dataclass(frozen=True)
class LognormalMixture:
    components: tuple[LognormalFit, ...]
    weights: tuple[float, ...]

    def pdf(self, x: float) -> float:
        return sum(
            weight * component.pdf(x)
            for component, weight in zip(self.components, self.weights)
        )


@dataclass(frozen=True)
class DatasetSummary:
    jobs: int
    state_counts: Counter[str]
    req_nodes_counts: Counter[int]
    req_cpus_counts: Counter[int]
    elapsed_median: float
    elapsed_mean: float
    elapsed_min: int
    elapsed_max: int


def read_jobs(path: Path = DEFAULT_DATASET) -> list[SourceJob]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file, delimiter="|")
        jobs: list[SourceJob] = []
        for row in reader:
            if not row.get("JobIDRaw"):
                continue
            jobs.append(
                SourceJob(
                    job_id=row["JobIDRaw"],
                    uid=row["UID"],
                    job_name=row["JobName"],
                    partition=row["Partition"],
                    req_nodes=int(row["ReqNodes"]),
                    req_cpus=int(row["ReqCPUS"]),
                    alloc_nodes=int(row["AllocNodes"]),
                    alloc_cpus=int(row["AllocCPUS"]),
                    elapsed_seconds=int(row["ElapsedRaw"]),
                    timelimit_minutes=int(row["TimelimitRaw"]),
                    priority=int(row["Priority"]),
                    state=row["State"],
                )
            )
    return jobs


def elapsed_times(jobs: list[SourceJob]) -> list[int]:
    return [job.elapsed_seconds for job in jobs if job.elapsed_seconds > 0]


def completed_elapsed_times(jobs: list[SourceJob]) -> list[int]:
    return [
        job.elapsed_seconds
        for job in jobs
        if job.state == "COMPLETED" and job.elapsed_seconds > 0
    ]


def forecast_errors(jobs: list[SourceJob]) -> list[int]:
    return [job.forecast_error_seconds for job in jobs if job.forecast_error_seconds > 0]


def completed_forecast_errors(jobs: list[SourceJob]) -> list[int]:
    return [
        job.forecast_error_seconds
        for job in jobs
        if job.state == "COMPLETED" and job.forecast_error_seconds > 0
    ]


def fit_lognormal(values: list[int] | list[float]) -> LognormalFit:
    if len(values) < 2:
        raise ValueError(
            "need at least two positive values to fit a lognormal distribution"
        )
    log_values = [math.log(value) for value in values]
    return LognormalFit(
        mu=statistics.mean(log_values),
        sigma=statistics.stdev(log_values),
        count=len(values),
    )


def trim_tails(
    values: list[int] | list[float], lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> list[int] | list[float]:
    if len(values) < 10:
        return values
    sorted_values = sorted(values)
    lower_index = int(len(sorted_values) * lower_quantile)
    upper_index = int(len(sorted_values) * upper_quantile)
    trimmed = sorted_values[lower_index:upper_index]
    return trimmed or values


def fit_lognormal_mixture(
    values: list[int] | list[float],
    component_count: int = 2,
    trim_outliers: bool = True,
) -> LognormalMixture:
    fit_values = trim_tails(values) if trim_outliers else values
    if len(fit_values) < component_count * 2:
        fit = fit_lognormal(fit_values)
        return LognormalMixture((fit,), (1.0,))

    try:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"joblib\.externals\.loky\.backend\.context",
            )
            warnings.filterwarnings(
                "ignore", message="Could not find the number of physical cores.*"
            )
            import numpy as np
            from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError as error:
        raise SystemExit(
            "scikit-learn is required for lognormal mixture fitting. "
            "Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from error

    log_values = np.array([[math.log(value)] for value in fit_values if value > 0])
    model = GaussianMixture(
        n_components=component_count,
        covariance_type="full",
        n_init=1,
        random_state=50728,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        model.fit(log_values)

    components: list[LognormalFit] = []
    weights: list[float] = []
    for weight, mean, covariance in zip(
        model.weights_, model.means_.ravel(), model.covariances_.reshape(-1)
    ):
        components.append(
            LognormalFit(mu=float(mean), sigma=math.sqrt(float(covariance)), count=0)
        )
        weights.append(float(weight))

    ordered = sorted(zip(components, weights), key=lambda item: item[0].mu)
    return LognormalMixture(
        components=tuple(component for component, _ in ordered),
        weights=tuple(weight for _, weight in ordered),
    )


def fit_elapsed_lognormal(jobs: list[SourceJob]) -> LognormalFit:
    return fit_lognormal(elapsed_times(jobs))


def fit_elapsed_lognormal_mixture(jobs: list[SourceJob]) -> LognormalMixture:
    return fit_lognormal_mixture(completed_elapsed_times(jobs), component_count=2)


def fit_forecast_error_lognormal(jobs: list[SourceJob]) -> LognormalFit:
    return fit_lognormal(forecast_errors(jobs))


def fit_forecast_error_lognormal_mixture(jobs: list[SourceJob]) -> LognormalMixture:
    return fit_lognormal_mixture(completed_forecast_errors(jobs), component_count=2)


def summarize_jobs(jobs: list[SourceJob]) -> DatasetSummary:
    elapsed_values = elapsed_times(jobs)
    return DatasetSummary(
        jobs=len(jobs),
        state_counts=Counter(job.state for job in jobs),
        req_nodes_counts=Counter(job.req_nodes for job in jobs),
        req_cpus_counts=Counter(job.req_cpus for job in jobs),
        elapsed_median=statistics.median(elapsed_values),
        elapsed_mean=statistics.mean(elapsed_values),
        elapsed_min=min(elapsed_values),
        elapsed_max=max(elapsed_values),
    )
