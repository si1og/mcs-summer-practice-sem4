from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET = Path("dataset/acct_0921-0923_uid50728_qe7")


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
        return sum(weight * component.pdf(x) for component, weight in zip(self.components, self.weights))

    @property
    def component_count(self) -> int:
        return len(self.components)


@dataclass(frozen=True)
class DatasetSummary:
    jobs: int
    positive_forecast_errors: int
    state_counts: Counter[str]
    req_nodes_counts: Counter[int]
    req_cpus_counts: Counter[int]
    elapsed_median: float
    elapsed_mean: float
    elapsed_min: int
    elapsed_max: int
    error_median: float
    error_mean: float
    error_min: int
    error_max: int


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


def forecast_errors(jobs: list[SourceJob]) -> list[int]:
    return [job.forecast_error_seconds for job in jobs if job.forecast_error_seconds > 0]


def elapsed_times(jobs: list[SourceJob]) -> list[int]:
    return [job.elapsed_seconds for job in jobs if job.elapsed_seconds > 0]


def fit_lognormal(values: list[int]) -> LognormalFit:
    if len(values) < 2:
        raise ValueError("need at least two positive values to fit a lognormal distribution")
    log_values = [math.log(value) for value in values]
    return LognormalFit(
        mu=statistics.mean(log_values),
        sigma=statistics.stdev(log_values),
        count=len(values),
    )


def fit_forecast_error_distribution(jobs: list[SourceJob]) -> LognormalFit:
    return fit_lognormal(forecast_errors(jobs))


def fit_lognormal_mixture(
    values: list[int],
    bins: int = 40,
    min_component_weight: float = 0.05,
) -> LognormalMixture:
    if len(values) < 4:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    log_values = [math.log(value) for value in values]
    min_log = min(log_values)
    max_log = max(log_values)
    if min_log == max_log:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    step = (max_log - min_log) / bins
    counts = [0] * bins
    for value in log_values:
        index = min(bins - 1, int((value - min_log) / step))
        counts[index] += 1

    peaks = [
        index
        for index in range(1, bins - 1)
        if counts[index] > counts[index - 1] and counts[index] >= counts[index + 1]
    ]
    if len(peaks) < 2:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    strongest = sorted(peaks, key=lambda index: counts[index], reverse=True)[:2]
    left_peak, right_peak = sorted(strongest)
    if counts[right_peak] < counts[left_peak] * 0.2:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    valley = min(range(left_peak + 1, right_peak), key=lambda index: counts[index], default=None)
    if valley is None:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    split_log = min_log + (valley + 1) * step
    left = [value for value in values if math.log(value) <= split_log]
    right = [value for value in values if math.log(value) > split_log]
    if len(left) < 2 or len(right) < 2:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    total = len(values)
    if len(left) / total < min_component_weight or len(right) / total < min_component_weight:
        fit = fit_lognormal(values)
        return LognormalMixture((fit,), (1.0,))

    return LognormalMixture(
        components=(fit_lognormal(left), fit_lognormal(right)),
        weights=(len(left) / total, len(right) / total),
    )


def fit_elapsed_mixture(jobs: list[SourceJob], bins: int = 40) -> LognormalMixture:
    return fit_lognormal_mixture(elapsed_times(jobs), bins=bins)


def fit_forecast_error_mixture(jobs: list[SourceJob], bins: int = 40) -> LognormalMixture:
    return fit_lognormal_mixture(forecast_errors(jobs), bins=bins)


def summarize_jobs(jobs: list[SourceJob]) -> DatasetSummary:
    errors = forecast_errors(jobs)
    elapsed_values = elapsed_times(jobs)
    return DatasetSummary(
        jobs=len(jobs),
        positive_forecast_errors=len(errors),
        state_counts=Counter(job.state for job in jobs),
        req_nodes_counts=Counter(job.req_nodes for job in jobs),
        req_cpus_counts=Counter(job.req_cpus for job in jobs),
        elapsed_median=statistics.median(elapsed_values),
        elapsed_mean=statistics.mean(elapsed_values),
        elapsed_min=min(elapsed_values),
        elapsed_max=max(elapsed_values),
        error_median=statistics.median(errors),
        error_mean=statistics.mean(errors),
        error_min=min(errors),
        error_max=max(errors),
    )


def print_summary(jobs: list[SourceJob]) -> None:
    summary = summarize_jobs(jobs)
    elapsed_mixture = fit_elapsed_mixture(jobs)

    print(f"jobs={summary.jobs}")
    print("task_runtime_distribution=ElapsedRaw")
    for index, component in enumerate(elapsed_mixture.components, start=1):
        print(f"runtime_component_{index}_mu={component.mu:.6f}")
        print(f"runtime_component_{index}_sigma={component.sigma:.6f}")
        print(f"runtime_component_{index}_weight={elapsed_mixture.weights[index - 1]:.6f}")
    print(f"runtime_components={elapsed_mixture.component_count}")
    print(f"median_elapsed_seconds={summary.elapsed_median:.0f}")
    print(f"mean_elapsed_seconds={summary.elapsed_mean:.2f}")
    print(f"min_elapsed_seconds={summary.elapsed_min}")
    print(f"max_elapsed_seconds={summary.elapsed_max}")
    print("states=" + ", ".join(f"{key}:{value}" for key, value in summary.state_counts.items()))
