from __future__ import annotations

import csv
import datetime as dt
import math
import os
import re
import shutil
import statistics
from pathlib import Path

from src.core.jobs import (
    LognormalFit,
    LognormalMixture,
    NormalMixture,
    SourceJob,
    elapsed_times,
    fit_elapsed_normal_mixture,
    fit_forecast_error_lognormal_mixture,
    fit_lognormal,
    fit_lognormal_mixture,
    forecast_errors,
    summarize_jobs,
)


def _import_pyplot():
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path("generated/matplotlib_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit(
            "matplotlib is required for graphs. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from error
    return plt


def _histogram(
    values: list[int] | list[float], bins: int
) -> tuple[list[float], list[int], float]:
    lower = min(values)
    upper = max(values)
    width = (upper - lower) / bins if upper != lower else 1.0
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lower) / width))
        counts[index] += 1
    centers = [lower + width * (index + 0.5) for index in range(bins)]
    return centers, counts, width


def _quantile(values: list[float] | list[int], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("cannot calculate quantile for an empty sequence")
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    return float(
        sorted_values[lower] * (upper - index) + sorted_values[upper] * (index - lower)
    )


def _main_x_limits(
    values: list[int] | list[float],
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.995,
) -> tuple[float, float]:
    lower_value = _quantile(values, lower_quantile)
    upper_value = _quantile(values, upper_quantile)
    padding = (upper_value - lower_value) * 0.15
    lower = max(0.0, lower_value - padding)
    upper = upper_value + padding
    if upper <= lower:
        return min(values), max(values)
    return lower, upper


def _padded_x_limits(
    x_limits: tuple[float, float], padding_ratio: float = 0.035
) -> tuple[float, float]:
    lower, upper = x_limits
    padding = (upper - lower) * padding_ratio
    return lower - padding, upper + padding


def _compact_hist_bins(
    values: list[int] | list[float],
    max_bins: int,
    target_bin_width: float = 1.2,
    min_bins: int = 24,
) -> int:
    lower = min(values)
    upper = max(values)
    if upper <= lower:
        return 1
    compact_bins = math.ceil((upper - lower) / target_bin_width)
    return min(max_bins, max(min_bins, compact_bins))


def _plot_pdf(
    ax,
    fit: LognormalFit | LognormalMixture,
    values: list[int] | list[float],
    bins: int,
    label: str = "логнормальная аппроксимация",
    x_limits: tuple[float, float] | None = None,
) -> None:
    if x_limits:
        lower, upper = x_limits
    else:
        lower = max(1, min(values))
        upper = max(values)
    lower = max(1, lower)
    step = (upper - lower) / 300
    xs = [lower + step * index for index in range(301)]
    bin_width = (upper - lower) / bins if upper != lower else 1.0
    ys = [fit.pdf(x) * len(values) * bin_width for x in xs]
    ax.plot(xs, ys, color="#d62728", linewidth=2.0, label=label)


def _plot_normal_mixture_on_values(
    ax,
    mixture: NormalMixture,
    values: list[int] | list[float],
    bins: int,
    label: str = "смесь нормальных распределений",
    x_limits: tuple[float, float] | None = None,
) -> None:
    if x_limits:
        lower, upper = x_limits
    else:
        lower = min(values)
        upper = max(values)
    step = (upper - lower) / 300 if upper != lower else 1.0
    xs = [lower + step * index for index in range(301)]
    bin_width = (upper - lower) / bins if upper != lower else 1.0
    ys = [mixture.pdf(x) * len(values) * bin_width for x in xs]
    ax.plot(xs, ys, color="#d62728", linewidth=2.0, label=label)


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    denominator = sigma * math.sqrt(2.0 * math.pi)
    exponent = -((x - mu) ** 2) / (2.0 * sigma**2)
    return math.exp(exponent) / denominator


def _plot_normal_pdf(
    ax,
    values: list[float],
    mu: float,
    sigma: float,
    bins: int,
    label: str = "нормальная аппроксимация",
    x_limits: tuple[float, float] | None = None,
) -> None:
    if x_limits:
        lower, upper = x_limits
    else:
        lower = min(values)
        upper = max(values)
    step = (upper - lower) / 300 if upper != lower else 1.0
    xs = [lower + step * index for index in range(301)]
    bin_width = (upper - lower) / bins if upper != lower else 1.0
    ys = [_normal_pdf(x, mu, sigma) * len(values) * bin_width for x in xs]
    ax.plot(xs, ys, color="#d62728", linewidth=2.0, label=label)


def _plot_normal_mixture_pdf(
    ax,
    mixture: LognormalMixture,
    values: list[float],
    bins: int,
    x_limits: tuple[float, float] | None = None,
) -> None:
    if x_limits:
        lower, upper = x_limits
    else:
        lower = min(values)
        upper = max(values)
    step = (upper - lower) / 300 if upper != lower else 1.0
    xs = [lower + step * index for index in range(301)]
    bin_width = (upper - lower) / bins if upper != lower else 1.0

    total_ys = [0.0 for _ in xs]
    colors = ["#9467bd", "#2ca02c"]
    for index, (component, weight) in enumerate(
        zip(mixture.components, mixture.weights), start=1
    ):
        ys = [
            weight
            * _normal_pdf(x, component.mu, component.sigma)
            * len(values)
            * bin_width
            for x in xs
        ]
        total_ys = [total + value for total, value in zip(total_ys, ys)]
        ax.plot(
            xs,
            ys,
            color=colors[(index - 1) % len(colors)],
            linewidth=1.4,
            linestyle="--",
            label=f"компонента {index}",
        )

    ax.plot(
        xs,
        total_ys,
        color="#d62728",
        linewidth=2.2,
        label="смесь нормальных распределений",
    )


def _log_values(values: list[float] | list[int], base: float) -> list[float]:
    if base <= 0 or base == 1:
        raise ValueError("log base must be positive and not equal to 1")
    return [math.log(value, base) for value in values]


def _log_axis_label(base: float, expression: str) -> str:
    if math.isclose(base, math.e):
        return f"ln({expression})"
    if float(base).is_integer():
        return f"log{int(base)}({expression})"
    return f"log base {base:g}({expression})"


def _rescale_lognormal_mixture(
    mixture: LognormalMixture, base: float
) -> LognormalMixture:
    if math.isclose(base, math.e):
        return mixture
    scale = math.log(base)
    return LognormalMixture(
        components=tuple(
            LognormalFit(
                mu=component.mu / scale,
                sigma=component.sigma / scale,
                count=component.count,
            )
            for component in mixture.components
        ),
        weights=mixture.weights,
    )


def _mixture_text(mixture: LognormalMixture) -> str:
    return "\n".join(
        f"комп. {index}: mu={component.mu:.4f}, sigma={component.sigma:.4f}, вес={weight:.3f}"
        for index, (component, weight) in enumerate(
            zip(mixture.components, mixture.weights), start=1
        )
    )


def _normal_mixture_text(mixture: NormalMixture) -> str:
    return "\n".join(
        f"комп. {index}: mean={component.mean:.1f}, sigma={component.sigma:.1f}, вес={weight:.3f}"
        for index, (component, weight) in enumerate(
            zip(mixture.components, mixture.weights), start=1
        )
    )


def _style_large_plot(fig, ax) -> None:
    ax.grid(True, alpha=0.18)
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.14)


def plot_source_summary(jobs: list[SourceJob], output_dir: Path) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_jobs(jobs)
    path = output_dir / "source_summary.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    charts = [
        ("Состояние", summary.state_counts),
        ("Запрошенные узлы", summary.req_nodes_counts),
        ("Запрошенные процессоры", summary.req_cpus_counts),
    ]
    for ax, (title, counts) in zip(axes, charts):
        labels = [str(key) for key in counts.keys()]
        values = list(counts.values())
        ax.bar(labels, values, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel("задачи")
        ax.tick_params(axis="x", rotation=20)
        for index, value in enumerate(values):
            ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Сводка исходных данных")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def clean_graphs_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def plot_runtime_fit(jobs: list[SourceJob], output_dir: Path, bins: int = 40) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_fit.png"
    runtimes = elapsed_times(jobs)
    mixture = fit_elapsed_normal_mixture(jobs)
    x_limits = _main_x_limits(runtimes)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        runtimes,
        bins=bins,
        range=x_limits,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.85,
        label="исходные длительности задач",
    )
    _plot_normal_mixture_on_values(
        ax,
        mixture,
        runtimes,
        bins,
        label="смесь нормальных распределений",
        x_limits=x_limits,
    )
    ax.set_title("Распределение длительности задач")
    ax.set_xlabel("ElapsedRaw, секунд")
    ax.set_ylabel("количество задач")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()

    text = _normal_mixture_text(mixture)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _manifest_scaled_forecast_errors(manifest_path: Path | None) -> list[float]:
    if not manifest_path or not manifest_path.exists():
        return []
    errors: list[float] = []
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("scaled_forecast_seconds"):
                forecast = float(row["scaled_forecast_seconds"])
            else:
                sampled_elapsed = float(row["sampled_elapsed_seconds"])
                if sampled_elapsed <= 0:
                    continue
                scale = float(row["scaled_sleep_seconds"]) / sampled_elapsed
                forecast = float(row["generated_forecast_seconds"]) * scale
            error = forecast - float(row["scaled_sleep_seconds"])
            if error > 0:
                errors.append(error)
    return errors


def plot_forecast_error_fit(
    jobs: list[SourceJob],
    output_dir: Path,
    manifest_path: Path | None = None,
    cluster_results_path: Path | None = None,
    bins: int = 40,
    log_base: float = math.e,
) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "forecast_error_fit.png"
    errors = _manifest_scaled_forecast_errors(manifest_path) or forecast_errors(jobs)
    actual_errors: list[float] = []
    if cluster_results_path:
        actual_errors = _cluster_actual_errors_from_manifest(
            _read_cluster_runtime_rows(cluster_results_path), manifest_path
        )

    mixture = fit_lognormal_mixture(errors, component_count=2)
    log_mixture = _rescale_lognormal_mixture(mixture, log_base)

    x_values = errors + actual_errors if actual_errors else errors
    x_limits = _main_x_limits(x_values, upper_quantile=0.98)
    histogram_bins = _compact_hist_bins(
        x_values, bins, target_bin_width=1.2, min_bins=20
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        errors,
        bins=histogram_bins,
        range=x_limits,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.55,
        label="расчетные ошибки прогноза",
    )
    if actual_errors:
        ax.hist(
            actual_errors,
            bins=histogram_bins,
            range=x_limits,
            color="#f58518",
            edgecolor="white",
            alpha=0.55,
            label="фактические ошибки прогноза",
        )
    _plot_pdf(
        ax,
        mixture,
        errors,
        histogram_bins,
        label="смесь логнормальных распределений",
        x_limits=x_limits,
    )
    ax.set_title("Распределение ошибки прогноза времени")
    ax.set_xlabel("прогноз - время выполнения, секунд")
    ax.set_ylabel("количество задач")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()

    text = _mixture_text(log_mixture)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_log_forecast_error_fit(
    jobs: list[SourceJob], output_dir: Path, bins: int = 40, log_base: float = math.e
) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "log_forecast_error_fit.png"
    errors = forecast_errors(jobs)
    log_errors = _log_values(errors, log_base)
    mixture = _rescale_lognormal_mixture(
        fit_forecast_error_lognormal_mixture(jobs), log_base
    )
    x_limits = _main_x_limits(log_errors, upper_quantile=0.98)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        log_errors,
        bins=bins,
        range=x_limits,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.85,
        label="логарифм исходной ошибки",
    )
    _plot_normal_mixture_pdf(ax, mixture, log_errors, bins, x_limits=x_limits)
    ax.set_title("Распределение логарифма ошибки прогноза времени")
    ax.set_xlabel(_log_axis_label(log_base, "TimelimitRaw * 60 - ElapsedRaw"))
    ax.set_ylabel("количество задач")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()

    text = _mixture_text(mixture)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_generated_vs_source(
    jobs: list[SourceJob], manifest_path: Path, output_dir: Path, bins: int = 40
) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "generated_runtime_vs_source.png"

    source_runtimes = elapsed_times(jobs)
    generated_runtimes = []
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            generated_runtimes.append(int(row["sampled_elapsed_seconds"]))
    x_limits = _main_x_limits(source_runtimes + generated_runtimes)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        source_runtimes,
        bins=bins,
        range=x_limits,
        density=True,
        color="#4c78a8",
        alpha=0.45,
        label="исходные длительности задач",
    )
    ax.hist(
        generated_runtimes,
        bins=bins,
        range=x_limits,
        density=True,
        color="#f58518",
        alpha=0.65,
        label="сгенерированные длительности задач",
    )
    ax.set_title("Сравнение исходных и сгенерированных длительностей задач")
    ax.set_xlabel("длительность задачи, секунд")
    ax.set_ylabel("плотность")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _parse_log_times(log_path: Path) -> dict[str, dict[str, dt.datetime]]:
    pattern = re.compile(
        r"job_index=(?P<index>\d+).* (?P<kind>start|end)=(?P<time>\\S+)"
    )
    result: dict[str, dict[str, dt.datetime]] = {}
    if not log_path.exists():
        return result
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        index = match.group("index")
        result.setdefault(index, {})[match.group("kind")] = dt.datetime.fromisoformat(
            match.group("time")
        )
    return result


def plot_execution_overhead(
    manifest_path: Path, log_path: Path, output_dir: Path
) -> Path | None:
    plt = _import_pyplot()
    log_times = _parse_log_times(log_path)
    if not log_times:
        return None

    planned = {}
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            index = row["script"].split("_")[1]
            planned[index] = int(row["scaled_sleep_seconds"])

    indexes: list[int] = []
    overhead_percent: list[float] = []
    for index, times in sorted(log_times.items(), key=lambda item: int(item[0])):
        if "start" not in times or "end" not in times or index not in planned:
            continue
        actual_seconds = (times["end"] - times["start"]).total_seconds()
        expected_seconds = planned[index]
        if expected_seconds <= 0:
            continue
        indexes.append(int(index))
        overhead_percent.append(
            (actual_seconds - expected_seconds) / expected_seconds * 100.0
        )

    if not indexes:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "execution_overhead.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(indexes, overhead_percent, color="#54a24b")
    ax.axhline(3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%")
    ax.set_title("Накладные расходы фактического запуска")
    ax.set_xlabel("номер сгенерированной задачи")
    ax.set_ylabel("(факт - план) / план, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _read_cluster_runtime_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []

    rows = []
    with results_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if not row.get("actual_slurm_runtime_seconds"):
                continue
            rows.append(row)
    return rows


def _parse_slurm_time(value: str) -> int:
    if "-" in value:
        days_raw, time_raw = value.split("-", 1)
        days = int(days_raw)
    else:
        days = 0
        time_raw = value
    hours_raw, minutes_raw, seconds_raw = time_raw.split(":")
    return (
        days * 24 * 60 * 60
        + int(hours_raw) * 60 * 60
        + int(minutes_raw) * 60
        + int(seconds_raw)
    )


def _cluster_runtime_series(
    rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    sorted_rows = sorted(rows, key=lambda row: int(row["script"].split("_")[1]))
    indexes = [int(row["script"].split("_")[1]) for row in sorted_rows]
    planned = [float(row["planned_sleep_seconds"]) for row in sorted_rows]
    actual = [float(row["actual_slurm_runtime_seconds"]) for row in sorted_rows]
    overhead = [float(row["overhead_vs_sleep_percent"]) for row in sorted_rows]
    return indexes, planned, actual, overhead


def _cluster_actual_errors(rows: list[dict[str, str]]) -> list[float]:
    errors: list[float] = []
    for row in rows:
        if not row.get("requested_slurm_time") or not row.get(
            "actual_slurm_runtime_seconds"
        ):
            continue
        error = _parse_slurm_time(row["requested_slurm_time"]) - float(
            row["actual_slurm_runtime_seconds"]
        )
        if error > 0:
            errors.append(error)
    return errors


def _manifest_scaled_forecasts(manifest_path: Path | None) -> dict[str, float]:
    if not manifest_path or not manifest_path.exists():
        return {}
    result: dict[str, float] = {}
    with manifest_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("scaled_forecast_seconds"):
                result[row["script"]] = float(row["scaled_forecast_seconds"])
                continue
            sampled_elapsed = float(row["sampled_elapsed_seconds"])
            if sampled_elapsed <= 0:
                continue
            scale = float(row["scaled_sleep_seconds"]) / sampled_elapsed
            result[row["script"]] = float(row["generated_forecast_seconds"]) * scale
    return result


def _cluster_actual_errors_from_manifest(
    rows: list[dict[str, str]], manifest_path: Path | None
) -> list[float]:
    forecasts = _manifest_scaled_forecasts(manifest_path)
    errors: list[float] = []
    for row in rows:
        forecast = forecasts.get(row["script"])
        if forecast is None or not row.get("actual_slurm_runtime_seconds"):
            continue
        error = forecast - float(row["actual_slurm_runtime_seconds"])
        if error > 0:
            errors.append(error)
    return errors


def plot_actual_error_fit(
    results_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    bins: int = 30,
    log_base: float = math.e,
) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)
    errors = _cluster_actual_errors_from_manifest(rows, manifest_path)

    if len(errors) < 2:
        return None

    mixture = fit_lognormal_mixture(errors, component_count=2)
    log_mixture = _rescale_lognormal_mixture(mixture, log_base)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_error_fit.png"
    x_limits = _main_x_limits(errors)
    histogram_bins = _compact_hist_bins(errors, bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        errors,
        bins=histogram_bins,
        range=x_limits,
        color="#f58518",
        edgecolor="white",
        alpha=0.85,
        label="фактические ошибки прогноза",
    )
    _plot_pdf(
        ax,
        mixture,
        errors,
        histogram_bins,
        label="смесь логнормальных распределений",
        x_limits=x_limits,
    )
    ax.set_title("Фактическое распределение ошибки прогноза времени")
    ax.set_xlabel("расчетный прогноз из manifest - фактическое время, секунд")
    ax.set_ylabel("количество задач")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()

    text = _mixture_text(log_mixture)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    # return path


def plot_log_actual_error_fit(
    results_path: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    bins: int = 30,
    log_base: float = math.e,
) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)
    errors = _cluster_actual_errors_from_manifest(rows, manifest_path)
    if len(errors) < 2:
        return None

    mixture = _rescale_lognormal_mixture(
        fit_lognormal_mixture(errors, component_count=2), log_base
    )
    log_errors = _log_values(errors, log_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "log_actual_error_fit.png"
    x_limits = _main_x_limits(log_errors)
    histogram_bins = _compact_hist_bins(errors, bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        log_errors,
        bins=histogram_bins,
        range=x_limits,
        color="#f58518",
        edgecolor="white",
        alpha=0.85,
        label="логарифм фактической ошибки",
    )
    _plot_normal_mixture_pdf(ax, mixture, log_errors, histogram_bins, x_limits=x_limits)
    ax.set_title("Фактическое распределение логарифма ошибки прогноза времени")
    ax.set_xlabel(
        _log_axis_label(log_base, "расчетный прогноз из manifest - фактическое время")
    )
    ax.set_ylabel("количество задач")
    ax.set_xlim(*_padded_x_limits(x_limits))
    ax.legend()

    text = _mixture_text(mixture)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    _style_large_plot(fig, ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cluster_runtime_spread(
    results_path: Path, output_dir: Path, max_group_size: int
) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)

    if not rows:
        return None

    print(f"rows={len(rows)}")

    indexes, planned, actual, overhead = _cluster_runtime_series(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_runtime_spread.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    flat_axes = axes.flatten()

    lower = min(min(planned), min(actual))
    upper = max(max(planned), max(actual))
    flat_axes[0].scatter(planned, actual, s=26, alpha=0.65, color="#4c78a8")
    flat_axes[0].plot(
        [lower, upper],
        [lower, upper],
        color="#d62728",
        linewidth=1.5,
        linestyle="--",
        label="факт = план",
    )
    flat_axes[0].set_title("Плановое и фактическое время")
    flat_axes[0].set_xlabel("запланированный sleep, секунд")
    flat_axes[0].set_ylabel("фактическое время Slurm, секунд")
    flat_axes[0].legend()

    flat_axes[1].hist(
        overhead,
        bins=min(14, max(5, len(set(overhead)))),
        color="#f58518",
        edgecolor="white",
    )
    overhead_xlim = _padded_x_limits((min(overhead), max(overhead)), padding_ratio=0.08)
    flat_axes[1].set_xlim(*overhead_xlim)
    if overhead_xlim[0] <= 3.0 <= overhead_xlim[1]:
        flat_axes[1].axvline(
            3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%"
        )
    flat_axes[1].set_title("Распределение отклонения")
    flat_axes[1].set_xlabel("(факт - sleep) / sleep, %")
    flat_axes[1].set_ylabel("количество задач")
    if flat_axes[1].get_legend_handles_labels()[0]:
        flat_axes[1].legend()

    group_size = max(1, math.ceil(len(indexes) / max_group_size))
    grouped_indexes: list[float] = []
    grouped_planned: list[float] = []
    grouped_actual: list[float] = []
    grouped_overhead: list[float] = []
    for start in range(0, len(indexes), group_size):
        end = start + group_size
        grouped_indexes.append(statistics.mean(indexes[start:end]))
        grouped_planned.append(statistics.median(planned[start:end]))
        grouped_actual.append(statistics.median(actual[start:end]))
        grouped_overhead.append(statistics.median(overhead[start:end]))

    flat_axes[2].plot(
        grouped_indexes,
        grouped_planned,
        marker="o",
        linewidth=1.8,
        label="медиана planned sleep",
    )
    flat_axes[2].plot(
        grouped_indexes,
        grouped_actual,
        marker="o",
        linewidth=1.8,
        label="медиана факта",
    )
    flat_axes[2].set_title(f"Медианы по группам задач, группа до {group_size}")
    flat_axes[2].set_xlabel("номер задачи")
    flat_axes[2].set_ylabel("секунд")
    flat_axes[2].legend()

    flat_axes[3].plot(
        grouped_indexes,
        grouped_overhead,
        marker="o",
        linewidth=1.8,
        color="#54a24b",
        label="медиана отклонения",
    )
    flat_axes[3].axhline(
        3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%"
    )
    flat_axes[3].set_title("Медианное отклонение по группам")
    flat_axes[3].set_xlabel("номер задачи")
    flat_axes[3].set_ylabel("(факт - sleep) / sleep, %")
    flat_axes[3].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cluster_planned_actual_by_job(
    results_path: Path,
    output_dir: Path,
    max_group_size: int,
) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)
    if not rows:
        return None

    indexes, planned, actual, _ = _cluster_runtime_series(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "planned_vs_actual_by_job.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(indexes, planned, s=18, alpha=0.55, color="#4c78a8")
    ax.scatter(indexes, actual, s=18, alpha=0.55, color="#f58518")

    group_size = max(1, math.ceil(len(indexes) / max_group_size))
    grouped_indexes: list[float] = []
    grouped_planned: list[float] = []
    grouped_actual: list[float] = []
    for start in range(0, len(indexes), group_size):
        end = start + group_size
        grouped_indexes.append(statistics.mean(indexes[start:end]))
        grouped_planned.append(statistics.median(planned[start:end]))
        grouped_actual.append(statistics.median(actual[start:end]))

    ax.plot(
        grouped_indexes,
        grouped_planned,
        color="#4c78a8",
        linewidth=2.2,
        marker="o",
        label="запланированный sleep",
    )
    ax.plot(
        grouped_indexes,
        grouped_actual,
        color="#f58518",
        linewidth=2.2,
        marker="o",
        label="фактическое время Slurm",
    )
    ax.set_title("Плановое и фактическое время по задачам")
    ax.set_xlabel("номер задачи")
    ax.set_ylabel("секунд")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cluster_overhead_by_job(
    results_path: Path, output_dir: Path, max_group_size: int
) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)
    if not rows:
        return None

    indexes, _, _, overhead = _cluster_runtime_series(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_overhead_by_job.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(indexes, overhead, s=18, alpha=0.55, color="#54a24b")

    group_size = max(1, math.ceil(len(indexes) / max_group_size))
    grouped_indexes: list[float] = []
    grouped_overhead: list[float] = []
    for start in range(0, len(indexes), group_size):
        end = start + group_size
        grouped_indexes.append(statistics.mean(indexes[start:end]))
        grouped_overhead.append(statistics.median(overhead[start:end]))

    ax.plot(
        grouped_indexes,
        grouped_overhead,
        color="#54a24b",
        linewidth=2.2,
        marker="o",
        label="медиана по группам",
    )
    ax.axhline(3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%")
    ax.set_title("Отклонение фактического времени по задачам")
    ax.set_xlabel("номер задачи")
    ax.set_ylabel("(факт - sleep) / sleep, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_graphs(
    jobs: list[SourceJob],
    max_group_size: int,
    output_dir: Path,
    manifest_path: Path | None,
    log_path: Path | None,
    cluster_results_path: Path | None = None,
    log_base: float = math.e,
) -> list[Path]:
    clean_graphs_dir(output_dir)
    paths = [
        plot_source_summary(jobs, output_dir),
        plot_runtime_fit(jobs, output_dir),
        plot_forecast_error_fit(
            jobs,
            output_dir,
            manifest_path=manifest_path,
            cluster_results_path=cluster_results_path,
            log_base=log_base,
        ),
        plot_log_forecast_error_fit(jobs, output_dir, log_base=log_base),
    ]
    if manifest_path and manifest_path.exists():
        paths.append(plot_generated_vs_source(jobs, manifest_path, output_dir))
    if manifest_path and log_path:
        overhead = plot_execution_overhead(manifest_path, log_path, output_dir)
        if overhead:
            paths.append(overhead)
    if cluster_results_path:
        planned_actual = plot_cluster_planned_actual_by_job(
            cluster_results_path, output_dir, max_group_size
        )
        if planned_actual:
            paths.append(planned_actual)
        overhead_by_job = plot_cluster_overhead_by_job(
            cluster_results_path, output_dir, max_group_size
        )
        if overhead_by_job:
            paths.append(overhead_by_job)
        actual_error = plot_actual_error_fit(
            cluster_results_path,
            output_dir,
            manifest_path=manifest_path,
            log_base=log_base,
        )
        if actual_error:
            paths.append(actual_error)
        log_actual_error = plot_log_actual_error_fit(
            cluster_results_path,
            output_dir,
            manifest_path=manifest_path,
            log_base=log_base,
        )
        if log_actual_error:
            paths.append(log_actual_error)
    return paths
