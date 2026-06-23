from __future__ import annotations

import csv
import datetime as dt
import math
import os
import re
import shutil
from pathlib import Path

from src.analyze.analyze import (
    LognormalMixture,
    SourceJob,
    elapsed_times,
    fit_elapsed_mixture,
    fit_forecast_error_mixture,
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


def _histogram(values: list[int], bins: int) -> tuple[list[float], list[int], float]:
    lower = min(values)
    upper = max(values)
    width = (upper - lower) / bins if upper != lower else 1.0
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lower) / width))
        counts[index] += 1
    centers = [lower + width * (index + 0.5) for index in range(bins)]
    return centers, counts, width


def _plot_pdf(ax, mixture: LognormalMixture, values: list[int], bins: int) -> None:
    lower = max(1, min(values))
    upper = max(values)
    step = (upper - lower) / 300
    xs = [lower + step * index for index in range(301)]
    _, _, bin_width = _histogram(values, bins)
    ys = [mixture.pdf(x) * len(values) * bin_width for x in xs]
    label = (
        "логнормальная аппроксимация"
        if mixture.component_count == 1
        else "смесь логнормальных распределений"
    )
    ax.plot(xs, ys, color="#d62728", linewidth=2.0, label=label)


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


def plot_forecast_error_fit(
    jobs: list[SourceJob], output_dir: Path, bins: int = 40
) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "forecast_error_fit.png"
    errors = forecast_errors(jobs)
    mixture = fit_forecast_error_mixture(jobs, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        errors,
        bins=bins,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.85,
        label="исходные ошибки прогноза",
    )
    _plot_pdf(ax, mixture, errors, bins)
    ax.set_title("Распределение ошибки прогноза времени")
    ax.set_xlabel("TimelimitRaw * 60 - ElapsedRaw, секунд")
    ax.set_ylabel("количество задач")
    ax.legend()

    text = "\n".join(
        f"компонента {index}: mu={component.mu:.4f}, sigma={component.sigma:.4f}, вес={weight:.3f}"
        for index, (component, weight) in enumerate(
            zip(mixture.components, mixture.weights), start=1
        )
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_runtime_fit(jobs: list[SourceJob], output_dir: Path, bins: int = 40) -> Path:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_fit.png"
    runtimes = elapsed_times(jobs)
    mixture = fit_elapsed_mixture(jobs, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        runtimes,
        bins=bins,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.85,
        label="исходные длительности задач",
    )
    _plot_pdf(ax, mixture, runtimes, bins)
    ax.set_title("Распределение длительности задач")
    ax.set_xlabel("ElapsedRaw, секунд")
    ax.set_ylabel("количество задач")
    ax.legend()

    text = "\n".join(
        f"компонента {index}: mu={component.mu:.4f}, sigma={component.sigma:.4f}, вес={weight:.3f}"
        for index, (component, weight) in enumerate(
            zip(mixture.components, mixture.weights), start=1
        )
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        source_runtimes,
        bins=bins,
        density=True,
        color="#4c78a8",
        alpha=0.45,
        label="исходные длительности задач",
    )
    ax.hist(
        generated_runtimes,
        bins=bins,
        density=True,
        color="#f58518",
        alpha=0.65,
        label="сгенерированные длительности задач",
    )
    ax.set_title("Сравнение исходных и сгенерированных длительностей задач")
    ax.set_xlabel("длительность задачи, секунд")
    ax.set_ylabel("плотность")
    ax.legend()
    fig.tight_layout()
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
    for index, times in log_times.items():
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


def plot_cluster_runtime_spread(results_path: Path, output_dir: Path) -> Path | None:
    plt = _import_pyplot()
    if not results_path.exists():
        return None

    rows = []
    with results_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if not row.get("actual_slurm_runtime_seconds"):
                continue
            rows.append(row)

    if not rows:
        return None

    print(f"rows={len(rows)}")

    indexes = [int(row["script"].split("_")[1]) for row in rows]
    planned = [float(row["planned_sleep_seconds"]) for row in rows]
    actual = [float(row["actual_slurm_runtime_seconds"]) for row in rows]
    overhead = [float(row["overhead_vs_sleep_percent"]) for row in rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_runtime_spread.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        indexes, planned, marker="o", linewidth=1.5, label="запланированный sleep"
    )
    axes[0].plot(
        indexes, actual, marker="o", linewidth=1.5, label="фактическое время Slurm"
    )
    axes[0].set_title("Плановое и фактическое время выполнения")
    axes[0].set_xlabel("номер задачи")
    axes[0].set_ylabel("секунд")
    axes[0].legend()

    axes[1].bar(indexes, overhead, color="#f58518")
    axes[1].axhline(
        3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%"
    )
    axes[1].set_title("Отклонение фактического времени")
    axes[1].set_xlabel("номер задачи")
    axes[1].set_ylabel("(факт - sleep) / sleep, %")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_graphs(
    jobs: list[SourceJob],
    output_dir: Path,
    manifest_path: Path | None,
    log_path: Path | None,
    cluster_results_path: Path | None = None,
) -> list[Path]:
    clean_graphs_dir(output_dir)
    paths = [
        plot_source_summary(jobs, output_dir),
        plot_runtime_fit(jobs, output_dir),
    ]
    if manifest_path and manifest_path.exists():
        paths.append(plot_generated_vs_source(jobs, manifest_path, output_dir))
    if manifest_path and log_path:
        overhead = plot_execution_overhead(manifest_path, log_path, output_dir)
        if overhead:
            paths.append(overhead)
    if cluster_results_path:
        spread = plot_cluster_runtime_spread(cluster_results_path, output_dir)
        if spread:
            paths.append(spread)
    return paths
