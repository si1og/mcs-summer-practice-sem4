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
    SourceJob,
    elapsed_times,
    fit_elapsed_lognormal,
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


def _plot_pdf(ax, fit: LognormalFit, values: list[int], bins: int) -> None:
    lower = max(1, min(values))
    upper = max(values)
    step = (upper - lower) / 300
    xs = [lower + step * index for index in range(301)]
    _, _, bin_width = _histogram(values, bins)
    ys = [fit.pdf(x) * len(values) * bin_width for x in xs]
    ax.plot(xs, ys, color="#d62728", linewidth=2.0, label="логнормальная аппроксимация")


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
    fit = fit_elapsed_lognormal(jobs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        runtimes,
        bins=bins,
        color="#72b7b2",
        edgecolor="white",
        alpha=0.85,
        label="исходные длительности задач",
    )
    _plot_pdf(ax, fit, runtimes, bins)
    ax.set_title("Распределение длительности задач")
    ax.set_xlabel("ElapsedRaw, секунд")
    ax.set_ylabel("количество задач")
    ax.legend()

    text = f"mu={fit.mu:.4f}\nsigma={fit.sigma:.4f}"
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


def _cluster_runtime_series(
    rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    indexes = [int(row["script"].split("_")[1]) for row in rows]
    planned = [float(row["planned_sleep_seconds"]) for row in rows]
    actual = [float(row["actual_slurm_runtime_seconds"]) for row in rows]
    overhead = [float(row["overhead_vs_sleep_percent"]) for row in rows]
    return indexes, planned, actual, overhead


def plot_cluster_runtime_spread(results_path: Path, output_dir: Path) -> Path | None:
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
        bins=min(30, max(5, len(overhead) // 8)),
        color="#f58518",
        edgecolor="white",
    )
    flat_axes[1].axvline(
        3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%"
    )
    flat_axes[1].set_title("Распределение отклонения")
    flat_axes[1].set_xlabel("(факт - sleep) / sleep, %")
    flat_axes[1].set_ylabel("количество задач")
    flat_axes[1].legend()

    group_size = max(1, math.ceil(len(indexes) / 20))
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
    results_path: Path, output_dir: Path
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

    group_size = max(1, math.ceil(len(indexes) / 20))
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


def plot_cluster_overhead_by_job(results_path: Path, output_dir: Path) -> Path | None:
    plt = _import_pyplot()
    rows = _read_cluster_runtime_rows(results_path)
    if not rows:
        return None

    indexes, _, _, overhead = _cluster_runtime_series(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_overhead_by_job.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(indexes, overhead, s=18, alpha=0.55, color="#54a24b")

    group_size = max(1, math.ceil(len(indexes) / 20))
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
    ax.axhline(
        3.0, color="#d62728", linestyle="--", linewidth=1.5, label="порог 3%"
    )
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
        planned_actual = plot_cluster_planned_actual_by_job(
            cluster_results_path, output_dir
        )
        if planned_actual:
            paths.append(planned_actual)
        overhead_by_job = plot_cluster_overhead_by_job(cluster_results_path, output_dir)
        if overhead_by_job:
            paths.append(overhead_by_job)
    return paths
