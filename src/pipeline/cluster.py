from __future__ import annotations

import csv
import datetime as dt
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


RESULT_FIELDS = [
    "script",
    "slurm_job_id",
    "source_job_id",
    "planned_sleep_seconds",
    "requested_slurm_time",
    "submitted_observed_at",
    "completed_observed_at",
    "slurm_submit_time",
    "slurm_start_time",
    "slurm_end_time",
    "slurm_state",
    "node_list",
    "queue_wait_seconds",
    "actual_slurm_runtime_seconds",
    "observed_wall_seconds",
    "overhead_vs_sleep_percent",
]


@dataclass(frozen=True)
class RemoteRun:
    local_run_dir: Path
    remote_run_dir: str
    manifest_path: Path


def shell_quote(value: str | Path) -> str:
    raw = str(value)
    if raw == "~":
        return raw
    if raw.startswith("~/"):
        return "~/" + shlex.quote(raw[2:])
    return shlex.quote(raw)


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def ssh(host: str, command: str) -> str:
    return run_command(["ssh", host, command])


def try_ssh(host: str, command: str) -> str:
    try:
        return ssh(host, command)
    except subprocess.CalledProcessError:
        return ""


def scp_from(source: str, target: Path) -> None:
    subprocess.run(["scp", source, str(target)], check=True)


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_slurm_fields(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_datetime(value: str) -> dt.datetime | None:
    if not value or value in {"Unknown", "N/A", "None"}:
        return None
    return dt.datetime.fromisoformat(value)


def seconds_between(start: dt.datetime | None, end: dt.datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return {row["script"]: row for row in csv.DictReader(input_file)}


def run_remote_generator(
    host: str,
    generator_dir: str,
    generator_command: str,
    results_root: Path,
) -> RemoteRun:
    run_id = "run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_run_dir = results_root / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    remote_run_dir = ssh(
        host,
        "cd "
        f"{shell_quote(generator_dir)} && "
        f"{generator_command} --run-id {shell_quote(run_id)}",
    )
    if not remote_run_dir:
        raise SystemExit("remote generator did not print remote run directory")

    manifest_path = local_run_dir / "manifest.csv"
    scp_from(f"{host}:{remote_run_dir}/manifest.csv", manifest_path)
    return RemoteRun(
        local_run_dir=local_run_dir,
        remote_run_dir=remote_run_dir,
        manifest_path=manifest_path,
    )


def copy_existing_manifest(
    host: str,
    remote_run_dir: str,
    results_root: Path,
) -> RemoteRun:
    run_id = Path(remote_run_dir.rstrip("/")).name
    local_run_dir = results_root / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_run_dir / "manifest.csv"
    scp_from(f"{host}:{remote_run_dir}/manifest.csv", manifest_path)
    return RemoteRun(
        local_run_dir=local_run_dir,
        remote_run_dir=remote_run_dir,
        manifest_path=manifest_path,
    )


def build_result_row(
    host: str,
    submitted_row: dict[str, str],
    manifest_row: dict[str, str],
) -> dict[str, str | int]:
    completed_at = timestamp()
    slurm_raw = try_ssh(host, f"scontrol show job {submitted_row['slurm_job_id']} -o")
    slurm = parse_slurm_fields(slurm_raw)

    submit_time = parse_datetime(slurm.get("SubmitTime", ""))
    start_time = parse_datetime(slurm.get("StartTime", ""))
    end_time = parse_datetime(slurm.get("EndTime", ""))
    planned_sleep = int(manifest_row["scaled_sleep_seconds"])
    actual_runtime = seconds_between(start_time, end_time)
    queue_wait = seconds_between(submit_time, start_time)
    observed_wall = seconds_between(
        dt.datetime.fromisoformat(submitted_row["submitted_observed_at"]),
        dt.datetime.fromisoformat(completed_at),
    )
    overhead = None
    if actual_runtime is not None:
        overhead = (actual_runtime - planned_sleep) / planned_sleep * 100.0

    return {
        "script": submitted_row["script"],
        "slurm_job_id": submitted_row["slurm_job_id"],
        "source_job_id": manifest_row["source_job_id"],
        "planned_sleep_seconds": planned_sleep,
        "requested_slurm_time": manifest_row["slurm_time"],
        "submitted_observed_at": submitted_row["submitted_observed_at"],
        "completed_observed_at": completed_at,
        "slurm_submit_time": slurm.get("SubmitTime", ""),
        "slurm_start_time": slurm.get("StartTime", ""),
        "slurm_end_time": slurm.get("EndTime", ""),
        "slurm_state": slurm.get("JobState", "UNKNOWN"),
        "node_list": slurm.get("NodeList", ""),
        "queue_wait_seconds": f"{queue_wait:.3f}" if queue_wait is not None else "",
        "actual_slurm_runtime_seconds": (
            f"{actual_runtime:.3f}" if actual_runtime is not None else ""
        ),
        "observed_wall_seconds": (
            f"{observed_wall:.3f}" if observed_wall is not None else ""
        ),
        "overhead_vs_sleep_percent": (
            f"{overhead:.3f}" if overhead is not None else ""
        ),
    }


def collect_results_as_jobs_finish(
    host: str,
    submitted: list[dict[str, str]],
    manifest: dict[str, dict[str, str]],
    result_path: Path,
    poll_interval: float,
) -> None:
    pending = {row["slurm_job_id"]: row for row in submitted}
    with result_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        while pending:
            ids = ",".join(sorted(pending))
            raw = ssh(host, f"squeue -h -j {ids} -o '%i' || true")
            active = {line.strip() for line in raw.splitlines() if line.strip()}
            finished = sorted(set(pending) - active, key=int)

            for job_id in finished:
                submitted_row = pending.pop(job_id)
                writer.writerow(
                    build_result_row(
                        host=host,
                        submitted_row=submitted_row,
                        manifest_row=manifest[submitted_row["script"]],
                    )
                )
                output_file.flush()

            if pending:
                time.sleep(poll_interval)


def submit_remote_run(
    host: str,
    remote_run_dir: str,
    local_run_dir: Path,
    manifest_path: Path,
    poll_interval: float,
) -> Path:
    manifest = read_manifest(manifest_path)
    submitted: list[dict[str, str]] = []
    for script_name in sorted(manifest):
        submitted_at = timestamp()
        job_id = ssh(
            host,
            "cd "
            f"{shell_quote(remote_run_dir)} && "
            f"sbatch --parsable {shell_quote(Path('slurm_jobs') / script_name)}",
        )
        submitted.append(
            {
                "script": script_name,
                "slurm_job_id": job_id,
                "submitted_observed_at": submitted_at,
            }
        )

    result_path = local_run_dir / "results.csv"
    collect_results_as_jobs_finish(
        host=host,
        submitted=submitted,
        manifest=manifest,
        result_path=result_path,
        poll_interval=poll_interval,
    )
    return result_path
