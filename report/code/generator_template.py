from __future__ import annotations

from pathlib import Path


def render_batch_script(
    index: int,
    source_job_id: str,
    partition: str,
    nodes: int,
    ntasks: int,
    slurm_time: str,
    sleep_seconds: int,
    log_file: Path,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=qe7_{index:04d}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={slurm_time}
#SBATCH --output=generated/logs/%x-%j.out
#SBATCH --error=generated/logs/%x-%j.err

set -euo pipefail

mkdir -p "$(dirname "{log_file}")"
start_time="$(date --iso-8601=seconds)"
echo "job_index={index} source_job_id={source_job_id} start=${{start_time}} planned_sleep={sleep_seconds}" >> "{log_file}"
sleep {sleep_seconds}
end_time="$(date --iso-8601=seconds)"
echo "job_index={index} source_job_id={source_job_id} end=${{end_time}} planned_sleep={sleep_seconds}" >> "{log_file}"
"""
