from __future__ import annotations

from pathlib import Path

from src.analyze.analyze import SourceJob


def render_batch_script(
    index: int,
    source: SourceJob,
    partition: str,
    nodes: int,
    ntasks: int,
    slurm_time: str,
    sleep_seconds: int,
    log_file: Path,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={source.job_name}_{index:04d}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={slurm_time}
#SBATCH --output=generated/logs/%x-%j.out
#SBATCH --error=generated/logs/%x-%j.err

set -euo pipefail

mkdir -p "$(dirname "{log_file}")"

start_time="$(date --iso-8601=seconds)"
echo "job_index={index} source_job_id={source.job_id} host=$(hostname) start=${{start_time}}" | tee -a "{log_file}"
sleep {sleep_seconds}
end_time="$(date --iso-8601=seconds)"
echo "job_index={index} source_job_id={source.job_id} host=$(hostname) end=${{end_time}}" | tee -a "{log_file}"
"""
