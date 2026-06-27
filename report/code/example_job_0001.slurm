#!/usr/bin/env bash
#SBATCH --job-name=qe7_0001
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:17:00
#SBATCH --output=generated/logs/%x-%j.out
#SBATCH --error=generated/logs/%x-%j.err

set -euo pipefail

mkdir -p "$(dirname "generated/logs/generated-jobs.log")"
start_time="$(date --iso-8601=seconds)"
echo "job_index=1 source_job_id=507280001 start=${start_time} planned_sleep=74" >> "generated/logs/generated-jobs.log"
sleep 74
end_time="$(date --iso-8601=seconds)"
echo "job_index=1 source_job_id=507280001 end=${end_time} planned_sleep=74" >> "generated/logs/generated-jobs.log"
