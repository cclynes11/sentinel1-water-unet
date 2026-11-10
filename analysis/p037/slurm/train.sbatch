#!/bin/bash
#SBATCH --job-name=p037-train
#SBATCH --account=EDIT_ME
#SBATCH --partition=EDIT_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Stage 2. Train. Usage: sbatch train.sbatch CONFIG   (CONFIG: absolute, or relative to the repository root)
# The job stops cleanly before the time limit (`max_hours` in the config) and writes last.pt;
# submitting it again resumes from there, and a job that finds training finished exits at once.
# Expected cost, from the paper's 4.5 h per 100 epochs of 59,387 patches on an A10G: about 1 min per
# epoch of the 18,004 UrbanSARFloods patches plus validation, so 200 epochs in roughly 3-4 h.
# That is an estimate; read the real figure from the `seconds` column of log.csv after the first epochs.
set -euo pipefail
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/analysis/p037/slurm" "$(dirname "$0")"; do
  if [ -f "$d/env.sh" ]; then source "$d/env.sh"; break; fi
done
command -v p037 >/dev/null || { echo "environment not set up: see analysis/p037/slurm/env.sh" >&2; exit 1; }
p037 train --config "$1" --set num_workers="$SLURM_CPUS_PER_TASK"
