#!/bin/bash
#SBATCH --job-name=p037-eval
#SBATCH --account=EDIT_ME
#SBATCH --partition=EDIT_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Stages 3-5. Predict and score the test scenes (1 + 20 MC-dropout passes per window), check proxy
# coverage, run the analysis. Usage: sbatch evaluate.sbatch CONFIG
# Scenes already scored are reused, so a job that hits the time limit can simply be resubmitted.
set -euo pipefail
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/analysis/p037/slurm" "$(dirname "$0")"; do
  if [ -f "$d/env.sh" ]; then source "$d/env.sh"; break; fi
done
command -v p037 >/dev/null || { echo "environment not set up: see analysis/p037/slurm/env.sh" >&2; exit 1; }
p037 all --config "$1"
