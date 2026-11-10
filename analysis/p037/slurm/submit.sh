#!/bin/bash
# Submit one experiment end to end: data -> train (x N, each resuming the last) -> evaluate + analysis.
# Usage: bash analysis/p037/slurm/submit.sh CONFIG [N_TRAIN_JOBS=2] [SOURCE [SCENES]]
# First: set --account and --partition (EDIT_ME) in the three .sbatch files and the environment in env.sh.
# Jobs are submitted from this folder, so their .out logs are written here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=$(realpath "$1")
N=${2:-2}
cd "$HERE"
if grep -q EDIT_ME ./*.sbatch; then echo "set --account and --partition in the .sbatch files first" >&2; exit 1; fi
job=$(sbatch --parsable data.sbatch "$CONFIG" "${3:-}" "${4:-}")
echo "data      $job"
for i in $(seq "$N"); do
  job=$(sbatch --parsable --dependency=afterok:"$job" train.sbatch "$CONFIG")
  echo "train $i   $job"
done
job=$(sbatch --parsable --dependency=afterok:"$job" evaluate.sbatch "$CONFIG")
echo "evaluate  $job"
