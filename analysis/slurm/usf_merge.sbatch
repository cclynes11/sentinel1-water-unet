#!/bin/bash
#SBATCH --job-name=usf-merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Concatenate the shard CSVs into analysis/usf_results.csv and write the summary tables.
set -euo pipefail
source "$(dirname "$0")/usf_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/usf_env.sh"
python analysis/usf_eval.py merge
