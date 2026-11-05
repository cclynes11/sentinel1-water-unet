#!/bin/bash
#SBATCH --job-name=usf-chips
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%A_%a.out
# One shard of the 256 px chips of one event: deterministic pass + 20 MC-dropout passes on pre and post.
# Usage: sbatch --array=0-$((NSHARDS-1)) usf_chips.sbatch EVENT NSHARDS
# Cost in float32 on CPU: ~7 s per chip on 4 cores (2 images x 21 forward passes of a 34.6 M-parameter
# U-Net), ~28 core-seconds per chip, ~72 core-hours for the 9176 chips of the test set; 200 chips per task
# take ~30-60 min on 8 cores. On a GPU node add  --gres=gpu:1  and set USF_DEVICE=cuda.
set -euo pipefail
source "$(dirname "$0")/usf_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/usf_env.sh"
EVENT=$1
NSHARDS=$2
python analysis/usf_eval.py --device "${USF_DEVICE:-cpu}" chips "$EVENT" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" \
    --mc-passes 20 --dtype float32 --mc-dtype "${USF_MC_DTYPE:-float32}" \
    --download-workers 4
