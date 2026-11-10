#!/bin/bash
#SBATCH --job-name=p037-data
#SBATCH --account=EDIT_ME
#SBATCH --partition=EDIT_ME
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Stage 1. Build the 8-bit patch cache and prepare the 4 test scenes.
# Usage: sbatch data.sbatch CONFIG [SOURCE [SCENE_SOURCE]]
#   SOURCE        extracted dataset folder or urban_sar_floods.tar.gz (training / validation chips)
#   SCENE_SOURCE  folder holding testing_case_orig/ (the 4 full scenes)
# Without SOURCE the files are downloaded from Hugging Face (38 GB + 31 GB): compute nodes usually have
# no internet, so in that case run the same command on a login or data-transfer node instead:
#   p037 data --config CONFIG
set -euo pipefail
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/env.sh"
CONFIG=$1
ARGS=()
[ -n "${2:-${P037_SOURCE:-}}" ] && ARGS+=(--source "${2:-$P037_SOURCE}")
[ -n "${3:-}" ] && ARGS+=(--scene-source "$3")
p037 data --config "$CONFIG" --workers "$SLURM_CPUS_PER_TASK" ${ARGS[@]+"${ARGS[@]}"}
