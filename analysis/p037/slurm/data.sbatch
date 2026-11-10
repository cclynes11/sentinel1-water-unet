#!/bin/bash
#SBATCH --job-name=p037-data
#SBATCH --account=EDIT_ME
#SBATCH --partition=EDIT_ME
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Stage 1. Build the 8-bit patch cache, then prepare the test scenes.
# Usage: sbatch data.sbatch CONFIG [SOURCE [SCENES]]
#   SOURCE  extracted dataset folder or urban_sar_floods.tar.gz (default: $P037_SOURCE, see env.sh)
#   SCENES  folder holding the full test scenes (default: $P037_SCENES)
# Anything not on disk is downloaded from Hugging Face (38 GB archive, 31 GB of scenes). If the compute
# nodes have no internet, run the two p037 commands below on a login node instead.
# Both steps skip what already exists, so the job can be resubmitted.
set -euo pipefail
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/analysis/p037/slurm" "$(dirname "$0")"; do
  if [ -f "$d/env.sh" ]; then source "$d/env.sh"; break; fi
done
command -v p037 >/dev/null || { echo "environment not set up: see analysis/p037/slurm/env.sh" >&2; exit 1; }
CONFIG=$1
SOURCE=${2:-${P037_SOURCE:-}}
SCENES=${3:-${P037_SCENES:-}}
p037 data --config "$CONFIG" --what cache --workers "${SLURM_CPUS_PER_TASK:-4}" ${SOURCE:+--source "$SOURCE"}
p037 data --config "$CONFIG" --what scenes ${SCENES:+--scene-source "$SCENES"}
