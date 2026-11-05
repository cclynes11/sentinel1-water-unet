#!/bin/bash
#SBATCH --job-name=usf-prepare
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# Band identification + 8-bit pre/post scene for one event. Downloads the 6-10 GB scene unless already
# downloaded with `usf_eval.py download` (USF_OFFLINE=1). Usage: sbatch usf_prepare.sbatch EVENT
set -euo pipefail
source "$(dirname "$0")/usf_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/usf_env.sh"
EVENT=$1
python analysis/usf_eval.py bands "$EVENT"
python analysis/usf_eval.py prepare "$EVENT"
