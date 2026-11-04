#!/bin/bash
#SBATCH --job-name=usf-scene
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Full-scene pre and post prediction (original 4224 px tiling), pixel error map, scene metrics and the
# per-tile error-map figure (uses the chip CSVs, so it runs after the chips array).
# Memory: ~6 GB for the largest scene (NovaKakhovka, 22596 x 12226 px). Usage: sbatch usf_scene.sbatch EVENT
set -euo pipefail
source "$(dirname "$0")/usf_env.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/usf_env.sh"
python analysis/usf_eval.py --device "${USF_DEVICE:-cpu}" scene "$1" --dtype "${USF_SCENE_DTYPE:-float32}"
