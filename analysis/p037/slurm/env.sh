# Sourced by every P037 Slurm job. EDIT the three marked lines for your account, then leave it alone.
#
#   P037_REPO   clone of this repository (default: three levels above this file)
#   P037_DATA   large files on scratch: caches (9 GB), test scenes (2 GB), checkpoints (0.4 GB per run)
#   P037_SOURCE the extracted UrbanSARFloods folder or urban_sar_floods.tar.gz, if already on disk
P037_REPO=${P037_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export P037_DATA=${P037_DATA:-${SCRATCH:-$HOME/scratch}/p037}          # EDIT if your scratch is elsewhere
export HF_HOME=${HF_HOME:-$P037_DATA/hf_home}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$P037_DATA/logs"
cd "$P037_REPO"
# --- environment: EDIT to activate the environment that has `pip install -e ".[p037]"` ---
# module load languages/python/3.12.3
# source ~/miniforge3/bin/activate s1water
