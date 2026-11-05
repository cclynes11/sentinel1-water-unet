# Sourced by every UrbanSARFloods Slurm job. Edit for your cluster.
#   USF_REPO      clone of this repository (default: two levels above this file)
#   USF_DATA      work directory on scratch, ~48 GB if `download` is used, ~12 GB peak otherwise
#   USF_OFFLINE=1 compute nodes have no internet: run `submit_usf.sh` after
#                 `python analysis/usf_eval.py download EVENT` for each event on a login/transfer node
USF_REPO=${USF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export USF_DATA=${USF_DATA:-${SCRATCH:-$HOME/scratch}/usf}
export HF_HOME=${HF_HOME:-$USF_DATA/hf_home}      # weights cache (Fabien-Ctrees/sentinel1-water-unet)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$USF_DATA/logs"
cd "$USF_REPO"
# --- environment (edit) ---
# module load python/3.11
# source ~/miniforge3/bin/activate s1water     # env with: pip install -e ".[viz]" pandas
if [ "${USF_OFFLINE:-0}" = 1 ]; then export HF_HUB_OFFLINE=1; fi
