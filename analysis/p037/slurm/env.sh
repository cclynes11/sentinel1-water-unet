# Sourced by every P037 Slurm job. Check the EDIT lines once for your account, then leave it alone.
#
#   P037_REPO    clone of this repository (default: three levels above this file)
#   P037_DATA    where the pipeline keeps its large files: patch cache (9 GB), prepared test scenes (2 GB),
#                one folder per run under runs/ (checkpoints 0.4 GB each, tables, figures)
#   P037_SOURCE  UrbanSARFloods training / validation chips already on disk: the extracted folder or the
#                urban_sar_floods.tar.gz archive. Unset = download from Hugging Face (38 GB).
#   P037_SCENES  folder holding the 4 full test scenes (testing_case_orig/). Unset = download (31 GB).
P037_REPO=${P037_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}

# BluePebble work space. The pipeline writes to its own folder, next to (not inside) the files of the notebooks.
export P037_DATA=${P037_DATA:-/user/work/$USER/p037/pipeline}                  # EDIT if you want it elsewhere
# The copy of the dataset extracted earlier in the project, if it is there.
if [ -z "${P037_SOURCE:-}" ] && [ -d "/user/work/$USER/p037/data/urbansarfloods/extracted" ]; then
  export P037_SOURCE=/user/work/$USER/p037/data/urbansarfloods/extracted
fi

export HF_HOME=${HF_HOME:-$P037_DATA/hf_home}        # pretrained weights and dataset downloads are cached here
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$P037_DATA"
cd "$P037_REPO"

# --- Python environment: EDIT to activate the one where you ran  pip install -e ".[p037]"  ---
# module load languages/python/3.12.3
# source ~/miniforge3/bin/activate s1water
if ! command -v p037 >/dev/null; then
  echo "p037 command not found: activate the environment in analysis/p037/slurm/env.sh (EDIT lines)" >&2
  return 1 2>/dev/null || exit 1
fi
