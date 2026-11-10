#!/bin/bash
# Submit the whole UrbanSARFloods evaluation as a Slurm dependency chain:
#   prepare(event) -> chips(event)[array of shards] -> scene(event) -> merge
# Usage:  cd analysis/slurm && ./submit_usf.sh [CHIPS_PER_SHARD]   (default 200 chips per array task)
# Options through the environment: USF_DATA, USF_DEVICE=cuda (+ add --gres to the sbatch lines),
# USF_MC_DTYPE (default float32), USF_OFFLINE=1, SBATCH_PARTITION / SBATCH_ACCOUNT (read by sbatch).
set -euo pipefail
cd "$(dirname "$0")"
PER=${1:-200}
# number of 256 px chips per event in the test set (Hugging Face, revision 0892ffa)
declare -A NCHIPS=([20210727_Weihui]=2475 [20230609_NovaKakhovka]=2893 [20231201_Jubba_1]=1779 [20231201_Jubba_2]=2029)
deps=""
for EVENT in "${!NCHIPS[@]}"; do
  N=$(( (NCHIPS[$EVENT] + PER - 1) / PER ))
  jp=$(sbatch --parsable -J "usf-prep-$EVENT" usf_prepare.sbatch "$EVENT")
  jc=$(sbatch --parsable -J "usf-chips-$EVENT" --dependency=afterok:$jp --array=0-$((N-1)) usf_chips.sbatch "$EVENT" "$N")
  js=$(sbatch --parsable -J "usf-scene-$EVENT" --dependency=afterok:$jc usf_scene.sbatch "$EVENT")
  echo "$EVENT: prepare $jp, chips $jc (array 0-$((N-1))), scene $js"
  deps="$deps:$js"
done
jm=$(sbatch --parsable --dependency=afterok$deps usf_merge.sbatch)
echo "merge $jm  ->  analysis/usf_results.csv"
