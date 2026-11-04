# UrbanSARFloods test-set evaluation

Zero-shot evaluation of the Sentinel-1 water U-Net as a **flood** mapper on the UrbanSARFloods test
events (Hugging Face `S1Floodbenchmark/UrbanSARFloods_v1`, revision `0892ffa`):
`20210727_Weihui`, `20230609_NovaKakhovka`, `20231201_Jubba_1`, `20231201_Jubba_2`.

- flood = water(post) AND NOT water(pre), threshold 0.5 on each image. Raw post-event water is scored
  too.
- Scored separately against GT class 1 (open flood) and class 2 (urban flood), one-vs-rest. Pixels of
  the *other* flood class are left out, because the model has a single flood output. NaN (no SAR data)
  pixels are left out.
- Per 256 px chip: the deterministic pass (dropout off), plus 20 MC-dropout passes (dropout layers in
  train mode, batch norm in eval mode) on both pre and post.

| file | content |
|---|---|
| `usf_eval.py` | the whole pipeline (subcommands `download`, `bands`, `prepare`, `chips`, `scene`, `merge`) |
| `band_identification.md`, `band_check/*.json` | which band is pre/post VV/VH and why (the GeoTIFFs have no band descriptions) |
| `preprocessing_shift.md` | every known difference from the model's training inputs |
| `usf_results.csv` | one row per 256 px chip (columns below) |
| `usf_summary_chips.md/.csv` | pixel-pooled scores per event and overall, from the chips |
| `scene_metrics/*.json` | pixel scores of the full-scene prediction (original 4224 px tiling) |
| `figures/error_map_<event>.png` | per-scene error map: pixel errors of the full-scene prediction, per-tile error rate and per-tile MC variance |
| `results_parts/` | per-shard CSVs (merged into `usf_results.csv`) |
| `slurm/` | Slurm job scripts for an HPC run |

## `usf_results.csv` columns

| column | meaning |
|---|---|
| `event`, `tile_id`, `tile_row`, `tile_col` | chip `<event>_ID_<row>_<col>`: window at row 256·row, col 256·col of the full scene |
| `chip_source` | `hub_chip`, or `scene_crop(empty_hub_file)` for the 997 Jubba_2 chips that are 0-byte files on the Hub |
| `chip_equals_scene_crop` | the downloaded chip is bit-identical to the same window of the full scene (8-bit) |
| `n_valid`, `n_gt_open`, `n_gt_urban`, `n_gt_nodata` | valid (non-NaN) pixels and GT counts inside them |
| `{flood,post}_{open,urban}_{tp,fp,fn,tn}` | confusion counts; `flood` = water(post) AND NOT water(pre), `post` = raw water(post) |
| `{flood,post}_{open,urban}_{f1,iou,precision,recall}` | per-chip scores. Empty (NaN) when undefined (e.g. no GT and no predicted flood) |
| `frac_water_pre`, `frac_water_post`, `frac_flood` | fraction of valid pixels predicted water / flood |
| `mean_prob_pre`, `mean_prob_post` | mean water probability, deterministic pass |
| `mean_prob_flood` | mean of P(post)·(1 − P(pre)), deterministic pass |
| `mean_entropy_post_det` | mean binary entropy (bits) of the deterministic P(post) |
| `mc_mean_prob_{pre,post,flood}` | mean over pixels of the MC-dropout mean probability (20 passes) |
| `mean_entropy_{pre,post,flood}` | **mean predictive entropy**: mean over pixels of H(p̄), p̄ = MC mean, in bits |
| `mc_var_{pre,post,flood}` | **MC-dropout variance**: mean over pixels of the per-pixel variance (unbiased, 20 passes) of P(pre), P(post) and P(flood) = P(post)·(1 − P(pre)) |
| `mc_passes` | 20 |

All pixel means are over the valid pixels of the chip. Chips with no valid pixel have NaN metrics.

## Running it

```bash
pip install -e ".[viz]" pandas            # from the repository root; CPU PyTorch is enough
python analysis/usf_eval.py bands 20231201_Jubba_2      # band check (HTTP range reads, no download)
python analysis/usf_eval.py prepare 20231201_Jubba_2    # downloads the 6-10 GB scene, keeps 8-bit only
python analysis/usf_eval.py chips 20231201_Jubba_2      # chips: download -> convert -> delete, + inference
python analysis/usf_eval.py scene 20231201_Jubba_2      # full-scene prediction, error maps, figure
python analysis/usf_eval.py merge
```

### Does it fit a 4 CPU / 16 GB RAM / 30 GB disk machine?

| resource | need | fits? |
|---|---|---|
| download | test set = 48.3 GB (scenes 31.1 GB + chips 17.2 GB) | **no, not at once**. The scripts stream it: one scene at a time (≤ 8.8 GB, deleted after conversion to 8-bit), and each chip deleted right after reading. Peak disk ≈ 10 GB + 1.6 GB of 8-bit scenes |
| RAM | ≤ 6 GB (full-scene inference of Nova Kakhovka, 22596 × 12226 px) | yes |
| compute, float32 | ~7 s per chip on 4 cores (2 images × 21 forward passes of a 34.6 M-parameter U-Net at ~0.2 s per 256 px image), **~18 h for the 9176 chips**, + ~2 h for the 4 × 2 full scenes | **too large** for this machine |
| compute, as run here | float32 deterministic pass, **bfloat16 autocast for the MC passes and the full scenes** (3.6× faster; mean abs. probability difference 6 × 10⁻⁴, ~9× below the MC spread): ~3-4 s per chip, **~9 h in total** | ran, see the PR description |

So the full float32 run is meant for a **Slurm HPC**: `analysis/slurm/submit_usf.sh` submits, per event,
`prepare` → a `chips` job array (200 chips per task, 8 CPUs, 16 GB) → `scene`, then one `merge`.
Edit `analysis/slurm/usf_env.sh` (environment activation, `USF_DATA` on scratch) first. If compute
nodes have no internet, run `python analysis/usf_eval.py download EVENT` for each event on a login or
data-transfer node (48 GB), and set `USF_OFFLINE=1`. On GPU nodes set `USF_DEVICE=cuda` and add
`--gres=gpu:1` to `usf_chips.sbatch` / `usf_scene.sbatch`.
