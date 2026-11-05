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
| `scene_metrics/*.json`, `usf_summary_scenes.md` | pixel scores of the full-scene prediction (tiling with 64 px crop; tile size and dtype recorded in the JSON) |
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
| `det_dtype`, `mc_dtype` | precision of the deterministic pass and of the MC-dropout passes (`float32` or `bfloat16` autocast) |

All pixel means are over the valid pixels of the chip. Chips with no valid pixel have NaN metrics.

## Results (local run, 9176 chips, 4 events)

Pixel-pooled F1 over all chips (`usf_summary_chips.md`), and over the full-scene prediction
(`usf_summary_scenes.md`, 64 px crop, so no chip-border effects):

| event | flood, open (chips / scene) | flood, urban | post water, open | post water, urban |
|---|---|---|---|---|
| Weihui | 0.272 / 0.317 | 0.001 / 0.001 | 0.232 / 0.265 | 0.001 / 0.001 |
| Nova Kakhovka | 0.043 / 0.037 | 0.006 / 0.003 | 0.041 / 0.036 | 0.003 / 0.002 |
| Jubba 1 | 0.543 / 0.539 | 0.000 / 0.000 | 0.514 / 0.510 | 0.000 / 0.000 |
| Jubba 2 | 0.463 / 0.441 | 0.000 / 0.000 | 0.291 / 0.274 | 0.000 / 0.000 |
| all (chips) | **0.269** (P 0.554, R 0.177) | 0.002 | 0.240 (P 0.348, R 0.183) | 0.001 |

- **Change detection helps precision, not recall.** `water(post) AND NOT water(pre)` removes permanent
  water: precision on open flood rises from 0.35 to 0.55 overall (Jubba 2: 0.22 → 0.56), recall barely
  moves (0.183 → 0.177). Recall is the main failure.
- **Urban flood is not detected** (F1 ≤ 0.006 everywhere). Expected: urban flood raises backscatter
  (double bounce) and the model looks for dark water (see `preprocessing_shift.md` §6).
- **Nova Kakhovka fails almost completely** (open-flood recall 2 %). Over its open-flood GT pixels the
  post-event VH is dark (median -22.0 dB, 77 % below -20 dB; every 4th pixel of the 8-bit scene) but **VV is not** (median -15.0 dB, only
  7 % below -20 dB), whereas the water the model does find there has VV ≈ -16 to -18 dB. The model
  needs VV to be dark. Possible causes: rough or turbulent flood water and debris after the dam breach,
  flooded vegetation and fields, and VH at the noise floor (thermal noise removal is unstated, see
  `preprocessing_shift.md` §1). In the 22 chips with > 20 000 open-flood pixels the model predicts
  water on 0.06 % of the post image (median 0). The Kakhovka reservoir drained, so water(pre) > water(post)
  there: of the pixels that are water before the event, 24 % are water after.
- **Uncertainty tracks error, except at Nova Kakhovka.** Spearman correlation between a chip's
  `mc_var_flood` and its flood error rate ((FP + FN) / valid pixels) is 0.76 at Weihui, 0.59 at Jubba 1
  and 0.53 at Jubba 2, but -0.00 at Nova Kakhovka, where the model is *confidently* wrong (missed flood
  with P(post) ≈ 0). Mean predictive entropy behaves the same (0.77, 0.59, 0.52, -0.03).
- Chips vs full scene: scores differ by ≤ 0.045 F1. Chip inference (256 px with no context margin) is
  slightly worse at Weihui and slightly better at Jubba 2.

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
| compute, as run here | float32 deterministic pass; **bfloat16 autocast for the MC passes and the full scenes** while the host CPU had native bfloat16 (3.6× faster; mean abs. probability difference 6 × 10⁻⁴, ~9× below the MC spread): ~2.5-3.4 s per chip. After a container restart the host had no bfloat16 instructions (emulated bfloat16: 17.5 s/chip), so the last 461 Nova Kakhovka chips and the Nova Kakhovka full scene ran in **float32** (9.5 s/chip; 15 min per full image). Precision is recorded per chip (`det_dtype`, `mc_dtype`) and per scene (`dtype` in the JSON). **~11 h in total** | ran |
| full-scene tile, float32, 4224 px | ~15 GB of activations per tile on CPU | **no** (OOM next to the chip job). Run here with `--tile 2176` (same 64 px crop), bfloat16; the Slurm job keeps 4224 px with 32 GB |

So the full float32 run is meant for a **Slurm HPC**: `analysis/slurm/submit_usf.sh` submits, per event,
`prepare` → a `chips` job array (200 chips per task, 8 CPUs, 16 GB) → `scene`, then one `merge`.
Edit `analysis/slurm/usf_env.sh` (environment activation, `USF_DATA` on scratch) first. If compute
nodes have no internet, run `python analysis/usf_eval.py download EVENT` for each event on a login or
data-transfer node (48 GB), and set `USF_OFFLINE=1`. On GPU nodes set `USF_DEVICE=cuda` and add
`--gres=gpu:1` to `usf_chips.sbatch` / `usf_scene.sbatch`.
