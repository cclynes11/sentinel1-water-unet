# Input shift: UrbanSARFloods test set vs the model's training inputs

The U-Net (Wagner et al. 2024) was trained on 256 × 256 patches of **Sentinel-1 IW GRDH σ⁰ in dB,
10 m, from the Amazon basin**, produced by the SNAP chain in `src/s1water/preprocess.py`. This
evaluation feeds it the **pre-event and post-event intensity bands of UrbanSARFloods**, converted with
the model's own `db_to_uint8` and stacked VV, VH, VV, VH. Nothing else was changed: no resampling to
10 m, no recalibration, no speckle filter, no fine-tuning. Every difference below is therefore in the
inputs the model saw.

Sources: `src/s1water/preprocess.py` and the README (training side); Zhao et al. 2024,
*UrbanSARFloods* (arXiv:2406.04111, §3), the files themselves, and `analysis/band_identification.md`
(test side). Where the paper is silent this is marked **unstated**. An unstated item is not a measured
difference.

## 1. Product level and processing chain

| | Training inputs (model) | UrbanSARFloods intensity | Effect / note |
|---|---|---|---|
| Sentinel-1 product | **IW GRDH** (Level-1 Ground Range Detected, High resolution), dual pol VV+VH | **IW SLC** (Level-1 Single Look Complex), dual pol VV+VH, from ASF | Different Level-1 product. GRD is detected, multilooked and projected to ground range by ESA's IPF. SLC keeps complex slant-range bursts. |
| Multilooking | Done by the ESA IPF for GRDH: ~5 looks range × 1 azimuth, ENL ≈ 4.4, native resolution ≈ 20 × 22 m | **4 looks range × 1 azimuth** applied by the authors on the SLC stack ("to get square pixels"); ENL ≲ 4 (adjacent SLC samples are correlated) | Similar number of looks, so the speckle level is similar or slightly higher. Different looks geometry and filter. |
| Interferometric steps | none | Coregistration of the pre and post SLCs (needed for the coherence bands), deburst, interferogram, Goldstein filter (9 × 9, phase only), coherence 9 × 9 | The post-event (secondary) image is **resampled onto the reference geometry** before detection. This interpolation does not exist in a GRD chain. Which date is the reference is unstated. |
| Orbit | Apply-Orbit-File, precise orbits (restituted fallback) | **unstated** (orbits are needed for coregistration, so some orbit file was used) | Sub-pixel geolocation differences at most. |
| GRD border noise removal | `Remove-GRD-Border-Noise` | not applicable to SLC; burst/sub-swath seams handled by deburst | Different edge artefacts at swath borders and burst seams. |
| Thermal noise removal | `ThermalNoiseRemoval` (removeThermalNoise = true) | **unstated**; usually skipped in InSAR chains | If noise is not removed, **VH over dark surfaces (water!) sits on the noise floor (~-22 to -25 dB)** and sub-swath noise patterns are not flattened. Water then looks *brighter in VH* than in training. This matters most in VH, which drives water detection. |
| Radiometric calibration | `Calibration` to **σ⁰** (sigma nought), linear, then `LinearToFromdB` | "calibrated and transformed into intensity (in dB)": **calibration type (σ⁰, β⁰ or γ⁰) unstated** | σ⁰ = β⁰ + 10·log10(sin θ_i), i.e. β⁰ is 1.4-3.1 dB brighter over the IW swath (θ_i ≈ 29-46°). γ⁰ = σ⁰ − 10·log10(cos θ_i), i.e. γ⁰ is 0.6-1.6 dB brighter. Either would shift the whole histogram *across* the 0.5 decision boundary the model learned. Observed land medians (VV ≈ -7 dB, VH ≈ -12 to -13.5 dB at Weihui/Jubba) fit σ⁰ or γ⁰, but cannot separate them. |
| Terrain correction / geocoding | `Terrain-Correction`, Range-Doppler, **GETASSE30** DEM, bilinear resampling, WGS84 lat/lon | Geocoded to WGS84 lat/lon at 20 m; **DEM, method and resampling unstated** | Different DEM and interpolation. Small geometric effect in flat terrain (all three sites are flat), larger in relief. |
| Radiometric terrain flattening | none | unstated (none implied) | - |
| Speckle filter | none | none on intensity (the Goldstein filter acts on phase only) | Same. |
| dB conversion | 10·log10(σ⁰) | 10·log10(·), **values clipped to [-60, +6.99] dB** in the files | Harmless: `db_to_uint8` clips again to [-49, 1] dB. |

## 2. Pixel grid and resolution (20 m vs 10 m)

| | Training | UrbanSARFloods |
|---|---|---|
| CRS | EPSG:4326 | EPSG:4326 |
| Pixel spacing | SNAP default **8.983 × 10⁻⁵ °** (10.0 m at the equator); Amazon ~3° S → **10.0 × 10.0 m** | **1.6666 × 10⁻⁴ °** (1/6000°) → 18.5 m N-S for all events; E-W: Weihui (35.7° N) **15.1 m**, Nova Kakhovka (46.8° N) **12.7 m**, Jubba (≈0-1.4° N) **18.5 m** |
| Ratio to training (N-S × E-W) | 1 × 1 | Weihui 1.85 × 1.51, Nova Kakhovka 1.85 × 1.27, Jubba 1.85 × 1.85 |
| Pixel shape on the ground | square | **anisotropic** outside the tropics (Nova Kakhovka 18.5 × 12.7 m) |

Consequences:

- In pixel units every object is **1.3-1.9× smaller** than in training, and **anisotropically** at
  Weihui and Nova Kakhovka. The model's receptive field and the 256 px chip cover 1.3-1.9× more ground
  in each direction, so the chip holds ~2.4-3.4× the training area.
- Narrow water bodies (canals, flooded streets, small channels) lose pixels. The README already notes
  that 1-2 px (10-20 m) rivers are often missed at 10 m, and at 20 m the threshold in metres roughly
  doubles.
- The effective resolution is less degraded than the spacing. GRDH native resolution is ~20 × 22 m
  anyway (sampled at 10 m). The 4 × 1 multilooked SLC has ~9.3 m slant-range spacing (≈ 13-19 m in
  ground range) × 14 m azimuth spacing (~22 m azimuth resolution) before geocoding to 18.5 m. The main change is the **sampling**: training images were oversampled ~2×,
  the test images are not, so textures, edges and speckle autocorrelation look different per pixel.
- **No resampling was done here**, as requested. Resampling the test images to 8.983 × 10⁻⁵ ° (bilinear,
  like the training Terrain-Correction) would restore object sizes in pixels. It is the first ablation
  to try.

## 3. Values given to the network

| | Training | This evaluation |
|---|---|---|
| Scaling | `(max(-49, min(1, dB)) + 50) × 5` → uint8, /255 | **same function** (`s1water.preprocess.db_to_uint8`) |
| Band order | VV, VH, VV, VH | VV, VH, VV, VH (bands 6, 5 for pre; 8, 7 for post; see `band_identification.md`) |
| No data | SNAP no-data 0 dB → 250; tile padding 255 | NaN → 250 (`db_to_uint8` NaN rule). 34-46 % of each scene is NaN (outside the swath). These pixels are **excluded from every metric** but the network still sees 250 as context at chip borders |
| Values clipped by the scaling | (not reported for training) | ≤ -49 dB: < 0.001 % of valid pixels. ≥ 1 dB: VV 0.07-1.9 % (Weihui 1.8-1.9 %, urban double bounce), VH < 0.07 % |
| Precision | float16 on GPU (original), float32 port validated | deterministic chip pass **float32**. MC-dropout passes and full-scene runs **bfloat16 autocast** on CPU for speed (mean abs. probability difference vs float32 6 × 10⁻⁴, 0.06 % of pixels flip, ~9× smaller than the MC-dropout spread). The Slurm scripts default to float32 everywhere |

## 4. Context and tiling

| | Training / original inference | This evaluation |
|---|---|---|
| Training patches | 256 × 256 | - |
| Inference, chips | original: 4224 px tiles, 64 px cropped on each side (no border effect) | **each 256 px chip on its own, no crop**: predictions within ~32-64 px of a chip edge lack context. The chips are what `usf_results.csv` scores |
| Inference, full scenes | 4224 / 4096 / 64 tiling | **same tiling** (`s1water.predict.predict_image`). Used for the error maps and `scene_metrics/`. The first 64 rows/cols are unpredicted, as in the original workflow, and excluded |

## 5. Scene content, sensor and acquisition

| | Training | UrbanSARFloods test |
|---|---|---|
| Region / biome | Amazon basin: tropical forest, rivers, floodplains | Weihui (Henan, China): dense urban + irrigated cropland, monsoon flood, Jul 2021. Nova Kakhovka (Ukraine): steppe/cropland + towns, dam-breach flood, Jun 2023. Jubba (Somalia): semi-arid shrubland/cropland along the Jubba river, Dec 2023 |
| Sensor | Sentinel-1A (README: 2021-2022 images) | Sentinel-1A only (all 12 SLC files: S1A) |
| Dates per event | single dates | 3 SLCs 12-60 days apart per event (Weihui 3/15/27 Jul 2021; Nova Kakhovka 17 Mar/10 Apr/9 Jun 2023; Jubba 2 Oct/7 Nov/1 Dec 2023), giving one pre-event and one post-event intensity image (which pre date is used is unstated) |
| Orbit geometry pre/post | n/a | same relative orbit and incidence angles (repeat-pass InSAR pairs), so pre/post differences are not geometric |
| Urban areas | rare in the Amazon training area | central at Weihui and present at Nova Kakhovka; double bounce pushes VV above 1 dB (clipped by the scaling) |

## 6. Task and label shift (not preprocessing, but it limits what any score can mean)

- **The model maps water, the labels mark flood.** GT 0 includes permanent water (rivers, reservoirs,
  ponds), so raw post-event water is penalised there by design. The derived flood mask
  `water(post) AND NOT water(pre)` removes water present in both images.
- **Urban flood (GT 2) is mostly not dark.** In cities, flood water raises backscatter (double bounce:
  +3.1 dB in post VV over GT 2 at Weihui). A water segmentation model trained to find low backscatter
  cannot detect it from intensity alone. The paper's urban-flood annotation relies on the coherence drop
  (pre-event minus co-event coherence, §3.2.1), which the model never sees. Recall on class 2 is expected to be near zero and measures this task mismatch rather
  than calibration.
- **Label rule:** a pixel is flood if it is flooded "in at least one polarization" (paper §3). The model
  combines both polarisations its own way.
- **Pre-event water state**: the pre-event image is a different season (Nova Kakhovka: March/April
  vs June), so water(pre) also carries seasonal changes (reservoir level: the Kakhovka reservoir
  *drained* after the breach, so water(pre) AND NOT water(post) is large there but is not scored).

## 7. What was not changed (on purpose)

No resampling to 10 m, no recalibration (σ⁰ conversion is impossible without knowing the calibration
used and the incidence angle per pixel, which the dataset does not provide), no thermal-noise
correction, no threshold tuning (0.5 as trained), no post-processing (the scene-border mask of
`s1water.postprocess` assumes GRD scene borders and is not applicable to these mosaics; NaN pixels
are excluded instead).
