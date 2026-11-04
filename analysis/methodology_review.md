# Methodology review: Sentinel-1 water U-Net (Wagner et al., 2024)

*Reviewed: repository `s1water` v1.0.5 (commit `0205285`, fork of `fabien-w/sentinel1-water-unet`), the
arXiv preprint [2401.16393v1](https://arxiv.org/abs/2401.16393) (published as Remote Sensing 16(6):1056), and
UrbanSARFloods ([arXiv 2406.04111v1](https://arxiv.org/abs/2406.04111), CVPRW 2024) with its GitHub README.
Date of review: 2026-11-04.*

---

## 0. Summary of findings

1. **The labels are optical model outputs, not ground truth.** The training and validation masks come from an
   earlier, unpublished-in-detail water model run on Planet NICFI monthly mosaics (4.78 m), matched to Sentinel-1
   scenes acquired up to 25 days apart. The paper cites Wagner et al. (2023) for that model, but that reference is a
   forest-cover/deforestation paper for Mato Grosso, so how the water labels were made is not documented in a paper
   anyone can check. F1 = 0.93 measures agreement with that optical labeller.
2. **The reported F1 is in-distribution.** Training (59,387) and validation (6,057) patches come from the same 579
   NICFI tiles, the same years (2021–2022), the same region, the same sensor, the same preprocessing and the same
   labeller. The split procedure is not described, and the patch arithmetic suggests a patch-level split
   (§3). Train and validation F1 are identical (0.930/0.930). No test set is held out by space, time or biome. The
   only out-of-sample checks are the comparisons against JRC GSW (F1 0.708) and MapBiomas (F1 0.686) at 30 m, and
   those are confounded by differences in resolution and date.
3. **The preprocessing contract is narrow and exact.** GRDH → orbit → border noise → thermal noise → σ⁰ →
   terrain correction on GETASSE30 (no radiometric terrain flattening, no speckle filter) → dB → clip to [−49, 1] dB →
   `(x+50)*5` to uint8 (0.2 dB steps) → `/255` → channels `[VV, VH, VV, VH]`. DN 250 is used both for no data and
   for real 0 dB pixels, and the +1 dB clip saturates bright targets such as urban double bounce.
4. **Uncertainty is not implemented, but MC dropout is cheap to add.** There are five `Dropout(0.5)` layers on the
   encoder path. They do not touch the skip connections, so MC-dropout variance will under-represent fine-scale
   uncertainty. Probabilities have never been checked for calibration, and the BCE + Dice loss tends to make them
   overconfident.
5. **UrbanSARFloods does not match the model's input contract, and its labels mean something different.** The
   dataset is SLC-based (20 m, 4×1 looks, calibration type unspecified, coherence included). Its labels mark
   *flood = change*: permanent water is labelled 0, and class 2 (urban flood) is defined by a coherence drop
   inside WSF2019. An intensity-only, single-date water model cannot see class 2 by construction. Using WSF2019
   building density as a failure proxy would be circular, because WSF2019 defines the class-2 labels and is itself
   derived partly from Sentinel-1.

---

## 1. Scope, sources and assumptions

**Sources read.** All Python modules in `src/s1water/`, the scripts in `scripts/`, the tests, the SNAP graph
`snap_graphs/s1_water_preprocessing.xml`, `docs/validation_*.json`, the README and the notebook builder; the full
text of arXiv 2401.16393v1; the full text of arXiv 2406.04111v1 and the corrected event table in the UrbanSARFloods
GitHub README.

**Not available.** The original R training code, the NICFI-derived label masks, the training and validation patches
(`data_validation_for_f1_paper` is referenced by `scripts/validate_port.py` but is not in the repo), and the Keras `.h5`
weights. Nothing in this review was re-run against data. The repository's own validation JSONs are quoted as
reported.

**Assumptions** (I was asked not to raise questions during the work, so I recorded these instead):

| # | Assumption | Why / how to verify |
|---|---|---|
| A1 | The arXiv v1 text is methodologically identical to the published MDPI version. | The README cites "Table 2" for F1 while arXiv v1 has it as Table 1, so the numbering at least changed. Diff against the published PDF. |
| A2 | The code in this repository is the authoritative preprocessing and inference contract. Where it disagrees with the paper text, the code wins. | `docs/validation_preproc_snap9.json` shows bit-exact agreement with the original tiles (98 tiles), and `docs/validation_port.json` shows identical masks. The paper's "2-band (VH and VV)" order contradicts the code's VV, VH order; the tiles assert `[VV, VH, VV, VH]` (`scripts/validate_preprocessing.py`). |
| A3 | The Dice term is the RStudio-blog form, `loss = BCE + (1 − Dice)`, with Dice computed over the whole batch and smoothing 1. | The paper says only "Dice coefficient-related loss", and the README credits the RStudio blog (Falbel, 2019). The training code is not released. |
| A4 | "The model with the best accuracy was kept" means best **validation** pixel accuracy, selected over 2000 epochs. | The paper does not say whether this is training or validation accuracy. If it is validation, the validation set was used for model selection and is not a test set. |
| A5 | The 59,387/6,057 split was a random split at patch level, not at tile, date or region level. | Not stated. Inferred from the arithmetic in §3. **Highest-priority question for the authors.** |
| A6 | Some NICFI masks were paired with more than one Sentinel-1 date. | Inferred from patch counts (§3) and from the paper's statement about "natural data augmentation provided by the acquisition of sampling images at different dates". |
| A7 | The training region is the Rio Negro basin / Brazilian Amazon. | The paper says NICFI "Amazon water bodies" and the README says "trained in the Amazon". The exact footprint of the 579 tiles is not published. |
| A8 | UrbanSARFloods band order is `[pre-int VV, pre-int VH, post-int VV, post-int VH, pre-coh VV, pre-coh VH, co-coh VV, co-coh VH]`, with intensity in dB and coherence in [0, 1]. | The paper lists the bands in this order but calls the last pair "intensity acquired co-event", which looks like a typo for coherence. The dataset card does not state the order. Check by value ranges on one chip. |
| A9 | UrbanSARFloods intensity is σ⁰ without thermal-noise removal or speckle filtering. | The paper says only "calibrated and transformed into intensity (in dB)". The calibration type (σ⁰/β⁰/γ⁰), TNR and software are not stated. §11.3 quantifies what each alternative would change. |
| A10 | Sentinel-1 IW GRDH has 10 m pixel spacing, about 20 × 22 m (range × azimuth) resolution and ENL ≈ 4.4. IW SLC pixel spacing is about 2.3 m (slant range) × 14 m (azimuth). | ESA Sentinel-1 User Handbook (ESA, 2013) and product specification. These are nominal values that vary with sub-swath. |

---

## 2. Training data and label provenance

### 2.1 What the paper says (arXiv §2.5)

- **Label source:** "a water mask at 4.78 m resolution derived from a previous model using Planet NICFI
  [Wagner et al., 2023]". The dataset covers Amazon water bodies from 2015 to 2023 and is tiled on the NICFI grid.
- **Selection:** NICFI masks from 2021–2022 that were (i) fully covered by a Sentinel-1 image acquired **≤ 25 days**
  from the NICFI mosaic date and (ii) had **< 25 % cloud**. The authors then drew masks at random for each monthly
  mosaic to get **550 non-overlapping tiles**, and added **29 tiles with sandbanks** (dry season) by hand, for a
  total of **579 tiles**.
- **Co-registration:** Sentinel-1 was preprocessed (§4) and clipped to each mask's extent. The 4.78 m mask was
  "projected onto a raster with the same resolution and extent" (10 m; resampling method not stated). Both rasters
  were cropped to 1792 × 1792 px and cut into 256 × 256 patches (49 per tile).
- **Totals:** 59,387 training and 6,057 validation patches. Augmentation was random horizontal and vertical flips
  only.

### 2.2 Provenance gaps and label-noise sources

| Issue | Consequence |
|---|---|
| **The cited label model is not a water model.** Reference [Wagner et al., 2023] is *"Mapping tropical forest cover and deforestation with Planet NICFI satellite images and deep learning in Mato Grosso State (Brazil) from 2015 to 2021"* (Remote Sensing 15(2):521). | The accuracy, training data and failure modes of the labeller are undocumented. Every F1 in the paper is bounded by, and biased toward, that labeller. |
| **Optical labels cannot see water under canopy.** | Flooded forest and flooded shrub are labelled *non-water* in training. The model's "flooded vegetation omission" (§8) is partly a **label definition**, not just a SAR limitation. The model was taught to call it land. |
| **Up to 25-day offset between label and image.** The Rio Negro stage moves by metres over a season, and NICFI monthly mosaics are themselves composites of many acquisition dates. | Shoreline positions disagree between label and input, which adds noise to edge pixels and thin channels. That penalises recall on small rivers. |
| **Resampling 4.78 m → 10 m with an unstated method.** | Small channels may be removed or widened depending on the rule (nearest, majority, or any-water). This interacts with the documented 1–2 px river omission. |
| **Clouds in labels.** Below 25 % cloud is allowed, and Fig. 4f shows a cloud error in the *label*. | Some validation false negatives and false positives are label errors. They are not separated from model errors in the reported F1. |
| **Sandbank tiles chosen by hand (29).** | A small, deliberately chosen subpopulation. It helps dry-season performance but is not a random sample. |
| **Sensor:** the README says Sentinel-1A, 2021–2022. Sentinel-1B was operating until December 2021. | It is unclear whether S1B scenes are in the training set. Inference now runs on S1C/D (since 2025). The design is the same, but this is still unvalidated domain shift. |

---

## 3. Train/val split and why the reported F1 is in-distribution

**What we know.** There are 59,387 + 6,057 = 65,444 patches, a 90.7 / 9.3 % split. Each 1792 × 1792 tile yields
49 patches, so 65,444 / 49 ≈ 1,335.6 tile-equivalents from 579 masks. Either some patches were dropped (no data or
borders) or, more likely, many masks were paired with **2–3 Sentinel-1 dates** (assumption A6). The paper does not
say how patches were assigned to train or validation.

**Why the 0.93 F1 is in-distribution, and probably optimistic even within the Amazon:**

1. **Same population.** Train and validation share years, region, sensor, preprocessing chain and labeller. Nothing
   is held out by space, time, biome or label source.
2. **Spatial autocorrelation.** If the split is at patch level (A5), a validation patch's neighbours (sharing its
   river, its edges and its Sentinel-1 scene) are in training. Random splits of autocorrelated data are known to
   inflate accuracy relative to spatially blocked validation (Roberts et al., 2017; Ploton et al., 2020).
3. **Near-duplicates across dates.** If one mask was paired with several Sentinel-1 dates (A6), a validation patch
   can have *the same label* and a very similar image as a training patch from another date.
4. **Model selection on the validation set.** Per A4, the best of 2000 epochs was chosen by accuracy. If that was
   validation accuracy, the validation set is a tuning set. Accuracy is also a poor selection metric at about 3.6 %
   water prevalence: `docs/validation_preproc_*.json` reports a water fraction of 0.0356 on its 98 tiles.
5. **No generalisation gap.** Training and validation F1 are both 0.930 (precision 0.935/0.934, recall
   0.926/0.926). For a 34.6 M-parameter network trained for 2000 epochs, that is what an i.i.d. split of a homogeneous
   population looks like. It is not evidence of transfer.
6. **Agreement with the labeller, not with truth** (§2.2). Errors shared by the optical labeller and the SAR model,
   such as water under canopy, are invisible to F1.
7. **The repository check verifies the port, not generalisation.** `docs/validation_port.json` reproduces F1 0.927
   on **2,379** tiles, a subset of the paper's 6,057. The reason for the subset is not given.

**Ambiguity in the metric.** Eq. 1 defines F1 "for each sample i", but Table 1 reads like a pooled (micro) F1.
`scripts/validate_port.py` pools TP, FP and FN over all tiles. Mean-of-per-patch F1 and pooled F1 can differ a lot
when many patches contain little water.

**Independent evidence that does exist.** At 30 m and 2022–2023, agreement is F1 0.708 against JRC GSW (precision
0.614, recall 0.836) and F1 0.686 against MapBiomas 2022. The model's water area correlates with the Manaus port
stage at r = 0.887. These comparisons are out of sample, but they mix resolution (10 m vs 30 m), time (multi-year
GSW vs a 12-day mosaic) and label definition, so they bound performance loosely rather than measure it.

**What a defensible estimate would need:** a split blocked by NICFI tile or larger region, and by year
(train 2021, test 2022, or the reverse). It would also need a test set labelled independently of the NICFI
labeller, for example a manually digitised sample stratified by the proxies in §9, and area-weighted accuracy and
area estimation following Olofsson et al. (2014).

---

## 4. Preprocessing contract (exact)

The model is only valid on inputs produced *exactly* like this. File references are to this repository.

### 4.1 Scene to 8-bit raster (`src/s1water/preprocess.py`, `snap_graphs/s1_water_preprocessing.xml`)

| Step | Operator / operation | Parameters | Notes |
|---|---|---|---|
| 0 | Input | Sentinel-1 **IW GRDH, 1SDV** (VV+VH) | Other polarisation modes are rejected (`polarisations_from_name`, `run_snap_chain`). |
| 1 | Apply-Orbit-File | `Sentinel Precise (Auto Download)`, polyDegree 3, continueOnFail false | **Deviation from the paper:** this repo falls back to *restituted* orbits for scenes under about 20 days old. |
| 2 | Remove-GRD-Border-Noise | `selectedPolarisations=VH,VV`, other parameters default | |
| 3 | ThermalNoiseRemoval | `removeThermalNoise=true` | Matters most for VH over water (§11.3). |
| 4 | Calibration | `outputSigmaBand=true`, linear | **σ⁰**, not β⁰ or γ⁰. No radiometric terrain flattening. |
| 5 | Terrain-Correction | DEM **GETASSE30** (30″, about 1 km), **bilinear**, default projection **WGS84 geographic**, default spacing 10 m = 8.983 × 10⁻⁵ ° | A coarse DEM means geolocation errors in relief, and no layover or shadow masks are produced. |
| 6 | LinearToFromdB | 10·log₁₀ | |
| 7 | Subset | scene footprint (from `preview/map-overlay.kml`) or an AOI | Band order in SNAP output is VH, VV; the code reorders by band name. |
| 8 | 8-bit scaling | `DN = round_half_up((max(−49, min(1, dB)) + 50) × 5)` | Range **5–255**, step **0.2 dB**. `NaN → 0 dB → 250`. Rounds to nearest, to match GDAL Byte conversion (`db_to_uint8`, `preprocess.py:172`). |
| — | **No speckle filter, no multitemporal filtering, no incidence-angle normalisation** | | The model learned on raw GRDH speckle (ENL ≈ 4.4). |

Reproducibility: SNAP 9 reproduces the original tiles bit-exactly. SNAP 14 gives a median difference of 0.36 dB
(p99 1.79 dB), within 1 DN on only 43 % of pixels, but water masks agree on 99.93 % of pixels (IoU 0.98)
(`docs/validation_preproc_snap14.json`).

### 4.2 Raster to tensor (`src/s1water/predict.py`)

- Two uint8 bands, **band 1 = VV, band 2 = VH**, shape `(2, H, W)`.
- `x = uint8 / 255.0` (`predict.py:99`). There is **no mean/std normalisation**, so the model's effective input is
  `(dB + 50) / 51`.
- Channels are duplicated to **`[VV, VH, VV, VH]`** (`predict.py:100`). The original model was built with 4 input
  channels, and the duplication is part of the contract. Feeding `[VV, VH, X, Y]` with different X, Y is off-contract.
- Tiling: padding to `k·4096 + 128` with **fill 255** (= +1 dB, a bright value), 4224 × 4224 tiles at a 4096 stride,
  and **64 px cropped** on each side. The first and last 64 rows and columns of an AOI are no data.
- Precision: **float16 on GPU** (the whole model is cast); the sigmoid is computed in float32.
- The theoretical receptive field is about 470 px (≈ 4.7 km). The 64 px crop is smaller than half of it, so pixels
  near a tile edge see zero-padded context. This matches the original workflow (part of the contract, not a bug),
  but probabilities near 4096-px tile seams carry slightly different context.

### 4.3 Contract hazards

- **DN 250 is overloaded.** It means both no data and a real 0 dB. `valid_data_mask` resolves this only at a 16×
  coarse level (`postprocess.py`). Any external data with real 0 dB targets, or NaN in the middle of a scene, will
  be miscoded.
- **The +1 dB clip saturates bright scatterers.** Urban double bounce and corner reflectors often exceed +1 dB and
  collapse to DN 255. That information is unrecoverable (relevant to §11).
- **Below −49 dB all values clip to DN 5.** After TNR, VH over calm water can approach this floor.
- **The grid must be geographic at 10 m spacing.** The model has never seen UTM or 20 m grids. Feature scale in
  pixels is what matters (§11.2).

---

## 5. Architecture (`src/s1water/model.py`)

- A classic U-Net (Ronneberger et al., 2015), ported from R/Keras `get_unet_128()`.
- **Encoder:** 5 levels of 32, 64, 128, 256 and 512 filters. Each level is 2 × [Conv 3×3 (same) → BatchNorm → ReLU],
  then MaxPool 2×2, then **Dropout(0.5)** (`model.py:73`). The skip connection is taken *before* pooling and dropout
  (`model.py:72`).
- **Centre:** 2 × [Conv 3×3 → BN → ReLU] with 1024 filters.
- **Decoder:** 5 levels, each **nearest-neighbour upsampling ×2** (not transposed convolution), then
  `concat[skip, up]`, then **3** × [Conv 3×3 → BN → ReLU] (512 down to 32 filters).
- **Head:** Conv 1×1 to 1 channel → sigmoid in float32.
- 34,600,641 parameters, 28 conv layers (`tests/test_model.py`). BN ε = 1e-3 and momentum 0.01 (Keras 0.99), to
  match Keras.
- Trained on 256 × 256 patches and run fully convolutionally on 4224 × 4224 tiles. BN running statistics come from
  256-px batches. That is fine in eval mode, but it matters if BN is accidentally left in train mode during MC
  sampling (§10).
- There are no attention modules, no deep supervision and no auxiliary inputs (DEM, incidence angle, time series).

## 6. Loss and optimisation

- **Loss:** binary cross-entropy + Dice-based loss (assumption A3: `BCE + (1 − Dice)`).
- **Optimiser:** Adam, lr 1e-4. The paper also writes "standard stochastic gradient descent", which presumably means
  mini-batch training with Adam. Batch size 32, **2000 epochs**, about 90 GPU-hours on an A10G.
- **Selection:** the checkpoint with the best accuracy (assumption A4).
- **Augmentation:** horizontal and vertical flips only. No rotation, no radiometric jitter, no speckle augmentation,
  no incidence-angle augmentation. The authors rely on multiple dates as natural augmentation.
- **Class imbalance:** no explicit weighting; the Dice term partly compensates.
- **Implication for calibration:** Dice-trained segmentation networks tend toward overconfident, near-binary
  probabilities (Mehrtash et al., 2020). Expect a bimodal probability histogram and poor reliability near 0.5.

## 7. Thresholding and post-processing

- **Threshold:** a fixed **p ≥ 0.5** in inference (`predict.py:32`, `pipeline.py:98`). The validation script uses a
  strict **p > 0.5** (`scripts/validate_port.py:102`), following the original R code. The difference is immaterial in
  practice but should be harmonised. The threshold was never tuned on precision–recall, never adapted per scene and
  never calibrated.
- **Output coding:** 0 = not water, 1 = water, 255 = no data. Probabilities are saved only with `--probability`.
- **Valid-data mask:** built on a 16× coarse grid. A cell is no data only if all of its pixels are 250. Holes are
  filled, then scene edges are eroded by **1000 m** in the "range" direction and **500 m** in "azimuth". The
  erosion runs along image x and y of a geographic grid, which is only an approximation of range and azimuth for a
  near-polar orbit about 10° off meridian.
- **Shade mask (optional, off by default):** SRTM 1″ aggregated by min to 90 m, slope ≥ 20°, hole-filled, convex
  hulls. The paper's **manual edits** of shade polygons over rivers are not reproduced.
- **Not reproduced from the paper:** the convective-cloud correction (§2.7 of the paper: anomaly detection on 68
  tiles, 111 images replaced by hand from the previous mosaic) and the 12-day max-composite mosaicking.

---

## 8. Documented failure modes

| # | Failure mode | Direction | Source in paper / repo | Mechanism (per the authors) |
|---|---|---|---|---|
| F1 | Small rivers 1–2 px (10–20 m) | Omission | arXiv §3.1, Fig. 3–4; README | Below effective GRDH resolution (about 20 m); worsened by label resampling. |
| F2 | Flooded forest, woodland and shrub | Omission | §3.2–3.4, §4.2, Table 2; README | C-band volume and double-bounce scattering. **Also a label definition** (§2.2). Against GSW, flooded woodland plus seasonally flooded shrub make up 36.7 % of false negatives; against MapBiomas, 47.6 %. |
| F3 | Convective rain cells | Omission | §2.7, §4.2; README | VV/VH increase over water (Doblas et al., 2020). 111 images were corrected by hand. |
| F4 | Mountain radar shadow | Commission | §2.3, §3.1, §4.2; README | Shadow looks like water. The GETASSE30 DEM is too coarse to model it. |
| F5 | Airport runways and wide roads (about 4 lanes) | Commission | §3.1, §4.2 | Specular reflection from smooth surfaces. |
| F6 | Urbanised riverfront (Manaus) | Omission | §3.1, §4.2 | Bright urban context and missing urban training samples. |
| F7 | Same-orbit slice boundary (about 200 m band) | Omission | §3.5 Fig. 6f, §4.2 | No overlap between consecutive slices; border effect from the model, the raw data or GRD border-noise removal. |
| F8 | Very low water, shoaling, wind roughening; sandbank look-alikes | Omission | §3.5 Fig. 6b, §4.2 | Bragg roughening raises VV (Pham-Duc et al., 2017); shallow water and sand confusion. |
| F9 | Orbit overlap areas and GRD border noise | Omission | §3.5, §4.2 | Border-noise removal artefacts. |
| F10 | Other biomes, arid land, cities | Unevaluated | README "Limitations" | Out of domain. In C-band, dry sand looks dark, so commission is expected (Martinis et al., 2018). |
| F11 | Sensor change (S1A to S1C/D) and SNAP version | Unvalidated shift | README | Same design by intent. SNAP 14 vs 9 differs by about 3 m sub-pixel shift and 0.36 dB median. |
| F12 | Label errors (clouds in NICFI) | Evaluation artefact | §3.1, Fig. 4f | Inflates apparent model error. |

---

## 9. Failure modes mapped to measurable geospatial proxies (open data)

Each proxy can be computed per pixel or per patch and used to (a) stratify the error analysis, (b) define an area of
applicability (Meyer & Pebesma, 2021), or (c) serve as covariates for an error or uncertainty model.

| Failure mode | Proxy (what to compute) | Open dataset | Suggested derivation | Citation(s) |
|---|---|---|---|---|
| F4 shadow / layover | **Slope**, **aspect relative to the look direction**, **local incidence angle**, geometric shadow/layover flag | Copernicus DEM GLO-30; SRTM 1″; FABDEM (forest- and building-removed) | Horn slope at 30 m, then a shadow flag where local incidence ≥ 90° and a layover flag where foreshortening is negative, using scene heading and incidence from annotation XML. | ESA (2021) Copernicus DEM; Farr et al. (2007); Small (2011); Hawker et al. (2022) |
| F4, F10 terrain implausibility | **HAND** (height above nearest drainage) | MERIT Hydro `hnd` band; or HAND computed from FABDEM | Per-pixel HAND; flag water predicted at high HAND as implausible (exclusion-mask logic). | Rennó et al. (2008); Nobre et al. (2011); Yamazaki et al. (2019); Twele et al. (2016) |
| F2 flooded vegetation | **Land cover** (tree, shrub, herbaceous wetland classes), **NDVI** at the image date, **canopy height**, **wetland class** | ESA WorldCover 2021 (10 m); Dynamic World (S2-only, near-daily); Sentinel-2 L2A NDVI; GEDI-based canopy height; LBA-ECO LC-07 wetlands (Amazon) | Fraction of WorldCover class 10/20/90 per patch; nearest cloud-free S2 NDVI within ±10 days; canopy height above 5 m; LC-07 "flooded forest/shrub" flag. | Zanaga et al. (2022); Brown et al. (2022); Drusch et al. (2012); Tucker (1979); Potapov et al. (2021); Hess et al. (2015); Tsyganskaya et al. (2018); Hess et al. (1995) |
| F3 convective rain | **Precipitation rate at overpass time** | GPM IMERG half-hourly, 0.1°; CHIRPS daily (too coarse in time for single cells, so a fallback only) | IMERG rate in the 30 min slot containing the S1 sensing time, and its maximum within ±1 h. | Huffman et al. (IMERG V07); Funk et al. (2015); Doblas et al. (2020); Danklmayer et al. (2009) |
| F1 small rivers | **Channel width**, or, below width-product limits, **upstream drainage area** and **distance to the nearest mapped channel** | GRWL (Landsat, ≥ 30 m); SWORD (SWOT reaches); MERIT Hydro width and upstream area; OSM waterways for less than 30 m | Width at the nearest centreline; for channels narrower than 30 m use MERIT upstream-area classes or OSM `waterway=stream/river`. | Allen & Pavelsky (2018); Altenau et al. (2021); Yamazaki et al. (2019); OpenStreetMap contributors |
| F6 urban riverfront | **Building density / built-up fraction** and **distance from the water edge to the nearest building** | Google Open Buildings (optical, sub-metre); GHS-BUILT-S R2023A (optical); OSM buildings; WSF2019 *(not to be used with UrbanSARFloods labels, see §11.6)* | Building footprint area per 100 m cell; buffer distance from predicted or reference water. | Sirko et al. (2021); Pesaresi & Politis (2023); Marconcini et al. (2021) |
| F5 runways and roads | **Runway polygons**, **road density** | OSM `aeroway=runway`, `highway=*`; GRIP global roads | Rasterise runways; road length per km² for major classes. | OpenStreetMap contributors; Meijer et al. (2018) |
| F8 wind roughening | **10 m wind speed** at overpass (and fetch) | ERA5 hourly single levels | Wind speed interpolated to the sensing time; flag above a few m/s over open water. | Hersbach et al. (2020); Pham-Duc et al. (2017) |
| F8 shoaling / sandbanks / transitional water | **Water-occurrence percentile** (transition zone), **river stage** at the image date | JRC Global Surface Water occurrence; gauge records (e.g. the Manaus port gauge used by the paper) | 0 < occurrence < 100 marks the seasonal zone; stage anomaly relative to the seasonal median. | Pekel et al. (2016); Wagner et al. (2024) |
| F7, F9 scene, slice and orbit borders | **Distance to scene and slice edges**, **orbit-overlap flag** | Sentinel-1 product footprints (manifest/KML, ASF catalogue) | Distance in metres from each pixel to the nearest slice edge; count of overlapping orbits. | Filipponi (2019) |
| F10 arid / sand look-alikes | **Aridity index**, **bare/sparse cover fraction** | Global Aridity Index v3; WorldCover class 60 | Per-patch aridity; bare-cover fraction. | Zomer et al. (2022); Martinis et al. (2018); Zhao et al. (2021) |

**Caveats on the proxies.**

- **DEM in forest.** Copernicus DEM and SRTM are surface models. Over the Amazon canopy, HAND and slope are noisy:
  canopy steps, and pits over water. The repo already turns its SRTM shade mask off for this reason. Use FABDEM or
  MERIT Hydro for HAND in forest. FABDEM's licence is non-commercial (CC BY-NC-SA).
- **WorldCover also uses Sentinel-1.** ESA WorldCover is classified from Sentinel-1 and Sentinel-2. As a vegetation
  proxy it is not fully independent of the SAR input. Dynamic World (Sentinel-2 only) is the independent
  alternative.
- **Building density from WSF2019 is circular** for any analysis against UrbanSARFloods labels, and partly circular
  for the model input (§11.6).

---

## 10. What the model needs for uncertainty estimates

### 10.1 MC dropout: feasible, with known limits

**What exists.** Five `nn.Dropout(0.5)` layers sit after each encoder pooling (`model.py:56, 73`). They are
element-wise, matching Keras `Dropout` on 4-D tensors. The weights were trained with dropout active, so test-time
dropout is the approximate-Bayesian setting of Gal & Ghahramani (2016).

**What to implement:**

1. Keep **BatchNorm in eval mode** and switch **only `nn.Dropout` modules** to train mode. `predict_image` calls
   `model.eval()` internally (`predict.py:96`), so this needs a dedicated function rather than a flag set
   beforehand. `torch.inference_mode()` is compatible with dropout sampling.
2. For *T* stochastic passes (T ≈ 10–30) per tile, keep running sums of `p` and of `p·log p` per pixel. This avoids
   holding T full-size probability maps.
3. Export per pixel:
   - predictive mean p̄;
   - predictive entropy H[p̄];
   - expected entropy E_t[H(p_t)], the aleatoric part;
   - **mutual information** H[p̄] − E_t[H(p_t)], the epistemic part, also called BALD;
   - optionally the standard deviation of p.
4. Expose **logits** (an option in `forward` to skip the sigmoid) so that calibration (§10.2) can work in logit
   space without fp16 saturation.
5. Cost scales linearly with T. The README's 55 × 55 km AOI takes about 55 s, so T = 20 is roughly 20 minutes on
   one T4. Reduce cost by sampling only tiles that contain water or that sit within the 0.05 < p̄ < 0.95 band of a
   deterministic pass.

**Known limits:**

- **Skip connections are deterministic.** Dropout perturbs only the down-path and the bottleneck. The full-resolution
  skip from level 1 is untouched, so MC variance will **under-represent uncertainty at edges and in 1–2 px channels**
  (F1), which is where the errors are.
- **p = 0.5 is aggressive at inference.** Samples may be noisier than the deterministic prediction. Check that
  p̄ ≈ p_det on validation tiles before trusting the variance.
- **MC dropout is not calibration.** Under dataset shift, MC dropout and temperature scaling both degrade, and deep
  ensembles tend to hold up better (Ovadia et al., 2019; Lakshminarayanan et al., 2017). Ensembles need retraining,
  and the training code is not released.
- **Flip test-time augmentation** is a cheap complement. Flips were the training augmentation, so flipped inputs are
  in-distribution. It captures some input-equivariance uncertainty, not model uncertainty.
- **Aleatoric heads** (predicting a variance, as in Kendall & Gal, 2017) need retraining.

### 10.2 Calibration

1. **Data needed:** a labelled set with per-pixel probabilities that is **spatially and temporally independent** of
   the training tiles (§3). Ideally, add an out-of-domain set stratified by the §9 proxies. Calibrating on the
   in-distribution validation tiles fixes in-distribution calibration only.
2. **Diagnostics:** reliability diagrams, ECE (binned, and **class-conditional** because water prevalence is about
   3.6 %), Brier score and NLL. Report them per proxy stratum (vegetation, rain, slope, built-up), not just pooled.
3. **Recalibration:** temperature scaling on logits (Guo et al., 2017) is one parameter and keeps the ranking.
   Fit it on a calibration split and evaluate on a disjoint one. If reliability differs by stratum, fit
   **stratified temperatures** or isotonic regression on (logit, proxy).
4. **Threshold after recalibration:** p = 0.5 on calibrated probabilities is no longer F1-optimal by construction.
   Choose the threshold for the use case (area estimation needs unbiased area; alerting needs high recall) and
   record it in the GeoTIFF tags. They currently hard-code `threshold="0.5"` in `pipeline.write_mask`.
5. **Area-level uncertainty:** pixel calibration does not give confidence intervals on water area, because errors are
   spatially correlated. Use stratified-sample area estimation (Olofsson et al., 2014), with strata from p̄ and the
   proxies.
6. **Optional:** conformal prediction on calibrated scores for pixel-level coverage guarantees. This is only valid
   under exchangeability with the calibration set, so the same spatial-independence requirement applies.

### 10.3 What has to come from the authors

The validation patches with **geolocation and acquisition dates**, the patch-to-tile-to-date mapping, the training
code (for ensembles and retraining with logits or aleatoric heads), and ideally the NICFI-derived masks. NICFI
licence terms may restrict redistribution of the masks.

---

## 11. Comparison with the UrbanSARFloods input contract

### 11.1 Side-by-side

| Aspect | Wagner U-Net (this repo) | UrbanSARFloods (Zhao et al., 2024) |
|---|---|---|
| Product | IW **GRDH** (detected, ground range) | IW **SLC** (complex, slant range) |
| Multilooking | Done in the ESA IPF: 5 × 1 looks, ENL ≈ 4.4, about 20 × 22 m resolution | **4 (range) × 1 (azimuth)** looks on SLC (about 2.3 × 14 m spacing), giving about 9 m slant ≈ 13–20 m ground × 14 m ("square pixels"); ENL ≤ 4 before geocoding |
| Output grid | WGS84 geographic, **10 m** spacing | WGS84 geographic, **20 m** spacing |
| Calibration | **σ⁰** (SNAP), after **thermal-noise removal** | "calibrated … intensity (in dB)". **σ⁰/β⁰/γ⁰ not stated; TNR not stated** |
| Terrain correction | Range–Doppler with GETASSE30 | Geocoding; DEM not stated |
| Speckle filter | None | None stated for intensity; **Goldstein filter (9 × 9) on the interferogram** |
| Value encoding | 8-bit `(clip(dB,−49,1)+50)·5`, then /255 | Float dB intensity; coherence in [0, 1] |
| Bands | 2 (VV, VH), fed as `[VV,VH,VV,VH]` | **8**: pre/post intensity VV, VH; pre/co-event coherence VV, VH (A8) |
| Temporal | **Single date** | **Bi-temporal**: two pre-event SLCs and one post-event SLC per event |
| Coherence | Not used | 9 × 9 window after Goldstein filtering |
| Label semantics | **Water** (any open surface water, permanent or not) | **Flood = change**: 0 non-flood (**includes permanent water**), 1 open flood, 2 urban flood |
| Label generation | Optical (Planet NICFI) model, ≤ 25 days offset | Semi-automatic from **the SAR data itself**: HSBA change detection on intensity (class 1); Δcoherence > 0.3 inside **WSF2019** built-up (class 2); manual small-object removal. Manual labels (PlanetScope 3 m, UAV) for 3 test events |
| Patch size | 256 × 256 at 10 m (2.56 km) | 512 × 512 at 20 m (10.24 km); baselines train on random 256 crops |
| Split | Unspecified, probably random patch-level (A5) | Stratified random 70/30 within 15 events (train/val) **plus 3 held-out test events** (Weihui, Jubba, NovaKakhovka) |
| Domain | Amazon, tropical forest and rivers | 18–20 events on 5 continents, mostly urban/peri-urban floods |

### 11.2 Resolution and multilooking

- **The information content is closer than the pixel spacing suggests.** GRDH's 10 m pixels oversample about
  20 × 22 m resolution, while UrbanSARFloods' 20 m pixels slightly undersample about 15 × 14 m multilooked SLC.
  Both carry information at roughly 15–22 m.
- **Feature scale in pixels is what the CNN learned.** The model's filters and its 470 px receptive field are tuned
  to 10 m spacing. A 40 m river is 4 px in training but 2 px in UrbanSARFloods. To feed the model, **resample
  UrbanSARFloods to 10 m**. Bilinear on linear power, then convert to dB, avoids dB-averaging bias. Upsampling adds
  no information and produces spatially correlated pixels, so speckle texture will look smoother than GRDH. That is
  a texture domain shift the model has never seen.
- **Speckle statistics differ.** GRDH is ENL ≈ 4.4 per resolution cell with spatially correlated 10 m samples. The
  4 × 1 SLC is ENL ≈ 4, geocoded to 20 m (ENL rises if geocoding averages). The model was trained without speckle
  augmentation, so expect sensitivity.

### 11.3 Calibration and radiometry: how far off-contract each unknown would put the input

| If UrbanSARFloods intensity is… | Offset vs the model's σ⁰ | In model DN (5 DN per dB) |
|---|---|---|
| σ⁰ | 0 | 0 |
| β⁰ (σ⁰ = β⁰ · sin θ) | β⁰ is **+1.4 to +3.0 dB** brighter at θ = 30°–46° | +7 to +15 DN, an incidence-dependent across-swath ramp |
| γ⁰ ellipsoid (γ⁰ = σ⁰ / cos θ) | **+0.6 to +1.6 dB** | +3 to +8 DN |
| γ⁰ terrain-flattened (Small, 2011) | Depends on terrain; large on slopes | Variable |
| **No thermal-noise removal** | VH over calm water is floored near the NESZ (about −22 dB), and sub-swath scalloping appears | Water in VH reads **brighter** than in training, pushing it toward non-water |

The thermal-noise question matters most. In training, VH over water was TNR-corrected and often well below the
noise floor. Untreated SLC VH over a flood would sit at the noise floor and partially lose the contrast the model
relies on. **Verify A9 before any transfer experiment**, for example by checking the VH histogram over permanent
water (JRC occurrence = 100) in a pre-event chip.

Other contract items:

- **NaN and no data:** map them explicitly. Do not let NaN become 0 dB → DN 250.
- **Quantisation:** apply `db_to_uint8` exactly.
- **Channel construction:** to stay on contract, feed one date as `[VV, VH, VV, VH]`. Putting pre-event bands in
  channels 3–4 is off-contract, because the model has only ever seen duplicates there.

### 11.4 Water vs flood: label mismatch

- **The model outputs water; UrbanSARFloods labels flood.** Permanent rivers and lakes are labelled **0** in
  UrbanSARFloods. Running the model on the post-event image and scoring it against the labels turns every permanent
  water pixel into a "false positive".
- **Minimum adapter:** `flood_hat = water(post) ∧ ¬water(pre)`, applying the model to both dates. An alternative is
  `water(post) ∧ ¬JRC_permanent`. Bi-temporal differencing **doubles the error exposure**: errors on either date
  propagate, and speckle-driven flips do not cancel.
- **Class 1 (open flood) is the only class the model can be expected to detect**, and only where flood water is
  open, calm and specular. HSBA labels flag intensity *decrease*, so flooded vegetation (which brightens through
  double bounce) is also largely outside class 1. That is consistent with the model's blind spot (F2), but the
  agreement comes from both sides missing the same thing, which is not a strength.
- **Class 2 (urban flood) should be scored separately as out of scope** or merged with 0, with that choice stated.
  Reporting a 3-class F1 for a binary water model is not meaningful.
- **Label independence:** UrbanSARFloods semi-automatic labels come from the **same SAR acquisitions**, by
  thresholding. Agreement with them partly measures agreement with HSBA. Only the manually labelled test subsets
  (Weihui, Jubba, NovaKakhovka) are independent of SAR.

### 11.5 Why intensity-only models struggle with urban flood

1. **Urban flood often brightens.** Water in streets creates a water–wall dihedral, so partially submerged buildings
   show *increased* backscatter through double bounce. An intensity water model looks for darkening (Chini et al.,
   2019; Zhao et al., 2024 review).
2. **Strength depends on orientation and look direction.** Double-bounce enhancement varies with building
   orientation relative to the line of sight and with polarisation. Streets parallel to the LOS behave differently
   from perpendicular ones, so a single-date intensity rule is unstable (Pelich et al., 2022).
3. **Shadow and layover hide streets.** At 10–20 m, narrow streets sit in building shadow or layover, so flood water
   there is often not observable in intensity at all.
4. **Intensity change can be too small to detect.** Coherence drops sharply when water replaces a stable
   built-up scatterer even when intensity barely changes. This is why UrbanSARFloods includes coherence and why class
   2 is *defined* by Δcoherence (Chini et al., 2019; Zhao et al., 2024).
5. **Model-specific problems.** The Wagner model (a) has almost no urban training data (the Manaus omission, F6),
   (b) clips at **+1 dB**, so bright double-bounce returns saturate at DN 255 and the brightening signal is erased,
   and (c) has no coherence input. It cannot detect class 2 by design. Even the UrbanSARFloods baselines with
   coherence reach F1 below 0.1 on class 2 (their Table 4), so this is hard for any model, not only an out-of-domain
   one.

### 11.6 WSF2019 circularity if building density is used as a proxy

- **Labels are a function of WSF2019.** Class 2 can exist only where WSF2019 = built-up (§3.2.1 of the UrbanSARFloods
  paper). Any analysis of "error vs building density" using WSF2019 is **tautological** for class-2 recall: outside
  WSF2019 there are no class-2 labels to miss, and inside it the class definition changes.
- **WSF2019 is itself derived from Sentinel-1** (2019 multitemporal backscatter statistics, with Sentinel-2)
  (Marconcini et al., 2021). Its density is therefore correlated with high mean backscatter in the very inputs being
  analysed. A WSF-based proxy is not independent of either the input or the label.
- **Temporal mismatch.** WSF2019 is a 2019 snapshot, while the events run from 2016 to 2023. Urban growth or change
  before or after 2019 is misclassified, and errors there would be wrongly blamed on "urban density".
- **Mitigations:**
  - Use a building-density proxy that is **independent of SAR and of the label rule**: Google Open Buildings or OSM
    footprints, or optical GHS-BUILT-S.
  - Report results **conditioned on** the WSF mask rather than "explained by" it.
  - Run a sensitivity analysis that swaps WSF2019 for an independent built-up layer when defining strata.
  - Prefer the **manually labelled** test events for any urban-density conclusions.
  - Note that **ESA WorldCover's built-up class also uses Sentinel-1**, so it does not break the circularity.

### 11.7 If a transfer experiment is run

1. Verify A8 and A9 from the data: band value ranges and the VH histogram over permanent water.
2. Convert intensity to σ⁰ dB if needed. Resample to 10 m with bilinear on linear power. Apply `db_to_uint8`.
   Build `[VV, VH, VV, VH]` per date.
3. Predict pre and post. Compute `flood_hat = water(post) ∧ ¬water(pre)`.
4. Score **class 1 only**, per event, on the 3 test events, against **both** the semi-automatic and the manual
   labels. Report class 2 recall separately as a known blind spot.
5. Stratify by the §9 proxies using independent building data.

---

## 12. Open questions for my supervisor

1. **Split protocol.** Should I ask the authors directly how the 59,387/6,057 split was made (patch, tile, date or
   region) and whether masks were paired with several Sentinel-1 dates? Without that, can we describe 0.93 only as
   "in-distribution agreement with the NICFI labeller"?
2. **Label model provenance.** The cited source for the water labels is a forest-cover paper. Is there an
   unpublished report on the NICFI water model, and do we treat its masks as reference data or as a second model?
3. **Independent test set.** Is it worth building a small manually digitised test set (e.g. stratified by flooded
   vegetation, rain, slope and urban fraction), and from which imagery, given NICFI licence limits?
4. **Scope of "water".** For our purposes, is flooded vegetation in scope? If so, the current model is the wrong
   starting point and an L-band or coherence-based approach should be considered (NISAR L-band, launched in 2025).
5. **Uncertainty method.** Is MC dropout acceptable as a first pass, given that it misses fine-scale uncertainty
   through deterministic skips? Or should we push for the training code so we can run ensembles?
6. **Calibration target.** Should calibration optimise pixel reliability (ECE) or unbiased area estimates
   (Olofsson-style)? The two lead to different thresholds and different validation designs.
7. **UrbanSARFloods use.** Is the goal to (a) test transfer of the Wagner model to open-area floods, (b) train a new
   model on UrbanSARFloods, or (c) use UrbanSARFloods only as an out-of-domain stress test? Each needs a different
   adapter and metric.
8. **Calibration type in UrbanSARFloods.** Should I contact the dataset authors to confirm σ⁰/β⁰/γ⁰ and thermal-noise
   removal, or reprocess from the released SLCs with a documented chain (SNAP: TOPS split → orbit → calibration σ⁰ →
   deburst → multilook → TC) so that both datasets share a contract?
9. **Re-processing to a GRD contract.** Alternatively, should we generate GRDH-equivalent inputs for the UrbanSARFloods
   event dates with this repo's exact chain? The GRDH products should exist for the same passes. That removes most
   radiometric mismatch and leaves only the label-semantics mismatch.
10. **Building-density proxy.** Which independent building layer is acceptable (Open Buildings, OSM, GHS-BUILT-S),
    given its coverage gaps and dates, so we avoid the WSF2019 circularity?
11. **Precipitation proxy.** Is IMERG at 0.1° and 30 min an acceptable proxy for convective cells that are often
    smaller than one IMERG cell? Or should we use radar where available, or detect the signature in the SAR itself?
12. **Sensor shift.** Should we budget a check of S1A-era versus S1C/D-era predictions over stable permanent water
    (JRC occurrence = 100) before using 2025+ scenes?

---

## References

*DOIs are given where I could match them reliably. Other entries give full bibliographic details instead.*

- Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. *Science*, 361(6402), 585–588. doi:10.1126/science.aat0636
- Altenau, E. H., et al. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD). *Water Resources Research*, 57, e2021WR030054. doi:10.1029/2021WR030054
- Brown, C. F., et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping. *Scientific Data*, 9, 251. doi:10.1038/s41597-022-01307-4
- Chini, M., Hostache, R., Giustarini, L., & Matgen, P. (2017). A hierarchical split-based approach for parametric thresholding of SAR images: flood inundation as a test case. *IEEE TGRS*, 55(12), 6975–6988. doi:10.1109/TGRS.2017.2737664
- Chini, M., Pelich, R., Pulvirenti, L., Pierdicca, N., Hostache, R., & Matgen, P. (2019). Sentinel-1 InSAR coherence to detect floodwater in urban areas: Houston and Hurricane Harvey as a test case. *Remote Sensing*, 11(2), 107. doi:10.3390/rs11020107
- Danklmayer, A., Döring, B. J., Schwerdt, M., & Chandra, M. (2009). Assessment of atmospheric propagation effects in SAR images. *IEEE TGRS*, 47(10), 3507–3518. doi:10.1109/TGRS.2009.2022271
- Doblas, J., Carneiro, A., Shimabukuro, Y., Sant'Anna, S., & Aragão, L. (2020). Assessment of rainfall influence on Sentinel-1 time series on Amazonian tropical forests aiming deforestation detection improvement. *IEEE Latin American GRSS & ISPRS Remote Sensing Conference (LAGIRS)*.
- Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25–36. doi:10.1016/j.rse.2011.11.026
- ESA (2013). *Sentinel-1 User Handbook*, GMES-S1OP-EOPG-TN-13-0001.
- ESA (2021). *Copernicus DEM: Copernicus Digital Elevation Model*. doi:10.5270/ESA-c5d3d65
- Falbel, D. (2019). Image segmentation with U-Net. RStudio AI Blog. https://blogs.rstudio.com/ai/posts/2019-08-23-unet/
- Farr, T. G., et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics*, 45, RG2004. doi:10.1029/2005RG000183
- Filipponi, F. (2019). Sentinel-1 GRD preprocessing workflow. *Proceedings*, 18(1), 11. doi:10.3390/ECRS-3-06201
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations (CHIRPS). *Scientific Data*, 2, 150066. doi:10.1038/sdata.2015.66
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML*, PMLR 48, 1050–1059.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, PMLR 70, 1321–1330.
- Hawker, L., et al. (2022). A 30 m global map of elevation with forests and buildings removed (FABDEM). *Environmental Research Letters*, 17, 024016. doi:10.1088/1748-9326/ac4d4f
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. doi:10.1002/qj.3803
- Hess, L. L., Melack, J. M., Filoso, S., & Wang, Y. (1995). Delineation of inundated area and vegetation along the Amazon floodplain with the SIR-C synthetic aperture radar. *IEEE TGRS*, 33(4), 896–904. doi:10.1109/36.406675
- Hess, L. L., et al. (2015). *LBA-ECO LC-07 Wetland Extent, Vegetation, and Inundation: Lowland Amazon Basin*. ORNL DAAC. doi:10.3334/ORNLDAAC/1284
- Huffman, G. J., et al. *GPM IMERG Final Precipitation L3 Half Hourly 0.1° × 0.1° V07*. NASA GES DISC. doi:10.5067/GPM/IMERG/3B-HH/07
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Marconcini, M., Metz-Marconcini, A., Esch, T., & Gorelick, N. (2021). Understanding current trends in global urbanisation: the World Settlement Footprint suite. *GI_Forum*, 9(1), 33–38. doi:10.1553/giscience2021_01_s33
- Martinis, S., Plank, S., & Ćwik, K. (2018). The use of Sentinel-1 time-series data to improve flood monitoring in arid areas. *Remote Sensing*, 10(4), 583. doi:10.3390/rs10040583
- Mehrtash, A., Wells, W. M., Tempany, C. M., Abolmaesumi, P., & Kapur, T. (2020). Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE Transactions on Medical Imaging*, 39(12), 3868–3878. doi:10.1109/TMI.2020.3006437
- Meijer, J. R., Huijbregts, M. A. J., Schotten, K. C. G. J., & Schipper, A. M. (2018). Global patterns of current and future road infrastructure. *Environmental Research Letters*, 13, 064006. doi:10.1088/1748-9326/aabd42
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633. doi:10.1111/2041-210X.13650
- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: fully convolutional neural networks for volumetric medical image segmentation. *3DV*.
- Nobre, A. D., et al. (2011). Height Above the Nearest Drainage – a hydrologically relevant new terrain model. *Journal of Hydrology*, 404(1–2), 13–29. doi:10.1016/j.jhydrol.2011.03.051
- Olofsson, P., et al. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42–57. doi:10.1016/j.rse.2014.02.015
- OpenStreetMap contributors. OpenStreetMap. https://www.openstreetmap.org (ODbL).
- Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS*.
- Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature*, 540, 418–422. doi:10.1038/nature20584
- Pelich, R., Chini, M., Hostache, R., Matgen, P., Pulvirenti, L., & Pierdicca, N. (2022). Mapping floods in urban areas from dual-polarization InSAR coherence data. *IEEE Geoscience and Remote Sensing Letters*, 19.
- Pesaresi, M., & Politis, P. (2023). *GHS-BUILT-S R2023A – GHS built-up surface grid, derived from Sentinel-2 composite and Landsat, multitemporal (1975–2030)*. European Commission, JRC. doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA
- Pham-Duc, B., Prigent, C., & Aires, F. (2017). Surface water monitoring within Cambodia and the Vietnamese Mekong Delta over a year, with Sentinel-1 SAR observations. *Water*, 9(6), 366. doi:10.3390/w9060366
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540. doi:10.1038/s41467-020-18321-y
- Potapov, P., et al. (2021). Mapping global forest canopy height through integration of GEDI and Landsat data. *Remote Sensing of Environment*, 253, 112165. doi:10.1016/j.rse.2020.112165
- Rennó, C. D., et al. (2008). HAND, a new terrain descriptor using SRTM-DEM: mapping terra-firme rainforest environments in Amazonia. *Remote Sensing of Environment*, 112(9), 3469–3481. doi:10.1016/j.rse.2008.03.018
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*.
- Sirko, W., et al. (2021). Continental-scale building detection from high resolution satellite imagery. arXiv:2107.12283.
- Small, D. (2011). Flattening gamma: radiometric terrain correction for SAR imagery. *IEEE TGRS*, 49(8), 3081–3093. doi:10.1109/TGRS.2011.2120616
- Tsyganskaya, V., Martinis, S., Marzahn, P., & Ludwig, R. (2018). SAR-based detection of flooded vegetation – a review of characteristics and approaches. *International Journal of Remote Sensing*, 39(8), 2255–2293. doi:10.1080/01431161.2017.1420938
- Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150. doi:10.1016/0034-4257(79)90013-0
- Twele, A., Cao, W., Plank, S., & Martinis, S. (2016). Sentinel-1-based flood mapping: a fully automated processing chain. *International Journal of Remote Sensing*, 37(13), 2990–3004. doi:10.1080/01431161.2016.1192304
- Wagner, F. H., et al. (2023). Mapping tropical forest cover and deforestation with Planet NICFI satellite images and deep learning in Mato Grosso State (Brazil) from 2015 to 2021. *Remote Sensing*, 15(2), 521.
- Wagner, F. H., Favrichon, S., Dalagnol, R., Hirye, M. C. M., Mullissa, A., & Saatchi, S. (2024). The Amazon's 2023 drought: Sentinel-1 reveals extreme Rio Negro River contraction. *Remote Sensing*, 16(6), 1056. doi:10.3390/rs16061056 (arXiv:2401.16393)
- Yamazaki, D., et al. (2019). MERIT Hydro: a high-resolution global hydrography map based on latest topography datasets. *Water Resources Research*, 55(6), 5053–5073. doi:10.1029/2019WR024873
- Zanaga, D., et al. (2022). *ESA WorldCover 10 m 2021 v200*. doi:10.5281/zenodo.7254221
- Zhao, J., Pelich, R., Hostache, R., Matgen, P., Cao, S., Wagner, W., & Chini, M. (2021). Deriving exclusion maps from C-band SAR time-series in support of floodwater mapping. *Remote Sensing of Environment*, 265, 112668. doi:10.1016/j.rse.2021.112668
- Zhao, J., Li, M., Li, Y., Matgen, P., & Chini, M. (2024). Urban flood mapping using satellite synthetic aperture radar data: a review of characteristics, approaches, and datasets. *IEEE Geoscience and Remote Sensing Magazine*. doi:10.1109/MGRS.2024.3496075
- Zhao, J., Xiong, Z., & Zhu, X. X. (2024). UrbanSARFloods: Sentinel-1 SLC-based benchmark dataset for urban and open-area flood mapping. *CVPR Workshops (EarthVision)*, 419–429. doi:10.1109/CVPRW63382.2024.00047 (arXiv:2406.04111)
- Zomer, R. J., Xu, J., & Trabucco, A. (2022). Version 3 of the Global Aridity Index and Potential Evapotranspiration Database. *Scientific Data*, 9, 409. doi:10.1038/s41597-022-01493-1
