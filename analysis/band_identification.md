# UrbanSARFloods test set: which band is which

**Result:** in all four test scenes, and in the 256 px chips (they are exact crops of the scenes), the
8 float32 bands are:

| band | content | how it was established |
|---|---|---|
| 1-4 | interferometric **coherence** (values in [0, 1]) | value range |
| 5 | **pre-event intensity, VH** (dB) | dB range; ~5-7 dB below band 6; does not darken over open flood |
| 6 | **pre-event intensity, VV** (dB) | higher median of the pair 5/6 |
| 7 | **post-event intensity, VH** (dB) | darkens by 4-11 dB over open-flood GT pixels |
| 8 | **post-event intensity, VV** (dB) | higher median of the pair 7/8; darkens over open flood |

The model input was therefore built as `pre = (band 6, band 5)` and `post = (band 8, band 7)` →
`db_to_uint8` → stacked **VV, VH, VV, VH**. The pre/co-event split and VV/VH order of the coherence
bands 1-4 are not needed (the model does not use coherence) and are **not reliably identifiable** from
the data (see the end of this note).

## What the files say (nothing)

- `rasterio` `descriptions` is `(None,) * 8` for every SAR file, scene and chip; there are no band-level
  tags, no `.aux.xml` sidecars, no `ImageDescription`. Dataset-level tags are only
  `TIFFTAG_XRESOLUTION=1`, `TIFFTAG_YRESOLUTION=1`, `AREA_OR_POINT=Area`.
- The Hugging Face dataset (revision `0892ffa`) has no README / dataset card.
- The authors' GitHub repository (`jie666-6/UrbanSARFloods`) has only a README with no band order and
  no data-loader code.
- The paper (Zhao et al. 2024, arXiv:2406.04111) lists the groups as *pre-event intensity, post-event
  intensity, pre-event coherence, co-event intensity (sic)*. **This is not the file order**: in the files
  the coherence bands come first. Relying on the paper's order would have fed coherence to the model.

So the order was inferred from the data, with `python analysis/usf_eval.py bands EVENT`, which reads
12 strips of 16 rows across each full scene (3.0-4.3 M pixels) and the matching GT. The rules are coded
in `identify_bands()` and the full statistics are stored in `analysis/band_check/<event>.json`. The
`prepare` and `chips` steps refuse to run unless the inferred intensity bands match `EXPECTED_BANDS`
for that event.

## Evidence

Median and mean by GT class (0 non-flood, 1 open flood, 2 urban flood) of the intensity bands, in dB:

| event | band | median | p99 | GT 0 | GT 1 | GT 2 |
|---|---|---|---|---|---|---|
| Weihui | 5 | -13.5 | -5.0 | -13.6 | -14.5 | -11.8 |
| | 6 | -7.0 | 2.4 | -7.1 | -7.6 | -5.2 |
| | 7 | -13.2 | -4.8 | -13.3 | **-21.6** | -11.0 |
| | 8 | -7.0 | 2.6 | -7.1 | **-16.4** | -4.0 |
| Nova Kakhovka | 5 | -16.8 | -7.7 | -17.1 | -15.9 | -14.7 |
| | 6 | -10.4 | -1.1 | -10.8 | -7.5 | -7.5 |
| | 7 | -17.8 | -8.0 | -17.8 | **-21.8** | -16.2 |
| | 8 | -11.3 | -2.2 | -11.2 | **-15.2** | -8.4 |
| Jubba 1 | 5 | -11.9 | -5.5 | -12.1 | -15.7 | - |
| | 6 | -6.7 | -0.5 | -6.9 | -9.3 | - |
| | 7 | -12.1 | -5.9 | -12.3 | **-22.7** | - |
| | 8 | -6.7 | -0.6 | -6.9 | **-19.2** | - |
| Jubba 2 | 5 | -12.1 | -5.9 | -12.3 | -14.6 | - |
| | 6 | -7.0 | -0.9 | -7.2 | -8.0 | - |
| | 7 | -12.5 | -6.4 | -12.6 | **-21.8** | - |
| | 8 | -7.3 | -1.2 | -7.4 | **-17.8** | - |

(No urban-flood pixel fell in the sampled strips of the Jubba scenes.)

1. **Coherence vs intensity.** Bands 1-4 have min 0 and max ≤ 1 in every scene, which fits coherence.
   Bands 5-8 range from -60 to +6.99 dB. The dataset appears to clip at [-60, 6.99] dB.
2. **VV vs VH.** In each pair (5, 6) and (7, 8) one band is 5.1-6.6 dB brighter, and only that band
   reaches positive dB at the 99th percentile. Cross-pol VH is normally 5-8 dB below co-pol VV over land,
   so 6 and 8 are VV and 5 and 7 are VH. This VH-before-VV order is also SNAP's band order (see the
   `Subset` comment in `s1water/preprocess.py`).
3. **Pre vs post.** An open flood is dark (specular) only after the event. Over GT = 1 pixels, bands 7/8
   are 4.0-11.3 dB darker than over GT = 0. Bands 5/6 change by -3.0 to +2.3 dB. So 7/8 are post-event.
   The urban-flood pixels agree: post VV (band 8) is 3.1 dB *brighter* than non-flood at Weihui, the
   expected double-bounce increase from water in streets.
4. **Consistency.** The same mapping is found independently in all four scenes. The chips are
   bit-identical crops of the scenes (`chip_equals_scene_crop` is True for every downloaded chip in
   `usf_results.csv`), so the mapping also holds for the chips.

## Coherence bands (not used; tentative only)

Flooding should decorrelate the co-event pair. At Weihui and Nova Kakhovka, bands 1/2 are much higher
than 3/4 over urban-flood pixels (0.64-0.73 vs 0.26-0.40), which fits *1/2 = pre-event, 3/4 =
co-event*. At Jubba coherence is about 0.1 everywhere (semi-arid, vegetated) and the difference is
within ±0.02, so the split cannot be confirmed there. VV/VH inside the coherence pairs cannot be told
apart from these statistics.

## Other facts about the files

- CRS EPSG:4326, pixel 0.00016666° (1/6000°): 18.5 m north-south, 12.7-18.5 m east-west depending on
  latitude.
- No-data is NaN, with no declared nodata value. 34-47 % of each scene is NaN (outside the swath).
- Weihui is pixel-interleaved and the other scenes band-interleaved. All are uncompressed strips of one
  row. The GTs are float32 0/1/2, LZW-compressed for Jubba.
- 997 of the 2029 Jubba_2 chip files on the Hub are **0 bytes**. Those tiles were evaluated on the
  identical window of the full scene (`chip_source = scene_crop(empty_hub_file)` in the CSV).
- The chips have no GT files (`testing_case_256` holds `*_SAR.tif` only). Chip GT is the same 256 px
  window of `testing_case_orig/<event>/<event>_GT.tif`: `ID_<r>_<c>` is the window at row `256·r`,
  column `256·c`. This was checked from the chip georeferencing.
