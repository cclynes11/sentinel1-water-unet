# MEng interim report: skeleton

**Working title:** Landscape proxies for out-of-distribution performance of SAR flood models
**Due:** Thursday 26 November 2026
**Sources:** [`literature_matrix.md`](literature_matrix.md) · [`datasets.md`](datasets.md) · [`citation_verification.md`](citation_verification.md)

---

## 0. Front matter
- Title, name, supervisor, date, word count
- Abstract (≤ 200 words, written last)
  - Problem: SAR flood models fail silently outside their training landscapes
  - Aim: test whether cheap landscape descriptors predict where they fail
  - Progress so far plus one headline preliminary number
- Evidence: none new; summarises sections 1–6
- Figures: none

## 1. Introduction and motivation
- Why SAR for floods: all-weather, day/night; operational rapid mapping
  - Evidence: Nemni 2020 (about 80% faster map production)
- Why OOD matters: models are trained on few events or biomes and deployed globally
  - Evidence: Wagner 2024 U-Net trained only on the Amazon; repo README "Limitations"
  - Evidence: Kolluru 2026, near-zero IoU in tree cover and built-up areas on unseen events
- Gap: OOD detectors are usually image- or model-based; **landscape context is available before the image and is rarely used as a failure predictor**
  - Evidence: Meyer & Pebesma 2021 (AOA in covariate space); Thasma 2026 (land-context priors help, and similarity to training tracks per-event performance)
- Contributions (planned): proxy set, evaluation protocol, benchmark against MSP / ensemble / kNN baselines
- Figures
  - **Fig 1**: Concept diagram: landscape covariates → proxy score → predicted IoU loss, beside a model-confidence route

## 2. Background and literature review
- 2.1 Deep learning for SAR flood and water mapping
  - Evidence: Nemni 2020, Wagner 2024, Zhao 2024, Portalés-Julià 2025 ⚠️ (read the full text first)
  - Point: performance varies strongly across events and datasets; label definitions differ
- 2.2 Distribution shift in Earth observation
  - Evidence: WILDS (Koh 2021): ID→OOD gap, domain vs subpopulation shift
  - Evidence: Thasma 2026 (per-event variation), Kolluru 2026 (land cover and flood type)
- 2.3 OOD detection and uncertainty
  - Evidence: Hendrycks & Gimpel 2017 (MSP), Sun 2022 ⚠️ (kNN), Le Bellier & Audebert 2024 (diffusion, near-OOD floods)
  - Evidence: Ovadia 2019 (calibration breaks under shift; ensembles best)
  - Evidence: de Jorge 2023 (segmentation reliability; robust ≠ well-calibrated)
- 2.4 From OOD detection to failure prediction
  - Evidence: Guérin 2023 (out-of-model-scope detection: judge monitors on error detection)
  - Evidence: Meyer & Pebesma 2021 (AOA: error valid only inside the training covariate space)
- 2.5 Summary of the gap
  - Literature matrix condensed into one table (key columns only)
- Figures and tables
  - **Table 1**: Condensed literature matrix (method / OOD approach / relevance)
  - **Fig 2** (optional): Taxonomy tree of proxies: landscape vs representation vs model-output

## 3. Aims, research questions and hypotheses
- RQ1: Do landscape covariates predict per-tile IoU loss of a SAR flood model on unseen events?
  - H1: tree-cover and built-up fractions correlate negatively with IoU (Kolluru 2026, Zhao 2024)
  - H2: terrain (slope / HAND / shadow likelihood) raises false positives (Wagner 2024)
  - H3: landscape dissimilarity to training (AOA DI) predicts IoU loss (Meyer & Pebesma 2021)
- RQ2: Do landscape proxies add information beyond model-based uncertainty (MSP, ensemble, kNN)?
  - H4: landscape and model-based proxies are complementary; combining them improves AURC
- RQ3: Are conclusions stable across datasets (UrbanSARFloods → Sen1Floods11)?
- Objectives with measurable success criteria (e.g. Spearman ρ, AURC reduction vs the MSP baseline)
- Figures: none (maybe a small table mapping RQ → hypothesis → metric)

## 4. Data
- 4.1 Primary: UrbanSARFloods
  - Evidence: 8,879 chips, 18 events, 20 land-cover classes, NF / FO / FU labels, CC BY 4.0
  - Issue: SLC intensity vs the GRD σ⁰ input of the Wagner U-Net; the preprocessing plan
- 4.2 Secondary: Sen1Floods11
  - Evidence: 4,831 chips, 11 events, 14 biomes; GRD; hand-labelled subset; Bolivia hold-out
  - Role: zero-shot test of the Wagner U-Net with minimal preprocessing
- 4.3 Landscape covariate layers (to choose and cite: land cover, DEM / HAND, permanent water, climate zone)
- 4.4 Splits: leave-events-out; list of training vs OOD events
- 4.5 Known label issues: weak labels, optical-derived references, definition mismatches (Kolluru 2026)
- Figures and tables
  - **Table 2**: Dataset summary (from `datasets.md`, UrbanSARFloods first)
  - **Fig 3**: World map of event locations, coloured by train / test split, with Amazon training area of Wagner 2024 overlaid
  - **Fig 4**: Land-cover composition per event (stacked bars), showing the covariate shift between splits

## 5. Methodology
- 5.1 Models
  - M0: Wagner 2024 U-Net, off-the-shelf (this repo, `src/s1water/`)
  - M1: U-Net trained or fine-tuned on UrbanSARFloods training events
  - (Optional) M2: Portalés-Julià 2025 public S1 U-Net, as a second architecture
  - Small ensemble or MC dropout for model-based uncertainty (Ovadia 2019)
- 5.2 Proxies, computed per tile
  - Landscape: tree-cover %, built-up %, cropland %, permanent-water %, mean slope / HAND, flood type
  - Landscape AOA DI over covariates (Meyer & Pebesma 2021)
  - Representation: kNN distance in bottleneck features (Sun 2022)
  - Model output: mean MSP / entropy (Hendrycks & Gimpel 2017); ensemble disagreement (Ovadia 2019)
- 5.3 Target: per-tile IoU / F1 and IoU drop vs in-distribution mean; separate FO and FU where labels allow
- 5.4 Evaluation of proxies (Guérin 2023 framing)
  - Spearman ρ between proxy and IoU loss
  - Risk-coverage curves and AURC (reject tiles by proxy score)
  - AUROC for "failure" (IoU below a threshold) vs in-distribution / OOD separation
  - Per-event and per-stratum reporting (WILDS 2021, Thasma 2026); bootstrap CIs over events
- 5.5 Controls: label-source confound, class prevalence (Thasma 2026), tile size
- Figures
  - **Fig 5**: Pipeline flowchart: S1 → preprocessing → model → prediction plus covariates → proxies → evaluation
  - **Fig 6** (optional): Schematic of the AOA DI in a 2-D covariate space

## 6. Preliminary results (progress to date)
- Repo set up; Wagner U-Net running; preprocessing validated (repo `docs/validation_*.json`)
- Zero-shot M0 on Sen1Floods11 (or a subset): overall and per-event IoU
- First stratified look: IoU by land-cover class / biome
- First correlation of 1–2 landscape proxies with IoU, if available by the deadline
- State honestly what is not done yet
- Figures and tables
  - **Fig 7**: Qualitative examples: S1 VV, prediction, reference, error map for one good tile and one failure tile (e.g. urban, forest, mountain shadow)
  - **Fig 8**: IoU distribution per event (box / strip plot), sorted by IoU
  - **Fig 9**: IoU vs tree-cover % and vs built-up % (scatter with LOESS / binned means)
  - **Table 3**: Per-event metrics for M0 (IoU, F1, precision, recall)

## 7. Discussion of preliminary findings
- Agreement or disagreement with Kolluru 2026 (land cover) and Wagner 2024 (documented failure modes)
- Threats to validity: label noise, few events, SLC vs GRD mismatch, spatial autocorrelation of tiles
- What the preliminary results change in the plan
- Figures: none (refer back to Figs 7–9)

## 8. Plan for remaining work
- Work packages: data prep → M1 training → proxy computation → evaluation → write-up
- Milestones and dates up to the final submission
- Stretch goals: Kuro Siwo for more events; diffusion OOD score (Le Bellier & Audebert 2024); combined proxy (learned regressor)
- Figures
  - **Fig 10**: Gantt chart (Nov 2026 → final deadline)

## 9. Risks and mitigation
- SLC/GRD input mismatch → fall back to Sen1Floods11 for M0; train M1 on UrbanSARFloods inputs
- Too few events for statistics → add Kuro Siwo; tile-level mixed-effects model with event as a random effect
- Compute limits → small ensemble (≤ 5) or MC dropout; Colab notebooks already in the repo
- Unresolved citations → see the verification log action items
- Table
  - **Table 4**: Risk register (risk / likelihood / impact / mitigation)

## 10. References
- Only entries ✅ or ⚠️ in `citation_verification.md`
- **Do not include "Zimmermann" until it is identified**
- Mark arXiv preprints as such (Thasma 2026, Kolluru 2026)
- One consistent style (e.g. Harvard / IEEE, per department guidance)

## Appendices (optional)
- Full literature matrix
- Full dataset table
- Preprocessing parameters (SNAP graph from `snap_graphs/`)

---

## Figure / table checklist
| ID | Content | Source of data | Status |
|---|---|---|---|
| Fig 1 | Concept diagram | drawn | ☐ |
| Fig 2 | Proxy taxonomy (optional) | literature matrix §C | ☐ |
| Fig 3 | Event map with splits | dataset metadata | ☐ |
| Fig 4 | Land-cover composition per event | covariate layers | ☐ |
| Fig 5 | Pipeline flowchart | drawn | ☐ |
| Fig 6 | AOA schematic (optional) | drawn | ☐ |
| Fig 7 | Qualitative success / failure tiles | M0 predictions | ☐ |
| Fig 8 | IoU per event | M0 on Sen1Floods11 | ☐ |
| Fig 9 | IoU vs landscape covariates | M0 plus covariates | ☐ |
| Fig 10 | Gantt chart | plan | ☐ |
| Table 1 | Condensed literature matrix | `literature_matrix.md` | ☐ |
| Table 2 | Dataset summary | `datasets.md` | ☐ |
| Table 3 | Per-event metrics | M0 | ☐ |
| Table 4 | Risk register | §9 | ☐ |

## Countdown to 26 Nov (suggested)
- By 11 Nov: resolve the citation action items; read Portalés-Julià 2025 in full; download Sen1Floods11
- By 16 Nov: zero-shot M0 on Sen1Floods11; Figs 7–8, Table 3
- By 20 Nov: land-cover covariates per tile; Fig 9; Figs 3–4
- By 23 Nov: draft sections 1–5 from this skeleton
- By 25 Nov: sections 6–9, abstract, proofread; supervisor check if possible
