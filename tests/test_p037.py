"""Tests of the P037 pipeline (src/p037). Everything runs on generated data, on CPU, in seconds."""

from __future__ import annotations

import tarfile

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")

from p037 import ood  # noqa: E402
from p037.config import Config, load_config  # noqa: E402
from p037.datasets import get_dataset  # noqa: E402
from p037.datasets.base import INVALID, channel_indices, uint8_to_db  # noqa: E402
from p037.metrics import class_sets, confusion, scores  # noqa: E402

SMOKE = dict(dataset="synthetic", epochs=1, batch_size=4, num_workers=0, bn_momentum=0.1, tile=64, window=128,
             crop=32, mc_passes=3, patience=0, n_bootstrap=5, min_positive_px=20,
             proxies=["slope_mean", "tree_frac", "built_frac"])


def make_cfg(tmp_path, **kw) -> Config:
    return Config(name="t", data_root=str(tmp_path / "data"), **{**SMOKE, **kw})


# ------------------------------------------------------------------------------------- config
def test_config_yaml_overrides_and_unknown_keys(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("name: a\nepochs: 5\n")
    cfg = load_config(f, ["lr=0.01", "val_events=[x, y]", "amp=false"])
    assert (cfg.name, cfg.epochs, cfg.lr, cfg.val_events, cfg.amp) == ("a", 5, 0.01, ["x", "y"], False)
    f.write_text("epocs: 5\n")
    with pytest.raises(KeyError):
        load_config(f)
    with pytest.raises(ValueError):
        Config(window=1000, tile=256)


# ------------------------------------------------------------------------------------- metrics
def test_confusion_excludes_other_class_and_invalid():
    gt = np.array([[0, 1, 2, INVALID], [0, 1, 2, 0]], np.uint8)
    pred = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], bool)
    valid = np.ones_like(pred)
    valid[1, 3] = False
    sets = class_sets({1: "open", 2: "urban"})
    assert confusion(pred, gt, valid, *sets["open"]) == (1, 1, 1, 1)  # urban pixels left out
    assert confusion(pred, gt, valid, *sets["any"]) == (2, 1, 2, 1)
    s = scores(1, 1, 1, 1)
    assert s["f1"] == pytest.approx(0.5) and s["iou"] == pytest.approx(1 / 3) and s["accuracy"] == pytest.approx(0.5)
    assert np.isnan(scores(0, 0, 0, 5)["f1"])


# ------------------------------------------------------------------------------------- OOD numerics
def test_percentile_and_placement():
    ref = pd.DataFrame({"a__tile": np.arange(100.0)})
    tiles = pd.DataFrame({"a__tile": [-5.0, 49.5, 200.0, np.nan]})
    out = ood.place(tiles, ref, {"a": "a__tile"})
    assert out["pct__a"].tolist()[:3] == [0.0, 50.0, 100.0] and np.isnan(out["pct__a"][3])
    assert out["pos__a"].tolist()[:3] == [-1.0, 0.0, 1.0]
    assert out["dist__a"].tolist()[:3] == [1.0, 0.0, 1.0]


def test_prune_collinear_keeps_first_of_a_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    ref = pd.DataFrame({"a": a, "b": a + 0.01 * rng.normal(size=500), "c": rng.normal(size=500)})
    kept, dropped = ood.prune_collinear(ref, {"a": "a", "b": "b", "c": "c"}, 0.85)
    assert kept == ["a", "c"] and dropped["b"][0] == "a"


def test_linear_recovers_weights_and_partial_correlation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3)) * [1.0, 5.0, 0.2]
    y = 0.3 + X @ [0.5, -0.1, 0.0] + 0.01 * rng.normal(size=2000)
    m = ood.Linear().fit(X, y)
    assert m.weights == pytest.approx(np.array([0.5, -0.1, 0.0]) * X.std(0), abs=2e-3)
    assert ood.r2(y, m.predict(X)) > 0.99
    # b depends on a only through c: the partial correlation given c vanishes
    c = rng.normal(size=3000)
    a, b = c + 0.3 * rng.normal(size=3000), c + 0.3 * rng.normal(size=3000)
    assert ood.spearman(a, b) > 0.8 and abs(ood.partial_spearman(a, b, c)) < 0.06


def test_triage_curve_perfect_and_random():
    err = np.array([10.0, 0, 0, 0, 30, 0, 0, 0, 0, 0])
    assert ood.found_at(err, err, 0.2) == pytest.approx(1.0)
    assert ood.found_at(-err, err, 0.2) == pytest.approx(0.0)
    x, y = ood.triage_curve(np.arange(10.0), err)
    assert y[0] == 0 and y[-1] == pytest.approx(1.0) and np.all(np.diff(y) >= 0)


def test_held_out_event_test_finds_a_transferable_signal():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"tile_id": [f"t{i}" for i in range(n)], "event": rng.choice(["a", "b", "c"], n),
                       "signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    df["event_group"] = df["event"]
    df["err"] = 0.2 * df["signal"] + 0.05 * rng.normal(size=n)
    table, preds = ood.held_out_event_test(df, {"good": ["signal"], "bad": ["noise"]}, "err")
    good, bad = (table[table["model"] == m]["spearman"] for m in ("good", "bad"))
    assert len(table) == 6 and good.min() > 0.8 and bad.abs().max() < 0.3
    assert preds["pred__good"].notna().all()


# ------------------------------------------------------------------------------------- synthetic data
def test_synthetic_cache_and_patch_dataset(tmp_path):
    pytest.importorskip("torch")
    from p037.data import PatchDataset, select_patches

    cfg = make_cfg(tmp_path, val_mode="events", val_events=["train_0"])
    ds = get_dataset(cfg)
    ds.build_cache(n_train=12, n_val=6)
    meta, patches = ds.meta(), ds.patches()
    assert meta.channels == ["pre_VV", "pre_VH", "post_VV", "post_VH"] and len(patches) == 18
    train, val = select_patches(cfg, ds)
    assert set(val["event"]) == {"train_0"} and "train_0" not in set(train["event"])
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"train", "val"}
    x, y, m = PatchDataset(cfg, ds, train, augment=True)[0]
    assert x.shape == (4, 64, 64) and str(x.dtype) == "torch.uint8" and y.shape == m.shape == (1, 64, 64)
    assert channel_indices(["post_VV", "post_VH", "post_VV", "post_VH"], meta.channels) == [2, 3, 2, 3]
    with pytest.raises(KeyError):
        channel_indices(["VV"], meta.channels)


# ------------------------------------------------------------------------------------- UrbanSARFloods adapter
def _write_chip(folder, chip, rng, swap_dates=False, nan_corner=False):
    """One 128 px chip in the dataset's layout: 8 float32 bands (4 coherence, pre VH, pre VV, post VH, post VV)."""
    import rasterio
    from rasterio.transform import from_origin

    n = 128
    gt = np.zeros((n, n), np.float32)
    gt[: n // 2, n // 2:] = 1  # open flood in the top-right quadrant only
    vh_pre, vv_pre = -16 + rng.normal(size=(n, n)), -9 + rng.normal(size=(n, n))
    vh_post, vv_post = vh_pre.copy(), vv_pre.copy()
    vh_post[gt == 1] -= 8
    vv_post[gt == 1] -= 8
    bands = [rng.uniform(0, 1, (n, n)) for _ in range(4)]
    bands += [vh_post, vv_post, vh_pre, vv_pre] if swap_dates else [vh_pre, vv_pre, vh_post, vv_post]
    arr = np.stack(bands).astype(np.float32)
    if nan_corner:
        arr[:, -8:, :8] = np.nan
    prof = dict(driver="GTiff", width=n, height=n, crs="EPSG:4326", transform=from_origin(10, 5, 1e-4, 1e-4))
    (folder / "SAR").mkdir(parents=True, exist_ok=True)
    (folder / "GT").mkdir(parents=True, exist_ok=True)
    with rasterio.open(folder / "SAR" / f"{chip}_SAR.tif", "w", count=8, dtype="float32", **prof) as o:
        o.write(arr)
    with rasterio.open(folder / "GT" / f"{chip}_GT.tif", "w", count=1, dtype="float32", **prof) as o:
        o.write(gt, 1)


def _fake_index(chips):
    """Rows in the format of analysis/proxy/outputs/usf_chip_footprints.csv.gz."""
    return [dict(chip=c, event=c.split("_ID_")[0], folder="02_FO", left=10.0, top=5.0, sx=1e-4, sy=1e-4, w=128, h=128,
                 split=s) for c, s in chips]


@pytest.mark.parametrize("as_tar", [False, True])
def test_usf_cache_from_folder_and_archive(tmp_path, as_tar):
    pytest.importorskip("rasterio")
    from p037.datasets.urbansarfloods import UrbanSARFloods, chip_index

    rng = np.random.default_rng(0)
    chips = [("20200101_Ev_ID_0_0", "train"), ("20200101_Ev_ID_0_1", "train"), ("20200101_Ev_ID_3_2", "valid")]
    src = tmp_path / "urban_sar_floods"
    for i, (c, _) in enumerate(chips):
        _write_chip(src / "02_FO", c, rng, nan_corner=i == 0)
    _write_chip(src / "02_FO", "20990101_NotListed_ID_0_0", rng)  # not in the official lists: ignored
    fp = tmp_path / "fp.csv"
    pd.DataFrame(_fake_index(chips)).to_csv(fp, index=False)
    source = src
    if as_tar:
        source = tmp_path / "urban_sar_floods.tar.gz"
        with tarfile.open(source, "w:gz") as tar:
            tar.add(src, arcname="urban_sar_floods")
    ds = UrbanSARFloods(tmp_path / "data")
    ds.patch = 64
    ds.build_cache(str(source), index=chip_index(fp), workers=2)

    patches = ds.patches()
    assert len(patches) == 12 and patches.groupby("split").size().to_dict() == {"train": 8, "val": 4}
    assert patches["patch_id"].iloc[1] == "20200101_Ev_ID_0_0_q1"
    images, labels = ds.open_arrays("train")
    q = patches[patches["chip"] == "20200101_Ev_ID_0_0"].set_index("quadrant")
    # the flood was drawn in the top-right quadrant = q1, and nowhere else
    assert q["n_open"].to_dict() == {0: 0, 1: 64 * 64, 2: 0, 3: 0}
    # the NaN corner (bottom-left quadrant = q2) is invalid in the labels
    assert q.loc[2, "n_valid"] == 64 * 64 - 64 and (np.asarray(labels[q.loc[2, "index"]]) == INVALID).sum() == 64
    # stored channel order is pre VV, pre VH, post VV, post VH: VV brighter than VH, flood darker after
    db = uint8_to_db(np.asarray(images[q.loc[1, "index"]]))
    assert db[0].mean() > db[1].mean() + 4 and db[2].mean() < db[0].mean() - 5
    assert ds.meta().channels == ["pre_VV", "pre_VH", "post_VV", "post_VH"]
    # centre of q1 of a 128 px chip at 1e-4 deg: 96 px east, 32 px south of the corner
    assert q.loc[1, "lon"] == pytest.approx(10 + 96e-4) and q.loc[1, "lat"] == pytest.approx(5 - 32e-4)


def test_usf_cache_refuses_swapped_dates_and_missing_chips(tmp_path):
    pytest.importorskip("rasterio")
    from p037.datasets.urbansarfloods import UrbanSARFloods, check_bands, chip_index

    rng = np.random.default_rng(0)
    chips = [(f"20200101_Ev_ID_0_{i}", "train") for i in range(6)]
    src = tmp_path / "src"
    for c, _ in chips:
        _write_chip(src / "02_FO", c, rng, swap_dates=True)
    fp = tmp_path / "fp.csv"
    pd.DataFrame(_fake_index(chips)).to_csv(fp, index=False)
    ds = UrbanSARFloods(tmp_path / "data")
    ds.patch = 64
    with pytest.raises(SystemExit, match="band order check failed"):
        ds.build_cache(str(src), index=chip_index(fp), workers=1)
    assert not (ds.cache_dir / "patches.csv").exists()

    (src / "02_FO" / "GT" / "20200101_Ev_ID_0_0_GT.tif").unlink()
    ds2 = UrbanSARFloods(tmp_path / "data2")
    ds2.patch = 64
    with pytest.raises(SystemExit, match="have no SAR or no GT"):
        ds2.build_cache(str(src), index=chip_index(fp), workers=1)

    stats = {c: {"0": v, "1": v - 8 if c.startswith("post") else v}
             for c, v in {"pre_VV": -9.0, "pre_VH": -9.5, "post_VV": -9.0, "post_VH": -9.5}.items()}
    stats["n_open"] = 50_000
    ok, reasons = check_bands(stats)
    assert not ok and "TOO SMALL" in reasons[0]


def test_usf_scene_preparation(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from p037.datasets.urbansarfloods import UrbanSARFloods

    event = "20231201_Jubba_1"
    raw = tmp_path / "raw" / "testing_case_orig" / event
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    arr = np.concatenate([rng.uniform(0, 1, (4, 70, 90)), -12 + rng.normal(size=(4, 70, 90))]).astype(np.float32)
    arr[4:, :, :5] = np.nan
    gt = rng.integers(0, 3, (70, 90)).astype(np.float32)
    prof = dict(driver="GTiff", width=90, height=70, crs="EPSG:4326", transform=from_origin(42, 1, 1e-4, 1e-4))
    with rasterio.open(raw / "20231201_Jubba_SAR.tif", "w", count=8, dtype="float32", **prof) as o:
        o.write(arr)
    with rasterio.open(raw / "20231201_Jubba_1_GT.tif", "w", count=1, dtype="float32", **prof) as o:
        o.write(gt, 1)
    ds = UrbanSARFloods(tmp_path / "data")
    ds.prepare_scene(event, str(tmp_path / "raw"))
    f = ds.scene_files(event)
    with rasterio.open(f["image"]) as im, rasterio.open(f["valid"]) as v, rasterio.open(f["gt"]) as g:
        assert list(im.descriptions) == ["pre_VV", "pre_VH", "post_VV", "post_VH"]
        # file bands 5-8 are pre VH, pre VV, post VH, post VV; stored order is pre VV, pre VH, post VV, post VH
        np.testing.assert_allclose(uint8_to_db(im.read(1))[:, 5:], arr[5][:, 5:], atol=0.11)
        np.testing.assert_allclose(uint8_to_db(im.read(2))[:, 5:], arr[4][:, 5:], atol=0.11)
        assert v.read(1)[:, :5].sum() == 0 and v.read(1)[:, 5:].all()
        np.testing.assert_array_equal(g.read(1), gt.astype(np.uint8))


# ------------------------------------------------------------------------------------- end to end
def _proxy_table(cfg, ds, path):
    """Random proxy values for every synthetic training patch and test tile."""
    rng = np.random.default_rng(0)
    p = ds.patches()
    ids = [(i, "b_usf_train") for i in p[p["split"] == "train"]["patch_id"]]
    tiles = pd.read_csv(cfg.out / "tiles.csv")
    ids += [(i, "test") for i in tiles["tile_id"]]
    df = pd.DataFrame(ids, columns=["sample_id", "set"])
    df["lon"], df["lat"], df["mostly_water"] = 0.0, 0.0, False
    for proxy in cfg.proxies:
        df[f"{proxy}__tile"] = rng.normal(size=len(df))
    df.to_csv(path, index=False)


def test_pipeline_end_to_end_on_synthetic_data(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rasterio")
    pytest.importorskip("matplotlib")
    from p037.analyze import run_analysis
    from p037.evaluate import evaluate_patches, evaluate_scenes
    from p037.proxies import check_coverage
    from p037.train import train

    torch.set_num_threads(2)
    cfg = make_cfg(tmp_path, proxy_table=str(tmp_path / "proxies.csv"))
    ds = get_dataset(cfg)
    ds.build_cache(n_train=16, n_val=8)
    for event in ds.test_events:
        ds.prepare_scene(event, shape=(200, 280))

    state = train(cfg, "cpu")
    assert state["epoch"] == 1
    log = pd.read_csv(cfg.out / "log.csv")
    assert len(log) == 1 and np.isfinite(log["train_loss"][0]) and (cfg.out / "last.pt").exists()
    assert train(cfg, "cpu")["epoch"] == 1  # finished: a second call does nothing
    cfg2 = make_cfg(tmp_path, epochs=2, proxy_table=cfg.proxy_table)
    assert train(cfg2, "cpu")["epoch"] == 2 and len(pd.read_csv(cfg.out / "log.csv")) == 2  # resumed

    tiles = pd.read_csv(evaluate_scenes(cfg, "cpu"))
    # 200 x 280 px scene, 64 px tiles, the last 20 columns without SAR data: 4 rows x 5 columns of tiles
    assert len(tiles) == 2 * 20 and tiles["tile_id"].iloc[0] == "synth_A_ID_0_0"
    edge = tiles[(tiles["event"] == "synth_A") & (tiles["tile_row"] == 3) & (tiles["tile_col"] == 4)].iloc[0]
    assert edge["n_valid"] == (200 - 192) * (260 - 256)  # the corner tile: 8 rows x 4 valid columns
    for cls in ("open", "urban", "any"):
        n = tiles[[f"flood_{cls}_{k}" for k in ("tp", "fp", "fn", "tn")]].sum(axis=1)
        other = tiles["n_gt_urban"] if cls == "open" else tiles["n_gt_open"] if cls == "urban" else 0
        assert (n == tiles["n_valid"] - other).all()  # every valid pixel is scored exactly once
    assert (tiles["mc_var_flood"] >= 0).all() and tiles["mean_entropy_flood"].between(0, 1).all()
    assert (cfg.out / "prob_synth_A.tif").exists()

    patches = pd.read_csv(evaluate_patches(cfg, "val", "cpu"))
    assert len(patches) == 8 and (patches["tile_row"] == -1).all()

    _proxy_table(cfg, ds, cfg.proxy_table)
    assert check_coverage(cfg)["missing"] == 0
    report = run_analysis(cfg)
    text = report.read_text()
    assert "## 6. Held-out-event test" in text and (report.parent / "figures" / "placement.png").exists()
    scored = pd.read_csv(report.parent / "tiles_scored.csv.gz")
    assert len(scored) == 40 and {"err_rate", "pct__slope_mean", "dist__built_frac"} <= set(scored.columns)


def test_zero_shot_rule_matches_its_definition(tmp_path):
    torch = pytest.importorskip("torch")
    from p037.evaluate import Predictor
    from p037.model import build_model

    cfg = make_cfg(tmp_path, flood_rule="post_and_not_pre", mc_passes=0)
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    channels = ["pre_VV", "pre_VH", "post_VV", "post_VH"]
    img = np.random.default_rng(0).integers(0, 255, (2, 4, 64, 64), dtype=np.uint8)
    out = Predictor(cfg, model, channels, "cpu")(img, crop=16)
    assert out["prob"].shape == (2, 32, 32) and "mc_var" not in out
    with torch.no_grad():
        x = torch.from_numpy(img).float() / 255
        pre, post = model(x[:, [0, 1, 0, 1]], crop=16)[:, 0], model(x[:, [2, 3, 2, 3]], crop=16)[:, 0]
    np.testing.assert_allclose(out["prob"], (post * (1 - pre)).numpy(), atol=1e-6)
    np.testing.assert_array_equal(out["pred"], ((post >= 0.5) & ~(pre >= 0.5)).numpy())


def test_any_class_is_derived_for_older_result_tables(tmp_path):
    from p037.analyze import prepare

    cfg = make_cfg(tmp_path, dataset="urbansarfloods", score_class="any", proxies=["slope_mean"],
                   proxy_table=str(tmp_path / "p.csv"))
    n = 60
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"event": "e", "tile_id": [f"t{i}" for i in range(n)], "tile_row": 0, "tile_col": range(n),
                      "n_valid": 100, "mc_var_flood": 1e-3, "mean_entropy_flood": 0.1, "frac_flood": 0.1})
    for c in ("open", "urban"):
        for k in ("tp", "fp", "fn", "tn"):
            t[f"flood_{c}_{k}"] = rng.integers(0, 20, n)
    t.to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"sample_id": [*t["tile_id"], *[f"r{i}" for i in range(80)]], "set": ["test"] * n + ["b_usf_train"] * 80,
                  "lon": 0.0, "lat": 0.0, "slope_mean__tile": rng.normal(size=n + 80)}).to_csv(tmp_path / "p.csv", index=False)
    df, *_ = prepare(cfg, tmp_path / "t.csv")
    expect = (t["flood_open_fp"] + t["flood_open_fn"] + t["flood_urban_fn"]).to_numpy()
    np.testing.assert_array_equal(df["err_px"].to_numpy(), expect)


# ------------------------------------------------------------------------------------- Experiment A adapter
def test_label_rules():
    from p037.datasets.amazon_public import LABEL_KINDS, rule_for

    covered = np.array([True, True, True, False])
    monthly = LABEL_KINDS["gsw_monthly"].apply(np.array([0, 1, 2, 2]), covered)
    assert monthly.tolist() == [INVALID, 0, 1, INVALID]  # no data, not water, water, outside the label file
    with pytest.raises(ValueError, match="wrong label_kind"):
        LABEL_KINDS["gsw_monthly"].apply(np.array([0, 1, 2, 77]), np.ones(4, bool))
    occ = rule_for({"label_kind": "gsw_occurrence", "water_min": 80}).apply(np.array([0, 79, 80, 255]), np.ones(4, bool))
    assert occ.tolist() == [0, 0, 1, INVALID]


def _amazon_source(tmp_path, label_values=(1, 2)):
    """A 10 m UTM 8-bit scene and a ~30 m geographic label whose water is everything east of a meridian."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_points

    src = tmp_path / "amazon_src"
    src.mkdir()
    h, w = 384, 512
    crs, tr = "EPSG:32720", from_origin(700000, 9700000, 10, 10)
    rng = np.random.default_rng(0)
    img = rng.integers(60, 240, (2, h, w)).astype(np.uint8)
    with rasterio.open(src / "s1.tif", "w", driver="GTiff", width=w, height=h, count=2, dtype="uint8", crs=crs,
                       transform=tr) as o:
        o.write(img)
        o.set_band_description(1, "VV (8-bit)")
        o.set_band_description(2, "VH (8-bit)")
    # corners of the scene in lon / lat, and a label grid that covers it with a margin
    lon, lat = warp_points(crs, "EPSG:4326", [700000, 700000 + 10 * w], [9700000, 9700000 - 10 * h])
    step = 0.00027
    left, top = min(lon) - 0.01, max(lat) + 0.01
    n_cols, n_rows = int((max(lon) - min(lon) + 0.02) / step) + 1, int((max(lat) - min(lat) + 0.02) / step) + 1
    split_lon = (lon[0] + lon[1]) / 2
    cols_lon = left + (np.arange(n_cols) + 0.5) * step
    label = np.where(cols_lon[None, :] > split_lon, label_values[1], label_values[0]) * np.ones((n_rows, 1), int)
    with rasterio.open(src / "label.tif", "w", driver="GTiff", width=n_cols, height=n_rows, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(left, top, step, step)) as o:
        o.write(label.astype(np.uint8), 1)
    (src / "scenes.yaml").write_text(
        "rio_test:\n  image: s1.tif\n  labels: ['label*.tif']\n  label_kind: gsw_monthly\n  note: test\n")
    return src, split_lon, (crs, tr, h, w)


def test_amazon_scene_labels_land_on_the_image_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import json

    from rasterio.warp import transform as warp_points

    from p037.datasets.amazon_public import AmazonPublic

    src, split_lon, (crs, tr, h, w) = _amazon_source(tmp_path)
    ds = AmazonPublic(tmp_path / "data" / "amazon_public")
    assert ds.list_events(str(src)) == ["rio_test"] and ds.test_events == []
    ds.prepare_scene("rio_test", str(src))
    assert ds.test_events == ["rio_test"] and ds.meta().channels == ["VV", "VH"]
    f = ds.scene_files("rio_test")
    with rasterio.open(f["gt"]) as g, rasterio.open(f["valid"]) as v, rasterio.open(f["image"]) as im:
        gt, valid = g.read(1), v.read(1).astype(bool)
        assert list(im.descriptions) == ["VV", "VH"] and (g.crs, g.transform) == (im.crs, im.transform)
    # expected label of every pixel from its own centre longitude
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = tr.c + tr.a * cols.ravel(), tr.f + tr.e * rows.ravel()
    lon, _ = warp_points(crs, "EPSG:4326", x.tolist(), y.tolist())
    lon = np.array(lon).reshape(h, w)
    sure = valid & (np.abs(lon - split_lon) > 0.0003)  # more than one label cell away from the boundary
    np.testing.assert_array_equal(gt[sure], (lon[sure] > split_lon).astype(np.uint8))
    assert valid.mean() > 0.5 and (gt[~valid] == INVALID).all()
    info = json.loads((ds.scene_dir("rio_test") / "info.json").read_text())
    assert set(info["label_value_counts"]) == {"1", "2"} and info["labelled_share_of_valid"] == pytest.approx(1.0)


def test_amazon_refuses_a_wrong_label_coding_and_labels_elsewhere(tmp_path):
    pytest.importorskip("rasterio")
    from p037.datasets.amazon_public import AmazonPublic

    src, *_ = _amazon_source(tmp_path, label_values=(0, 100))  # an occurrence raster declared as monthly history
    with pytest.raises(ValueError, match="wrong label_kind"):
        AmazonPublic(tmp_path / "d1").prepare_scene("rio_test", str(src))
    (src / "scenes.yaml").write_text("rio_test:\n  image: s1.tif\n  labels: ['nothing*.tif']\n  label_kind: binary\n")
    with pytest.raises(FileNotFoundError):
        AmazonPublic(tmp_path / "d2").prepare_scene("rio_test", str(src))
    (src / "scenes.yaml").write_text("rio_test:\n  image: s1.tif\n  labels: ['label.tif']\n  label_kind: nonsense\n")
    with pytest.raises(KeyError, match="label_kind"):
        AmazonPublic(tmp_path / "d3").prepare_scene("rio_test", str(src))


def test_experiment_a_runs_from_scene_to_report(tmp_path, monkeypatch):
    """The pretrained route end to end: scene + public label -> tiles with coordinates -> proxies -> report."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("rasterio")
    safetensors = pytest.importorskip("safetensors.torch")
    from p037.analyze import run_analysis
    from p037.evaluate import evaluate_scenes
    from p037.model import build_model
    from p037.proxies import check_coverage, merge_new

    torch.set_num_threads(2)
    src, *_ = _amazon_source(tmp_path)
    cfg = Config(name="a", data_root=str(tmp_path / "data"), dataset="amazon_public", init="pretrained",
                 input_channels=["VV", "VH", "VV", "VH"], target_classes=[1], score_class="water",
                 reference_set="a_amazon", tile=64, window=128, crop=32, mc_passes=3, n_bootstrap=5,
                 min_positive_px=20, proxies=["slope_mean", "tree_frac"], proxy_table=str(tmp_path / "proxies.csv"))
    # stand-in for the published weights (no download in tests)
    torch.manual_seed(0)
    weights = tmp_path / "w.safetensors"
    safetensors.save_file(build_model(cfg, init="random").state_dict(), str(weights))
    monkeypatch.setenv("S1WATER_WEIGHTS", str(weights))

    ds = get_dataset(cfg)
    ds.prepare_scene("rio_test", str(src))
    tiles = pd.read_csv(evaluate_scenes(cfg, "cpu"))
    assert {"water_water_f1", "mc_var_water", "lon", "lat"} <= set(tiles.columns)
    assert tiles["lon"].between(-62, -60).all() and tiles["lat"].between(-3.5, -2).all()  # UTM 20S, near the scene
    # a reference distribution exists, the new tiles have no proxies yet
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"sample_id": [f"AMZ_{i}" for i in range(50)], "set": "a_amazon", "lon": -60.0, "lat": -3.0})
    for p in cfg.proxies:
        ref[f"{p}__tile"] = rng.normal(size=50)
    ref["water_frac__tile"] = 0.1
    ref.to_csv(cfg.proxy_table, index=False)
    cover = check_coverage(cfg)
    assert cover["missing"] == len(tiles)
    todo = pd.read_csv(cover["missing_file"])
    assert {"sample_id", "set", "region", "lon", "lat"} <= set(todo.columns) and todo["lon"].notna().all()
    # what analysis/proxy/compute_proxies.py would return for them
    values = pd.DataFrame({"sample_id": todo["sample_id"], "water_frac__tile": 0.7})
    for p in cfg.proxies:
        values[f"{p}__tile"] = rng.normal(size=len(todo))
    values.to_csv(tmp_path / "new.csv", index=False)
    merge_new(cfg, tmp_path / "new.csv")
    assert check_coverage(cfg)["missing"] == 0
    table = pd.read_csv(cfg.proxy_table)
    assert len(table) == 50 + len(tiles) and table.loc[table["set"] == "test", "mostly_water"].all()
    report = run_analysis(cfg)
    assert "## 2. Where the test tiles sit" in report.read_text()


def test_compare_runs_uses_common_tiles_only(tmp_path):
    from p037.analyze import compare_runs

    a = pd.DataFrame({"tile_id": ["t1", "t2", "t3"], "event": "e", "flood_open_tp": [10, 0, 5], "flood_open_fp": [0, 0, 5],
                      "flood_open_fn": [10, 0, 0], "flood_open_tn": [80, 100, 90]})
    b = a[a["tile_id"] != "t3"].assign(flood_open_tp=[20, 0], flood_open_fn=[0, 0])
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    out = compare_runs({"a": tmp_path / "a.csv", "b": tmp_path / "b.csv"}).set_index(["event", "run"])
    assert out.loc[("all", "a"), "tiles"] == 2 and out.loc[("all", "a"), "recall"] == pytest.approx(0.5)
    assert out.loc[("e", "b"), "f1"] == pytest.approx(1.0)
