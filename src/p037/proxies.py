"""Stage 4: make sure every training patch and test tile has proxy values.

The proxies themselves are computed by the scripts of PR #3 (``analysis/proxy/``): Copernicus DEM
GLO-30 (slope, elevation range), GLO-30 HAND, ESA WorldCover 2021 (vegetation, tree cover), GHSL
(built-up fraction, building height), each at tile scale (~4.7 km) and neighbourhood scale (~24 km).
WSF2019 is left out on purpose: UrbanSARFloods builds its urban-flood labels from it.

For UrbanSARFloods that table already exists (``analysis/proxy/outputs/proxy_values.csv.gz``:
18,004 training patches, 9,176 test tiles, 600 Amazon reference points). This stage checks that
it covers what an experiment needs, and for anything missing writes a sample table in the format
``analysis/proxy/compute_proxies.py`` reads, with the command to run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, Config
from .datasets import get_dataset

log = logging.getLogger("p037.proxies")


def needed_samples(cfg: Config) -> pd.DataFrame:
    """Every location this experiment needs proxies for: training patches and scored tiles."""
    ds = get_dataset(cfg)
    parts = []
    if ds.has_cache():
        p = ds.patches()
        p = p[p["split"] == "train"]
        parts.append(pd.DataFrame({"sample_id": p["patch_id"], "set": cfg.reference_set, "event": p["event"],
                                   "lon": p["lon"], "lat": p["lat"], "region": p["event"]}))
    tiles = cfg.out / "tiles.csv"
    if tiles.exists():
        t = pd.read_csv(tiles, usecols=["tile_id", "event"])
        parts.append(pd.DataFrame({"sample_id": t["tile_id"], "set": "test", "event": t["event"],
                                   "region": t["event"]}))
    if not parts:
        raise FileNotFoundError("nothing to check yet: build the cache or run `p037 evaluate` first")
    return pd.concat(parts, ignore_index=True)


def check_coverage(cfg: Config) -> dict:
    """Compare what the experiment needs with the proxy table; write the missing samples if any."""
    table_path = cfg.resolve(cfg.proxy_table)
    need = needed_samples(cfg)
    have = pd.read_csv(table_path, usecols=["sample_id", "set"]) if table_path.exists() else pd.DataFrame(
        {"sample_id": [], "set": []})
    missing = need[~need["sample_id"].isin(have["sample_id"])]
    out = {"needed": len(need), "covered": len(need) - len(missing), "missing": len(missing),
           "by_set": need.assign(covered=need["sample_id"].isin(have["sample_id"]))
           .groupby("set")["covered"].agg(["size", "sum"]).rename(columns={"size": "needed", "sum": "covered"})}
    log.info("proxy coverage of %s: %d of %d samples\n%s", cfg.name, out["covered"], out["needed"], out["by_set"])
    if len(missing):
        path = cfg.out / "proxy_samples_missing.csv"
        located = missing.dropna(subset=["lon", "lat"]) if "lon" in missing else missing.iloc[:0]
        located.to_csv(path, index=False)
        script: Path = REPO_ROOT / "analysis" / "proxy" / "compute_proxies.py"
        log.warning("%d samples have no proxies. %d of them have coordinates and were written to %s.\n"
                    "Compute them (needs internet: AWS Open Data and the JRC server) with:\n"
                    "  cd %s && python %s %s %s --sets %s\n"
                    "then append the result to %s. Test tiles of a new dataset need lon / lat first "
                    "(see analysis/proxy/build_samples.py).",
                    len(missing), len(located), path, script.parent, script.name, path,
                    cfg.out / "proxies_new.csv", ",".join(sorted(set(missing["set"]))), table_path)
        out["missing_file"] = path
    return out
