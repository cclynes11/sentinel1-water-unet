"""Proxy set (a): random water-adjacent locations in the Amazon basin.

Stand-in for the training patches of the Wagner et al. (2024) U-Net (Amazon basin, 2021-2022,
59,387 patches of 256 x 256 px at 10 m whose exact locations are not published).

1. Points are drawn uniformly (in degrees) inside the Natural Earth "AMAZON BASIN" polygon.
2. A point is kept when the tile window centred on it (one UrbanSARFloods test chip, ~4.7 km)
   holds between 1 % and 95 % open water in a coarse (~280 m) ESA WorldCover 2021 water map
   built from the COG overviews; this is the "water-adjacent" criterion.

Usage: python sample_amazon.py OUT_CSV [N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import TILE_DEG, WC_WATER, Grid, load_layer, three_deg_tiles, wc_url
from rasterio.enums import Resampling
from shapely.geometry import Point, shape
from shapely.prepared import prep

HERE = Path(__file__).parent
SEED = 20240306  # fixed seed so the sample is reproducible
COARSE = 1 / 400.0  # 0.0025 deg


def main(out_csv: str, n: int = 600) -> None:
    gj = json.loads((HERE / "amazon_basin_ne10m.geojson").read_text())
    basin = shape(gj["features"][0]["geometry"])
    pb = prep(basin)
    g = Grid(basin.bounds, res=COARSE)
    wc = load_layer([wc_url(*t) for t in three_deg_tiles(g.bounds)], g, np.uint8, 0,
                    Resampling.nearest, decimate=True, workers=8)
    water = (wc == WC_WATER).astype(np.float32)
    valid = wc > 0
    # fraction of water in a TILE_DEG box around every coarse cell (box filter via cumsum)
    k = int(round(TILE_DEG / COARSE))
    def boxsum(a):
        c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        h = k // 2
        out = np.full(a.shape, np.nan, np.float32)
        out[h:-h or None, h:-h or None] = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k])[
            : a.shape[0] - 2 * h, : a.shape[1] - 2 * h]
        return out
    wfrac = boxsum(water) / np.maximum(boxsum(valid.astype(np.float32)), 1)

    rng = np.random.default_rng(SEED)
    w, s, e, nn = basin.bounds
    rows, tried = [], 0
    while len(rows) < n:
        lon, lat = rng.uniform(w, e), rng.uniform(s, nn)
        tried += 1
        if not pb.contains(Point(lon, lat)):
            continue
        r, c = g.rowcol(lon, lat)
        f = wfrac[min(r, g.height - 1), min(c, g.width - 1)]
        if np.isfinite(f) and 0.01 <= f <= 0.95:
            rows.append(dict(sample_id=f"AMZ_{len(rows):04d}", lon=lon, lat=lat,
                             coarse_water_frac=float(f)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    inside = sum(pb.contains(Point(rng.uniform(w, e), rng.uniform(s, nn))) for _ in range(20000))
    print(f"kept {len(df)} of {tried} draws; basin fill of bbox ~{inside / 20000:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600)
