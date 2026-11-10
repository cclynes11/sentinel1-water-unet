"""Google Open Buildings v3 footprint fraction, where the dataset has coverage.

Open Buildings v3 covers Africa, South / South-East Asia and Latin America and the Caribbean.
Among our samples that means: all of set (a) (Amazon), the African UrbanSARFloods training
scenes (Beledweyne 2018 and 2023, Beira 2019, Lokoja/Niger 2022) and the Jubba test scenes.
Everything else gets NaN ("not covered"), never zero.

The S2 level-4 CSV files are streamed and never stored. Buildings with confidence >= 0.70 are
binned by centroid on a 0.0025 deg grid (area in m2), and window sums are taken from that grid.

Usage: python open_buildings.py SAMPLES_CSV OUT_CSV
"""

from __future__ import annotations

import gzip
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import s2sphere as s2
from common import NEIGH_DEG, TILE_DEG

BASE = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/"
COVERED_EVENTS = {"Amazon", "20180508_Somalia", "20231114_Beledweyne", "20190320_Beira",
                  "20221013_Niger", "20231201_Jubba_1", "20231201_Jubba_2"}
CELL = 0.0025
MIN_CONF = 0.70


def tokens_for(df):
    h = NEIGH_DEG / 2 + CELL
    out = set()
    for r in df.itertuples():
        for dx in (-h, 0, h):
            for dy in (-h, 0, h):
                ll = s2.LatLng.from_degrees(r.lat + dy, r.lon + dx)
                out.add(s2.CellId.from_lat_lng(ll).parent(4).to_token())
    return out


def keep_mask(df):
    """Set of 0.1 deg cells touched by some neighbourhood window (to drop far-away rows)."""
    h = NEIGH_DEG / 2 + 0.1
    keep = set()
    for r in df.itertuples():
        for i in range(math.floor((r.lat - h) * 10), math.floor((r.lat + h) * 10) + 1):
            for j in range(math.floor((r.lon - h) * 10), math.floor((r.lon + h) * 10) + 1):
                keep.add((i, j))
    return keep


def stream_cell(token, keep, acc, lock):
    url = BASE + f"{token}_buildings.csv.gz"
    r = requests.get(url, stream=True, timeout=600)
    if r.status_code == 404:
        return token, 0
    r.raise_for_status()
    n = 0
    with gzip.GzipFile(fileobj=r.raw) as gz:
        for ch in pd.read_csv(gz, usecols=["latitude", "longitude", "area_in_meters",
                                           "confidence"], chunksize=2_000_000):
            ch = ch[ch.confidence >= MIN_CONF]
            k = list(zip(np.floor(ch.latitude * 10).astype(int),
                         np.floor(ch.longitude * 10).astype(int), strict=True))
            ch = ch[[x in keep for x in k]]
            if ch.empty:
                continue
            gi = np.floor(ch.latitude / CELL).astype(np.int64)
            gj = np.floor(ch.longitude / CELL).astype(np.int64)
            s = ch.groupby([gi, gj]).area_in_meters.sum()
            with lock:
                for (a, b), v in s.items():
                    acc[(a, b)] += v
            n += len(ch)
    return token, n


def main(samples_csv, out_csv):
    import threading

    df = pd.read_csv(samples_csv)
    df = df[df.event.isin(COVERED_EVENTS)]
    toks = sorted(tokens_for(df))
    keep = keep_mask(df)
    acc: dict = defaultdict(float)
    lock = threading.Lock()
    print(f"{len(df)} samples, {len(toks)} S2 cells", flush=True)
    with ThreadPoolExecutor(3) as ex:
        for tok, n in ex.map(lambda t: stream_cell(t, keep, acc, lock), toks):
            print(f"  cell {tok}: {n:,} buildings kept", flush=True)
    rows = []
    for r in df.itertuples():
        out = {"sample_id": r.sample_id}
        for scale, size in (("tile", TILE_DEG), ("neigh", NEIGH_DEG)):
            h = size / 2
            i0, i1 = math.floor((r.lat - h) / CELL), math.floor((r.lat + h) / CELL)
            j0, j1 = math.floor((r.lon - h) / CELL), math.floor((r.lon + h) / CELL)
            tot = sum(acc.get((i, j), 0.0) for i in range(i0, i1) for j in range(j0, j1))
            area = ((i1 - i0) * CELL * 111_320) * ((j1 - j0) * CELL * 111_320
                                                   * math.cos(math.radians(r.lat)))
            out[f"ob_bldg_frac__{scale}"] = tot / area
        rows.append(out)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
