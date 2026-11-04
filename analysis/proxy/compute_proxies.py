"""Compute every proxy at two scales for every sample of a sample table.

* tile scale        : a TILE_DEG box (one 256 x 256 px UrbanSARFloods test chip, ~4.7 km at the
                      equator) centred on the sample, every 1" pixel used;
* neighbourhood     : a NEIGH_DEG box (5 x 5 test chips, ~24 km) centred on the sample, every
                      3rd pixel in each direction (~90 m sampling) to keep it fast.

Samples are processed region by region: one region per Amazon point, one per UrbanSARFloods
event (the layers of the whole event are loaded once).

Usage: python compute_proxies.py SAMPLES_CSV OUT_CSV [--sets a_amazon,b_usf_train,test]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from common import NEIGH_DEG, TILE_DEG, Stack

PAD = NEIGH_DEG / 2 + 0.01


def region_rows(sub: pd.DataFrame) -> list[dict]:
    b = (sub.lon.min() - PAD, sub.lat.min() - PAD, sub.lon.max() + PAD, sub.lat.max() + PAD)
    st = Stack(b)

    def one(row):
        out = {"sample_id": row.sample_id}
        for scale, size, step in (("tile", TILE_DEG, 1), ("neigh", NEIGH_DEG, 3)):
            for k, v in st.stats(row.lon, row.lat, size, step).items():
                out[f"{k}__{scale}"] = v
        return out

    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, sub.itertuples()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--sets", default="a_amazon,b_usf_train,test")
    args = ap.parse_args()
    df = pd.read_csv(args.samples)
    df = df[df.set.isin(args.sets.split(","))]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["sample_id"]).sample_id)
    regions = [(r, g) for r, g in df.groupby("region", sort=False) if not set(g.sample_id) <= done]
    print(f"{len(regions)} regions to do")
    small = [(r, g) for r, g in regions if len(g) == 1]
    big = [(r, g) for r, g in regions if len(g) > 1]
    header = not out_path.exists()

    def flush(rows):
        nonlocal header
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=header, index=False)
        header = False

    t0 = time.time()
    # single-point regions (Amazon): several in parallel
    with ThreadPoolExecutor(10) as ex:
        buf = []
        for i, rows in enumerate(ex.map(lambda rg: region_rows(rg[1]), small)):
            buf += rows
            if len(buf) >= 25:
                flush(buf)
                buf = []
                print(f"  {i + 1}/{len(small)} points  {time.time() - t0:.0f}s", flush=True)
        if buf:
            flush(buf)
    for r, g in big:
        t1 = time.time()
        flush(region_rows(g))
        print(f"  region {r}: {len(g)} samples in {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
