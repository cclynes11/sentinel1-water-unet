"""Build the sample table (one row per tile-scale window) for all sets.

* a_amazon   : random water-adjacent Amazon points (sample_amazon.py)
* b_usf_train: UrbanSARFloods training chips (512 x 512 px, listed in Train_dataset.txt).
               Each chip is split into its four 256 x 256 px quadrants so that the tile window
               has the same footprint as a test chip.
* test       : every UrbanSARFloods test chip (testing_case_256, 256 x 256 px), footprint from
               the chip id (row, col) and the georeferencing of testing_case_orig.

Usage: python build_samples.py USF_META_DIR TEST_META_DIR OUT_CSV
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RES = 0.00016666
ID_RE = re.compile(r"^(?P<event>.+?)_ID_(?P<r>\d+)_(?P<c>\d+)_(?:SAR|GT)\.tif$")


def usf_train(meta_dir: Path) -> pd.DataFrame:
    train = {Path(line.strip()).name.replace("_GT.tif", "").replace("_SAR.tif", "")
             for line in (meta_dir / "urban_sar_floods__Train_dataset.txt").read_text().split()
             if line.strip()}
    valid = {Path(line.strip()).name.replace("_GT.tif", "").replace("_SAR.tif", "")
             for line in (meta_dir / "urban_sar_floods__Valid_dataset.txt").read_text().split()
             if line.strip()}
    recs = {}
    for line in (meta_dir / "chips.jsonl").read_text().splitlines():
        d = json.loads(line)
        if "/SAR/" not in d["name"] or "tiepoint" not in d:
            continue
        name = Path(d["name"]).name
        m = ID_RE.match(name)
        key = name.replace("_SAR.tif", "")
        recs[key] = dict(chip=key, event=m["event"], folder=d["name"].split("/")[1],
                         left=d["tiepoint"][3], top=d["tiepoint"][4],
                         sx=d["scale"][0], sy=d["scale"][1],
                         w=d["width"][0], h=d["height"][0])
    chips = pd.DataFrame(recs.values())
    chips["split"] = ["train" if c in train else "valid" if c in valid else "none"
                      for c in chips["chip"]]
    print("chips in archive by split:", chips["split"].value_counts().to_dict())
    print("train list entries:", len(train), " found in archive:",
          int((chips["split"] == "train").sum()))
    chips.to_csv(HERE / "outputs" / "usf_chip_footprints.csv.gz", index=False)
    rows = []
    for ch in chips[chips["split"] == "train"].itertuples():
        for qi, (fx, fy) in enumerate(((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))):
            rows.append(dict(sample_id=f"{ch.chip}_q{qi}", set="b_usf_train", event=ch.event,
                             chip=ch.chip, folder=ch.folder,
                             lon=ch.left + fx * ch.w * ch.sx, lat=ch.top - fy * ch.h * ch.sy,
                             region=ch.event))
    return pd.DataFrame(rows)


def test_chips(test_dir: Path) -> pd.DataFrame:
    geo = json.loads((test_dir / "orig_geo.json").read_text())
    rows = []
    for ev, g in geo.items():
        a, b, x0, d, e, y0 = g["transform"]
        left0, top0 = x0, y0  # rasterio transform of the full scene (pixel corner)
        for fn in json.loads((test_dir / f"{ev}_files.json").read_text()):
            m = ID_RE.match(fn)
            r, c = int(m["r"]), int(m["c"])
            rows.append(dict(sample_id=fn.replace("_SAR.tif", ""), set="test", event=ev,
                             chip=fn.replace("_SAR.tif", ""), folder="",
                             lon=left0 + (c + 0.5) * 256 * a, lat=top0 + (r + 0.5) * 256 * e,
                             region=ev))
    return pd.DataFrame(rows)


def main(usf_meta: str, test_meta: str, out_csv: str) -> None:
    amz = pd.read_csv(HERE / "outputs" / "amazon_samples.csv")
    amz = amz.assign(set="a_amazon", event="Amazon", chip="", folder="", region=amz["sample_id"])
    parts = [amz[["sample_id", "set", "event", "chip", "folder", "lon", "lat", "region"]]]
    if (Path(usf_meta) / "urban_sar_floods__Train_dataset.txt").exists():
        parts.append(usf_train(Path(usf_meta)))
    else:
        print("WARNING: Train_dataset.txt not available yet; training set skipped")
    parts.append(test_chips(Path(test_meta)))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(df.groupby(["set", "event"]).size().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
