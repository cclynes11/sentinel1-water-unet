"""List the UrbanSARFloods test chips and read the scene georeferencing (headers only).

For each test event, writes <event>_files.json (the testing_case_256 chip names, from the
Hugging Face tree API) and orig_geo.json (size and transform of testing_case_orig/<event>
SAR.tif, read over HTTP range requests: only the GeoTIFF header is transferred). Two chips per
event are opened to check that chip ID_r_c covers rows r*256.. and columns c*256.. of the scene.

Usage: python fetch_test_meta.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rasterio
import requests

REPO = "S1Floodbenchmark/UrbanSARFloods_v1"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main/testing_case_256/"
FILE = f"/vsicurl/https://huggingface.co/datasets/{REPO}/resolve/main/"
ORIG = {
    "20210727_Weihui": "20210727_Weihui_SAR.tif",
    "20230609_NovaKakhovka": "20230609_NovaKakhovka_SAR.tif",
    "20231201_Jubba_1": "20231201_Jubba_SAR.tif",
    "20231201_Jubba_2": "20231201_Jubba_2_SAR.tif",
}


def open_retry(path):
    for k in range(6):
        try:
            return rasterio.open(path)
        except rasterio.errors.RasterioIOError:
            time.sleep(2 ** k)
    raise RuntimeError(path)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geo = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for ev, fn in ORIG.items():
            files, url = [], TREE + ev
            while url:
                r = requests.get(url, params=None if "cursor" in url else {"limit": 1000},
                                 timeout=120)
                r.raise_for_status()
                files += [x["path"].split("/")[-1] for x in r.json()]
                url = r.links.get("next", {}).get("url")
            (out / f"{ev}_files.json").write_text(json.dumps(files))
            with open_retry(FILE + f"testing_case_orig/{ev}/{fn}") as src:
                t = src.transform
                geo[ev] = dict(width=src.width, height=src.height, transform=list(t)[:6])
            for chip in (files[0], files[len(files) // 2]):
                rr, cc = (int(v) for v in chip.replace("_SAR.tif", "").split("_ID_")[1].split("_"))
                with open_retry(FILE + f"testing_case_256/{ev}/{chip}") as c:
                    exp = (t.c + cc * 256 * t.a, t.f + rr * 256 * t.e)
                    err = max(abs(c.bounds.left - exp[0]), abs(c.bounds.top - exp[1]))
                    print(f"{ev} {chip}: grid check error {err:.2e} deg")
                    assert err < 1e-6
            print(ev, len(files), "chips", geo[ev])
    (out / "orig_geo.json").write_text(json.dumps(geo, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
