"""Shared raster access and per-window proxy statistics.

Every proxy is computed on a common 1 arc-second (~30 m) latitude/longitude grid:

* Copernicus DEM GLO-30 (AWS Open Data, COG)          -> elevation, slope
* GLO-30 HAND (ASF, AWS Open Data, COG, derived from GLO-30) -> height above nearest drainage
* ESA WorldCover 2021 v200 (AWS Open Data, COG, 10 m)  -> vegetation / tree / water fractions
* GHSL GHS-BUILT-S 2020 R2023A (JRC, 100 m Mollweide)   -> built-up surface fraction
* GHSL GHS-BUILT-H ANBH 2018 R2023A (JRC, 100 m)        -> average net building height

WSF2019 is deliberately not used: UrbanSARFloods uses it to build its urban-flood labels.
"""

from __future__ import annotations

import math
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import from_bounds

ARCSEC = 1.0 / 3600.0
TILE_DEG = 256 * 0.00016666  # one UrbanSARFloods test chip (256 px at 0.00016666 deg)
NEIGH_DEG = 5 * TILE_DEG  # neighbourhood: 5 x 5 test chips centred on the tile
EARTH_R = 6371008.8

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="268435456",
    GDAL_CACHEMAX=1024,
)

GHSL_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
GHSL_S = (GHSL_BASE + "/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0/"
          "tiles/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
GHSL_H = (GHSL_BASE + "/GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100/"
          "V1-0/tiles/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")

# WorldCover 2021 classes
WC_TREE = 10
WC_VEG = (10, 20, 30, 40, 90, 95, 100)  # tree, shrub, grass, crop, herb. wetland, mangrove, moss
WC_WATER = 80

CACHE = Path(os.environ.get("PROXY_CACHE", "/tmp/proxy_cache"))
_lock = threading.Lock()
_dl_locks: dict[str, threading.Lock] = {}


# ----------------------------------------------------------------------------- tile names
def _ns(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"


def _ew(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"


def dem_url(lat: int, lon: int) -> str:
    n = f"Copernicus_DSM_COG_10_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


def hand_url(lat: int, lon: int) -> str:
    n = f"Copernicus_DSM_COG_10_{_ns(lat)}_00_{_ew(lon)}_00_HAND.tif"
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/{n}"


def wc_url(lat: int, lon: int) -> str:
    return ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{_ns(lat)}{_ew(lon)}_Map.tif")


def one_deg_tiles(bounds):
    w, s, e, n = bounds
    return [(la, lo) for la in range(math.floor(s), math.ceil(n))
            for lo in range(math.floor(w), math.ceil(e))]


def three_deg_tiles(bounds):
    w, s, e, n = bounds
    las = range(math.floor(s / 3) * 3, math.ceil(n / 3) * 3, 3)
    los = range(math.floor(w / 3) * 3, math.ceil(e / 3) * 3, 3)
    return [(la, lo) for la in las for lo in los]


# ----------------------------------------------------------------------------- grid
class Grid:
    """A 1 arc-second lat/lon grid covering `bounds` (west, south, east, north)."""

    def __init__(self, bounds, res=ARCSEC):
        w, s, e, n = bounds
        self.res = res
        self.west = math.floor(w / res) * res
        self.north = math.ceil(n / res) * res
        self.width = int(math.ceil((e - self.west) / res))
        self.height = int(math.ceil((self.north - s) / res))
        self.transform = from_origin(self.west, self.north, res, res)
        self.bounds = (self.west, self.north - self.height * res,
                       self.west + self.width * res, self.north)

    @property
    def shape(self):
        return (self.height, self.width)

    def rowcol(self, lon, lat):
        return (int(round((self.north - lat) / self.res)), int(round((lon - self.west) / self.res)))

    def lats(self):
        return self.north - (np.arange(self.height) + 0.5) * self.res


def _read_into(url, grid: Grid, dest: np.ndarray, resampling, decimate: bool, dst_nodata):
    """Read the part of one COG tile overlapping `grid` and warp it into `dest` (in place)."""
    try:
        src = rasterio.open(url)
    except rasterio.errors.RasterioIOError:
        return False  # no tile here (ocean) or not covered
    with src:
        w, s, e, n = grid.bounds
        sw, ss, se, sn = src.bounds
        ib = (max(w, sw), max(s, ss), min(e, se), min(n, sn))
        if ib[0] >= ib[2] or ib[1] >= ib[3]:
            return False
        win = from_bounds(*ib, transform=src.transform).round_offsets().round_lengths()
        if win.width <= 0 or win.height <= 0:
            return False
        if decimate:  # let GDAL use overviews: read at ~ the destination resolution
            oh = max(1, int(round((ib[3] - ib[1]) / grid.res)))
            ow = max(1, int(round((ib[2] - ib[0]) / grid.res)))
            arr = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest)
            wt = src.window_transform(win) * rasterio.Affine.scale(win.width / ow, win.height / oh)
        else:
            arr = src.read(1, window=win)
            wt = src.window_transform(win)
        # destination sub-window covering the intersection (plus one pixel of margin)
        r0 = max(int(math.floor((grid.north - ib[3]) / grid.res)) - 1, 0)
        r1 = min(int(math.ceil((grid.north - ib[1]) / grid.res)) + 1, grid.height)
        c0 = max(int(math.floor((ib[0] - grid.west) / grid.res)) - 1, 0)
        c1 = min(int(math.ceil((ib[2] - grid.west) / grid.res)) + 1, grid.width)
        sub_t = grid.transform * rasterio.Affine.translation(c0, r0)
        tmp = np.full((r1 - r0, c1 - c0), dst_nodata, dtype=dest.dtype)
        reproject(arr, tmp, src_transform=wt, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=sub_t, dst_crs="EPSG:4326", dst_nodata=dst_nodata,
                  resampling=resampling)
        valid = tmp != dst_nodata if not np.isnan(dst_nodata) else ~np.isnan(tmp)
        with _lock:
            dest[r0:r1, c0:c1][valid] = tmp[valid]
    return True


def load_layer(urls, grid: Grid, dtype, dst_nodata, resampling=Resampling.nearest,
               decimate=False, workers=6):
    dest = np.full(grid.shape, dst_nodata, dtype=dtype)
    with rasterio.Env(**GDAL_ENV):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda u: _read_into(u, grid, dest, resampling, decimate, dst_nodata),
                        urls))
    return dest


# ----------------------------------------------------------------------------- GHSL
MOLL_X0, MOLL_Y0 = -18041000.0, 9000000.0


def ghsl_tiles(bounds):
    """GHSL 1000 km Mollweide tiles (row, col) covering a lat/lon box."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    w, s, e, n = bounds
    xs, ys = [], []
    for lon in np.linspace(w, e, 25):
        for lat in np.linspace(s, n, 25):
            x, y = t.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    cs = range(int((min(xs) - MOLL_X0) // 1e6) + 1, int((max(xs) - MOLL_X0) // 1e6) + 2)
    rs = range(int((MOLL_Y0 - max(ys)) // 1e6) + 1, int((MOLL_Y0 - min(ys)) // 1e6) + 2)
    return [(r, c) for r in rs for c in cs]


def ghsl_local(template: str, r: int, c: int) -> str | None:
    """Download (once) a GHSL tile zip and return a /vsizip/ path to its GeoTIFF."""
    url = template.format(r=r, c=c)
    path = CACHE / "ghsl" / url.rsplit("/", 1)[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with _dl_locks.setdefault(path.name, threading.Lock()):
        for attempt in range(8):
            if path.exists():
                break
            try:
                resp = requests.get(url, timeout=600)
                if resp.status_code == 404:
                    path.write_bytes(b"")  # no tile (ocean)
                else:
                    resp.raise_for_status()
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(resp.content)
                    zipfile.ZipFile(tmp).testzip()
                    tmp.rename(path)
            except (requests.RequestException, zipfile.BadZipFile):
                time.sleep(2 ** min(attempt, 5))
        else:
            raise RuntimeError(f"could not download {url}")
    if path.stat().st_size == 0:
        return None
    tif = [m for m in zipfile.ZipFile(path).namelist() if m.endswith(".tif")][0]
    return f"/vsizip/{path}/{tif}"


def load_ghsl(grid: Grid):
    """Built-up fraction (0-1) and ANBH building height (m) on `grid` (nearest, 100 m cells)."""
    built = np.full(grid.shape, np.nan, dtype=np.float32)
    height = np.full(grid.shape, np.nan, dtype=np.float32)
    for r, c in ghsl_tiles(grid.bounds):
        for tmpl, dest, scale in ((GHSL_S, built, 1e-4), (GHSL_H, height, 1.0)):
            p = ghsl_local(tmpl, r, c)
            if p is None:
                continue
            with rasterio.Env(**GDAL_ENV), rasterio.open(p) as src:
                tmp = np.full(grid.shape, np.nan, dtype=np.float32)
                reproject(rasterio.band(src, 1), tmp, src_nodata=src.nodata,
                          dst_transform=grid.transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                          resampling=Resampling.nearest)
                ok = ~np.isnan(tmp)
                dest[ok] = tmp[ok] * scale
    return built, height


# ----------------------------------------------------------------------------- stack
def horn_slope_deg(dem: np.ndarray, grid: Grid) -> np.ndarray:
    """Slope (degrees) with Horn's 3x3 kernel on a lat/lon grid (metric spacing per row)."""
    z = dem.astype(np.float32)
    dy = EARTH_R * math.radians(grid.res)
    dx = (dy * np.cos(np.radians(grid.lats()))).astype(np.float32)[1:-1, None]
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    s = np.full(z.shape, np.nan, dtype=np.float32)
    s[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return s


class Stack:
    """All proxy layers for one region, on a 1 arc-second grid."""

    def __init__(self, bounds):
        self.grid = g = Grid(bounds)
        dem = load_layer([dem_url(*t) for t in one_deg_tiles(g.bounds)], g, np.float32, np.nan,
                         Resampling.nearest)
        self.dem = dem
        self.slope = horn_slope_deg(dem, g)
        self.hand = load_layer([hand_url(*t) for t in one_deg_tiles(g.bounds)], g, np.float32,
                               np.nan, Resampling.nearest)
        self.wc = load_layer([wc_url(*t) for t in three_deg_tiles(g.bounds)], g, np.uint8, 0,
                             Resampling.nearest, decimate=True)
        self.built, self.height = load_ghsl(g)

    def window(self, lon, lat, size_deg, step=1):
        r, c = self.grid.rowcol(lon, lat)
        h = int(round(size_deg / self.grid.res / 2))
        r0, r1 = max(r - h, 0), min(r + h, self.grid.height)
        c0, c1 = max(c - h, 0), min(c + h, self.grid.width)
        return slice(r0, r1, step), slice(c0, c1, step)

    def stats(self, lon, lat, size_deg, step=1) -> dict:
        sl = self.window(lon, lat, size_deg, step)
        return window_stats(self.dem[sl], self.slope[sl], self.hand[sl], self.wc[sl],
                            self.built[sl], self.height[sl])


PROXIES = ["slope_mean", "slope_p90", "elev_range", "hand_median", "hand_lt5_frac",
           "veg_frac", "tree_frac", "built_frac", "bldg_height"]
EXTRA = ["water_frac", "dem_valid_frac", "wc_valid_frac"]


def window_stats(dem, slope, hand, wc, built, height) -> dict:
    out = {k: np.nan for k in PROXIES + EXTRA}
    n = dem.size
    if n == 0:
        return out
    z = dem[np.isfinite(dem)]
    out["dem_valid_frac"] = z.size / n
    if z.size > 0.5 * n:
        p1, p99 = np.percentile(z, [1, 99])
        out["elev_range"] = float(p99 - p1)
        s = slope[np.isfinite(slope)]
        if s.size:
            out["slope_mean"] = float(s.mean())
            out["slope_p90"] = float(np.percentile(s, 90))
    hh = hand[np.isfinite(hand)]
    if hh.size > 0.5 * n:
        out["hand_median"] = float(np.median(hh))
        out["hand_lt5_frac"] = float((hh < 5).mean())
    w = wc[wc > 0]
    out["wc_valid_frac"] = w.size / n
    if w.size > 0.5 * n:
        out["veg_frac"] = float(np.isin(w, WC_VEG).mean())
        out["tree_frac"] = float((w == WC_TREE).mean())
        out["water_frac"] = float((w == WC_WATER).mean())
    b = np.isfinite(built)
    if b.sum() > 0.5 * n:
        bs = built[b]
        out["built_frac"] = float(bs.mean())
        hb = b & np.isfinite(height) & (built > 0)
        wsum = built[hb].sum()
        # built-surface-weighted mean ANBH; undefined (NaN) with < 0.1 % built-up surface
        if wsum > 0.001 * b.sum():
            out["bldg_height"] = float((height[hb] * built[hb]).sum() / wsum)
    return out
