"""Stream urban_sar_floods.tar.gz from Hugging Face and keep only metadata.

The archive is a single 38 GB gzip stream, so it cannot be read at random. This script
downloads it with parallel HTTP range requests, decompresses it in order and walks the tar
members, keeping:

* the small text members (Train_dataset.txt, Valid_dataset.txt, data_norm.txt) in full;
* for every GeoTIFF chip, the georeferencing parsed from the first 64 KiB of the file
  (width, height, pixel scale, tie point), written to a JSON-lines file.

Nothing else is written to disk. Usage:

    python stream_usf_headers.py OUT_DIR
"""

from __future__ import annotations

import json
import queue
import struct
import sys
import tarfile
import threading
import time
import zlib
from pathlib import Path

import requests

URL = (
    "https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/"
    "urban_sar_floods.tar.gz"
)
CHUNK = 32 * 1024 * 1024
WORKERS = 8
HEADER_BYTES = 65536


def total_size() -> int:
    r = requests.head(URL, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch(start: int, end: int) -> bytes:
    for attempt in range(8):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** min(attempt, 5))
    raise RuntimeError(f"range {start}-{end} failed")


def chunk_stream(size: int):
    """Yield the archive bytes in order while WORKERS ranges download ahead."""
    starts = list(range(0, size, CHUNK))
    results: dict[int, bytes] = {}
    cond = threading.Condition()
    todo: queue.Queue[int] = queue.Queue()
    for i in range(len(starts)):
        todo.put(i)
    next_needed = [0]

    def worker():
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            with cond:  # do not run more than 3*WORKERS chunks ahead (memory bound)
                cond.wait_for(lambda i=i: i < next_needed[0] + 3 * WORKERS)
            s = starts[i]
            data = fetch(s, min(s + CHUNK, size) - 1)
            with cond:
                results[i] = data
                cond.notify_all()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for i in range(len(starts)):
        with cond:
            cond.wait_for(lambda i=i: i in results)
            data = results.pop(i)
            next_needed[0] = i + 1
            cond.notify_all()
        yield data


def parse_tiff_header(buf: bytes) -> dict | None:
    """Return width, height, pixel scale and tie point from a little/big-endian TIFF."""
    if len(buf) < 8:
        return None
    bo = {b"II": "<", b"MM": ">"}.get(buf[:2])
    if bo is None or struct.unpack(bo + "H", buf[2:4])[0] != 42:
        return None
    ifd = struct.unpack(bo + "I", buf[4:8])[0]
    if ifd + 2 > len(buf):
        return None
    n = struct.unpack(bo + "H", buf[ifd : ifd + 2])[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}
    fmt = {1: "B", 2: "c", 3: "H", 4: "I", 11: "f", 12: "d", 16: "Q"}
    out: dict = {}
    for k in range(n):
        e = ifd + 2 + 12 * k
        if e + 12 > len(buf):
            return None
        tag, typ, cnt = struct.unpack(bo + "HHI", buf[e : e + 8])
        if tag not in (256, 257, 277, 33550, 33922, 34264):
            continue
        nbytes = sizes.get(typ, 1) * cnt
        if nbytes <= 4:
            raw = buf[e + 8 : e + 8 + nbytes]
        else:
            off = struct.unpack(bo + "I", buf[e + 8 : e + 12])[0]
            if off + nbytes > len(buf):
                return None
            raw = buf[off : off + nbytes]
        vals = struct.unpack(bo + fmt[typ] * cnt, raw)
        out[{256: "width", 257: "height", 277: "bands", 33550: "scale",
             33922: "tiepoint", 34264: "transform"}[tag]] = list(vals)
    return out


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = total_size()
    log = open(out / "progress.log", "a", buffering=1)
    meta = open(out / "chips.jsonl", "w", buffering=1)
    log.write(f"archive size {size}\n")

    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = bytearray()
    state = {"need": 512, "mode": "header", "member": None, "remaining": 0, "pad": 0,
             "head": bytearray()}
    n_members = 0
    t0 = time.time()
    consumed = 0
    for raw in chunk_stream(size):
        consumed += len(raw)
        buf += dec.decompress(raw)
        while True:
            if state["mode"] == "header":
                if len(buf) < 512:
                    break
                h = bytes(buf[:512])
                del buf[:512]
                if h == b"\0" * 512:
                    continue
                ti = tarfile.TarInfo.frombuf(h, "utf-8", "surrogateescape")
                n_members += 1
                state.update(member=ti, remaining=ti.size, pad=(-ti.size) % 512,
                             head=bytearray(), mode="data")
            else:
                ti = state["member"]
                take = min(len(buf), state["remaining"])
                keep_text = ti.name.endswith(".txt")
                limit = ti.size if keep_text else HEADER_BYTES
                if len(state["head"]) < limit:
                    state["head"] += buf[: min(take, limit - len(state["head"]))]
                del buf[:take]
                state["remaining"] -= take
                if state["remaining"] > 0:
                    break
                if len(buf) < state["pad"]:  # wait for the padding bytes
                    break
                del buf[: state["pad"]]
                state["pad"] = 0
                if keep_text:
                    name = ti.name.replace("/", "__")
                    (out / name).write_bytes(bytes(state["head"]))
                    log.write(f"saved {ti.name}\n")
                elif ti.name.lower().endswith((".tif", ".tiff")):
                    rec = {"name": ti.name, "size": ti.size}
                    rec.update(parse_tiff_header(bytes(state["head"])) or {"parse": "failed"})
                    meta.write(json.dumps(rec) + "\n")
                state["mode"] = "header"
        if n_members and (consumed // CHUNK) % 16 == 0:
            el = time.time() - t0
            log.write(f"{consumed/1e9:.2f}/{size/1e9:.2f} GB  members={n_members}  "
                      f"{consumed/el/1e6:.1f} MB/s\n")
    log.write(f"done members={n_members} in {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main(sys.argv[1])
