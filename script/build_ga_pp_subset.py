"""Build a Grasp-Anything++ subset large enough to train on, without downloading 150 GB.

How it works:

1. Download 4 small zips (~10.6 GB): GA++'s 3 label directories + `scene_description` from the
   base repo.
2. Pick a random-but-deterministic set of scenes by hashing the scene name. Because the hash
   depends only on the name, all 4 zips select the **same** scenes without comparing lists.
3. Extract *every* sample of those scenes. Taking whole scenes (rather than individual samples)
   is required: `M_union` needs every part of the same object, and with loose samples the union
   degenerates into that single part.
4. The images live in a 65 GB archive, so they are not downloaded; each needed file is read
   directly over an HTTP range request.

The result has exactly the layout `GraspAnythingPPDataset` expects; run
`split/build_grasp_anything_pp.py` next and it is ready to train.

    python script/build_ga_pp_subset.py --out data/ga-pp-subset --scenes 2000
"""
import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
REPO_BASE = "airvlab/Grasp-Anything"
REPO_PP = "airvlab/Grasp-Anything-pp"

# Zips small enough to download in full -> (repo, filename, inner directory, extension)
LOCAL_ZIPS = [
    (REPO_PP, "grasp_instructions.zip", "grasp_instructions", ".pkl"),
    (REPO_PP, "grasp_label_positive.zip", "grasp_label_positive", ".pt"),
    (REPO_PP, "part_mask.zip", "part_mask", ".npy"),
    (REPO_BASE, "scene_description.zip", "scene_description", ".pkl"),
]

TOTAL_SCENES = 994_860  # used to estimate progress before the scan completes


# --------------------------------------------------------------- zip reading --
def parse_eocd(read_at, size):
    """(n_entries, cd_size, cd_offset). Every archive here is ZIP64."""
    tail = read_at(max(0, size - 65536), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("End Of Central Directory not found")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10 : i + 20])
    j = tail.rfind(b"PK\x06\x07")
    if j >= 0:
        z64 = struct.unpack("<Q", tail[j + 8 : j + 16])[0]
        z = read_at(z64, z64 + 55)
        if z[:4] == b"PK\x06\x06":
            n, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])
    return n, cd_size, cd_off


def zip64_fix(extra, csize, usize, lho):
    if 0xFFFFFFFF not in (csize, usize, lho):
        return csize, usize, lho
    q = 0
    while q + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[q : q + 4])
        if hid == 0x0001:
            f, k = extra[q + 4 : q + 4 + hsz], 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", f[k : k + 8])[0]; k += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", f[k : k + 8])[0]; k += 8
            if lho == 0xFFFFFFFF:
                lho = struct.unpack("<Q", f[k : k + 8])[0]
            break
        q += 4 + hsz
    return csize, usize, lho


def iter_central_dir(read_at, size, chunk=16 << 20):
    """Yield (name, offset, csize, method) per entry, read in chunks to keep RAM down."""
    n_entries, cd_size, cd_off = parse_eocd(read_at, size)
    end, pos, buf = cd_off + cd_size, cd_off, b""
    while pos < end:
        stop = min(pos + chunk, end)
        buf += read_at(pos, stop - 1)
        pos = stop
        p = 0
        while len(buf) - p >= 46 and buf[p : p + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", buf[p + 10 : p + 12])[0]
            csize, usize = struct.unpack("<II", buf[p + 20 : p + 28])
            nlen, elen, clen = struct.unpack("<HHH", buf[p + 28 : p + 34])
            if len(buf) - p < 46 + nlen + elen + clen:
                break
            lho = struct.unpack("<I", buf[p + 42 : p + 46])[0]
            name = buf[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
            extra = buf[p + 46 + nlen : p + 46 + nlen + elen]
            csize, usize, lho = zip64_fix(extra, csize, usize, lho)
            p += 46 + nlen + elen + clen
            if not name.endswith("/"):
                yield name, lho, csize, method
        buf = buf[p:]


def read_member(read_at, offset, csize, method):
    if csize == 0:
        return b""
    head = read_at(offset, offset + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = offset + 30 + nlen + elen
    raw = read_at(start, start + csize - 1)
    return zlib.decompressobj(-15).decompress(raw) if method == 8 else raw


def local_reader_multi(paths):
    """Like local_reader but for an archive split into several parts (image_part_aa + ab),
    mapping a logical offset onto (file, real offset) without concatenating 65 GB."""
    handles = [open(p, "rb") for p in paths]
    sizes = [os.path.getsize(p) for p in paths]
    lock = threading.Lock()

    def read_at(a, b):
        out, base = b"", 0
        for fh, n in zip(handles, sizes):
            lo, hi = base, base + n - 1
            if a <= hi and b >= lo:
                with lock:
                    fh.seek(max(a, lo) - base)
                    out += fh.read(min(b, hi) - max(a, lo) + 1)
            base += n
        return out

    return read_at, sum(sizes)


def local_reader(path):
    lock = threading.Lock()
    fh = open(path, "rb")

    def read_at(a, b):
        with lock:
            fh.seek(a)
            return fh.read(b - a + 1)

    return read_at, os.path.getsize(path)


class RemoteReader:
    """Range reads against HuggingFace. Keeps the session and the resolved CDN URL -- a request
    that has to follow the redirect again is twice as slow."""

    def __init__(self, repo, *names):
        self.session = requests.Session()
        self.urls, self.sizes = [], []
        for name in names:
            r = self.session.head(HF.format(repo=repo, name=name), allow_redirects=True,
                                  timeout=60)
            r.raise_for_status()
            self.urls.append(r.url)
            self.sizes.append(int(r.headers["Content-Length"]))
        self.size = sum(self.sizes)

    def read_at(self, a, b):
        """A range on the logical archive (the parts joined end to end)."""
        out, base = b"", 0
        for url, n in zip(self.urls, self.sizes):
            lo, hi = base, base + n - 1
            if a <= hi and b >= lo:
                r = self.session.get(url, timeout=300, headers={
                    "Range": f"bytes={max(a, lo) - base}-{min(b, hi) - base}"})
                r.raise_for_status()
                out += r.content
            base += n
        return out


# ------------------------------------------------------------------ download --
def download(repo, name, dest):
    if os.path.isfile(dest):
        print(f"  [skip ] {name} ({os.path.getsize(dest) / 1e9:.2f} GB)")
        return
    url = HF.format(repo=repo, name=name)
    for cli in ("hf", "huggingface-cli"):
        if shutil.which(cli):
            print(f"  [{cli}] {name}")
            cmd = [cli, "download", repo, name, "--repo-type", "dataset",
                   "--local-dir", os.path.dirname(dest)]
            subprocess.run(cmd, check=True)
            return
    print(f"  [get  ] {name}")
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=300) as r:
        r.raise_for_status()
        total = pos + int(r.headers.get("Content-Length", 0))
        with open(tmp, "ab") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                pos += len(block)
                print(f"\r         {pos / 1e9:5.2f} / {total / 1e9:5.2f} GB", end="")
    print()
    os.rename(tmp, dest)


# -------------------------------------------------------------- selection ----
def pack_mask(payload):
    """A binary 416x416 uint8 part_mask -> a bit-packed array (173 KB -> 21 KB)."""
    import io

    import numpy as np

    mask = np.load(io.BytesIO(payload))
    buf = io.BytesIO()
    np.save(buf, np.packbits(mask.astype(bool).ravel()))
    return buf.getvalue()


def write_atomic(dest, payload):
    """
    Write through a temp file, then `os.replace`.

    Writing straight to the destination means one dropped connection or Ctrl-C leaves a
    truncated file behind, and the next run only checks `os.path.isfile` and skips it -- a
    corrupt JPEG goes straight into training with no error anywhere. `os.replace` is atomic
    within one filesystem.
    """
    # A different suffix from download()'s `.part`, so the two mechanisms never collide.
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, dest)


def keeps_scene(scene, every):
    """Deterministic hash selection: with the same `every`, every archive picks the same scenes."""
    return int(hashlib.md5(scene.encode()).hexdigest()[:8], 16) % every == 0


def scene_of(name):
    return os.path.splitext(os.path.basename(name))[0].rsplit("_", 2)[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/ga-pp-subset", help="Subset directory")
    p.add_argument("--zips-dir", default=None,
                   help="Where the zips are stored/downloaded (default <out>/_archives)")
    p.add_argument("--scenes", type=int, default=2000, help="Approximate number of scenes to keep")
    p.add_argument("--workers", type=int, default=16, help="Threads used to fetch images")
    p.add_argument("--skip-images", action="store_true",
                   help="Skip fetching images (when they are already present)")
    p.add_argument("--images-from-zip", action="store_true",
                   help="Download image_part_aa+ab in full (65 GB) and extract images from "
                        "disk, instead of reading each image over HTTP. Worth it above ~40k "
                        "scenes.")
    p.add_argument("--pack-masks", action="store_true",
                   help="Store part_mask bit-packed: 21 KB instead of 173 KB per file (8x "
                        "smaller, the loader detects it automatically). Use for large subsets.")
    p.add_argument("--keep-zips", action="store_true", help="Keep the zips after extraction")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out
    zips_dir = args.zips_dir or os.path.join(out, "_archives")
    os.makedirs(zips_dir, exist_ok=True)

    print("== 1/4  downloading label zips (~10.6 GB, once) ==")
    for repo, name, _, _ in LOCAL_ZIPS:
        download(repo, name, os.path.join(zips_dir, name))

    print("\n== 2/4  extracting the subset ==")
    every = max(1, round(TOTAL_SCENES / max(1, args.scenes)))
    print(f"  keeping 1 scene in every {every} (target ~{args.scenes:,} scenes)")

    scenes = set()
    for repo, name, folder, ext in LOCAL_ZIPS:
        path = os.path.join(zips_dir, name)
        read_at, size = local_reader(path)
        target = os.path.join(out, folder)
        os.makedirs(target, exist_ok=True)

        n_kept = 0
        for entry_name, off, csize, method in iter_central_dir(read_at, size):
            if not entry_name.endswith(ext):
                continue
            scene = scene_of(entry_name)
            if not keeps_scene(scene, every):
                continue
            dest = os.path.join(target, os.path.basename(entry_name))
            if not os.path.isfile(dest):
                payload = read_member(read_at, off, csize, method)
                if args.pack_masks and folder == "part_mask":
                    payload = pack_mask(payload)
                write_atomic(dest, payload)
            scenes.add(scene)
            n_kept += 1
            if n_kept % 2000 == 0:
                print(f"\r  {folder:22} {n_kept:>8,} file", end="")
        print(f"\r  {folder:22} {n_kept:>8,} file")

    print(f"\n  {len(scenes):,} scenes selected")
    if not scenes:
        raise SystemExit("No scene was selected -- try a larger --scenes.")

    if args.skip_images:
        print("\n== 3/4  skipping images (--skip-images) ==")
    elif args.images_from_zip:
        print(f"\n== 3/4  downloading the image archive (65 GB), extracting {len(scenes):,} images ==")
        fetch_images_from_zip(scenes, os.path.join(out, "image"), zips_dir)
    else:
        print(f"\n== 3/4  fetching {len(scenes):,} images over HTTP range (no 65 GB download) ==")
        fetch_images(scenes, os.path.join(out, "image"), args.workers)

    print("\n== 4/4  done ==")
    for folder in sorted(os.listdir(out)):
        path = os.path.join(out, folder)
        if os.path.isdir(path) and not folder.startswith("_"):
            print(f"  {folder:22} {len(os.listdir(path)):>8,} file")
    if not args.keep_zips:
        shutil.rmtree(zips_dir, ignore_errors=True)
        print(f"  (removed {zips_dir}; use --keep-zips to keep it)")
    print(f"\nNext: python split/build_grasp_anything_pp.py --data-dir {out}")


def fetch_images_from_zip(scenes, target, zips_dir):
    """Download image_part_aa + ab, then extract the needed images. For large subsets this is
    far cheaper than reading each image over HTTP: 65 GB downloaded once is a bandwidth
    question, whereas ~550 ms per image is latency and does not shrink with bandwidth."""
    os.makedirs(target, exist_ok=True)
    parts = []
    for name in ("image_part_aa", "image_part_ab"):
        dest = os.path.join(zips_dir, name)
        download(REPO_BASE, name, dest)
        parts.append(dest)

    read_at, size = local_reader_multi(parts)
    need = {s for s in scenes if not os.path.isfile(os.path.join(target, s + ".jpg"))}
    print(f"  scanning the central directory ({size / 1e9:.0f} GB)...")
    done = 0
    for name, off, csize, method in iter_central_dir(read_at, size):
        scene = os.path.splitext(os.path.basename(name))[0]
        if scene not in need:
            continue
        write_atomic(os.path.join(target, scene + ".jpg"),
                     read_member(read_at, off, csize, method))
        done += 1
        if done % 500 == 0:
            print(f"\r  {done:,}/{len(need):,} images", end="")
    print(f"\r  {done:,}/{len(need):,} images")


def fetch_images(scenes, target, workers):
    os.makedirs(target, exist_ok=True)
    need = {s for s in scenes if not os.path.isfile(os.path.join(target, s + ".jpg"))}
    if not need:
        print("  all images already present")
        return

    reader = RemoteReader(REPO_BASE, "image_part_aa", "image_part_ab")
    print(f"  scanning image.zip's central directory ({reader.size / 1e9:.0f} GB archive)...")
    entries = {}
    for name, off, csize, method in iter_central_dir(reader.read_at, reader.size):
        scene = os.path.splitext(os.path.basename(name))[0]
        if scene in need:
            entries[scene] = (off, csize, method)
    print(f"  matched {len(entries):,}/{len(need):,} scenes")

    # One RemoteReader per thread: requests.Session is not safe to share.
    local = threading.local()

    def grab(item):
        scene, (off, csize, method) = item
        if not hasattr(local, "reader"):
            local.reader = RemoteReader(REPO_BASE, "image_part_aa", "image_part_ab")
        data = read_member(local.reader.read_at, off, csize, method)
        write_atomic(os.path.join(target, scene + ".jpg"), data)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(grab, entries.items()):
            done += 1
            if done % 50 == 0:
                print(f"\r  {done:,}/{len(entries):,} images", end="")
    print(f"\r  {done:,}/{len(entries):,} images")


if __name__ == "__main__":
    main()
