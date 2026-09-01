"""Dựng một subset Grasp-Anything++ đủ dùng để train, không cần tải 150 GB.

Cách làm:

1. Tải 4 zip nhỏ (~10.6 GB): 3 thư mục label của GA++ + `scene_description` của base repo.
2. Chọn ngẫu nhiên-nhưng-xác-định một tập scene bằng hash tên scene. Vì hash chỉ phụ thuộc
   tên, cả 4 zip chọn ra **cùng** tập scene mà không cần so danh sách.
3. Trích ra *mọi* sample của những scene đó. Lấy trọn scene (chứ không phải sample rời) là
   bắt buộc: `M_∪` cần mọi part của cùng một object, lấy sample rời thì union suy biến thành
   chính part đó.
4. Ảnh nằm trong archive 65 GB nên không tải; đọc thẳng từng file cần qua HTTP range.

Kết quả có layout đúng như `GraspAnythingPPDataset` mong đợi, chạy tiếp
`split/build_grasp_anything_pp.py` là train được.

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

# zip cần tải hẳn về (nhỏ) -> (repo, tên file, thư mục bên trong, đuôi)
LOCAL_ZIPS = [
    (REPO_PP, "grasp_instructions.zip", "grasp_instructions", ".pkl"),
    (REPO_PP, "grasp_label_positive.zip", "grasp_label_positive", ".pt"),
    (REPO_PP, "part_mask.zip", "part_mask", ".npy"),
    (REPO_BASE, "scene_description.zip", "scene_description", ".pkl"),
]

TOTAL_SCENES = 994_860  # dùng để ước lượng khi chưa quét xong


# ------------------------------------------------------------------ zip đọc --
def parse_eocd(read_at, size):
    """(n_entries, cd_size, cd_offset). Mọi archive ở đây đều ZIP64."""
    tail = read_at(max(0, size - 65536), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("không thấy End Of Central Directory")
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
    """Yield (name, offset, csize, method) cho từng entry, đọc theo chunk để khỏi ngốn RAM."""
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
    """Như local_reader nhưng cho archive bị chẻ nhiều part (image_part_aa + ab), map offset
    logic sang (file, offset thật) mà không phải nối 65 GB lại."""
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
    """Đọc range trên HuggingFace. Giữ session và URL CDN đã resolve -- mỗi request mới mà
    phải đi lại redirect thì chậm gấp đôi."""

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
        """Range trên archive logic (các part nối lại)."""
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


# ------------------------------------------------------------------ tải zip --
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


# ------------------------------------------------------------------- chọn ----
def pack_mask(payload):
    """part_mask nhị phân 416x416 uint8 -> mảng bit đóng gói (173 KB -> 21 KB)."""
    import io

    import numpy as np

    mask = np.load(io.BytesIO(payload))
    buf = io.BytesIO()
    np.save(buf, np.packbits(mask.astype(bool).ravel()))
    return buf.getvalue()


def write_atomic(dest, payload):
    """
    Ghi qua file tạm rồi `os.replace`.

    Ghi thẳng vào đích thì một lần đứt mạng/Ctrl-C giữa chừng để lại file cụt, mà lần chạy
    sau chỉ kiểm tra `os.path.isfile` nên sẽ bỏ qua nó -- JPEG hỏng đi thẳng vào training và
    không báo lỗi ở đâu cả. `os.replace` là atomic trên cùng một filesystem.
    """
    # Đuôi khác `.part` của download() để hai cơ chế không bao giờ giẫm lên nhau.
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, dest)


def keeps_scene(scene, every):
    """Chọn xác định theo hash: cùng `every` thì mọi archive chọn ra cùng tập scene."""
    return int(hashlib.md5(scene.encode()).hexdigest()[:8], 16) % every == 0


def scene_of(name):
    return os.path.splitext(os.path.basename(name))[0].rsplit("_", 2)[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/ga-pp-subset", help="Thư mục subset")
    p.add_argument("--zips-dir", default=None,
                   help="Nơi chứa/tải các zip (mặc định <out>/_archives)")
    p.add_argument("--scenes", type=int, default=2000, help="Số scene muốn giữ (xấp xỉ)")
    p.add_argument("--workers", type=int, default=16, help="Số luồng tải ảnh")
    p.add_argument("--skip-images", action="store_true",
                   help="Bỏ bước tải ảnh (khi ảnh đã có sẵn)")
    p.add_argument("--images-from-zip", action="store_true",
                   help="Tải hẳn image_part_aa+ab (65 GB) rồi trích ảnh từ đĩa, thay vì đọc "
                        "từng ảnh qua HTTP. Đáng dùng khi > ~40k scene.")
    p.add_argument("--pack-masks", action="store_true",
                   help="Lưu part_mask dưới dạng bit đóng gói: 21 KB thay vì 173 KB mỗi file "
                        "(nhỏ hơn 8 lần, loader tự nhận ra). Bật khi subset lớn.")
    p.add_argument("--keep-zips", action="store_true", help="Giữ lại zip sau khi trích")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out
    zips_dir = args.zips_dir or os.path.join(out, "_archives")
    os.makedirs(zips_dir, exist_ok=True)

    print("== 1/4  tải label zip (~10.6 GB, chỉ một lần) ==")
    for repo, name, _, _ in LOCAL_ZIPS:
        download(repo, name, os.path.join(zips_dir, name))

    print("\n== 2/4  trích subset ==")
    every = max(1, round(TOTAL_SCENES / max(1, args.scenes)))
    print(f"  giữ 1 trong mỗi {every} scene (mục tiêu ~{args.scenes:,} scene)")

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

    print(f"\n  {len(scenes):,} scene được chọn")
    if not scenes:
        raise SystemExit("Không chọn được scene nào -- thử --scenes lớn hơn.")

    if args.skip_images:
        print("\n== 3/4  bỏ qua ảnh (--skip-images) ==")
    elif args.images_from_zip:
        print(f"\n== 3/4  tải archive ảnh (65 GB) rồi trích {len(scenes):,} ảnh ==")
        fetch_images_from_zip(scenes, os.path.join(out, "image"), zips_dir)
    else:
        print(f"\n== 3/4  tải {len(scenes):,} ảnh qua HTTP range (không tải 65 GB) ==")
        fetch_images(scenes, os.path.join(out, "image"), args.workers)

    print("\n== 4/4  xong ==")
    for folder in sorted(os.listdir(out)):
        path = os.path.join(out, folder)
        if os.path.isdir(path) and not folder.startswith("_"):
            print(f"  {folder:22} {len(os.listdir(path)):>8,} file")
    if not args.keep_zips:
        shutil.rmtree(zips_dir, ignore_errors=True)
        print(f"  (đã xoá {zips_dir}; --keep-zips để giữ lại)")
    print(f"\nTiếp: python split/build_grasp_anything_pp.py --data-dir {out}")


def fetch_images_from_zip(scenes, target, zips_dir):
    """Tải image_part_aa + ab về rồi trích các ảnh cần. Với subset lớn thì rẻ hơn hẳn đọc
    từng ảnh qua HTTP: 65 GB tải một lần là chuyện băng thông, còn ~550 ms/ảnh là chuyện
    latency và không co lại theo băng thông."""
    os.makedirs(target, exist_ok=True)
    parts = []
    for name in ("image_part_aa", "image_part_ab"):
        dest = os.path.join(zips_dir, name)
        download(REPO_BASE, name, dest)
        parts.append(dest)

    read_at, size = local_reader_multi(parts)
    need = {s for s in scenes if not os.path.isfile(os.path.join(target, s + ".jpg"))}
    print(f"  quét central directory ({size / 1e9:.0f} GB)...")
    done = 0
    for name, off, csize, method in iter_central_dir(read_at, size):
        scene = os.path.splitext(os.path.basename(name))[0]
        if scene not in need:
            continue
        write_atomic(os.path.join(target, scene + ".jpg"),
                     read_member(read_at, off, csize, method))
        done += 1
        if done % 500 == 0:
            print(f"\r  {done:,}/{len(need):,} ảnh", end="")
    print(f"\r  {done:,}/{len(need):,} ảnh")


def fetch_images(scenes, target, workers):
    os.makedirs(target, exist_ok=True)
    need = {s for s in scenes if not os.path.isfile(os.path.join(target, s + ".jpg"))}
    if not need:
        print("  ảnh đã có đủ")
        return

    reader = RemoteReader(REPO_BASE, "image_part_aa", "image_part_ab")
    print(f"  quét central directory của image.zip ({reader.size / 1e9:.0f} GB archive)...")
    entries = {}
    for name, off, csize, method in iter_central_dir(reader.read_at, reader.size):
        scene = os.path.splitext(os.path.basename(name))[0]
        if scene in need:
            entries[scene] = (off, csize, method)
    print(f"  khớp {len(entries):,}/{len(need):,} scene")

    # Mỗi luồng một RemoteReader riêng: requests.Session không an toàn khi dùng chung.
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
                print(f"\r  {done:,}/{len(entries):,} ảnh", end="")
    print(f"\r  {done:,}/{len(entries):,} ảnh")


if __name__ == "__main__":
    main()
