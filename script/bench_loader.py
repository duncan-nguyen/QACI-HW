"""Đo throughput của dataloader theo số worker, để chọn --num-workers bằng số liệu.

Với GR-ConvNet (1,9M tham số) thì GPU gần như không bao giờ là chỗ nghẽn -- kể cả H200. Cái
quyết định thời gian train là dataloader trên CPU: mỗi sample phải decode JPEG, rotate/zoom
ảnh + part_mask, và rasterize M_union. Script này quét vài giá trị worker rồi in sample/s.

    python script/bench_loader.py --dataset-path data/grasp-anything-pp-full \
        --split-path split/grasp-anything-pp --workers 8,16,32,48
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--split-path", default=None)
    p.add_argument("--dataset", default="grasp-anything-pp")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batches", type=int, default=30, help="Số batch đo mỗi mức worker")
    p.add_argument("--workers", default=None,
                   help="Danh sách cách nhau bởi dấu phẩy (mặc định: suy từ số core)")
    p.add_argument("--seen", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    nproc = os.cpu_count() or 4
    if args.workers:
        levels = [int(w) for w in args.workers.split(",")]
    else:
        levels = sorted({2, 4, 8, nproc // 4, nproc // 2, max(1, nproc - 8)})
        levels = [w for w in levels if 0 < w <= nproc]

    ds_kwargs = dict(output_size=args.input_size, include_depth=False, include_rgb=True,
                     random_rotate=True, random_zoom=True, seen=bool(args.seen))
    if args.split_path:
        ds_kwargs["split_path"] = args.split_path
    dataset = get_dataset(args.dataset)(args.dataset_path, **ds_kwargs)
    print(f"{len(dataset):,} sample, {nproc} core, batch {args.batch_size}\n")

    print(f"{'workers':>8} {'sample/s':>10} {'ms/batch':>10}   {'1 lượt qua dữ liệu':>20}")
    best = (0, 0.0)
    for workers in levels:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers,
            persistent_workers=workers > 0, prefetch_factor=4 if workers else None,
            pin_memory=True)

        t0, n, nb = None, 0, 0
        for batch in loader:
            if t0 is None:      # batch đầu gánh chi phí spawn worker, không tính
                t0 = time.time()
                continue
            n += batch[0].shape[0]
            nb += 1
            if nb >= args.batches:
                break
        dt = time.time() - t0
        rate = n / dt if dt > 0 else 0.0
        hours = len(dataset) / rate / 3600 if rate else float("inf")
        print(f"{workers:>8} {rate:>10.0f} {dt / max(nb, 1) * 1000:>10.0f}   {hours:>17.2f} giờ")
        if rate > best[1]:
            best = (workers, rate)
        del loader

    print(f"\nTốt nhất: --num-workers {best[0]}  ({best[1]:.0f} sample/s)")
    print("Nếu sample/s vẫn tăng ở mức worker cao nhất thì thử thêm mức cao hơn nữa.")


if __name__ == "__main__":
    main()
