import argparse
import hashlib
import inspect
import logging
import time

import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_extras(net, batch, device):
    """Phần tử thứ 6 (prompt/part_mask/union_pos) của Grasp-Anything++; xem train_network.py."""
    if len(batch) < 6 or not getattr(net, "accepts_extras", False):
        return {}
    extra = batch[5]
    kwargs = {}
    if "prompt" in extra:
        kwargs["prompts"] = extra["prompt"]
    for key in ("part_mask", "union_pos", "align_weight"):
        if key in extra:
            kwargs[key] = extra[key].to(device)
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate networks")

    # Network
    parser.add_argument(
        "--network",
        metavar="N",
        type=str,
        nargs="+",
        help="Path to saved networks to evaluate",
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help="Input image size for the network"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, help='Dataset Name ("cornell" or "jaquard")'
    )
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Thư mục chứa seen.obj/unseen.obj (mặc định dò theo SPLIT_DIRS của loader)",
    )
    parser.add_argument(
        "--use-depth", type=int, default=1, help="Use Depth image for evaluation (1/0)"
    )
    parser.add_argument(
        "--use-rgb", type=int, default=1, help="Use RGB image for evaluation (1/0)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether data augmentation should be applied",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.01,
        help="Fraction of data for training (remainder is validation)",
    )
    parser.add_argument(
        "--ds-shuffle", action="store_true", default=False, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--ds-rotate",
        type=float,
        default=0.0,
        help="Shift the start point of the dataset to use a different test/train split",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Dataset workers")

    # Evaluation
    parser.add_argument(
        "--n-grasps", type=int, default=1, help="Number of grasps to consider per image"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.25, help="Threshold for IOU matching"
    )
    parser.add_argument(
        "--iou-eval", action="store_true", help="Compute success based on IoU metric."
    )
    parser.add_argument(
        "--align-eval",
        action="store_true",
        help="Đo luôn chất lượng A_T (IoU/Dice với part_mask, fg vs bg, token thắng có phải "
        "từ chỉ part). Cần nạp part_mask nên chậm hơn; không bật thì loader bỏ qua "
        "part_mask/M_union và evaluation nhanh hơn đáng kể.",
    )
    parser.add_argument(
        "--jacquard-output", action="store_true", help="Jacquard-dataset style output"
    )

    # Misc.
    parser.add_argument(
        "--vis", action="store_true", help="Visualise the network output"
    )
    parser.add_argument(
        "--cpu",
        dest="force_cpu",
        action="store_true",
        default=False,
        help="Force code to run in CPU mode",
    )
    parser.add_argument(
        "--random-seed", type=int, default=123, help="Random seed for numpy"
    )
    parser.add_argument(
        "--seen",
        type=int,
        default=1,
        help="Flag for using seen classes, only work for Grasp-Anything dataset",
    )

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != "jacquard":
        raise ValueError(
            "--jacquard-output can only be used with the --dataset jacquard option."
        )
    if args.jacquard_output and args.augment:
        raise ValueError("--jacquard-output can not be used with data augmentation.")

    return args


if __name__ == "__main__":
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info(f"Loading {args.dataset.title()} Dataset...")
    Dataset = get_dataset(args.dataset)
    ds_kwargs = dict(
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=args.augment,
        random_zoom=args.augment,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
        seen=args.seen,
    )
    # Chỉ truyền khi có, vì các loader khác (cornell/jacquard/...) không nhận kwarg này.
    if args.split_path:
        ds_kwargs["split_path"] = args.split_path
    # Metric IoU chỉ cần pos/cos/sin/width. Nạp part_mask và dựng M_union (đọc N file .pt cho
    # mỗi sample) là phần lớn thời gian mỗi sample -- bỏ đi khi không đo alignment.
    if not args.align_eval:
        params = inspect.signature(Dataset.__init__).parameters
        for key in ("include_mask", "include_union"):
            if key in params:
                ds_kwargs[key] = False
    test_dataset = Dataset(args.dataset_path, **ds_kwargs)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info(f"Validation size: {len(val_indices)}")
    logging.info(
        "Eval split: seen=%s split_frac=%s shuffle=%s augment=%s indices_sha256=%s",
        args.seen,
        args.split,
        args.ds_shuffle,
        args.augment,
        hashlib.sha256(",".join(map(str, val_indices)).encode()).hexdigest()[:16],
    )

    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.num_workers, sampler=val_sampler
    )
    logging.info("Done")

    for network in args.network:
        logging.info(f"\nEvaluating model {network}")

        # Load Network
        net = torch.load(network, map_location=device, weights_only=False)
        # Bản cũ tin rằng checkpoint được lưu sau validate() nên đã ở eval mode. Một checkpoint
        # lưu giữa lúc train sẽ được đánh giá với dropout bật mà không có cảnh báo nào.
        was_training = net.training
        net.eval()
        logging.info(
            "checkpoint training_mode_on_load=%s -> eval(); sha256=%s",
            was_training,
            sha256_file(network),
        )
        if args.align_eval and hasattr(net, "set_collect_stats"):
            net.set_collect_stats(True)

        results = {"correct": 0, "failed": 0}
        align_stats, align_n = {}, 0
        token_part, token_punct, token_n, token_batches = 0.0, 0.0, 0, 0

        if args.jacquard_output:
            jo_fn = network + "_jacquard_output.txt"
            with open(jo_fn, "w") as f:
                pass

        start_time = time.time()

        with torch.no_grad():
            for idx, batch in enumerate(test_data):
                x, y, didx, rot, zoom = batch[:5]
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                extras = eval_extras(net, batch, device)
                lossd = net.compute_loss(xc, yc, **extras)
                pred = lossd["pred"]

                if args.align_eval:
                    for sn, sv in lossd.get("stats", {}).items():
                        align_stats[sn] = align_stats.get(sn, 0.0) + float(sv)
                    align_n += 1
                    # `accepts_extras` vẫn True khi use_text=0 (baseline no-text) nên prompts
                    # vẫn có mặt, nhưng nhánh alignment thì không tồn tại.
                    if (
                        "prompts" in extras
                        and getattr(net, "use_text", False)
                        and hasattr(net, "align_token_report")
                    ):
                        rep = net.align_token_report(xc, extras["prompts"])
                        if rep["n"]:
                            token_n += rep["n"]
                            token_part += rep["part_frac"] * rep["n"]
                        token_punct += rep["punct_frac"]
                        token_batches += 1

                q_img, ang_img, width_img = post_process_output(
                    pred["pos"],
                    pred["cos"],
                    pred["sin"],
                    pred["width"],
                )

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(
                        q_img,
                        ang_img,
                        test_data.dataset.get_gtbb(didx, rot, zoom),
                        no_grasps=args.n_grasps,
                        grasp_width=width_img,
                        threshold=args.iou_threshold,
                    )
                    if s:
                        results["correct"] += 1
                    else:
                        results["failed"] += 1

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(
                        q_img, ang_img, width_img=width_img, no_grasps=1
                    )
                    with open(jo_fn, "a") as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + "\n")
                            f.write(g.to_jacquard(scale=1024 / 300) + "\n")

                if args.vis:
                    save_results(
                        rgb_img=test_data.dataset.get_rgb(
                            didx, rot, zoom, normalise=False
                        ),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=args.n_grasps,
                        grasp_width_img=width_img,
                    )

        avg_time = (time.time() - start_time) / len(test_data)
        logging.info(f"Average evaluation time per image: {avg_time * 1000}ms")

        if args.iou_eval:
            logging.info(
                "IOU Results: %d/%d = %f"
                % (
                    results["correct"],
                    results["correct"] + results["failed"],
                    results["correct"] / (results["correct"] + results["failed"]),
                )
            )

        if align_n:
            # Chất lượng A_T -- loss không nói lên nó có hoạt động hay không. fg ≈ bg nghĩa là
            # gate vô dụng; argmax_last cao nghĩa là token thắng luôn là token cuối câu.
            logging.info(
                "ALIGN Results: %s",
                " ".join(
                    f"{n}={v / align_n:.4f}" for n, v in sorted(align_stats.items())
                ),
            )
        if token_batches:
            logging.info(
                "TOKEN Results: argmax_is_part=%.4f argmax_is_punct=%.4f n=%d",
                token_part / token_n if token_n else 0.0,
                token_punct / token_batches,
                token_n,
            )

        if args.jacquard_output:
            logging.info(f"Jacquard output saved to {jo_fn}")

        del net
        torch.cuda.empty_cache()
