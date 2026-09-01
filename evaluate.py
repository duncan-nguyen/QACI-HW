import argparse
import logging
import time

import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.data.index_split import SUBSETS, describe, index_splits, select_subset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def eval_extras(net, batch, device):
    """Phần tử thứ 6 (prompt/part_mask/union_pos) của Grasp-Anything++; xem train_network.py."""
    if len(batch) < 6 or not getattr(net, "accepts_extras", False):
        return {}
    extra = batch[5]
    kwargs = {}
    if "prompt" in extra:
        kwargs["prompts"] = extra["prompt"]
    for key in ("part_mask", "union_pos"):
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
        help="Fraction of the dev set used for training; must match the value passed to "
        "train_network.py so the subsets line up",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Phải khớp --test-split của train_network.py để tái lập đúng lát cắt.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="val",
        choices=list(SUBSETS),
        help="Lát cắt đem đánh giá. 'val' = hành vi cũ (mọi index sau --split), nhưng đó là "
        "tập đã dùng chọn checkpoint nên số ra sẽ lạc quan. Dùng 'test' cho tập độc lập "
        "(cần train với --test-split > 0), 'all' cho toàn bộ split seen/unseen.",
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
    test_dataset = Dataset(args.dataset_path, **ds_kwargs)

    # Dùng chung utils/data/index_split.py với train_network.py: hai file cắt index bằng đúng
    # một hàm nên không thể lệch nhau nữa.
    splits = index_splits(
        test_dataset.length,
        train_frac=args.split,
        test_frac=args.test_split,
        shuffle=args.ds_shuffle,
        seed=args.random_seed,
    )
    val_indices = select_subset(splits, args.subset)
    if not val_indices:
        raise SystemExit(
            f"subset '{args.subset}' rỗng ({describe(splits)}). Với --subset test cần train "
            "bằng --test-split > 0 và truyền lại đúng --split/--test-split/--ds-shuffle."
        )
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info(f"Index splits: {describe(splits)}")
    logging.info(f"Evaluating subset '{args.subset}': {len(val_indices)} samples")
    if args.subset == "val":
        logging.warning(
            "subset='val' là tập đã dùng để chọn checkpoint trong train_network.py -- con số "
            "thu được lạc quan, không phải kết quả trên tập độc lập."
        )

    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.num_workers, sampler=val_sampler
    )
    logging.info("Done")

    for network in args.network:
        logging.info(f"\nEvaluating model {network}")

        # Load Network. `.eval()` là bắt buộc: model có dropout p=0.1 và BatchNorm, mà cờ
        # training được lưu kèm checkpoint -- trước đây chỉ đúng nhờ may mắn (train_network.py
        # tình cờ save ngay sau validate()).
        net = torch.load(network, map_location=device, weights_only=False).eval()

        results = {"correct": 0, "failed": 0}

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
                lossd = net.compute_loss(xc, yc, **eval_extras(net, batch, device))

                q_img, ang_img, width_img = post_process_output(
                    lossd["pred"]["pos"],
                    lossd["pred"]["cos"],
                    lossd["pred"]["sin"],
                    lossd["pred"]["width"],
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

        if args.jacquard_output:
            logging.info(f"Jacquard output saved to {jo_fn}")

        del net
        torch.cuda.empty_cache()
