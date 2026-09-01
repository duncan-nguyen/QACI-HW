import argparse
import logging
import time

import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.checkpoint import load_network
from utils.data import get_dataset
from utils.data.index_split import SUBSETS, describe, index_splits, select_subset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def eval_extras(net, batch, device):
    """Phần tử thứ 6 (prompt/part_mask) của Grasp-Anything++; xem train_network.py."""
    if len(batch) < 6 or not getattr(net, "accepts_extras", False):
        return {}
    extra = batch[5]
    kwargs = {}
    if "prompt" in extra:
        kwargs["prompts"] = extra["prompt"]
    if "part_mask" in extra:
        kwargs["part_mask"] = extra["part_mask"].to(device)
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size lúc evaluate. Forward chạy theo lô, nhưng post-process và IoU "
        "vẫn được tính riêng từng ảnh để giữ nguyên metric.",
    )

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
    parser.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="ĐỐI CHỨNG: ghép mỗi ảnh với prompt của một object khác. Nếu accuracy gần như "
        "không đổi thì model đang bỏ qua ngôn ngữ. Chỉ dùng được với grasp-anything-pp.",
    )

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != "jacquard":
        raise ValueError(
            "--jacquard-output can only be used with the --dataset jacquard option."
        )
    if args.jacquard_output and args.augment:
        raise ValueError("--jacquard-output can not be used with data augmentation.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

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

    if args.shuffle_prompts:
        if not hasattr(test_dataset, "shuffle_prompts"):
            raise SystemExit(
                f"--shuffle-prompts chỉ có nghĩa với dataset có prompt; {args.dataset} không có."
            )
        test_dataset.shuffle_prompts(seed=args.random_seed)
        logging.warning(
            "--shuffle-prompts: prompt đã bị hoán vị (seed %d). Con số dưới đây là ĐỐI CHỨNG "
            "ÂM, không phải kết quả của model.",
            args.random_seed,
        )

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
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )
    logging.info(f"Evaluation batch size: {args.batch_size}")
    logging.info("Done")

    for network in args.network:
        logging.info(f"\nEvaluating model {network}")

        # load_network đọc cả checkpoint state_dict (mới) lẫn module đã pickle (cũ) và luôn
        # trả về model ở .eval() -- bắt buộc vì model có dropout 0,1 và BatchNorm, mà cờ
        # training được lưu kèm checkpoint.
        net = load_network(network, map_location=device)

        results = {"correct": 0, "failed": 0}

        if args.jacquard_output:
            jo_fn = network + "_jacquard_output.txt"
            with open(jo_fn, "w") as f:
                pass

        start_time = time.time()
        n_samples = 0

        with torch.no_grad():
            for idx, batch in enumerate(test_data):
                x, y, didx, rot, zoom = batch[:5]
                batch_size = x.shape[0]
                n_samples += batch_size
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc, **eval_extras(net, batch, device))
                pred = lossd["pred"]

                # post_process_output dùng Gaussian filter 2D. Cắt từng sample trước khi
                # gọi để filter không trộn chiều batch vào chiều không gian.
                for i in range(batch_size):
                    sample_idx = int(didx[i])
                    sample_rot = float(rot[i])
                    sample_zoom = float(zoom[i])
                    q_img, ang_img, width_img = post_process_output(
                        pred["pos"][i : i + 1].float(),
                        pred["cos"][i : i + 1].float(),
                        pred["sin"][i : i + 1].float(),
                        pred["width"][i : i + 1].float(),
                    )

                    if args.iou_eval:
                        s = evaluation.calculate_iou_match(
                            q_img,
                            ang_img,
                            test_data.dataset.get_gtbb(
                                sample_idx, sample_rot, sample_zoom
                            ),
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
                                f.write(test_data.dataset.get_jname(sample_idx) + "\n")
                                f.write(g.to_jacquard(scale=1024 / 300) + "\n")

                    if args.vis:
                        save_results(
                            rgb_img=test_data.dataset.get_rgb(
                                sample_idx, sample_rot, sample_zoom, normalise=False
                            ),
                            depth_img=test_data.dataset.get_depth(
                                sample_idx, sample_rot, sample_zoom
                            ),
                            grasp_q_img=q_img,
                            grasp_angle_img=ang_img,
                            no_grasps=args.n_grasps,
                            grasp_width_img=width_img,
                        )

        avg_time = (time.time() - start_time) / max(1, n_samples)
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
