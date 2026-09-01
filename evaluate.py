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
    """The 6th element (prompt/part_mask) of Grasp-Anything++; see train_network.py."""
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
        help="Directory holding seen.obj/unseen.obj (default: the loader's SPLIT_DIRS)",
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
        help="Must match train_network.py's --test-split to reproduce the same slice.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="val",
        choices=list(SUBSETS),
        help="Which slice to evaluate. 'val' = the upstream behaviour (every index after "
        "--split), but that is the set used to select the checkpoint, so the number will be "
        "optimistic. Use 'test' for an independent set (requires training with --test-split "
        "> 0), or 'all' for the entire seen/unseen split.",
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
        help="Evaluation batch size. The forward pass is batched, but post-processing and IoU "
        "are still computed per image so the metric is unchanged.",
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
        help="COUNTERFACTUAL: pair every image with another object's prompt. If accuracy barely "
        "moves, the model is ignoring the language. Only works with grasp-anything-pp.",
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
    # Only passed when set, because the other loaders (cornell/jacquard/...) reject this kwarg.
    if args.split_path:
        ds_kwargs["split_path"] = args.split_path
    test_dataset = Dataset(args.dataset_path, **ds_kwargs)

    if args.shuffle_prompts:
        if not hasattr(test_dataset, "shuffle_prompts"):
            raise SystemExit(
                f"--shuffle-prompts only makes sense for datasets with prompts; "
                f"{args.dataset} has none."
            )
        test_dataset.shuffle_prompts(seed=args.random_seed)
        logging.warning(
            "--shuffle-prompts: prompts have been permuted (seed %d). The numbers below are a "
            "NEGATIVE CONTROL, not the model's result.",
            args.random_seed,
        )

    # Shares utils/data/index_split.py with train_network.py: both slice indices through the
    # very same function, so they can no longer drift apart.
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
            f"subset '{args.subset}' is empty ({describe(splits)}). --subset test requires "
            "training with --test-split > 0 and passing back the same "
            "--split/--test-split/--ds-shuffle."
        )
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info(f"Index splits: {describe(splits)}")
    logging.info(f"Evaluating subset '{args.subset}': {len(val_indices)} samples")
    if args.subset == "val":
        logging.warning(
            "subset='val' is the set used to select the checkpoint in train_network.py -- the "
            "resulting number is optimistic, not a result on an independent set."
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

        # load_network reads both the state_dict checkpoints and the pickled modules, and
        # always returns the model in .eval() -- required because the model has dropout 0.1 and
        # BatchNorm, and the training flag is stored with the checkpoint.
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

                # post_process_output applies a 2-D gaussian filter. Slice per sample before
                # calling it, so the filter does not mix the batch dimension into the spatial
                # ones.
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
