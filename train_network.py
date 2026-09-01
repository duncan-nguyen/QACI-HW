import argparse
import contextlib
import datetime
import inspect
import json
import logging
import os

# HuggingFace's "fast" tokenizers open a Rust thread pool; when the DataLoader forks workers
# afterwards, tokenizers prints a warning and can deadlock. We tokenize *inside* the workers,
# so no parallelism is needed in the main process. Must be set before transformers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import tensorboardX
import torch
import torch.utils.data
from torch import optim

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.checkpoint import save_checkpoint
from utils.data import get_dataset
from utils.data.index_split import describe, index_splits
from utils.dataset_processing import evaluation
from utils.diagnostics import TokenTally, align_stats, grad_norm_split, grad_norms
from utils.visualisation.gridshow import gridshow

# Default epochs for probe figures: dense early (attention learns fastest there), then sparse.
PROBE_EPOCHS = "0,1,3,10,25"

GRASP_LOSSES = ("p_loss", "cos_loss", "sin_loss", "width_loss")


def parse_args():
    parser = argparse.ArgumentParser(description="Train network")

    # Network
    parser.add_argument(
        "--network",
        type=str,
        default="grconvnet3",
        help="Network name in inference/models",
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help="Input image size for the network"
    )
    parser.add_argument(
        "--use-depth", type=int, default=1, help="Use Depth image for training (1/0)"
    )
    parser.add_argument(
        "--use-rgb", type=int, default=1, help="Use RGB image for training (1/0)"
    )
    parser.add_argument(
        "--use-dropout", type=int, default=1, help="Use dropout for training (1/0)"
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=0.1,
        help="Dropout prob for training (0-1)",
    )
    parser.add_argument(
        "--channel-size",
        type=int,
        default=32,
        help="Internal channel size for the network",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.25, help="Threshold for IOU matching"
    )

    # Datasets
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

    # Text-visual alignment branch (only used by the stag network)
    parser.add_argument(
        "--use-text",
        type=int,
        default=1,
        help="Enable the language branch (1/0). 0 = image-only baseline",
    )
    parser.add_argument(
        "--w-align", type=float, default=1.0, help="Weight of L_align(A_T, part_mask)"
    )
    parser.add_argument(
        "--align-mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="How tokens are pooled: 'soft' = softmax over tokens, 'hard' = argmax",
    )
    parser.add_argument(
        "--region-text",
        type=int,
        default=1,
        help="Feed R_T (the region's text feature) to the decoder (1/0)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="residual",
        choices=["residual", "gate"],
        help="'residual' = F + phi([F, F*A_T, R_T]); 'gate' = F*(1+lambda*A_T)",
    )
    parser.add_argument(
        "--align-stage",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "conv4"],
        help="Where A_T is computed: after res5 (56x56) or after conv4 (113x113, finer for small parts)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs over which the L_align weight is warmed up",
    )

    # Diagnostics (see utils/diagnostics.py and the "Reading the logs" section of the README)
    parser.add_argument(
        "--diag-interval",
        type=int,
        default=500,
        help="Log fusion/token/gradient stats to TensorBoard every N batches. 0 = off.",
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=8,
        help="Number of fixed samples used for alignment figures across epochs. 0 = no figures.",
    )
    parser.add_argument(
        "--probe-epochs",
        type=str,
        default=PROBE_EPOCHS,
        help="Comma-separated epochs at which the probe set is drawn (the best epoch is always added).",
    )
    parser.add_argument(
        "--counterfactual-every",
        type=int,
        default=5,
        help="Every N epochs, run validation with shuffled and fixed prompts to measure "
        "Delta_prompt. 0 = off. Each run costs two extra validation passes.",
    )
    parser.add_argument(
        "--fixed-prompt",
        type=str,
        default="Grasp the object.",
        help="The single prompt used for every image in the 'fixed' counterfactual arm",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Fraction of the dev set (everything outside --test-split) used for training; "
        "the remainder is validation",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Fraction of the dataset held out as test, taken from the end of the list and NOT "
        "used for training or checkpoint selection. Evaluate it with `evaluate.py --subset "
        "test` using the same --split/--test-split/--ds-shuffle. Default 0.0 = no held-out test.",
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
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches each DataLoader worker prepares ahead. Large values multiply quickly with "
        "batch-size and num-workers and can get the process SIGKILLed for running out of RAM.",
    )
    parser.add_argument(
        "--tokenize-in-loader",
        type=int,
        default=1,
        help="Tokenize prompts inside the DataLoader workers instead of the main process (1/0). "
        "Tokenizing 64 prompts costs 6.3 ms, straight on the critical path of every step.",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        help="Validation batch size. Upstream hard-codes 1, i.e. one image per GPU forward. "
        "Results are unchanged (post-processing and IoU are still computed per sample).",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "off", "bf16", "fp16"],
        help="Mixed precision. 'auto' = bf16 if the GPU supports it, otherwise off. 'fp16' adds a "
        "GradScaler. NOTE: enabling AMP shifts the numbers away from pure fp32 -- say so when "
        "reporting.",
    )
    parser.add_argument(
        "--channels-last",
        type=str,
        default="auto",
        choices=["auto", "0", "1"],
        help="NHWC layout for convolutions. 'auto' = on when AMP is on under CUDA (without tensor "
        "cores NHWC is usually slower).",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        type=int,
        default=1,
        help="Let cuDNN benchmark for the fastest conv algorithm. The input size is fixed, so this "
        "costs a few batches once and pays off for the rest of the run.",
    )
    parser.add_argument(
        "--tf32",
        type=int,
        default=1,
        help="Allow TF32 in matmul/cuDNN on Ampere and newer.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="Learning-rate schedule. 'none' keeps the upstream behaviour; 'cosine' decays smoothly "
        "to 0; 'step' divides by 10 at 60%% and 85%% of the epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3 for adam, 0.01 for sgd). Should be adjusted with the "
        "batch size; linear scaling is the rough rule.",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int, default=1000, help="Batches per Epoch"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="Optmizer for the training. (adam or SGD)",
    )

    # Logging etc.
    parser.add_argument(
        "--description", type=str, default="", help="Training description"
    )
    parser.add_argument("--logdir", type=str, default="logs/", help="Log directory")
    parser.add_argument(
        "--vis", action="store_true", help="Visualise the training process"
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
    return args


def resolve_amp(mode, device):
    """
    The autocast dtype, or None for pure fp32.

    'auto' only ever picks bf16 -- it has the same exponent range as fp32, so it needs no loss
    scaling and does not overflow. fp16 is about as fast but requires a GradScaler and can be
    unstable, so it is only used when explicitly asked for.
    """
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else None


def autocast_ctx(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def to_device(x, device, channels_last=False):
    """
    `non_blocking=True` only helps when the source memory is pinned -- the DataLoader has
    `pin_memory` on, so the H2D copy overlaps with the previous batch's compute.
    """
    x = x.to(device, non_blocking=True)
    if channels_last and x.dim() == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


class LossMeter:
    """
    Accumulate losses *on the GPU*, synchronising only once at the end of the epoch.

    Upstream calls `.item()` on the total and on every component inside the loop: 5-7
    `cudaStreamSynchronize` calls per batch, each blocking the CPU until the whole kernel
    queue drains -- exactly what breaks the overlap between data loading and compute.

    Accumulation is in float64 to keep the precision of the original Python summation
    (important under autocast, where the loss is bf16/fp16).
    """

    def __init__(self, device):
        self.device = device
        self.total = torch.zeros((), dtype=torch.float64, device=device)
        self.parts = {}
        self.weight = 0.0

    def update(self, loss, losses, weight=1.0):
        self.total += loss.detach().double() * weight
        for name, value in losses.items():
            if name not in self.parts:
                self.parts[name] = torch.zeros(
                    (), dtype=torch.float64, device=self.device
                )
            self.parts[name] += value.detach().double() * weight
        self.weight += weight

    def result(self):
        denominator = self.weight if self.weight else 1.0
        return (
            float(self.total) / denominator,
            {name: float(value) / denominator for name, value in self.parts.items()},
        )


def batch_extras(net, batch, device):
    """
    Pull prompt / part_mask from the 6th element of the batch (only Grasp-Anything++ has it).
    Models that do not declare `accepts_extras` get an empty dict, so the original
    compute_loss signature still works.
    """
    if len(batch) < 6 or not getattr(net, "accepts_extras", False):
        return {}
    extra = batch[5]
    kwargs = {}
    if "prompt_tokens" in extra:
        # The loader already tokenized in a worker -> hand the tensors straight to
        # CLIPTextEncoder (it moves them to the device). The "prompt" key keeps the raw
        # strings, for the figures.
        kwargs["prompts"] = extra["prompt_tokens"]
    elif "prompt" in extra:
        kwargs["prompts"] = extra["prompt"]
    if "part_mask" in extra:
        kwargs["part_mask"] = extra["part_mask"].to(device, non_blocking=True)
    return kwargs


def validate(
    net,
    device,
    val_data,
    iou_threshold,
    diagnostics=False,
    amp_dtype=None,
    channels_last=False,
):
    """
    Run validation.

    Batched (`--val-batch-size`): the forward pass runs on batches, while post-processing and
    IoU are still computed *per sample* on the `[i:i+1]` slice -- `post_process_output` applies
    a 2-D gaussian filter, so feeding it a whole batch would blur across the batch dimension.
    Keeping the slice on the GPU (rather than moving to CPU and slicing there) makes the
    numbers identical to the batch_size=1 version when AMP is off.

    Every average is weighted by the *sample count* of its batch, so a short final batch does
    not skew the result.

    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :param diagnostics: also collect alignment quality + token distribution
                        (utils/diagnostics.py). Only meaningful for models with a language
                        branch.
    :param amp_dtype: autocast dtype, None = fp32
    :param channels_last: move inputs to NHWC
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        "correct": 0,
        "failed": 0,
        "loss": 0,
        "losses": {},
        "align": {},
        "token": {},
        "tally": TokenTally(),
    }

    meter = LossMeter(device)
    collect = diagnostics and getattr(net, "use_text", False)
    n_align = 0

    with (
        torch.no_grad(),
        net.collecting_stats() if collect else contextlib.nullcontext(),
    ):
        for batch in val_data:
            # GA++ returns a 6th element (prompt / part_mask); other datasets return exactly 5.
            x, y, didx, rot, zoom_factor = batch[:5]
            batch_size = x.shape[0]
            xc = to_device(x, device, channels_last)
            yc = [to_device(yy, device) for yy in y]
            extras = batch_extras(net, batch, device)
            with autocast_ctx(device, amp_dtype):
                lossd = net.compute_loss(xc, yc, **extras)

            meter.update(lossd["loss"], lossd["losses"], weight=batch_size)

            if (
                collect
                and lossd.get("align_logits") is not None
                and "part_mask" in extras
            ):
                # Measured over the *whole* validation set, not the probe set: these answer
                # "is the selected region right", which needs statistics, not nice examples.
                n_align += batch_size
                for k, v in align_stats(
                    lossd["align_logits"], extras["part_mask"]
                ).items():
                    results["align"][k] = results["align"].get(k, 0.0) + v * batch_size
                stats = lossd.get("stats", {})
                for k, v in stats.items():
                    if k.startswith("token/") or k.startswith("fusion/"):
                        results["token"][k] = (
                            results["token"].get(k, 0.0) + v * batch_size
                        )
                if "_winning_tokens" in stats and "prompts" in extras:
                    results["tally"].update(
                        stats["_winning_tokens"],
                        stats["_text_mask"],
                        net.text_encoder.token_strings(extras["prompts"]),
                    )

            pred = lossd["pred"]
            for i in range(batch_size):
                # `.float()` is a no-op with AMP off; under bf16 it is required, because
                # numpy has no bfloat16 dtype.
                q_out, ang_out, w_out = post_process_output(
                    pred["pos"][i : i + 1].float(),
                    pred["cos"][i : i + 1].float(),
                    pred["sin"][i : i + 1].float(),
                    pred["width"][i : i + 1].float(),
                )

                s = evaluation.calculate_iou_match(
                    q_out,
                    ang_out,
                    val_data.dataset.get_gtbb(
                        int(didx[i]), float(rot[i]), float(zoom_factor[i])
                    ),
                    no_grasps=1,
                    grasp_width=w_out,
                    threshold=iou_threshold,
                )

                if s:
                    results["correct"] += 1
                else:
                    results["failed"] += 1

    results["loss"], results["losses"] = meter.result()
    for group in ("align", "token"):
        results[group] = (
            {k: v / n_align for k, v in results[group].items()} if n_align else {}
        )
    results["accuracy"] = results["correct"] / max(
        1, results["correct"] + results["failed"]
    )
    return results


def train(
    epoch,
    net,
    device,
    train_data,
    optimizer,
    batches_per_epoch,
    vis=False,
    tb=None,
    diag_interval=0,
    amp_dtype=None,
    scaler=None,
    channels_last=False,
):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :param tb: SummaryWriter for per-*step* diagnostics (fusion/token/gradient)
    :param diag_interval: measure once every N batches. 0 = off. Measuring every step is
                          wasteful: entropy + argmax over (B, L, H, W), and grad_norm_split
                          costs two extra backward passes.
    :param amp_dtype: autocast dtype (None = fp32)
    :param scaler: GradScaler, only really enabled when amp_dtype is fp16
    :param channels_last: move inputs to NHWC
    :return:  Average Losses for Epoch
    """
    results = {"loss": 0, "losses": {}}
    meter = LossMeter(device)

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    # The stop condition sits at the top of the inner loop: upstream increments `batch_idx`
    # before breaking, so the outer loop re-enters and fetches one more batch (copying
    # part_mask to the GPU) only to throw it away -- 1,999 optimizer steps for
    # --batches-per-epoch 2000.
    while batch_idx < batches_per_epoch:
        for batch in train_data:
            if batch_idx >= batches_per_epoch:
                break
            x, y = batch[0], batch[1]
            extras = batch_extras(net, batch, device)
            batch_idx += 1
            step = epoch * batches_per_epoch + batch_idx
            diag = (
                diag_interval
                and tb is not None
                and batch_idx % diag_interval == 0
                and getattr(net, "use_text", False)
            )

            xc = to_device(x, device, channels_last)
            yc = [to_device(yy, device) for yy in y]
            with net.collecting_stats() if diag else contextlib.nullcontext():
                with autocast_ctx(device, amp_dtype):
                    lossd = net.compute_loss(xc, yc, **extras)

            loss = lossd["loss"]

            if batch_idx % 100 == 0:
                # `.item()` synchronises the GPU, so it is only called every 100 batches. The
                # loss accumulation itself stays on the GPU in LossMeter until the epoch ends.
                logging.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():0.4f}"
                )

            meter.update(loss, lossd["losses"])

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if diag:
                    # grad_norms reads `.grad` directly, and at this point the gradients are
                    # still multiplied by the fp16 scale factor -> unscale before measuring,
                    # otherwise the numbers are meaningless.
                    scaler.unscale_(optimizer)
                    log_step_diagnostics(tb, net, lossd, step, xc, yc, extras)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if diag:
                    log_step_diagnostics(tb, net, lossd, step, xc, yc, extras)
                optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend(
                        [x[idx,].numpy().squeeze()]
                        + [yi[idx,].numpy().squeeze() for yi in y]
                        + [x[idx,].numpy().squeeze()]
                        + [
                            pc[idx,].detach().float().cpu().numpy().squeeze()
                            for pc in lossd["pred"].values()
                        ]
                    )
                gridshow(
                    "Display",
                    imgs,
                    [
                        (xc.min().item(), xc.max().item()),
                        (0.0, 1.0),
                        (0.0, 1.0),
                        (-1.0, 1.0),
                        (0.0, 1.0),
                    ]
                    * 2
                    * n_img,
                    [cv2.COLORMAP_BONE] * 10 * n_img,
                    10,
                )
                cv2.waitKey(2)

    results["loss"], results["losses"] = meter.result()

    return results


def log_step_diagnostics(tb, net, lossd, step, xc, yc, extras):
    """
    Per-step diagnostics: is fusion having an effect, are tokens collapsing, how large is the
    L_align gradient relative to L_grasp. Call *between* backward() and optimizer.step().
    """
    # The "step/" prefix: same metric names as on validation, but the x axis is the *step*, not
    # the epoch -- sharing a tag would overlay two curves on two different scales.
    for name, value in lossd.get("stats", {}).items():
        if not name.startswith("_"):
            tb.add_scalar(f"step/{name}", value, step)

    for group, norm in grad_norms(net).items():
        tb.add_scalar(f"gradient/{group}", norm, step)

    # Two extra backward passes on this very batch: we need to know where lambda puts the
    # align gradient relative to the grasp gradient *on the same data*, not on another batch.
    if "part_mask" in extras and "prompts" in extras:
        split = grad_norm_split(net, xc, yc, extras["prompts"], extras["part_mask"])
        for name, value in split.items():
            tb.add_scalar(f"gradient/visual_from_{name}", value, step)


def log_epoch(tb, epoch, train_results, val_results, lr, lambda_align):
    """All per-epoch scalars, named by group so TensorBoard collates the right charts."""

    def grasp_part(losses):
        return sum(losses.get(k, 0.0) for k in GRASP_LOSSES)

    tb.add_scalar("loss/train/total", train_results["loss"], epoch)
    tb.add_scalar("loss/train/grasp", grasp_part(train_results["losses"]), epoch)
    tb.add_scalar("loss/val/total", val_results["loss"], epoch)
    tb.add_scalar("loss/val/grasp", grasp_part(val_results["losses"]), epoch)
    for name in ("align_bce", "align_dice"):
        # Warmup pushes the *total* up while each component goes down -- reading the total
        # alone is misleading, so these two components get their own curves.
        if name in train_results["losses"]:
            tb.add_scalar(f"loss/train/{name}", train_results["losses"][name], epoch)
        if name in val_results["losses"]:
            tb.add_scalar(f"loss/val/{name}", val_results["losses"][name], epoch)

    tb.add_scalar("weight/lambda_align", lambda_align, epoch)
    tb.add_scalar("optimizer/lr", lr, epoch)
    tb.add_scalar("metric/val_iou", val_results["accuracy"], epoch)

    for name, value in val_results.get("align", {}).items():
        tb.add_scalar(f"align/{name}", value, epoch)
    for name, value in val_results.get("token", {}).items():
        # These already carry the token/ or fusion/ prefix; this is the average over the
        # *whole* validation set, unlike step/... which is measured on one training batch.
        tb.add_scalar(name, value, epoch)

    tally = val_results.get("tally")
    if tally is not None and tally.total:
        # Tokens winning the most pixels across the validation set. If the top entries are all
        # object nouns, the model is localising the object rather than the part.
        tb.add_text("token/winning", tally.as_text(20).replace("\n", "\n\n"), epoch)
        tb.add_scalar("token/punctuation_mass", tally.punctuation_mass(), epoch)
        logging.info(
            "top winning tokens: "
            + ", ".join(f"{t} {p:.1%}" for t, p in tally.top(5))
        )


def log_counterfactual(
    tb, net, device, loaders, epoch, iou_threshold, amp_dtype=None, channels_last=False
):
    """
    Counterfactual: correct prompt / shuffled prompt / one fixed prompt for every image.

    Delta_prompt = S_normal - S_shuffled. Near 0 means nothing can be concluded about the
    language branch being useful, however low align_loss got -- this number matters more than
    any loss curve.
    """
    scores = {}
    for name, loader in loaders.items():
        if loader is None:
            continue
        scores[name] = validate(
            net,
            device,
            loader,
            iou_threshold,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )["accuracy"]
        tb.add_scalar(f"counterfactual/{name}_success", scores[name], epoch)

    if "normal" in scores and "shuffled" in scores:
        drop = scores["normal"] - scores["shuffled"]
        tb.add_scalar("counterfactual/prompt_drop", drop, epoch)
        logging.info(
            "counterfactual: "
            + "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            + f"  Δ_prompt={drop:+.4f}"
        )
    return scores


def save_probe_figures(net, dataset, indices, folder, epoch, tb=None):
    """
    A *fixed* probe set: the same samples, redrawn at every epoch.

    Changing samples between epochs makes them incomparable -- and what we need to see is
    exactly when attention starts to learn, whether it collapses after warmup, and whether the
    grasp improves along with it or only the mask gets prettier.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.visualisation.alignment import plot_prompt_grid, plot_sample_diagnostics

    os.makedirs(folder, exist_ok=True)
    was_training = net.training
    net.eval()
    try:
        for k, idx in enumerate(indices):
            fig, info = plot_sample_diagnostics(net, dataset, idx)
            fig.savefig(
                os.path.join(folder, f"probe_{k:02d}_epoch_{epoch:03d}.png"),
                dpi=110,
                bbox_inches="tight",
            )
            if tb is not None:
                tb.add_figure(f"probe/{k:02d}", fig, epoch)
            plt.close(fig)
            if k == 0:
                logging.info(f"probe 00: IoU={info['iou']:.3f} tokens={info['tokens']}")

        # Same image, different prompts -- the most important figure: if the three rows are
        # identical, the model is ignoring the language.
        if indices and getattr(net, "use_text", False):
            idx = indices[0]
            prompts = probe_prompts(dataset, idx)
            fig = plot_prompt_grid(net, dataset, idx, prompts)
            fig.savefig(
                os.path.join(folder, f"prompt_grid_epoch_{epoch:03d}.png"),
                dpi=110,
                bbox_inches="tight",
            )
            if tb is not None:
                tb.add_figure("probe/prompt_grid", fig, epoch)
            plt.close(fig)
    finally:
        net.train(was_training)


def probe_prompts(dataset, idx):
    """
    The real prompt + prompts of other parts of the *same object* + a neutral prompt.

    These three separate three levels: right part, wrong part but right object, and nothing
    specific said at all.
    """
    prompts = [("real", dataset.get_prompt(idx, use_permutation=False))]
    object_id = dataset._object_id(dataset.sample_id(idx))
    for path in dataset._files_by_object.get(object_id, [])[:3]:
        other = dataset.grasp_files.index(path)
        if other != idx:
            prompts.append(
                ("other part", dataset.get_prompt(other, use_permutation=False))
            )
    prompts.append(("neutral", "Grasp the object."))
    return prompts[:4]


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime("%y%m%d_%H%M")
    net_desc = "{}_{}".format(dt, "_".join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, "commandline_args.json")
        with open(params_path, "w") as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, "log"),
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # Backend configuration. The input size is fixed for the whole run, so `cudnn.benchmark`
    # only searches during the first few batches and reuses the result; with constantly
    # changing shapes the flag would backfire.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    amp_dtype = resolve_amp(args.amp, device)
    # fp16 needs a GradScaler; bf16 does not (same exponent range as fp32). The GradScaler is
    # created anyway with `enabled=False`, so the code below needs only one branch.
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)
    if args.channels_last == "auto":
        channels_last = amp_dtype is not None and device.type == "cuda"
    else:
        channels_last = bool(int(args.channels_last))
    logging.info(
        "Precision: %s | channels_last=%s | cudnn.benchmark=%s",
        "fp32" if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        channels_last,
        torch.backends.cudnn.benchmark,
    )
    if amp_dtype is not None:
        logging.warning(
            "AMP is ON: loss and accuracy will differ slightly from a pure fp32 run. Say so "
            "when comparing tables with older runs (the setting is in commandline_args.json)."
        )

    # Load Dataset
    logging.info(f"Loading {args.dataset.title()} Dataset...")
    Dataset = get_dataset(args.dataset)
    ds_kwargs = dict(
        output_size=args.input_size,
        ds_rotate=args.ds_rotate,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb,
        seen=args.seen,
    )
    # Only passed when set, because the other loaders (cornell/jacquard/...) reject this kwarg.
    if args.split_path:
        ds_kwargs["split_path"] = args.split_path

    # The tokenizer has to be attached to the dataset *before* the DataLoader is built: with
    # persistent_workers, once the workers have forked, attributes set in the main process no
    # longer propagate (the same trap noted for the counterfactual loaders below).
    if (
        args.use_text
        and args.tokenize_in_loader
        and args.num_workers > 0
        and "prompt_tokenizer" in inspect.signature(Dataset.__init__).parameters
    ):
        try:
            from inference.models.stag import PromptTokenizer

            ds_kwargs["prompt_tokenizer"] = PromptTokenizer()
            logging.info("Tokenizing prompts inside the DataLoader workers.")
        except Exception as exc:  # transformers missing / CLIP checkpoint not downloadable
            logging.warning(
                "Could not build a PromptTokenizer (%s); falling back to tokenizing in the "
                "main process.",
                exc,
            )

    dataset = Dataset(args.dataset_path, **ds_kwargs)
    # Validation must run on *un-augmented* data: see GraspDatasetBase.eval_view().
    val_dataset = dataset.eval_view()
    logging.info(f"Dataset size is {dataset.length}")

    # Creating data indices for training and validation splits
    splits = index_splits(
        dataset.length,
        train_frac=args.split,
        test_frac=args.test_split,
        shuffle=args.ds_shuffle,
        seed=args.random_seed,
    )
    train_indices, val_indices = splits["train"], splits["val"]
    logging.info(f"Index splits: {describe(splits)}")
    if splits["test"]:
        # Logged explicitly so `evaluate.py --subset test` can reproduce this exact slice.
        logging.info(
            "Test set held out, used neither for training nor checkpoint selection. "
            "Evaluate it with: "
            f"evaluate.py --subset test --split {args.split} --test-split {args.test_split}"
            f"{' --ds-shuffle' if args.ds_shuffle else ''} --random-seed {args.random_seed}"
        )
    else:
        logging.warning(
            "--test-split 0.0: no held-out test set. Any 'test' number reported later will be "
            "the very validation set used to select the checkpoint."
        )

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    # train() rebuilds the iterator on every pass of `while batch_idx < batches_per_epoch`, so
    # persistent_workers saves respawning the workers each time. The GA++ loader is CPU-heavy
    # (JPEG decode + rotate/zoom for the image, part_mask and M_union), so prefetching pays too.
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")

    train_loader_kwargs = dict(num_workers=args.num_workers)
    eval_loader_kwargs = dict(num_workers=args.num_workers)
    if args.num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            pin_memory=True,
        )
        # There are three eval loaders (normal/shuffled/fixed). If they were all persistent,
        # after a counterfactual pass we would hold up to 4*num_workers processes along with
        # their queues and pinned batches. Letting eval workers exit after each pass keeps RAM
        # from growing with the number of conditions.
        eval_loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=min(args.prefetch_factor, 2),
            pin_memory=True,
        )

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **train_loader_kwargs,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        **eval_loader_kwargs,
    )

    # The counterfactual loaders must be built *here*: with persistent_workers, changing a
    # dataset attribute in the main process after the workers have forked is invisible to them
    # -- the prompts would still be the real ones and the measurement meaningless.
    counterfactual_loaders = {"normal": val_data}
    if (
        args.counterfactual_every
        and hasattr(val_dataset, "shuffle_prompts")
        and args.use_text
    ):
        shuffled = val_dataset.eval_view().shuffle_prompts(seed=args.random_seed)
        fixed = val_dataset.eval_view().set_fixed_prompt(args.fixed_prompt)
        counterfactual_loaders["shuffled"] = torch.utils.data.DataLoader(
            shuffled,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            **eval_loader_kwargs,
        )
        counterfactual_loaders["fixed"] = torch.utils.data.DataLoader(
            fixed,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            **eval_loader_kwargs,
        )
    logging.info("Done")

    # Load the network
    logging.info("Loading Network...")
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net_kwargs = dict(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size,
    )
    if getattr(network, "accepts_extras", False):
        net_kwargs.update(
            use_text=bool(args.use_text),
            w_align=args.w_align,
            align_mode=args.align_mode,
            region_text=bool(args.region_text),
            fusion=args.fusion,
            align_stage=args.align_stage,
        )
    net = network(**net_kwargs)

    net = net.to(device)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    logging.info("Done")

    # The CLIP text encoder is frozen -> keep it out of the optimizer.
    params = [p for p in net.parameters() if p.requires_grad]
    if args.optim.lower() == "adam":
        optimizer = optim.Adam(params, lr=args.lr if args.lr else 1e-3)
    elif args.optim.lower() == "sgd":
        optimizer = optim.SGD(params, lr=args.lr if args.lr else 0.01, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {args.optim} is not implemented")

    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_schedule == "step":
        milestones = [int(args.epochs * 0.6), int(args.epochs * 0.85)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    else:
        scheduler = None

    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()

    # Fixed probe set, taken from the validation split: the same samples at every epoch.
    probe_indices = (
        sorted(val_indices)[: args.probe_samples] if args.probe_samples else []
    )
    probe_epochs = {int(e) for e in args.probe_epochs.split(",") if e.strip()}
    figures_dir = os.path.join(save_folder, "figures")
    if probe_indices:
        logging.info(
            f"Probe set: {len(probe_indices)} samples, drawn at epochs "
            f"{sorted(probe_epochs)} and at every new best epoch"
        )

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info(f"Beginning Epoch {epoch:02d}")
        if hasattr(net, "set_loss_warmup"):
            # Phase L_align in: A_T is random at first, forcing it hard would wreck the grasp branch.
            net.set_loss_warmup(min(1.0, (epoch + 1) / max(1, args.warmup_epochs)))
        train_results = train(
            epoch,
            net,
            device,
            train_data,
            optimizer,
            args.batches_per_epoch,
            vis=args.vis,
            tb=tb,
            diag_interval=args.diag_interval,
            amp_dtype=amp_dtype,
            scaler=scaler,
            channels_last=channels_last,
        )

        # Run Validation
        logging.info("Validating...")
        test_results = validate(
            net,
            device,
            val_data,
            args.iou_threshold,
            diagnostics=True,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        iou = test_results["accuracy"]
        logging.info(
            "%d/%d = %f"
            % (
                test_results["correct"],
                test_results["correct"] + test_results["failed"],
                iou,
            )
        )
        if test_results["align"]:
            logging.info(
                "align: "
                + "  ".join(f"{k}={v:.4f}" for k, v in test_results["align"].items())
            )

        log_epoch(
            tb,
            epoch,
            train_results,
            test_results,
            optimizer.param_groups[0]["lr"],
            getattr(net, "lambda_align", 0.0),
        )

        if args.counterfactual_every and (epoch + 1) % args.counterfactual_every == 0:
            log_counterfactual(
                tb,
                net,
                device,
                counterfactual_loaders,
                epoch,
                args.iou_threshold,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )

        # Save best performing network
        # `best_iou` is only updated on a genuine improvement. Upstream updates it inside the
        # `or (epoch % 10) == 0` branch, so every tenth epoch lowers the bar to a possibly
        # worse value and later epochs get saved as spurious "best" checkpoints.
        is_best = iou > best_iou
        if is_best or epoch == 0 or (epoch % 10) == 0:
            # Four decimals: %0.2f rounds several epochs to the same filename, forcing a log
            # grep to find which checkpoint is actually best.
            # save_checkpoint stores state_dict + kwargs without the CLIP text tower: ~9 MB
            # per file instead of 265 MB (utils/checkpoint.py).
            save_checkpoint(
                net, os.path.join(save_folder, "epoch_%02d_iou_%0.4f" % (epoch, iou))
            )
        if is_best:
            best_iou = iou
            logging.info(f"New best IoU {best_iou:.4f} at epoch {epoch}")

        if probe_indices and (epoch in probe_epochs or is_best):
            save_probe_figures(
                net, val_dataset, probe_indices, figures_dir, epoch, tb=tb
            )

        if scheduler is not None:
            scheduler.step()

    # One final counterfactual on the final model, even when the --counterfactual-every
    # schedule does not land on the last epoch: this is the number that goes into the report.
    if args.counterfactual_every:
        log_counterfactual(
            tb,
            net,
            device,
            counterfactual_loaders,
            args.epochs - 1,
            args.iou_threshold,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
    tb.flush()


if __name__ == "__main__":
    run()
