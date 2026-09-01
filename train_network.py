import argparse
import contextlib
import datetime
import json
import logging
import os

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

# Epoch mặc định để lưu hình probe: dày lúc đầu (attention học nhanh nhất ở đó), thưa dần.
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
        help="Thư mục chứa seen.obj/unseen.obj (mặc định dò theo SPLIT_DIRS của loader)",
    )

    # Nhánh text-visual alignment (chỉ dùng bởi network grconvnet3_align)
    parser.add_argument(
        "--use-text",
        type=int,
        default=1,
        help="Bật nhánh ngôn ngữ (1/0). 0 = baseline no-text",
    )
    parser.add_argument(
        "--w-align", type=float, default=1.0, help="Trọng số L_align(A_T, part_mask)"
    )
    parser.add_argument(
        "--align-mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Cách gộp token: 'soft' = softmax theo token (V2), 'hard' = max (V1)",
    )
    parser.add_argument(
        "--region-text",
        type=int,
        default=1,
        help="Đưa R_T (text feature của vùng) vào decoder (1/0)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="residual",
        choices=["residual", "gate"],
        help="'residual' = F + phi([F, F*A_T, R_T]) (V2); 'gate' = F*(1+lambda*A_T) (V1)",
    )
    parser.add_argument(
        "--align-stage",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "conv4"],
        help="Nơi tính A_T: sau res5 (56x56) hay sau conv4 (113x113, mịn hơn cho part nhỏ)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Số epoch warmup trọng số L_align",
    )

    # Chẩn đoán (xem utils/diagnostics.py và README mục "Đọc log")
    parser.add_argument(
        "--diag-interval",
        type=int,
        default=500,
        help="Cứ mỗi bấy nhiêu batch thì ghi fusion/token/gradient vào TensorBoard. 0 = tắt.",
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=8,
        help="Số sample cố định để vẽ hình alignment qua các epoch. 0 = không vẽ.",
    )
    parser.add_argument(
        "--probe-epochs",
        type=str,
        default=PROBE_EPOCHS,
        help="Các epoch vẽ probe set, ngăn bằng dấu phẩy (epoch tốt nhất luôn được vẽ thêm).",
    )
    parser.add_argument(
        "--counterfactual-every",
        type=int,
        default=5,
        help="Cứ mỗi bấy nhiêu epoch thì chạy validation với prompt hoán vị và prompt cố định "
        "để đo Δ_prompt. 0 = tắt. Mỗi lần tốn thêm hai lượt validate.",
    )
    parser.add_argument(
        "--fixed-prompt",
        type=str,
        default="Grasp the object.",
        help="Prompt dùng chung cho mọi ảnh trong nhánh đối chứng 'fixed'",
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
        help="Tỉ lệ dataset giữ lại làm test, cắt ở cuối danh sách và KHÔNG dùng lúc train "
        "hay chọn checkpoint. Đánh giá bằng `evaluate.py --subset test` với cùng --split/"
        "--test-split/--ds-shuffle. Mặc định 0.0 = hành vi cũ (không có tập test độc lập).",
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

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="Lịch giảm learning rate. 'none' giữ nguyên hành vi cũ; 'cosine' giảm mượt về 0; "
        "'step' chia 10 ở 60%% và 85%% số epoch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (mặc định: 1e-3 cho adam, 0.01 cho sgd). Đổi batch size thì nên "
        "đổi theo, quy tắc thô là scale tuyến tính.",
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


def batch_extras(net, batch, device):
    """
    Lấy prompt / part_mask ở phần tử thứ 6 của batch (chỉ Grasp-Anything++ có).
    Model không khai báo `accepts_extras` thì trả dict rỗng -> chữ ký compute_loss cũ giữ nguyên.
    """
    if len(batch) < 6 or not getattr(net, "accepts_extras", False):
        return {}
    extra = batch[5]
    kwargs = {}
    if "prompt" in extra:
        kwargs["prompts"] = extra["prompt"]
    if "part_mask" in extra:
        kwargs["part_mask"] = extra["part_mask"].to(device)
    return kwargs


def validate(net, device, val_data, iou_threshold, diagnostics=False):
    """
    Run validation.

    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :param diagnostics: gom thêm chất lượng alignment + phân bố token (utils/diagnostics.py).
                        Chỉ có ý nghĩa với model có nhánh ngôn ngữ.
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {"correct": 0, "failed": 0, "loss": 0, "losses": {}, "align": {}, "token": {},
               "tally": TokenTally()}

    ld = len(val_data)
    collect = diagnostics and getattr(net, "use_text", False)
    n_align = 0

    with torch.no_grad(), (net.collecting_stats() if collect else contextlib.nullcontext()):
        for batch in val_data:
            # GA++ trả thêm phần tử thứ 6 (prompt / part_mask); các dataset khác trả đúng 5.
            x, y, didx, rot, zoom_factor = batch[:5]
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            extras = batch_extras(net, batch, device)
            lossd = net.compute_loss(xc, yc, **extras)

            loss = lossd["loss"]

            results["loss"] += loss.item() / ld
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item() / ld

            if collect and lossd.get("align_logits") is not None and "part_mask" in extras:
                # Đo trên *toàn bộ* val, không phải trên probe set: hai thứ này trả lời câu
                # "vùng chọn có đúng không" nên cần thống kê chứ không cần ví dụ đẹp.
                n_align += 1
                for k, v in align_stats(lossd["align_logits"], extras["part_mask"]).items():
                    results["align"][k] = results["align"].get(k, 0.0) + v
                stats = lossd.get("stats", {})
                for k, v in stats.items():
                    if k.startswith("token/") or k.startswith("fusion/"):
                        results["token"][k] = results["token"].get(k, 0.0) + v
                if "_winning_tokens" in stats and "prompts" in extras:
                    results["tally"].update(stats["_winning_tokens"], stats["_text_mask"],
                                            net.text_encoder.token_strings(extras["prompts"]))

            q_out, ang_out, w_out = post_process_output(
                lossd["pred"]["pos"],
                lossd["pred"]["cos"],
                lossd["pred"]["sin"],
                lossd["pred"]["width"],
            )

            s = evaluation.calculate_iou_match(
                q_out,
                ang_out,
                val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                no_grasps=1,
                grasp_width=w_out,
                threshold=iou_threshold,
            )

            if s:
                results["correct"] += 1
            else:
                results["failed"] += 1

    for group in ("align", "token"):
        results[group] = {k: v / n_align for k, v in results[group].items()} if n_align else {}
    results["accuracy"] = results["correct"] / max(1, results["correct"] + results["failed"])
    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False,
          tb=None, diag_interval=0):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :param tb: SummaryWriter để ghi chẩn đoán theo *step* (fusion/token/gradient)
    :param diag_interval: cứ mỗi bấy nhiêu batch thì đo một lần. 0 = tắt. Đo mỗi bước là phí:
                          entropy + argmax trên (B, L, H, W) và grad_norm_split tốn thêm hai
                          lượt backward.
    :return:  Average Losses for Epoch
    """
    results = {"loss": 0, "losses": {}}

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    # Điều kiện dừng nằm ở đầu vòng trong: bản cũ tăng `batch_idx` rồi mới break, nên vòng
    # ngoài vào lại một lần nữa, nạp thêm một batch (và copy part_mask lên GPU) chỉ để vứt đi
    # -- kết quả là 1.999 optimizer step cho --batches-per-epoch 2000.
    while batch_idx < batches_per_epoch:
        for batch in train_data:
            if batch_idx >= batches_per_epoch:
                break
            x, y = batch[0], batch[1]
            extras = batch_extras(net, batch, device)
            batch_idx += 1
            step = epoch * batches_per_epoch + batch_idx
            diag = (diag_interval and tb is not None and batch_idx % diag_interval == 0
                    and getattr(net, "use_text", False))

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            with net.collecting_stats() if diag else contextlib.nullcontext():
                lossd = net.compute_loss(xc, yc, **extras)

            loss = lossd["loss"]

            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():0.4f}"
                )

            results["loss"] += loss.item()
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item()

            optimizer.zero_grad()
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
                            pc[idx,].detach().cpu().numpy().squeeze()
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

    results["loss"] /= batch_idx
    for l in results["losses"]:
        results["losses"][l] /= batch_idx

    return results


def log_step_diagnostics(tb, net, lossd, step, xc, yc, extras):
    """
    Chẩn đoán theo step: fusion có tác động không, token có collapse không, gradient của
    L_align so với L_grasp lớn cỡ nào. Gọi *giữa* backward() và optimizer.step().
    """
    # Tiền tố "step/": cùng tên với chỉ số đo trên validation nhưng trục x là *step* chứ không
    # phải epoch -- để chung một tag thì hai đường chồng lên nhau ở hai thang đo khác nhau.
    for name, value in lossd.get("stats", {}).items():
        if not name.startswith("_"):
            tb.add_scalar(f"step/{name}", value, step)

    for group, norm in grad_norms(net).items():
        tb.add_scalar(f"gradient/{group}", norm, step)

    # Hai backward phụ trên chính batch này: cần biết lambda đang đặt gradient align ở đâu so
    # với gradient grasp *trên cùng dữ liệu*, chứ không phải trên một batch khác.
    if "part_mask" in extras and "prompts" in extras:
        split = grad_norm_split(net, xc, yc, extras["prompts"], extras["part_mask"])
        for name, value in split.items():
            tb.add_scalar(f"gradient/visual_from_{name}", value, step)


def log_epoch(tb, epoch, train_results, val_results, lr, lambda_align):
    """Toàn bộ scalar theo epoch, đặt tên theo nhóm để TensorBoard gộp đúng biểu đồ."""
    def grasp_part(losses):
        return sum(losses.get(k, 0.0) for k in GRASP_LOSSES)

    tb.add_scalar("loss/train/total", train_results["loss"], epoch)
    tb.add_scalar("loss/train/grasp", grasp_part(train_results["losses"]), epoch)
    tb.add_scalar("loss/val/total", val_results["loss"], epoch)
    tb.add_scalar("loss/val/grasp", grasp_part(val_results["losses"]), epoch)
    for name in ("align_bce", "align_dice"):
        # Warmup làm *tổng* đi lên trong lúc từng thành phần đang đi xuống -- nhìn tổng thôi
        # thì đọc nhầm, nên hai thành phần này phải có đường riêng.
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
        # Đã mang sẵn tiền tố token/ hoặc fusion/; đây là trung bình trên *toàn bộ* tập
        # validation, khác với step/... đo trên một batch train.
        tb.add_scalar(name, value, epoch)

    tally = val_results.get("tally")
    if tally is not None and tally.total:
        # Token thắng nhiều pixel nhất trên cả tập val. Nếu top toàn danh từ object thì model
        # đang định vị object chứ không phải part.
        tb.add_text("token/winning", tally.as_text(20).replace("\n", "\n\n"), epoch)
        tb.add_scalar("token/punctuation_mass", tally.punctuation_mass(), epoch)
        logging.info("token thắng nhiều nhất: "
                     + ", ".join(f"{t} {p:.1%}" for t, p in tally.top(5)))


def log_counterfactual(tb, net, device, loaders, epoch, iou_threshold):
    """
    Đối chứng: prompt đúng / prompt hoán vị / một prompt cố định cho mọi ảnh.

    Δ_prompt = S_normal - S_shuffled. Gần 0 thì không kết luận được nhánh ngôn ngữ có ích,
    dù align_loss có thấp đến đâu -- đây là con số quan trọng hơn mọi đường loss.
    """
    scores = {}
    for name, loader in loaders.items():
        if loader is None:
            continue
        scores[name] = validate(net, device, loader, iou_threshold)["accuracy"]
        tb.add_scalar(f"counterfactual/{name}_success", scores[name], epoch)

    if "normal" in scores and "shuffled" in scores:
        drop = scores["normal"] - scores["shuffled"]
        tb.add_scalar("counterfactual/prompt_drop", drop, epoch)
        logging.info("counterfactual: " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
                     + f"  Δ_prompt={drop:+.4f}")
    return scores


def save_probe_figures(net, dataset, indices, folder, epoch, tb=None):
    """
    Probe set *cố định*: cùng những sample đó, epoch nào cũng vẽ lại.

    Đổi sample giữa các epoch thì không so sánh được -- mà thứ cần thấy chính là attention bắt
    đầu học từ lúc nào, có collapse sau warmup không, và grasp có tốt lên theo hay chỉ mỗi mask
    đẹp lên.
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
            fig.savefig(os.path.join(folder, f"probe_{k:02d}_epoch_{epoch:03d}.png"),
                        dpi=110, bbox_inches="tight")
            if tb is not None:
                tb.add_figure(f"probe/{k:02d}", fig, epoch)
            plt.close(fig)
            if k == 0:
                logging.info(f"probe 00: IoU={info['iou']:.3f} tokens={info['tokens']}")

        # Cùng một ảnh, đổi prompt: hình quan trọng nhất: nếu ba hàng giống hệt nhau thì model
        # đang bỏ qua ngôn ngữ.
        if indices and getattr(net, "use_text", False):
            idx = indices[0]
            prompts = probe_prompts(dataset, idx)
            fig = plot_prompt_grid(net, dataset, idx, prompts)
            fig.savefig(os.path.join(folder, f"prompt_grid_epoch_{epoch:03d}.png"),
                        dpi=110, bbox_inches="tight")
            if tb is not None:
                tb.add_figure("probe/prompt_grid", fig, epoch)
            plt.close(fig)
    finally:
        net.train(was_training)


def probe_prompts(dataset, idx):
    """
    Prompt thật + prompt của các part khác *cùng object* + một prompt trung tính.

    Ba nguồn này tách được ba mức: đúng part, sai part nhưng đúng vật, và không nói gì cụ thể.
    """
    prompts = [("thật", dataset.get_prompt(idx, use_permutation=False))]
    object_id = dataset._object_id(dataset.sample_id(idx))
    for path in dataset._files_by_object.get(object_id, [])[:3]:
        other = dataset.grasp_files.index(path)
        if other != idx:
            prompts.append(("part khác", dataset.get_prompt(other, use_permutation=False)))
    prompts.append(("trung tính", "Grasp the object."))
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
    # Chỉ truyền khi có, vì các loader khác (cornell/jacquard/...) không nhận kwarg này.
    if args.split_path:
        ds_kwargs["split_path"] = args.split_path
    dataset = Dataset(args.dataset_path, **ds_kwargs)
    # Validation phải chạy trên dữ liệu *không* augmentation: xem GraspDatasetBase.eval_view().
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
        # Ghi rõ ra log để `evaluate.py --subset test` tái lập được đúng lát cắt này.
        logging.info(
            "Test set giữ nguyên, không dùng để train hay chọn checkpoint. Đánh giá bằng: "
            f"evaluate.py --subset test --split {args.split} --test-split {args.test_split}"
            f"{' --ds-shuffle' if args.ds_shuffle else ''} --random-seed {args.random_seed}"
        )
    else:
        logging.warning(
            "--test-split 0.0: không có tập test độc lập. Chỉ số 'test' báo cáo sau này sẽ "
            "chính là tập validation đã dùng để chọn checkpoint."
        )

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    # train() tạo lại iterator sau mỗi vòng `while batch_idx <= batches_per_epoch`, nên
    # persistent_workers tiết kiệm hẳn việc spawn lại worker. Loader của GA++ nặng CPU
    # (decode JPEG + rotate/zoom cho ảnh, part_mask và M_union) nên prefetch cũng đáng.
    loader_kwargs = dict(num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=4, pin_memory=True
        )

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler, **loader_kwargs
    )

    # Loader đối chứng phải dựng *ở đây*: với persistent_workers, đổi thuộc tính dataset ở
    # tiến trình chính sau khi worker đã fork thì worker không thấy -- prompt vẫn là prompt
    # thật và số đo sẽ vô nghĩa.
    counterfactual_loaders = {"normal": val_data}
    if args.counterfactual_every and hasattr(val_dataset, "shuffle_prompts") and args.use_text:
        shuffled = val_dataset.eval_view().shuffle_prompts(seed=args.random_seed)
        fixed = val_dataset.eval_view().set_fixed_prompt(args.fixed_prompt)
        counterfactual_loaders["shuffled"] = torch.utils.data.DataLoader(
            shuffled, batch_size=1, sampler=val_sampler, **loader_kwargs)
        counterfactual_loaders["fixed"] = torch.utils.data.DataLoader(
            fixed, batch_size=1, sampler=val_sampler, **loader_kwargs)
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
    logging.info("Done")

    # CLIP text encoder bị đóng băng -> lọc ra khỏi optimizer cho gọn.
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

    # Probe set cố định, lấy từ tập validation: cùng những sample này ở mọi epoch.
    probe_indices = sorted(val_indices)[: args.probe_samples] if args.probe_samples else []
    probe_epochs = {int(e) for e in args.probe_epochs.split(",") if e.strip()}
    figures_dir = os.path.join(save_folder, "figures")
    if probe_indices:
        logging.info(f"Probe set: {len(probe_indices)} sample, vẽ ở epoch "
                     f"{sorted(probe_epochs)} và ở mỗi epoch tốt nhất")

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info(f"Beginning Epoch {epoch:02d}")
        if hasattr(net, "set_loss_warmup"):
            # L_align vào dần: A_T lúc đầu ngẫu nhiên, ép mạnh sẽ kéo hỏng nhánh grasp.
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
        )

        # Run Validation
        logging.info("Validating...")
        test_results = validate(net, device, val_data, args.iou_threshold, diagnostics=True)
        iou = test_results["accuracy"]
        logging.info("%d/%d = %f" % (test_results["correct"],
                                     test_results["correct"] + test_results["failed"], iou))
        if test_results["align"]:
            logging.info("align: " + "  ".join(f"{k}={v:.4f}"
                                               for k, v in test_results["align"].items()))

        log_epoch(tb, epoch, train_results, test_results,
                  optimizer.param_groups[0]["lr"], getattr(net, "lambda_align", 0.0))

        if args.counterfactual_every and (epoch + 1) % args.counterfactual_every == 0:
            log_counterfactual(tb, net, device, counterfactual_loaders, epoch,
                               args.iou_threshold)

        # Save best performing network
        # `best_iou` chỉ được cập nhật khi thật sự tốt hơn. Bản cũ đặt nó bên trong nhánh
        # `or (epoch % 10) == 0`, nên mỗi epoch chia hết cho 10 lại hạ mốc xuống một giá trị
        # có thể tệ hơn, khiến các epoch sau đó được lưu như "best" giả.
        is_best = iou > best_iou
        if is_best or epoch == 0 or (epoch % 10) == 0:
            # 4 chữ số thập phân: %0.2f làm tròn khiến nhiều epoch trùng tên và phải grep log
            # mới biết checkpoint nào thật sự tốt nhất.
            # save_checkpoint lưu state_dict + kwargs, bỏ CLIP text tower: ~9 MB thay vì
            # 265 MB mỗi file (utils/checkpoint.py).
            save_checkpoint(
                net, os.path.join(save_folder, "epoch_%02d_iou_%0.4f" % (epoch, iou))
            )
        if is_best:
            best_iou = iou
            logging.info(f"New best IoU {best_iou:.4f} at epoch {epoch}")

        if probe_indices and (epoch in probe_epochs or is_best):
            save_probe_figures(net, val_dataset, probe_indices, figures_dir, epoch, tb=tb)

        if scheduler is not None:
            scheduler.step()

    # Đối chứng lần cuối trên model cuối cùng, kể cả khi lịch --counterfactual-every không rơi
    # đúng epoch cuối: đây là con số đi vào báo cáo.
    if args.counterfactual_every:
        log_counterfactual(tb, net, device, counterfactual_loaders, args.epochs - 1,
                           args.iou_threshold)
    tb.flush()


if __name__ == "__main__":
    run()
