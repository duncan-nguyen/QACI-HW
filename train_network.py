import argparse
import copy
import datetime
import hashlib
import inspect
import json
import logging
import os

import cv2
import numpy as np
import tensorboardX
import torch
import torch.utils.data
from torch import optim

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


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
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Checkpoint để khởi tạo backbone (vd weights/model_grasp_anything). Nạp bằng "
        "strict=False nên các nhánh mới (align/graspability) giữ khởi tạo ngẫu nhiên. "
        "Số layer khớp/không khớp được ghi vào log -- luôn kiểm tra con số đó.",
    )
    parser.add_argument(
        "--gate-lambda",
        type=float,
        default=1.0,
        help="Hệ số gating F' = F ⊙ (1 + λ·A_T). Chỉ dùng bởi grconvnet3_align.",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=128,
        help="Số chiều không gian chiếu chung của alignment token-pixel.",
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
        "--w-agnostic",
        type=float,
        default=1.0,
        help="Trọng số L_agnostic(Q_g, M_union)",
    )
    parser.add_argument(
        "--w-token",
        type=float,
        default=0.0,
        help="Trọng số L_token -- contrastive theo token ở vùng part (Update 3). L_align chỉ "
        "giám sát max_j nên đa dạng token luôn bị phạt; đây là số hạng duy nhất thưởng cho "
        "nó. 0 = tắt. Chỉ có nghĩa khi part_mask thật sự phân biệt part.",
    )
    parser.add_argument(
        "--mask-angle-loss",
        type=int,
        default=1,
        help="Chỉ tính cos/sin/width trong vùng grasp (1/0, Update 1). Target của chúng bằng "
        "0 ở ~97% pixel nền; để dày đặc thì nhánh góc sụp về hằng số. p_loss luôn dày đặc.",
    )
    parser.add_argument(
        "--align-weight-file",
        type=str,
        default=None,
        help="File .npz d_i do script/build_align_weights.py sinh (Update 2). Mặc định dò "
        "<dataset-path>/align_weights.npz; không có thì mọi sample có trọng số 1.",
    )
    parser.add_argument(
        "--single-part-weight",
        type=float,
        default=1.0,
        help="Trọng số L_align cho object chỉ có một part, nơi d_i không xác định (hợp các "
        "part bằng chính nó). Mặc định giữ nguyên 1.0 vì không có bằng chứng để loại.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Số epoch warmup trọng số hai loss phụ",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
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
    parser.add_argument(
        "--val-augment",
        type=int,
        default=0,
        help="Bật random rotate/zoom cho *validation* (1/0). Mặc định 0. Bật lên thì IoU "
        "validation đo trên ảnh xoay/zoom ngẫu nhiên khác nhau mỗi epoch, nhiễu tới ~1pp, "
        "và việc chọn checkpoint theo max của nó là chọn đỉnh nhiễu. Chỉ bật để tái lập "
        "hành vi cũ.",
    )
    parser.add_argument(
        "--align-diag-samples",
        type=int,
        default=64,
        help="Số sample validation dùng cho chẩn đoán token alignment (0 = tắt). Cần chạy "
        "tokenizer nên đắt hơn các thống kê còn lại.",
    )

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
    Lấy prompt / part_mask / union_pos ở phần tử thứ 6 của batch (chỉ Grasp-Anything++ có).
    Model không khai báo `accepts_extras` thì trả dict rỗng -> chữ ký compute_loss cũ giữ nguyên.
    """
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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_backbone(net, path):
    """
    Khởi tạo backbone từ một checkpoint có sẵn (vd `weights/model_grasp_anything`, chính là
    GR-ConvNet3 đã train trên toàn bộ Grasp-Anything -- cùng kiến trúc, cùng phân phối ảnh).

    Nạp `strict=False` để các nhánh mới (`align.*`, `graspability_head`, `text_encoder.*`)
    giữ khởi tạo ngẫu nhiên. Số layer khớp / không khớp được ghi ra log: nếu một lần đổi tên
    layer làm hỏng việc nạp, bạn sẽ âm thầm train from scratch mà không có dấu hiệu nào.
    """
    src = torch.load(path, map_location="cpu", weights_only=False)
    sd = src.state_dict() if hasattr(src, "state_dict") else src

    own = net.state_dict()
    # conv1 khác số kênh vào (RGB vs RGB-D) là trường hợp duy nhất đáng tự động chỉnh.
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape:
            if k == "conv1.weight" and own[k].shape[1] < v.shape[1]:
                sd[k] = v[:, v.shape[1] - own[k].shape[1] :]
                logging.warning(
                    "init_from: cắt %s %s -> %s", k, tuple(v.shape), tuple(sd[k].shape)
                )
            else:
                logging.warning(
                    "init_from: bỏ %s (shape %s != %s)",
                    k,
                    tuple(v.shape),
                    tuple(own[k].shape),
                )
                del sd[k]

    missing, unexpected = net.load_state_dict(sd, strict=False)
    loaded = len(own) - len(missing)
    logging.info(
        "init_from=%s loaded=%d/%d missing=%d unexpected=%d",
        path,
        loaded,
        len(own),
        len(missing),
        len(unexpected),
    )
    logging.info(
        "init_from missing (giữ khởi tạo ngẫu nhiên): %s", sorted(missing)[:20]
    )
    if unexpected:
        logging.warning(
            "init_from unexpected (không dùng đến): %s", sorted(unexpected)[:20]
        )
    if loaded == 0:
        raise SystemExit(
            f"--init-from {path}: không nạp được layer nào. Kiểm tra kiến trúc/tên layer."
        )


def log_align_weights(dataset):
    """
    Ghi ra tóm tắt d_i (Update 2). `mean_d_multi_part` là con số quyết định:

    * ≈ 0   -> part_mask của dataset ở mức OBJECT. `L_align` không thể dạy grounding mức part
              và đang dạy ngược lại; nó sẽ tự teo về 0 nhờ trọng số. Cần đổi nguồn nhãn.
    * ≈ 0.7 -> nhãn phân biệt part tốt; nút thắt nằm ở kiến trúc chứ không phải dữ liệu.
    """
    summary = getattr(dataset, "align_weight_summary", None)
    if not summary:
        return
    if not summary.get("source"):
        logging.warning(
            "Không thấy file trọng số alignment -- mọi sample dùng d_i = 1 (hành vi cũ). "
            "Chạy script/build_align_weights.py để bật Update 2."
        )
        return
    logging.info("align_weights: %s", json.dumps(summary, ensure_ascii=False))
    mean_d = summary.get("mean_d_multi_part")
    # NaN < 0.1 là False, nên trường hợp "không có object nào ≥2 part" tự loại.
    if mean_d is not None and mean_d < 0.1:
        logging.warning(
            "mean_d_multi_part=%.4f: part_mask gần như luôn bằng cả object. L_align sẽ tự "
            "teo về 0. Đây là câu trả lời cho tầng 1 -- cân nhắc chuyển sang giám sát A_T "
            "bằng grasp map của chính part đó.",
            mean_d,
        )


def log_split_identity(dataset, train_indices, val_indices, args):
    """
    Ghi rõ tập validation *là cái gì* -- "Validation size: 1763" không cho biết đó là block
    đuôi cố định của danh sách file hay một mẫu ngẫu nhiên, cũng không cho biết nó có trùng
    scene với tập train hay không, và bản thân nó có được dùng lại làm test hay không.
    """
    digest = hashlib.sha256(",".join(map(str, val_indices)).encode()).hexdigest()[:16]
    logging.info(
        "Split: shuffle=%s split_frac=%s val_is_tail_block=%s val_indices_sha256=%s",
        bool(args.ds_shuffle),
        args.split,
        not args.ds_shuffle,
        digest,
    )
    if not hasattr(dataset, "scene_id"):
        return
    try:
        val_scenes = {dataset.scene_id(i) for i in val_indices}
        train_scenes = {dataset.scene_id(i) for i in train_indices}
    except Exception as exc:  # loader khác không có scene_id
        logging.warning("Không lấy được scene_id để kiểm rò rỉ: %s", exc)
        return
    shared = val_scenes & train_scenes
    logging.info(
        "Split: val_scenes=%d train_scenes=%d shared_scenes=%d (%.2f%% scene validation "
        "cũng có trong train)",
        len(val_scenes),
        len(train_scenes),
        len(shared),
        100.0 * len(shared) / len(val_scenes) if val_scenes else 0.0,
    )
    logging.info(
        "LƯU Ý: tập validation này dùng để *chọn* checkpoint. Đừng báo cáo lại chính nó như "
        "một tập test độc lập."
    )


def validate(net, device, val_data, iou_threshold, align_diag_samples=0):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :param align_diag_samples: số sample đầu dùng cho chẩn đoán token alignment (0 = tắt)
    :return: Successes, Failures and Losses
    """
    net.eval()
    if hasattr(net, "set_collect_stats"):
        net.set_collect_stats(True)

    results = {
        "correct": 0,
        "failed": 0,
        "loss": 0,
        "loss_full": 0,
        "losses": {},
        "stats": {},
        "token_report": None,
    }
    token_hits = {"n": 0, "part": 0.0, "punct": 0.0, "batches": 0}

    ld = len(val_data)

    with torch.no_grad():
        for seen_n, batch in enumerate(val_data):
            # GA++ trả thêm phần tử thứ 6 (prompt / part_mask); các dataset khác trả đúng 5.
            x, y, didx, rot, zoom_factor = batch[:5]
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            extras = batch_extras(net, batch, device)
            lossd = net.compute_loss(xc, yc, **extras)

            loss = lossd["loss"]

            results["loss"] += loss.item() / ld
            results["loss_full"] += float(lossd.get("loss_full", loss)) / ld
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item() / ld
            for sn, sv in lossd.get("stats", {}).items():
                if sn not in results["stats"]:
                    results["stats"][sn] = 0
                results["stats"][sn] += float(sv) / ld

            # Token thắng có phải từ chỉ part không -- metric trực tiếp của luận điểm method.
            # Cần tokenizer nên chỉ chạy trên vài chục sample đầu.
            if (
                seen_n < align_diag_samples
                and "prompts" in extras
                and getattr(net, "use_text", False)
                and hasattr(net, "align_token_report")
            ):
                rep = net.align_token_report(xc, extras["prompts"])
                if rep["n"]:
                    token_hits["n"] += rep["n"]
                    token_hits["part"] += rep["part_frac"] * rep["n"]
                token_hits["punct"] += rep["punct_frac"]
                token_hits["batches"] += 1

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

    if token_hits["batches"]:
        results["token_report"] = {
            "n": token_hits["n"],
            "part_frac": token_hits["part"] / token_hits["n"]
            if token_hits["n"]
            else 0.0,
            "punct_frac": token_hits["punct"] / token_hits["batches"],
        }
    if hasattr(net, "set_collect_stats"):
        net.set_collect_stats(False)

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        "loss": 0,
        "loss_full": 0,
        "losses": {},
        "grad_norm": 0.0,
        "grad_norm_n": 0,
        "steps": 0,
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    # Điều kiện thoát phải kiểm *trước* khi lấy batch: đặt ở sau sẽ nạp thêm batch rồi vứt đi,
    # và `while <=` cho vòng lặp chạy lại thêm một nhịp nữa. Bản cũ vì thế chỉ chạy
    # batches_per_epoch - 1 bước cập nhật nhưng lại chia trung bình cho batches_per_epoch + 1.
    while batch_idx < batches_per_epoch:
        for batch in train_data:
            if batch_idx >= batches_per_epoch:
                break

            x, y = batch[0], batch[1]
            extras = batch_extras(net, batch, device)
            batch_idx += 1

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, **extras)

            loss = lossd["loss"]

            results["loss"] += loss.item()
            results["loss_full"] += float(lossd.get("loss_full", loss))
            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()

            # Batch cuối luôn được đo, nếu không thì với batches_per_epoch < 100 sẽ không có
            # lần đo nào và grad_norm báo 0.0 -- dễ đọc nhầm thành "gradient bằng 0".
            if batch_idx % 100 == 0 or batch_idx == batches_per_epoch:
                # max_norm=inf: clip_grad_norm_ trả về chuẩn tổng mà không cắt gì. Đo thưa vì
                # mỗi lần đọc là một lần đồng bộ device.
                gn = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in net.parameters() if p.grad is not None],
                        float("inf"),
                    )
                )
                results["grad_norm"] += gn
                results["grad_norm_n"] += 1
                logging.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():0.4f}, "
                    f"GradNorm: {gn:0.3f}"
                )

            optimizer.step()
            results["steps"] += 1

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
    results["loss_full"] /= batch_idx
    for l in results["losses"]:
        results["losses"][l] /= batch_idx
    if results["grad_norm_n"]:
        results["grad_norm"] /= results["grad_norm_n"]

    return results


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
    ds_params = inspect.signature(Dataset.__init__).parameters
    if "align_weight_path" in ds_params:
        ds_kwargs["align_weight_path"] = args.align_weight_file
        ds_kwargs["single_part_weight"] = args.single_part_weight
    dataset = Dataset(args.dataset_path, **ds_kwargs)
    logging.info(f"Dataset size is {dataset.length}")
    log_align_weights(dataset)

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info(f"Training size: {len(train_indices)}")
    logging.info(f"Validation size: {len(val_indices)}")
    log_split_identity(dataset, train_indices, val_indices, args)

    # Bản cũ dùng *chung một* dataset cho cả train lẫn validation, nên validation cũng chịu
    # random rotate/zoom: IoU validation nhiễu ~1pp mỗi epoch và chọn checkpoint theo max của
    # nó là chọn đỉnh nhiễu. copy nông để hai instance dùng chung grasp_files/_files_by_object
    # (danh sách ~880k phần tử, không nhân đôi bộ nhớ) mà vẫn tắt được augmentation riêng.
    val_dataset = copy.copy(dataset)
    val_dataset.random_rotate = bool(args.val_augment)
    val_dataset.random_zoom = bool(args.val_augment)
    logging.info(
        "Validation augmentation: %s",
        "ON (nhiễu, chỉ để tái lập hành vi cũ)" if args.val_augment else "OFF",
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
            w_agnostic=args.w_agnostic,
            w_token=args.w_token,
            mask_angle_loss=bool(args.mask_angle_loss),
            gate_lambda=args.gate_lambda,
            proj_dim=args.proj_dim,
        )
    net = network(**net_kwargs)

    if args.init_from:
        load_backbone(net, args.init_from)

    net = net.to(device)
    logging.info("Done")

    # CLIP text encoder bị đóng băng -> lọc ra khỏi optimizer cho gọn.
    params = [p for p in net.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    logging.info(
        "Params: trainable=%d total=%d",
        n_train,
        sum(p.numel() for p in net.parameters()),
    )
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

    best_iou = 0.0
    total_steps = 0
    ckpt_log = os.path.join(save_folder, "checkpoints.jsonl")
    for epoch in range(args.epochs):
        logging.info(f"Beginning Epoch {epoch:02d}")
        warmup = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))
        if hasattr(net, "set_loss_warmup"):
            # L_align/L_agnostic vào dần: A_T lúc đầu ngẫu nhiên, ép mạnh sẽ kéo hỏng nhánh grasp.
            net.set_loss_warmup(warmup)
        train_results = train(
            epoch,
            net,
            device,
            train_data,
            optimizer,
            args.batches_per_epoch,
            vis=args.vis,
        )
        total_steps += train_results["steps"]

        # Log training losses to tensorboard
        tb.add_scalar("loss/train_loss", train_results["loss"], epoch)
        tb.add_scalar("loss/train_loss_full", train_results["loss_full"], epoch)
        tb.add_scalar("grad_norm", train_results["grad_norm"], epoch)
        tb.add_scalar("warmup", warmup, epoch)
        tb.add_scalar("optimizer_steps", total_steps, epoch)
        for n, l in train_results["losses"].items():
            tb.add_scalar("train_loss/" + n, l, epoch)

        # Breakdown phải nằm trong *log text*, không chỉ tensorboard: từ train.log của bản cũ
        # không thể biết align_loss có giảm hay không, mà nó chiếm ~63% tổng loss.
        # `loss_full` là loss ở trọng số đầy đủ -- đại lượng duy nhất so sánh được xuyên
        # warmup, vì `loss` bị nhân warmup ở các epoch đầu.
        logging.info(
            "epoch %d train loss=%.4f loss_full=%.4f grad_norm=%.3f steps=%d | %s",
            epoch,
            train_results["loss"],
            train_results["loss_full"],
            train_results["grad_norm"],
            train_results["steps"],
            " ".join(
                f"{n}={l:.4f}" for n, l in sorted(train_results["losses"].items())
            ),
        )

        # Run Validation
        logging.info("Validating...")
        test_results = validate(
            net,
            device,
            val_data,
            args.iou_threshold,
            align_diag_samples=args.align_diag_samples,
        )
        logging.info(
            "%d/%d = %f"
            % (
                test_results["correct"],
                test_results["correct"] + test_results["failed"],
                test_results["correct"]
                / (test_results["correct"] + test_results["failed"]),
            )
        )
        logging.info(
            "epoch %d val   loss=%.4f loss_full=%.4f | %s",
            epoch,
            test_results["loss"],
            test_results["loss_full"],
            " ".join(f"{n}={l:.4f}" for n, l in sorted(test_results["losses"].items())),
        )
        if test_results["stats"]:
            # fg ≈ bg nghĩa là gate vô dụng dù loss có giảm; argmax_last cao nghĩa là token
            # thắng luôn là token cuối câu, tức alignment đã thoái hoá về mức câu.
            logging.info(
                "epoch %d align | %s",
                epoch,
                " ".join(
                    f"{n}={v:.4f}" for n, v in sorted(test_results["stats"].items())
                ),
            )
            for n, v in test_results["stats"].items():
                tb.add_scalar("align/" + n, v, epoch)
        if test_results["token_report"]:
            rep = test_results["token_report"]
            logging.info(
                "epoch %d token | argmax_is_part=%.4f argmax_is_punct=%.4f (n=%d)",
                epoch,
                rep["part_frac"],
                rep["punct_frac"],
                rep["n"],
            )
            tb.add_scalar("align/argmax_is_part", rep["part_frac"], epoch)
            tb.add_scalar("align/argmax_is_punct", rep["punct_frac"], epoch)
        if hasattr(net, "text_encoder") and hasattr(
            net.text_encoder, "truncation_report"
        ):
            n_p, n_t, rate = net.text_encoder.truncation_report()
            if n_t:
                logging.warning("prompt bị cắt: %d/%d (%.2f%%)", n_t, n_p, 100 * rate)

        # Log validation results to tensorbaord
        tb.add_scalar(
            "loss/IOU",
            test_results["correct"]
            / (test_results["correct"] + test_results["failed"]),
            epoch,
        )
        tb.add_scalar("loss/val_loss", test_results["loss"], epoch)
        tb.add_scalar("loss/val_loss_full", test_results["loss_full"], epoch)
        for n, l in test_results["losses"].items():
            tb.add_scalar("val_loss/" + n, l, epoch)

        # Save best performing network
        iou = test_results["correct"] / (
            test_results["correct"] + test_results["failed"]
        )
        # `best_iou` chỉ được cập nhật khi thật sự tốt hơn. Bản cũ gán trong cùng nhánh với
        # điều kiện lưu định kỳ, nên cứ epoch chia hết cho 10 là ngưỡng bị kéo tụt xuống giá
        # trị hiện tại và một loạt checkpoint không phải "best" vẫn được lưu.
        is_best = iou > best_iou
        path = None
        if is_best or epoch == 0 or (epoch % 10) == 0:
            path = os.path.join(save_folder, "epoch_%02d_iou_%0.2f" % (epoch, iou))
            torch.save(net, path)
        if is_best:
            best_iou = iou

        # Tên file làm tròn tới 2 chữ số nên không phân biệt được 0.3313 với 0.3296. Ghi kèm
        # metadata máy đọc để không phải regex lại log khi chọn checkpoint.
        record = {
            "epoch": epoch,
            "iou": iou,
            "is_best": is_best,
            "path": path,
            "val_loss": test_results["loss"],
            "val_loss_full": test_results["loss_full"],
            "train_loss": train_results["loss"],
            "train_loss_full": train_results["loss_full"],
            "grad_norm": train_results["grad_norm"],
            "warmup": warmup,
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": total_steps,
            "val_losses": test_results["losses"],
            "align": test_results["stats"],
            "token_report": test_results["token_report"],
        }
        if path:
            record["sha256"] = _sha256(path)
        with open(ckpt_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        if scheduler is not None:
            scheduler.step()
            tb.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)


if __name__ == "__main__":
    run()
