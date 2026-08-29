# Grasp-Anything
This is the repository of the paper "Grasp-Anything: Large-scale Grasp Dataset from Foundation Models"

> **This fork adds language-driven grasp detection on Grasp-Anything++.** The added method,
> data tooling and run scripts are documented in
> [Language-driven grasping trên GA++](#language-driven-grasping-trên-ga) (Vietnamese);
> the full design note is [idea.md](idea.md).

## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)
   1. [Language-driven grasping trên GA++](#language-driven-grasping-trên-ga)

## Installation
- Create a virtual environment
```bash
$ conda create -n granything python=3.9
$ conda activate granything
```

- Install pytorch
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Datasets
Our dataset can be accessed via [this link](https://airvlab.github.io/grasp-anything/docs/download/).

## Training
We use GR-ConvNet as our default deep network. To train GR-ConvNet on different datasets, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0
```
For example, if you want to train a GR-ConvNet on Cornell, use the following command:
```bash
$ python train_network.py --dataset cornell --dataset-path data/cornell --description training_cornell --use-depth 0
```
We also provide training for other baselines, you can use the following command:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0 --network <baseline_name>
```
For instance, if you want to train GG-CNN on Cornell, use the following command:
```bash
python train_network.py --dataset cornell --dataset-path data/cornell/ --description training_ggcnn_on_cornell --use-depth 0 --network ggcnn
```

## Testing
For testing procedure, we can apply the similar commands to test different baselines on different datasets:
```bash
python evaluate.py --network <path_to_pretrained_network> --dataset <dataset> --dataset-path data/<dataset> --iou-eval
```
Important note: `<path_to_pretrained_network>` is the path to the pretrained model obtained by training procedure. Usually, the pretrained models obtained by training are stored at `logs/<timstamp>_<training_description>`. You can select the desired pretrained model to evaluate. We do not have to specify neural architecture as the codebase will automatically detect the neural architecture. Pretrained weights are available at [this link](https://drive.google.com/file/d/1OXVFXqv0rgxiVLz89tnSj0Xb-20ZJ4fH/view?usp=sharing).


## Language-driven grasping trên GA++

Grasp-Anything++ gắn cho mỗi *part* của mỗi object một câu lệnh gắp ("Grasp the mug at its
handle"). Nhánh này thêm một model discriminative, một forward pass, trong đó việc *định vị
part theo ngôn ngữ* được supervise **tường minh** bằng `part_mask` của GA++ thay vì để fusion
module tự học ngầm.

### Method — `grconvnet3_align`

```
prompt ──► CLIP text (đóng băng, per-token) ──► t_1..t_L
                                                    │
ảnh ────► conv1..res5 ──► F (56×56×128) ────────────┤
                                                    ▼
                       A_T = σ( max_j cos(W_v F_xy, W_t t_j) · exp(τ) )
                                                    │
                       F' = F ⊙ (1 + λ·A_T) ──► conv4..conv6 ──► pos/cos/sin/width
                                                    │
                       Q_g = head_g(F)   (graspability, không điều kiện text)

L = L_grasp + w_agnostic·BCE(Q_g, M_∪) + w_align·[BCE + Dice](A_T, part_mask)
```

Gate đặt sau `res5` nên cả 4 head đều được condition. `A_T` và `Q_g` supervise ngay ở 56×56;
target hạ xuống bằng avg-pool chứ không upsample logits lên 224. `use_text=False` biến đúng
file này thành baseline GR-ConvNet không ngôn ngữ, dùng chung ngân sách train.

Chi tiết: [inference/models/grconvnet3_align.py](inference/models/grconvnet3_align.py),
[utils/data/grasp_anything_pp_data.py](utils/data/grasp_anything_pp_data.py).

### Dữ liệu

GA++ chỉ chứa phần ngôn ngữ + label; ảnh nằm ở repo Grasp-Anything gốc. Giải nén trọn bộ là
**~830 GB** (riêng `part_mask` đã 764 GB), nên `script/build_ga_pp_subset.py` dựng subset:

- `--pack-masks` — lưu mask dạng bit đóng gói, 21 KB thay vì 173 KB, không mất mát
- `--images-from-zip` — tải archive ảnh 65 GB rồi trích tại chỗ (rẻ hơn HTTP từng ảnh khi
  trên ~40k scene); dưới ngưỡng đó thì đọc từng ảnh qua HTTP range, không cần tải 65 GB

| scene | sample | % GA++ | đĩa (packed) |
|---|---|---|---|
| 10k | 44k | 1% | 1,8 GB |
| 100k | 445k | 10% | 18 GB |
| 300k | 1,34M | 30% | 55 GB |
| 994k (full) | 4,41M | 100% | 183 GB |

Split seen/unseen dựng theo protocol paper LGD §5.1 (70% category theo tần suất vào Base):
`split/build_grasp_anything_pp.py`.

Schema đã verify trực tiếp trên archive — xem [notebooks/ga_pp_schema.ipynb](notebooks/ga_pp_schema.ipynb),
đọc bằng HTTP range nên chỉ tốn ~5 MB.

### Chạy

```bash
# 1. tải + dựng dataset và split (bước lâu nhất)
SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

# 2. đo num_workers trên chính máy đó -- GPU gần như không bao giờ là chỗ nghẽn
python script/bench_loader.py --dataset-path data/grasp-anything-pp-full \
    --split-path split/grasp-anything-pp --workers 8,16,32,48,64

# 3. train + eval + gom kết quả (build/split đã cache nên bỏ qua)
NUM_WORKERS=<số đo được> bash script/run_paper_setting.sh
```

Mặc định bám những gì paper nêu rõ: GA++ đầy đủ · split Base/New 70/30 · 100 epoch ·
success khi IoU ≥ 0.25 **và** lệch góc ≤ 30° · báo cáo Seen / Unseen / harmonic mean H.
Batch size, optimizer, learning rate thì paper không công bố trong bản chính — giá trị dùng ở
đây ghi trong header script và trong `config.txt` của mỗi lần chạy.

Lưu ý: repo này định nghĩa "epoch" = `BATCHES_PER_EPOCH` batch, không phải một lượt qua hết dữ
liệu. 100 × 2000 × 64 = 12,8M lượt sample ≈ 2,9 lượt qua 4,41M.

Có nhiều GPU thì chạy trọn bảng ablation song song (`train_network.py` không có DDP, một run
chỉ dùng một GPU):

```bash
bash script/run_ablation_4gpu.sh                    # 4 arm trên 4 GPU
GPUS="0 0 0 0" bash script/run_ablation_4gpu.sh     # 4 arm trên 1 GPU
```

### Kết quả

Mỗi lần chạy tự gom vào `results/<timestamp>_<description>/`:

```
summary.md          bảng Seen/Unseen/H cạnh Table 2 của paper, + token mạnh nhất từng hình
config.txt          đúng cấu hình đã chạy
train.log  eval_seen.log  eval_unseen.log
tensorboard/
figures/            prediction_*.png, tokens_*.png, parts_same_object.png, prompts_free_form.png
```

Hai loại hình:

- `prediction_{seen,unseen}_*.png` — grasp dự đoán (đỏ) cạnh ground truth (lục), kèm bản đồ Q,
  góc và `A_T`. Tiêu đề ghi IoU cao nhất và đạt/trượt theo đúng metric của paper.
- `tokens_{seen,unseen}_*.png` — bản đồ alignment của **từng token** trong prompt cạnh
  `part_mask` ground truth, cho thấy nhánh ngôn ngữ có thật sự chọn đúng vùng hay không. Vẽ lại mà không
train lại: `python script/export_results.py --checkpoint <ckpt> --dataset-path <data> --out <dir>`.

`results/` nằm trong `.gitignore`.

### Notebook

- [notebooks/train_ga_pp.ipynb](notebooks/train_ga_pp.ipynb) — clone, dựng subset, train, eval,
  visualize alignment. Chạy được trên Colab.
- [notebooks/ga_pp_schema.ipynb](notebooks/ga_pp_schema.ipynb) — kiểm tra schema GA++ trực tiếp
  trên HuggingFace, không cần tải dữ liệu.

### Sửa trong fork (so với upstream)

Những lỗi có sẵn khiến codebase không chạy được trên môi trường hiện đại:

- `np.int` ×5 / `np.float` ×1 trong `utils/dataset_processing/grasp.py` — numpy ≥ 1.24 đã gỡ
- `torch.load(..., weights_only=False)` ở `evaluate.py` và `inference/grasp_generator.py` —
  torch ≥ 2.6 mặc định `weights_only=True`
- `grasp_generator.load_model()` dùng `self.device` trước khi gán
- `validate()` truyền `rot`/`zoom` dạng tensor vào `get_gtbb`, `np.cos(tensor)` làm hỏng ma trận
  xoay (loader GA++ ép `float()`; cornell/jacquard/grasp-anything vẫn còn lỗi này)

Thêm vào: `--split-path`, `--lr`, `--lr-schedule` cho `train_network.py`; DataLoader dùng
`persistent_workers`/`prefetch`/`pin_memory`; loader GA++ đổi thứ tự augmentation sang
`crop → resize → rotate` (nhanh 3,2×, tương đương về hình học với góc bội số 90°).


## Acknowledgement
Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
