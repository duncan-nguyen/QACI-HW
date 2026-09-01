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
   1. [Đọc log](#đọc-log)

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
                   S_pj = cos(W_v F_p, W_t t_j) · exp(τ)
                   α_pj = softmax_j(S_pj / τ_t)        token quan trọng tại pixel p
                   A_T(p) = σ( Σ_j α_pj S_pj )         vùng quan trọng
                   R_T(p) = Σ_j α_pj W_r t_j           text feature của vùng đó
                                                    │
              F' = F + φ([F, F ⊙ A_T, R_T]) ──► conv4..conv6 ──► pos/cos/sin/width

L = L_grasp + w_align · [BCE + Dice](A_T, part_mask)
```

Soft-select token tại **từng pixel** thay vì lấy `max`, và đưa cả *vùng* lẫn *text feature của
vùng* vào decoder. Fusion đặt sau `res5` nên cả 4 head đều được condition; conv cuối của φ khởi
tạo bằng 0 nên ở bước 0 thì `F' = F` đúng bằng baseline, nhánh ngôn ngữ đi vào dần. `A_T`
supervise ngay ở 56×56, target hạ xuống bằng avg-pool chứ không upsample logits lên 224.
Token SOT/EOT/padding **và dấu câu** bị loại khỏi candidate.

Cùng một file, bốn arm của bảng ablation bật/tắt bằng cờ:

| arm | `--use-text` | `--align-mode` | `--region-text` | `--fusion` |
|---|---|---|---|---|
| GR-ConvNet (no text) | 0 | – | – | – |
| Hard-max (V1) | 1 | `hard` | 0 | `gate` |
| Soft alignment | 1 | `soft` | 0 | `residual` |
| **V2 full** | 1 | `soft` | 1 | `residual` |

`--align-stage conv4` chuyển chỗ tính `A_T` từ 56×56 lên 113×113 (mịn hơn cho part nhỏ, đổi
lại conv1..res5 không còn được prompt điều kiện).

Chi tiết: [inference/models/grconvnet3_align.py](inference/models/grconvnet3_align.py),
[utils/data/grasp_anything_pp_data.py](utils/data/grasp_anything_pp_data.py).

Checkpoint lưu bằng [utils/checkpoint.py](utils/checkpoint.py): state_dict + kwargs, bỏ CLIP
text tower — **9 MB** thay vì 265 MB. `load_network()` đọc được cả hai định dạng, kể cả
checkpoint V1 (nó tự khôi phục cấu hình `hard`/`gate` để số đo lại đúng là số của model đó).

Kiểm tra nhanh trước khi tốn GPU, không cần dataset:

```bash
python script/smoke_test_align.py     # 4 arm, checkpoint round-trip, overfit một batch
```

### Dữ liệu

GA++ chỉ chứa phần ngôn ngữ + label; ảnh nằm ở repo Grasp-Anything gốc. Giải nén trọn bộ là
**~830 GB** (riêng `part_mask` đã 764 GB), nên `script/build_ga_pp_subset.py` dựng subset:

- `--pack-masks` — lưu mask dạng bit đóng gói, 21 KB thay vì 173 KB, không mất mát
- `--images-from-zip` — tải archive ảnh 65 GB rồi trích tại chỗ (rẻ hơn HTTP từng ảnh khi
  trên ~40k scene); dưới ngưỡng đó thì đọc từng ảnh qua HTTP range, không cần tải 65 GB

| scene | sample | % GA++ | đĩa (packed) |
|---|---|---|---|
| 10k | 44k | 1% | 1,8 GB |
| **100k** (mặc định) | **445k** | **10%** | **18 GB** |
| 300k | 1,34M | 30% | 55 GB |
| 994k (full) | 4,41M | 100% | 183 GB |

Split seen/unseen dựng theo protocol paper LGD §5.1 (70% category theo tần suất vào Base):
`split/build_grasp_anything_pp.py`.

Schema đã verify trực tiếp trên archive — xem [notebooks/ga_pp_schema.ipynb](notebooks/ga_pp_schema.ipynb),
đọc bằng HTTP range nên chỉ tốn ~5 MB.

### Chạy

Cấu hình mặc định: **một GPU · 100.000 scene (~445k sample, 10% GA++) · 50 epoch**.

```bash
# 1. tải + dựng dataset, split, và kiểm dữ liệu (bước lâu nhất, có cache)
SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

# 2. đo num_workers trên chính máy đó -- GPU gần như không bao giờ là chỗ nghẽn
#    (xem mục "Tối ưu tốc độ train" bên dưới: một sample đã từ 36,5 ms xuống 9,6 ms)
python script/bench_loader.py --dataset-path data/grasp-anything-pp-full \
    --split-path split/grasp-anything-pp --workers 8,16,32,48,64

# 3. train + eval + đối chứng + gom kết quả (build/split đã cache nên bỏ qua)
NUM_WORKERS=<số đo được> bash script/run_paper_setting.sh
```

Giữ nguyên theo paper: split Base/New 70/30 theo category · success khi IoU ≥ 0.25 **và** lệch
góc ≤ 30° · báo cáo Seen / Unseen / harmonic mean H. Khác paper — vì chỉ có một GPU: **10% dữ
liệu** (paper dùng cả 4,41M sample) và **50 epoch** (paper 100, Fig 6). Batch size, optimizer,
learning rate thì paper không công bố trong bản chính. Mọi con số này ghi trong header script
và trong `config.txt` của mỗi lần chạy — báo cáo phải nêu rõ, đừng đặt cạnh Table 2 như thể
cùng điều kiện.

Lưu ý: repo này định nghĩa "epoch" = `BATCHES_PER_EPOCH` batch, không phải một lượt qua hết dữ
liệu. 50 × 2000 × 64 = **6,4M lượt sample** ≈ 14 lượt qua 445k sample của subset.

Bảng ablation §6 chạy **tuần tự trên một GPU** (`train_network.py` không có DDP; nhét bốn tiến
trình vào một GPU chỉ khiến chúng giành CPU của nhau, tổng thời gian không đổi mà không arm nào
xong sớm):

```bash
bash script/run_ablation.sh                      # 4 arm, tuần tự -- ≈ 4 lần một run đơn
GPU=1 bash script/run_ablation.sh                # chọn GPU khác
ARMS="soft full" bash script/run_ablation.sh     # chạy lại vài arm
RESUME=1 bash script/run_ablation.sh             # đứt giữa chừng thì chạy tiếp
```

Ngân sách train của mọi arm giống hệt nhau — điều kiện bắt buộc để bảng có nghĩa; script in con
số đó ra đầu và cuối.

### Kết quả

Mỗi lần chạy tự gom vào `results/<timestamp>_<description>/`:

```
summary.md                  bảng Seen/Unseen/H cạnh Table 2, bảng đối chứng prompt, token/hình
config.txt                  đúng cấu hình đã chạy
dataset-check/              thống kê part_mask + prompt, lưới ảnh kiểm tra bằng mắt
train.log  eval_seen.log  eval_unseen.log  audit_text_reliance.{log,json}
tensorboard/
figures/                    diagnostic_*, prompt_grid, failures_*, prediction_*, tokens_*
```

Bốn loại hình, theo thứ tự nên nhìn:

- `prompt_grid.png` — **hình quan trọng nhất**: cùng một ảnh, đổi prompt. Mỗi hàng là
  `part_mask` GT · `A_T` · token mạnh nhất kèm attention mass · Q · grasp. Ba hàng giống hệt
  nhau nghĩa là model đang bỏ qua ngôn ngữ, bất kể `align_loss` thấp đến đâu.
- `diagnostic_{seen,unseen}_*.png` — một hàng đầy đủ cho một sample: RGB · `part_mask` GT ·
  `A_T` · **bản đồ sai số** (xanh TP / đỏ FP / vàng FN) · Q · GT+dự đoán · token mạnh nhất.
- `failures_{seen,unseen}.png` — gallery lỗi bốn nhóm (align đúng/grasp đúng · align đúng/grasp
  sai · align sai/grasp sai · không phát hiện grasp). Nhóm nào đông cho biết lỗi nằm ở
  grounding hay ở grasp decoder — hai hướng sửa khác hẳn nhau.
- `prediction_*.png`, `tokens_*.png`, `parts_same_object.png` — như trước.

Vẽ lại mà không train lại:
`python script/export_results.py --checkpoint <ckpt> --dataset-path <data> --out <dir>`.

`results/` nằm trong `.gitignore`.

### Đọc log

Ba câu hỏi cần trả lời khi debug nhánh ngôn ngữ, và chỗ trả lời từng câu.

**Trước khi train — dữ liệu có dạy được không?**

```bash
python script/check_dataset.py --data-dir data/grasp-anything-pp-full     --split-path split/grasp-anything-pp --out results/dataset-check
```

In ra: tỉ lệ foreground của `part_mask`, IoU giữa các part **cùng object**, tỉ lệ part có mask
trùng nhau, số token còn lại mỗi prompt sau khi lọc, và một lưới ảnh + prompt + mask + grasp GT.
Cảnh báo quan trọng: nếu phần lớn cặp part cùng object có IoU > 0.9 thì supervision thực tế ở
mức *object* chứ không phải mức part — `L_align` không thể dạy grounding part-level, và Δ trong
bảng đối chứng sẽ nhỏ dù model không có lỗi gì. Báo cáo sâu hơn kèm ví dụ:
`script/diagnose_part_masks.py`. Bước này đã nằm sẵn trong `run_paper_setting.sh` (bước 3/7).

**Trong lúc train — TensorBoard** (`--diag-interval`, `--probe-samples`, `--counterfactual-every`)

| nhóm tag | đọc thế nào |
|---|---|
| `loss/{train,val}/{total,grasp,align_bce,align_dice}` | tách riêng vì warmup làm *tổng* đi lên trong lúc từng thành phần đang đi xuống |
| `weight/lambda_align`, `optimizer/lr` | giá trị λ *thực tế* của epoch đó (đã nhân warmup) |
| `align/{iou,dice,score_margin,foreground_score,background_score,predicted_area}` | `score_margin = mean(A_T ∣ trong mask) − mean(A_T ∣ ngoài mask)`; gần 0 = attention chưa phân biệt fg/bg, kể cả khi BCE đã nhỏ |
| `token/{attention_entropy,top1_mass,top3_mass,valid_count,temperature,logit_scale}` | entropy ≈ 0 là collapse vào một token; ≈ `token/max_entropy` là không token nào quan trọng; `top1_mass ≈ 1` ngay từ đầu = soft attention đang chạy y hệt hard max; `logit_scale` chạm 100 là temperature có vấn đề |
| `token/winning` (text) | bảng "token nào thắng bao nhiêu % pixel" trên cả tập val. Top toàn danh từ object (`apple`, `mug`) = đang định vị *vật* chứ không phải *part*. `token/punctuation_mass` phải bằng **0** |
| `fusion/{feature_norm,residual_norm,residual_ratio}` | `r_F = ‖F'−F‖/‖F‖`: ≈ 0 là fusion gần như không tác động (khởi tạo đúng bằng 0, phải thấy nó *lớn dần*); ≫ 1 là fusion lấn át feature thị giác |
| `gradient/{visual_encoder,text_projection,alignment,fusion,grasp_decoder}` | chuẩn L2 gradient theo khối, ghi mỗi `--diag-interval` batch |
| `gradient/visual_from_{grasp,align,ratio}` | gradient của `L_grasp` và `L_align` **riêng rẽ** lên visual encoder, trên cùng một batch. `ratio` hàng chục lần trở lên = giảm `--w-align` hoặc kéo dài `--warmup-epochs` |
| `counterfactual/{normal,shuffled,fixed}_success`, `counterfactual/prompt_drop` | `Δ_prompt = S_normal − S_shuffled`. Gần 0 thì **không kết luận được** nhánh ngôn ngữ có ích |
| `probe/*` (hình) | probe set **cố định** 8 sample, vẽ lại ở epoch 0, 1, 3, 10, 25 và mỗi epoch tốt nhất — thấy được attention bắt đầu học lúc nào, có collapse sau warmup không |
| `step/token/*`, `step/fusion/*` | cùng chỉ số nhưng đo trên batch train theo *step*, không phải theo epoch |

**Sau khi train — đối chứng, quan trọng hơn mọi đường loss**

```bash
python script/audit_text_reliance.py --checkpoint logs/.../epoch_67_iou_0.3313     --dataset-path data/grasp-anything-pp-full --split-path split/grasp-anything-pp     --n-samples 500 --figures results/audit
```

Bốn điều kiện trên cùng một sample: prompt thật · prompt của object khác (`shuffled`) · một câu
chung cho mọi ảnh (`fixed`) · prompt của part khác cùng object (`other_part`). Đọc:
`Δ_shuffled ≈ 0` là grasp không phụ thuộc prompt; `Δ_shuffled` lớn nhưng `Δ_other_part ≈ 0` là
model mới nhận ra *vật*, chưa phân biệt *part*. `evaluate.py --shuffle-prompts` làm phiên bản
gọn của phép này cho một split.

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

### Sửa sau khi soi kết quả run 260830 (`results/`)

Những lỗi làm *sai số liệu* chứ không làm chương trình chết, phát hiện khi đọc lại run GA++
200k. Chi tiết ở docstring từng chỗ:

| | Lỗi | Sửa |
|---|---|---|
| A1 | `split/build_grasp_anything_pp.py` lấy 70% category *nhiều nhất* cho Base, để lại 956/882.214 sample (0,1%) cho New; 295 scene nằm ở cả hai tập | `--split-mode balanced` (mặc định) chia theo **số sample**, category vẫn rời nhau; `--drop-shared-from seen` (mặc định) bỏ ảnh giao nhau khỏi train; cảnh báo khi một tập < 5% |
| A2 | `evaluate.py` và `train_network.py` cắt index bằng hai đoạn code lặp lại nên "test set" trùng khít tập chọn checkpoint | Dùng chung `utils/data/index_split.py`; thêm `--test-split` (train) và `--subset {train,val,test,dev,all}` (evaluate) |
| A3 | Validation chạy trên dataset bật `random_rotate`/`random_zoom`; `zoom ~ U(0.5, 1)` cắt mất grasp mà `get_gtbb` không loại → target rỗng, mẫu tự động fail | `GraspDatasetBase.eval_view()` — bản sao nông với augmentation tắt |
| A4 | `best_iou` bị nhánh lưu định kỳ `epoch % 10` hạ xuống → lưu "best" giả | Chỉ cập nhật khi thật sự tốt hơn; tên file `%0.4f` thay vì `%0.2f` |
| A5 | `evaluate.py` không gọi `net.eval()` (model có dropout 0,1 + BatchNorm) | `.eval()` ngay khi load |
| A6 | Nhãn width chuẩn hoá `/(output_size/2)` nhưng decode `×150` — hằng 150 là `output_size/2` của GR-ConvNet gốc 300×300, với input 224 làm grasp dài hơn thật 1,34× | `post_process_output` suy `width_scale` từ kích thước ảnh (`×112` cho 224, vẫn `×150` cho 300) |
| A7 | Vòng train nạp thừa một batch mỗi epoch rồi vứt → 1.999 step cho `--batches-per-epoch 2000` | Kiểm tra điều kiện dừng ở đầu vòng trong |
| A8 | `GraspRectangle.iou` đánh index âm vào canvas → wrap-around, IoU sai ở grasp sát mép | Dịch cả hai polygon về gốc 0 |
| A9 | `script/build_ga_pp_subset.py` ghi ảnh/label không atomic → đứt giữa chừng để lại file cụt mà lần chạy sau bỏ qua | `write_atomic()` qua file tạm + `os.replace` |
| A10 | `script/diagnose_part_masks.py` (đo `part_mask` ở mức part hay mức object) bị gỡ khỏi branch này | Khôi phục từ branch `text-image-aware` |

### Tối ưu tốc độ train

Chỗ nghẽn là **CPU dataloader**, không phải GPU (GR-ConvNet chỉ 1,9M tham số). Một sample GA++
đi từ **20,5 ms xuống 4,7 ms** trên một core — nhanh **4,3×**. Số đo lấy trên 16 core với
`cv2.setNumThreads(0)`, ảnh 416×416×3 → 224×224, trung bình 3 lượt đo xen kẽ hai bản code.

| | Chỗ | Trước | Sau | Sai khác so với bản cũ |
|---|---|---|---|---|
| P1 | `Image.resize` — `skimage.transform.resize` | 12,5 ms | **0,40 ms** | ≤ 1/255 trên 0,02-0,4% pixel (làm tròn float32). Tái tạo đúng hai bước của skimage bằng cv2: lọc Gauss `σ=(tỉ_lệ−1)/2` rồi bilinear. Hai chi tiết dễ sai: (a) tên chế độ biên không trùng nhau — skimage `reflect`→`BORDER_REFLECT_101`, `symmetric`→`BORDER_REFLECT`; (b) khi **phóng to**, hàng/cột ngoài cùng có lấy mẫu ngoài biên mà `cv2.resize` luôn nhân bản biên → lệch tới **13/255** ở khung viền, nên đường phóng to phải dùng `warpAffine` có `borderMode`. Đừng thay bằng `INTER_AREA`: lệch 0,74/255 mỗi pixel **và** chậm hơn (0,69 ms) |
| P2 | `Image.rotate` ở góc bội 90° | 2,32 ms | **0,22 ms** | ≤ 1/255, theo hướng tốt hơn: `np.rot90` là hoán vị chỉ số nên *chính xác tuyệt đối*, còn skimage nội suy rồi làm tròn. Góc bất kỳ → `cv2.warpAffine` (≤ 1/255) |
| P3 | `Image.from_file` — `imageio.imread` | 1,94 ms | **1,51 ms** | Không có (đối chiếu trên JPEG: giống hệt). `cv2.imread` trả `None` khi lỗi thay vì ném exception, nên bọc thêm `FileNotFoundError` — không thì train âm thầm trên ảnh đen |
| P4 | `GraspRectangles.draw` — `skimage.draw.polygon` từng rect | 2,24 ms | **0,25 ms** | **2,2% diện tích `pos`.** Hình học được vector hoá (chính xác), nhưng `cv2.fillConvexPoly` tô cả pixel mà cạnh đi qua còn skimage chỉ lấy pixel có *tâm* bên trong; thu rect vào 0,5 px để bù, còn lệch 2,2% diện tích và 0,2% pixel đổi rect thắng ở `ang`/`width`. **Đây là thay đổi duy nhất đụng vào nhãn train** |
| P5 | 5-7 lần `.item()` mỗi step trong vòng train | mỗi lần chặn CPU tới khi GPU chạy xong hàng đợi | `LossMeter` cộng dồn trên GPU (float64), đồng bộ 1 lần/epoch | Không có |
| P6 | Tokenize CLIP trong tiến trình chính | 6,3 ms/batch nằm trên đường găng | `PromptTokenizer` chạy trong DataLoader worker (0,042 ms/sample) | Không có — cùng tokenizer, cùng `max_length` |
| P7 | `validate()` cố định `batch_size=1` | forward 1 ảnh mỗi lần | `--val-batch-size` (mặc định 32); post-process và IoU vẫn chạy từng sample | Không có — đã kiểm ở bs 1/5/N: `correct`/`failed` giống hệt, loss lệch < 1e-8 |
| P8 | Không có AMP / channels_last / `cudnn.benchmark` | fp32, NCHW | `--amp auto` (bf16 nếu GPU hỗ trợ), `--channels-last auto`, `--cudnn-benchmark`, `--tf32` | **Có, đáng kể.** `--amp off` để quay về fp32 thuần |
| P9 | `rot` bị `default_collate` ép về float32 khi phần tử đầu batch là `0` (int) | mất 8 chữ số của các góc khác trong batch | `float(random.choice(...))` trong `GraspDatasetBase.__getitem__` | Sửa lỗi. Không lộ ra khi `batch_size=1`, nhưng làm `validate` dựng GT bằng góc khác góc đã xoay ảnh |

Sau khi sửa, phần tốn nhất còn lại của một sample là giải mã JPEG (1,5 ms) — muốn giảm nữa thì
phải đổi định dạng dữ liệu trên đĩa, không phải đổi code.

Kiểm tra lại các phép biến đổi này (không cần dataset, không cần GPU):

```bash
python script/smoke_test_loader.py
```

Muốn tái lập đúng số liệu của các run cũ: đặt `AMP=off` và hoàn nguyên P4 (hunk `draw` trong
`utils/dataset_processing/grasp.py`). Các mục còn lại chỉ khác ở mức làm tròn, hoặc chính xác
hơn bản cũ.

## Acknowledgement
Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
