# Grasp-Aware Text-Visual Alignment

Language-driven grasp detection trên Grasp-Anything++, chỉ dùng dữ liệu có sẵn —
không sinh ảnh, không LLM-generate prompt, không synthetic data.

## 0. Dữ liệu

GA++ **không phải dataset độc lập**: nó chỉ chứa phần ngôn ngữ + label, ảnh nằm ở repo
Grasp-Anything gốc. Cả hai repo trên HuggingFace đều **không gate** (`gated: false`) —
tải anonymous được, đã verify bằng HTTP HEAD.

| Repo | File | Size | Dùng làm gì |
|---|---|---|---|
| `airvlab/Grasp-Anything` | `image_part_aa` + `image_part_ab` | 65.0 GB | ảnh 416×416 |
| | `scene_description.zip` | 0.34 GB | `(caption, [object])`, để dựng split |
| `airvlab/Grasp-Anything-pp` | `grasp_instructions.zip` | 1.54 GB | prompt (.pkl) |
| | `grasp_label_positive.zip` | 3.95 GB | grasp **part-level** (.pt) |
| | `part_mask.zip` | 4.79 GB | mask **part-level** (.npy) |
| | `grasp_label_negative.zip` | 5.12 GB | negative grasp (.pt) |

Script: `script/download_grasp_anything_pp.sh` (`--check` để verify layout).

> **Va chạm tên:** cả hai repo đều có `grasp_label_positive.zip` — base là object-level,
> GA++ là part-level. Không trộn. Script chỉ lấy bản GA++.

Tác giả yêu cầu điền form + đồng ý MIT trước khi dùng:
https://airvlab.github.io/grasp-anything/docs/download/

### Schema — đã verify trực tiếp trên archive (`notebooks/ga_pp_schema.ipynb`)

```
image/<scene>.jpg                             416×416 RGB, 994.860 scene, dùng chung
grasp_instructions/<scene>_<obj>_<part>.pkl   pickle của MỘT str
grasp_label_positive/<scene>_<obj>_<part>.pt  float32 (N,6) = [q, x, y, w, h, theta_deg]
grasp_label_negative/<scene>_<obj>_<part>.pt  cùng layout, q ≤ 0
part_mask/<scene>_<obj>_<part>.npy            uint8 (416,416), giá trị {0,1}
```

4.412.384 sample part-level trên 994.860 scene (≈4,4 part/scene). Cột `q` là điểm antipodal
`T̃ = (cos α₁ + cos α₂)/R` của paper §3.2 — ngưỡng `T̃ > 0` chính là thứ chia positive/negative.
`_grasp_anything_format` bỏ qua cột này.

Notebook đọc schema thẳng từ zip trên HuggingFace bằng HTTP range (~5 MB), không cần tải
150 GB; kèm chế độ dựng mini-subset khớp id để dev loader.

## 1. Vấn đề

Một ảnh sinh ra nhiều sample, mỗi sample là một grasp target khác nhau, và GR-ConvNet
gốc không nhận được tín hiệu nào để phân biệt chúng. Nó buộc phải regress về trung bình.

Text chính là tín hiệu còn thiếu đó.

Mức độ ambiguity phụ thuộc mức annotation — đo trên split của **base** GA:

```
seen    15,089 samples / 12,036 scenes   objs/scene: mean 1.25, max 4
unseen   8,009 samples /  7,459 scenes   objs/scene: mean 1.07, max 3
```

76% scene ở split seen chỉ có **một** object. Nên ở mức *object*, ambiguity chỉ tồn tại
ở 24% scene — yếu hơn nhiều so với giả định ban đầu.

Ở mức *part* thì khác hẳn: GA++ có nhiều prompt part-level cho cùng một object
("at its handle" / "at its rim"), mỗi prompt ứng với một vùng grasp khác nhau. Ambiguity
này có mặt ở gần như mọi sample. **Đây mới là chỗ conditioning thực sự có việc để làm**,
và toàn bộ thiết kế dưới đây làm việc ở mức part.

> Số trên là của base GA. Trên GA++ đã verify: id **là** part-level
> (`<scene>_<object>_<part>`), nên `split/grasp-anything/*.obj` (id hai phần) khớp **0**
> sample — phải dùng `split/build_grasp_anything_pp.py`.

## 2. Ý tưởng

**Tăng độ liên quan giữa token quan trọng và vùng quan trọng.**

Mô hình tính relevance giữa mọi token và mọi pixel, rồi soft-select token phù hợp tại từng
vùng. Không cần trích target phrase. Alignment map được supervise bằng `part_mask`; grasp
decoder nhận cả vùng quan trọng lẫn text feature tương ứng với vùng đó.

## 3. Kiến trúc

```
Prompt ──► CLIP text encoder ──► token embeddings [t_1..t_L]
                                        │
Image ──► GR-Conv encoder ──► F ────────┤
                                        ▼
                     token–region relevance S_xyj
                                  │
                     α_xyj = softmax_j(S_xyj / τ_t)
                            ┌─────┴─────┐
                            ▼           ▼
                    region map A_T   region text R_T
                            └─────┬─────┘
                                        │
                         residual feature fusion
                  F' = F + φ([F, F ⊙ A_T, R_T])
                                        │
                              GR-Conv decoder
                                        │
                            ┌───────┬───────┬───────┐
                           Q_T    cos2θ   sin2θ    W
```

Với pixel `p=(x,y)`:

```
S_pj = cos(W_v F_p, W_t t_j) / τ
α_pj = softmax_j(S_pj / τ_t)           # token quan trọng tại pixel p
A_T(p) = σ(Σ_j α_pj S_pj)              # vùng quan trọng
R_T(p) = Σ_j α_pj W_r t_j              # text feature của vùng đó
```

Loại SOT/EOT/padding/punctuation khỏi candidate tokens. Fuse trước decoder để prompt
condition cả quality, angle và width. `A_T` supervise ở resolution thật của feature map.

> **Trạng thái:** V2 đã implement trong `inference/models/grconvnet3_align.py`. V1 (hard `max`
> + gate `F⊙(1+A_T)`) vẫn còn dưới dạng cờ `--align-mode hard --fusion gate` để làm một arm
> của ablation; checkpoint V1 cũ vẫn load và chạy đúng như lúc train.

## 4. Loss

```
L = L_grasp(Q_T, cos2θ, sin2θ, W)
  + λ · L_align(A_T, part_mask)
```

`L_align` = BCE + Dice (Dice đỡ cho việc part_mask chỉ chiếm ~2–5% pixel). `τ` là temperature
học được, khởi tạo 0.07. BCE tính trên logits; Dice tính sau sigmoid. Warmup `λ` trong vài
epoch đầu. Bỏ `Q_g`, `M_∪` và `L_agnostic` để objective gọn và dễ ablate.

## 5. Đánh giá

Theo đúng protocol của LGD (paper §5.1) để số đặt cạnh Table 2 được:

- **Split** — seen = Base, unseen = New, chia theo *category object*: 70% category theo tần
  suất giảm dần vào Base, 30% còn lại vào New. `split/build_grasp_anything_pp.py`, category
  lấy từ `scene_description/<scene>.pkl` = `(caption, [object_0, object_1, ...])` index theo
  `<object_idx>`.
- **Metric** — success khi IoU ≥ 0.25 **và** lệch góc ≤ 30°. Repo đã đúng sẵn:
  `--iou-threshold 0.25` và `angle_threshold=np.pi/6` ở `utils/dataset_processing/grasp.py:351`.
- **Báo cáo** — Seen / Unseen / harmonic mean H. `script/eval_grasp_anything_pp.sh <checkpoint>`
  chạy cả hai split rồi in H.

Train trên seen, eval cả hai. Unseen là category chưa từng thấy lúc train, nên nó đo đúng thứ
nhánh ngôn ngữ đáng lẽ phải giúp: từ vựng part dùng chung giữa các category (đo trên 4000
prompt thật: chỉ **197 part-phrase** khác nhau, `handle`/`cap`/`skin`/`stem`... lặp lại khắp
nơi), còn danh từ object thì theo định nghĩa của split là không.

## 6. Ablation

Cùng một file `grconvnet3_align.py`, bật/tắt bằng cờ. Giữ **nguyên** ngân sách train giữa các
arm (`--epochs × --batches-per-epoch × --batch-size`) và ghi rõ con số đó trong report.

Cấu hình chạy: **một GPU · 100.000 scene (~445k sample, 10% GA++) · 50 epoch**, tức
50 × 2000 × 64 = **6,4M lượt sample** mỗi arm ≈ 14 lượt qua subset. Đây là subsample của GA++,
không phải train hết — và khác paper ở cả hai chiều (paper: 100% dữ liệu, 100 epoch), nên bảng
dưới đây so *giữa các arm với nhau*, không so thẳng với Table 2. Bốn arm chạy tuần tự bằng
`script/run_ablation.sh`.

| Arm | Token–region | Region text | `L_align` | Seen | Unseen | H |
|---|---|---|---|---|---|---|
| GR-ConvNet (no text) | ✗ | ✗ | ✗ | | | |
| Hard-max V1 | max | ✗ | ✓ | | | |
| Soft alignment | softmax | ✗ | ✓ | | | |
| **V2 full** | softmax | ✓ | ✓ | | | |

Số tham chiếu từ Table 2 của paper (cùng split, cùng metric) — để đối chiếu, không phải để
thắng: GR-ConvNet + CLIP `0.37 / 0.18 / 0.24` · CLIP-Fusion `0.40 / 0.29 / 0.33` ·
LGD `0.48 / 0.42 / 0.45`.

## 7. Định vị so với related work

- **LGD** (Vuong et al., CVPR 2024) — chính là paper giới thiệu GA++. Generative, diffusion,
  mỗi lần suy luận chạy T bước denoising.
- **CLIPORT** — two-stream semantic / spatial, là gốc của ý "decouple".
- **LAVT / CLIPSeg / referring segmentation** — token-pixel alignment + auxiliary grounding
  loss là công thức chuẩn của nhánh này.

Cái ta làm: một baseline **discriminative, một forward pass**. Mô hình học relevance
token–region tường minh, soft-select token quan trọng tại từng vùng và đưa region-specific
text feature vào grasp decoder.

Phạm vi: đây là bài tập, không đặt mục tiêu vượt LGD. Kết quả cần có là bảng Seen/Unseen/H của
bốn arm ở §6, đủ để thấy từng thành phần đóng góp gì.

## 8. Ghi chú implement

**Đã xong:**

- `hardware/` (`device.py` + `camera.py`) — trước đó thiếu hẳn, cả ba entry point ImportError ngay.
- `utils/data/grasp_anything_data.py`: bỏ glob song song `prompt_files`/`rgb_files` (chúng
  không được lọc theo seen/unseen như `grasp_files`, và nhiều grasp file dùng chung một ảnh);
  `get_depth()` raise thông báo rõ thay vì AttributeError.
- `script/download_grasp_anything_pp.sh` — ảnh + `scene_description` + ba thư mục label GA++.
- `notebooks/ga_pp_schema.ipynb` — verify schema qua HTTP range, dựng mini-subset.
- `utils/data/grasp_anything_pp_data.py` — loader GA++: `grasp_instructions/`, `part_mask/`.
  `part_mask` chịu **đúng** chuỗi rotate→zoom→resize của ảnh (đo lệch tâm 0.00px trên 8
  tổ hợp rot/zoom; control âm: nếu mask bỏ qua rot thì lệch 97–106px). `M_∪` vẫn dựng được
  (`include_union=True`) nhưng mặc định tắt từ khi V2 bỏ `Q_g`.
- `inference/models/grconvnet3_align.py` — V1: CLIP per-token + hard-max `A_T` + residual
  gate + `Q_g`. `use_text=False` là baseline no-text. (V2 bên dưới thay phần lõi.)
- `split/build_grasp_anything_pp.py`, `script/eval_grasp_anything_pp.sh`.
- `train_network.py` / `evaluate.py` nhận batch 6 phần tử (tương thích ngược với 5), thêm
  `--use-text --w-align --warmup-epochs` (V2 thêm `--align-mode --region-text --fusion
  --align-stage` và nhóm cờ chẩn đoán; `--w-agnostic` bỏ cùng với `Q_g`).

**Sửa lỗi có sẵn (không sửa thì không chạy được trên môi trường mới):**

- `np.int` ×5 / `np.float` ×1 trong `utils/dataset_processing/grasp.py` — numpy ≥ 1.24 đã gỡ.
- `torch.load(..., weights_only=False)` ở `evaluate.py` và `inference/grasp_generator.py` —
  torch ≥ 2.6 mặc định `weights_only=True`, không load nổi checkpoint lưu cả module.
- `grasp_generator.load_model()` dùng `self.device` trước khi gán.
- `validate()` truyền `rot`/`zoom` dạng tensor vào `get_gtbb`, `np.cos(tensor)` làm ma trận
  xoay thành `(2,2,1)` → ValueError. Loader GA++ ép `float()`; **ba loader cũ (cornell,
  jacquard, grasp-anything) vẫn còn lỗi này**.

**V2 — đã xong:**

- `inference/models/grconvnet3_align.py` — soft token–region attention (`α_pj = softmax_j`),
  `R_T`, residual fusion `F' = F + φ([F, F⊙A_T, R_T])` với conv cuối khởi tạo 0 (bước 0 đúng
  bằng baseline). Bỏ hẳn `Q_g`/`L_agnostic`. Loại thêm token dấu câu khỏi candidate — hình
  alignment của run V1 cho thấy `.` thường là token "mạnh nhất". Bốn arm của §6 bật/tắt bằng
  cờ; `--align-stage conv4` tính `A_T` ở 113×113 thay vì 56×56.
- `utils/checkpoint.py` — checkpoint 9 MB thay vì 265 MB (state_dict + kwargs, bỏ CLIP text
  tower). `load_network()` đọc cả hai định dạng; checkpoint V1 tự khôi phục cấu hình `hard`/
  `gate` nên số đo lại vẫn là số của đúng model đó.
- `utils/diagnostics.py` + logging trong `train_network.py` — `align/score_margin`,
  `token/{entropy,top1_mass,winning}`, `fusion/residual_ratio`, gradient của `L_grasp` và
  `L_align` riêng rẽ lên visual encoder, probe set cố định vẽ lại qua các epoch, và
  `counterfactual/prompt_drop` ngay trong lúc train. Cách đọc từng nhóm: mục "Đọc log" của
  README.
- `script/check_dataset.py` — kiểm dữ liệu trước khi train: foreground, IoU giữa các part cùng
  object, tỉ lệ mask trùng nhau, số token mỗi prompt, lưới ảnh.
- `script/audit_text_reliance.py` — bốn điều kiện đối chứng (thật / hoán vị / cố định / part
  khác cùng object) + gallery lỗi bốn nhóm. `evaluate.py --shuffle-prompts` là bản gọn.
- `script/smoke_test_align.py` — chạy được không cần dataset, không cần GPU.

**Còn lại:**

- Chạy ablation ở §6 với cùng ngân sách train (cần GPU + dữ liệu). `script/run_ablation.sh`
  chạy sẵn bốn arm `notext / hardmax / soft / full` tuần tự trên một GPU, 100k scene, 50 epoch.
- Kết quả `check_dataset.py` trên dữ liệu thật quyết định mức trần của cả hướng này: nếu phần
  lớn part cùng object có mask trùng nhau thì `L_align` không dạy được grounding part-level.
- `grasp_label_negative/` chưa dùng.
