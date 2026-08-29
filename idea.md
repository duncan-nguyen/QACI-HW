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

Tách hai câu hỏi và supervise bằng **hai loại nhãn khác hẳn nhau**:

| Nhánh | Trả lời | Target | Lấy từ |
|---|---|---|---|
| `Q_g` (geometry) | grasp *thế nào* | `M_∪` — hợp grasp của **mọi part** của object | union grasp label của object |
| `A_T` (language) | grasp *chỗ nào* | `part_mask` của part được prompt nhắc | `part_mask/*.npy` |
| `Q_T` (output) | kết quả | grasp rectangle của part đó | `grasp_label_positive` |

Điểm mấu chốt: `A_T` được supervise bằng **segmentation mask**, còn `Q` bằng **grasp
rectangle**. Hai nhãn khác loại → decoupling là thật.

Nếu supervise `A_T` bằng chính grasp rectangle đã rasterize thì `L_align` chỉ là bản sao
của `L_pos` (target của `Q` cũng chính là rectangle rasterize, xem
`utils/dataset_processing/grasp.py:252-259`), và "decouple WHAT/HOW" thành khẩu hiệu rỗng.
`part_mask/` của GA++ tránh được đúng cái bẫy đó.

## 3. Kiến trúc

```
Prompt ──► CLIP text encoder ──► token embeddings [t_1..t_L]
                                        │
Image ──► GR-Conv encoder ──► F ────────┤
                                        ▼
                          token-pixel alignment
                          A_T(x,y) = σ( max_j cos(W_v F_xy, W_t t_j) / τ )
                                        │
                          F' = F ⊙ (1 + A_T)      ◄── residual gating
                                        │
                              GR-Conv decoder
                                        │
                            ┌───────┬───────┬───────┐
                           Q_T    cos2θ   sin2θ    W
```

**Fuse ở feature level, không phải output level.** Prompt là part-level — handle và vành
cốc khác cả angle lẫn width, nên cả 4 head đều phải được condition. Chỉ nhân `A_T` vào `Q`
là ablation, không phải method.

Residual gating `(1 + A_T)` thay vì `A_T` để tránh gradient triệt tiêu khi `A_T ≈ 0` lúc
đầu training. Kèm warmup λ từ 0.

Cài đặt: `inference/models/grconvnet3_align.py`, tên network `grconvnet3_align`. Gate đặt sau
`res5` trước `conv4` nên cả 4 head đều được condition. Với `--input-size 224` thì `F` là
56×56×128; `A_T` và `Q_g` được supervise ngay ở 56×56 — target hạ xuống bằng avg-pool, không
upsample logits lên 224 để tạo độ chính xác giả.

## 4. Loss

```
L = L_grasp   (Q_T, cos2θ, sin2θ, W)    # loss GR-ConvNet gốc
  + λ1 · L_agnostic(Q_g, M_∪)            # graspability, bất kể part nào
  + λ2 · L_align   (A_T, part_mask)      # grounding: prompt → đúng part
```

`L_align` = BCE + Dice (Dice đỡ cho việc part_mask chỉ chiếm ~2–5% pixel). `τ` là temperature
học được, khởi tạo 0.07 — bắt buộc, vì cosine ∈ [-1,1] không đủ dải cho BCE. `L_agnostic` là
BCE. Cả hai tính trên logits (`BCEWithLogits`), không sigmoid trước.

Cờ: `--w-align` (λ2), `--w-agnostic` (λ1), `--warmup-epochs` (warmup tuyến tính 0→1),
`--use-text 0` để tắt hẳn nhánh ngôn ngữ. Đặt trọng số về 0 là bỏ hẳn loss phụ đó, không chỉ
nhân 0 — dùng cho các arm ablation ở §6.

Optional — contrastive dùng **`grasp_label_negative/`** làm hard negatives thật, thay cho
in-batch negatives (vốn nhiều false negative vì rất nhiều prompt trùng nhau). Chưa cài.

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

> **Cảnh báo leak:** một scene có thể xuất hiện ở **cả** seen lẫn unseen — "unseen" nghĩa là
> unseen *category*, không phải unseen *ảnh*. Khi dựng `M_∪` chỉ được union các label nằm
> trong split đang train. `GraspAnythingPPDataset` gom `_files_by_object` **sau** khi lọc
> split nên tự động đúng; đừng thay bằng glob thẳng trên đĩa.

## 6. Ablation

Cùng một file `grconvnet3_align.py`, bật/tắt bằng cờ. Giữ **nguyên** ngân sách train giữa các
arm (`--epochs × --batches-per-epoch × --batch-size`) và ghi rõ con số đó trong report — mặc
định của repo (1000×50×8 = 400k sample) chỉ chạm ~9% của 4,4M, đó là subsample chứ không phải
train hết.

| Arm | Cờ | text→spatial | L_align | Q_g union | Seen | Unseen | H |
|---|---|---|---|---|---|---|---|
| GR-ConvNet (no text) | `--use-text 0 --w-agnostic 0` | ✗ | ✗ | ✗ | | | |
| + spatial attention | `--w-align 0 --w-agnostic 0` | ✓ | ✗ | ✗ | | | |
| + align loss | `--w-agnostic 0` | ✓ | ✓ | ✗ | | | |
| **Ours (full)** | mặc định | ✓ | ✓ | ✓ | | | |

Số tham chiếu từ Table 2 của paper (cùng split, cùng metric) — để đối chiếu, không phải để
thắng: GR-ConvNet + CLIP `0.37 / 0.18 / 0.24` · CLIP-Fusion `0.40 / 0.29 / 0.33` ·
LGD `0.48 / 0.42 / 0.45`.

## 7. Định vị so với related work

- **LGD** (Vuong et al., CVPR 2024) — chính là paper giới thiệu GA++. Generative, diffusion,
  mỗi lần suy luận chạy T bước denoising.
- **CLIPORT** — two-stream semantic / spatial, là gốc của ý "decouple".
- **LAVT / CLIPSeg / referring segmentation** — token-pixel alignment + auxiliary grounding
  loss là công thức chuẩn của nhánh này.

Cái ta làm: một baseline **discriminative, một forward pass**, trong đó grounding part-level
được supervise **tường minh** bằng `part_mask` thay vì để fusion module tự học ngầm. Nhãn của
`A_T` (segmentation mask) khác loại với nhãn của `Q` (grasp rectangle) nên việc decouple
WHAT/HOW là thật chứ không phải hai bản sao của cùng một supervision.

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
- `utils/data/grasp_anything_pp_data.py` — loader GA++: `grasp_instructions/`, `part_mask/`,
  `M_∪`. `part_mask` chịu **đúng** chuỗi rotate→zoom→resize của ảnh (đo lệch tâm 0.00px trên 8
  tổ hợp rot/zoom; control âm: nếu mask bỏ qua rot thì lệch 97–106px).
- `inference/models/grconvnet3_align.py` — CLIP text (đóng băng, per-token) + `A_T` + gating +
  hai loss phụ. `use_text=False` là baseline no-text dùng chung file, chung ngân sách train.
- `split/build_grasp_anything_pp.py`, `script/eval_grasp_anything_pp.sh`.
- `train_network.py` / `evaluate.py` nhận batch 6 phần tử (tương thích ngược với 5), thêm
  `--use-text --w-align --w-agnostic --warmup-epochs`.

**Sửa lỗi có sẵn (không sửa thì không chạy được trên môi trường mới):**

- `np.int` ×5 / `np.float` ×1 trong `utils/dataset_processing/grasp.py` — numpy ≥ 1.24 đã gỡ.
- `torch.load(..., weights_only=False)` ở `evaluate.py` và `inference/grasp_generator.py` —
  torch ≥ 2.6 mặc định `weights_only=True`, không load nổi checkpoint lưu cả module.
- `grasp_generator.load_model()` dùng `self.device` trước khi gán.
- `validate()` truyền `rot`/`zoom` dạng tensor vào `get_gtbb`, `np.cos(tensor)` làm ma trận
  xoay thành `(2,2,1)` → ValueError. Loader GA++ ép `float()`; **ba loader cũ (cornell,
  jacquard, grasp-anything) vẫn còn lỗi này**.

**Còn lại:**

- Tải dữ liệu → chạy `split/build_grasp_anything_pp.py` → train. Chưa có số thật nào.
- `--w-align` / `--w-agnostic` đang 1.0 / 1.0, chưa tune.
- Checkpoint 265 MB vì `torch.save(net)` pickle cả CLIP text tower.
- `F` ở bottleneck là 56×56 (stride 4) — hơi thô cho part-level; nếu `A_T` mờ thì thử tính sau
  `conv4` (112×112).
- `grasp_label_negative/` chưa dùng.
