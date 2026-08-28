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
| `airvlab/Grasp-Anything-pp` | `grasp_instructions.zip` | 1.54 GB | prompt (.pkl) |
| | `grasp_label_positive.zip` | 3.95 GB | grasp **part-level** (.pt) |
| | `part_mask.zip` | 4.79 GB | mask **part-level** (.npy) |
| | `grasp_label_negative.zip` | 5.12 GB | negative grasp (.pt) |

Script: `script/download_grasp_anything_pp.sh` (`--check` để verify layout).

> **Va chạm tên:** cả hai repo đều có `grasp_label_positive.zip` — base là object-level,
> GA++ là part-level. Không trộn. Script chỉ lấy bản GA++.

Tác giả yêu cầu điền form + đồng ý MIT trước khi dùng:
https://airvlab.github.io/grasp-anything/docs/download/

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

> Số trên là của base GA. Phải đo lại trên GA++ sau khi tải — id trong
> `grasp_label_positive` của GA++ có thể là part-level, khi đó `split/grasp-anything/*.obj`
> không khớp và phải dựng split mới.

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
                          A_T(x,y) = σ( agg_j cos(W_v F_xy, W_t t_j) / τ )
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

## 4. Loss

```
L = L_grasp   (Q_T, cos2θ, sin2θ, W)    # loss GR-ConvNet gốc
  + λ1 · L_agnostic(Q_g, M_∪)            # graspability, bất kể part nào
  + λ2 · L_align   (A_T, part_mask)      # grounding: prompt → đúng part
```

`L_align`: BCE hoặc Dice. Bắt buộc có sigmoid + temperature học được (cosine ∈ [-1,1],
BCE cần [0,1]).

Optional — contrastive dùng **`grasp_label_negative/`** làm hard negatives thật, thay cho
in-batch negatives (vốn nhiều false negative vì rất nhiều prompt trùng nhau).

## 5. Đánh giá: prompt-swap

`calculate_iou_match` match với **bất kỳ** GT nào trong ảnh
(`utils/dataset_processing/evaluation.py:75-79`), nên model hoàn toàn có thể ignore text
mà vẫn ăn điểm. Phải chứng minh ngược lại.

Giữ nguyên ảnh, thay prompt của part `i` bằng prompt của **part khác cùng object**:

```
Δ = Acc(prompt đúng) − Acc(prompt đã swap)
```

- `Δ ≈ 0`  → model ignore text, mọi con số còn lại vô nghĩa.
- `Δ` lớn  → bằng chứng trực tiếp rằng alignment loss có tác dụng.

Swap ở mức **part** chứ không phải object, vì hai lý do:

1. **Coverage.** Swap object-level chỉ chạy được trên scene có ≥2 object — 39% samples ở
   split seen, **14%** ở unseen. Part-level dùng được gần như mọi sample.
2. **Độ khó.** Cùng object, chỉ khác chỗ cầm → negative khó hơn hẳn, loại trừ khả năng
   model chỉ học "object nào to nhất trong ảnh".

> **Cảnh báo leak:** 1,712 scene xuất hiện ở **cả** seen lẫn unseen — "unseen" nghĩa là
> unseen *category*, không phải unseen *ảnh*. Khi dựng `M_∪`, chỉ được union các label
> nằm trong split đang train. Union tất cả file trên đĩa là nhét category unseen vào
> target training → số unseen đẹp một cách vô nghĩa.

## 6. Ablation

| Method | text→spatial | L_align (mask) | union Q_g | Acc | **Δ swap** |
|---|---|---|---|---|---|
| GR-ConvNet (no text) | ✗ | ✗ | ✗ | | 0 (by design) |
| + CLIP concat | ✗ | ✗ | ✗ | | |
| + spatial attention | ✓ | ✗ | ✗ | | |
| + align loss | ✓ | ✓ | ✗ | | |
| **Ours (full)** | ✓ | ✓ | ✓ | | |

Ba cột đầu là ba contribution riêng biệt — không gộp.

## 7. Định vị so với related work

Phải cite và so sánh trực diện:

- **CLIPORT** — "decouple semantic / spatial" chính là two-stream của nó.
- **LAVT / CLIPSeg / referring segmentation** — token-pixel alignment + auxiliary
  grounding loss là công thức chuẩn của nhánh này.
- **LGD** (Vuong et al., CVPR 2024) — chính là paper giới thiệu GA++, dùng diffusion.
  Đây là đối thủ trực tiếp nhất.

Delta trung thực — **không** phải "grasp rectangle là supervision miễn phí" (GA++ ship
`part_mask` sẵn, luận điểm đó không đứng được):

> Một baseline discriminative nhẹ cho language-driven grasping, trong đó grounding
> part-level được supervise **tường minh** thay vì để model tự học ngầm. Kèm một
> diagnostic (prompt-swap) cho thấy các baseline hiện có phần lớn **ignore text**.

Nói thẳng: delta về *method* là khiêm tốn. Phần sắc nhất là **diagnostic** — nếu
prompt-swap cho thấy CLIP-concat có `Δ ≈ 0` còn method này thì không, đó là kết quả đáng
báo cáo hơn cả bảng accuracy. Đủ cho HW/workshop; muốn lên hội nghị lớn thì cần thêm.

## 8. Ghi chú implement

**Đã fix:**

- Thêm package `hardware/` (`device.py` + `camera.py`) — trước đó thiếu hẳn, khiến
  `train_network.py`, `evaluate.py`, `train_network_grasp_det_seg.py` ImportError ngay.
- `utils/data/grasp_anything_data.py`: bỏ `prompt_files` / `rgb_files` glob song song
  (chúng **không** được filter theo seen/unseen như `grasp_files`, và nhiều grasp file
  dùng chung một ảnh). Mọi path giờ derive từ `grasp_files[idx]` qua
  `get_rgb_file()` / `get_prompt_file()`.
- Regex `_\d{1}\.pt` → `_\d+\.pt`. *Lưu ý: object index tối đa trong cả hai split base GA
  là 4, nên bug này chưa từng kích hoạt — fix mang tính phòng thủ cho GA++.*
- `get_depth()` raise NotImplementedError với thông báo rõ thay vì AttributeError trên
  `self.depth_files` (attribute không bao giờ được gán — Grasp-Anything chỉ có RGB).
- `script/download_grasp_anything_pp.sh` — tải đúng phần cần cho task language-driven.

**Còn lại:**

- Loader GA++ mới: đọc `grasp_instructions/` (không phải `scene_description/`, vốn là mô
  tả scene-level của base GA), `part_mask/`, `grasp_label_positive/` part-level.
- `utils/data/grasp_data.py:95` trả `x, (pos,cos,sin,width), idx, rot, zoom` — cần mở rộng
  để trả thêm prompt tokens, `part_mask`, `M_∪`; `inference/models/grasp_model.py:16`
  hard-code chữ ký 4 tensor.
- Augmentation: `part_mask` và `M_∪` phải chịu **cùng** rot/zoom với ảnh và grasp label,
  nếu không alignment loss học nhầm.
- CLIP `encode_text()` chỉ trả EOT token; muốn per-token phải chạy
  `transformer → ln_final → @ text_projection` cho cả chuỗi, và mask SOT/EOT/padding
  trước khi aggregate.
- `F` ở bottleneck là 56×56 (stride 4) — hơi thô cho part-level; cân nhắc tính `A_T`
  sau `conv4`/`conv5`.
- Dựng split mới nếu id GA++ là part-level.
