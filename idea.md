# Grasp-Aware Text-Visual Alignment

Language-driven grasp detection trên Grasp-Anything++, chỉ dùng dữ liệu có sẵn —
không sinh ảnh, không LLM-generate prompt, không synthetic data.

## 1. Vấn đề

Mỗi sample của Grasp-Anything là `<scene>_<obj>.pt` + `<scene>.jpg`: một ảnh sinh ra
N sample, mỗi sample chỉ chứa grasp của **một** object. GR-ConvNet gốc vì thế nhận
supervision **mâu thuẫn** — cùng một ảnh, lúc bảo "grasp ở đây", lúc bảo "grasp ở kia",
không có tín hiệu nào phân biệt.

Text chính là tín hiệu còn thiếu đó.

## 2. Ý tưởng

Tách rõ hai câu hỏi và supervise chúng bằng **hai nhãn khác nhau**:

| Nhánh | Trả lời | Target | Lấy từ |
|---|---|---|---|
| `Q_g` (geometry) | grasp *thế nào* | `M_∪` — hợp grasp của **mọi** object trong scene | union `<scene>_*.pt` |
| `A_T` (language) | grasp *cái gì* | `M_i` — grasp của object được prompt nhắc | `<scene>_<i>.pt` |
| `Q_T` (output) | kết quả | `M_i` | nhãn hiện tại |

`Q_g` học class-agnostic graspability; `A_T` học grounding từ prompt. Cả hai nhãn đều
đã nằm sẵn trong dataset — không tạo thêm sample nào.

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

**Fuse ở feature level, không phải output level.** Prompt của GA++ là part-level
("grasp the mug *at its handle*") — handle và vành cốc khác cả angle lẫn width, nên
cả 4 head đều phải được condition. Chỉ nhân `A_T` vào `Q` là ablation, không phải method.

Residual gating `(1 + A_T)` thay vì `A_T` để tránh gradient triệt tiêu khi `A_T ≈ 0`
lúc đầu training. Kèm warmup λ từ 0.

## 4. Loss

```
L = L_grasp(Q_T, cos2θ, sin2θ, W)      # loss GR-ConvNet gốc
  + λ1 · L_agnostic(Q_g,  M_∪)          # graspability, bất kể object nào
  + λ2 · L_align   (A_T,  M_i)          # grounding: prompt → đúng vùng
```

`L_align`: BCE hoặc Dice. Bắt buộc có sigmoid + temperature học được (cosine ∈ [-1,1],
BCE cần [0,1]).

Optional — contrastive với **intra-image negatives** (các object khác trong *cùng* scene,
không phải in-batch): negatives khó hơn hẳn và không có false negative.

## 5. Đánh giá: prompt-swap

Metric IoU@0.25 match với **bất kỳ** GT nào trong ảnh, nên model hoàn toàn có thể
ignore text mà vẫn ăn điểm. Phải chứng minh ngược lại:

Giữ nguyên ảnh, thay prompt của object `i` bằng prompt của object `j` **cùng scene**:

```
Δ = Acc(prompt đúng) − Acc(prompt đã swap)
```

- `Δ ≈ 0`  → model ignore text, mọi con số còn lại vô nghĩa.
- `Δ` lớn  → bằng chứng trực tiếp rằng alignment loss có tác dụng.

Zero dữ liệu mới. Đây là figure quan trọng nhất của paper.

## 6. Ablation

| Method | text→spatial | L_align | union Q_g | Acc | **Δ swap** |
|---|---|---|---|---|---|
| GR-ConvNet (no text) | ✗ | ✗ | ✗ | | 0 (by design) |
| + CLIP concat | ✗ | ✗ | ✗ | | |
| + spatial attention | ✓ | ✗ | ✗ | | |
| + align loss | ✓ | ✓ | ✗ | | |
| **Ours (full)** | ✓ | ✓ | ✓ | | |

Hai cột cuối là hai contribution riêng biệt — không gộp.

## 7. Định vị so với related work

"Decouple semantic / spatial" là **CLIPORT**; token-pixel alignment + auxiliary grounding
loss là **LAVT / CLIPSeg / referring segmentation**. Phải cite và so sánh trực diện.

Delta thật sự:

> Referring segmentation cần segmentation mask. Chúng tôi cho thấy **grasp rectangle là
> supervision grounding miễn phí**, và ở mức **part-level** — thứ mà mask object-level
> không cung cấp được.

Câu này thuộc về abstract.

## 8. Ghi chú implement

Repo hiện là Grasp-Anything gốc, chưa có gì liên quan text.

**Đã fix:**

- Thêm package `hardware/` (`device.py` + `camera.py`) — trước đó thiếu hẳn, khiến
  `train_network.py`, `evaluate.py`, `train_network_grasp_det_seg.py` ImportError ngay.
- `utils/data/grasp_anything_data.py`: bỏ `prompt_files` / `rgb_files` glob song song
  (chúng **không** được filter theo seen/unseen như `grasp_files`, và nhiều grasp file
  dùng chung một ảnh). Mọi path giờ derive từ `grasp_files[idx]` qua
  `get_rgb_file()` / `get_prompt_file()`.
- Regex `_\d{1}\.pt` → `_\d+\.pt`. Regex cũ **không khớp gì cả** với scene ≥10 object
  (`abc_10.pt` giữ nguyên) → đường dẫn ảnh thành `image/abc_10.pt` → FileNotFoundError.
  Toàn bộ object thứ 10 trở đi của mọi scene lớn bị hỏng.
- `get_depth()` raise NotImplementedError với thông báo rõ thay vì AttributeError trên
  `self.depth_files` (attribute không bao giờ được gán — Grasp-Anything chỉ có RGB).

**Còn lại:**

- `utils/data/grasp_data.py:92` trả `x, (pos,cos,sin,width), idx, rot, zoom` — cần mở rộng
  để trả thêm prompt tokens và `M_∪`; `inference/models/grasp_model.py:19` hard-code
  chữ ký 4 tensor.
- `scene_description/*.pkl` là mô tả scene-level, không phải grasping prompt của GA++.
- CLIP `encode_text()` chỉ trả EOT token; muốn per-token phải chạy
  `transformer → ln_final → @ text_projection` cho cả chuỗi, và mask SOT/EOT/padding
  trước khi aggregate.
- `F` ở bottleneck là 56×56 (stride 4) — hơi thô cho part-level; cân nhắc tính `A_T`
  sau `conv4`/`conv5`.
