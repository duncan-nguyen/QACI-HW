# Spatial Text Alignment for Language-Driven Grasp Detection (STAG)

This repository is a fork of the [Grasp-Anything](https://airvlab.github.io/grasp-anything/)
codebase. It adds **STAG**, a single-pass model for part-level language-driven grasp detection
on Grasp-Anything++ (GA++), together with the data tooling, ablation scripts and diagnostics
used to produce the accompanying paper.

STAG computes token--region relevance between a frozen CLIP text encoder and the GR-ConvNet
feature map, supervises the resulting spatial map with the annotated target-part mask, and
conditions all four grasp outputs on that map and on a region-specific text feature. On a
held-out test partition of a 100K-scene GA++ subset it reaches a harmonic mean of **0.273**,
exceeding the image-only and unsupervised-alignment variants by 6.1 and 3.8 points.

## Table of contents
   1. [Method](#method)
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)
   1. [Results](#results)
   1. [Reproducing the reported setting](#reproducing-the-reported-setting)
   1. [Reading the logs](#reading-the-logs)
   1. [Fixes in this fork](#fixes-in-this-fork)
   1. [Loader performance](#loader-performance)

## Method

Given an RGB image `I` and prompt `T`, the model predicts dense grasp quality `Q`, angle
components `(C, S) = (cos 2θ, sin 2θ)` and normalized width `W`; local maxima of `Q` are
decoded as rectangular grasps.

Three properties matter:

- **Soft token--region alignment.** Tokens are selected per pixel by a softmax rather than an
  argmax, so several contextual tokens can explain one region. Start, end, padding **and
  punctuation** tokens are excluded from the candidates.
- **Residual grasp conditioning.** The final convolution of `φ` is zero-initialized, so at
  step 0 `F' = F` is exactly the image-only backbone and the language branch phases in as
  `A_T` becomes meaningful. Fusion sits after `res5`, so all four heads are conditioned.
- **Mask-supervised alignment.** `A_T` is supervised at its own 56×56 resolution with the GA++
  `part_mask`, average-pooled down rather than upsampling the logits. This target is not a
  rasterization of the grasp rectangle, so the language branch does not merely relearn what
  the grasp branch already knows.

Implementation: [inference/models/stag.py](inference/models/stag.py),
[utils/data/grasp_anything_pp_data.py](utils/data/grasp_anything_pp_data.py).

### Ablation arms

All arms are the same file and the same training budget; only flags change:

| arm | `--use-text` | `--w-align` | `--align-mode` | `--region-text` | `--fusion` |
|---|---|---|---|---|---|
| **STAG (full)** | 1 | 0.3 | `soft` | 1 | `residual` |
| w/o `L_align` | 1 | 0 | `soft` | 1 | `residual` |
| w/o Text | 0 | 0 | – | – | – |

Two further variants exist but are not reported: `--align-mode hard` (argmax over tokens) with
`--fusion gate` (`F ⊙ (1+λA_T)`), and `--align-stage conv4`, which moves the computation of
`A_T` from 56×56 up to 113×113 — finer for small parts, at the cost of `conv1..res5` no longer
being conditioned on the prompt.

### Checkpoints

Checkpoints are saved by [utils/checkpoint.py](utils/checkpoint.py) as `state_dict` + kwargs
with the CLIP text tower stripped: **~9 MB** instead of 265 MB per file. `load_network()` reads
both that format and the upstream pickled-module format.

A dataset-free, GPU-free sanity check of every arm, the checkpoint round-trip, and overfitting
a single batch:

```bash
python script/smoke_test_align.py
```

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

The Grasp-Anything datasets can be accessed via
[this link](https://airvlab.github.io/grasp-anything/docs/download/).

GA++ ships only the language annotations and labels; the images live in the base
Grasp-Anything repository. Extracting everything is **~830 GB** (`part_mask` alone is 764 GB),
so [script/build_ga_pp_subset.py](script/build_ga_pp_subset.py) builds a subset instead:

- `--pack-masks` — store masks bit-packed, 21 KB instead of 173 KB, lossless
- `--images-from-zip` — download the 65 GB image archive and extract locally (cheaper than
  per-image HTTP above ~40k scenes; below that, images are read through HTTP range requests
  and the 65 GB download is avoided entirely)

| scenes | samples | % of GA++ | disk (packed) |
|---|---|---|---|
| 10k | 44k | 1% | 1.8 GB |
| **100k** (default) | **445k** | **10%** | **18 GB** |
| 300k | 1.34M | 30% | 55 GB |
| 994k (full) | 4.41M | 100% | 183 GB |

The seen/unseen split follows the LGD protocol (Sec. 5.1) and is built by
[split/build_grasp_anything_pp.py](split/build_grasp_anything_pp.py). Two deliberate
departures from a literal reading of that protocol are documented in that file's docstring:
the split is balanced by **sample count** rather than category count (the literal reading
leaves 0.1% of samples for New on GA++'s skewed distribution), and scenes containing both Base
and New objects are dropped from the training side to avoid an image-level leak.

## Training

We use GR-ConvNet as the default deep network. To train GR-ConvNet on different datasets:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0
```
For example, to train a GR-ConvNet on Cornell:
```bash
$ python train_network.py --dataset cornell --dataset-path data/cornell --description training_cornell --use-depth 0
```
Other baselines are selected with `--network`:
```bash
$ python train_network.py --dataset <dataset> --dataset-path <dataset> --description <your_description> --use-depth 0 --network <baseline_name>
```
For instance, to train GG-CNN on Cornell:
```bash
python train_network.py --dataset cornell --dataset-path data/cornell/ --description training_ggcnn_on_cornell --use-depth 0 --network ggcnn
```

STAG is `--network stag` on `--dataset grasp-anything-pp`:
```bash
python train_network.py --dataset grasp-anything-pp \
    --dataset-path data/grasp-anything-pp-full \
    --split-path split/grasp-anything-pp \
    --network stag --use-depth 0 --use-rgb 1 \
    --use-text 1 --w-align 0.3 --align-mode soft --region-text 1 --fusion residual \
    --warmup-epochs 3 --description stag-full
```

## Testing

The same commands apply to any baseline on any dataset:
```bash
python evaluate.py --network <path_to_pretrained_network> --dataset <dataset> --dataset-path data/<dataset> --iou-eval
```
`<path_to_pretrained_network>` is a checkpoint produced by training, normally under
`logs/<timestamp>_<training_description>`. The architecture does not need to be specified — the
codebase detects it from the checkpoint. Upstream pretrained weights are available at
[this link](https://drive.google.com/file/d/1OXVFXqv0rgxiVLz89tnSj0Xb-20ZJ4fH/view?usp=sharing).

`--subset test` evaluates the independent test partition rather than the validation set used
for checkpoint selection; it requires the same `--split`/`--test-split`/`--random-seed` that
training used.

## Results

Independent-test grasp success on the held-out partition. All arms share the same data, split,
seed and training budget. A prediction succeeds when rectangle IoU exceeds 0.25 **and** the
angular error is at most 30°.

| Method | Seen | Unseen | H |
|---|---|---|---|
| **STAG** | **0.354** | **0.222** | **0.273** |
| w/o `L_align` | 0.344 | 0.179 | 0.235 |
| w/o Text | 0.278 | 0.171 | 0.212 |

The full model leads on all three metrics, by 7.6 / 5.1 / 6.1 points over the image-only
baseline. The gap to the unsupervised-alignment arm is 3.8 points in `H`, and it is larger on
Unseen than on Seen (4.3 versus 1.0 points) — consistent with mask supervision helping more on
new categories, though **one seed is not enough to establish a robust generalization
advantage**.

For orientation only, Table 2 of the LGD paper (full GA++, 4.41M samples, 100 epochs, the
original split over LVIS labels) reports GR-ConvNet + CLIP at 0.37/0.18/0.24, CLIP-Fusion at
0.40/0.29/0.33 and LGD at 0.48/0.42/0.45. Those numbers come from a different data scale and
split and are not comparable to the table above.

## Reproducing the reported setting

The reported numbers come from a 100K-scene GA++ subset containing 441,298 part-level samples,
with images resized to 224×224. Each arm receives **640K sampled training instances**
(20 × 250 × 128) with Adam, initial learning rate 2×10⁻³, cosine decay and seed 123.
Validation is used only for checkpoint selection; 0.5% of the data is held out for testing.

[notebooks/train_ga_pp.ipynb](notebooks/train_ga_pp.ipynb) runs exactly that pipeline on Colab
— build the subset, build the split, train `notext / noalign / full` sequentially, evaluate
`--subset test` on Seen and Unseen, and print the LaTeX macros for the results table.

On a single-GPU workstation, [script/run_paper_setting.sh](script/run_paper_setting.sh) drives
the same pipeline end to end. Its defaults are a larger budget than the reported runs
(50 × 2000 × 64 sample draws, lr 1×10⁻³) — override them to match:

```bash
# 1. download + build the dataset, the split, and check the data (slowest step, cached)
SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

# 2. measure num_workers on this machine -- the GPU is almost never the bottleneck
python script/bench_loader.py --dataset-path data/grasp-anything-pp-full \
    --split-path split/grasp-anything-pp --workers 8,16,32,48,64

# 3. train + eval + counterfactual + collect results (build/split are cached and skipped)
NUM_WORKERS=<measured> EPOCHS=20 BATCHES_PER_EPOCH=250 BATCH_SIZE=128 LR=2e-3 \
    bash script/run_paper_setting.sh
```

Note that this repo defines an "epoch" as `BATCHES_PER_EPOCH` batches, not one pass over the
data. What must be held constant across arms — and reported — is the total number of sample
draws.

The ablation table runs **sequentially on one GPU** (`train_network.py` has no DDP; four
processes on one GPU only fight over the CPU dataloader, giving the same total time with no
arm finishing early):

```bash
bash script/run_ablation.sh                      # all arms, sequential
GPU=1 bash script/run_ablation.sh                # pick another GPU
ARMS="soft full" bash script/run_ablation.sh     # re-run a couple of arms
RESUME=1 bash script/run_ablation.sh             # continue after an interruption
```

Every run collects itself into `results/<timestamp>_<description>/`:

```
summary.md                  Seen/Unseen/H table, prompt counterfactual table, tokens, figures
config.txt                  the exact configuration that ran
dataset-check/              part_mask + prompt statistics, image grid for visual inspection
train.log  eval_seen.log  eval_unseen.log  audit_text_reliance.{log,json}
tensorboard/
figures/                    diagnostic_*, prompt_grid, failures_*, prediction_*, tokens_*
```

The figures, in the order worth looking at:

- `prompt_grid.png` — **the most important figure**: one image, different prompts. Each row is
  GT `part_mask` · `A_T` · strongest tokens with attention mass · Q · grasp. Three identical
  rows mean the model is ignoring the language, however low `align_loss` got.
- `diagnostic_{seen,unseen}_*.png` — a complete row for one sample: RGB · GT `part_mask` ·
  `A_T` · **error map** (green TP / red FP / yellow FN) · Q · GT+prediction · strongest tokens.
- `failures_{seen,unseen}.png` — a four-class failure gallery (align ok/grasp ok · align
  ok/grasp wrong · align wrong/grasp wrong · no grasp detected). Which class is crowded tells
  you whether the fault is grounding or grasp decoding — two entirely different fixes.
- `prediction_*.png`, `tokens_*.png`, `parts_same_object.png`.

Redraw without retraining:
`python script/export_results.py --checkpoint <ckpt> --dataset-path <data> --out <dir>`.
The paper's same-image/different-part figure comes from
[script/visualize_alignment_comparison.py](script/visualize_alignment_comparison.py), which
selects candidates purely by the distance between two GT part-mask centroids, never by model
score. `results/` is in `.gitignore`.

## Reading the logs

Three questions come up when debugging the language branch, and this is where each is answered.

**Before training — can the data teach anything?**

```bash
python script/check_dataset.py --data-dir data/grasp-anything-pp-full \
    --split-path split/grasp-anything-pp --out results/dataset-check
```

Reports the foreground fraction of `part_mask`, the IoU between parts of the **same object**,
the rate of duplicate masks, the tokens left per prompt after filtering, and an image grid with
prompts, masks and GT grasps. The important warning: if most same-object part pairs have
IoU > 0.9, the supervision is effectively at *object* level, not part level — `L_align` cannot
then teach part-level grounding, and the counterfactual deltas will be small even with a
perfectly good model. A deeper report with examples: `script/diagnose_part_masks.py`. This step
is already step 3/7 of `run_paper_setting.sh`.

**During training — TensorBoard** (`--diag-interval`, `--probe-samples`, `--counterfactual-every`)

| tag group | how to read it |
|---|---|
| `loss/{train,val}/{total,grasp,align_bce,align_dice}` | separated because warmup pushes the *total* up while each component goes down |
| `weight/lambda_align`, `optimizer/lr` | the *effective* λ of that epoch (warmup already applied) |
| `align/{iou,dice,score_margin,foreground_score,background_score,predicted_area}` | `score_margin = mean(A_T ∣ inside mask) − mean(A_T ∣ outside)`; near 0 means attention does not separate fg/bg, even when BCE is already low |
| `token/{attention_entropy,top1_mass,top3_mass,valid_count,temperature,logit_scale}` | entropy ≈ 0 is collapse onto one token; ≈ `token/max_entropy` means no token matters; `top1_mass ≈ 1` from the start means soft attention is behaving like a hard argmax; `logit_scale` hitting 100 means the temperature is broken |
| `token/winning` (text) | a "which token wins what percentage of pixels" table over the whole val set. Top entries that are all object nouns (`apple`, `mug`) mean the model localises the *object*, not the *part*. `token/punctuation_mass` must be **0** |
| `fusion/{feature_norm,residual_norm,residual_ratio}` | `r_F = ‖F'−F‖/‖F‖`: ≈ 0 means fusion has almost no effect (it is exactly 0 at init by construction — what matters is seeing it *grow*); ≫ 1 means fusion is overwhelming the visual features |
| `gradient/{visual_encoder,text_projection,alignment,fusion,grasp_decoder}` | per-block L2 gradient norms, logged every `--diag-interval` batches |
| `gradient/visual_from_{grasp,align,ratio}` | the gradients of `L_grasp` and `L_align` on the visual encoder **separately**, over the same batch. A `ratio` in the tens or above means lowering `--w-align` or lengthening `--warmup-epochs` |
| `counterfactual/{normal,shuffled,fixed}_success`, `counterfactual/prompt_drop` | `Δ_prompt = S_normal − S_shuffled`. Near 0 means **nothing can be concluded** about the language branch being useful |
| `probe/*` (figures) | a **fixed** 8-sample probe set, redrawn at epochs 0, 1, 3, 10, 25 and at every new best epoch — showing when attention starts to learn and whether it collapses after warmup |
| `step/token/*`, `step/fusion/*` | the same metrics measured on training batches per *step* rather than per epoch |

**After training — the counterfactual, which matters more than any loss curve**

```bash
python script/audit_text_reliance.py --checkpoint logs/.../epoch_67_iou_0.3313 \
    --dataset-path data/grasp-anything-pp-full --split-path split/grasp-anything-pp \
    --n-samples 500 --figures results/audit
```

Four conditions on the same sample: the real prompt · another object's prompt (`shuffled`) ·
one sentence shared by every image (`fixed`) · another part of the same object (`other_part`).
Reading it: `Δ_shuffled ≈ 0` means the grasp does not depend on the prompt; a large
`Δ_shuffled` with `Δ_other_part ≈ 0` means the model recognises the *object* but not the
*part*. `evaluate.py --shuffle-prompts` is a compact version of this for one split.

## Fixes in this fork

Bugs that stopped the upstream codebase from running on a modern environment:

- `np.int` ×5 / `np.float` ×1 in `utils/dataset_processing/grasp.py` — removed in numpy ≥ 1.24
- `torch.load(..., weights_only=False)` in `evaluate.py` and `inference/grasp_generator.py` —
  torch ≥ 2.6 defaults to `weights_only=True`
- `grasp_generator.load_model()` used `self.device` before assigning it
- `validate()` passed `rot`/`zoom` to `get_gtbb` as tensors; `np.cos(tensor)` corrupts the
  rotation matrix (the GA++ loader coerces to `float()`; cornell/jacquard/grasp-anything still
  carry this bug)

Added: `--split-path`, `--lr`, `--lr-schedule` for `train_network.py`; `persistent_workers` /
prefetch / `pin_memory` in the DataLoader; the GA++ loader reorders augmentation to
`crop → resize → rotate` (3.2× faster, geometrically equivalent for rotations that are
multiples of 90°).

Bugs that produced *wrong numbers* rather than crashes, found while re-reading a 200k GA++ run.
Details are in each docstring:

| | Bug | Fix |
|---|---|---|
| A1 | `split/build_grasp_anything_pp.py` gave the 70% most frequent categories to Base, leaving 956/882,214 samples (0.1%) for New; 295 scenes appeared in both sets | `--split-mode balanced` (default) splits by **sample count** with categories still disjoint; `--drop-shared-from seen` (default) removes overlapping images from training; warns when a set falls below 5% |
| A2 | `evaluate.py` and `train_network.py` sliced indices with two duplicated code paths, so the "test set" coincided exactly with the checkpoint-selection set | a shared `utils/data/index_split.py`; added `--test-split` (train) and `--subset {train,val,test,dev,all}` (evaluate) |
| A3 | validation ran on a dataset with `random_rotate`/`random_zoom` on; `zoom ~ U(0.5, 1)` crops grasps away that `get_gtbb` does not drop → empty targets, automatic failures | `GraspDatasetBase.eval_view()` — a shallow copy with augmentation off |
| A4 | `best_iou` was lowered by the periodic `epoch % 10` save branch, producing spurious "best" checkpoints | update only on a genuine improvement; filenames use `%0.4f` instead of `%0.2f` |
| A5 | `evaluate.py` never called `net.eval()` (the model has dropout 0.1 + BatchNorm) | `.eval()` on load |
| A6 | width labels are normalized by `/(output_size/2)` but decoded with `×150` — 150 is `output_size/2` for the original 300×300 GR-ConvNet, so at input 224 every grasp came out 1.34× too long | `post_process_output` derives `width_scale` from the image size (`×112` at 224, still `×150` at 300) |
| A7 | the training loop fetched one extra batch per epoch and discarded it → 1,999 steps for `--batches-per-epoch 2000` | check the stop condition at the top of the inner loop |
| A8 | `GraspRectangle.iou` indexed the canvas with negative values → wrap-around, wrong IoU for grasps near the border | shift both polygons to origin 0 |
| A9 | `script/build_ga_pp_subset.py` wrote images/labels non-atomically → an interruption left truncated files that later runs skipped over | `write_atomic()` via a temp file + `os.replace` |
| A10 | `script/diagnose_part_masks.py` (measuring whether `part_mask` is at part or object level) had been dropped from this branch | restored from the `text-image-aware` branch |

## Loader performance

The bottleneck is the **CPU dataloader**, not the GPU (GR-ConvNet has only 1.9M parameters).
One GA++ sample went from **20.5 ms to 4.7 ms** on a single core — **4.3× faster**. Measured on
16 cores with `cv2.setNumThreads(0)`, 416×416×3 images resized to 224×224, averaged over three
interleaved runs of both code paths.

| | Where | Before | After | Difference vs. the old code |
|---|---|---|---|---|
| P1 | `Image.resize` — `skimage.transform.resize` | 12.5 ms | **0.40 ms** | ≤ 1/255 on 0.02-0.4% of pixels (float32 rounding). Reproduces skimage's two steps in cv2: a gaussian filter with `σ=(ratio−1)/2`, then bilinear. Two easy mistakes: (a) the border-mode names do not correspond — skimage `reflect`→`BORDER_REFLECT_101`, `symmetric`→`BORDER_REFLECT`; (b) when **upscaling**, the outermost rows/columns do sample outside the border and `cv2.resize` always replicates it → up to **13/255** of error along the frame, so the upscaling path must use `warpAffine` with an explicit `borderMode`. Do not substitute `INTER_AREA`: it is off by 0.74/255 per pixel **and** slower (0.69 ms) |
| P2 | `Image.rotate` at multiples of 90° | 2.32 ms | **0.22 ms** | ≤ 1/255, in the better direction: `np.rot90` is an index permutation and therefore *exact*, whereas skimage interpolates and then rounds. Arbitrary angles → `cv2.warpAffine` (≤ 1/255) |
| P3 | `Image.from_file` — `imageio.imread` | 1.94 ms | **1.51 ms** | None (verified identical on JPEG). `cv2.imread` returns `None` on error instead of raising, so it is wrapped in a `FileNotFoundError` — otherwise training would silently run on black images |
| P4 | `GraspRectangles.draw` — `skimage.draw.polygon` per rect | 2.24 ms | **0.25 ms** | **2.2% of the `pos` area.** The geometry is vectorized (exact), but `cv2.fillConvexPoly` fills every pixel an edge crosses while skimage takes only pixels whose *centre* is inside; insetting the rect by 0.5 px compensates, leaving 2.2% of area and 0.2% of pixels changing which rect wins in `ang`/`width`. **This is the only change that touches the training labels** |
| P5 | 5-7 `.item()` calls per training step | each blocks the CPU until the GPU queue drains | `LossMeter` accumulates on the GPU (float64), syncing once per epoch | None |
| P6 | CLIP tokenization in the main process | 6.3 ms/batch on the critical path | `PromptTokenizer` runs in the DataLoader workers (0.042 ms/sample) | None — same tokenizer, same `max_length` |
| P7 | `validate()` hard-coded `batch_size=1` | one image per forward | `--val-batch-size` (default 32); post-processing and IoU stay per sample | None — verified at bs 1/5/N: identical `correct`/`failed`, loss differing by < 1e-8 |
| P8 | no AMP / channels_last / `cudnn.benchmark` | fp32, NCHW | `--amp auto` (bf16 where supported), `--channels-last auto`, `--cudnn-benchmark`, `--tf32` | **Yes, significant.** Use `--amp off` for pure fp32 |
| P9 | `rot` was coerced to float32 by `default_collate` when the batch's first element was `0` (an int) | lost 8 digits of the other angles in the batch | `float(random.choice(...))` in `GraspDatasetBase.__getitem__` | A bug fix. Invisible at `batch_size=1`, but it made `validate` build the GT with a different angle than the loader used to rotate the image |

After these changes the largest remaining per-sample cost is JPEG decoding (1.5 ms); reducing
it further means changing the on-disk data format, not the code.

Re-verify these transforms (no dataset, no GPU required):

```bash
python script/smoke_test_loader.py
```

To reproduce numbers from older runs exactly: set `AMP=off` and revert P4 (the `draw` hunk in
`utils/dataset_processing/grasp.py`). Everything else differs only at rounding level, or is
more accurate than before.

## Acknowledgement
Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
