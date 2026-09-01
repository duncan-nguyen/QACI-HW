# Spatial Text Alignment for Language-Driven Grasp Detection (STAG)

A fork of the [Grasp-Anything](https://airvlab.github.io/grasp-anything/) codebase adding
**STAG**, a single-pass model for part-level language-driven grasp detection on
Grasp-Anything++ (GA++).

STAG computes token–region relevance between a frozen CLIP text encoder and the GR-ConvNet
feature map, supervises that spatial map with the annotated target-part mask, and conditions
all four grasp outputs on the map and on a region-specific text feature.

## Method


Tokens are selected per pixel by a softmax rather than an argmax (punctuation excluded). The
last conv of `φ` is zero-initialized, so at step 0 `F' = F` is exactly the image-only backbone
and language phases in as `A_T` becomes meaningful. `A_T` is supervised at its own 56×56
resolution — a target that is *not* a rasterization of the grasp rectangle, so the language
branch does not merely relearn what the grasp branch already knows.

All arms are the same file ([inference/models/stag.py](inference/models/stag.py)) on the same
training budget; only flags change:

| arm | `--use-text` | `--w-align` | `--align-mode` | `--region-text` | `--fusion` |
|---|---|---|---|---|---|
| **STAG (full)** | 1 | 0.3 | `soft` | 1 | `residual` |
| w/o `L_align` | 1 | 0 | `soft` | 1 | `residual` |
| w/o Text | 0 | 0 | – | – | – |

## Results

Independent-test grasp success; same data, split, seed and budget for every arm. Success =
rectangle IoU > 0.25 **and** angular error ≤ 30°.

| Method | Seen | Unseen | H |
|---|---|---|---|
| **STAG** | **0.354** | **0.222** | **0.273** |
| w/o `L_align` | 0.344 | 0.179 | 0.235 |
| w/o Text | 0.278 | 0.171 | 0.212 |

The full model leads by 7.6 / 5.1 / 6.1 points over the image-only baseline. Its 3.8-point `H`
gap over the unsupervised-alignment arm is larger on Unseen than Seen (4.3 vs 1.0), consistent
with mask supervision helping more on new categories — but **one seed is not enough to
establish a robust generalization advantage**.

## Setup

```bash
conda create -n granything python=3.9 && conda activate granything
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

GA++ ships only labels and language; images live in the base Grasp-Anything repo and the full
extraction is ~830 GB. [script/build_ga_pp_subset.py](script/build_ga_pp_subset.py) builds a
subset (100k scenes ≈ 445k samples ≈ 18 GB with `--pack-masks`), and
[split/build_grasp_anything_pp.py](split/build_grasp_anything_pp.py) builds the seen/unseen
split following the LGD protocol — balanced by sample count rather than category count, with
scenes shared between Base and New dropped from training. Both docstrings explain why.

## Training and testing

```bash
python train_network.py --dataset grasp-anything-pp \
    --dataset-path data/grasp-anything-pp-full --split-path split/grasp-anything-pp \
    --network stag --use-depth 0 --use-rgb 1 \
    --use-text 1 --w-align 0.3 --align-mode soft --region-text 1 --fusion residual \
    --warmup-epochs 3 --description stag-full

python evaluate.py --network <checkpoint> --dataset grasp-anything-pp \
    --dataset-path data/grasp-anything-pp-full --iou-eval --subset test
```

Checkpoints land in `logs/<timestamp>_<description>` as `state_dict` + kwargs without the CLIP
text tower (~9 MB, not 265 MB); the architecture is detected on load. `--subset test` needs the
same `--split`/`--test-split`/`--random-seed` as training. Upstream baselines still work via
`--network ggcnn` etc. on the original datasets.

**Reproducing the reported numbers:** a 100K-scene subset (441,298 part-level samples) at
224×224, 640K sampled training instances per arm (20 × 250 × 128), Adam, lr 2×10⁻³, cosine,
seed 123, 0.5% held out for test. [notebooks/train_ga_pp.ipynb](notebooks/train_ga_pp.ipynb)
runs exactly that on Colab. [script/run_paper_setting.sh](script/run_paper_setting.sh) drives
the same pipeline on one GPU, but defaults to a larger budget — override `EPOCHS`,
`BATCHES_PER_EPOCH`, `BATCH_SIZE`, `LR`. `script/run_ablation.sh` runs all arms sequentially.
Note an "epoch" here is `BATCHES_PER_EPOCH` batches, not a pass over the data; what must match
across arms is the total sample draws.

## Diagnostics

| | |
|---|---|
| `script/smoke_test_align.py` | every arm + checkpoint round-trip, no dataset or GPU |
| `script/smoke_test_loader.py` | image/label transforms vs. the skimage reference |
| `script/check_dataset.py` | is `part_mask` really part-level? If most same-object pairs have IoU > 0.9 it is not, and `L_align` cannot teach part grounding |
| `script/audit_text_reliance.py` | real vs. shuffled vs. fixed vs. other-part prompt. `Δ_shuffled ≈ 0` means the grasp ignores the prompt |
| `script/export_results.py` | redraw figures without retraining |

During training, TensorBoard carries `align/score_margin` (fg−bg separation of `A_T`),
`token/*` (attention entropy, winning tokens, `punctuation_mass` must be 0),
`fusion/residual_ratio` (0 at init by construction — it must *grow*),
`gradient/visual_from_{grasp,align}` (for tuning `--w-align`) and `counterfactual/prompt_drop`.
Among the figures, `prompt_grid.png` comes first: three identical rows mean the model is
ignoring the language, however low `align_loss` got.

## Fixes in this fork

Beyond making the upstream code run on modern numpy/torch, several bugs produced *wrong
numbers* rather than crashes: the seen/unseen split left 0.1% of samples for New and shared
scenes across both sets; the "test set" coincided with the checkpoint-selection set;
validation ran with augmentation on, so cropped-away grasps counted as automatic failures;
the periodic save branch lowered `best_iou`, producing spurious "best" checkpoints; and width
was decoded with a constant valid only for the original 300×300 net, making grasps 1.34× too
long at input 224. Each fix is documented in the docstring at its site.

The loader was also the real bottleneck, not the GPU — one sample went from 20.5 ms to 4.7 ms
by replacing skimage with cv2. Only two changes alter results: `GraspRectangles.draw` shifts
2.2% of the `pos` area (the only change touching training labels), and AMP (`--amp off`
restores fp32).

## Acknowledgement

Our codebase is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
