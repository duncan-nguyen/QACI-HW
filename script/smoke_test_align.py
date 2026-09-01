"""Check that `stag` works -- no dataset, no GPU required.

Run this before submitting a training job to a GPU machine: every arm of the ablation table
must forward and backward, a checkpoint must reload to exactly the same output, and the
language branch must actually be able to learn `part_mask` on a toy problem.

    python script/smoke_test_align.py            # ~1 minute on CPU
    python script/smoke_test_align.py --quick    # skip the overfitting test

Requires CLIP `openai/clip-vit-base-patch32` in the HuggingFace cache (a one-off 250 MB
download).
"""

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.models.stag import STAG, CLIPTextEncoder
from utils.checkpoint import load_network, save_checkpoint

# The ablation arms, with exactly the flags script/run_ablation.sh passes.
ARMS = {
    "notext": dict(use_text=False),
    "hardmax": dict(use_text=True, align_mode="hard", region_text=False, fusion="gate"),
    "soft": dict(
        use_text=True, align_mode="soft", region_text=False, fusion="residual"
    ),
    "full": dict(use_text=True, align_mode="soft", region_text=True, fusion="residual"),
}

PROMPTS = ["Grasp the mug at its handle.", "Lift the apple by its stem."]


def fake_batch(batch_size=2, size=224, seed=0):
    """(x, yc, part_mask) shaped exactly like the GA++ loader's output, but random."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, size, size, generator=g)
    yc = [
        torch.rand(batch_size, 1, size, size, generator=g),
        torch.rand(batch_size, 1, size, size, generator=g) * 2 - 1,
        torch.rand(batch_size, 1, size, size, generator=g) * 2 - 1,
        torch.rand(batch_size, 1, size, size, generator=g),
    ]
    part_mask = torch.zeros(batch_size, 1, size, size)
    part_mask[:, :, 40:90, 60:120] = 1.0  # ~3% of pixels, the size of a real part
    return x, yc, part_mask


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        raise SystemExit(f"smoke test failed: {name}")


def build(arm, encoder, **extra):
    return STAG(
        input_channels=3,
        dropout=True,
        prob=0.1,
        text_encoder=encoder,
        **ARMS[arm],
        **extra,
    )


def test_arms(encoder):
    print("\n1. all arms: forward + backward")
    x, yc, part_mask = fake_batch()
    for arm in ARMS:
        net = build(arm, encoder).train()
        prompts = PROMPTS if ARMS[arm]["use_text"] else None
        out = net.compute_loss(x, yc, prompts=prompts, part_mask=part_mask)
        out["loss"].backward()

        grads = [
            p.grad for p in net.parameters() if p.requires_grad and p.grad is not None
        ]
        has_align = "align_bce" in out["losses"] and "align_dice" in out["losses"]
        check(
            f"{arm}: finite loss",
            torch.isfinite(out["loss"]).item(),
            f"loss={out['loss'].item():.4f}",
        )
        check(
            f"{arm}: gradients present",
            bool(grads) and all(torch.isfinite(g).all() for g in grads),
        )
        check(
            f"{arm}: align_bce/dice {'present' if arm != 'notext' else 'absent'}",
            has_align == (arm != "notext"),
        )
        # CLIP must stay frozen: any gradient in the text tower is a design bug.
        frozen = (
            all(p.grad is None for p in net.text_encoder.parameters())
            if ARMS[arm]["use_text"]
            else True
        )
        check(f"{arm}: CLIP frozen", frozen)


def test_identity_at_init(encoder):
    print(
        "\n2. residual fusion is the identity at init ('full' starts exactly at the baseline)"
    )
    x, _, _ = fake_batch()
    net = build("full", encoder).eval()
    with torch.no_grad():
        with_text = net(x, PROMPTS)[0]
        without = net(x, None)[0]
    check(
        "F' = F at step 0",
        torch.allclose(with_text, without, atol=1e-6),
        f"max|Δ|={(with_text - without).abs().max().item():.2e}",
    )


def test_align_stage(encoder):
    print("\n3. align_stage")
    x, yc, part_mask = fake_batch()
    for stage, expect in (("bottleneck", 56), ("conv4", 113)):
        net = build("full", encoder, align_stage=stage).eval()
        with torch.no_grad():
            align = net(x, PROMPTS)[4]
        check(
            f"{stage}: A_T at {expect}x{expect}",
            align.shape[-1] == expect,
            str(tuple(align.shape)),
        )
        loss = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
        check(
            f"{stage}: align_loss computable",
            torch.isfinite(loss["losses"]["align_bce"] + loss["losses"]["align_dice"]),
        )


def test_degenerate_prompts(encoder):
    print("\n4. degenerate prompts (empty / all punctuation) do not produce NaN")
    x, yc, part_mask = fake_batch()
    net = build("full", encoder).train()
    out = net.compute_loss(x, yc, prompts=["", ". . ."], part_mask=part_mask)
    out["loss"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check(
        "finite loss",
        torch.isfinite(out["loss"]).item(),
        f"loss={out['loss'].item():.4f}",
    )
    check("finite gradients", all(torch.isfinite(g).all() for g in grads))


def test_checkpoint(encoder):
    print("\n5. checkpoint: small, and reloads to the same output")
    x, _, _ = fake_batch()
    net = build("full", encoder).eval()
    with torch.no_grad():
        before = net(x, PROMPTS)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")
        save_checkpoint(net, path)
        size_mb = os.path.getsize(path) / 1e6
        # Pickling the whole CLIP text tower would give 265 MB. The learned part is a few MB.
        check("size < 20 MB", size_mb < 20, f"{size_mb:.1f} MB")

        loaded = load_network(path, map_location="cpu", text_encoder=encoder)
        with torch.no_grad():
            after = loaded(x, PROMPTS)
        same = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, after))
        check("outputs match", same)
        check(
            "kwargs preserved",
            loaded.init_kwargs["align_mode"] == "soft"
            and loaded.init_kwargs["region_text"] is True,
        )


def test_prompt_sensitivity(encoder, steps=60):
    print(
        f"\n6. overfit one batch ({steps} steps): A_T must track part_mask, and changing the "
        "prompt must change the output"
    )
    x, yc, part_mask = fake_batch()
    net = build("full", encoder).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-3)

    first = last = None
    for step in range(steps):
        out = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        value = (out["losses"]["align_bce"] + out["losses"]["align_dice"]).item()
        first = value if step == 0 else first
        last = value
    check("align_loss decreases", last < first, f"{first:.4f} -> {last:.4f}")

    net.eval()
    with torch.no_grad():
        a = net.predict(x, PROMPTS)
        b = net.predict(x, ["Grasp the mug at its rim.", "Lift the apple by its skin."])
    delta = (a["pos"] - b["pos"]).abs().mean().item()
    check("changing the prompt changes Q", delta > 0, f"mean|dQ|={delta:.2e}")

    with torch.no_grad():
        iou = ((a["align"] > 0.5) & (part_mask[:, :, ::4, ::4] > 0.5)).sum() / (
            ((a["align"] > 0.5) | (part_mask[:, :, ::4, ::4] > 0.5)).sum() + 1e-6
        )
    print(
        f"       IoU(A_T > 0.5, part_mask) after {steps} steps: {iou.item():.3f} "
        "(indicative only -- the input is noise)"
    )


def test_stats(encoder):
    print("\n8. diagnostics: token/fusion stats + per-block gradients")
    from utils.diagnostics import align_stats, grad_norm_split, grad_norms

    x, yc, part_mask = fake_batch()
    net = build("full", encoder).train()
    with net.collecting_stats():
        out = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
    stats = out["stats"]
    check(
        "token/fusion stats present",
        "token/attention_entropy" in stats and "fusion/residual_ratio" in stats,
    )
    check(
        "entropy <= log(L)",
        stats["token/attention_entropy"] <= stats["token/max_entropy"] + 1e-5,
        f"H={stats['token/attention_entropy']:.3f} <= {stats['token/max_entropy']:.3f}",
    )
    check("residual_ratio = 0 at init", stats["fusion/residual_ratio"] == 0.0)
    check(
        "winning-token shares sum to 1",
        abs(float(stats["_winning_tokens"].sum(1).mean()) - 1.0) < 1e-4,
    )

    a = align_stats(out["align_logits"], part_mask)
    check(
        "align_stats has all keys",
        {"iou", "dice", "score_margin", "predicted_area"} <= set(a),
        f"margin={a['score_margin']:+.4f}",
    )

    out["loss"].backward()
    g = grad_norms(net)
    check(
        "per-block gradients",
        {"visual_encoder", "grasp_decoder", "fusion"} <= set(g),
        " ".join(f"{k}={v:.2e}" for k, v in sorted(g.items())),
    )
    split = grad_norm_split(net, x, yc, PROMPTS, part_mask)
    check(
        "L_grasp and L_align gradients separable",
        "ratio" in split,
        f"grasp={split.get('grasp', 0):.2e} align={split.get('align', 0):.2e} "
        f"ratio={split.get('ratio', 0):.2f}",
    )


def test_token_alignment(encoder):
    print("\n7. token_alignment() for the alignment figures")
    x, _, _ = fake_batch()
    net = build("full", encoder).eval()
    out = net.token_alignment(x, PROMPTS)
    check(
        "SOT/EOT/pad/punctuation dropped",
        out["tokens"][0] == ["grasp", "the", "mug", "at", "its", "handle"],
        str(out["tokens"][0]),
    )
    check(
        "per-token alpha present", out["weights"][0].shape[0] == len(out["tokens"][0])
    )
    check(
        "alpha sums to 1 at every pixel",
        torch.allclose(
            out["weights"][0].sum(0), torch.ones_like(out["weights"][0][0]), atol=1e-4
        ),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--quick", action="store_true", help="skip the overfitting test (item 6)"
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    print("loading the CLIP text encoder (shared by every model below)...")
    encoder = CLIPTextEncoder()

    test_arms(encoder)
    test_identity_at_init(encoder)
    test_align_stage(encoder)
    test_degenerate_prompts(encoder)
    test_checkpoint(encoder)
    if not args.quick:
        test_prompt_sensitivity(encoder)
    test_token_alignment(encoder)
    test_stats(encoder)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
