"""Kiểm tra `grconvnet3_align` V2 chạy đúng -- không cần dataset, không cần GPU.

Chạy trước khi đẩy job train lên máy có GPU: bốn arm của bảng ablation (idea.md §6) đều phải
forward/backward được, checkpoint phải load lại ra đúng cùng một output, và nhánh ngôn ngữ
phải thật sự học được `part_mask` trên một bài toán đồ chơi.

    python script/smoke_test_align.py            # ~1 phút trên CPU
    python script/smoke_test_align.py --quick    # bỏ phần overfit

Cần CLIP `openai/clip-vit-base-patch32` trong cache HuggingFace (tải một lần, 250 MB).
"""
import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.models.grconvnet3_align import (CLIPTextEncoder,  # noqa: E402
                                               GenerativeResnetAlign)
from utils.checkpoint import load_network, save_checkpoint  # noqa: E402

# Bốn arm của idea.md §6, đúng cờ mà script/run_ablation.sh truyền vào.
ARMS = {
    'notext': dict(use_text=False),
    'hardmax': dict(use_text=True, align_mode='hard', region_text=False, fusion='gate'),
    'soft': dict(use_text=True, align_mode='soft', region_text=False, fusion='residual'),
    'full': dict(use_text=True, align_mode='soft', region_text=True, fusion='residual'),
}

PROMPTS = ["Grasp the mug at its handle.", "Lift the apple by its stem."]


def fake_batch(batch_size=2, size=224, seed=0):
    """(x, yc, part_mask) giống hệt thứ loader GA++ trả về, chỉ khác là ngẫu nhiên."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, size, size, generator=g)
    yc = [torch.rand(batch_size, 1, size, size, generator=g),
          torch.rand(batch_size, 1, size, size, generator=g) * 2 - 1,
          torch.rand(batch_size, 1, size, size, generator=g) * 2 - 1,
          torch.rand(batch_size, 1, size, size, generator=g)]
    part_mask = torch.zeros(batch_size, 1, size, size)
    part_mask[:, :, 40:90, 60:120] = 1.0          # ~3% pixel, đúng cỡ part thật
    return x, yc, part_mask


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        raise SystemExit(f"smoke test thất bại: {name}")


def build(arm, encoder, **extra):
    return GenerativeResnetAlign(input_channels=3, dropout=True, prob=0.1,
                                 text_encoder=encoder, **ARMS[arm], **extra)


def test_arms(encoder):
    print("\n1. bốn arm: forward + backward")
    x, yc, part_mask = fake_batch()
    for arm in ARMS:
        net = build(arm, encoder).train()
        prompts = PROMPTS if ARMS[arm]['use_text'] else None
        out = net.compute_loss(x, yc, prompts=prompts, part_mask=part_mask)
        out['loss'].backward()

        grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
        has_align = 'align_bce' in out['losses'] and 'align_dice' in out['losses']
        check(f"{arm}: loss hữu hạn", torch.isfinite(out['loss']).item(),
              f"loss={out['loss'].item():.4f}")
        check(f"{arm}: có gradient", bool(grads) and all(torch.isfinite(g).all() for g in grads))
        check(f"{arm}: align_bce/dice {'có' if arm != 'notext' else 'không'}",
              has_align == (arm != 'notext'))
        # CLIP phải đóng băng: một gradient nào đó ở text tower là lỗi thiết kế.
        frozen = all(p.grad is None for p in net.text_encoder.parameters()) \
            if ARMS[arm]['use_text'] else True
        check(f"{arm}: CLIP đóng băng", frozen)


def test_identity_at_init(encoder):
    print("\n2. residual fusion khởi tạo = identity (arm 'full' bắt đầu đúng bằng baseline)")
    x, _, _ = fake_batch()
    net = build('full', encoder).eval()
    with torch.no_grad():
        with_text = net(x, PROMPTS)[0]
        without = net(x, None)[0]
    check("F' = F ở bước 0", torch.allclose(with_text, without, atol=1e-6),
          f"max|Δ|={(with_text - without).abs().max().item():.2e}")


def test_align_stage(encoder):
    print("\n3. align_stage")
    x, yc, part_mask = fake_batch()
    for stage, expect in (('bottleneck', 56), ('conv4', 113)):
        net = build('full', encoder, align_stage=stage).eval()
        with torch.no_grad():
            align = net(x, PROMPTS)[4]
        check(f"{stage}: A_T ở {expect}×{expect}", align.shape[-1] == expect,
              str(tuple(align.shape)))
        loss = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
        check(f"{stage}: align_loss tính được",
              torch.isfinite(loss['losses']['align_bce'] + loss['losses']['align_dice']))


def test_degenerate_prompts(encoder):
    print("\n4. prompt suy biến (rỗng / toàn dấu câu) không làm NaN")
    x, yc, part_mask = fake_batch()
    net = build('full', encoder).train()
    out = net.compute_loss(x, yc, prompts=["", ". . ."], part_mask=part_mask)
    out['loss'].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check("loss hữu hạn", torch.isfinite(out['loss']).item(), f"loss={out['loss'].item():.4f}")
    check("gradient hữu hạn", all(torch.isfinite(g).all() for g in grads))


def test_checkpoint(encoder):
    print("\n5. checkpoint: nhỏ, và load lại ra đúng output")
    x, _, _ = fake_batch()
    net = build('full', encoder).eval()
    with torch.no_grad():
        before = net(x, PROMPTS)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ckpt')
        save_checkpoint(net, path)
        size_mb = os.path.getsize(path) / 1e6
        # V1 pickle cả CLIP text tower -> 265 MB. Phần học được chỉ vài MB.
        check("kích thước < 20 MB", size_mb < 20, f"{size_mb:.1f} MB")

        loaded = load_network(path, map_location='cpu', text_encoder=encoder)
        with torch.no_grad():
            after = loaded(x, PROMPTS)
        same = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, after))
        check("output khớp bit-đối-bit", same)
        check("kwargs được giữ", loaded.init_kwargs['align_mode'] == 'soft'
              and loaded.init_kwargs['region_text'] is True)


def test_prompt_sensitivity(encoder, steps=60):
    print(f"\n6. overfit một batch ({steps} bước): A_T phải bám part_mask, và đổi prompt phải "
          "đổi output")
    x, yc, part_mask = fake_batch()
    net = build('full', encoder).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-3)

    first = last = None
    for step in range(steps):
        out = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
        opt.zero_grad()
        out['loss'].backward()
        opt.step()
        value = (out['losses']['align_bce'] + out['losses']['align_dice']).item()
        first = value if step == 0 else first
        last = value
    check("align_loss giảm", last < first, f"{first:.4f} -> {last:.4f}")

    net.eval()
    with torch.no_grad():
        a = net.predict(x, PROMPTS)
        b = net.predict(x, ["Grasp the mug at its rim.", "Lift the apple by its skin."])
    delta = (a['pos'] - b['pos']).abs().mean().item()
    check("đổi prompt -> đổi Q", delta > 0, f"mean|ΔQ|={delta:.2e}")

    with torch.no_grad():
        iou = ((a['align'] > 0.5) & (part_mask[:, :, ::4, ::4] > 0.5)).sum() / \
            (((a['align'] > 0.5) | (part_mask[:, :, ::4, ::4] > 0.5)).sum() + 1e-6)
    print(f"       IoU(A_T > 0.5, part_mask) sau {steps} bước: {iou.item():.3f} "
          "(chỉ để tham khảo -- input là nhiễu)")


def test_stats(encoder):
    print("\n8. chẩn đoán: token/fusion stats + gradient theo khối")
    from utils.diagnostics import align_stats, grad_norm_split, grad_norms

    x, yc, part_mask = fake_batch()
    net = build('full', encoder).train()
    with net.collecting_stats():
        out = net.compute_loss(x, yc, prompts=PROMPTS, part_mask=part_mask)
    stats = out['stats']
    check("có token/fusion stats", 'token/attention_entropy' in stats
          and 'fusion/residual_ratio' in stats)
    check("entropy <= log(L)", stats['token/attention_entropy'] <= stats['token/max_entropy'] + 1e-5,
          f"H={stats['token/attention_entropy']:.3f} <= {stats['token/max_entropy']:.3f}")
    check("residual_ratio = 0 lúc khởi tạo", stats['fusion/residual_ratio'] == 0.0)
    check("token thắng cộng lại = 1",
          abs(float(stats['_winning_tokens'].sum(1).mean()) - 1.0) < 1e-4)

    a = align_stats(out['align_logits'], part_mask)
    check("align_stats đủ khoá",
          {'iou', 'dice', 'score_margin', 'predicted_area'} <= set(a),
          f"margin={a['score_margin']:+.4f}")

    out['loss'].backward()
    g = grad_norms(net)
    check("gradient theo khối", {'visual_encoder', 'grasp_decoder', 'fusion'} <= set(g),
          " ".join(f"{k}={v:.2e}" for k, v in sorted(g.items())))
    split = grad_norm_split(net, x, yc, PROMPTS, part_mask)
    check("tách được grad L_grasp và L_align", 'ratio' in split,
          f"grasp={split.get('grasp', 0):.2e} align={split.get('align', 0):.2e} "
          f"ratio={split.get('ratio', 0):.2f}")


def test_token_alignment(encoder):
    print("\n7. token_alignment() cho hình alignment")
    x, _, _ = fake_batch()
    net = build('full', encoder).eval()
    out = net.token_alignment(x, PROMPTS)
    check("bỏ SOT/EOT/pad/dấu câu",
          out['tokens'][0] == ['grasp', 'the', 'mug', 'at', 'its', 'handle'],
          str(out['tokens'][0]))
    check("có α theo token", out['weights'][0].shape[0] == len(out['tokens'][0]))
    check("α cộng lại bằng 1 tại mỗi pixel",
          torch.allclose(out['weights'][0].sum(0), torch.ones_like(out['weights'][0][0]),
                         atol=1e-4))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='bỏ phần overfit (mục 6)')
    args = ap.parse_args()

    torch.manual_seed(0)
    print("nạp CLIP text encoder (dùng chung cho mọi model bên dưới)...")
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
    print("\nTất cả đều qua.")


if __name__ == '__main__':
    main()
