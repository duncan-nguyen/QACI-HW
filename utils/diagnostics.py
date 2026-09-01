"""Metrics answering the three debugging questions of the text-visual alignment branch.

    1. Which tokens are being selected?   -> token_stats()
    2. Is the selected region correct?    -> align_stats()
    3. Does the grasp follow the prompt?  -> counterfactual (script/audit_text_reliance.py)

Plus two more groups for when the language branch looks present but inert:

    fusion_stats()  -- r_F = ||F' - F|| / ||F||, does fusion actually affect the features
    grad_norms()    -- per-block gradients of L_grasp and L_align, for tuning lambda

Every function here returns plain `float`/`dict[str, float]` and holds no tensors, so it can be
called inside the training loop without retaining the graph or leaking memory.
"""
import collections

import torch
import torch.nn.functional as F

# Parameter groups for gradient/<group>. Matched by name prefix in the state_dict.
PARAM_GROUPS = {
    'visual_encoder': ('conv1.', 'bn1.', 'conv2.', 'bn2.', 'conv3.', 'bn3.',
                       'res1.', 'res2.', 'res3.', 'res4.', 'res5.'),
    'text_projection': ('align.t_proj.', 'align.r_proj.'),
    'alignment': ('align.',),
    'fusion': ('fuse.',),
    'grasp_decoder': ('conv4.', 'bn4.', 'conv5.', 'bn5.', 'conv6.',
                      'pos_output.', 'cos_output.', 'sin_output.', 'width_output.'),
}


def match_resolution(target, like):
    """Downsample the target onto `like`'s grid (exactly what compute_loss uses as label)."""
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)


@torch.no_grad()
def align_stats(align_logits, part_mask, threshold=0.5):
    """
    Quality of `A_T` against `part_mask`, measured at the resolution the loss uses.

    :param align_logits: (B, 1, H, W) logits of A_T
    :param part_mask: (B, 1, H', W') binary label
    :return: dict -- iou, dice, foreground_score, background_score, score_margin,
             predicted_area, target_area

    `score_margin` = mean(A_T | inside mask) - mean(A_T | outside mask). Near 0 means attention
    does not yet separate foreground from background, even when BCE is already low (the label
    covers only ~2-5% of pixels, so predicting all-zero already gives a low BCE).
    """
    prob = torch.sigmoid(align_logits)
    target = match_resolution(part_mask, align_logits) > 0.5
    pred = prob > threshold

    inter = (pred & target).flatten(1).sum(1).float()
    union = (pred | target).flatten(1).sum(1).float()
    p_sum = pred.flatten(1).sum(1).float()
    t_sum = target.flatten(1).sum(1).float()

    fg = torch.where(t_sum > 0, (prob.flatten(1) * target.flatten(1)).sum(1) / t_sum.clamp(min=1),
                     torch.zeros_like(t_sum))
    bg_count = (~target).flatten(1).sum(1).float()
    bg = torch.where(bg_count > 0,
                     (prob.flatten(1) * (~target).flatten(1)).sum(1) / bg_count.clamp(min=1),
                     torch.zeros_like(bg_count))

    n = align_logits[0].numel()
    return {
        'iou': float((inter / union.clamp(min=1)).mean()),
        'dice': float((2 * inter / (p_sum + t_sum).clamp(min=1)).mean()),
        'foreground_score': float(fg.mean()),
        'background_score': float(bg.mean()),
        'score_margin': float((fg - bg).mean()),
        'predicted_area': float((p_sum / n).mean()),
        'target_area': float((t_sum / n).mean()),
    }


@torch.no_grad()
def token_stats(attn, text_mask):
    """
    Distribution of alpha over tokens: is attention concentrated or spread out?

    :param attn: (B, L, H, W) alpha_pj (already softmaxed over tokens)
    :param text_mask: (B, L) bool, True on content tokens
    :return: dict -- attention_entropy, top1_mass, top3_mass, valid_count, max_entropy

    How to read it: entropy ~= 0 means collapse onto a single token; entropy ~= log(L) means no
    token matters; top1_mass ~= 1 from the start means soft attention is behaving exactly like
    a hard argmax.
    """
    valid = text_mask.float().sum(1)                                   # (B,)
    p = attn.clamp(min=1e-12)
    entropy = -(attn * p.log()).sum(1)                                 # (B, H, W)

    sorted_attn = attn.sort(dim=1, descending=True).values
    top1 = sorted_attn[:, 0]
    top3 = sorted_attn[:, :3].sum(1)

    return {
        'attention_entropy': float(entropy.mean()),
        'top1_mass': float(top1.mean()),
        'top3_mass': float(top3.mean()),
        'valid_count': float(valid.mean()),
        # log(L_valid): the upper bound on entropy, telling whether the observed value is high or low.
        'max_entropy': float(valid.clamp(min=1).log().mean()),
    }


@torch.no_grad()
def winning_tokens(attn, text_mask):
    """
    Which token wins the most pixels, per sample.

    :return: (B, L) the fraction of pixels where token j is argmax_j alpha_pj
    """
    winner = attn.argmax(dim=1)                                        # (B, H, W)
    b, l = text_mask.shape
    counts = torch.zeros(b, l, device=attn.device)
    counts.scatter_add_(1, winner.flatten(1), torch.ones_like(winner.flatten(1),
                                                              dtype=counts.dtype))
    return counts / winner[0].numel()


@torch.no_grad()
def fusion_stats(feat_in, feat_out):
    """
    r_F = ||F' - F|| / ||F||: does fusion actually change the features.

    r_F ~= 0 means the language branch has almost no effect (with zero-initialised residual
    fusion it is exactly 0 at the first step -- what matters is seeing it *grow*). r_F >> 1
    means fusion is overwhelming the visual features.
    """
    fn = feat_in.flatten(1).norm(dim=1)
    rn = (feat_out - feat_in).flatten(1).norm(dim=1)
    return {
        'feature_norm': float(fn.mean()),
        'residual_norm': float(rn.mean()),
        'residual_ratio': float((rn / fn.clamp(min=1e-8)).mean()),
    }


def _group_of(name):
    for group, prefixes in PARAM_GROUPS.items():
        if group == 'alignment':
            continue                # 'alignment' is the catch-all group, checked last
        if name.startswith(prefixes):
            return group
    if name.startswith(PARAM_GROUPS['alignment']):
        return 'alignment'
    return None


def grad_norms(net):
    """
    L2 norm of the current gradient per block (call *after* loss.backward()).

    :return: dict group -> norm
    """
    sums = {}
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        group = _group_of(name)
        if group is None:
            continue
        sums[group] = sums.get(group, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: v ** 0.5 for k, v in sums.items()}


def grad_norm_split(net, xc, yc, prompts, part_mask, group='visual_encoder'):
    """
    Gradients of L_grasp and L_align *separately* on one block, over the same batch.

    This is the number that decides `--w-align`: if ||grad L_align|| is tens of times larger
    than ||grad L_grasp||, the language branch is wrecking the grasp branch and lambda must come
    down (or the warmup be lengthened).

    Uses `torch.autograd.grad`, so it does **not** touch the optimizer's `.grad`.

    :return: dict -- grasp, align, ratio (align/grasp)
    """
    params = [p for name, p in net.named_parameters()
              if p.requires_grad and _group_of(name) == group]
    if not params:
        return {}

    was_training = net.training
    net.eval()                      # dropout off, so both measurements share a deterministic graph
    try:
        out = net.compute_loss(xc, yc, prompts=prompts, part_mask=part_mask)
        grasp = sum(out['losses'][k] for k in ('p_loss', 'cos_loss', 'sin_loss', 'width_loss'))
        align = out['losses'].get('align_bce', None)
        if align is not None:
            align = align + out['losses']['align_dice']

        g_grasp = torch.autograd.grad(grasp, params, retain_graph=align is not None,
                                      allow_unused=True)
        result = {'grasp': _norm(g_grasp)}
        if align is not None:
            g_align = torch.autograd.grad(align, params, allow_unused=True)
            result['align'] = _norm(g_align)
            result['ratio'] = result['align'] / max(result['grasp'], 1e-12)
    finally:
        net.train(was_training)
    return result


def _norm(grads):
    total = sum(float(g.detach().pow(2).sum()) for g in grads if g is not None)
    return total ** 0.5


class TokenTally:
    """
    Aggregates "which token wins what percentage of pixels" over the whole validation set.

    The table diagnoses quickly: if the top entries are all object nouns (`apple`, `mug`), the
    model is localising the *object* rather than the *part*. Punctuation must be exactly 0;
    anything else means the token mask is broken.
    """

    def __init__(self):
        self.mass = collections.Counter()
        self.total = 0.0

    def update(self, winning, text_mask, token_strings):
        """
        :param winning: (B, L) fraction of pixels won, from winning_tokens()
        :param text_mask: (B, L) bool
        :param token_strings: list[list[str]] of matching shape, from
                              CLIPTextEncoder.token_strings()
        """
        for b in range(winning.shape[0]):
            for j in text_mask[b].nonzero(as_tuple=True)[0].tolist():
                token = token_strings[b][j].replace('</w>', '')
                self.mass[token] += float(winning[b, j])
            self.total += 1.0

    def top(self, k=20):
        """[(token, mean pixel fraction)] in descending order."""
        if not self.total:
            return []
        return [(t, m / self.total) for t, m in self.mass.most_common(k)]

    def punctuation_mass(self):
        """Must be 0: punctuation tokens are excluded from the candidates in CLIPTextEncoder."""
        if not self.total:
            return 0.0
        return sum(m for t, m in self.mass.items()
                   if not any(ch.isalnum() for ch in t)) / self.total

    def as_text(self, k=20):
        lines = [f"{t}: {p:.1%}" for t, p in self.top(k)]
        return "\n".join(lines) if lines else "(no data yet)"
