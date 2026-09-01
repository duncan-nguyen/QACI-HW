"""Số đo để trả lời ba câu hỏi debug của nhánh text-visual alignment.

    1. Token nào đang được chọn?          -> token_stats()
    2. Vùng được chọn có đúng không?      -> align_stats()
    3. Grasp có đổi theo prompt không?    -> counterfactual (script/audit_text_reliance.py)

Cộng thêm hai nhóm chẩn đoán khi nghi ngờ nhánh ngôn ngữ "có mà như không":

    fusion_stats()  -- r_F = ||F' - F|| / ||F||, fusion có thật sự tác động lên feature không
    grad_norms()    -- gradient của L_grasp và L_align lên từng khối, để chỉnh lambda

Mọi hàm ở đây trả `float`/`dict[str, float]` thuần, không giữ tensor, để gọi trong vòng train
mà không giữ đồ thị hay chiếm bộ nhớ.
"""
import collections

import torch
import torch.nn.functional as F

# Nhóm tham số cho gradient/<group>. Khớp theo tiền tố tên trong state_dict.
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
    """Hạ target về đúng lưới của `like` (giống hệt thứ compute_loss dùng làm nhãn)."""
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)


@torch.no_grad()
def align_stats(align_logits, part_mask, threshold=0.5):
    """
    Chất lượng `A_T` so với `part_mask`, đo ở đúng resolution mà loss dùng.

    :param align_logits: (B, 1, H, W) logits của A_T
    :param part_mask: (B, 1, H', W') nhãn nhị phân
    :return: dict -- iou, dice, foreground_score, background_score, score_margin,
             predicted_area, target_area

    `score_margin` = mean(A_T | trong mask) - mean(A_T | ngoài mask). Gần 0 nghĩa là attention
    chưa phân biệt được foreground/background, kể cả khi BCE đã nhỏ (nhãn lệch ~2-5% nên
    "đoán toàn 0" đã cho BCE thấp).
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
    Phân bố alpha theo token: attention đang tập trung hay đang trải đều?

    :param attn: (B, L, H, W) alpha_pj (đã softmax theo token)
    :param text_mask: (B, L) bool, True ở token nội dung
    :return: dict -- attention_entropy, top1_mass, top3_mass, valid_count, max_entropy

    Cách đọc: entropy ≈ 0 là collapse vào một token; entropy ≈ log(L) là không token nào
    quan trọng; top1_mass ≈ 1 ngay từ đầu nghĩa là soft attention đang chạy y hệt hard max.
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
        # log(L_valid): mốc trên của entropy, để biết entropy quan sát được là cao hay thấp.
        'max_entropy': float(valid.clamp(min=1).log().mean()),
    }


@torch.no_grad()
def winning_tokens(attn, text_mask):
    """
    Token nào thắng ở nhiều pixel nhất, tính theo từng sample.

    :return: (B, L) tỉ lệ pixel mà token j là argmax_j alpha_pj
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
    r_F = ||F' - F|| / ||F||: fusion có thật sự đổi feature không.

    r_F ≈ 0 nghĩa là nhánh ngôn ngữ gần như không tác động (với residual fusion khởi tạo 0 thì
    đúng bằng 0 ở bước đầu -- phải thấy nó *lớn dần*). r_F >> 1 nghĩa là fusion lấn át hẳn
    feature thị giác.
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
            continue                # 'alignment' là nhóm bao, xét sau cùng
        if name.startswith(prefixes):
            return group
    if name.startswith(PARAM_GROUPS['alignment']):
        return 'alignment'
    return None


def grad_norms(net):
    """
    Chuẩn L2 của gradient đang có trên từng khối (gọi *sau* loss.backward()).

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
    Gradient của L_grasp và L_align *riêng rẽ* lên một khối, trên cùng một batch.

    Đây là con số quyết định `--w-align`: nếu ‖∇L_align‖ lớn hơn ‖∇L_grasp‖ hàng chục lần thì
    nhánh ngôn ngữ đang kéo hỏng nhánh grasp, phải giảm lambda (hoặc kéo dài warmup).

    Dùng `torch.autograd.grad` nên **không** đụng vào `.grad` của optimizer.

    :return: dict -- grasp, align, ratio (align/grasp)
    """
    params = [p for name, p in net.named_parameters()
              if p.requires_grad and _group_of(name) == group]
    if not params:
        return {}

    was_training = net.training
    net.eval()                      # tắt dropout để hai lần đo dùng chung một đồ thị xác định
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
    Gom "token nào thắng ở bao nhiêu phần trăm pixel" trên cả tập validation.

    Bảng này đọc rất nhanh ra bệnh: nếu top toàn là danh từ object (`apple`, `mug`) thì model
    đang định vị *object* chứ không phải *part* -- đúng thứ hình alignment của run V1 cho thấy.
    Dấu câu phải bằng 0 tuyệt đối; khác 0 nghĩa là mask token bị hỏng.
    """

    def __init__(self):
        self.mass = collections.Counter()
        self.total = 0.0

    def update(self, winning, text_mask, token_strings):
        """
        :param winning: (B, L) tỉ lệ pixel thắng, từ winning_tokens()
        :param text_mask: (B, L) bool
        :param token_strings: list[list[str]] cùng chiều, từ CLIPTextEncoder.token_strings()
        """
        for b in range(winning.shape[0]):
            for j in text_mask[b].nonzero(as_tuple=True)[0].tolist():
                token = token_strings[b][j].replace('</w>', '')
                self.mass[token] += float(winning[b, j])
            self.total += 1.0

    def top(self, k=20):
        """[(token, tỉ lệ pixel trung bình)] giảm dần."""
        if not self.total:
            return []
        return [(t, m / self.total) for t, m in self.mass.most_common(k)]

    def punctuation_mass(self):
        """Phải bằng 0: token dấu câu đã bị loại khỏi candidate ở CLIPTextEncoder."""
        if not self.total:
            return 0.0
        return sum(m for t, m in self.mass.items()
                   if not any(ch.isalnum() for ch in t)) / self.total

    def as_text(self, k=20):
        lines = [f"{t}: {p:.1%}" for t, p in self.top(k)]
        return "\n".join(lines) if lines else "(chưa có dữ liệu)"
