"""GR-ConvNet-3 với text-visual alignment ở mức part (language-driven grasping).

Ý tưởng: prompt của Grasp-Anything++ nhắm vào một *part* cụ thể ("grasp the mug at its
handle"), nên tín hiệu ngôn ngữ phải chọn *vùng* trên ảnh chứ không chỉ đổi output cuối. Ta
tính một bản đồ alignment token-pixel `A_T` ngay tại bottleneck rồi dùng nó gate feature,
và supervise `A_T` **tường minh** bằng `part_mask` của GA++ -- nhãn khác loại với grasp
rectangle, nên nhánh ngôn ngữ không chỉ học lại đúng thứ mà nhánh grasp đã học.

    prompt ──► CLIP text (frozen) ──► t_1..t_L
                                          │
    ảnh ────► conv1..res5 ──► F (56×56) ──┤
                                          ▼
                        A_T = σ( max_j cos(W_v F_xy, W_t t_j) / τ )
                                          │
                        F' = F ⊙ (1 + λ·A_T)  ──► conv4..conv6 ──► pos/cos/sin/width
                                          │
                        Q_g = head_g(F)  (graspability, không điều kiện text)

    L = L_grasp + λ1·BCE(Q_g, M_union) + λ2·[BCE + Dice](A_T, part_mask)
"""

import logging
import re
import string

import torch
import torch.nn.functional as F
from torch import nn

from inference.models.grconvnet3 import GenerativeResnet

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
MAX_PROMPT_TOKENS = 16  # prompt GA++ dài nhất ~11 từ, 77 của CLIP là phí
# Thay cho -inf ở token bị mask khi cần logsumexp: -inf làm gradient thành NaN.
NEG_INF = -1e4


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder đóng băng, trả embedding **từng token** (không phải một vector EOT).

    `CLIPTextModel.last_hidden_state` cho cả chuỗi; ta chiếu qua `text_projection` để về cùng
    không gian với image embedding của CLIP, rồi mask bỏ SOT/EOT/padding.
    """

    def __init__(self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS):
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        from transformers import logging as hf_logging

        # Checkpoint CLIP chứa cả vision tower; ta chỉ lấy text tower nên transformers sẽ liệt
        # kê một loạt key "UNEXPECTED" -- đúng như mong đợi, không cần in ra.
        hf_logging.set_verbosity_error()
        # train_network.py bật basicConfig(INFO) toàn cục, kéo theo log HTTP của hub.
        for name in ("httpx", "huggingface_hub", "filelock"):
            logging.getLogger(name).setLevel(logging.WARNING)

        self.max_length = max_length
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Prompt dài hơn max_length bị cắt: khi đó chuỗi *không còn EOT*, và dòng mask EOT ở
        # forward() sẽ bỏ nhầm một token nội dung. Im lặng thì không ai biết -- nên đếm.
        self._n_prompts = 0
        self._n_truncated = 0
        self._warned_truncation = False

    def truncation_report(self):
        """(số prompt đã encode, số bị cắt, tỉ lệ) -- gọi sau vài batch đầu là đủ biết."""
        rate = self._n_truncated / self._n_prompts if self._n_prompts else 0.0
        return self._n_prompts, self._n_truncated, rate

    @property
    def dim(self):
        return self.model.config.projection_dim

    def train(self, mode=True):
        # Giữ text encoder ở eval kể cả khi net.train() -- nó đóng băng.
        super().train(mode)
        self.model.eval()
        return self

    def token_strings(self, prompts):
        """
        Chuỗi token thật sự đưa vào encoder, để gán nhãn cho hình alignment.

        :return: list[list[str]] cùng chiều dài với chiều L của forward() (kể cả SOT/EOT/pad,
                 dùng chung mask với forward để lọc).
        """
        tok = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return [
            [self.tokenizer.convert_ids_to_tokens(i) for i in ids]
            for ids in tok["input_ids"]
        ]

    def part_token_mask(self, input_ids, content_mask, prompts):
        """
        (B, L) bool: token nào thuộc cụm từ chỉ part của prompt.

        Dùng cho `L_token` (Update 3) -- số hạng duy nhất nói cho model biết token *nào* phải
        thắng, thay vì chỉ giám sát giá trị `max` như `L_align`.

        Token là mảnh BPE nên so bằng "nằm trong": `bbed` ⊂ `webbed`. Prompt không khớp mẫu
        "... its <part>." trả về hàng toàn False và sẽ bị `L_token` bỏ qua.
        """
        out = torch.zeros_like(content_mask)
        for b, prompt in enumerate(prompts):
            words = _part_words(prompt)
            if not words:
                continue
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            for j, tk in enumerate(tokens):
                if not content_mask[b, j]:
                    continue
                clean = _clean_token(tk)
                if clean and any(clean in w for w in words):
                    out[b, j] = True
        return out

    @torch.no_grad()
    def forward(self, prompts, want_part_tokens=False):
        """
        :param prompts: list[str] độ dài B
        :param want_part_tokens: trả thêm mask token thuộc cụm chỉ part (đắt: cần detokenize)
        :return: (B, L, D) embedding từng token, (B, L) mask token nội dung,
                 (B, L) mask token chỉ part hoặc None
        """
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Đếm truncate *trước* khi chuyển sang GPU: làm sau sẽ ép đồng bộ device mỗi batch.
        # Bị cắt <=> dùng hết max_length mà token cuối không phải EOT.
        n_trunc = int(
            (
                (tok["attention_mask"].sum(dim=1) == self.max_length)
                & (tok["input_ids"][:, -1] != self.tokenizer.eos_token_id)
            ).sum()
        )
        self._n_prompts += tok["input_ids"].shape[0]
        self._n_truncated += n_trunc
        if n_trunc and not self._warned_truncation:
            self._warned_truncation = True
            logging.warning(
                "CLIPTextEncoder: %d prompt trong batch này vượt MAX_PROMPT_TOKENS=%d và bị "
                "cắt -- chuỗi mất EOT nên mask EOT bỏ nhầm một token nội dung. Tăng "
                "MAX_PROMPT_TOKENS.",
                n_trunc,
                self.max_length,
            )

        # Mask token nội dung dựng trên CPU: bỏ SOT (luôn ở vị trí 0) và EOT (token cuối cùng
        # còn attend được). Làm ở đây để `part_token_mask` đọc được mà không đồng bộ device.
        am = tok["attention_mask"]
        mask = am.bool().clone()
        mask[:, 0] = False
        eot = am.sum(dim=1) - 1
        mask[torch.arange(mask.shape[0]), eot] = False

        part_tok = None
        if want_part_tokens:
            part_tok = self.part_token_mask(tok["input_ids"], mask, prompts)

        tok = tok.to(device)
        out = self.model.text_model(**tok).last_hidden_state  # (B, L, hidden)
        emb = self.model.text_projection(out)  # (B, L, D)

        return (
            emb,
            mask.to(device),
            part_tok.to(device) if part_tok is not None else None,
        )


class TokenPixelAlignment(nn.Module):
    """
    Cosine similarity giữa mọi token văn bản và mọi pixel của feature map, max theo token.

    Trả *logits* (chưa sigmoid) để dùng thẳng với BCEWithLogits -- ổn định số hơn.
    """

    def __init__(self, visual_dim, text_dim, proj_dim=128, init_temperature=0.07):
        super().__init__()
        self.v_proj = nn.Conv2d(visual_dim, proj_dim, kernel_size=1)
        self.t_proj = nn.Linear(text_dim, proj_dim)
        # Cosine ∈ [-1, 1] không đủ dải cho BCE -> cần nhiệt độ học được.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())
        # Bật trong validate() để lấy token thắng; tắt lúc train vì mỗi lần đọc là một lần
        # đồng bộ device. Chỉ an toàn khi chạy một luồng (không DataParallel).
        self.collect_stats = False
        self.last_token_scores = None
        # keep_sim do GenerativeResnetAlign bật khi w_token > 0.
        self.keep_sim = False
        self.last_sim = None

    def per_token(self, feat, text_emb, text_mask):
        """
        Bản đồ alignment của **từng** token -- đây là thứ để visualize "token nào ăn vùng nào".

        :return: (B, L, H, W) logits; token bị mask có giá trị -inf
        """
        v = F.normalize(self.v_proj(feat), dim=1)  # (B, P, H, W)
        t = F.normalize(self.t_proj(text_emb), dim=-1)  # (B, L, P)

        sim = torch.einsum("bphw,blp->blhw", v, t) * self.logit_scale.exp().clamp(
            max=100.0
        )
        return sim.masked_fill(~text_mask[:, :, None, None], float("-inf"))

    def forward(self, feat, text_emb, text_mask):
        """
        :param feat: (B, C, H, W)
        :param text_emb: (B, L, D)
        :param text_mask: (B, L) bool, True ở token nội dung
        :return: (B, 1, H, W) logits alignment (max theo token)
        """
        sim = self.per_token(feat, text_emb, text_mask)
        if self.collect_stats:
            # Điểm cao nhất của từng token trên toàn ảnh -- "token nào đang thắng".
            self.last_token_scores = sim.flatten(2).max(dim=2).values  # (B, L)
        if self.keep_sim:
            # `L_token` cần toàn bộ (B, L, H, W); giữ lại để compute_loss khỏi forward lần hai.
            # -inf ở token bị mask phải thay bằng số hữu hạn, nếu không logsumexp sinh NaN.
            self.last_sim = torch.nan_to_num(sim, neginf=NEG_INF)
        logits = sim.max(dim=1).values  # (B, H, W)
        # Prompt rỗng (không còn token nào) -> -inf; đưa về 0 để loss không thành NaN.
        logits = torch.nan_to_num(logits, neginf=0.0)
        return logits.unsqueeze(1)


def dice_loss(logits, target, eps=1.0, reduce=True):
    """:param reduce: False trả (B,) để đánh trọng số theo từng sample (xem Update 2)."""
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    per = 1 - (2 * inter + eps) / (prob.sum(dim=1) + target.sum(dim=1) + eps)
    return per.mean() if reduce else per


def token_contrastive_loss(sim, part_tok, weight, neg=-1e4):
    """
    Số hạng nói cho model biết token *nào* phải thắng ở vùng part (Update 3).

    `L_align` chỉ giám sát `max_j`, nên đa dạng token luôn bị phạt: tại một pixel nền, chỉ cần
    một token tình cờ hợp hướng là max bị đẩy lên. Áp lực đó dồn mọi token về một hướng chung
    -- đúng hiện tượng "token collapse" quan sát được trong figures/tokens_*.png. Đây là số
    hạng duy nhất thưởng cho đa dạng.

        L_token = -mean_{xy ∈ part} log( Σ_{j ∈ part} e^{s_j} / Σ_j e^{s_j} )

    :param sim: (B, L, H, W) similarity từng token, đã thay -inf bằng `neg` hữu hạn
    :param part_tok: (B, L) bool, token thuộc cụm chỉ part
    :param weight: (B, 1, H, W) trọng số pixel, chính là target part_mask ở độ phân giải A_T
    :return: (B,) loss từng sample; sample không có token part trả 0
    """
    has_part = part_tok.any(dim=1)                                  # (B,)
    log_z = torch.logsumexp(sim, dim=1)                             # (B, H, W)
    log_p = torch.logsumexp(
        sim.masked_fill(~part_tok[:, :, None, None], neg), dim=1
    )                                                               # (B, H, W)
    nll = log_z - log_p                                             # >= 0, hữu hạn nhờ `neg`

    w = weight.squeeze(1) * has_part[:, None, None].to(weight.dtype)
    den = w.flatten(1).sum(dim=1).clamp(min=1e-6)
    return (nll * w).flatten(1).sum(dim=1) / den


class GenerativeResnetAlign(GenerativeResnet):
    """
    GR-ConvNet-3 + nhánh ngôn ngữ. `use_text=False` biến nó thành GR-ConvNet-3 gốc (baseline
    no-text dùng chung file, chung số bước train).
    """

    accepts_extras = True

    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
        prob=0.0,
        use_text=True,
        proj_dim=128,
        gate_lambda=1.0,
        w_align=1.0,
        w_agnostic=1.0,
        w_token=0.0,
        mask_angle_loss=True,
        text_encoder=None,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            channel_size=channel_size,
            dropout=dropout,
            prob=prob,
        )

        self.use_text = use_text
        self.gate_lambda = gate_lambda
        self.w_align = w_align
        self.w_agnostic = w_agnostic
        self.w_token = w_token
        self.mask_angle_loss = mask_angle_loss
        # Warmup do vòng train set: 0 -> 1 để gradient của L_align không lấn lúc A_T còn ngẫu nhiên.
        self.register_buffer("loss_warmup", torch.tensor(1.0), persistent=False)

        bottleneck = channel_size * 4
        # Q_g: graspability không điều kiện text, supervise bằng M_union.
        self.graspability_head = nn.Conv2d(bottleneck, 1, kernel_size=1)

        if use_text:
            self.text_encoder = (
                text_encoder if text_encoder is not None else CLIPTextEncoder()
            )
            self.align = TokenPixelAlignment(
                bottleneck, self.text_encoder.dim, proj_dim
            )
            self.align.keep_sim = w_token > 0

        self.collect_stats = False

    def set_collect_stats(self, value):
        """
        Bật/tắt việc tính chẩn đoán `A_T` trong compute_loss.

        Mặc định tắt: mỗi thống kê là một lần `.item()` tức một lần đồng bộ device, không đáng
        trả giá ở mọi batch train. Bật quanh vòng validate().
        """
        self.collect_stats = bool(value)
        if self.use_text:
            self.align.collect_stats = bool(value)

    # --------------------------------------------------------------- forward --
    def encode(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        return x

    def decode(self, x):
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            return (
                self.pos_output(self.dropout_pos(x)),
                self.cos_output(self.dropout_cos(x)),
                self.sin_output(self.dropout_sin(x)),
                self.width_output(self.dropout_wid(x)),
            )
        return (
            self.pos_output(x),
            self.cos_output(x),
            self.sin_output(x),
            self.width_output(x),
        )

    def forward(self, x_in, prompts=None):
        """
        :return: (pos, cos, sin, width, align_logits, graspability_logits); hai phần tử cuối là
                 None khi không có text / không bật nhánh tương ứng.
        """
        feat = self.encode(x_in)
        graspability_logits = self.graspability_head(feat)

        align_logits = None
        self._part_tok = None
        if self.use_text and prompts is not None:
            text_emb, text_mask, part_tok = self.text_encoder(
                prompts, want_part_tokens=self.w_token > 0
            )
            self._part_tok = part_tok
            align_logits = self.align(feat, text_emb.to(feat.dtype), text_mask)
            # Residual gating: A_T ≈ 0 lúc đầu vẫn cho gradient đi qua nhánh thị giác.
            feat = feat * (1.0 + self.gate_lambda * torch.sigmoid(align_logits))

        return self.decode(feat) + (align_logits, graspability_logits)

    # ------------------------------------------------------------------ loss --
    def compute_loss(
        self, xc, yc, prompts=None, part_mask=None, union_pos=None, align_weight=None
    ):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred, align_logits, grasp_logits = self(
            xc, prompts
        )

        # --- Update 1: cos/sin/width chỉ tính trong vùng grasp -------------------
        # Target của ba đại lượng này bằng 0 ở mọi pixel ngoài grasp rectangle, mà nền chiếm
        # ~97% ảnh. Để dày đặc thì ~97% gradient chỉ dạy model xuất 0, và nhánh góc sụp về
        # hằng số -- đúng bản đồ góc phẳng thấy trong figures/prediction_seen_00.png.
        # `p_loss` PHẢI giữ dày đặc: model cần học chỗ nào *không* được gắp.
        losses = {"p_loss": F.smooth_l1_loss(pos_pred, y_pos)}
        if self.mask_angle_loss:
            fg = (y_pos > 0.5).to(pos_pred.dtype)
            # clamp thay cho `if fg.any()`: tránh một lần đồng bộ device mỗi batch, và batch
            # không có foreground cho loss 0 thay vì NaN.
            den = fg.sum().clamp(min=1.0)
            for name, pred, tgt in (
                ("cos_loss", cos_pred, y_cos),
                ("sin_loss", sin_pred, y_sin),
                ("width_loss", width_pred, y_width),
            ):
                losses[name] = (
                    F.smooth_l1_loss(pred, tgt, reduction="none") * fg
                ).sum() / den
        else:
            losses["cos_loss"] = F.smooth_l1_loss(cos_pred, y_cos)
            losses["sin_loss"] = F.smooth_l1_loss(sin_pred, y_sin)
            losses["width_loss"] = F.smooth_l1_loss(width_pred, y_width)

        grasp_loss = sum(losses.values())
        warmup = float(self.loss_warmup)
        loss = grasp_loss
        # `loss` bị warmup nhân vào nên các epoch trong giai đoạn warmup có *định nghĩa loss
        # khác* các epoch sau -- không so sánh được xuyên epoch, và argmin của nó luôn rơi vào
        # epoch 0. `loss_full` là cùng biểu thức ở trọng số đầy đủ, dùng để so sánh và để chọn
        # checkpoint.
        loss_full = grasp_loss
        stats = {}

        if align_logits is not None and part_mask is not None and self.w_align > 0:
            target = _match_resolution(part_mask, align_logits)

            # --- Update 2: trọng số theo độ phân biệt của part_mask -------------
            # d_i = 1 - IoU(mask_i, hợp mask mọi part cùng object). Sample nào có mask bằng
            # cả object (quả táo: skin/flesh/seeds/stem cùng một mask) thì d_i = 0 và không
            # đóng góp gì -- target của nó không chứa thông tin phân biệt part, ép học nó là
            # dạy model *bất biến theo part*, đúng ngược luận điểm của method.
            # Chuẩn hoá bằng .mean() chứ không phải mean có trọng số: khi phần lớn sample có
            # d_i ≈ 0 thì L_align teo lại và trả ngân sách gradient về nhánh grasp.
            d = _sample_weight(align_weight, align_logits)             # (B,)

            # Tách BCE và Dice: gộp một số thì không phân biệt được "đoán gần 0 khắp nơi"
            # (Dice ≈ 1, BCE nhỏ) với "phủ quá rộng" (BCE lớn, Dice vừa).
            bce_per = F.binary_cross_entropy_with_logits(
                align_logits, target, reduction="none"
            ).mean(dim=(1, 2, 3))
            dice_per = dice_loss(align_logits, target, reduce=False)
            bce = (bce_per * d).mean()
            dice = (dice_per * d).mean()
            align = bce + dice
            losses["align_bce"] = bce
            losses["align_dice"] = dice
            losses["align_loss"] = align
            losses["align_weight"] = d.mean()
            loss = loss + self.w_align * warmup * align
            loss_full = loss_full + self.w_align * align

            # --- Update 3: contrastive theo token -------------------------------
            sim = getattr(self.align, "last_sim", None)
            if self.w_token > 0 and sim is not None and self._part_tok is not None:
                token = (
                    token_contrastive_loss(sim, self._part_tok, target, neg=NEG_INF) * d
                ).mean()
                losses["token_loss"] = token
                loss = loss + self.w_token * warmup * token
                loss_full = loss_full + self.w_token * token
            self.align.last_sim = None      # đừng giữ graph sang batch sau

            if self.collect_stats:
                stats.update(self._align_stats(align_logits, target))

        if union_pos is not None and self.w_agnostic > 0:
            target = _match_resolution(union_pos, grasp_logits)
            agnostic = F.binary_cross_entropy_with_logits(grasp_logits, target)
            losses["agnostic_loss"] = agnostic
            loss = loss + self.w_agnostic * warmup * agnostic
            loss_full = loss_full + self.w_agnostic * agnostic

        pred = {"pos": pos_pred, "cos": cos_pred, "sin": sin_pred, "width": width_pred}
        if align_logits is not None:
            pred["align"] = torch.sigmoid(align_logits)
        pred["graspability"] = torch.sigmoid(grasp_logits)
        return {
            "loss": loss,
            "loss_full": loss_full,
            "losses": losses,
            "pred": pred,
            "stats": stats,
        }

    @torch.no_grad()
    def _align_stats(self, align_logits, target):
        """
        Chẩn đoán `A_T` -- loss không cho biết nhánh alignment có hoạt động hay không.

        `fg`/`bg` bằng nhau nghĩa là gate vô dụng dù loss có giảm. `argmax_last` cao nghĩa là
        token thắng luôn là token cuối câu: CLIP text dùng causal attention nên token cuối là
        token duy nhất thấy cả câu, và `A_T = max` theo token sẽ luôn chọn nó -- alignment
        thoái hoá về mức câu, không còn là mức token/part.
        """
        prob = torch.sigmoid(align_logits)
        fg = target > 0.5
        bg = ~fg
        pred = prob > 0.5
        inter = (pred & fg).sum().float()
        stats = {
            "align_fg_mean": prob[fg].mean() if fg.any() else prob.new_zeros(()),
            "align_bg_mean": prob[bg].mean() if bg.any() else prob.new_zeros(()),
            "align_iou": inter / ((pred | fg).sum().float() + 1e-6),
            "align_dice_coef": 2 * inter / ((pred.sum() + fg.sum()).float() + 1e-6),
            "logit_scale": self.align.logit_scale.exp().detach(),
        }
        scores = getattr(self.align, "last_token_scores", None)
        if scores is not None:
            valid = torch.isfinite(scores)  # (B, L)
            pos = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
            last = pos.masked_fill(~valid, -1).max(dim=1).values  # (B,)
            winner = scores.masked_fill(~valid, float("-inf")).argmax(dim=1)
            stats["argmax_last_frac"] = ((winner == last) & (last >= 0)).float().mean()
        return stats

    def predict(self, xc, prompts=None):
        pos, cos, sin, width, align_logits, grasp_logits = self(xc, prompts)
        out = {
            "pos": pos,
            "cos": cos,
            "sin": sin,
            "width": width,
            "graspability": torch.sigmoid(grasp_logits),
        }
        if align_logits is not None:
            out["align"] = torch.sigmoid(align_logits)
        return out

    @torch.no_grad()
    def token_alignment(self, x_in, prompts):
        """
        Alignment token-pixel cho việc visualize.

        :return: dict
            tokens     list[list[str]] -- token nội dung của từng prompt (đã bỏ SOT/EOT/pad)
            maps       (B, L_i, H, W) similarity của từng token, đã chuẩn hoá về [0, 1]
            attention  (B, 1, H, W) A_T = sigmoid(max theo token), chính thứ dùng để gate
        """
        if not self.use_text:
            raise RuntimeError(
                "Model khởi tạo với use_text=False, không có nhánh alignment."
            )

        feat = self.encode(x_in)
        text_emb, text_mask, _ = self.text_encoder(prompts)
        sim = self.align.per_token(
            feat, text_emb.to(feat.dtype), text_mask
        )  # (B, L, H, W)
        attention = torch.sigmoid(self.align(feat, text_emb.to(feat.dtype), text_mask))

        all_tokens = self.text_encoder.token_strings(prompts)
        tokens, maps = [], []
        for b in range(sim.shape[0]):
            keep = text_mask[b].nonzero(as_tuple=True)[0]
            tokens.append([_clean_token(all_tokens[b][i]) for i in keep.tolist()])
            m = sim[b, keep]  # (L_i, H, W)
            lo, hi = m.amin(), m.amax()
            maps.append((m - lo) / (hi - lo + 1e-8))
        return {"tokens": tokens, "maps": maps, "attention": attention}

    @torch.no_grad()
    def align_token_report(self, x_in, prompts):
        """
        Token thắng có phải từ chỉ part hay không -- metric trực tiếp của luận điểm
        "part-level grounding", thứ mà `align_loss` hoàn toàn không đo.

        Đắt hơn `_align_stats` (phải chạy tokenizer và lấy chuỗi token) nên chỉ chạy trên một
        subset nhỏ của validation, không phải mọi batch.

        :return: dict
            n           số sample tính được (prompt có cụm part nhận dạng được)
            part_frac   tỉ lệ token thắng nằm trong cụm chỉ part
            punct_frac  tỉ lệ token thắng là dấu câu
            winners     list[(token thắng, cụm part)] để soi tay
        """
        out = self.token_alignment(x_in, prompts)
        n = part_hit = punct = total = 0
        winners = []
        for tokens, m, prompt in zip(out["tokens"], out["maps"], prompts):
            if not tokens:
                continue
            total += 1
            top = tokens[int(m.flatten(1).max(dim=1).values.argmax())]
            part = _part_words(prompt)
            winners.append((top, sorted(part)))
            if top and all(ch in string.punctuation for ch in top):
                punct += 1
            if part:
                n += 1
                # token là mảnh BPE nên so bằng "nằm trong": `bbed` ⊂ `webbed`.
                if top and any(top in w for w in part):
                    part_hit += 1
        return {
            "n": n,
            "part_frac": part_hit / n if n else 0.0,
            "punct_frac": punct / total if total else 0.0,
            "winners": winners,
        }

    def set_loss_warmup(self, value):
        """Gọi mỗi epoch: 0 -> 1 trong vài epoch đầu."""
        self.loss_warmup.fill_(float(value))


def _clean_token(token):
    """CLIP BPE gắn hậu tố `</w>` ở token cuối từ -- bỏ đi cho dễ đọc."""
    return token.replace("</w>", "")


def _part_words(prompt):
    """
    Cụm từ chỉ part trong prompt GA++, lấy phần sau "its" cuối cùng.

        "Pick up duck by its webbed feet."  ->  {'webbed', 'feet'}
        "Hold book at its binding."         ->  {'binding'}

    Trả set rỗng nếu prompt không theo mẫu đó -- sample đó bị loại khỏi mẫu số.
    """
    words = re.findall(r"[a-z]+", prompt.lower())
    if "its" not in words:
        return set()
    return set(words[len(words) - 1 - words[::-1].index("its") + 1 :])


def _sample_weight(weight, like):
    """
    Trọng số theo từng sample cho `L_align`, dạng (B,).

    `weight` vắng mặt (dataset chưa có file d_i, hoặc dataset khác GA++) -> toàn 1, tức đúng
    hành vi cũ. Nhận cả (B,) lẫn (B, 1).
    """
    b = like.shape[0]
    if weight is None:
        return like.new_ones(b)
    return weight.reshape(b).to(device=like.device, dtype=like.dtype)


def _match_resolution(target, like):
    """
    Hạ target (224×224) xuống đúng lưới của A_T / Q_g (56×56) thay vì upsample logits lên --
    supervise ở resolution thật của feature map, không tạo độ chính xác giả.
    """
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)
