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

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grconvnet3 import GenerativeResnet

CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_DIM = 512
MAX_PROMPT_TOKENS = 16      # prompt GA++ dài nhất ~11 từ, 77 của CLIP là phí


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder đóng băng, trả embedding **từng token** (không phải một vector EOT).

    `CLIPTextModel.last_hidden_state` cho cả chuỗi; ta chiếu qua `text_projection` để về cùng
    không gian với image embedding của CLIP, rồi mask bỏ SOT/EOT/padding.
    """

    def __init__(self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS):
        super(CLIPTextEncoder, self).__init__()
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

    @property
    def dim(self):
        return self.model.config.projection_dim

    def train(self, mode=True):
        # Giữ text encoder ở eval kể cả khi net.train() -- nó đóng băng.
        super(CLIPTextEncoder, self).train(mode)
        self.model.eval()
        return self

    def token_strings(self, prompts):
        """
        Chuỗi token thật sự đưa vào encoder, để gán nhãn cho hình alignment.

        :return: list[list[str]] cùng chiều dài với chiều L của forward() (kể cả SOT/EOT/pad,
                 dùng chung mask với forward để lọc).
        """
        tok = self.tokenizer(list(prompts), padding='max_length', truncation=True,
                             max_length=self.max_length)
        return [[self.tokenizer.convert_ids_to_tokens(i) for i in ids]
                for ids in tok['input_ids']]

    @torch.no_grad()
    def forward(self, prompts):
        """
        :param prompts: list[str] độ dài B
        :return: (B, L, D) embedding từng token, (B, L) mask bool của token nội dung
        """
        device = next(self.model.parameters()).device
        tok = self.tokenizer(list(prompts), padding='max_length', truncation=True,
                             max_length=self.max_length, return_tensors='pt').to(device)
        out = self.model.text_model(**tok).last_hidden_state          # (B, L, hidden)
        emb = self.model.text_projection(out)                         # (B, L, D)

        mask = tok['attention_mask'].bool()
        # Bỏ SOT (luôn ở vị trí 0) và EOT (token cuối cùng còn attend được).
        mask[:, 0] = False
        eot = tok['attention_mask'].sum(dim=1) - 1
        mask[torch.arange(mask.shape[0], device=device), eot] = False
        return emb, mask


class TokenPixelAlignment(nn.Module):
    """
    Cosine similarity giữa mọi token văn bản và mọi pixel của feature map, max theo token.

    Trả *logits* (chưa sigmoid) để dùng thẳng với BCEWithLogits -- ổn định số hơn.
    """

    def __init__(self, visual_dim, text_dim, proj_dim=128, init_temperature=0.07):
        super(TokenPixelAlignment, self).__init__()
        self.v_proj = nn.Conv2d(visual_dim, proj_dim, kernel_size=1)
        self.t_proj = nn.Linear(text_dim, proj_dim)
        # Cosine ∈ [-1, 1] không đủ dải cho BCE -> cần nhiệt độ học được.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    def per_token(self, feat, text_emb, text_mask):
        """
        Bản đồ alignment của **từng** token -- đây là thứ để visualize "token nào ăn vùng nào".

        :return: (B, L, H, W) logits; token bị mask có giá trị -inf
        """
        v = F.normalize(self.v_proj(feat), dim=1)                     # (B, P, H, W)
        t = F.normalize(self.t_proj(text_emb), dim=-1)                # (B, L, P)

        sim = torch.einsum('bphw,blp->blhw', v, t) * self.logit_scale.exp().clamp(max=100.0)
        return sim.masked_fill(~text_mask[:, :, None, None], float('-inf'))

    def forward(self, feat, text_emb, text_mask):
        """
        :param feat: (B, C, H, W)
        :param text_emb: (B, L, D)
        :param text_mask: (B, L) bool, True ở token nội dung
        :return: (B, 1, H, W) logits alignment (max theo token)
        """
        sim = self.per_token(feat, text_emb, text_mask)
        logits = sim.max(dim=1).values                                # (B, H, W)
        # Prompt rỗng (không còn token nào) -> -inf; đưa về 0 để loss không thành NaN.
        logits = torch.nan_to_num(logits, neginf=0.0)
        return logits.unsqueeze(1)


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    return (1 - (2 * inter + eps) / (prob.sum(dim=1) + target.sum(dim=1) + eps)).mean()


class GenerativeResnetAlign(GenerativeResnet):
    """
    GR-ConvNet-3 + nhánh ngôn ngữ. `use_text=False` biến nó thành GR-ConvNet-3 gốc (baseline
    no-text dùng chung file, chung số bước train).
    """

    accepts_extras = True

    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False,
                 prob=0.0, use_text=True, proj_dim=128, gate_lambda=1.0,
                 w_align=1.0, w_agnostic=1.0, text_encoder=None):
        super(GenerativeResnetAlign, self).__init__(
            input_channels=input_channels, output_channels=output_channels,
            channel_size=channel_size, dropout=dropout, prob=prob)

        self.use_text = use_text
        self.gate_lambda = gate_lambda
        self.w_align = w_align
        self.w_agnostic = w_agnostic
        # Warmup do vòng train set: 0 -> 1 để gradient của L_align không lấn lúc A_T còn ngẫu nhiên.
        self.register_buffer('loss_warmup', torch.tensor(1.0), persistent=False)

        bottleneck = channel_size * 4
        # Q_g: graspability không điều kiện text, supervise bằng M_union.
        self.graspability_head = nn.Conv2d(bottleneck, 1, kernel_size=1)

        if use_text:
            self.text_encoder = text_encoder if text_encoder is not None else CLIPTextEncoder()
            self.align = TokenPixelAlignment(bottleneck, self.text_encoder.dim, proj_dim)

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
            return (self.pos_output(self.dropout_pos(x)), self.cos_output(self.dropout_cos(x)),
                    self.sin_output(self.dropout_sin(x)), self.width_output(self.dropout_wid(x)))
        return (self.pos_output(x), self.cos_output(x),
                self.sin_output(x), self.width_output(x))

    def forward(self, x_in, prompts=None):
        """
        :return: (pos, cos, sin, width, align_logits, graspability_logits); hai phần tử cuối là
                 None khi không có text / không bật nhánh tương ứng.
        """
        feat = self.encode(x_in)
        graspability_logits = self.graspability_head(feat)

        align_logits = None
        if self.use_text and prompts is not None:
            text_emb, text_mask = self.text_encoder(prompts)
            align_logits = self.align(feat, text_emb.to(feat.dtype), text_mask)
            # Residual gating: A_T ≈ 0 lúc đầu vẫn cho gradient đi qua nhánh thị giác.
            feat = feat * (1.0 + self.gate_lambda * torch.sigmoid(align_logits))

        return self.decode(feat) + (align_logits, graspability_logits)

    # ------------------------------------------------------------------ loss --
    def compute_loss(self, xc, yc, prompts=None, part_mask=None, union_pos=None):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred, align_logits, grasp_logits = self(xc, prompts)

        losses = {
            'p_loss': F.smooth_l1_loss(pos_pred, y_pos),
            'cos_loss': F.smooth_l1_loss(cos_pred, y_cos),
            'sin_loss': F.smooth_l1_loss(sin_pred, y_sin),
            'width_loss': F.smooth_l1_loss(width_pred, y_width),
        }
        loss = sum(losses.values())
        warmup = float(self.loss_warmup)

        if align_logits is not None and part_mask is not None and self.w_align > 0:
            target = _match_resolution(part_mask, align_logits)
            align = F.binary_cross_entropy_with_logits(align_logits, target) \
                + dice_loss(align_logits, target)
            losses['align_loss'] = align
            loss = loss + self.w_align * warmup * align

        if union_pos is not None and self.w_agnostic > 0:
            target = _match_resolution(union_pos, grasp_logits)
            agnostic = F.binary_cross_entropy_with_logits(grasp_logits, target)
            losses['agnostic_loss'] = agnostic
            loss = loss + self.w_agnostic * warmup * agnostic

        pred = {'pos': pos_pred, 'cos': cos_pred, 'sin': sin_pred, 'width': width_pred}
        if align_logits is not None:
            pred['align'] = torch.sigmoid(align_logits)
        pred['graspability'] = torch.sigmoid(grasp_logits)
        return {'loss': loss, 'losses': losses, 'pred': pred}

    def predict(self, xc, prompts=None):
        pos, cos, sin, width, align_logits, grasp_logits = self(xc, prompts)
        out = {'pos': pos, 'cos': cos, 'sin': sin, 'width': width,
               'graspability': torch.sigmoid(grasp_logits)}
        if align_logits is not None:
            out['align'] = torch.sigmoid(align_logits)
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
            raise RuntimeError('Model khởi tạo với use_text=False, không có nhánh alignment.')

        feat = self.encode(x_in)
        text_emb, text_mask = self.text_encoder(prompts)
        sim = self.align.per_token(feat, text_emb.to(feat.dtype), text_mask)   # (B, L, H, W)
        attention = torch.sigmoid(self.align(feat, text_emb.to(feat.dtype), text_mask))

        all_tokens = self.text_encoder.token_strings(prompts)
        tokens, maps = [], []
        for b in range(sim.shape[0]):
            keep = text_mask[b].nonzero(as_tuple=True)[0]
            tokens.append([_clean_token(all_tokens[b][i]) for i in keep.tolist()])
            m = sim[b, keep]                                                   # (L_i, H, W)
            lo, hi = m.amin(), m.amax()
            maps.append((m - lo) / (hi - lo + 1e-8))
        return {'tokens': tokens, 'maps': maps, 'attention': attention}

    def set_loss_warmup(self, value):
        """Gọi mỗi epoch: 0 -> 1 trong vài epoch đầu."""
        self.loss_warmup.fill_(float(value))


def _clean_token(token):
    """CLIP BPE gắn hậu tố `</w>` ở token cuối từ -- bỏ đi cho dễ đọc."""
    return token.replace('</w>', '')


def _match_resolution(target, like):
    """
    Hạ target (224×224) xuống đúng lưới của A_T / Q_g (56×56) thay vì upsample logits lên --
    supervise ở resolution thật của feature map, không tạo độ chính xác giả.
    """
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)
