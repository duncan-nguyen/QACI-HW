"""GR-ConvNet-3 + grasp-aware text-visual alignment ở mức part (language-driven grasping).

Ý tưởng: prompt của Grasp-Anything++ nhắm vào một *part* cụ thể ("grasp the mug at its
handle"), nên tín hiệu ngôn ngữ phải chọn *vùng* trên ảnh chứ không chỉ đổi output cuối. Ta
tính relevance giữa **mọi token** và **mọi pixel**, soft-select token quan trọng tại từng
pixel, rồi đưa cả *vùng quan trọng* `A_T` lẫn *text feature của vùng đó* `R_T` vào decoder.
`A_T` được supervise **tường minh** bằng `part_mask` của GA++ -- nhãn khác loại với grasp
rectangle, nên nhánh ngôn ngữ không chỉ học lại đúng thứ mà nhánh grasp đã học.

    prompt ──► CLIP text (frozen) ──► t_1..t_L
                                          │
    ảnh ────► conv1..res5 ──► F (56×56) ──┤
                                          ▼
                         S_pj = cos(W_v F_p, W_t t_j) / τ
                         α_pj = softmax_j(S_pj / τ_t)      token quan trọng tại p
                         A_T(p) = σ(Σ_j α_pj S_pj)         vùng quan trọng
                         R_T(p) = Σ_j α_pj W_r t_j         text feature của vùng đó
                                          │
                    F' = F + φ([F, F ⊙ A_T, R_T])  ──► conv4..conv6 ──► pos/cos/sin/width

    L = L_grasp + w_align · [BCE + Dice](A_T, part_mask)

Cờ bật/tắt cho bảng ablation của idea.md §6 -- cùng một file, cùng ngân sách train:

    arm            use_text  align_mode  region_text  fusion     w_align
    no-text        False     -           -            -          0
    hard-max (V1)  True      hard        False        gate       > 0
    soft           True      soft        False        residual   > 0
    V2 full        True      soft        True         residual   > 0

So với V1 (commit trước): `max` theo token -> softmax; thêm `R_T`; gate nhân
`F ⊙ (1+λA_T)` -> residual fusion; bỏ hẳn `Q_g`/`L_agnostic`; loại thêm token dấu câu khỏi
candidate (V1 chỉ loại SOT/EOT/pad, nên `.` thường là token "mạnh nhất" trong hình alignment).
Checkpoint V1 vẫn load và chạy được -- xem `__setstate__`.
"""
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grconvnet3 import GenerativeResnet

CLIP_MODEL = 'openai/clip-vit-base-patch32'
MAX_PROMPT_TOKENS = 16      # prompt GA++ dài nhất ~11 từ, 77 của CLIP là phí

# masked_fill bằng -inf làm softmax ra NaN khi *mọi* token của một prompt đều bị loại; -1e4 đủ
# nhỏ để α ≈ 0 mà vẫn hữu hạn. Trường hợp prompt rỗng được xử lý riêng bằng cờ `valid`.
NEG_INF = -1e4


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder đóng băng, trả embedding **từng token** (không phải một vector EOT).

    `CLIPTextModel.last_hidden_state` cho cả chuỗi; ta chiếu qua `text_projection` để về cùng
    không gian với image embedding của CLIP, rồi mask bỏ SOT/EOT/padding **và dấu câu**.
    """

    def __init__(self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS,
                 drop_punctuation=True):
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
        self.drop_punctuation = drop_punctuation
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Tra cứu "token này có nội dung không" theo id, dựng một lần cho cả vocab (49.408
        # mục) thay vì decode lại mỗi batch. Hình alignment của run V1 cho thấy token mạnh
        # nhất thường là `.` -- dấu câu hút hết attention mà không mang thông tin part nào.
        vocab = self.tokenizer.get_vocab()
        is_content = torch.zeros(max(vocab.values()) + 1, dtype=torch.bool)
        for token, idx in vocab.items():
            is_content[idx] = any(ch.isalnum() for ch in token.replace('</w>', ''))
        self.register_buffer('is_content', is_content, persistent=False)

    @property
    def dim(self):
        return self.model.config.projection_dim

    def __setstate__(self, state):
        """
        Encoder trong checkpoint V1 không có `drop_punctuation` lẫn bảng `is_content`.

        Giữ đúng hành vi V1 (dấu câu *được* tính là candidate) để con số đo lại từ checkpoint
        cũ vẫn là con số của model đó, không phải của một biến thể khác.
        """
        super(CLIPTextEncoder, self).__setstate__(state)
        if not hasattr(self, 'drop_punctuation'):
            self.drop_punctuation = False
            logging.getLogger(__name__).info(
                'Checkpoint V1: giữ nguyên hành vi cũ, không loại token dấu câu.')
        if 'is_content' not in self._buffers:
            vocab = self.tokenizer.get_vocab()
            is_content = torch.zeros(max(vocab.values()) + 1, dtype=torch.bool)
            for token, idx in vocab.items():
                is_content[idx] = any(ch.isalnum() for ch in token.replace('</w>', ''))
            self.register_buffer('is_content', is_content, persistent=False)

    def train(self, mode=True):
        # Giữ text encoder ở eval kể cả khi net.train() -- nó đóng băng.
        super(CLIPTextEncoder, self).train(mode)
        self.model.eval()
        return self

    def _tokenize(self, prompts):
        """
        Nhận cả hai dạng prompt:

        * `list[str]` -- tokenize tại chỗ, như trước.
        * `dict` có `input_ids`/`attention_mask` -- đã tokenize sẵn trong DataLoader worker
          (xem `PromptTokenizer`). Tokenize 64 prompt tốn 6,3 ms; chạy trong tiến trình chính
          thì 6,3 ms đó nằm thẳng trên đường găng của mỗi step, chia về worker thì mất hẳn.

        :return: dict[str, Tensor] hai key input_ids / attention_mask
        """
        if isinstance(prompts, dict):
            return {'input_ids': prompts['input_ids'],
                    'attention_mask': prompts['attention_mask']}
        return self.tokenizer(list(prompts), padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')

    def token_strings(self, prompts):
        """
        Chuỗi token thật sự đưa vào encoder, để gán nhãn cho hình alignment.

        :return: list[list[str]] cùng chiều dài với chiều L của forward() (kể cả SOT/EOT/pad,
                 dùng chung mask với forward để lọc).
        """
        ids = self._tokenize(prompts)['input_ids']
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return [[self.tokenizer.convert_ids_to_tokens(i) for i in seq] for seq in ids]

    @torch.no_grad()
    def forward(self, prompts):
        """
        :param prompts: list[str] độ dài B, hoặc dict đã tokenize sẵn (xem `_tokenize`)
        :return: (B, L, D) embedding từng token, (B, L) mask bool của token nội dung
        """
        device = next(self.model.parameters()).device
        tok = {k: v.to(device, non_blocking=True) for k, v in self._tokenize(prompts).items()}
        out = self.model.text_model(**tok).last_hidden_state          # (B, L, hidden)
        emb = self.model.text_projection(out)                         # (B, L, D)

        ids = tok['input_ids']
        # `.bool()` đã tạo tensor mới nên các phép ghi đè bên dưới không đụng vào attention_mask
        # gốc -- quan trọng khi tensor đó đến từ DataLoader và còn được dùng lại ở chỗ khác.
        mask = tok['attention_mask'].bool()
        # Bỏ SOT (luôn ở vị trí 0) và EOT (token cuối cùng còn attend được).
        mask[:, 0] = False
        eot = tok['attention_mask'].sum(dim=1) - 1
        mask[torch.arange(mask.shape[0], device=device), eot] = False
        if self.drop_punctuation:
            mask &= self.is_content.to(device)[ids]
        return emb, mask


class PromptTokenizer:
    """
    Tokenize prompt *trong DataLoader worker* thay vì trong tiến trình chính.

    Dataset gọi object này cho từng sample; `default_collate` gộp các tensor 1 chiều thành
    (B, L), và `CLIPTextEncoder._tokenize` nhận thẳng dict đó. Kết quả token giống hệt đường
    cũ -- cùng tokenizer, cùng `padding='max_length'`, cùng `max_length`.

    Phải picklable để `num_workers > 0` gửi được sang worker; `CLIPTokenizerFast` thoả mãn.
    """

    def __init__(self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS):
        from transformers import CLIPTokenizerFast
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.max_length = max_length
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    def __call__(self, prompt):
        tok = self.tokenizer(prompt, padding='max_length', truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        return {'input_ids': tok['input_ids'][0],
                'attention_mask': tok['attention_mask'][0]}


class TokenRegionAlignment(nn.Module):
    """
    Relevance giữa mọi token văn bản và mọi pixel của feature map (idea.md §3).

    Trả *logits* của `A_T` (chưa sigmoid) để dùng thẳng với BCEWithLogits -- ổn định số hơn --
    kèm `R_T` (text feature của vùng) và trọng số token `α` để visualize.
    """

    def __init__(self, visual_dim, text_dim, proj_dim=128, region_dim=64, mode='soft',
                 region_text=True, token_temperature=1.0, init_temperature=0.07):
        super(TokenRegionAlignment, self).__init__()
        if mode not in ('soft', 'hard'):
            raise ValueError(f"align_mode phải là 'soft' hoặc 'hard', nhận '{mode}'")

        self.mode = mode
        self.token_temperature = token_temperature
        self.v_proj = nn.Conv2d(visual_dim, proj_dim, kernel_size=1)
        self.t_proj = nn.Linear(text_dim, proj_dim)
        # W_r: text feature đưa vào decoder, tách khỏi W_t để không gian dùng cho *so khớp* và
        # không gian dùng để *mô tả* không phải là một.
        self.r_proj = nn.Linear(text_dim, region_dim) if region_text else None
        # Cosine ∈ [-1, 1] không đủ dải cho BCE -> cần nhiệt độ học được.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    @property
    def region_dim(self):
        return 0 if self.r_proj is None else self.r_proj.out_features

    def per_token(self, feat, text_emb, text_mask):
        """
        S_pj cho **từng** token -- đây là thứ để visualize "token nào ăn vùng nào".

        :return: (B, L, H, W) logits; token bị mask nhận NEG_INF
        """
        v = F.normalize(self.v_proj(feat), dim=1)                     # (B, P, H, W)
        t = F.normalize(self.t_proj(text_emb), dim=-1)                # (B, L, P)

        sim = torch.einsum('bphw,blp->blhw', v, t) * self.logit_scale.exp().clamp(max=100.0)
        return sim.masked_fill(~text_mask[:, :, None, None], NEG_INF)

    def token_weights(self, sim, text_mask):
        """α_pj: soft-select token tại từng pixel (hoặc one-hot khi mode='hard')."""
        if self.mode == 'hard':
            # Bản V1: chỉ token thắng mới đóng góp. Giữ lại để làm một arm của ablation.
            hard = torch.zeros_like(sim)
            return hard.scatter_(1, sim.argmax(dim=1, keepdim=True), 1.0)
        return torch.softmax(sim / self.token_temperature, dim=1)

    def forward(self, feat, text_emb, text_mask):
        """
        :param feat: (B, C, H, W)
        :param text_emb: (B, L, D)
        :param text_mask: (B, L) bool, True ở token nội dung
        :return: (align_logits (B,1,H,W), region (B,R,H,W) hoặc None, α (B,L,H,W))
        """
        sim = self.per_token(feat, text_emb, text_mask)
        attn = self.token_weights(sim, text_mask)
        logits = (attn * sim).sum(dim=1)                              # (B, H, W)

        # Prompt mà mọi token đều bị loại (rỗng, hoặc toàn dấu câu): α vô nghĩa -> trả 0 để
        # loss không bị kéo bởi NEG_INF và fusion không nhận nhiễu.
        valid = text_mask.any(dim=1)
        logits = torch.where(valid[:, None, None], logits, torch.zeros_like(logits))
        attn = attn * valid[:, None, None, None]

        region = None
        if self.r_proj is not None:
            r = self.r_proj(text_emb)                                 # (B, L, R)
            region = torch.einsum('blhw,blr->brhw', attn, r)
        return logits.unsqueeze(1), region, attn


class RegionFusion(nn.Module):
    """
    F' = F + φ([F, F ⊙ A_T, R_T]).

    Residual chứ không phải gate nhân: conv cuối khởi tạo 0 nên lúc bắt đầu F' = F đúng bằng
    baseline, nhánh ngôn ngữ đi vào dần khi `A_T` bắt đầu có nghĩa. Gate `F ⊙ (1+λA_T)` của V1
    thì ngay từ bước 0 đã nhân feature với một bản đồ ngẫu nhiên.
    """

    def __init__(self, feat_dim, region_dim=0):
        super(RegionFusion, self).__init__()
        self.conv1 = nn.Conv2d(2 * feat_dim + region_dim, feat_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(feat_dim)
        self.conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, feat, attention, region=None):
        parts = [feat, feat * attention]
        if region is not None:
            parts.append(region)
        return feat + self.conv2(F.relu(self.bn(self.conv1(torch.cat(parts, dim=1)))))


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
    # Tên trong inference/models/__init__.py -- utils/checkpoint.py dựng lại model từ đây.
    network_name = 'grconvnet3_align'

    # Giá trị của V1, dùng khi unpickle checkpoint cũ (xem __setstate__).
    _V1_DEFAULTS = dict(align_mode='hard', fusion='gate', use_region_text=False,
                        align_stage='bottleneck', gate_lambda=1.0, w_align=1.0)

    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False,
                 prob=0.0, use_text=True, align_mode='soft', region_text=True,
                 fusion='residual', align_stage='bottleneck', proj_dim=128, region_dim=64,
                 token_temperature=1.0, gate_lambda=1.0, w_align=1.0, text_encoder=None):
        """
        :param align_mode: 'soft' = α_pj softmax theo token (V2); 'hard' = max theo token (V1)
        :param region_text: đưa `R_T` vào decoder hay không
        :param fusion: 'residual' = F + φ([F, F⊙A_T, R_T]); 'gate' = F ⊙ (1 + λ·A_T) như V1
        :param align_stage: 'bottleneck' (56×56, sau res5) hoặc 'conv4' (113×113, sau conv4).
                            Mức part khá nhỏ so với stride 4; nếu `A_T` mờ thì thử 'conv4'.
        :param token_temperature: τ_t của softmax theo token. S đã được chia τ học được rồi,
                                  nên τ_t chỉ điều chỉnh độ "nhọn" của việc chọn token.
        :param text_encoder: dùng lại một CLIPTextEncoder có sẵn (tiết kiệm khi load nhiều model)
        """
        super(GenerativeResnetAlign, self).__init__(
            input_channels=input_channels, output_channels=output_channels,
            channel_size=channel_size, dropout=dropout, prob=prob)

        if fusion not in ('residual', 'gate'):
            raise ValueError(f"fusion phải là 'residual' hoặc 'gate', nhận '{fusion}'")
        if align_stage not in ('bottleneck', 'conv4'):
            raise ValueError(
                f"align_stage phải là 'bottleneck' hoặc 'conv4', nhận '{align_stage}'")

        # Đủ để dựng lại model rỗng khi load checkpoint dạng state_dict (utils/checkpoint.py).
        self.init_kwargs = dict(
            input_channels=input_channels, output_channels=output_channels,
            channel_size=channel_size, dropout=dropout, prob=prob, use_text=use_text,
            align_mode=align_mode, region_text=region_text, fusion=fusion,
            align_stage=align_stage, proj_dim=proj_dim, region_dim=region_dim,
            token_temperature=token_temperature, gate_lambda=gate_lambda, w_align=w_align)

        self.use_text = use_text
        self.align_mode = align_mode
        self.use_region_text = bool(region_text)
        self.fusion = fusion
        self.align_stage = align_stage
        self.gate_lambda = gate_lambda
        self.w_align = w_align
        # Bật bằng `with net.collecting_stats():` -- xem utils/diagnostics.py. Mặc định tắt để
        # vòng train nóng không phải tính entropy/argmax trên (B, L, H, W) mỗi bước.
        self.collect_stats = False
        self.last_stats = {}
        # Warmup do vòng train set: 0 -> 1 để gradient của L_align không lấn lúc A_T còn ngẫu nhiên.
        self.register_buffer('loss_warmup', torch.tensor(1.0), persistent=False)

        if use_text:
            feat_dim = channel_size * 4 if align_stage == 'bottleneck' else channel_size * 2
            self.text_encoder = text_encoder if text_encoder is not None else CLIPTextEncoder()
            self.align = TokenRegionAlignment(
                feat_dim, self.text_encoder.dim, proj_dim=proj_dim, region_dim=region_dim,
                mode=align_mode, region_text=region_text,
                token_temperature=token_temperature)
            if fusion == 'residual':
                self.fuse = RegionFusion(feat_dim, self.align.region_dim)

    # --------------------------------------------------------------- forward --
    def encode(self, x_in):
        """conv1..res5 -> F ở bottleneck (56×56 với input 224)."""
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        return x

    def heads(self, x):
        if self.dropout:
            return (self.pos_output(self.dropout_pos(x)), self.cos_output(self.dropout_cos(x)),
                    self.sin_output(self.dropout_sin(x)), self.width_output(self.dropout_wid(x)))
        return (self.pos_output(x), self.cos_output(x),
                self.sin_output(x), self.width_output(x))

    def condition(self, feat, prompts):
        """
        Một lượt token-region alignment + fusion tại đúng feature map đang có.

        :return: (F', align_logits) -- align_logits là None khi không có prompt
        """
        if not self.use_text or prompts is None:
            return feat, None

        text_emb, text_mask = self.text_encoder(prompts)
        align_logits, region, attn = self.align(feat, text_emb.to(feat.dtype), text_mask)
        attention = torch.sigmoid(align_logits)

        if self.fusion == 'gate':
            # Bản V1: gate nhân, không dùng R_T.
            fused = feat * (1.0 + self.gate_lambda * attention)
        else:
            fused = self.fuse(feat, attention, region)

        if self.collect_stats:
            self.last_stats = self._forward_stats(feat, fused, attn, text_mask)
        return fused, align_logits

    def _forward_stats(self, feat, fused, attn, text_mask):
        from utils.diagnostics import fusion_stats, token_stats, winning_tokens

        stats = {f'token/{k}': v for k, v in token_stats(attn, text_mask).items()}
        stats.update({f'fusion/{k}': v for k, v in fusion_stats(feat, fused).items()})
        stats['token/logit_scale'] = float(self.align.logit_scale.exp())
        stats['token/temperature'] = 1.0 / max(stats['token/logit_scale'], 1e-8)
        # Giữ ở dạng tensor CPU: vòng train gộp lại rồi mới đổi sang tên token.
        stats['_winning_tokens'] = winning_tokens(attn, text_mask).cpu()
        stats['_text_mask'] = text_mask.cpu()
        return stats

    @contextlib.contextmanager
    def collecting_stats(self):
        """Bật thu thập chẩn đoán cho các forward bên trong khối `with`."""
        previous = self.collect_stats
        self.collect_stats = True
        try:
            yield self
        finally:
            self.collect_stats = previous

    def forward(self, x_in, prompts=None):
        """
        :return: (pos, cos, sin, width, align_logits); phần tử cuối là None khi không có text.
        """
        x = self.encode(x_in)
        align_logits = None
        if self.align_stage == 'bottleneck':
            x, align_logits = self.condition(x, prompts)

        x = F.relu(self.bn4(self.conv4(x)))
        if self.align_stage == 'conv4':
            # Sau conv4 lưới là 113×113 (stride ~2): mịn gấp đôi cho part nhỏ, đổi lại
            # conv1..res5 không còn được điều kiện bởi prompt.
            x, align_logits = self.condition(x, prompts)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return self.heads(x) + (align_logits,)

    # ------------------------------------------------------------------ loss --
    def compute_loss(self, xc, yc, prompts=None, part_mask=None, **ignored):
        """
        L = L_grasp(Q_T, cos2θ, sin2θ, W) + w_align · warmup · [BCE + Dice](A_T, part_mask).

        :param ignored: nuốt các key thừa của loader (vd `union_pos` từ bản V1) để không phải
                        đổi loader khi đổi objective.
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred, align_logits = self(xc, prompts)

        losses = {
            'p_loss': F.smooth_l1_loss(pos_pred, y_pos),
            'cos_loss': F.smooth_l1_loss(cos_pred, y_cos),
            'sin_loss': F.smooth_l1_loss(sin_pred, y_sin),
            'width_loss': F.smooth_l1_loss(width_pred, y_width),
        }
        loss = sum(losses.values())

        if align_logits is not None and part_mask is not None and self.w_align > 0:
            target = _match_resolution(part_mask, align_logits)
            # BCE trên logits, Dice sau sigmoid: part_mask chỉ chiếm ~2-5% pixel nên chỉ BCE
            # thì nghiệm "toàn 0" đã đủ tốt. Log riêng hai thành phần: warmup làm tổng đi lên
            # trong lúc từng phần đang đi xuống, nhìn tổng thôi thì đọc nhầm.
            losses['align_bce'] = F.binary_cross_entropy_with_logits(align_logits, target)
            losses['align_dice'] = dice_loss(align_logits, target)
            loss = loss + self.lambda_align * (losses['align_bce'] + losses['align_dice'])

        pred = {'pos': pos_pred, 'cos': cos_pred, 'sin': sin_pred, 'width': width_pred}
        if align_logits is not None:
            pred['align'] = torch.sigmoid(align_logits)
        return {'loss': loss, 'losses': losses, 'pred': pred,
                'align_logits': align_logits, 'stats': dict(self.last_stats)}

    @property
    def lambda_align(self):
        """Trọng số L_align *thực tế* của bước hiện tại (đã nhân warmup)."""
        return self.w_align * float(self.loss_warmup)

    def predict(self, xc, prompts=None):
        pos, cos, sin, width, align_logits = self(xc, prompts)
        out = {'pos': pos, 'cos': cos, 'sin': sin, 'width': width}
        if align_logits is not None:
            out['align'] = torch.sigmoid(align_logits)
        return out

    @torch.no_grad()
    def token_alignment(self, x_in, prompts):
        """
        Alignment token-pixel cho việc visualize.

        :return: dict
            tokens     list[list[str]] -- token nội dung của từng prompt (đã bỏ SOT/EOT/pad/dấu câu)
            maps       list[(L_i, H, W)] similarity của từng token, đã chuẩn hoá về [0, 1]
            weights    list[(L_i, H, W)] α_pj -- token nào được chọn tại pixel nào
            attention  (B, 1, H, W) A_T = σ(Σ_j α_pj S_pj), chính thứ đưa vào fusion
        """
        if not self.use_text:
            raise RuntimeError('Model khởi tạo với use_text=False, không có nhánh alignment.')

        feat = self.encode(x_in)
        if self.align_stage == 'conv4':
            feat = F.relu(self.bn4(self.conv4(feat)))
        text_emb, text_mask = self.text_encoder(prompts)
        text_emb = text_emb.to(feat.dtype)
        sim = self.align.per_token(feat, text_emb, text_mask)          # (B, L, H, W)
        align_logits, _, attn = self.align(feat, text_emb, text_mask)

        all_tokens = self.text_encoder.token_strings(prompts)
        tokens, maps, weights = [], [], []
        for b in range(sim.shape[0]):
            keep = text_mask[b].nonzero(as_tuple=True)[0]
            tokens.append([_clean_token(all_tokens[b][i]) for i in keep.tolist()])
            m = sim[b, keep]                                           # (L_i, H, W)
            lo, hi = m.amin(), m.amax()
            maps.append((m - lo) / (hi - lo + 1e-8))
            weights.append(attn[b, keep])
        return {'tokens': tokens, 'maps': maps, 'weights': weights,
                'attention': torch.sigmoid(align_logits)}

    def set_loss_warmup(self, value):
        """Gọi mỗi epoch: 0 -> 1 trong vài epoch đầu."""
        self.loss_warmup.fill_(float(value))

    # ------------------------------------------------------- tương thích V1 --
    def __setstate__(self, state):
        """
        Checkpoint V1 là `torch.save(net)`, tức pickle cả object -- nó không mang những thuộc
        tính V2 thêm vào (`fusion`, `align_mode`, ...). Điền giá trị V1 vào để checkpoint cũ
        (hard-max + gate) vẫn chạy đúng như lúc train dưới code V2.
        """
        super(GenerativeResnetAlign, self).__setstate__(state)
        for name, value in self._V1_DEFAULTS.items():
            if not hasattr(self, name):
                setattr(self, name, value)
        if not hasattr(self, 'init_kwargs'):
            self.init_kwargs = None
        # `align` của V1 là TokenPixelAlignment (không có .mode/.r_proj); pickle giữ nguyên
        # trọng số nhưng class đã đổi tên -> chỉ vá thuộc tính còn thiếu.
        align = getattr(self, 'align', None)
        if align is not None:
            if not hasattr(align, 'mode'):
                align.mode = 'hard'
            if not hasattr(align, 'r_proj'):
                align.r_proj = None
            if not hasattr(align, 'token_temperature'):
                align.token_temperature = 1.0
        if hasattr(self, 'graspability_head'):
            # V2 bỏ Q_g; giữ tham số trong object thì vô hại, nhưng không ai đọc nó nữa.
            logging.getLogger(__name__).info(
                'Checkpoint V1: bỏ qua graspability_head (V2 không dùng Q_g).')


# `TokenPixelAlignment` là tên class của V1. Pickle lưu theo tên module + tên class, nên alias
# này là thứ giữ cho `torch.load` một checkpoint V1 không chết vì AttributeError.
TokenPixelAlignment = TokenRegionAlignment


def _clean_token(token):
    """CLIP BPE gắn hậu tố `</w>` ở token cuối từ -- bỏ đi cho dễ đọc."""
    return token.replace('</w>', '')


def _match_resolution(target, like):
    """
    Hạ target (224×224) xuống đúng lưới của A_T thay vì upsample logits lên -- supervise ở
    resolution thật của feature map, không tạo độ chính xác giả.
    """
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)
