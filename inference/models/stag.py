import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import nn

from inference.models.grconvnet3 import GenerativeResnet

CLIP_MODEL = "openai/clip-vit-base-patch32"
MAX_PROMPT_TOKENS = 16  # the longest GA++ prompt is ~11 words; CLIP's 77 is wasteful

# masked_fill with -inf makes softmax return NaN when *every* token of a prompt is dropped;
# -1e4 is small enough that alpha ~= 0 while staying finite. The empty-prompt case is handled
# separately through the `valid` flag.
NEG_INF = -1e4


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder returning **per-token** embeddings (not a single EOT vector).

    `CLIPTextModel.last_hidden_state` gives the whole sequence; we project it through
    `text_projection` into the same space as CLIP's image embedding, then mask out SOT/EOT/
    padding **and punctuation**.
    """

    def __init__(
        self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS, drop_punctuation=True
    ):
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        from transformers import logging as hf_logging

        # The CLIP checkpoint also carries the vision tower; we only take the text tower, so
        # transformers lists a pile of "UNEXPECTED" keys -- expected, no need to print them.
        hf_logging.set_verbosity_error()
        # train_network.py enables a global basicConfig(INFO), which drags in the hub's HTTP logs.
        for name in ("httpx", "huggingface_hub", "filelock"):
            logging.getLogger(name).setLevel(logging.WARNING)

        self.max_length = max_length
        self.drop_punctuation = drop_punctuation
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # A "does this token carry content" lookup by id, built once for the whole vocabulary
        # (49,408 entries) instead of decoding again every batch. Alignment figures from early
        # runs showed the strongest token was usually `.` -- punctuation absorbs all the
        # attention while carrying no information about the part.
        vocab = self.tokenizer.get_vocab()
        is_content = torch.zeros(max(vocab.values()) + 1, dtype=torch.bool)
        for token, idx in vocab.items():
            is_content[idx] = any(ch.isalnum() for ch in token.replace("</w>", ""))
        self.register_buffer("is_content", is_content, persistent=False)

    @property
    def dim(self):
        return self.model.config.projection_dim

    def train(self, mode=True):
        # Keep the text encoder in eval even under net.train() -- it is frozen.
        super().train(mode)
        self.model.eval()
        return self

    def _tokenize(self, prompts):
        """
        Accepts prompts in either form:

        * `list[str]` -- tokenized here and now.
        * `dict` with `input_ids`/`attention_mask` -- already tokenized inside a DataLoader
          worker (see `PromptTokenizer`). Tokenizing 64 prompts costs 6.3 ms; run in the main
          process those 6.3 ms sit straight on the critical path of every step, moved to the
          workers they disappear.

        :return: dict[str, Tensor] with the two keys input_ids / attention_mask
        """
        if isinstance(prompts, dict):
            return {
                "input_ids": prompts["input_ids"],
                "attention_mask": prompts["attention_mask"],
            }
        return self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def token_strings(self, prompts):
        """
        The tokens actually fed to the encoder, used to label alignment figures.

        :return: list[list[str]] with the same length as forward()'s L dimension (SOT/EOT/pad
                 included; use the same mask as forward to filter them).
        """
        ids = self._tokenize(prompts)["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return [[self.tokenizer.convert_ids_to_tokens(i) for i in seq] for seq in ids]

    @torch.no_grad()
    def forward(self, prompts):
        """
        :param prompts: list[str] of length B, or a pre-tokenized dict (see `_tokenize`)
        :return: (B, L, D) per-token embeddings, (B, L) bool mask of content tokens
        """
        device = next(self.model.parameters()).device
        tok = {
            k: v.to(device, non_blocking=True)
            for k, v in self._tokenize(prompts).items()
        }
        out = self.model.text_model(**tok).last_hidden_state  # (B, L, hidden)
        emb = self.model.text_projection(out)  # (B, L, D)

        ids = tok["input_ids"]
        # `.bool()` already allocates a new tensor, so the writes below do not touch the
        # original attention_mask -- important when it comes from the DataLoader and is reused
        # elsewhere.
        mask = tok["attention_mask"].bool()
        # Drop SOT (always at position 0) and EOT (the last attended token).
        mask[:, 0] = False
        eot = tok["attention_mask"].sum(dim=1) - 1
        mask[torch.arange(mask.shape[0], device=device), eot] = False
        if self.drop_punctuation:
            mask &= self.is_content.to(device)[ids]
        return emb, mask


class PromptTokenizer:
    """
    Tokenize prompts *inside the DataLoader workers* instead of the main process.

    The dataset calls this object per sample; `default_collate` stacks the 1-D tensors into
    (B, L), and `CLIPTextEncoder._tokenize` consumes that dict directly. The resulting tokens
    are identical to the in-process path -- same tokenizer, same `padding='max_length'`, same
    `max_length`.

    Must be picklable so `num_workers > 0` can ship it to the workers; `CLIPTokenizerFast` is.
    """

    def __init__(self, model_name=CLIP_MODEL, max_length=MAX_PROMPT_TOKENS):
        from transformers import CLIPTokenizerFast
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.max_length = max_length
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    def __call__(self, prompt):
        tok = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
        }


class TokenRegionAlignment(nn.Module):
    """
    Relevance between every text token and every pixel of the feature map.

    Returns the *logits* of `A_T` (before sigmoid) so they can go straight into
    BCEWithLogits -- numerically more stable -- together with `R_T` (the region's text
    feature) and the token weights `alpha` for visualisation.
    """

    def __init__(
        self,
        visual_dim,
        text_dim,
        proj_dim=128,
        region_dim=64,
        mode="soft",
        region_text=True,
        token_temperature=1.0,
        init_temperature=0.07,
    ):
        super().__init__()
        if mode not in ("soft", "hard"):
            raise ValueError(f"align_mode must be 'soft' or 'hard', got '{mode}'")

        self.mode = mode
        self.token_temperature = token_temperature
        self.v_proj = nn.Conv2d(visual_dim, proj_dim, kernel_size=1)
        self.t_proj = nn.Linear(text_dim, proj_dim)
        # W_r: the text feature handed to the decoder, kept separate from W_t so that the
        # space used for *matching* and the space used for *describing* are not the same one.
        self.r_proj = nn.Linear(text_dim, region_dim) if region_text else None
        # A cosine in [-1, 1] has too little range for BCE -> a learnable temperature.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    @property
    def region_dim(self):
        return 0 if self.r_proj is None else self.r_proj.out_features

    def per_token(self, feat, text_emb, text_mask):
        """
        S_pj for **each** token -- this is what visualises "which token claims which region".

        :return: (B, L, H, W) logits; masked tokens receive NEG_INF
        """
        v = F.normalize(self.v_proj(feat), dim=1)  # (B, P, H, W)
        t = F.normalize(self.t_proj(text_emb), dim=-1)  # (B, L, P)

        sim = torch.einsum("bphw,blp->blhw", v, t) * self.logit_scale.exp().clamp(
            max=100.0
        )
        return sim.masked_fill(~text_mask[:, :, None, None], NEG_INF)

    def token_weights(self, sim, text_mask):
        """alpha_pj: soft token selection per pixel (or one-hot when mode='hard')."""
        if self.mode == "hard":
            # Hard variant: only the winning token contributes. Kept as an ablation arm.
            hard = torch.zeros_like(sim)
            return hard.scatter_(1, sim.argmax(dim=1, keepdim=True), 1.0)
        return torch.softmax(sim / self.token_temperature, dim=1)

    def forward(self, feat, text_emb, text_mask):
        """
        :param feat: (B, C, H, W)
        :param text_emb: (B, L, D)
        :param text_mask: (B, L) bool, True on content tokens
        :return: (align_logits (B,1,H,W), region (B,R,H,W) or None, alpha (B,L,H,W))
        """
        sim = self.per_token(feat, text_emb, text_mask)
        attn = self.token_weights(sim, text_mask)
        logits = (attn * sim).sum(dim=1)  # (B, H, W)

        # Prompts whose tokens are all dropped (empty, or pure punctuation): alpha is
        # meaningless -> return 0 so the loss is not dragged by NEG_INF and fusion gets no noise.
        valid = text_mask.any(dim=1)
        logits = torch.where(valid[:, None, None], logits, torch.zeros_like(logits))
        attn = attn * valid[:, None, None, None]

        region = None
        if self.r_proj is not None:
            r = self.r_proj(text_emb)  # (B, L, R)
            region = torch.einsum("blhw,blr->brhw", attn, r)
        return logits.unsqueeze(1), region, attn


class RegionFusion(nn.Module):
    """
    F' = F + phi([F, F ⊙ A_T, R_T]).

    A residual rather than a multiplicative gate: the last conv is zero-initialised, so at the
    start F' = F is exactly the baseline and the language branch phases in as `A_T` becomes
    meaningful. A gate `F ⊙ (1+λA_T)` would multiply the features by a random map from step 0.
    """

    def __init__(self, feat_dim, region_dim=0):
        super().__init__()
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


class STAG(GenerativeResnet):
    """
    GR-ConvNet-3 plus the language branch. `use_text=False` turns it back into a plain
    GR-ConvNet-3 (the image-only baseline shares this file and its training budget).
    """

    accepts_extras = True
    # The name registered in inference/models/__init__.py -- utils/checkpoint.py rebuilds the
    # model from it.
    network_name = "stag"

    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        channel_size=32,
        dropout=False,
        prob=0.0,
        use_text=True,
        align_mode="soft",
        region_text=True,
        fusion="residual",
        align_stage="bottleneck",
        proj_dim=128,
        region_dim=64,
        token_temperature=1.0,
        gate_lambda=1.0,
        w_align=1.0,
        text_encoder=None,
    ):
        """
        :param align_mode: 'soft' = alpha_pj softmax over tokens; 'hard' = argmax over tokens
        :param region_text: whether to feed `R_T` to the decoder
        :param fusion: 'residual' = F + phi([F, F⊙A_T, R_T]); 'gate' = F ⊙ (1 + λ·A_T)
        :param align_stage: 'bottleneck' (56×56, after res5) or 'conv4' (113×113, after conv4).
                            Parts are small relative to a stride of 4; if `A_T` looks blurry,
                            try 'conv4'.
        :param token_temperature: tau_t of the softmax over tokens. S is already divided by a
                                  learned tau, so tau_t only tunes how peaked the token
                                  selection is.
        :param text_encoder: reuse an existing CLIPTextEncoder (saves memory when loading
                             several models)
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            channel_size=channel_size,
            dropout=dropout,
            prob=prob,
        )

        if fusion not in ("residual", "gate"):
            raise ValueError(f"fusion must be 'residual' or 'gate', got '{fusion}'")
        if align_stage not in ("bottleneck", "conv4"):
            raise ValueError(
                f"align_stage must be 'bottleneck' or 'conv4', got '{align_stage}'"
            )

        # Enough to rebuild an empty model when loading a state_dict checkpoint
        # (utils/checkpoint.py).
        self.init_kwargs = dict(
            input_channels=input_channels,
            output_channels=output_channels,
            channel_size=channel_size,
            dropout=dropout,
            prob=prob,
            use_text=use_text,
            align_mode=align_mode,
            region_text=region_text,
            fusion=fusion,
            align_stage=align_stage,
            proj_dim=proj_dim,
            region_dim=region_dim,
            token_temperature=token_temperature,
            gate_lambda=gate_lambda,
            w_align=w_align,
        )

        self.use_text = use_text
        self.align_mode = align_mode
        self.use_region_text = bool(region_text)
        self.fusion = fusion
        self.align_stage = align_stage
        self.gate_lambda = gate_lambda
        self.w_align = w_align
        # Enabled with `with net.collecting_stats():` -- see utils/diagnostics.py. Off by
        # default so the hot training loop does not compute entropy/argmax over (B, L, H, W)
        # at every step.
        self.collect_stats = False
        self.last_stats = {}
        # Warmup driven by the training loop: 0 -> 1 so the gradient of L_align does not
        # dominate while A_T is still random.
        self.register_buffer("loss_warmup", torch.tensor(1.0), persistent=False)

        if use_text:
            feat_dim = (
                channel_size * 4 if align_stage == "bottleneck" else channel_size * 2
            )
            self.text_encoder = (
                text_encoder if text_encoder is not None else CLIPTextEncoder()
            )
            self.align = TokenRegionAlignment(
                feat_dim,
                self.text_encoder.dim,
                proj_dim=proj_dim,
                region_dim=region_dim,
                mode=align_mode,
                region_text=region_text,
                token_temperature=token_temperature,
            )
            if fusion == "residual":
                self.fuse = RegionFusion(feat_dim, self.align.region_dim)

    # --------------------------------------------------------------- forward --
    def encode(self, x_in):
        """conv1..res5 -> F at the bottleneck (56×56 for a 224 input)."""
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

    def condition(self, feat, prompts):
        """
        One pass of token-region alignment plus fusion on the feature map at hand.

        :return: (F', align_logits) -- align_logits is None when there is no prompt
        """
        if not self.use_text or prompts is None:
            return feat, None

        text_emb, text_mask = self.text_encoder(prompts)
        align_logits, region, attn = self.align(
            feat, text_emb.to(feat.dtype), text_mask
        )
        attention = torch.sigmoid(align_logits)

        if self.fusion == "gate":
            # Multiplicative gate, without R_T.
            fused = feat * (1.0 + self.gate_lambda * attention)
        else:
            fused = self.fuse(feat, attention, region)

        if self.collect_stats:
            self.last_stats = self._forward_stats(feat, fused, attn, text_mask)
        return fused, align_logits

    def _forward_stats(self, feat, fused, attn, text_mask):
        from utils.diagnostics import fusion_stats, token_stats, winning_tokens

        stats = {f"token/{k}": v for k, v in token_stats(attn, text_mask).items()}
        stats.update({f"fusion/{k}": v for k, v in fusion_stats(feat, fused).items()})
        stats["token/logit_scale"] = float(self.align.logit_scale.exp())
        stats["token/temperature"] = 1.0 / max(stats["token/logit_scale"], 1e-8)
        # Kept as CPU tensors: the training loop aggregates first, then maps to token names.
        stats["_winning_tokens"] = winning_tokens(attn, text_mask).cpu()
        stats["_text_mask"] = text_mask.cpu()
        return stats

    @contextlib.contextmanager
    def collecting_stats(self):
        """Collect diagnostics for the forward passes inside the `with` block."""
        previous = self.collect_stats
        self.collect_stats = True
        try:
            yield self
        finally:
            self.collect_stats = previous

    def forward(self, x_in, prompts=None):
        """
        :return: (pos, cos, sin, width, align_logits); the last item is None without text.
        """
        x = self.encode(x_in)
        align_logits = None
        if self.align_stage == "bottleneck":
            x, align_logits = self.condition(x, prompts)

        x = F.relu(self.bn4(self.conv4(x)))
        if self.align_stage == "conv4":
            # After conv4 the grid is 113×113 (stride ~2): twice as fine for small parts, at
            # the cost of conv1..res5 no longer being conditioned on the prompt.
            x, align_logits = self.condition(x, prompts)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return self.heads(x) + (align_logits,)

    # ------------------------------------------------------------------ loss --
    def compute_loss(self, xc, yc, prompts=None, part_mask=None, **ignored):
        """
        L = L_grasp(Q_T, cos2θ, sin2θ, W) + w_align · warmup · [BCE + Dice](A_T, part_mask).

        :param ignored: swallows extra keys from the loader so the loader need not change when
                        the objective does.
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred, align_logits = self(xc, prompts)

        losses = {
            "p_loss": F.smooth_l1_loss(pos_pred, y_pos),
            "cos_loss": F.smooth_l1_loss(cos_pred, y_cos),
            "sin_loss": F.smooth_l1_loss(sin_pred, y_sin),
            "width_loss": F.smooth_l1_loss(width_pred, y_width),
        }
        loss = sum(losses.values())

        if align_logits is not None and part_mask is not None and self.w_align > 0:
            target = _match_resolution(part_mask, align_logits)
            # BCE on logits, Dice after sigmoid: part_mask covers only ~2-5% of the pixels, so
            # BCE alone is already well served by the all-zero solution. The two terms are
            # logged separately: warmup pushes the *total* up while each part goes down, and
            # reading the total alone is misleading.
            losses["align_bce"] = F.binary_cross_entropy_with_logits(
                align_logits, target
            )
            losses["align_dice"] = dice_loss(align_logits, target)
            loss = loss + self.lambda_align * (
                losses["align_bce"] + losses["align_dice"]
            )

        pred = {"pos": pos_pred, "cos": cos_pred, "sin": sin_pred, "width": width_pred}
        if align_logits is not None:
            pred["align"] = torch.sigmoid(align_logits)
        return {
            "loss": loss,
            "losses": losses,
            "pred": pred,
            "align_logits": align_logits,
            "stats": dict(self.last_stats),
        }

    @property
    def lambda_align(self):
        """The *effective* L_align weight at the current step (warmup already applied)."""
        return self.w_align * float(self.loss_warmup)

    def predict(self, xc, prompts=None):
        pos, cos, sin, width, align_logits = self(xc, prompts)
        out = {"pos": pos, "cos": cos, "sin": sin, "width": width}
        if align_logits is not None:
            out["align"] = torch.sigmoid(align_logits)
        return out

    @torch.no_grad()
    def token_alignment(self, x_in, prompts):
        """
        Token-pixel alignment, for visualisation.

        :return: dict
            tokens     list[list[str]] -- content tokens of each prompt (SOT/EOT/pad/punctuation
                       already removed)
            maps       list[(L_i, H, W)] per-token similarity, normalised to [0, 1]
            weights    list[(L_i, H, W)] alpha_pj -- which token is picked at which pixel
            attention  (B, 1, H, W) A_T = sigma(sum_j alpha_pj S_pj), exactly what fusion sees
        """
        if not self.use_text:
            raise RuntimeError(
                "Model was built with use_text=False; there is no alignment branch."
            )

        feat = self.encode(x_in)
        if self.align_stage == "conv4":
            feat = F.relu(self.bn4(self.conv4(feat)))
        text_emb, text_mask = self.text_encoder(prompts)
        text_emb = text_emb.to(feat.dtype)
        sim = self.align.per_token(feat, text_emb, text_mask)  # (B, L, H, W)
        align_logits, _, attn = self.align(feat, text_emb, text_mask)

        all_tokens = self.text_encoder.token_strings(prompts)
        tokens, maps, weights = [], [], []
        for b in range(sim.shape[0]):
            keep = text_mask[b].nonzero(as_tuple=True)[0]
            tokens.append([_clean_token(all_tokens[b][i]) for i in keep.tolist()])
            m = sim[b, keep]  # (L_i, H, W)
            lo, hi = m.amin(), m.amax()
            maps.append((m - lo) / (hi - lo + 1e-8))
            weights.append(attn[b, keep])
        return {
            "tokens": tokens,
            "maps": maps,
            "weights": weights,
            "attention": torch.sigmoid(align_logits),
        }

    def set_loss_warmup(self, value):
        """Called every epoch: 0 -> 1 over the first few epochs."""
        self.loss_warmup.fill_(float(value))


def _clean_token(token):
    """CLIP BPE appends `</w>` to the last token of a word -- strip it for readability."""
    return token.replace("</w>", "")


def _match_resolution(target, like):
    """
    Downsample the target (224×224) onto A_T's grid instead of upsampling the logits --
    supervise at the feature map's real resolution rather than inventing precision.
    """
    if target.shape[-2:] == like.shape[-2:]:
        return target
    return F.adaptive_avg_pool2d(target, like.shape[-2:]).clamp(0.0, 1.0)
