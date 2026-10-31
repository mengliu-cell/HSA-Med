# Copyright 2025. Apache-2.0.
"""Anatomical Prototype Alignment (APA) and Disease-aware Spatial Enhancement (DSE).

APA learns K shared Gaussian *spatial* prototypes, without organ labels or a
one-to-one disease/prototype assignment. Use a separate module instance for each
dataset. Disease embeddings come from an external text encoder; this module
neither tokenizes disease names nor detaches their embeddings.

For one image, APA accepts row-major visual tokens (M, D) and disease embeddings
(C, text_dim). It returns disease-conditioned anatomical features (C, D) and
spatial responses (C, M). DSE returns enhanced visual tokens (C, M, D). The
bbox loss uses normalized xyxy coordinates and indices into the C disease
queries, averaging BCE over annotated instances, including repeated diseases.
DSE needs no separate loss: train with L_gnd + lambda_APA * L_APA.

ADSA wraps APA -> DSE for concatenated Qwen2.5-VL merged image tokens. Its
single-query interface is compatible with existing callers. For multiple
queries, use category_image_embeds explicitly: these sequences cannot be
scattered into the original M image-token positions without a decoder-side
integration. No disease averaging or detection/regression head is introduced.
This architecture changes the parameter layout of older ADSA checkpoints.

Qwen2_5_VLADSAConnector connects these modules to a native Qwen2.5-VL model:

    image -> Qwen visual encoder/merger -> APA -> DSE --+
    candidate names -> Qwen text backbone -> APA       +-> Qwen LLM decoder
    full instruction --------------------------------+

Qwen2.5-VL has a decoder-only text backbone, not a separate text encoder.
The connector reuses that backbone in a text-only, cache-free pass to encode
each candidate name. All candidates (including absent diseases and Unknown)
are supplied during both training and inference, without reading GT answers.
Each enhanced image sequence remains independent and is preceded by its
category name. Shared prototype features are included once per image.
"""

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Previous X-ray categories:
# DEFAULT_DISEASE_CATEGORIES = [
#     "Pneumonia", "Cardiomegaly", "Pneumothorax", "Atelectasis", "Pleural Effusion",
#     "Pulmonary Edema", "Consolidation", "Infiltration", "Nodule", "Pleural Thickening",
#     "Mass", "Unknown",
# ]

DEFAULT_DISEASE_CATEGORIES = [
    "Hepatocellular Carcinoma",
    "Intrahepatic Cholangiocarcinoma",
    "Liver Metastasis",
    "Hepatic Hemangioma",
    "Focal Nodular Hyperplasia",
    "Hepatocellular Adenoma",
    "Hepatic Cyst",
    "Liver Abscess",
    "Unknown",
]


def _grid_coordinates(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Global normalized token centers (M, 2), ordered as (x, y)."""
    if h <= 0 or w <= 0:
        raise ValueError("Grid dimensions must be positive.")
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten()), dim=-1)


@dataclass
class APAOutput:
    anatomy_features: torch.Tensor       # (C, D): h_hat, input to DSE
    prototype_features: torch.Tensor     # (K, D): z_k
    prototype_attention: torch.Tensor    # (C, K): alpha_ck
    gaussian_responses: torch.Tensor     # (K, M): g_nk transposed, NOT normalized
    token_association: torch.Tensor      # (K, M): a_nk transposed
    spatial_maps: torch.Tensor           # (C, M): sum_k alpha_ck * g_nk
    coordinates: torch.Tensor            # (M, 2): normalized global (x, y)


class APA(nn.Module):
    """Learn coarse regions from disease labels and boxes, without organ priors.

    K is independent of the disease count C. For K=16, Gaussian centers start
    at the centers of a uniform 4x4 grid. Centers use sigmoid and scales use
    softplus + eps. There are no separate learnable semantic prototype vectors:
    z_k is the Gaussian-weighted aggregation of the current image's tokens.

    ``proto_dim=None`` uses D-dimensional query/key projections as in the
    manuscript; an explicit value permits a smaller attention matching space.
    ``disease_embeds=None`` is an unconditioned compatibility mode with uniform
    prototype attention and a zero text residual, not disease-aware alignment.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        text_dim: int = 2048,
        num_prototypes: int = 16,
        proto_dim: Optional[int] = None,
        initial_sigma: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        if min(embed_dim, text_dim, num_prototypes) <= 0:
            raise ValueError("Feature dimensions and num_prototypes must be positive.")
        if proto_dim is not None and proto_dim <= 0:
            raise ValueError("proto_dim must be positive.")
        if eps <= 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive.")
        if not math.isfinite(initial_sigma) or initial_sigma <= eps:
            raise ValueError("initial_sigma must be finite and greater than eps.")
        self.embed_dim = embed_dim
        self.text_dim = text_dim
        self.num_prototypes = num_prototypes
        self.proto_dim = embed_dim if proto_dim is None else proto_dim
        self.eps = eps

        cols = math.ceil(math.sqrt(num_prototypes))
        rows = math.ceil(num_prototypes / cols)
        indices = torch.arange(num_prototypes)
        centers = torch.stack(
            ((indices % cols + 0.5) / cols, (indices // cols + 0.5) / rows),
            dim=-1,
        )
        self.proto_center = nn.Parameter(torch.logit(centers))
        # Stable inverse softplus so the initial standard deviation is exact.
        sigma = torch.tensor(initial_sigma - eps)
        raw_sigma = sigma + torch.log(-torch.expm1(-sigma))
        self.proto_raw_sigma = nn.Parameter(raw_sigma.expand(num_prototypes, 2).clone())

        self.text_proj = (
            nn.Identity() if text_dim == embed_dim else nn.Linear(text_dim, embed_dim)
        )
        self.query_proj = nn.Linear(embed_dim, self.proto_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, self.proto_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_norm = nn.LayerNorm(embed_dim)

    @property
    def centers(self) -> torch.Tensor:
        return self.proto_center.float().sigmoid()

    @property
    def scales(self) -> torch.Tensor:
        return F.softplus(self.proto_raw_sigma.float()) + self.eps

    def gaussian_responses(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Unnormalized exp(-d_nk), in float32 for stable geometry and BCE."""
        diff = coordinates.float().unsqueeze(0) - self.centers.unsqueeze(1)
        distance = 0.5 * (diff / self.scales.unsqueeze(1)).square().sum(dim=-1)
        return torch.exp(-distance)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        h: int,
        w: int,
        disease_embeds: Optional[torch.Tensor] = None,
    ) -> APAOutput:
        if visual_tokens.ndim != 2 or visual_tokens.shape != (h * w, self.embed_dim):
            raise ValueError(f"Expected visual_tokens shape ({h * w}, {self.embed_dim}).")
        coordinates = _grid_coordinates(h, w, visual_tokens.device)
        gaussian = self.gaussian_responses(coordinates)
        # Normalize over TOKENS for each prototype, never over prototypes.
        association = gaussian / (gaussian.sum(dim=-1, keepdim=True) + self.eps)
        with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
            prototype_features = association @ visual_tokens.float()
        prototype_features = prototype_features.to(visual_tokens.dtype)

        if disease_embeds is None:
            text = visual_tokens.new_zeros((1, self.embed_dim))
            attention = gaussian.new_full((1, self.num_prototypes), 1 / self.num_prototypes)
        else:
            if disease_embeds.ndim == 1:
                disease_embeds = disease_embeds.unsqueeze(0)
            if (disease_embeds.ndim != 2 or disease_embeds.shape[0] == 0
                    or disease_embeds.shape[1] != self.text_dim):
                raise ValueError(f"Expected disease_embeds shape (C, {self.text_dim}), C > 0.")
            text = self.text_proj(disease_embeds.to(visual_tokens))
            query = self.query_proj(text)
            key = self.key_proj(prototype_features)
            with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
                logits = query.float() @ key.float().t() / math.sqrt(self.proto_dim)
                attention = logits.softmax(dim=-1)

        values = self.value_proj(prototype_features)
        with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
            context = attention @ values.float()
            spatial_maps = attention @ gaussian
        anatomy_features = self.output_norm(text + context.to(text.dtype))
        return APAOutput(
            anatomy_features=anatomy_features,
            prototype_features=prototype_features,
            prototype_attention=attention,
            gaussian_responses=gaussian,
            token_association=association,
            spatial_maps=spatial_maps,
            coordinates=coordinates,
        )


class DSE(nn.Module):
    """FiLM(h_hat) followed by feature-wise gating and residual fusion.

    v_tilde_cn = gamma_c * LN(v_n) + beta_c
    s_cn = sigmoid(W_g [v_n; v_tilde_cn; h_hat_c] + b_g)
    v_enh_cn = v_n + s_cn * v_tilde_cn

    With identity_init, only the final FiLM layer starts at zero. This preserves
    the pretrained tokens initially, while allowing nonzero FiLM gradients on
    the first update. There is no additional zero-initialized residual scale
    that could block learning of the entire modulation branch.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        film_hidden: int = 512,
        dropout: float = 0.0,
        identity_init: bool = True,
    ):
        super().__init__()
        if embed_dim <= 0 or film_hidden <= 0:
            raise ValueError("Feature dimensions must be positive.")
        self.embed_dim = embed_dim
        self.film = nn.Sequential(
            nn.Linear(embed_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * embed_dim),
        )
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(3 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if identity_init:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, visual_tokens: torch.Tensor, anatomy_features: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 2 or visual_tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected visual_tokens shape (M, {self.embed_dim}).")
        if (anatomy_features.ndim != 2 or anatomy_features.shape[0] == 0
                or anatomy_features.shape[-1] != self.embed_dim):
            raise ValueError(f"Expected anatomy_features shape (C, {self.embed_dim}), C > 0.")
        gamma, beta = self.film(anatomy_features).chunk(2, dim=-1)
        modulated = gamma[:, None, :] * self.visual_norm(visual_tokens)[None, :, :] + beta[:, None, :]
        # Exactly W_g [v; v_tilde; h_hat], without materializing a (C,M,3D) tensor.
        w_visual, w_modulated, w_anatomy = self.gate.weight.split(self.embed_dim, dim=-1)
        gate_logits = (
            F.linear(visual_tokens, w_visual)[None, :, :]
            + F.linear(modulated, w_modulated)
            + F.linear(anatomy_features, w_anatomy, self.gate.bias)[:, None, :]
        )
        return visual_tokens[None, :, :] + self.dropout(gate_logits.sigmoid() * modulated)


@dataclass
class ADSAOutput:
    """Per-image outputs, with disease dimensions preserved.

    image_embeds: concatenated (sum_M, D) only when each image has one query
        (or uses the unconditioned fallback); None if any image has C > 1.
    saliency: (H, W) for a single query, otherwise (C, H, W), without min-max
        normalization. These are the raw Gaussian mixtures required by BCE.
    category_image_embeds: list of (C_i, M_i, D) enhanced token sequences.
    apa_outputs: intermediate features/attention for inspection or decoding.
    """

    image_embeds: Optional[torch.Tensor]
    saliency: List[torch.Tensor]
    grids: List[Tuple[int, int]]
    category_image_embeds: List[torch.Tensor]
    apa_outputs: List[APAOutput]


class ADSA(nn.Module):
    """APA -> DSE wrapper for static images and external disease embeddings.

    The default query/key dimension is D. ``proto_dim`` remains an optional
    smaller attention space. ``spatial_prior_weight`` is accepted only as None
    or 1 for call compatibility: APA uses pure Gaussian pooling, with no
    learned semantic/spatial assignment mixing coefficient.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        text_dim: int = 2048,
        num_prototypes: int = 16,
        proto_dim: Optional[int] = None,
        spatial_prior_weight: Optional[float] = None,
        film_hidden: int = 512,
        dropout: float = 0.0,
        *,
        initial_sigma: float = 0.25,
        identity_init: bool = True,
    ):
        super().__init__()
        if spatial_prior_weight not in (None, 1.0):
            raise ValueError("APA uses Gaussian pooling; spatial_prior_weight must be None or 1.")
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.apa = APA(embed_dim, text_dim, num_prototypes, proto_dim, initial_sigma)
        self.dse = DSE(embed_dim, film_hidden, dropout, identity_init)

    @property
    def gamma(self) -> torch.Tensor:
        """FiLM scale bias for legacy ``adsa_gamma`` logging, not a residual multiplier."""
        return self.dse.film[-1].bias[:self.embed_dim]

    def forward(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
        disease_embeds: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> ADSAOutput:
        """Process concatenated, row-major merged tokens.

        disease_embeds[i] is (text_dim,) or (C_i, text_dim). Query rows define
        the category indices used by apa_alignment_loss. All candidate queries
        should be available at inference too; no boxes are used during forward.
        With no query, routing is uniform and enhancement is unconditioned.
        A temporal grid t != 1 is rejected rather than treated as a taller
        image with incorrect global coordinates.
        """
        if image_embeds.ndim != 2 or image_embeds.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected image_embeds shape (sum_M, {self.embed_dim}).")
        if (image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3
                or image_grid_thw.shape[0] == 0):
            raise ValueError("image_grid_thw must have shape (num_images, 3), num_images > 0.")
        if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be a positive integer.")
        if disease_embeds is not None and len(disease_embeds) != len(image_grid_thw):
            raise ValueError("Provide exactly one disease embedding entry per image.")

        grids, split_sizes = [], []
        m = spatial_merge_size
        for t, h, w in image_grid_thw.tolist():
            if t != 1:
                raise ValueError("APA expects static images (t=1), not stacked video frames.")
            if h <= 0 or w <= 0 or h % m or w % m or int(h) != h or int(w) != w:
                raise ValueError("Patch grid dimensions must be positive integers divisible by spatial_merge_size.")
            hh, ww = int(h) // m, int(w) // m
            grids.append((hh, ww))
            split_sizes.append(hh * ww)
        if sum(split_sizes) != image_embeds.shape[0]:
            raise ValueError("Merged token count does not match image_grid_thw.")

        enhanced, saliency, apa_outputs = [], [], []
        for i, (tokens, (h, w)) in enumerate(zip(image_embeds.split(split_sizes), grids)):
            queries = None if disease_embeds is None else disease_embeds[i]
            result = self.apa(tokens, h, w, queries)
            enhanced.append(self.dse(tokens, result.anatomy_features))
            maps = result.spatial_maps.reshape(-1, h, w)
            saliency.append(maps[0] if maps.shape[0] == 1 else maps)
            apa_outputs.append(result)
        single_query = all(tokens.shape[0] == 1 for tokens in enhanced)
        return ADSAOutput(
            image_embeds=torch.cat([tokens[0] for tokens in enhanced]) if single_query else None,
            saliency=saliency,
            grids=grids,
            category_image_embeds=enhanced,
            apa_outputs=apa_outputs,
        )


def extract_disease_categories(instruction: str) -> List[str]:
    """Read a literal candidate list from the USER instruction, without eval.

    Example: "Please determine ... ['Hepatic Cyst', 'Liver Abscess', 'Unknown']".
    Pass only the user prompt, never the assistant answer or full conversation.
    A complete bracketed list is required; ambiguous/malformed input is rejected.
    Explicit lists of disease names can also be passed directly to the connector.
    """
    candidates = []
    for match in re.finditer(r"\[[^\[\]]*\]", instruction):
        try:
            value = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list) and value and all(isinstance(name, str) for name in value):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("User instruction must contain exactly one complete disease-name list.")
    return _validate_category_names(candidates[0])


def _validate_category_names(names: List[str]) -> List[str]:
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Provide a nonempty list of nonempty disease names.")
    names = [name.strip() for name in names]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Candidate disease names must be unique within each image.")
    return names


@dataclass
class QwenADSAInputs:
    """Expanded decoder input and APA outputs, without re-encoding the image.

    input_ids includes virtual feature slots; inputs_embeds holds their actual
    prototype/enhanced features. Use forward_kwargs() for a native model forward.
    Use connector.generate() for generation so cached M-RoPE stays consistent.
    labels has -100 at ALL inserted feature/name/delimiter slots. Original
    response labels remain aligned after expansion, and padding is masked.
    """

    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    image_grid_thw: torch.Tensor         # repeated per disease, for native M-RoPE
    labels: Optional[torch.Tensor]
    adsa_output: ADSAOutput
    candidate_categories: List[List[str]]  # per original image; defines loss indices

    def forward_kwargs(self) -> dict:
        # Pixels/grids are intentionally omitted: features are already embedded,
        # and position_ids already incorporates all repeated image grids.
        kwargs = dict(
            input_ids=self.input_ids,
            inputs_embeds=self.inputs_embeds,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
        )
        if self.labels is not None:
            kwargs["labels"] = self.labels
        return kwargs


class Qwen2_5_VLADSAConnector(nn.Module):
    """Insert APA/DSE between native Qwen2.5-VL encoders and its LLM decoder.

    This is a model connection interface, not an SFT trainer. It neither freezes
    parameters nor loads another text model. Both encoding and generation use
    the SAME Qwen text backbone. Gradients follow the caller's requires_grad
    settings, including LoRA. Tested with the HF layout exposing
    model.get_image_features, model.language_model and model.get_rope_index.

    Disease queries can be a user instruction containing a literal list, a
    shared list of names, or a list of name lists aligned to original images.
    Omitting disease_queries uses DEFAULT_DISEASE_CATEGORIES (9 liver classes).
    Query rows define bbox category_indices. The complete instruction remains
    in input_ids; only candidate NAMES are used for the query-encoding pass.

    Each original image block becomes:
        [K shared prototype features]
        disease_1: <vision_start> [M enhanced tokens] <vision_end>
        ...
        disease_C: <vision_start> [M enhanced tokens] <vision_end>

    Enhanced tokens remain in row-major order. Native M-RoPE is recomputed
    with C copies of the original spatial grid; no temporal stretching or
    averaging across diseases is used. The visual encoder runs once per image.

    Example (processor inputs must contain images, prompt and optionally labels):
        connector = Qwen2_5_VLADSAConnector(qwen, processor.tokenizer)
        packed = connector.prepare_inputs(**batch, disease_queries=user_prompt)
        outputs = connector(**packed.forward_kwargs())  # or qwen(**...)
        loss_apa = apa_alignment_loss(
            packed.adsa_output.saliency, packed.adsa_output.grids,
            normalized_boxes, category_indices,
        )
        loss = outputs.loss + 0.5 * loss_apa

    forward() delegates to Qwen; prepare_inputs() performs the encoder ->
    APA/DSE -> decoder packing. generate() accepts the original processor
    inputs and returns generated continuation IDs ONLY, excluding the expanded
    prefix, ready for tokenizer.batch_decode(..., skip_special_tokens=True).
    """

    def __init__(self, qwen_model: nn.Module, tokenizer: Any, adsa: Optional[ADSA] = None):
        super().__init__()
        base = getattr(qwen_model, "model", None)
        required = ("get_image_features", "language_model", "get_rope_index")
        if base is None or not all(hasattr(base, name) for name in required):
            raise ValueError(
                "Expected the Qwen2.5-VL layout with model.get_image_features, "
                "model.language_model and model.get_rope_index."
            )
        self.qwen_model = qwen_model
        self.tokenizer = tokenizer
        embedding = qwen_model.get_input_embeddings()
        # ZeRO-3 can partition embedding.weight into an empty 1-D tensor
        # outside its forward; architecture dimensions still live in config.
        dim = getattr(qwen_model.config, "text_config", qwen_model.config).hidden_size
        if qwen_model.config.vision_config.out_hidden_size != dim:
            raise ValueError("Qwen merged vision features must use the LLM embedding dimension.")
        if adsa is None:
            adsa = ADSA(embed_dim=dim, text_dim=dim).to(embedding.weight)
        if adsa.embed_dim != dim or adsa.apa.text_dim != dim:
            raise ValueError("ADSA visual/text dimensions must match this Qwen model.")
        self.adsa = adsa

    def forward(self, **decoder_kwargs):
        """Run the native decoder on packed.forward_kwargs(), preserving its outputs."""
        return self.qwen_model(**decoder_kwargs)

    def _category_lists(self, queries, num_images: int) -> List[List[str]]:
        if queries is None:
            names = _validate_category_names(DEFAULT_DISEASE_CATEGORIES)
            return [list(names) for _ in range(num_images)]
        if isinstance(queries, str):
            names = extract_disease_categories(queries)
            return [list(names) for _ in range(num_images)]
        if not isinstance(queries, (list, tuple)) or not queries:
            raise ValueError("disease_queries must be a user instruction or candidate-name lists.")
        if all(isinstance(name, str) for name in queries):
            names = _validate_category_names(list(queries))
            return [list(names) for _ in range(num_images)]
        if len(queries) != num_images or not all(isinstance(names, (list, tuple)) for names in queries):
            raise ValueError("Per-image candidate lists must align with image_grid_thw.")
        return [_validate_category_names(list(names)) for names in queries]

    def _encode_categories(self, categories: List[List[str]]) -> List[torch.Tensor]:
        # Encode identical names once per call, retaining gradients and query order.
        names = list(dict.fromkeys(name for group in categories for name in group))
        device = self.qwen_model.get_input_embeddings().weight.device
        tokens = self.tokenizer(
            names, padding=True, return_tensors="pt", add_special_tokens=False,
        )
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        if (mask.sum(-1) == 0).any():
            raise ValueError("Each candidate disease must tokenize to at least one token.")
        # Direct text-only backbone call avoids image features, GT answer tokens,
        # and the outer multimodal model's mutable generation rope_deltas.
        positions = mask.long().cumsum(-1).sub(1).clamp_min(0)
        result = self.qwen_model.model.language_model(
            input_ids=ids, attention_mask=mask, position_ids=positions,
            use_cache=False, return_dict=True,
        )
        hidden = result.last_hidden_state
        weights = mask.to(hidden.device, hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        index = {name: i for i, name in enumerate(names)}
        return [pooled[[index[name] for name in group]] for group in categories]

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        disease_queries: Optional[Union[str, List[str], List[List[str]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *,
        padding_side: str = "left",
    ) -> QwenADSAInputs:
        """Encode images/names and expand image placeholders before the decoder.

        Supply the original processor batch (static images only). Existing
        position_ids are not accepted because expansion requires new positions.
        Optional labels must already mask the original prompt with -100.
        Boxes are not inputs to this method and are never needed at inference.
        Left padding is the default and is required by generate().
        """
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            raise ValueError("input_ids must have shape (B,L), B > 0.")
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or (labels is not None and labels.shape != input_ids.shape):
            raise ValueError("attention_mask and labels must match input_ids.")
        if not ((attention_mask == 0) | (attention_mask == 1)).all() or (attention_mask.sum(-1) == 0).any():
            raise ValueError("attention_mask must be binary with at least one active token per sample.")
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or len(image_grid_thw) == 0:
            raise ValueError("image_grid_thw must have shape (num_images,3), num_images > 0.")
        config = self.qwen_model.config
        image_id = config.image_token_id
        start_id = config.vision_start_token_id
        end_id = config.vision_end_token_id
        video_id = getattr(config, "video_token_id", None)
        if video_id is not None and ((input_ids == video_id) & attention_mask.bool()).any():
            raise ValueError("The APA/DSE connector currently accepts static images only.")
        categories = self._category_lists(disease_queries, len(image_grid_thw))
        merge = config.vision_config.spatial_merge_size
        sizes = []
        for t, h, w in image_grid_thw.tolist():
            if t != 1 or h <= 0 or w <= 0 or int(h) != h or int(w) != w or h % merge or w % merge:
                raise ValueError("Expected t=1 and positive patch-grid dimensions divisible by spatial_merge_size.")
            sizes.append(int(h * w) // merge**2)

        # Validate all processor image spans before running either backbone.
        rows, spans, image_index = [], [], 0
        for row, mask in zip(input_ids, attention_mask):
            active = row[mask.bool()]
            row_spans = []
            cursor = 0
            while cursor < len(active):
                if active[cursor].item() != image_id:
                    cursor += 1
                    continue
                if image_index >= len(sizes) or cursor == 0 or active[cursor - 1].item() != start_id:
                    raise ValueError("Image token blocks must follow vision_start and align with image_grid_thw.")
                size = sizes[image_index]
                end = cursor + size
                if (end >= len(active) or not (active[cursor:end] == image_id).all()
                        or active[end].item() != end_id):
                    raise ValueError("Image placeholder count does not match its merged spatial grid.")
                row_spans.append((cursor - 1, end + 1, image_index))
                image_index += 1
                cursor = end + 1
            rows.append(active)
            spans.append(row_spans)
        if image_index != len(sizes):
            raise ValueError("Image blocks in input_ids do not match image_grid_thw.")

        queries = self._encode_categories(categories)
        features = self.qwen_model.model.get_image_features(pixel_values, image_grid_thw)
        if isinstance(features, torch.Tensor):
            visual_tokens = features
        else:
            visual_tokens = torch.cat(list(features), dim=0)
        adsa_parameter = next(self.adsa.parameters())
        result = self.adsa(
            visual_tokens.to(adsa_parameter), image_grid_thw,
            spatial_merge_size=merge, disease_embeds=queries,
        )
        embedding = self.qwen_model.get_input_embeddings()
        device = embedding.weight.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id for packing.")

        packed_ids, packed_embeds, packed_labels, repeated_grids = [], [], [], []
        for batch_i, (row, row_spans) in enumerate(zip(rows, spans)):
            row = row.to(device)
            row_labels = None if labels is None else labels[batch_i][attention_mask[batch_i].bool()].to(device)
            ids_parts, embed_parts, label_parts = [], [], []

            def append_text(ids, target=None):
                ids = ids.to(device=device, dtype=torch.long)
                ids_parts.append(ids)
                embed_parts.append(embedding(ids))
                if labels is not None:
                    label_parts.append(torch.full_like(ids, -100) if target is None else target)

            def append_features(values, token_id):
                values = values.to(embedding.weight)
                ids = torch.full((len(values),), token_id, device=device, dtype=torch.long)
                ids_parts.append(ids)
                embed_parts.append(values)
                if labels is not None:
                    label_parts.append(torch.full_like(ids, -100))

            cursor = 0
            for begin, end, i in row_spans:
                append_text(row[cursor:begin], None if row_labels is None else row_labels[cursor:begin])
                # Prototype slots use an ordinary placeholder ID, not image IDs:
                # native M-RoPE must only count the C enhanced image sequences.
                append_features(result.apa_outputs[i].prototype_features, pad_id)
                for name, enhanced in zip(categories[i], result.category_image_embeds[i]):
                    name_ids = self.tokenizer.encode(name + ": ", add_special_tokens=False)
                    append_text(torch.tensor(name_ids, dtype=torch.long, device=device))
                    append_text(row.new_tensor([start_id]))
                    append_features(enhanced, image_id)
                    append_text(row.new_tensor([end_id]))
                    repeated_grids.append(image_grid_thw[i])
                cursor = end
            append_text(row[cursor:], None if row_labels is None else row_labels[cursor:])
            packed_ids.append(torch.cat(ids_parts))
            packed_embeds.append(torch.cat(embed_parts))
            if labels is not None:
                packed_labels.append(torch.cat(label_parts))

        length = max(len(ids) for ids in packed_ids)
        max_positions = getattr(getattr(config, "text_config", config), "max_position_embeddings", None)
        if max_positions is not None and length > max_positions:
            raise ValueError(f"Expanded input length {length} exceeds the model context limit {max_positions}.")
        ids_batch, embed_batch, mask_batch, label_batch = [], [], [], []
        for i, (ids, embeds) in enumerate(zip(packed_ids, packed_embeds)):
            pad = length - len(ids)
            left, right = (pad, 0) if padding_side == "left" else (0, pad)
            ids_batch.append(F.pad(ids, (left, right), value=pad_id))
            embed_batch.append(F.pad(embeds, (0, 0, left, right)))
            mask_batch.append(F.pad(torch.ones_like(ids), (left, right)))
            if labels is not None:
                label_batch.append(F.pad(packed_labels[i], (left, right), value=-100))
        ids_batch = torch.stack(ids_batch)
        embed_batch = torch.stack(embed_batch)
        mask_batch = torch.stack(mask_batch)
        grids = torch.stack(repeated_grids).to(device)
        position_ids, _ = self.qwen_model.model.get_rope_index(
            ids_batch, image_grid_thw=grids, attention_mask=mask_batch,
        )
        return QwenADSAInputs(
            input_ids=ids_batch, inputs_embeds=embed_batch, attention_mask=mask_batch,
            position_ids=position_ids, image_grid_thw=grids,
            labels=torch.stack(label_batch) if labels is not None else None,
            adsa_output=result, candidate_categories=categories,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        disease_queries: Optional[Union[str, List[str], List[List[str]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate continuation IDs using native Qwen cache and multimodal positions.

        A scoped embedding hook substitutes prepared features during prefill
        (or at the prefix on each pass with use_cache=False). Keeping native
        input_ids visible lets Qwen compute and update its own M-RoPE/cache;
        static prefill position_ids must not be reused for generated tokens.
        The hook is always removed. As with native Qwen's rope_deltas, use one
        model instance sequentially, not for concurrent generation requests.
        """
        if generation_kwargs.get("return_dict_in_generate", False):
            raise ValueError("generate() returns continuation IDs; return_dict_in_generate must be False.")
        # Override a model generation_config that might otherwise return a
        # structured object instead of the continuation tensor promised here.
        generation_kwargs["return_dict_in_generate"] = False
        if any(key in generation_kwargs for key in (
            "inputs_embeds", "position_ids", "labels", "pixel_values_videos", "video_grid_thw",
        )):
            raise ValueError("Pass original image/prompt inputs; the connector constructs decoder embeddings/positions.")
        if self.training:
            raise ValueError("Call connector.eval() before generation.")
        packed = self.prepare_inputs(
            input_ids, pixel_values, image_grid_thw, disease_queries,
            attention_mask=attention_mask, padding_side="left",
        )
        prefix_len = packed.input_ids.shape[1]
        injected = False

        def inject_features(module, args, output):
            nonlocal injected
            ids = args[0]
            if ids.ndim != 2 or ids.shape[1] < prefix_len:
                return output
            repeats, remainder = divmod(ids.shape[0], packed.input_ids.shape[0])
            if remainder or repeats == 0:
                raise ValueError("Unexpected generation batch expansion.")
            expected = packed.input_ids.repeat_interleave(repeats, dim=0)
            if not torch.equal(ids[:, :prefix_len], expected):
                raise ValueError("Generation prefix changed before visual feature injection.")
            injected = True
            features = packed.inputs_embeds.repeat_interleave(repeats, dim=0).to(output)
            return torch.cat((features, output[:, prefix_len:]), dim=1)

        handle = self.qwen_model.get_input_embeddings().register_forward_hook(inject_features)
        try:
            sequences = self.qwen_model.generate(
                input_ids=packed.input_ids,
                attention_mask=packed.attention_mask,
                image_grid_thw=packed.image_grid_thw,
                **generation_kwargs,
            )
        finally:
            handle.remove()
        if not injected:
            raise RuntimeError("The native Qwen generation path did not consume the prepared features.")
        return sequences[:, prefix_len:]


def apa_alignment_loss(
    saliency: List[torch.Tensor],
    grids: List[Tuple[int, int]],
    boxes_xyxy: List[Optional[torch.Tensor]],
    category_indices: Optional[List[Optional[torch.Tensor]]] = None,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    """Instance-averaged BCE of disease-conditioned responses against bbox masks.

    saliency[i]: (H_i,W_i) or (C_i,H_i,W_i), raw APA spatial responses.
    boxes_xyxy[i]: (N_i,4) normalized xyxy, a single (4,) box, or None.
    category_indices[i]: (N_i,) integer indices into image i's disease queries;
        required for images with C_i > 1 and usable boxes. These are query-row
        indices, not arbitrary dataset class IDs. Repeated indices are allowed.

    Every instance gets its own binary mask and BCE, as in L_APA; boxes of the
    same disease are NOT unioned. Small boxes with no token center inside them
    still contribute their all-zero target. Empty annotations give a connected
    zero when saliency is available. Dice is an opt-in legacy extension; the
    default is the manuscript's BCE alone.
    """
    if not (len(saliency) == len(grids) == len(boxes_xyxy)):
        raise ValueError("saliency, grids and boxes_xyxy must have equal lengths.")
    if category_indices is not None and len(category_indices) != len(saliency):
        raise ValueError("category_indices must have one entry per image.")
    losses = []
    for i, (response, (h, w), boxes) in enumerate(zip(saliency, grids, boxes_xyxy)):
        maps = response.unsqueeze(0) if response.ndim == 2 else response
        if maps.ndim != 3 or maps.shape[0] == 0 or maps.shape[1:] != (h, w):
            raise ValueError("Saliency shape must match (H,W) or (C,H,W) and the supplied grid.")
        if boxes is None:
            continue
        boxes = torch.as_tensor(boxes, device=response.device, dtype=torch.float32)
        if boxes.shape == (4,):
            boxes = boxes.unsqueeze(0)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Boxes must have shape (N,4) or (4,).")
        if boxes.shape[0] == 0:
            continue
        if (not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any()
                or (boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError("Boxes must be finite normalized xyxy coordinates with x1<=x2, y1<=y2.")
        indices = None if category_indices is None else category_indices[i]
        if indices is None:
            if maps.shape[0] != 1:
                raise ValueError("Multi-disease bbox supervision requires category_indices.")
            indices = torch.zeros(len(boxes), device=response.device, dtype=torch.long)
        else:
            indices = torch.as_tensor(indices, device=response.device)
            if indices.shape != (len(boxes),) or indices.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
            ):
                raise ValueError("category_indices must contain one integer query index per box.")
            indices = indices.long()
            if (indices < 0).any() or (indices >= maps.shape[0]).any():
                raise ValueError("Category index is outside the image's disease-query range.")
        coords = _grid_coordinates(h, w, response.device)
        targets = (
            (coords[None, :, :] >= boxes[:, None, :2])
            & (coords[None, :, :] <= boxes[:, None, 2:])
        ).all(dim=-1).float()
        predictions = maps[indices].flatten(1).float().clamp(1e-6, 1 - 1e-6)
        instance_losses = bce_weight * F.binary_cross_entropy(predictions, targets, reduction="none").mean(dim=-1)
        if dice_weight:
            intersection = (predictions * targets).sum(dim=-1)
            dice = 1 - (2 * intersection + 1) / (predictions.sum(dim=-1) + targets.sum(dim=-1) + 1)
            instance_losses = instance_losses + dice_weight * dice
        losses.append(instance_losses)
    if losses:
        return torch.cat(losses).mean()
    if saliency:
        return sum(response.float().sum() * 0 for response in saliency)
    return torch.tensor(0.0)


def adsa_alignment_loss(
    saliency: List[torch.Tensor],
    grids: List[Tuple[int, int]],
    boxes_xyxy: List[Optional[torch.Tensor]],
    bce_weight: float = 1.0,
    dice_weight: float = 0.0,
    *,
    category_indices: Optional[List[Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    """Compatibility entry point; the default now follows the APA BCE objective."""
    return apa_alignment_loss(
        saliency, grids, boxes_xyxy, category_indices,
        bce_weight=bce_weight, dice_weight=dice_weight,
    )
