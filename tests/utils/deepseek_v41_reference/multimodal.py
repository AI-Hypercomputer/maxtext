# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal CPU reference adapter, fp64 upcast context, and mini-twin builders.

Adapts the vendored DeepSeek-V4.1-Flash vision tower (`vision.py`), image
processor (`image_processor.py`), and text reference (`model.py`, `engram.py`)
into a differentiable CPU oracle without modifying the vendored files. All
deviations R1-R8 are documented in `provenance.json`.
"""

from collections.abc import Iterator
import contextlib
import dataclasses
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image
from PIL import ImageOps
import torch
import torch.nn.functional as F
from torch import nn

from tests.utils.parity import layer_parity

from . import engram as ref_engram
from . import image_processor
from . import model as ref_model
from . import vision

MINI_SEQ_LEN = 40
MINI_BATCH_SIZE = 2
MINI_COMPRESS_RATIOS = (0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1)
MINI_KV_SOURCE_LAYERS = (2, 4, 6, 8)
MINI_INDEX_SOURCE_LAYERS = (2, 4, 6, 8, 10, 12)
MINI_CANDIDATE_SOURCE_LAYER = 8
MINI_ENGRAM_LAYER_IDS = (1, 6)
MINI_GRID_PADDED = (7, 5)
MINI_GRID_EXACT = (6, 9)
MINI_DEFAULT_SPAN_STARTS = ((0, 16), (4, 22))


def compute_engram_num_embeddings(
    layer_ids: tuple[int, ...],
    max_ngram_size: int,
    n_heads: int,
    vocab_size: int,
) -> tuple[int, ...]:
  """Computes total unpadded embedding rows per engram layer from prime bucket sizes."""
  seen: set[int] = set()
  totals: list[int] = []
  for _ in layer_ids:
    layer_total = 0
    for _ in range(max_ngram_size - 1):
      current = vocab_size - 1
      for _ in range(n_heads):
        current = ref_engram.find_next_prime(current, seen)
        seen.add(current)
        layer_total += current
    totals.append(layer_total)
  return tuple(totals)


def mini_model_args(**overrides: Any) -> ref_model.ModelArgs:
  """Returns the 14-layer text + 2-layer vision mini-twin ModelArgs."""
  kwargs: dict[str, Any] = {
      "dtype": "fp32",
      "max_batch_size": MINI_BATCH_SIZE,
      "max_seq_len": 64,
      "vocab_size": 129280,
      "dim": 64,
      "moe_inter_dim": 32,
      "n_layers": len(MINI_COMPRESS_RATIOS),
      "n_mtp_layers": 0,
      "n_heads": 4,
      "n_routed_experts": 4,
      "n_shared_experts": 1,
      "n_activated_experts": 2,
      "score_func": "sqrtsoftplus",
      "route_scale": 1.5,
      "swiglu_limit": 10.0,
      "q_lora_rank": 16,
      "head_dim": 32,
      "rope_head_dim": 8,
      "norm_eps": 1e-20,
      "o_groups": 2,
      "o_lora_rank": 8,
      "window_size": 8,
      "compress_ratios": MINI_COMPRESS_RATIOS,
      "kv_source_layers": MINI_KV_SOURCE_LAYERS,
      "index_source_layers": MINI_INDEX_SOURCE_LAYERS,
      "compress_rope_theta": 160000.0,
      "original_seq_len": 65536,
      "rope_theta": 10000.0,
      "rope_factor": 16,
      "beta_fast": 32,
      "beta_slow": 1,
      "index_n_heads": 16,
      "index_head_dim": 16,
      "index_topk": 3,
      "candidate_source_layer": MINI_CANDIDATE_SOURCE_LAYER,
      "candidate_topk_blocks": 2,
      "candidate_block_size": 4,
      "hc_mult": 4,
      "hc_sinkhorn_iters": 20,
      "hc_eps": 1e-6,
      "engram_layer_ids": MINI_ENGRAM_LAYER_IDS,
      "engram_max_ngram_size": 3,
      "engram_vocab_size": 50,
      "engram_n_heads": 2,
      "engram_head_dim": 8,
      "engram_pad_id": 2,
      "engram_compressed_vocab_size": 99092,
      "dspark_block_size": 0,
      "vision_n_layers": 2,
      "vision_dim": 128,
      "vision_n_heads": 2,
      "vision_inter_dim": 352,
      "vision_patch_size": 14,
      "vision_rope_theta": 10000.0,
      "vision_downsample_ratio": 3,
      "vision_max_n_token": 1024,
      "vision_min_pixels": 0,
      "vision_max_wh_ratio": None,
      "image_token_id": 129264,
  }
  kwargs.update(overrides)
  if "engram_num_embeddings" not in overrides:
    kwargs["engram_num_embeddings"] = compute_engram_num_embeddings(
        tuple(kwargs["engram_layer_ids"]),
        int(kwargs["engram_max_ngram_size"]),
        int(kwargs["engram_n_heads"]),
        int(kwargs["engram_vocab_size"]),
    )
  return ref_model.ModelArgs(**kwargs)


class MultimodalTransformer(ref_model.Transformer):
  """DeepSeek-V4.1 reference Transformer with vision tower, aligner, and bias_vl.

  Implements deviations R1-R5 over the text-only `ref_model.Transformer` and
  release `inference/model.py`:
    R1: Relative package imports.
    R2: No `@torch.inference_mode()` on `encode_image` or `forward`.
    R3: `full_logits=True` returns `[B, S, vocab]` logits directly.
    R4: Base built with `vision_n_layers=0`, then per-layer `gate.bias_vl` and
        `vision`, `aligner`, `image_start`, `image_end`, `image_newline` attached.
    R5: `engram_hashes` `.clone()`d outside inference mode for autograd.
  """

  def __init__(self, args: ref_model.ModelArgs, tokenizer: Any = None) -> None:
    text_args = dataclasses.replace(args, vision_n_layers=0)
    super().__init__(text_args, tokenizer=tokenizer)
    self.vision_enabled = args.vision_enabled
    self.vision: vision.ViT | None = None
    self.aligner: vision.Aligner | None = None
    if args.vision_enabled:
      # Deviation R4: attach per-layer float32 routing bias_vl onto each Gate.
      for layer_id, layer in enumerate(self.layers):
        n_routed_experts, _ = args.get_moe_config(layer_id)
        layer.ffn.gate.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
      self.vision = vision.ViT(args)
      self.aligner = vision.Aligner(args)
      self.image_start = nn.Parameter(torch.empty(args.dim))
      self.image_end = nn.Parameter(torch.empty(args.dim))
      self.image_newline = nn.Parameter(torch.empty(args.dim))

  def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
    """Encodes one image's ViT patches through ViT and 3x3 Aligner (R2: differentiable)."""
    if self.vision is None or self.aligner is None:
      raise RuntimeError("encode_image called on a model with vision_n_layers=0")
    return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

  def merge_image_embeddings(
      self,
      images: list[list[image_processor.ImageInput] | None],
      h: torch.Tensor,
  ) -> torch.Tensor:
    """Overwrites each image's token span in `h` with its ViT/aligner features and delimiters."""
    for i, sample in enumerate(images):
      if sample is None:
        continue
      for img in sample:
        types = img.types.to(h.device)
        span = h[i, img.start : img.start + types.numel()]
        span[types == image_processor.IMAGE_START] = self.image_start.to(h.dtype)
        span[types == image_processor.IMAGE_END] = self.image_end.to(h.dtype)
        span[types == image_processor.IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
        embeds = self.encode_image(img.patches.to(h.device), img.n_vit_h, img.n_vit_w)
        span[types == image_processor.IMAGE] = embeds.to(h.dtype)
    return h

  def forward(  # pylint: disable=arguments-renamed
      self,
      input_ids: torch.Tensor,
      start_pos: int = 0,
      images: list[list[image_processor.ImageInput] | None] | None = None,
      token_types: torch.Tensor | None = None,
      full_logits: bool = True,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Differentiable multimodal forward pass (deviations R2, R3, R5)."""
    image_mask = None if token_types is None else token_types >= 0
    engram_mask = None if image_mask is None else ~image_mask
    engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask).clone() if self.engram_hash is not None else None
    h = self.embed(input_ids)
    if images is not None:
      assert start_pos == 0, "image spans must be prefilled in a single chunk"
      h = h.clone()
      self.merge_image_embeddings(images, h)
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    main_hiddens = []
    pre_mix = ref_model.make_identity_pre_mix(h, self.hc_mult)
    for i, layer in enumerate(self.layers):
      if layer.engram is not None and engram_hashes is not None:
        h = layer.engram(
            h,
            engram_hashes[:, :, layer.engram.layer_hash_index, :],
            engram_mask,
        )
      if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))
      h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
    h = self.layers[-1].hc_pre(h, pre_mix)
    logits = self.head(self.norm(h), full_logits=full_logits)
    if full_logits:
      return logits
    output_ids = ref_model.sample(logits, self.temperature)
    main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
    return output_ids, logits, main_hidden

  @torch.inference_mode()
  def forward_inference_mode(
      self,
      input_ids: torch.Tensor,
      start_pos: int = 0,
      images: list[list[image_processor.ImageInput] | None] | None = None,
      token_types: torch.Tensor | None = None,
      full_logits: bool = True,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Release inference-mode forward pass without autograd clones (for R2 bitwise verification)."""
    image_mask = None if token_types is None else token_types >= 0
    engram_mask = None if image_mask is None else ~image_mask
    engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask) if self.engram_hash is not None else None
    h = self.embed(input_ids)
    if images is not None:
      assert start_pos == 0, "image spans must be prefilled in a single chunk"
      self.merge_image_embeddings(images, h)
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
    main_hiddens = []
    pre_mix = ref_model.make_identity_pre_mix(h, self.hc_mult)
    for i, layer in enumerate(self.layers):
      if layer.engram is not None and engram_hashes is not None:
        h = layer.engram(
            h,
            engram_hashes[:, :, layer.engram.layer_hash_index, :],
            engram_mask,
        )
      if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))
      h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
    h = self.layers[-1].hc_pre(h, pre_mix)
    logits = self.head(self.norm(h), full_logits=full_logits)
    if full_logits:
      return logits
    output_ids = ref_model.sample(logits, self.temperature)
    main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
    return output_ids, logits, main_hidden


def build_multimodal_model(args: ref_model.ModelArgs, tokenizer: Any = None) -> MultimodalTransformer:
  """Builds `MultimodalTransformer` with torch's default dtype matching `args.dtype`."""
  prev = torch.get_default_dtype()
  torch.set_default_dtype(torch.float32 if args.dtype == "fp32" or args.fp32_compute else torch.bfloat16)
  try:
    model = MultimodalTransformer(args, tokenizer=tokenizer)
  finally:
    torch.set_default_dtype(prev)
  return model.eval()


def init_mini_reference(model: nn.Module, *, seed: int = 1) -> None:
  """Initializes a mini-twin reference model with deterministic non-degenerate weights.

  Matches `deepseek_v41_vs_reference_test._init_reference` (`1 / sqrt(fan_in)`
  weights, `1 + N(0, 0.1)` norms, `N(0, 0.1)` routing biases) extended to the
  multimodal parameters (`N(0, 0.02)` vision/aligner biases, `N(0, 1)` image
  delimiters, independent `gate.bias` and `gate.bias_vl` draws) so every routed
  expert receives tokens (`grad != None` on all compared params) and `bias_vl`
  changes image-token top-k routing.
  """
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      r = torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype)
      if name.endswith("norm.weight") or name.endswith(("norm1.weight", "norm2.weight", "q_weight", "k_weight")):
        p.copy_(1.0 + 0.1 * r)
      elif name in ("image_start", "image_end", "image_newline"):
        p.copy_(r)
      elif name.endswith(("gate.bias", "gate.bias_vl")):
        p.copy_(0.1 * r)
      elif name.endswith("attn_sink"):
        p.copy_(0.5 * r)
      elif name.endswith("_base"):
        p.copy_(0.1 * r)
      elif name.endswith("_scale") and "hc_" in name:
        p.copy_(0.5 + 0.1 * r)
      elif "embed" in name and not name.startswith("vision."):
        p.copy_(r)
      elif name.endswith(".bias"):
        p.copy_(0.02 * r)
      else:
        p.copy_(r / (p.shape[-1] ** 0.5))


def make_synthetic_image_patches(
    args: ref_model.ModelArgs,
    n_vit_h: int,
    n_vit_w: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int, int]:
  """Builds ViT patches from a seeded synthetic RGB PIL image via `image_processor` logic.

  Deviation R6: skips the final `.to(torch.bfloat16)` cast of
  `image_processor.load_image` line 127 when `dtype` is float32 or float64.
  """
  p = args.vision_patch_size
  raw_pixels = np.random.default_rng(seed).integers(0, 256, size=(n_vit_h * p, n_vit_w * p, 3), dtype=np.uint8)
  image = Image.fromarray(raw_pixels, mode="RGB")
  n_llm_h, n_llm_w, best_height, best_width = image_processor.plan_image_grid(image.width, image.height, args)
  planned_vit_h, planned_vit_w = best_height // p, best_width // p
  if (planned_vit_h, planned_vit_w) != (n_vit_h, n_vit_w):
    raise ValueError(f"Planned ViT grid {(planned_vit_h, planned_vit_w)} != requested {(n_vit_h, n_vit_w)}")
  if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
    image = image.resize((best_width, best_height))
  else:
    image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
  work_dtype = torch.promote_types(dtype, torch.float32)
  x = torch.from_numpy(np.asarray(image, dtype=np.float32)).to(work_dtype).permute(2, 0, 1) / 255.0
  x = ((x - 0.5) / 0.5).to(dtype)
  patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
  return patches, n_llm_h, n_llm_w


@dataclasses.dataclass(frozen=True)
class MiniVLBatch:
  """Synthetic multimodal batch for the 14-layer mini-twin."""

  input_ids: torch.Tensor
  token_types: torch.Tensor
  image_mask: torch.Tensor
  images: list[list[image_processor.ImageInput]]
  patches_bnpd: torch.Tensor
  n_vit_h: int
  n_vit_w: int
  n_llm_h: int
  n_llm_w: int


def build_mini_vl_batch(
    args: ref_model.ModelArgs,
    *,
    grid: tuple[int, int] = MINI_GRID_PADDED,
    batch_size: int = MINI_BATCH_SIZE,
    seq_len: int = MINI_SEQ_LEN,
    span_starts: tuple[tuple[int, ...], ...] = MINI_DEFAULT_SPAN_STARTS,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> MiniVLBatch:
  """Builds a seeded image+text batch with two image spans per sequence."""
  if len(span_starts) != batch_size:
    raise ValueError(f"span_starts length {len(span_starts)} must equal batch_size {batch_size}")
  n_vit_h, n_vit_w = grid
  rng = np.random.default_rng(seed)
  # Draw text tokens in [3, vocab_size) avoiding pad_id=2 and image_token_id.
  raw_ids = rng.integers(3, args.vocab_size - 1, size=(batch_size, seq_len))
  raw_ids = np.where(raw_ids >= args.image_token_id, raw_ids + 1, raw_ids)
  input_ids = torch.as_tensor(raw_ids, dtype=torch.int64)
  token_types = torch.full((batch_size, seq_len), image_processor.TEXT, dtype=torch.int64)

  images: list[list[image_processor.ImageInput]] = []
  patches_per_sample: list[torch.Tensor] = []
  n_llm_h_out = 0
  n_llm_w_out = 0
  for b in range(batch_size):
    sample_images: list[image_processor.ImageInput] = []
    sample_patches: list[torch.Tensor] = []
    prev_end = 0
    for img_idx, start in enumerate(span_starts[b]):
      patches, n_llm_h, n_llm_w = make_synthetic_image_patches(
          args,
          n_vit_h,
          n_vit_w,
          seed=seed * 100 + b * 10 + img_idx + 1,
          dtype=dtype,
      )
      n_llm_h_out, n_llm_w_out = n_llm_h, n_llm_w
      types = image_processor.image_token_types(n_llm_h, n_llm_w)
      span_len = int(types.numel())
      if start < prev_end or start + span_len > seq_len:
        raise ValueError(f"Invalid image span [{start}, {start + span_len}) for seq_len={seq_len} (prev_end={prev_end})")
      prev_end = start + span_len
      input_ids[b, start:prev_end] = args.image_token_id
      token_types[b, start:prev_end] = types
      sample_images.append(
          image_processor.ImageInput(
              start=start,
              patches=patches,
              n_vit_h=n_vit_h,
              n_vit_w=n_vit_w,
              types=types,
          )
      )
      sample_patches.append(patches.flatten(1))
    images.append(sample_images)
    patches_per_sample.append(torch.stack(sample_patches, dim=0))

  patches_bnpd = torch.stack(patches_per_sample, dim=0)
  image_mask = token_types >= 0
  return MiniVLBatch(
      input_ids=input_ids,
      token_types=token_types,
      image_mask=image_mask,
      images=images,
      patches_bnpd=patches_bnpd,
      n_vit_h=n_vit_h,
      n_vit_w=n_vit_w,
      n_llm_h=n_llm_h_out,
      n_llm_w=n_llm_w_out,
  )


def reference_gradient_classes(model: nn.Module, *, freeze_vision: bool = False) -> layer_parity.GradientClasses:
  """Returns the explicit five-class parameter partition for `model`."""
  compared: set[str] = set()
  non_gradient: set[str] = set()
  discrete_only: set[str] = set()
  frozen: set[str] = set()
  padded: set[str] = set()
  for name, _ in model.named_parameters():
    if name.endswith((".ffn.gate.bias", ".ffn.gate.bias_vl")):
      non_gradient.add(name)
    elif ".attn.indexer." in name:
      discrete_only.add(name)
    elif freeze_vision and name.startswith("vision."):
      frozen.add(name)
    elif name.endswith(".engram.embed.weight"):
      padded.add(name)
    else:
      compared.add(name)
  return layer_parity.GradientClasses(
      compared=frozenset(compared),
      non_gradient=frozenset(non_gradient),
      discrete_only=frozenset(discrete_only),
      frozen=frozenset(frozen),
      padded=frozenset(padded),
  )


def count_bias_vl_routing_changes(model: MultimodalTransformer, batch: MiniVLBatch) -> int:
  """Counts `(layer, image_token)` positions whose top-k expert set changes under `bias_vl`.

  Runs the model forward on `batch` and compares each decoder layer's `Gate`
  top-k selection on image-token positions (`image_mask=True`) against the
  selection that same `Gate` makes on the same hidden states when `image_mask`
  is `None` (using `gate.bias`).
  """
  changed_positions = 0
  flat_mask = batch.image_mask.reshape(-1)
  captured_inputs: list[tuple[ref_model.Gate, torch.Tensor]] = []

  def make_hook(gate_mod: ref_model.Gate):
    def hook(_module: nn.Module, args: tuple[Any, ...], _output: Any) -> None:
      captured_inputs.append((gate_mod, args[0].detach().clone()))

    return hook

  handles = [layer.ffn.gate.register_forward_hook(make_hook(layer.ffn.gate)) for layer in model.layers]
  try:
    with torch.no_grad():
      model(
          batch.input_ids,
          images=batch.images,
          token_types=batch.token_types,
          full_logits=True,
      )
  finally:
    for handle in handles:
      handle.remove()

  with torch.no_grad():
    for gate_mod, x_flat in captured_inputs:
      _, idx_vl = gate_mod(x_flat, image_mask=flat_mask)
      _, idx_text = gate_mod(x_flat, image_mask=None)
      diff_rows = (idx_vl[flat_mask] != idx_text[flat_mask]).any(dim=-1)
      changed_positions += int(diff_rows.sum().item())
  return changed_positions


def _make_vision_cos_sin_upcast(default_dtype: torch.dtype):
  """Builds a cache-free `get_vision_cos_sin` using `promote_types(dtype, float32)`."""

  def _get_vision_cos_sin(
      n_h: int,
      n_w: int,
      dim: int,
      theta: float,
      dtype: torch.dtype | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    target = default_dtype if dtype is None else dtype
    work_dtype = torch.promote_types(target, torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=work_dtype) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).to(work_dtype) * inv_freq
    freqs = freqs.flatten(1)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    if work_dtype != torch.float32:
      inv_freq32 = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
      freqs32 = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq32).flatten(1)
      cos._fp32_exact = freqs32.cos().unsqueeze(1)  # pylint: disable=protected-access
      sin._fp32_exact = freqs32.sin().unsqueeze(1)  # pylint: disable=protected-access
    return cos, sin

  return _get_vision_cos_sin


def _vision_apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Upcast-only replacement for `vision.apply_rotary` (deviation R8)."""
  dtype = x.dtype
  work_dtype = torch.promote_types(dtype, torch.float32)
  x1, x2 = x.to(work_dtype).chunk(2, dim=-1)
  if work_dtype == torch.float32 and cos.dtype != torch.float32:
    cos_w = cos._fp32_exact if hasattr(cos, "_fp32_exact") else cos.to(work_dtype)  # pylint: disable=protected-access
    sin_w = sin._fp32_exact if hasattr(sin, "_fp32_exact") else sin.to(work_dtype)  # pylint: disable=protected-access
  else:
    cos_w = cos.to(work_dtype)
    sin_w = sin.to(work_dtype)
  return torch.cat([x1 * cos_w - x2 * sin_w, x2 * cos_w + x1 * sin_w], dim=-1).to(dtype)


def _vision_rmsnorm_forward(self: vision.RMSNorm, x: torch.Tensor) -> torch.Tensor:
  """Upcast-only replacement for `vision.RMSNorm.forward` (deviation R8)."""
  dtype = x.dtype
  work_dtype = torch.promote_types(dtype, torch.float32)
  x_work = x.to(work_dtype)
  x_work = x_work * torch.rsqrt(x_work.square().mean(-1, keepdim=True) + self.eps)
  return (self.weight.to(work_dtype) * x_work).to(dtype)


def _ref_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
  """Upcast-only replacement for `ref_model.linear` admitting float64 (deviation R8)."""
  assert bias is None
  assert weight.dtype in (
      torch.float64,
      torch.float32,
      torch.bfloat16,
      torch.float16,
  ), weight.dtype
  if ref_model.compute_fp32:
    work_dtype = torch.promote_types(x.dtype, torch.float32)
    return F.linear(x.to(work_dtype), weight.to(work_dtype))
  return F.linear(x.to(weight.dtype), weight)


def _ref_engram_forward(
    self: ref_model.Engram,
    x: torch.Tensor,
    hash_ids: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Upcast-only replacement for `ref_model.Engram.forward` (deviation R8)."""
  work_dtype = torch.promote_types(x.dtype, torch.float32)
  kv = self.wkv(self.embed(hash_ids).flatten(-2))
  key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
  key = key.to(work_dtype).unflatten(-1, (self.hc_mult, self.dim))
  weight = self.q_weight.to(work_dtype) * self.k_weight.to(work_dtype)
  h, eps = x.to(work_dtype), self.eps
  rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
  dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
  gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
  if token_mask is not None:
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
  return (h + gate.unsqueeze(-1) * value.to(work_dtype).unsqueeze(-2)).to(x.dtype)


def _ref_gate_forward(
    self: ref_model.Gate,
    x: torch.Tensor,
    image_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Upcast-only replacement for `ref_model.Gate.forward` (deviation R8)."""
  work_dtype = torch.promote_types(x.dtype, torch.float32)
  scores = ref_model.linear(x.to(work_dtype), self.weight.to(work_dtype)) / self.gate_temp
  if self.score_func == "softmax":
    scores = scores.softmax(dim=-1)
  elif self.score_func == "sigmoid":
    scores = scores.sigmoid()
  else:
    scores = F.softplus(scores).sqrt()
  bias = self.bias
  if image_mask is not None and self.bias_vl is not None:
    bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
  indices = (scores + bias).topk(self.topk, dim=-1)[1]
  weights = scores.gather(1, indices)
  if self.norm_topk_prob and self.topk > 1:
    weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
  weights *= self.route_scale
  return weights, indices


@contextlib.contextmanager
def fp64_reference_context(
    dtype: torch.dtype = torch.float64,
) -> Iterator[None]:
  """Test-scoped context manager replacing fp32 downcasts with upcast-only casts (R8).

  Patches `vision.get_vision_cos_sin` (cache-free copy), `vision.apply_rotary`,
  `vision.RMSNorm.forward`, `ref_model.linear`, `ref_model.Engram.forward`, and
  `ref_model.Gate.forward`. Clears `vision.get_vision_cos_sin`'s LRU cache on
  both entry and exit so cached fp32 tables never leak into fp64 runs.
  """
  orig_cos_sin = vision.get_vision_cos_sin
  orig_cos_sin.cache_clear()
  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.object(vision, "get_vision_cos_sin", _make_vision_cos_sin_upcast(dtype)))
    stack.enter_context(mock.patch.object(vision, "apply_rotary", _vision_apply_rotary))
    stack.enter_context(mock.patch.object(vision.RMSNorm, "forward", _vision_rmsnorm_forward))
    stack.enter_context(mock.patch.object(ref_model, "linear", _ref_linear))
    stack.enter_context(mock.patch.object(ref_model.Engram, "forward", _ref_engram_forward))
    stack.enter_context(mock.patch.object(ref_model.Gate, "forward", _ref_gate_forward))
    try:
      yield
    finally:
      orig_cos_sin.cache_clear()
