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

"""vLLM multimodal-registry integration for MaxText models.

This module hosts the four vLLM-side classes required to register a MaxText
multimodal model with vLLM's `MULTIMODAL_REGISTRY`:

  * `MaxTextProcessingInfo`        — declares per-modality limits + token budgets
  * `MaxTextDummyInputsBuilder`    — synthesizes dummy inputs for warmup/profiling
  * `MaxTextMultiModalProcessor`   — routes processor outputs into MultiModalKwargs
  * `MaxTextForConditionalGeneration` — the registered model class

It also exposes `get_dummy_mm_input_info()`, a config-driven helper that returns
the per-image / per-audio shapes, token counts, and placeholder IDs for any
multimodal MaxText model. The helper is the single source of per-model
dispatch; the four classes above are model-agnostic.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import jax
import numpy as np
from PIL import Image
from flax import nnx
from flax import linen as nn
from jax import numpy as jnp

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)

from .adapter import AttentionMetadata, MaxTextForCausalLM


@dataclasses.dataclass(frozen=True)
class DummyMMInputInfo:
  """Per-model multimodal shape, token, and placeholder metadata.

  Built once per engine startup by `get_dummy_mm_input_info()` and reused by
  the ProcessingInfo / DummyInputsBuilder / MultiModalProcessor classes below.
  """

  image_shape: tuple[int, ...]
  image_pil_size: tuple[int, int]  # (width, height) for PIL.Image.new
  max_images: int | None
  tokens_per_image: int
  image_placeholder_id: int
  audio_waveform_samples: int | None
  audio_shape: tuple[int, ...] | None
  tokens_per_audio: int | None
  audio_placeholder_id: int | None
  max_audios: int | None
  audio_sample_rate: int = 16000


def get_dummy_mm_input_info(
    model_name: str,
    mm_overrides: Mapping[str, int | None] | None = None,
) -> DummyMMInputInfo:
  """Resolve per-image / per-audio metadata for a MaxText multimodal model.

  Args:
    model_name: MaxText model identifier (e.g. "gemma3-4b", "llama4-17b-16e").
    mm_overrides: Optional dict to override `max_images` / `max_audios`.

  Returns:
    A `DummyMMInputInfo` populated for the given model family.

  Raises:
    ValueError: If `model_name` does not match any known multimodal family.
  """
  mm_overrides = dict(mm_overrides or {})

  if model_name.startswith("gemma3"):
    info = _gemma3_info()
  elif model_name.startswith("gemma4"):
    info = _gemma4_info()
  elif model_name.startswith("llama4"):
    info = _llama4_info()
  elif model_name.startswith("qwen3-omni"):
    info = _qwen3_omni_info()
  else:
    raise ValueError(
        f"MaxText model_name {model_name!r} is not a known multimodal model."
    )

  return dataclasses.replace(
      info,
      max_images=mm_overrides.get("max_images", info.max_images),
      max_audios=mm_overrides.get("max_audios", info.max_audios),
  )


def _gemma3_info() -> DummyMMInputInfo:
  from maxtext.multimodal.processor_gemma3 import (  # pylint: disable=import-outside-toplevel
      GEMMA_DEFAULT_IMAGE_SIZE,
      GEMMA_NUM_TOKENS_PER_MEDIA,
      GEMMA_TOKEN_PLACEHOLDER,
      get_dummy_image_shape_for_init_gemma3,
  )

  full_shape = get_dummy_image_shape_for_init_gemma3(
      batch_size=1, num_image_per_sequence=1
  )
  # full_shape = (B, N, H, W, C); strip B and N for the per-image shape.
  return DummyMMInputInfo(
      image_shape=full_shape[2:],
      image_pil_size=(GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE),
      max_images=1,
      tokens_per_image=GEMMA_NUM_TOKENS_PER_MEDIA,
      image_placeholder_id=GEMMA_TOKEN_PLACEHOLDER,
      audio_waveform_samples=None,
      audio_shape=None,
      tokens_per_audio=None,
      audio_placeholder_id=None,
      max_audios=None,
  )


def _gemma4_info() -> DummyMMInputInfo:
  from maxtext.multimodal.processor_gemma4 import (  # pylint: disable=import-outside-toplevel
      GEMMA4_IMAGE_HEIGHT,
      GEMMA4_IMAGE_WIDTH,
      GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA,
      GEMMA4_PATCH_SIZE,
      GEMMA4_POOLING_KERNEL,
      GEMMA4_TOKEN_PLACEHOLDER,
      get_dummy_image_shape_for_init_gemma4,
  )

  full_shape = get_dummy_image_shape_for_init_gemma4(
      batch_size=1, num_image_per_sequence=1
  )
  num_patches = (GEMMA4_IMAGE_HEIGHT // GEMMA4_PATCH_SIZE) * (
      GEMMA4_IMAGE_WIDTH // GEMMA4_PATCH_SIZE
  )
  soft_tokens = num_patches // (GEMMA4_POOLING_KERNEL ** 2)
  return DummyMMInputInfo(
      image_shape=full_shape[2:],
      image_pil_size=(GEMMA4_IMAGE_WIDTH, GEMMA4_IMAGE_HEIGHT),
      max_images=1,
      tokens_per_image=soft_tokens + GEMMA4_NUM_EXTRA_TOKENS_PER_MEDIA,
      image_placeholder_id=GEMMA4_TOKEN_PLACEHOLDER,
      audio_waveform_samples=None,
      audio_shape=None,
      tokens_per_audio=None,
      audio_placeholder_id=None,
      max_audios=None,
  )


def _llama4_info() -> DummyMMInputInfo:
  from maxtext.multimodal.processor_llama4 import (  # pylint: disable=import-outside-toplevel
      LLAMA4_PATCH_SIZE,
      LLAMA4_PATCH_TOKEN,
      LLAMA4_PIXEL_SHUFFLE_RATIO,
      LLAMA4_TILE_SIZE,
      LLAMA4_TILES_NUM,
      get_dummy_image_shape_for_init_llama4,
      get_tokens_for_this_image,
  )

  full_shape = get_dummy_image_shape_for_init_llama4(
      batch_size=1, num_image_per_sequence=1
  )
  # full_shape = (B*N, TILES_PAD_TO, C, TILE, TILE); strip the leading batch.
  image_shape = full_shape[1:]

  downsample_ratio = int(round(1.0 / (LLAMA4_PIXEL_SHUFFLE_RATIO ** 2)))
  num_patches_per_chunk = int(
      (LLAMA4_TILE_SIZE // LLAMA4_PATCH_SIZE) ** 2 // downsample_ratio
  )
  # Worst-case aspect ratio that fully utilizes LLAMA4_TILES_NUM=16: a 4x4 grid.
  # Token budget includes BEGIN/END/FAKE markers + tile separators + patch tokens.
  worst_case_tokens = len(get_tokens_for_this_image([4, 4], num_patches_per_chunk))
  return DummyMMInputInfo(
      image_shape=image_shape,
      image_pil_size=(LLAMA4_TILE_SIZE * 4, LLAMA4_TILE_SIZE * 4),
      max_images=1,
      tokens_per_image=worst_case_tokens,
      image_placeholder_id=LLAMA4_PATCH_TOKEN,
      audio_waveform_samples=None,
      audio_shape=None,
      tokens_per_audio=None,
      audio_placeholder_id=None,
      max_audios=None,
  )


def _qwen3_omni_info() -> DummyMMInputInfo:
  from maxtext.multimodal.processor_qwen3_omni import (  # pylint: disable=import-outside-toplevel
      QWEN3_OMNI_AUDIO_TOKEN,
      QWEN3_OMNI_IMAGE_SIZE,
      QWEN3_OMNI_IMAGE_TOKEN,
      SAMPLE_RATE,
      get_dummy_image_shape_for_init_qwen3_omni,
  )

  full_shape = get_dummy_image_shape_for_init_qwen3_omni(batch_size=1)
  # full_shape = (B, C, T, H, W); strip the leading batch.
  image_shape = full_shape[1:]

  # 768/16 = 48 patches per side → 2304 patches per frame; 2x2 spatial merge → 576 tokens.
  spatial_merge = 2
  vit_patch = 16
  tokens_per_image = (QWEN3_OMNI_IMAGE_SIZE // vit_patch // spatial_merge) ** 2

  # Conservative dummy: 30 seconds of audio at 16 kHz.
  dummy_seconds = 30
  audio_waveform_samples = SAMPLE_RATE * dummy_seconds
  # Audio encoder produces ~13 tokens per 100-sample chunk
  # (see _get_feat_extract_output_lengths in processor_qwen3_omni).
  tokens_per_audio = (audio_waveform_samples // 100) * 13

  return DummyMMInputInfo(
      image_shape=image_shape,
      image_pil_size=(QWEN3_OMNI_IMAGE_SIZE, QWEN3_OMNI_IMAGE_SIZE),
      max_images=4,
      tokens_per_image=tokens_per_image,
      image_placeholder_id=QWEN3_OMNI_IMAGE_TOKEN,
      audio_waveform_samples=audio_waveform_samples,
      audio_shape=(audio_waveform_samples,),
      tokens_per_audio=tokens_per_audio,
      audio_placeholder_id=QWEN3_OMNI_AUDIO_TOKEN,
      max_audios=2,
      audio_sample_rate=SAMPLE_RATE,
  )


# Hidden-size discriminators for Gemma3 model families. Verified against
# `src/maxtext/configs/models/gemma3-*.yml` text-config defaults.
_GEMMA3_BY_HIDDEN_SIZE = {
    2560: "gemma3-4b",
    3840: "gemma3-12b",
    5376: "gemma3-27b",
}

_GEMMA4_BY_HIDDEN_SIZE = {
    5376: "gemma4-26b",
    5760: "gemma4-31b",
}

_LLAMA4_BY_NUM_EXPERTS = {
    16: "llama4-17b-16e",
    128: "llama4-17b-128e",
}


def _hf_to_maxtext_model_name(model_config) -> str:
  """Resolve a MaxText model_name string from a vLLM `ModelConfig`.

  Strategy: prefer an explicit `additional_config["maxtext_config"]["model_name"]`
  override; otherwise dispatch on `hf_config.architectures[0]` plus a model-family
  discriminator (hidden_size / num_local_experts).
  """
  hf_config = model_config.hf_config
  text_cfg = getattr(hf_config, "text_config", hf_config)
  model_type = getattr(hf_config, "model_type", None).lower()
  hidden_size = getattr(text_cfg, "hidden_size", None)
  num_experts = getattr(text_cfg, "num_local_experts", None)

  if model_type.startswith("gemma3"):
    return _GEMMA3_BY_HIDDEN_SIZE.get(hidden_size, "gemma3-4b")
  if model_type.startswith("gemma4"):
    return _GEMMA4_BY_HIDDEN_SIZE.get(hidden_size, "gemma4-26b")
  if model_type.startswith("llama4"):
    return _LLAMA4_BY_NUM_EXPERTS.get(num_experts, "llama4-17b-16e")
  if model_type.startswith("qwen3_omni"):
    return "qwen3-omni-30b-a3b"

  raise ValueError(
      f"Cannot infer MaxText model_name from HF model_type {model_type!r}. "
      "Pass it explicitly via additional_config['maxtext_config']['model_name']."
  )


class _MaxTextStubProcessor:
  """Minimal HF-compatible processor that vLLM can call into.

  Delegates per-model image preprocessing to MaxText's existing preprocessors
  (`processor_gemma3.preprocess_mm_data_gemma3`, etc.) so the resulting
  `pixel_values` shape matches what the MaxText vision encoder expects. This
  avoids pulling in `transformers`' image processors (and torch) for the JAX
  worker.
  """

  def __init__(self, model_name: str, mm_info: DummyMMInputInfo, tokenizer):
    self.model_name = model_name
    self.mm_info = mm_info
    self.tokenizer = tokenizer

  def __call__(self, text=None, images=None, audios=None, return_tensors=None, **_):
    from maxtext.utils import max_logging
    max_logging.log(f"_MaxTextStubProcessor called with text={text is not None}, images={type(images)} len={len(images) if images else 0}, audios={type(audios)} len={len(audios) if audios else 0}")
    out: dict = {}
    if text is not None:
      tokenized = self.tokenizer(text, return_tensors=return_tensors)
      out["input_ids"] = tokenized["input_ids"]
      if "attention_mask" in tokenized:
        out["attention_mask"] = tokenized["attention_mask"]
    if images:
      try:
        out["pixel_values"] = self._preprocess_images(images)
        max_logging.log(f"_MaxTextStubProcessor returned pixel_values shape={out['pixel_values'].shape}")
      except Exception as e:
        max_logging.log(f"_MaxTextStubProcessor _preprocess_images failed: {e}")
        raise
    if audios:
      out["input_audio_features"] = self._stack_audios(audios)
    try:
      from transformers.feature_extraction_utils import BatchFeature  # pylint: disable=import-outside-toplevel
      res = BatchFeature(out)
    except ImportError:
      res = out
    max_logging.log(f"_MaxTextStubProcessor returning keys={list(res.keys())}")
    return res

  def _preprocess_images(self, images):
    """Convert PIL images → MaxText per-model `pixel_values` numpy array."""
    np_images = [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in images]
    if self.model_name.startswith("gemma3"):
      from maxtext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

      return preprocess_mm_data_gemma3(np_images).pixel_values
    if self.model_name.startswith("gemma4"):
      from maxtext.multimodal.processor_gemma4 import preprocess_mm_data_gemma4  # pylint: disable=import-outside-toplevel

      return preprocess_mm_data_gemma4(np_images).pixel_values
    if self.model_name.startswith("llama4"):
      from maxtext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

      return preprocess_mm_data_llama4(np_images).pixel_values
    if self.model_name.startswith("qwen3-omni"):
      # Qwen3-Omni's preprocessor takes a config object rather than raw images.
      # Resize-and-stack to the expected per-image shape so the vision encoder
      # receives a usable batch during profiling.
      target_w, target_h = self.mm_info.image_pil_size
      arrays = []
      for img in images:
        pil = img.convert("RGB").resize((target_w, target_h))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arrays.append(arr)
      return np.stack(arrays, axis=0)
    raise ValueError(f"Unsupported model_name {self.model_name!r} in stub processor.")

  def _stack_audios(self, audios):
    waveforms = [item[0] if isinstance(item, tuple) else item for item in audios]
    if self.model_name.startswith("qwen3-omni"):
      from maxtext.multimodal.processor_qwen3_omni import _np_extract_fbank_features
      max_len = max(len(w) for w in waveforms)
      padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
      for i, w in enumerate(waveforms):
        padded[i, : len(w)] = w
      return _np_extract_fbank_features(padded)
    max_len = max(len(w) for w in waveforms)
    out = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
      out[i, : len(w)] = w
    return out


class MaxTextProcessingInfo(BaseProcessingInfo):
  """Declares per-modality limits and token budgets for MaxText multimodal models."""

  def __init__(self, ctx):
    super().__init__(ctx)
    model_name = _hf_to_maxtext_model_name(self.ctx.model_config)
    additional = getattr(self.ctx.model_config, "additional_config", None) or {}
    overrides = (
        additional.get("maxtext_mm_limits", {})
        if isinstance(additional, dict)
        else {}
    )
    self._mm_info = get_dummy_mm_input_info(model_name, overrides)
    self._model_name = model_name

  @property
  def mm_info(self) -> DummyMMInputInfo:
    return self._mm_info

  def get_supported_mm_limits(self) -> Mapping[str, int | None]:
    limits: dict[str, int | None] = {"image": self._mm_info.max_images}
    if self._mm_info.audio_placeholder_id is not None:
      limits["audio"] = self._mm_info.max_audios
    return limits

  def get_mm_max_tokens_per_item(
      self, seq_len: int, mm_counts: Mapping[str, int]
  ) -> Mapping[str, int]:
    del seq_len, mm_counts
    out: dict[str, int] = {"image": self._mm_info.tokens_per_image}
    if self._mm_info.tokens_per_audio is not None:
      out["audio"] = self._mm_info.tokens_per_audio
    return out

  def get_hf_processor(self, **kwargs):
    del kwargs
    return _MaxTextStubProcessor(self._model_name, self._mm_info, self.get_tokenizer())


class MaxTextDummyInputsBuilder(BaseDummyInputsBuilder[MaxTextProcessingInfo]):
  """Builds dummy multimodal inputs for vLLM warmup/profiling."""

  def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
    info = self.info.mm_info
    tokenizer = self.info.get_tokenizer()
    img_tok = tokenizer.decode([info.image_placeholder_id])
    text = img_tok * mm_counts.get("image", 0)
    if info.audio_placeholder_id is not None:
      aud_tok = tokenizer.decode([info.audio_placeholder_id])
      text += aud_tok * mm_counts.get("audio", 0)
    return text + " describe the input."

  def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
    del seq_len, mm_options
    info = self.info.mm_info
    out: dict = {}
    n_img = mm_counts.get("image", 0)
    if n_img:
      w, h = info.image_pil_size
      out["image"] = [Image.new("RGB", (w, h), color=128) for _ in range(n_img)]
    n_aud = mm_counts.get("audio", 0)
    if n_aud and info.audio_waveform_samples is not None:
      base = np.zeros(info.audio_waveform_samples, dtype=np.float32)
      out["audio"] = [(base.copy(), info.audio_sample_rate) for _ in range(n_aud)]
    return out


class MaxTextMultiModalProcessor(BaseMultiModalProcessor[MaxTextProcessingInfo]):
  """Routes processor outputs into the MultiModalKwargs vLLM expects.

  Placeholder substitution is a no-op: the input prompt already contains the
  raw placeholder token IDs, and the adapter's `__call__` reads
  `multimodal_embeddings` directly from `attention_metadata` (populated by
  tpu-inference's runner from the kwargs declared in `_get_mm_fields_config`).
  """

  def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs=None):
    from maxtext.utils import max_logging
    max_logging.log(f"_call_hf_processor called with mm_data keys={list(mm_data.keys()) if mm_data else None}")
    del tok_kwargs
    processor = self.info.get_hf_processor(**(mm_kwargs or {}))
    images = None
    audios = None
    if mm_data:
      images = mm_data.get("image") or mm_data.get("images")
      audios = mm_data.get("audio") or mm_data.get("audios")
    return processor(
        text=prompt,
        images=images,
        audios=audios,
        return_tensors="np",
    )

  def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
    del hf_processor_mm_kwargs
    from vllm.multimodal.inputs import MultiModalFieldConfig  # pylint: disable=import-outside-toplevel

    fields: dict = {}
    if "pixel_values" in hf_inputs:
      fields["pixel_values"] = MultiModalFieldConfig.batched("image")
    if "input_audio_features" in hf_inputs:
      fields["input_audio_features"] = MultiModalFieldConfig.batched("audio")
    return fields

  def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
    del hf_processor_mm_kwargs, out_mm_kwargs
    updates = []
    
    # Image no-op replacement
    img_id = self.info.mm_info.image_placeholder_id
    if img_id is not None:
      img_id = int(img_id)
      try:
        images = mm_items.get_items("image", object)
      except KeyError:
        images = []
      if images:
        target_id = 2 if self.info._model_name.startswith("gemma3") else img_id
        repl_len = 1 if self.info._model_name.startswith("gemma3") else int(self.info.mm_info.tokens_per_image)
        updates.append(
            PromptReplacement(
                modality="image",
                target=[target_id],
                replacement=[target_id] * repl_len,
            )
        )
        
    # Audio no-op replacement
    audio_id = self.info.mm_info.audio_placeholder_id
    if audio_id is not None:
      try:
        audios = mm_items.get_items("audio", object)
      except KeyError:
        audios = []
      if audios:
        tokens_per_audio = self.info.mm_info.tokens_per_audio
        if self.info._model_name.startswith("qwen3-omni"):
          waveforms = [item[0] if isinstance(item, tuple) else item for item in audios]
          max_len = max(len(w) for w in waveforms)
          from maxtext.multimodal.processor_qwen3_omni import _np_extract_fbank_features, _get_feat_extract_output_lengths
          dummy_fbank = _np_extract_fbank_features(np.zeros((1, max_len), dtype=np.float32))
          tokens_per_audio = int(_get_feat_extract_output_lengths(np.array(dummy_fbank.shape[2])))
          
        updates.append(
            PromptReplacement(
                modality="audio",
                target=[audio_id],
                replacement=[audio_id] * tokens_per_audio,
            )
        )
      
    return updates


@MULTIMODAL_REGISTRY.register_processor(
    MaxTextMultiModalProcessor,
    info=MaxTextProcessingInfo,
    dummy_inputs=MaxTextDummyInputsBuilder,
)
class MaxTextForConditionalGeneration(MaxTextForCausalLM):
  """A vLLM-compatible conditional-generation wrapper for MaxText models.

  Extends `MaxTextForCausalLM` with multimodal input handling: parses
  `pixel_values` from vLLM kwargs, runs the MaxText vision encoder, and
  passes precomputed embeddings into the decoder forward pass.
  """

  _self_manages_sharding: bool = True
  supports_multimodal = True

  def _parse_and_validate_image_input(self, **kwargs) -> jnp.ndarray | None:
    """Extract `pixel_values` from vLLM kwargs and convert to a JAX array."""
    pixel_values = kwargs.pop("pixel_values", None)
    if pixel_values is None:
      return None

    # vLLM may pass torch tensors; convert without forcing a dtype-specific path.
    if hasattr(pixel_values, "contiguous") and hasattr(pixel_values, "numpy"):
      import torch  # pylint: disable=import-outside-toplevel

      if pixel_values.dtype == torch.bfloat16:
        # numpy doesn't support bfloat16; round-trip via int16 view.
        pixel_values = (
            pixel_values.contiguous().view(torch.int16).numpy().view(jnp.bfloat16)
        )
      else:
        pixel_values = pixel_values.numpy()

    return jnp.asarray(pixel_values)

  def _parse_and_validate_audio_input(self, **kwargs) -> jnp.ndarray | None:
    input_audio_features = kwargs.pop("input_audio_features", None)
    if input_audio_features is None:
      return None
    if hasattr(input_audio_features, "numpy"):
      input_audio_features = input_audio_features.numpy()
    return jnp.asarray(input_audio_features)

  def embed_multimodal(self, **kwargs) -> list[jax.Array]:
    """Generate multimodal embeddings for vLLM's two-stage execution."""
    embeds = []
    pixel_values = self._parse_and_validate_image_input(**kwargs)
    if pixel_values is not None:
      if not isinstance(self.model, nnx.Module):
        raise ValueError("Model is not initialized.")

      with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
        image_embeddings, _ = self.model.vision_encoder(
            input_images=pixel_values,
            deterministic=True,
        )
        batch_size = image_embeddings.shape[0]
        embeds.extend([image_embeddings[i].reshape(-1, image_embeddings.shape[-1]) for i in range(batch_size)])

    input_audio_features = self._parse_and_validate_audio_input(**kwargs)
    if input_audio_features is not None:
      if not isinstance(self.model, nnx.Module):
        raise ValueError("Model is not initialized.")

      with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
        audio_embeddings = self.model.audio_encoder(
            input_audio=input_audio_features,
            deterministic=True,
        )
        batch_size = audio_embeddings.shape[0]
        embeds.extend([audio_embeddings[i].reshape(-1, audio_embeddings.shape[-1]) for i in range(batch_size)])

    return embeds

  def __call__(self, *args, **kwargs):
    from maxtext.utils import max_logging
    max_logging.log(f"__call__ received {len(args)} positional args: {[type(x).__name__ for x in args]}")
    
    kv_caches = args[0]
    attention_metadata = kwargs.get("attention_metadata")
    real_input_ids = None
    
    for arg in args[1:]:
      if hasattr(arg, "multimodal_embeddings") or type(arg).__name__ == "AttentionMetadata":
        attention_metadata = arg
      elif real_input_ids is None and (hasattr(arg, "ndim") or type(arg).__name__ in ("DynamicJaxprTracer", "ArrayImpl")):
        real_input_ids = arg
          
    if attention_metadata is not None:
      multimodal_embeddings = getattr(attention_metadata, "multimodal_embeddings", None)
      if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
        if self.maxtext_config.model_name.startswith("qwen3-omni"):
          image_embeds = []
          audio_embeds = []
          for emb in multimodal_embeddings:
            if emb.shape[0] == 576:
              image_embeds.append(emb)
            else:
              audio_embeds.append(emb)
              
          if image_embeds:
            kwargs["precomputed_multimodal_embeddings"] = jnp.concatenate(image_embeds, axis=0)
          if audio_embeds:
            kwargs["precomputed_audio_embeddings"] = jnp.concatenate(audio_embeds, axis=0)
        else:
          if isinstance(multimodal_embeddings, list):
            multimodal_embeddings = jnp.concatenate(multimodal_embeddings, axis=0)
          kwargs["precomputed_multimodal_embeddings"] = multimodal_embeddings

    res = super().__call__(
        kv_caches,
        real_input_ids,
        attention_metadata,
        **kwargs,
    )
    if isinstance(res, tuple) and len(res) == 4:
      return res[:3]
    return res
