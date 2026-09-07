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

"""Multimodal processor adapter registry for MaxText models in vLLM."""

from typing import Any, Protocol

import flax.linen as nn
import jax
from tpu_inference.models.jax.utils.multi_modal_utils import convert_torch_tensor_to_jax, normalize_mm_grid_thw


class MultimodalHandler(Protocol):
  """Connects one MaxText model family to vLLM's multimodal APIs.

  vLLM processes raw media and passes the resulting tensors to this handler.
  The handler converts those tensors and runs the MaxText modality encoder.
  The registered processor may come from vLLM or MaxText.
  """

  def register_processor(self, model_class: type) -> None:
    """Registers a vLLM-compatible processor for the MaxText model class."""

  def embed_multimodal(
      self,
      model: Any,
      maxtext_config: Any,
      mesh: jax.sharding.Mesh,
      **processor_outputs: Any,
  ) -> list[jax.Array]:
    """Runs the MaxText modality encoder on processor output tensors."""

  def placeholder_token_ids(self, hf_config: Any) -> list[int]:
    """Returns token IDs whose embeddings are replaced by modality values."""


class Qwen3VLMultimodalHandler:
  """Connects vLLM's Qwen3-VL processor to MaxText's vision encoder."""

  def register_processor(self, model_class: type) -> None:
    """Registers vLLM's Qwen3-VL processor for the MaxText wrapper."""
    # These imports are intentionally local. Text-only MaxText users should not
    # need vLLM's multimodal model modules during adapter import.
    from vllm.model_executor.models.qwen3_vl import (  # pylint: disable=import-outside-toplevel
        Qwen3VLDummyInputsBuilder,
        Qwen3VLMultiModalProcessor,
        Qwen3VLProcessingInfo,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY  # pylint: disable=import-outside-toplevel

    MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3VLProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )(model_class)

  def embed_multimodal(
      self,
      model: Any,
      maxtext_config: Any,
      mesh: jax.sharding.Mesh,
      **processor_outputs: Any,
  ) -> list[jax.Array]:
    """Runs the MaxText vision encoder on vLLM Qwen3-VL processor outputs."""
    pixel_values = convert_torch_tensor_to_jax(processor_outputs["pixel_values"], dtype=maxtext_config.dtype)
    image_grid_thw = normalize_mm_grid_thw(processor_outputs["image_grid_thw"])

    embeddings = []
    current_idx = 0
    patch_size = maxtext_config.patch_size_for_vit
    temporal_patch_size = maxtext_config.temporal_patch_size_for_vit
    channels = maxtext_config.num_channels_for_vit

    for grid_t, grid_h, grid_w in image_grid_thw:
      image_size = grid_t * grid_h * grid_w
      image_pixels = pixel_values[current_idx : current_idx + image_size, :]
      input_images = image_pixels.reshape(
          1,
          channels,
          grid_t * temporal_patch_size,
          grid_h * patch_size,
          grid_w * patch_size,
      )

      with mesh, nn.logical_axis_rules(maxtext_config.logical_axis_rules):
        image_embeddings, _ = model.vision_encoder(input_images=input_images, deterministic=True)
      embeddings.append(image_embeddings.squeeze(0))
      current_idx += image_size

    return embeddings

  def placeholder_token_ids(self, hf_config: Any) -> list[int]:
    """Returns Qwen3-VL image and video placeholder token IDs."""
    return [
        token_id
        for name in ("image_token_id", "video_token_id")
        if (token_id := getattr(hf_config, name, None)) is not None
    ]


class Gemma4MultimodalHandler:
  """Connects vLLM's Gemma4 processor to MaxText's vision encoder."""

  def register_processor(self, model_class: type) -> None:
    """Registers vLLM's Gemma4 processor for the MaxText wrapper."""
    from vllm.model_executor.models.gemma4_mm import (  # pylint: disable=import-outside-toplevel
        Gemma4DummyInputsBuilder,
        Gemma4MultiModalProcessor,
        Gemma4ProcessingInfo,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY  # pylint: disable=import-outside-toplevel

    MULTIMODAL_REGISTRY.register_processor(
        Gemma4MultiModalProcessor,
        info=Gemma4ProcessingInfo,
        dummy_inputs=Gemma4DummyInputsBuilder,
    )(model_class)

  def embed_multimodal(
      self,
      model: Any,
      maxtext_config: Any,
      mesh: jax.sharding.Mesh,
      **processor_outputs: Any,
  ) -> list[jax.Array]:
    """Runs the MaxText vision encoder on vLLM Gemma4 processor outputs."""
    raw_pixel_values = processor_outputs.get("pixel_values")
    raw_pixel_position_ids = processor_outputs.get("pixel_position_ids", processor_outputs.get("image_position_ids"))

    pixel_values = convert_torch_tensor_to_jax(raw_pixel_values, dtype=maxtext_config.dtype)
    pixel_position_ids = convert_torch_tensor_to_jax(raw_pixel_position_ids, dtype=jax.numpy.int32)

    if isinstance(pixel_values, list):
      embeddings = []
      for pv, ppos in zip(pixel_values, pixel_position_ids):
        if pv.ndim == 2:
          pv = jax.numpy.expand_dims(pv, 0)
        if ppos.ndim == 2:
          ppos = jax.numpy.expand_dims(ppos, 0)
        with mesh, nn.logical_axis_rules(maxtext_config.logical_axis_rules):
          image_embeddings, _ = model.vision_encoder(
              input_images=pv,
              positions_xy=ppos,
              deterministic=True,
          )
        embeddings.append(image_embeddings.reshape(-1, maxtext_config.emb_dim))
      return embeddings
    else:
      if pixel_values.ndim == 2:
        pixel_values = jax.numpy.expand_dims(pixel_values, 0)
      if pixel_position_ids.ndim == 2:
        pixel_position_ids = jax.numpy.broadcast_to(
            pixel_position_ids, (pixel_values.shape[0],) + pixel_position_ids.shape
        )

      embeddings = []
      batch_size = pixel_values.shape[0]
      for b in range(batch_size):
        pv = pixel_values[b : b + 1]
        ppos = pixel_position_ids[b : b + 1]
        with mesh, nn.logical_axis_rules(maxtext_config.logical_axis_rules):
          image_embeddings, _ = model.vision_encoder(
              input_images=pv,
              positions_xy=ppos,
              deterministic=True,
          )
        embeddings.append(image_embeddings.reshape(-1, maxtext_config.emb_dim))
      return embeddings

  def placeholder_token_ids(self, hf_config: Any) -> list[int]:
    """Returns Gemma4 image placeholder token IDs."""
    token_id = getattr(hf_config, "image_token_id", None)
    if token_id is not None:
      return [token_id]
    return [258880]


_MULTIMODAL_HANDLERS: tuple[tuple[str, MultimodalHandler], ...] = (
    ("qwen3-vl", Qwen3VLMultimodalHandler()),
    ("gemma4", Gemma4MultimodalHandler()),
)


def get_multimodal_handler(model_name: str | None) -> MultimodalHandler | None:
  """Returns the handler for a MaxText model name, or ``None``."""
  if not model_name:
    return None
  for model_prefix, handler in _MULTIMODAL_HANDLERS:
    if model_name.startswith(model_prefix):
      return handler
  return None
