# Copyright 2026 Google LLC
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

"""Model-specific helpers shared by MaxText's vLLM adapter."""

import math
from typing import Any

import jax.numpy as jnp


def normalize_vllm_input_positions(input_positions: Any):
  """Converts vLLM's position layout to MaxText decode/prefill layout.

  vLLM supplies ordinary positions as ``(num_tokens,)``, decode MRoPE positions
  as ``(3, num_tokens)``, and batched MRoPE positions as ``(3, batch, seq)``.
  MaxText treats flattened decode tokens as a batch with a singleton sequence dimension,
  expecting ``(num_tokens, 1)`` or ``(num_tokens, 1, 3)``, and batched MRoPE as
  ``(batch, seq, 3)``.
  """
  if input_positions is None:
    return None
  if input_positions.ndim == 1:
    return input_positions.reshape((-1, 1))
  if input_positions.ndim == 2 and input_positions.shape[0] == 3:
    return input_positions.T[:, None, :]
  if input_positions.ndim == 3 and input_positions.shape[0] == 3:
    return jnp.transpose(input_positions, (1, 2, 0))
  raise ValueError(
      "vLLM input positions must have shape (num_tokens,), (3, num_tokens), or (3, batch, seq); "
      f"got {input_positions.shape}."
  )


def build_qwen_gdn_cache_layout(cfg: Any, torch_module: Any):
  """Returns the shapes, dtypes, and unpadded bytes for a Qwen GDN cache."""
  key_dim = cfg.gdn_key_head_dim * cfg.gdn_num_key_heads
  value_dim = cfg.gdn_value_head_dim * cfg.gdn_num_value_heads
  conv_dim = key_dim * 2 + value_dim

  shapes = (
      (cfg.gdn_conv_kernel_dim - 1, conv_dim),
      (cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim),
  )
  # This is the TPU Inference / upstream vLLM contract regardless of model
  # weight or attention-KV dtype.
  dtypes = (torch_module.bfloat16, torch_module.float32)
  page_size_bytes = sum(
      math.prod(shape) * torch_module.empty((), dtype=dtype).element_size()
      for shape, dtype in zip(shapes, dtypes, strict=True)
  )
  return shapes, dtypes, page_size_bytes


_LAYER_NAME_PREFIX = "layer."


def resolve_layer_kv_cache_indices(layer_name_to_kvcache_index: Any, num_kv_caches: int) -> list[int] | None:
  """Maps decoder layer index -> physical index into vLLM's ``kv_caches`` list.

  tpu-inference allocates one physical cache per unique slot and reports which
  cache each layer uses through ``layer_name_to_kvcache_index`` (JAX-side layer
  names are ``layer.{i}``). The physical list is *not* guaranteed to be in layer
  order: hybrid models group the Mamba/GDN caches ahead of the attention caches,
  KV-sharing layers redirect to another layer's cache, and the legacy aliased
  layout backs several layers with one cache. MaxText decoders index the list
  positionally (``kv_caches[lyr]``), so the adapter uses this mapping to present
  them a layer-ordered view.

  Args:
    layer_name_to_kvcache_index: ``{layer_name: physical_index}``, or the
      ``tuple(dict.items())`` form tpu-inference passes as a static jit
      argument. ``None``/empty means "no mapping": callers fall back to
      positional indexing, matching the native tpu-inference models.
    num_kv_caches: Length of the physical ``kv_caches`` list, for validation.

  Returns:
    ``physical_indices`` with ``physical_indices[lyr]`` the position of layer
    ``lyr``'s cache in ``kv_caches``, or ``None`` when no mapping was given.

  Raises:
    ValueError: If the ``layer.{i}`` entries do not cover a contiguous
      ``0..n-1`` range, or an index falls outside ``kv_caches``.
  """
  if not layer_name_to_kvcache_index:
    return None
  if not isinstance(layer_name_to_kvcache_index, dict):
    layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)

  by_layer: dict[int, int] = {}
  for name, physical_index in layer_name_to_kvcache_index.items():
    # Other entries (e.g. a per-layer auxiliary cache suffix) are not decoder
    # layers and are left to the model that declared them.
    if not name.startswith(_LAYER_NAME_PREFIX):
      continue
    suffix = name[len(_LAYER_NAME_PREFIX) :]
    if not suffix.isdigit():
      continue
    by_layer[int(suffix)] = int(physical_index)

  if not by_layer:
    return None

  num_layers = max(by_layer) + 1
  missing = sorted(set(range(num_layers)) - set(by_layer))
  if missing:
    raise ValueError(
        f"layer_name_to_kvcache_index is missing decoder layers {missing} (expected layer.0..layer.{num_layers - 1})."
    )

  physical_indices = [by_layer[lyr] for lyr in range(num_layers)]
  out_of_range = [idx for idx in physical_indices if not 0 <= idx < num_kv_caches]
  if out_of_range:
    raise ValueError(
        f"layer_name_to_kvcache_index points outside the kv_caches list (len={num_kv_caches}): {sorted(set(out_of_range))}."
    )
  return physical_indices


def gather_layer_kv_caches(kv_caches: list[Any], physical_indices: list[int] | None) -> list[Any]:
  """Returns ``kv_caches`` re-ordered so that entry ``lyr`` is layer ``lyr``'s cache.

  A no-op (same list object) when ``physical_indices`` is ``None``. This is
  pure Python list indexing on traced values, so it adds no XLA ops.
  """
  if physical_indices is None:
    return kv_caches
  return [kv_caches[idx] for idx in physical_indices]


def scatter_layer_kv_caches(
    kv_caches: list[Any], layer_kv_caches: list[Any], physical_indices: list[int] | None
) -> list[Any]:
  """Writes layer-ordered updated caches back into the physical ``kv_caches`` layout.

  The returned list has the same length and order as the physical list the
  runner handed us, which is what its donation / out_shardings expect. Physical
  caches no decoder layer maps to are passed through unchanged. If several
  layers share one physical cache, the highest-numbered layer's update wins.
  """
  if physical_indices is None:
    return layer_kv_caches
  updated = list(kv_caches)
  for lyr, idx in enumerate(physical_indices):
    updated[idx] = layer_kv_caches[lyr]
  return updated
