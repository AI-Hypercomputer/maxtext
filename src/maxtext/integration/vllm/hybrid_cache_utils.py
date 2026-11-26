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

import importlib
import inspect
import math
from typing import Any

import jax.numpy as jnp

from maxtext.utils import max_logging


def call_with_supported_kwargs(func: Any, **kwargs) -> Any:
  """Calls ``func`` with only the keyword arguments its signature accepts.

  Private tpu-inference helpers change signatures between revisions, and the
  MaxText adapter has to work against whichever revision the runtime image
  installs. Rather than pinning one argument list, the caller supplies every
  value it can compute and this helper forwards the subset the installed
  callable declares. Callables that accept ``**kwargs`` get everything.

  Args:
    func: The callable to invoke.
    **kwargs: Candidate keyword arguments.

  Returns:
    Whatever ``func`` returns.
  """
  try:
    parameters = inspect.signature(func).parameters
  except (TypeError, ValueError):
    # C-implemented callables may not expose a signature; pass everything and
    # let the callable complain if it disagrees.
    return func(**kwargs)

  if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
    return func(**kwargs)

  supported = {name: value for name, value in kwargs.items() if name in parameters}
  dropped = sorted(set(kwargs) - set(supported))
  if dropped:
    max_logging.log(f"{getattr(func, '__name__', repr(func))} does not accept {dropped}; omitting them.")
  return func(**supported)


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


def get_hf_text_config(model_config: Any) -> Any:
  """Returns the HF text config of a vLLM ``ModelConfig``.

  Prefers ``model_config.hf_text_config``; otherwise unwraps
  ``hf_config.text_config`` (multimodal configs such as Qwen3.5-VL keep the
  text model's fields, e.g. ``layer_types``, there), falling back to
  ``hf_config`` itself. Returns None if ``model_config`` has neither.
  """
  if model_config is None:
    return None
  hf_text_config = getattr(model_config, "hf_text_config", None)
  if hf_text_config is not None:
    return hf_text_config
  hf_config = getattr(model_config, "hf_config", None)
  return getattr(hf_config, "text_config", None) or hf_config


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
    if not isinstance(name, str) or not name.startswith(_LAYER_NAME_PREFIX):
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
  caches no decoder layer maps to are passed through unchanged.

  With KV sharing several layers map to one physical cache, but only the donor
  layer replaces its entry; the sharing layers' entries are still the gathered
  input object. Those are skipped, otherwise they would overwrite the donor's
  update (gemma4-e2b: layers 15-34 share the caches of layers 13/14). Among
  replaced entries the highest-numbered layer wins.
  """
  if physical_indices is None:
    return layer_kv_caches
  if len(layer_kv_caches) != len(physical_indices):
    raise ValueError(
        f"Expected layer_kv_caches to have length {len(physical_indices)}, " f"but got {len(layer_kv_caches)}."
    )
  updated = list(kv_caches)
  for lyr, idx in enumerate(physical_indices):
    if layer_kv_caches[lyr] is kv_caches[idx]:
      # Not updated by this layer; writing it back would clobber the donor's update.
      continue
    updated[idx] = layer_kv_caches[lyr]
  return updated


# Per-layer type spellings of recurrent (stateful, non-KV-cache) layers. These
# are the non-attention kinds of vLLM's ``LayerBlockType``
# (``vllm.config.model``). Hybrid detection matches these *inclusively* rather
# than excluding a list of attention spellings: attention spellings keep
# growing (``sliding_attention``, ``indexed_attention``,
# ``deepseek_sparse_attention``, ``qwen_sparse_attention``, ...), and treating
# an unknown one as recurrent would silently switch a pure-attention model into
# vLLM's Mamba cache mode. An unknown recurrent spelling instead leaves the
# model non-hybrid, which fails loudly (attention caches for recurrent layers)
# rather than silently.
_RECURRENT_LAYER_TYPES = frozenset({"linear_attention", "mamba"})

# The architecture name MaxText models are registered under in vLLM. Callers
# select it with ``hf_overrides={"architectures": ["MaxTextForCausalLM"]}``.
MAXTEXT_VLLM_ARCHITECTURE = "MaxTextForCausalLM"


def is_hybrid_hf_config(hf_text_config: Any) -> bool:
  """Returns True if ``hf_text_config`` mixes recurrent layers with other layers.

  Recurrent layers are Mamba, GDN, linear attention, and similar. The check is
  model-agnostic and reads the per-layer fields HF / vLLM configs use for
  hybrid models, in this order:

  * ``layer_types`` (Qwen3-Next / Qwen3.5, Granite-4) and ``layers_block_type``
    (Jamba, NemotronH, Zamba2): hybrid if at least one entry is a recurrent
    type (``_RECURRENT_LAYER_TYPES``) and at least one is not. Attention-only
    models are not hybrid whatever their attention spelling (``full_attention``,
    ``sliding_attention``, ``deepseek_sparse_attention``, ...), and neither are
    configs with only attention-plus-MLP/MoE entries.
  * ``full_attention_interval`` (Qwen3-Next family, when ``layer_types`` is not
    materialized, e.g. a generic ``PretrainedConfig`` on a transformers version
    that does not know the model type): every N-th layer is full attention and
    the rest are linear attention, so hybrid if N > 1.
  * ``linear_attn_config`` (Kimi Linear): hybrid if both ``kda_layers`` and
    ``full_attn_layers`` are non-empty.
  * ``attn_type_list`` (MiniMax): hybrid if it has both ``0`` (linear
    attention) and ``1`` (softmax attention) entries.

  Args:
    hf_text_config: The HF text config (``ModelConfig.hf_text_config``).

  Returns:
    Whether the model has both recurrent and non-recurrent layers.
  """
  if hf_text_config is None:
    return False
  for field in ("layer_types", "layers_block_type"):
    layer_types = getattr(hf_text_config, field, None)
    if layer_types:
      is_recurrent = [t in _RECURRENT_LAYER_TYPES for t in layer_types]
      return any(is_recurrent) and not all(is_recurrent)
  full_attention_interval = getattr(hf_text_config, "full_attention_interval", None)
  if isinstance(full_attention_interval, int):
    return full_attention_interval > 1
  linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
  if isinstance(linear_attn_config, dict):
    return bool(linear_attn_config.get("kda_layers")) and bool(linear_attn_config.get("full_attn_layers"))
  attn_type_list = getattr(hf_text_config, "attn_type_list", None)
  if attn_type_list:
    return 0 in attn_type_list and 1 in attn_type_list
  return False


# MaxText decoder blocks whose recurrent layers the vLLM adapter gives Mamba
# KV cache specs.
HYBRID_DECODER_BLOCKS = frozenset({"qwen3_next", "qwen3_5"})


def validate_hybrid_detection(decoder_block: Any, hf_text_config: Any) -> None:
  """Fails if the HF config and the MaxText decoder block disagree on hybrid-ness.

  vLLM's hybrid cache config (``mamba_cache_mode``, ``mamba_block_size``) is
  derived from the HF config (``is_hybrid_hf_config``), but the KV cache specs
  come from the MaxText decoder block. If they disagree, vLLM would serve the
  model with the wrong cache mode, so raise with an actionable message instead.

  Args:
    decoder_block: The MaxText ``decoder_block`` (string or enum).
    hf_text_config: The HF text config (``ModelConfig.hf_text_config``).

  Raises:
    ValueError: if exactly one of the two describes a hybrid model.
  """
  name = getattr(decoder_block, "value", decoder_block) or ""
  maxtext_hybrid = name in HYBRID_DECODER_BLOCKS
  hf_hybrid = is_hybrid_hf_config(hf_text_config)
  if maxtext_hybrid and not hf_hybrid:
    raise ValueError(
        f"MaxText decoder_block={name!r} has recurrent (Mamba/GDN) layers, but the HF config "
        f"(model_type={getattr(hf_text_config, 'model_type', None)!r}) does not describe a hybrid model, so vLLM "
        "would not enable its hybrid cache mode. Add the per-layer types to the HF config (e.g. `layer_types` or "
        "`full_attention_interval` via hf_overrides), or extend `is_hybrid_hf_config` for this config format."
    )
  if hf_hybrid and not maxtext_hybrid:
    raise ValueError(
        f"The HF config (model_type={getattr(hf_text_config, 'model_type', None)!r}) describes a hybrid "
        f"attention + recurrent model, but MaxText decoder_block={name!r} has no hybrid KV cache support in the vLLM "
        f"adapter (supported: {sorted(HYBRID_DECODER_BLOCKS)})."
    )


def register_maxtext_hybrid_model_config(
    models_config_map: dict[str, Any],
    verify_base_cls: type[Any],
    hybrid_config_cls: Any,
) -> bool:
  """Registers ``MaxTextForCausalLM`` in vLLM's ``MODELS_CONFIG_MAP``.

  vLLM >= 0.33 deprecated ``EngineArgs.mamba_cache_mode``. With prefix caching
  enabled it now derives ``cache_config.mamba_cache_mode = "align"`` and
  ``mamba_block_size`` in ``HybridAttentionMambaModelConfig``, which
  ``VllmConfig.try_verify_and_update_config`` only runs when
  ``model_config.is_hybrid``. That flag comes from the registered model class,
  and the MaxText wrapper class serves every MaxText model, so it is not set.

  The registered verifier detects hybrid models from the HF config
  (``is_hybrid_hf_config``) and applies vLLM's generic hybrid hook to them. No
  model family is hard-coded, and non-hybrid MaxText models are left untouched.

  Args:
    models_config_map: Mutable architecture-to-verifier mapping from vLLM.
    verify_base_cls: Base ``VerifyAndUpdateConfig`` class from vLLM.
    hybrid_config_cls: ``HybridAttentionMambaModelConfig`` class from vLLM.

  Returns:
    True if ``MaxTextForCausalLM`` was newly registered, False if an entry for
    it already exists (vLLM's, a previous call's, or another plugin's).
  """
  if MAXTEXT_VLLM_ARCHITECTURE in models_config_map:
    return False

  class MaxTextForCausalLMConfig(verify_base_cls):
    """Applies vLLM's hybrid attention/Mamba config update to hybrid models."""

    @staticmethod
    def verify_and_update_config(vllm_config: Any) -> None:
      """Runs HybridAttentionMambaModelConfig if the HF config is hybrid."""
      model_config = getattr(vllm_config, "model_config", None)
      if model_config is None:
        return
      # If vLLM already classifies the model as hybrid, it runs the hybrid hook
      # itself right after this verifier; don't apply it twice.
      if getattr(model_config, "is_hybrid", False):
        return
      if is_hybrid_hf_config(get_hf_text_config(model_config)):
        hybrid_config_cls.verify_and_update_config(vllm_config)

  models_config_map[MAXTEXT_VLLM_ARCHITECTURE] = MaxTextForCausalLMConfig
  return True


def maybe_install_tpu_hybrid_coordinator(
    vllm_config: Any,
    import_module_fn: Any = importlib.import_module,
) -> bool:
  """Installs tpu-inference's dual-pool hybrid KV cache coordinator hooks.

  When ``AsyncLLMEngine`` spawns an ``EngineCore`` subprocess on a single host
  (``UniProcExecutor`` with ``enable_dp_attention=False``), ``TPUPlatform``'s
  ``check_and_update_config`` ran only in the parent process and neither
  ``UniProcExecutor`` nor vLLM's default ``Scheduler`` re-installs
  ``TPUHybridKVCacheCoordinator`` in ``EngineCore``. Calling this during
  ``get_kv_cache_spec`` (after ``_mamba_num_blocks`` is populated and before
  ``EngineCore`` constructs ``Scheduler``) activates the dual attention/Mamba
  block pools in ``EngineCore`` instead of falling back to upstream vLLM's
  single-pool ``HybridKVCacheCoordinator``.

  Args:
    vllm_config: Active vLLM configuration passed to the hook installer.
    import_module_fn: Module importer callable (injectable for unit tests).

  Returns:
    True if the hook installer was invoked, False if unavailable or skipped.
  """
  if vllm_config is None:
    return False
  try:
    hybrid_coord_mod = import_module_fn("tpu_inference.core.hybrid_coordinator")
    install_fn = hybrid_coord_mod.maybe_install_hybrid_coordinator_hooks
  except (ImportError, AttributeError) as e:
    max_logging.log(f"Skipping TPUHybridKVCacheCoordinator hook installation: {e}")
    return False

  install_fn(vllm_config)
  return True
