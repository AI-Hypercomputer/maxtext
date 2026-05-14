# Copyright 2023–2025 Google LLC
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

"""MaxText vLLM adapter package."""

from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model
from .adapter import MaxTextForCausalLM


logger = init_logger(__name__)


def _patch_vllm_uses_mrope_for_maxtext() -> None:
  """Suppress vLLM's M-RoPE detection for MaxTextForCausalLM.

  vLLM's ``uses_mrope`` returns True whenever the HF config has
  ``rope_parameters.mrope_section``, which is the case for the Qwen3 family
  even for the text-only models MaxText currently serves. That flag then
  drives tpu-inference to call ``get_mrope_input_positions`` on the JIT'd
  model (which MaxTextForCausalLM doesn't define, so it ends up as None and
  the persistent batch manager dereferences it on the first request) and to
  precompile mrope-shaped position tensors that our text-only Jax model
  can't consume. Force-return False when the HF config is targeting
  ``MaxTextForCausalLM``. Drop once MaxText supports true multimodal serving
  via vLLM, or vLLM gains a per-architecture mrope opt-out.
  """
  # pylint: disable=import-outside-toplevel
  import vllm.config.model as _vllm_config_model
  import vllm.transformers_utils.config as _vllm_config_utils

  orig_uses_mrope = _vllm_config_utils.uses_mrope

  def _maxtext_uses_mrope(config) -> bool:
    architectures = getattr(config, "architectures", None) or []
    if "MaxTextForCausalLM" in architectures:
      return False
    return orig_uses_mrope(config)

  _vllm_config_utils.uses_mrope = _maxtext_uses_mrope
  # vllm.config.model imported uses_mrope as a local name; rebind that too so
  # ModelConfig.uses_mrope picks up the patch.
  _vllm_config_model.uses_mrope = _maxtext_uses_mrope


def _patch_tpu_inference_jax_kv_spec_for_maxtext() -> None:
  """Have tpu-inference's JAX kv_cache_spec builder honor ``layer_types == 'linear_attention'``.

  Upstream ``tpu_inference.runner.kv_cache_manager.KVCacheManager.get_kv_cache_spec``
  has a TODO to unify the hybrid kv-cache path with torchax. Until that lands,
  any ``"linear_attention"`` entry in the HF config's ``layer_types`` list silently
  becomes a ``FullAttentionSpec`` when no torch attention modules are registered
  (the JAX case — MaxTextForCausalLM is an ``nnx.Module``, registers nothing in
  ``static_forward_context``). That breaks Qwen3-Next / Qwen3.5 served via MaxText:
  the GDN layers get paged-attention caches instead of mamba ``(conv_state,
  recurrent_state)`` tuples. Replace those slots with a ``MambaSpec`` built from
  the model's ``get_mamba_state_shape_from_config``. Drop once tpu-inference's
  upstream JAX path supports MambaSpec natively.
  """
  # pylint: disable=import-outside-toplevel
  import dataclasses

  from tpu_inference.runner.kv_cache_manager import KVCacheManager
  from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
  from vllm.v1.kv_cache_interface import MambaSpec

  orig_get_kv_cache_spec = KVCacheManager.get_kv_cache_spec

  def patched(self):
    spec = orig_get_kv_cache_spec(self)
    text_config = getattr(
        self.runner.model_config,
        "hf_text_config",
        getattr(self.runner.model_config, "hf_config", None),
    )
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
      return spec

    # Architectures live on the top-level HF config (not the text sub-config).
    # Check both so we don't miss it when the override is applied at either layer.
    architectures = list(
        getattr(self.runner.model_config.hf_config, "architectures", None) or []
    ) + list(getattr(text_config, "architectures", None) or [])
    if "MaxTextForCausalLM" not in architectures:
      # Don't disturb foreign architectures sharing this process.
      return spec

    shapes = MaxTextForCausalLM.get_mamba_state_shape_from_config(self.runner.vllm_config)
    dtypes = MaxTextForCausalLM.get_mamba_state_dtype_from_config(self.runner.vllm_config)
    block_size = (
        self.runner.cache_config.block_size
        * self.runner.vllm_config.parallel_config.decode_context_parallel_size
    )

    # vLLM requires every layer's page_size_bytes to match before grouping
    # (vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size). Full-attn
    # and mamba state shapes give very different natural page sizes, so we
    # mirror tpu-inference's `update_mamba_page_size_padded` (only invoked
    # in the torch path) and pad both families to a common
    # per-`shared_by`-group footprint.
    attn_page_size = next(
        (s.page_size_bytes for s in spec.values() if not isinstance(s, MambaSpec)),
        None,
    )
    probe_mamba = MambaSpec(
        block_size=block_size, shapes=tuple(shapes), dtypes=tuple(dtypes)
    )
    mamba_unpadded = probe_mamba.page_size_bytes

    num_attn = sum(1 for lt in layer_types if lt != "linear_attention")
    num_mamba = sum(1 for lt in layer_types if lt == "linear_attention")
    if attn_page_size is None or num_attn == 0 or num_mamba == 0:
      # Pure mamba or pure attn; nothing to unify.
      uniform = None
    else:
      mn = min(num_attn, num_mamba)
      mx = max(num_attn, num_mamba)
      group_size = mx if mx < mn * 1.5 else mn
      num_attn_groups = (num_attn + group_size - 1) // group_size
      num_mamba_groups = (num_mamba + group_size - 1) // group_size
      uniform = int(
          num_attn_groups * attn_page_size + num_mamba_groups * mamba_unpadded
      )
      # Persist the same value tpu-inference's torch path would have stored,
      # so the per-layer allocator math at kv_cache_manager.py:700-720 lines up.
      self._hybrid_uniform_page_size_bytes = uniform
      self.runner.cache_config.mamba_page_size_padded = uniform
      for key, s in list(spec.items()):
        if not isinstance(s, MambaSpec):
          spec[key] = dataclasses.replace(s, page_size_padded=uniform)

    replaced = 0
    for i, layer_type in enumerate(layer_types):
      if layer_type != "linear_attention":
        continue
      key = f"layer.{i}"
      if key not in spec:
        continue
      spec[key] = MambaSpec(
          block_size=block_size,
          shapes=tuple(shapes),
          dtypes=tuple(dtypes),
          page_size_padded=uniform,
          mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
      )
      replaced += 1
    logger.info(
        "[mt-kv-spec-patch] replaced %d entries with MambaSpec (uniform_page_size=%s)",
        replaced,
        uniform,
    )
    return spec

  KVCacheManager.get_kv_cache_spec = patched


def register():
  """Register MaxTextForCausalLM model with tpu_inference and vllm.

  Note, this function is invoked directly by the vLLM engine during startup. As such,
  it leverages vLLM logging to report its status.
  """
  logger.info("Registering MaxTextForCausalLM model with tpu_inference and vllm.")
  _patch_vllm_uses_mrope_for_maxtext()
  _patch_tpu_inference_jax_kv_spec_for_maxtext()
  register_model("MaxTextForCausalLM", MaxTextForCausalLM)
  logger.info("Successfully registered MaxTextForCausalLM model.")
