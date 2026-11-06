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

"""vLLM adapter for MaxText models."""

import dataclasses
import os
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
import numpy as np
try:
  import omegaconf
except ImportError:  # pragma: no cover
  omegaconf = None
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.convert_utils import DEFAULT_TPU_NUM_LANES, compute_padded_moe_mlp_dim
from maxtext.integration.vllm.hybrid_cache_utils import (
    build_qwen_gdn_cache_layout,
    call_with_supported_kwargs,
    gather_layer_kv_caches,
    normalize_vllm_input_positions,
    resolve_layer_kv_cache_indices,
    scatter_layer_kv_caches,
)
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tpu_inference.models.jax.utils.multi_modal_utils import merge_multimodal_embeddings
from .multimodal import get_multimodal_handler


try:
  from tpu_inference.layers.common.attention_metadata import AttentionMetadata
except ImportError:
  # Mock for documentation build or environments without tpu_inference
  class AttentionMetadata:
    input_positions: jax.Array


from vllm.config import VllmConfig
try:
  from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig
except ImportError:  # pragma: no cover
  MODELS_CONFIG_MAP = {}

  class VerifyAndUpdateConfig:  # type: ignore[no-redef]
    pass

# Threshold to determine if the ratio of attention to mamba layers is highly imbalanced.
# If max_count / min_count >= this threshold, we group KV cache allocations by the
# smaller count to prevent excessive memory padding for the minority layer type.
_HYBRID_LAYER_IMBALANCE_THRESHOLD = 1.5

# MaxText decoder blocks that interleave GatedDeltaNet (linear attention) layers
# with full attention, i.e. vLLM hybrid (attention + mamba-style state) models.
HYBRID_DECODER_BLOCKS = ("qwen3_next", "qwen3_5")


def is_hybrid_maxtext_model(model_name: str | None) -> bool:
  """Whether the MaxText model `model_name` is a hybrid attention + GDN model.

  Reads `decoder_block` from the model's config (`configs/models/<name>.yml`)
  without building the full MaxText config, since vLLM needs the answer when
  the model class is registered, before any `VllmConfig` exists.
  """
  if not model_name:
    return False
  # -Instruct variants share the base model's architecture (as in pyconfig).
  model_name = model_name.replace("-Instruct", "")
  model_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "models", f"{model_name}.yml")
  if not os.path.isfile(model_config_path):
    return False
  if omegaconf is not None:
    decoder_block = omegaconf.OmegaConf.load(model_config_path).get("decoder_block", "")
  else:
    import yaml  # pylint: disable=import-outside-toplevel

    with open(model_config_path, "r", encoding="utf-8") as f:
      decoder_block = (yaml.safe_load(f) or {}).get("decoder_block", "")
  return str(decoder_block) in HYBRID_DECODER_BLOCKS


class MaxTextModelConfig(VerifyAndUpdateConfig):
  """vLLM config hook for `MaxTextForCausalLM`.

  When `vllm serve` (or `LLM`/`AsyncLLMEngine`) launches without
  `MAXTEXT_MODEL_NAME` set in the environment before `register()` runs —
  passing `model_name` via `additional_config["maxtext_config"]["model_name"]`
  instead — `MaxTextForCausalLM.is_hybrid` is still `False` when `ModelConfig`
  inspects the registered class. Updating `_model_info.is_hybrid` here (inside
  `VllmConfig.try_verify_and_update_config`, before vLLM checks
  `if self.model_config.is_hybrid:`) lets vLLM's own
  `HybridAttentionMambaModelConfig.verify_and_update_config` default
  `cache_config.mamba_cache_mode = "align"` and `mamba_block_size` when prefix
  caching is enabled.
  """

  @staticmethod
  def verify_and_update_config(vllm_config: VllmConfig) -> None:
    additional_config = getattr(vllm_config, "additional_config", None)
    maxtext_cfg = (
        additional_config.get("maxtext_config", {})
        if isinstance(additional_config, dict)
        else {}
    )
    if not isinstance(maxtext_cfg, dict):
      maxtext_cfg = {}

    model_name = maxtext_cfg.get("model_name") or os.environ.get("MAXTEXT_MODEL_NAME")
    if model_name:
      os.environ.setdefault("MAXTEXT_MODEL_NAME", str(model_name))

    explicit_decoder_block = maxtext_cfg.get("decoder_block")
    if explicit_decoder_block is not None:
      is_hybrid = str(explicit_decoder_block) in HYBRID_DECODER_BLOCKS
    else:
      is_hybrid = is_hybrid_maxtext_model(model_name)

    if MaxTextForCausalLM.is_hybrid != is_hybrid:
      MaxTextForCausalLM.is_hybrid = is_hybrid
      try:
        from tpu_inference.models.common.model_loader import register_model  # pylint: disable=import-outside-toplevel

        register_model("MaxTextForCausalLM", MaxTextForCausalLM)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is not None:
      model_info = getattr(model_config, "_model_info", None)
      if model_info is not None and getattr(model_info, "is_hybrid", None) != is_hybrid:
        if dataclasses.is_dataclass(model_info):
          model_config._model_info = dataclasses.replace(model_info, is_hybrid=is_hybrid)
        else:
          model_info.is_hybrid = is_hybrid
      elif model_info is None and not isinstance(getattr(type(model_config), "is_hybrid", None), property):
        model_config.is_hybrid = is_hybrid

    if is_hybrid:
      cache_config = getattr(vllm_config, "cache_config", None)
      if (
          cache_config is not None
          and getattr(cache_config, "enable_prefix_caching", False)
          and getattr(cache_config, "mamba_cache_mode", "none") == "none"
      ):
        cache_config.mamba_cache_mode = "align"
        if getattr(cache_config, "mamba_block_size", None) is None:
          cache_config.mamba_block_size = getattr(cache_config, "block_size", None)


def next_power_of_two(x: int) -> int:
  """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
  assert x > 0
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


def generate_maxtext_config(vllm_config: VllmConfig) -> pyconfig.HyperParameters:
  """Generates a MaxText configuration from a vLLM configuration.

  This function takes a vLLM configuration object and translates relevant
  parameters into a MaxText `HyperParameters` object. It handles loading
  paths and model names from the vLLM config, and applies a base MaxText
  vLLM configuration file.

  Args:
    vllm_config: The vLLM configuration object containing model and load
      parameters.

  Returns:
    A `pyconfig.HyperParameters` object configured for MaxText.

  Raises:
    ValueError: If `hf_config_path` is not provided in the vLLM model config.
  """
  if "maxtext_config" in vllm_config.additional_config:
    overrides = dict(vllm_config.additional_config["maxtext_config"])
    overrides.pop("mtp_hf_checkpoint_path", None)
  else:
    overrides = {}

  if vllm_config.load_config.load_format == "dummy":
    if overrides.get("load_parameters_path") is not None:
      max_logging.log(
          "Warning: load_parameters_path is set when using dummy load format. Checkpoint loading will be skipped."
      )
      overrides["load_parameters_path"] = None

  if overrides.get("attention") == "vllm_batched_rpa" or overrides.get("use_batched_rpa", False):
    os.environ["USE_BATCHED_RPA_KERNEL"] = "1"

  # Add base config path to positional args
  base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  argv_list = ["", str(base_config_path)]

  # Gather sharding information from vLLM config to determine transformations to apply
  sharding_config = vllm_config.sharding_config
  tp = sharding_config.tp_size
  ep = sharding_config.expert_size
  attn_dp = sharding_config.attn_dp_size

  # Calculate the maximum TP size across attention and MLP dimensions
  kv_tp_size = tp * ep
  moe_mlp_tp_size = tp * attn_dp

  # Gather information on the hidden size of MoE models to determine if padding is needed
  # to meet MLP MoE requirements for tpu-inference GMM_v2 kernel.
  hf_config = (
      vllm_config.model_config.hf_config.text_config
      if hasattr(vllm_config.model_config.hf_config, "text_config")
      else vllm_config.model_config.hf_config
  )
  hidden_size = getattr(hf_config, "moe_intermediate_size", None)
  try:
    num_lanes = pltpu.get_tpu_info().num_lanes
  except Exception:  # pylint: disable=broad-exception-caught
    num_lanes = DEFAULT_TPU_NUM_LANES
  num_kv_heads = hf_config.num_key_value_heads

  # Number of KV heads in global attention layers (None if the field is absent or unset).
  num_global_kv_heads = getattr(hf_config, "num_global_key_value_heads", None)
  use_global_kv_heads = num_global_kv_heads is not None

  max_logging.log(
      f"vLLM sharding config: hidden_size={hidden_size}, kv_heads={num_kv_heads}, global_kv_heads={num_global_kv_heads}, "
      f"num_lanes={num_lanes}, tp={tp}, attn_dp={attn_dp}, ep={ep}, moe_mlp_tp_size={moe_mlp_tp_size}"
  )

  # The native tpu-inference model paths derive use_ep from
  # parallel_config.enable_expert_parallel, but the mesh built by
  # ShardingConfigManager.from_vllm_config takes expert parallelism only from
  # additional_config's sharding_strategy. MaxText derives use_ep from the mesh, so
  # --enable-expert-parallel alone leaves the experts unsharded here while the native
  # implementation of the same model runs expert-parallel. Warn rather than fail, since
  # the run is still correct, only sharded differently than the flag suggests.
  expert_shard_degree = ep * sharding_config.attn_dp_expert_size
  if vllm_config.parallel_config.enable_expert_parallel and expert_shard_degree == 1:
    max_logging.warning(
        "--enable-expert-parallel was requested but the mesh has no expert shards "
        f"(expert={ep}, attn_dp_expert={sharding_config.attn_dp_expert_size}), so MaxText will shard the MoE over "
        "the MLP dimension instead of the expert dimension. The vLLM flag does not reach the JAX mesh; set "
        'expert_parallelism in the vLLM additional_config sharding_strategy, e.g. \'{"sharding": '
        '{"sharding_strategy": {"expert_parallelism": <num_devices>}}}\', to actually shard experts.'
    )

  # Replicate the number of KV heads if its less than the total degree of model parallelism
  if kv_tp_size % num_kv_heads == 0 and num_kv_heads < kv_tp_size:
    max_logging.log(
        f"Padding num_kv_heads from {num_kv_heads} to {kv_tp_size} to match the degree of tensor parallelism."
    )
    overrides["base_num_kv_heads"] = kv_tp_size

  # Replicate the number of global KV heads if its less than the total degree of model parallelism
  if use_global_kv_heads and kv_tp_size % num_global_kv_heads == 0 and num_global_kv_heads < kv_tp_size:
    max_logging.log(
        f"Padding num_global_kv_heads from {num_global_kv_heads} "
        f"to {kv_tp_size} to match the degree of tensor parallelism."
    )
    overrides["global_num_kv_heads"] = kv_tp_size

  # Pad the hidden size of MoE models if the MLP dimension is less than expected by the GMM_v2 kernel in tpu-inference.
  # The GMM_v2 kernel requires the MLP dimension per expert to be at least 2x the number of TPU lanes
  # to ensure efficient execution. See the validate_inputs() method in the following file for more details:
  # https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/megablox/gmm_v2.py
  padded_hidden_size = compute_padded_moe_mlp_dim(hidden_size, moe_mlp_tp_size, num_lanes)
  if padded_hidden_size is not None and padded_hidden_size != hidden_size:

    # This inflates every expert weight, so it is a real memory/FLOP cost rather than a
    # cosmetic reshape: at moe_mlp_tp_size=4 a 512-wide MoE is padded to 1024 (2x the MoE
    # weights), and at moe_mlp_tp_size=8 to 2048 (4x). Log it at WARNING so it is visible
    # under vLLM's logging configuration, which does not surface absl INFO records.
    max_logging.warning(
        f"Padding moe_intermediate_size from {hidden_size} to {padded_hidden_size} to match MLP MoE requirements "
        f"(moe_mlp_tp_size={moe_mlp_tp_size}, 2*num_lanes={2 * num_lanes}). This multiplies the MoE weights and "
        f"MoE FLOPs by {padded_hidden_size / hidden_size:g}x. Consider sharding the MoE over the expert axis "
        f"instead, by setting expert_parallelism in the vLLM additional_config sharding_strategy."
    )
    overrides["padded_base_moe_mlp_dim"] = padded_hidden_size

  # Align mode pins the mamba block size to the attention one.
  cache_config = vllm_config.cache_config
  if getattr(cache_config, "mamba_cache_mode", "none") == "align":
    overrides["gdn_mamba_block_size"] = cache_config.block_size

  # Enable 1-layer Multi-Token Prediction (MTP) when vLLM speculative_config uses method="mtp".
  spec_config = getattr(vllm_config, "speculative_config", None)
  if getattr(spec_config, "method", None) == "mtp":
    overrides.setdefault("mtp_num_layers", 1)

  # Propagate vLLM's --enable_return_routed_experts flag to MaxText's MoE router capture.
  aux_output_config = getattr(vllm_config, "aux_output_config", None)
  if getattr(aux_output_config, "enable_return_routed_experts", False):
    overrides.setdefault("return_routed_experts", True)

  maxtext_config = pyconfig.initialize(argv_list, **overrides)
  return maxtext_config


class MaxTextForCausalLM(nnx.Module):
  """A vLLM-compatible causal language model wrapper for MaxText.

  This class serves as the primary interface for integrating MaxText models
  into the vLLM serving framework, specifically for causal language modeling
  tasks. It handles configuration generation, model initialization, and execution
  of the decoding step.
  """

  # Signal to tpu-inference model_loader that this class manages its own
  # JIT-sharded initialization (via create_nnx_model with out_shardings).
  # When True, model_loader skips wrapping __init__ in an outer bare @jax.jit,
  _self_manages_sharding: bool = True
  supports_multimodal: bool = False
  # vLLM's `IsHybrid` flag (attention + mamba-style recurrent state). vLLM
  # reads it from the registered class to enable mamba prefix caching
  # ("align" cache mode) and the hybrid KV cache layout. Set per model by
  # `register()`, since one class serves every MaxText model.
  is_hybrid: bool = False

  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
    """Initializes the MaxTextForCausalLM model.

    Args:
      vllm_config: The vLLM configuration object.
      rng_key: A JAX random key for model initialization.
      mesh: The JAX mesh device for model sharding.
    """
    self.vllm_config = vllm_config
    self.cfg = vllm_config.model_config
    self.maxtext_config = generate_maxtext_config(vllm_config)
    self.multimodal_handler = get_multimodal_handler(self.maxtext_config.model_name)

    # Model configuration
    self.mesh = mesh
    self.model_mode = MODEL_MODE_AUTOREGRESSIVE
    self.is_text_generation_model = True

    # Model creation
    self.model: nnx.Module | None = None

    # Indicates that the model handles its own sharding logic
    self._self_manages_sharding = True

    # Handle dummy weight loading during initialization
    if vllm_config.load_config.load_format == "dummy":
      self.load_weights(rng_key)

    elif self.maxtext_config.load_parameters_path is None:
      max_logging.log("Warning: No load_parameters_path provided. The model will be initialized with random weights.")

  def modules(self):
    """Dummy method to satisfy vLLM's internal cleanup logic."""
    return []

  # pylint: disable=keyword-arg-before-vararg
  def __call__(
      self,
      kv_caches: list[jax.Array],
      input_ids: jax.Array,
      attention_metadata: AttentionMetadata,
      inputs_embeds: jax.Array | None = None,
      _input_positions=None,
      _layer_name_to_kvcache_index=None,
      *args,
      **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array], list[jax.Array] | None]:
    """Performs a forward pass through the causal language model.

    The positional layout mirrors the tpu-inference runner's model call
    (``kv_caches, input_ids, attention_metadata, inputs_embeds, input_positions,
    layer_name_to_kvcache_index, lora_metadata, ...``), the same contract the
    native JAX models such as Gemma4 consume.

    Args:
      kv_caches: The physical KV caches allocated by tpu-inference, in the
        runner's slot order. This is not necessarily layer order (see
        ``_layer_name_to_kvcache_index``).
      input_ids: A JAX array of input token IDs.
      attention_metadata: Attention metadata for the decoding process.
      inputs_embeds: Optional precomputed input embeddings.
      _input_positions: Unused; positions are read from ``attention_metadata``.
      _layer_name_to_kvcache_index: ``layer.{i} -> index into kv_caches``, as a
        static tuple of pairs (or dict). MaxText decoders index caches by layer
        (``kv_caches[lyr]``), so the list is re-ordered by this map before the
        forward pass and scattered back afterwards. ``None`` keeps the
        positional interpretation.
      *args: Remaining positional runner arguments (unused).
      **kwargs: Arbitrary keyword arguments.

    Returns:
      A tuple containing:
        - updated_kv_caches: A list of updated KV caches.
        - hidden: The hidden states.
        - aux_hidden_states: A list of auxiliary hidden states.
        - expert_indices: A list of expert indices or None.

    Raises:
      ValueError: If the model is not an instance of `nnx.Module`.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model must be an instance of type nnx.Module.")

    # For hybrid models, attention_metadata can arrive as a GroupedAttentionMetadata
    # (a mapping from layer names like "layer.0" to per-group AttentionMetadata).
    # Keep the mapping so each decoder layer (GDN or full attention) receives its
    # own group's block tables and state indices.

    # Present the decoder a layer-ordered view of the physical cache list. With
    # the vLLM hybrid layout all Mamba/GDN caches precede the attention caches,
    # so kv_caches[lyr] would otherwise hand an attention layer a 3D conv state.
    if kv_caches is not None:
      physical_indices = resolve_layer_kv_cache_indices(_layer_name_to_kvcache_index, len(kv_caches))
      layer_kv_caches = gather_layer_kv_caches(kv_caches, physical_indices)
    else:
      physical_indices = None
      layer_kv_caches = None

    # MaxText decode treats vLLM's flattened tokens as a batch with seq_len=1.
    # MRoPE positions arrive channel-first and must also move their 3 channels
    # to MaxText's trailing dimension.
    if inputs_embeds is not None:
      decoder_input_embeddings = inputs_embeds[:, None, :].astype(self.maxtext_config.dtype)
      # tpu-inference passes input_ids=None with precomputed embeddings. The
      # decoder still requires a shape-compatible token array, but does not use
      # its values when decoder_input_embeddings is present.
      input_ids = jnp.zeros((inputs_embeds.shape[0], 1), dtype=jnp.int32)
    else:
      decoder_input_embeddings = None
      input_ids = jnp.expand_dims(input_ids, axis=1)

    if isinstance(attention_metadata, dict):
      first_meta = next(iter(attention_metadata.values()))
      positions = getattr(first_meta, "input_positions", None)
    else:
      positions = getattr(attention_metadata, "input_positions", None)
    if positions is None:
      positions = _input_positions
    input_positions = normalize_vllm_input_positions(positions)

    # Filter kwargs to only those accepted by self.model.
    model_kwargs = dict(kwargs)
    for extra_key in (
        "inputs_embeds",
        "input_positions",
        "layer_name_to_kvcache_index",
        "_layer_name_to_kv_cache",
        "shared_attention_metadata",
        "intermediate_tensors",
        "lora_metadata",
        "is_first_rank",
        "is_last_rank",
    ):
      model_kwargs.pop(extra_key, None)

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      expert_indices = None
      res = self.model(
          decoder_input_tokens=input_ids,
          decoder_input_embeddings=decoder_input_embeddings,
          decoder_positions=input_positions,
          kv_caches=layer_kv_caches,
          attention_metadata=attention_metadata,
          model_mode=self.model_mode,
          **model_kwargs,
      )

      if isinstance(res, tuple) and len(res) == 3:
        hidden, layer_kv_caches, expert_indices = res
        if expert_indices is not None and isinstance(self.mesh, Mesh):
          expert_indices = jax.lax.with_sharding_constraint(
              expert_indices, jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
          )
      else:
        hidden, layer_kv_caches = res

      # Hand the updated caches back in the runner's physical order.
      if kv_caches is not None:
        kv_caches = scatter_layer_kv_caches(kv_caches, layer_kv_caches, physical_indices)

      # To be compatible with vLLM, we reshape to (batch * seq, dim).
      hidden = hidden.reshape((-1, hidden.shape[-1]))
      if getattr(self.vllm_config, "speculative_config", None) is not None:
        spec_method = getattr(self.vllm_config.speculative_config, "method", None)
        decoder = getattr(self.model, "decoder", None)
        if spec_method == "mtp" and decoder is not None and hasattr(decoder, "decoder_norm"):
          normed_hidden = decoder.decoder_norm(hidden)
          aux_hidden_states = [normed_hidden.reshape((-1, normed_hidden.shape[-1]))]
        else:
          aux_hidden_states = [hidden]

    return kv_caches, hidden, aux_hidden_states, expert_indices

  def forward(self, *args, **kwargs):
    """Alias for __call__ for compatibility.

    Args:
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of the `__call__` method.
    """
    return self(*args, **kwargs)

  def get_input_embeddings(self) -> jax.Array:
    """Returns the input embeddings of the model.

    Returns:
      A JAX array representing the input embeddings.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder.embedding

  def embed_multimodal(self, **kwargs) -> list[jax.Array]:
    """Computes embeddings using the configured model family's handler."""
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    if self.multimodal_handler is None:
      raise ValueError(f"No vLLM multimodal handler is registered for {self.maxtext_config.model_name!r}.")
    return self.multimodal_handler.embed_multimodal(self.model, self.maxtext_config, self.mesh, **kwargs)

  def embed_input_ids(
      self,
      input_ids: jax.Array,
      multimodal_embeddings: jax.Array | None = None,
      is_multimodal: jax.Array | None = None,
  ) -> jax.Array:
    """Embeds the input token IDs using the model's token embedder.

    Args:
      input_ids: A JAX array of input token IDs.
      multimodal_embeddings: Optional modality embeddings to merge.
      is_multimodal: Optional mask identifying multimodal inputs.

    Returns:
      A JAX array of embedded input tokens.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      inputs_embeds = self.model.token_embedder(input_ids)

      if multimodal_embeddings is not None:
        if self.multimodal_handler is None:
          raise ValueError(f"No vLLM multimodal handler is registered for {self.maxtext_config.model_name!r}.")
        placeholder_ids = self.multimodal_handler.placeholder_token_ids(self.cfg.hf_config)

        inputs_embeds = merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            placeholder_ids,
        )
      return inputs_embeds

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Computes the logits from the hidden states using the underlying decoder model.

    Args:
      hidden_states: A JAX array of hidden states.

    Returns:
      A JAX array of logits.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      # Reshape to (num_tokens, 1, hidden_dim) for decoder output head
      y = jnp.expand_dims(hidden_states, axis=1)

      # Compute logits using the MaxText decoder's output head
      logits = self.model.decoder.apply_output_head(self.model.token_embedder, y, True, self.model_mode)

      # Reshape back to (num_tokens, vocab_size)
      return logits.squeeze(1)

  @property
  def mtp_layer(self) -> nnx.Module | None:
    """Returns the first Multi-Token Prediction layer from the underlying Transformer, if present."""
    mtp_block = getattr(self.model, "mtp_block", None)
    return getattr(mtp_block, "mtp_layer_1", None)

  def has_mtp(self) -> bool:
    """Returns True if an MTP draft layer is attached to this model."""
    return self.mtp_layer is not None

  # pylint: disable=keyword-arg-before-vararg
  def mtp_forward(
      self,
      kv_caches: list[jax.Array],
      input_ids: jax.Array,
      target_hidden_states: jax.Array | None = None,
      attention_metadata: AttentionMetadata | dict | None = None,
      _layer_name_to_kvcache_index: tuple | dict | None = None,
      spec_step_idx: int = 0,
      *args,
      **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array], None]:
    """Executes one Multi-Token Prediction (MTP) draft step using the target model's shared embedding."""
    del spec_step_idx, args
    if not isinstance(self.model, nnx.Module) or self.mtp_layer is None:
      raise ValueError("MTP layer is not initialized on MaxTextForCausalLM.")
    if target_hidden_states is None:
      target_hidden_states = kwargs.pop("target_hidden_states", kwargs.pop("hidden_states", None))
    if _layer_name_to_kvcache_index is None:
      _layer_name_to_kvcache_index = kwargs.pop("layer_name_to_kvcache_index", None)

    attn_meta = (
        attention_metadata.get("draft_layer.0") or next(iter(attention_metadata.values()))
        if isinstance(attention_metadata, dict)
        else attention_metadata
    )
    input_positions = normalize_vllm_input_positions(getattr(attn_meta, "input_positions", None))

    draft_kv_index = (
        dict(_layer_name_to_kvcache_index).get("draft_layer.0", len(kv_caches) - 1)
        if (kv_caches and _layer_name_to_kvcache_index is not None)
        else (len(kv_caches) - 1 if kv_caches else None)
    )
    draft_kv_cache = kv_caches[draft_kv_index] if draft_kv_index is not None else None

    input_ids_2d = jnp.expand_dims(input_ids, axis=1) if input_ids.ndim == 1 else input_ids
    prev_hidden_3d = (
        jnp.expand_dims(target_hidden_states, axis=1) if target_hidden_states.ndim == 2 else target_hidden_states
    )

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      target_embeds_3d = self.model.token_embedder(input_ids_2d.astype(jnp.int32), model_mode=self.model_mode)
      raw_hidden, updated_draft_kv = self.mtp_layer(
          prev_hidden_state=prev_hidden_3d.astype(self.maxtext_config.dtype),
          target_token_embedding=target_embeds_3d.astype(self.maxtext_config.dtype),
          position_ids=input_positions,
          decoder_segment_ids=None,
          deterministic=True,
          model_mode=self.model_mode,
          kv_cache=draft_kv_cache,
          attention_metadata=attn_meta,
          return_kv_cache=True,
      )
      normed_hidden = self.mtp_layer.final_norm(raw_hidden)
      normed_hidden = normed_hidden.reshape((-1, normed_hidden.shape[-1]))

    if kv_caches and draft_kv_index is not None and updated_draft_kv is not None:
      kv_caches = list(kv_caches)
      kv_caches[draft_kv_index] = updated_draft_kv

    return kv_caches, normed_hidden, [normed_hidden], None

  def compute_mtp_logits(self, hidden_states: jax.Array, lora_metadata=None) -> jax.Array:
    """Projects already-normalized MTP hidden states (`mtp.norm`) to vocab logits."""
    del lora_metadata
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")
    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      y = jnp.expand_dims(hidden_states, axis=1) if hidden_states.ndim == 2 else hidden_states
      logits = self.model.decoder.apply_output_head(
          self.model.token_embedder, y, True, self.model_mode, normalize_y=False, reduce_mhc=False
      )
      return logits.squeeze(1)

  def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
    """Identity pass-through for EAGLE3/MTP proposer compatibility."""
    return hidden_states

  def load_weights(self, rng_key: jax.Array) -> None:
    """Loads model weights using the underlying decoder model.

    Args:
      rng_key: A JAX random key for model initialization.
    """
    if self.model is not None:
      return

    with self.mesh, logical_axis_rules(self.maxtext_config.logical_axis_rules):
      model = model_creation_utils.from_pretrained(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      if getattr(model, "mtp_block", None) is not None and not getattr(model, "_mtp_block_restored", False):
        add_cfg = getattr(self.vllm_config, "additional_config", None) or {}
        mt_add_cfg = add_cfg.get("maxtext_config", {}) if isinstance(add_cfg, dict) else {}
        explicit_mtp_hf = mt_add_cfg.get("mtp_hf_checkpoint_path") if isinstance(mt_add_cfg, dict) else None
        model_path = getattr(self.cfg, "model", None)
        load_format = getattr(getattr(self.vllm_config, "load_config", None), "load_format", None)
        hf_ckpt_path = explicit_mtp_hf or (
            model_path
            if (
                getattr(model, "_mtp_block_restored", None) is False
                or (load_format == "dummy" and model_path and os.path.isdir(str(model_path)))
            )
            else None
        )
        if hf_ckpt_path:
          hf_cfg_obj = getattr(self.cfg, "hf_config", None)
          hf_cfg_dict = hf_cfg_obj.to_dict() if hasattr(hf_cfg_obj, "to_dict") else None
          model_creation_utils.load_mtp_block_from_hf(
              model, self.maxtext_config, self.mesh, str(hf_ckpt_path), hf_config_dict=hf_cfg_dict
          )
      if self.maxtext_config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, self.mesh, self.maxtext_config)
        if self.maxtext_config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, self.maxtext_config)
      self.model = nnx.data(model)
    patch_raiden_worker_h2d()

  def get_mrope_input_positions(
      self,
      input_tokens: list[int],
      mm_features: list = None,
  ) -> tuple[np.ndarray, int]:
    """Get dummy mrope input positions and delta value for text-only MaxText."""
    seq_len = len(input_tokens)
    # Use NumPy rather than JAX (jnp) to avoid triggering XLA compilations on every distinct seq_len.
    pos_range = np.arange(seq_len, dtype=np.int32)
    # M-RoPE expects 3D position vectors (3, seq_len) and position_delta (int)
    positions = np.stack([pos_range, pos_range, pos_range], axis=0)
    return positions, 0


# Monkey-patch KVCacheManager.get_kv_cache_spec to support GDN/Mamba layers in Pure JAX path.
def patch_kv_cache_manager():
  """Monkey-patches KVCacheManager to support hybrid Attention + GDN/Mamba models."""
  # pylint: disable=import-outside-toplevel,protected-access
  try:
    from tpu_inference.runner.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import MambaSpec
    import torch
  except ImportError as e:
    # Gracefully handle missing imports in standard JAX environments (e.g. unit tests on CPU)
    max_logging.log(f"Skipping KVCacheManager patch (tpu_inference or dependencies not installed): {e}")
    return

  try:
    original_get_kv_cache_spec = KVCacheManager.get_kv_cache_spec
  except AttributeError as e:
    # Raise a clear error if packages exist but patch target is missing (indicating API change or mismatch)
    raise RuntimeError(
        "Failed to apply KVCacheManager patch: KVCacheManager.get_kv_cache_spec not found. "
        "This usually indicates a vLLM / tpu-inference API change or version mismatch."
    ) from e

  def patched_get_kv_cache_spec(self):
    runner = self.runner
    if not hasattr(runner, "model"):
      return original_get_kv_cache_spec(self)

    model = runner.model
    if not hasattr(model, "maxtext_config"):
      return original_get_kv_cache_spec(self)

    cfg = model.maxtext_config
    decoder_block = getattr(cfg, "decoder_block", "")

    decoder_block_str = ""
    if isinstance(decoder_block, str):
      decoder_block_str = decoder_block
    elif hasattr(decoder_block, "value"):
      decoder_block_str = decoder_block.value

    if decoder_block_str in HYBRID_DECODER_BLOCKS:
      mamba_cache_mode = getattr(self.runner.cache_config, "mamba_cache_mode", "none")
      interval = cfg.inhomogeneous_layer_cycle_interval

      # Qwen GDN keeps its short convolution history in BF16, but recurrence is
      # accumulated and persisted in FP32. Declaring both caches as the model
      # dtype silently quantizes the recurrent state after every generated token.
      mamba_shapes, mamba_dtypes, unpadded_mamba_page_size = build_qwen_gdn_cache_layout(cfg, torch)

      # Calculate attn_page_size_bytes
      from tpu_inference.layers.common.sharding import ShardingAxisName
      from tpu_inference import utils as common_utils

      tp_axis_name = getattr(ShardingAxisName, "KV_HEAD", ShardingAxisName.ATTN_HEAD)
      model_cnt = common_utils.get_mesh_shape_product(self.runner.mesh, tp_axis_name)

      model_config = self.runner.model_config
      text_config = getattr(model_config, "hf_text_config", getattr(model_config, "hf_config", None))
      base_num_kv_heads = model_config.get_total_num_kv_heads()
      base_head_size = model_config.get_head_size()

      num_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or base_num_kv_heads
      head_size = getattr(text_config, "global_head_dim", None) or base_head_size

      num_kv_heads = common_utils.get_padded_num_heads(num_kv_heads, model_cnt)
      head_size = common_utils.get_padded_head_dim(head_size)

      from tpu_inference.runner.kv_cache import get_attention_page_size_bytes

      block_size = self.runner.cache_config.block_size

      attn_page_size_bytes = get_attention_page_size_bytes(
          self.runner.mesh, block_size, num_kv_heads, head_size, self.runner.kv_cache_dtype, False
      )

      # Calculate groups
      num_layers = cfg.base_num_decoder_layers
      num_base_attn = num_layers // interval
      num_mamba = num_layers - num_base_attn
      vllm_cfg = getattr(self.runner, "vllm_config", None)
      spec_cfg = getattr(vllm_cfg, "speculative_config", None) or getattr(self.runner, "speculative_config", None)
      spec_method = getattr(spec_cfg, "method", None) if spec_cfg is not None else None
      use_gemma4_mtp = bool(
          spec_cfg is not None
          and hasattr(spec_cfg, "use_gemma4_mtp")
          and spec_cfg.use_gemma4_mtp()
      )
      num_draft_attn = 1 if (spec_method in ("mtp", "eagle3", "dflash") and not use_gemma4_mtp) else 0
      num_attn = num_base_attn + num_draft_attn

      # To allocate memory uniformly for a hybrid model's KV/recurrent cache page table,
      # we group layers together. The uniform page size must support both attention and
      # mamba layers.
      # If the ratio of attention to mamba layers is relatively balanced (less than _HYBRID_LAYER_IMBALANCE_THRESHOLD),
      # we use the larger count as the group size to minimize the total number of groups.
      # If they are highly imbalanced (>= _HYBRID_LAYER_IMBALANCE_THRESHOLD), we group by the smaller count to prevent
      # the page size from being inflated by excessive padding for the minority layer type.
      min_count = min(num_attn, num_mamba)
      max_count = max(num_attn, num_mamba)
      if max_count < min_count * _HYBRID_LAYER_IMBALANCE_THRESHOLD:
        group_size = max_count
      else:
        group_size = min_count
      num_attn_groups = (num_attn + group_size - 1) // group_size
      num_mamba_groups = (num_mamba + group_size - 1) // group_size

      uniform_page_size_bytes = num_attn_groups * attn_page_size_bytes + num_mamba_groups * unpadded_mamba_page_size

      # Set the padded page size on manager and config
      self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
      self.runner.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)

      # Size the mamba/attention block pools. The signature of this
      # tpu-inference private helper has changed across versions: older
      # revisions also take the vLLM kv-cache group layout
      # (`num_attn_groups`, `num_mamba_groups`, `group_size`), newer ones
      # re-derive it internally and only take the page sizes and layer
      # counts. Pass by keyword and keep only the parameters the installed
      # version declares, so the adapter works with either revision instead
      # of dying with a TypeError before the KV cache spec is built.
      call_with_supported_kwargs(
          self._maybe_set_compact_mamba_num_blocks_override,
          attn_page_size_bytes=attn_page_size_bytes,
          unpadded_mamba_page_size_bytes=int(unpadded_mamba_page_size),
          num_attn_groups=num_attn_groups,
          num_mamba_groups=num_mamba_groups,
          num_attn_layers=num_attn,
          num_mamba_layers=num_mamba,
          group_size=group_size,
      )

    kv_cache_spec = original_get_kv_cache_spec(self)

    if decoder_block_str in HYBRID_DECODER_BLOCKS:
      for i in range(cfg.base_num_decoder_layers):
        layer_name = f"layer.{i}"
        if (i + 1) % interval != 0:
          if layer_name in kv_cache_spec:
            kv_cache_spec[layer_name] = MambaSpec(
                block_size=kv_cache_spec[layer_name].block_size,
                shapes=mamba_shapes,
                dtypes=mamba_dtypes,
                page_size_padded=self._hybrid_uniform_page_size_bytes,
                mamba_cache_mode=mamba_cache_mode,
            )
        elif layer_name in kv_cache_spec:
          kv_cache_spec[layer_name] = self._create_attention_spec(
              kv_cache_spec[layer_name].block_size,
              num_kv_heads,
              head_size,
              sliding_window=getattr(kv_cache_spec[layer_name], "sliding_window", None),
          )
      if "draft_layer.0" in kv_cache_spec:
        kv_cache_spec["draft_layer.0"] = self._create_attention_spec(
            kv_cache_spec["draft_layer.0"].block_size,
            num_kv_heads,
            head_size,
            sliding_window=getattr(kv_cache_spec["draft_layer.0"], "sliding_window", None),
        )

    return kv_cache_spec

  KVCacheManager.get_kv_cache_spec = patched_get_kv_cache_spec
  max_logging.log("Successfully applied KVCacheManager patch for hybrid GDN models.")


def patch_raiden_worker_h2d():
  """Monkey-patches TPUWorker.raiden_h2d and RaidenWorkerSync to apply Raiden weights to runner."""
  try:
    from tunix.experimental.weight_sync.raiden_synchronizer import patch_raiden_worker_sync  # pylint: disable=import-outside-toplevel

    patch_raiden_worker_sync()
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Skipping raiden worker sync patch: {e}")
