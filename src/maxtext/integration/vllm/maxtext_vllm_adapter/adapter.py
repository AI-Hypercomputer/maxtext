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

import os
from flax import nnx
import flax.linen as nn
import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.hybrid_cache_utils import build_qwen_gdn_cache_layout, normalize_vllm_input_positions
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


try:
  from tpu_inference.layers.common.attention_metadata import AttentionMetadata
except ImportError:
  # Mock for documentation build or environments without tpu_inference
  class AttentionMetadata:
    input_positions: jax.Array


from vllm.config import VllmConfig

# Threshold to determine if the ratio of attention to mamba layers is highly imbalanced.
# If max_count / min_count >= this threshold, we group KV cache allocations by the
# smaller count to prevent excessive memory padding for the minority layer type.
_HYBRID_LAYER_IMBALANCE_THRESHOLD = 1.5


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
    overrides = vllm_config.additional_config["maxtext_config"]
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
  num_lanes = pltpu.get_tpu_info().num_lanes
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
  if hidden_size is not None and (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
    padded_hidden_size = next_power_of_two(hidden_size)
    while (padded_hidden_size // moe_mlp_tp_size) < (2 * num_lanes):
      padded_hidden_size = next_power_of_two(padded_hidden_size + 1)

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

  def __call__(
      self,
      kv_caches: list[jax.Array],
      input_ids: jax.Array,
      attention_metadata: AttentionMetadata,
      *args,
      **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array], list[jax.Array] | None]:
    """Performs a forward pass through the causal language model.

    Args:
      kv_caches: A list of JAX arrays representing the KV caches.
      input_ids: A JAX array of input token IDs.
      attention_metadata: Attention metadata for the decoding process.
      *args: Variable length argument list.
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

    # below, GDN layers don't touch block_tables — they index via
    # ``mamba_state_indices`` — and all full-attn layers belong to the same
    # kv_cache_group so they share one block_tables. Pick a metadata from a
    # full-attn (non-linear_attention) layer when possible; otherwise any
    # value works.
    if isinstance(attention_metadata, dict):
      hf_text_config = getattr(self.cfg, "hf_text_config", getattr(self.cfg, "hf_config", None))
      layer_types = getattr(hf_text_config, "layer_types", None) or []
      attention_metadata_picked = None
      for i, lt in enumerate(layer_types):
        if lt != "linear_attention":
          attention_metadata_picked = attention_metadata.get(f"layer.{i}")
          if attention_metadata_picked is not None:
            break
      if attention_metadata_picked is None:
        attention_metadata_picked = next(iter(attention_metadata.values()))
      attention_metadata = attention_metadata_picked

    # MaxText decode treats vLLM's flattened tokens as a batch with seq_len=1.
    # MRoPE positions arrive channel-first and must also move their 3 channels
    # to MaxText's trailing dimension.
    input_ids = jnp.expand_dims(input_ids, axis=1)
    input_positions = normalize_vllm_input_positions(attention_metadata.input_positions)

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      expert_indices = None
      res = self.model(
          decoder_input_tokens=input_ids,
          decoder_positions=input_positions,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          model_mode=self.model_mode,
          **kwargs,
      )

      if isinstance(res, tuple) and len(res) == 3:
        hidden, kv_caches, expert_indices = res
      else:
        hidden, kv_caches = res

      # To be compatible with vLLM, we reshape to (batch * seq, dim).
      hidden = hidden.reshape((-1, hidden.shape[-1]))

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

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder.embedding

  def embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
    """Embeds the input token IDs using the model's token embedder.

    Args:
      input_ids: A JAX array of input token IDs.

    Returns:
      A JAX array of embedded input tokens.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder(input_ids)

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Computes the logits from the hidden states using the underlying decoder model.

    Args:
      hidden_states: A JAX array of hidden states.

    Returns:
      A JAX array of logits.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      # Reshape to (num_tokens, 1, hidden_dim) for decoder output head
      y = jnp.expand_dims(hidden_states, axis=1)

      # Compute logits using the MaxText decoder's output head
      logits = self.model.decoder.apply_output_head(self.model.token_embedder, y, True, self.model_mode)

      # Reshape back to (num_tokens, vocab_size)
      return logits.squeeze(1)

  def load_weights(self, rng_key: jax.Array) -> None:
    """Loads model weights using the underlying decoder model.

    Args:
      rng_key: A JAX random key for model initialization.
    """
    if self.model is not None:
      return

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      model = model_creation_utils.from_pretrained(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      if self.maxtext_config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, self.mesh, self.maxtext_config)
        if self.maxtext_config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, self.maxtext_config)
      self.model = nnx.data(model)

  def get_mrope_input_positions(
      self,
      input_tokens: list[int],
      mm_features: list = None,
  ) -> tuple[jax.Array, int]:
    """Get dummy mrope input positions and delta value for text-only MaxText."""
    seq_len = len(input_tokens)
    pos_range = jnp.arange(seq_len, dtype=jnp.int32)
    # M-RoPE expects 3D position vectors (3, seq_len) and position_delta (int)
    positions = jnp.stack([pos_range, pos_range, pos_range], axis=0)
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

    if decoder_block_str in ("qwen3_next", "qwen3_5"):
      interval = cfg.inhomogeneous_layer_cycle_interval

      # Qwen GDN keeps its short convolution history in BF16, but recurrence is
      # accumulated and persisted in FP32. Declaring both caches as the model
      # dtype silently quantizes the recurrent state after every generated token.
      mamba_shapes, mamba_dtypes, unpadded_mamba_page_size = build_qwen_gdn_cache_layout(cfg, torch)

      # Calculate attn_page_size_bytes
      from tpu_inference.layers.common.sharding import ShardingAxisName
      from tpu_inference import utils as common_utils

      tp_axis_name = ShardingAxisName.ATTN_HEAD
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
      num_attn = num_layers // interval
      num_mamba = num_layers - num_attn

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

      self._maybe_set_compact_mamba_num_blocks_override(
          attn_page_size_bytes,
          int(unpadded_mamba_page_size),
          num_attn_groups,
          num_mamba_groups,
          num_attn,
          num_mamba,
          group_size,
      )

    kv_cache_spec = original_get_kv_cache_spec(self)

    if decoder_block_str in ("qwen3_next", "qwen3_5"):
      for i in range(cfg.base_num_decoder_layers):
        if (i + 1) % interval != 0:
          layer_name = f"layer.{i}"
          if layer_name in kv_cache_spec:
            kv_cache_spec[layer_name] = MambaSpec(
                block_size=kv_cache_spec[layer_name].block_size,
                shapes=mamba_shapes,
                dtypes=mamba_dtypes,
                page_size_padded=self._hybrid_uniform_page_size_bytes,
            )

    return kv_cache_spec

  KVCacheManager.get_kv_cache_spec = patched_get_kv_cache_spec
  max_logging.log("Successfully applied KVCacheManager patch for hybrid GDN models.")
