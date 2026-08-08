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
import torch
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.layers.embeddings import Embed

# Class-level patch for Embed to support dynamic inputs_embeds override
original_embed_call = Embed.__call__


def patched_embed_call(self, x, model_mode=None):
  """Uses active input embeddings when the vLLM multimodal path enables them."""
  if hasattr(self, "use_inputs_embeds") and x.ndim == 2:

    def true_fn():
      return jnp.broadcast_to(self.active_inputs_embeds[...], (x.shape[0], 1, self.active_inputs_embeds.shape[-1]))

    def false_fn():
      return original_embed_call(self, x, model_mode=model_mode)

    return jax.lax.cond(self.use_inputs_embeds[...], true_fn, false_fn)
  return original_embed_call(self, x, model_mode=model_mode)


Embed.__call__ = patched_embed_call


try:
  from tpu_inference.layers.common.attention_metadata import AttentionMetadata
except ImportError:
  # Mock for documentation build or environments without tpu_inference
  class AttentionMetadata:
    input_positions: jax.Array


try:
  from tpu_inference.models.jax.utils.multi_modal_utils import merge_multimodal_embeddings
except ImportError:
  merge_multimodal_embeddings = None


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

    max_logging.log(
        f"Padding moe_intermediate_size from {hidden_size} to {padded_hidden_size} to match MLP MoE requirements."
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
  supports_multimodal: bool = False

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

    # Extract inputs_embeds if passed as positional arg (index 3 in model_fn, so args[0] here)
    inputs_embeds = args[0] if len(args) > 0 else None

    # Handle dummy inputs for multimodal path when inputs_embeds is provided
    if input_ids is None and inputs_embeds is not None:
      num_tokens = inputs_embeds.shape[0]
      # Construct dummy input_ids of shape (num_tokens, 1)
      input_ids = jnp.zeros((num_tokens, 1), dtype=jnp.int32)

      # Reshape inputs_embeds to (num_tokens, 1, dim) and cast to model dtype
      inputs_embeds_3d = jnp.expand_dims(inputs_embeds, axis=1).astype(self.maxtext_config.dtype)

      if hasattr(self.model, "token_embedder"):
        # Recreate the variable to change its shape dynamically
        self.model.token_embedder.active_inputs_embeds = nnx.Variable(inputs_embeds_3d)
        self.model.token_embedder.use_inputs_embeds[...] = True
    else:
      # Ensure inputs are at least 2D with a batch dimension
      input_ids = jnp.expand_dims(input_ids, axis=1)
      if hasattr(self.model, "token_embedder") and hasattr(self.model.token_embedder, "use_inputs_embeds"):
        self.model.token_embedder.use_inputs_embeds[...] = False
        # Reset shape to (1, 1, dim) to avoid broadcasting errors in JIT tracing of unused branch
        self.model.token_embedder.active_inputs_embeds = nnx.Variable(
            jnp.zeros((1, 1, self.maxtext_config.emb_dim), dtype=self.maxtext_config.dtype)
        )

    input_positions = jnp.expand_dims(attention_metadata.input_positions, axis=-1)

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      expert_indices = None
      hidden, kv_caches = self.model(
          decoder_input_tokens=input_ids,
          decoder_positions=input_positions,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          model_mode=self.model_mode,
          **kwargs,
      )

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

  def embed_multimodal(self, **kwargs) -> list[jax.Array]:
    """Computes embeddings for multimodal inputs (images)."""
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    pixel_values = kwargs.get("pixel_values", None)
    image_grid_thw = kwargs.get("image_grid_thw", None)

    if pixel_values is None or image_grid_thw is None:
      return []

    if isinstance(pixel_values, torch.Tensor):
      if pixel_values.dtype == torch.bfloat16:
        pixel_values = pixel_values.to(torch.float32).numpy().astype(jnp.bfloat16)
      else:
        pixel_values = pixel_values.numpy()
      pixel_values = jnp.asarray(pixel_values)

    if isinstance(image_grid_thw, torch.Tensor):
      image_grid_thw = jnp.asarray(image_grid_thw.numpy())

    image_grid_thw = image_grid_thw.reshape(-1, 3)

    multimodal_embeddings = []
    current_idx = 0

    patch_size = self.maxtext_config.patch_size_for_vit
    temp_patch = self.maxtext_config.temporal_patch_size_for_vit
    channel = self.maxtext_config.num_channels_for_vit

    for image_thw in image_grid_thw:
      grid_t, grid_h, grid_w = image_thw
      image_size = grid_t * grid_h * grid_w
      end_idx = current_idx + image_size

      single_pixel_values = pixel_values[current_idx:end_idx, :]

      # Reconstruct 5D shape: (1, channel, T, H, W) using direct reshape matching MaxText preprocessor
      x_5d = single_pixel_values.reshape(1, channel, grid_t * temp_patch, grid_h * patch_size, grid_w * patch_size)

      with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
        img_embeds, _ = self.model.vision_encoder(input_images=x_5d, deterministic=True)
        img_embeds = img_embeds.squeeze(0)
        multimodal_embeddings.append(img_embeds)

      current_idx = end_idx

    return multimodal_embeddings

  def embed_input_ids(
      self,
      input_ids: jax.Array,
      multimodal_embeddings: jax.Array | None = None,
      is_multimodal: jax.Array | None = None,
  ) -> jax.Array:
    """Embeds the input token IDs and merges multimodal embeddings if present."""
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      inputs_embeds = self.model.token_embedder(input_ids)

      if multimodal_embeddings is not None:
        if merge_multimodal_embeddings is None:
          raise ImportError("tpu_inference multimodal utilities are required to merge multimodal embeddings.")

        placeholder_ids = []
        if hasattr(self.cfg.hf_config, "image_token_id"):
          placeholder_ids.append(self.cfg.hf_config.image_token_id)
        if hasattr(self.cfg.hf_config, "video_token_id"):
          placeholder_ids.append(self.cfg.hf_config.video_token_id)

        if not placeholder_ids:
          max_logging.log(
              "Warning: No image_token_id or video_token_id found in hf_config. Cannot merge multimodal embeddings."
          )
          return inputs_embeds

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
      if hasattr(self.model, "token_embedder"):
        self.model.token_embedder.active_inputs_embeds = nnx.Variable(
            jnp.zeros((1, 1, self.maxtext_config.emb_dim), dtype=self.maxtext_config.dtype)
        )
        self.model.token_embedder.use_inputs_embeds = nnx.Variable(jnp.array(False))

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
    import numpy as np
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

      num_v_heads = cfg.gdn_num_value_heads
      num_k_heads = cfg.gdn_num_key_heads
      head_k_dim = cfg.gdn_key_head_dim
      head_v_dim = cfg.gdn_value_head_dim
      conv_kernel_size = cfg.gdn_conv_kernel_dim

      key_dim = head_k_dim * num_k_heads
      value_dim = head_v_dim * num_v_heads
      conv_dim = key_dim * 2 + value_dim

      conv_state_shape = (conv_kernel_size - 1, conv_dim)
      recurrent_state_shape = (num_v_heads, head_k_dim, head_v_dim)

      mamba_shapes = (conv_state_shape, recurrent_state_shape)

      torch_dtype = torch.bfloat16
      if str(cfg.dtype) == "float32":
        torch_dtype = torch.float32
      elif str(cfg.dtype) == "float16":
        torch_dtype = torch.float16
      mamba_dtypes = (torch_dtype, torch_dtype)

      # Calculate unpadded mamba page size
      dtype_size = 4 if torch_dtype == torch.float32 else 2
      unpadded_mamba_page_size = sum(int(np.prod(shape)) * dtype_size for shape in mamba_shapes)

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
