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

import jax
import jax.numpy as jnp
import os

from flax import nnx
import flax.linen as nn
from jax.sharding import Mesh
from MaxText import model_creation_utils
from MaxText import max_logging
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE
from MaxText.globals import MAXTEXT_PKG_DIR

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from vllm.config import VllmConfig


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

  if vllm_config.model_config.hf_config_path is None:
    raise ValueError("hf_config_path must be provided when using MaxTextForCausalLM.")

  # Add base config path to positional args
  base_config_path = os.path.join(MAXTEXT_PKG_DIR, "configs", "vllm.yml")
  argv_list = ["", str(base_config_path)]

  maxtext_config = pyconfig.initialize(argv_list, **overrides)
  return maxtext_config


class MaxTextDecoderModel(nnx.Module):
  """A vLLM-compatible decoder model wrapper for MaxText.

  This class adapts a MaxText model for use within the vLLM framework,
  handling configuration generation, model initialization, and execution
  of the decoding step.
  """

  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh) -> None:
    """Initializes the MaxTextDecoderModel.

    Args:
      vllm_config: The vLLM configuration object.
      rng_key: A JAX random key for model initialization.
      mesh: The JAX mesh device for model sharding.
    """
    self.vllm_config = vllm_config
    self.maxtext_config = generate_maxtext_config(vllm_config)

    # Model configuration
    self.mesh = mesh
    self.model_mode = MODEL_MODE_AUTOREGRESSIVE

    # Model creation
    self.model: nnx.Module | None = None
    self.logits: jax.Array | None = None

    # Handle dummy weight loading during initialization
    if vllm_config.load_config.load_format == "dummy":
      with self.mesh:
        self.load_weights(rng_key)

    elif self.maxtext_config.load_parameters_path is None:
      max_logging.log("Warning: No load_parameters_path provided. The model will be initialized with random weights.")

  def __call__(
      self,
      kv_caches: list[jax.Array],
      input_ids: jax.Array,
      attention_metadata: AttentionMetadata,
      *args,
      **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array]]:
    """Performs a forward pass through the decoder model.

    Args:
      kv_caches: A list of JAX arrays representing the KV caches.
      input_ids: A JAX array of input token IDs.
      attention_metadata: Attention metadata for the decoding process.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      A tuple containing:
        - updated_kv_caches: A list of updated KV caches.
        - hidden: The hidden states (Q, d_model).
        - aux_hidden_states: A list of auxiliary hidden states.

    Raises:
      ValueError: If the model is not an instance of `nnx.Module`.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model must be an instance of type nnx.Module.")

    if input_ids.ndim < 2:
      input_ids = jnp.expand_dims(input_ids, axis=0)

    input_positions = attention_metadata.input_positions
    if input_positions.ndim < 2:
      input_positions = jnp.expand_dims(input_positions, axis=0)

    with nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      logits, hidden, kv_caches = self.model(
          decoder_input_tokens=input_ids,
          decoder_positions=input_positions,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          model_mode=self.model_mode,
          **kwargs,
      )

    if hidden.ndim > 1:
      hidden = jnp.squeeze(hidden, axis=0)
      logits = jnp.squeeze(logits, axis=0)

    self.logits = nnx.data(logits)  # cache logits for compute_logits call

    return kv_caches, hidden, aux_hidden_states

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Computes the logits from the hidden states.

    Args:
      hidden_states: A JAX array of hidden states.

    Returns:
      A JAX array of logits (Q, vocab_size).
    """
    if self.logits is not None:
      return self.logits

    with nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      embeddings = self.model.token_embedder
      return self.model.decoder.apply_output_head(embeddings, hidden_states, True, self.model_mode)

  def load_weights(self, rng_key: jax.Array) -> None:
    """Loads model parameters on the provided mesh.

    Args:
      rng_key: A JAX random key for model initialization.
    """
    if self.model is not None:
      return

    with nn.logical_axis_rules(""):
      model, _ = model_creation_utils.create_nnx_model(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      self.model = nnx.data(model)


class MaxTextForCausalLM(nnx.Module):
  """A vLLM-compatible causal language model wrapper for MaxText.

  This class serves as the primary interface for integrating MaxText models
  into the vLLM serving framework, specifically for causal language modeling
  tasks. It wraps the `MaxTextDecoderModel` and exposes methods expected
  by vLLM.
  """

  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
    """Initializes the MaxTextForCausalLM model.

    Args:
      vllm_config: The vLLM configuration object.
      rng_key: A JAX random key for model initialization.
      mesh: The JAX mesh device for model sharding.
    """
    self.cfg = vllm_config.model_config
    self.mesh = mesh
    self.model = MaxTextDecoderModel(vllm_config, rng_key, mesh)
    self.is_text_generation_model = True

  def __call__(
      self, kv_caches: list[jax.Array], input_ids: jax.Array, attention_metadata: AttentionMetadata, *args, **kwargs
  ) -> tuple[list[jax.Array], jax.Array]:
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
    """
    with self.mesh:
      kv_caches, hidden, aux_hidden_states = self.model(kv_caches, input_ids, attention_metadata, *args, **kwargs)
    return kv_caches, hidden, aux_hidden_states

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
    with self.mesh:
      return self.model.model.token_embedder.embedding

  def embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
    """Embeds the input token IDs using the model's token embedder.

    Args:
      input_ids: A JAX array of input token IDs.

    Returns:
      A JAX array of embedded input tokens.
    """
    with self.mesh:
      return self.model.model.token_embedder(input_ids)

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Computes the logits from the hidden states using the underlying decoder model.

    Args:
      hidden_states: A JAX array of hidden states.

    Returns:
      A JAX array of logits.
    """
    with self.mesh:
      return self.model.compute_logits(hidden_states)

  def load_weights(self, rng_key: jax.Array) -> None:
    """Loads model weights using the underlying decoder model.

    Args:
      rng_key: A JAX random key for model initialization.
    """
    with self.mesh:
      self.model.load_weights(rng_key)
