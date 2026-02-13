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
import jax

from flax import nnx
import flax.linen as nn
from jax import numpy as jnp
from jax.sharding import Mesh
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE
from MaxText.globals import MAXTEXT_CONFIGS_DIR
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils


try:
  from tpu_inference.layers.common.attention_metadata import AttentionMetadata
except ImportError:
  # Mock for documentation build or environments without tpu_inference
  class AttentionMetadata:
    input_positions: jax.Array


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
  base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  argv_list = ["", str(base_config_path)]

  maxtext_config = pyconfig.initialize(argv_list, **overrides)
  return maxtext_config


class MaxTextForCausalLM(nnx.Module):
  """A vLLM-compatible causal language model wrapper for MaxText.

  This class serves as the primary interface for integrating MaxText models
  into the vLLM serving framework, specifically for causal language modeling
  tasks. It handles configuration generation, model initialization, and execution
  of the decoding step.
  """

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

    # Handle dummy weight loading during initialization
    if vllm_config.load_config.load_format == "dummy":
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

    Raises:
      ValueError: If the model is not an instance of `nnx.Module`.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model must be an instance of type nnx.Module.")

    # Ensure inputs are at least 2D with a batch dimension
    input_ids = jnp.atleast_2d(input_ids)
    input_positions = jnp.atleast_2d(attention_metadata.input_positions)

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
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
      y = hidden_states[:, jnp.newaxis, :]

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

    with self.mesh, nn.logical_axis_rules(""):
      model, _ = model_creation_utils.create_nnx_model(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      self.model = nnx.data(model)
