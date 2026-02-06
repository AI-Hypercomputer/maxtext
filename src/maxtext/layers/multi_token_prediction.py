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

"""JAX implementation of the Multi Token Prediction https://arxiv.org/pdf/2412.19437 """

from typing import Type

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config, MODEL_MODE_TRAIN
from MaxText.globals import EPS
from maxtext.layers import nnx_wrappers
from maxtext.layers.decoders import DecoderLayer
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding


# Custom Variable types for MTP intermediate outputs
# These will be automatically converted to Linen mutable collections by ToLinen wrapper
# The class names become collection names directly (no case conversion)
class mtp_losses(nnx.Variable):  # pylint: disable=invalid-name
  """Variable type for storing MTP loss components -> 'mtp_losses' collection."""


class mtp_acceptance(nnx.Variable):  # pylint: disable=invalid-name
  """Variable type for storing MTP acceptance predictions -> 'mtp_acceptance' collection."""


def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """Performs a leftward roll on sequence axis and masks invalid positions.

  Args:
    x: Input array of shape [batch, seq_len, ...].
    shift: Number of positions to shift left.

  Returns:
    Rolled array with masked positions set to zero.
  """
  if shift == 0:
    return x
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


class MultiTokenPredictionLayer(nnx.Module):
  """Multi-Token Prediction layer: normalize, concatenate, project, and transform.

  Implements: h_next = TransformerLayer(W_p(concat(RMSNorm(h_prev), RMSNorm(e_target))))
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      layer_number: int,
      transformer_layer_module: Type[DecoderLayer],
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.layer_number = layer_number
    self.transformer_layer_module = transformer_layer_module
    self.rngs = rngs
    k = layer_number
    cfg = self.config

    self.embedding_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.hidden_state_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.projection_layer = DenseGeneral(
        in_features_shape=2 * cfg.emb_dim,
        out_features_shape=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        use_bias=False,
        kernel_axes=("concat_embed", "embed"),
        rngs=rngs,
    )
    # Use MODEL_MODE_TRAIN for initialization; runtime model_mode is passed dynamically.
    if cfg.pure_nnx_decoder:
      mtp_transformer_layer = transformer_layer_module(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          name=f"mtp_{k}_transformer_layer",
          rngs=rngs,
      )
    else:
      mtp_transformer_layer = transformer_layer_module(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          name=f"mtp_{k}_transformer_layer",
      )

    self.transformer_layer = nnx_wrappers.ToNNX(mtp_transformer_layer, rngs=rngs)

    # ToNNX requires explicit initialization with sample inputs for proper parameter setup.
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config=cfg, model_mode=MODEL_MODE_TRAIN)
    self.transformer_layer.lazy_init(
        inputs=jnp.zeros((batch_size, seq_len, self.config.emb_dim), dtype=self.config.dtype),
        decoder_segment_ids=None,
        decoder_positions=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

  @property
  def embedding_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_embedding_norm")

  @embedding_norm.setter
  def embedding_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_embedding_norm", module)

  @property
  def hidden_state_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_hidden_state_norm")

  @hidden_state_norm.setter
  def hidden_state_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_hidden_state_norm", module)

  @property
  def projection_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_projection")

  @projection_layer.setter
  def projection_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_projection", module)

  @property
  def transformer_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_transformer_layer")

  @transformer_layer.setter
  def transformer_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_transformer_layer", module)

  def __call__(
      self,
      prev_hidden_state: jnp.ndarray,
      target_token_embedding: jnp.ndarray,
      *,
      position_ids: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    """Applies MTP combination, projection, and transformer processing.

    Args:
        prev_hidden_state: Shape [batch, seq_len, hidden_size].
        target_token_embedding: Embedding for token t+k. Shape [batch, seq_len, embed_dim].
        position_ids: Shape [batch, seq_len].
        decoder_segment_ids: Shape [batch, seq_len] or None.
        deterministic: Whether to disable dropout.
        model_mode: Operational mode (train, eval, decode).

    Returns:
        Processed hidden state. Shape [batch, seq_len, hidden_size].
    """
    target_token_embedding = sharding.maybe_shard_with_logical(
        target_token_embedding,
        ("activation_batch", "activation_length", "activation_embed"),
        self.mesh,
        self.config.shard_mode,
        self.config.logical_axis_rules,
    )

    embedding_norm = self.embedding_norm(target_token_embedding)
    hidden_state_norm = self.hidden_state_norm(prev_hidden_state)
    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)
    projected_features = self.projection_layer(concatenated_features)

    output = self.transformer_layer(
        inputs=projected_features,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=position_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    return output[0] if isinstance(output, tuple) else output


class MultiTokenPredictionBlock(nnx.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      transformer_layer_module: Type[DecoderLayer],
      decoder: nnx.Module,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.transformer_layer_module = transformer_layer_module
    self.decoder = decoder
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    # 1-indexed to match paper convention.
    for k in range(1, config.mtp_num_layers + 1):
      layer = MultiTokenPredictionLayer(
          config=config,
          mesh=mesh,
          layer_number=k,
          transformer_layer_module=transformer_layer_module,
          rngs=rngs.fork(),
      )
      setattr(self, f"mtp_layer_{k}", layer)

  def __call__(
      self,
      shared_embedding,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      *,
      position_ids,
      decoder_segment_ids,
      model_mode,
      deterministic,
  ) -> dict:
    cfg = self.config
    mtp_hidden_state = main_hidden_state

    # Rolling variables move prediction window one token to the right per iteration.
    rolled_input_ids = input_ids
    rolled_target_ids = target_ids
    rolled_target_mask = target_mask
    rolled_position_id = position_ids

    mtp_losses_list = []
    mtp_weights_list = []
    mtp_preds_list = []
    mtp_masks_list = []

    for k in range(1, cfg.mtp_num_layers + 1):
      rolled_input_ids = roll_and_mask(rolled_input_ids)
      rolled_target_ids = roll_and_mask(rolled_target_ids)
      rolled_target_mask = roll_and_mask(rolled_target_mask)
      rolled_position_id = roll_and_mask(rolled_position_id)

      target_token_embedding = self.decoder._apply_embedding(
          shared_embedding,
          rolled_input_ids,
          rolled_position_id,
          deterministic,
          model_mode=self.decoder.model_mode,
      )

      mtp_layer = getattr(self, f"mtp_layer_{k}")
      mtp_hidden_state = mtp_layer(
          prev_hidden_state=mtp_hidden_state,
          target_token_embedding=target_token_embedding,
          position_ids=position_ids,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=self.decoder.model_mode,
      )

      mtp_logits = self.decoder.apply_output_head(shared_embedding, mtp_hidden_state, deterministic, model_mode)

      mtp_xent, _ = max_utils.cross_entropy_with_logits(
          mtp_logits, jax.nn.one_hot(rolled_target_ids, cfg.vocab_size), 0.0
      )
      mtp_xent_masked = mtp_xent * rolled_target_mask

      if model_mode == MODEL_MODE_TRAIN:
        mtp_losses_list.append(jnp.sum(mtp_xent_masked))
        mtp_weights_list.append(jnp.sum(rolled_target_mask).astype(jnp.float32))

      if cfg.mtp_eval_target_module == k:
        # Float32 to avoid gradient errors; converted back to int32 in acceptance calculation.
        mtp_preds_list.append(jnp.argmax(mtp_logits, axis=-1).astype(jnp.float32))
        mtp_masks_list.append(rolled_target_mask)

    if mtp_losses_list:
      # Not part of checkpoints, don't declare in __init__
      self.losses = mtp_losses(jnp.stack(mtp_losses_list))
      self.weights = mtp_losses(jnp.stack(mtp_weights_list))
    if mtp_preds_list:
      # Not part of checkpoints, don't declare in __init__
      self.mtp_preds = mtp_acceptance(jnp.stack(mtp_preds_list))
      self.mtp_mask = mtp_acceptance(jnp.stack(mtp_masks_list))

    return {}


def calculate_mtp_loss(intermediate_outputs, config):
  """Calculates Multi-Token Prediction loss from intermediate outputs."""
  mtp_losses_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "losses"), default=None
  )
  mtp_weights_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "weights"), default=None
  )

  if mtp_losses_data is None:
    return 0.0

  # Handle both tuple (Linen sow) and array (NNX Variable) formats.
  if isinstance(mtp_losses_data, (tuple, list)):
    if not mtp_losses_data:
      return 0.0
    mtp_losses_array = jnp.array(mtp_losses_data)
    mtp_weights_array = jnp.array(mtp_weights_data)
  else:
    if mtp_losses_data.size == 0:
      return 0.0
    mtp_losses_array = mtp_losses_data
    mtp_weights_array = mtp_weights_data

  avg_mtp_loss = jnp.sum(mtp_losses_array) / (jnp.sum(mtp_weights_array) + EPS)
  return avg_mtp_loss * config.mtp_loss_scaling_factor


def calculate_mtp_acceptance_rate(intermediate_outputs, config):
  """Calculates MTP acceptance rate from intermediate outputs."""
  sown_data = maxtext_utils.get_nested_value(intermediate_outputs, ("mtp_acceptance", "mtp_block"), {})

  # Handle both tuple (Linen sow) and array (NNX Variable) formats.
  mtp_preds_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_preds",), None)
  valid_mask_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_mask",), None)

  mtp_preds = mtp_preds_raw[0] if isinstance(mtp_preds_raw, (tuple, list)) and mtp_preds_raw else mtp_preds_raw
  valid_mask = valid_mask_raw[0] if isinstance(valid_mask_raw, (tuple, list)) and valid_mask_raw else valid_mask_raw

  # Only populated during eval for the target MTP module.
  if mtp_preds is None or valid_mask is None:
    return 0.0

  mtp_preds = mtp_preds.astype(jnp.int32)
  main_model_preds = jnp.argmax(intermediate_outputs["logits"], axis=-1)

  # Align main model predictions with MTP head target by rolling k steps.
  rolled_main_preds = main_model_preds
  for _ in range(config.mtp_eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)

  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  return (correct_predictions / (total_valid_tokens + EPS)) * 100


def multi_token_prediction_block_as_linen(
    *,
    config: Config,
    mesh: Mesh,
    transformer_layer_module: Type[DecoderLayer],
    decoder: nnx.Module,
    rngs: nnx.Rngs,
    name: str | None = None,
) -> nn.Module:
  """Initializes MultiTokenPredictionBlock as a Linen module.

  Args:
    config: Configuration object containing model hyperparameters.
    mesh: JAX Mesh for model parallelism.
    transformer_layer_module: The Transformer Decoder Layer class to use.
    decoder: The decoder module that provides embedding and output head.
    rngs: Random number generators for initialization.
    name: Optional name for the module.

  Returns:
    An instance of MultiTokenPredictionBlock wrapped as a Linen module.
  """
  return nnx.bridge.to_linen(
      MultiTokenPredictionBlock,
      config=config,
      mesh=mesh,
      transformer_layer_module=transformer_layer_module,
      decoder=decoder,
      rngs=rngs,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )
