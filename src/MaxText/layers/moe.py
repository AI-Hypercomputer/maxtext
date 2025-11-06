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


"""MoE related Layers."""

import enum
import functools
import math
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
from flax import nnx
import flax.linen as nn
import jax
from jax import ad_checkpoint as adc
from jax.experimental import xla_metadata
import jax.numpy as jnp
from MaxText import common_types as ctypes
from MaxText import max_logging
from MaxText import max_utils
from MaxText.kernels import megablox as mblx
from MaxText.layers import attentions, linears, nnx_wrappers, quantizations
from MaxText.layers.initializers import NdInitializer, default_bias_init, nd_dense_init, variable_to_logically_partitioned
import numpy as np
import tokamax

set_xla_metadata = xla_metadata.set_xla_metadata


DISPATCH = "dispatch"
COMBINE = "combine"


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
  """Sort activations by `sort_indices`.

  If `use_custom_vjp=True`, then we use a custom backward pass that
  reverses the sort order. Specifically, this unsort operation is simply a sort
  with `jnp.argsort(sort_indices)` as the sort indices. This is only needed in
  the case where the compiler generates a less efficient backward pass op.

  Note that `use_custom_vjp=True` assumes that `sort_indices` is a permutation
  of `jnp.arange(inputs.shape[0])`.

  Args:
    inputs: `(tokens, ...)`-shaped array of input activations to sort.
    sort_indices: `(tokens,)`-shaped array containing the sort order.
    use_custom_vjp: Whether to use the explicit backward pass.

  Returns:
    `(tokens, ...)`-shaped array of input activations sorted by `sort_indices`.
  """
  assert inputs.shape[0] == sort_indices.shape[0]

  with jax.named_scope("sort_activations"):
    if use_custom_vjp:
      return _sort_activations_custom(inputs, sort_indices)
    return inputs[sort_indices, ...]


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
  """Sort functions with custom vjp."""
  return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Forward pass of the custom vjp for `_sort_activations()`."""
  return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  """Backward pass of the custom vjp for `_sort_activations()`."""
  sort_indices = residuals
  return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
      representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the indices of the selected experts for each
                       token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  indices = jnp.arange(num_experts).repeat(bs * seq_len)
  selected_num = bs * seq_len * num_experts_per_tok
  top_k_indices = jax.random.choice(rng_key, indices, shape=(selected_num,)).reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


class GateLogit(nnx.Module):
  """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing."""

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      model_name: str,
      rngs: nnx.Rngs,
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      use_bias: bool = False,
      score_func: str = "",
      quant: Optional[quantizations.AqtQuantization] = None,
      matmul_precision: str = "default",
  ):
    """Initializes the GateLogit module.

    Attributes:
      in_features_shape: The shape of the input features.
      out_features_shape: The shape of the output features, typically the number of experts.
      model_name: The name of the model.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      axis: The axis or axes over transformation is applied.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      use_bias: Whether to add learnable bias in gate logit scores. When enabled,
        this bias aids expert load balancing (like in DeepSeek V3), and is not
        part of the loss calculation.
      score_func: Scoring function for output normalization before applying bias.
      quant: The quantization configuration. If None, no quantization is applied.
      matmul_precision: The precision level for the matrix multiplication.
    """
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.model_name = model_name
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.score_func = score_func
    self.quant = quant
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      dummy_inputs = jnp.zeros((1, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: jax.Array, _initializing: bool = False) -> Tuple[jax.Array, Optional[jax.Array]]:

    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(norm_axis)))
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)
      if self.model_name.startswith("deepseek3"):
        pre_bias_logits = output

    if self.use_bias:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nnx.Module):
  """Implements a routed MoE block."""

  def __init__(
      self,
      config: ctypes.Config,
      num_experts: int,
      num_experts_per_tok: int,
      mesh: jax.sharding.Mesh,
      kernel_init: attentions.NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      intermediate_dim: int = 2048,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    """Initializes the RoutedMoE module.

    Attributes:
      config: The main config setting.
      num_experts: Number of experts.
      num_experts_per_tok: Number of experts for each token.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      intermediate_dim: Intermediate dimension of MoE.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    """
    self.config = config
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.intermediate_dim = intermediate_dim
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs

    if self.config.fsdp_shard_on_exp:
      # special sharding for dsv3
      self.wi_kernel_axes = ("embed_no_exp", None, "mlp")
      self.wo_kernel_axes = ("embed_no_exp", "mlp", None)
    else:
      self.wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
      self.wo_kernel_axes = ("exp", "mlp", "embed_no_exp")

    self.gate = GateLogit(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.num_experts,
        model_name=self.config.model_name,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        use_bias=self.config.routed_bias,
        score_func=self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    # pylint: disable=protected-access
    self.activation_fn = linears._convert_to_activation_function(self.config.mlp_activations[0])

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save
      # memory. Instead they are retrieved from the tensors stored in the 'aqt'
      # collection.
      self.wi_0 = jnp.zeros((num_experts, self.config.emb_dim, intermediate_dim))
      self.wi_1 = jnp.zeros((num_experts, self.config.emb_dim, intermediate_dim))
      self.wo = jnp.zeros((num_experts, intermediate_dim, self.config.emb_dim))
    else:
      self.wi_0 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.config.emb_dim, intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.wi_kernel_axes,
      )
      self.wi_1 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.config.emb_dim, intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (self.num_experts, self.intermediate_dim, self.config.emb_dim),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.wo_kernel_axes,
      )

    if self.config.mlp_bias:
      wi_bias_axes = ("exp", "activation_mlp")
      wo_bias_axes = ("exp", "activation_embed")
      wi_bias_shape = (self.num_experts, self.intermediate_dim)
      wo_bias_shape = (self.num_experts, self.config.emb_dim)
      self.wi_0_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          sharding=wi_bias_axes,
      )
      self.wi_1_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          sharding=wi_bias_axes,
      )
      self.wo_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wo_bias_shape, self.weight_dtype),
          sharding=wo_bias_axes,
      )
    else:
      self.wi_0_bias = None
      self.wi_1_bias = None
      self.wo_bias = None

  def get_expert_parallelism_size(self):
    return self.mesh.shape.get("expert", 1)

  def get_tensor_parallelism_size(self):
    return self.mesh.shape.get("tensor", 1)

  def get_tensor_transpose_parallelism_size(self):
    return self.mesh.shape.get("tensor_transpose", 1)

  def get_context_autoregressive_parallelism_size(self):
    return self.mesh.shape.get("context_autoregressive", 1)

  def get_topk(self, gate_logits, pre_bias_logits, rngs=None):
    """get topk."""
    # shape of top_k_weights & top_k_indices:
    # (batch, sequence, num_experts_per_tok).
    if self.config.use_random_routing:
      if rngs is None:
        raise ValueError("The random key cannot be None for random routing.")
      # Reuse the 'dropout' RNG stream to ensure random routing
      rng = rngs.dropout()
      top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
      return top_k_weights, top_k_indices

    if self.config.model_name.startswith("deepseek3"):
      top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block == ctypes.DecoderBlockType.DEEPSEEK:
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    elif self.config.decoder_block != ctypes.DecoderBlockType.LLAMA4:
      top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

    # This is the Qwen3-specific normalization of router weights.
    if self.config.norm_topk_prob:
      top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True)

    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights):
    """Scales weights according to DeepSeek's v3 reference implementation."""
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594.
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
    """Returns a mask that selects only the top-k groups of experts.

    Groups of experts are selected based on the sum of the top-2 expert scores
    for each group.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.

    Returns:
      Array of shape `(batch, seq, num_experts)` that is 1 for experts in the
      top-k groups and 0 elsewhere.
    """
    # Find top groups based on each group's top-2 expert scores, where
    # `scores_grouped.shape =
    # (batch * seq, n_routing_groups, experts_per_group)`.
    scores_grouped = jnp.reshape(
        gate_logits,
        gate_logits.shape[:-1] + (self.config.n_routing_groups, -1),
    )
    top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
    group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, k=self.config.topk_routing_group)

    # Mask selected groups so that only those experts are considered.
    group_mask = jax.nn.one_hot(group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32)
    group_mask = jnp.sum(group_mask, axis=-2)

    # Apply masks and get top-k indices.
    score_mask_expanded = jnp.broadcast_to(
        group_mask[..., None],
        group_mask.shape + (self.num_experts // self.config.n_routing_groups,),
    )
    return jnp.reshape(
        score_mask_expanded,
        score_mask_expanded.shape[:-2] + (self.num_experts,),
    )

  def deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weights use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.
      pre_bias_logits: Array of shape `(batch, seq,num_experts)`.

    Returns:
      - top_k_weights: `(batch, seq, num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(batch, seq, num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    """
    expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def apply_ffn_activation(self, layer_w0, layer_w1):
    """Applies FFN activation function."""
    with jax.named_scope("ffn_act"):
      if self.config.decoder_block == ctypes.DecoderBlockType.GPT_OSS:
        layer_w0 = jnp.clip(layer_w0, a_min=None, a_max=self.config.mlp_activations_limit)
        layer_w1 = jnp.clip(layer_w1, a_min=-self.config.mlp_activations_limit, a_max=self.config.mlp_activations_limit)
        layer_act = self.activation_fn(layer_w0 * 1.702)
        glu = jnp.multiply(layer_w0, layer_act)
        intermediate_layer = jnp.multiply(glu, (layer_w1 + 1))
      else:
        layer_act = self.activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      return intermediate_layer.astype(self.dtype)

  def permute(self, inputs, gate_logits, pre_bias_logits, use_custom_sort_vjp=True, rngs=None, roll_to_expert_id=None):
    """Permute tokens to group by expert to fit gmm call."""
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = self.get_topk(gate_logits, pre_bias_logits, rngs)

    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      # weights will be of shape (batch_size, seq_len, num_experts_per_tok)
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))  # weights are top_k_weights here
      # Squeeze router_scores to (batch_size * seq_len, num_experts_per_tok)
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    flatten_selected_experts = jnp.ravel(selected_experts)
    if roll_to_expert_id is not None:
      flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % self.num_experts
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    # sort inputs for number of selected experts
    replicated_inputs_2d = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
    sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp).astype(
        self.dtype
    )
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    # Return the experts for each sorted input.
    expert_indices = jnp.arange(self.num_experts)
    sorted_experts = jnp.repeat(
        expert_indices,
        repeats=group_size,
        total_repeat_length=flatten_selected_experts.shape[0],
    )
    return (
        sorted_inputs,
        sorted_selected_experts,
        weights,
        group_size,
        sorted_experts,
    )

  def unpermute(
      self,
      intermediate,
      sorted_selected_experts,
      weights,
      batch_size,
      sequence_length,
      use_custom_sort_vjp=True,
  ):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = _sort_activations(
        intermediate,
        jnp.argsort(sorted_selected_experts),
        use_custom_sort_vjp,
    )
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = jax.lax.Precision(self.config.matmul_precision)
      if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
        # For Llama4, combine using weights of 1 for selected experts
        reshaped_weights = jnp.ones_like(reshaped_weights)
      if self.config.float32_weight_sum:
        reshaped_intermediate = reshaped_intermediate.astype(jnp.float32)
        reshaped_weights = reshaped_weights.astype(jnp.float32)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate,
          reshaped_weights,
          precision=matmul_precision,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
  ):
    """Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts
    located
    on the current shard. It groups the tokens by their assigned local expert
    index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert
        across all the batch shards. Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to
        num_expert_parallelism - 1).
      is_offset: If True, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If
        False, assumes that `inputs` corresponding to the shard's experts start
        from the beginning of the tensor but need to be permuted by expert ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when
        `is_offset` is True. Shape `[total_tokens_for_this_shard]`.

    Returns:
      A tuple containing:
        sorted_inputs: Input data permuted local expert ID.
        sorted_indices: Indices used to permute the inputs.
        local_group_size: Number of tokens assigned to each local expert on this
          shard.
        sorted_experts_ids: expert ID corresponding to each token of the permuted
        inputs.
    """

    # Slice the count of local expert IDs in each batch shard.
    # all_shard_local_sizes.shape: [expert_shard, local_expert_size]
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        shard_index * local_expert_size,
        local_expert_size,
        axis=1,
    )
    local_sizes = all_shard_local_sizes.reshape(-1)

    # Total count of the local expert IDs is the sum of the counts across all
    # batch shards, since all batch shards will send their contributions to the
    # current expert shard.
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    # In this case, the data that needs to be processed by the local shard
    # does not start from row 0 but actually starts at
    # (jnp.concatenate((jnp.array([0]),
    #  jnp.cumsum(local_group_sizes[:-1]))[shard_id]).
    # This happens if batches (`inputs`) are replicated across expert shards and
    # pre-sorted by global Expert ID (via permute()).
    if is_offset:
      divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
      expert_indices = jnp.where(
          divided_assignments == shard_index,
          jnp.mod(global_sorted_experts, local_expert_size),
          local_expert_size,
      )

    # In this case the `input` data has been received from the batch shards and
    # needs to be reorganized in order of local Expert IDs.
    else:
      base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
      expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    sorted_inputs = _sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return (
        sorted_inputs,
        sorted_indices,
        local_group_size,
        sorted_experts_ids,
    )

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
  ):
    """Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all."""

    class TransformStrategy(enum.Enum):
      INPUT_OFFSET = enum.auto()
      SEND_SIZE = enum.auto()
      OUTPUT_OFFSET = enum.auto()
      RECV_SIZE = enum.auto()

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
      """Transforms the input array based on the specified strategy."""
      # Prepares it for the usage with `ragged_all_to_all` API. The
      # transformation determines how data is sent and received between shards.
      if is_batch_sharded:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # Index of input array for the send
          local_array = input_array[shard_id]
          return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
          # Size of input array for the send
          return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # Received index in the target output
          zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
          array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
          cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
          return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

      # If the batch is unsharded then we send the same data slice to all other
      # shards. We also assume each shard will have the local processed inputs
      # sorted to start from index 0. Finally, len(input_array.shape) == 1 since
      # there is only one batch shard.
      else:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # The data on each shard always starts at 0.
          return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
        elif strategy == TransformStrategy.SEND_SIZE:
          # The send amount is always the amount of data the current expert
          # shard needs to process.
          return jnp.repeat(input_array[shard_id], num_expert_parallelism)
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # The offset in each shard will just be the start of the group which
          # that shard is responsible for.
          output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        # The amount that each shard receives from all other shards is
        # equivalent to the group sizes (aka input_array).
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

    input_offsets = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.INPUT_OFFSET,
        is_batch_sharded,
    )
    send_sizes = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.SEND_SIZE,
        is_batch_sharded,
    )
    output_offsets = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.OUTPUT_OFFSET,
        is_batch_sharded,
    )
    recv_sizes = transform_array(
        all_shards_group_sizes,
        shard_id,
        TransformStrategy.RECV_SIZE,
        is_batch_sharded,
    )
    return input_offsets, send_sizes, output_offsets, recv_sizes

  def transform_bias(self, experts_index, *biases):
    """Selects bias values for a variable number of bias tensors based on chosen experts."""
    return tuple(bias[experts_index] for bias in biases)

  def sparse_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ):
    """Perform sparse matrix multiplication of inputs and Experts."""

    def gmm(inputs, kernel, tiling, group_sizes, expert_assignments):
      pad_length = self.config.tile_batch_seq
      hs_shape = inputs.shape
      # pad length is the 1st dimension of tiling size in gmm call
      if inputs.shape[0] != expert_assignments.shape[0]:
        raise ValueError("The number of input tokens must match the number of expert" " assignments!")
      padding_amount = 0
      if hs_shape[0] % pad_length:
        padding_amount = pad_length - hs_shape[0] % pad_length
        inputs = jax.lax.pad(inputs, jnp.array(0.0, dtype=inputs.dtype), [(0, padding_amount, 0), (0, 0, 0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.dtype)

      lhs_quantize_dtype, rhs_quantize_dtype = None, None
      if self.quant is not None:
        quant_dg = self.quant.quant_dg
        lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
        rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()
      m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
      tiling = (
          min(tiling[0], m),
          min(tiling[1], k),
          min(tiling[2], n),
      )
      if self.config.use_tokamax_gmm:
        output = tokamax.ragged_dot(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=self.dtype,
            implementation="mosaic",
        )
      else:
        if self.config.megablox:
          output = mblx.gmm(
              lhs=inputs,
              rhs=kernel,
              group_sizes=group_sizes,
              preferred_element_type=self.dtype,
              tiling=tiling,
              lhs_quantize_dtype=lhs_quantize_dtype,
              rhs_quantize_dtype=rhs_quantize_dtype,
              use_qwix_quantization=self.config.use_qwix_quantization,
          )
        else:
          rhs_inputs = kernel
          if isinstance(kernel, aqt.QTensor):
            if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
              raise ValueError("Unsupported usecase for ragged_dot with quantized kernel.")
            rhs_inputs = kernel.qvalue
          with set_xla_metadata(ragged_dot_tiling=",".join([str(t) for t in tiling])):
            output = jax.lax.ragged_dot(
                lhs=inputs,
                rhs=rhs_inputs,
                group_sizes=group_sizes,
                preferred_element_type=self.dtype,
            )
          if isinstance(kernel, aqt.QTensor):
            # Multiply outputs by the kernely scale
            scales = jnp.take(kernel.scale[0].squeeze(), indices=expert_assignments, axis=0)
            if padding_amount > 0:
              scales = jax.lax.pad(
                  scales,
                  jnp.array(0.0, dtype=scales.dtype),
                  [(0, padding_amount, 0), (0, 0, 0)],
              )
            output *= scales
      if padding_amount > 0:
        output = output[: hs_shape[0]]
      return output

    # Currently, we support data, tensor, and expert parallelism with Megablox.
    # We all gather the input activations over tensor parallelism to follow
    # https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.

    # Check if the batch should be sharded by expert and whether the batch_size
    # supports this. For example, for interleaved inference, prefill always has
    # batch_size=1 while decode can have batch_size > 1.
    try:
      is_batch_sharded_by_expert = (
          "expert"
          in tuple(
              filter(
                  lambda tup: tup[0] == "activation_batch",
                  self.config.logical_axis_rules,
              )
          )[
              0
          ][1]
      )
    except:  # pylint: disable=bare-except
      is_batch_sharded_by_expert = False
    if is_batch_sharded_by_expert and inputs.shape[0] > 1:
      batch_logical_axis = "activation_batch"
    else:
      batch_logical_axis = "activation_batch_no_exp"

    if self.get_tensor_transpose_parallelism_size() > 1:
      input_partition_pspec = nn.logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", "activation_embed"))
      w0_bias_pspec = nn.logical_to_mesh_axes(("exp", None))
      w1_bias_pspec = nn.logical_to_mesh_axes(("exp", None))
      wo_bias_pspec = nn.logical_to_mesh_axes(("exp", "activation_embed"))
    else:
      input_partition_pspec = nn.logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
      w0_bias_pspec = nn.logical_to_mesh_axes(("exp", "activation_mlp"))
      w1_bias_pspec = nn.logical_to_mesh_axes(("exp", "activation_mlp"))
      wo_bias_pspec = nn.logical_to_mesh_axes(("exp", "activation_embed"))

    gate_logits_pspec = nn.logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
    if self.config.model_name.startswith("deepseek3"):
      pre_bias_logits_pspec = nn.logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
    else:
      # pre_bias_logits is None for non-DeepSeek v3 models
      pre_bias_logits_pspec = None

    # w0, w1, wo needs to be un sharded on fsdp / fsdp_transpose axis, so use
    # mlp_no_fsdp axis
    if self.config.fsdp_shard_on_exp:
      # special sharding for dsv3 to remove overhead between gmm/AG
      w0_pspec = nn.logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
      w1_pspec = nn.logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
      wo_pspec = nn.logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
    else:
      w0_pspec = nn.logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
      w1_pspec = nn.logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
      wo_pspec = nn.logical_to_mesh_axes(("exp", "mlp_no_fsdp", "embed_tensor_transpose"))
    if isinstance(w0_kernel, aqt.QTensor):
      w0_pspec = aqt.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
    if isinstance(w1_kernel, aqt.QTensor):
      w1_pspec = aqt.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
    if isinstance(wo_kernel, aqt.QTensor):
      wo_pspec = aqt.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            input_partition_pspec,
            gate_logits_pspec,
            pre_bias_logits_pspec,
            w0_pspec,
            w1_pspec,
            wo_pspec,
            w0_bias_pspec,
            w1_bias_pspec,
            wo_bias_pspec,
            None,
        ),
        out_specs=(nn.logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", "activation_embed"))),
        check_vma=False,
    )
    def wrapper(x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, rngs):
      batch_size, sequence_length, _ = x.shape
      expert_axis_name = "expert"
      num_expert_parallelism = self.get_expert_parallelism_size()
      if num_expert_parallelism > 1:
        expert_shard_id = jax.lax.axis_index(expert_axis_name)
      else:
        expert_shard_id = 0
      num_expert_parallelism = self.get_expert_parallelism_size()
      if self.config.use_ring_of_experts:
        # The ring-of-experts strategy first duplicates the inputs to all
        # expert shards, and then routes within each shard.

        # Duplicate inputs to all expert shards.
        x, logits, pre_bias_logits = tuple(
            jax.lax.all_gather(z, axis_name=expert_axis_name, tiled=True) for z in (x, logits, pre_bias_logits)
        )

        # "Route" tokens within each shard.
        num_experts_per_shard = self.config.num_experts // num_expert_parallelism
        x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(
            x,
            logits,
            pre_bias_logits,
            self.config.use_custom_sort_vjp,
            roll_to_expert_id=num_experts_per_shard * expert_shard_id,
        )

        # Filter down to the group sizes that apply to only the experts in the
        # current shard.
        group_sizes = group_sizes[:num_experts_per_shard]
        mask = jnp.arange(x.shape[0]) < jnp.sum(group_sizes)
        x = jnp.where(mask[:, None], x, 0)
      else:
        x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(
            x, logits, pre_bias_logits, self.config.use_custom_sort_vjp, rngs
        )

        if num_expert_parallelism > 1:
          batch_axis = "expert" if is_batch_sharded_by_expert else "data"
          # get group sizes for all shards
          local_expert_size = self.config.num_experts // num_expert_parallelism
          reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
          global_group_sizes = group_sizes
          if is_batch_sharded_by_expert:
            all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                all_shards_group_sizes,
                expert_shard_id,
                num_expert_parallelism,
            )

            # TODO(ranran): For better performance, we could update output buffer to a smaller
            # size to replace self.get_expert_parallelism_size() for efficiency,
            # Or we could apply capacity_factor for excessive experts.
            # Note: Reducing buffer increase the risk of token dropping under unbalanced distribution.

            # In the worst case, all of the global input data is assigned to each expert in the current shard.
            # This would result in num_expert_shards * input_size * experts_per_shard assignments. However, if
            # experts_per_shard > num_experts_per_tok we cannot assign more than num_experts_per_tok to all of the inputs.
            max_local_experts_per_tok = min(local_expert_size, self.config.num_experts_per_tok)
            buffer_size = int(
                num_expert_parallelism
                * self.config.per_device_batch_size
                * self.config.max_target_length
                * max_local_experts_per_tok
            )
            output_shape = jnp.zeros((buffer_size, self.config.emb_dim), dtype=x.dtype)

            x = jax.lax.ragged_all_to_all(
                x,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=expert_axis_name,
            )
            global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=expert_axis_name)
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes,
                local_expert_size,
                shard_index=expert_shard_id,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
            )
          else:
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes[None, :],
                local_expert_size,
                shard_index=expert_shard_id,
                is_offset=True,
                global_sorted_experts=selected_experts,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
            )

      if self.config.mlp_bias:
        w0_bias, w1_bias, wo_bias = self.transform_bias(selected_experts, w0_bias, w1_bias, wo_bias)

      gmm_fn = functools.partial(
          gmm,
          group_sizes=group_sizes,
          expert_assignments=selected_experts,
      )
      wi_tile_size = (
          self.config.tile_batch_seq,
          self.config.tile_embed_dim,
          self.config.tile_mlp_dim,
      )
      wo_tile_size = (
          self.config.tile_batch_seq,
          self.config.tile_mlp_dim,
          self.config.tile_embed_dim,
      )
      layer_w0 = gmm_fn(x, w0, tiling=wi_tile_size)
      if self.get_tensor_transpose_parallelism_size() > 1:
        layer_w0 = jax.lax.psum(layer_w0, "tensor_transpose")
      if self.config.mlp_bias:
        layer_w0 = layer_w0 + w0_bias
      layer_w0 = adc.checkpoint_name(layer_w0, "mlpwi_0")

      layer_w1 = gmm_fn(x, w1, tiling=wi_tile_size)
      if self.get_tensor_transpose_parallelism_size() > 1:
        layer_w1 = jax.lax.psum(layer_w1, "tensor_transpose")
      if self.config.mlp_bias:
        layer_w1 = layer_w1 + w1_bias
      layer_w1 = adc.checkpoint_name(layer_w1, "mlpwi_1")
      intermediate_layer = self.apply_ffn_activation(layer_w0, layer_w1)

      intermediate_output = gmm_fn(intermediate_layer, wo, tiling=wo_tile_size)
      if self.get_tensor_parallelism_size() > 1:
        intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)
      if self.config.mlp_bias:
        intermediate_output = intermediate_output + wo_bias
      intermediate_output = adc.checkpoint_name(intermediate_output, "mlpwo")

      if self.config.use_ring_of_experts:
        # Set the outputs of tokens which were not processed to 0.
        mask = jnp.arange(intermediate_output.shape[0]) < jnp.sum(group_sizes)
        intermediate_output = jnp.where(mask[:, None], intermediate_output, 0)

        # Unsort and deduplicate the outputs locally.
        output = self.unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_custom_sort_vjp=self.config.use_custom_sort_vjp,
        )

        # Sum up the partial outputs across the expert shards.
        output = jnp.reshape(output, (-1, sequence_length, self.config.emb_dim))
        output = jax.lax.psum_scatter(output, expert_axis_name, scatter_dimension=0, tiled=True)

      else:
        if num_expert_parallelism > 1:
          original_inputs_first_dim = batch_size * sequence_length * self.config.num_experts_per_tok
          if sorted_selected_experts.shape[0] != original_inputs_first_dim:
            raise ValueError("original_inputs_first_dim does not match the original tensor" " shape!")
          output_shape = jnp.zeros(
              (
                  original_inputs_first_dim,
                  self.config.emb_dim // self.get_tensor_parallelism_size(),
              ),
              dtype=intermediate_output.dtype,
          )
          if is_batch_sharded_by_expert:
            # locally unpermute back to the original order
            local_output = _sort_activations(
                intermediate_output,
                jnp.argsort(local_sorted_indices),  # pylint: disable=undefined-variable
                self.config.use_custom_sort_vjp,
            )
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                jnp.transpose(all_shards_group_sizes),  # pylint: disable=undefined-variable
                expert_shard_id,
                num_expert_parallelism,
            )
            intermediate_output = jax.lax.ragged_all_to_all(
                local_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=expert_axis_name,
            )
          else:
            # If bach is replicated across EP shards then each shard should send
            # 0..local_shard_size data to the other shards and receive the
            # local_shard data from all of the other shards using
            # ragged_all_to_all.
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                reshaped_group_sizes,  # pylint: disable=undefined-variable
                expert_shard_id,
                num_expert_parallelism,
                is_batch_sharded=False,
            )
            intermediate_output = jax.lax.ragged_all_to_all(
                intermediate_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=expert_axis_name,
            )

        output = self.unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_custom_sort_vjp=self.config.use_custom_sort_vjp,
        )

      return output, None

    if self.config.moe_fsdp_use_two_stage_all_gather:
      # Unshard on fsdp axis
      w0_kernel = nn.with_logical_constraint(w0_kernel, ("exp", "embed_tensor_transpose", "mlp"))
      w1_kernel = nn.with_logical_constraint(w1_kernel, ("exp", "embed_tensor_transpose", "mlp"))

      # Unshard on fsdp_transpose axis
      wo_kernel = nn.with_logical_constraint(wo_kernel, ("exp", "mlp", "embed_tensor_transpose"))

      # Make sure XLA does not optimize by combining above All-Gather to unshard
      # on FSDP axis and the subsequent unshard on fsdp_transpose axis
      w0_kernel = jax.lax.optimization_barrier(w0_kernel)
      w1_kernel = jax.lax.optimization_barrier(w1_kernel)
      wo_kernel = jax.lax.optimization_barrier(wo_kernel)

      # Unshard on both fsdp and fsdp_transpose transpose
      w0_kernel = nn.with_logical_constraint(w0_kernel, ("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
      w1_kernel = nn.with_logical_constraint(w1_kernel, ("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
      wo_kernel = nn.with_logical_constraint(wo_kernel, ("exp", "mlp_no_fsdp", "embed_tensor_transpose"))

    return wrapper(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, self.rngs
    )

  def reshape_and_update_weights(self, weights, indices):
    """reshape and update weights."""
    # input of weights and indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (
        jnp.arange(weights.shape[0])[:, None, None],
        jnp.arange(weights.shape[1])[:, None],
        indices,
    )
    update_weights = update_weights.at[index_update].set(weights)
    return update_weights

  def get_context_partition_and_sub_seq(self, seq_len):
    cp = self.get_context_autoregressive_parallelism_size()
    if seq_len % cp != 0:
      cp = 1
    sub_seq = seq_len // cp
    return cp, sub_seq

  def generate_masks_subgroup(self, top_k_indices, softmax_probs):
    """Subgroup mask generation for inference only."""
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    # Break sequence into subsequences (groups) of tokens, and route only within
    # each group.
    top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))

    tokens_per_batch = sub_seq * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=3)

    # reshape & update weights
    softmax_probs = jnp.reshape(
        softmax_probs,
        ((batch_size, cp, sub_seq, self.num_experts)),
    )
    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=3) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    # ici_context_parallelism
    dispatch_mask = jnp.reshape(
        dispatch_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )
    combine_mask = jnp.reshape(
        combine_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )

    return dispatch_mask, combine_mask

  def generate_masks(self, top_k_indices, softmax_probs):
    """Generate masks."""
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape

    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, seq_len * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:
    """Compute the load balance loss."""
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(
      self,
      rhs_mesh_axes: Tuple[Optional[str], ...] = (),
      einsum_name: str | None = None,
  ):
    """Get the Einstein summation."""

    # the check is to prevent aqteinsum as einsum op for dispatch and combine
    # einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and einsum_name in (
        DISPATCH,
        COMBINE,
    ):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):  # pylint: disable=unused-argument
        # simply skip kwargs, since aqt einsum doesn't support any kwargs
        # like precision
        is_aqt = not isinstance(self.quant, quantizations.Fp8Quantization)
        kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
        return self.quant.einsum(**kw)(*args)  # pytype: disable=attribute-error

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def maybe_all_gather_kernel_weight_in_expert_parallelism(
      self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...]
  ):
    """All-gather kernel weight in expert parallelism if needed."""
    if self.get_expert_parallelism_size() > 1:
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead
      # of weight_dtype
      kernel = nn.with_logical_constraint(kernel, kernel_axes)
    return kernel

  def dense_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[jax.Array, Optional[jax.Array]]:
    """Dense matrix multiplication."""
    # gate_logits: batch, length, expert
    gate_logits = nn.with_logical_constraint(gate_logits, ("activation_batch", "activation_norm_length", None))
    if self.config.model_name.startswith("deepseek3"):
      # pre_bias_logits is None for non-DeepSeek v3 models
      pre_bias_logits = nn.with_logical_constraint(pre_bias_logits, ("activation_batch", "activation_norm_length", None))
    top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits, self.rngs)
    is_llama4_decoder_layer = self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4
    if is_llama4_decoder_layer:
      router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(self.dtype)
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
    matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    if self.config.model_call_mode != "inference":
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      loss = self.load_balance_loss(top_k_indices, softmax_probs)
    else:
      loss = None
    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]

    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      if self.config.model_call_mode != "inference":
        # TODO(b/425930949): remove this pylint by refactoring the logic here.
        dispatch_mask, combine_mask = self.generate_masks(
            top_k_indices, weights  # pylint: disable=undefined-variable,possibly-used-before-assignment
        )
        mask_axes = ("activation_batch", "activation_norm_length", None, None)
        dispatch_axis = (
            "activation_exp",
            "activation_batch_no_exp",
            None,
            "activation_embed",
        )
        mlp_axis = (
            "activation_exp",
            "activation_batch_no_exp",
            None,
            "activation_mlp",
        )
        dispatch_eimsum = "BSM,BSEC -> EBCM"
        mlp_up_einsum = "EBCM,EMH -> EBCH"
        mlp_down_einsum = "EBCH,EHM -> EBCM"
        output_einsum = "EBCM,BSEC -> BSM"
      else:
        # TODO(b/425930507): Try replacing `softmax_probs` with padded weights
        # and verify with decode acc tests.
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = (
              "activation_norm_length",
              "activation_batch",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_norm_length",
              "activation_batch",
              None,
              "activation_embed",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_no_exp",
              None,
              None,
              "activation_embed",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_no_exp",
              None,
              None,
              "activation_mlp",
          )
        else:
          mask_axes = (
              "activation_batch",
              "activation_norm_length",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_batch",
              "activation_norm_length",
              None,
              "activation_embed",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_no_exp",
              None,
              None,
              "activation_embed",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_no_exp",
              None,
              None,
              "activation_mlp",
          )
        dispatch_eimsum = "BNSM,BNSEC -> EBNCM"
        mlp_up_einsum = "EBNCM,EMH -> EBNCH"
        mlp_down_einsum = "EBNCH,EHM -> EBNCM"
        output_einsum = "EBNCM,BNSEC -> BNSM"

        inputs = jnp.reshape(inputs, (batch_size, cp, sub_seq, inputs.shape[2]))
        inputs = nn.with_logical_constraint(inputs, input_axis)

      dispatch_mask = nn.with_logical_constraint(dispatch_mask, mask_axes)
      combine_mask = nn.with_logical_constraint(combine_mask, mask_axes)

      with jax.named_scope("dispatch"):
        # only cp during prefill
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            dispatch_eimsum, inputs, dispatch_mask, precision=matmul_precision
        )
        if cp > 1:
          dispatch = nn.with_logical_constraint(
              dispatch,
              (
                  None,
                  "activation_batch_no_exp",
                  "activation_norm_length",
                  None,
                  "activation_embed",
              ),
          )
        dispatch = nn.with_logical_constraint(
            dispatch,
            dispatch_axis,
        )
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            mlp_up_einsum, dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w0_bias = w0_bias[:, None, None, :]
          layer_w0 = layer_w0 + w0_bias

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = nn.with_logical_constraint(
            layer_w0,
            mlp_axis,
        )
        layer_w0 = adc.checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            mlp_up_einsum, dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w1_bias = w1_bias[:, None, None, :]
          layer_w1 = layer_w1 + w1_bias
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = nn.with_logical_constraint(
            layer_w1,
            mlp_axis,
        )
        layer_w1 = adc.checkpoint_name(layer_w1, "mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            mlp_down_einsum,
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          wo_bias = wo_bias[:, None, None, :]
          intermediate_layer = intermediate_layer + wo_bias
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        if self.config.model_call_mode != "inference":
          intermediate_layer = nn.with_logical_constraint(
              intermediate_layer,
              (
                  "activation_exp",
                  "activation_batch_no_exp",
                  None,
                  "activation_embed",
              ),
          )
        intermediate_layer = adc.checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            output_einsum,
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        )
        if output.ndim == 4:
          output = jnp.reshape(
              output,
              (
                  output.shape[0],
                  output.shape[1] * output.shape[2],
                  output.shape[3],
              ),
          )
      return output, loss
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = adc.checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w1 = layer_w1 + w1_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = adc.checkpoint_name(layer_w1, "mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)

      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM",
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          intermediate_layer = intermediate_layer + wo_bias[None, None, :, :]
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = adc.checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("weight_sum"):
        if is_llama4_decoder_layer:
          weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
        if self.config.float32_weight_sum:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
          weights = weights.astype(jnp.float32)
        # cast to f32 for sum up in einsum op
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, None

  def retrieve_quantized_weight(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    """Retrieve quantized weights."""
    # This is called only during tracing. This is to invoke creation of
    # quantized tensor inside AqtEinsum.  After jit, this will become no-op and
    # will not affect performance.
    _ = self.dense_matmul(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
    )

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, Optional[jax.Array]]:
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits, pre_bias_logits = self.gate(inputs)

    w0_kernel = jnp.asarray(self.wi_0[...], self.dtype)
    w1_kernel = jnp.asarray(self.wi_1[...], self.dtype)
    wo_kernel = jnp.asarray(self.wo[...], self.dtype)

    if cfg.mlp_bias:
      w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
      w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
      wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
    else:
      w0_bias, w1_bias, wo_bias = None, None, None

    if cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_bias,
            w1_bias,
            wo_bias,
        )
      return self.sparse_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )
    else:
      return self.dense_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )


class RoutedAndSharedMoE(nnx.Module):
  """Implements a block which combines shared and routed experts."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      kernel_init: NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    """nitializes the RoutedAndSharedMoE module.

    Attributes:
      config: The main config setting.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    """
    self.config = config
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    # NOTE: the name MoeBlock_0 is to ensure reverse compatibility with
    # existing checkpoints for routed experts.
    self.MoeBlock_0 = RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
    )
    self.shared_experts = linears.MlpBlock(
        mesh=self.mesh,
        in_features=self.config.emb_dim,
        intermediate_dim=self.config.shared_experts * self.config.moe_mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=self.quant,
        rngs=self.rngs,
    )

  @property
  def routed_moe(self):
    return self.MoeBlock_0

  def __call__(self, inputs: jax.Array) -> jax.Array:
    routed_experts, _ = self.routed_moe(inputs)
    shared_experts = self.shared_experts(inputs)
    return routed_experts + shared_experts


def get_gate_logit(
    inputs_shape: tuple[int, ...],
    out_features_shape: Union[Iterable[int], int],
    model_name: str,
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    use_bias: bool = False,
    score_func: str = "",
    quant: Optional[quantizations.AqtQuantization] = None,
    matmul_precision: str = "default",
    name: Optional[str] = None,
):
  """Creates a GateLogit Linen module."""

  axis = linears.canonicalize_tuple(axis)
  in_features_shape = tuple(inputs_shape[ax] for ax in linears.normalize_axes(axis, len(inputs_shape)))

  module = nnx_wrappers.to_linen(
      GateLogit,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      model_name=model_name,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      use_bias=use_bias,
      score_func=score_func,
      quant=quant,
      matmul_precision=matmul_precision,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_moe(
    config: ctypes.Config,
    num_experts: int,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    intermediate_dim: int = 2048,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  """Creates a RoutedMoE Linen module."""

  module = nnx_wrappers.to_linen(
      RoutedMoE,
      config=config,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      intermediate_dim=intermediate_dim,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_and_shared_moe(
    config: ctypes.Config,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  """Creates a RoutedAndSharedMoE Linen module."""

  module = nnx_wrappers.to_linen(
      RoutedAndSharedMoE,
      config=config,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
