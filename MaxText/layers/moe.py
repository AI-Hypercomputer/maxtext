#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""MoE-related layers."""

import enum
import functools
import math
from typing import Any, Optional, Tuple

from aqt.jax.v2 import aqt_tensor as aqt
import flax.linen as nn
import jax
from jax import ad_checkpoint as adc
from jax.experimental import shard_map
import jax.numpy as jnp
from MaxText import common_types as ctypes
from MaxText import max_logging
from MaxText import max_utils
from MaxText.kernels import megablox as mblx
from MaxText.layers import attentions
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import quantizations
import numpy as np


DISPATCH = "dispatch"
COMBINE = "combine"


def random_routing(
    rng_key: jax.Array,
    gate_logits: jax.Array,
    num_experts_per_tok: int,
) -> tuple[jax.Array, jax.Array]:
  """Performs random routing of tokens to experts.

  Args:
    rng_key: PRNGKey for randomnly selecting experts.
    gate_logits: Array of shape `(batch, seq, num_experts)` representing the
      logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    top_k_weights: Array of shape `(batch, seq, num_experts_per_tok)`.
    top_k_indices: Array of shape `(batch, seq, num_experts_per_tok)`.
  """
  num_tokens = math.prod(gate_logits.shape[:-1])
  num_experts = gate_logits.shape[-1]
  indices = jnp.arange(num_experts).repeat(num_tokens)
  selected_num = num_tokens * num_experts_per_tok
  top_k_indices = jnp.reshape(
      jax.random.choice(rng_key, indices, shape=(selected_num,)),
      gate_logits.shape[:-1] + (num_experts_per_tok,),
  )
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def _maybe_make_param_in_module(
    module: nn.Module,
    name: str,
    kernel_init: attentions.NdInitializer,
    kernel_axes: Tuple[str, ...],
    kernel_shape: tuple[int, ...],
    kernel_in_axis: tuple[int, ...],
    kernel_out_axis: tuple[int, ...],
) -> jax.Array:
  """Creates parameter `name` in `module` unless in quantized serve mode.

  Args:
    module: The module to create the parameter in.
    name: The name of the parameter.
    kernel_init: The initializer function for the parameter.
    kernel_axes: The axes of the parameter.
    kernel_shape: The shape of the parameter.
    kernel_in_axis: The axes of the parameter that are inputs.
    kernel_out_axis: The axes of the parameter that are outputs.

  Returns:
    The parameter as a JAX array.

    If `module.quant == True`, then we instead return a placeholder array of
    zeros. This is done to save memory by storing the weight tensors from the
    'aqt' collection instead of from `params`.
  """
  if quantizations.in_serve_mode(module.quant):
    return jnp.zeros(kernel_shape)
  else:
    return jnp.asarray(
        module.param(
            name,
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            kernel_shape,
            module.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        )
    )


class GateLogit(nn.Module):
  """Computes gate logits with pre-bias values neeeded for DeepSeek routing.

  Attributes:
    features: the number of output features.
    model_name: the model to run.
    axis: axis to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: tuple with axes to apply kernel function.
    use_bias: whether to add learnable bias in gate logit scores. When enabled,
      this bias aids expert load balancing (like in DeepSeek V3), and is not
      part of the loss calculation.
    score_func: scoring function for output normalization before applying bias.
    quant: quantization config, defaults to None implying no quantization.
    matmul_precision: precision for JAX functions.
  """

  features: int
  model_name: str
  axis: int = -1
  weight_dtype: ctypes.DType = jnp.float32
  dtype: ctypes.DType = jnp.float32
  kernel_init: attentions.NdInitializer = attentions.nd_dense_init(
      1.0, "fan_in", "truncated_normal"
  )
  kernel_axes: Tuple[Optional[str], ...] = ()
  use_bias: bool = False
  score_func: str = ""
  quant: Optional[quantizations.AqtQuantization] = None
  matmul_precision: str = "default"

  @nn.compact
  def __call__(
      self, inputs: ctypes.Array
  ) -> Tuple[ctypes.Array, Optional[ctypes.Array]]:

    features = linears._canonicalize_tuple(self.features)
    axis = linears._canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = linears._normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))

    kernel = _maybe_make_param_in_module(
        module=self,
        name="kernel",
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        kernel_shape=kernel_shape,
        kernel_in_axis=kernel_in_axis,
        kernel_out_axis=kernel_out_axis,
    )

    output = linears._compute_dot_general(
        inputs=inputs,
        kernel=kernel,
        kernel_axes=self.kernel_axes,
        axis=axis,
        contract_ind=tuple(range(0, len(axis))),
        matmul_precision=self.matmul_precision,
        quant=self.quant,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)
      if self.model_name.startswith("deepseek3"):
        pre_bias_logits = output

    if self.use_bias:
      bias_axes, bias_shape = (
          self.kernel_axes[-len(features) :],
          kernel_shape[-len(features) :],
      )
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(
              initializers.default_bias_init, bias_axes
          ),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nn.Module):
  """Implements a routed MoE block.

  Attributes:
    config: Configuration for the MoE block.
    num_experts: Total number of experts.
    num_experts_per_tok: Number of routed experts per token.
    mesh: Device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    intermediate_dim: Size of the intermediate/hidden expert dimension.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: ctypes.Config
  num_experts: int
  num_experts_per_tok: int
  mesh: jax.sharding.Mesh
  kernel_init: attentions.NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  intermediate_dim: int = 2048
  weight_dtype: ctypes.DType = jnp.float32
  dtype: ctypes.DType = jnp.float32
  quant: Optional[quantizations.AqtQuantization] = None

  # The first axes is expert
  wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
  wo_kernel_axes = ("exp", "mlp", "embed_no_exp")

  def is_ds3(self):
    return self.config.model_name.startswith("deepseek3")

  def matmul_precision(self):
    return jax.lax.Precision(self.config.matmul_precision)

  def get_expert_parallelism_size(self) -> int:
    return self.mesh.shape["expert"]

  def get_tensor_parallelism_size(self) -> int:
    return self.mesh.shape["tensor"]

  def get_context_autoregressive_parallelism_size(self) -> int:
    return self.mesh.shape["context_autoregressive"]

  def is_batch_sharded_by_expert(self) -> bool:
    """Returns `True` if `activation_batch` contains the `expert` axis."""
    rules = dict(self.config.logical_axis_rules)
    return "activation_batch" in rules and "expert" in rules["activation_batch"]

  def generate_kernels(
      self, num_experts: int, emb_dim: int, mlp_dim: int
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generates the params for the weights in the MoE block.

    Args:
      num_experts: The number of experts.
      emb_dim: The embedding dimension.
      mlp_dim: The MLP dimension.

    Returns:
      `wi_0`: Array of shape `(num_experts, emb_dim, mlp_dim)`.
      `wi_1`: Array of shape `(num_experts, emb_dim, mlp_dim)`.
      `wo`: Array of shape `(num_experts, mlp_dim, emb_dim)`.
    """

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    kernel_init = attentions.nd_dense_init(1.0, "fan_in", "truncated_normal")

    maybe_make_param = functools.partial(
        _maybe_make_param_in_module,
        module=self,
        kernel_init=kernel_init,
        kernel_in_axis=kernel_in_axis,
        kernel_out_axis=kernel_out_axis,
    )

    w0_kernel = maybe_make_param(
        name="wi_0",
        kernel_axes=self.wi_kernel_axes,
        kernel_shape=(num_experts, emb_dim, mlp_dim),
    )

    w1_kernel = maybe_make_param(
        name="wi_1",
        kernel_axes=self.wi_kernel_axes,
        kernel_shape=(num_experts, emb_dim, mlp_dim),
    )

    wo_kernel = maybe_make_param(
        name="wo",
        kernel_axes=self.wo_kernel_axes,
        kernel_shape=(num_experts, mlp_dim, emb_dim),
    )
    return w0_kernel, w1_kernel, wo_kernel

  def get_topk(
      self, gate_logits: jax.Array, pre_bias_logits: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Returns the per-token top-k expert weights and indices.

    Args:
      gate_logits: Array of shape `(..., num_experts)`.
      pre_bias_logits: Array of shape `(..., num_experts)`.

    Returns:
      - top_k_weights: `(..., num_experts_per_tok)` array of weights for experts
        selected for each token.
      - top_k_indices: `(..., num_experts_per_tok)` array of indices identifying
        the selected experts for each token.
    """
    if self.config.use_random_routing:
      return random_routing(
          rng_key=self.make_rng("random_routing"),
          gate_logits=gate_logits,
          num_experts_per_tok=self.num_experts_per_tok,
      )

    if self.config.model_name.startswith("deepseek3"):
      top_k_weights, top_k_indices = self.deepseek_routing(
          gate_logits, pre_bias_logits
      )
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(
          gate_logits, self.num_experts_per_tok
      )

    if self.config.decoder_block == ctypes.DecoderBlockType.DEEPSEEK:
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    elif self.config.decoder_block != ctypes.DecoderBlockType.LLAMA4:
      top_k_weights = jnp.astype(
          jax.nn.softmax(jnp.astype(top_k_weights, jnp.float32), axis=-1),
          self.dtype,
      )

    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights: jax.Array) -> jax.Array:
    """Scales weights according to DeepSeek's v3 reference implementation.

    Reference:
    https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594

    Args:
      weights: Array of weights.

    Returns:
      Array of scaled weights.
    """
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def _expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
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
    group_mask = jax.nn.one_hot(
        group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32
    )
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

  def deepseek_routing(
      self, gate_logits: jax.Array, pre_bias_logits: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weigths use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(..., num_experts)`.
      pre_bias_logits: Array of shape `(..., num_experts)`.

    Returns:
      - top_k_weights: `(..., num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(..., num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    """
    expert_mask = (
        1
        if self.config.n_routing_groups == -1
        else self._expert_group_mask(gate_logits)
    )
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def selected_experts(self, group_sizes: jax.Array, num_tokens: int):
    return jnp.repeat(
        jnp.arange(self.num_experts),
        repeats=group_sizes,
        total_repeat_length=num_tokens,
    )

  def permute(
      self, inputs: jax.Array, expert_selection: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Sort and group tokens by expert.

    Args:
      inputs: `(..., emb_dim)` array of input tokens.
      expert_selection: `(..., num_experts_per_tok)` array of expert ids.

    Returns:
      - sorted_and_duplicated_tokens: `(num_duplicated_tokens, emb_dim)` array
        of tokens that contains `num_experts_per_tok` copies of each token from
        `inputs`, and sorted by expert id.
      - group_sizes: `(num_experts,)` array of the number of tokens assigned to
        each expert.
    """
    group_sizes = jnp.bincount(
        jnp.ravel(expert_selection), length=self.num_experts
    )
    sorted_inputs = jnp.astype(
        jnp.take(
            jnp.reshape(inputs, (-1, inputs.shape[-1])),
            indices=(
                jnp.argsort(expert_selection, axis=None)
                // self.num_experts_per_tok
            ),
            axis=0,
        ),
        self.dtype,
    )
    return sorted_inputs, group_sizes

  def unpermute(
      self,
      inputs: jax.Array,
      expert_selection: jax.Array,
      expert_affinity: jax.Array,
  ) -> jax.Array:
    """Undoes the transformation induced by `permute()`.

    Args:
      inputs: `(num_duplicated_tokens, emb_dim)` array of input tokens.
      expert_selection: `(..., num_experts_per_tok)` array of expert ids.
      expert_affinity: `(..., num_experts_per_tok)` array of expert weights.

    Returns:
      `inputs` permuted back to the original order and shape (as encoded in
      `routing_table`), where the intermediate values from separate experts are
      combined using `weights`.
    """
    # Unpermute tokens back to the original order, with added dimension for
    # the intermediate values from separate experts.
    sorted_selected_experts = jnp.argsort(expert_selection, axis=None)
    inputs = jnp.take(
        inputs, indices=jnp.argsort(sorted_selected_experts), axis=0
    )
    inputs = jnp.reshape(inputs, expert_selection.shape + (-1,))

    # Combine weights for each token.
    with jax.named_scope("weight_sum"):
      if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
        # For Llama4, combine using weights of 1 for selected experts
        expert_affinity = jnp.ones_like(expert_affinity)
      inputs = jnp.einsum(
          "...KE,...K -> ...E",
          jnp.astype(inputs, jnp.float32),
          jnp.astype(expert_affinity, jnp.float32),
          precision=jax.lax.Precision(self.config.matmul_precision),
      )

    return jnp.astype(inputs, self.dtype)

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
  ):
    """Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts
    located on the current shard. It groups the tokens by their assigned local
    expert index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert
        across all the batch shards. Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to
        num_expert_parallelism - 1).
      is_offset: If `True`, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If
        `False`, assumes that `inputs` corresponding to the shard's experts
        start from the beginning of the tensor but need to be permuted by expert
        ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when
        `is_offset` is `True`. Shape `[total_tokens_for_this_shard]`.

    Returns:
      A tuple containing:
      - sorted_inputs: Input data permuted local expert ID.
      - sorted_indices: Indices used to permute the inputs.
      - local_group_size: Number of tokens assigned to each local expert on this
          shard.
      - sorted_experts_ids: expert ID corrsponding to each token of the permuted
        inputs.
    """

    # Slice the count of local expert IDs in each batch shard.
    # all_shard_local_sizes.shape: [expert_shard, local_expert_size]
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        start_index=shard_index * local_expert_size,
        slice_size=local_expert_size,
        axis=1,
    )
    local_sizes = all_shard_local_sizes.reshape(-1)

    # Total count of the local expert IDs is the sum of the counts across all
    # batch shards, since all batch shards will send their contributions to the
    # current expert shard.
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    # In this case, the data that needs to be processed by the local shard
    # does not start from row 0 but actually starts at
    # `(jnp.concatenate((jnp.array([0]),
    #   jnp.cumsum(local_group_sizes[:-1]))[shard_id])`.
    # This happens if batches (`inputs`) are replicated across expert shards and
    # pre-sorted by global Expert ID (via permute()).
    if is_offset:
      divided_assignments = jnp.floor_divide(
          global_sorted_experts, local_expert_size
      )
      expert_indices = jnp.where(
          divided_assignments == shard_index,
          jnp.mod(global_sorted_experts, local_expert_size),
          local_expert_size,
      )

    # In this case the `input` data has been received from the batch shards and
    # needs to be reorganized in order of local Expert IDs.
    else:
      base_indices = jnp.mod(
          jnp.arange(local_sizes.shape[0]), local_expert_size
      )
      expert_indices = jnp.repeat(
          base_indices, local_sizes, total_repeat_length=inputs.shape[0]
      )

    sorted_indices = jnp.argsort(expert_indices)
    sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
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
          zero_row = jnp.zeros(
              (1,) + input_array.shape[1:], dtype=input_array.dtype
          )
          array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
          cumulated_array = jnp.cumsum(
              array_with_zeros, axis=0, dtype=input_array.dtype
          )
          return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the traget output
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown tranform array strategy: {strategy}")

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
          output_offset = jnp.concatenate(
              (jnp.array([0]), jnp.cumsum(input_array[:-1]))
          )[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        # The amount that each shard receives from all other shards is
        # equivalent to the group sizes (aka input_array).
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the traget output
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

  def route(
      self,
      inputs: jax.Array,
      expert_selection: jax.Array,
  ) -> tuple[jax.Array, jax.Array, Any]:
    """Route the inputs to the corresponding experts.

    Args:
      inputs: `(..., emb_dim)` array of tokens.
      expert_selection: `(..., num_experts_per_tok)` array of expert ids.

    Returns:
      - routed_tokens: `(..., emb_dim)` array of tokens which have been
        duplicated `num_experts_per_tok` times each and routed to the
        corresponding expert shard.
      - group_sizes: `(num_experts,)` array of group sizes for `routed_tokens`.
      - routing_info: A dictionary containing additional information about the
        routing process, so that the tokens can be unrouted later.
    """

    # Create a copy of each token for each expert that it is assigned to. Sort
    # the copies by expert id and track the size of each expert's group.
    inputs, group_sizes = self.permute(inputs, expert_selection)

    if self.get_expert_parallelism_size() == 1:  # No expert parallelism.
      return inputs, group_sizes, None

    else:  # Route to the corresponding expert shards.
      batch_axis = "expert" if self.is_batch_sharded_by_expert() else "data"
      expert_shard_id = jax.lax.axis_index("expert")

      # Get group sizes for all shards.
      local_expert_size = (
          self.config.num_experts // self.get_expert_parallelism_size()
      )
      all_shards_group_sizes = None
      reshaped_group_sizes = jnp.sum(
          group_sizes.reshape(-1, local_expert_size), axis=1
      )
      global_group_sizes = group_sizes

      if self.is_batch_sharded_by_expert():
        all_shards_group_sizes = jax.lax.all_gather(
            reshaped_group_sizes, axis_name=batch_axis
        )
        input_offsets, send_sizes, output_offsets, recv_sizes = (
            RoutedMoE.get_all_to_all_params(
                all_shards_group_sizes,
                expert_shard_id,
                self.get_expert_parallelism_size(),
            )
        )
        # TODO(ranran): For better performance, we could update output buffer
        # to a smaller size to replace self.get_expert_parallelism_size() for
        # efficiency, or we could apply capacity_factor for excessive experts.
        # Note: Reducing buffer increase the risk of token dropping under
        # unbalanced distribution.
        buffer_size = int(
            self.get_expert_parallelism_size()
            * self.config.per_device_batch_size
            * self.config.max_target_length
            * self.config.num_experts_per_tok
        )
        output_shape = jnp.zeros(
            (buffer_size, self.config.emb_dim), dtype=inputs.dtype
        )

        inputs = jax.lax.ragged_all_to_all(
            inputs,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        global_group_sizes = jax.lax.all_gather(group_sizes, axis_name="expert")
        inputs, local_sorted_indices, group_sizes, _ = RoutedMoE.local_permute(
            inputs,
            global_group_sizes,
            local_expert_size,
            shard_index=expert_shard_id,
        )
      else:
        inputs, local_sorted_indices, group_sizes, _ = RoutedMoE.local_permute(
            inputs,
            global_group_sizes[None, :],
            local_expert_size,
            shard_index=expert_shard_id,
            is_offset=True,
            global_sorted_experts=self.selected_experts(
                group_sizes, inputs.shape[0]
            ),
        )
      return (
          inputs,
          group_sizes,
          {
              "local_sorted_indices": local_sorted_indices,
              "all_shards_group_sizes": all_shards_group_sizes,
              "reshaped_group_sizes": reshaped_group_sizes,
          },
      )

  def unroute(
      self,
      intermediate_output: jax.Array,
      expert_affinity: jax.Array,
      expert_selection: jax.Array,
      routing_info: Any,
  ) -> jax.Array:
    """Undo `route()`.

    Args:
      intermediate_output: `(..., emb_dim)` array of output activations.
      expert_affinity: `(..., num_experts_per_tok)` array of expert affinities.
      expert_selection: `(..., num_experts_per_tok)` array of expert ids.
      routing_info: Routing information returned by `route()`.

    Returns:
      `(..., emb_dim)` array of de-duplicated output activations.
    """

    if self.get_expert_parallelism_size() == 1:
      return self.unpermute(
          intermediate_output,
          expert_selection,
          expert_affinity,
      )

    else:
      local_sorted_indices = routing_info["local_sorted_indices"]
      all_shards_group_sizes = routing_info["all_shards_group_sizes"]
      reshaped_group_sizes = routing_info["reshaped_group_sizes"]
      expert_shard_id = jax.lax.axis_index("expert")

      original_inputs_first_dim = (
          expert_selection.shape[0]
          * expert_selection.shape[1]
          * self.config.num_experts_per_tok
      )

      output_shape = jnp.zeros(
          (
              original_inputs_first_dim,
              self.config.emb_dim // self.get_tensor_parallelism_size(),
          ),
          dtype=intermediate_output.dtype,
      )
      if self.is_batch_sharded_by_expert():
        # locally unpermute back to the original order
        local_output = jnp.take(
            intermediate_output,
            indices=jnp.argsort(local_sorted_indices),
            axis=0,
        )
        input_offsets, send_sizes, output_offsets, recv_sizes = (
            RoutedMoE.get_all_to_all_params(
                jnp.transpose(all_shards_group_sizes),
                expert_shard_id,
                self.get_expert_parallelism_size(),
            )
        )
        intermediate_output = jax.lax.ragged_all_to_all(
            local_output,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
      else:
        # If batch is replicated across EP shards then each shard should send
        # 0..local_shard_size data to the other shards and receive the
        # local_shard data from all of the other shards using
        # ragged_all_to_all.
        input_offsets, send_sizes, output_offsets, recv_sizes = (
            RoutedMoE.get_all_to_all_params(
                reshaped_group_sizes,
                expert_shard_id,
                self.get_expert_parallelism_size(),
                is_batch_sharded=False,
            )
        )
        intermediate_output = jax.lax.ragged_all_to_all(
            intermediate_output,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )

      final_output = self.unpermute(
          intermediate_output,
          expert_selection,
          expert_affinity,
      )
      return final_output

  def process_tokens(
      self,
      inputs: jax.Array,
      w0: jax.Array | aqt.QTensor,
      w1: jax.Array | aqt.QTensor,
      wo: jax.Array | aqt.QTensor,
      group_sizes: jax.Array,
  ) -> jax.Array:
    """Process tokens for each expert shard.

    Here, `num_experts` refers to the number of experts on the current shard.

    Args:
      inputs: `(num_tokens, emb_dim)` array of input activations.
      w0: `(num_experts, emb_dim, mlp_dim)` array of weights.
      w1: `(num_experts, emb_dim, mlp_dim)` array of weights.
      wo: `(num_experts, mlp_dim, emb_dim)` array of weights.
      group_sizes: `(num_experts,)` array of group sizes.

    Returns:
      `(num_tokens, emb_dim)` array of output activations.
    """
    layer_w0 = adc.checkpoint_name(self.gmm(inputs, w0, group_sizes), "mlpwi_0")
    layer_w1 = adc.checkpoint_name(self.gmm(inputs, w1, group_sizes), "mlpwi_1")
    # pylint: disable=protected-access
    layer_act = linears._convert_to_activation_function(
        self.config.mlp_activations[0]
    )(layer_w0)
    intermediate_layer = jnp.multiply(layer_act, layer_w1)
    intermediate_output = adc.checkpoint_name(
        self.gmm(intermediate_layer, wo, group_sizes), "mlpwo"
    )

    if self.get_tensor_parallelism_size() > 1:
      intermediate_output = jax.lax.psum_scatter(
          intermediate_output, "tensor", scatter_dimension=1, tiled=True
      )

    return intermediate_output

  def gmm(
      self,
      inputs: jax.Array,
      kernel: jax.Array | aqt.QTensor,
      group_sizes: jax.Array,
  ) -> jax.Array:
    """Grouped matrix multiplication.

    Args:
      inputs: `(num_tokens, dim0)` array of input activations.
      kernel: `(num_experts, dim0, dim1)` array of weights.
      group_sizes: `(num_experts,)` array of group sizes.

    Returns:
      `(num_tokens, dim1)` array of output activations.
    """
    padding = inputs.shape[0] - (inputs.shape[0] % self.config.tile_batch_seq)

    def pad(x):
      """Pad `x.shape[0]` to next multiple of `self.config.tile_batch_seq`."""
      return jax.lax.pad(
          x, jnp.array(0.0, dtype=x.dtype), [(0, padding, 0), (0, 0, 0)]
      )

    def unpad(x):
      """Undo `pad(x)`."""
      return x[: x.shape[0] - padding]

    inputs = pad(inputs.astype(self.dtype))
    kernel = kernel.astype(self.dtype)

    if self.quant is None:
      lhs_quantize_dtype, rhs_quantize_dtype = None, None
    else:
      quant_dg = self.quant.quant_dg
      lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
      rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()

    if self.config.megablox:
      output = mblx.gmm(
          lhs=inputs,
          rhs=kernel,
          group_sizes=group_sizes,
          preferred_element_type=jnp.bfloat16,
          tiling=(
              min(inputs.shape[0], self.config.tile_batch_seq),
              min(inputs.shape[1], self.config.tile_activation_dim),
              min(kernel.shape[2], self.config.tile_weight_dim),
          ),
          lhs_quantize_dtype=lhs_quantize_dtype,
          rhs_quantize_dtype=rhs_quantize_dtype,
      )
    else:
      if isinstance(kernel, aqt.QTensor):
        if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
          raise ValueError(
              "Unsupported usecase for ragged_dot with quantized kernel."
          )
        else:
          rhs_inputs = kernel.qvalue
      else:
        rhs_inputs = kernel

      output = jax.lax.ragged_dot(
          lhs=inputs,
          rhs=rhs_inputs,
          group_sizes=group_sizes,
          preferred_element_type=jnp.bfloat16,
      )

      if isinstance(kernel, aqt.QTensor):
        # Multiply outputs by the kernel scale.
        scales = jnp.take(
            kernel.scale[0].squeeze(),
            indices=self.selected_experts(group_sizes, inputs.shape[0]),
            axis=0,
        )
        output *= pad(scales)

    return unpad(output)

  def sparse_matmul(
      self,
      inputs: jax.Array,
      gate_logits: jax.Array,
      pre_bias_logits: jax.Array,
      w0_kernel: jax.Array,
      w1_kernel: jax.Array,
      wo_kernel: jax.Array,
  ) -> ctypes.Array:
    """Perform sparse matrix multiplication of inputs and experts.

    Currently, we support data, tensor, and expert parallelism with Megablox.
    We all gather the input activations over tensor parallelism to follow
    https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.

    Args:
      inputs: `(..., emb_dim)` array of input activations.
      gate_logits: `(..., num_experts)` array of routing activations.
      pre_bias_logits: `(..., num_experts)` array of routing activations.
      w0_kernel: `(num_experts, emb_dim, mlp_dim)` array of weights.
      w1_kernel: `(num_experts, emb_dim, mlp_dim)` array of weights.
      wo_kernel: `(num_experts, mlp_dim, emb_dim)` array of weights.

    Returns:
      `(..., emb_dim)` array of output activations of same shape as `inputs`.
    """

    # Check if the batch should be sharded by expert and whether the batch_size
    # supports this. For example, for interleaved inference, prefill always has
    # batch_size=1 while decode can have batch_size > 1.
    if self.is_batch_sharded_by_expert() and inputs.shape[0] > 1:
      batch_logical_axis = "activation_batch"
    else:
      batch_logical_axis = "activation_batch_no_exp"

    activation_map = (batch_logical_axis, "activation_length", None)

    def weight_pspec(kernel, axis_map):
      pspec = nn.logical_to_mesh_axes(axis_map)
      if isinstance(kernel, aqt.QTensor):
        return aqt.partition_spec(pspec, (1,), kernel.dtype, use_bias=False)
      return pspec

    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(
            nn.logical_to_mesh_axes(activation_map),
            nn.logical_to_mesh_axes(activation_map),
            nn.logical_to_mesh_axes(activation_map) if self.is_ds3() else None,
            weight_pspec(w0_kernel, ("exp", None, "mlp")),
            weight_pspec(w1_kernel, ("exp", None, "mlp")),
            weight_pspec(wo_kernel, ("exp", "mlp", None)),
        ),
        out_specs=(
            nn.logical_to_mesh_axes(
                (batch_logical_axis, "activation_length", "activation_embed")
            )
        ),
        check_rep=False,
    )
    def wrapper(inputs, logits, pre_bias_logits, w0, w1, wo):
      # Compute the token-to-expert affinity and selection.
      expert_affinity, expert_selection = self.get_topk(logits, pre_bias_logits)

      # Pre-scale the inputs for the LLAMA4 decoder block.
      if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
        inputs *= jax.nn.sigmoid(jnp.astype(expert_affinity, jnp.float32))

      # Gather and sort tokens for each expert shard.
      inputs, group_sizes, routing_info = self.route(inputs, expert_selection)

      # Process tokens for each expert shard.
      intermediate_output = self.process_tokens(inputs, w0, w1, wo, group_sizes)

      final_output = self.unroute(
          intermediate_output, expert_affinity, expert_selection, routing_info
      )

      return final_output

    return wrapper(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
    )

  def reshape_and_update_weights(self, weights, indices):
    """reshape and update weights."""
    # input of weights and indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros(
        (weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype
    )
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
    top_k_indices = jnp.reshape(
        top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2])
    )

    tokens_per_batch = sub_seq * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts)
            * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(
        "Applying potential token dropping with a batch expert_capacity of"
        f" {expert_capacity_per_batch}"
    )

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
    expert_mask = jax.nn.one_hot(
        top_k_indices, num_classes=self.num_experts, dtype=jnp.int32
    )
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = nn.with_logical_constraint(
        expert_mask_fused, ("activation_batch", None, None, None)
    )
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count,
        ("activation_batch", "activation_length", None, None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(
        expert_token_count, expert_capacity_per_batch
    )
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
    combined_expert_token_position = (
        jnp.sum(expert_token_position, axis=3) * combined_expert_mask
    )
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
            math.ceil(tokens_per_batch / self.num_experts)
            * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(
        "Applying potential token dropping with a batch expert_capacity of"
        f" {expert_capacity_per_batch}"
    )

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
    expert_mask = jax.nn.one_hot(
        top_k_indices, num_classes=self.num_experts, dtype=jnp.int32
    )
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, seq_len * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = nn.with_logical_constraint(
        expert_mask_fused, ("activation_batch", None, None)
    )
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count,
        ("activation_batch", "activation_length", None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(
        expert_token_count, expert_capacity_per_batch
    )
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = (
        jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    )
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
  def load_balance_loss(self, top_k_indices, logits) -> ctypes.Array:
    """Compute the load balance loss."""
    expert_mask = jax.nn.one_hot(
        top_k_indices, num_classes=self.num_experts, dtype=jnp.int32
    )
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = (
        jnp.mean(density * density_prob)
        * (self.num_experts**2)
        * self.config.load_balance_loss_weight
    )
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
      self, kernel: ctypes.Array, kernel_axes: Tuple[Optional[str], ...]
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
  ) -> tuple[ctypes.Array, Optional[ctypes.Array]]:
    """Dense matrix multiplication."""
    # gate_logits: batch, length, expert
    gate_logits = nn.with_logical_constraint(
        gate_logits, ("activation_batch", "activation_length", None)
    )
    if self.config.model_name.startswith("deepseek3"):
      # pre_bias_logits is `None` for non-DeepSeek v3 models.
      pre_bias_logits = nn.with_logical_constraint(
          pre_bias_logits, ("activation_batch", "activation_length", None)
      )
    top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits)
    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      router_scores = jnp.astype(
          jax.nn.sigmoid(jnp.astype(top_k_weights, jnp.float32)), jnp.bfloat16
      )
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)

    if self.config.model_call_mode != "inference":
      softmax_probs = jax.nn.softmax(
          gate_logits.astype(jnp.float32), axis=-1
      ).astype(self.dtype)
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
        mask_axes = ("activation_batch", "activation_length", None, None)
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
        softmax_probs = jax.nn.softmax(
            gate_logits.astype(jnp.float32), axis=-1
        ).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(
            top_k_indices, softmax_probs
        )
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = (
              "activation_length",
              "activation_batch",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_length",
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
              "activation_length",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_batch",
              "activation_length",
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
        dispatch = self.get_einsum(
            rhs_mesh_axes=mask_axes, einsum_name=DISPATCH
        )(
            dispatch_eimsum,
            inputs,
            dispatch_mask,
            precision=self.matmul_precision(),
        )
        if cp > 1:
          dispatch = nn.with_logical_constraint(
              dispatch,
              (
                  None,
                  "activation_batch_no_exp",
                  "activation_length",
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
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(
            w0_kernel, w0_kernel_axes
        )
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            mlp_up_einsum,
            dispatch,
            w0_kernel,
            precision=self.matmul_precision(),
        )

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = nn.with_logical_constraint(
            layer_w0,
            mlp_axis,
        )
        layer_w0 = adc.checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(
            w1_kernel, w1_kernel_axes
        )
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            mlp_up_einsum,
            dispatch,
            w1_kernel,
            precision=self.matmul_precision(),
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = nn.with_logical_constraint(
            layer_w1,
            mlp_axis,
        )
        layer_w1 = adc.checkpoint_name(layer_w1, "mlpwi_1")
      # pylint: disable=protected-access
      layer_w0_act = linears._convert_to_activation_function(
          self.config.mlp_activations[0]
      )(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(
            wo_kernel, wo_kernel_axes
        )
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            mlp_down_einsum,
            layer_multiply,
            wo_kernel,
            precision=self.matmul_precision(),
        )
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
            precision=self.matmul_precision(),
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
      inputs = nn.with_logical_constraint(
          inputs, ("activation_batch", "activation_length", "activation_embed")
      )
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH",
            inputs,
            w0_kernel,
            precision=self.matmul_precision(),
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = adc.checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH",
            inputs,
            w1_kernel,
            precision=self.matmul_precision(),
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = adc.checkpoint_name(layer_w1, "mlpwi_1")
      # pylint: disable=protected-access
      layer_w0_act = linears._convert_to_activation_function(
          self.config.mlp_activations[0]
      )(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM",
            layer_multiply,
            wo_kernel,
            precision=self.matmul_precision(),
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = adc.checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("w_sum"):
        if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
          weights = self.reshape_and_update_weights(
              jnp.ones_like(top_k_weights), top_k_indices
          )
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,  # pylint: disable=undefined-variable,possibly-used-before-assignment
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
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    """Retrieve quantized weights."""
    # This is called only during tracing. This is to invoke creation of
    # quantized tensor inside AqtEinsum.  After jit, this will become no-op and
    # will not affect performance.
    _ = self.dense_matmul(
        inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
    )

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"][
        "frozen"
    ]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"][
        "frozen"
    ]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"][
        "frozen"
    ]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  @nn.compact
  def __call__(
      self, inputs: ctypes.Array
  ) -> tuple[ctypes.Array, Optional[ctypes.Array]]:
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits, pre_bias_logits = GateLogit(
        self.num_experts,
        model_name=cfg.model_name,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        name="gate",
        use_bias=cfg.routed_bias,
        score_func=cfg.routed_score_func,
        matmul_precision=cfg.matmul_precision,
    )(inputs)

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(
        cfg.num_experts, cfg.emb_dim, self.intermediate_dim
    )
    if cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
        )
      return self.sparse_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
      ), None
    else:
      return self.dense_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
      )


class RoutedAndSharedMoE(nn.Module):
  """Implements a block which combines shared and routed experts.

  Attributes:
    config: Model configs.
    mesh: device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: ctypes.Config
  mesh: jax.sharding.Mesh
  kernel_init: attentions.NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  weight_dtype: ctypes.DType = jnp.float32
  dtype: ctypes.DType = jnp.float32
  quant: Optional[quantizations.AqtQuantization] = None

  @nn.compact
  def __call__(self, inputs: ctypes.Array) -> tuple[ctypes.Array, Optional[ctypes.Array]]:
    cfg = self.config
    # NOTE: the naming mismatch here is to ensure reverse compatibility with
    # existing checkpoints. The `name` represents the weight name in
    # JAX/checkpoints and so the class name is just for readability.
    routed_experts, _ = RoutedMoE(
        name="MoeBlock_0",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(
            1.0, "fan_in", "truncated_normal"
        ),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
    )(inputs)

    shared_experts = linears.MlpBlock(
        intermediate_dim=cfg.shared_experts * cfg.moe_mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="shared_experts",
        config=cfg,
        quant=self.quant,
    )(inputs)

    return routed_experts + shared_experts, None
