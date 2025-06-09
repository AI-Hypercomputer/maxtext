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


"""MoE related Layers."""

import functools
from typing import Iterable, Tuple, Union, Optional
from enum import Enum, auto
import math

import numpy as np

from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.sharding import Mesh
import jax
import jax.numpy as jnp

import flax.linen as nn

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.aqt_tensor import QTensor

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import DType, Array, Config, DecoderBlockType
from MaxText.kernels import megablox as mblx
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import quantizations
from MaxText.layers.attentions import NdInitializer, nd_dense_init
from MaxText.layers.initializers import default_bias_init
from MaxText.layers.quantizations import AqtQuantization as Quant


DISPATCH = "dispatch"
COMBINE = "combine"


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """
  Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
                 representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length, num_experts_per_tok)
                       representing the indices of the selected experts for each token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length, num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  indices = jnp.arange(num_experts).repeat(bs * seq_len)
  selected_num = bs * seq_len * num_experts_per_tok
  top_k_indices = jax.random.choice(rng_key, indices, shape=(selected_num,)).reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


class GateLogit(nn.Module):
  """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing.

  Attributes:
    features: tuple with numbers of output features.
    model_name: which model to run.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: tuple with axes to apply kernel function.
    use_bias: whether to add learnable bias in gate logit scores.
      When enabled, this bias aids expert load balancing (like in DeepSeek V3),
      and is not part of the loss calculation.
    score_func: scoring function for output normalization before applying bias.
    quant: quantization config, defaults to None implying no quantization.
    matmul_precision: precision for JAX functions.
  """

  features: Union[Iterable[int], int]
  model_name: str
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[Optional[str], ...] = ()
  use_bias: bool = False
  score_func: str = ""
  quant: Optional[Quant] = None
  matmul_precision: str = "default"

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[Array, Optional[Array]]:

    features = linears._canonicalize_tuple(self.features)
    axis = linears._canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = linears._normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      kernel = jnp.zeros(kernel_shape)
    else:
      kernel = self.param(
          "kernel",
          nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    output = linears._compute_dot_general(
        inputs, kernel, self.kernel_axes, axis, contract_ind, self.matmul_precision, self.quant
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
          nn.with_logical_partitioning(default_bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nn.Module):
  """Implements a routed MoE block.

  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    intermediate_dim: Intermediate dimension of MoE.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  intermediate_dim: int = 2048
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  # The first axes is expert
  wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
  wo_kernel_axes = ("exp", "mlp", "embed_no_exp")

  def get_expert_parallelism_size(self):
    return self.mesh.shape["expert"]

  def get_tensor_parallelism_size(self):
    return self.mesh.shape["tensor"]

  def get_context_autoregressive_parallelism_size(self):
    return self.mesh.shape["context_autoregressive"]

  def generate_kernels(self, num_experts, emb_dim, mlp_dim):
    """generates kernels"""

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    kernel_init = nd_dense_init(1.0, "fan_in", "truncated_normal")

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w0_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w0_kernel = self.param(
          "wi_0",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )

    w0_kernel = jnp.asarray(w0_kernel, self.dtype)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w1_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w1_kernel = self.param(
          "wi_1",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      wo_kernel = jnp.zeros((num_experts, mlp_dim, emb_dim))
    else:
      wo_kernel = self.param(
          "wo",
          nn.with_logical_partitioning(kernel_init, self.wo_kernel_axes),
          (num_experts, mlp_dim, emb_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  def get_topk(self, gate_logits, pre_bias_logits):
    """get topk. shape of top_k_weights & top_k_indices: (batch, sequence, num_experts_per_tok)"""
    if self.config.use_random_routing:
      rng = self.make_rng("random_routing")
      top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
      return top_k_weights, top_k_indices

    if self.config.model_name.startswith("deepseek3"):
      top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block == DecoderBlockType.DEEPSEEK:
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    elif self.config.decoder_block != DecoderBlockType.LLAMA4:
      top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)
    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights):
    """Scales weights according to DeepSeek's v3 reference implementation.
    https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594
    """
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def deepseek_routing(self, gate_logits, pre_bias_logits):
    """DeepSeek routing logit.

    When the configuration specifies a number of routing groups (n_routing_groups is not -1),
    it involves two-stage selection process:

    1) Group Scoring: Experts are partitioned into n_routing_groups.
    Within each group, the logits of the top-2 scoring experts are summed to create an aggregate score for the group.
    2) The top-K (topk_routing_group) groups are identified based on their aggregate scores.
    The final set of selected experts is chosen only from within these top-K groups.

    If the configuration does not specify routing groups (n_routing_groups is -1),
    using a standard top-k routing mechanism.

    The selection uses post_bias logits, but the return weigths are based on pre_bias logits.
    """
    # Reshape
    batch_size, seq_len = gate_logits.shape[0], gate_logits.shape[1]
    n = batch_size * seq_len
    gate_logits_flat = jnp.reshape(gate_logits, (n, self.num_experts))
    pre_bias_logits_flat = jnp.reshape(pre_bias_logits, (n, self.num_experts))

    if self.config.n_routing_groups != -1:
      # Enable device-limited routing
      experts_per_group = self.num_experts // self.config.n_routing_groups
      scores_grouped = jnp.reshape(gate_logits_flat, (n, self.config.n_routing_groups, experts_per_group))

      # Group selection: select top2 from each group, sum values, then select top groups
      top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
      group_scores = jnp.sum(top2_in_group_vals.astype(jnp.float32), axis=-1)
      group_idx = jax.lax.top_k(group_scores, k=self.config.topk_routing_group)[1]

      # Create masks for selected groups
      group_mask = jax.nn.one_hot(group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32)
      group_mask = jnp.sum(group_mask, axis=1)

      # Apply masks and get topk indices
      score_mask_grouped = jnp.expand_dims(group_mask, axis=-1)
      score_mask_expanded = jnp.broadcast_to(score_mask_grouped, (n, self.config.n_routing_groups, experts_per_group))
      score_mask = jnp.reshape(score_mask_expanded, (n, self.num_experts))
      negative_infinity = -jax.numpy.inf
      masked_scores = jnp.where(score_mask > 0, gate_logits_flat, negative_infinity)
      top_k_indices = jax.lax.top_k(masked_scores, k=self.num_experts_per_tok)[1]
    else:
      top_k_indices = jax.lax.top_k(gate_logits_flat, k=self.num_experts_per_tok)[1]

    # Get topk weights from pre bias logits
    top_k_weights = jnp.take_along_axis(pre_bias_logits_flat, top_k_indices, axis=-1)

    # Reshape
    top_k_indices = jnp.reshape(top_k_indices, (batch_size, seq_len, self.num_experts_per_tok))
    top_k_weights = jnp.reshape(top_k_weights, (batch_size, seq_len, self.num_experts_per_tok))
    return top_k_weights, top_k_indices

  def permute(self, inputs, gate_logits, pre_bias_logits):
    """Permute tokens to group by expert to fit gmm call."""
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = self.get_topk(gate_logits, pre_bias_logits)

    if self.config.decoder_block == DecoderBlockType.LLAMA4:
      # weights will be of shape (batch_size, seq_len, num_experts_per_tok)
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))  # weights are top_k_weights here
      # Squeeze router_scores to (batch_size * seq_len, num_experts_per_tok)
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    # Return the experts for each sorted input.
    expert_indices = jnp.arange(self.num_experts)
    sorted_experts = jnp.repeat(expert_indices, repeats=group_size, total_repeat_length=flatten_selected_experts.shape[0])
    return sorted_inputs, sorted_selected_experts, weights, group_size, sorted_experts

  def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = lax.Precision(self.config.matmul_precision)
      if self.config.decoder_block == DecoderBlockType.LLAMA4:
        # For Llama4, combine using weights of 1 for selected experts
        reshaped_weights = jnp.ones_like(reshaped_weights)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate.astype(jnp.float32),
          reshaped_weights.astype(jnp.float32),
          precision=matmul_precision,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  @staticmethod
  def local_permute(inputs, global_group_sizes, local_expert_size, shard_index, is_offset=False, global_sorted_experts=None):
    """Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts located
    on the current shard. It groups the tokens by their assigned local expert
    index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert across all the batch shards.
        Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to num_expert_parallelism - 1).
      is_offset: If True, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If False, assumes
        that `inputs` corresponding to the shard's experts start from the beginning of the tensor
        but need to be permuted by expert ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when `is_offset`
        is True. Shape `[total_tokens_for_this_shard]`.

    Returns:
      A tuple containing:
        sorted_inputs: Input data permuted local expert ID.
        sorted_indices: Indices used to permute the inputs.
        local_group_size: Number of tokens assigned to each local expert on this
          shard.
        sorted_experts_ids: expert ID corrsponding to each token of the permuted inputs.
    """

    # Slice the count of local expert IDs in each batch shard.
    # all_shard_local_sizes.shape: [expert_shard, local_expert_size]
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes, shard_index * local_expert_size, local_expert_size, axis=1
    )
    local_sizes = all_shard_local_sizes.reshape(-1)

    # Total count of the local expert IDs is the sum of the counts across all batch shards,
    # since all batch shards will send their contributions to the current expert shard.
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    # In this case, the data that needs to be processed by the local shard
    # does not start from row 0 but actually starts at
    # jnp.concatenate((jnp.array([0]), jnp.cumsum(local_group_sizes[:-1]))[shard_id]
    # This happens if batches (`inputs`) are replicated across expert shards and pre-sorted
    # by global Expert ID (via permute()).
    if is_offset:
      divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
      expert_indices = jnp.where(
          divided_assignments == shard_index, jnp.mod(global_sorted_experts, local_expert_size), local_expert_size
      )

    # In this case the `input` data has been received from the batch shards and needs to be
    # reorganized in order of local Expert IDs.
    else:
      base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
      expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

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
  def get_all_to_all_params(all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True):
    """Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all."""

    class TransformStrategy(Enum):
      INPUT_OFFSET = auto()
      SEND_SIZE = auto()
      OUTPUT_OFFSET = auto()
      RECV_SIZE = auto()

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
      """This function transforms the input array based on the specified strategy,
      preparing it for the usage with `ragged_all_to_all` API. The transformation
      determines how data is sent and received between shards.
      """
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
          # Received size in the traget output
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown tranform array strategy: {strategy}")

      # If the batch is unsharded then we send the same data slice to all other shards.
      # We also assume each shard will have the local processed inputs sorted to start from index 0.
      # Finally, len(input_array.shape) == 1 since there is only one batch shard.
      else:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # The data on each shard always starts at 0.
          return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
        elif strategy == TransformStrategy.SEND_SIZE:
          # The send amount is always the amount of data the current expert shard needs to process.
          return jnp.repeat(input_array[shard_id], num_expert_parallelism)
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # The offset in each shard will just be the start of the group which that shard is
          # responsible for.
          output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        # The amount that each shard receives from all other shards is equivalent to the group sizes
        # (aka input_array).
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the traget output
          return input_array
        else:
          raise ValueError(f"Unknown tranform array strategy: {strategy}")

    input_offsets = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.INPUT_OFFSET, is_batch_sharded)
    send_sizes = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.SEND_SIZE, is_batch_sharded)
    output_offsets = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.OUTPUT_OFFSET, is_batch_sharded)
    recv_sizes = transform_array(all_shards_group_sizes, shard_id, TransformStrategy.RECV_SIZE, is_batch_sharded)
    return input_offsets, send_sizes, output_offsets, recv_sizes

  def sparse_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):
    """Perform sparse matrix multiplication of inputs and Experts."""

    def gmm(inputs, kernel, group_sizes, expert_assignments):
      tile_size = (self.config.tile_batch_seq, self.config.tile_activation_dim, self.config.tile_weight_dim)
      PAD_LENGTH = self.config.tile_batch_seq
      hs_shape = inputs.shape
      # pad length is the 1st dimension of tiling size in gmm call
      if inputs.shape[0] != expert_assignments.shape[0]:
        raise ValueError("The number of input tokens must match the number of expert assignments!")
      pad_length = PAD_LENGTH
      if hs_shape[0] % PAD_LENGTH:
        pad_length = PAD_LENGTH - hs_shape[0] % PAD_LENGTH
        inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.dtype)

      lhs_quantize_dtype, rhs_quantize_dtype = None, None
      if self.quant is not None:
        quant_dg = self.quant.quant_dg
        lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
        rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()

      if self.config.megablox:
        m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
        output = mblx.gmm(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            preferred_element_type=jnp.bfloat16,
            tiling=(min(tile_size[0], m), min(tile_size[1], k), min(tile_size[2], n)),
            lhs_quantize_dtype=lhs_quantize_dtype,
            rhs_quantize_dtype=rhs_quantize_dtype,
        )
      else:
        rhs_inputs = kernel
        if isinstance(kernel, QTensor):
          if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
            raise ValueError("Unsupported usecase for ragged_dot with quantized kernel.")
          rhs_inputs = kernel.qvalue
        output = jax.lax.ragged_dot(
            lhs=inputs,
            rhs=rhs_inputs,
            group_sizes=group_sizes,
            preferred_element_type=jnp.bfloat16,
        )
        if isinstance(kernel, QTensor):
          # Multiply outputs by the kernely scale
          scales = jnp.take(kernel.scale[0].squeeze(), indices=expert_assignments, axis=0)
          if hs_shape[0] % PAD_LENGTH:
            scales = jax.lax.pad(scales, jnp.array(0.0, dtype=scales.dtype), [(0, pad_length, 0), (0, 0, 0)])
          output *= scales
      if hs_shape[0] % PAD_LENGTH:
        output = output[: hs_shape[0]]
      return output

    # Currently, we only support data and tensor parallelism with Megablox.
    # We all gather the input activations over tensor parallelism to follow strategy
    # in https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.

    # Check if the batch should be sharded by expert and whether the batch_size supports this.
    # E.g. for Interleaved Inference, Prefill always has batch_size=1 while Decode
    # can have batch_size > 1.
    try:
      is_batch_sharded_by_expert = (
          "expert" in tuple(filter(lambda tup: tup[0] == "activation_batch", self.config.logical_axis_rules))[0][1]
      )
    except:  # pylint: disable=bare-except
      is_batch_sharded_by_expert = False
    if is_batch_sharded_by_expert and inputs.shape[0] > 1:
      batch_logical_axis = "activation_batch"
    else:
      batch_logical_axis = "activation_batch_no_exp"

    input_partition_pspec = nn.logical_to_mesh_axes((batch_logical_axis, None, None))
    gate_logits_pspec = nn.logical_to_mesh_axes((batch_logical_axis, None, None))
    if self.config.model_name.startswith("deepseek3"):
      pre_bias_logits_pspec = nn.logical_to_mesh_axes((batch_logical_axis, None, None))
    else:
      # pre_bias_logits is None for non-DeepSeek v3 models
      pre_bias_logits_pspec = None
    w0_pspec = nn.logical_to_mesh_axes(("exp", None, "mlp"))
    w1_pspec = nn.logical_to_mesh_axes(("exp", None, "mlp"))
    wo_pspec = nn.logical_to_mesh_axes(("exp", "mlp", None))
    if isinstance(w0_kernel, QTensor):
      w0_pspec = aqt_tensor.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
    if isinstance(w1_kernel, QTensor):
      w1_pspec = aqt_tensor.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
    if isinstance(wo_kernel, QTensor):
      wo_pspec = aqt_tensor.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)

    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(input_partition_pspec, gate_logits_pspec, pre_bias_logits_pspec, w0_pspec, w1_pspec, wo_pspec),
        out_specs=(nn.logical_to_mesh_axes((batch_logical_axis, None, "activation_embed"))),
        check_rep=False,
    )
    def wrapper(x, logits, pre_bias_logits, w0, w1, wo):
      batch_size, sequence_length, _ = x.shape
      x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(x, logits, pre_bias_logits)
      expert_axis_name = "expert"
      expert_shard_id = jax.lax.axis_index(expert_axis_name)
      num_expert_parallelism = self.get_expert_parallelism_size()
      if num_expert_parallelism > 1:
        batch_axis = "expert" if is_batch_sharded_by_expert else "data"
        # get group sizes for all shards
        local_expert_size = self.config.num_experts // num_expert_parallelism
        reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
        global_group_sizes = group_sizes
        if is_batch_sharded_by_expert:
          all_shards_group_sizes = lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
          input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
              all_shards_group_sizes, expert_shard_id, num_expert_parallelism
          )
          # TODO(ranran): For better performance, we could update output buffer to a smaller
          # size to replace self.get_expert_parallelism_size() for efficiency,
          # Or we could apply capacity_factor for excessive experts.
          # Note: Reducing buffer increase the risk of token dropping under unbalanced distribution.
          buffer_size = int(
              num_expert_parallelism
              * self.config.per_device_batch_size
              * self.config.max_target_length
              * self.config.num_experts_per_tok
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
          global_group_sizes = lax.all_gather(group_sizes, axis_name=expert_axis_name)
          x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
              x, global_group_sizes, local_expert_size, shard_index=expert_shard_id
          )
        else:
          x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
              x,
              global_group_sizes[None, :],
              local_expert_size,
              shard_index=expert_shard_id,
              is_offset=True,
              global_sorted_experts=selected_experts,
          )

      layer_w0 = gmm(x, w0, group_sizes, selected_experts)
      layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      layer_w1 = gmm(x, w1, group_sizes, selected_experts)
      layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      # pylint: disable=protected-access
      layer_act = linears._convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      intermediate_layer = jnp.multiply(layer_act, layer_w1)
      intermediate_output = gmm(intermediate_layer, wo, group_sizes, selected_experts)
      intermediate_output = checkpoint_name(intermediate_output, "mlpwo")

      if self.get_tensor_parallelism_size() > 1:
        intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)

      if num_expert_parallelism > 1:
        original_inputs_first_dim = batch_size * sequence_length * self.config.num_experts_per_tok
        if sorted_selected_experts.shape[0] != original_inputs_first_dim:
          raise ValueError("original_inputs_first_dim does not match the original tensor shape!")
        output_shape = jnp.zeros(
            (original_inputs_first_dim, self.config.emb_dim // self.get_tensor_parallelism_size()),
            dtype=intermediate_output.dtype,
        )
        if is_batch_sharded_by_expert:
          # locally unpermute back to the original order
          local_output = jnp.take(intermediate_output, indices=jnp.argsort(local_sorted_indices), axis=0)
          input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
              jnp.transpose(all_shards_group_sizes), expert_shard_id, num_expert_parallelism
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
          # 0..local_shard_size data to the other shards and receive the local_shard data from
          # all of the other shards using ragged_all_to_all.
          input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
              reshaped_group_sizes, expert_shard_id, num_expert_parallelism, is_batch_sharded=False
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
          intermediate_output, sorted_selected_experts, weights, batch_size=batch_size, sequence_length=sequence_length
      )

      return output, None

    return wrapper(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)

  def reshape_and_update_weights(self, weights, indices):
    """
    reshape and update weights.

    input of weights and indices: (batch_size, seq_len, num_experts_per_tok)
    output of updated weights: (batch_size, seq_len, num_experts)
    """
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
    """subgroup mask generation for inference only"""
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    # Break sequence into subsequences (groups) of tokens, and route only within each group.
    top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))

    tokens_per_batch = sub_seq * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(f"Applying potential token dropping with a batch expert_capacity of {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to expert [0, 1] & [1, 3],
    # then expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts))
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count, ("activation_batch", "activation_length", None, None, None)
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

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    # ici_context_parallelism
    dispatch_mask = jnp.reshape(dispatch_mask, (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch))
    combine_mask = jnp.reshape(combine_mask, (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch))

    return dispatch_mask, combine_mask

  def generate_masks(self, top_k_indices, softmax_probs):
    """generate masks"""
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape

    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(f"Applying potential token dropping with a batch expert_capacity of {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to expert [0, 1] & [1, 3],
    # then expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch_size, seq_len * self.num_experts_per_tok, self.num_experts))
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count, ("activation_batch", "activation_length", None, None)
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

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits):
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(self, rhs_mesh_axes: Tuple[Optional[str], ...] = (), einsum_name=None):
    """get the Einstein summation"""

    # the check is to prevent aqteinsum as einsum op for dispatch and combine einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and einsum_name in (DISPATCH, COMBINE):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):
        # simply skip kwargs, since aqt einsum doesn't support any kwargs like precision
        is_aqt = not isinstance(self.quant, quantizations.Fp8Quantization)
        kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
        return self.quant.einsum(**kw)(*args)

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def maybe_all_gather_kernel_weight_in_expert_parallelism(self, kernel, kernel_axes):
    if self.get_expert_parallelism_size() > 1:
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead of weight_dtype
      kernel = nn.with_logical_constraint(kernel, kernel_axes)
    return kernel

  def dense_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel):
    """dense matrix multiplication"""
    # gate_logits: batch, length, expert
    gate_logits = nn.with_logical_constraint(gate_logits, ("activation_batch", "activation_length", None))
    if self.config.model_name.startswith("deepseek3"):
      # pre_bias_logits is None for non-DeepSeek v3 models
      pre_bias_logits = nn.with_logical_constraint(pre_bias_logits, ("activation_batch", "activation_length", None))
    top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits)
    is_llama4_decoder_layer = self.config.decoder_block == DecoderBlockType.LLAMA4
    if is_llama4_decoder_layer:
      router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(jnp.bfloat16)
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
    matmul_precision = lax.Precision(self.config.matmul_precision)

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
        # TODO: remove this pylint by refactoring the logic here
        dispatch_mask, combine_mask = self.generate_masks(
            top_k_indices, weights  # pylint: disable=possibly-used-before-assignment
        )
        mask_axes = ("activation_batch", "activation_length", None, None)
        input_axis = ("activation_batch", "activation_length", "activation_embed")
        dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, "activation_embed")
        mlp_axis = ("activation_exp", "activation_batch_no_exp", None, "activation_mlp")
        dispatch_eimsum = "BSM,BSEC -> EBCM"
        mlp_up_einsum = "EBCM,EMH -> EBCH"
        mlp_down_einsum = "EBCH,EHM -> EBCM"
        output_einsum = "EBCM,BSEC -> BSM"
      else:
        # todo: try replace softmax_probs with padded weights and verify with decode acc tests
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = ("activation_length", "activation_batch", None, None, None)
          input_axis = ("activation_length", "activation_batch", None, "activation_embed")
          dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_embed")
          mlp_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_mlp")
        else:
          mask_axes = ("activation_batch", "activation_length", None, None, None)
          input_axis = ("activation_batch", "activation_length", None, "activation_embed")
          dispatch_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_embed")
          mlp_axis = ("activation_exp", "activation_batch_no_exp", None, None, "activation_mlp")
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
              (None, "activation_batch_no_exp", "activation_length", None, "activation_embed"),
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

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = nn.with_logical_constraint(
            layer_w0,
            mlp_axis,
        )
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            mlp_up_einsum, dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = nn.with_logical_constraint(
            layer_w1,
            mlp_axis,
        )
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      # pylint: disable=protected-access
      layer_w0_act = linears._convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            mlp_down_einsum, layer_multiply, wo_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        if self.config.model_call_mode != "inference":
          intermediate_layer = nn.with_logical_constraint(
              intermediate_layer,
              ("activation_exp", "activation_batch_no_exp", None, "activation_embed"),
          )
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            output_einsum,
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        )
        if output.ndim == 4:
          output = jnp.reshape(output, (output.shape[0], output.shape[1] * output.shape[2], output.shape[3]))
      return output, loss
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      # pylint: disable=protected-access
      layer_w0_act = linears._convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("w_sum"):
        if is_llama4_decoder_layer:
          weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
        ).astype(self.dtype)
      return output, None

  def retrieve_quantized_weight(
      self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
  ) -> tuple[QTensor, QTensor, QTensor]:
    """retrieve quantized weight"""
    # This is called only during tracing. This is to invoke creation of quantized tensor inside AqtEinsum.
    # After jit, this will become no-op and will not affect performance.
    _ = self.dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  @nn.compact
  def __call__(self, inputs):
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

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts, cfg.emb_dim, self.intermediate_dim)
    if cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel
        )
      return self.sparse_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)
    else:
      return self.dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)


class RoutedAndSharedMoE(nn.Module):
  """Implements a block which combines shared and routed experts,

  Attributes:
    config: Model configs.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    # NOTE: the naming mismatch here is to ensure reverse compatibility with existing checkpoints.
    # The `name` represents the weight name in JAX/checkpoints and so the class name
    # is just for readability.
    routed_experts, _ = RoutedMoE(
        name="MoeBlock_0",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
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

    return routed_experts + shared_experts
