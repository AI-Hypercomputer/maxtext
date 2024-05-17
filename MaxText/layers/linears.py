#  Copyright 2023 Google LLC
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

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import common_types
from layers import initializers
from layers import normalizations
from layers import quantizations
import numpy as np
from jax.ad_checkpoint import checkpoint_name
import megablox as mblx
from jax.experimental import shard_map
from jax.sharding import Mesh


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization


def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[str, ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    def compute_dot_general(inputs, kernel, axis, contract_ind):
      """Computes a dot_general operation that may be quantized."""
      dot_general = lax.dot_general
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls()
        dot_general = dot_general_cls()
      return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))

    # print(f"for loop.......")
    # print(f"features: {features}")
    # print(f"axis: {axis}")
    # print(f"kernel_shape: {kernel_shape}")
    # print(f"kernel_in_axis: {kernel_in_axis}")
    # print(f"kernel_out_axis: {kernel_out_axis}")
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
    # print(f"inputs.shape: {inputs.shape}")
    # print(f"kernel.shape: {kernel.shape}")
    output = compute_dot_general(inputs, kernel, axis, contract_ind)
    # print(f"output.shape: {output.shape}")

    if self.use_bias:
      bias_axes, bias_shape = self.kernel_axes[-len(features) :], kernel_shape[-len(features) :]
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ("relu",)
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = self.get_norm_layer()(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = DenseGeneral(
          (len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="wi",
          quant=self.quant,
          use_bias=self.use_bias,
      )(inputs)
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        x = DenseGeneral(
            self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
        )(inputs)
        # print("dense_name", dense_name)
        # print("before activation", x)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    # print("activations", activations)
    x = functools.reduce(operator.mul, activations)
    x = checkpoint_name(x, "mlpwi")
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output


class MoeBlock(nn.Module):
  """Mixture of Experts (MoE) block.

  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    dtype: Type for the dense layer.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32

  def generate_kernels(self, num_experts, base_emb_dim, mlp_dim, inputs):
    
    features = _canonicalize_tuple(mlp_dim)
    wo_features = _canonicalize_tuple(inputs.shape[-1])
    axis = _canonicalize_tuple(-1)
    axis = _normalize_axes(axis, inputs.ndim)
    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    wo_kernel_shape = tuple(inputs.shape[ax] for ax in axis) + wo_features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    wo_kernel_out_axis = np.arange(len(axis), len(axis) + len(wo_features))

    # print("mega.......")
    # print(f"features: {features}")
    # print(f"axis: {axis}")
    # print(f"kernel_shape: {kernel_shape}")
    # print(f"kernel_shape_0: {wo_kernel_shape}")
    # print(f"kernel_in_axis: {kernel_in_axis}")
    # print(f"kernel_out_axis: {kernel_out_axis}")
    # print(f"wo_kernel_out_axis: {wo_kernel_out_axis}")

    # kernel_in_axis = np.arange(1)
    # kernel_out_axis = np.arange(1, 2)
    kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')

    kernel_axes = ('exp', 'embed', 'mlp')
    wo_kernel_axes = ('exp', 'mlp', 'embed')
    
    w0_kernel = self.param(
        'wi_0',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, base_emb_dim, mlp_dim),
        self.weight_dtype,
        np.arange(1),
        np.arange(1, 2),
      )
    w0_kernel = jnp.asarray(w0_kernel, self.dtype)
    w1_kernel = self.param(
        'wi_1',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, base_emb_dim, mlp_dim),
        self.weight_dtype,
        np.arange(1),
        np.arange(1, 2),
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)
    wo_kernel = self.param(
        'wo',
        nn.with_logical_partitioning(kernel_init, wo_kernel_axes),
        (num_experts, mlp_dim, base_emb_dim),
        self.weight_dtype,
        np.arange(1),
        np.arange(1, 3),
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  # def call_gmm(self,
  #              inputs,
  #              num_experts,
  #              base_emb_dim,
  #              mlp_dim,
  #              mlp_activation,
  #              group_sizes):

  #   @functools.partial(
  #       shard_map.shard_map,
  #       mesh=self.mesh,
  #       in_specs=(
  #             (nn.logical_to_mesh_axes((None, None))),
  #             (nn.logical_to_mesh_axes((None, None, None))),
  #             (nn.logical_to_mesh_axes((None,))),
  #         ),
  #       out_specs=(nn.logical_to_mesh_axes((None, None))),
  #       # in_specs=(
  #       #       (nn.logical_to_mesh_axes(("activation_batch_length", "embed"))),
  #       #       (nn.logical_to_mesh_axes((None, "activation_embed", "mlp"))),
  #       #       (nn.logical_to_mesh_axes((None,))),
  #       #   ),
  #       # out_specs=(nn.logical_to_mesh_axes(("activation_batch_length", "mlp"))),
  #       check_rep=False,
  #   )
  #   def gmm(inputs, kernel, group_sizes):
  #     hs_shape = inputs.shape
  #     # pad lengh is the 1st dimension of tiling size in gmm call
  #     pad_length = 512
  #     if hs_shape[0] % pad_length:
  #       # padding
  #       pad_length = pad_length - hs_shape[0] % pad_length

  #       inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0,0,0)])
  #       inputs = inputs.astype(self.dtype)

  #     # print(f"{inputs.shape} {kernel.shape} {group_sizes.shape}")
  #     #jax.debug.print("{group_sizes}", group_sizes=group_sizes)
  #     inputs = inputs.astype(self.dtype)
  #     kernel = kernel.astype(self.weight_dtype)
  #     output = mblx.gmm(lhs=inputs, 
  #                       rhs=kernel, 
  #                       group_sizes=group_sizes,
  #                       tiling=(512, 512, 512))
      
  #     if hs_shape[0] % pad_length:
  #       output = output[:hs_shape[0]]

  #     return output
  
  #   w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(num_experts,base_emb_dim,mlp_dim,inputs.ndim)

  #   # jax.debug.print("group sizes: {group_sizes}", group_sizes=group_sizes)
  #   layer_1 = gmm(inputs, w0_kernel, group_sizes)
  #   layer_2 = gmm(inputs, w1_kernel, group_sizes)
  #   layer_1_act = _convert_to_activation_function(mlp_activation)(layer_1)
  #   intermediate_layer = jnp.multiply(layer_1_act, layer_2)
  #   output = gmm(intermediate_layer, wo_kernel, group_sizes)
  #   print("running gmm")
  #   return output

  # def permute(self, inputs, gate_logits, base_emb_dim):
  #   inputs_2d = jnp.reshape(inputs, (-1, base_emb_dim))
  #   weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
  #   weights = jax.nn.softmax(weights.astype(self.weight_dtype), axis=-1).astype(self.dtype)
  #   flatten_selected_experts = jnp.ravel(selected_experts)
  #   sorted_selected_experts = jnp.argsort(flatten_selected_experts)
  #   repeat_hidden_states = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
  #   sorted_output = jnp.take(repeat_hidden_states, indices=sorted_selected_experts, axis=0).astype(self.dtype)
  #   expert_order, expert_size = jnp.unique(flatten_selected_experts, return_counts=True, size=self.num_experts)
  #   group_size = jnp.zeros(self.num_experts, dtype=jnp.int32)

  #   expert_is_unused = expert_size == 0
  #   indices = jnp.where(expert_is_unused, len(group_size), expert_order)
  #   group_size = group_size.at[indices].set(jnp.where(expert_is_unused, group_size[indices], expert_size))

  #   return sorted_output, sorted_selected_experts, weights, group_size

  # def unpermute(self, layer_3, inputs, sorted_selected_experts, weights):
  #   unsort_output = jnp.take(layer_3, indices=jnp.argsort(sorted_selected_experts), axis=0)
  #   flatten_weights = jnp.ravel(weights)
  #   combined_output = jnp.multiply(unsort_output, flatten_weights[:, None])
  #   groups = jnp.reshape(combined_output, (-1, self.num_experts_per_tok, combined_output.shape[1]))
  #   return jnp.sum(groups, axis=1).reshape(inputs.shape).astype(self.dtype)

  # @nn.compact
  # def __call__(self, inputs):
  #   cfg = self.config

  #   gate_logits = DenseGeneral(
  #           self.num_experts,
  #           dtype=self.dtype,
  #           kernel_init=self.kernel_init,
  #           kernel_axes=self.kernel_axes,
  #           name='gate')(inputs)
  #   sorted_hidden_states, sorted_selected_experts, weights, group_size = self.permute(inputs,
  #                                                                                     gate_logits,
  #                                                                                     cfg.base_emb_dim)
  
  #   intermediate_output = self.call_gmm(sorted_hidden_states,
  #                                       cfg.num_experts,
  #                                       cfg.base_emb_dim,
  #                                       cfg.mlp_dim,
  #                                       cfg.mlp_activations[0],
  #                                       group_size)

  #   output = self.unpermute(intermediate_output, inputs, sorted_selected_experts, weights)

  #   return output

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name='gate')(inputs)
    
    top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    flattened_top_k_weights = top_k_weights.reshape(-1, self.num_experts_per_tok)

    # print(f"flattened_top_k_weights.shape: {flattened_top_k_weights.shape}")
    # print(f"flattened_top_k_weights: {flattened_top_k_weights}")

    softmax_probs = jax.nn.softmax(flattened_top_k_weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    softmax_probs = softmax_probs.reshape(gate_logits.shape[:-1] + (self.num_experts_per_tok,))

    weights = jnp.zeros_like(gate_logits)
    index_update = (jnp.arange(gate_logits.shape[0])[:, None, None], jnp.arange(gate_logits.shape[1])[:, None], top_k_indices)
    weights = weights.at[index_update].set(softmax_probs)

    # print(f"weights.shape: {weights.shape}")
    # print(f"weights: {weights}")

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts,
                                                            cfg.base_emb_dim,
                                                            cfg.mlp_dim,
                                                            inputs)
    
    with jax.named_scope("wi_0"):
      layer_w0 = jnp.einsum("BLE,NEH -> BLNH", inputs, w0_kernel)
    with jax.named_scope("wi_1"):
      layer_w1 = jnp.einsum("BLE,NEH -> BLNH", inputs, w1_kernel)
    layer_w0_act = _convert_to_activation_function(cfg.mlp_activations[0])(layer_w0)
    layer_multiply = jnp.multiply(layer_w0_act, layer_w1)
    with jax.named_scope("wo"):
      intermediate_layer = jnp.einsum("BLNH,NHE -> BLNE", layer_multiply, wo_kernel)
    with jax.named_scope("w_sum"):
      output = jnp.einsum("BLNE,BLN -> BLE", intermediate_layer, weights)
    
    return output
