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
from jax.experimental import shard_map
import max_logging

try:
  from jax.experimental.pallas.ops.tpu import megablox as mblx
except ImportError:
  max_logging.log("JAX megablox is available for TPU only.")
  pass

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization

BATCH = "activation_batch"

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
    output = compute_dot_general(inputs, kernel, axis, contract_ind)

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
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    x = checkpoint_name(x, "mlpwi")
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, (BATCH, "activation_length", "activation_mlp"))
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
    weight_dtype: Type for the weights.
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

  def generate_kernels(self, num_experts, emb_dim, mlp_dim):

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')

    # The first axes is expert
    kernel_axes = (None, 'embed', 'mlp')
    wo_kernel_axes = (None, 'mlp', 'embed')

    w0_kernel = self.param(
        'wi_0',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, emb_dim, mlp_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    w0_kernel = jnp.asarray(w0_kernel, self.dtype)
    w1_kernel = self.param(
        'wi_1',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, emb_dim, mlp_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)
    wo_kernel = self.param(
        'wo',
        nn.with_logical_partitioning(kernel_init, wo_kernel_axes),
        (num_experts, mlp_dim, emb_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  def permute(self, inputs, gate_logits, emb_dim):
    """Permute tokens to group by expert to fit gmm call."""

    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_2d = jnp.reshape(inputs, (-1, emb_dim))
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(self.weight_dtype), axis=-1).astype(self.dtype)
    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts) 
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size

  def unpermute(self, intermediate, sorted_selected_experts, weights):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(unsort_intermediate, (-1, self.num_experts_per_tok, self.config.emb_dim)) 
    with jax.named_scope("weight_sum"):
      output = jnp.einsum("BKE,BK -> BE", reshaped_intermediate, reshaped_weights)
    return output.reshape(-1, self.config.max_target_length, self.config.emb_dim).astype(self.dtype) 

  def megablox(self, inputs, gate_logits, config, w0_kernel, w1_kernel, wo_kernel):
    # TODO(ranran): need to changes in JAX repo to enable optimized tile_size 
    #               instead of the static default tile_size (512, 512, 512)
    tile_size = (512, 512, 512)

    def gmm(inputs, kernel, group_sizes):
      hs_shape = inputs.shape
      # pad lengh is the 1st dimension of tiling size in gmm call
      pad_length = 512
      if hs_shape[0] % pad_length:
        pad_length = pad_length - hs_shape[0] % pad_length
        inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0,0,0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.weight_dtype)
      output = mblx.gmm(lhs=inputs,
                        rhs=kernel,
                        group_sizes=group_sizes,
                        preferred_element_type=jnp.bfloat16,
                        tiling=tile_size)

      if hs_shape[0] % pad_length:
        output = output[:hs_shape[0]]
      return output

    # Currently, we only support data parallelism with Megablox (sharding on batch dimensions)
    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(
              (nn.logical_to_mesh_axes((BATCH, None, None))),
              (nn.logical_to_mesh_axes((BATCH, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
          ),
        out_specs=(nn.logical_to_mesh_axes((BATCH, None, None))),
        check_rep=False,
    )
    def wrapper(x, logits, w0, w1, wo):
      x, sorted_selected_experts, weights, group_sizes = self.permute(x, logits, config.emb_dim)

      layer_w0 = gmm(x, w0, group_sizes)
      layer_w1 = gmm(x, w1, group_sizes)
      layer_act = _convert_to_activation_function(config.mlp_activations[0])(layer_w0)
      intermediate_layer = jnp.multiply(layer_act, layer_w1)
      intermediate_output = gmm(intermediate_layer, wo, group_sizes)
      output = self.unpermute(intermediate_output,
                              sorted_selected_experts,
                              weights)
      return output
    return wrapper(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name="gate")(inputs)

    top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    flattened_top_k_weights = top_k_weights.reshape(-1, self.num_experts_per_tok)

    softmax_probs = jax.nn.softmax(flattened_top_k_weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    softmax_probs = softmax_probs.reshape(gate_logits.shape[:-1] + (self.num_experts_per_tok,))

    weights = jnp.zeros_like(gate_logits)
    index_update = (jnp.arange(gate_logits.shape[0])[:, None, None], jnp.arange(gate_logits.shape[1])[:, None], top_k_indices)
    weights = weights.at[index_update].set(softmax_probs)

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts,
                                                            cfg.emb_dim,
                                                            cfg.mlp_dim)

    if cfg.megablox:
      max_logging.log("Running MoE megablox implementation.")
      return self.megablox(inputs, gate_logits, cfg, w0_kernel, w1_kernel, wo_kernel)
    else:
      max_logging.log("Running MoE matmul implementation.")
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
