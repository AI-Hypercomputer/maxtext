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

"""nxx module overrides and utility methods for LTI distillation"""

import jax
from flax import nnx
from maxtext.layers import linears, initializers
from maxtext.common.common_types import Config
from jax.sharding import Mesh, NamedSharding
import jax.numpy as jnp
from typing import Iterable, Optional

from maxtext.common.common_types import DType, ShardMode, Array
from maxtext.layers.nnx_wrappers import ToNNX
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.layers.initializers import NdInitializer, nd_dense_init
from maxtext.utils import max_logging, max_utils


class LearnToInitDecoderLayer(nnx.Module):
  """
  A generic wrapper that initializes a base decoder layer and dynamically swaps
  its DenseGeneral modules for learn-to-init distillation.

  This class instantiates a standard base decoder layer (e.g., LlamaDecoderLayer)
  and replaces specific attention projection sub-modules ("query", "key", "value",
  "out") with customized `LearnToInitDense` modules.

  Attributes:
      learn_to_init_wrapper: The instantiated base decoder layer containing the mutable NNX graph.
      config: The model configuration parameters.
      rngs: The random number generator state used for initialization.
      self_attention_module_name: The target name of the attention module to customize.
  """

  def __init__(
      self,
      base_layer_cls,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant=None,
      **kwargs,
  ):
    # Instantiate the original layer (e.g., LlamaDecoderLayer)
    self.learn_to_init_wrapper = base_layer_cls(
        config=config, model_mode=model_mode, mesh=mesh, rngs=rngs, quant=quant, **kwargs
    )

    self.config = config
    self.rngs = rngs

    self.self_attention_module_name = "self_attention"

    # replace relevant nnx modules with customized LearnToInit modules
    self._customize_attention_modules(self.learn_to_init_wrapper)

  def _customize_attention_modules(self, module: nnx.Module):
    """Replaces specific DenseGeneral modules (q, k, v projections) in the attention module."""
    attention_module = getattr(module, self.self_attention_module_name, None)
    if attention_module is None:
      return

    # Target Q, K, V projections sub module names
    target_names = ["query", "key", "value", "out"]

    use_general_linear_map = self.config.lti_use_general_linear_map
    teacher_config = self.config.teacher_config

    for name in target_names:
      child = getattr(attention_module, name, None)
      if isinstance(child, linears.DenseGeneral):
        orig_proj_shape = child.kernel.shape
        assert len(orig_proj_shape) == 3
        if name in ("query", "key", "value"):
          teacher_heads_num = teacher_config.base_num_query_heads if name == "query" else teacher_config.base_num_kv_heads
          teacher_shape = (orig_proj_shape[0], teacher_heads_num, teacher_config.head_dim)
        elif name == "out":
          teacher_shape = (teacher_config.base_num_query_heads, teacher_config.head_dim, orig_proj_shape[2])
        else:
          max_logging.warning(f"Non handled LTI projection type {name}")
          continue
        new_module = LearnToInitDense(
            in_features_shape=child.in_features_shape,
            out_features_shape=child.out_features_shape,
            C=jnp.empty(teacher_shape),
            axis=child.axis,
            weight_dtype=child.weight_dtype,
            dtype=child.dtype,
            kernel_init=child.kernel_init,
            kernel_axes=child.kernel_axes,
            quant=child.quant,
            use_bias=child.use_bias,
            shard_mode=child.shard_mode,
            matmul_precision=child.matmul_precision,
            is_output_projection=(name == "out"),
            use_general_linear_map=use_general_linear_map,
            rngs=self.rngs,  # Reuse the layer's RNG stream
        )
        # Swap the module in the mutable NNX graph
        setattr(attention_module, name, new_module)

  def __call__(self, *args, **kwargs):
    # Just forward the forward pass arguments to the base layer
    return self.learn_to_init_wrapper(*args, **kwargs)


class LearnToInitDense(nnx.Module):
  """
  A customized Dense layer used exclusively during the learn-to-init phase of distillation.

  This module replaces standard `DenseGeneral` projections within the attention mechanism.
  Instead of a single standard kernel, it computes the effective projection weights
  dynamically during the forward pass by combining learnable student parameters
  (either A and B matrices, or a general linear map W) with frozen teacher weights (C).

  The projection math adapts automatically based on whether the layer is used for
  Q/K/V projections or the final output projection.

  Attributes:
      C: The frozen, pre-trained teacher tensor.
      A: The first learnable projection matrix (used if use_general_linear_map is False).
      B: The second learnable projection matrix (used if use_general_linear_map is False).
      W: A single, general learnable linear map (used if use_general_linear_map is True).
      bias: An optional learnable bias parameter.
  """

  TENSOR_A = "A"
  TENSOR_B = "B"
  TENSOR_C = "C"
  TENSOR_W = "W"

  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      C: Optional[jax.Array] = None,  # C is assumed to be the teacher tensor
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      is_output_projection: bool = False,
      use_general_linear_map: bool = False,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = (),
      quant: None | Quant = None,
      use_bias: bool = False,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs = None,
  ):
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.is_output_projection = is_output_projection
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.quant = quant
    self.use_bias = use_bias
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.use_general_linear_map = use_general_linear_map

    self.C = nnx.Param(C, sharding=self.kernel_axes)

    kernel_shape = self.in_features_shape + self.out_features_shape
    assert len(kernel_shape) == 3, "LearnToInitDense currently only supports 3D kernels for attention."
    assert len(self.C.value.shape) == 3, "The teacher tensor C must be 3D."

    if self.is_output_projection:
      # For output projection: student(u,v,b_s), teacher(x,y,b_t)
      u, v, b_s = kernel_shape
      x, y, b_t = self.C.value.shape
      assert b_s == b_t, f"Embedding dimension mismatch for output projection: {b_s} != {b_t}"
      if self.use_general_linear_map:
        self.W = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (x, y, u, v), self.weight_dtype),
            sharding=(None, None, None, None),
        )
      else:
        self.A = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (x, u), self.weight_dtype),
            sharding=(None, None),
        )
        self.B = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (v, y), self.weight_dtype),
            sharding=(None, None),
        )
    else:
      # For Q,K,V projections: student(b_s,u,v), teacher(b_t,x,y)
      b_s, u, v = kernel_shape
      b_t, x, y = self.C.value.shape

      assert b_s == b_t, f"Dimension mismatch for QKV projection: {b_s} != {b_t}"
      if self.use_general_linear_map:
        self.W = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (x, y, u, v), self.weight_dtype),
            sharding=(None, None, None, None),
        )
      else:
        self.A = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (x, u), self.weight_dtype),
            sharding=(None, None),
        )
        self.B = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (y, v), self.weight_dtype),
            sharding=(None, None),
        )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = self.out_features_shape
      self.bias = nnx.Param(
          initializers.default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          sharding=bias_axes,
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array, _initializing: bool = False, out_sharding: NamedSharding | None = None) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
            f"Input dimension {inputs.shape[ax]} at axis {ax} "
            f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    if self.C.value.shape[0] == 0:
      raise ValueError(
          "The 'C' tensor in LearnToInitDense has not been initialized. "
          "Please inject the teacher weights before training."
      )

    if self.use_general_linear_map:
      kernel = _calc_attn_weight(
          None,
          None,
          self.C,
          general_map=self.W,
          is_output_projection=self.is_output_projection,
          matmul_precision=self.matmul_precision,
      )
    else:
      kernel = _calc_attn_weight(
          self.A, self.B, self.C, is_output_projection=self.is_output_projection, matmul_precision=self.matmul_precision
      )

    if self.parameter_memory_host_offload:
      max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
      kernel = jax.device_put(kernel, max_utils.device_space())
    kernel = jnp.asarray(kernel, self.dtype)

    # out_sharding should be None for auto mesh axis
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    contract_ind = tuple(range(0, len(self.axis)))
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        None,
        _initializing,
        out_sharding,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output


def _calc_attn_weight(
    A: jax.Array | nnx.Param | None,
    B: jax.Array | nnx.Param | None,
    C: jax.Array | nnx.Param | None,
    general_map: Optional[jax.Array | nnx.Param] = None,
    is_output_projection: bool = False,
    matmul_precision: str = "default",
    scan_dim: str = "",
):
  """Computes the effective attention weights from teacher weight and learnable projection(s).
  See the description of calculate_attn_weight() below for details.
  """
  if general_map is not None:
    if is_output_projection:
      kernel = jnp.einsum(f"x{scan_dim}yb,x{scan_dim}yuv->u{scan_dim}vb", C, general_map, precision=matmul_precision)
    else:
      kernel = jnp.einsum(f"b{scan_dim}xy,x{scan_dim}yuv->b{scan_dim}uv", C, general_map, precision=matmul_precision)
    return kernel

  if is_output_projection:
    intermediate = jnp.einsum(f"x{scan_dim}yb,x{scan_dim}u->y{scan_dim}ub", C, A, precision=matmul_precision)
    kernel = jnp.einsum(f"y{scan_dim}ub,v{scan_dim}y->u{scan_dim}vb", intermediate, B, precision=matmul_precision)
  else:
    intermediate = jnp.einsum(f"b{scan_dim}xy,x{scan_dim}u->b{scan_dim}uy", C, A, precision=matmul_precision)
    kernel = jnp.einsum(f"b{scan_dim}uy,y{scan_dim}v->b{scan_dim}uv", intermediate, B, precision=matmul_precision)
  return kernel


def calculate_attn_weight(
    A: jax.Array | None,
    B: jax.Array | None,
    C: jax.Array,
    general_map: Optional[jax.Array] = None,
    is_output_projection: bool = False,
    matmul_precision: str = "default",
) -> jax.Array:
  """
  Helper function to dynamically compute the effective attention weights using `jnp.einsum`.

  Computes the kernel by contracting the frozen teacher tensor (C) with the learnable
  student representations. It handles both factorized maps (A and B) and general linear
  maps (general_map/W), adjusting the tensor contractions based on whether the module
  is an output projection or a Q/K/V projection.

  Args:
      A: The first learned factorized matrix.
      B: The second learned factorized matrix.
      C: The frozen teacher tensor.
      general_map: An optional unified learnable projection tensor used instead of A and B.
      is_output_projection: Boolean flag indicating if this computes the output projection weight.
      matmul_precision: The precision for the einsum matrix multiplications.
      scan_dim: A string representing the scan dimension for einsum (e.g., "l" for scanned layers, or "").

  Returns:
      The computed effective kernel tensor.
  """

  # In scan mode, tensors have an extra 2-nd dimension for the layer.
  # We add 'l' to the einsum string to handle this batch dimension.
  scan_dim = "l" if C.ndim == 4 else ""
  return _calc_attn_weight(
      A,
      B,
      C,
      general_map=general_map,
      is_output_projection=is_output_projection,
      matmul_precision=matmul_precision,
      scan_dim=scan_dim,
  )


def apply_lti_model_update(student_model, student_config):
  """
  Applies the finalized learn-to-init weights to the student model and cleans up the NNX graph.

  This function iterates over the `LearnToInitDense` layers in the trained student model,
  calculates their final, static effective kernels using `calculate_attn_weight`, and
  replaces the dynamically-computed LTI modules with standard kernel representations.
  It effectively collapses the learn-to-init parameterization back into a standard
  decoder architecture, modifying the `student_model` in-place.

  NOTE: works for ToNXX decoder model and layer-scan mode only

  Args:
      student_model: The trained student model to be updated in-place.
      student_config: The configuration of the student model containing parameters like `matmul_precision`.
  """

  # Access the nested ToNNX dictionary directly
  assert isinstance(student_model.decoder, ToNNX), "LTI now only supports ToNNX as the student_model's decoder type"
  lti_wrapped_node = student_model.decoder.layers["learn_to_init_wrapper"]
  attn_state_dict = lti_wrapped_node["self_attention"]

  # Iterate through known projections and compute final weights
  for proj_name in ["query", "key", "value", "out"]:
    if proj_name not in attn_state_dict:
      raise ValueError("Unsupported structure of LTI-augmented Attention module.")

    proj_params = attn_state_dict[proj_name]
    is_output_proj = proj_name == "out"

    C_param = proj_params.get(LearnToInitDense.TENSOR_C)

    if C_param is None:
      raise ValueError("Attention LTI-augmented module has no C parameter.")

    if LearnToInitDense.TENSOR_W in proj_params:
      max_logging.log(f"Computing final learn-to-init weight (general map) for: {proj_name}")
      W_param = proj_params[LearnToInitDense.TENSOR_W]
      final_kernel = calculate_attn_weight(
          A=None,
          B=None,
          C=C_param,
          general_map=W_param,
          is_output_projection=is_output_proj,
          matmul_precision=student_config.matmul_precision,
      )
    elif LearnToInitDense.TENSOR_A in proj_params and LearnToInitDense.TENSOR_B in proj_params:
      max_logging.log(f"Computing final learn-to-init weight for: {proj_name}")
      A_param = proj_params[LearnToInitDense.TENSOR_A]
      B_param = proj_params[LearnToInitDense.TENSOR_B]
      final_kernel = calculate_attn_weight(
          A=A_param,
          B=B_param,
          C=C_param,
          is_output_projection=is_output_proj,
          matmul_precision=student_config.matmul_precision,
      )
    else:
      continue

    # 3. Overwrite C with the final computed kernel
    C_param.set_value(final_kernel)

    # 4. Standardize the structure by placing it under the 'kernel' key
    proj_params["kernel"] = C_param

    # 5. Clean up the LTI-specific parameters using .pop()
    # Using pop(key, None) avoids KeyErrors if a tensor was omitted or already shared/deleted
    proj_params.pop(LearnToInitDense.TENSOR_W, None)
    proj_params.pop(LearnToInitDense.TENSOR_A, None)
    proj_params.pop(LearnToInitDense.TENSOR_B, None)
    proj_params.pop(LearnToInitDense.TENSOR_C, None)

  # unpack the learn_to_init_wrapper to match the standard model structure
  del student_model.decoder.layers["learn_to_init_wrapper"]
  student_model.decoder.layers.update(lti_wrapped_node)
