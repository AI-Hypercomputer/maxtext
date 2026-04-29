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
from jax.sharding import NamedSharding
import jax.numpy as jnp
from typing import Iterable, Optional

from maxtext.common.common_types import DType, ShardMode, Array
from maxtext.layers.nnx_wrappers import ToNNX
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.layers.initializers import NdInitializer, nd_dense_init
from maxtext.utils import max_logging, max_utils


LTI_MODIFIED_ATTENTION_PARAM_NAMES = ["query", "key", "value", "out"]
LTI_ORIGINAL_ATTENTION_PARAMS_NAME = "kernel"
LTI_LAYER_PATH_PREFIXES = ("layers_", "dense_layers_", "moe_layers_")


def apply_lti_modification(module: nnx.Module, module_name: str | None = None):
  """
  Applies Learn-To-Init structural modifications to an instantiated NNX module.
  Checks the config to determine if LTI is enabled.
  """

  config = getattr(module, "config", None)
  if not config or not getattr(config, "learn_to_init_mode", False):
    return module

  lti_layer_indices = getattr(config, "lti_layer_indices", None)
  if lti_layer_indices is not None and not getattr(config, "scan_layers", True):
    if module_name is not None and "layers_" in module_name:
      try:
        local_idx = int(module_name.split("_")[-1])
        if module_name.startswith("dense_layers_"):
          layer_idx = local_idx
        elif module_name.startswith("moe_layers_"):
          first_dense = getattr(config, "first_num_dense_layers", 0)
          layer_idx = first_dense + local_idx
        else:
          layer_idx = local_idx

        if layer_idx not in lti_layer_indices:
          max_logging.info(
              f"apply_lti_modification: skipping module={module_name} since its index "
              f"{layer_idx} is not in lti_layer_indices."
          )
          return module
      except ValueError:
        pass

  attn_module_name = config.attn_module_name

  if attn_module_name:
    max_logging.info(f"apply_lti_modification: customizing module={attn_module_name} in {module_name}")
    _customize_attention_modules(config, attn_module_name, module)
  return module


def _customize_attention_modules(config: Config, attn_module_name: str, module: nnx.Module):
  """Replaces specific DenseGeneral modules (q, k, v projections) in the attention module."""
  attention_module = getattr(module, attn_module_name, None)
  if attention_module is None:
    return

  # Target Q, K, V projections sub module names
  target_names = LTI_MODIFIED_ATTENTION_PARAM_NAMES

  use_general_linear_map = config.lti_use_general_linear_map
  teacher_config = config.teacher_config

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
          rngs=attention_module.rngs,  # Reuse the original module RNGs
      )
      # Swap the module in the mutable NNX graph
      setattr(attention_module, name, new_module)
      max_logging.info(f"Replaced {attn_module_name}.{name} with LearnToInitDense.{name}")


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

  NOTE: works for ToNXX decoder model

  Args:
      student_model: The trained student model to be updated in-place.
      student_config: The configuration of the student model containing parameters like `matmul_precision`.
  """

  assert isinstance(student_model.decoder, ToNNX), "LTI now only supports ToNNX as the student_model's decoder type"

  if student_config.attn_module_name is None:
    return

  if getattr(student_config, "scan_layers", True):
    layer_modules = [student_model.decoder.layers]
  else:
    layer_modules = []
    # Collect all possible layer names (e.g. layers_0, dense_layers_0, moe_layers_0)
    for name, module in vars(student_model.decoder).items():
      if name.startswith(LTI_LAYER_PATH_PREFIXES):
        layer_modules.append(module)

  for layer_module in layer_modules:
    attn_state_dict = layer_module.get(student_config.attn_module_name)
    if attn_state_dict is None:
      raise ValueError("LTI: attn_state_dict wasn't found in the model state dict")

    for proj_name in LTI_MODIFIED_ATTENTION_PARAM_NAMES:
      proj_params = attn_state_dict.get(proj_name)
      if proj_params is None:
        raise ValueError("Non LTI supported Attention module state.")

      is_output_proj = proj_name == "out"
      C_param = proj_params.get(LearnToInitDense.TENSOR_C)
      if C_param is None:
        continue  # Not an LTI augmented module
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
        raise ValueError("Non LTI supported Attention module state.")

      C_param.set_value(final_kernel)
      # inject as a regular parameter
      proj_params[LTI_ORIGINAL_ATTENTION_PARAMS_NAME] = C_param
      # Clean up the LTI-specific parameters
      proj_params.pop(LearnToInitDense.TENSOR_W, None)
      proj_params.pop(LearnToInitDense.TENSOR_A, None)
      proj_params.pop(LearnToInitDense.TENSOR_B, None)
      proj_params.pop(LearnToInitDense.TENSOR_C, None)
