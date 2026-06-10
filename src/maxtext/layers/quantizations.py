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

"""Quantization library."""

import functools
import qwix.pallas as qpl
from typing import Tuple, Callable
from dataclasses import dataclass


import qwix
from qwix._src.core import dot_general_qt
from qwix._src.core import sparsity

import jax
import jax.numpy as jnp

from flax.linen import fp8_ops
from flax.linen import initializers as flax_initializers
import flax.linen as nn

from maxtext.common.common_types import DType, Config


# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""


@dataclass
class QwixQuantization:
  """Configures Qwix quantization github.com/google/qwix, for training only."""

  quant_mode = "train"  # needed by external call
  act_calibration_method: str = "absmax"
  weight_calibration_method: str = "absmax"
  bwd_calibration_method: str = "absmax"

  def _get_fp8_full_qwix_config(self) -> dot_general_qt.DotGeneralQtConfig:
    """Returns Qwix dot_general config for fp8_full quantization."""
    return dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,  # activation
        rhs_qtype=jnp.float8_e4m3fn,  # weight
        dlhs_grad_qtype=jnp.float8_e5m2,  # activation gradient
        drhs_grad_qtype=jnp.float8_e5m2,  # weight gradient
        lhs_calibration_method=self.act_calibration_method,
        rhs_calibration_method=self.weight_calibration_method,
        dlhs_grad_calibration_method=self.bwd_calibration_method,
        drhs_grad_calibration_method=self.bwd_calibration_method,
        tile_size=None,
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns Qwix dot_general."""
    return functools.partial(QwixDotGeneral, config=self._get_fp8_full_qwix_config())

  def einsum(self, mesh_axes: Tuple[str, ...] = (), **kwargs):
    """Returns Qwix einsum."""
    return QwixEinsum(config=self._get_fp8_full_qwix_config())


class QwixDotGeneral(nn.Module):
  """A callable class for Qwix dot_general."""

  config: dot_general_qt.DotGeneralQtConfig

  @nn.compact
  def __call__(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding=None,
  ) -> jax.Array:

    return dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, self.config)


class QwixEinsum(nn.Module):
  """A callable class for Qwix einsum."""

  config: dot_general_qt.DotGeneralQtConfig

  @nn.compact
  def __call__(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] | None = None,
      out_sharding=None,
  ) -> jax.Array:

    def custom_dot_general(*args, **kwargs):
      return dot_general_qt.dot_general_qt(*args[:3], self.config)

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=custom_dot_general,
          out_sharding=out_sharding,
      )


@dataclass
class Fp8Quantization(Quantization):
  """Configures Fp8 quantization for NVIDIA GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.Fp8DirectDotGeneralOp

  def einsum(self, dtype: DType = jnp.float32):
    return _Fp8EinsumWrapper(dtype=dtype)


class _Fp8EinsumWrapper(nn.Module):
  """Wrapper for nn.Fp8Einsum to handle computation dtype."""

  dtype: DType

  @nn.compact
  def __call__(self, eqn, lhs, rhs, **kwargs):
    # nn.Fp8Einsum determines compute dtype from rhs.
    # We cast rhs to the desired computation dtype.
    # nn.Fp8Einsum will then cast lhs to the same dtype.
    rhs = rhs.astype(self.dtype)
    return nn.Fp8Einsum(name="fp8_einsum")(eqn, lhs, rhs, **kwargs)


class Fp8Einsum(nn.Module):
  """An fp8 einsum op."""

  #: size of the amax history.
  amax_history_length: int = 1024
  #: e4m3 variants, e.g., e4m3fn, e4m3fnuz.
  e4m3_dtype: DType = jnp.float8_e4m3fn
  #: e5m2 variants, e.g., e5m2, e5m2fnuz.
  e5m2_dtype: DType = jnp.float8_e5m2
  #: computation dtype.
  dtype: DType = jnp.float32

  def setup(self) -> None:
    """init with input_amax_history, kernel_amax_history, output_grad_amax_history,
    input_scale, kernel_scale, output_grad_scale"""
    scale_args = (
        flax_initializers.ones_init(),
        jax.random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        flax_initializers.zeros_init(),
        jax.random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
    self.input_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "input_amax_history", *amax_history_args)
    self.kernel_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_amax_history", *amax_history_args)
    self.output_grad_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_amax_history", *amax_history_args)

    self.input_scale = self.variable(OVERWRITE_WITH_GRADIENT, "input_scale", *scale_args)
    self.kernel_scale = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_scale", *scale_args)
    self.output_grad_scale = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_scale", *scale_args)

  def __call__(self, eqn, *args, **kwargs):
    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.dtype
    k = jnp.asarray(k, comp_dtype)
    x = jnp.asarray(x, comp_dtype)

    x_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, x, self.input_scale.value, self.input_amax_history.value)
    k_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, k, self.kernel_scale.value, self.kernel_amax_history.value)

    y_qdq = jnp.einsum(eqn, x_qdq, k_qdq, _dot_general=fp8_ops.dot_general_with_precision)

    y = fp8_ops.out_qdq(
        comp_dtype,
        self.e5m2_dtype,
        y_qdq,
        self.output_grad_scale.value,
        self.output_grad_amax_history.value,
    )
    return y


@dataclass
class NANOOFp8Quantization(Quantization):
  """Configures NANOO Fp8 quantization for AMD MI300/MI325 GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.NANOOFp8DotGeneralOp


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization == "fp8":
    return "fp8"
  if config.quantization == "nanoo_fp8":
    return "nanoo_fp8"
  if config.quantization.startswith("te_"):
    return config.quantization
  return None


def configure_quantization(config: Config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  del quant_mode_str  # Unused since AQT is removed
  if config.use_batch_split_schedule and config.quantization:
    # The older version of batch-split that fully uses qwix quantization.
    if config.quantization == "fp8_full" and not config.use_manual_quantization:
      return QwixQuantization(
          weight_calibration_method=config.weight_quantization_calibration_method,
          act_calibration_method=config.act_quantization_calibration_method,
          bwd_calibration_method=config.bwd_quantization_calibration_method,
      )
    # The pure JAX version of batch-split that uses manual quantization for dot general.
    return None

  if config.use_qwix_quantization:
    return None
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
      return Fp8Quantization()
    elif quant_cfg == "nanoo_fp8":
      return NANOOFp8Quantization()
    elif isinstance(quant_cfg, str) and quant_cfg.startswith("te_"):
      return TransformerEngineQuantization(config)
  return None


def configure_kv_quant(config):
  if config.quantize_kvcache:
    raise ValueError(
        "KV cache quantization (quantize_kvcache=True) is no longer supported "
        "because Accurate Quantized Training (AQT) has been deprecated and removed from MaxText."
    )


class NvidaFp8Provider(qwix.QtProvider):
  """Wraps nn.Fp8DirectDotGeneralOp with Qwix's provider interface."""

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.Fp8DirectDotGeneralOp(name=op_id)(*args, **kwargs)

  def einsum(self, *args, **kwargs):
    rule, op_id = self._get_current_rule_and_op_id("einsum")
    if rule is None:
      return jnp.einsum(*args, **kwargs)
    return nn.Fp8Einsum(name=op_id)(*args, **kwargs)


class NANOOFp8Provider(qwix.QtProvider):

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.NANOOFp8DotGeneralOp(name=op_id)(*args, **kwargs)


def get_fp8_full_qwix_rule_w_sparsity(config: Config):
  sparsity_rule = None
  if config.weight_sparsity_n and config.weight_sparsity_m:
    sparsity_rule = sparsity.SparsityRule(
        weight_sparsity_n=config.weight_sparsity_n,
        weight_sparsity_m=config.weight_sparsity_m,
        weight_sparsity_update_step=config.weight_sparsity_update_step,
        weight_sparsity_start_step=config.weight_sparsity_start_step,
    )
  return [
      qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method=config.weight_quantization_calibration_method,
          act_calibration_method=config.act_quantization_calibration_method,
          bwd_calibration_method=config.bwd_quantization_calibration_method,
          additional_qt_config={"sparsity_rule": sparsity_rule},
          op_names=("dot_general", "gmm", "ragged_dot"),
      ),
  ]


def get_quantization_rule(config: Config):
  """Returns a list of qwix.QtRule from `dtype`."""

  def make_qt_rule(dtype) -> list[qwix.QtRule]:
    return [
        qwix.QtRule(
            module_path="decoder/.*layers.*",
            weight_qtype=dtype,
            act_qtype=dtype,
            bwd_qtype=dtype,
            bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
            op_names=("dot_general",),
        )
    ]

  match config.quantization:
    case "int4":
      return make_qt_rule(jnp.int4)

    case "int8":
      return make_qt_rule(jnp.int8)

    case "fp8_e5m2":
      return make_qt_rule(jnp.float8_e5m2)

    case "fp8" | "fp8_e4m3" | "fp8_gpu" | "fp8_nanoo":
      return make_qt_rule(jnp.float8_e4m3fn)

    case "fp8_full":
      return get_fp8_full_qwix_rule_w_sparsity(config)
    case "fp8_gpu":
      return make_qt_rule(jnp.float8_e4m3fn)
    case "":
      return None


def get_qt_provider(config):
  """Get quantization rules based on the config."""
  match config.quantization:
    case "int4" | "int8" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | "fp8_full":
      return qwix.QtProvider(get_quantization_rule(config))
    case "fp8_gpu":
      return NvidaFp8Provider(get_quantization_rule(config))
    case "fp8_nanoo":
      return NANOOFp8Provider(get_quantization_rule(config))
  return None


def maybe_quantize_model(model, config):
  """Quantize the model if quantization is enabled."""
  # Batch split is not using Qwix's interception feature but manual plumbing
  if config.use_qwix_quantization and not config.use_batch_split_schedule and not config.pure_nnx:
    quantization_provider = get_qt_provider(config)
    if quantization_provider:
      model = qwix.quantize_model(model, quantization_provider)
  return model


def _cast_reduced_from(arr, reduced_arr):
  aval = jax.typeof(reduced_arr)
  # In shard map
  if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
    for axis in aval.mat.reduced:
      arr = jax.lax.pcast(arr, axis, to="reduced")
    return arr
  # Outside shard map
  return jax.reshard(arr, aval.sharding)


def _make_scale_tensor(scale, arr):
  scale_tensor = jnp.full_like(arr, scale, dtype=jnp.bfloat16)
  return _cast_reduced_from(scale_tensor, arr)


def _get_max_min(target_dtype):
  if target_dtype in (jnp.int4, jnp.int8):
    return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min
  else:
    return jnp.finfo(target_dtype).max.astype(jnp.bfloat16), jnp.finfo(target_dtype).min.astype(jnp.bfloat16)


def manual_quantize(tensor, calibration_method, dtype=jnp.float8_e4m3fn):
  """Manually quantizes a tensor based on a fixed calibration method.

  Args:
    tensor: The tensor to quantize.
    calibration_method: A string specifying the calibration method. Expected
      format is "fixed,{scale},{max_val}".

  Returns:
    A qwix.QArray containing the quantized value and the scale.

  Raises:
    ValueError: If calibration_method is None or has an unexpected format.
  """
  calib_method = calibration_method
  if calib_method is None:
    raise ValueError("calibration_method cannot be None for manual quantization")
  if not calib_method.startswith("fixed"):
    raise ValueError("Only static weight/activation quantization is supported, but got" f" {calib_method}")

  parts = calib_method.split(",")
  if len(parts) != 3:
    raise ValueError(f"Unexpected format for weight calibration method: {calib_method}")

  dtype_max, dtype_min = _get_max_min(dtype)
  max_val = float(parts[2])
  scale = max_val / dtype_max
  scale = jnp.where(scale == 0, 1.0, scale)
  # scale must be converted to a tensor because grad has reduced axes.
  scale_tensor = _make_scale_tensor(scale, tensor)
  min_bound = _make_scale_tensor(dtype_min, tensor)
  max_bound = _make_scale_tensor(dtype_max, tensor)
  q_tensor = jnp.clip(tensor / scale_tensor, min_bound, max_bound).astype(dtype)

  # get scale for QArray
  scale_shape = [1] * tensor.ndim
  # It must stay fully replicated for the backward pass and Pallas.
  scale_tensor_qpl = jnp.full(scale_shape, scale, dtype=tensor.dtype)
  # wrap in QArray
  return qpl.QArray(qvalue=q_tensor, scale=scale_tensor_qpl)


class TransformerEngineQuantization(Quantization):
  """Class for TransformerEngine quantization recipes."""

  def __init__(self, config):
    """Initialize TransformerEngine quantization."""

    self.quant_mode = "train"

    if not config.quantization.startswith("te_"):
      raise ValueError(f"Invalid TransformerEngine quantization config: {config.quantization}")

    self._recipe = TransformerEngineQuantization._get_recipe(config.quantization)

  def __hash__(self):
    return hash((self.quant_mode, self._recipe))

  def __eq__(self, other):
    if not isinstance(other, TransformerEngineQuantization):
      return False
    return (self.quant_mode, self._recipe) == (other.quant_mode, other._recipe)

  @staticmethod
  def _get_recipe(recipe_name: str):
    """Get the TransformerEngine recipe based on the name."""
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    RECIPES = {
        "te_fp8_delayedscaling": recipe.DelayedScaling,
        "te_fp8_currentscaling": recipe.Float8CurrentScaling,
        "te_mxfp8": recipe.MXFP8BlockScaling,
        "te_nvfp4": recipe.NVFP4BlockScaling,  # pytype: disable=module-attr
        "te_nvfp4_no_rht": functools.partial(recipe.NVFP4BlockScaling, disable_rht=True),  # pytype: disable=module-attr
    }
    if recipe_name not in RECIPES:
      raise ValueError(f"Invalid TransformerEngine recipe: {recipe_name}")
    return RECIPES[recipe_name]()

  def get_block_size(self):
    """Get the block size for quantization for recipes that require blocks.

    If there is no block requirement for the current recipe, returns 1.
    """
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    if isinstance(self._recipe, recipe.MXFP8BlockScaling):
      return 32
    if isinstance(self._recipe, recipe.NVFP4BlockScaling):  # pytype: disable=module-attr
      return 128  # TODO(set this to 16 when unfused RHT is supported)
    return 1

  def _wrap(self, f, name=None):
    """Wraps the given function `f` to support TransformerEngine quantization.

    This method does a couple things:


    1. Wraps the given function in a context that specifies MaxText's physical mesh axes to
    TransformerEngine. This ensures our collective operations in TransformerEngine are using
    the correct axes.

    2. Wraps the given function in a Flax linen module. This module does not store any Flax
    parameters but can store Flax variables for quantizers if required by the recipe.

    3. When the wrapper is called, it provides an additional argument to the given function `f`,
    'generate_quantizer_set' as the first argument. 'generate_quantizer_set' is a function that
    can be called to generate a TransformerEngine/JAX quantizer set object used in
    TransformerEngine/JAX APIs. 'generate_quantizer_set' will generate quantizers based on the
    recipe of this TransformerEngineQuantizer object.

    Args:
      f: The function to wrap. The first argument must be 'generate_quantizer_set'.
      name: The name of this wrapped operation. If unspecified, will use `f.__name__`.

    Returns:
      A Flax linen module that wraps the given function.
    """

    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    fp8_recipe = self._recipe

    class TEWrapper(transformer_engine.jax.flax.module.TransformerEngineBase):
      """Wrapper module for TransformerEngine quantization."""

      def generate_quantizer_set(self, postfix: str = ""):
        OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
        return super().generate_quantizer_set(  # pytype: disable=wrong-keyword-args
            postfix=postfix,
            variable_collection=OVERWRITE_WITH_GRADIENT,
            quantization_checkpoint_name="quantization",
            fp8_recipe=fp8_recipe,
        )

      @nn.compact
      def __call__(self, *args, **kwargs):
        return f(self.generate_quantizer_set, *args, **kwargs)

    TEWrapper.__name__ = f"TEWrapper_{name if name else f.__name__}"

    return TEWrapper

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    def te_dot_general(generate_quantizer_set, x, kernel, dims, **kwargs):
      contracting_dims, batch_dims = dims
      assert batch_dims == ((), ()), "Batch dimensions must be empty for TransformerEngine dot."

      quantizer_set = generate_quantizer_set()
      return transformer_engine.jax.dense.dense(
          x,
          kernel,
          contracting_dims=contracting_dims,
          quantizer_set=quantizer_set,
      )

    return self._wrap(te_dot_general, "dot_general")

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""
    # quant.einsum is only required for MoE or for inference with KVCache.
    raise ValueError("Einsum is not yet supported for TransformerEngine quantization.")
