# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Model configurations for research into backprop quantization."""
import dataclasses
import enum
import functools
import itertools

from typing import Optional, Tuple

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
import numpy as onp


DotDimensionNumbers = lax.DotDimensionNumbers

_TMP_DTYPE = jnp.float32

# pylint: skip-file

class Rounding(enum.Enum):
  """Specifies rounding mode to use."""
  # Deterministic, using round-ties-to-nearest-even
  DETERMINISTIC_RTNE = 'deterministic_rtne'
  # Use exactly the number of random bits specified by rounding_custom_bits,
  # and add trailing 0.5, shifted appropriately, to fully debias.
  STOCHASTIC_CUSTOM_WITH_HALF = 'stochastic_custom_with_half'


@dataclasses.dataclass
class HParamsNonDifferentiable:
  """Hyper-params for a single (non-differentiable) dot product."""
  # Integer precision for the matrix multiply.
  bits: int = 8
  # What kind of stochastic rounding to do for the lhs of this multiply.
  rounding_lhs: Rounding = Rounding.DETERMINISTIC_RTNE
  # What kind of stochastic rounding to do for the rhs of this multiply.
  rounding_rhs: Rounding = Rounding.DETERMINISTIC_RTNE
  rounding_custom_bits: int = 23
  use_hardware_integers: bool = True
  matmul_output_type: jnp.dtype = jnp.float32
  # Overided by using fp8 if fp8_type is specified
  fp8_type: Optional[jnp.dtype] = None

HparamBaseTypes = Optional[HParamsNonDifferentiable]


@dataclasses.dataclass
class HParams:
  """Hyper-params for a dot product and its derivatives."""
  fwd: HparamBaseTypes = None
  lhs_gradient: HparamBaseTypes = None
  rhs_gradient: HparamBaseTypes = None


def std_int_hparams(bits: int) -> HParams:
  """Standard int quantization hparams."""

  def std_hparams_nd(bits, rounding_mode):
    return HParamsNonDifferentiable(
        bits=bits,
        rounding_lhs=rounding_mode,
        rounding_rhs=rounding_mode,
        use_hardware_integers=True,
        matmul_output_type=jnp.bfloat16,
        rounding_custom_bits=22,
        fp8_type=None,
    )

  return HParams(
      fwd=std_hparams_nd(bits, Rounding.DETERMINISTIC_RTNE),
      lhs_gradient=std_hparams_nd(bits, Rounding.STOCHASTIC_CUSTOM_WITH_HALF),
      rhs_gradient=std_hparams_nd(bits, Rounding.STOCHASTIC_CUSTOM_WITH_HALF),
  )


def get_original_dot_general() ->...:
  return jax.lax.dot_general


def cast_tmp(x: jnp.ndarray) -> jnp.ndarray:
  return jax.lax.convert_element_type(x, _TMP_DTYPE)


def _rand_round(
    shape: tuple[int, ...], bits: jnp.ndarray) -> jnp.ndarray:
  """Random floats in nearly [-0.5, 0.5] of shape `shape`.

  Accepts random bits as unsigned ints which should have their 0th to
  `hparams.rand_bits` bits set to random values (the special case of 1 bit
  is reinterpreted as 8 bits, since 1 bit is converted to 8 correlated
  bits in `_random`).

  Shifts the bits to the leading mantissa bit of the `f32` representation,
  and binary ORs with `1.0`, resulting in a normal number between
  `[1.0, 1.0 + (x - 1) / x]`, where `x = 2 ** rand_bits`.

  From there we adjust to `[-0.5 + 1/(2x), 0.5 - 1/(2x)]`, but don't bother
  with scaling up (converges fine for 8bits+).

  Args:
    shape: Shape of desired f32 tensor.
    bits: Random `u8` or `u16` bits in shape `shape`

  Returns:
    Random floats seeded by `bits` in approximately the range `[-0.5, 0.5]`.
  """
  nbits = 16
  assert bits.shape == shape, (bits.shape, bits.shape)
  if nbits == 1:
    nbits = 8  # Upscaled by _random already.
  assert bits.dtype == {8: jnp.uint8, 16: jnp.uint16}[nbits], bits.dtype
  # Align bits with the mantissa of f32.
  nmant = jnp.finfo(jnp.float32).nmant
  r_bitpattern = jnp.uint32(bits) << (nmant - nbits)
  r_bitpattern = r_bitpattern | jnp.float32(1).view(jnp.uint32)
  assert r_bitpattern.dtype == jnp.uint32
  rand_floats = jax.lax.bitcast_convert_type(r_bitpattern, jnp.float32)
  shift = 2 ** (-1 - nbits)
  centered = rand_floats - (1.5 - shift)
  # NOTE(vladf): Consider stretching low-bit random numbers to cover full range.
  # return centered / (1 - 2 ** (-nbits)) -- may improve quality.
  return centered


def _random(
    shape: tuple[int, ...], rng: jax.Array) -> jnp.ndarray:
  """Generates random floats for stochastic rounding."""
  bits = jax.random.bits(rng, shape=shape, dtype='uint16')
  return _rand_round(shape, bits)


def _round(x: jnp.ndarray, random, dtype, rounding: Rounding) -> jnp.ndarray:
  """Rounds x to an integer, using specified rounding mode."""
  if rounding == Rounding.DETERMINISTIC_RTNE:
    return dtype(jnp.round(x))

  # stochastic rounding
  assert(rounding == Rounding.STOCHASTIC_CUSTOM_WITH_HALF)
  assert x.shape == random.shape, (x.shape, random.shape)
  return jnp.round(jnp.float32(x) + random).astype(jnp.int8)


def _dezero(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.where(x == 0.0, 1.0, x)


def _get_scaling_shape(shape, contracting):
  """Returns the expected shape for scaling given the shape of a tensor."""
  return tuple([
      1 if dim in contracting else length
      for dim, length in enumerate(shape)
  ])


def _get_scaling_factor(static_max, tensor, contracting, mi):
  """Returns the scaling factor for a given tensor."""
  if static_max is None:
    max_val = jnp.max(jnp.abs(tensor), axis=contracting, keepdims=True)
  else:
    assert static_max.shape == _get_scaling_shape(tensor.shape, contracting), (
        f'the static_max shape was {static_max.shape} but expected to be '
        f'{_get_scaling_shape(tensor.shape, contracting)}')
    max_val = static_max
  # mrasquinha: Scales are currently in f32; possibly move to bf16
  return jax.lax.convert_element_type(mi / _dezero(max_val), _TMP_DTYPE)


def _stochastic_dot_general(hparams: HParamsNonDifferentiable,
                            stoc_noise,
                            lhs: jnp.ndarray,
                            rhs: jnp.ndarray,
                            dimension_numbers,
                            use_dot_general: bool = False,
                            static_max_lhs: Optional[jnp.ndarray] = None,
                            static_max_rhs: Optional[jnp.ndarray] = None,
                            ) -> jnp.ndarray:
  """Dynamically quantized DotGeneral. No custom gradient.

  Args:
    hparams: The parameters that determine the quantization.
    stoc_noise: Key to use for random number generation (stochastic rounding).
    lhs: The left-hand side of the operation.
    rhs: The left-hand side of the operation.
    dimension_numbers: Specifies contracting and batch dimensions.
      Analogous to lax.dot_general.
    use_dot_general: Whether to use the use_dot_general for the product.
    static_max_lhs: The max (in absolute value) values for scaling the LHS.
      If set to None, dynamic scaling is used to determine the value.
      If this is specified, it must have the same rank as the LHS.
      It should also have length 1 along contracted dimensions.
      If scale_pow2 is True, it will also apply to static scales.
    static_max_rhs: Analogous to s1, but for the RHS.

  Returns:
    The output of the quantized multiplication.
  """
  del use_dot_general
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_kept = _remaining(range(lhs.ndim), lhs_contracting, lhs_batch)
  rhs_kept = _remaining(range(rhs.ndim), rhs_contracting, rhs_batch)

  # We do the dot_general computation in the same precision as
  # jax.lax.dot_general does: if lhs or rhs is float32, result is float32
  mi = ((2**hparams.bits) - 2) / 2.

  s1 = _get_scaling_factor(static_max_lhs, lhs, lhs_contracting, mi)
  s3 = _get_scaling_factor(static_max_rhs, rhs, rhs_contracting, mi)

  if hparams.use_hardware_integers:
    assert hparams.bits <= 8, (f'Requested {hparams.bits}-bit integers,'
                               'and use_hardware_integers=True, which the '
                               'hardware does not support')
    matmul_input_type = jnp.int8
  else:
    matmul_input_type = jnp.bfloat16

  qlhs = _round(
      cast_tmp(jnp.multiply(s1, cast_tmp(lhs))), stoc_noise, matmul_input_type,
      hparams.rounding_lhs)
  qrhs = _round(
      cast_tmp(jnp.multiply(cast_tmp(rhs), s3)), stoc_noise, matmul_input_type,
      hparams.rounding_rhs)
  # Apply clipping to ensure we are in the [-mi, mi] range. This is only
  # necessary if specifying a static max.
  mi = int(mi)  # Make sure any clipping doesn't change the type.
  if static_max_lhs is not None:
    qlhs = jnp.clip(qlhs, -mi, mi)
  if static_max_rhs is not None:
    qrhs = jnp.clip(qrhs, -mi, mi)

  qres = get_original_dot_general()(
      qlhs,
      qrhs,
      dimension_numbers,
      preferred_element_type=hparams.matmul_output_type)
  qres = jax.lax.convert_element_type(qres, jnp.bfloat16)

  # Embed s1 dimensions (lhs batch and lhs kept dims) into qres dims
  # (batch <> lhs kept <> rhs kept)
  # Step 1: reorder dims to (batch <> lhs kept <> lhs contract)
  output_lhs_dims = list(lhs_batch) + lhs_kept + list(lhs_contracting)
  s1 = jax.lax.transpose(s1, tuple(output_lhs_dims))
  # Step 2: squeeze out lhs_contract. Now (batch <> lhs kept)
  s1 = jax.lax.squeeze(s1, range(s1.ndim - len(lhs_contracting), s1.ndim))
  # Step 3: add in rhs_kept as 1s. Now (batch <> lhs kept <> rhs kept as 1s)
  s1 = jax.lax.expand_dims(s1, range(s1.ndim, s1.ndim + len(rhs_kept)))

  # Embed s3 dimensions (rhs batch and rhs kept dims) into qres dims
  # (batch <> lhs kept <> rhs kept)
  # Step 1: reorder dims to (batch <> rhs kept <> rhs contract)
  output_rhs_dims = list(rhs_batch) + rhs_kept + list(rhs_contracting)
  s3 = jax.lax.transpose(s3, tuple(output_rhs_dims))
  # Step 2: squeeze out rhs_contract. Now (batch <> rhs kept).
  s3 = jax.lax.squeeze(s3, range(s3.ndim - len(rhs_contracting), s3.ndim))
  # Step 3: add in lhs_kept as 1s. Now (batch <> lhs kept as 1s <> rhs kept)
  s3 = jax.lax.expand_dims(
      s3, range(len(lhs_batch),
                len(lhs_batch) + len(lhs_kept)))

  return jax.lax.convert_element_type((1 / s1) * qres * (1 / s3), jnp.bfloat16)


def _fp8_dot_general(hparams: HParamsNonDifferentiable,
                     stoc_noise,
                     lhs: jnp.ndarray,
                     rhs: jnp.ndarray,
                     dimension_numbers,
                     static_max_lhs: Optional[jnp.ndarray] = None,
                     static_max_rhs: Optional[jnp.ndarray] = None,
                     ) -> jnp.ndarray:
  """FP8 quantized DotGeneral with the support of E4M3 and E5M2.

  Args:
    hparams: The parameters that determine the FP8 quantization.
    stoc_noise: Not used in FP8
    lhs: The left-hand side of the operation.
    rhs: The right-hand side of the operation.
    dimension_numbers: Specifies contracting and batch dimensions.
    static_max_lhs: The max (in absolute value) values for scaling the LHS.
    static_max_rhs: The max (in absolute value) values for scaling the RHS.
  Returns:
    The output of the quantized multiplication.

  """
  del stoc_noise
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_kept = _remaining(range(lhs.ndim), lhs_contracting, lhs_batch)
  rhs_kept = _remaining(range(rhs.ndim), rhs_contracting, rhs_batch)

  if hparams.fp8_type == jnp.float8_e4m3fn:
    mi = 448.0
  elif hparams.fp8_type == jnp.float8_e5m2:
    mi = 57344.0
  else:
    assert False, f'Unrecognized FP8 type: {hparams}'

  s1 = _get_scaling_factor(static_max_lhs, lhs, lhs_contracting, mi)
  s3 = _get_scaling_factor(static_max_rhs, rhs, rhs_contracting, mi)
  qlhs = jax.lax.convert_element_type(
      cast_tmp(jnp.multiply(s1, cast_tmp(lhs))), hparams.fp8_type)
  qrhs = jax.lax.convert_element_type(
      cast_tmp(jnp.multiply(s3, cast_tmp(rhs))), hparams.fp8_type)

  qres = get_original_dot_general()(
      qlhs,
      qrhs,
      dimension_numbers,
      preferred_element_type=jnp.bfloat16)
  qres = jax.lax.convert_element_type(qres, jnp.bfloat16)

  # Embed s1 dimensions (lhs batch and lhs kept dims) into qres dims
  # (batch <> lhs kept <> rhs kept)
  # Step 1: reorder dims to (batch <> lhs kept <> lhs contract)
  output_lhs_dims = list(lhs_batch) + lhs_kept + list(lhs_contracting)
  s1 = jax.lax.transpose(s1, tuple(output_lhs_dims))
  # Step 2: squeeze out lhs_contract. Now (batch <> lhs kept)
  s1 = jax.lax.squeeze(s1, range(s1.ndim - len(lhs_contracting), s1.ndim))
  # Step 3: add in rhs_kept as 1s. Now (batch <> lhs kept <> rhs kept as 1s)
  s1 = jax.lax.expand_dims(s1, range(s1.ndim, s1.ndim + len(rhs_kept)))

  # Embed s3 dimensions (rhs batch and rhs kept dims) into qres dims
  # (batch <> lhs kept <> rhs kept)
  # Step 1: reorder dims to (batch <> rhs kept <> rhs contract)
  output_rhs_dims = list(rhs_batch) + rhs_kept + list(rhs_contracting)
  s3 = jax.lax.transpose(s3, tuple(output_rhs_dims))
  # Step 2: squeeze out rhs_contract. Now (batch <> rhs kept).
  s3 = jax.lax.squeeze(s3, range(s3.ndim - len(rhs_contracting), s3.ndim))
  # Step 3: add in lhs_kept as 1s. Now (batch <> lhs kept as 1s <> rhs kept)
  s3 = jax.lax.expand_dims(
      s3, range(len(lhs_batch),
                len(lhs_batch) + len(lhs_kept)))

  return jax.lax.convert_element_type((1 / s1) * qres * (1 / s3), jnp.bfloat16)


def _unquantized_dot_general(prng_key, noise, lhs: jnp.ndarray,
                             rhs: jnp.ndarray,
                             dimension_numbers) -> jnp.ndarray:
  del prng_key, noise
  return get_original_dot_general()(lhs, rhs, dimension_numbers)


# From lax.dot_general gradient implementation:
def _ranges_like(*xs):
  start = 0
  for x in xs:
    x_len = len(x)
    yield range(start, start + x_len)
    start += x_len


# From lax.dot_general gradient implementation:
def _remaining(original, *removed_lists):
  removed = set(itertools.chain(*removed_lists))
  return [i for i in original if i not in removed]


def _custom_dot_general(fwd_result_dot, lhs_gradient_dot, rhs_gradient_dot):
  """DotGeneral with configurable custom gradient implementations."""

  # Adapted from lax.dot_general gradient implementation.
  def _dot_general_transpose_base(bits, g, y, *, dimension_numbers,
                                  swap_ans, dot_impl):
    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    x_ndim = g.ndim - y.ndim + len(x_batch) + 2 * len(x_contract)
    x_kept = _remaining(range(x_ndim), x_contract, x_batch)
    y_kept = _remaining(range(y.ndim), y_contract, y_batch)
    if swap_ans:
      ans_batch, ans_y, _ = _ranges_like(x_batch, y_kept, x_kept)
    else:
      ans_batch, _, ans_y = _ranges_like(x_batch, x_kept, y_kept)
    dims = ((tuple(ans_y), tuple(y_kept)), (tuple(ans_batch), tuple(y_batch)))
    x_contract_sorted_by_y = list(
        onp.take(x_contract, onp.argsort(y_contract)))  # type: ignore[arg-type]
    out_axes = onp.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)

    return jax.lax.transpose(dot_impl(bits, g, y, dims),
                             tuple(out_axes))

  @functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
  def trace_impl(prng_key: ..., lhs: jnp.ndarray,
                 rhs: jnp.ndarray, dimension_numbers) -> jnp.ndarray:
    del prng_key
    return fwd_result_dot(None, lhs, rhs, dimension_numbers)

  def fwd_impl(prng_key: ..., lhs: jnp.ndarray,
               rhs: jnp.ndarray, dimension_numbers) ->...:
    return (trace_impl(prng_key, lhs, rhs, dimension_numbers),
            (prng_key, lhs, rhs))

  def bwd_impl(dimension_numbers, residual, g: jnp.ndarray) -> ...:
    (prng_key, lhs, rhs) = residual
    bits = _random(g.shape, prng_key)
    lhs_g = _dot_general_transpose_base(
        bits,
        g,
        rhs,
        dimension_numbers=dimension_numbers,
        swap_ans=False,
        dot_impl=lhs_gradient_dot)

    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
    rhs_g = _dot_general_transpose_base(
        bits,
        g,
        lhs,
        dimension_numbers=swapped_dimension_numbers,
        swap_ans=True,
        dot_impl=rhs_gradient_dot)
    return (None, lhs_g, rhs_g)

  trace_impl.defvjp(fwd_impl, bwd_impl)
  return trace_impl


def _pick_dot(hparams: HparamBaseTypes) -> ...:
  """Picks between quantized and unquantized DotGeneral implementations."""
  if hparams is None:
    return _unquantized_dot_general
  elif isinstance(hparams, HParamsNonDifferentiable) and (
      hparams.fp8_type is None):
    return functools.partial(_stochastic_dot_general, hparams)
  elif isinstance(hparams, HParamsNonDifferentiable) and (
      hparams.fp8_type is not None):
    return functools.partial(_fp8_dot_general, hparams)
  else:
    assert False, f'Unrecognized hparams: {hparams}'


# signature: (prng_key, lhs, rhs, dimension_numbers)
def dq_dot_general(hparams: HParams) -> ...:
  """DotGeneral with optional dynamic quantization in forwards and backwards."""
  return _custom_dot_general(
      _pick_dot(hparams.fwd), _pick_dot(hparams.lhs_gradient),
      _pick_dot(hparams.rhs_gradient))


def dq_matmul(hparams: HParams) -> ...:
  """MatMul with optional dynamic quantization in forwards and backwards."""
  dg = dq_dot_general(hparams)

  def f(prng_key, a: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    n_a = len(a.shape)
    dimension_numbers = (((n_a - 1,), (0,)), ((), ()))
    return dg(prng_key, a, w, dimension_numbers)

  return f


_INT8 = HParamsNonDifferentiable(
    bits=8,
    rounding_lhs=Rounding.DETERMINISTIC_RTNE,
    rounding_rhs=Rounding.DETERMINISTIC_RTNE,
    use_hardware_integers=True,
    matmul_output_type=jnp.bfloat16,
    rounding_custom_bits=16,
)

_STOCHASTIC_ROUNDED_GRADIENT = HParamsNonDifferentiable(
    bits=8,
    rounding_lhs=Rounding.STOCHASTIC_CUSTOM_WITH_HALF,
    rounding_rhs=Rounding.DETERMINISTIC_RTNE,
    use_hardware_integers=True,
    matmul_output_type=jnp.bfloat16,
    rounding_custom_bits=16,
)


F8B8 = HParams(
    fwd=_INT8,
    lhs_gradient=_STOCHASTIC_ROUNDED_GRADIENT,
    rhs_gradient=_STOCHASTIC_ROUNDED_GRADIENT)


_E4M3 = HParamsNonDifferentiable(
    fp8_type=jnp.float8_e4m3fn,
)

_E5M2 = HParamsNonDifferentiable(
    fp8_type=jnp.float8_e5m2,
)

FP8 = HParams(
    fwd=_E4M3,
    lhs_gradient=_E5M2,
    rhs_gradient=_E5M2)


def make_aqt_dq_dg():
  """Create dq_dg.
  
  In maxtext DenseGeneral, use the following:
    aqt_dq_dg = make_aqt_dq_dg()
    aqt_key = self.make_rng('aqt')
    return aqt_dq_dg(aqt_key, inputs, kernel, ((axis, contract_ind), ((), ())))
  """
  hparams = HParams(
      fwd=HParamsNonDifferentiable(),
      lhs_gradient=HParamsNonDifferentiable(
          rounding_lhs=Rounding.STOCHASTIC_CUSTOM_WITH_HALF),
      rhs_gradient=HParamsNonDifferentiable(
          rounding_lhs=Rounding.STOCHASTIC_CUSTOM_WITH_HALF),
  )
  dg = dq_dot_general(hparams)  # interface: key, lhs, rhs, dimension
  return dg