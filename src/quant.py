"""Language Model configurations for research into backprop quantization.

copied from
google3/third_party/tensorflow_models/mlperf/models/rough/gpt3/quant.py
"""
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
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import transformers

DotDimensionNumbers = lax.DotDimensionNumbers

_TMP_DTYPE = jnp.float32


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
  fp8_type: jnp.dtype = None


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


def get_original_dot_general() -> ...:
  return jax.lax.dot_general


def cast_tmp(x: jnp.ndarray) -> jnp.ndarray:
  return jax.lax.convert_element_type(x, _TMP_DTYPE)


def _jax_random_bits(key, bit_width, shape):
  from jax._src import prng  # pylint: disable=g-import-not-at-top

  assert hasattr(prng, 'random_wrap')
  assert not jax.config.jax_enable_custom_prng
  key = prng.random_wrap(key, impl=jax.random.default_prng_impl())
  return prng.random_bits(key, bit_width=bit_width, shape=shape)  # pytype: disable=module-attr


def _rand_round(shape: tuple[int, ...], bits: jnp.ndarray) -> jnp.ndarray:
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


def _random(shape: tuple[int, ...], rng: jax.random.KeyArray) -> jnp.ndarray:
  """Generates random floats for stochastic rounding."""
  nbits = 16
  assert nbits in [8, 16]
  bits = _jax_random_bits(rng, nbits, shape)
  return _rand_round(shape, bits)


def _round(x: jnp.ndarray, random, dtype, rounding: Rounding) -> jnp.ndarray:
  """Rounds x to an integer, using specified rounding mode."""
  if rounding == Rounding.DETERMINISTIC_RTNE:
    return dtype(jnp.round(x))

  # stochastic rounding
  assert rounding == Rounding.STOCHASTIC_CUSTOM_WITH_HALF
  assert x.shape == random.shape, (x.shape, random.shape)
  return jnp.round(jnp.float32(x) + random).astype(jnp.int8)


def _dezero(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.where(x == 0.0, 1.0, x)


def _get_scaling_shape(shape, contracting):
  """Returns the expected shape for scaling given the shape of a tensor."""
  return tuple(
      [1 if dim in contracting else length for dim, length in enumerate(shape)]
  )


def _get_scaling_factor(static_max, tensor, contracting, mi):
  """Returns the scaling factor for a given tensor."""
  if static_max is None:
    max_val = jnp.max(jnp.abs(tensor), axis=contracting, keepdims=True)
  else:
    assert static_max.shape == _get_scaling_shape(tensor.shape, contracting), (
        f'the static_max shape was {static_max.shape} but expected to be '
        f'{_get_scaling_shape(tensor.shape, contracting)}'
    )
    max_val = static_max
  # mrasquinha: Scales are currently in f32; possibly move to bf16
  return jax.lax.convert_element_type(mi / _dezero(max_val), _TMP_DTYPE)


def _stochastic_dot_general(
    hparams: HParamsNonDifferentiable,
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
    dimension_numbers: Specifies contracting and batch dimensions. Analogous to
      lax.dot_general.
    use_dot_general: Whether to use the use_dot_general for the product.
    static_max_lhs: The max (in absolute value) values for scaling the LHS. If
      set to None, dynamic scaling is used to determine the value. If this is
      specified, it must have the same rank as the LHS. It should also have
      length 1 along contracted dimensions. If scale_pow2 is True, it will also
      apply to static scales.
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
  mi = ((2**hparams.bits) - 2) / 2.0

  s1 = _get_scaling_factor(static_max_lhs, lhs, lhs_contracting, mi)
  s3 = _get_scaling_factor(static_max_rhs, rhs, rhs_contracting, mi)

  if hparams.use_hardware_integers:
    assert hparams.bits <= 8, (
        f'Requested {hparams.bits}-bit integers,'
        'and use_hardware_integers=True, which the '
        'hardware does not support'
    )
    matmul_input_type = jnp.int8
  else:
    matmul_input_type = jnp.bfloat16

  qlhs = _round(
      cast_tmp(jnp.multiply(s1, cast_tmp(lhs))),
      stoc_noise,
      matmul_input_type,
      hparams.rounding_lhs,
  )
  qrhs = _round(
      cast_tmp(jnp.multiply(cast_tmp(rhs), s3)),
      stoc_noise,
      matmul_input_type,
      hparams.rounding_rhs,
  )
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
      preferred_element_type=hparams.matmul_output_type,
  )
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
      s3, range(len(lhs_batch), len(lhs_batch) + len(lhs_kept))
  )

  return jax.lax.convert_element_type((1 / s1) * qres * (1 / s3), jnp.bfloat16)


def _fp8_dot_general(
    hparams: HParamsNonDifferentiable,
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
      cast_tmp(jnp.multiply(s1, cast_tmp(lhs))), hparams.fp8_type
  )
  qrhs = jax.lax.convert_element_type(
      cast_tmp(jnp.multiply(s3, cast_tmp(rhs))), hparams.fp8_type
  )

  qres = get_original_dot_general()(
      qlhs, qrhs, dimension_numbers, preferred_element_type=jnp.bfloat16
  )
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
      s3, range(len(lhs_batch), len(lhs_batch) + len(lhs_kept))
  )

  return jax.lax.convert_element_type((1 / s1) * qres * (1 / s3), jnp.bfloat16)


def _unquantized_dot_general(
    prng_key, noise, lhs: jnp.ndarray, rhs: jnp.ndarray, dimension_numbers
) -> jnp.ndarray:
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
  def _dot_general_transpose_base(
      bits, g, y, *, dimension_numbers, swap_ans, dot_impl
  ):
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
        onp.take(x_contract, onp.argsort(y_contract))
    )  # type: ignore[arg-type]
    out_axes = onp.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)

    return jax.lax.transpose(dot_impl(bits, g, y, dims), tuple(out_axes))

  @functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
  def trace_impl(
      prng_key: ..., lhs: jnp.ndarray, rhs: jnp.ndarray, dimension_numbers
  ) -> jnp.ndarray:
    del prng_key
    return fwd_result_dot(None, lhs, rhs, dimension_numbers)

  def fwd_impl(
      prng_key: ..., lhs: jnp.ndarray, rhs: jnp.ndarray, dimension_numbers
  ) -> ...:
    return (
        trace_impl(prng_key, lhs, rhs, dimension_numbers),
        (prng_key, lhs, rhs),
    )

  def bwd_impl(dimension_numbers, residual, g: jnp.ndarray) -> ...:
    (prng_key, lhs, rhs) = residual
    bits = _random(g.shape, prng_key)
    lhs_g = _dot_general_transpose_base(
        bits,
        g,
        rhs,
        dimension_numbers=dimension_numbers,
        swap_ans=False,
        dot_impl=lhs_gradient_dot,
    )

    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
    rhs_g = _dot_general_transpose_base(
        bits,
        g,
        lhs,
        dimension_numbers=swapped_dimension_numbers,
        swap_ans=True,
        dot_impl=rhs_gradient_dot,
    )
    return (None, lhs_g, rhs_g)

  trace_impl.defvjp(fwd_impl, bwd_impl)
  return trace_impl


def _pick_dot(hparams: HparamBaseTypes) -> ...:
  """Picks between quantized and unquantized DotGeneral implementations."""
  if hparams is None:
    return _unquantized_dot_general
  elif isinstance(hparams, HParamsNonDifferentiable) and (
      hparams.fp8_type is None
  ):
    return functools.partial(_stochastic_dot_general, hparams)
  elif isinstance(hparams, HParamsNonDifferentiable) and (
      hparams.fp8_type is not None
  ):
    return functools.partial(_fp8_dot_general, hparams)
  else:
    assert False, f'Unrecognized hparams: {hparams}'


# signature: (prng_key, lhs, rhs, dimension_numbers)
def dq_dot_general(hparams: HParams) -> ...:
  """DotGeneral with optional dynamic quantization in forwards and backwards."""
  return _custom_dot_general(
      _pick_dot(hparams.fwd),
      _pick_dot(hparams.lhs_gradient),
      _pick_dot(hparams.rhs_gradient),
  )


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
    rhs_gradient=_STOCHASTIC_ROUNDED_GRADIENT,
)


_E4M3 = HParamsNonDifferentiable(
    fp8_type=jnp.float8_e4m3fn,
)

_E5M2 = HParamsNonDifferentiable(
    fp8_type=jnp.float8_e5m2,
)

FP8 = HParams(fwd=_E4M3, lhs_gradient=_E5M2, rhs_gradient=_E5M2)


WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor


class QuanztizedLayer(base_layer.BaseLayer):
  """Base class for sharing prng key."""

  dim: int = 2
  rng_key = None

  def setup(self) -> None:
    # mrasquinha: This is currently not used; retaining for a subsequent
    # fix related to sharing rng generation across quantized matmuls.
    self.rng_key = jax.random.PRNGKey(112129)


class AttentionProjection(QuanztizedLayer):
  """Layer that computes multi heads projection.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    input_dim: Input dimension.
    num_heads: Number of attention heads.
    dim_per_head: Size of each head.
    is_output_projection: Whether it is out projection or not. If False, we use
      "...D,DNH->...NH" for query,key,value projection. Otherwise we use
      "...NH,DNH->...D" for output projection.
    use_bias: Whether to add bias in projection or not.
    attention_combine_dims: The heads and key/value dimensions are combined in
      the variables and the computation.
    use_nhd_shape: Whether to use NHD shape for the variable, useful for dot
      attention output layer.
    explicit_fan_in_fan_out_axes: Set true except for backward compatibility.
  """

  input_dim: int = 0
  num_heads: int = 0
  dim_per_head: int = 0
  is_output_projection: bool = False
  use_bias: bool = True
  attention_combine_dims: bool = False
  use_nhd_shape: bool = False
  explicit_fan_in_fan_out_axes: bool = False  # TODO(b/232864754) switch to True
  quantization: HParams = HParams()

  def setup(self) -> None:
    super().setup()
    wp = self.weight_split_dims_mapping
    has_sharding = self.mesh_shape is not None and wp.wt is not None
    if self.attention_combine_dims:
      assert not self.use_bias
      hd_shape = [self.num_heads * self.dim_per_head]
    else:
      hd_shape = [self.num_heads, self.dim_per_head]

    if self.attention_combine_dims and has_sharding:
      if len(wp.wt) == 3:
        if self.is_output_projection and self.use_nhd_shape:
          h_sharding = ()
          for axes in (wp.wt[0], wp.wt[1]):
            if isinstance(axes, (str, int)):
              h_sharding += (axes,)
            elif axes is not None:
              h_sharding += tuple(axes)
          wt = [h_sharding, wp.wt[2]]
        else:
          h_sharding = ()
          for axes in (wp.wt[1], wp.wt[2]):
            if isinstance(axes, (str, int)):
              h_sharding += (axes,)
            elif axes is not None:
              h_sharding += tuple(axes)
          wt = [wp.wt[0], h_sharding]
      assert len(wt) == 2
    else:
      wt = wp.wt

    if self.is_output_projection and self.use_nhd_shape:
      pc_shape = hd_shape + [self.input_dim]
      if self.attention_combine_dims:
        fan_in_axes, fan_out_axes = [-1], [-2]
      else:
        fan_in_axes, fan_out_axes = [-1], [-2, -3]
    else:
      pc_shape = [self.input_dim] + hd_shape
      if self.attention_combine_dims:
        fan_in_axes, fan_out_axes = [-2], [-1]
      else:
        fan_in_axes, fan_out_axes = [-3], [-1, -2]

    pc = WeightHParams(
        shape=pc_shape,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wt,
        fan_in_axes=(
            fan_in_axes if self.explicit_fan_in_fan_out_axes else None
        ),
        fan_out_axes=(
            fan_out_axes if self.explicit_fan_in_fan_out_axes else None
        ),
    )
    self.create_variable('w', pc)
    if self.use_bias:
      if self.is_output_projection:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[0]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[self.input_dim],
            init=WeightInit.Constant(0.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping,
        )
      else:
        if has_sharding:
          bias_split_dims_mapping = [wp.wt[1], wp.wt[2]]
        else:
          bias_split_dims_mapping = None
        pc_bias = WeightHParams(
            shape=[self.num_heads, self.dim_per_head],
            init=WeightInit.Constant(0.0),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=bias_split_dims_mapping,
        )
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      inputs: A JTensor of shape [..., num_heads, dim_per_head] if
        p.is_output_projection is True or [..., p.input_dim] otherwise..

    Returns:
      The projected JTensor with shape [..., p.input_dim] if
      p.is_output_projection is True or [..., num_heads, dim_per_head]
      otherwise.
    """
    theta = self.theta

    shape = inputs.shape
    rank = len(shape)

    inputs = self._cast_to_fprop_dtype(inputs)
    if self.attention_combine_dims:
      pc_shape = [self.input_dim, self.num_heads, self.dim_per_head]
      if self.is_output_projection and self.use_nhd_shape:
        pc_shape = [self.num_heads, self.dim_per_head, self.input_dim]
      w = jnp.reshape(theta.w, pc_shape)
    else:
      w = theta.w

    if self.is_output_projection:
      assert shape[-2:] == (self.num_heads, self.dim_per_head)
      if self.use_nhd_shape:
        # ...NH,NHD->...D
        dimension_numbers = (((rank - 2, rank - 1), (0, 1)), ((), ()))
      else:
        # '...NH,DNH->...D'
        dimension_numbers = (((rank - 2, rank - 1), (1, 2)), ((), ()))
    else:
      assert (
          shape[-1] == self.input_dim
      ), f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
      # '...D,DNH->...NH'
      dimension_numbers = (((rank - 1,), (0,)), ((), ()))

    # Use a sharded prng key
    ret = dq_dot_general(self.quantization)(
        self.next_prng_key(), inputs, w, dimension_numbers
    )
    if self.use_bias:
      ret += theta.b
    return ret


class CombinedQKVProjectionLayer(QuanztizedLayer):
  """Layer that computes QKV projection with a combined weight.

  It may lead to faster collectives and step-time on TPU.

  This layer is expected to be used within DotProductAttention below.

  Attributes:
    input_dim: Input dimension.
    num_heads: Number of heads.
    dim_per_head: Size of each head.
    use_bias: Whether to add bias in the projection layer.
    attention_combine_dims: If set, the heads and key/value dimensions are
      combined in the variables and the computation.
    explicit_fan_in_fan_out_axes: Set true except for backward compatibility.
  """

  input_dim: int = 0
  num_heads: int = 0
  dim_per_head: int = 0
  use_bias: bool = True
  attention_combine_dims: bool = False
  explicit_fan_in_fan_out_axes: bool = False  # TODO(b/232864754) switch to True
  quantization: HParams = HParams()

  def setup(self) -> None:
    super().setup()
    # Sharding has the same convention of AttentionProjection, which doesn't
    # contain the leading stacking dimension.
    wt = self.weight_split_dims_mapping.wt
    if wt is not None:
      assert isinstance(wt, (list, tuple))
      if self.attention_combine_dims:
        if len(wt) == 3:
          hd_sharding = ()
          for s in wt[1:]:
            if isinstance(s, (list, tuple)):
              hd_sharding += tuple(s)
            elif s is not None:
              hd_sharding += (s,)
          wt = [wt[0], hd_sharding]
        else:
          assert len(wt) == 2
      else:
        # Replicate the concat axis.
        assert len(wt) == 3, (
            'wp.wt only specifies the sharding for '
            'the last three dims of the weight tensor.'
        )
      weight_split_dims_mapping = [None] + list(wt)
      if self.attention_combine_dims:
        bias_split_dims_mapping = [None, wt[1]]
      else:
        bias_split_dims_mapping = [None, wt[1], wt[2]]
    else:
      weight_split_dims_mapping = None
      bias_split_dims_mapping = None

    if self.attention_combine_dims:
      hd_shape = [self.num_heads * self.dim_per_head]
      fan_in_axes, fan_out_axes = [-2], [-1]
    else:
      hd_shape = [self.num_heads, self.dim_per_head]
      fan_in_axes, fan_out_axes = [-3], [-1, -2]

    pc_shape = [3, self.input_dim] + hd_shape
    # Combined weight for q, k, v projections.
    pc = WeightHParams(
        shape=pc_shape,
        init=self.params_init,
        dtype=self.dtype,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=weight_split_dims_mapping,
        fan_in_axes=(
            fan_in_axes if self.explicit_fan_in_fan_out_axes else None
        ),
        fan_out_axes=(
            fan_out_axes if self.explicit_fan_in_fan_out_axes else None
        ),
    )
    self.create_variable('w', pc)
    if self.use_bias:
      # Combined bias weight for q, k, v projections.
      pc_bias = WeightHParams(
          shape=[3] + hd_shape,
          init=WeightInit.Constant(0.0),
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=bias_split_dims_mapping,
      )
      self.create_variable('b', pc_bias)

  def __call__(self, inputs: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    """Computes the QKV projection for inputs.

    Args:
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The three projected JTensor with shape [..., num_heads, dim_per_head]
      in q_proj, k_proj and v_proj order.
    """
    theta = self.theta

    shape = inputs.shape
    rank = len(shape)
    assert rank > 0

    assert shape[-1] == self.input_dim
    batch_dims_rank = rank - 1
    # batch_eqn = eqn_sym[:batch_dims_rank] if rank else '...'
    if self.attention_combine_dims:
      pc_shape = [3, self.input_dim, self.num_heads, self.dim_per_head]
      w = jnp.reshape(theta.w, pc_shape)
      if self.use_bias:
        b_shape = [3, self.num_heads, self.dim_per_head]
        b = jnp.reshape(theta.b, b_shape)
    else:
      w = theta.w
      if self.use_bias:
        b = theta.b

    # K indexes qkv.
    # Intended einsum: '...D,KDNH->K...NH'.
    # DotGeneral gives us '...D,KDNH->...KNH', after which we transpose it to
    # the target shape.
    dimension_numbers = (((rank - 1,), (1,)), ((), ()))

    # Use a sharded prng key
    ret = dq_dot_general(self.quantization)(
        self.next_prng_key(), inputs, w, dimension_numbers
    )
    # Transpose '...KNH->K...NH'.
    permutation = tuple(
        [batch_dims_rank]
        + list(range(batch_dims_rank))
        + [batch_dims_rank + 1, batch_dims_rank + 2]
    )
    ret = jax.lax.transpose(ret, permutation)

    ret = checkpoint_name(ret, 'combined_qkv_proj')
    if self.use_bias:
      # Add newaxis to bias weight for each batch dim since ret is K...NH
      # and theta.b is KNH. Need to reshape theta.b to K...NH
      ret += jnp.expand_dims(b, list(range(1, batch_dims_rank + 1)))
    # Split into three projections.
    query_proj, key_proj, value_proj = ret
    query_proj = checkpoint_name(query_proj, 'query_proj')
    key_proj = checkpoint_name(key_proj, 'key_proj')
    value_proj = checkpoint_name(value_proj, 'value_proj')
    return query_proj, key_proj, value_proj


def copy_attention_projection_fields_from_base(
    hparams: pax_fiddle.Config[AttentionProjection],
    base: pax_fiddle.Config[AttentionProjection],
):
  """Copies common fields from `base` to `hparams`."""
  hparams.input_dim = base.input_dim
  hparams.num_heads = base.num_heads
  hparams.dim_per_head = base.dim_per_head
  hparams.is_output_projection = base.is_output_projection
  hparams.use_bias = base.use_bias
  hparams.attention_combine_dims = base.attention_combine_dims
  hparams.use_nhd_shape = base.use_nhd_shape
  hparams.explicit_fan_in_fan_out_axes: bool = base.explicit_fan_in_fan_out_axes


def copy_qkv_attention_projection_fields_from_base(
    hparams: pax_fiddle.Config[CombinedQKVProjectionLayer],
    base: pax_fiddle.Config[CombinedQKVProjectionLayer],
):
  """Copies common fields from `base` to `hparams`."""
  hparams.input_dim = base.input_dim
  hparams.num_heads = base.num_heads
  hparams.dim_per_head = base.dim_per_head
  hparams.use_bias = base.use_bias
  hparams.attention_combine_dims = base.attention_combine_dims
  hparams.explicit_fan_in_fan_out_axes: bool = base.explicit_fan_in_fan_out_axes


class LinearQuantized(QuanztizedLayer):
  """Linear layer without bias, with integer quantization.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """

  input_dims: int = 0
  output_dims: int = 0
  weight_init: Optional[WeightInit] = None
  quantization: HParams = HParams()

  def setup(self) -> None:
    super().setup()
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dims, self.output_dims],
            init=self.weight_init,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    # Use a sharded prng key
    out = dq_matmul(self.quantization)(
        self.next_prng_key(), inputs, self.theta.w
    )
    # Adjust sharding annotation during decoding.
    if ap.out is not None and len(ap.out) == 3 and out.ndim == 2:
      ap.out = [ap.out[0], ap.out[2]]
    out = base_layer.maybe_shard(out, ap.out, self.mesh_axis_names)
    return out


def apply_quantized_layers_sharded(model, quantization):
  """Changes the model to use quantized Transformer layers."""
  if hasattr(model, 'lm_tpl'):
    xformer_p = model.lm_tpl.stacked_transformer_tpl
    if xformer_p.cls == transformers.PipelinedTransformer:
      xformer_p = xformer_p.pipeline_stage

    if xformer_p.cls == transformers.StackedTransformerRepeated:
      xformer_p = xformer_p.block
    xformer_p = xformer_p.transformer_layer_params_tpl

    lq = pax_fiddle.Config(LinearQuantized)
    lq.quantization = quantization
    xformer_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl = lq

    qkv_proj = pax_fiddle.Config(CombinedQKVProjectionLayer)
    copy_qkv_attention_projection_fields_from_base(
        qkv_proj, xformer_p.tr_atten_tpl.combined_qkv_proj_tpl
    )
    qkv_proj.quantization = quantization
    xformer_p.tr_atten_tpl.combined_qkv_proj_tpl = qkv_proj

    for atten_p in (xformer_p.tr_atten_tpl, xformer_p.cross_atten_tpl):
      if atten_p is None:
        continue
      proj = pax_fiddle.Config(AttentionProjection)
      copy_attention_projection_fields_from_base(proj, atten_p.proj_tpl)
      proj.quantization = quantization
      atten_p.proj_tpl = proj