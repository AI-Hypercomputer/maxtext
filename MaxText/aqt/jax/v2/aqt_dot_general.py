# Copyright 2022 Google LLC
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
"""Quantized dot_general."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes


# pylint: disable=g-explicit-bool-comparison

import copy
import functools
from typing import Callable, Optional, Union
from aqt.jax.v2 import config
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp


@flax.struct.dataclass
class Context:
  key: Optional[jax.random.KeyArray]
  train_step: Optional[int]


def _context_split(context: Context) -> tuple:
  def mk_ctx(key):
    return Context(key=key, train_step=context.train_step)
  if context.key is not None:
    key1, key2 = jax.random.split(context.key)
    return mk_ctx(key1), mk_ctx(key2)
  return mk_ctx(None), mk_ctx(None)


def _get_edge_of_last_int_bucket(cfg: config.IntNumerics):
  ret = 2.0 ** (cfg.bits - 1)
  if cfg.preserve_zero:
    # Lose one bucket.
    ret -= 0.5
  return ret


def _int_fresh_scale(x, cfg: config.Tensor) -> jnp.ndarray:
  """Calibration scale."""
  assert isinstance(cfg.numerics, config.IntNumerics)
  assert cfg.numerics.bits <= 22, 'Too many bits, float32 has less precision.'
  msg = 'Perhaps you are using fake_quant and forgot to set them.'
  assert cfg.calib_shared_axes is not None, msg

  if cfg.bound is None:
    # If you want to use different calibration, modify _make_int_quant.vjp_bwd.
    abs_max = jnp.max(jnp.abs(x), axis=cfg.calib_shared_axes, keepdims=True)
  else:
    assert cfg.bound > 0, 'Static quantization bound should be positive.'
    abs_max = jnp.asarray(cfg.bound).reshape((1,) * len(x.shape))
  abs_max = jnp.where(abs_max == 0.0, 1.0, abs_max)
  if cfg.bound_stop_grad:
    # TODO(lew): Does not matter in DG, because we are using custom gradient.
    #   We should take that into account somehow.
    abs_max = lax.stop_gradient(abs_max)

  abs_max_mapped_to = _get_edge_of_last_int_bucket(cfg.numerics)
  if cfg.preserve_max_val:
    # In this case we are mapping abs_max onto center of the last bucket
    # Lose half of last bucket
    abs_max_mapped_to -= 0.5
  # Now abs_max_mapped_to is either center or edge of the last bucket.

  # Verifying the correctness of this function amounts to verifying this table:
  # pylint: disable=line-too-long
  # if preserve_zero == F, zero might be rounded either to [-1, 0] bucket or to [0, 1] bucket
  # preserve_zero, preserve_max_val, 8b, 2b, 1b
  # F, F, 128.0, 2.0, 1.0  # bucket count is even; map onto the far edge of the last bucket
  # F, T, 127.5, 1.5, 0.5  # bucket count is even; map onto the center of the last bucket
  # T, F, 127.5, 1.5, 0.5  # bucket count is odd;  map onto the far edge of the last bucket
  # T, T, 127.0, 1.0, 0.0  # bucket count is odd;  map onto the center of the last bucket

  new_scale = abs_max_mapped_to / abs_max
  if cfg.po2_scale:
    # With floor the bigges value (we are using jnp.max) is in the range of
    # clipping and therefore have a correct gradinet.
    new_scale = 2 ** jnp.floor(jnp.log2(new_scale))
  return new_scale


def _make_int_quant(cfg: config.Tensor):
  """Function make_quant."""
  # This function is supposed to round values in a bucket to its center.
  # The way to check for correctness is to check that the values between
  # the buckets are "hard to decide".
  # Let's look at 0 or 0.5 (depending on preserve zero),
  # and lets look at the edge of the last bucket (table in fresh_scale_).

  assert isinstance(cfg.numerics, config.IntNumerics)
  assert cfg.numerics.bits <= 22, 'Too many bits, float32 has less precision.'

  # preserve_max_val does not affect the edge of the last bucket.
  edge_of_last_bucket = _get_edge_of_last_int_bucket(cfg.numerics)

  def fwd(x, context):
    # Maybe noise
    if cfg.noise_fn:
      assert context.key is not None, (
          'noise_fn is set, requestic stochastic rounding, but key key was not'
          ' passed.'
      )
      x = x + cfg.noise_fn(x.shape, context.key)

    # Maybe clip
    if cfg.clip:
      # If we are not rounding, we just clip to bucket edges.
      fwd_clip_bound = edge_of_last_bucket
      # If, after clip, we are rounding, we need to make sure that
      # we won't round values at the edge_of_last_bucket away to the
      # non-existing bucket.
      if cfg.round:
        # Reducing fwd_clip_bound by any value in (0.0, 1.0) is correct.
        fwd_clip_bound -= 0.5
      x = jnp.clip(x, -fwd_clip_bound, fwd_clip_bound)

    # Maybe round
    if cfg.round:
      # TODO(lew): Have bucket centers at 2*k + 1, not at halves.
      round_to_halves = not cfg.numerics.preserve_zero
      if round_to_halves:
        x = jnp.floor(x) + 0.5
      else:
        x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    return x

  def vjp_fwd(x, context):
    res = (x,)
    return fwd(x, context), res

  def vjp_bwd(res, grad):
    # This is gradient of clip. For boundary values we will have full graindent.
    # We might use something like this for calibrations other than abs(max(x))
    # (x,) = res
    # ret = (x <= edge_of_last_bucket) * (x >= -edge_of_last_bucket) * grad
    del res
    ret = grad
    return (ret, None)

  vjp = jax.custom_vjp(fwd)
  vjp.defvjp(vjp_fwd, vjp_bwd)
  return vjp


def _scale_quant(x, *, cfg, ca, context):
  """The core quantizing function."""
  msg = (
      'use_fake_quant mode is used in tests and it is exactly equal when'
      ' po2_scale == True; Did you forget to set it?'
  )
  assert (not cfg.use_fake_quant) or cfg.po2_scale, msg

  # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
  # TODO(lew): After we implement optimization to not double-quantize,
  #   what would happen if we pass fq value (xhs_q2) in residual?
  if cfg.use_fake_quant:
    assert cfg.dtype == jnp.bfloat16

  if isinstance(cfg.numerics, config.NoNumerics):
    return x, None, None
  if cfg.calib_shared_axes is None:
    cfg.calib_shared_axes = ca
  fresh_scale_fn = cfg.fresh_scale or functools.partial(
      _int_fresh_scale, cfg=cfg
  )
  scale = fresh_scale_fn(x)
  x_s = _maybe_mul(x, scale)
  quant = cfg.clip_and_round or _make_int_quant(cfg)
  quant = functools.partial(quant, context=context)
  x_q, quant_grad = jax.vjp(quant, x_s)
  # We are passing quant_grad (and not more) ot the backward pass.
  # That is equivalent to having:
  # scale = stop_gradient(scale)
  #
  # This is not the only possible choice and we intend to allow experimentation.
  # However for today we hardcoded this choice.
  #
  # In order to achevie no-stop-gradiend solution, we should take vjp
  # of a larger piece of code like the whole _scale_quant.
  #
  # TODO(lew): Implement configuration of stop-gradient.
  inv_scale = _maybe_inv(scale)

  return x_q, inv_scale, quant_grad


def _residual(
    *, x, x_q, inv_scale, quant_grad, side, dg_dims, lhs_shape, rhs_shape
):
  """Assembles TensorRes."""
  assert side in ['lhs', 'rhs']
  transpose_scale = (
      _rhs_scale_transpose if side == 'rhs' else _lhs_scale_transpose
  )
  inv_scale_t = transpose_scale(inv_scale, dg_dims, lhs_shape, rhs_shape)
  # TODO(lew): Remove when we have appropriate tests.
  # quant_grad won't cause additional computations if it is STE.
  quant_grad = None
  return TensorRes(
      value=x,
      qvalue=x_q,
      qvalue_scale=inv_scale_t,
      quant_grad=quant_grad,
  )


def make_fake_quant(cfg: config.Tensor, ca=None):
  def fake_quant(x, context):
    x_q, inv_scale, _ = _scale_quant(x, cfg=cfg, ca=ca, context=context)
    return _maybe_mul(x_q, inv_scale)

  return fake_quant


@flax.struct.dataclass
class TensorRes:
  """All the things we pass from the forward pass to the backward pass."""
  value: jnp.ndarray
  qvalue: jnp.ndarray
  qvalue_scale: Union[jnp.ndarray, None]
  quant_grad: Union[Callable[[jnp.ndarray], tuple], None]


@flax.struct.dataclass
class DotGeneralRes:
  context_bwd: Context
  lhs: TensorRes
  rhs: TensorRes


def _scale_trans(x, ca, ba):
  for i in ca:
    assert x.shape[i] == 1
  ra = tuple(i for i in range(len(x.shape)) if i not in ba + ca)
  x = jnp.transpose(x, ba + ra + ca)
  # TODO(lew): x = jnp.squeeze(x, axis=range(len(ba+ra): len(x.shape))
  shape_ba = x.shape[: len(ba)]
  shape_ra = x.shape[len(ba) : len(x.shape) - len(ca)]
  # Will need to add additional axes (size 1) for the other shape_ra
  x = x.reshape(shape_ba + shape_ra)
  return x


def _lhs_scale_transpose(lhs_scale, dimension_numbers, lhs_shape, rhs_shape):
  """Transposes lhs_scale to output dimension order."""
  if lhs_scale is None:
    return None
  # The axis order in out is as follows: batch, lhs_ra, rhs_ra
  # - batch axes order is uniquely determined by either lhs_ba or rhs_ba
  # - contraction axes ca disappear from the output
  # - order of the remaining axes (ra) is preserved.
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qlhs_scale_t = _scale_trans(lhs_scale, lhs_ca, lhs_ba)
  # inserting dummy axes for rhs_ra
  assert len(qlhs_scale_t.shape) == len(lhs_shape) - len(lhs_ca)
  start = len(qlhs_scale_t.shape)
  end = len(rhs_shape) - len(rhs_ca) - len(rhs_ba) + start
  lhs_dummy_axes = range(start, end)
  qlhs_scale_t = jnp.expand_dims(qlhs_scale_t, axis=lhs_dummy_axes)
  return qlhs_scale_t


def _rhs_scale_transpose(rhs_scale, dimension_numbers, lhs_shape, rhs_shape):
  if rhs_scale is None:
    return None
  del rhs_shape
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qrhs_scale_t = _scale_trans(rhs_scale, rhs_ca, rhs_ba)
  start = len(rhs_ba)
  end = len(lhs_shape) - len(lhs_ca) - len(lhs_ba) + start
  rhs_dummy_axes = range(start, end)
  qrhs_scale_t = jnp.expand_dims(qrhs_scale_t, axis=rhs_dummy_axes)
  return qrhs_scale_t


def _maybe_mul(x, scale):
  if scale is None:
    return x
  return x * scale


def _maybe_inv(x):
  if x is None:
    return None
  return 1.0 / x


def _make_dot_general_raw(cfg: config.DotGeneralRaw):
  """Makes quantized lax.dot_general replacement."""

  def my_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      context,
  ):
    """Creates a fake_quant function."""

    (lhs_ca, rhs_ca), _ = dimension_numbers

    context, context_bwd = _context_split(context)
    context_lhs, context_rhs = _context_split(context)
    del context

    lhs_q, lhs_inv_scale, lhs_quant_grad = _scale_quant(
        lhs, cfg=cfg.lhs, ca=lhs_ca, context=context_lhs
    )
    lhs_q2 = (
        _maybe_mul(lhs_q, lhs_inv_scale) if cfg.lhs.use_fake_quant else lhs_q
    )
    lhs_res = _residual(
        x=lhs,
        x_q=lhs_q,
        inv_scale=lhs_inv_scale,
        quant_grad=lhs_quant_grad,
        side='lhs',
        dg_dims=dimension_numbers,
        lhs_shape=lhs.shape,
        rhs_shape=rhs.shape,
    )

    rhs_q, rhs_inv_scale, rhs_quant_grad = _scale_quant(
        rhs, cfg=cfg.rhs, ca=rhs_ca, context=context_rhs
    )
    rhs_q2 = (
        _maybe_mul(rhs_q, rhs_inv_scale) if cfg.rhs.use_fake_quant else rhs_q
    )
    rhs_res = _residual(
        x=rhs,
        x_q=rhs_q,
        inv_scale=rhs_inv_scale,
        quant_grad=rhs_quant_grad,
        side='rhs',
        dg_dims=dimension_numbers,
        lhs_shape=lhs.shape,
        rhs_shape=rhs.shape,
    )

    # These types match default TPU behavior. GPU would need some work.
    # Relevant: https://github.com/google/jax/issues/14022
    if cfg.lhs.dtype == jnp.int8 and cfg.rhs.dtype == jnp.int8:
      lax_dg_in_dtype = jnp.int8
      lax_dg_out_dtype = jnp.int32
      if cfg.lhs.clip_and_round is None and cfg.lhs.clip_and_round is None:
        msg = 'Need cfg.xhs.clip and cfg.xhs.round to use HW int8'
        assert cfg.lhs.round and cfg.lhs.clip, msg
        assert cfg.rhs.round and cfg.rhs.clip, msg
      else:
        assert False, "For now, we don't allow HW int8 with clip_and_round"
    else:
      lax_dg_in_dtype = jnp.bfloat16
      lax_dg_out_dtype = jnp.float32

    out = lax.dot_general(
        lhs_q2.astype(lax_dg_in_dtype),
        rhs_q2.astype(lax_dg_in_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=lax_dg_out_dtype,
        precision=lax.Precision.DEFAULT,
    )

    if not cfg.lhs.use_fake_quant:
      out = _maybe_mul(out, lhs_res.qvalue_scale)

    if not cfg.rhs.use_fake_quant:
      out = _maybe_mul(out, rhs_res.qvalue_scale)

    res = DotGeneralRes(
        context_bwd=context_bwd,
        lhs=lhs_res,
        rhs=rhs_res,
    )
    return out, res

  return my_dot_general


def _dot_general_raw_attach_gradient(
    fwd_dot_general_raw,
    dlhs_dot_general_raw,
    drhs_dot_general_raw,
    dlhs_use_fwd_quant=False,
    drhs_use_fwd_quant=False,
):
  """Makes quantized lax.dot_general replacement with attached gradients."""

  def make_fwd(return_residual):
    def fwd(
        lhs,
        rhs,
        dimension_numbers,
        context,
    ):
      assert lhs.dtype == rhs.dtype
      ret, res = fwd_dot_general_raw(
          lhs,
          rhs,
          dimension_numbers,
          context,
      )
      ret = ret.astype(lhs.dtype)
      return (ret, res) if return_residual else ret

    return fwd

  def vjp_bwd(
      fwd_dimension_numbers,
      res: DotGeneralRes,
      g,
  ):
    def ranges_like(*xs):
      start = 0
      for x in xs:
        yield tuple(range(start, start + len(x)))
        start += len(x)

    def grad_dot_general(
        y_res: TensorRes,
        quant_grad,
        dot_general,
        y_is_lhs,
        context,
        use_fwd_quant,
    ):
      y_ndim = y_res.value.ndim

      (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
      if y_is_lhs:
        (y_ca, x_ca) = (x_ca, y_ca)
        (y_ba, x_ba) = (x_ba, y_ba)
      g_ndim = g.ndim - y_ndim + len(x_ba) + 2 * len(x_ca)
      x_ra = tuple(i for i in range(g_ndim) if i not in x_ca and i not in x_ba)
      y_ra = tuple(i for i in range(y_ndim) if i not in y_ca and i not in y_ba)
      if y_is_lhs:
        g_ba, g_ca, _ = ranges_like(x_ba, y_ra, x_ra)
      else:
        g_ba, _, g_ca = ranges_like(x_ba, x_ra, y_ra)
      dims = ((g_ca, y_ra), (g_ba, y_ba))

      if use_fwd_quant:
        gv = _maybe_mul(g, y_res.qvalue_scale)
        yv = y_res.qvalue
      else:
        gv = g
        yv = y_res.value
      out, _ = dot_general(gv, yv, dims, context)

      x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
      out_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
      transposed_out = jax.lax.transpose(out, out_axes)
      if quant_grad is not None:
        transposed_out = quant_grad(transposed_out)[0]
      return transposed_out

    context1, context2 = _context_split(res.context_bwd)
    dlhs = grad_dot_general(
        res.rhs,
        res.lhs.quant_grad,
        dlhs_dot_general_raw,
        False,
        context1,
        dlhs_use_fwd_quant,
    )
    drhs = grad_dot_general(
        res.lhs,
        res.rhs.quant_grad,
        drhs_dot_general_raw,
        True,
        context2,
        drhs_use_fwd_quant,
    )
    return (
        dlhs.astype(res.lhs.value.dtype),
        drhs.astype(res.rhs.value.dtype),
        None,
    )

  vjp = jax.custom_vjp(make_fwd(False), nondiff_argnums=(2,))
  vjp.defvjp(make_fwd(True), vjp_bwd)
  return vjp


def make_dot_general(cfg: Optional[config.DotGeneral]):
  """Makes quantized lax.dot_general replacement with attached gradients."""
  if cfg is None:
    def ret_lax_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        *,
        context=Context(key=None, train_step=None),
    ):
      del context
      return jax.lax.dot_general(
          lhs, rhs, dimension_numbers, precision, preferred_element_type
      )

    return ret_lax_dg

  dg = _dot_general_raw_attach_gradient(
      fwd_dot_general_raw=_make_dot_general_raw(cfg.fwd),
      dlhs_dot_general_raw=_make_dot_general_raw(cfg.dlhs),
      drhs_dot_general_raw=_make_dot_general_raw(cfg.drhs),
      dlhs_use_fwd_quant=cfg.dlhs.use_fwd_quant,
      drhs_use_fwd_quant=cfg.drhs.use_fwd_quant,
  )

  def ret_dg(
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
      *,
      context=Context(key=None, train_step=None),
  ):
    assert (
        precision is None
    ), f'Precision {precision} requested together with quantization.'
    assert preferred_element_type is None, (
        f'Preferred_element_typerecision {preferred_element_type} requested'
        ' together with quantization.'
    )
    assert lhs.dtype == rhs.dtype, (
        'The only reason we need that, is because we need to determine return'
        ' type.'
    )
    out = dg(
        lhs=lhs,
        rhs=rhs,
        dimension_numbers=dimension_numbers,
        context=context,
    )
    return out.astype(lhs.dtype)

  return ret_dg


def make_conv_general_dilated(cfg: config.DotGeneralRaw):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific cfg.
  cfg = copy.deepcopy(cfg)
  if cfg is None:
    cfg = config.DotGeneralRaw.make()

  def my_conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation=None,
      rhs_dilation=None,
      dimension_numbers=None,
      feature_group_count=1,
      batch_group_count=1,
      precision=None,
      preferred_element_type=None,
  ) -> jax.Array:
    msg1 = """
To simplify the code, we currently assume a Flax-particular layout of the data.
This makes sense, because this is the main use-case of this function.
However if there is any other use, we will drop that assumption."""
    rank = len(lhs.shape)
    assert len(rhs.shape) == rank
    assert dimension_numbers is not None, msg1
    assert dimension_numbers.lhs_spec[0:2] == (0, rank - 1), msg1
    assert dimension_numbers.rhs_spec[0:2] == (rank - 1, rank - 2), msg1
    assert dimension_numbers.out_spec[0:2] == (0, rank - 1), msg1
    # In Flax, lhs is the inputs, rhs is the kernel.
    # lhs layout is B, spatials..., Ci
    # rhs layout is: spatials..., Ci, Co
    # out layous it: B, spatials..., Co
    #
    # we need to share these axes: lhs[1:] , rhs[:-1]
    # we have a scale/invscale per: lhs[0] / out[0] and rhs[-1] / out[-1]

    if isinstance(cfg.lhs.numerics, config.NoNumerics):
      pass
    elif isinstance(cfg.lhs.numerics, config.IntNumerics):
      # Flax assumptions.
      assert cfg.lhs.calib_shared_axes == list(range(1, rank))
      lhs_scale = _int_fresh_scale(lhs, cfg.lhs)
      lhs = lhs * lhs_scale
      quant = cfg.lhs.clip_and_round or _make_int_quant(cfg.lhs)
      lhs = quant(lhs, None)
    else:
      assert False, cfg.lhs.numerics

    if isinstance(cfg.rhs.numerics, config.NoNumerics):
      pass
    elif isinstance(cfg.rhs.numerics, config.IntNumerics):
      assert cfg.rhs.calib_shared_axes == list(range(0, rank - 1))
      rhs_scale = _int_fresh_scale(rhs, cfg.rhs)
      rhs = rhs * rhs_scale
      quant = cfg.rhs.clip_and_round or _make_int_quant(cfg.rhs)
      rhs = quant(rhs, None)
    else:
      assert False, cfg.rhs.numerics

    out = lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    if isinstance(cfg.lhs.numerics, config.NoNumerics):
      pass
    elif isinstance(cfg.lhs.numerics, config.IntNumerics):
      out /= lhs_scale
    else:
      assert False

    if isinstance(cfg.rhs.numerics, config.NoNumerics):
      pass
    elif isinstance(cfg.rhs.numerics, config.IntNumerics):
      out /= rhs_scale
    else:
      assert False
    # # Future scale granularity optimization.
    # In 1x1 conv, each pixel (spatial location) can have different scales
    # in 1xN (rows x colums) conv each row can have different scale, but
    # columns need to share the scales ,  because we are adding pixels across.
    #
    # For patch convs we could have separate scales per patch.
    # We don't do that optimization, because there is a  Flax op: ConvLocal
    # using lax.conv_general_dilated_local which uses lax.dot_general.
    #
    # Dilations: If a dilation of LHS is bigger than the total spatial size of
    # RHS, we could use separe (per LHS pixel) scales.
    # The same applies to dilated RHS.
    # We don't do that optimization yet.
    #
    # We can have different scales across different groups.
    # This applies to both feature and batch.
    return out

  return my_conv_general_dilated
