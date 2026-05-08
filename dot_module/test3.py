"""
dot with scale=1 for both fwd and bwd
check against bf16 tensordot
"""


import os
import jax

import unittest
from unittest.mock import MagicMock
import jax.numpy as jnp
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


import contextlib
import functools
import math
from typing import Sequence

import jax
import jax.numpy as jnp
#from maxtext.src.maxtext.kernels import attention, sort_activations

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

# Import your dot function here
# from your_module import dot 
import numpy as np

#from dot_module.dot import dot
from absl.testing import parameterized

##############



def cast_reduced_from(arr, reduced_arr):
  aval = jax.typeof(reduced_arr)
  # In shard map
  if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
    for axis in aval.mat.reduced:
      arr = jax.lax.pcast(arr, axis, to="reduced")
    return arr
  # Outside shard map
  return jax.reshard(arr, aval.sharding)


def make_scale_tensor(scale, arr):
  scale_tensor = jnp.full_like(arr, scale, dtype=jnp.bfloat16)
  return cast_reduced_from(scale_tensor, arr)


# NOTE: Only w can have reduced axes, which is true for how we use dot.
@functools.partial(jax.custom_vjp, nondiff_argnames=["config", "axes"])
def dot(act, w, config, axes=1):
  """Quantized tensordot replacement."""
  return dot_fwd(act, w, config=config, axes=axes)[0]


def _get_max_min(target_dtype):
  if target_dtype == jnp.int4 or target_dtype == jnp.int8:
    return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min
  else:
    return jnp.finfo(target_dtype).max.astype(jnp.bfloat16), jnp.finfo(
        target_dtype
    ).min.astype(jnp.bfloat16)


def dot_fwd(act, w, config, axes=1):
  """Forward pass for the quantized tensordot.

  This function performs a tensordot operation between `act` and `w`.
  If quantization is enabled in the `config`, it quantizes `act` and `w`
  using static float8 quantization before performing the dot product.
  The result is then dequantized. If quantization is disabled, it
  behaves like a standard `jnp.tensordot`.

  Args:
    act: The first array.
    w: The second array (weights).
    config: Configuration object containing quantization settings.
    axes: The axes to contract over, as in `jnp.tensordot`.

  Returns:
    A tuple containing:
      - The result of the dot product.
      - Residuals for the backward pass. If quantization is disabled, this
        is the VJP function. Otherwise, it contains `(q_act, q_w, a_scale,
        w_scale)`.
  """
  if not config.quantization:
    result, vjp_fn = jax.vjp(
        functools.partial(jnp.tensordot, axes=axes), act, w
    )
    return result, vjp_fn

  w_calib_method = config.weight_quantization_calibration_method
  if not w_calib_method.startswith("fixed"):
    raise ValueError(
        "Only static weight quantization is supported, but got"
        f" {w_calib_method}"
    )

  w_parts = w_calib_method.split(",")
  if len(w_parts) != 3:
    raise ValueError(
        f"Unexpected format for weight calibration method: {w_calib_method}"
    )

  a_calib_method = config.act_quantization_calibration_method
  if not a_calib_method.startswith("fixed"):
    raise ValueError(
        f"Only static act quantization is supported, but got {a_calib_method}"
    )

  a_parts = a_calib_method.split(",")
  if len(a_parts) != 3:
    raise ValueError(
        f"Unexpected format for act calibration method: {a_calib_method}"
    )

  #fwd_dtype = jnp.float8_e4m3fn
  fwd_dtype = jnp.bfloat16
  dtype_max, dtype_min = _get_max_min(fwd_dtype)
  w_max_val = float(w_parts[2])
  w_scale = w_max_val / dtype_max
  w_scale = jnp.where(w_scale == 0, 1.0, w_scale)
  w_scale = 1.0
  # scale_w must be converted to a tensor because w_grad has reduced axes.
  scale_w_tensor = make_scale_tensor(w_scale, w)
  min_w_bound = make_scale_tensor(dtype_min, w)
  max_w_bound = make_scale_tensor(dtype_max, w)
  q_w = jnp.clip(w / scale_w_tensor, min_w_bound, max_w_bound).astype(fwd_dtype)

  a_max_val = float(a_parts[2])
  a_scale = a_max_val / dtype_max
  a_scale = jnp.where(a_scale == 0, 1.0, a_scale)
  a_scale = 1.0
  q_act = jnp.clip(act / a_scale, dtype_min, dtype_max).astype(fwd_dtype)

  # Call tensordot on quantized inputs
  qres = jnp.tensordot(q_act, q_w, axes=axes)
  result = qres.astype(jnp.bfloat16) * a_scale * w_scale
  return result.astype(jnp.bfloat16), (q_act, q_w, a_scale, w_scale)


def dot_bwd(config, axes, res, g):
  """Backward pass for the quantized tensordot.

  This function computes the gradients for the inputs of the `dot` function,
  handling both quantized and non-quantized scenarios. For quantized
  operations, it performs dynamic quantization on the gradient `g` and uses
  quantized activations and weights from the forward pass residuals (`res`)
  to compute the gradients for `act` and `w`.

  Args:
    config: Configuration object containing quantization settings.
    axes: The axes to contract over, as in `jnp.tensordot`.
    res: Residuals from the forward pass (`dot_fwd`). If quantization is
      disabled, this is the VJP function. Otherwise, it contains `(q_act, q_w,
      a_scale, w_scale)`.
    g: The gradient of the output of the `dot` function.

  Returns:
    A tuple containing the gradients for `act` and `w`.
  """

  grad_calib_method = config.bwd_quantization_calibration_method
  if grad_calib_method and not grad_calib_method.startswith("absmax"):
    raise ValueError(
        "Only dynamic quantization is supported for bwd pass, but got"
        f" {grad_calib_method}"
    )
  grad = g
  if not config.quantization:
    vjp_fn = res
    act_grad, w_grad = vjp_fn(grad)
    return act_grad, w_grad

  q_act, q_w, a_scale, w_scale = res

  act_grad_axes = q_w.ndim - axes
  w_grad_axes = q_act.ndim - axes

  aval = jax.typeof(q_w)

  #bwd_dtype = jnp.float8_e5m2
  bwd_dtype = jnp.bfloat16
  dtype_max, dtype_min = _get_max_min(bwd_dtype)
  # TODO: check this
  def quantize_grad(contracting_axes, expand_dims_count=0):
    abs_max_g = jnp.max(jnp.abs(grad), axis=contracting_axes, keepdims=True)
    if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
      for axis_name in aval.sharding.mesh.axis_names:
        abs_max_g = jax.lax.pmax(abs_max_g, axis_name)
    scale_g = abs_max_g / dtype_max
    scale_g = jnp.where(scale_g == 0, 1.0, scale_g)
    q_grad = jnp.clip(grad / scale_g, dtype_min, dtype_max).astype(bwd_dtype)
    scale_g = jnp.squeeze(scale_g, axis=tuple(contracting_axes))
    for _ in range(expand_dims_count):
      scale_g = jnp.expand_dims(scale_g, axis=-1)
    return grad, 1.0

  # For act_grad computation, we contract over the last act_grad_axes axes of grad.
  # So we quantize along the non-contracting axes (the first ones).
  # We reduce over the contracting axes.
  contracting_axes_for_act = list(range(grad.ndim - act_grad_axes, grad.ndim))
  q_grad_for_act, scale_g_for_act = quantize_grad(
      contracting_axes_for_act, expand_dims_count=axes
  )

  # For w_grad computation, we contract over the first w_grad_axes axes of grad.
  # So we quantize along the non-contracting axes (the remaining ones).
  # We reduce over the contracting axes.
  contracting_axes_for_w = list(range(w_grad_axes))
  q_grad_for_w, scale_g_for_w = quantize_grad(
      contracting_axes_for_w, expand_dims_count=0
  )

  act_grad_raw = jax.lax.dot_general(
      q_grad_for_act,
      q_w,
      dimension_numbers=(
          (list(range(grad.ndim - act_grad_axes, grad.ndim)), list(range(axes, q_w.ndim))),
          ([], []),
      ),
  )

  aval = jax.typeof(q_w)
  # In shard map
  if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
    # Call dot_general without out_sharding. We cast the proper axes to unreduced
    # after the dot.
    w_grad_raw = jax.lax.dot_general(
        q_act,
        q_grad_for_w,
        dimension_numbers=((list(range(w_grad_axes)), list(range(w_grad_axes))), ([], [])),
    )
    for axis in aval.mat.reduced:
      w_grad_raw = jax.lax.pcast(w_grad_raw, axis, to="unreduced")
  # Outside shard map
  else:
    w_grad_raw = jax.lax.dot_general(
        q_act,
        q_grad_for_w,
        dimension_numbers=((list(range(w_grad_axes)), list(range(w_grad_axes))), ([], [])),
        out_sharding=jax.sharding.NamedSharding(
            aval.sharding.mesh,
            jax.sharding.PartitionSpec(unreduced=aval.sharding.spec.reduced),
        ),
    )
  act_grad = act_grad_raw.astype(jnp.bfloat16) * w_scale * scale_g_for_act
  # scale_z_for_w has shape NC_y. w_grad_raw has shape (C_x, NC_y).
  # We need to make sure it broadcasts.
  scale_g_tensor = make_scale_tensor(scale_g_for_w, q_w)
  w_grad = (
      w_grad_raw.astype(jnp.bfloat16)
      * make_scale_tensor(a_scale, q_w)
      * scale_g_tensor
  )

  return act_grad.astype(jnp.bfloat16), w_grad.astype(jnp.bfloat16)


dot.defvjp(dot_fwd, dot_bwd)

###########################


class MockQuantConfig:
  """Mocks the config attributes required by dot"""
  def __init__(self, quantize=True):
    self.quantization = quantize
    self.weight_quantization_calibration_method = "fixed,-224,224"
    self.act_quantization_calibration_method = "fixed,-224,224"
    self.bwd_quantization_calibration_method = "absmax"

class DeepSeekBatchsplitDotTest(parameterized.TestCase):
  """Tests the custom quantized dot function for forward and backward passes."""

  def setUp(self):
    super().setUp()
    # 1. Explicitly define axis_types for the mesh to prevent the IndexError
    devices = np.array(jax.devices())
    self.mesh = jax.sharding.Mesh(
        devices, 
        ("data",), 
        axis_types=(jax.sharding.AxisType.Auto,)
    )

  def assert_close(self, a, b, rtol=2e-01, atol=2e-01):
    self.assertTrue(
        jax.numpy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False),
        msg=(f"The following two arrays are not close\n{a=}\n{b=}\n" 
             f"total difference is {jnp.sum(jnp.abs(a - b))=}"),
    )


  def test_quantized_dot_backward(self):
    cfg_unquant = MockQuantConfig(quantize=False)
    cfg_quant = MockQuantConfig(quantize=True)
    
    rng = jax.random.PRNGKey(1337)
    k1, k2 = jax.random.split(rng)
    
    act = jax.random.normal(k1, (2, 16, 64), dtype=jnp.bfloat16) * 0.2
    w = jax.random.normal(k2, (64, 32), dtype=jnp.bfloat16) * 0.2

    # Bind the arrays to the mesh
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    act = jax.device_put(act, sharding)
    w = jax.device_put(w, sharding)


    # 1. GROUND TRUTH (Expected): Pure tensordot backward pass
    def pure_loss_fn(act_input, w_input):
        out = jnp.tensordot(act_input, w_input, axes=1)
        # Squaring out here gives varied gradients so rows aren't identical!
        return jnp.sum(out ** 2) 

    # Calculate exact mathematical gradients
    expected_grad_act, expected_grad_w = jax.grad(pure_loss_fn, argnums=(0, 1))(act, w)
    expect_out = pure_loss_fn(act, w)



    def loss_fn(act_input, w_input, config):
      out = dot(act_input, w_input, config=config, axes=1)
      return jnp.sum(out ** 2)


    with jax.set_mesh(self.mesh):
      grad_fn = jax.jit(
          jax.value_and_grad(loss_fn, argnums=(0, 1)), 
          static_argnames=["config"]
      )
      
      #expect_out, (expected_grad_act, expected_grad_w) = grad_fn(act, w, config=cfg_unquant)
      actual_out, (actual_grad_act, actual_grad_w) = grad_fn(act, w, config=cfg_quant)
      



    print("\nexpect_out", expect_out)
    print("actual_out", actual_out)

    self.assertEqual(actual_grad_act.shape, expected_grad_act.shape)
    self.assertEqual(actual_grad_w.shape, expected_grad_w.shape)
    self.assertEqual(actual_grad_act.dtype, jnp.bfloat16)
    self.assertEqual(actual_grad_w.dtype, jnp.bfloat16)

    print("\nactual_grad_act", actual_grad_act)
    print("expected_grad_act", expected_grad_act)

    print("\nactual_grad_w", actual_grad_w)
    print("expected_grad_w", expected_grad_w)

    # Backward pass uses fp8_e5m2 which is heavily truncated - tolerance relaxed to 20%
    self.assert_close(actual_grad_act, expected_grad_act, rtol=1e-5, atol=1e-5)
    self.assert_close(actual_grad_w, expected_grad_w, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
