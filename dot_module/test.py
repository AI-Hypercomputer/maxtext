import unittest
from unittest.mock import MagicMock
import jax
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

from dot_module.dot import dot

# def cast_reduced_from(arr, reduced_arr):
#   aval = jax.typeof(reduced_arr)
#   # In shard map
#   if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
#     for axis in aval.mat.reduced:
#       arr = jax.lax.pcast(arr, axis, to="reduced")
#     return arr
#   # Outside shard map
#   return jax.reshard(arr, aval.sharding)


# def make_scale_tensor(scale, arr):
#   scale_tensor = jnp.full_like(arr, scale, dtype=jnp.bfloat16)
#   return cast_reduced_from(scale_tensor, arr)


# # NOTE: Only w can have reduced axes, which is true for how we use dot.
# @functools.partial(jax.custom_vjp, nondiff_argnames=["config", "axes"])
# def dot(act, w, config, axes=1):
#   """Quantized tensordot replacement."""
#   return dot_fwd(act, w, config=config, axes=axes)[0]


# def _get_max_min(target_dtype):
#   if target_dtype == jnp.int4 or target_dtype == jnp.int8:
#     return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min
#   else:
#     return jnp.finfo(target_dtype).max.astype(jnp.bfloat16), jnp.finfo(
#         target_dtype
#     ).min.astype(jnp.bfloat16)


# def dot_fwd(act, w, config, axes=1):
#   """Forward pass for the quantized tensordot.

#   This function performs a tensordot operation between `act` and `w`.
#   If quantization is enabled in the `config`, it quantizes `act` and `w`
#   using static float8 quantization before performing the dot product.
#   The result is then dequantized. If quantization is disabled, it
#   behaves like a standard `jnp.tensordot`.

#   Args:
#     act: The first array.
#     w: The second array (weights).
#     config: Configuration object containing quantization settings.
#     axes: The axes to contract over, as in `jnp.tensordot`.

#   Returns:
#     A tuple containing:
#       - The result of the dot product.
#       - Residuals for the backward pass. If quantization is disabled, this
#         is the VJP function. Otherwise, it contains `(q_act, q_w, a_scale,
#         w_scale)`.
#   """
#   if not config.quantization:
#     result, vjp_fn = jax.vjp(
#         functools.partial(jnp.tensordot, axes=axes), act, w
#     )
#     return result, vjp_fn

#   w_calib_method = config.weight_quantization_calibration_method
#   if not w_calib_method.startswith("fixed"):
#     raise ValueError(
#         "Only static weight quantization is supported, but got"
#         f" {w_calib_method}"
#     )

#   w_parts = w_calib_method.split(",")
#   if len(w_parts) != 3:
#     raise ValueError(
#         f"Unexpected format for weight calibration method: {w_calib_method}"
#     )

#   a_calib_method = config.act_quantization_calibration_method
#   if not a_calib_method.startswith("fixed"):
#     raise ValueError(
#         f"Only static act quantization is supported, but got {a_calib_method}"
#     )

#   a_parts = a_calib_method.split(",")
#   if len(a_parts) != 3:
#     raise ValueError(
#         f"Unexpected format for act calibration method: {a_calib_method}"
#     )

#   fwd_dtype = jnp.float8_e4m3fn
#   dtype_max, dtype_min = _get_max_min(fwd_dtype)
#   w_max_val = float(w_parts[2])
#   w_scale = w_max_val / dtype_max
#   w_scale = jnp.where(w_scale == 0, 1.0, w_scale)
#   # scale_w must be converted to a tensor because w_grad has reduced axes.
#   scale_w_tensor = make_scale_tensor(w_scale, w)
#   min_w_bound = make_scale_tensor(dtype_min, w)
#   max_w_bound = make_scale_tensor(dtype_max, w)
#   q_w = jnp.clip(w / scale_w_tensor, min_w_bound, max_w_bound).astype(fwd_dtype)

#   a_max_val = float(a_parts[2])
#   a_scale = a_max_val / dtype_max
#   a_scale = jnp.where(a_scale == 0, 1.0, a_scale)
#   q_act = jnp.clip(act / a_scale, dtype_min, dtype_max).astype(fwd_dtype)

#   # Call tensordot on quantized inputs
#   qres = jnp.tensordot(q_act, q_w, axes=axes)
#   result = qres.astype(jnp.bfloat16) * a_scale * w_scale
#   return result.astype(jnp.bfloat16), (q_act, q_w, a_scale, w_scale)


# def dot_bwd(config, axes, res, g):
#   """Backward pass for the quantized tensordot.

#   This function computes the gradients for the inputs of the `dot` function,
#   handling both quantized and non-quantized scenarios. For quantized
#   operations, it performs dynamic quantization on the gradient `g` and uses
#   quantized activations and weights from the forward pass residuals (`res`)
#   to compute the gradients for `act` and `w`.

#   Args:
#     config: Configuration object containing quantization settings.
#     axes: The axes to contract over, as in `jnp.tensordot`.
#     res: Residuals from the forward pass (`dot_fwd`). If quantization is
#       disabled, this is the VJP function. Otherwise, it contains `(q_act, q_w,
#       a_scale, w_scale)`.
#     g: The gradient of the output of the `dot` function.

#   Returns:
#     A tuple containing the gradients for `act` and `w`.
#   """

#   grad_calib_method = config.bwd_quantization_calibration_method
#   if grad_calib_method and not grad_calib_method.startswith("absmax"):
#     raise ValueError(
#         "Only dynamic quantization is supported for bwd pass, but got"
#         f" {grad_calib_method}"
#     )
#   grad = g
#   if not config.quantization:
#     vjp_fn = res
#     act_grad, w_grad = vjp_fn(grad)
#     return act_grad, w_grad

#   q_act, q_w, a_scale, w_scale = res

#   act_grad_axes = q_w.ndim - axes
#   w_grad_axes = q_act.ndim - axes

#   aval = jax.typeof(q_w)

#   bwd_dtype = jnp.float8_e5m2
#   dtype_max, dtype_min = _get_max_min(bwd_dtype)
#   def quantize_grad(contracting_axes, expand_dims_count=0):
#     abs_max_g = jnp.max(jnp.abs(grad), axis=contracting_axes, keepdims=True)
#     if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
#       for axis_name in aval.sharding.mesh.axis_names:
#         abs_max_g = jax.lax.pmax(abs_max_g, axis_name)
#     scale_g = abs_max_g / dtype_max
#     scale_g = jnp.where(scale_g == 0, 1.0, scale_g)
#     q_grad = jnp.clip(grad / scale_g, dtype_min, dtype_max).astype(bwd_dtype)
#     scale_g = jnp.squeeze(scale_g, axis=tuple(contracting_axes))
#     for _ in range(expand_dims_count):
#       scale_g = jnp.expand_dims(scale_g, axis=-1)
#     return q_grad, scale_g

#   # For act_grad computation, we contract over the last act_grad_axes axes of grad.
#   # So we quantize along the non-contracting axes (the first ones).
#   # We reduce over the contracting axes.
#   contracting_axes_for_act = list(range(grad.ndim - act_grad_axes, grad.ndim))
#   q_grad_for_act, scale_g_for_act = quantize_grad(
#       contracting_axes_for_act, expand_dims_count=axes
#   )

#   # For w_grad computation, we contract over the first w_grad_axes axes of grad.
#   # So we quantize along the non-contracting axes (the remaining ones).
#   # We reduce over the contracting axes.
#   contracting_axes_for_w = list(range(w_grad_axes))
#   q_grad_for_w, scale_g_for_w = quantize_grad(
#       contracting_axes_for_w, expand_dims_count=0
#   )

#   act_grad_raw = jax.lax.dot_general(
#       q_grad_for_act,
#       q_w,
#       dimension_numbers=(
#           (list(range(grad.ndim - act_grad_axes, grad.ndim)), list(range(axes, q_w.ndim))),
#           ([], []),
#       ),
#   )

#   aval = jax.typeof(q_w)
#   # In shard map
#   if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
#     # Call dot_general without out_sharding. We cast the proper axes to unreduced
#     # after the dot.
#     w_grad_raw = jax.lax.dot_general(
#         q_act,
#         q_grad_for_w,
#         dimension_numbers=((list(range(w_grad_axes)), list(range(w_grad_axes))), ([], [])),
#     )
#     for axis in aval.mat.reduced:
#       w_grad_raw = jax.lax.pcast(w_grad_raw, axis, to="unreduced")
#   # Outside shard map
#   else:
#     w_grad_raw = jax.lax.dot_general(
#         q_act,
#         q_grad_for_w,
#         dimension_numbers=((list(range(w_grad_axes)), list(range(w_grad_axes))), ([], [])),
#         out_sharding=jax.sharding.NamedSharding(
#             aval.sharding.mesh,
#             jax.sharding.PartitionSpec(unreduced=aval.sharding.spec.reduced),
#         ),
#     )
#   act_grad = act_grad_raw.astype(jnp.bfloat16) * w_scale * scale_g_for_act
#   # scale_z_for_w has shape NC_y. w_grad_raw has shape (C_x, NC_y).
#   # We need to make sure it broadcasts.
#   scale_g_tensor = make_scale_tensor(scale_g_for_w, q_w)
#   w_grad = (
#       w_grad_raw.astype(jnp.bfloat16)
#       * make_scale_tensor(a_scale, q_w)
#       * scale_g_tensor
#   )

#   return act_grad.astype(jnp.bfloat16), w_grad.astype(jnp.bfloat16)


# dot.defvjp(dot_fwd, dot_bwd)

def get_mock_config(quantized=True):
    config = MagicMock()
    config.quantization = quantized
    if quantized:
        config.weight_quantization_calibration_method = "fixed,-224,224"
        config.act_quantization_calibration_method = "fixed,-224,224"
        config.bwd_quantization_calibration_method = "absmax"
    return config


class TestDotFunction(unittest.TestCase):

  def test_unquantized_forward_and_backward(self):
      config = get_mock_config(quantized=False)
      key = jax.random.PRNGKey(0)
      k1, k2 = jax.random.split(key)
      
      act = jax.random.normal(k1, (4, 16))
      w = jax.random.normal(k2, (16, 8))
      
      # Test Forward
      custom_res = dot(act, w, config=config, axes=1)
      jnp_res = jnp.tensordot(act, w, axes=1)
      
      self.assertTrue(jnp.allclose(custom_res, jnp_res, atol=1e-5))
      
      # Test Backward
      def loss_fn(a, w_):
          return jnp.sum(dot(a, w_, config=config, axes=1))
          
      def ref_loss_fn(a, w_):
          return jnp.sum(jnp.tensordot(a, w_, axes=1))
          
      custom_grad_act, custom_grad_w = jax.grad(loss_fn, argnums=(0, 1))(act, w)
      ref_grad_act, ref_grad_w = jax.grad(ref_loss_fn, argnums=(0, 1))(act, w)
      
      self.assertTrue(jnp.allclose(custom_grad_act, ref_grad_act, atol=1e-5))
      self.assertTrue(jnp.allclose(custom_grad_w, ref_grad_w, atol=1e-5))

  def test_quantized_forward_and_backward_local(self):
      config = get_mock_config(quantized=True)
      key = jax.random.PRNGKey(1)
      
      # Use values that will trigger the fixed scaling (-224 to 224)
      act = jax.random.uniform(key, (8, 32), minval=-100, maxval=100)
      w = jax.random.uniform(key, (32, 16), minval=-100, maxval=100)
      
      # Forward pass
      res = dot(act, w, config=config, axes=1)
      
      # Ensure output is cast back to bfloat16 as defined in your code
      self.assertEqual(res.dtype, jnp.bfloat16)
      self.assertEqual(res.shape, (8, 16))
      
      # Backward pass (Ensure VJP doesn't crash on shapes/types)
      def loss_fn(a, w_):
          return jnp.sum(dot(a, w_, config=config, axes=1))
          
      grad_act, grad_w = jax.grad(loss_fn, argnums=(0, 1))(act, w)
      
      self.assertEqual(grad_act.dtype, jnp.bfloat16)
      self.assertEqual(grad_w.dtype, jnp.bfloat16)
      self.assertEqual(grad_act.shape, act.shape)
      self.assertEqual(grad_w.shape, w.shape)


  def test_quantized_shard_map_backward(self):
      config = get_mock_config(quantized=True)
      
      # Create a mock 2-device mesh (works on CPU for testing)
      devices = jax.devices()
      if len(devices) < 2:
          self.skipTest("Requires at least 2 devices (use CPU with flags if needed)")
          
      mesh = Mesh(np.array(devices[:2]), ('data',))
      
      act_shape = (4, 16)
      w_shape = (16, 8)
      
      # Data parallel partition spec
      act_spec = P('data', None)
      w_spec = P(None, None) # Replicated weights
      out_spec = P('data', None)

      @functools.partial(
          jax.shard_map, 
          mesh=mesh, 
          in_specs=(act_spec, w_spec), 
          out_specs=out_spec,
          check_vma=True
      )
      def sharded_fwd(a, w_):
          return dot(a, w_, config=config, axes=1)

      # To test the backward pass inside the shard map, we define 
      # a loss function that itself is mapped.
      @functools.partial(
          jax.shard_map, 
          mesh=mesh, 
          in_specs=(act_spec, w_spec), 
          out_specs=(act_spec, P(None, None)), # Weight grads should be replicated/psummed later
          check_vma=True
      )
      def sharded_bwd(a, w_):
          def loss(a_inner, w_inner):
              return jnp.sum(dot(a_inner, w_inner, config=config, axes=1))
          return jax.grad(loss, argnums=(0, 1))(a, w_)

      key = jax.random.PRNGKey(2)
      act = jax.device_put(jax.random.normal(key, act_shape), NamedSharding(mesh, act_spec))
      w = jax.device_put(jax.random.normal(key, w_shape), NamedSharding(mesh, w_spec))

      # 1. Test it compiles and runs forward
      out = sharded_fwd(act, w)
      self.assertEqual(out.shape, (4, 8))

      # 2. Test the backward branch for AxisType.Manual is hit and succeeds
      grad_act, grad_w = sharded_bwd(act, w)
      
      self.assertEqual(grad_act.shape, act_shape)
      # Because local chunk size for 'data' axis on 2 devices is half the batch:
      self.assertEqual(grad_act.addressable_data(0).shape, (2, 16)) 
      self.assertEqual(grad_w.shape, w_shape)


if __name__ == "__main__":
  unittest.main()
