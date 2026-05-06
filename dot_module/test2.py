#from maxtext.src.maxtext.models import deepseek_batchsplit

import os
#os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=16"

import jax
# jax.config.update("jax_platforms", "cpu")


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

from dot_module.dot import dot
from absl.testing import parameterized


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

  # def test_quantized_dot_forward(self):
  #   cfg_unquant = MockQuantConfig(quantize=False)
  #   cfg_quant = MockQuantConfig(quantize=True)
    
  #   rng = jax.random.PRNGKey(42)
  #   k1, k2 = jax.random.split(rng)
    
  #   # Scale down inputs to prevent aggressive clipping at max_val=1.0
  #   act = jax.random.normal(k1, (2, 16, 64), dtype=jnp.bfloat16) * 0.2
  #   w = jax.random.normal(k2, (64, 32), dtype=jnp.bfloat16) * 0.2

  #   # 2. Explicitly bind the arrays to the mesh so `aval.sharding.mesh` populates correctly
  #   sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
  #   act = jax.device_put(act, sharding)
  #   w = jax.device_put(w, sharding)

  #   with jax.set_mesh(self.mesh):
  #     # JIT to ensure XLA sharding annotations are respected
  #     dot_fn = jax.jit(dot, static_argnames=["config", "axes"])
      
  #     expected_out = jnp.tensordot(act, w, axes=1)
  #     #expected_out = dot_fn(act, w, config=cfg_unquant, axes=1)
  #     actual_out = dot_fn(act, w, config=cfg_quant, axes=1)
      

  #   self.assertEqual(actual_out.shape, expected_out.shape)
  #   self.assertEqual(actual_out.dtype, jnp.bfloat16)
    
  #   # Forward pass uses fp8_e4m3fn - tolerance relaxed to 10%
  #   self.assert_close(actual_out, expected_out, rtol=1e-1, atol=1e-1)

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
    print("actual_out", jnp.maxexpect_out-actual_out)

    self.assertEqual(actual_grad_act.shape, expected_grad_act.shape)
    self.assertEqual(actual_grad_w.shape, expected_grad_w.shape)
    self.assertEqual(actual_grad_act.dtype, jnp.bfloat16)
    self.assertEqual(actual_grad_w.dtype, jnp.bfloat16)

    print("\nactual_grad_act", actual_grad_act)
    print("expected_grad_act", expected_grad_act)

    print("\nactual_grad_w", actual_grad_w)
    print("expected_grad_w", expected_grad_w)

    # Backward pass uses fp8_e5m2 which is heavily truncated - tolerance relaxed to 20%
    self.assert_close(actual_grad_act, expected_grad_act, rtol=2e-1, atol=2e-1)
    self.assert_close(actual_grad_w, expected_grad_w, rtol=2e-1, atol=2e-1)


if __name__ == "__main__":
  unittest.main()
