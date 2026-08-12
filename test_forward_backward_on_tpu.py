# Copyright 2026 Google LLC
"""Verification script for forward-backward execution on 4 TPU chips with 2-layer parameter buffer host offloading."""

import os
import sys
import numpy as np

# Ensure jax uses TPU
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
import optax

from maxtext.layers import nnx_scan
from maxtext.common.common_types import HyperConnectionType, MODEL_MODE_TRAIN
from maxtext.layers import mhc
from maxtext.models import qwen3
from maxtext import pyconfig


def test_scanned_layer_forward_backward_tpu():
  print("=" * 80)
  print(f"[TEST 1] Testing Scanned Layer Forward-Backward on {len(jax.devices())} TPU Devices")
  print(f"TPU Devices: {jax.devices()}")
  print("=" * 80)

  devices = np.array(jax.devices())
  mesh = Mesh(devices, ("data",))

  length = 4
  dim = 128
  batch_size = 4

  class ScannedTransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
      w_init = jax.random.normal(rngs.params(), (dim, dim)) * 0.02
      self.kernel = nnx.Param(
          w_init,
          out_sharding=NamedSharding(mesh, P(None, None)),
      )
      self.bias = nnx.Param(
          jnp.zeros((dim,)),
          out_sharding=NamedSharding(mesh, P(None)),
      )
      self.scale = nnx.Param(
          jnp.ones((dim,)),
          out_sharding=NamedSharding(mesh, P(None)),
      )

    def __call__(self, x):
      # Layer norm
      mean = jnp.mean(x, axis=-1, keepdims=True)
      var = jnp.var(x, axis=-1, keepdims=True)
      x_norm = (x - mean) / jnp.sqrt(var + 1e-6) * self.scale[...]
      # Linear + GELU + residual
      y = jnp.matmul(x_norm, self.kernel[...]) + self.bias[...]
      y = jax.nn.gelu(y)
      return x + y

  rngs = nnx.Rngs(params=42)

  # Create scanned stack of 4 layers
  stacked_model = nnx_scan.create_scanned_layers(
      ScannedTransformerBlock,
      length=length,
      param_scan_axis=0,
      metadata_axis_name="layers",
      rngs=rngs,
  )

  # Split into graphdef and params
  graphdef, params = nnx.split(stacked_model, nnx.Param)

  print(f"Stacked params structure:")
  for path, val in jax.tree_util.tree_leaves_with_path(params):
    p_str = "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path)
    print(f"  {p_str}: shape={val.shape}, dtype={val.dtype}")

  # Host offload with 2-layer ping-pong double buffer
  def loss_fn(current_params, inputs, targets):
    model = nnx.merge(graphdef, current_params)
    out = nnx_scan.apply_scanned_layers(
        model,
        inputs,
        length=length,
        param_scan_axis=0,
        apply_fn=lambda layer, carry: layer(carry),
        remat=True,
        parameter_memory_host_offload=True,
        parameter_memory_two_layer_buffer=True,
    )
    loss = jnp.mean((out - targets) ** 2)
    return loss, out

  optimizer = optax.adamw(learning_rate=1e-3)
  opt_state = optimizer.init(params)

  @jax.jit
  def train_step(current_params, current_opt_state, inputs, targets):
    (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params, inputs, targets)
    updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
    next_params = optax.apply_updates(current_params, updates)
    return next_params, next_opt_state, loss

  # Input data sharded across data axis
  data_sharding = NamedSharding(mesh, P("data", None, None))
  inputs = jax.device_put(
      jax.random.normal(jax.random.PRNGKey(0), (batch_size, 32, dim)),
      data_sharding,
  )
  targets = jax.device_put(
      jax.random.normal(jax.random.PRNGKey(1), (batch_size, 32, dim)),
      data_sharding,
  )

  print("\nCompiling train_step on TPU...")
  lowered = train_step.lower(params, opt_state, inputs, targets)
  compiled = lowered.compile()
  print("Compilation successful! Running 5 training steps...")

  curr_params = params
  curr_opt_state = opt_state
  for step in range(5):
    curr_params, curr_opt_state, loss_val = compiled(curr_params, curr_opt_state, inputs, targets)
    loss_float = float(loss_val)
    print(f"  Step {step}: Loss = {loss_float:.6f}")
    assert not np.isnan(loss_float), f"Loss is NaN at step {step}"

  print("Test 1 PASSED: Scanned layer forward-backward with 2-layer buffer executed cleanly on TPU!\n")


def test_mhc_layer_on_tpu():
  print("=" * 80)
  print(f"[TEST 2] Testing mHC Layer Forward-Backward on {len(jax.devices())} TPU Devices")
  print("=" * 80)

  devices = np.array(jax.devices())
  mesh = Mesh(devices, ("data",))

  config_dict = {
      "model_name": "qwen3-next-80b-a3b",
      "mhc_expansion_rate": 4,
      "enable_mhc_lite": True,
      "sinkhorn_iterations": 3,
      "normalization_layer_epsilon": 1e-6,
      "parameter_memory_host_offload": True,
      "parameter_memory_two_layer_buffer": True,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "matmul_precision": "default",
  }
  mock_config = type("Config", (), config_dict)()
  rngs = nnx.Rngs(params=0)

  mhc_layer = mhc.ManifoldConstrainedHyperConnections(
      config=mock_config,
      dim=64,
      mesh=mesh,
      rngs=rngs,
  )

  graphdef, params, *rest = nnx.split(mhc_layer, nnx.Param, ...)

  def mhc_loss_fn(current_params, x):
    layer = nnx.merge(graphdef, current_params, *rest)
    def dummy_norm(inp):
      return inp
    def dummy_branch(inputs, **kwargs):
      return inputs, None, None

    out, _ = layer(
        norm_fn=dummy_norm,
        branch_fn=dummy_branch,
        x=x,
        mhc_type=HyperConnectionType.MLP_MOE,
    )
    return jnp.mean(out ** 2)

  @jax.jit
  def mhc_step(current_params, x):
    loss, grads = jax.value_and_grad(mhc_loss_fn)(current_params, x)
    return loss, grads

  batch, seq, k, dim = 4, 16, 4, 64
  data_sharding = NamedSharding(mesh, P("data", None, None, None))
  x = jax.device_put(
      jnp.ones((batch, seq, k, dim), dtype=jnp.bfloat16),
      data_sharding,
  )

  print("Compiling mHC train step...")
  loss_val, grads = mhc_step(params, x)
  loss_float = float(loss_val)
  print(f"  mHC Loss = {loss_float:.6f}")
  assert not np.isnan(loss_float), "mHC Loss is NaN"
  print("Test 2 PASSED: mHC forward-backward executed cleanly on TPU!\n")


def test_qwen3_scannable_block_on_tpu():
  print("=" * 80)
  print(f"[TEST 3] Testing Qwen3-Next Scannable Block with 2-Layer Buffer on {len(jax.devices())} TPU Devices")
  print("=" * 80)

  devices = np.array(jax.devices())
  mesh = Mesh(devices, ("data",))

  config = pyconfig.initialize(
      [
          None,
          "src/maxtext/configs/base.yml",
          "model_name=qwen3-next-80b-a3b",
          "override_model_config=True",
          "base_num_decoder_layers=3",
          "base_emb_dim=256",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "base_mlp_dim=512",
          "base_moe_mlp_dim=256",
          "num_experts=4",
          "num_experts_per_tok=2",
          "vocab_size=256",
          "max_target_length=16",
          "per_device_batch_size=1",
          "param_scan_axis=0",
          "parameter_memory_host_offload=True",
          "parameter_memory_two_layer_buffer=True",
          "use_gdn_kernel=False",
          "use_hybrid_gdn=False",
          "attention=dot_product",
          "ici_fsdp_parallelism=4",
          "skip_jax_distributed_system=True",
      ]
  )

  rngs = nnx.Rngs(params=0)
  block = qwen3.Qwen3NextScannableBlock(
      config=config,
      mesh=mesh,
      model_mode=MODEL_MODE_TRAIN,
      quant=None,
      num_of_layers=2,  # 2 local GatedDeltaNet layers
      apply_internal_remat=False,
      rngs=rngs,
  )
  nnx.pop(block, (nnx.RngCount, nnx.RngKey, nnx.Intermediate))
  graphdef, params, *rest = nnx.split(block, nnx.Param, ...)

  batch, seq, dim = 4, 16, config.emb_dim
  positions = jnp.tile(jnp.arange(seq)[None, :], (batch, 1))
  segment_ids = jnp.zeros((batch, seq), dtype=jnp.int32)

  def block_loss_fn(current_params, hidden_states):
    reconstructed_block = nnx.merge(graphdef, current_params, *rest)
    out = reconstructed_block(
        hidden_states,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    if isinstance(out, tuple):
      out = out[0]
    return jnp.mean(out ** 2)

  @jax.jit
  def block_train_step(current_params, hidden_states):
    loss, grads = jax.value_and_grad(block_loss_fn)(current_params, hidden_states)
    return loss, grads

  data_sharding = NamedSharding(mesh, P("data", None, None))
  hidden_states = jax.device_put(
      jnp.ones((batch, seq, dim), dtype=jnp.bfloat16),
      data_sharding,
  )

  print("Compiling Qwen3NextScannableBlock train step with 2-layer buffer on TPU...")
  loss_val, grads = block_train_step(params, hidden_states)
  loss_float = float(loss_val)
  print(f"  Qwen3NextScannableBlock Loss = {loss_float:.6f}")
  assert not np.isnan(loss_float), "Block Loss is NaN"
  print("Test 3 PASSED: Qwen3NextScannableBlock forward-backward with 2-layer buffer executed cleanly on TPU!\n")


if __name__ == "__main__":
  test_scanned_layer_forward_backward_tpu()
  test_mhc_layer_on_tpu()
  test_qwen3_scannable_block_on_tpu()
  print("=" * 80)
  print("ALL TPU FORWARD-BACKWARD INTEGRATION TESTS PASSED SUCCESSFULLY!")
  print("=" * 80)
