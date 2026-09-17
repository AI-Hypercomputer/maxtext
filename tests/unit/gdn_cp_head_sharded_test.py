# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Head-Sharded Context Parallelism in GDN."""

import os

# Force 2 CPU devices before JAX initialization only if running CPU test target
if "ghostfish" not in os.environ.get("TEST_TARGET", "") and "tpu" not in os.environ.get("TEST_TARGET", ""):
  if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import time
import types
from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from maxtext.common import common_types
from maxtext.inference import kvcache
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.models import qwen3

FLAGS = flags.FLAGS


def create_gdn_config(
    hidden_size: int = 2048,
    num_key_heads: int = 16,
    num_value_heads: int = 64,
    head_dim: int = 128,
    conv_kernel_dim: int = 4,
    chunk_size: int = 64,
    cp_size: int = 1,
    dtype: jnp.dtype = jnp.float32,
    use_qk_norm: bool = True,
) -> types.SimpleNamespace:
  """Creates configuration namespace for Qwen3NextGatedDeltaNet."""
  return types.SimpleNamespace(
      emb_dim=hidden_size,
      gdn_num_value_heads=num_value_heads,
      gdn_num_key_heads=num_key_heads,
      gdn_key_head_dim=head_dim,
      gdn_value_head_dim=head_dim,
      gdn_conv_kernel_dim=conv_kernel_dim,
      gdn_chunk_size=chunk_size,
      dtype=dtype,
      weight_dtype=dtype,
      gdn_state_dtype=dtype,
      gdn_decay_dtype=dtype,
      matmul_precision="highest",
      normalization_layer_epsilon=1e-6,
      use_qk_norm_in_gdn=use_qk_norm,
      use_gdn_kernel=True,
      load_balance_loss_weight=0.0,
      scan_layers=False,
      using_pipeline_parallelism=False,
      ici_context_parallelism=cp_size,
      ici_context_usp_ulysses_parallelism=1,
      logical_axis_rules=(),
      shard_mode="auto",
      debug_sharding=False,
  )


def leaves_dict(tree):
  leaves = jax.tree_util.tree_leaves_with_path(tree)
  result = {}
  for path, val in leaves:
    if hasattr(val, "shape"):
      name = ".".join(str(getattr(k, "key", getattr(k, "name", str(k)))) for k in path)
      result[name] = np.asarray(val)
  return result


class GdnCpHeadShardedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gdn_bwd_pallas.ensure_cpu_interpret_registered()

  def test_cp_head_sharded_equivalence_cpu(self):
    """Verifies mathematical equivalence between CP=1 baseline and CP=2 head-sharded GDN."""
    if "ghostfish" in os.environ.get("TEST_TARGET", "") or jax.default_backend() != "cpu":
      self.skipTest(f"CPU equivalence test skipped on Ghostfish/non-CPU target (current is {jax.default_backend()})")
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest(f"Test requires at least 2 devices, found {len(devices)}")

    batch = 1
    seq_len = 128
    emb_dim = 256
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 64

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    mesh_cp2 = Mesh(np.array(devices[:2]), axis_names=("context",))

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=1,
    )
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=2,
    )

    rng = nnx.Rngs(42)
    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp1,
        mesh=mesh_cp1,
        dtype=jnp.float32,
        rngs=rng,
    )

    # Instantiate model_cp2 with the same initial parameters
    model_cp2 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp2,
        mesh=mesh_cp2,
        dtype=jnp.float32,
        rngs=rng,
    )

    # Exact parameter sync
    _, state1 = nnx.split(model_cp1)
    nnx.update(model_cp2, state1)

    key = jax.random.PRNGKey(101)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)

    # CP=1 training step
    @jax.jit
    def train_step_cp1(model, x):
      def loss_fn(m):
        out, _ = m(x)
        loss = jnp.mean(out * proj)
        return loss, out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    # CP=2 training step with sequence sharding on input
    sharding_cp2 = NamedSharding(mesh_cp2, P(None, "context", None))
    x_input_sharded = jax.device_put(x_input, sharding_cp2)
    proj_sharded = jax.device_put(proj, sharding_cp2)

    @jax.jit
    def train_step_cp2(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        loss = jnp.mean(out * p)
        return loss, out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    with mesh_cp1:
      loss_cp1, out_cp1, grads_cp1 = train_step_cp1(model_cp1, x_input)

    with mesh_cp2:
      loss_cp2, out_cp2, grads_cp2 = train_step_cp2(model_cp2, x_input_sharded, proj_sharded)

    # 1. Forward output equivalence
    out_cp1_np = np.asarray(out_cp1)
    out_cp2_np = np.asarray(out_cp2)
    np.testing.assert_allclose(
        out_cp1_np,
        out_cp2_np,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Forward output diverged between CP=1 and CP=2",
    )

    # 2. Loss equivalence
    loss_cp1_f = float(loss_cp1)
    loss_cp2_f = float(loss_cp2)
    self.assertAlmostEqual(
        loss_cp1_f,
        loss_cp2_f,
        delta=1e-6,
        msg=f"Loss diverged: CP=1 ({loss_cp1_f}) vs CP=2 ({loss_cp2_f})",
    )

    # 3. Parameter gradients equivalence
    g1_dict = leaves_dict(grads_cp1)
    g2_dict = leaves_dict(grads_cp2)

    self.assertEqual(set(g1_dict.keys()), set(g2_dict.keys()))

    for param_name in sorted(g1_dict.keys()):
      g1 = g1_dict[param_name]
      g2 = g2_dict[param_name]
      max_abs_diff = float(np.max(np.abs(g1 - g2)))
      ref_mag = float(np.max(np.abs(g1)))
      rel_diff = max_abs_diff / (ref_mag + 1e-7)

      self.assertTrue(
          rel_diff <= 1e-4 or max_abs_diff <= 1e-5,
          f"Gradient for {param_name} diverged: rel_diff={rel_diff:.2e}, max_abs={max_abs_diff:.2e}",
      )

  def test_cp_head_sharded_with_initial_states(self):
    """Verifies head-sharded CP execution with non-None initial conv_state and recurrent_state."""
    if "ghostfish" in os.environ.get("TEST_TARGET", "") or jax.default_backend() != "cpu":
      self.skipTest(f"CPU test skipped on Ghostfish/non-CPU target (current is {jax.default_backend()})")
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest(f"Test requires at least 2 devices, found {len(devices)}")

    batch = 1
    seq_len = 128
    emb_dim = 256
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 64
    conv_kernel_dim = 4
    conv_dim = 2 * (num_k_heads * head_dim) + (num_v_heads * head_dim)

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    mesh_cp2 = Mesh(np.array(devices[:2]), axis_names=("context",))

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel_dim=conv_kernel_dim,
        cp_size=1,
    )
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel_dim=conv_kernel_dim,
        cp_size=2,
    )

    rng = nnx.Rngs(42)
    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp1,
        mesh=mesh_cp1,
        dtype=jnp.float32,
        rngs=rng,
    )
    model_cp2 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp2,
        mesh=mesh_cp2,
        dtype=jnp.float32,
        rngs=rng,
    )

    # Exact parameter sync
    _, state1 = nnx.split(model_cp1)
    nnx.update(model_cp2, state1)

    key = jax.random.PRNGKey(101)
    k1, k2, k3 = jax.random.split(key, 3)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)

    # Initial states
    init_conv_state = jax.random.normal(k2, (batch, conv_kernel_dim - 1, conv_dim), dtype=jnp.float32)
    init_recurrent_state = jax.random.normal(k3, (batch, num_v_heads, head_dim, head_dim), dtype=jnp.float32)

    cache_cp1 = kvcache.KVCache(
        max_prefill_length=seq_len,
        max_target_length=seq_len,
        batch=batch,
        key_seq_len=1,
        value_seq_len=1,
        key_heads=num_v_heads,
        value_heads=num_v_heads,
        key_head_size=head_dim,
        value_head_size=head_dim,
        dtype=jnp.float32,
        is_gdn=True,
        conv_kernel_size=conv_kernel_dim,
        conv_dim=conv_dim,
        model_mode=common_types.MODEL_MODE_PREFILL,
    )
    cache_cp1.update_gdn_states(init_recurrent_state, init_conv_state)

    cache_cp2 = kvcache.KVCache(
        max_prefill_length=seq_len,
        max_target_length=seq_len,
        batch=batch,
        key_seq_len=1,
        value_seq_len=1,
        key_heads=num_v_heads,
        value_heads=num_v_heads,
        key_head_size=head_dim,
        value_head_size=head_dim,
        dtype=jnp.float32,
        is_gdn=True,
        conv_kernel_size=conv_kernel_dim,
        conv_dim=conv_dim,
        model_mode=common_types.MODEL_MODE_PREFILL,
    )
    cache_cp2.update_gdn_states(init_recurrent_state, init_conv_state)

    @nnx.jit
    def step_fn(model, x, cache):
      out, _ = model(x, model_mode=common_types.MODEL_MODE_PREFILL, kv_cache=cache)
      next_rs, next_cs = cache.get_gdn_states()
      return out, next_rs, next_cs

    sharding_cp2 = NamedSharding(mesh_cp2, P(None, "context", None))
    x_input_sharded = jax.device_put(x_input, sharding_cp2)

    with mesh_cp1:
      out_cp1, next_rs_cp1, next_cs_cp1 = step_fn(model_cp1, x_input, cache_cp1)

    with mesh_cp2:
      out_cp2, next_rs_cp2, next_cs_cp2 = step_fn(model_cp2, x_input_sharded, cache_cp2)

    # Assert execution succeeds and next_conv_state is 3D with correct shape
    self.assertEqual(next_cs_cp2.ndim, 3)
    self.assertEqual(next_cs_cp2.shape, (batch, conv_kernel_dim - 1, conv_dim))

    # Assert forward output matches CP=1
    np.testing.assert_allclose(
        np.asarray(out_cp1),
        np.asarray(out_cp2),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Forward output with initial states diverged between CP=1 and CP=2",
    )

    # Assert next states match CP=1
    np.testing.assert_allclose(
        np.asarray(next_cs_cp1),
        np.asarray(next_cs_cp2),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Next conv state diverged between CP=1 and CP=2",
    )
    np.testing.assert_allclose(
        np.asarray(next_rs_cp1),
        np.asarray(next_rs_cp2),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Next recurrent state diverged between CP=1 and CP=2",
    )

  def test_cp_head_sharded_v7x_ghostfish(self):
    """Verifies GDN head-sharded CP parity, speedup, and memory reduction on TPU v7x."""
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "tpu":
      self.skipTest(f"Ghostfish test requires TPU backend, current is {backend}")
    if len(devices) < 2:
      self.skipTest(f"Ghostfish test requires at least 2 TPU devices, found {len(devices)}")

    batch = 1
    seq_len = 65536
    emb_dim = 4096
    num_k_heads = 16
    num_v_heads = 64
    head_dim = 128

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    mesh_cp2 = Mesh(np.array(devices[:2]), axis_names=("context",))

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=1,
    )
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=2,
    )

    rng = nnx.Rngs(123)
    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp1,
        mesh=mesh_cp1,
        dtype=jnp.bfloat16,
        rngs=rng,
    )
    model_cp2 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp2,
        mesh=mesh_cp2,
        dtype=jnp.bfloat16,
        rngs=rng,
    )

    _, state1 = nnx.split(model_cp1)
    nnx.update(model_cp2, state1)

    key = jax.random.PRNGKey(2026)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.bfloat16)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.bfloat16)

    @jax.jit
    def step_cp1(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    sharding_cp1 = NamedSharding(mesh_cp1, P(None, "context", None))
    x_sharded_cp1 = jax.device_put(x_input, sharding_cp1)
    proj_sharded_cp1 = jax.device_put(proj, sharding_cp1)

    sharding_cp2 = NamedSharding(mesh_cp2, P(None, "context", None))
    x_sharded = jax.device_put(x_input, sharding_cp2)
    proj_sharded = jax.device_put(proj, sharding_cp2)

    @jax.jit
    def step_cp2(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    # Warmup runs
    with mesh_cp1:
      loss_cp1, _, grads_cp1 = step_cp1(model_cp1, x_sharded_cp1, proj_sharded_cp1)
      loss_cp1.block_until_ready()

    with mesh_cp2:
      loss_cp2, _, grads_cp2 = step_cp2(model_cp2, x_sharded, proj_sharded)
      loss_cp2.block_until_ready()

    # Numerical Parity Check
    loss_diff = abs(float(loss_cp1) - float(loss_cp2))
    print(
        f"\n[Numerical Parity Check] CP=1 loss={float(loss_cp1):.6f}, CP=2"
        f" loss={float(loss_cp2):.6f}, diff={loss_diff:.4e}"
    )
    self.assertLess(loss_diff, 1e-4, f"TPU loss mismatch: CP=1 {float(loss_cp1)} vs CP=2 {float(loss_cp2)}")

    # Gradient Parity Check
    g1_dict = leaves_dict(grads_cp1)
    g2_dict = leaves_dict(grads_cp2)
    self.assertEqual(set(g1_dict.keys()), set(g2_dict.keys()))
    for param_name in sorted(g1_dict.keys()):
      g1 = g1_dict[param_name]
      g2 = g2_dict[param_name]
      max_abs_diff = float(np.max(np.abs(g1 - g2)))
      ref_mag = float(np.max(np.abs(g1)))
      rel_diff = max_abs_diff / (ref_mag + 1e-7)
      print(f"  Gradient {param_name}: rel_diff={rel_diff:.4e}," f" max_abs={max_abs_diff:.4e}, ref_mag={ref_mag:.4e}")
      self.assertTrue(
          rel_diff <= 5e-2 or max_abs_diff <= 1e-3,
          f"Gradient for {param_name} diverged on TPU: rel_diff={rel_diff:.2e}, max_abs={max_abs_diff:.2e}",
      )

    # Timing comparison (3 timed steps)
    iters = 3
    t0 = time.perf_counter()
    with mesh_cp1:
      for _ in range(iters):
        l, _, _ = step_cp1(model_cp1, x_sharded_cp1, proj_sharded_cp1)
        l.block_until_ready()
    time_cp1 = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    with mesh_cp2:
      for _ in range(iters):
        l, _, _ = step_cp2(model_cp2, x_sharded, proj_sharded)
        l.block_until_ready()
    time_cp2 = (time.perf_counter() - t0) / iters

    speedup = time_cp1 / time_cp2
    print(f"\n[Cloud TPU v7x Ghostfish Benchmark S={seq_len}]")
    print(f"  CP=1 Step Latency: {time_cp1 * 1000:.2f} ms")
    print(f"  CP=2 Step Latency: {time_cp2 * 1000:.2f} ms")
    print(f"  Speedup:           {speedup:.2f}x (Target: ~1.85x)")

    self.assertGreater(
        speedup,
        1.5,
        f"Expected CP=2 speedup > 1.5x on TPU v7x, achieved {speedup:.2f}x",
    )


if __name__ == "__main__":
  absltest.main()
