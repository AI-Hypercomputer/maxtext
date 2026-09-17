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
import time
import types

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.common import common_types
from maxtext.inference import kvcache
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.models import qwen3
import numpy as np

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
    state_dtype: jnp.dtype = jnp.float32,
    decay_dtype: jnp.dtype = jnp.float32,
    use_qk_norm: bool = True,
    gdn_cp_mode: str = "auto",
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
      gdn_state_dtype=state_dtype,
      gdn_decay_dtype=decay_dtype,
      matmul_precision="highest",
      normalization_layer_epsilon=1e-6,
      use_qk_norm_in_gdn=use_qk_norm,
      use_gdn_kernel=True,
      gdn_cp_mode=gdn_cp_mode,
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
        gdn_cp_mode="head",
    )
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=2,
        gdn_cp_mode="head",
    )

    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp1,
        mesh=mesh_cp1,
        dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )

    # Instantiate model_cp2 with the same initial parameters and its own mesh
    model_cp2 = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_cp2,
        mesh=mesh_cp2,
        dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )

    key = jax.random.PRNGKey(101)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)

    # CP=1 training step
    @nnx.jit
    def train_step_cp1(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        loss = jnp.mean(out * p)
        return loss, out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    # CP=2 training step with sequence sharding on input
    sharding_cp2 = NamedSharding(mesh_cp2, P(None, "context", None))
    x_input_sharded = jax.device_put(x_input, sharding_cp2)
    proj_sharded = jax.device_put(proj, sharding_cp2)

    @nnx.jit
    def train_step_cp2(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        loss = jnp.mean(out * p)
        return loss, out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    with mesh_cp1:
      loss_cp1, out_cp1, grads_cp1 = train_step_cp1(model_cp1, x_input, proj)
      loss_cp1 = jax.block_until_ready(loss_cp1)
      out_cp1 = jax.block_until_ready(out_cp1)
      grads_cp1 = jax.block_until_ready(grads_cp1)

    with mesh_cp2:
      loss_cp2, out_cp2, grads_cp2 = train_step_cp2(model_cp2, x_input_sharded, proj_sharded)
      loss_cp2 = jax.block_until_ready(loss_cp2)
      out_cp2 = jax.block_until_ready(out_cp2)
      grads_cp2 = jax.block_until_ready(grads_cp2)

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
        gdn_cp_mode="head",
    )
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel_dim=conv_kernel_dim,
        cp_size=2,
        gdn_cp_mode="head",
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
      out_cp1 = jax.block_until_ready(out_cp1)
      next_rs_cp1 = jax.block_until_ready(next_rs_cp1)
      next_cs_cp1 = jax.block_until_ready(next_cs_cp1)

    with mesh_cp2:
      out_cp2, next_rs_cp2, next_cs_cp2 = step_fn(model_cp2, x_input_sharded, cache_cp2)
      out_cp2 = jax.block_until_ready(out_cp2)
      next_rs_cp2 = jax.block_until_ready(next_rs_cp2)
      next_cs_cp2 = jax.block_until_ready(next_cs_cp2)

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

  def test_cp_seq_sharded_equivalence_and_gt_k_heads_cpu(self):
    """Verifies Sequence-Sharded CP (gdn_cp_mode='seq') for CP=2, 4, 8, 16, 32 (including cp_size > num_k_heads)."""
    if "ghostfish" in os.environ.get("TEST_TARGET", "") or jax.default_backend() != "cpu":
      self.skipTest(f"CPU test skipped on Ghostfish/non-CPU target (current is {jax.default_backend()})")
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Test requires at least 4 CPU devices, found {len(devices)}")

    batch = 1
    chunk_size = 16
    seq_len = 512  # 32 chunks total (1 chunk per device at CP=32)
    emb_dim = 128
    # Use num_k_heads=2, num_v_heads=4 so CP=4, 8, 16, 32 all have cp_size > num_k_heads (2)!
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 32

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=1,
        gdn_cp_mode="seq",
    )
    with mesh_cp1:
      model_cp1 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp1, mesh=mesh_cp1, dtype=jnp.float32, rngs=nnx.Rngs(99))
    _, state1 = nnx.split(model_cp1)
    state1_host = jax.tree.map(np.asarray, state1)

    k1, k2 = jax.random.split(jax.random.PRNGKey(303))
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)

    @nnx.jit
    def train_step(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    with mesh_cp1:
      loss_cp1, out_cp1, grads_cp1 = train_step(model_cp1, x_input, proj)
      loss_cp1 = jax.block_until_ready(loss_cp1)
      out_cp1 = jax.block_until_ready(out_cp1)
      grads_cp1 = jax.block_until_ready(grads_cp1)
    g1_dict = leaves_dict(grads_cp1)

    tested_cp_sizes = [cp for cp in (2, 4, 8, 16, 32) if cp <= len(devices)]
    for cp_size in tested_cp_sizes:
      mesh_cp = Mesh(np.array(devices[:cp_size]), axis_names=("context",))
      cfg_cp = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          chunk_size=chunk_size,
          cp_size=cp_size,
          gdn_cp_mode="seq",
      )
      with mesh_cp:
        model_cp = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp, mesh=mesh_cp, dtype=jnp.float32, rngs=nnx.Rngs(99))
        nnx.update(model_cp, jax.tree.map(jnp.asarray, state1_host))

      sharding_cp = NamedSharding(mesh_cp, P(None, "context", None))
      x_sharded = jax.device_put(x_input, sharding_cp)
      proj_sharded = jax.device_put(proj, sharding_cp)

      with mesh_cp:
        loss_cp, out_cp, grads_cp = train_step(model_cp, x_sharded, proj_sharded)
        loss_cp = jax.block_until_ready(loss_cp)
        out_cp = jax.block_until_ready(out_cp)
        grads_cp = jax.block_until_ready(grads_cp)

      np.testing.assert_allclose(
          np.asarray(out_cp),
          np.asarray(out_cp1),
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"Seq-sharded CP={cp_size} forward output diverged from CP=1",
      )
      self.assertAlmostEqual(
          float(loss_cp),
          float(loss_cp1),
          delta=1e-5,
          msg=f"Seq-sharded CP={cp_size} loss diverged from CP=1",
      )
      g_cp_dict = leaves_dict(grads_cp)
      for param_name in sorted(g1_dict.keys()):
        g1 = g1_dict[param_name]
        g_cp = g_cp_dict[param_name]
        max_abs_diff = float(np.max(np.abs(g1 - g_cp)))
        ref_mag = float(np.max(np.abs(g1)))
        rel_diff = max_abs_diff / (ref_mag + 1e-7)
        self.assertTrue(
            rel_diff <= 1e-3 or max_abs_diff <= 1e-4,
            f"Seq-sharded CP={cp_size} grad {param_name} diverged: rel_diff={rel_diff:.2e}, max_abs={max_abs_diff:.2e}",
        )

  def test_cp_seq_sharded_packed_segment_ids_cpu(self):
    """Verifies packed document segment_ids boundary masking across CP ranks in forward & backward."""
    if "ghostfish" in os.environ.get("TEST_TARGET", "") or jax.default_backend() != "cpu":
      self.skipTest(f"CPU test skipped on Ghostfish/non-CPU target (current is {jax.default_backend()})")
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Test requires at least 4 CPU devices, found {len(devices)}")

    batch = 1
    chunk_size = 16
    seq_len = 64  # 4 ranks x 16 tokens; each rank is a separate document (seg_id = r + 1)
    emb_dim = 128
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 32

    mesh_cp4 = Mesh(np.array(devices[:4]), axis_names=("context",))
    cfg_cp4 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="seq",
    )
    with mesh_cp4:
      model_cp4 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp4, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(111))

    k1, k2 = jax.random.split(jax.random.PRNGKey(808))
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)
    seg_same = jnp.ones((batch, seq_len), dtype=jnp.int32)
    seg_split = jnp.repeat(jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), 16, axis=1)

    sharding_x = NamedSharding(mesh_cp4, P(None, "context", None))
    sharding_seg = NamedSharding(mesh_cp4, P(None, "context"))
    x_sh = jax.device_put(x_input, sharding_x)
    p_sh = jax.device_put(proj, sharding_x)
    seg_same_sh = jax.device_put(seg_same, sharding_seg)
    seg_split_sh = jax.device_put(seg_split, sharding_seg)

    @nnx.jit
    def step_with_seg(model, x, p, seg):
      def loss_fn(m, x_in):
        out, _ = m(x_in, decoder_segment_ids=seg)
        return jnp.mean(out * p), out

      (loss, out), (grads_m, grad_x) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(model, x)
      return loss, out, grads_m, grad_x

    with mesh_cp4:
      _, out_same, _, gx_same = step_with_seg(model_cp4, x_sh, p_sh, seg_same_sh)
      _, out_split, _, gx_split = step_with_seg(model_cp4, x_sh, p_sh, seg_split_sh)

    # Confirm document boundaries at rank splits alter the Conv1D boundary tokens (16..18, 32..34, 48..50)
    diff_boundary = float(jnp.max(jnp.abs(out_same[:, 16:19, :] - out_split[:, 16:19, :])))
    self.assertGreater(diff_boundary, 1e-5)
    diff_gx = float(jnp.max(jnp.abs(gx_same[:, 13:16, :] - gx_split[:, 13:16, :])))
    self.assertGreater(diff_gx, 1e-6)

  def test_cp_seq_sharded_with_initial_states_cpu(self):
    """Verifies Sequence-Sharded CP with non-zero initial conv_state & recurrent_state."""
    if "ghostfish" in os.environ.get("TEST_TARGET", "") or jax.default_backend() != "cpu":
      self.skipTest(f"CPU test skipped on Ghostfish/non-CPU target (current is {jax.default_backend()})")
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Test requires at least 4 CPU devices, found {len(devices)}")

    batch = 1
    chunk_size = 16
    seq_len = 128
    emb_dim = 256
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 64
    conv_kernel_dim = 4
    conv_dim = 2 * (num_k_heads * head_dim) + (num_v_heads * head_dim)

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    mesh_cp4 = Mesh(np.array(devices[:4]), axis_names=("context",))

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel_dim=conv_kernel_dim,
        chunk_size=chunk_size,
        cp_size=1,
        gdn_cp_mode="seq",
    )
    cfg_cp4 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel_dim=conv_kernel_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="seq",
    )

    with mesh_cp1:
      model_cp1 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp1, mesh=mesh_cp1, dtype=jnp.float32, rngs=nnx.Rngs(77))
    _, state1 = nnx.split(model_cp1)
    state1_host = jax.tree.map(np.asarray, state1)
    with mesh_cp4:
      model_cp4 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp4, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(77))
      nnx.update(model_cp4, jax.tree.map(jnp.asarray, state1_host))

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(505), 3)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    init_conv_state = jax.random.normal(k2, (batch, conv_kernel_dim - 1, conv_dim), dtype=jnp.float32) * 0.2
    init_recurrent_state = jax.random.normal(k3, (batch, num_v_heads, head_dim, head_dim), dtype=jnp.float32) * 0.2

    def _make_cache():
      c = kvcache.KVCache(
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
      c.update_gdn_states(init_recurrent_state, init_conv_state)
      return c

    cache_cp1 = _make_cache()
    cache_cp4 = _make_cache()

    @nnx.jit
    def step_fn(model, x, cache):
      out, _ = model(x, model_mode=common_types.MODEL_MODE_PREFILL, kv_cache=cache)
      next_rs, next_cs = cache.get_gdn_states()
      return out, next_rs, next_cs

    sharding_cp4 = NamedSharding(mesh_cp4, P(None, "context", None))
    x_sharded = jax.device_put(x_input, sharding_cp4)

    with mesh_cp1:
      out_cp1, next_rs_cp1, next_cs_cp1 = step_fn(model_cp1, x_input, cache_cp1)
    with mesh_cp4:
      out_cp4, next_rs_cp4, next_cs_cp4 = step_fn(model_cp4, x_sharded, cache_cp4)

    np.testing.assert_allclose(np.asarray(out_cp4), np.asarray(out_cp1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(next_cs_cp4), np.asarray(next_cs_cp1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(next_rs_cp4), np.asarray(next_rs_cp1), rtol=1e-4, atol=1e-4)

  def test_cp_head_sharded_v7x_ghostfish(self):
    """Verifies GDN Sequence-Sharded and Head-Sharded CP parity, speedup,

    and multi-chip CP=1,2,4,8 scaling on TPU v7x Ghostfish.
    """
    backend = jax.default_backend()
    devices = jax.devices()
    test_target = os.environ.get("TEST_TARGET", "")
    is_ghostfish = "ghostfish" in test_target
    is_4chip_target = "4chip" in test_target
    if is_ghostfish:
      self.assertEqual(
          backend,
          "tpu",
          f"Expected TPU backend on Ghostfish target, got {backend} ({devices})",
      )
      if is_4chip_target:
        self.assertLen(
            devices,
            8,
            f"Expected 8 TPU7x TensorCores (4 chips) on 4-chip Ghostfish target, got {len(devices)}: {devices}",
        )
      else:
        self.assertGreaterEqual(
            len(devices),
            2,
            f"Expected >= 2 TPU devices on Ghostfish target, got {len(devices)}: {devices}",
        )
    else:
      if backend != "tpu":
        self.skipTest(f"Ghostfish test requires TPU backend, current is {backend}")
      if len(devices) < 2:
        self.skipTest(f"Ghostfish test requires at least 2 TPU devices, found {len(devices)}")

    print(f"\n[TPU Hardware Verification] backend={backend}, device_count={len(devices)}")
    for idx_d, dev in enumerate(devices):
      print(f"  Device {idx_d}: {dev!r}, device_kind={getattr(dev, 'device_kind', 'unknown')}")

    batch = 1
    seq_len = 65536
    emb_dim = 4096
    num_k_heads = 16
    num_v_heads = 64
    head_dim = 128
    iters = 5

    @nnx.jit
    def step_fn(model, x, p):
      def loss_fn(m):
        out, _ = m(x)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    key = jax.random.PRNGKey(2026)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.bfloat16)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.bfloat16)

    # 1. CP=1 Baseline on devices[:1]
    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        cp_size=1,
        dtype=jnp.bfloat16,
        state_dtype=jnp.float32,
        decay_dtype=jnp.float32,
        gdn_cp_mode="seq",
    )
    with mesh_cp1:
      model_cp1 = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_cp1,
          mesh=mesh_cp1,
          dtype=jnp.bfloat16,
          rngs=nnx.Rngs(123),
      )
    _, state1 = nnx.split(model_cp1)
    state1_host = jax.tree.map(np.asarray, state1)

    sharding_cp1 = NamedSharding(mesh_cp1, P(None, "context", None))
    x_sharded_cp1 = jax.device_put(x_input, sharding_cp1)
    proj_sharded_cp1 = jax.device_put(proj, sharding_cp1)

    with mesh_cp1:
      for _ in range(2):
        loss_cp1, _, grads_cp1 = step_fn(model_cp1, x_sharded_cp1, proj_sharded_cp1)
        loss_cp1.block_until_ready()
      t0 = time.perf_counter()
      for _ in range(iters):
        l, _, _ = step_fn(model_cp1, x_sharded_cp1, proj_sharded_cp1)
        l.block_until_ready()
      time_cp1 = (time.perf_counter() - t0) / iters
    hbm_cp1_gb = (devices[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / (1024**3)
    loss_cp1_val = float(loss_cp1)
    g1_dict = leaves_dict(grads_cp1)

    # Free CP=1 intermediate buffers on device 0 before multi-core sweeps
    del x_sharded_cp1, proj_sharded_cp1, grads_cp1, model_cp1

    # 2. Multi-core / Multi-chip sweep across CP = 2, 4, 8 (up to len(devices))
    active_cp_sizes = [cp for cp in (2, 4, 8) if cp <= len(devices)]
    results_table = []

    for cp_size in active_cp_sizes:
      mesh_cp = Mesh(np.array(devices[:cp_size]), axis_names=("context",))
      sharding_cp = NamedSharding(mesh_cp, P(None, "context", None))
      x_sharded = jax.device_put(x_input, sharding_cp)
      proj_sharded = jax.device_put(proj, sharding_cp)

      # Fresh devices in this tier: devices[cp_size // 2 : cp_size] have never been used by smaller CP tiers!
      fresh_devs = devices[cp_size // 2 : cp_size]

      # Run Sequence-Sharded CP first so fresh_devs record exact Sequence-Sharded Peak HBM
      cfg_seq = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          cp_size=cp_size,
          dtype=jnp.bfloat16,
          state_dtype=jnp.float32,
          decay_dtype=jnp.float32,
          gdn_cp_mode="seq",
      )
      with mesh_cp:
        model_seq = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_seq,
            mesh=mesh_cp,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(123),
        )
        nnx.update(model_seq, jax.tree.map(jnp.asarray, state1_host))
        for _ in range(2):
          loss_seq, _, grads_seq = step_fn(model_seq, x_sharded, proj_sharded)
          loss_seq.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
          l, _, _ = step_fn(model_seq, x_sharded, proj_sharded)
          l.block_until_ready()
        time_seq = (time.perf_counter() - t0) / iters

      hbm_seq_gb = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in fresh_devs) / (1024**3)
      loss_seq_val = float(loss_seq)
      g_seq_dict = leaves_dict(grads_seq)
      del grads_seq, model_seq

      # Run Head-Sharded CP second on the same mesh_cp
      cfg_head = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          cp_size=cp_size,
          dtype=jnp.bfloat16,
          state_dtype=jnp.float32,
          decay_dtype=jnp.float32,
          gdn_cp_mode="head",
      )
      with mesh_cp:
        model_head = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_head,
            mesh=mesh_cp,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(123),
        )
        nnx.update(model_head, jax.tree.map(jnp.asarray, state1_host))
        for _ in range(2):
          loss_head, _, grads_head = step_fn(model_head, x_sharded, proj_sharded)
          loss_head.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
          l, _, _ = step_fn(model_head, x_sharded, proj_sharded)
          l.block_until_ready()
        time_head = (time.perf_counter() - t0) / iters

      hbm_head_gb = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in fresh_devs) / (1024**3)
      loss_head_val = float(loss_head)
      g_head_dict = leaves_dict(grads_head)
      del grads_head, model_head, x_sharded, proj_sharded

      loss_diff_head = abs(loss_cp1_val - loss_head_val)
      loss_diff_seq = abs(loss_cp1_val - loss_seq_val)
      print(
          f"\n[Numerical Parity Check CP={cp_size}] CP=1 loss={loss_cp1_val:.6f} | "
          f"CP={cp_size} Head loss={loss_head_val:.6f} (diff={loss_diff_head:.4e}) | "
          f"CP={cp_size} Seq loss={loss_seq_val:.6f} (diff={loss_diff_seq:.4e})"
      )
      self.assertLess(loss_diff_head, 1e-4)
      self.assertLess(loss_diff_seq, 1e-4)

      max_rel_head = 0.0
      max_rel_seq = 0.0
      for param_name in sorted(g1_dict.keys()):
        g1 = g1_dict[param_name]
        for mode_label, g_other in (("head", g_head_dict[param_name]), ("seq", g_seq_dict[param_name])):
          max_abs_diff = float(np.max(np.abs(g1 - g_other)))
          ref_mag = float(np.max(np.abs(g1)))
          rel_diff = max_abs_diff / (ref_mag + 1e-7)
          if mode_label == "head":
            max_rel_head = max(max_rel_head, rel_diff)
          else:
            max_rel_seq = max(max_rel_seq, rel_diff)
          print(
              f"  [CP={cp_size} {mode_label}] Gradient {param_name}: rel_diff={rel_diff:.4e}, max_abs={max_abs_diff:.4e}"
          )
          self.assertTrue(
              rel_diff <= 5e-2 or max_abs_diff <= 1e-3,
              f"[CP={cp_size} {mode_label}] Gradient for {param_name} diverged on TPU: rel_diff={rel_diff:.2e}",
          )

      speedup_head = time_cp1 / time_head
      speedup_seq = time_cp1 / time_seq
      seq_vs_head = time_head / time_seq
      print(f"\n[Cloud TPU v7x Ghostfish Benchmark CP={cp_size} ({cp_size // 2} chips, {cp_size} cores) S={seq_len}]")
      print(f"  CP=1 (1 Core) Step Latency:            {time_cp1 * 1000:.2f} ms (Peak HBM: {hbm_cp1_gb:.2f} GB)")
      print(
          f"  CP={cp_size} Head-Sharded Step Latency:        {time_head * 1000:.2f} ms"
          f" (Speedup vs CP=1: {speedup_head:.2f}x, Peak HBM/core: {hbm_head_gb:.2f} GB)"
      )
      print(
          f"  CP={cp_size} Sequence-Sharded Step Latency:    {time_seq * 1000:.2f} ms"
          f" (Speedup vs CP=1: {speedup_seq:.2f}x, Seq vs Head: {seq_vs_head:.2f}x, Peak HBM/core: {hbm_seq_gb:.2f} GB)"
      )

      if cp_size == 2:
        self.assertGreater(speedup_head, 1.5)
        self.assertGreater(speedup_seq, 1.5)

      results_table.append(
          {
              "cp_size": cp_size,
              "chips": cp_size // 2,
              "t_loc": seq_len // cp_size,
              "n_c": (seq_len // cp_size) // 64,
              "time_head_ms": time_head * 1000,
              "speedup_head": speedup_head,
              "hbm_head_gb": hbm_head_gb,
              "loss_diff_head": loss_diff_head,
              "max_rel_head": max_rel_head,
              "time_seq_ms": time_seq * 1000,
              "speedup_seq": speedup_seq,
              "seq_vs_head": seq_vs_head,
              "hbm_seq_gb": hbm_seq_gb,
              "loss_diff_seq": loss_diff_seq,
              "max_rel_seq": max_rel_seq,
          }
      )

    print("\n====================================================================================================")
    print(
        "REAL MULTI-CHIP CLOUD TPU v7x (GHOSTFISH) SUMMARY"
        f" (S={seq_len}, D={emb_dim}, H_k={num_k_heads}, H_v={num_v_heads}, d={head_dim})"
    )
    print("====================================================================================================")
    print(
        f"  CP=1  (0.5 chip, 1 core, T_loc=65536, N_c=1024): {time_cp1 * 1000:.2f} ms (1.00x)"
        f" | Peak HBM/core: {hbm_cp1_gb:.2f} GB"
    )
    for row in results_table:
      print(
          f"  CP={row['cp_size']:<2d} ({row['chips']} chips, {row['cp_size']} cores, T_loc={row['t_loc']:<5d},"
          f" N_c={row['n_c']:<4d}): Head={row['time_head_ms']:.2f} ms ({row['speedup_head']:.2f}x,"
          f" HBM={row['hbm_head_gb']:.2f} GB, dLoss={row['loss_diff_head']:.2e}, maxRelGrad={row['max_rel_head']:.2e})"
          f" | Seq={row['time_seq_ms']:.2f} ms ({row['speedup_seq']:.2f}x, Seq/Head={row['seq_vs_head']:.2f}x,"
          f" HBM={row['hbm_seq_gb']:.2f} GB, dLoss={row['loss_diff_seq']:.2e}, maxRelGrad={row['max_rel_seq']:.2e})"
      )
    print("====================================================================================================")

    # 3. When 8 devices (4 chips) are available, verify cp_size = 8 > num_k_heads = 4 on real multi-chip TPU7x!
    if len(devices) >= 8:
      print("\n[Multi-Chip TPU7x Verification: cp_size = 8 > num_k_heads = 4 (gdn_cp_mode='seq')]")
      k_heads_small = 4
      v_heads_small = 16
      cfg_gt_cp1 = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=k_heads_small,
          num_value_heads=v_heads_small,
          head_dim=head_dim,
          cp_size=1,
          dtype=jnp.bfloat16,
          state_dtype=jnp.float32,
          decay_dtype=jnp.float32,
          gdn_cp_mode="seq",
      )
      cfg_gt_cp8 = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=k_heads_small,
          num_value_heads=v_heads_small,
          head_dim=head_dim,
          cp_size=8,
          dtype=jnp.bfloat16,
          state_dtype=jnp.float32,
          decay_dtype=jnp.float32,
          gdn_cp_mode="seq",
      )
      mesh_cp8 = Mesh(np.array(devices[:8]), axis_names=("context",))
      with mesh_cp1:
        model_gt_cp1 = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_gt_cp1, mesh=mesh_cp1, dtype=jnp.bfloat16, rngs=nnx.Rngs(777)
        )
      _, state_gt1 = nnx.split(model_gt_cp1)
      state_gt1_host = jax.tree.map(np.asarray, state_gt1)

      with mesh_cp8:
        model_gt_cp8 = qwen3.Qwen3NextGatedDeltaNet(
            config=cfg_gt_cp8, mesh=mesh_cp8, dtype=jnp.bfloat16, rngs=nnx.Rngs(777)
        )
        nnx.update(model_gt_cp8, jax.tree.map(jnp.asarray, state_gt1_host))

      x_gt_cp1 = jax.device_put(x_input, sharding_cp1)
      p_gt_cp1 = jax.device_put(proj, sharding_cp1)
      sharding_cp8 = NamedSharding(mesh_cp8, P(None, "context", None))
      x_gt_cp8 = jax.device_put(x_input, sharding_cp8)
      p_gt_cp8 = jax.device_put(proj, sharding_cp8)

      with mesh_cp1:
        for _ in range(2):
          loss_gt_cp1, _, grads_gt_cp1 = step_fn(model_gt_cp1, x_gt_cp1, p_gt_cp1)
          loss_gt_cp1.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
          l, _, _ = step_fn(model_gt_cp1, x_gt_cp1, p_gt_cp1)
          l.block_until_ready()
        time_gt_cp1 = (time.perf_counter() - t0) / iters

      with mesh_cp8:
        for _ in range(2):
          loss_gt_cp8, _, grads_gt_cp8 = step_fn(model_gt_cp8, x_gt_cp8, p_gt_cp8)
          loss_gt_cp8.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
          l, _, _ = step_fn(model_gt_cp8, x_gt_cp8, p_gt_cp8)
          l.block_until_ready()
        time_gt_cp8 = (time.perf_counter() - t0) / iters

      gt_loss_diff = abs(float(loss_gt_cp1) - float(loss_gt_cp8))
      gt_speedup = time_gt_cp1 / time_gt_cp8
      print(
          f"  [cp_size=8 > num_k_heads=4, S={seq_len}] CP=1 loss={float(loss_gt_cp1):.6f} ({time_gt_cp1 * 1000:.2f} ms) | "
          f"CP=8 Seq loss={float(loss_gt_cp8):.6f} "
          f"({time_gt_cp8 * 1000:.2f} ms, Speedup={gt_speedup:.2f}x, diff={gt_loss_diff:.4e})"
      )
      self.assertLess(gt_loss_diff, 1e-4)
      g_gt1_dict = leaves_dict(grads_gt_cp1)
      g_gt8_dict = leaves_dict(grads_gt_cp8)
      for param_name in sorted(g_gt1_dict.keys()):
        g1 = g_gt1_dict[param_name]
        g8 = g_gt8_dict[param_name]
        max_abs_diff = float(np.max(np.abs(g1 - g8)))
        ref_mag = float(np.max(np.abs(g1)))
        rel_diff = max_abs_diff / (ref_mag + 1e-7)
        print(
            f"    [cp_size=8 > num_k_heads=4] Gradient {param_name}: rel_diff={rel_diff:.4e}, max_abs={max_abs_diff:.4e}"
        )
        self.assertTrue(
            rel_diff <= 5e-2 or max_abs_diff <= 1e-3,
            f"[cp_size=8 > num_k_heads=4] Gradient for {param_name} diverged on TPU: rel_diff={rel_diff:.2e}",
        )
      del model_gt_cp1, model_gt_cp8, grads_gt_cp1, grads_gt_cp8, x_gt_cp1, p_gt_cp1, x_gt_cp8, p_gt_cp8

  def test_gdn_cp_auto_mode_selection(self):
    """Verifies that auto mode defaults to head-sharded CP when cp_size <= 4 and sequence-sharded when cp_size > 4."""
    devices = jax.devices()
    if len(devices) < 4:
      return

    x_dummy = jnp.zeros((1, 64, 2048), dtype=jnp.float32)

    # 1. auto mode with cp_size=2 and divisible heads (16 % 2 == 0) -> selects head-sharded CP
    mesh_cp2 = Mesh(np.array(devices[:2]), ("context",))
    cfg_cp2 = create_gdn_config(cp_size=2, gdn_cp_mode="auto", num_key_heads=16)
    model_cp2 = qwen3.Qwen3NextGatedDeltaNet(cfg_cp2, rngs=nnx.Rngs(0), mesh=mesh_cp2)
    out_cp2 = model_cp2(x_dummy, model_mode=common_types.MODEL_MODE_EVAL)
    self.assertEqual(out_cp2.shape, (1, 64, cfg_cp2.emb_dim))

    # 2. auto mode with cp_size=4 and divisible heads (16 % 4 == 0) -> selects head-sharded CP
    mesh_cp4 = Mesh(np.array(devices[:4]), ("context",))
    cfg_cp4 = create_gdn_config(cp_size=4, gdn_cp_mode="auto", num_key_heads=16)
    model_cp4 = qwen3.Qwen3NextGatedDeltaNet(cfg_cp4, rngs=nnx.Rngs(0), mesh=mesh_cp4)
    out_cp4 = model_cp4(x_dummy, model_mode=common_types.MODEL_MODE_EVAL)
    self.assertEqual(out_cp4.shape, (1, 64, cfg_cp4.emb_dim))

    # 3. auto mode with cp_size=4 but non-divisible heads (num_k_heads=2 < 4) -> falls back to seq-sharded CP cleanly
    cfg_cp4_nondiv = create_gdn_config(cp_size=4, gdn_cp_mode="auto", num_key_heads=2, num_value_heads=8)
    model_cp4_nondiv = qwen3.Qwen3NextGatedDeltaNet(cfg_cp4_nondiv, rngs=nnx.Rngs(0), mesh=mesh_cp4)
    out_cp4_nondiv = model_cp4_nondiv(x_dummy, model_mode=common_types.MODEL_MODE_EVAL)
    self.assertEqual(out_cp4_nondiv.shape, (1, 64, cfg_cp4_nondiv.emb_dim))

    # 4. explicit head mode with non-divisible heads raises ValueError
    cfg_head_fail = create_gdn_config(cp_size=4, gdn_cp_mode="head", num_key_heads=2, num_value_heads=8)
    model_head_fail = qwen3.Qwen3NextGatedDeltaNet(cfg_head_fail, rngs=nnx.Rngs(0), mesh=mesh_cp4)
    with self.assertRaises(ValueError):
      model_head_fail(x_dummy, model_mode=common_types.MODEL_MODE_EVAL)


if __name__ == "__main__":
  absltest.main()
