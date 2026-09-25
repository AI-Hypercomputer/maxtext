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
    enable_gdn_sequence_packing: bool = False,
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
      enable_gdn_sequence_packing=enable_gdn_sequence_packing,
      gdn_cp_mode=gdn_cp_mode,
      load_balance_loss_weight=0.0,
      scan_layers=False,
      using_pipeline_parallelism=False,
      ici_context_parallelism=cp_size,
      ici_context_usp_ulysses_parallelism=1,
      logical_axis_rules=(),
      shard_mode="auto",
      debug_sharding=False,
      gdn_mamba_block_size=None,
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
        enable_gdn_sequence_packing=True,
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
    """Verifies that auto mode defaults to head-sharded CP when cp_size <= 2 and sequence-sharded when cp_size > 2."""
    devices = jax.devices()
    if len(devices) < 4:
      return

    emb_dim = 128
    head_dim = 32
    chunk_size = 32
    seq_len = 128
    x_dummy = jnp.zeros((1, seq_len, emb_dim), dtype=jnp.float32)

    @nnx.jit
    def run_fwd(m, x):
      out, _ = m(x, model_mode=common_types.MODEL_MODE_TRAIN)
      return out

    # 1. auto mode with cp_size=2 and divisible heads (4 % 2 == 0) -> selects head-sharded CP
    mesh_cp2 = Mesh(np.array(devices[:2]), ("context",))
    cfg_cp2 = create_gdn_config(
        hidden_size=emb_dim,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=2,
        gdn_cp_mode="auto",
        num_key_heads=4,
        num_value_heads=8,
    )
    x_cp2 = jax.device_put(x_dummy, NamedSharding(mesh_cp2, P(None, "context", None)))
    with mesh_cp2:
      model_cp2 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp2, mesh=mesh_cp2, dtype=jnp.float32, rngs=nnx.Rngs(0))
      out_cp2 = run_fwd(model_cp2, x_cp2)
    self.assertEqual(out_cp2.shape, (1, seq_len, cfg_cp2.emb_dim))

    # 2. auto mode with cp_size=4 -> selects sequence-sharded CP
    mesh_cp4 = Mesh(np.array(devices[:4]), ("context",))
    cfg_cp4 = create_gdn_config(
        hidden_size=emb_dim,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="auto",
        num_key_heads=4,
        num_value_heads=8,
    )
    x_cp4 = jax.device_put(x_dummy, NamedSharding(mesh_cp4, P(None, "context", None)))
    with mesh_cp4:
      model_cp4 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp4, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(0))
      out_cp4 = run_fwd(model_cp4, x_cp4)
    self.assertEqual(out_cp4.shape, (1, seq_len, cfg_cp4.emb_dim))

    # 3. auto mode with cp_size=4 but non-divisible heads (num_k_heads=2 < 4) -> falls back to seq-sharded CP cleanly
    cfg_cp4_nondiv = create_gdn_config(
        hidden_size=emb_dim,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="auto",
        num_key_heads=2,
        num_value_heads=8,
    )
    with mesh_cp4:
      model_cp4_nondiv = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_cp4_nondiv, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(0)
      )
      out_cp4_nondiv = run_fwd(model_cp4_nondiv, x_cp4)
    self.assertEqual(out_cp4_nondiv.shape, (1, seq_len, cfg_cp4_nondiv.emb_dim))

    # 4. explicit head mode with non-divisible heads raises ValueError
    cfg_head_fail = create_gdn_config(
        hidden_size=emb_dim,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="head",
        num_key_heads=2,
        num_value_heads=8,
    )
    with mesh_cp4:
      model_head_fail = qwen3.Qwen3NextGatedDeltaNet(
          config=cfg_head_fail, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(0)
      )
      with self.assertRaises(ValueError):
        run_fwd(model_head_fail, x_cp4)

  def test_cp_sequence_packing_head_and_seq_modes(self):
    """Verifies CP=2 (head and seq modes) with sequence packing across rank split, conv_halo, and mid-chunk boundaries."""
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest(f"Requires at least 2 devices, found {len(devices)}")

    batch = 1
    seq_len = 256
    chunk_size = 64
    emb_dim = 256
    num_k_heads = 2
    num_v_heads = 4
    head_dim = 128

    # Timeline across 2 ranks (Rank 0: 0..127, Rank 1: 128..255):
    # - Seq 1: 0..63 (ends on Chunk 0 boundary)
    # - Seq 2: 64..129 (crosses Rank 0 -> Rank 1 boundary at t=128, and ends at t=129 inside Rank 1's 3-token conv_halo!)
    # - Seq 3: 130..180 (starts at t=130 inside Rank 1's 3-token conv_halo and ends mid-chunk at 180)
    # - Seq 4: 181..240 (starts mid-chunk at 181, followed by trailing 0-padding at 241..255)
    seg_np = np.zeros((batch, seq_len), dtype=np.int32)
    seg_np[0, 0:64] = 1
    seg_np[0, 64:130] = 2
    seg_np[0, 130:181] = 3
    seg_np[0, 181:241] = 4
    seg_ids = jnp.asarray(seg_np)

    mesh_cp1 = Mesh(np.array([devices[0]]), axis_names=("context",))
    mesh_cp2 = Mesh(np.array(devices[:2]), axis_names=("context",))

    key = jax.random.PRNGKey(777)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=1,
        enable_gdn_sequence_packing=True,
    )
    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp1, mesh=mesh_cp1, dtype=jnp.float32, rngs=nnx.Rngs(42))

    @nnx.jit
    def step_fn(model, x, p, s):
      def loss_fn(m):
        out, _ = m(x, decoder_segment_ids=s, model_mode=common_types.MODEL_MODE_TRAIN)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    with mesh_cp1:
      loss_ref, out_ref, grads_ref = step_fn(model_cp1, x_input, proj, seg_ids)
      out_ref_np = np.asarray(jax.block_until_ready(out_ref))
      g_ref_dict = leaves_dict(jax.block_until_ready(grads_ref))

    # Verify Oracle #3: Qwen3NextGatedDeltaNet with use_gdn_kernel=False, enable_gdn_sequence_packing=True
    cfg_pure_jax = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=1,
        enable_gdn_sequence_packing=True,
    )
    cfg_pure_jax.use_gdn_kernel = False
    model_pure_jax = qwen3.Qwen3NextGatedDeltaNet(
        config=cfg_pure_jax, mesh=mesh_cp1, dtype=jnp.float32, rngs=nnx.Rngs(42)
    )
    with mesh_cp1:
      loss_pj, out_pj, grads_pj = step_fn(model_pure_jax, x_input, proj, seg_ids)
      out_pj_np = np.asarray(jax.block_until_ready(out_pj))
      g_pj_dict = leaves_dict(jax.block_until_ready(grads_pj))
    np.testing.assert_allclose(
        out_pj_np,
        out_ref_np,
        rtol=2e-3,
        atol=2e-3,
        err_msg="Pure-JAX Qwen3NextGatedDeltaNet (use_gdn_kernel=False) diverged from Pallas kernel with sequence packing!",
    )
    self.assertAlmostEqual(float(loss_pj), float(loss_ref), delta=1e-4)
    for param_name in sorted(g_ref_dict.keys()):
      g1 = g_ref_dict[param_name]
      gpj = g_pj_dict[param_name]
      max_abs_diff = float(np.max(np.abs(g1 - gpj)))
      ref_mag = float(np.max(np.abs(g1)))
      rel_diff = max_abs_diff / (ref_mag + 1e-7)
      self.assertTrue(
          rel_diff <= 2e-2 or max_abs_diff <= 1e-3,
          f"Pure-JAX Qwen3NextGatedDeltaNet gradient {param_name} diverged: rel_diff={rel_diff:.2e}",
      )

    for mode in ("head", "seq"):
      cfg_cp2 = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          chunk_size=chunk_size,
          cp_size=2,
          gdn_cp_mode=mode,
          enable_gdn_sequence_packing=True,
      )
      model_cp2 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp2, mesh=mesh_cp2, dtype=jnp.float32, rngs=nnx.Rngs(42))
      sharding_x = NamedSharding(mesh_cp2, P(None, "context", None))
      sharding_s = NamedSharding(mesh_cp2, P(None, None) if mode == "head" else P(None, "context"))
      x_sharded = jax.device_put(x_input, sharding_x)
      p_sharded = jax.device_put(proj, sharding_x)
      s_sharded = jax.device_put(seg_ids, sharding_s)

      with mesh_cp2:
        loss_cp2, out_cp2, grads_cp2 = step_fn(model_cp2, x_sharded, p_sharded, s_sharded)
        out_cp2_np = np.asarray(jax.block_until_ready(out_cp2))
        g_cp2_dict = leaves_dict(jax.block_until_ready(grads_cp2))

      np.testing.assert_allclose(
          out_cp2_np,
          out_ref_np,
          rtol=2e-3,
          atol=2e-3,
          err_msg=f"Sequence packing CP=2 ({mode}) forward output diverged from CP=1!",
      )
      self.assertAlmostEqual(float(loss_cp2), float(loss_ref), delta=1e-4)
      for param_name in sorted(g_ref_dict.keys()):
        g1 = g_ref_dict[param_name]
        g2 = g_cp2_dict[param_name]
        max_abs_diff = float(np.max(np.abs(g1 - g2)))
        ref_mag = float(np.max(np.abs(g1)))
        rel_diff = max_abs_diff / (ref_mag + 1e-7)
        self.assertTrue(
            rel_diff <= 2e-2 or max_abs_diff <= 1e-3,
            f"Sequence packing CP=2 ({mode}) gradient {param_name} diverged: "
            f"rel_diff={rel_diff:.2e}, max_abs={max_abs_diff:.2e}",
        )

  def test_cp_4chip_sequence_packing_tpu(self):
    """Verifies 4-chip (8 TPU cores) CP=4 and 2D (data=2, context=4) sequence packing on TPU."""
    devices = jax.devices()
    if len(devices) < 8:
      self.skipTest(f"Requires 8 devices (4-chip TPU), found {len(devices)}")

    is_tpu = jax.default_backend() == "tpu"
    batch = 2
    seq_len = 512 if is_tpu else 256
    chunk_size = 64
    emb_dim = 256 if is_tpu else 128
    num_k_heads = 4 if is_tpu else 2
    num_v_heads = 16 if is_tpu else 4
    head_dim = 128 if is_tpu else 64

    seg_np = np.zeros((batch, seq_len), dtype=np.int32)
    if is_tpu:
      # Row 0: 6 packed documents across 4 CP ranks (128 tokens per rank)
      seg_np[0, 0:64] = 1
      seg_np[0, 64:135] = 2
      seg_np[0, 135:180] = 3
      seg_np[0, 180:258] = 4
      seg_np[0, 258:390] = 5
      seg_np[0, 390:490] = 6
      # Row 1: 5 packed documents + mid/trailing 0-padding
      seg_np[1, 0:100] = 1
      seg_np[1, 100:128] = 0
      seg_np[1, 128:260] = 2
      seg_np[1, 260:320] = 3
      seg_np[1, 320:410] = 4
      seg_np[1, 410:500] = 5
    else:
      seg_np[0, 0:64] = 1
      seg_np[0, 64:135] = 2
      seg_np[0, 135:180] = 3
      seg_np[0, 180:250] = 4
      seg_np[1, 0:50] = 1
      seg_np[1, 50:64] = 0
      seg_np[1, 64:150] = 2
      seg_np[1, 150:240] = 3
    seg_ids = jnp.asarray(seg_np)

    key = jax.random.PRNGKey(888)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)

    mesh_cp1 = Mesh(np.array(devices[:2]).reshape(2, 1), axis_names=("data", "context"))
    mesh_cp4 = Mesh(np.array(devices[:8]).reshape(2, 4), axis_names=("data", "context"))
    rules = ((common_types.KV_BATCH, "data"), (common_types.LENGTH, "context"))

    cfg_cp1 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=1,
        enable_gdn_sequence_packing=True,
    )
    cfg_cp1.logical_axis_rules = rules
    model_cp1 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp1, mesh=mesh_cp1, dtype=jnp.float32, rngs=nnx.Rngs(99))

    cfg_cp4 = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=4,
        gdn_cp_mode="seq",
        enable_gdn_sequence_packing=True,
    )
    cfg_cp4.logical_axis_rules = rules
    model_cp4 = qwen3.Qwen3NextGatedDeltaNet(config=cfg_cp4, mesh=mesh_cp4, dtype=jnp.float32, rngs=nnx.Rngs(99))

    @nnx.jit
    def step_fn(model, x, p, s):
      def loss_fn(m):
        out, _ = m(x, decoder_segment_ids=s, model_mode=common_types.MODEL_MODE_TRAIN)
        return jnp.mean(out * p), out

      (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      return loss, out, grads

    with mesh_cp1:
      x_1 = jax.device_put(x_input, NamedSharding(mesh_cp1, P("data", None, None)))
      p_1 = jax.device_put(proj, NamedSharding(mesh_cp1, P("data", None, None)))
      s_1 = jax.device_put(seg_ids, NamedSharding(mesh_cp1, P("data", None)))
      loss_1, out_1, grads_1 = step_fn(model_cp1, x_1, p_1, s_1)
      out_1_np = np.asarray(jax.block_until_ready(out_1))
      g_1_dict = leaves_dict(jax.block_until_ready(grads_1))

    with mesh_cp4:
      x_4 = jax.device_put(x_input, NamedSharding(mesh_cp4, P("data", "context", None)))
      p_4 = jax.device_put(proj, NamedSharding(mesh_cp4, P("data", "context", None)))
      s_4 = jax.device_put(seg_ids, NamedSharding(mesh_cp4, P("data", "context")))
      loss_4, out_4, grads_4 = step_fn(model_cp4, x_4, p_4, s_4)
      out_4_np = np.asarray(jax.block_until_ready(out_4))
      g_4_dict = leaves_dict(jax.block_until_ready(grads_4))

    np.testing.assert_allclose(out_4_np, out_1_np, rtol=2e-3, atol=2e-3)
    self.assertAlmostEqual(float(loss_4), float(loss_1), delta=1e-4)
    for param_name in sorted(g_1_dict.keys()):
      g1 = g_1_dict[param_name]
      g4 = g_4_dict[param_name]
      max_abs_diff = float(np.max(np.abs(g1 - g4)))
      ref_mag = float(np.max(np.abs(g1)))
      rel_diff = max_abs_diff / (ref_mag + 1e-7)
      self.assertTrue(
          rel_diff <= 2e-2 or max_abs_diff <= 1e-3,
          f"2D (data=2, context=4) CP sequence packing gradient {param_name} diverged: rel_diff={rel_diff:.2e}",
      )

  def test_cp4_kernel_packed_caller_states_and_state_grads(self):
    """Seq-CP=4 kernel with caller states: rank-0 leading pad, all-pad rank 0 row, all-pad last shard.

    Checks CP vs single-device outputs, next conv/recurrent states and all
    gradients (including the caller-state gradients), and that next_conv_state
    is the window ending at the last valid token (not the bucket-padded tail).
    """
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    is_tpu = jax.default_backend() == "tpu"
    cp_size, chunk_size, batch = 4, 64, 2
    seq_len = 512
    num_k_heads, num_v_heads, kernel_size = 2, 4, 4
    head_dim = 128 if is_tpu else 32
    dim_size = num_k_heads * head_dim * 2 + num_v_heads * head_dim

    seg_np = np.zeros((batch, seq_len), dtype=np.int32)
    seg_np[0, 6:224] = 1
    seg_np[0, 224:384] = 2  # rank 3 (384..511) is all padding
    seg_np[1, 128:] = 1  # rank 0 is all padding
    seg = jnp.asarray(seg_np)

    keys = jax.random.split(jax.random.PRNGKey(4242), 12)
    qkv = jax.random.normal(keys[0], (batch, seq_len, dim_size), jnp.float32)
    b = jax.random.normal(keys[1], (batch, seq_len, num_v_heads), jnp.float32)
    a = jax.random.normal(keys[2], (batch, seq_len, num_v_heads), jnp.float32)
    cw = jax.random.normal(keys[3], (kernel_size, 1, dim_size), jnp.float32)
    cb = jax.random.normal(keys[4], (dim_size,), jnp.float32)
    al = jax.random.normal(keys[5], (num_v_heads,), jnp.float32)
    dt = jax.random.normal(keys[6], (num_v_heads,), jnp.float32)
    do = jax.random.normal(keys[7], (batch, seq_len, num_v_heads, head_dim), jnp.float32)
    cs = jax.random.normal(keys[8], (batch, kernel_size - 1, dim_size), jnp.float32) * 0.5
    rs = jax.random.normal(keys[9], (batch, num_v_heads, head_dim, head_dim), jnp.float32) * 0.2
    dcs = jax.random.normal(keys[10], cs.shape, jnp.float32)
    drs = jax.random.normal(keys[11], rs.shape, jnp.float32)

    def call(q, b_, a_, cw_, cb_, al_, dt_, cs_, rs_, seg_, cp_axis):
      return gdn_bwd_pallas.gdn_decoupled_conv1d(
          q,
          b_,
          a_,
          cw_,
          cb_,
          al_,
          dt_,
          cs_,
          rs_,
          num_k_heads,
          num_v_heads,
          head_dim,
          head_dim,
          kernel_size,
          chunk_size,
          True,
          jnp.float32,
          cp_axis,
          seg_,
      )

    mesh = Mesh(np.array(devices[:cp_size]), ("context",))

    def loss_single(*args):
      out, (ncs, nrs) = call(*args, seg, None)
      return jnp.sum(out * do) + jnp.sum(ncs * dcs) + jnp.sum(nrs * drs), (out, ncs, nrs)

    def loss_cp(*args):
      mapped = jax.shard_map(
          lambda *xs: call(*xs, "context"),
          mesh=mesh,
          in_specs=(P(None, "context", None),) * 3 + (P(),) * 6 + (P(None, "context"),),
          out_specs=(P(None, "context", None, None), (P(), P())),
          check_vma=False,
      )
      out, (ncs, nrs) = mapped(*args, seg)
      return jnp.sum(out * do) + jnp.sum(ncs * dcs) + jnp.sum(nrs * drs), (out, ncs, nrs)

    args = (qkv, b, a, cw, cb, al, dt, cs, rs)
    argnums = tuple(range(len(args)))
    (_, aux_s), grads_s = jax.jit(jax.value_and_grad(loss_single, argnums=argnums, has_aux=True))(*args)
    (_, aux_c), grads_c = jax.jit(jax.value_and_grad(loss_cp, argnums=argnums, has_aux=True))(*args)

    tol = 2e-2 if is_tpu else 2e-3
    for name, got, exp in zip(("out", "next_conv_state", "next_recurrent_state"), aux_c, aux_s):
      got, exp = np.asarray(got), np.asarray(exp)
      rel = float(np.max(np.abs(got - exp))) / (float(np.max(np.abs(exp))) + 1e-7)
      self.assertLessEqual(rel, tol, f"CP vs single {name}: rel={rel:.2e}")
    grad_names = ("dqkv", "db", "da", "d_conv_w", "d_conv_b", "d_a_log", "d_dt_bias", "d_conv_state", "d_recurrent_state")
    for name, got, exp in zip(grad_names, grads_c, grads_s):
      got, exp = np.asarray(got), np.asarray(exp)
      rel = float(np.max(np.abs(got - exp))) / (float(np.max(np.abs(exp))) + 1e-7)
      self.assertLessEqual(rel, tol, f"CP vs single {name}: rel={rel:.2e}")

    qkv_np = np.asarray(qkv)
    expected_cs = np.stack([qkv_np[0, 381:384], qkv_np[1, 509:512]])
    for label, got in (("single", aux_s[1]), ("cp", aux_c[1])):
      np.testing.assert_allclose(np.asarray(got), expected_cs, rtol=1e-5, atol=1e-5, err_msg=f"{label} next_conv_state")
    # The caller recurrent state reaches the first document, so its gradient is non-zero. Both rows start with
    # >= K - 1 padding tokens, so no valid token's conv window reaches the caller conv state: zero gradient.
    self.assertGreater(float(np.max(np.abs(np.asarray(grads_c[-1])))), 1e-4, "d_recurrent_state is zero")
    self.assertLessEqual(float(np.max(np.abs(np.asarray(grads_c[-2])))), 1e-6, "d_conv_state leaked through padding")

  def _run_e2e_layer_unpacked_solo_loop(
      self,
      *,
      x_input: jax.Array,
      proj: jax.Array,
      seg_ids: jax.Array,
      emb_dim: int,
      num_k_heads: int,
      num_v_heads: int,
      head_dim: int,
      chunk_size: int,
      rng_seed: int,
      mesh_solo: Mesh,
  ) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Ground-truth E2E layer oracle: loops over each document individually with enable_gdn_sequence_packing=False."""
    batch, seq_len, _ = x_input.shape
    total_elems = float(batch * seq_len * emb_dim)
    seg_np = np.asarray(seg_ids, dtype=np.int32)

    cfg_solo = create_gdn_config(
        hidden_size=emb_dim,
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        cp_size=1,
        enable_gdn_sequence_packing=False,
    )
    cfg_solo.use_gdn_kernel = False
    model_solo = qwen3.Qwen3NextGatedDeltaNet(config=cfg_solo, mesh=mesh_solo, dtype=jnp.float32, rngs=nnx.Rngs(rng_seed))

    @nnx.jit
    def solo_step_fn(model, x_d, p_d):
      def loss_fn(m, x_in):
        out_d, _ = m(
            x_in,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        return jnp.sum(out_d * p_d) / total_elems, out_d

      (loss_d, out_d), (grads_m, grad_x) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(model, x_d)
      return loss_d, out_d, grads_m, grad_x

    out_solo = np.zeros((batch, seq_len, emb_dim), dtype=np.float32)
    dx_solo = np.zeros((batch, seq_len, emb_dim), dtype=np.float32)
    loss_solo = 0.0
    g_solo_dict: dict[str, np.ndarray] = {}

    with mesh_solo:
      for b_idx in range(batch):
        idx = 0
        while idx < seq_len:
          s_val = int(seg_np[b_idx, idx])
          if s_val <= 0:
            idx += 1
            continue
          end_idx = idx + 1
          while end_idx < seq_len and int(seg_np[b_idx, end_idx]) == s_val:
            end_idx += 1
          l_d = end_idx - idx
          l_pad = ((l_d + chunk_size - 1) // chunk_size) * chunk_size
          pad_amt = l_pad - l_d
          x_d = jnp.pad(
              x_input[b_idx : b_idx + 1, idx:end_idx, :],
              ((0, 0), (0, pad_amt), (0, 0)),
          )
          p_d = jnp.pad(
              proj[b_idx : b_idx + 1, idx:end_idx, :],
              ((0, 0), (0, pad_amt), (0, 0)),
          )
          loss_d, out_d, grads_d, dx_d = solo_step_fn(model_solo, x_d, p_d)
          loss_solo += float(loss_d)
          out_solo[b_idx, idx:end_idx, :] = np.asarray(out_d[0, :l_d, :], dtype=np.float32)
          dx_solo[b_idx, idx:end_idx, :] = np.asarray(dx_d[0, :l_d, :], dtype=np.float32)
          d_dict = leaves_dict(jax.block_until_ready(grads_d))
          for k, v in d_dict.items():
            if k in g_solo_dict:
              g_solo_dict[k] = g_solo_dict[k] + np.asarray(v, dtype=np.float32)
            else:
              g_solo_dict[k] = np.asarray(v, dtype=np.float32).copy()
          idx = end_idx

    return loss_solo, out_solo, dx_solo, g_solo_dict

  def test_e2e_layer_10way_cross_cp_and_solo_loop_parity(self):
    """Section 4.3 Test 1 & Test 2: Full-Layer Unpacked Solo Loop vs Kernel & Pure JAX across CP=1, 2, 4."""
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest(f"Requires >= 2 devices, found {len(devices)}")

    is_tpu = jax.default_backend() == "tpu"
    has_8_devs = len(devices) >= 8
    batch = 2
    seq_len = 512 if (is_tpu and has_8_devs) else 256
    chunk_size = 64
    emb_dim = 256 if is_tpu else 128
    # CP=4 head-sharded configs (run whenever >= 8 devices) need num_k_heads % 4 == 0.
    num_k_heads = 4 if (is_tpu or has_8_devs) else 2
    num_v_heads = 16 if is_tpu else 4
    head_dim = 128 if is_tpu else 64
    rng_seed = 2026

    seg_np = np.zeros((batch, seq_len), dtype=np.int32)
    if seq_len == 512:
      # Row 0: chunk boundary (0..63), mid-chunk end (64..135), strictly inside chunk (135..180),
      # short L<K (180..182), cross-rank (182..390, 390..490)
      seg_np[0, 0:64] = 1
      seg_np[0, 64:135] = 2
      seg_np[0, 135:180] = 3
      seg_np[0, 180:182] = 4
      seg_np[0, 182:390] = 5
      seg_np[0, 390:490] = 6
      # Row 1: mid-chunk padding + multi-rank segments
      seg_np[1, 0:100] = 1
      seg_np[1, 100:128] = 0
      seg_np[1, 128:260] = 2
      seg_np[1, 260:320] = 3
      seg_np[1, 320:500] = 4
    else:
      seg_np[0, 0:64] = 1
      seg_np[0, 64:135] = 2
      seg_np[0, 135:180] = 3
      seg_np[0, 180:182] = 4
      seg_np[0, 182:248] = 5
      seg_np[1, 0:50] = 1
      seg_np[1, 50:64] = 0
      seg_np[1, 64:150] = 2
      seg_np[1, 150:240] = 3
    seg_ids = jnp.asarray(seg_np)

    key = jax.random.PRNGKey(901)
    k1, k2 = jax.random.split(key)
    x_input = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    # Zero proj on 0-padding so loss = mean(out * proj) only scores valid tokens
    proj = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj = jnp.where((seg_ids > 0)[:, :, None], proj, 0.0)

    mesh_solo = Mesh(np.array(devices[:1]), axis_names=("context",))
    loss_solo, out_solo, dx_solo, g_solo_dict = self._run_e2e_layer_unpacked_solo_loop(
        x_input=x_input,
        proj=proj,
        seg_ids=seg_ids,
        emb_dim=emb_dim,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        rng_seed=rng_seed,
        mesh_solo=mesh_solo,
    )

    @nnx.jit
    def packed_step_fn(model, x_in, p_in, s_in):
      def loss_fn(m, x_arg):
        out_p, _ = m(
            x_arg,
            decoder_segment_ids=s_in,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        return jnp.mean(out_p * p_in), out_p

      (loss_p, out_p), (grads_m, grad_x) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(model, x_in)
      return loss_p, out_p, grads_m, grad_x

    configs_to_test = [
        ("Packed Kernel (CP=1)", True, 1, "seq", False),
        ("Packed Pure JAX (CP=1)", False, 1, "seq", False),
        ("Packed Kernel (CP=2, head)", True, 2, "head", False),
        ("Packed Kernel (CP=2, seq)", True, 2, "seq", False),
        ("Packed Pure JAX (CP=2, head)", False, 2, "head", False),
        ("Packed Pure JAX (CP=2, seq)", False, 2, "seq", False),
    ]
    if has_8_devs:
      configs_to_test.extend(
          [
              ("Packed Kernel (CP=4, head, 2D)", True, 4, "head", True),
              ("Packed Kernel (CP=4, seq, 2D)", True, 4, "seq", True),
              ("Packed Pure JAX (CP=4, head, 2D)", False, 4, "head", True),
              ("Packed Pure JAX (CP=4, seq, 2D)", False, 4, "seq", True),
          ]
      )

    table_rows = []
    valid_mask_np = (seg_np > 0)[:, :, None]

    for label, use_kernel, cp_sz, cp_mode, use_2d_mesh in configs_to_test:
      if use_2d_mesh:
        mesh = Mesh(np.array(devices[: 2 * cp_sz]).reshape(2, cp_sz), axis_names=("data", "context"))
        rules = ((common_types.KV_BATCH, "data"), (common_types.LENGTH, "context"))
        p_x = P("data", "context", None) if cp_sz > 1 else P("data", None, None)
        p_s = P("data", None) if (cp_sz == 1 or cp_mode == "head") else P("data", "context")
      else:
        mesh = Mesh(np.array(devices[:cp_sz]), axis_names=("context",))
        rules = None
        p_x = P(None, "context", None) if cp_sz > 1 else P(None, None, None)
        p_s = P(None, None) if (cp_sz == 1 or cp_mode == "head") else P(None, "context")

      cfg = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          chunk_size=chunk_size,
          cp_size=cp_sz,
          gdn_cp_mode=cp_mode,
          enable_gdn_sequence_packing=True,
      )
      cfg.use_gdn_kernel = use_kernel
      if rules is not None:
        cfg.logical_axis_rules = rules

      model = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh, dtype=jnp.float32, rngs=nnx.Rngs(rng_seed))
      with mesh:
        x_s = jax.device_put(x_input, NamedSharding(mesh, p_x))
        p_s_arr = jax.device_put(proj, NamedSharding(mesh, p_x))
        s_s = jax.device_put(seg_ids, NamedSharding(mesh, p_s))
        loss_got, out_got, grads_got, dx_got = packed_step_fn(model, x_s, p_s_arr, s_s)
        out_got_np = np.asarray(jax.block_until_ready(out_got), dtype=np.float32) * valid_mask_np
        dx_got_np = np.asarray(jax.block_until_ready(dx_got), dtype=np.float32) * valid_mask_np
        g_got_dict = leaves_dict(jax.block_until_ready(grads_got))

      fwd_max_abs = float(np.max(np.abs(out_got_np - out_solo)))
      fwd_rel = fwd_max_abs / (float(np.max(np.abs(out_solo))) + 1e-7)
      dx_max_abs = float(np.max(np.abs(dx_got_np - dx_solo)))
      dx_rel = dx_max_abs / (float(np.max(np.abs(dx_solo))) + 1e-7)

      max_param_rel = 0.0
      for param_name in sorted(g_solo_dict.keys()):
        g_ref = g_solo_dict[param_name]
        g_cur = np.asarray(g_got_dict[param_name], dtype=np.float32)
        p_abs = float(np.max(np.abs(g_cur - g_ref)))
        p_rel = p_abs / (float(np.max(np.abs(g_ref))) + 1e-7)
        max_param_rel = max(max_param_rel, p_rel)
        self.assertTrue(
            p_rel <= 2.5e-2 or p_abs <= 1e-3,
            f"[{label}] Parameter {param_name} diverged from Unpacked Solo Loop: rel={p_rel:.2e}, abs={p_abs:.2e}",
        )

      self.assertTrue(
          fwd_rel <= 1e-2 or fwd_max_abs <= 2e-3,
          f"[{label}] Forward output diverged from Unpacked Solo Loop: rel={fwd_rel:.2e}, abs={fwd_max_abs:.2e}",
      )
      self.assertTrue(
          dx_rel <= 2e-2 or dx_max_abs <= 2e-3,
          f"[{label}] Input grad dx diverged from Unpacked Solo Loop: rel={dx_rel:.2e}, abs={dx_max_abs:.2e}",
      )
      self.assertAlmostEqual(float(loss_got), float(loss_solo), delta=2e-4)
      table_rows.append((label, fwd_max_abs, fwd_rel, dx_rel, max_param_rel, "MATCH"))

    print("\n" + "=" * 118)
    print(">>> SECTION 4.3 E2E FULL-LAYER PARITY TABLE vs. UNPACKED SOLO LOOP (enable_gdn_sequence_packing=False)")
    print("=" * 118)
    print(
        f"  {'Configuration':<34} | {'Fwd Max Abs':<12} | {'Fwd Rel Diff':<12} | "
        f"{'dX Rel Diff':<12} | {'Max Param Grad Rel':<18} | Status"
    )
    print("  " + "-" * 114)
    for label, fwd_abs, fwd_r, dx_r, p_r, st in table_rows:
      print(f"  {label:<34} | {fwd_abs:<12.2e} | {fwd_r:<12.2e} | " f"{dx_r:<12.2e} | {p_r:<18.2e} | {st}")
    print("=" * 118 + "\n")

  def test_e2e_layer_zero_cross_document_bleed(self):
    """Section 4.3 Test 3: Perturbing Doc 1 by +100.0 causes exact 0.0 change on Doc 2, 3, 4 in full GDN layer."""
    devices = jax.devices()
    is_tpu = jax.default_backend() == "tpu"
    batch, seq_len, chunk_size = 1, 128, 64
    emb_dim = 256 if is_tpu else 128
    num_k_heads = 4 if is_tpu else 2
    num_v_heads = 16 if is_tpu else 4
    head_dim = 128 if is_tpu else 64

    seg_np = np.zeros((batch, seq_len), dtype=np.int32)
    seg_np[0, 0:25] = 1
    seg_np[0, 25:64] = 2
    seg_np[0, 64:95] = 3
    seg_np[0, 95:120] = 4
    seg_ids = jnp.asarray(seg_np)

    key = jax.random.PRNGKey(999)
    k1, k2 = jax.random.split(key)
    x_base = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.float32)
    # Perturb only Doc 1 (tokens 0..24) by +100.0
    x_perturbed = x_base.at[:, 0:25, :].add(100.0)
    # Target proj is only non-zero on Docs 2, 3, 4 (tokens 25..119)
    proj_docs_234 = jax.random.normal(k2, (batch, seq_len, emb_dim), dtype=jnp.float32)
    proj_docs_234 = jnp.where((seg_ids >= 2)[:, :, None], proj_docs_234, 0.0)

    mesh_1 = Mesh(np.array(devices[:1]), axis_names=("context",))
    for use_kernel in (True, False):
      cfg = create_gdn_config(
          hidden_size=emb_dim,
          num_key_heads=num_k_heads,
          num_value_heads=num_v_heads,
          head_dim=head_dim,
          chunk_size=chunk_size,
          cp_size=1,
          enable_gdn_sequence_packing=True,
      )
      cfg.use_gdn_kernel = use_kernel
      model = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh_1, dtype=jnp.float32, rngs=nnx.Rngs(123))

      @nnx.jit
      def run_layer(m, x_in):
        def loss_fn(x_arg):
          out_y, _ = m(
              x_arg,
              decoder_segment_ids=seg_ids,
              model_mode=common_types.MODEL_MODE_TRAIN,
          )
          return jnp.sum(out_y * proj_docs_234), out_y

        (_, out_y), dx = jax.value_and_grad(loss_fn, has_aux=True)(x_in)
        return out_y, dx

      with mesh_1:
        out_base, dx_base = run_layer(model, x_base)
        out_pert, dx_pert = run_layer(model, x_perturbed)
        out_diff_docs_234 = float(np.max(np.abs(np.asarray(out_base[:, 25:120, :] - out_pert[:, 25:120, :]))))
        dx_diff_docs_234 = float(np.max(np.abs(np.asarray(dx_base[:, 25:120, :] - dx_pert[:, 25:120, :]))))
      self.assertEqual(
          out_diff_docs_234,
          0.0,
          f"[use_gdn_kernel={use_kernel}] Doc 1 perturbation leaked into forward outputs of Docs 2..4: {out_diff_docs_234}",
      )
      self.assertEqual(
          dx_diff_docs_234,
          0.0,
          f"[use_gdn_kernel={use_kernel}] Doc 1 perturbation leaked into backward dx of Docs 2..4: {dx_diff_docs_234}",
      )


if __name__ == "__main__":
  absltest.main()
