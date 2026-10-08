# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for per-layer CP strategy and halo sliding-window CP attention."""

from __future__ import annotations

import dataclasses
import functools
import os
import types
from unittest import mock

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
from flax import linen as nn  # pylint: disable=g-import-not-at-top
from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top
from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top
from maxtext.common.common_types import AttentionType  # pylint: disable=g-import-not-at-top
from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=g-import-not-at-top
from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
try:
  import tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel  # pylint: disable=g-import-not-at-top,unused-import
except ModuleNotFoundError:
  import sys  # pylint: disable=g-import-not-at-top
  from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as _tk_kernel  # pylint: disable=g-import-not-at-top
  from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as _tk_mask  # pylint: disable=g-import-not-at-top
  for _mod_name in (
      "tokamax._src",
      "tokamax._src.ops",
      "tokamax._src.ops.attention",
      "tokamax._src.ops.attention.base",
      "tokamax._src.ops.attention.pallas_triton",
      "tokamax._src.ops.experimental",
      "tokamax._src.ops.experimental.tpu",
      "tokamax._src.ops.experimental.tpu.splash_attention",
  ):
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
  sys.modules["tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel"] = _tk_kernel
  sys.modules["tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask"] = _tk_mask
from maxtext.layers import attention_op as attention_op_lib  # pylint: disable=g-import-not-at-top
from maxtext.utils import max_utils  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top


def _count_primitives(jaxpr, primitive_name):
  """Counts primitive occurrences in a jaxpr, recursing into sub-jaxprs."""
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive_name:
      count += 1
    for value in eqn.params.values():
      values = value if isinstance(value, (list, tuple)) else (value,)
      for entry in values:
        entry = getattr(entry, "jaxpr", entry)
        if hasattr(entry, "eqns"):
          count += _count_primitives(entry, primitive_name)
  return count


def _dense_sliding_window_reference(q, k, v, seg_ids, window_size):
  """Dense FP32 causal sliding-window attention reference.

  Args:
    q: [B, S, H_q, D]
    k: [B, S, H_kv, D]
    v: [B, S, H_kv, D]
    seg_ids: [B, S] or None
    window_size: int sliding window size W (attends to [i - (W - 1), i])
  """
  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  v_f32 = v.astype(jnp.float32)
  b, s, h_q, d = q_f32.shape
  h_kv = k_f32.shape[2]
  if h_q != h_kv:
    repeats = h_q // h_kv
    k_f32 = jnp.repeat(k_f32, repeats, axis=2)
    v_f32 = jnp.repeat(v_f32, repeats, axis=2)
  logits = jnp.einsum("bshd,bthd->bhst", q_f32, k_f32)
  idx = jnp.arange(s)
  causal = idx[:, None] >= idx[None, :]
  local = (idx[:, None] - idx[None, :]) < window_size
  valid = causal & local
  valid = valid[None, None, :, :]
  if seg_ids is not None:
    same_seg = seg_ids[:, :, None] == seg_ids[:, None, :]
    valid = valid & same_seg[:, None, :, :]
  logits = jnp.where(valid, logits, -1e30)
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.where(valid, weights, 0.0)
  out = jnp.einsum("bhst,bthd->bshd", weights, v_f32)
  return out.reshape(b, s, h_q, d)


def _make_attention_op_config(
    *,
    cp_strategy: str = "all_gather",
    local_cp_strategy: str = "halo",
    load_balance: bool = True,
    block_size: int = 128,
):
  """Creates a lightweight config namespace for AttentionOp testing."""
  return types.SimpleNamespace(
      context_parallel_strategy=cp_strategy,
      local_context_parallel_strategy=local_cp_strategy,
      context_parallel_load_balance=load_balance,
      context_sharding="context",
      ulysses_context_sharding="context_usp_ulysses",
      use_tokamax_splash=True,
      use_jax_splash=False,
      sa_block_q=block_size,
      sa_block_kv=block_size,
      sa_block_kv_compute=block_size,
      eval_sa_block_q=block_size,
      eval_sa_block_kv=block_size,
      eval_sa_block_kv_compute=block_size,
      sa_block_q_dkv=block_size,
      sa_block_kv_dkv=block_size,
      sa_block_kv_dkv_compute=block_size,
      sa_block_q_dq=block_size,
      sa_block_kv_dq=block_size,
      sa_use_fused_bwd_kernel=True,
      sa_q_layout="HEAD_DIM_MINOR",
      sa_k_layout="HEAD_DIM_MINOR",
      sa_v_layout="HEAD_DIM_MINOR",
      eval_sa_q_layout="HEAD_DIM_MINOR",
      eval_sa_k_layout="HEAD_DIM_MINOR",
      eval_sa_v_layout="HEAD_DIM_MINOR",
      use_splash_scheduler=False,
      sa_fuse_reciprocal=True,
      sa_use_base2_exp=True,
      local_sa_block_q=block_size,
      local_sa_block_kv=block_size,
      local_sa_block_kv_compute=block_size,
      eval_local_sa_block_q=block_size,
      eval_local_sa_block_kv=block_size,
      eval_local_sa_block_kv_compute=block_size,
      local_sa_block_q_dkv=block_size,
      local_sa_block_kv_dkv=block_size,
      local_sa_block_kv_dkv_compute=block_size,
      local_sa_block_q_dq=block_size,
      local_sa_block_kv_dq=block_size,
      local_sa_use_fused_bwd_kernel=True,
      local_sa_q_layout="HEAD_DIM_MINOR",
      local_sa_k_layout="HEAD_DIM_MINOR",
      local_sa_v_layout="HEAD_DIM_MINOR",
      eval_local_sa_q_layout="HEAD_DIM_MINOR",
      eval_local_sa_k_layout="HEAD_DIM_MINOR",
      eval_local_sa_v_layout="HEAD_DIM_MINOR",
      local_use_splash_scheduler=False,
      local_sa_fuse_reciprocal=True,
      local_sa_use_base2_exp=True,
      use_max_logit_estimate=-1,
      cost_estimate_flops_fwd=-1,
      cost_estimate_flops_bwd=-1,
      dq_reduction_steps=0,
      enable_dropout=False,
      head_dim=64,
      shard_mode="explicit",
      debug_sharding=False,
      logical_axis_rules=[
          ("activation_batch_attn", ("data", "fsdp")),
          ("activation_heads", ("tensor",)),
          ("activation_kv_heads", ("tensor",)),
          ("activation_q_length", ("context",)),
          ("activation_kv_length", ()),
          ("activation_kv", ()),
      ],
      is_eval=False,
      run_as_eval=False,
      float32_qk_product=True,
      float32_logits=True,
  )


class CpHaloAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 8:
      self.skipTest(f"Requires 8 CPU devices, found {len(jax.devices())}")
    if jax.default_backend() == "cpu":
      orig_make_splash_mha = attention_op_lib.tokamax_splash_kernel.make_splash_mha

      def _make_splash_mha_cpu(*args, **kwargs):
        if "config" in kwargs and kwargs["config"] is not None:
          kwargs["config"] = dataclasses.replace(kwargs["config"], interpret=True)
        return orig_make_splash_mha(*args, **kwargs)

      self.enterContext(
          mock.patch.object(
              attention_op_lib.tokamax_splash_kernel,
              "make_splash_mha",
              side_effect=_make_splash_mha_cpu,
          )
      )

  def _make_mesh(self, cp_size: int) -> Mesh:
    fsdp_size = 8 // cp_size
    devices = np.asarray(jax.devices()[:8]).reshape(1, fsdp_size, cp_size, 1)
    return Mesh(devices, ("data", "fsdp", "context", "tensor"))

  def test_per_layer_cp_strategy_resolution(self):
    mesh = self._make_mesh(cp_size=2)
    cfg = _make_attention_op_config(
        cp_strategy="ring",
        local_cp_strategy="halo",
        load_balance=True,
    )
    local_op = attention_op_lib.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=1024,
        num_query_heads=4,
        num_kv_heads=2,
        float32_qk_product=True,
        max_prefill_predict_length=512,
        float32_logits=True,
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=128,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(local_op.context_parallel_strategy, "halo")

    cfg_ag = _make_attention_op_config(
        cp_strategy="ring",
        local_cp_strategy="all_gather",
        load_balance=True,
    )
    local_ag_op = attention_op_lib.AttentionOp(
        config=cfg_ag,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=1024,
        num_query_heads=4,
        num_kv_heads=2,
        float32_qk_product=True,
        max_prefill_predict_length=512,
        float32_logits=True,
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=128,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(local_ag_op.context_parallel_strategy, "all_gather")

    with self.assertRaisesRegex(ValueError, "only supported for AttentionType.LOCAL_SLIDING"):
      attention_op_lib.AttentionOp(
          config=_make_attention_op_config(cp_strategy="halo"),
          mesh=mesh,
          attention_kernel="flash",
          max_target_length=1024,
          num_query_heads=4,
          num_kv_heads=2,
          float32_qk_product=True,
          max_prefill_predict_length=512,
          float32_logits=True,
          attention_type=AttentionType.GLOBAL,
          rngs=nnx.Rngs(0),
      )

  @parameterized.named_parameters(
      dict(testcase_name="cp2_lb_false_unpacked", cp_size=2, load_balance=False, packed=False),
      dict(testcase_name="cp2_lb_false_packed", cp_size=2, load_balance=False, packed=True),
      dict(testcase_name="cp2_lb_true_unpacked", cp_size=2, load_balance=True, packed=False),
      dict(testcase_name="cp2_lb_true_packed", cp_size=2, load_balance=True, packed=True),
      dict(testcase_name="cp4_lb_false_unpacked", cp_size=4, load_balance=False, packed=False),
      dict(testcase_name="cp4_lb_false_packed", cp_size=4, load_balance=False, packed=True),
      dict(testcase_name="cp4_lb_true_unpacked", cp_size=4, load_balance=True, packed=False),
      dict(testcase_name="cp4_lb_true_packed", cp_size=4, load_balance=True, packed=True),
  )
  def test_halo_attention_forward_and_grad_parity(self, cp_size: int, load_balance: bool, packed: bool):
    mesh = self._make_mesh(cp_size=cp_size)
    b = 4
    s = 1024
    h_q = 4
    h_kv = 2
    d = 64
    w = 128
    block_size = 128

    cfg = _make_attention_op_config(
        cp_strategy="ring",
        local_cp_strategy="halo",
        load_balance=load_balance,
        block_size=block_size,
    )
    op = attention_op_lib.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=s,
        num_query_heads=h_q,
        num_kv_heads=h_kv,
        float32_qk_product=True,
        max_prefill_predict_length=s,
        float32_logits=True,
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=w,
        rngs=nnx.Rngs(0),
    )

    key_rng = jax.random.PRNGKey(42 + cp_size * 10 + int(load_balance) * 2 + int(packed))
    kq, kk, kv, kdo = jax.random.split(key_rng, 4)
    q_contig = jax.random.normal(kq, (b, s, h_q, d), dtype=jnp.float32) * 0.2
    k_contig = jax.random.normal(kk, (b, s, h_kv, d), dtype=jnp.float32) * 0.2
    v_contig = jax.random.normal(kv, (b, s, h_kv, d), dtype=jnp.float32) * 0.2
    do_contig = jax.random.normal(kdo, (b, s, h_q, d), dtype=jnp.float32) * 0.2

    if packed:
      pos = jnp.arange(s)[None, :]
      seg_contig = jnp.where(pos < 300, 1, jnp.where(pos < 700, 2, 3))
      seg_contig = jnp.broadcast_to(seg_contig, (b, s)).astype(jnp.int32)
    else:
      seg_contig = None

    def ref_loss(q_, k_, v_):
      out_ = _dense_sliding_window_reference(q_, k_, v_, seg_contig, w)
      return jnp.sum(out_ * do_contig), out_

    (_, out_ref), (dq_ref, dk_ref, dv_ref) = jax.value_and_grad(
        ref_loss, argnums=(0, 1, 2), has_aux=True
    )(q_contig, k_contig, v_contig)

    if load_balance:
      q_in = max_utils.reorder_sequence(q_contig, cp_size=cp_size, seq_dim=1, to_contiguous=False)
      k_in = max_utils.reorder_sequence(k_contig, cp_size=cp_size, seq_dim=1, to_contiguous=False)
      v_in = max_utils.reorder_sequence(v_contig, cp_size=cp_size, seq_dim=1, to_contiguous=False)
      do_in = max_utils.reorder_sequence(do_contig, cp_size=cp_size, seq_dim=1, to_contiguous=False)
      seg_in = (
          max_utils.reorder_sequence(seg_contig, cp_size=cp_size, seq_dim=1, to_contiguous=False)
          if seg_contig is not None
          else None
      )
    else:
      q_in, k_in, v_in, do_in, seg_in = q_contig, k_contig, v_contig, do_contig, seg_contig

    qkv_sharding = NamedSharding(mesh, P("fsdp", "context", None, None))
    seg_sharding = NamedSharding(mesh, P("fsdp", "context"))
    q_in = jax.device_put(q_in, qkv_sharding)
    k_in = jax.device_put(k_in, qkv_sharding)
    v_in = jax.device_put(v_in, qkv_sharding)
    do_in = jax.device_put(do_in, qkv_sharding)
    if seg_in is not None:
      seg_in = jax.device_put(seg_in, seg_sharding)

    def halo_loss(q_, k_, v_):
      out_, _ = op.tpu_flash_attention(
          q_,
          k_,
          v_,
          seg_in,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_.astype(jnp.float32) * do_in.astype(jnp.float32)), out_

    with nn.logical_axis_rules(cfg.logical_axis_rules):
      (_, out_halo), (dq_halo, dk_halo, dv_halo) = jax.jit(
          jax.value_and_grad(halo_loss, argnums=(0, 1, 2), has_aux=True)
      )(q_in, k_in, v_in)

    if load_balance:
      out_halo = max_utils.reorder_sequence(out_halo, cp_size=cp_size, seq_dim=1, to_contiguous=True)
      dq_halo = max_utils.reorder_sequence(dq_halo, cp_size=cp_size, seq_dim=1, to_contiguous=True)
      dk_halo = max_utils.reorder_sequence(dk_halo, cp_size=cp_size, seq_dim=1, to_contiguous=True)
      dv_halo = max_utils.reorder_sequence(dv_halo, cp_size=cp_size, seq_dim=1, to_contiguous=True)

    np.testing.assert_allclose(np.asarray(out_halo), np.asarray(out_ref), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(dq_halo), np.asarray(dq_ref), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(dk_halo), np.asarray(dk_ref), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(dv_halo), np.asarray(dv_ref), rtol=2e-3, atol=2e-3)

  def test_halo_uses_ppermute_and_no_all_gather(self):
    mesh = self._make_mesh(cp_size=4)
    b, s, h_q, h_kv, d, w = 4, 1024, 4, 2, 64, 128
    cfg = _make_attention_op_config(
        cp_strategy="ring",
        local_cp_strategy="halo",
        load_balance=True,
        block_size=128,
    )
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      op = attention_op_lib.AttentionOp(
          config=cfg,
          mesh=mesh,
          attention_kernel="flash",
          max_target_length=s,
          num_query_heads=h_q,
          num_kv_heads=h_kv,
          float32_qk_product=True,
          max_prefill_predict_length=s,
          float32_logits=True,
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=w,
          rngs=nnx.Rngs(0),
      )
      q = jnp.zeros((b, s, h_q, d), dtype=jnp.bfloat16)
      k = jnp.zeros((b, s, h_kv, d), dtype=jnp.bfloat16)
      v = jnp.zeros((b, s, h_kv, d), dtype=jnp.bfloat16)
      seg = jnp.ones((b, s), dtype=jnp.int32)

      jaxpr = jax.make_jaxpr(
          lambda q_, k_, v_, s_: op.tpu_flash_attention(q_, k_, v_, s_, model_mode=MODEL_MODE_TRAIN)[0]
      )(q, k, v, seg).jaxpr

    self.assertEqual(_count_primitives(jaxpr, "all_gather"), 0)
    self.assertGreaterEqual(_count_primitives(jaxpr, "ppermute"), 2)

  def test_halo_with_context_checkpoint_policy(self):
    mesh = self._make_mesh(cp_size=2)
    b, s, h_q, h_kv, d, w = 4, 1024, 4, 2, 64, 128
    cfg = _make_attention_op_config(
        cp_strategy="ring",
        local_cp_strategy="halo",
        load_balance=True,
        block_size=128,
    )
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      op = attention_op_lib.AttentionOp(
          config=cfg,
          mesh=mesh,
          attention_kernel="flash",
          max_target_length=s,
          num_query_heads=h_q,
          num_kv_heads=h_kv,
          float32_qk_product=True,
          max_prefill_predict_length=s,
          float32_logits=True,
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=w,
          rngs=nnx.Rngs(0),
      )
      qkv_sharding = NamedSharding(mesh, P("fsdp", "context", None, None))
      q = jax.device_put(jnp.ones((b, s, h_q, d), dtype=jnp.float32) * 0.1, qkv_sharding)
      k = jax.device_put(jnp.ones((b, s, h_kv, d), dtype=jnp.float32) * 0.1, qkv_sharding)
      v = jax.device_put(jnp.ones((b, s, h_kv, d), dtype=jnp.float32) * 0.1, qkv_sharding)

      @functools.partial(
          jax.checkpoint,
          policy=jax.checkpoint_policies.save_only_these_names("context"),
      )
      def step_fn(q_, k_, v_):
        out_, _ = op.tpu_flash_attention(q_, k_, v_, None, model_mode=MODEL_MODE_TRAIN)
        return jnp.sum(out_)

      dq, dk, dv = jax.jit(jax.grad(step_fn, argnums=(0, 1, 2)))(q, k, v)
      self.assertEqual(dq.shape, q.shape)
      self.assertEqual(dk.shape, k.shape)
      self.assertEqual(dv.shape, v.shape)

  def test_pyconfig_validates_local_context_parallel_strategy(self):
    base_yml = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "maxtext", "configs", "base.yml"
    )
    config = pyconfig.initialize(
        [
            "train.py",
            base_yml,
            "run_name=test_cp_halo",
            "compile_topology=v6e-8",
            "compile_topology_num_slices=1",
            "ici_fsdp_parallelism=2",
            "ici_context_parallelism=4",
            "attention=flash",
            "use_tokamax_splash=True",
            "context_parallel_strategy=ring",
            "local_context_parallel_strategy=halo",
            "sliding_window_size=1024",
            "max_target_length=8192",
            "per_device_batch_size=1",
        ]
    )
    self.assertEqual(config.context_parallel_strategy, "ring")
    self.assertEqual(config.local_context_parallel_strategy, "halo")


if __name__ == "__main__":
  absltest.main()
