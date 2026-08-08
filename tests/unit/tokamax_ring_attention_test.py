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
"""Tests for MaxText's Tokamax ring attention adapter."""

# pylint: disable=protected-access

from __future__ import annotations

import functools
import types
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.attention import tokamax_ring_attention


def _count_primitives(jaxpr, primitive_name, name_param=None):
  """Counts primitive occurrences in a jaxpr, recursing into sub-jaxprs."""
  count = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive_name and (name_param is None or name_param in str(eqn.params.get("name", ""))):
      count += 1
    for value in eqn.params.values():
      values = value if isinstance(value, (list, tuple)) else (value,)
      for entry in values:
        entry = getattr(entry, "jaxpr", entry)
        if hasattr(entry, "eqns"):
          count += _count_primitives(entry, primitive_name, name_param)
  return count


class TokamaxRingAttentionTest(absltest.TestCase):

  def test_is_context_parallel_ring_requested_accepts_case_insensitive_strategy(self):
    config = types.SimpleNamespace(context_parallel_strategy="Ring")

    self.assertTrue(tokamax_ring_attention.is_context_parallel_ring_requested(config))

  def test_make_causal_mask_uses_local_tokamax_causal_mask(self):
    mask = tokamax_ring_attention._make_causal_mask((16, 16), 4)

    self.assertEqual(mask.shape, (16, 16))
    self.assertEqual(mask.q_sequence.tolist(), list(range(16)))
    self.assertIsNone(mask.kv_sequence)

  def test_make_causal_mask_sets_load_balanced_original_positions(self):
    mask = tokamax_ring_attention._make_causal_mask((16, 16), 4, load_balanced=True)

    expected_sequence = [0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9]
    self.assertEqual(mask.q_sequence.tolist(), expected_sequence)
    self.assertEqual(mask.kv_sequence.tolist(), expected_sequence)

  def test_validate_ring_mesh_axis_requires_key_value_sequence_sharding(self):
    mesh = types.SimpleNamespace(shape={"context": 4})

    with self.assertRaisesRegex(ValueError, "K/V sequence sharding"):
      tokamax_ring_attention.validate_ring_mesh_axis(
          axis_names_q=(None, None, "context", None),
          axis_names_kv=(None, None, None, None),
          sequence_dim_q=2,
          sequence_dim_kv=2,
          mesh=mesh,
          ring_axis="context",
      )

  def test_validate_head_sharding_rejects_mismatched_gqa_axes(self):
    mesh = types.SimpleNamespace(shape={"tensor": 2})

    with self.assertRaisesRegex(ValueError, "Q and KV head sharding"):
      tokamax_ring_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=8,
          num_kv_heads=4,
          head_dim_q=1,
          head_dim_kv=1,
      )

  def test_call_ring_attention_uses_ring_kernel_segment_ids(self):
    captured = {}

    class RingSegmentIds:

      def __init__(self, q, kv):
        self.q = q
        self.kv = kv

    def kernel(q, k, v, segment_ids):
      captured["segment_ids_type"] = type(segment_ids)
      captured["q_segment_shape"] = segment_ids.q.shape
      captured["kv_segment_shape"] = segment_ids.kv.shape
      return q + k + v

    query = jnp.ones((1, 2, 4, 2))
    key = jnp.ones((1, 2, 4, 2))
    value = jnp.ones((1, 2, 4, 2))
    segment_ids = jnp.ones((1, 4), dtype=jnp.int32)

    with mock.patch.object(tokamax_ring_attention.ring_attention_kernel, "SegmentIds", RingSegmentIds):
      out = tokamax_ring_attention.call_ring_attention(
          query,
          key,
          value,
          segment_ids,
          segment_ids,
          kernel,
      )

    self.assertEqual(out.shape, query.shape)
    self.assertIs(captured["segment_ids_type"], RingSegmentIds)
    self.assertEqual(captured["q_segment_shape"], (4,))
    self.assertEqual(captured["kv_segment_shape"], (4,))

  def test_with_sequence_axis_preserves_partition_spec_type(self):
    spec = jax.sharding.PartitionSpec("data", None, None, "model")

    out = tokamax_ring_attention.with_sequence_axis(spec, "context", sequence_dim=2)

    self.assertIsInstance(out, jax.sharding.PartitionSpec)
    self.assertEqual(tuple(out), ("data", None, "context", "model"))

  def test_build_splash_config_keeps_staged_dq_for_larger_kv_shards(self):
    config = types.SimpleNamespace(
        dq_reduction_steps=3,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        use_splash_scheduler=False,
        ring_scan_unroll=2,
        context_parallel_load_balance=False,
        sa_bwd_dkv_megacore=False,
    )

    splash_config = tokamax_ring_attention.build_splash_config(
        config,
        q_seq_len=1024,
        kv_seq_len=1024,
        context_parallel_size=2,
    )

    self.assertEqual(splash_config.dq_reduction_steps, 3)
    self.assertEqual(splash_config.ring_scan_unroll, 2)

  def test_build_splash_config_disables_staged_dq_for_small_kv_shards(self):
    config = types.SimpleNamespace(
        dq_reduction_steps=3,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        use_splash_scheduler=False,
        ring_scan_unroll=1,
        context_parallel_load_balance=False,
        sa_bwd_dkv_megacore=False,
    )

    splash_config = tokamax_ring_attention.build_splash_config(
        config,
        q_seq_len=512,
        kv_seq_len=512,
        context_parallel_size=2,
    )

    self.assertIsNone(splash_config.dq_reduction_steps)

  def test_residual_checkpoint_name_enables_context_remat_policy(self):
    """The named ring residuals let a save-context policy skip the forward recompute."""
    if len(jax.devices()) < 2:
      self.skipTest("Requires at least 2 devices for a ring mesh.")
    config = types.SimpleNamespace(
        context_parallel_strategy="ring",
        context_parallel_load_balance=True,
        dq_reduction_steps=-1,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        use_splash_scheduler=False,
        ring_scan_unroll=1,
        use_max_logit_estimate=-1,
    )
    devices = np.asarray(jax.devices()[:2]).reshape(1, 2)
    # The ring axis is deliberately not named "context" so that the string
    # only appears in the jaxpr through the residual checkpoint name.
    mesh = jax.sharding.Mesh(devices, ("data", "ring"))
    query = jnp.zeros((1, 1, 256, 128), jnp.bfloat16)

    def shard_with_pspec(arr, spec):
      return jax.device_put(arr, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)))

    _, ring_kernel, ring_kernel_spec = tokamax_ring_attention.make_sharded_ring_attention_kernel(
        config,
        query=query,
        key=query,
        context_parallel_size=2,
        ring_axis="ring",
        attn_logits_soft_cap=None,
        maybe_shard_with_pspec=shard_with_pspec,
    )
    qkv_spec = jax.sharding.PartitionSpec(None, None, "ring", None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(ring_kernel_spec, qkv_spec, qkv_spec, qkv_spec),
        out_specs=qkv_spec,
        check_vma=False,
    )
    def ring_attn(ring_kernel, q, k, v):
      return tokamax_ring_attention.call_ring_attention(q, k, v, None, None, ring_kernel)

    def grad_jaxpr(policy):
      remat = jax.checkpoint(lambda q, k, v: ring_attn(ring_kernel, q, k, v).astype(jnp.float32).sum(), policy=policy)
      return jax.make_jaxpr(jax.grad(remat))(query, query, query).jaxpr

    saved = grad_jaxpr(jax.checkpoint_policies.save_only_these_names("context"))
    recomputed = grad_jaxpr(jax.checkpoint_policies.nothing_saveable)
    self.assertGreaterEqual(_count_primitives(saved, "name", "context"), 2)
    # Saving the named residuals removes the ring forward recompute, and with
    # it that recompute's collective permutes, from the backward program.
    self.assertLess(_count_primitives(saved, "ppermute"), _count_primitives(recomputed, "ppermute"))


if __name__ == "__main__":
  absltest.main()
