# Copyright 2023–2026 Google LLC
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

"""CPU unit tests for the Muse Spark expert-parallel dispatch and fused post-expert norm kernels.

The tests pass `is_tpu=False`, so they cover the routing logic and the pure-JAX fallbacks of the kernels, not the
Pallas TPU kernels themselves.
"""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import muse_spark_post_expert_norm
from maxtext.kernels.ragged import ragged_sort_tc

pytestmark = [pytest.mark.cpu_only]


def _input_scaled_rms_norm(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """`RMSNorm((1 + scale) * x)` computed in float32 with the scaled input rounded to `x.dtype` first."""
  scale_f32 = (jnp.asarray(1.0, dtype=scale.dtype) + scale).astype(jnp.float32)
  scaled = (x.astype(jnp.float32) * scale_f32).astype(x.dtype).astype(jnp.float32)
  mean2 = jnp.mean(jnp.square(scaled), axis=-1, keepdims=True)
  return (scaled * jax.lax.rsqrt(mean2 + epsilon)).astype(x.dtype)


def _reference_local_moe_shard(
    ep_x: jax.Array,
    ep_expert_ids: jax.Array,
    ep_weights: jax.Array,
    scale: jax.Array,
    *,
    expert_start: int,
    num_local_experts: int,
    topk: int,
    buffer_size: int,
    ep_valid: jax.Array | None = None,
    epsilon: float = 1e-8,
):
  """Plain-JAX reference of one expert-parallel shard: truncated dispatch, norm and float32 combine."""
  ep_tokens, dim = ep_x.shape
  flat_ids = ep_expert_ids.reshape(-1).astype(jnp.int32)
  local_ids = flat_ids - expert_start
  is_local = (local_ids >= 0) & (local_ids < num_local_experts)
  if ep_valid is not None:
    is_local = is_local & ep_valid.reshape(-1).astype(jnp.bool_)
  bucket_ids = jnp.where(is_local, local_ids, num_local_experts)
  perm = jnp.argsort(bucket_ids, stable=True).astype(jnp.int32)
  raw_group_sizes = jnp.bincount(bucket_ids, length=num_local_experts + 1)[:num_local_experts].astype(jnp.int32)
  cum_groups = jnp.minimum(jnp.cumsum(raw_group_sizes, dtype=jnp.int32), buffer_size)
  group_sizes = jnp.concatenate([cum_groups[:1], cum_groups[1:] - cum_groups[:-1]]).astype(jnp.int32)
  count = cum_groups[-1]

  slot_ids = perm[:buffer_size]
  valid = jnp.arange(buffer_size, dtype=jnp.int32) < count
  tok_ids = jnp.where(valid, slot_ids // topk, 0)
  # Gather in float32 so the VJP accumulates a token's top-k duplicates in float32, as the gather-reduce kernel does.
  x_sorted = jnp.where(valid[:, None], ep_x.astype(jnp.float32)[tok_ids].astype(ep_x.dtype), jnp.zeros((), ep_x.dtype))
  # An elementwise nonlinearity stands in for the expert MLP.
  expert_out = jnp.tanh(x_sorted.astype(jnp.float32) * 1.25).astype(ep_x.dtype)
  expert_out = jnp.where(valid[:, None], expert_out, jnp.zeros((), ep_x.dtype))
  normed = _input_scaled_rms_norm(expert_out, scale, epsilon)
  w_flat = ep_weights.reshape(-1).astype(jnp.float32)
  w_rows = jnp.where(valid, w_flat[jnp.where(valid, slot_ids, 0)], 0.0)
  weighted = jnp.where(
      valid[:, None],
      (normed.astype(jnp.float32) * w_rows[:, None]).astype(ep_x.dtype),
      jnp.zeros((), ep_x.dtype),
  )
  out_f32 = (
      jnp.zeros((ep_tokens + 1, dim), jnp.float32)
      .at[jnp.where(valid, tok_ids, ep_tokens)]
      .add(weighted.astype(jnp.float32))[:ep_tokens]
  )
  return x_sorted, group_sizes, raw_group_sizes, out_f32.astype(ep_x.dtype)


def _assert_close(actual, expected, *, rel: float, name: str):
  actual = np.asarray(actual, dtype=np.float32)
  expected = np.asarray(expected, dtype=np.float32)
  err = float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-30))
  assert err < rel, f"{name}: relative error {err} >= {rel}"


class MuseSparkKernelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="shard0_2d", shard_idx=0, keep_3d=False, tokens_3d=False, buffer_size=96, use_valid=False),
      dict(testcase_name="shard2_invalid_slots_2d", shard_idx=2, keep_3d=False, tokens_3d=False, buffer_size=96, use_valid=True),
      dict(testcase_name="shard1_truncated_3d", shard_idx=1, keep_3d=True, tokens_3d=True, buffer_size=40, use_valid=False),
      dict(testcase_name="shard3_full_3d", shard_idx=3, keep_3d=True, tokens_3d=False, buffer_size=128, use_valid=True),
  )
  def test_local_dispatch_norm_and_combine_match_reference(self, shard_idx, keep_3d, tokens_3d, buffer_size, use_valid):
    ep_tokens, dim, num_experts, ep_size, topk = 64, 256, 32, 4, 4
    num_local_experts = num_experts // ep_size
    expert_start = shard_idx * num_local_experts

    kx, ke, kw, ks, kv, kct = jax.random.split(jax.random.PRNGKey(17 + shard_idx), 6)
    ep_x = jax.random.normal(kx, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)
    logits = jax.random.normal(ke, (ep_tokens, num_experts), jnp.float32)
    _, ep_expert_ids = jax.lax.top_k(logits, topk)
    ep_expert_ids = ep_expert_ids.astype(jnp.int32)
    ep_weights = jax.random.uniform(kw, (ep_tokens, topk), jnp.float32, minval=0.05, maxval=0.95)
    scale = (0.1 * jax.random.normal(ks, (dim,), jnp.float32)).astype(jnp.bfloat16)
    ep_valid = jax.random.bernoulli(kv, 0.85, (ep_tokens, topk)) if use_valid else None
    cotangent = jax.random.normal(kct, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)

    def kernel_loss(x_in, w_in, s_in):
      x_disp = x_in.reshape(ep_tokens, dim // 128, 128) if tokens_3d else x_in
      x_sorted, local_routing, full_perm = ragged_sort_tc.local_ep_ragged_sort_tc(
          x_disp,
          ep_expert_ids,
          num_local_experts=num_local_experts,
          expert_start=expert_start,
          topk=topk,
          buffer_size=buffer_size,
          ep_valid=ep_valid,
          mask_padding=False,
          keep_3d=keep_3d,
          return_full_perm=True,
          is_tpu=False,
      )
      expert_out = jnp.tanh(x_sorted.astype(jnp.float32) * 1.25).astype(x_in.dtype)
      w_rows = ragged_sort_tc.tc_local_ep_row_weights(local_routing.tc_routing, w_in, buffer_size, is_tpu=False)
      weighted = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          expert_out,
          s_in,
          local_routing.tc_routing.count,
          sorted_weights=w_rows,
          epsilon=1e-8,
          is_tpu=False,
      )
      combined = ragged_sort_tc.ring_ragged_unsort_tc(
          weighted,
          local_routing.tc_routing,
          topk,
          w_in,
          mask_padding=False,
          prescaled=True,
          out_3d=tokens_3d,
          is_tpu=False,
      )
      combined_2d = combined.reshape(ep_tokens, dim)
      loss = jnp.sum(combined_2d.astype(jnp.float32) * cotangent.astype(jnp.float32))
      return loss, (x_sorted, local_routing, full_perm, combined_2d)

    def reference_loss(x_in, w_in, s_in):
      x_sorted, group_sizes, raw_group_sizes, combined_2d = _reference_local_moe_shard(
          x_in,
          ep_expert_ids,
          w_in,
          s_in,
          expert_start=expert_start,
          num_local_experts=num_local_experts,
          topk=topk,
          buffer_size=buffer_size,
          ep_valid=ep_valid,
      )
      loss = jnp.sum(combined_2d.astype(jnp.float32) * cotangent.astype(jnp.float32))
      return loss, (x_sorted, group_sizes, raw_group_sizes, combined_2d)

    (loss_k, (xs_k, routing_k, full_perm, y_k)), (dx_k, dw_k, ds_k) = jax.value_and_grad(
        kernel_loss, argnums=(0, 1, 2), has_aux=True
    )(ep_x, ep_weights, scale)
    (loss_r, (xs_r, gs_r, raw_gs_r, y_r)), (dx_r, dw_r, ds_r) = jax.value_and_grad(
        reference_loss, argnums=(0, 1, 2), has_aux=True
    )(ep_x, ep_weights, scale)

    count = int(routing_k.tc_routing.count)
    self.assertEqual(full_perm.shape, (ep_tokens * topk,))
    np.testing.assert_array_equal(np.asarray(routing_k.group_sizes), np.asarray(gs_r))
    np.testing.assert_array_equal(np.asarray(routing_k.raw_group_sizes), np.asarray(raw_gs_r))
    np.testing.assert_array_equal(
        np.asarray(xs_k.reshape(buffer_size, dim)[:count].astype(jnp.float32)),
        np.asarray(xs_r[:count].astype(jnp.float32)),
    )
    # The forward pass is the same arithmetic in the same order, so it is exact. The backward pass of the norm is a
    # hand-written VJP in the kernel and autodiff in the reference, so it agrees to bfloat16 rounding.
    np.testing.assert_array_equal(np.asarray(y_k.astype(jnp.float32)), np.asarray(y_r.astype(jnp.float32)))
    self.assertEqual(float(loss_k), float(loss_r))
    _assert_close(dx_k, dx_r, rel=2e-2, name="d_tokens")
    _assert_close(dw_k, dw_r, rel=2e-2, name="d_routing_weights")
    _assert_close(ds_k, ds_r, rel=2e-2, name="d_norm_scale")

  def test_local_row_weights_vjp_saves_no_float_residuals(self):
    num_tokens, topk, cap = 32, 4, 64
    ep_x = jnp.ones((num_tokens, 128), jnp.bfloat16)
    ep_ids = jnp.zeros((num_tokens, topk), jnp.int32)
    _, local_routing, _ = ragged_sort_tc.local_ep_ragged_sort_tc(
        ep_x,
        ep_ids,
        num_local_experts=8,
        expert_start=0,
        topk=topk,
        buffer_size=cap,
        is_tpu=False,
    )
    w_flat = jnp.linspace(0.1, 1.0, num_tokens * topk, dtype=jnp.float32)
    _, vjp_fn = jax.vjp(
        lambda w_in: ragged_sort_tc.tc_local_ep_row_weights(local_routing.tc_routing, w_in, cap, is_tpu=False),
        w_flat,
    )
    float_residuals = [
        leaf
        for leaf in jax.tree.leaves(vjp_fn)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    self.assertEmpty(float_residuals)

  def test_rows_beyond_count_do_not_affect_norm_or_combine(self):
    ep_tokens, dim, topk, cap = 32, 256, 4, 80
    kx, kw, ks, kg = jax.random.split(jax.random.PRNGKey(99), 4)
    ep_x = jax.random.normal(kx, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)
    # Half of the slots go to local expert 1; the rest go to expert 7, which this shard does not hold.
    ep_ids = jnp.where((jnp.arange(ep_tokens * topk) % 2 == 0).reshape(ep_tokens, topk), 1, 7).astype(jnp.int32)
    ep_weights = jax.random.uniform(kw, (ep_tokens, topk), jnp.float32, minval=0.1, maxval=0.9)
    scale = (0.05 * jax.random.normal(ks, (dim,), jnp.float32)).astype(jnp.bfloat16)
    dy = jax.random.normal(kg, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)

    _, local_routing, _ = ragged_sort_tc.local_ep_ragged_sort_tc(
        ep_x,
        ep_ids,
        num_local_experts=4,
        expert_start=0,
        topk=topk,
        buffer_size=cap,
        mask_padding=False,
        is_tpu=False,
    )
    count = int(local_routing.tc_routing.count)
    self.assertLess(count, cap)

    clean = jnp.ones((cap, dim), jnp.bfloat16) * 0.5
    nan_tail = jnp.concatenate([clean[:count], jnp.full((cap - count, dim), jnp.nan, dtype=jnp.bfloat16)], axis=0)
    zero_tail = jnp.concatenate([clean[:count], jnp.zeros((cap - count, dim), jnp.bfloat16)], axis=0)

    def run_pipeline(buf_in, w_in, s_in):
      w_rows = ragged_sort_tc.tc_local_ep_row_weights(local_routing.tc_routing, w_in, cap, is_tpu=False)
      weighted = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          buf_in, s_in, local_routing.tc_routing.count, sorted_weights=w_rows, epsilon=1e-8, is_tpu=False
      )
      out = ragged_sort_tc.ring_ragged_unsort_tc(
          weighted,
          local_routing.tc_routing,
          topk,
          w_in,
          mask_padding=False,
          prescaled=True,
          is_tpu=False,
      )
      return jnp.sum(out.astype(jnp.float32) * dy.astype(jnp.float32))

    loss_nan, (dbuf_nan, dw_nan, ds_nan) = jax.value_and_grad(run_pipeline, argnums=(0, 1, 2))(
        nan_tail, ep_weights, scale
    )
    loss_zero, (dbuf_zero, dw_zero, ds_zero) = jax.value_and_grad(run_pipeline, argnums=(0, 1, 2))(
        zero_tail, ep_weights, scale
    )

    self.assertTrue(np.isfinite(float(loss_nan)))
    self.assertEqual(float(loss_nan), float(loss_zero))
    for grad in (dbuf_nan, dw_nan, ds_nan):
      self.assertTrue(bool(np.all(np.isfinite(np.asarray(grad, np.float32)))))
    np.testing.assert_array_equal(np.asarray(dbuf_nan[:count], np.float32), np.asarray(dbuf_zero[:count], np.float32))
    np.testing.assert_array_equal(np.asarray(dw_nan), np.asarray(dw_zero))
    np.testing.assert_array_equal(np.asarray(ds_nan, np.float32), np.asarray(ds_zero, np.float32))

  def test_empty_shard_produces_zero_gradients(self):
    cap, dim = 256, 256
    expert_out = jnp.full((cap, dim), jnp.nan, dtype=jnp.bfloat16)
    scale = jnp.ones((dim,), dtype=jnp.bfloat16) * 0.2
    weights = jnp.ones((cap,), dtype=jnp.float32) * 0.5

    def loss_fn(x_in, s_in, w_in):
      out = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          x_in, s_in, jnp.int32(0), sorted_weights=w_in, is_tpu=False
      )
      # No row is valid, so nothing may be read from the (all-NaN) buffer.
      return jnp.sum(jnp.where(jnp.arange(cap)[:, None] < 0, out.astype(jnp.float32), 0.0))

    loss, (dx, dscale, dw) = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(expert_out, scale, weights)
    self.assertEqual(float(loss), 0.0)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(dscale, np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(dw, np.float32), 0.0)


if __name__ == "__main__":
  absltest.main()
