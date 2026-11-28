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

"""Unit tests for the Muse Spark expert-parallel dispatch and fused post-expert norm kernels."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import muse_spark_post_expert_norm
from maxtext.kernels.ragged import ragged_sort_tc


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


@pytest.mark.cpu_only
class MuseSparkKernelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "shard0_2d",
          "shard_idx": 0,
          "keep_3d": False,
          "tokens_3d": False,
          "buffer_size": 96,
          "use_valid": False,
      },
      {
          "testcase_name": "shard2_invalid_slots_2d",
          "shard_idx": 2,
          "keep_3d": False,
          "tokens_3d": False,
          "buffer_size": 96,
          "use_valid": True,
      },
      {
          "testcase_name": "shard1_truncated_3d",
          "shard_idx": 1,
          "keep_3d": True,
          "tokens_3d": True,
          "buffer_size": 40,
          "use_valid": False,
      },
      {
          "testcase_name": "shard3_full_3d",
          "shard_idx": 3,
          "keep_3d": True,
          "tokens_3d": False,
          "buffer_size": 128,
          "use_valid": True,
      },
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
        leaf for leaf in jax.tree.leaves(vjp_fn) if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
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

  @parameterized.named_parameters(("2d", False), ("3d", True))
  def test_fused_post_expert_norm_without_sorted_weights(self, is_3d):
    cap, count, dim = 48, 31, 256
    kx, ks, kg = jax.random.split(jax.random.PRNGKey(211 if is_3d else 210), 3)
    x_2d = jax.random.normal(kx, (cap, dim), jnp.float32).astype(jnp.bfloat16)
    x_2d = x_2d.at[count:].set(jnp.nan)
    scale = (0.1 * jax.random.normal(ks, (dim,), jnp.float32)).astype(jnp.bfloat16)
    cot = jax.random.normal(kg, (cap, dim), jnp.float32).astype(jnp.bfloat16)

    def kernel_fn(x_in, s_in):
      buf = x_in.reshape(cap, dim // 128, 128) if is_3d else x_in
      out = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          buf, s_in, jnp.int32(count), sorted_weights=None, epsilon=1e-8, is_tpu=False
      ).reshape(cap, dim)
      return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32)), out

    def ref_fn(x_in, s_in):
      valid = (jnp.arange(cap) < count)[:, None]
      clean_x = jnp.where(valid, x_in, jnp.zeros((), x_in.dtype))
      out = jnp.where(valid, _input_scaled_rms_norm(clean_x, s_in, 1e-8), jnp.zeros((), x_in.dtype))
      return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32)), out

    (loss_k, out_k), (dx_k, ds_k) = jax.value_and_grad(kernel_fn, argnums=(0, 1), has_aux=True)(x_2d, scale)
    (loss_r, out_r), (dx_r, ds_r) = jax.value_and_grad(ref_fn, argnums=(0, 1), has_aux=True)(x_2d, scale)
    np.testing.assert_array_equal(np.asarray(out_k, np.float32), np.asarray(out_r, np.float32))
    self.assertEqual(float(loss_k), float(loss_r))
    _assert_close(dx_k, dx_r, rel=2e-2, name="dx_no_weights")
    _assert_close(ds_k, ds_r, rel=2e-2, name="ds_no_weights")

  def test_row_weights_scatter_sort_large_slots_matches_direct_gather(self):
    """`is_tpu=True` with `num_slots >= 4096` exercises the pure-JAX scatter+sort path on CPU."""
    ep_tokens, topk, cap = 1024, 4, 512  # num_slots = 4096 >= 4096
    ke, kw, kg = jax.random.split(jax.random.PRNGKey(303), 3)
    ep_x = jnp.zeros((ep_tokens, 128), jnp.bfloat16)
    ep_ids = jax.random.randint(ke, (ep_tokens, topk), 0, 16, dtype=jnp.int32)
    ep_weights = jax.random.uniform(kw, (ep_tokens, topk), jnp.float32, minval=0.05, maxval=0.95)
    cot = jax.random.normal(kg, (cap,), jnp.float32)

    _, local_routing, _ = ragged_sort_tc.local_ep_ragged_sort_tc(
        ep_x, ep_ids, num_local_experts=4, expert_start=0, topk=topk, buffer_size=cap, is_tpu=False
    )

    def run_weights(w_in, use_tpu_path: bool):
      w_rows = ragged_sort_tc.tc_local_ep_row_weights(local_routing.tc_routing, w_in, cap, is_tpu=use_tpu_path)
      return jnp.sum(w_rows * cot), w_rows

    (_, w_tpu), dw_tpu = jax.value_and_grad(lambda w: run_weights(w, True), has_aux=True)(ep_weights)
    (_, w_cpu), dw_cpu = jax.value_and_grad(lambda w: run_weights(w, False), has_aux=True)(ep_weights)
    np.testing.assert_array_equal(np.asarray(w_tpu), np.asarray(w_cpu))
    np.testing.assert_array_equal(np.asarray(dw_tpu), np.asarray(dw_cpu))


@pytest.mark.tpu_only
class MuseSparkKernelsTpuTest(parameterized.TestCase):
  """Parity tests for the real Pallas / TC TPU kernels (`is_tpu=True`) against the pure-JAX reference."""

  def setUp(self):
    super().setUp()
    try:
      tpu_devices = jax.devices("tpu")
    except RuntimeError:
      tpu_devices = []
    if not tpu_devices:
      self.skipTest("Pallas TPU kernels require TPU hardware.")

  @parameterized.named_parameters(
      {"testcase_name": "2d_weighted", "is_3d": False, "dim": 512, "has_weights": True, "count": 290},
      {"testcase_name": "3d_weighted_4096", "is_3d": True, "dim": 4096, "has_weights": True, "count": 300},
      {"testcase_name": "3d_unweighted_non8_tiles", "is_3d": True, "dim": 2560, "has_weights": False, "count": 140},
      {"testcase_name": "3d_empty_count", "is_3d": True, "dim": 1024, "has_weights": True, "count": 0},
  )
  def test_pallas_post_expert_norm_matches_reference(self, is_3d, dim, has_weights, count):
    cap = 384
    kx, ks, kw, kg = jax.random.split(jax.random.PRNGKey(400 + count + int(is_3d)), 4)
    x_2d = jax.random.normal(kx, (cap, dim), jnp.float32).astype(jnp.bfloat16)
    x_2d = x_2d.at[count:].set(jnp.nan)
    scale = (0.1 * jax.random.normal(ks, (dim,), jnp.float32)).astype(jnp.bfloat16)
    w_rows = jax.random.uniform(kw, (cap,), jnp.float32, minval=0.1, maxval=0.9)
    cot = jax.random.normal(kg, (cap, dim), jnp.float32).astype(jnp.bfloat16)

    def run_norm(x_in, s_in, w_in, *, is_tpu: bool):
      buf = x_in.reshape(cap, dim // 128, 128) if is_3d else x_in
      out = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          buf,
          s_in,
          jnp.int32(count),
          sorted_weights=w_in if has_weights else None,
          epsilon=1e-8,
          is_tpu=is_tpu,
      ).reshape(cap, dim)
      valid = jnp.arange(cap, dtype=jnp.int32)[:, None] < count
      masked_out = jnp.where(valid, out.astype(jnp.float32), 0.0)
      return jnp.sum(masked_out * cot.astype(jnp.float32)), out

    (loss_tpu, out_tpu), grads_tpu = jax.value_and_grad(
        lambda x, s, w: run_norm(x, s, w, is_tpu=True), argnums=(0, 1, 2), has_aux=True
    )(x_2d, scale, w_rows)
    (_, out_ref), grads_ref = jax.value_and_grad(
        lambda x, s, w: run_norm(x, s, w, is_tpu=False), argnums=(0, 1, 2), has_aux=True
    )(x_2d, scale, w_rows)

    if count == 0:
      # Unvisited row blocks stay untouched in HBM; verify loss and deterministically zeroed `dscale`/`dw`.
      self.assertEqual(float(loss_tpu), 0.0)
      np.testing.assert_array_equal(np.asarray(grads_tpu[1], np.float32), 0.0)
      if has_weights:
        np.testing.assert_array_equal(np.asarray(grads_tpu[2], np.float32), 0.0)
    else:
      _assert_close(out_tpu[:count], out_ref[:count], rel=2e-4, name="out")
      _assert_close(grads_tpu[0][:count], grads_ref[0][:count], rel=5e-4, name="dx")
      _assert_close(grads_tpu[1], grads_ref[1], rel=1e-3, name="dscale")
      if has_weights:
        _assert_close(grads_tpu[2], grads_ref[2], rel=5e-4, name="dw")

  @parameterized.named_parameters(("2d", False), ("3d", True))
  def test_tpu_local_ep_dispatch_and_combine_match_reference(self, keep_3d):
    ep_tokens, dim, num_experts, ep_size, topk, cap = 256, 512, 16, 4, 4, 384
    num_local_experts = num_experts // ep_size
    expert_start = num_local_experts
    kx, ke, kw, ks, kct = jax.random.split(jax.random.PRNGKey(500 + int(keep_3d)), 5)
    ep_x = jax.random.normal(kx, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)
    _, ep_ids = jax.lax.top_k(jax.random.normal(ke, (ep_tokens, num_experts), jnp.float32), topk)
    ep_ids = ep_ids.astype(jnp.int32)
    ep_weights = jax.random.uniform(kw, (ep_tokens, topk), jnp.float32, minval=0.05, maxval=0.95)
    scale = (0.1 * jax.random.normal(ks, (dim,), jnp.float32)).astype(jnp.bfloat16)
    cot = jax.random.normal(kct, (ep_tokens, dim), jnp.float32).astype(jnp.bfloat16)

    def tpu_loss(x_in, w_in, s_in):
      x_sorted, local_rt, _ = ragged_sort_tc.local_ep_ragged_sort_tc(
          x_in,
          ep_ids,
          num_local_experts=num_local_experts,
          expert_start=expert_start,
          topk=topk,
          buffer_size=cap,
          mask_padding=False,
          keep_3d=keep_3d,
          is_tpu=True,
      )
      expert_out = jnp.tanh(x_sorted.astype(jnp.float32) * 1.25).astype(x_in.dtype)
      w_rows = ragged_sort_tc.tc_local_ep_row_weights(local_rt.tc_routing, w_in, cap, is_tpu=True)
      weighted = muse_spark_post_expert_norm.fused_ragged_post_expert_norm(
          expert_out, s_in, local_rt.tc_routing.count, sorted_weights=w_rows, epsilon=1e-8, is_tpu=True
      )
      combined = ragged_sort_tc.ring_ragged_unsort_tc(
          weighted, local_rt.tc_routing, topk, w_in, mask_padding=False, prescaled=True, is_tpu=True
      )
      return jnp.sum(combined.astype(jnp.float32) * cot.astype(jnp.float32)), combined

    def ref_loss(x_in, w_in, s_in):
      _, _, _, combined = _reference_local_moe_shard(
          x_in,
          ep_ids,
          w_in,
          s_in,
          expert_start=expert_start,
          num_local_experts=num_local_experts,
          topk=topk,
          buffer_size=cap,
      )
      return jnp.sum(combined.astype(jnp.float32) * cot.astype(jnp.float32)), combined

    (_, y_tpu), (dx_tpu, dw_tpu, ds_tpu) = jax.value_and_grad(tpu_loss, argnums=(0, 1, 2), has_aux=True)(
        ep_x, ep_weights, scale
    )
    (_, y_ref), (dx_ref, dw_ref, ds_ref) = jax.value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(
        ep_x, ep_weights, scale
    )
    _assert_close(y_tpu, y_ref, rel=1e-2, name="combined")
    _assert_close(dx_tpu, dx_ref, rel=2e-2, name="dx")
    _assert_close(dw_tpu, dw_ref, rel=2e-2, name="dw")
    _assert_close(ds_tpu, ds_ref, rel=2e-2, name="dscale")


if __name__ == "__main__":
  absltest.main()
