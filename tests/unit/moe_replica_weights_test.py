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

"""CPU and TPU unit tests for `moe_replica_weights.extend_with_replicas`."""

import functools
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import qwix.pallas as qpl

from maxtext.layers import moe_replica_weights as mrw
from tests.utils import moe_replica_test_utils

AXIS = "ep"
NUM_EP = 4
LOCAL = 3
NUM_REPLICAS = 2
DIM = 5


def _hand_plan():
  """4-device plan where home 0 sends local expert 2 to peers 1 and 3, and home 2 sends local expert 0 to peer 1."""
  # Receivers:
  #   dev 0: [-1, -1]
  #   dev 1: [home 0 slot 2, home 2 slot 0]
  #   dev 2: [-1, -1]
  #   dev 3: [home 0 slot 2, -1]
  return types.SimpleNamespace(
      rep_home=jnp.array([[-1, -1], [0, 2], [-1, -1], [0, -1]], jnp.int32),
      rep_slot=jnp.array([[0, 0], [2, 0], [0, 0], [2, 0]], jnp.int32),
      send_slots=jnp.array([[2, 2], [0, 0], [0, 0], [0, 0]], jnp.int32),
      send_peer_index=jnp.array(
          [[-1, 0, -1, 1], [-1, -1, -1, -1], [-1, 0, -1, -1], [-1, -1, -1, -1]],
          jnp.int32,
      ),
      num_slots=LOCAL + NUM_REPLICAS,
      num_replicas=NUM_REPLICAS,
  )


def _plan_from_transfers(num_ep, local, num_replicas, max_send, transfers):
  """Builds a deterministic plan from `(home, local_slot, receiver)` triples satisfying all plan invariants."""
  rep_home = np.full((num_ep, num_replicas), -1, dtype=np.int32)
  rep_slot = np.zeros((num_ep, num_replicas), dtype=np.int32)
  send_slots = np.zeros((num_ep, max_send), dtype=np.int32)
  send_peer_index = np.full((num_ep, num_ep), -1, dtype=np.int32)
  recv_fill = np.zeros((num_ep,), dtype=np.int32)
  send_fill = np.zeros((num_ep,), dtype=np.int32)
  for h, loc, r in sorted(transfers, key=lambda t: (t[0], t[2])):
    assert 0 <= h < num_ep and 0 <= r < num_ep and h != r and 0 <= loc < local
    assert send_peer_index[h, r] == -1 and recv_fill[r] < num_replicas and send_fill[h] < max_send
    s = recv_fill[r]
    recv_fill[r] += 1
    rep_home[r, s] = h
    rep_slot[r, s] = loc
    k = send_fill[h]
    send_fill[h] += 1
    send_slots[h, k] = loc
    send_peer_index[h, r] = k
  return types.SimpleNamespace(
      rep_home=jnp.asarray(rep_home),
      rep_slot=jnp.asarray(rep_slot),
      send_slots=jnp.asarray(send_slots),
      send_peer_index=jnp.asarray(send_peer_index),
      num_slots=local + num_replicas,
      num_replicas=num_replicas,
  )


class MoveParamsTest(absltest.TestCase):

  def test_send_and_recv_sizes_match_across_peers(self):
    plan = _hand_plan()
    params = [mrw._direct_move_params(plan, me, NUM_EP) for me in range(NUM_EP)]  # pylint: disable=protected-access
    for i in range(NUM_EP):
      for j in range(NUM_EP):
        self.assertEqual(int(params[i][1][j]), int(params[j][3][i]))
    self.assertEqual(sum(int(p[1].sum()) for p in params), int((np.asarray(plan.rep_home) >= 0).sum()))
    # Verify direct input offsets point to the exact home-local slots sent to active receivers.
    self.assertEqual(int(params[0][0][1]), 2)
    self.assertEqual(int(params[0][0][3]), 2)
    self.assertEqual(int(params[2][0][1]), 0)


class ExtendWithReplicasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.plan = _hand_plan()
    self.w = rng.standard_normal((NUM_EP, LOCAL, DIM, 4)).astype(np.float32)
    self.g = rng.standard_normal((NUM_EP, LOCAL + NUM_REPLICAS, DIM, 4)).astype(np.float32)
    patcher = mock.patch.object(jax.lax, "ragged_all_to_all", moe_replica_test_utils.emulated_ragged_all_to_all)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _vjp(self, fn, w_in, cotangent):
    def per_device(w, g):
      out, vjp = jax.vjp(fn, w)
      return out, vjp(g)[0]

    return jax.jit(jax.vmap(per_device, axis_name=AXIS))(jnp.asarray(w_in), jnp.asarray(cotangent))

  def test_forward_and_backward_sum_multiple_replicas_to_home(self):
    out, dw = self._vjp(
        lambda w: mrw.extend_with_replicas(w, self.plan, AXIS, NUM_EP, jnp.float32),
        self.w,
        self.g,
    )
    reps, dw_rep = moe_replica_test_utils.replica_reference(self.w, self.g[:, LOCAL:], self.plan)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([self.w, reps], axis=1))
    np.testing.assert_allclose(np.asarray(dw), self.g[:, :LOCAL] + dw_rep, rtol=1e-6, atol=1e-6)

  def test_corrupted_bwd_recv_size_raises(self):
    """Verifies the CPU emulator rejects a mismatched backward `recv_sizes` argument."""
    orig_extend_direct = mrw._extend_direct  # pylint: disable=protected-access

    def corrupt_bwd_recv(*args):
      idx = args[6]
      corrupted_idx = (*idx[:8], jnp.zeros_like(idx[8]))
      return orig_extend_direct(*args[:6], corrupted_idx)

    with mock.patch.object(mrw, "_extend_direct", side_effect=corrupt_bwd_recv):
      with self.assertRaisesRegex((ValueError, jax.errors.JaxRuntimeError), "recv_sizes"):
        out, dw = self._vjp(
            lambda w: mrw.extend_with_replicas(w, self.plan, AXIS, NUM_EP, jnp.float32),
            self.w,
            self.g,
        )
        jax.block_until_ready((out, dw))

  def test_slot0_with_unused_send_slots_padded_to_zero(self):
    """Regression test: home 0 sends local slot 0 while max_send=4 pads unused send_slots with 0s."""
    plan = types.SimpleNamespace(
        rep_home=jnp.array([[-1, -1], [0, -1], [-1, -1], [-1, -1]], jnp.int32),
        rep_slot=jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32),
        send_slots=jnp.zeros((NUM_EP, 4), jnp.int32),
        send_peer_index=jnp.array([[-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]], jnp.int32),
        num_slots=LOCAL + NUM_REPLICAS,
        num_replicas=NUM_REPLICAS,
    )
    out, dw = self._vjp(
        lambda w: mrw.extend_with_replicas(w, plan, AXIS, NUM_EP, jnp.float32),
        self.w,
        self.g,
    )
    reps, dw_rep = moe_replica_test_utils.replica_reference(self.w, self.g[:, LOCAL:], plan)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([self.w, reps], axis=1))
    np.testing.assert_allclose(np.asarray(dw), self.g[:, :LOCAL] + dw_rep, rtol=1e-6, atol=1e-6)

  def test_empty_plan_leaves_replicas_zero_and_home_grad_exact(self):
    empty_plan = types.SimpleNamespace(
        rep_home=jnp.full((NUM_EP, NUM_REPLICAS), -1, jnp.int32),
        rep_slot=jnp.zeros((NUM_EP, NUM_REPLICAS), jnp.int32),
        send_slots=jnp.zeros((NUM_EP, 2), jnp.int32),
        send_peer_index=jnp.full((NUM_EP, NUM_EP), -1, jnp.int32),
        num_slots=LOCAL + NUM_REPLICAS,
        num_replicas=NUM_REPLICAS,
    )
    out, dw = self._vjp(
        lambda w: mrw.extend_with_replicas(w, empty_plan, AXIS, NUM_EP, jnp.float32),
        self.w,
        self.g,
    )
    np.testing.assert_array_equal(np.asarray(out[:, :LOCAL]), self.w)
    np.testing.assert_array_equal(np.asarray(out[:, LOCAL:]), 0.0)
    np.testing.assert_array_equal(np.asarray(dw), self.g[:, :LOCAL])

  def test_mixed_precision_fp32_master_weights_to_bf16_compute(self):
    g_bf16 = jnp.asarray(self.g, dtype=jnp.bfloat16)
    out, dw = self._vjp(
        lambda w: mrw.extend_with_replicas(w, self.plan, AXIS, NUM_EP, jnp.bfloat16),
        self.w,
        g_bf16,
    )
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(dw.dtype, jnp.float32)
    w_bf16 = np.asarray(jnp.asarray(self.w, dtype=jnp.bfloat16), dtype=np.float32)
    g_f32 = np.asarray(g_bf16, dtype=np.float32)
    reps, dw_rep = moe_replica_test_utils.replica_reference(w_bf16, g_f32[:, LOCAL:], self.plan)
    np.testing.assert_array_equal(np.asarray(out, dtype=np.float32), np.concatenate([w_bf16, reps], axis=1))
    np.testing.assert_allclose(np.asarray(dw), g_f32[:, :LOCAL] + dw_rep, rtol=1e-2, atol=1e-2)

  def test_noop_when_zero_replicas_or_single_device(self):
    w_single = jnp.asarray(self.w[0], dtype=jnp.float32)
    g_single = jnp.asarray(self.g[0, :LOCAL], dtype=jnp.bfloat16)
    zero_rep_plan = types.SimpleNamespace(num_replicas=0, send_slots=jnp.zeros((NUM_EP, 2), jnp.int32))
    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=AssertionError("must not call ra2a")):
      out_r0, vjp_r0 = jax.vjp(lambda w: mrw.extend_with_replicas(w, zero_rep_plan, AXIS, NUM_EP, jnp.bfloat16), w_single)
      (dw_r0,) = vjp_r0(g_single)
      out_ep1, vjp_ep1 = jax.vjp(lambda w: mrw.extend_with_replicas(w, self.plan, AXIS, 1, jnp.bfloat16), w_single)
      (dw_ep1,) = vjp_ep1(g_single)
    self.assertEqual(out_r0.shape, w_single.shape)
    self.assertEqual(out_r0.dtype, jnp.bfloat16)
    self.assertEqual(dw_r0.dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(out_r0), np.asarray(w_single.astype(jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(out_ep1), np.asarray(w_single.astype(jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(dw_r0), np.asarray(g_single.astype(jnp.float32)))
    np.testing.assert_array_equal(np.asarray(dw_ep1), np.asarray(g_single.astype(jnp.float32)))

  def test_invalid_plan_parameters_raise_value_error(self):
    w_single = jnp.asarray(self.w[0], dtype=jnp.float32)
    bad_cases = [
        types.SimpleNamespace(num_replicas=-1, send_slots=jnp.zeros((NUM_EP, 2), jnp.int32)),
        types.SimpleNamespace(num_replicas=2, send_slots=jnp.zeros((NUM_EP,), jnp.int32)),
        types.SimpleNamespace(num_replicas=2, send_slots=jnp.zeros((NUM_EP, 0), jnp.int32)),
        types.SimpleNamespace(num_replicas=2, send_slots=jnp.zeros((NUM_EP, 9), jnp.int32)),
        types.SimpleNamespace(num_replicas=2, send_slots=jnp.full((NUM_EP, 2), LOCAL, jnp.int32)),
    ]
    for bad_plan in bad_cases:
      with self.assertRaises(ValueError):
        mrw.extend_with_replicas(w_single, bad_plan, AXIS, NUM_EP, jnp.float32)
    with self.assertRaises(ValueError):
      mrw.extend_with_replicas(w_single, self.plan, AXIS, 0, jnp.float32)


@pytest.mark.tpu_only
class ExtendWithReplicasTpuTest(parameterized.TestCase):
  """Runs `extend_with_replicas` under `jax.shard_map` with the real TPU `ragged_all_to_all` collective."""

  def _run_shard_map_vjp(self, devs, plan, w_np, g_np, *, compute_dtype):
    """Executes forward and backward `extend_with_replicas` inside `jax.shard_map`."""
    self.assertIsNot(jax.lax.ragged_all_to_all, moe_replica_test_utils.emulated_ragged_all_to_all)
    num_ep = len(devs)
    mesh = Mesh(np.asarray(devs), (AXIS,))
    w_sharded = jax.device_put(jnp.asarray(w_np), jax.sharding.NamedSharding(mesh, P(AXIS)))
    g_sharded = jax.device_put(jnp.asarray(g_np, dtype=compute_dtype), jax.sharding.NamedSharding(mesh, P(AXIS)))

    @jax.jit
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(AXIS), P(AXIS)), out_specs=(P(AXIS), P(AXIS)))
    def step(w_shard, g_shard):
      w_local = w_shard[0]
      g_local = g_shard[0]
      out_local, vjp_fn = jax.vjp(lambda w: mrw.extend_with_replicas(w, plan, AXIS, num_ep, compute_dtype), w_local)
      (dw_local,) = vjp_fn(g_local)
      return out_local[None], dw_local[None]

    return step(w_sharded, g_sharded)

  @parameterized.named_parameters(
      ("hand_plan_f32", "hand", 2, 2, jnp.float32),
      ("hand_plan_bf16", "hand", 2, 2, jnp.bfloat16),
      ("r1_maxsend1_f32", "r1_ms1", 1, 1, jnp.float32),
      ("r2_maxsend3_fanout_f32", "r2_ms3", 2, 3, jnp.float32),
      ("r3_maxsend2_multi_home_bf16", "r3_ms2", 3, 2, jnp.bfloat16),
  )
  def test_real_ragged_all_to_all_forward_and_backward_match_reference(self, kind, num_replicas, max_send, compute_dtype):
    """Checks forward replica rows and home-summed backward grads on real TPU `ragged_all_to_all`."""
    devs = jax.devices()
    if kind == "hand":
      # _hand_plan() has fixed EP=4; slice an exact 4-device mesh (skipping if fewer than 4 TPU devices exist).
      if len(devs) < 4:
        self.skipTest(f"_hand_plan requires 4 TPU devices, got {len(devs)}.")
      devs = devs[:4]
      plan = _hand_plan()
    else:
      # Hand-built transfer fixtures below address device indices up to 3 (EP >= 4).
      if len(devs) < 4:
        self.skipTest(f"Transfer fixture {kind} requires >= 4 TPU devices, got {len(devs)}.")
      num_ep = len(devs)
      if kind == "r1_ms1":
        transfers = [(h, h % LOCAL, (h + 1) % num_ep) for h in range(0, num_ep, 2)]
      elif kind == "r2_ms3":
        transfers = [(0, 2, 1), (0, 2, 2), (0, 1, 3), (2, 0, 1)]
      else:
        transfers = [(0, 1, 2), (0, 1, 3), (1, 2, 2), (2, 0, 3), (3, 0, 2)]
      plan = _plan_from_transfers(num_ep, LOCAL, num_replicas, max_send, transfers)

    num_ep = len(devs)
    rng = np.random.default_rng(42 + num_replicas * 10 + max_send)
    # Test weight tensor shape [EP, L, 16, 128].
    w_np = rng.standard_normal((num_ep, LOCAL, 16, 128)).astype(np.float32)
    g_np = rng.standard_normal((num_ep, LOCAL + num_replicas, 16, 128)).astype(np.float32)

    out, dw = self._run_shard_map_vjp(devs, plan, w_np, g_np, compute_dtype=compute_dtype)
    self.assertEqual(out.dtype, compute_dtype)
    self.assertEqual(dw.dtype, jnp.float32)

    w_ref = np.asarray(jnp.asarray(w_np, dtype=compute_dtype), dtype=np.float32)
    g_ref = np.asarray(jnp.asarray(g_np, dtype=compute_dtype), dtype=np.float32)
    reps, dw_rep = moe_replica_test_utils.replica_reference(w_ref, g_ref[:, LOCAL:], plan)
    np.testing.assert_array_equal(np.asarray(out, dtype=np.float32), np.concatenate([w_ref, reps], axis=1))
    tol = 1e-2 if compute_dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(np.asarray(dw), g_ref[:, :LOCAL] + dw_rep, rtol=tol, atol=tol)


@jax.custom_vjp
def _ste_consume(q, g):
  """sum(dequant(q) * g) whose bwd mimics gmm: float d(dequantized weight) as the QArray qvalue cotangent."""
  return jnp.sum(q.qvalue.astype(jnp.float32) * q.scale.astype(jnp.float32) * g.astype(jnp.float32))


def _ste_consume_fwd(q, g):
  return _ste_consume(q, g), (q, g)


def _ste_consume_bwd(res, ct):
  q, g = res
  dq = qpl.QArray(qvalue=(ct * g).astype(g.dtype), scale=jnp.zeros_like(q.scale), zero_point=None, qtype=q.qtype)
  return dq, jnp.zeros_like(g)


_ste_consume.defvjp(_ste_consume_fwd, _ste_consume_bwd)


class ExtendQuantizedWithReplicasTest(parameterized.TestCase):
  """CPU tests for `extend_quantized_with_replicas` (dynamic & static FP8, channelwise & per-tensor, STE grads)."""

  def setUp(self):
    """Initializes deterministic 4-shard weights/grads and patches `ragged_all_to_all` on CPU."""
    super().setUp()
    rng = np.random.default_rng(1)
    self.plan = _hand_plan()
    # Home 0 (the sender) holds 8x larger weights than its receivers 1 and 3.
    mag = np.array([8.0, 1.0, 1.0, 1.0], np.float32).reshape((NUM_EP, 1, 1, 1))
    self.w = (rng.standard_normal((NUM_EP, LOCAL, DIM, 4)) * mag).astype(np.float32)
    self.g = rng.standard_normal((NUM_EP, LOCAL + NUM_REPLICAS, DIM, 4)).astype(np.float32)
    patcher = mock.patch.object(jax.lax, "ragged_all_to_all", moe_replica_test_utils.emulated_ragged_all_to_all)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _extend(self, calibration, channelwise_axes=(2,), dtype=jnp.float32):
    def per_device(w):
      return mrw.extend_quantized_with_replicas(
          w, self.plan, AXIS, NUM_EP, dtype, jnp.float8_e4m3fn, calibration, channelwise_axes
      )

    return jax.jit(jax.vmap(per_device, axis_name=AXIS))(jnp.asarray(self.w))

  @parameterized.named_parameters(
      ("absmax_channelwise", "absmax", (2,)),
      ("absmax_pertensor", "absmax", ()),
      ("fixed_channelwise_arg", "fixed,-224,224", (2,)),
      ("fixed_pertensor", "fixed,224", ()),
  )
  def test_home_and_replica_codes_and_scales(self, calibration, channelwise_axes):
    out = self._extend(calibration, channelwise_axes)
    self.assertEqual(out.qvalue.dtype, jnp.float8_e4m3fn)
    self.assertEqual(out.qvalue.shape, (NUM_EP, LOCAL + NUM_REPLICAS, DIM, 4))
    is_static = calibration.startswith("fixed")
    expected_scale_shape = (
        (NUM_EP, 1, 1, 1) if is_static else (NUM_EP, LOCAL + NUM_REPLICAS, 1, 4 if channelwise_axes else 1)
    )
    self.assertEqual(out.scale.shape, expected_scale_shape)

    codes = np.asarray(out.qvalue.astype(jnp.float32))
    scale = np.asarray(out.scale)
    rep_home, rep_slot = np.asarray(self.plan.rep_home), np.asarray(self.plan.rep_slot)
    for d in range(NUM_EP):
      ref = qpl.quantize(
          jnp.asarray(self.w[d]),
          jnp.float8_e4m3fn,
          channelwise_axes=list(channelwise_axes),
          calibration_method=calibration,
      )
      np.testing.assert_array_equal(codes[d, :LOCAL], np.asarray(ref.qvalue.astype(jnp.float32)))
      if is_static:
        np.testing.assert_array_equal(scale[d], np.asarray(ref.scale))
      else:
        ref_scale = np.broadcast_to(np.asarray(ref.scale), (LOCAL, *ref.scale.shape[1:]))
        np.testing.assert_allclose(scale[d, :LOCAL], ref_scale, rtol=1e-6, atol=1e-7)
      for s in range(NUM_REPLICAS):
        h = int(rep_home[d, s])
        if h < 0:
          np.testing.assert_array_equal(codes[d, LOCAL + s], 0.0)
          continue
        home = qpl.quantize(
            jnp.asarray(self.w[h]),
            jnp.float8_e4m3fn,
            channelwise_axes=list(channelwise_axes),
            calibration_method=calibration,
        )
        loc = int(rep_slot[d, s])
        np.testing.assert_array_equal(codes[d, LOCAL + s], np.asarray(home.qvalue.astype(jnp.float32))[loc])
        if not is_static:
          np.testing.assert_allclose(scale[d, LOCAL + s], np.asarray(home.scale)[0], rtol=1e-6, atol=1e-7)

  def test_static_calibration_skips_scale_collective(self):
    """Static fixed calibration shares a constant (1, 1, 1) scale, so only codes use ragged_all_to_all."""
    calls = 0
    orig = moe_replica_test_utils.emulated_ragged_all_to_all

    def counting_ra2a(*args, **kwargs):
      nonlocal calls
      calls += 1
      return orig(*args, **kwargs)

    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=counting_ra2a):
      self._extend("fixed,-224,224", ())
    self.assertEqual(calls, 1)
    calls = 0
    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=counting_ra2a):
      self._extend("absmax", (2,))
    self.assertEqual(calls, 2)

  def test_naive_requantization_would_change_receiver_home_scale(self):
    """Documents why scales move: requantizing [home | replica] on receiver 1 inflates its home scale."""
    dense = jax.jit(
        jax.vmap(lambda w: mrw.extend_with_replicas(w, self.plan, AXIS, NUM_EP, jnp.float32), axis_name=AXIS)
    )(jnp.asarray(self.w))
    naive = qpl.quantize(dense[1], jnp.float8_e4m3fn, channelwise_axes=[2], calibration_method="absmax")
    home = jax.jit(lambda w: qpl.quantize(w, jnp.float8_e4m3fn, channelwise_axes=[2], calibration_method="absmax"))(
        jnp.asarray(self.w[1])
    )
    self.assertGreater(float(np.min(np.asarray(naive.scale) / np.asarray(home.scale))), 2.0)
    np.testing.assert_array_equal(
        np.asarray(self._extend("absmax").scale[1, :LOCAL]), np.broadcast_to(home.scale, (LOCAL, 1, 4))
    )

  @parameterized.named_parameters(
      ("absmax_f32", "absmax", (2,), jnp.float32),
      ("absmax_pertensor_bf16", "absmax", (), jnp.bfloat16),
      ("fixed_bf16", "fixed,-224,224", (), jnp.bfloat16),
  )
  def test_straight_through_grads_return_home_like_dense_move(self, calibration, channelwise_axes, dtype):
    g_in = jnp.asarray(self.g, dtype=dtype)

    def per_device(w, g):
      def loss_q(w_):
        q = mrw.extend_quantized_with_replicas(
            w_, self.plan, AXIS, NUM_EP, dtype, jnp.float8_e4m3fn, calibration, channelwise_axes
        )
        return _ste_consume(q, g)

      def loss_dense(w_):
        extended = mrw.extend_with_replicas(w_, self.plan, AXIS, NUM_EP, dtype)
        return jnp.sum(extended.astype(jnp.float32) * g.astype(jnp.float32))

      return jax.grad(loss_q)(w), jax.grad(loss_dense)(w)

    dw_q, dw_dense = jax.jit(jax.vmap(per_device, axis_name=AXIS))(jnp.asarray(self.w), g_in)
    self.assertEqual(dw_q.dtype, jnp.float32)
    g_ref = np.asarray(g_in, dtype=np.float32)
    _, dw_rep = moe_replica_test_utils.replica_reference(self.w, g_ref[:, LOCAL:], self.plan)
    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-6
    np.testing.assert_allclose(np.asarray(dw_q), g_ref[:, :LOCAL] + dw_rep, rtol=tol, atol=tol)
    np.testing.assert_array_equal(np.asarray(dw_q), np.asarray(dw_dense))

  def test_noop_when_zero_replicas_or_single_device(self):
    w_single = jnp.asarray(self.w[0], dtype=jnp.float32)
    g_single = jnp.asarray(self.g[0, :LOCAL], dtype=jnp.bfloat16)
    zero_rep_plan = types.SimpleNamespace(num_replicas=0, send_slots=jnp.zeros((NUM_EP, 2), jnp.int32))
    with mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=AssertionError("must not call ra2a")):
      out_r0 = mrw.extend_quantized_with_replicas(w_single, zero_rep_plan, AXIS, NUM_EP, jnp.bfloat16, jnp.float8_e4m3fn)
      dw_r0 = jax.grad(
          lambda w: _ste_consume(
              mrw.extend_quantized_with_replicas(w, zero_rep_plan, AXIS, NUM_EP, jnp.bfloat16, jnp.float8_e4m3fn),
              g_single,
          )
      )(w_single)
      out_ep1 = mrw.extend_quantized_with_replicas(w_single, self.plan, AXIS, 1, jnp.bfloat16, jnp.float8_e4m3fn)
      dw_ep1 = jax.grad(
          lambda w: _ste_consume(
              mrw.extend_quantized_with_replicas(w, self.plan, AXIS, 1, jnp.bfloat16, jnp.float8_e4m3fn),
              g_single,
          )
      )(w_single)
    self.assertEqual(out_r0.qvalue.shape, w_single.shape)
    self.assertEqual(out_ep1.qvalue.shape, w_single.shape)
    self.assertEqual(dw_r0.dtype, jnp.float32)
    self.assertEqual(dw_ep1.dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(dw_r0), np.asarray(g_single.astype(jnp.float32)))
    np.testing.assert_array_equal(np.asarray(dw_ep1), np.asarray(g_single.astype(jnp.float32)))

  def test_invalid_inputs_raise(self):
    w_single = jnp.asarray(self.w[0], dtype=jnp.float32)
    with self.assertRaises(ValueError):
      mrw.extend_quantized_with_replicas(
          w_single[0], self.plan, AXIS, NUM_EP, jnp.float32, jnp.float8_e4m3fn, "absmax", (2,)
      )
    with self.assertRaises(ValueError):
      mrw.extend_quantized_with_replicas(
          w_single, self.plan, AXIS, NUM_EP, jnp.float32, jnp.float8_e4m3fn, "absmax", (0, 2)
      )
    with self.assertRaises(NotImplementedError):
      jax.jit(
          jax.vmap(
              lambda w: mrw.extend_quantized_with_replicas(
                  w, self.plan, AXIS, NUM_EP, jnp.float32, jnp.int8, "minmax", ()
              ),
              axis_name=AXIS,
          )
      )(jnp.asarray(self.w))


@pytest.mark.tpu_only
class ExtendQuantizedWithReplicasTpuTest(parameterized.TestCase):
  """Runs `extend_quantized_with_replicas` under `jax.shard_map` with real TPU `ragged_all_to_all`."""

  @parameterized.named_parameters(
      ("absmax_channelwise_bf16", "absmax", (2,), jnp.bfloat16),
      ("absmax_pertensor_f32", "absmax", (), jnp.float32),
      ("fixed_bf16", "fixed,-224,224", (), jnp.bfloat16),
  )
  def test_real_ragged_all_to_all_quantized_forward_and_backward(self, calibration, channelwise_axes, compute_dtype):
    devs = jax.devices()
    if len(devs) < 4:
      self.skipTest(f"_hand_plan requires 4 TPU devices, got {len(devs)}.")
    self.assertIsNot(jax.lax.ragged_all_to_all, moe_replica_test_utils.emulated_ragged_all_to_all)
    devs = devs[:4]
    plan = _hand_plan()
    rng = np.random.default_rng(77)
    mag = np.array([8.0, 1.0, 1.0, 1.0], np.float32).reshape((NUM_EP, 1, 1, 1))
    w_np = (rng.standard_normal((NUM_EP, LOCAL, 16, 128)) * mag).astype(np.float32)
    g_np = rng.standard_normal((NUM_EP, LOCAL + NUM_REPLICAS, 16, 128)).astype(np.float32)
    mesh = Mesh(np.asarray(devs), (AXIS,))
    w_sharded = jax.device_put(jnp.asarray(w_np), jax.sharding.NamedSharding(mesh, P(AXIS)))
    g_sharded = jax.device_put(jnp.asarray(g_np, dtype=compute_dtype), jax.sharding.NamedSharding(mesh, P(AXIS)))

    @jax.jit
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(AXIS), P(AXIS)), out_specs=(P(AXIS), P(AXIS)))
    def step(w_shard, g_shard):
      def loss_q(w_):
        q = mrw.extend_quantized_with_replicas(
            w_, plan, AXIS, NUM_EP, compute_dtype, jnp.float8_e4m3fn, calibration, channelwise_axes
        )
        return _ste_consume(q, g_shard[0]), q

      (_, q_out), dw_local = jax.value_and_grad(loss_q, has_aux=True)(w_shard[0])
      return jax.tree.map(lambda a: a[None], q_out), dw_local[None]

    @jax.jit
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS))
    def quant_r0(w_shard):
      ref = qpl.quantize(
          w_shard[0].astype(compute_dtype),
          jnp.float8_e4m3fn,
          channelwise_axes=list(channelwise_axes),
          calibration_method=calibration,
      )
      return jax.tree.map(lambda a: a[None], ref)

    ref_r0 = quant_r0(w_sharded)
    ref_codes = np.asarray(ref_r0.qvalue.astype(jnp.float32))
    ref_scale = np.asarray(ref_r0.scale, dtype=np.float32)

    q_out, dw = step(w_sharded, g_sharded)
    codes = np.asarray(q_out.qvalue.astype(jnp.float32))
    scale = np.asarray(q_out.scale, dtype=np.float32)
    is_static = calibration.startswith("fixed")
    rep_home, rep_slot = np.asarray(plan.rep_home), np.asarray(plan.rep_slot)
    for d in range(NUM_EP):
      np.testing.assert_array_equal(codes[d, :LOCAL], ref_codes[d])
      if is_static:
        np.testing.assert_array_equal(scale[d], ref_scale[d])
      else:
        np.testing.assert_array_equal(
            scale[d, :LOCAL],
            np.broadcast_to(ref_scale[d], (LOCAL, *ref_scale.shape[2:])),
        )
      for s in range(NUM_REPLICAS):
        h = int(rep_home[d, s])
        if h >= 0:
          np.testing.assert_array_equal(codes[d, LOCAL + s], ref_codes[h, int(rep_slot[d, s])])
          if not is_static:
            np.testing.assert_array_equal(scale[d, LOCAL + s], ref_scale[h, 0])

    g_ref = np.asarray(jnp.asarray(g_np, dtype=compute_dtype), dtype=np.float32)
    _, dw_rep = moe_replica_test_utils.replica_reference(w_np, g_ref[:, LOCAL:], plan)
    tol = 1e-2 if compute_dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(np.asarray(dw), g_ref[:, :LOCAL] + dw_rep, rtol=tol, atol=tol)


if __name__ == "__main__":
  absltest.main()
