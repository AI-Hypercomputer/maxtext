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


if __name__ == "__main__":
  absltest.main()
