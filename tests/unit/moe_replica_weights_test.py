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

"""CPU tests for `moe_replica_weights.extend_with_replicas`."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import moe_replica_weights as mrw

AXIS = "ep"
NUM_EP = 4
LOCAL = 3
NUM_REPLICAS = 2
DIM = 5


def _emulated_ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
  """Emulates `jax.lax.ragged_all_to_all` across a vmapped `axis_name` on CPU."""
  del recv_sizes
  ep = input_offsets.shape[0]
  me = jax.lax.axis_index(axis_name)
  all_op = jax.lax.all_gather(operand, axis_name=axis_name)
  all_in = jax.lax.all_gather(input_offsets, axis_name=axis_name)
  all_sz = jax.lax.all_gather(send_sizes, axis_name=axis_name)
  all_out = jax.lax.all_gather(output_offsets, axis_name=axis_name)
  rows = jnp.arange(output.shape[0])
  buf = output
  for src in range(ep):
    off_in = all_in[src, me]
    sz = all_sz[src, me]
    off_out = all_out[src, me]
    idx = jnp.clip(rows - off_out + off_in, 0, operand.shape[0] - 1)
    mask = ((rows >= off_out) & (rows < off_out + sz)).reshape((output.shape[0],) + (1,) * (output.ndim - 1))
    buf = jnp.where(mask, all_op[src, idx], buf)
  return buf


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


def _reference(w, g_rep, plan):
  """NumPy reference for replica rows `[EP, R, ...]` and home-grad contributions `[EP, L, ...]`."""
  rep_home = np.asarray(plan.rep_home)
  rep_slot = np.asarray(plan.rep_slot)
  reps = np.zeros((NUM_EP, NUM_REPLICAS) + w.shape[2:], dtype=w.dtype)
  dw_rep = np.zeros_like(w)
  for r in range(NUM_EP):
    for s in range(NUM_REPLICAS):
      h = int(rep_home[r, s])
      if h >= 0:
        loc = int(rep_slot[r, s])
        reps[r, s] = w[h, loc]
        dw_rep[h, loc] += g_rep[r, s]
  return reps, dw_rep


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
    patcher = mock.patch.object(jax.lax, "ragged_all_to_all", _emulated_ragged_all_to_all)
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
    reps, dw_rep = _reference(self.w, self.g[:, LOCAL:], self.plan)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([self.w, reps], axis=1))
    np.testing.assert_allclose(np.asarray(dw), self.g[:, :LOCAL] + dw_rep, rtol=1e-6, atol=1e-6)

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
    reps, dw_rep = _reference(self.w, self.g[:, LOCAL:], plan)
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
    reps, dw_rep = _reference(w_bf16, g_f32[:, LOCAL:], self.plan)
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


if __name__ == "__main__":
  absltest.main()
