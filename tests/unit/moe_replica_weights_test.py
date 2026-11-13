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

"""CPU tests for the hot-expert replica weight moves.

XLA:CPU does not implement ragged all-to-all, so these tests run the ops under `jax.vmap` with an
EP axis name and replace `jax.lax.ragged_all_to_all` by an all-gather based emulation with the same
semantics. They check the forward values and the custom-VJP gradients against a numpy reference on
a hand-built replica plan.
"""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import moe_replica_weights as mrw

NUM_EP = 4
LOCAL = 4
NUM_REPLICAS = 2
MAX_SEND = 2
DIM = 3
AXIS = "ep"


def _hand_plan():
  """4 devices, 4 home experts and 2 replica slots each.

  Device 1 hosts replicas of (home 0, slot 3) and (home 2, slot 0); device 2 hosts (home 0, slot 1);
  device 3 hosts (home 1, slot 2). Each home's send buffer is ordered by receiver.
  """
  return types.SimpleNamespace(
      rep_home=jnp.array([[-1, -1], [0, 2], [0, -1], [1, -1]], jnp.int32),
      rep_slot=jnp.array([[0, 0], [3, 0], [1, 0], [2, 0]], jnp.int32),
      send_slots=jnp.array([[3, 1], [2, 0], [0, 0], [0, 0]], jnp.int32),
      send_peer_index=jnp.array([[-1, 0, 1, -1], [-1, -1, -1, 0], [-1, 0, -1, -1], [-1, -1, -1, -1]], jnp.int32),
      num_slots=LOCAL + NUM_REPLICAS,
      num_replicas=NUM_REPLICAS,
  )


def _emulated_ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
  """ragged_all_to_all semantics via all_gather, usable under vmap (no CPU kernel needed)."""
  del recv_sizes
  me = jax.lax.axis_index(axis_name)
  ops = jax.lax.all_gather(operand, axis_name)  # [EP(sender), N, ...]
  in_off = jax.lax.all_gather(input_offsets, axis_name)[:, me]  # what each sender sends to me
  size = jax.lax.all_gather(send_sizes, axis_name)[:, me]
  out_off = jax.lax.all_gather(output_offsets, axis_name)[:, me]
  rows = jnp.arange(output.shape[0])
  out = output
  for i in range(ops.shape[0]):
    hit = (rows >= out_off[i]) & (rows < out_off[i] + size[i])
    src = jnp.clip(in_off[i] + rows - out_off[i], 0, operand.shape[0] - 1)
    out = jnp.where(hit.reshape((-1,) + (1,) * (out.ndim - 1)), ops[i][src], out)
  return out


def _reference(w, g_rep, plan):
  """Replica rows [EP, R, D] and their gradient contribution to the home weights [EP, L, D]."""
  rep_home, rep_slot = np.asarray(plan.rep_home), np.asarray(plan.rep_slot)
  reps = np.zeros((NUM_EP, plan.num_replicas, DIM), np.float32)
  dw = np.zeros_like(w)
  for r in range(NUM_EP):
    for s in range(plan.num_replicas):
      if rep_home[r, s] >= 0:
        reps[r, s] = w[rep_home[r, s], rep_slot[r, s]]
        dw[rep_home[r, s], rep_slot[r, s]] += g_rep[r, s]
  return reps, dw


class MoveParamsTest(absltest.TestCase):

  def test_send_and_recv_sizes_match(self):
    plan = _hand_plan()
    params = [mrw._move_params(plan, me, NUM_EP) for me in range(NUM_EP)]  # pylint: disable=protected-access
    for i in range(NUM_EP):
      for j in range(NUM_EP):
        self.assertEqual(int(params[i][1][j]), int(params[j][3][i]))  # send_sizes[i->j] == recv_sizes[j<-i]
    self.assertEqual(sum(int(p[1].sum()) for p in params), int((np.asarray(plan.rep_home) >= 0).sum()))

  def test_direct_offsets_index_the_weights(self):
    plan = _hand_plan()
    for me in range(NUM_EP):
      gather = mrw._move_params(plan, me, NUM_EP)  # pylint: disable=protected-access
      direct = mrw._direct_move_params(plan, me, NUM_EP)  # pylint: disable=protected-access
      for a, b in zip(gather[1:], direct[1:]):
        np.testing.assert_array_equal(a, b)
      # Direct input offset = the home-local slot that the gather path puts at that send-buffer position.
      send = np.asarray(gather[1]) > 0
      np.testing.assert_array_equal(
          np.asarray(direct[0])[send], np.asarray(plan.send_slots[me])[np.asarray(gather[0])[send]]
      )


class ReplicaWeightOpsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.plan = _hand_plan()
    self.w = rng.standard_normal((NUM_EP, LOCAL, DIM)).astype(np.float32)
    self.g = rng.standard_normal((NUM_EP, LOCAL + NUM_REPLICAS, DIM)).astype(np.float32)
    patcher = mock.patch.object(jax.lax, "ragged_all_to_all", _emulated_ragged_all_to_all)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _vjp(self, fn, cotangent):
    """Per-device forward and VJP, both inside the vmapped EP axis (the backward pass uses collectives too)."""

    def per_device(w, g):
      out, vjp = jax.vjp(fn, w)
      return out, vjp(g)[0]

    return jax.jit(jax.vmap(per_device, axis_name=AXIS))(jnp.asarray(self.w), cotangent)

  @parameterized.parameters("gather", "direct")
  def test_extend_with_replicas(self, move):
    extend = mrw.extend_with_replicas_direct if move == "direct" else mrw.extend_with_replicas
    out, dw = self._vjp(lambda w: extend(w, self.plan, AXIS, NUM_EP, jnp.float32), jnp.asarray(self.g))
    reps, dw_rep = _reference(self.w, self.g[:, LOCAL:], self.plan)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([self.w, reps], axis=1))
    np.testing.assert_allclose(np.asarray(dw), self.g[:, :LOCAL] + dw_rep, rtol=1e-6, atol=1e-6)

  def test_home_and_replicas(self):
    g_home, g_rep = self.g[:, :LOCAL], self.g[:, LOCAL:]
    (home, rep), dw = self._vjp(
        lambda w: mrw.home_and_replicas(w, self.plan, AXIS, NUM_EP, jnp.float32),
        (jnp.asarray(g_home), jnp.asarray(g_rep)),
    )
    reps, dw_rep = _reference(self.w, g_rep, self.plan)
    np.testing.assert_array_equal(np.asarray(home), self.w)
    np.testing.assert_array_equal(np.asarray(rep), reps)
    np.testing.assert_allclose(np.asarray(dw), g_home + dw_rep, rtol=1e-6, atol=1e-6)

  def test_home_and_replicas_slot0_with_unused_send_slots(self):
    """Regression test: home 0 sends local slot 0 while max_send=4 pads unused send_slots with 0s."""
    plan = types.SimpleNamespace(
        rep_home=jnp.array([[-1, -1], [0, -1], [-1, -1], [-1, -1]], jnp.int32),
        rep_slot=jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]], jnp.int32),
        # Home 0 sends only 1 replica (slot 0 -> peer 1), with 3 unused send slots padded with 0.
        send_slots=jnp.zeros((NUM_EP, 4), jnp.int32),
        send_peer_index=jnp.array([[-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]], jnp.int32),
        num_slots=LOCAL + NUM_REPLICAS,
        num_replicas=NUM_REPLICAS,
    )
    g_home = np.full((NUM_EP, LOCAL, DIM), 2.0, dtype=np.float32)
    g_rep = np.zeros((NUM_EP, NUM_REPLICAS, DIM), dtype=np.float32)
    g_rep[1, 0] = np.array([10.0, -5.0, 3.5], dtype=np.float32)

    (home, rep), dw = self._vjp(
        lambda w: mrw.home_and_replicas(w, plan, AXIS, NUM_EP, jnp.float32),
        (jnp.asarray(g_home), jnp.asarray(g_rep)),
    )
    np.testing.assert_array_equal(np.asarray(home), self.w)
    np.testing.assert_array_equal(np.asarray(rep)[1, 0], self.w[0, 0])
    # Slot 0 on home 0 must equal home_grad[0] + replica_grad (never clobbered by padded send_slots[1:] == 0).
    np.testing.assert_allclose(np.asarray(dw)[0, 0], g_home[0, 0] + g_rep[1, 0], rtol=1e-6, atol=1e-6)
    reps, dw_rep = _reference(self.w, g_rep, plan)
    np.testing.assert_array_equal(np.asarray(rep), reps)
    np.testing.assert_allclose(np.asarray(dw), g_home + dw_rep, rtol=1e-6, atol=1e-6)

  def test_move_replicas(self):
    g_rep = self.g[:, LOCAL:]
    out, dw = self._vjp(lambda w: mrw.move_replicas(w, self.plan, AXIS, NUM_EP), jnp.asarray(g_rep))
    reps, dw_rep = _reference(self.w, g_rep, self.plan)
    np.testing.assert_array_equal(np.asarray(out), reps)
    np.testing.assert_allclose(np.asarray(dw), dw_rep, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
