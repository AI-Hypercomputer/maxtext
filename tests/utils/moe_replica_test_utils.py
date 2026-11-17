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

"""Shared CPU collective emulation and NumPy reference helpers for MoE expert-replication tests."""

import jax
import jax.numpy as jnp
import numpy as np


def assert_matching_recv_sizes(recv_sz, expected_sz, me):
  """Raises `ValueError` if a receiver's `recv_sizes` disagrees with incoming `send_sizes`."""
  recv_np = np.asarray(recv_sz)
  exp_np = np.asarray(expected_sz)
  if not np.array_equal(recv_np, exp_np):
    raise ValueError(f"ragged_all_to_all recv_sizes {recv_np} != incoming send_sizes {exp_np} on rank {int(me)}")


def emulated_ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name
):  # pylint: disable=too-many-positional-arguments
  """Emulates `jax.lax.ragged_all_to_all` across a vmapped `axis_name` on CPU, honoring and checking `recv_sizes`."""
  ep = input_offsets.shape[0]
  me = jax.lax.axis_index(axis_name)
  all_op = jax.lax.all_gather(operand, axis_name=axis_name)
  all_in = jax.lax.all_gather(input_offsets, axis_name=axis_name)
  all_sz = jax.lax.all_gather(send_sizes, axis_name=axis_name)
  all_out = jax.lax.all_gather(output_offsets, axis_name=axis_name)
  jax.debug.callback(assert_matching_recv_sizes, recv_sizes, all_sz[:, me], me, ordered=False)
  rows = jnp.arange(output.shape[0])
  buf = output
  for src in range(ep):
    off_in = all_in[src, me]
    sz = recv_sizes[src]
    off_out = all_out[src, me]
    idx = jnp.clip(rows - off_out + off_in, 0, operand.shape[0] - 1)
    mask = ((rows >= off_out) & (rows < off_out + sz)).reshape((output.shape[0],) + (1,) * (output.ndim - 1))
    buf = jnp.where(mask, all_op[src, idx], buf)
  return buf


def replica_reference(w, g_rep, plan):
  """NumPy reference returning `(reps, dw_rep)` for `w` `[EP, L, ...]` and `g_rep` `[EP, R, ...]`."""
  rep_home = np.asarray(plan.rep_home)
  rep_slot = np.asarray(plan.rep_slot)
  num_ep, num_replicas = rep_home.shape
  reps = np.zeros((num_ep, num_replicas) + w.shape[2:], dtype=w.dtype)
  dw_rep = np.zeros_like(w)
  for r in range(num_ep):
    for s in range(num_replicas):
      h = int(rep_home[r, s])
      if h >= 0:
        loc = int(rep_slot[r, s])
        reps[r, s] = w[h, loc]
        dw_rep[h, loc] += g_rep[r, s]
  return reps, dw_rep
