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

"""Bidirectional store-and-forward ring all-gather (TensorCore Pallas, ICI DMA).

Validated on v7x: tiled semantics bit-exact vs ``jax.lax.all_gather(..., tiled=True)``. Written for
IN-shard_map use: `ring_all_gather` runs the pallas_call directly on the local shard
inside an enclosing shard_map, with a caller-supplied `collective_id` (a fixed id would collide
with other in-flight Pallas collectives' barrier semaphores) and an explicit `CostEstimate` so the latency-hiding scheduler can see the ~2x-bytes
DMA cost instead of treating the custom-call as free (kernels without a cost estimate are
invisible to the scheduler's overlap decisions).

Why a RING (not the direct-to-owner broadcast `_direct_all_gather` in moe.py): the direct pattern
sends every shard's block over multi-hop paths to all peers (bisection congestion, measured
regressing at EP=8); the ring moves each block over neighbor links only, at the validated
159-179 GB/s.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

MESH = pl.DeviceIdType.MESH


def _strides(sizes):
  """Tiled chunk strides for axes ordered outermost..innermost. strides[k]=prod(sizes[k+1:])."""
  st = [1] * len(sizes)
  for i in range(len(sizes) - 2, -1, -1):
    st[i] = st[i + 1] * sizes[i + 1]
  return st


def _full_index(ref, gather_dim, start, length, split_dim=None, s_start=0, s_len=None):
  """Index tuple: whole array, a [start:start+length] window on gather_dim, and (for bidi) a
  [s_start:s_start+s_len] window on split_dim. The split is on a NON-gather dim so the gather_dim
  offsets stay tile-aligned (splitting the gather/concat dim breaks Mosaic tile alignment)."""
  idx = [pl.ds(0, ref.shape[d]) for d in range(ref.ndim)]
  idx[gather_dim] = pl.ds(start, length)
  if split_dim is not None:
    idx[split_dim] = pl.ds(s_start, s_len)
  return tuple(idx)


def _pick_split_dim(shape, gather_dim):
  """First non-gather dim with an even extent (to halve for bidirectional). None -> force uni."""
  for d in range(len(shape)):
    if d != gather_dim and shape[d] % 2 == 0:
      return d
  return None


def _neighbor(all_axes, axis, delta):
  """device_id MESH dict: step `delta` along `axis` (wrap), hold every other mesh axis fixed."""
  size = lax.axis_size(axis)
  nxt = lax.rem(lax.axis_index(axis) + delta + size, size)
  return {a: (nxt if a == axis else lax.axis_index(a)) for a in all_axes}


def _ring_stage(o_ref, all_axes, axes, sizes, strides, k, chunk, gather_dim,
                send_sem, recv_sem, bidi, split_dim, pipe):
  """One store-and-forward ring over axes[k]; fills sizes[k] blocks of cur_len rows.

  Bidirectional splits each block along split_dim (a NON-gather dim): +dir carries the upper half,
  -dir the lower half, each over the FULL gather block -> gather_dim offsets stay tile-aligned.

  Pipelining: each direction's half is further sliced into `pipe` independent sub-chunks along
  split_dim, each with its OWN sem. Store-and-forward forces depth-1 PER sub-chunk (can't forward
  what hasn't arrived), but the 2*pipe sub-chunks progress independently and are issued
  round-robin, so at steady state 2*pipe DMAs are in flight. pipe=1 is the baseline."""
  s = sizes[k]
  if s == 1:
    return
  cur_rows = strides[k] * chunk           # gather-dim block length in rows
  # Base row of this axis-k group = (sum_{j<k} coord_j * strides[j]) * chunk (shared across the ring).
  base_rows = (sum(lax.axis_index(axes[j]) * strides[j] for j in range(k)) if k > 0 else 0) * chunk
  my = lax.axis_index(axes[k])
  use_bidi = bidi and split_dim is not None

  right_n = _neighbor(all_axes, axes[k], +1)
  left_n = _neighbor(all_axes, axes[k], -1)
  bsem = pltpu.get_barrier_semaphore()
  pl.semaphore_signal(bsem, inc=1, device_id=right_n, device_id_type=MESH)
  if use_bidi:
    pl.semaphore_signal(bsem, inc=1, device_id=left_n, device_id_type=MESH)
  pl.semaphore_wait(bsem, 2 if use_bidi else 1)

  # Pipeline "lanes": (sem_idx, neighbor, hop_sign, split_start, split_len). hop_sign=-1 forwards
  # source-coord (my-i) to the +1 neighbor; +1 forwards (my+i) to the -1 neighbor.
  lanes = []
  if split_dim is not None:
    E = o_ref.shape[split_dim]
    if use_bidi:
      half = E // 2
      assert half % pipe == 0, (E, pipe)
      w = half // pipe
      for j in range(pipe):
        lanes.append((j,        right_n, -1, half + j * w, w))   # +dir, upper half
        lanes.append((pipe + j, left_n,  +1, j * w,        w))   # -dir, lower half
    else:
      assert E % pipe == 0, (E, pipe)
      w = E // pipe
      for j in range(pipe):
        lanes.append((j, right_n, -1, j * w, w))                 # uni, full split sliced into pipe
  else:
    lanes.append((0, right_n, -1, None, None))                   # uni, no split_dim: single block

  def idx(start, ss, sl):
    if ss is None:
      return _full_index(o_ref, gather_dim, start, cur_rows)
    return _full_index(o_ref, gather_dim, start, cur_rows, split_dim=split_dim, s_start=ss, s_len=sl)

  def block_start(u):
    return base_rows + lax.rem(u + s, s) * cur_rows

  prev = [None] * len(lanes)
  for i in range(s - 1):
    for li, (si, nb, sign, ss, sl) in enumerate(lanes):
      start = block_start(my - i if sign < 0 else my + i)
      if prev[li] is not None:
        prev[li].wait()
      prev[li] = pltpu.async_remote_copy(
          o_ref.at[idx(start, ss, sl)], o_ref.at[idx(start, ss, sl)],
          send_sem.at[si], recv_sem.at[si], device_id=nb, device_id_type=MESH)
  for d in prev:
    if d is not None:
      d.wait()


def _kernel(w_ref, o_ref, send_sem, recv_sem, local_sem, *,
            all_axes, axes, sizes, strides, chunk, gather_dim, bidi, split_dim, pipe):
  # 1. place our own shard at its tiled slot: chunk index = sum coord_j * strides[j].
  my_chunk = sum(lax.axis_index(axes[j]) * strides[j] for j in range(len(axes)))
  my_start = my_chunk * chunk
  local_copy = pltpu.make_async_copy(
      w_ref.at[_full_index(w_ref, gather_dim, 0, chunk)],
      o_ref.at[_full_index(o_ref, gather_dim, my_start, chunk)],
      local_sem)
  local_copy.start()
  local_copy.wait()
  # 2. nested rings, innermost axis first (largest k -> smallest stride).
  for k in range(len(axes) - 1, -1, -1):
    _ring_stage(o_ref, all_axes, axes, sizes, strides, k, chunk, gather_dim,
                send_sem, recv_sem, bidi, split_dim, pipe)


def ring_all_gather(x, mesh, gather_axes, gather_dim, collective_id, *, bidi=True, pipe=1):
  """In-shard_map ring all-gather of the LOCAL shard `x` over `gather_axes` (mesh axis names,
  outermost..innermost), tiled on `gather_dim`. Semantics ==
  ``jax.lax.all_gather(x, gather_axes, axis=gather_dim, tiled=True)`` (bit-exact, pure data move).

  MUST be called INSIDE a shard_map spanning `mesh` (uses lax.axis_index on the mesh axes for MESH
  device_id addressing). `collective_id` selects the barrier semaphore and must be DISTINCT from
  every other concurrently in-flight Pallas collective's id."""
  if isinstance(gather_axes, str):
    gather_axes = (gather_axes,)
  sizes = tuple(mesh.shape[ax] for ax in gather_axes)
  strides = _strides(sizes)
  n_total = 1
  for s_ in sizes:
    n_total *= s_
  chunk = x.shape[gather_dim]  # local rows on the gather dim
  out_shape = list(x.shape)
  out_shape[gather_dim] = chunk * n_total
  out_shape = tuple(out_shape)
  all_axes = tuple(mesh.axis_names)
  split_dim = _pick_split_dim(out_shape, gather_dim) if bidi else None
  HBM = pltpu.MemorySpace.HBM

  def kern(w_ref, o_ref, send_sem, recv_sem, local_sem):
    _kernel(w_ref, o_ref, send_sem, recv_sem, local_sem,
            all_axes=all_axes, axes=tuple(gather_axes), sizes=sizes, strides=strides,
            chunk=chunk, gather_dim=gather_dim, bidi=bidi, split_dim=split_dim, pipe=pipe)

  nsem = 2 * pipe   # 2 directions x pipe sub-chunks (uni uses the first `pipe`)
  # Cost estimate: each device receives + forwards ~2x the full gathered bytes over the ring.
  # Without it the custom-call is invisible to the latency-hiding scheduler's overlap decisions.
  full_bytes = 1
  for d_ in out_shape:
    full_bytes *= d_
  full_bytes *= x.dtype.itemsize
  return pl.pallas_call(
      kern,
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
      in_specs=[pl.BlockSpec(memory_space=HBM)],
      out_specs=pl.BlockSpec(memory_space=HBM),
      scratch_shapes=[pltpu.SemaphoreType.DMA((nsem,)),   # send, per (dir,sub) lane
                      pltpu.SemaphoreType.DMA((nsem,)),   # recv, per (dir,sub) lane
                      pltpu.SemaphoreType.DMA],            # local copy
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
      cost_estimate=pl.CostEstimate(flops=0, bytes_accessed=2 * full_bytes, transcendentals=0),
  )(x)
