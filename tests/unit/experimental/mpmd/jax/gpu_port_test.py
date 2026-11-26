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

"""CPU checks of the GPU port: gmm_impl 'tokamax' on its compact local buffer, the --platform defaults, and what a GPU
pod adds -- its coordinator, its persistent cache, its profile options, cuDNN attention.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/gpu_port_test.py -q

The tests of ten seconds or more -- the Qwen3.8, bf16, four-stage and other variants of what the rest check once, a
minute or more of compiles each -- are marked scheduled_only: MaxText's scheduled CI runs them (`-m 'not
scheduled_only'` leaves them out).  NormReduce runs on several processes, so it also needs
MAXTEXT_MPMD_MULTIHOST_TESTS=1 (testing.require_multihost).  Run them before changing what they hold.

gmm_impl 'tokamax' runs here on tokamax's own lax.ragged_dot backend (tokamax_gmm 'xla'), which XLA:CPU lowers as it
lowers gmm_impl 'ragged_dot': a dense product over the groups, each row masked to its own.  So the pure-JAX path the
CPU suites already hold against the monolith -- gmm_impl 'ragged_dot' with the argsort dispatch (use_ragged_sort
False) -- is its reference, and in fp32 the two are bit for bit one: the compact buffer's rows are the argsort
buffer's local rows in the same order, its masked operands the same, and its gathers in and out of the buffer the
same products in the same order.  That holds at each grouped matmul (TokamaxGmm), the ring of experts
(TokamaxRingMoe), a chunk's programs (TokamaxChunks) and the whole pipeline (TokamaxPipeline) -- DeepSeek-V3 and
Qwen3.8 layers, batch split on and off, split_dw acc and sum, expert_weights stacked and sliced, EP 1 (where the
compact buffer's masks are skipped) -- but where XLA:CPU itself rounds the two programs apart, or the port adds in
another order:

  * bf16: XLA:CPU keeps a bf16 intermediate in fp32 across a fusion where it may (xla_allow_excess_precision, on by
    default), and the two programs fuse differently, so top-k routing's renormalised weights, and what they reach,
    can round apart in their last bits; with --xla_allow_excess_precision=false a bf16 step is bitwise again
    (TokamaxPipeline's slow test, under split_dw 'sum': under 'acc' the halves' routed-expert dW sum may round twice
    in tokamax_gmm -- each half's dW in the weight dtype first, as 'sum' rounds -- where ragged_dot's rounds once).
    Held to a few bf16 ulps of each leaf's largest magnitude: two at the ring of experts, four over a whole step.
  * expert_weights 'stacked' on a chunk of several layers: tokamax's forward reads the layer's block of the stacked
    weight with the other blocks' empty groups (named_gmm's trick, which spares a GPU kernel's custom call a copy of
    the slice), which widens the dense lowering's contraction, and XLA:CPU may block it differently: fp32 rounding.
    One-layer chunks -- a 2-stage pipeline of 4 layers at V=2, the runs the port was first measured on -- and
    'sliced' are bitwise, but for in-place dW (below).
  * in-place dW (--inplace-dw auto, the default, is 'on' off an sm_100 GPU; a ChunkPrograms' inplace_dw holds on a
    chunk of one DeepSeek-V3 layer, or of several under layer_tree 'per_layer', and on a Qwen3.8 chunk): under batch
    split a step's accumulating backward adds the halves' routed-expert dW into the accumulator one at a time,
    (acc + dW_mb1) + dW_mb0, where ragged_dot's adds their sum, acc + (dW_mb1 + dW_mb0): those leaves to 8 fp32 ulps
    of their largest magnitude (TokamaxPipeline.check), every other leaf bit for bit; unsplit, or with --inplace-dw
    off, every leaf bitwise.

--layer-tree per_layer, a GPU default, trains as the stacked layout does on the same weights (PerLayerTree): every
gradient leaf bit for bit on ragged_dot, the routed ones under tokamax's in-place dW -- which per_layer turns on for
a chunk of several DeepSeek-V3 layers -- as above; two AdamW steps' losses and norms to 1e-6.

--save-residuals on (SaveResiduals: a chunk's forward returns its layers' VJP residuals, its backward evaluates the
VJP instead of recomputing the chunk) gives the recompute programs' step -- DeepSeek-V3 layers in per-layer trees on
tokamax's kernels, batch split on and off, EP 1 with a dense layer, cp 2 -- to 1e-5: the loss, the stage sumsqs to 2e-5,
every gradient leaf to 1e-5 of its largest magnitude; its lean residual set holds no fp32 RMSNorm activation and no
gated product's intermediates.  An fp32 accumulator taking the routed experts' dW in place (--inplace-dw all:
tokamax_gmm's dw32), saved or recomputed, gives the separate add's gradients to 4 bf16 ulps of each leaf's largest
magnitude, and --buffer-masks off changes no bit, with the grouped matmuls leaving NaN in the rows past the groups
as a GPU kernel leaves them unwritten (unwritten_rows).

Everything else needs no kernel: ds_ep._local_plan against a numpy statement of the compact buffer and of MaxText's
truncated buffer, the dispatch / combine custom VJPs against autodiff of the naive gathers, and under combine_impl
'triton' (off GPU gpu_kernels.gather_sum's XLA sum, itself against numpy) against those (LocalPlan), the capped
buffer against ragged_buffer_test's numpy MoE (TokamaxRaggedBuffer), StageConfig's refusals (TokamaxValidation),
the --platform defaults, --inplace-dw auto and build()'s wiring (PlatformDefaults), init_distributed's coordinator
on a GPU pod (GpuCoordinator), cache_every_process (CacheEveryProcess), profiling.profile_options on GPU
(GpuProfileOptions), transfer's option-aware platforms (TransferPlatforms), cudnn_attention's refusals and call
(CudnnAttention; its numbers only on a GPU backend), gpu.xla_flags against this jaxlib's XLA (GpuXlaFlags),
--norm-reduce allreduce against the hops on jax.distributed CPU processes (NormReduce) and its program on a stand-in
pipeline (NormReduceProgram), what lean residuals keep of attention, RoPE and the RMSNorms (LeanResiduals), and
--offload-residuals's round trip through pinned host memory (OffloadResiduals).  The interleaved schedule's F&B
pairs under the GPU defaults are its unpaired order's bit for bit (InterleavedPairs); per-layer dense layers train as
the stacked ones do, bit for bit (DenseLayerTree); the gathered embedding's gradient scattered into the accumulator
is autodiff's table's where no token repeats, and in a bf16 accumulator, which takes a token's rows summed in fp32,
where tokens repeat too (fp32 compute; EmbedRows).  The tokamax classes skip where tokamax does not import.
"""

import argparse
import collections
import contextlib
import dataclasses
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax.deepseek import attention
from maxtext.experimental.mpmd.jax.deepseek import layers as dl
from maxtext.experimental.mpmd.jax.deepseek import rope
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_parity as par
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from tests.unit.experimental.mpmd.jax import multihost_test  # (the module: its TestCase stays its own)
from maxtext.experimental.mpmd.jax import profiling
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer
# Functions and constants only: a TestCase imported here would be collected, and run, a second time.
from tests.unit.experimental.mpmd.jax.ds_ep_test import (
    HAVE_TOKAMAX,
    TILE,
    TINY_ARGS,
    _count_eqns,
    _eqn_out_shapes,
    _sys_modules,
    assert_same_run,
    mode_run,
    run_pipe,
    stage_seqs,
    tiny_xla,
)
from tests.unit.experimental.mpmd.jax.ds_ep_qwen_test import Q_TINY, Q_TINY_ARGS, qtiny_xla
from tests.unit.experimental.mpmd.jax.ragged_buffer_test import (
    DROPPING,
    MESH_SPECS as RB_SPECS,
    T as RB_T,
    inputs as rb_inputs,
    numpy_moe,
    ring_moe as rb_ring_moe,
    routing as rb_routing,
    scfg as rb_scfg,
)
from maxtext.experimental.mpmd.jax.qwen35.config import Qwen35Config

needs_tokamax = unittest.skipUnless(HAVE_TOKAMAX, "needs tokamax (gmm_impl 'tokamax' runs its ragged_dot_general)")
if HAVE_TOKAMAX:
  # tokamax before 0.0.15 parses sys.argv with absl at its first config read (every ragged_dot_general call reads
  # one): under pytest, pytest's own options (-q, -p, -v -o ...), which absl refuses, so each test reaching a tokamax
  # kernel failed on a flag error.  0.0.15 on parses its --tokamax_* flags only.  Marked parsed here, before any test
  # runs, absl's flags keep their defaults (TokamaxGmm.test_pytest_argv).
  from absl import flags as absl_flags

  if not absl_flags.FLAGS.is_parsed():
    absl_flags.FLAGS.mark_as_parsed()


# gmm_impl 'tokamax' on tokamax's lax.ragged_dot backend, and its reference: the pure-JAX experts on the argsort
# dispatch's buffer.
TOKAMAX = dict(gmm_impl="tokamax", tokamax_gmm="xla", use_ragged_sort=False)
RAGGED = dict(gmm_impl="ragged_dot", use_ragged_sort=False)
BF16_EPS = 2.0**-7  # bf16's machine epsilon: one ulp of x is at most BF16_EPS x |x|
FP32_EPS = 2.0**-23  # fp32's


def leaves(tree):
  """The leaves of a tree as fp32 host arrays."""
  return [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.tree.leaves(tree)]


def assert_near(t, got, want, ulps, what, eps=BF16_EPS):
  """Every leaf within `ulps` bf16 ulps of its largest magnitude (`ulps` x BF16_EPS x max |leaf|; `eps`: another
  dtype's): a weight gradient sums many rounded terms, so its error scales with the leaf, not the element."""
  for i, (a, b) in enumerate(zip(leaves(got), leaves(want), strict=True)):
    err, top = float(np.max(np.abs(a - b), initial=0.0)), float(np.max(np.abs(b), initial=0.0))
    t.assertLessEqual(err, ulps * eps * top, f"{what} leaf {i}: {err:.3g} of {top:.3g}")


def residual_shapes(f, *args):
  """The shape of each array jax.vjp(f, *args) keeps for its backward, as a traced program (ChunkPrograms'
  saved-residual forward) keeps them: the VJP's leaves, its primal inputs among them."""
  return [tuple(a.shape) for a in jax.eval_shape(lambda *a: jax.tree_util.tree_leaves(jax.vjp(f, *a)[1]), *args)]


def assert_equal_leaves(t, got, want, what):
  for i, (a, b) in enumerate(zip(leaves(got), leaves(want), strict=True)):
    np.testing.assert_array_equal(a, b, err_msg=f"{what} leaf {i}")


# ------------------------------------------------------------------ the compact buffer
def numpy_plan(idx, shard, n_local, rows):
  """The compact buffer from first principles: this shard's (token, choice) pairs -- expert e in [shard x n_local,
  (shard + 1) x n_local) -- ordered by local expert, then by flat index t x K + k (token order, then choice order),
  the first R of them kept, R = `rows` (None: T x K) capped at T x K.  -> (the kept pairs' flat indices in row
  order, the local groups' sizes in the buffer, each pair's row (R for a pair not in the buffer), R)."""
  T, K = idx.shape
  R = T * K if rows is None else min(rows, T * K)
  le = idx.reshape(-1).astype(np.int64) - shard * n_local
  local = [j for j in range(T * K) if 0 <= le[j] < n_local]
  kept = np.asarray(sorted(local, key=lambda j: (le[j], j))[:R], np.int64)
  gs = np.bincount(le[kept], minlength=n_local) if kept.size else np.zeros(n_local, np.int64)
  pos = np.full(T * K, R, np.int64)
  pos[kept] = np.arange(kept.size)
  return kept, gs, pos, R


def maxtext_kept(idx, shard, n_local, rows):
  """MaxText's truncated buffer (ragged_sort.ring_ragged_sort's packed path, `_truncate_matrix` of the shard's group
  sizes), as ragged_buffer_test.numpy_moe states it: every (token, choice) pair stably sorted by GLOBAL expert,
  and of the shard's contiguous range the first `rows` kept (all of it for None).  -> [T x K] bool."""
  flat = idx.reshape(-1).astype(np.int64)
  order = np.argsort(flat, kind="stable")
  rank = np.empty_like(order)
  rank[order] = np.arange(order.size)
  counts = np.bincount(flat, minlength=int(flat.max()) + 1)
  start = int(counts[: shard * n_local].sum())
  mine = (flat >= shard * n_local) & (flat < (shard + 1) * n_local)
  return mine & ((rank - start < rows) if rows is not None else True)


def routings(seed):
  """(name, idx [T, K], E) cases: top-k-like (distinct experts a token), random with replacement (a token may take
  one expert twice, as MaxText's random routing draws), and skewed onto shard 0's experts (so a cap drops there)."""
  r = np.random.default_rng(seed)
  T, K, E = 48, 3, 12
  yield "distinct", np.stack([r.permutation(E)[:K] for _ in range(T)]).astype(np.int32), E
  yield "replacement", r.integers(0, E, (T, K)).astype(np.int32), E
  yield "skewed", np.where(r.random((T, K)) < 0.7, r.integers(0, E // 4, (T, K)), r.integers(0, E, (T, K))).astype(
      np.int32
  ), E


def assert_same(a, b, rtol=1e-5, atol=1e-5):
  """Two computations of one fp32 result: bit for bit on CPU, where both are XLA's and sum in one order; on any other
  backend (a GPU, or a TPU where JAX_PLATFORMS names it instead of cpu) to fp32 summation-order rounding
  (sums of up to 16 unit-scale terms here) -- on a GPU gpu_kernels' Triton kernels and XLA's fusions, autotuned per
  process, each sum in their own order."""
  if jax.default_backend() == "cpu":
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  else:
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


class LocalPlan(unittest.TestCase):
  """ds_ep._local_plan, the compact buffer gmm_impl 'tokamax' runs the ring's experts on, and the custom VJPs that
  move rows in and out of it -- no kernel, no tokamax."""

  def test_against_numpy(self):
    """For random routings, each shard of EP 4 and buffers dropless, capped below the shard's load, exactly at it and
    past T x K: the buffer holds exactly this shard's pairs, by local expert and stable in token order (`order`, and
    `src` its tokens), the group sizes are the shard's clamped to the buffer (truncate_group_sizes), `valid` marks
    the rows inside the groups, `pos` sends each kept pair to its row (its inverse on the kept rows) and every other
    pair one past the buffer -- and a cap drops exactly the pairs MaxText's truncated buffer drops.  Every row's
    `src` and `order` stay in bounds (the gathers past the groups read real rows, which `valid` then zeroes).
    Under jit with the shard traced, as moe_experts' lax.axis_index gives it."""
    plan = jax.jit(ds_ep._local_plan, static_argnums=(2, 3))
    for seed in range(2):
      for name, idx, E in routings(seed):
        T, K = idx.shape
        ep = 4
        n_local = E // ep
        for shard in range(ep):
          load = int(np.sum((idx >= shard * n_local) & (idx < (shard + 1) * n_local)))
          for rows in (None, max(load - 5, 1), load, T * K + 7):
            with self.subTest(seed=seed, routing=name, shard=shard, rows=rows):
              kept, gs, pos, R = numpy_plan(idx, shard, n_local, rows)
              src_, gs_, valid_, pos_, order_ = map(np.asarray, plan(jnp.asarray(idx), jnp.int32(shard), n_local, rows))
              n = kept.size
              self.assertEqual(
                  (src_.shape, gs_.shape, valid_.shape, pos_.shape, order_.shape),
                  ((R,), (n_local,), (R, 1), (T * K,), (R,)),
              )
              self.assertEqual([a.dtype for a in (src_, gs_, pos_, order_)], [np.int32] * 4)
              np.testing.assert_array_equal(order_[:n], kept)
              np.testing.assert_array_equal(src_[:n], kept // K)
              np.testing.assert_array_equal(gs_, gs)
              np.testing.assert_array_equal(valid_[:, 0], np.arange(R) < n)
              np.testing.assert_array_equal(pos_, pos)
              np.testing.assert_array_equal(pos_[order_[:n]], np.arange(n))
              counts = jnp.asarray([np.sum(idx == shard * n_local + e) for e in range(n_local)], jnp.int32)
              np.testing.assert_array_equal(gs_, np.asarray(ds_ep.truncate_group_sizes(counts, R)))
              np.testing.assert_array_equal(pos_ < R, maxtext_kept(idx, shard, n_local, rows))
              self.assertTrue(np.all((src_ >= 0) & (src_ < T)) and np.all((order_ >= 0) & (order_ < T * K)))
              if rows is not None and rows < load:
                self.assertEqual(n, rows)  # the cap binds: the shard's last rows are dropped

  def test_gather_sum(self):
    """gpu_kernels.gather_sum (off GPU: its XLA sum) is sum_k w[t, k] y[pos[t, k]] in fp32, a pos of len(y) adding
    nothing, weighted or not (to 1e-6 of a float64 sum), and with bf16 rows a bf16 result: the fp32 sum rounded once,
    bit for bit."""
    from maxtext.experimental.mpmd.jax import gpu_kernels

    r = np.random.default_rng(3)
    T, K, R, D = 9, 3, 7, 5
    y = r.standard_normal((R, D)).astype(np.float32)
    pos = r.integers(0, R + 1, (T, K)).astype(np.int32)  # R: no row
    w = r.standard_normal((T, K)).astype(np.float32)
    want_w = np.zeros((T, D))
    want_1 = np.zeros((T, D))
    for t in range(T):
      for k in range(K):
        if pos[t, k] < R:
          want_w[t] += w[t, k] * y[pos[t, k]]
          want_1[t] += y[pos[t, k]]
    np.testing.assert_allclose(
        np.asarray(gpu_kernels.gather_sum(jnp.asarray(y), jnp.asarray(pos), jnp.asarray(w))), want_w, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gpu_kernels.gather_sum(jnp.asarray(y), jnp.asarray(pos))), want_1, rtol=1e-6, atol=1e-6
    )
    got = gpu_kernels.gather_sum(jnp.asarray(y, jnp.bfloat16), jnp.asarray(pos), jnp.asarray(w))
    self.assertEqual(got.dtype, jnp.bfloat16)
    want16 = np.einsum(
        "tkd,tk->td",
        np.concatenate([np.asarray(jnp.asarray(y, jnp.bfloat16), np.float32), np.zeros((1, D), np.float32)])[pos],
        w,
    )
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(jnp.asarray(want16, jnp.float32).astype(jnp.bfloat16), np.float32)
    )

  def test_take_rows(self):
    """A row index one past the array reads zeros (the fill), not a clamped last row."""
    a = jnp.arange(12.0).reshape(4, 3) + 1
    got = np.asarray(ds_ep._take_rows(a, jnp.array([3, 4, 0, 4], jnp.int32)))
    np.testing.assert_array_equal(got, [[10, 11, 12], [0, 0, 0], [1, 2, 3], [0, 0, 0]])

  def _plan(self, K, rows, seed=0, E=8, ep=2, T=40, shard=1):
    r = np.random.default_rng(seed)
    idx = r.integers(0, E, (T, K)).astype(np.int32)  # with replacement: a token may repeat an expert
    return idx, ds_ep._local_plan(jnp.asarray(idx), jnp.int32(shard), E // ep, rows)

  def test_dispatch_rows_vjp(self):
    """_dispatch_rows forward is the naive masked gather; its backward -- each token's rows' gradients gathered
    through `pos` and summed over the token's choices in fp32, rounded once -- is autodiff's scatter-add of the naive
    gather (bit for bit at K = 2 on every backend, where the sum has one order: a token's two terms, the rest zeros,
    give one fp32 sum in any order; to 1e-6, relative and absolute, at K = 3), without a scatter in the program; in
    bf16 it is the fp32 computation of the same inputs, rounded once.  combine_impl 'triton' gives the same gradient:
    bit for bit on CPU, where it is gpu_kernels.gather_sum's XLA sum, and to fp32 rounding elsewhere -- on a GPU its
    kernel runs (assert_same: 1e-5, relative and absolute)."""
    D = 16
    for K, rows in ((2, None), (2, 30), (3, None), (3, 45)):
      idx, (src, gs, valid, pos, order) = self._plan(K, rows)
      T, R = idx.shape[0], src.shape[0]
      pos_h = np.asarray(pos)
      r = np.random.default_rng(K)
      xf = jnp.asarray(r.standard_normal((T, D)), jnp.float32)
      g = jnp.asarray(r.standard_normal((R, D)), jnp.float32)

      def naive(xf):
        return jnp.where(valid, jnp.take(xf, src, axis=0), 0)

      def custom(xf):
        return ds_ep._dispatch_rows(xf, src, pos, valid, K)

      with self.subTest(K=K, rows=rows):
        (ya, va), (yb, vb) = jax.vjp(custom, xf), jax.vjp(naive, xf)
        np.testing.assert_array_equal(np.asarray(ya), np.asarray(yb))
        da, db = va(g)[0], vb(g)[0]
        want = np.zeros((T, D))
        for j in np.flatnonzero(pos_h < R):  # token j // K's gradient: its kept rows'
          want[j // K] += np.asarray(g, np.float64)[pos_h[j]]
        np.testing.assert_allclose(np.asarray(da), want, rtol=1e-6, atol=1e-6)
        if K == 2:  # two terms, the rest zeros: one fp32 sum in any order, any backend
          np.testing.assert_array_equal(np.asarray(da), np.asarray(db))
        else:
          np.testing.assert_allclose(np.asarray(da), np.asarray(db), rtol=1e-6, atol=1e-6)
        dc = jax.vjp(lambda x: ds_ep._dispatch_rows(x, src, pos, valid, K, "triton"), xf)[1](g)[0]
        assert_same(dc, da)  # off GPU gather_sum's XLA sum: the same program
        grad = lambda f: jax.make_jaxpr(jax.grad(lambda x: jnp.sum(f(x) * g)))(xf).jaxpr
        self.assertEqual(_count_eqns(grad(custom), "scatter-add"), 0)
        self.assertGreater(_count_eqns(grad(naive), "scatter-add"), 0)  # the check can see one
        gb = g.astype(jnp.bfloat16)
        d16 = jax.vjp(lambda x: ds_ep._dispatch_rows(x, src, pos, valid, K), xf.astype(jnp.bfloat16))[1](gb)[0]
        d32 = jax.vjp(custom, xf)[1](gb.astype(jnp.float32))[0]
        self.assertEqual(d16.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(d16.astype(jnp.float32)), np.asarray(d32.astype(jnp.bfloat16).astype(jnp.float32))
        )

  def test_combine_rows_vjp(self):
    """_combine_rows is the ring's unsort and fp32 weighted sum over the buffer, the same as its naive formulation: a
    pair outside it (another shard's expert, or one a cap dropped) reads zero.  Its backward is the same as
    autodiff's of the naive formulation -- a row's gradient its token's times its weight (one product: no sum to
    reorder), a weight's its token's gradient dotted with its row -- with no scatter-add in the program; a weight
    whose pair is outside the buffer gets exactly zero, and the rows past the groups none.  combine_impl 'triton'
    (off GPU gpu_kernels.gather_sum's XLA sum) gives the same output and row gradients, and the weights' gradients to
    1e-6 on CPU, 1e-5 elsewhere (relative and absolute: the dots computed for the buffer's rows, then placed at their
    pairs), with the same exact zeros.  "The same" is bit for bit on CPU, and to fp32 rounding elsewhere (assert_same:
    1e-5, relative and absolute); the exact zeros are exact on every backend."""
    D = 16
    # dropless (EP 2); capped and full, no row past the groups (EP 2); capped with rows to spare past the groups
    # (EP 4, --ragged-buffer-factor 2: R 40 of T x K 80, 27 in the groups), where an index R + r would land in dW
    for K, rows, ep in ((2, None, 2), (3, 45, 2), (2, 12, 2), (2, 40, 4)):
      idx, (src, gs, valid, pos, order) = self._plan(K, rows, seed=1, ep=ep)
      T, R = idx.shape[0], src.shape[0]
      r = np.random.default_rng(K + 10)
      y = jnp.asarray(r.standard_normal((R, D)), jnp.float32)
      wts = jnp.asarray(r.standard_normal((T, K)), jnp.float32)
      g = jnp.asarray(r.standard_normal((T, D)), jnp.float32)
      inside = np.asarray(pos) < R

      def naive(y, wts):
        yk = jnp.concatenate([y, jnp.zeros((1, D), y.dtype)])[pos].reshape(T, K, D)
        return jnp.einsum("tkd,tk->td", yk.astype(jnp.float32), wts.astype(jnp.float32)).astype(y.dtype)

      def custom(y, wts):
        return ds_ep._combine_rows(y, wts, src, pos, valid, order, y.dtype)

      same = assert_same
      with self.subTest(K=K, rows=rows, ep=ep):
        self.assertFalse(inside.all())  # some pairs are other shards' (or dropped)
        (oa, va), (ob, vb) = jax.vjp(custom, y, wts), jax.vjp(naive, y, wts)
        same(oa, ob)
        (dya, dwa), (dyb, dwb) = va(g), vb(g)
        same(dya, dyb)
        same(dwa, dwb)
        np.testing.assert_array_equal(np.asarray(dwa).reshape(-1)[~inside], 0)
        (oc, vc) = jax.vjp(lambda y, w: ds_ep._combine_rows(y, w, src, pos, valid, order, y.dtype, "triton"), y, wts)
        same(oc, oa)
        dyc, dwc = vc(g)
        same(dyc, dya)
        tol = 1e-6 if jax.default_backend() == "cpu" else 1e-5  # dots of 16 terms, in two orders
        np.testing.assert_allclose(np.asarray(dwc), np.asarray(dwa), rtol=tol, atol=tol)
        np.testing.assert_array_equal(np.asarray(dwc).reshape(-1)[~inside], 0)
        np.testing.assert_array_equal(np.asarray(dya)[~np.asarray(valid[:, 0])], 0)
        self.assertEqual(
            _count_eqns(jax.make_jaxpr(jax.grad(lambda y: jnp.sum(custom(y, wts) * g)))(y).jaxpr, "scatter-add"), 0
        )
        # 'triton' places the weights' dots by one scatter that promises XLA unique indices: a row inside the groups
        # goes to its own (token, choice), every other row to an index of its own past the end (mode='drop' drops it)
        # -- read from the program: the values cannot show a repeat, an update out of range being dropped either way
        res = (y, wts, src, pos, valid, order)
        bwd = jax.make_jaxpr(lambda g: ds_ep._combine_rows_bwd(y.dtype, "triton", res, g))(g)
        (sc,) = [e for e in bwd.jaxpr.eqns if e.primitive.name == "scatter"]
        # the program evaluated up to the scatter's indices (its operands: operand, indices, updates)
        slots = np.asarray(jax.core.eval_jaxpr(bwd.jaxpr.replace(outvars=sc.invars[1:2]), bwd.consts, g)[0]).ravel()
        here = np.asarray(valid[:, 0])
        self.assertTrue(sc.params["unique_indices"])
        self.assertEqual(np.unique(slots).size, R)
        np.testing.assert_array_equal(slots[here], np.asarray(order)[here])
        self.assertTrue(np.all(slots[~here] >= T * K))
        if rows is None or ep == 4:
          self.assertGreater(np.sum(~here), 1)  # rows past the groups, which one shared index would repeat
        if ep == 4:
          self.assertLess(R + np.sum(here), T * K)  # past the buffer is not past dW: R + r would land inside

  def test_sum_rows_vjp(self):
    """_sum_rows, the combine under route_weights 'experts' (the rows weighted already): each token's rows summed in
    fp32 and rounded once, a pair outside the buffer adding nothing -- the naive formulation's output -- and its
    backward autodiff's of that formulation bit for bit (a row's gradient its token's, a copy), the rows past the
    groups getting none, with no scatter-add in the program and no [rows, emb] residual (it reads no row of y).
    combine_impl 'triton' gives the same: bit for bit on CPU, to fp32 rounding on a GPU (assert_same)."""
    D = 16
    for K, rows in ((2, None), (3, 45), (2, 12)):
      idx, (src, gs, valid, pos, order) = self._plan(K, rows, seed=2)
      T, R = idx.shape[0], src.shape[0]
      r = np.random.default_rng(K + 20)
      y = jnp.asarray(r.standard_normal((R, D)), jnp.float32)
      g = jnp.asarray(r.standard_normal((T, D)), jnp.float32)

      def naive(y):
        yk = jnp.concatenate([y, jnp.zeros((1, D), y.dtype)])[pos].reshape(T, K, D)
        return yk.astype(jnp.float32).sum(axis=1).astype(y.dtype)

      def custom(y, impl="xla"):
        return ds_ep._sum_rows(y, src, pos, valid, K, impl)

      with self.subTest(K=K, rows=rows):
        self.assertFalse((np.asarray(pos) < R).all())  # some pairs are other shards' (or dropped)
        (oa, va), (ob, vb) = jax.vjp(custom, y), jax.vjp(naive, y)
        assert_same(oa, ob)
        dya, dyb = va(g)[0], vb(g)[0]
        assert_same(dya, dyb)
        np.testing.assert_array_equal(np.asarray(dya)[~np.asarray(valid[:, 0])], 0)
        oc, vc = jax.vjp(lambda y: custom(y, "triton"), y)
        assert_same(oc, oa)
        assert_same(vc(g)[0], dya)
        self.assertNotIn((R, D), residual_shapes(custom, y))
        self.assertEqual(
            _count_eqns(jax.make_jaxpr(jax.grad(lambda y: jnp.sum(custom(y) * g)))(y).jaxpr, "scatter-add"), 0
        )
        y16, g16 = y.astype(jnp.bfloat16), g.astype(jnp.bfloat16)
        o16 = ds_ep._sum_rows(y16, src, pos, valid, K)
        self.assertEqual(o16.dtype, jnp.bfloat16)  # rounded once, from the fp32 sum of the bf16 rows
        np.testing.assert_array_equal(
            np.asarray(o16.astype(jnp.float32)),
            np.asarray(naive(y16.astype(jnp.float32)).astype(jnp.bfloat16), np.float32),
        )
        self.assertEqual(jax.vjp(lambda y: ds_ep._sum_rows(y, src, pos, valid, K), y16)[1](g16)[0].dtype, jnp.bfloat16)

  def test_row_weights_vjp(self):
    """_row_weights, each buffer row's routing weight under route_weights 'experts': wts[t, k] of the row's pair
    (`order`), and its backward places each row's gradient at the row's pair -- exactly, as autodiff's transpose of
    the gather does for the rows inside the groups -- and gives every other pair zero: whatever the rows past the
    groups receive (other pairs' weights' gradients, or garbage from an unmasked buffer: NaN here) reaches no pair,
    and no scatter-add is in the program."""
    past = []
    for K, rows in ((2, None), (3, 45), (2, 12)):
      idx, (src, gs, valid, pos, order) = self._plan(K, rows, seed=3)
      T, R = idx.shape[0], src.shape[0]
      r = np.random.default_rng(K + 30)
      wts = jnp.asarray(r.standard_normal((T, K)), jnp.float32)
      g = jnp.asarray(r.standard_normal((R,)), jnp.float32)
      inside = np.asarray(valid[:, 0])
      past.append(not inside.all())
      with self.subTest(K=K, rows=rows):
        w_row, v = jax.vjp(lambda w: ds_ep._row_weights(w, pos, order), wts)
        np.testing.assert_array_equal(np.asarray(w_row), np.asarray(wts).reshape(-1)[np.asarray(order)])
        want = np.zeros(T * K, np.float32)
        want[np.asarray(order)[inside]] = np.asarray(g)[inside]
        np.testing.assert_array_equal(np.asarray(v(g)[0]).reshape(-1), want)
        np.testing.assert_array_equal(np.asarray(v(jnp.where(inside, g, jnp.nan))[0]).reshape(-1), want)
        naive = jax.vjp(lambda w: jnp.take(w.reshape(-1), order, axis=0) * inside, wts)[1](g)[0]
        np.testing.assert_array_equal(np.asarray(v(g)[0]), np.asarray(naive))
        grad = jax.make_jaxpr(jax.grad(lambda w: jnp.sum(ds_ep._row_weights(w, pos, order) * g)))(wts).jaxpr
        self.assertEqual(_count_eqns(grad, "scatter-add"), 0)
    self.assertTrue(any(past))  # the dropless buffer has rows past its groups


# ------------------------------------------------------------------ the grouped matmuls
def compact_case(seed=0, dtype=jnp.float32, G=4, K=64, N=32, R=256, sizes=(40, 0, 90, 70)):
  """A compact buffer of R rows: groups `sizes` from row 0, the rows past them zero (as the dispatch leaves them)
  -> (lhs, rhs [G, K, N], group sizes, valid [R, 1], a cotangent)."""
  gs = jnp.asarray(sizes, jnp.int32)
  valid = (jnp.arange(R) < int(np.sum(sizes)))[:, None]
  ks = jax.random.split(jax.random.key(seed), 3)
  lhs = jnp.where(valid, jax.random.normal(ks[0], (R, K), jnp.float32), 0).astype(dtype)
  rhs = (0.1 * jax.random.normal(ks[1], (G, K, N), jnp.float32)).astype(dtype)
  ct = jax.random.normal(ks[2], (R, N), jnp.float32).astype(dtype)
  return lhs, rhs, gs, valid, ct


@contextlib.contextmanager
def unwritten_rows():
  """tokamax_gmm's forward and dlhs products (ds_ep._tokamax_dot) with NaN in the compact buffer's rows past the
  groups, where a GPU kernel (Mosaic GPU's, Triton's) leaves them unwritten -- tokamax's lax.ragged_dot here writes
  zeros, as the masks do.  dW ([G, K, N]) holds no row of the buffer.  Only what traces inside takes it: jax.jit
  replays a function's earlier trace (the same function, the same shapes) without calling it.  -> the poisoned
  products' shapes, as they trace."""
  real, poisoned = ds_ep._tokamax_dot, []

  def unwritten(lhs, rhs, gs, dims, out_dtype, impls):
    out = real(lhs, rhs, gs, dims, out_dtype, impls)
    if dims is ds_ep._CONTRACTING_DIMS:  # dW: [G, K, N], no rows of the buffer
      return out
    poisoned.append(out.shape)
    return jnp.where(jnp.arange(out.shape[0])[:, None] < jnp.sum(gs), out, jnp.nan).astype(out.dtype)

  with mock.patch.object(ds_ep, "_tokamax_dot", unwritten):
    yield poisoned


@needs_tokamax
class TokamaxGmm(unittest.TestCase):
  """ds_ep.tokamax_gmm, tokamax_gmm 'xla', against named_gmm's pure-JAX `ragged_dot` on the same rows (group offset
  0: a compact buffer holds this shard's groups from row 0)."""

  def vjp(self, f, *args, ct):
    y, v = jax.vjp(jax.jit(f), *args)
    return (y,) + tuple(v(ct))

  def test_matches_ragged_dot(self):
    """Forward, dlhs and dW bit for bit, in fp32 and bf16 (fp32 accumulation, rounded once, either way), for groups
    with an empty one and rows past them, and for groups filling the buffer."""
    for dtype in (jnp.float32, jnp.bfloat16):
      for sizes in ((40, 0, 90, 70), (64, 64, 64, 64)):
        lhs, rhs, gs, valid, ct = compact_case(dtype=dtype, sizes=sizes)
        with self.subTest(dtype=jnp.dtype(dtype).name, sizes=sizes):
          got = self.vjp(lambda x, w: ds_ep.tokamax_gmm("a", "xla", x, w, gs, valid), lhs, rhs, ct=ct)
          want = self.vjp(
              lambda x, w: ds_ep.named_gmm("a", TILE, True, x, w, gs, jnp.int32(0), impl="ragged_dot"), lhs, rhs, ct=ct
          )
          assert_equal_leaves(self, got, want, "out, dlhs, dW")
          self.assertEqual([a.dtype for a in got], [dtype] * 3)

  def test_rows_past_the_groups(self):
    """What the rows past the groups hold reaches nothing -- NaN in the operands there, or NaN the kernels leave there
    (Mosaic GPU's skip those rows, unwritten: unwritten_rows fills them): the forward's and dlhs's rows there come
    out zero, and every value is the clean buffer's bit for bit."""
    lhs, rhs, gs, valid, ct = compact_case()
    f = lambda x, w: ds_ep.tokamax_gmm("a", "xla", x, w, gs, valid)
    clean = self.vjp(f, lhs, rhs, ct=ct)
    dirty = self.vjp(f, jnp.where(valid, lhs, jnp.nan), rhs, ct=jnp.where(valid, ct, jnp.nan))
    with unwritten_rows() as poisoned:  # a function of its own: jit would replay f's clean trace
      garbage = self.vjp(lambda x, w: f(x, w), lhs, rhs, ct=ct)
    self.assertTrue(poisoned)
    for what, got in (("NaN operands", dirty), ("NaN kernel rows", garbage)):
      assert_equal_leaves(self, got, clean, f"{what}: out, dlhs, dW")
      for a in got[:2]:
        np.testing.assert_array_equal(np.asarray(a)[~np.asarray(valid[:, 0])], 0)

  def _pair(self, impl, dtype, acc):
    """d/d(w, x) of two grouped matmuls of one weight w -- the second on w as the first hands it on (acc) or on w
    itself -- the shape of a batch-split layer's mb0 and mb1 experts under split_dw 'acc' and 'sum'."""
    lhs, rhs, gs, valid, ct = compact_case(dtype=dtype)
    if impl == "tokamax":
      gmm = lambda nm, x, w, acc=False: ds_ep.tokamax_gmm(nm, "xla", x, w, gs, valid, acc=acc)
    else:
      gmm = lambda nm, x, w, acc=False: ds_ep.named_gmm(
          nm, TILE, True, x, w, gs, jnp.int32(0), impl="ragged_dot", acc=acc
      )

    def loss(w, x):
      y0, wp = gmm("a", x, w, acc=True) if acc else (gmm("a", x, w), w)
      y1 = gmm("b", (x * 0.5).astype(dtype), wp)
      return jnp.sum(y0.astype(jnp.float32) * ct.astype(jnp.float32)) + jnp.sum(y1.astype(jnp.float32) * 0.3)

    return jax.jit(jax.grad(loss, argnums=(0, 1)))(rhs, lhs), jax.make_jaxpr(jax.grad(loss))(rhs, lhs), rhs.shape

  def test_acc(self):
    """acc=True hands the rhs on, and the second product's dW joins this call's (named_gmm's acc), with no add_any
    over the weight where two plain calls have one: in fp32 bit for bit `ragged_dot`'s acc and the two plain calls'.
    In bf16 the halves' sum may round once (megablox's and `ragged_dot`'s acc: in fp32 before the rounding) or twice
    (each half's dW in the weight dtype first, as split_dw 'sum' rounds): within two ulps of the leaf of
    `ragged_dot`'s with and without acc either way."""
    acc, jaxpr, shape = self._pair("tokamax", jnp.float32, True)
    plain, plain_jaxpr, _ = self._pair("tokamax", jnp.float32, False)
    assert_equal_leaves(self, acc, self._pair("ragged_dot", jnp.float32, True)[0], "fp32 vs ragged_dot acc")
    assert_equal_leaves(self, acc, plain, "acc vs plain")
    self.assertEqual([_eqn_out_shapes(j, "add_any").count(shape) for j in (jaxpr, plain_jaxpr)], [0, 1])
    bf16 = self._pair("tokamax", jnp.bfloat16, True)[0]
    for acc in (True, False):
      assert_near(self, bf16, self._pair("ragged_dot", jnp.bfloat16, acc)[0], 2, f"bf16 vs ragged_dot acc={acc}")

  def test_stacked(self):
    """stacked=(ws, l): the forward and dlhs read block l of the stacked weight in place -- `rhs` carries only the
    weight gradient (its values are never read), and `ws` gets none -- with the group sizes padded by the other
    blocks' empty groups: bit for bit the slice's for one block; for three, whose padding widens the dense
    lowering's contraction, to 4 fp32 ulps of each leaf.  A block of another shape or dtype is refused."""
    lhs, _, gs, valid, ct = compact_case()
    for L, layer in ((1, 0), (3, 1), (3, 2)):
      ws = 0.1 * jax.random.normal(jax.random.key(L + layer), (L,) + (4, 64, 32), jnp.float32)
      f = lambda w, x, st: ds_ep.tokamax_gmm("a", "xla", x, w, gs, valid, stacked=st)
      with self.subTest(L=L, layer=layer):
        sliced = self.vjp(lambda w, x: f(w, x, None), ws[layer], lhs, ct=ct)
        stacked = self.vjp(lambda w, x: f(w, x, (ws, layer)), ws[layer], lhs, ct=ct)
        blind = self.vjp(lambda w, x: f(w, x, (ws, layer)), jnp.zeros_like(ws[layer]), lhs, ct=ct)
        assert_equal_leaves(self, blind, stacked, "rhs values unread")
        if L == 1:
          assert_equal_leaves(self, stacked, sliced, "stacked vs sliced")
        for i, (a, b) in enumerate(zip(leaves(stacked), leaves(sliced))):
          self.assertLessEqual(float(np.max(np.abs(a - b))), 2.0**-21 * float(np.max(np.abs(b))), i)
        g_ws = jax.grad(lambda ws: jnp.sum(f(ws[layer], lhs, (ws, layer)) * ct))(ws)
        g_sl = jax.grad(lambda w: jnp.sum(f(w, lhs, (ws, layer)) * ct))(ws[layer])
        np.testing.assert_array_equal(np.asarray(g_ws[layer]), np.asarray(g_sl))  # all of it through the slice
        np.testing.assert_array_equal(np.delete(np.asarray(g_ws), layer, axis=0), 0)
    with self.assertRaisesRegex(ValueError, "is not the"):
      ds_ep.tokamax_gmm("a", "xla", lhs, ws[0][:, :32], gs, valid, stacked=(ws, 0))
    with self.assertRaisesRegex(ValueError, "is not the"):
      ds_ep.tokamax_gmm("a", "xla", lhs, ws[0].astype(jnp.bfloat16), gs, valid, stacked=(ws, 0))

  def test_names_reach_the_lowered_program(self):
    """The three products carry named_gmm's names under 'tokamax' for xprof -- `jvp(tokamax)/gmm_wo`,
    `transpose(jvp(tokamax))/gmm_wo_dx` and `_dw`: the transform takes 'tokamax' and the kernel's own scope stays
    clean."""
    lhs, rhs, gs, valid, _ = compact_case()
    txt = (
        jax.jit(
            jax.value_and_grad(lambda x, w: jnp.sum(ds_ep.tokamax_gmm("gmm_wo", "xla", x, w, gs, valid)), argnums=(0, 1))
        )
        .lower(lhs, rhs)
        .as_text(debug_info=True)
    )
    for name in ("/jvp(tokamax)/gmm_wo/", "/transpose(jvp(tokamax))/gmm_wo_dx/", "/transpose(jvp(tokamax))/gmm_wo_dw/"):
      self.assertIn(name, txt)
    self.assertNotIn("(gmm_wo", txt)

  def test_keep(self):
    """keep=Recompute(fn, args), lean residuals' -- here the dispatch's rows, kept as the fewer tokens they gather
    and the plan: the VJP keeps `args` in the lhs's place and recomputes the lhs for the dW kernel.  The output and
    every gradient bit for bit the plain call's, for the plain, acc (a second product on the handed-on weight),
    stacked, own-dtype accumulator (dw32) and its pass-through variants, and no lhs-shaped residual left (without
    keep there is one: the check sees it).  The gradient reaches `args` through the lhs's own producer (the
    dispatch's VJP) only."""
    E, ep, K, T, Kd, N = 8, 2, 2, 100, 64, 32
    r = np.random.default_rng(5)
    src, gs, valid, pos, _ = ds_ep._local_plan(
        jnp.asarray(r.integers(0, E, (T, K)), jnp.int32), jnp.int32(1), E // ep, None
    )
    R = src.shape[0]  # T x K rows: more than the tokens
    xt = jnp.asarray(r.standard_normal((T, Kd)), jnp.float32)
    rhs = jnp.asarray(0.1 * r.standard_normal((E // ep, Kd, N)), jnp.float32)
    ws = jnp.stack([rhs * 0.5, rhs, rhs * 2.0])
    acc0 = jnp.asarray(r.standard_normal(rhs.shape), jnp.float32)
    ct = jnp.asarray(r.standard_normal((R, N)), jnp.float32)

    def gmm(name, scale, xt, w, keep, **kw):
      f = lambda xt: (ds_ep._dispatch_rows(xt, src, pos, valid, K) * scale).astype(xt.dtype)
      return ds_ep.tokamax_gmm(name, "xla", f(xt), w, gs, valid, keep=ds_ep.Recompute(f, (xt,)) if keep else None, **kw)

    def acc_pair(xt, w, acc, keep, **kw):
      o, wp = gmm("a", 1.0, xt, w, keep, acc=True, **kw)
      if "dw0" in kw:  # the handed-on accumulator start, the weight itself
        return o + gmm("b", 0.5, xt, w, keep, dw0=wp)
      return o + gmm("b", 0.5, xt, wp, keep)

    variants = {
        "plain": (jnp.float32, lambda xt, w, acc, keep: gmm("a", 1.0, xt, w, keep)),
        "acc": (jnp.float32, acc_pair),
        "stacked": (jnp.float32, lambda xt, w, acc, keep: gmm("a", 1.0, xt, w, keep, stacked=(ws, 1))),
        "dw32": (jnp.bfloat16, lambda xt, w, acc, keep: gmm("a", 1.0, xt, w, keep, dw0=acc)),
        "dw32 pass": (jnp.bfloat16, lambda xt, w, acc, keep: acc_pair(xt, w, acc, keep, dw0=acc)),
    }
    for name, (dt, f) in variants.items():
      args = (xt.astype(dt), rhs.astype(dt), acc0)
      with self.subTest(name):
        runs = {}
        for keep in (False, True):
          g = lambda *a, keep=keep: f(*a, keep)
          y, v = jax.vjp(g, *args)
          runs[keep] = ((y,) + tuple(v(ct.astype(dt))), residual_shapes(g, *args))
        assert_equal_leaves(self, runs[True][0], runs[False][0], f"{name}: out, d tokens, dW, d acc")
        self.assertIn((R, Kd), runs[False][1])
        self.assertNotIn((R, Kd), runs[True][1])
        self.assertIn((T, Kd), runs[True][1])

  def test_backends(self):
    """tokamax_gmm's backends as ragged_dot_general's implementation lists: Mosaic GPU has no dW kernel on sm80 /
    sm100, so dW stays on Triton there (and the forward / dlhs fall back to it); 'xla' is tokamax's lax.ragged_dot
    throughout.  Without tokamax the error names the package."""
    self.assertEqual(sorted(ds_ep.TOKAMAX_GMM), ["mosaic", "tokamax_mosaic", "triton", "xla"])
    self.assertEqual(
        {k: ds_ep._tokamax_impls(k) for k in ds_ep.TOKAMAX_GMM},
        {
            "triton": (("triton",), ("triton",)),
            "mosaic": (("mosaic", "triton"), ("triton",)),
            "tokamax_mosaic": (("mosaic", "triton"), ("triton",)),
            "xla": (("xla",), ("xla",)),
        },
    )
    with _sys_modules({"tokamax": None}), self.assertRaisesRegex(ImportError, "pip install tokamax"):
      ds_ep._tokamax()

  def test_pytest_argv(self):
    """With a pytest command line's own options in sys.argv (-q, -p no:cacheprovider, -v -o ...) as this module is
    imported, a grouped matmul runs: tokamax before 0.0.15 hands them all to absl at its first config read, which
    refuses them, unless this module has marked absl's flags parsed.  In a fresh interpreter, whose absl starts
    unparsed."""
    script = textwrap.dedent(
        """
        import sys
        sys.argv = [sys.argv[0], '-q', '-p', 'no:cacheprovider', '-v', '-o', 'faulthandler_timeout=600']
        from tests.unit.experimental.mpmd.jax import gpu_port_test as t
        lhs, rhs, gs, valid, _ = t.compact_case()
        t.ds_ep.tokamax_gmm('a', 'xla', lhs, rhs, gs, valid).block_until_ready()
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=testing.child_pythonpath(__name__))
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    self.assertEqual(out.returncode, 0, out.stderr[-3000:])


class TokamaxValidation(unittest.TestCase):
  """StageConfig's and named_gmm's refusals around gmm_impl 'tokamax' (no kernel runs)."""

  def test_refusals(self):
    """'tokamax' sorts into a buffer of its own, so MaxText's ring_ragged_sort is refused with it; tokamax_gmm must
    be one of TOKAMAX_GMM; the ring's named_gmm refuses impl 'tokamax' (it runs on the compact buffer, not on the
    ring's sorted rows); a capped buffer (ragged_buffer_factor > 0) needs the ragged sort, or 'tokamax', whose
    compact buffer it caps."""
    self.assertIn("tokamax", ds_ep.GMM_IMPLS)
    base = tiny_xla()
    ok = dataclasses.replace(base, **TOKAMAX)
    self.assertEqual((ok.gmm_impl, ok.tokamax_gmm, ok.use_ragged_sort), ("tokamax", "xla", False))
    default = dataclasses.replace(base, gmm_impl="tokamax").tokamax_gmm
    self.assertIn(default, ds_ep.TOKAMAX_GMM)
    self.assertEqual(dsp.make_parser().get_default("tokamax_gmm"), default)  # the CLI's is StageConfig's
    with self.assertRaisesRegex(ValueError, "gmm_impl='tokamax'.*use use_ragged_sort=False"):
      dataclasses.replace(ok, use_ragged_sort=True)
    for bad in ("cutlass", "Triton", ""):
      with self.subTest(tokamax_gmm=bad), self.assertRaisesRegex(ValueError, "tokamax_gmm must be one of"):
        dataclasses.replace(ok, tokamax_gmm=bad)
    x = jnp.zeros((8, 4), jnp.float32)
    with self.assertRaisesRegex(ValueError, "named_gmm: gmm_impl 'tokamax' runs on a compact local buffer"):
      ds_ep.named_gmm("gmm_wo", TILE, True, x, jnp.zeros((2, 4, 4)), jnp.array([4, 4]), jnp.int32(0), impl="tokamax")
    self.assertEqual(dataclasses.replace(ok, ragged_buffer_factor=1.5).ragged_buffer_factor, 1.5)
    with self.assertRaisesRegex(ValueError, "needs use_ragged_sort \\(or gmm_impl 'tokamax'"):
      dataclasses.replace(base, ragged_buffer_factor=1.5)
    # route_weights 'experts' needs each buffer row's (token, choice) pair: the compact buffer's or the argsort
    # dispatch's, not MaxText's ring_ragged_sort's
    self.assertEqual(ok.route_weights, "combine")
    self.assertEqual(dsp.make_parser().get_default("route_weights"), "combine")  # the CLI's is StageConfig's
    for good in (ok, base):
      self.assertEqual(dataclasses.replace(good, route_weights="experts").route_weights, "experts")
    with self.assertRaisesRegex(ValueError, "route_weights 'experts'.*use use_ragged_sort=False"):
      dataclasses.replace(base, use_ragged_sort=True, route_weights="experts")
    with self.assertRaisesRegex(ValueError, "route_weights must be one of"):
      dataclasses.replace(ok, route_weights="rows")


# ------------------------------------------------------------------ the ring of experts
RING_SPECS = ({"gate": {"kernel": P()}, "wi_0": P("expert"), "wi_1": P("expert"), "wo": P("expert")}, P(ds_ep.FE))


def ring_scfg(model, impl, dtype=jnp.float32, **kw):
  """tiny()'s DeepSeek-V3 MoE (top-k of the sigmoid scores, renormalised and scaled) or Q_TINY's Qwen3.8 one (top-k
  of the raw logits, a softmax over the k) at fsdp 2 x ep 2 -- 8 experts, top-2, emb and MoE width 256 -- with the
  kernels `impl`, in `dtype`."""
  scfg = (tiny_xla if model == "deepseek" else qtiny_xla)(**impl, **kw)
  return dataclasses.replace(scfg, model=scfg.model.replace(dtype=dtype))


def ring_inputs(scfg, seed=0, layers=1):
  """fp32 router and expert weights ([layers, E, ...] experts for layers > 1), tokens [8, 64, D] (two rows a device)
  and a probe that makes the output a scalar loss."""
  cfg = scfg.model
  E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_mlp_dim
  r = np.random.default_rng(seed)
  lead = (layers,) if layers > 1 else ()
  w = {
      "gate": {"kernel": (0.2 * r.standard_normal((D, E))).astype(np.float32)},
      "wi_0": (r.standard_normal(lead + (E, D, F)) / np.sqrt(D)).astype(np.float32),
      "wi_1": (r.standard_normal(lead + (E, D, F)) / np.sqrt(D)).astype(np.float32),
      "wo": (r.standard_normal(lead + (E, F, D)) / np.sqrt(F)).astype(np.float32),
  }
  x = r.standard_normal((8, 64, D)).astype(np.float32)
  return w, x, r.standard_normal(x.shape).astype(np.float32)


def ring_mesh():
  return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ds_ep.FE)


def ring_grads(scfg, w, x, probe, lean=False):
  """`ds_ep._ring_moe` (router, EP all-gather, routed experts, reduce-scatter) over every device's tokens -> its
  output and the gradients of sum(output x probe) in the router, the three expert weights and the tokens.  `lean`:
  traced under lean residuals, as a saved-residual forward is."""
  mesh = ring_mesh()
  wspec, xspec = RING_SPECS
  f = jax.shard_map(
      lambda w, x: ds_ep._ring_moe(w, x, jax.random.key(3), scfg)[0],
      mesh=mesh,
      in_specs=(wspec, xspec),
      out_specs=xspec,
      check_vma=False,
  )
  put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))

  def loss(w, x):
    y = f(w, x)
    return jnp.sum(y.astype(jnp.float32) * probe), y

  with dl.lean_residuals(lean):
    (_, y), g = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        jax.tree.map(put, w, wspec), put(jnp.asarray(x, scfg.model.dtype), xspec)
    )
  return y, g


def ring_kept(scfg, w, x, lean=True):
  """The shape of each array `ds_ep._ring_moe`'s VJP keeps, a device's view: traced abstractly inside the ring's
  shard_map, as a saved-residual forward traces its layers' VJP (ChunkPrograms' probe), each array once (`lean`:
  ring_grads'), the primal inputs left out -> [shape]."""
  mesh = ring_mesh()
  wspec, xspec = RING_SPECS
  kept = []

  def body(w, x):
    with dl.lean_residuals(lean):
      y, vjp = jax.vjp(lambda w, x: ds_ep._ring_moe(w, x, jax.random.key(3), scfg)[0], w, x)
    prim, seen = jax.tree.leaves((w, x)), []
    for a in jax.tree.leaves(vjp):
      if hasattr(a, "shape") and not any(a is q for q in seen + prim):
        seen.append(a)
        kept.append(tuple(a.shape))
    return y

  aval = lambda a, s, dt: jax.ShapeDtypeStruct(np.shape(a), dt, sharding=NamedSharding(mesh, s))
  jax.eval_shape(
      jax.shard_map(body, mesh=mesh, in_specs=(wspec, xspec), out_specs=xspec, check_vma=False),
      jax.tree.map(lambda a, s: aval(a, s, jnp.float32), w, wspec),
      aval(x, xspec, scfg.model.dtype),
  )
  return kept


def halves_grads(scfg, w, x, probe, layer, stacked, lean=False):
  """A batch-split layer's routed experts without its phase barriers (split_layers' experts): each device's rows as
  two halves, each routed with a key of its own, dispatched, through moe_experts and combined -- under split_dw
  'acc' mb0's experts handing their weights on to mb1's (pass_weights) -- reading layer `layer` of the stacked
  expert weights (w's [L, E, ...]) in place under `stacked`, else from its slice.  -> the output and the gradients
  of sum(output x probe) in the router, the stacked expert weights and the tokens.  `lean`: ring_grads'."""
  mesh = ring_mesh()
  wspec = {"gate": {"kernel": P()}, **{k: P(None, "expert") for k in ds_ep.EXPERT_LEAVES}}
  xspec = P(ds_ep.FE)

  def body(w, x):
    ws = {k: w[k].astype(scfg.model.dtype) for k in ds_ep.EXPERT_LEAVES}
    wl = {"gate": w["gate"], **{k: ws[k][layer] for k in ds_ep.EXPERT_LEAVES}}
    st = (ws, layer) if stacked else None
    r, out, passed = x.shape[0] // 2, [], None
    for i, xi in enumerate((x[:r], x[r:])):
      key = jax.random.fold_in(jax.random.key(3), i)
      xg, sg = ds_ep.moe_dispatch(xi, ds_ep.moe_scores(wl, xi, scfg), scfg)
      if scfg.split_dw == "acc" and i == 0:
        y, passed = ds_ep.moe_experts(wl, xg, sg, key, scfg, pass_weights=True, stacked=st)
      else:
        y = ds_ep.moe_experts({**wl, **(passed or {})}, xg, sg, key, scfg, stacked=st)
      out.append(ds_ep.moe_combine(y, scfg))
    return jnp.concatenate(out)

  f = jax.shard_map(body, mesh=mesh, in_specs=(wspec, xspec), out_specs=xspec, check_vma=False)
  put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))

  def loss(w, x):
    y = f(w, x)
    return jnp.sum(y.astype(jnp.float32) * probe), y

  with dl.lean_residuals(lean):
    (_, y), g = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        jax.tree.map(put, w, wspec), put(jnp.asarray(x, scfg.model.dtype), xspec)
    )
  return y, g


@needs_tokamax
@testing.needs_devices(4)
class TokamaxRingMoe(unittest.TestCase):
  """The ring of experts on a 2 x 2 mesh of fsdp x ep under gmm_impl 'tokamax' against 'ragged_dot' on the argsort
  dispatch: the output, and the gradients of the router, the expert weights and the tokens."""

  def pair(self, model, dtype=jnp.float32, **kw):
    w, x, probe = ring_inputs(ring_scfg(model, RAGGED))
    return [ring_grads(ring_scfg(model, impl, dtype, **kw), w, x, probe) for impl in (TOKAMAX, RAGGED)]

  def test_fp32_bitwise(self):
    """fp32, DeepSeek-V3's and Qwen3.8's routers, top-k and random routing (with replacement: a token may take an
    expert twice, so a buffer row is a (token, choice) pair, not a token): bit for bit."""
    for model in ("deepseek", "qwen"):
      for random_routing in (False, True):
        with self.subTest(model=model, random_routing=random_routing):
          got, want = self.pair(model, random_routing=random_routing)
          assert_equal_leaves(self, got, want, "output, router, experts, tokens")

  def test_bf16(self):
    """bf16 random routing, whose weights are the raw scores, bit for bit; top-k routing renormalises its weights
    in bf16 (DeepSeek-V3: by their sum, scaled; Qwen3.8: a softmax rounded to bf16, renormalised), which XLA:CPU
    computes in fp32 across a fusion where its excess precision allows, fused differently in the two programs:
    the output still bit for bit, every gradient within an ulp or two of its leaf."""
    got, want = self.pair("deepseek", jnp.bfloat16, random_routing=True)
    assert_equal_leaves(self, got, want, "bf16 random routing")
    for model in ("deepseek", "qwen"):
      with self.subTest(model=model):
        got, want = self.pair(model, jnp.bfloat16, random_routing=False)
        assert_equal_leaves(self, got[0], want[0], "bf16 top-k output")
        assert_near(self, got[1], want[1], 2, "bf16 top-k gradients")

  def test_split_dw_and_stacked(self):
    """A batch-split layer's two halves through the routed experts (`halves_grads`), split_dw 'acc' (mb0's dW kernels
    adding into mb1's, tokamax_gmm's acc) and 'sum', reading layer 1 of two-layer stacked weights from its slice
    ('sliced') or in place ('stacked'): bit for bit 'ragged_dot''s sliced, but tokamax's stacked forward, whose
    padded groups widen XLA:CPU's dense contraction, there to fp32 rounding."""
    model = "deepseek"
    w, x, probe = ring_inputs(ring_scfg(model, RAGGED), layers=2)
    for split_dw in ds_ep.SPLIT_DW:
      want = halves_grads(ring_scfg(model, RAGGED, split_dw=split_dw), w, x, probe, 1, False)
      for stacked in (False, True):
        with self.subTest(split_dw=split_dw, stacked=stacked):
          got = halves_grads(ring_scfg(model, TOKAMAX, split_dw=split_dw), w, x, probe, 1, stacked)
          if stacked:
            for i, (a, b) in enumerate(zip(leaves(got), leaves(want), strict=True)):
              self.assertLessEqual(float(np.max(np.abs(a - b))), 2.0**-18 * float(np.max(np.abs(b))), i)
          else:
            assert_equal_leaves(self, got, want, "sliced")

  def test_route_weights_experts(self):
    """route_weights 'experts' (the routing weights on wo's input rows, the combine an unweighted sum): tokamax's
    compact buffer (_row_weights, _sum_rows) bit for bit 'ragged_dot''s argsort rows (autodiff of the weights' gather
    and of the plain sum) in fp32, for DeepSeek-V3's and Qwen3.8's routers, top-k and random routing; and within fp32
    rounding of route_weights 'combine' -- the same sums, associated otherwise (up to 5.7e-7 of a leaf's largest
    magnitude measured, jax 0.11.1 and 0.12.0.1).  In bf16, random routing (top-k's choices flip against fp32's,
    under either), as close to the fp32 result as 'combine' comes: within 3 bf16 ulps of each leaf (up to 1.6
    measured on jax 0.11.1, 1.8 on 0.12.0.1; 'combine' 1.7 and 2.0)."""
    for model in ("deepseek", "qwen"):
      w, x, probe = ring_inputs(ring_scfg(model, RAGGED))
      for random_routing in (False, True):
        with self.subTest(model=model, random_routing=random_routing):
          got, want = self.pair(model, random_routing=random_routing, route_weights="experts")
          assert_equal_leaves(self, got, want, "output, router, experts, tokens")
          comb = ring_grads(ring_scfg(model, TOKAMAX, random_routing=random_routing), w, x, probe)
          for i, (a, b) in enumerate(zip(leaves(got), leaves(comb), strict=True)):
            self.assertLessEqual(float(np.max(np.abs(a - b))), 4e-6 * float(np.max(np.abs(b))), i)
          if random_routing:
            bf16 = ring_grads(
                ring_scfg(model, TOKAMAX, jnp.bfloat16, random_routing=True, route_weights="experts"), w, x, probe
            )
            assert_near(self, bf16, comb, 3, "bf16 against fp32")

  def test_lean_residuals(self):
    """Under lean residuals, as a saved-residual forward traces, the wi products keep the gathered tokens (and the
    plan) and wo's the gated product's inputs, and recompute their lhs in the backward (ds_ep.Recompute): the output
    and every gradient bit for bit the plain trace's -- tokamax's compact buffer dropless, and capped with its masks
    off, under both route_weights; 'ragged_dot''s argsort rows; a batch-split layer's halves under split_dw 'acc',
    reading stacked weights in place; fp32, and bf16 on CPU (a GPU's XLA may fuse the recomputed product with
    excess precision of its own: within an ulp there).  But in fp32 under route_weights 'experts', the router's and
    the tokens' gradients to 4 fp32 ulps of their largest magnitude: only they take the routing weights' gradient,
    a sum over each row of the gated product times wo's dlhs, which jax 0.11.1's XLA:CPU rounds apart in the two
    traces (up to 1.58 ulps measured, the halves' router gradient; jax 0.12.0.1 bit for bit).  And 'ragged_dot''s
    lean VJP keeps the EP group's T gathered tokens for the wi products, not the T x K sorted rows (`ring_kept`)."""

    def check_routed(got, want, name):
      """fp32 under route_weights 'experts': the output and the expert weights' gradients bit for bit, the router's
      and the tokens' to 4 fp32 ulps."""
      (y, (gw, gx)), (y0, (gw0, gx0)) = got, want
      rest = lambda gw: {k: v for k, v in gw.items() if k != "gate"}
      assert_equal_leaves(self, (y, rest(gw)), (y0, rest(gw0)), name)
      for what, a, b in (("router", gw["gate"], gw0["gate"]), ("tokens", gx, gx0)):
        [a], [b] = leaves(a), leaves(b)
        err, top = float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))
        self.assertLessEqual(err, 4 * FP32_EPS * top, f"{name}: the {what} gradient, {err:.3g} of {top:.3g}")

    model = "deepseek"
    w, x, probe = ring_inputs(ring_scfg(model, RAGGED))
    experts = dict(route_weights="experts")
    for name, impl, dt, kw in (
        ("tokamax", TOKAMAX, jnp.float32, {}),
        ("tokamax, experts", TOKAMAX, jnp.float32, experts),
        ("capped, unmasked", TOKAMAX, jnp.float32, dict(experts, ragged_buffer_factor=1.5, buffer_masks=False)),
        ("ragged_dot", RAGGED, jnp.float32, {}),
        ("ragged_dot, experts", RAGGED, jnp.float32, experts),
        ("bf16 experts", TOKAMAX, jnp.bfloat16, dict(experts, random_routing=False)),
    ):
      with self.subTest(name):
        sc = ring_scfg(model, impl, dt, **kw)
        got, want = ring_grads(sc, w, x, probe, lean=True), ring_grads(sc, w, x, probe)
        if dt == jnp.float32 and kw.get("route_weights") == "experts":
          check_routed(got, want, name)
        elif dt == jnp.float32 or jax.default_backend() == "cpu":
          assert_equal_leaves(self, got, want, name)
        else:
          assert_near(self, got, want, 1, name)
        if impl == RAGGED:
          # the argsort dispatch's wi products keep the EP group's T gathered tokens (moe_experts' keep_xs), not the
          # T x K sorted rows: of the rows' shape only h0 and h1 are kept (moe_mlp is emb, 256, here)
          T, K, D = x.shape[0] * x.shape[1] // sc.fsdp, sc.model.num_experts_per_tok, sc.model.emb_dim
          kept = ring_kept(sc, w, x)
          self.assertEqual((kept.count((T, D)), kept.count((T * K, D))), (1, 2), f"{name}: {kept}")
    w, x, probe = ring_inputs(ring_scfg(model, RAGGED), layers=2)
    sc = ring_scfg(model, TOKAMAX, route_weights="experts")
    check_routed(
        halves_grads(sc, w, x, probe, 1, True, lean=True), halves_grads(sc, w, x, probe, 1, True), "batch-split halves"
    )


# ------------------------------------------------------------------ a chunk's programs
def chunk_run(scfg, layers=(0, 1), mbs=8, seq=128, seed=0):
  """`layers` as one middle chunk on a one-stage mesh of scfg's fsdp x ep, the reference attention (the experts
  are what differ): its forward and backward programs on the stage's own init and random activations -> [y, dx,
  *the chunk's per-device gradient partials] as fp32 host arrays."""
  scfg = dataclasses.replace(scfg, model=scfg.model.replace(attention_kernel="reference"))
  mesh = dsp.stage_meshes(1, scfg.fsdp, scfg.ep)[0]
  ch = ds_ep.ChunkSpec(
      index=1, layers=tuple(layers), first=False, last=False, stage=0, kinds=ds_ep.layer_kinds(scfg.model, tuple(layers))
  )
  sp = ds_ep.StagePrograms(scfg, [ch], mesh)
  cp = ds_ep.ChunkPrograms(scfg, ch, mesh, mbs, seq, mbs * seq)
  rng = np.random.default_rng(seed + 1)
  put = lambda a: jax.device_put(jnp.asarray(a, scfg.model.dtype), NamedSharding(mesh, P(ds_ep.FE, None, None)))
  x, dy = (put(rng.standard_normal((mbs, seq, scfg.model.emb_dim))) for _ in range(2))
  wc = sp.gather(sp.init(jax.random.key(seed)))[0]
  y, res = cp.fwd(wc, x, jnp.int32(5))
  dx, gacc = cp.bwd(wc, x, res, dy, jnp.int32(5))
  return leaves([y, dx, gacc])


@needs_tokamax
@testing.needs_devices(4)
class TokamaxChunks(unittest.TestCase):
  """A middle chunk of two layers on a one-stage 2 x 2 mesh, fp32, top-k routing: its forward and backward
  programs -- the output, dx and every weight gradient -- under gmm_impl 'tokamax' against 'ragged_dot'.  Bit for
  bit, but DeepSeek's expert_weights 'stacked' (tokamax's forward reads each layer's block of the stacked weight in
  place, padded with the other layer's empty groups): fp32 rounding, within 1e-5 of each leaf (6e-7 measured).  A
  Qwen chunk's layers are subtrees of their own, so 'stacked' reads nothing in place there and is bitwise too."""

  def check(self, model, **kw):
    make = tiny_xla if model == "deepseek" else qtiny_xla
    if kw.get("batch_split"):
      kw = dict(kw, batch_split_impl="native")  # split_layers: the GPU default (SPLIT_ARGS)
    got, want = (chunk_run(make(random_routing=False, **impl, **kw)) for impl in (TOKAMAX, RAGGED))
    near = model == "deepseek" and kw.get("expert_weights", "stacked") == "stacked"
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
      if near:
        self.assertLessEqual(float(np.max(np.abs(a - b))), 1e-5 * float(np.max(np.abs(b))), i)
      else:
        np.testing.assert_array_equal(a, b, err_msg=f"{model} {kw}: leaf {i}")

  @pytest.mark.scheduled_only
  def test_deepseek(self):
    """DeepSeek-V3 layers: unsplit, and batch split under split_dw 'acc' and 'sum', each 'stacked' and 'sliced'."""
    for split in (False, True):
      for split_dw in ds_ep.SPLIT_DW if split else ("acc",):
        for ew in ds_ep.EXPERT_WEIGHTS:
          with self.subTest(batch_split=split, split_dw=split_dw, expert_weights=ew):
            self.check("deepseek", batch_split=split, split_dw=split_dw, expert_weights=ew)

  @pytest.mark.scheduled_only
  def test_qwen(self):
    """Qwen3.8 layers (two Gated DeltaNet ones): unsplit, and batch split under split_dw 'acc' and 'sum'."""
    for split, split_dw in ((False, "acc"), (True, "acc"), (True, "sum")):
      with self.subTest(batch_split=split, split_dw=split_dw):
        self.check("qwen", batch_split=split, split_dw=split_dw)


@testing.needs_devices(4)
class InplaceDwStacked(unittest.TestCase):
  """In-place dW accumulation's guard (ds_ep.ChunkPrograms' inplace_dw): it runs only where each layer's accumulator
  block is a whole buffer -- a Qwen chunk's layer subtrees, a DeepSeek chunk under layer_tree 'per_layer' or of one
  layer -- because blocks sliced out of one stacked [n, ...] leaf, each handed to gpu_kernels.ragged_dw (a jax.Ref
  jax-triton aliases), came out wrong on GB200 (jax 0.12.0.1, jax-triton 0.5.0: inf without donation, garbage with
  it).  On CPU ragged_dw is lax's and right on slices too, so the guard is pinned here by what the programs decide;
  the GPU case below is the minimal reproduction, an expected failure until jax / jax-triton fix it."""

  def flags(self, layers, layer_tree, model="deepseek"):
    args = (
        (TINY_ARGS if model == "deepseek" else Q_TINY_ARGS + ["--layer-split", ""])
        + [
            "--fsdp",
            "1",
            "--ep",
            "2",
            "--vv",
            "1",
            "--stages",
            "1",
            "--layers",
            str(layers),
            "--gbs",
            "8",
            "--mb",
            "2",
            *SPLIT_ARGS,
            "--acc-dtype",
            "bfloat16",
            "--layer-tree",
            layer_tree,
        ]
        + TOKAMAX_ARGS
    )
    p = dsp.build(dsp.make_parser().parse_args(args), meshes=dsp.stage_meshes(1, 1, 2))
    return [pr.inplace_dw for pr in p.progs]

  def test_stacked_chunks_of_several_layers_do_not(self):
    self.assertEqual(self.flags(2, "stacked"), [False])
    self.assertEqual(self.flags(1, "stacked"), [True])  # one layer: the block is the whole leaf
    self.assertEqual(self.flags(2, "per_layer"), [True])

  def test_qwen_chunks_do(self):
    self.assertEqual(self.flags(4, "stacked", model="qwen"), [True])  # a Qwen chunk is per layer either way

  @testing.requires_gpu("triton", "jax_triton", why="the in-place kernel (Triton)")
  def test_whole_buffer_on_gpu(self):
    from maxtext.experimental.mpmd.jax import gpu_kernels

    E, D, F, R = 8, 256, 128, 1024
    x = jax.random.normal(jax.random.key(0), (R, D), jnp.bfloat16)
    g = jax.random.normal(jax.random.key(1), (R, F), jnp.bfloat16)
    gs = jnp.full((E,), R // 2 // E, jnp.int32)
    acc = jax.random.normal(jax.random.key(2), (E, D, F), jnp.float32).astype(jnp.bfloat16)
    want = acc.astype(jnp.float32) + jax.lax.ragged_dot_general(
        x, g, gs, gpu_kernels._CONTRACTING, preferred_element_type=jnp.float32
    )
    got = jax.jit(gpu_kernels.ragged_dw, donate_argnums=(3,))(x, g, gs, acc + 0).astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-2, atol=2e-2)

  @unittest.expectedFailure
  @testing.requires_gpu("triton", "jax_triton", why="the in-place kernel (Triton)")
  def test_stacked_slices_on_gpu(self):
    """Two blocks of one stacked buffer through the in-place kernel: wrong on GB200 (the hazard the guard avoids).
    An unexpected success here means jax / jax-triton handle it now, and the guard can go."""
    from maxtext.experimental.mpmd.jax import gpu_kernels

    E, D, F, R = 8, 256, 128, 1024
    x = jax.random.normal(jax.random.key(0), (R, D), jnp.bfloat16)
    g = jax.random.normal(jax.random.key(1), (R, F), jnp.bfloat16)
    gs = jnp.full((E,), R // 2 // E, jnp.int32)
    stack = jax.random.normal(jax.random.key(2), (1, 2, E, D, F), jnp.float32).astype(jnp.bfloat16)

    def f(stack, use_kernel):
      outs = []
      for l in range(2):
        start = stack[0, l]
        if use_kernel:
          outs.append(gpu_kernels.ragged_dw(x * (l + 1), g, gs, start))
        else:
          dw = jax.lax.ragged_dot_general(
              x * (l + 1), g, gs, gpu_kernels._CONTRACTING, preferred_element_type=jnp.float32
          )
          outs.append((start.astype(jnp.float32) + dw).astype(start.dtype))
      return jnp.stack(outs)[None]

    want = jax.jit(lambda s: f(s, False))(stack).astype(jnp.float32)
    got = jax.jit(lambda s: f(s, True))(stack).astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-2, atol=2e-2)


# ------------------------------------------------------------------ the whole pipeline
# TINY_ARGS at the port's first measured setup: fp32 (compute and reduce-scatters), top-k routing, batch split,
# fsdp 2 x ep 2, two stages of two one-layer chunks (V = 2), two microbatches of 8 rows (two a device).
# --batch-split as it runs on GPU: ds_ep.split_layers (--batch-split-impl native, the GPU default: on_gpu), which these
# CPU tests name -- DeepSeek-V3's default elsewhere is Lineage's layers, whose TPU kernels do not run here.
SPLIT_ARGS = ["--batch-split", "--batch-split-impl", "native"]
E2E_ARGS = (
    TINY_ARGS
    + ("--fsdp 2 --ep 2 --vv 2 --layers 4 --gbs 16 --mb 2 --dtype float32 " "--rs-dtype float32").split()
    + SPLIT_ARGS
)
# Q_TINY's Qwen3.8 layers alike: three Gated DeltaNet chunks, then the attention one.
Q_E2E_ARGS = Q_TINY_ARGS + ["--layer-split", ""] + E2E_ARGS[len(TINY_ARGS) :] + ["--attn", "reference"]
TOKAMAX_ARGS = ["--gmm-impl", "tokamax", "--tokamax-gmm", "xla", "--ragged-sort", "off"]
RAGGED_ARGS = ["--gmm-impl", "ragged_dot", "--ragged-sort", "off"]


# A routed-expert weight's leaf in a chunk tree (DeepSeek-V3's 'moe', a Qwen3.5-family layer's 'routed_experts').
ROUTED_LEAF = re.compile(r"\['(moe|routed_experts)'\]\['(wi_0|wi_1|wo)'\]$")


def pipeline_step(args):
  """One loss-and-gradients pass, with the fusion plan, of the pipeline `args` builds, from init(key 1) on
  synthetic_batch(3) -> (its StageConfig, the loss, the stage sumsqs, every reduced
  gradient leaf by chunk as (path, fp32 host array), whether a chunk's accumulating backward adds the routed
  experts' dW into the step's accumulator inside the dW kernel (a ChunkPrograms' inplace_dw))."""
  p = dsp.build(dsp.make_parser().parse_args(args))
  grads, sqs, lacc = run_pipe(p, 1, 3)
  flat = jax.tree_util.tree_flatten_with_path(p.chunk_trees(grads))[0]
  return (
      p.scfg,
      float(jnp.sum(lacc)),
      [float(s) for s in sqs],
      [(jax.tree_util.keystr(k), leaves(v)[0]) for k, v in flat],
      any(getattr(pr, "inplace_dw", False) for pr in p.progs),
  )


@needs_tokamax
@pytest.mark.scheduled_only
@testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
class TokamaxPipeline(unittest.TestCase):
  """The whole pipeline from the command line, --gmm-impl tokamax --tokamax-gmm xla against --gmm-impl ragged_dot
  --ragged-sort off: a step's loss, stage sumsqs and every reduced gradient leaf (what --grad-digest digests)."""

  def check(self, args, rel=0.0, loss_rel=0.0, inplace=None):
    """Bit for bit, or the loss to `loss_rel` and every gradient leaf to `rel` of its largest magnitude (the stage
    sumsqs to twice that).  Where tokamax's accumulating backwards add the routed experts' dW into the step's
    accumulator inside the dW kernel (a ChunkPrograms' inplace_dw) under batch split -- the accumulator and the
    halves' dWs then associate as (acc + dW_mb1) + dW_mb0, not acc + (dW_mb1 + dW_mb0) -- those leaves to 8 fp32
    ulps; unsplit (one dW a layer) bit for bit.  `inplace`: whether a chunk's programs must (True) or must not
    (False) take in-place dW, None either.  -> tokamax's StageConfig."""
    got, want = pipeline_step(args + TOKAMAX_ARGS), pipeline_step(args + RAGGED_ARGS)
    self.assertEqual((got[0].gmm_impl, got[0].tokamax_gmm, got[0].use_ragged_sort), ("tokamax", "xla", False))
    self.assertEqual((want[0].gmm_impl, want[0].use_ragged_sort, want[4]), ("ragged_dot", False, False))
    if inplace is not None:
      self.assertEqual(got[4], inplace, "in-place dW")
    routed = 2.0**-20 if got[4] and got[0].batch_split else 0.0
    self.assertLessEqual(abs(got[1] - want[1]), loss_rel * abs(want[1]), "loss")
    np.testing.assert_allclose(got[2], want[2], rtol=2 * max(rel, routed), atol=0, err_msg="stage sumsqs")
    for (path, a), (path_b, b) in zip(got[3], want[3], strict=True):
      self.assertEqual(path, path_b)
      tol = max(rel, routed) if ROUTED_LEAF.search(path) else rel
      err, top = float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))
      self.assertLessEqual(err, tol * top, f"gradient leaf {path}: {err:.3g} of {top:.3g}")
    return got[0]

  def test_deepseek_batch_split(self):
    """DeepSeek-V3 layers at the port's first measured setup (E2E_ARGS: fp32, top-k, batch split with split_dw acc,
    fsdp 2 x ep 2, V = 2, expert_weights stacked over one-layer chunks): bit for bit, but the routed experts' dW
    leaves, to 8 fp32 ulps -- in-place dW (--inplace-dw auto, the default: 'on' here), which one-layer chunks take."""
    scfg = self.check(E2E_ARGS, inplace=True)
    self.assertEqual(
        (scfg.batch_split, scfg.split_dw, scfg.expert_weights, scfg.random_routing), (True, "acc", "stacked", False)
    )

  def test_deepseek_ep1(self):
    """EP 1 and dropless -- every compact-buffer row inside a group, so _tokamax_experts skips its masks, where
    ragged_dot runs on the argsort buffer -- with a dense layer before chunk 0's MoE layer (--layer-split 2,1,1,1),
    per-layer trees, batch split off: bit for bit (unsplit, in-place dW adds one dW a layer).
    SaveResiduals.test_ep1_dense_layer holds the same tokamax run with --save-residuals on to it."""
    # appended: E2E_ARGS holds --ep twice (TINY_ARGS's, then its own), and the last one counts
    args = [a for a in E2E_ARGS if a not in SPLIT_ARGS] + [
        "--ep",
        "1",
        "--layer-tree",
        "per_layer",
        "--layers",
        "5",
        "--dense-layers",
        "1",
        "--layer-split",
        "2,1,1,1",
    ]
    scfg = self.check(args, inplace=True)
    self.assertEqual((scfg.ep, scfg.model.first_num_dense_layers, scfg.batch_split), (1, 1, False))
    self.assertLessEqual(scfg.ragged_buffer_factor, 0)  # dropless

  @pytest.mark.scheduled_only
  def test_deepseek_batch_split_without_inplace_dw(self):
    """The same with --inplace-dw off (the halves' dW summed before the accumulator takes them, as ragged_dot's
    are): every leaf bit for bit."""
    scfg = self.check(E2E_ARGS + ["--inplace-dw", "off"], inplace=False)
    self.assertFalse(scfg.inplace_dw)

  @pytest.mark.scheduled_only
  def test_qwen_batch_split(self):
    """Qwen3.8 layers (Q_E2E_ARGS: three Gated DeltaNet chunks, then the attention one), the same setup: bit for
    bit, but the routed experts' dW leaves (in-place dW: 8 fp32 ulps)."""
    scfg = self.check(Q_E2E_ARGS, inplace=True)
    self.assertTrue(ds_ep.is_qwen(scfg.model) and scfg.batch_split)

  @pytest.mark.scheduled_only
  def test_variants(self):
    """DeepSeek-V3 layers unsplit (in-place dW adds one dW a layer), under split_dw 'sum' and expert_weights
    'sliced', random routing, route_weights 'experts' (the compact buffer's _row_weights and _sum_rows against autodiff
    of the argsort rows' weights and plain sum), and in two-layer chunks: bit for bit -- but the batch-split one-layer
    chunks' routed dW leaves (in-place dW: 8 fp32 ulps) and the two-layer chunks' 'stacked' reads (tokamax's padded
    groups: fp32 rounding; stacked chunks of several layers take no in-place dW)."""
    no_split = [a for a in E2E_ARGS if a != "--batch-split"]
    for what, args, rel in (
        ("unsplit", no_split, 0.0),
        ("sum, sliced", E2E_ARGS + ["--split-dw", "sum", "--expert-weights", "sliced"], 0.0),
        ("random routing", E2E_ARGS + ["--routing", "random"], 0.0),
        ("route weights experts", E2E_ARGS + ["--route-weights", "experts"], 0.0),
        ("two-layer chunks", E2E_ARGS + ["--layers", "8"], 1e-5),
    ):
      with self.subTest(what):
        self.check(args, rel=rel, loss_rel=rel / 10)

  @pytest.mark.scheduled_only
  def test_bf16(self):
    """bf16 compute (the reduce-scatters in fp32, as tiny()'s), DeepSeek-V3 and Qwen3.8 layers: the loss to 1e-6,
    every gradient leaf within four bf16 ulps of its largest magnitude -- XLA:CPU's excess precision rounds top-k's
    renormalised weights apart in the two programs (DeepSeek-V3's leaves by up to 1.9 ulps measured, Qwen3.8's not
    at all)."""
    for args in (E2E_ARGS, Q_E2E_ARGS):
      with self.subTest(qwen=args is Q_E2E_ARGS):
        self.check(args + ["--dtype", "bfloat16"], rel=4 * BF16_EPS, loss_rel=1e-6)

  @pytest.mark.scheduled_only
  def test_bf16_bitwise_without_excess_precision(self):
    """With XLA:CPU's excess precision off (--xla_allow_excess_precision=false, in a fresh interpreter: XLA reads its
    flags once) a bf16 step under split_dw 'sum' is bit for bit too: the bf16 differences are XLA's fusion-dependent
    rounding, not the port's."""
    script = textwrap.dedent(
        """
        import json
        from tests.unit.experimental.mpmd.jax import gpu_port_test as t
        args = t.E2E_ARGS + ['--dtype', 'bfloat16', '--rs-dtype', 'bfloat16', '--split-dw', 'sum']
        got, want = t.pipeline_step(args + t.TOKAMAX_ARGS), t.pipeline_step(args + t.RAGGED_ARGS)
        print(json.dumps(dict(loss=got[1] == want[1], sumsq=got[2] == want[2],
                              leaves=[bool((a == b).all()) for (_, a), (_, b) in zip(got[3], want[3])])))
        """
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        PYTHONPATH=testing.child_pythonpath(__name__),
        XLA_FLAGS="--xla_force_host_platform_device_count=8 --xla_allow_excess_precision=false",
    )
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=1200)
    self.assertEqual(out.returncode, 0, out.stderr[-3000:])
    res = json.loads(out.stdout.strip().splitlines()[-1])
    self.assertTrue(res["loss"] and res["sumsq"] and all(res["leaves"]), res)


# ------------------------------------------------------------------ --layer-tree per_layer
def relayout(trees, src, dst):
  """Chunk-order trees in pipeline `src`'s layer tree -> pipeline `dst`'s: each chunk's MoE layers taken out of its
  'layers' (ds_ep.layer_weights) and put back (stack_layers), the values untouched."""
  out = []
  for cs, cd, t in zip(src.chunks, dst.chunks, trees, strict=True):
    if cs.n_moe:
      t = dict(
          t, layers=ds_ep.stack_layers([ds_ep.layer_weights(t["layers"], l, cs.kinds) for l in range(cs.n_moe)], cd.kinds)
      )
    out.append(t)
  return out


@pytest.mark.scheduled_only
@testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
class PerLayerTree(unittest.TestCase):
  """--layer-tree per_layer (a GPU default): a DeepSeek-V3 chunk's MoE layers one subtree a layer, as a Qwen3.8
  chunk's, instead of [n, ...] stacks -- its own reads (layer_weights), gradients (stack_layers) and in-place dW
  blocks.  TINY on two stages of one two-layer chunk each (fsdp 1 x ep 2, batch split, fp32, --direct-experts on),
  from the stacked pipeline's weights laid out per layer: a step's loss, stage sumsqs and every reduced gradient leaf
  (laid out back), and two AdamW steps' losses and norms, against the stacked layout on ragged_dot."""

  TAIL = [
      "--stages",
      "2",
      "--vv",
      "1",
      "--layers",
      "4",
      "--fsdp",
      "1",
      "--ep",
      "2",
      "--gbs",
      "8",
      "--mb",
      "2",
      "--dtype",
      "float32",
      "--rs-dtype",
      "float32",
      "--weight-dtype",
      "float32",
      *SPLIT_ARGS,
      "--direct-experts",
      "on",
  ]

  def run_from(self, p, trees, steps=2):
    """From the chunk trees `trees`: one loss-and-gradients pass on synthetic_batch(3) -> {loss, the stage sumsqs,
    the reduced gradients as chunk trees of fp32 host arrays, then `steps` steps' losses and gradient norms}."""
    params, opts, wc0 = p.init_state(p.params_from_chunks(trees))
    batch = p.synthetic_batch(3)
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
    out = dict(
        loss=float(jnp.sum(lacc)),
        sqs=[float(s) for s in sqs],
        losses=[],
        norms=[],
        grads=[jax.tree.map(lambda a: leaves(a)[0], t) for t in p.chunk_trees(grads)],
    )
    del grads
    lrs = p.lr_arrays(1e-3)
    for _ in range(steps):
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      out["losses"].append(float(jnp.sum(lacc)))
      out["norms"].append(float(gn))
    return out

  def check(self, kernels, routed):
    """The per-layer pipeline on `kernels` against the stacked one on ragged_dot: the loss bit for bit, every
    gradient leaf too but the routed experts', those to `routed` of their largest magnitude; the stage sumsqs to
    twice that, and to 8 fp32 ulps at least (a per-layer tree sums each layer's squares on its own: another order,
    an ulp apart measured on jax 0.11.1); the steps' losses and norms to 1e-6.  -> the per-layer pipeline."""
    ap = dsp.make_parser()
    st = dsp.build(ap.parse_args(TINY_ARGS + self.TAIL + RAGGED_ARGS + ["--layer-tree", "stacked"]))
    pl = dsp.build(ap.parse_args(TINY_ARGS + self.TAIL + kernels + ["--layer-tree", "per_layer"]))
    trees = [jax.tree.map(np.asarray, t) for t in st.chunk_trees(st.init(jax.random.key(1))[0])]
    self.assertEqual(
        ({ch.kinds for ch in st.chunks}, {ch.kinds for ch in pl.chunks}), ({()}, {(ds_ep.DEEPSEEK_KIND,) * 2})
    )
    self.assertTrue(pl.direct and pl.scfg.batch_split)
    want, got = self.run_from(st, trees), self.run_from(pl, relayout(trees, st, pl))
    self.assertEqual(got["loss"], want["loss"])
    np.testing.assert_allclose(got["sqs"], want["sqs"], rtol=max(2 * routed, 2.0**-20), atol=0, err_msg="stage sumsqs")
    for c, (a, b) in enumerate(zip(relayout(got["grads"], pl, st), want["grads"], strict=True)):
      for (path, x), (path_b, y) in zip(
          jax.tree_util.tree_flatten_with_path(a)[0], jax.tree_util.tree_flatten_with_path(b)[0], strict=True
      ):
        name = jax.tree_util.keystr(path)
        self.assertEqual(name, jax.tree_util.keystr(path_b))
        err, top = float(np.max(np.abs(np.asarray(x) - y))), float(np.max(np.abs(y)))
        tol = routed if ROUTED_LEAF.search(name) else 0.0
        self.assertLessEqual(err, tol * top, f"chunk {c} gradient leaf {name}: {err:.3g} of {top:.3g}")
    np.testing.assert_allclose(got["losses"], want["losses"], rtol=1e-6, atol=0, err_msg="step losses")
    np.testing.assert_allclose(got["norms"], want["norms"], rtol=1e-6, atol=0, err_msg="step norms")
    return pl

  @needs_tokamax
  def test_tokamax_inplace_dw(self):
    """The GPU's combination: tokamax's kernels with in-place dW, which per_layer alone turns on for a chunk of
    several DeepSeek-V3 layers (each layer's accumulator block a buffer of its own): the routed experts' gradient
    leaves to 8 fp32 ulps -- the halves' dW added in another order, as in TokamaxPipeline (under one measured) --
    every other leaf bit for bit."""
    pl = self.check(TOKAMAX_ARGS, 2.0**-20)
    self.assertEqual([pr.inplace_dw for pr in pl.progs], [True] * 2)

  def test_ragged_dot(self):
    """On ragged_dot's kernels (no in-place dW): every gradient leaf bit for bit."""
    self.check(RAGGED_ARGS, 0.0)


# ------------------------------------------------------------------ a capped buffer
@needs_tokamax
@testing.needs_devices(4)
class TokamaxRaggedBuffer(unittest.TestCase):
  """ragged_buffer_factor > 0 under gmm_impl 'tokamax': the cap applies to the compact buffer, whose rows are the
  shard's (token, choice) pairs in local expert order -- so a shard routed more than its buffer keeps the same first
  rows MaxText's truncated buffer keeps.  ragged_buffer_test's ring (EP 2, top-2 of 8; its skewed routing overflows
  shard 0), fp32, against its numpy MoE."""

  def scfg(self, factor):
    return dataclasses.replace(rb_scfg(-1.0), **TOKAMAX, ragged_buffer_factor=factor)

  def grads(self, sc, w, x):
    f, mesh = rb_ring_moe(sc)
    wspec, xspec = RB_SPECS
    put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
    probe = np.random.default_rng(1).standard_normal(x.shape).astype(np.float32)

    def loss(w, x):
      y = f(w, x)
      return jnp.sum(y * probe), y

    (_, y), g = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        jax.tree.map(put, w, wspec), put(x, xspec)
    )
    return y, g, probe

  def test_roomy_buffer_is_dropless(self):
    """A cap under the worst case (factor 1.5: 384 rows of 512) but over every shard's load (unskewed routing): the
    dropless output and the router's and the tokens' gradients bit for bit; the expert weights' to 8 fp32 ulps of
    each leaf (their dW contracts over the buffer's rows, 384 against 512, which XLA:CPU blocks differently)."""
    w, x = rb_inputs(skew=False)
    sc = self.scfg(1.5)
    self.assertEqual(ds_ep.ragged_buffer_rows(sc, RB_T), 384)
    self.assertTrue(numpy_moe(sc, w, x)[1].all(), "the test wants no drops at factor 1.5")
    (y, (gw, gx), _), (y0, (gw0, gx0), _) = self.grads(sc, w, x), self.grads(self.scfg(-1.0), w, x)
    assert_equal_leaves(self, (y, gw["gate"], gx), (y0, gw0["gate"], gx0), "output, router, tokens")
    for k in ds_ep.EXPERT_LEAVES:
      a, b = np.asarray(gw[k]), np.asarray(gw0[k])
      self.assertLessEqual(float(np.max(np.abs(a - b))), 2.0**-20 * float(np.max(np.abs(b))), k)

  def test_small_buffer_drops_like_maxtext(self):
    """Factors 0.5, 0.7 and 1.3 (128, 179 and 332 rows) overflow shard 0: the output is the numpy MoE that keeps
    each shard's first rows in global expert order, to fp32 rounding, and the drops show far above that."""
    w, x = rb_inputs()
    dropless, _ = numpy_moe(rb_scfg(-1.0), w, x)
    for f in DROPPING:
      with self.subTest(factor=f):
        sc = self.scfg(f)
        want, kept = numpy_moe(sc, w, x)
        self.assertGreater((~kept).sum(), 0, f"the test wants drops at factor {f}")
        got = np.asarray(self.grads(sc, w, x)[0])
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        self.assertGreater(np.abs(got - dropless).max(), 0.5)

  def test_dropped_pairs_contribute_nothing(self):
    """The gradients of a dropping buffer (factors 0.5 and 0.7) in the tokens, the router and the expert weights:
    jax.grad of a dense statement that weighs each (token, choice) by whether MaxText's buffer keeps it -- a dropped
    pair moves no output, no expert weight and no routing weight -- to fp32 rounding."""
    w, x = rb_inputs()
    for factor in (0.5, 0.7):
      with self.subTest(factor=factor):
        sc = self.scfg(factor)
        _, kept = numpy_moe(sc, w, x)
        _, (gw, gx), probe = self.grads(sc, w, x)

        def ref(w, x, kept=kept):
          xf = x.reshape(RB_T, -1)
          idx, wts = rb_routing(sc, w, xf)
          wts = wts * jnp.asarray(kept.reshape(idx.shape), jnp.float32)
          hp = jax.lax.Precision.HIGHEST
          h0 = jnp.einsum("td,tkdf->tkf", xf, w["wi_0"][idx], precision=hp)
          h1 = jnp.einsum("td,tkdf->tkf", xf, w["wi_1"][idx], precision=hp)
          out = jnp.einsum("tkf,tkfd->tkd", jax.nn.silu(h0) * h1, w["wo"][idx], precision=hp)
          return jnp.einsum("tkd,tk->td", out, wts, precision=hp).reshape(x.shape)

        rw, rx = jax.grad(lambda w, x: jnp.sum(ref(w, x) * probe), argnums=(0, 1))(w, x)
        for name, a, b in (
            [("x", gx, rx)]
            + [(k, gw[k], rw[k]) for k in ds_ep.EXPERT_LEAVES]
            + [("gate", gw["gate"]["kernel"], rw["gate"]["kernel"])]
        ):
          a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
          self.assertLess(np.abs(a - b).max(), 1e-5 * np.abs(b).max(), name)


# ------------------------------------------------------------------ --platform
# TINY_ARGS's model and shape flags without its kernel flags, so that --platform's defaults show.
SHAPE_ARGS = (
    "--emb 256 --heads 2 --q-lora 128 --kv-lora 128 --moe-mlp 256 --experts 8 --topk 2 --vocab 512 --seq 256 "
    "--sa-block 128 --sa-block-kv-compute 128 --routing topk --stages 2 --vv 1 --layers 2 --fsdp 1 --ep 2 "
    "--gbs 4 --mb 2"
).split()
# The parser's own defaults of the flags GPU_DEFAULTS replaces -- TPU's kernels and collectives -- which tpu and cpu
# keep (every CPU test's command line relies on them).
TPU_DEFAULTS = {
    "attn": "tokamax_splash",
    "gmm_impl": "megablox",
    "ragged_sort": "auto",
    "embed": "iota",
    "fsdp_collectives": "auto",
    "ep_collectives": "auto",
    "route_weights": "combine",
}
# GPU_DEFAULTS as flags: every GPU default with its value (change the two together).
GPU_DEFAULT_FLAGS = (
    "--attn cudnn --gmm-impl tokamax --ragged-sort off --embed gather --fsdp-collectives xla "
    "--ep-collectives xla --layer-tree per_layer --norm-reduce allreduce --gdn-conv shift "
    "--gdn-scan triton --direct-experts on --combine-impl triton --route-weights experts"
)


@contextlib.contextmanager
def jax_platforms(value):
  """JAX_PLATFORMS set to `value` (None: unset) for the duration, and jax.config's jax_platforms -- which
  resolve_platform reads first, and which the process's JAX_PLATFORMS (cpu in a CPU test run) set when jax was
  imported -- to it too, as a process started with that environment would see it; jax's own backend, chosen at its
  first use, is not touched."""
  with mock.patch.dict(os.environ), mock.patch.object(dsp, "_config_platforms", lambda: value or None):
    os.environ.pop("JAX_PLATFORMS", None)
    if value is not None:
      os.environ["JAX_PLATFORMS"] = value
    yield


def flags_of(a, keys):
  return {k: getattr(a, k) for k in keys}


@testing.needs_devices(4)
class PlatformDefaults(unittest.TestCase):
  """ds_ep_pipeline's --platform: which platform a run is for, decided before any jax backend starts, and the kernel
  and collective defaults it brings (GPU_DEFAULTS); --inplace-dw auto resolved by device; build() wiring the GPU flags
  into the StageConfig.  Nothing is compiled."""

  def test_resolve_platform(self):
    """tpu, gpu and cpu are taken as given; auto reads JAX_PLATFORMS's first entry (cuda, rocm and gpu are 'gpu'),
    else what is installed -- found, not imported: jax.distributed must start before any backend -- libtpu first,
    then a jax CUDA / ROCm plugin, else 'cpu'.  A module whose spec cannot be read counts as absent."""
    for given in ("tpu", "gpu", "cpu"):
      with jax_platforms("rocm"):
        self.assertEqual(dsp.resolve_platform(given), given)
    for env, want in (
        ("cpu", "cpu"),
        ("cuda", "gpu"),
        ("cuda,cpu", "gpu"),
        (" CUDA ", "gpu"),
        ("rocm", "gpu"),
        ("gpu", "gpu"),
        ("tpu", "tpu"),
        ("tpu,cpu", "tpu"),
        ("cpu,cuda", "cpu"),
    ):
      with self.subTest(JAX_PLATFORMS=env), jax_platforms(env):
        self.assertEqual((dsp.resolve_platform("auto"), dsp.resolve_platform()), (want, want))

    def finder(present, broken):
      def find_spec(name, *a, **k):
        if name in broken:
          raise ValueError(f"{name}.__spec__ is None")
        return types.SimpleNamespace(name=name) if name in present else None

      return find_spec

    for present, broken, want in (
        ({"libtpu"}, (), "tpu"),
        ({"libtpu", "jax_cuda12_plugin"}, (), "tpu"),
        ({"jax_cuda13_plugin"}, (), "gpu"),
        ({"jax_cuda12_plugin"}, (), "gpu"),
        ({"jax_rocm7_plugin"}, (), "gpu"),
        ({"jax_rocm60_plugin"}, (), "gpu"),
        (set(), (), "cpu"),
        ({"jax_cuda12_plugin"}, ("libtpu",), "gpu"),
        (set(), ("libtpu", "jax_cuda13_plugin"), "cpu"),
    ):
      for env in (None, ""):
        with (
            self.subTest(installed=sorted(present), broken=broken, JAX_PLATFORMS=env),
            jax_platforms(env),
            mock.patch("importlib.util.find_spec", finder(present, broken)),
        ):
          self.assertEqual(dsp.resolve_platform("auto"), want)

  def test_gpu_defaults(self):
    """--platform gpu, or auto on a cuda process: every flag left out takes GPU_DEFAULTS -- cuDNN attention,
    tokamax's grouped matmuls on the compact buffer and so no MaxText ragged sort, a gathered embedding, lax (NCCL)
    collectives -- a flag given keeps its value, --tokamax-gmm parses, and the namespace keeps --platform as given
    (main resolves it again; the run's provenance records the flag).  Each GPU default is a value its flag accepts,
    and the help lists them."""
    ap = dsp.make_parser()
    gpu = dsp.GPU_DEFAULTS
    self.assertTrue(set(TPU_DEFAULTS) <= set(gpu))
    self.assertEqual((gpu["attn"], gpu["gmm_impl"], gpu["ragged_sort"]), ("cudnn", "tokamax", "off"))
    for act in ap._actions:
      if act.dest in gpu:
        if act.choices is not None:
          self.assertIn(gpu[act.dest], act.choices, act.dest)
        if callable(act.type):
          act.type(gpu[act.dest])
    self.assertIn("--gmm-impl tokamax", " ".join(ap.format_help().split()))
    given = ["--attn", "reference", "--gmm-impl", "ragged_dot", "--embed", "iota", "--ep-collectives", "ppermute"]
    for platform, env in (("gpu", "cpu"), ("auto", "cuda"), ("auto", "rocm")):
      with self.subTest(platform=platform, JAX_PLATFORMS=env), jax_platforms(env):
        a = ap.parse_args(SHAPE_ARGS + ["--platform", platform])
        self.assertEqual((a.platform, dsp.resolve_platform(a.platform), flags_of(a, gpu)), (platform, "gpu", gpu))
        self.assertEqual(a.tokamax_gmm, gpu.get("tokamax_gmm", ap.get_default("tokamax_gmm")))
        a = ap.parse_args(SHAPE_ARGS + ["--platform", platform] + given)
        self.assertEqual(
            flags_of(a, gpu),
            {**gpu, "attn": "reference", "gmm_impl": "ragged_dot", "embed": "iota", "ep_collectives": "ppermute"},
        )
        for impl in ds_ep.TOKAMAX_GMM:
          self.assertEqual(ap.parse_args(SHAPE_ARGS + ["--platform", platform, "--tokamax-gmm", impl]).tokamax_gmm, impl)
    with jax_platforms("cpu"):  # a Qwen3.5-family --model's dims and the GPU defaults together
      a = ap.parse_args(["--model", "qwen3.8-2.4t-a95b", "--platform", "gpu"])
      self.assertEqual((a.emb, a.experts, a.topk, flags_of(a, gpu)), (8192, 512, 10, gpu))
      self.assertEqual(ap.parse_args(["--model", "qwen3.8-2.4t-a95b", "--platform", "gpu", "--emb", "256"]).emb, 256)
    for bad in (["--platform", "metal"], ["--tokamax-gmm", "cutlass"]):
      with self.subTest(bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        ap.parse_args(SHAPE_ARGS + bad)

  def test_gpu_defaults_are_the_listed_ones(self):
    """GPU_DEFAULTS is GPU_DEFAULT_FLAGS, every GPU default with its value and nothing else (the dict once gained
    --direct-experts and then --combine-impl without the documentation's list, which this list mirrors)."""
    words = GPU_DEFAULT_FLAGS.split()
    flag = {
        act.dest: next(s for s in act.option_strings if s.startswith("--"))
        for act in dsp.make_parser()._actions
        if act.dest in dsp.GPU_DEFAULTS
    }
    self.assertEqual(len(words) % 2, 0, words)
    self.assertEqual(sorted(zip(words[::2], words[1::2])), sorted((flag[k], v) for k, v in dsp.GPU_DEFAULTS.items()))

  def test_inplace_dw_auto(self):
    """--inplace-dw auto (the default) is 'all' where gpu_kernels.ragged_dw's tcgen05 kernel takes an fp32
    accumulator -- an sm_100 GPU with MOSAIC_F32_ACC -- and 'on' elsewhere (this CPU, an A100, or the kernel turned
    off); a value given stays."""
    from maxtext.experimental.mpmd.jax import gpu_kernels

    self.assertEqual(dsp.make_parser().parse_args(SHAPE_ARGS).inplace_dw, "auto")
    self.assertEqual([dsp.resolve_inplace_dw(v) for v in ("auto", "on", "off", "all")], ["on", "on", "off", "all"])
    for supported, f32, want in ((True, True, "all"), (False, True, "on"), (True, False, "on")):
      with (
          self.subTest(sm100=supported, f32=f32),
          mock.patch.object(jax, "default_backend", return_value="gpu"),
          mock.patch.object(gpu_kernels, "mosaic_supported", return_value=supported),
          mock.patch.object(gpu_kernels, "MOSAIC_F32_ACC", f32),
      ):
        self.assertEqual(dsp.resolve_inplace_dw("auto"), want)
        self.assertEqual(dsp.resolve_inplace_dw("on"), "on")

  def test_tpu_and_cpu_keep_tpu_defaults(self):
    """--platform tpu and cpu, and auto on a cpu or tpu process (every CPU test's case), keep the TPU defaults: tokamax
    splash, megablox, the ragged sort at EP > 1, the iota embedding, collectives chosen per stage.  A GPU parse
    leaves the parser's defaults as they were (it swaps them for one parse only)."""
    ap = dsp.make_parser()
    self.assertEqual({k: ap.get_default(k) for k in TPU_DEFAULTS}, TPU_DEFAULTS)
    for platform, env, want in (
        ("tpu", "cuda", "tpu"),
        ("cpu", "cuda", "cpu"),
        ("auto", "cpu", "cpu"),
        ("auto", "tpu", "tpu"),
    ):
      with self.subTest(platform=platform, JAX_PLATFORMS=env), jax_platforms(env):
        ap.parse_args(SHAPE_ARGS + ["--platform", "gpu"])
        a = ap.parse_args(SHAPE_ARGS + ["--platform", platform])
        self.assertEqual(
            (a.platform, dsp.resolve_platform(a.platform), flags_of(a, TPU_DEFAULTS)), (platform, want, TPU_DEFAULTS)
        )
        self.assertEqual(
            {k: ap.get_default(k) for k in dsp.GPU_DEFAULTS},
            {k: dsp.make_parser().get_default(k) for k in dsp.GPU_DEFAULTS},
        )

  def test_build(self):
    """build() hands --tokamax-gmm to the StageConfig and resolves --ragged-sort auto to off under --gmm-impl tokamax
    (on under the others at EP > 1: MaxText's ring sort is megablox's), where an explicit --ragged-sort on is
    refused; a namespace without tokamax_gmm (a caller's own) takes StageConfig's default.  A CPU build of a
    --platform gpu command line holds every GPU default -- for a Qwen3.8 model too, whose config accepts 'cudnn' --
    and its explicit flags."""
    ap = dsp.make_parser()
    with jax_platforms("cpu"):
      for impl in ds_ep.TOKAMAX_GMM:
        s = dsp.build(
            ap.parse_args(TINY_ARGS + ["--gmm-impl", "tokamax", "--tokamax-gmm", impl, "--ragged-sort", "auto"])
        ).scfg
        self.assertEqual((s.gmm_impl, s.tokamax_gmm, s.use_ragged_sort), ("tokamax", impl, False))
      s = dsp.build(ap.parse_args(TINY_ARGS + ["--ragged-sort", "auto"])).scfg
      self.assertEqual((s.gmm_impl, s.use_ragged_sort), ("ragged_dot", True))
      with self.assertRaisesRegex(ValueError, "use use_ragged_sort=False"):
        dsp.build(ap.parse_args(TINY_ARGS + ["--gmm-impl", "tokamax", "--ragged-sort", "on"]))
      a = ap.parse_args(TINY_ARGS + ["--gmm-impl", "tokamax"])
      del a.tokamax_gmm
      self.assertEqual(dsp.build(a).scfg.tokamax_gmm, ds_ep.StageConfig.__dataclass_fields__["tokamax_gmm"].default)
      s = dsp.build(ap.parse_args(SHAPE_ARGS + ["--platform", "gpu"])).scfg
      self.assertEqual(
          (
              s.gmm_impl,
              s.tokamax_gmm,
              s.use_ragged_sort,
              s.iota_embed,
              s.fsdp_collectives,
              s.ep_collectives,
              s.model.attention_kernel,
              s.route_weights,
          ),
          (
              "tokamax",
              dsp.GPU_DEFAULTS.get("tokamax_gmm", ap.get_default("tokamax_gmm")),
              False,
              False,
              "xla",
              "xla",
              "cudnn",
              "experts",
          ),
      )
      s = dsp.build(ap.parse_args(Q_TINY_ARGS + ["--platform", "gpu"])).scfg
      self.assertTrue(ds_ep.is_qwen(s.model))
      self.assertEqual(
          (s.gmm_impl, s.use_ragged_sort, s.iota_embed, s.ep_collectives, s.model.attention_kernel, s.route_weights),
          ("ragged_dot", False, False, "xla", "cudnn", "experts"),
      )
      self.assertEqual(dsp.build(ap.parse_args(TINY_ARGS)).scfg.route_weights, "combine")

  def test_batch_split_impl(self):
    """--batch-split's layers by platform: Lineage's (DeepSeek-V3's default) run TPU kernels only, so on GPU
    (--platform gpu, or auto on a cuda process) the model's batch split is ds_ep.split_layers -- 'native', for
    DeepSeek-V3 as for the Qwen3.5 family; on tpu and cpu DeepSeek-V3's stays Lineage's.  An explicit
    --batch-split-impl stands, but native for DeepSeek-V3 on TPU devices -- a TPU VM's, or Pathways' (their client's
    runtime_type proxy/...), whatever --platform says -- where Lineage's layers run its batch split: build refuses it
    there (native_split_refusal), before anything is built.  build() of a GPU command line runs split_layers."""
    ap = dsp.make_parser()
    tpu = types.SimpleNamespace(platform="tpu", device_kind="TPU7x")
    pathways = types.SimpleNamespace(
        platform="tpu", device_kind="TPU7x", client=types.SimpleNamespace(runtime_type="proxy/pathways")
    )
    gpu = types.SimpleNamespace(platform="gpu", device_kind="NVIDIA GB200")
    for platform, env, want in (
        ("gpu", "cpu", "native"),
        ("auto", "cuda", "native"),
        ("tpu", "cuda", "lineage"),
        ("cpu", "cuda", "lineage"),
        ("auto", "cpu", "lineage"),
        ("auto", "tpu", "lineage"),
    ):
      with self.subTest(platform=platform, JAX_PLATFORMS=env), jax_platforms(env):
        a = ap.parse_args(SHAPE_ARGS + ["--platform", platform, "--batch-split"])
        self.assertEqual(
            (dsp.on_gpu(a), dsp.model_batch_split_impl(a), dsp.batch_split_impl(a)), (want == "native", want, want)
        )
        self.assertEqual(dsp.uses_lineage(a), want == "lineage")
        self.assertIsNone(dsp.native_split_refusal(a))  # the model's own batch split, wherever
        a = ap.parse_args(SHAPE_ARGS + ["--platform", platform, "--batch-split", "--batch-split-impl", "native"])
        # refused where Lineage's layers run it: by the devices, not --platform (this process's are CPU devices)
        self.assertIsNone(dsp.native_split_refusal(a))
        self.assertIsNone(dsp.native_split_refusal(a, [gpu] * 4))
        for devs in ([tpu] * 4, [pathways] * 4):
          self.assertIsNotNone(dsp.native_split_refusal(a, devs))
        q = ap.parse_args(Q_TINY_ARGS + ["--platform", platform, "--batch-split"])
        self.assertEqual(dsp.batch_split_impl(q), "native")
        self.assertIsNone(dsp.native_split_refusal(q, [tpu] * 4))
    native = ["--batch-split", "--batch-split-impl", "native"]
    with mock.patch.object(ds_ep, "StageConfig", side_effect=AssertionError("built")):
      with (
          mock.patch.object(dsp.jax, "devices", lambda *_: [pathways] * 8),
          self.assertRaisesRegex(ValueError, r"^--batch-split-impl native: on TPU DeepSeek-V3's batch split runs"),
      ):
        dsp.build(ap.parse_args(TINY_ARGS + ["--gbs", "8"] + native))
      for platform in ("auto", "tpu", "cpu", "gpu"):  # on this process's CPU devices not refused: the build goes on
        with self.subTest(platform=platform), self.assertRaisesRegex(AssertionError, "built"):
          dsp.build(ap.parse_args(TINY_ARGS + ["--platform", platform, "--gbs", "8"] + native))
    with jax_platforms("cpu"):
      s = dsp.build(
          ap.parse_args(
              TINY_ARGS + ["--platform", "gpu", "--batch-split", "--dtype", "float32", "--gbs", "8", "--mb", "2"]
          )
      ).scfg
      self.assertEqual((s.batch_split, s.batch_split_impl, s.lineage_split), (True, "native", False))

  def test_native_split_on_cpu_with_libtpu_installed(self):
    """MaxText's CPU CI: libtpu installed and JAX_PLATFORMS unset, so --platform auto resolves to tpu -- the TPU
    defaults, as on CPU -- while jax runs on CPU devices.  DeepSeek-V3's native batch split builds there: the refusal
    reads the devices, not resolve_platform."""
    ap = dsp.make_parser()
    found = lambda name, *a, **k: types.SimpleNamespace(name=name) if name == "libtpu" else None
    with jax_platforms(None):
      with mock.patch("importlib.util.find_spec", found):
        a = ap.parse_args(TINY_ARGS + ["--gbs", "8", "--batch-split", "--batch-split-impl", "native"])
        self.assertEqual((a.platform, dsp.resolve_platform(a.platform)), ("auto", "tpu"))
        self.assertIsNone(dsp.native_split_refusal(a))
      s = dsp.build(a).scfg
    self.assertEqual((s.batch_split, s.batch_split_impl, s.lineage_split), (True, "native", False))

  def test_platform_follows_argparse_precedence(self):
    """ModelParser's peek reads --model and --platform as argparse does: the arguments', else a given namespace's,
    else the default -- so a namespace's platform does not override the arguments' (nor its model theirs)."""
    ap = dsp.make_parser()
    with jax_platforms("cpu"):
      for ns_platform, args, want in (
          ("gpu", ["--platform", "tpu"], "tpu"),
          ("gpu", [], "gpu"),
          ("tpu", ["--platform", "gpu"], "gpu"),
          (None, [], "auto"),
      ):
        with self.subTest(namespace=ns_platform, args=args):
          ns = argparse.Namespace(platform=ns_platform) if ns_platform else None
          a = ap.parse_args(SHAPE_ARGS + args, namespace=ns)
          self.assertEqual(a.platform, want)
          self.assertEqual(a.gmm_impl, "tokamax" if dsp.resolve_platform(want) == "gpu" else TPU_DEFAULTS["gmm_impl"])


# ------------------------------------------------------------------ init_distributed on a GPU pod
class GpuCoordinator(unittest.TestCase):
  """init_distributed on a pod of a GKE GPU JobSet, where no cluster detection finds the coordinator:
  the JobSet names pod 0 (MAXTEXT_MPMD_COORD_HOST) and each step its port, which make JAX_COORDINATOR_ADDRESS --
  jax.distributed.initialize mocked."""

  READ = (
      "MAXTEXT_MPMD_COORD_HOST",
      "JAX_COORDINATOR_ADDRESS",
      "JAX_COORDINATOR_PORT",
      "MPMD_COORD_PORT",
      "MPMD_INIT_TIMEOUT",
      "JAX_NUM_PROCESSES",
      "JAX_PROCESS_ID",
      "JAX_CROSS_HOST_TRANSFER_SOCKET_ADDRESS",
      "JAX_CROSS_HOST_TRANSPORT_ADDRESSES",
      "JAX_FORCE_DCN_CROSS_HOST_TRANSFERS",
      "JAX_CROSS_HOST_TRANSFER_TRANSFER_SIZE",
      "JAX_CROSS_HOST_TRANSFER_TIMEOUT_SECONDS",
  )
  HOST = "mpmd-run-slice-job-0-0.mpmd-run"  # pod 0 of JobSet mpmd-run: <jobset>-<job>-<index>-<pod>.<subdomain>
  # what a GKE GPU JobSet gives every pod: pod 0's name, the process count, and JAX_PROCESS_ID, which the pod's entry
  # script sets to its completion index
  POD = dict(MAXTEXT_MPMD_COORD_HOST=HOST, JAX_NUM_PROCESSES="4", JAX_PROCESS_ID="2")

  def run_init(self, **env):
    """-> (JAX_COORDINATOR_ADDRESS as jax.distributed.initialize would read it, its keyword arguments, the output)."""
    seen, out = [], io.StringIO()
    with (
        mock.patch.dict(os.environ, env),
        contextlib.redirect_stdout(out),
        mock.patch.object(
            jax.distributed, "initialize", lambda **kw: seen.append((os.environ.get("JAX_COORDINATOR_ADDRESS"), kw))
        ),
    ):
      for k in set(self.READ) - set(env):
        os.environ.pop(k, None)
      dsp.init_distributed()
    ((address, kw),) = seen
    return address, kw, out.getvalue()

  def test_address(self):
    """MAXTEXT_MPMD_COORD_HOST at the step's MPMD_COORD_PORT; a JAX_COORDINATOR_PORT already set wins (with
    init_distributed's warning when the step gives another); 8482 without either; a JAX_COORDINATOR_ADDRESS already
    set stands (an empty one does not); no MAXTEXT_MPMD_COORD_HOST (or an empty one), no address.  The process count and
    id pass through, and the processes wait 30 minutes for each other."""
    address, kw, out = self.run_init(**self.POD, MPMD_COORD_PORT="8503")
    self.assertEqual(
        (address, kw, out), (f"{self.HOST}:8503", dict(num_processes=4, process_id=2, initialization_timeout=1800), "")
    )
    self.assertEqual(self.run_init(**self.POD, JAX_COORDINATOR_PORT="1234")[0], f"{self.HOST}:1234")
    self.assertEqual(self.run_init(**self.POD)[0], f"{self.HOST}:8482")
    address, _, out = self.run_init(**self.POD, MPMD_COORD_PORT="8503", JAX_COORDINATOR_PORT="1234")
    self.assertEqual(address, f"{self.HOST}:1234")
    self.assertIn("JAX_COORDINATOR_PORT=1234 takes the place of this step's MPMD_COORD_PORT=8503", out)
    self.assertEqual(
        self.run_init(**self.POD, MPMD_COORD_PORT="8503", JAX_COORDINATOR_ADDRESS="10.0.0.7:9999")[0], "10.0.0.7:9999"
    )
    self.assertEqual(
        self.run_init(**self.POD, MPMD_COORD_PORT="8503", JAX_COORDINATOR_ADDRESS="")[0], f"{self.HOST}:8503"
    )
    for host in ({}, {"MAXTEXT_MPMD_COORD_HOST": ""}):
      self.assertIsNone(self.run_init(MPMD_COORD_PORT="8503", **host)[0], host)

  def test_jax_keeps_it(self):
    """What jax.distributed's own parameter detection (jax/_src/clusters) makes of the pod's environment, no cluster of
    its kinds detected (a GKE GPU pod's): the coordinator address init_distributed composed, the process count and
    id given, all devices visible."""
    from jax._src import clusters
    from jax._src.clusters import cloud_tpu_cluster as ctc

    keep = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAX_", "MEGASCALE_", "TPU_", "MAXTEXT_MPMD_", "MPMD_", "SLURM_", "OMPI_"))
        and k not in ("KUBERNETES_SERVICE_HOST", "JOB_COMPLETION_INDEX")
    }
    seen = []

    def initialize(**kw):  # jax.distributed.initialize up to the point it would connect
      seen.append(
          clusters.ClusterEnv.auto_detect_unset_distributed_params(
              os.environ.get("JAX_COORDINATOR_ADDRESS"),
              kw.get("num_processes"),
              kw.get("process_id"),
              None,
              None,
              kw["initialization_timeout"],
          )
      )

    with (
        mock.patch.dict(os.environ, {**keep, **self.POD, "MPMD_COORD_PORT": "8503"}, clear=True),
        mock.patch.object(jax.distributed, "initialize", initialize),
        mock.patch.object(ctc, "running_in_cloud_tpu_vm", False),
    ):
      dsp.init_distributed()
    self.assertEqual(seen, [(f"{self.HOST}:8503", 4, 2, None)])


# ------------------------------------------------------------------ every process writes the cache
class CacheEveryProcess(unittest.TestCase):
  """ds_ep_pipeline.cache_every_process (MPMD_CACHE_EVERY_PROCESS=1, for a GPU job whose nodes each put
  JAX_CACHE_DIR on a local SSD of their own).  jax writes persistent-cache entries from process 0 only; the patched write
  is jax's own put behind the same minimum compile time, from any process.  jax's internals are mocked
  (compilation_cache.put_executable_and_time, the distributed process id) and restored."""

  def test_patched_write(self):
    """The patch replaces compiler._cache_write once (a second call leaves it), with jax's signature, and the write
    reaches put_executable_and_time with jax's arguments on process 3 -- where jax's own write does not -- unless the
    compile was under the minimum time; a failing put warns, as jax's does, and does not raise."""
    from jax._src import compilation_cache as cc
    from jax._src import compiler, distributed
    from jax._src import config as jax_config

    original, sig = compiler._cache_write, inspect.signature(cc.put_executable_and_time)
    puts = []
    with (
        mock.patch.object(compiler, "_cache_write", original),
        mock.patch.object(cc, "put_executable_and_time", lambda *a, **k: puts.append(sig.bind(*a, **k).arguments)),
        mock.patch.object(distributed.global_state, "process_id", 3),
        jax_config.persistent_cache_min_compile_time_secs(2.0),
    ):
      original("key", 5.7, "jit_f", "backend", "exe")
      self.assertEqual(puts, [])  # jax's rule: process 3 writes nothing
      self.assertTrue(dsp.cache_every_process())
      patched = compiler._cache_write
      self.assertIsNot(patched, original)
      self.assertEqual(list(inspect.signature(patched).parameters), list(inspect.signature(original).parameters))
      self.assertFalse(dsp.cache_every_process())
      self.assertIs(compiler._cache_write, patched)
      patched("key", 5.7, "jit_f", "backend", "exe")
      patched("key2", 1.5, "jit_g", "backend", "exe")  # under the minimum compile time
      self.assertEqual(
          puts, [dict(cache_key="key", module_name="jit_f", executable="exe", backend="backend", compile_time=5)]
      )
    self.assertIs(compiler._cache_write, original)

    def fail(*a, **k):
      raise OSError("disk full")

    with (
        mock.patch.object(compiler, "_cache_write", original),
        mock.patch.object(cc, "put_executable_and_time", fail),
        jax_config.persistent_cache_min_compile_time_secs(0.0),
        warnings.catch_warnings(record=True) as w,
    ):
      warnings.simplefilter("always")
      self.assertTrue(dsp.cache_every_process())
      compiler._cache_write("key", 3.0, "jit_h", "backend", "exe")
    self.assertEqual(
        [str(x.message) for x in w], ["Error writing persistent compilation cache entry for 'jit_h': OSError: disk full"]
    )

  def test_without_jax_internals(self):
    """A jax without _cache_write or put_executable_and_time keeps its own rule: nothing is patched."""
    from jax._src import compilation_cache as cc
    from jax._src import compiler

    original = compiler._cache_write
    with mock.patch.object(compiler, "_cache_write", None):
      self.assertFalse(dsp.cache_every_process())
      self.assertIsNone(compiler._cache_write)
    with mock.patch.object(compiler, "_cache_write", original), mock.patch.object(cc, "put_executable_and_time", None):
      self.assertFalse(dsp.cache_every_process())
      self.assertIs(compiler._cache_write, original)


# ------------------------------------------------------------------ profiles, the run-ahead cap
class GpuProfileOptions(unittest.TestCase):
  """profiling.profile_options for a GPU process: CUPTI's own keys, which the GPU tracer needs exactly (it refuses an
  unknown key -- a TPU one -- and then records no device at all)."""

  def test_gpu(self):
    """gpu, cuda and rocm (a device's .platform): gpu_num_chips_to_profile_per_task and GPU_EVENT_CAPS, no tpu_* key,
    no Python tracer; the chip count and its check as on TPU, and attrs still win."""
    self.assertEqual(
        profiling.GPU_EVENT_CAPS, {"gpu_max_activity_api_events": 1 << 24, "gpu_max_callback_api_events": 1 << 24}
    )
    for platform in ("gpu", "cuda", "rocm"):
      with self.subTest(platform=platform):
        o = profiling.profile_options(platform=platform)
        self.assertEqual(
            o.advanced_configuration, {"gpu_num_chips_to_profile_per_task": profiling.CHIPS, **profiling.GPU_EVENT_CAPS}
        )
        self.assertEqual(o.python_tracer_level, 0)
        self.assertFalse([k for k in o.advanced_configuration if k.startswith("tpu_")])
    o = profiling.profile_options(4, platform="cuda", host_tracer_level=3, python_tracer_level=1)
    self.assertEqual(
        (o.advanced_configuration["gpu_num_chips_to_profile_per_task"], o.host_tracer_level, o.python_tracer_level),
        (4, 3, 1),
    )
    with self.assertRaisesRegex(ValueError, "at least one chip, not 0"):
      profiling.profile_options(0, platform="gpu")

  def test_tpu_unchanged(self):
    """platform None (every caller before the port) and 'tpu' -- and 'cpu', a CPU run's device -- keep libtpu's keys,
    trimmed by SPARSECORE_TRIM, and jax's default Python tracer."""
    default = jax.profiler.ProfileOptions().python_tracer_level
    for platform in (None, "tpu", "cpu"):
      with self.subTest(platform=platform):
        o = profiling.profile_options(2, platform=platform)
        self.assertEqual(o.advanced_configuration, {"tpu_num_chips_to_profile_per_task": 2, **profiling.SPARSECORE_TRIM})
        self.assertEqual(o.python_tracer_level, default)


class ProfileHloProto(unittest.TestCase):
  """--profile-hlo-proto: a trace carries every compiled program's HLO proto, and one past protobuf's 2 GB is not
  written at all (a one-stage DeepSeek-V3 layout's 383 programs: 2.63 GB).  auto keeps them while this process's
  lowered text (precompile's `program_prints` bytes) stays within HLO_PROTO_AUTO_BYTES; on and off are as said."""

  def test_rule(self):
    stand_in = lambda *sizes: types.SimpleNamespace(
        program_prints=[(f"p{i}", 0, {"bytes": b}, None) for i, b in enumerate(sizes)]
    )
    lim = dsp.HLO_PROTO_AUTO_BYTES
    self.assertEqual(lim, 1 << 30)
    for p, auto in (
        (stand_in(), True),
        (stand_in(lim // 2, lim // 2), True),
        (stand_in(lim // 2, lim // 2 + 1), False),
        (types.SimpleNamespace(), True),
    ):  # no precompile: nothing counted
      with self.subTest(n=len(getattr(p, "program_prints", ()))):
        self.assertEqual(dsp.profile_hlo_proto("auto", p), auto)
        self.assertTrue(dsp.profile_hlo_proto("on", p))
        self.assertFalse(dsp.profile_hlo_proto("off", p))
    self.assertEqual(dsp.make_parser().parse_args([]).profile_hlo_proto, "auto")
    self.assertEqual(dsp.make_parser().parse_args(["--profile-hlo-proto", "off"]).profile_hlo_proto, "off")
    self.assertTrue(hasattr(jax.profiler.ProfileOptions(), "enable_hlo_proto"))


class TransferPlatforms(unittest.TestCase):
  """transfer's run-ahead cap (max_inflight_computations) on GPU: the CUDA / ROCm plugins are C-API clients, whose
  creation reads jax_pjrt_client_create_options, so a GPU process takes the cap as a TPU one does."""

  def test_option_aware_platforms(self):
    self.assertTrue({"tpu", "cuda", "gpu", "rocm"} <= set(transfer.OPTION_AWARE_PLATFORMS))
    self.assertNotIn("cpu", transfer.OPTION_AWARE_PLATFORMS)

  def test_cuda_process_takes_the_cap(self):
    """set_max_inflight on a process pinned to cuda reports the cap applied, silently; pinned to cpu it warns that
    the cap is not (backend initialization and the pin mocked; the client options restored after)."""
    from jax._src import xla_bridge as xb

    saved = jax.config.jax_pjrt_client_create_options
    try:
      for pinned, want in ((("cuda",), True), (("cuda", "cpu"), True), (("rocm",), True), (("cpu",), False)):
        jax.config.update("jax_pjrt_client_create_options", saved)
        with (
            self.subTest(pinned=pinned),
            mock.patch.object(xb, "backends_are_initialized", lambda: False),
            mock.patch.object(transfer, "pinned_platforms", lambda pinned=pinned: pinned),
            warnings.catch_warnings(record=True) as w,
        ):
          warnings.simplefilter("always")
          self.assertEqual(transfer.set_max_inflight(64, plugin="cuda"), want)
          self.assertEqual(transfer.pjrt_client_options("cuda")[transfer.MAX_INFLIGHT_OPTION], 64)
          self.assertEqual(any("NOT applied" in str(x.message) for x in w), not want)
    finally:
      jax.config.update("jax_pjrt_client_create_options", saved)


# ------------------------------------------------------------------ cuDNN attention
class CudnnAttention(unittest.TestCase):
  """deepseek.attention's 'cudnn' kernel, ds_ep_pipeline's GPU default: jax's cuDNN flash attention, causal, on the
  [b, t, n, d] operands as they come.  What it refuses needs no GPU; off GPU the cuDNN call itself is mocked; its
  numbers against reference_attention only on a GPU backend."""

  def operands(self, b=2, t=16, nq=2, nkv=2, dqk=192, dv=128, dtype=jnp.bfloat16):
    ks = jax.random.split(jax.random.key(0), 3)
    return (
        jax.random.normal(ks[0], (b, t, nq, dqk)).astype(dtype),
        jax.random.normal(ks[1], (b, t, nkv, dqk)).astype(dtype),
        jax.random.normal(ks[2], (b, t, nkv, dv)).astype(dtype),
    )

  def test_refusals(self):
    """A query offset or key segment ids of their own -- context parallelism's -- are refused before any cuDNN call:
    directly, through _attention_kernel's dispatch and through attention_op."""
    from jax._src.cudnn import fused_attention_stablehlo as cudnn_fa

    cfg = DeepSeekConfig(attention_kernel="cudnn")
    q, k, v = self.operands()
    seg = jnp.ones(q.shape[:2], jnp.int32)
    with mock.patch.object(cudnn_fa, "dot_product_attention", side_effect=AssertionError("reached cuDNN")):
      for kw in (dict(offset=8), dict(kv_segment_ids=seg)):
        for f in (attention.cudnn_attention, attention._attention_kernel, attention.attention_op):
          with (
              self.subTest(f=f.__name__, case=sorted(kw)),
              self.assertRaisesRegex(NotImplementedError, "attention_kernel 'cudnn': no query offset"),
          ):
            f(q, k, v, seg, cfg, **kw)

  def test_call(self):
    """The cuDNN call gets the operands themselves (BTNH: no transpose), scale 1.0 (MLA and Qwen fold theirs into the
    query), the causal mask, and no segment ids (ds_ep's rows are unpacked) -- MLA's unequal q / k and v head dims
    through attention_op, and Qwen3.8's GQA through qwen35's attention, which repeats the kv heads only for
    'reference': cuDNN takes the 2 kv heads of 4 query heads as they are."""
    from jax._src.cudnn import fused_attention_stablehlo as cudnn_fa
    from maxtext.experimental.mpmd.jax.qwen35 import _ops as q_ops

    calls = []

    def fake(query, key, value, *a, **kw):
      calls.append((query, key, value, a, kw))
      return jnp.zeros(query.shape[:3] + value.shape[3:], value.dtype)

    want_kw = dict(scale=1.0, mask_type=cudnn_fa.MaskType.CAUSAL, qkv_layout="BTNH")
    qcfg = Qwen35Config(**{**Q_TINY, "attention_kernel": "cudnn"})
    with mock.patch.object(cudnn_fa, "dot_product_attention", fake):
      q, k, v = self.operands()
      out = attention.attention_op(q, k, v, jnp.ones(q.shape[:2], jnp.int32), DeepSeekConfig(attention_kernel="cudnn"))
      self.assertEqual(out.shape, (2, 16, 2, 128))
      q, k, v = self.operands(nq=4, nkv=2, dqk=64, dv=64)
      q_ops.attention(q, k, v, jnp.ones(q.shape[:2], jnp.int32), qcfg)
    self.assertEqual(len(calls), 2)
    for (cq, ck, cv, a, kw), shapes in zip(
        calls, (((2, 16, 2, 192), (2, 16, 2, 192), (2, 16, 2, 128)), ((2, 16, 4, 64), (2, 16, 2, 64), (2, 16, 2, 64)))
    ):
      self.assertEqual(((cq.shape, ck.shape, cv.shape), a, kw), (shapes, (), want_kw))
    self.assertTrue(calls[1][0] is q and calls[1][1] is k and calls[1][2] is v)

  @testing.requires_gpu(why="cuDNN")
  def test_matches_reference_on_gpu(self):
    """On a GPU: MLA's head dims in bf16 (qk 192, v 128, on sm90 and later; sm80 takes equal dims of at most 128),
    the output and the three input gradients against reference_attention, to bf16 rounding."""
    cfg = DeepSeekConfig(attention_kernel="cudnn")
    mla = float(getattr(jax.devices()[0], "compute_capability", 0) or 0) >= 9
    q, k, v = (0.3 * a for a in self.operands(t=256, nq=4, nkv=4, dqk=192 if mla else 128))
    seg = jnp.ones(q.shape[:2], jnp.int32)
    ct = jax.random.normal(jax.random.key(1), q.shape[:3] + v.shape[3:]).astype(jnp.bfloat16)
    ya, va = jax.vjp(lambda q, k, v: attention.cudnn_attention(q, k, v, seg, cfg), q, k, v)
    yb, vb = jax.vjp(lambda q, k, v: attention.reference_attention(q, k, v, seg), q, k, v)
    for name, a, b in zip(("out", "dq", "dk", "dv"), (ya,) + va(ct), (yb,) + vb(ct)):
      a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
      self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(b), 2e-2, name)


# ------------------------------------------------------------------ gpu.xla_flags
class GpuXlaFlags(unittest.TestCase):
  """gpu.xla_flags, which ds_ep_multihost.sh puts in XLA_FLAGS on GPU (its non-comment lines joined by spaces).  XLA
  parses every backend's flags at startup and aborts on one it does not know, so a CPU process started with them
  proves each is a flag of this jaxlib's XLA; one unknown flag added aborts it (the control)."""

  def test_flags_parse(self):
    with open(os.path.join(testing.package_dir(), "gpu.xla_flags")) as f:
      lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    self.assertTrue(lines)
    for line in lines:  # one flag a line: the shell's join keeps them apart
      self.assertRegex(line, r"^--xla_[a-z0-9_]+=\S*$")

    def start(flags):
      return subprocess.run(
          [sys.executable, "-c", "import jax; jax.devices()"],
          capture_output=True,
          text=True,
          env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags),
          timeout=600,
      )

    out = start(" ".join(lines))
    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
    out = start(" ".join(lines) + " --xla_gpu_enable_no_such_flag=true")
    self.assertNotEqual(out.returncode, 0)
    self.assertIn("Unknown flag in XLA_FLAGS", out.stderr)


# ------------------------------------------------------------------ --norm-reduce allreduce
# One process of a jax.distributed run (multihost_test.run_multi's `cmd`): ds_ep_pipeline.main with
# Pipeline._global_sumsq counted, the count printed last -- so a run says whether its steps took the all-reduce.
COUNTED_MAIN = """if True:
    import sys
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
    calls = []
    real = dsp.Pipeline._global_sumsq
    def counted(self, sq):
        calls.append(len(sq))
        return real(self, sq)
    dsp.Pipeline._global_sumsq = counted
    dsp.main(sys.argv[1:])
    print(f'global_sumsq calls: {len(calls)}', flush=True)
"""


class NormReduce(unittest.TestCase):
  """--norm-reduce allreduce: under jax.distributed the stages' gradient sums of squares meet in one program over every
  device of the job (Pipeline._global_sumsq: each stage's first device gives its stage's sum, every other device 0),
  and each stage's update takes [total, 0, ..., 0] for the S partials the hops deliver.  Against --norm-reduce hops,
  the default, on local CPU processes (multihost_test.run_multi; gloo carries the all-reduce): the losses and the
  gradient norms of every step (TINY / Q_TINY's three warmup steps, fp32 weights at lr 1e-3: an update moves the loss
  by 5-10%, so a norm gone wrong shows in the next step's loss too).  In both runs every stage clips by one norm;
  every step of the allreduce run called _global_sumsq once on every process, and the hops run never.

  Two stages: the all-reduce's total is fl(sq0 + sq1) in any order -- the zeros add exactly, and the update's sum of
  [total, 0] is total -- the very sum the hops' update makes, so every step is the hops run's bit for bit (DeepSeek-V3
  and Qwen3.8 layers measured alike).  Four stages: the hops' updates add ((sq0 + sq1) + sq2) + sq3, the all-reduce
  in gloo's order, so the norms may differ in their last bit (one ulp at the first step on Qwen3.8 layers, at the
  third on DeepSeek-V3's, measured), which Adam's sign-like early steps grow through the updates: the first loss bit
  for bit (the first forward sees no norm), the first norm to 1e-6 and the later steps to multihost_test._compare's
  bounds for runs an ulp apart (losses 2e-3, norms 2e-2).  A stage's sum dropped, or counted twice, moved the first
  norm by 16-46% at two stages."""

  def setUp(self):
    testing.require_multihost()  # every test here starts several JAX processes

  def runs(self, args, procs, devs):
    """`args` under jax.distributed on `procs` local CPU processes of `devs` devices, once under --norm-reduce hops
    and once under allreduce -> {mode: [(the process's run record, its _global_sumsq calls)] in process order}."""
    out = {}
    for mode in ("hops", "allreduce"):
      d = self.enterContext(tempfile.TemporaryDirectory())
      outs = multihost_test.run_multi(
          None,
          procs,
          devs,
          collectives="gloo",
          cmd=[
              sys.executable,
              "-c",
              COUNTED_MAIN,
              "--distributed",
              *args,
              "--norm-reduce",
              mode,
              "--json",
              f"{d}/run.json",
              "--json-procs",
              "all",
          ],
      )
      bad = [i for i, (rc, _) in enumerate(outs) if rc]
      self.assertFalse(bad, "\n".join(f"----- {mode}, process {i}: exit {outs[i][0]}\n{outs[i][1][-3000:]}" for i in bad))
      runs = []
      for i, (_, o) in enumerate(outs):
        with open(f"{d}/run.json" if i == 0 else f"{d}/run.p{i}.json") as f:
          rec = json.load(f)
        calls = re.findall(r"^global_sumsq calls: (\d+)$", o, re.M)
        self.assertEqual(len(calls), 1, o[-3000:])
        runs.append((rec, int(calls[0])))
      out[mode] = runs
    return out

  def check(self, args, procs, devs, exact):
    runs = self.runs(args, procs, devs)
    got = {}
    for mode, rs in runs.items():
      n = len(rs[0][0]["warmup_gnorms"])
      self.assertGreaterEqual(n, 2, mode)
      for i, (rec, calls) in enumerate(rs):
        self.assertEqual((rec["process"], rec["processes"], rec["config"]["norm_reduce"]), (i, procs, mode))
        self.assertEqual(calls, n if mode == "allreduce" else 0, f"{mode}, process {i}")
      norms = {tuple(rec["warmup_gnorms"]) for rec, _ in rs}
      self.assertEqual(len(norms), 1, f"{mode}: the stages clip by different norms {norms}")
      losses = [rec["warmup_losses"] for rec, _ in rs if rec["warmup_losses"]]
      # the last stage's processes have them: one, or every one when that stage spans them all
      self.assertEqual(len(losses), sum(rec["stages"] == rs[-1][0]["stages"] for rec, _ in rs), mode)
      self.assertTrue(all(l == losses[0] for l in losses), f"{mode}: {losses}")
      got[mode] = (losses[0], list(norms.pop()))
    (loss, norm), (loss_h, norm_h) = got["allreduce"], got["hops"]
    if exact:
      self.assertEqual(loss, loss_h)
      self.assertEqual(norm, norm_h)
    else:
      self.assertEqual(loss[0], loss_h[0])
      np.testing.assert_allclose(norm[0], norm_h[0], rtol=1e-6)
      np.testing.assert_allclose(loss, loss_h, rtol=2e-3)
      np.testing.assert_allclose(norm, norm_h, rtol=2e-2)

  @pytest.mark.scheduled_only
  def test_two_stages(self):
    """DeepSeek-V3 layers (TINY), batch split, two stages of EP 2 on two processes of two devices -- each stage's
    second device gives the all-reduce 0: bit for bit."""
    flags = "--stages 2 --vv 1 --fsdp 1 --ep 2 --layers 2 --gbs 8 --mb 2 --batch-split --batch-split-impl native"
    self.check(multihost_test.TINY + flags.split(), procs=2, devs=2, exact=True)

  @pytest.mark.scheduled_only
  def test_qwen_two_stages(self):
    """The same on Qwen3.8 layers (Q_TINY, two a stage: three Gated DeltaNet layers, then the gated-attention one):
    bit for bit."""
    flags = "--stages 2 --vv 1 --fsdp 1 --ep 2 --layers 4 --gbs 8 --mb 2 --batch-split --batch-split-impl native"
    self.check(multihost_test.Q_TINY + flags.split(), procs=2, devs=2, exact=True)

  @pytest.mark.scheduled_only
  def test_one_stage_two_processes(self):
    """The data-parallel layout: one stage over both processes (FSDP 2 across them, EP 2 inside each, the breadth
    schedule, weights gathered a chunk's visit), so the stage's devices span processes and each process assembles
    its own shards of the total.  The hops run has no hops at one stage -- its update's norm is the stage's own sum
    -- and the all-reduce's total is that sum plus zeros: bit for bit."""
    flags = (
        "--stages 1 --vv 2 --fsdp 2 --ep 2 --layout ep_on_chip --schedule breadth --fsdp-scope repeat "
        "--layers 2 --gbs 32 --mb 4 --batch-split --batch-split-impl native"
    )
    self.check(multihost_test.TINY + flags.split(), procs=2, devs=2, exact=True)

  @pytest.mark.scheduled_only
  def test_four_stages(self):
    """TINY on four one-device stages, each a process of its own: four partials, summed in different orders."""
    flags = "--stages 4 --vv 1 --fsdp 1 --ep 1 --layers 4 --gbs 8 --mb 4 --batch-split --batch-split-impl native"
    self.check(multihost_test.TINY + flags.split(), procs=4, devs=1, exact=False)


class NormReduceProgram(unittest.TestCase):
  """Pipeline._global_sumsq in one process on a stand-in pipeline: stages of two devices (three over the first six of
  CPU's eight, so two devices hold no stage and give 0 too; two over four GPUs), each stage's sum of squares a
  replicated scalar on its mesh.  Every stage gets the total of all of them -- each stage counted once though it
  holds two devices -- exact (the sums are exact in fp32), replicated on its own mesh; the program is built once and
  reused; and the GPU-only compiler option that keeps the sum a NCCL all-reduce (not XLA's Triton one-shot kernel,
  whose compile fails at 64 ranks in one NVL72 rack) goes to it on GPU only: CPU's and TPU's XLA builds need not know
  the option."""

  def test_total_and_compiler_options(self):
    devs = jax.devices()
    n_stages = len(devs) // 2 - 1 if len(devs) >= 6 else len(devs) // 2
    if n_stages < 2:
      self.skipTest(f"needs four devices, not {len(devs)}")
    meshes = [Mesh(np.array(devs[2 * s : 2 * s + 2]), ("x",)) for s in range(n_stages)]
    vals = [1.5, 2.25, 4.125, 8.0625][:n_stages]
    sq = [jax.device_put(np.float32(v), NamedSharding(m, P())) for v, m in zip(vals, meshes)]
    stand_in = types.SimpleNamespace(_gsq=None, meshes=meshes, S=n_stages, local=frozenset(range(n_stages)))
    options, real_jit = [], jax.jit

    def spy(*args, **kw):
      options.append(kw.get("compiler_options"))
      return real_jit(*args, **kw)

    with mock.patch.object(jax, "jit", spy):
      totals = [dsp.Pipeline._global_sumsq(stand_in, sq) for _ in range(2)]
    gpu = devs[0].platform == "gpu"
    self.assertEqual(options, [{"xla_gpu_experimental_use_collective_kernels": ""} if gpu else None])
    for tot in totals:
      self.assertEqual(sorted(tot), list(range(n_stages)))
      for s, m in enumerate(meshes):
        self.assertEqual(tot[s].sharding, NamedSharding(m, P()))
        self.assertEqual([float(x.data) for x in tot[s].addressable_shards], [sum(vals)] * 2, s)
    for s, z in stand_in._gsq["zeros"].items():  # the partials the update takes besides the total
      self.assertEqual((z.sharding, float(z)), (NamedSharding(meshes[s], P()), 0.0))


# ------------------------------------------------------------------ --offload-residuals on
# TINY_ARGS's model on two stages of EP 2 (four devices: CPU's first four, or four GPUs), two one-layer chunks a stage
# (V = 2), batch split, fp32, two microbatches of four sequences; Q_TINY's Qwen3.8 layers alike.
OFFLOAD_ARGS = (
    TINY_ARGS
    + ("--fsdp 1 --ep 2 --vv 2 --layers 4 --gbs 8 --mb 2 --dtype float32 " "--rs-dtype float32").split()
    + SPLIT_ARGS
)
Q_OFFLOAD_ARGS = Q_TINY_ARGS + ["--layer-split", ""] + OFFLOAD_ARGS[len(TINY_ARGS) :] + ["--attn", "reference"]


@testing.needs_devices(4)
class DirectExperts(unittest.TestCase):
  """--direct-experts (the GPU default): at fsdp 1 a routed expert's compute copy would be its storage's bytes again,
  so the stages read the storage leaves in place (Pipeline.direct) -- no gather, no second copy of the step's largest
  weights.  On CPU, TINY's DeepSeek-V3 layers and Q_TINY's Qwen3.8 ones (2 stages x EP 2, V = 2, interleaved, batch
  split, fp32 storage and compute): every chunk's routed leaves are read in place, the prefetch plans and the update's
  gathers leave them out, the loop's weights hold the storage buffers themselves -- every step, the chunks the
  update gathers and those a prefetch brings (the first stage's later chunk, Pipeline._install) -- and three steps
  train bit for bit as with the copy (off): losses, norms, every parameter."""

  TAIL = [
      "--stages",
      "2",
      "--vv",
      "2",
      "--fsdp",
      "1",
      "--ep",
      "2",
      "--gbs",
      "8",
      "--mb",
      "2",
      "--dtype",
      "float32",
      "--rs-dtype",
      "float32",
      "--weight-dtype",
      "float32",
      *SPLIT_ARGS,
      "--schedule",
      "interleaved",
  ]

  @staticmethod
  def watch(p):
    """Watch the weights `p`'s step loop runs on: as the loop starts (the update's gathers, through Pipeline._whole)
    and as a prefetch completes a chunk's (Pipeline._install), each in-place leaf against the step's params, by
    buffer.  -> {(stage, slot, 'update' or 'prefetch'): [whether each was the storage's own buffer, every time]}."""
    ptr = lambda a: a.addressable_shards[0].data.unsafe_buffer_pointer()
    seen, run, install = {}, p._run, p._install

    def note(how, params, s, i, w):
      ids = p.direct.get(p.stage_chunks[s][i].index, ())
      if ids and w is not None:
        got, own = jax.tree.leaves(w), jax.tree.leaves(params[s][i])
        seen.setdefault((s, i, how), []).extend(ptr(got[j]) == ptr(own[j]) for j in ids)

    def watched_run(params, wc, batch, overlap):
      for s, ws in enumerate(wc):
        for i, w in enumerate(ws or ()):
          note("update", params, s, i, w)
      return run(params, wc, batch, overlap)

    def watched_install(st, op, delivered):
      install(st, op, delivered)
      b = op.pf[0]
      if b not in st.wparts:  # the gather's last host: the chunk's weights are in st.wc now
        note("prefetch", st.params, op.s, p.slot[b], st.wc[op.s][p.slot[b]])

    p._run, p._install = watched_run, watched_install
    return seen

  def assert_read_in_place(self, p, seen):
    """What `watch` saw: every chunk with leaves read in place, the storage's own buffers each time -- a prefetched
    chunk among them."""
    self.assertEqual(
        {k[:2] for k in seen},
        {(s, i) for s in range(p.S) for i, ch in enumerate(p.stage_chunks[s]) if ch.index in p.direct},
    )
    self.assertIn("prefetch", {k[2] for k in seen})
    self.assertTrue(all(all(v) for v in seen.values()), seen)

  def run_steps(self, base, mode, steps=3):
    """`steps` steps of the pipeline `base` + TAIL builds under --direct-experts `mode`, its loop watched -> (the
    pipeline, what `watch` saw, every step's (loss partials, gradient norm), the final params' leaves)."""
    p = dsp.build(dsp.make_parser().parse_args(base + self.TAIL + ["--direct-experts", mode]))
    seen = self.watch(p)
    params, opts, wc0 = p.init(jax.random.key(0))
    lrs, batch = p.lr_arrays(1e-3), p.synthetic_batch()
    out = []
    for _ in range(steps):
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      out.append((np.asarray(lacc).tolist(), float(gn)))
    leaves = [np.asarray(x) for x in jax.tree.leaves(params)]
    return p, seen, out, leaves

  def check(self, base):
    p0, _, n0, w0 = self.run_steps(base, "off")
    p1, seen, n1, w1 = self.run_steps(base, "on")
    self.assertEqual(p0.direct, {})
    self.assertEqual(p1.direct, {ch.index: ds_ep.routed_leaf_ids(ch) for ch in p1.chunks})
    self.assertTrue(all(len(v) == 3 for v in p1.direct.values()))  # wi_0, wi_1, wo
    self.assertFalse(p1.self_prefetch)  # nothing of chunk 0 left to gather late
    for b, hosts in p1.pf_gathers:
      for _, g in hosts:
        self.assertFalse(set(g) & set(p1.direct[b]), f"chunk {b}: a prefetch gathers {g}")
    for s, st in enumerate(p1.stages):
      for i in p1.gather_slots[s]:
        c = p1.stage_chunks[s][i].index
        self.assertEqual(set(st.keep[i]), set(range(p1.n_leaves[c])) - set(p1.direct[c]))
    self.assert_read_in_place(p1, seen)
    self.assertEqual(n0, n1)  # every step's losses and norm
    for a, b in zip(w0, w1, strict=True):
      np.testing.assert_array_equal(a, b)
    self.assertIn("read in place (fsdp 1: no compute copy)", p1.summary())

  @pytest.mark.scheduled_only
  def test_deepseek(self):
    self.check(TINY_ARGS + ["--layers", "4"])

  @pytest.mark.scheduled_only
  def test_qwen(self):
    self.check(Q_TINY_ARGS + ["--layer-split", "", "--attn", "reference"])

  @pytest.mark.scheduled_only
  def test_one_host_gather_and_repeat_scope(self):
    """fsdp_scope 'repeat' at M=1 (the update gathers a chunk with no earlier op to gather in; a prefetch with one
    host, whose part names its leaves when the in-place ones leave it short of the whole chunk): two steps bit for
    bit with the copy, the plan's weights whole at every op, the in-place leaves the storage's own buffers at every
    gather."""
    from tests.unit.experimental.mpmd.jax.ds_ep_test import check_repeat_plan

    kw = dict(
        n_stages=2,
        chunks_per_stage=2,
        n_micro=1,
        global_batch=2,
        seq=256,
        overlap=True,
        schedule="breadth",
        fsdp_scope="repeat",
    )
    scfg = tiny_xla(layers=4, fsdp=1, ep=2)
    p = dsp.Pipeline(scfg, direct_experts=True, **kw)
    q = dsp.Pipeline(scfg, **kw)
    self.assertTrue(p.direct and not q.direct)
    check_repeat_plan(self, p)
    self.assertTrue(any(p.pf_n[j] == 1 for j in p.pf), p.pf)
    seen = self.watch(p)
    assert_same_run(self, mode_run(p, 1, 2), mode_run(q, 1, 2), "direct vs copy, repeat, M=1")
    self.assert_read_in_place(p, seen)

  def test_where_it_applies(self):
    """Only at fsdp 1, under shard mode auto, where a routed leaf's storage and compute dtypes agree (a bf16 copy of
    fp32 weights is no alias), and with the flag on: a GPU default (GPU_DEFAULTS), off on TPU and CPU, whose plans
    (the pinned circular ones among them) stay as they were; Pipeline's own default is off too."""
    ap = dsp.make_parser()
    with jax_platforms("cpu"):
      self.assertEqual(ap.parse_args([]).direct_experts, "off")
      self.assertEqual(ap.parse_args(["--platform", "gpu"]).direct_experts, "on")
      self.assertEqual(ap.parse_args(["--platform", "tpu"]).direct_experts, "off")
    self.assertFalse(inspect.signature(dsp.Pipeline).parameters["direct_experts"].default)
    for extra, want in (
        ([], True),
        (["--direct-experts", "off"], False),
        (["--weight-dtype", "bfloat16"], False),
        (["--fsdp", "2", "--ep", "1"], False),
    ):
      with self.subTest(extra=extra):
        args = TINY_ARGS + ["--layers", "4"] + self.TAIL + ["--direct-experts", "on"] + extra
        p = dsp.build(ap.parse_args(args), meshes=dsp.stage_meshes(2, 2, 1) if "--fsdp" in extra else None)
        self.assertEqual(bool(p.direct), want)


@needs_tokamax
@testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
class SaveResiduals(unittest.TestCase):
  """--save-residuals on (Megatron-LM's default, no recompute): a chunk's forward returns its layers' VJP residuals and
  its backward evaluates the VJP on them, instead of recomputing the chunk.  On CPU, E2E_ARGS's DeepSeek-V3 layers
  (fsdp 2 x ep 2, V = 2, per-layer trees, tokamax's xla kernels) with and without batch split, at EP 1 with a dense
  layer, and under context parallelism (cp 2): one step's loss, stage sumsqs and every reduced gradient leaf as the
  recompute programs' to 1e-5 (XLA fuses the two programs differently: check); the lean residual set keeps no fp32
  [rows, seq, emb] RMSNorm intermediate and no silu / sigmoid intermediate of the gated MLPs; an fp32 accumulator
  taking the routed experts' dW in place (--inplace-dw all, tokamax_gmm's dw32), saved or recomputed, gives the
  separate add's gradients to 4 bf16 ulps (check_fp32_in_place); --buffer-masks off changes no bit, with NaN in the
  rows past the groups (unwritten_rows)."""

  BASE = list(E2E_ARGS) + TOKAMAX_ARGS + ["--layer-tree", "per_layer"]

  def check(self, base, extra=(), rel=1e-5):
    """`base` + `extra` with --save-residuals on against the recompute programs (off): the loss to `rel`, the stage
    sumsqs to twice that, every reduced gradient leaf to `rel` of its largest magnitude.  -> the saved run's
    StageConfig."""
    got = pipeline_step(base + ["--save-residuals", "on", *extra])
    want = pipeline_step(base)
    self.assertTrue(got[0].save_residuals and not want[0].save_residuals)
    self.assertLessEqual(abs(got[1] - want[1]), rel * abs(want[1]), "loss")
    np.testing.assert_allclose(got[2], want[2], rtol=2 * rel, atol=0, err_msg="stage sumsqs")
    for (path, a), (path_b, b) in zip(got[3], want[3], strict=True):
      self.assertEqual(path, path_b)
      err, top = float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))
      self.assertLessEqual(err, rel * top, f"gradient leaf {path}: {err:.3g} of {top:.3g}")
    return got[0]

  def check_fp32_in_place(self, base):
    """`base` (bf16 compute) with an fp32 accumulator taking the routed experts' dW in place (--inplace-dw all:
    tokamax_gmm's dw32) against tokamax's dW -- rounded to the bf16 weight's dtype -- plus the fp32 add (--inplace-dw
    off): in-place dW taken by the first run and not by the second, the loss bit for bit (the forward is the same
    program), every reduced gradient leaf within 4 bf16 ulps of its largest magnitude."""
    got = pipeline_step(base + ["--inplace-dw", "all", "--acc-dtype", "float32"])
    want = pipeline_step(base + ["--inplace-dw", "off", "--acc-dtype", "float32"])
    self.assertEqual((got[4], want[4]), (True, False))
    self.assertEqual(got[1], want[1])
    for (path, a), (_, b) in zip(got[3], want[3], strict=True):
      err, top = float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))
      self.assertLessEqual(err, 4 * BF16_EPS * top, f"gradient leaf {path}: {err:.3g} of {top:.3g}")

  @pytest.mark.scheduled_only
  def test_unsplit(self):
    """Batch split off: check's bounds."""
    self.check([a for a in self.BASE if a not in SPLIT_ARGS])

  @pytest.mark.scheduled_only
  def test_batch_split(self):
    """Batch split on (split_run's interleaved halves under the saved VJP): check's bounds."""
    self.check(self.BASE)

  @pytest.mark.scheduled_only
  def test_ep1_dense_layer(self):
    """EP 1 and dropless -- every compact-buffer row inside a group, so _tokamax_experts skips its masks, on both
    sides (against ragged_dot, which has no such skip: TokamaxPipeline.test_deepseek_ep1) -- with a dense layer
    before chunk 0's MoE layer (--layer-split 2,1,1,1), batch split off: check's bounds, the run's EP, buffer and
    dense layer as named."""
    # appended: BASE holds --ep twice (TINY_ARGS's, then E2E_ARGS's), and the last one counts
    base = [a for a in self.BASE if a not in SPLIT_ARGS] + ["--ep", "1"]
    scfg = self.check(base + ["--layers", "5", "--dense-layers", "1", "--layer-split", "2,1,1,1"])
    self.assertEqual((scfg.ep, scfg.model.first_num_dense_layers), (1, 1))
    self.assertLessEqual(scfg.ragged_buffer_factor, 0)  # dropless

  @pytest.mark.scheduled_only
  def test_context_parallel(self):
    """Context parallelism (--cp 2: each sequence of a microbatch on two devices of the 'expert' axis, a row of 128
    tokens each; two sequences a microbatch, a row a stage device), batch split off (StageConfig refuses it with cp):
    check's bounds -- the lean layers with the gathered key / value latent between its norm and wkv_b (attention.mla's
    kv_context: LeanResiduals.test_mla_context_parallel), the rows' positions and RoPE, the fused norms."""
    scfg = self.check([a for a in self.BASE if a not in SPLIT_ARGS] + ["--cp", "2", "--gbs", "4"])
    self.assertEqual(scfg.cp, 2)

  @pytest.mark.scheduled_only
  @testing.needs_ynn_off(
      "on jaxlib 0.12.0.1 XLA:CPU crashes running these programs with them on (SIGSEGV or SIGABRT in 4 of 4 runs, "
      "in the --inplace-dw off pipeline; on 0.12.1 2 of 2); with them off it passes (4 of 4 on 0.12.0.1, 1 of 1 on "
      "0.12.1), as on jaxlib 0.11.1",
      jaxlib_before=testing.YNN_BUG_JAXLIBS_BEFORE,
  )
  def test_fp32_accumulator_in_place(self):
    """Saved residuals, batch split off: check_fp32_in_place, under both --route-weights (the dW kernels reading their
    recomputed lhs either way)."""
    for route in ds_ep.ROUTE_WEIGHTS:
      with self.subTest(route_weights=route):
        self.check_fp32_in_place(
            [a for a in self.BASE if a not in SPLIT_ARGS]
            + ["--dtype", "bfloat16", "--save-residuals", "on", "--route-weights", route]
        )

  @pytest.mark.scheduled_only
  def test_fp32_accumulator_in_place_recompute(self):
    """The recompute path (no saved residuals) under batch split: check_fp32_in_place.  bwd_core_recompute traces the
    accumulator starts as primals of its VJP, whose cotangents carry the accumulated dW -- both halves' (mb0 hands
    its start on to mb1, and its dW continues mb1's accumulated value)."""
    self.check_fp32_in_place(self.BASE + ["--dtype", "bfloat16"])

  @pytest.mark.scheduled_only
  def test_fp32_accumulator_in_place_recompute_unsplit(self):
    """The recompute path with batch split off (each layer's start a primal of its own VJP): check_fp32_in_place."""
    self.check_fp32_in_place([a for a in self.BASE if a not in SPLIT_ARGS] + ["--dtype", "bfloat16"])

  def test_lean_residuals(self):
    """The saved set holds no fp32 activation-sized leaf (RMSNorm's x in fp32); of the bf16 ones only each layer's
    input but the chunk's own (the backward's argument) and its post-attention stream -- no RMSNorm output or
    normalised activation (LeanResiduals) -- in chunks of a dense and a MoE layer, of two MoE layers and of one; RoPE's
    rows unbroadcast, one head; and a gated product keeps only its two inputs (no silu / sigmoid intermediate).
    Built only: no step runs."""
    base = [a for a in self.BASE if a not in SPLIT_ARGS] + [
        "--dtype",
        "bfloat16",
        "--save-residuals",
        "on",
        "--layers",
        "6",
        "--dense-layers",
        "1",
        "--mlp",
        "384",
        "--moe-mlp",
        "128",
        "--layer-split",
        "2,2,1,1",
    ]
    p = dsp.build(dsp.make_parser().parse_args(base))
    self.assertEqual([pr.ch.dense + pr.ch.n_moe for pr in p.progs], [2, 2, 1, 1])
    for pr in p.progs:
      if pr.ch.last:
        continue
      local = pr._saved_meta["local"]
      act = NamedSharding(pr.mesh, pr.act_spec).shard_shape(pr.act_aval.shape)  # a device's [rows, seq, emb]
      acts = [r for r in local if r.shape == act]
      # an fp32 [rows, seq, emb] leaf: RMSNorm's input in fp32 (CPU's reference attention keeps 4-D fp32 scores)
      self.assertFalse([r for r in acts if r.dtype == jnp.float32], pr.ch.index)
      self.assertEqual(len(acts), 2 * (pr.ch.dense + pr.ch.n_moe) - 1, pr.ch.index)
      self.assertEqual({r.shape[-2] for r in local if r.dtype == jnp.complex64 and len(r.shape) == 4}, {1}, pr.ch.index)
    # the gated product under lean residuals keeps its two inputs only (no silu / sigmoid intermediate)
    a = jnp.ones((4, 8), jnp.bfloat16)

    def n_res(lean):
      with dl.lean_residuals(lean):
        _, vjp = jax.vjp(lambda x, y: dl.gated_product(x, y, "silu", jnp.bfloat16), a, a * 2)
      return len(jax.tree_util.tree_leaves(vjp))

    self.assertLess(n_res(True), n_res(False))
    self.assertLessEqual(n_res(True), 2)

  @pytest.mark.scheduled_only
  def test_buffer_masks_off(self):
    """--buffer-masks off against on, batch split off, at EP 2 with a capped buffer (--ragged-buffer-factor 1.5: rows
    past the groups exist) and the GPU's combine (--combine-impl triton: here gpu_kernels.gather_sum's XLA sum, and
    a backward that dots every row of the buffer with its token's gradient, then drops the rows past the groups):
    the loss and every reduced gradient leaf finite and bit for bit -- every reader of the compact buffer's rows past
    its groups is bounded by the group sizes or gathers valid rows only.  With masks off the dispatch leaves real
    tokens' rows there and the combine's backward their gradients; the grouped matmuls, whose lax.ragged_dot writes
    zeros there, leave NaN there in both runs (unwritten_rows), as a GPU kernel leaves those rows unwritten."""
    base = [a for a in self.BASE if a not in SPLIT_ARGS] + ["--ragged-buffer-factor", "1.5", "--combine-impl", "triton"]
    with unwritten_rows() as poisoned:
      got = pipeline_step(base + ["--buffer-masks", "off"])
      self.assertTrue(poisoned, "no grouped matmul of the masks-off run traced through unwritten_rows")
      want = pipeline_step(base)
    self.assertEqual((got[0].buffer_masks, want[0].buffer_masks, got[0].combine_impl), (False, True, "triton"))
    self.assertTrue(np.isfinite(want[1]) and all(np.isfinite(b).all() for _, b in want[3]))
    self.assertEqual(got[1], want[1])
    for (path, a), (_, b) in zip(got[3], want[3], strict=True):
      np.testing.assert_array_equal(a, b, err_msg=path)

  @pytest.mark.scheduled_only
  def test_route_weights_experts(self):
    """--route-weights experts (the GPU default) on the GPU's MoE path -- a capped buffer, so rows past the groups
    exist, its masks off, the combine kernel's code (combine_impl 'triton': off GPU gather_sum's XLA sum) -- with
    saved residuals, whose backward reads the experts' recomputed lhs (lean residuals), against the recompute
    programs."""
    self.check(
        [a for a in self.BASE if a not in SPLIT_ARGS]
        + [
            "--route-weights",
            "experts",
            "--ragged-buffer-factor",
            "1.5",
            "--buffer-masks",
            "off",
            "--combine-impl",
            "triton",
        ]
    )

  @pytest.mark.scheduled_only
  def test_route_weights_experts_batch_split(self):
    """--route-weights experts under batch split with saved residuals (both halves' experts recomputing their lhs in
    the backward) against the recompute programs."""
    self.check(self.BASE + ["--route-weights", "experts"])

  @pytest.mark.scheduled_only
  def test_lean_experts_off(self):
    """--lean-experts off (the experts' lhs kept as it is: the buffer rows and the gated product) on the GPU's MoE path
    against the recompute programs, as test_route_weights_experts runs it with the lhs recomputed."""
    self.check(
        [a for a in self.BASE if a not in SPLIT_ARGS]
        + [
            "--route-weights",
            "experts",
            "--ragged-buffer-factor",
            "1.5",
            "--buffer-masks",
            "off",
            "--combine-impl",
            "triton",
            "--lean-experts",
            "off",
        ]
    )

  def test_expert_residuals(self):
    """What a MoE layer's saved-residual forward keeps of its routed experts: the gathered tokens [T, emb] instead of
    the compact buffer's rows [R, emb] (R = T x top-k here, dropless at EP 2), the wi products' outputs h0 and h1
    [R, moe_mlp] but not their gated product -- and the expert output rows [R, emb] only under --route-weights
    combine, whose combine's backward reads them (one a layer).  --lean-experts off keeps the rows and the product
    instead: no row gather, and one evaluation of the gated product fewer, in the backward (under --route-weights
    experts the activation's own backward still evaluates the unweighted product, for the routing weights'
    gradient)."""
    base = [a for a in self.BASE if a not in SPLIT_ARGS] + [
        "--dtype",
        "bfloat16",
        "--save-residuals",
        "on",
        "--moe-mlp",
        "128",
    ]  # moe_mlp apart from emb
    for route, n_y, lean in (("combine", 1, "on"), ("experts", 0, "on"), ("experts", 0, "off")):
      with self.subTest(route_weights=route, lean_experts=lean):
        p = dsp.build(dsp.make_parser().parse_args(base + ["--route-weights", route, "--lean-experts", lean]))
        cfg, scfg = p.scfg.model, p.scfg
        T = scfg.ep * p.seq * p.gbs // (p.M * scfg.fsdp * scfg.ep)  # the EP group's tokens, a device's view
        R = ds_ep.ragged_buffer_rows(scfg, T) or T * cfg.num_experts_per_tok
        self.assertGreater(R, T)
        D, F = cfg.emb_dim, cfg.moe_mlp_dim
        moe = [pr for pr in p.progs if not pr.ch.last and pr.ch.n_moe]
        self.assertTrue(moe)
        for pr in moe:
          shapes = [tuple(r.shape) for r in pr._saved_meta["local"]]
          n = pr.ch.n_moe
          # --lean-experts off: the buffer rows instead of the gathered tokens, and the gated product beside its inputs
          want = (n_y * n, 2 * n, n) if lean == "on" else (n_y * n + n, 3 * n, 0)
          self.assertEqual((shapes.count((R, D)), shapes.count((R, F)), shapes.count((T, D))), want, pr.ch.index)


# ------------------------------------------------------------------ dense layers a subtree each
# E2E_ARGS's pipeline unsplit (fp32, top-k, fsdp 2 x ep 2, V = 2, tokamax's xla kernels) with DeepSeek-V3's layer
# structure: two dense layers (an MLP of width 384) before four MoE layers, in chunk 0.
DENSE_ARGS = (
    [a for a in E2E_ARGS if a not in SPLIT_ARGS] + TOKAMAX_ARGS + ["--layers", "6", "--dense-layers", "2", "--mlp", "384"]
)


def layer_tree_steps(args):
  """One loss-and-gradients pass (with the fusion plan) of the pipeline `args` builds under each --layer-tree, from one
  set of weights -- the stacked pipeline's init(key 1), its stacks cut into per-layer subtrees for the other
  (ds_ep_parity.relayout) -- on synthetic_batch(3) -> {layer tree: (its pipeline, the loss, the stage sumsqs, every
  reduced gradient leaf by chunk in the stacked layout, as (path, fp32 host array))}."""
  out = {}
  ps, pp = (dsp.build(dsp.make_parser().parse_args(args + ["--layer-tree", t])) for t in ("stacked", "per_layer"))
  host = [jax.tree.map(np.asarray, t) for t in ps.chunk_trees(ps.init(jax.random.key(1))[0])]
  for p in (ps, pp):
    params, _, wc0 = p.init_state(
        p.params_from_chunks([par.relayout(t, a, b) for t, a, b in zip(host, ps.chunks, p.chunks, strict=True)])
    )
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, p.synthetic_batch(3))
    trees = [
        par.relayout(jax.tree.map(np.asarray, t), ch, par.stacked(ch)) for t, ch in zip(p.chunk_trees(grads), p.chunks)
    ]
    out[p.scfg.layer_tree] = (
        p,
        float(jnp.sum(lacc)),
        [float(s) for s in sqs],
        [(jax.tree_util.keystr(k), leaves(v)[0]) for k, v in jax.tree_util.tree_flatten_with_path(trees)[0]],
    )
  return out


@needs_tokamax
@testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
class DenseLayerTree(unittest.TestCase):
  """--layer-tree per_layer for the dense layers too (ds_ep.ChunkSpec.dense_kinds): each dense layer a subtree with a
  layer axis of one, as the MoE layers are, so that a backward adds each layer's dW into an accumulator leaf of its own
  (on a GPU inside the dW GEMM) rather than stacking the layers' dWs into one, and a layer's weights are whole buffers,
  not slices of a stack that the lean residuals' checkpoints copy.  On CPU, from one set of weights, against 'stacked'
  (the TPU default, unchanged): one step's loss and every reduced gradient leaf bit for bit (fp32), the stage sumsqs
  to fp32 rounding (each sums its leaves' squares, which the two trees group differently) -- with saved residuals and
  recomputed, under batch split, in a chunk 0 of the embedding and dense layers alone, and with the dense layers over
  two chunks."""

  def check(self, args):
    got = layer_tree_steps(args)
    (ps, *a), (pp, *b) = got["stacked"], got["per_layer"]
    dense = [ch for ch in pp.chunks if ch.dense]
    self.assertTrue(dense)
    self.assertEqual([ch.dense_kinds for ch in dense], [(ds_ep.DEEPSEEK_KIND,) * ch.dense for ch in dense])
    self.assertEqual({ch.dense_kinds for ch in ps.chunks}, {()})
    self.assertEqual(a[0], b[0], "loss")
    np.testing.assert_allclose(b[1], a[1], rtol=1e-6, err_msg="stage sumsqs")
    for (path, x), (path_b, y) in zip(a[2], b[2], strict=True):
      self.assertEqual(path, path_b)
      np.testing.assert_array_equal(y, x, err_msg=path)
    return pp

  def test_trees(self):
    """A chunk 0 of the embedding and two dense layers alone (the 64-B200 recipe's layout): per layer its
    'dense_layers' are subtrees '00', '01' of [1, ...] leaves, stored and computed as the stacked ones' layers, their
    accumulator leaves [partials, 1, ...] (each a whole buffer) and their leaf names a layer's each; stacked, one
    [2, ...] leaf a weight.  A tree of the other layout is refused, not loaded."""
    ps, pp = (
        dsp.build(dsp.make_parser().parse_args(DENSE_ARGS + ["--layer-split", "0,2,1,1", "--layer-tree", t]))
        for t in ("stacked", "per_layer")
    )
    ch, chs = pp.chunks[0], ps.chunks[0]
    self.assertEqual((ch.layers, ch.dense, ch.kinds, ch.dense_kinds), ((), 2, (), ("mla", "mla")))
    shapes = ds_ep.chunk_shapes(pp.scfg, ch)["dense_layers"]
    self.assertEqual(sorted(shapes), ["00", "01"])
    self.assertEqual(shapes["01"]["mlp"]["wi_0"]["kernel"], (1, 256, 384))
    self.assertEqual(ds_ep.chunk_shapes(ps.scfg, chs)["dense_layers"]["mlp"]["wi_0"]["kernel"], (2, 256, 384))
    self.assertEqual(ds_ep.chunk_storage_specs(ch)["dense_layers"]["00"]["mlp"]["wo"]["kernel"], P(None, None, ds_ep.FE))
    self.assertEqual(
        ds_ep.chunk_compute_specs(ch)["dense_layers"]["01"]["self_attention"]["wq_a"]["kernel"], P(None, None, None)
    )
    self.assertEqual(ds_ep.acc_shapes(pp.scfg, ch)["dense_layers"]["00"]["mlp"]["wo"]["kernel"], (4, 1, 384, 256))
    names = ds_ep.leaf_names(ch)
    self.assertIn("dense.01.mlp.wi_0", names)
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(len(names), 2 * len(ds_ep.leaf_names(chs)) - 1)  # one table, each dense leaf twice
    host = [jax.tree.map(lambda a: np.zeros(a.shape, a.dtype), ps._avals(c.stage, "storage", c)) for c in ps.chunks]
    with self.assertRaisesRegex(ValueError, "chunk 0: the weights are not this model's"):
      pp.params_from_chunks(host)
    back = [par.relayout(par.relayout(t, a, b), b, a) for t, a, b in zip(host, ps.chunks, pp.chunks)]
    self.assertEqual(jax.tree.structure(back), jax.tree.structure(host))

  def test_saved_residuals(self):
    """Megatron-LM's recipe's path: saved residuals, unsplit.  'stacked' reads its routed experts sliced (its stacked
    reads are refused under saved residuals)."""
    self.check(DENSE_ARGS + ["--save-residuals", "on", "--expert-weights", "sliced"])

  @pytest.mark.scheduled_only
  def test_recompute(self):
    self.check(DENSE_ARGS)

  @pytest.mark.scheduled_only
  def test_batch_split(self):
    self.check(DENSE_ARGS + SPLIT_ARGS)

  @pytest.mark.scheduled_only
  def test_dense_chunks(self):
    """The recipe's chunk 0 (the embedding and the dense layers, no MoE layer: its kinds empty, its dense_kinds not),
    and a layer split that counts every layer, the dense layers over chunks 0 and 1, with saved residuals."""
    for split in ("0,2,1,1", "1,2,2,1"):
      with self.subTest(layer_split=split):
        pp = self.check(DENSE_ARGS + ["--layer-split", split, "--save-residuals", "on", "--expert-weights", "sliced"])
        self.assertEqual([c.dense for c in pp.chunks], [2, 0, 0, 0] if split == "0,2,1,1" else [1, 1, 0, 0])


# ------------------------------------------------------------------ the embedding's gradient
def dense_embed_grad(w, tokens, dh, scfg):
  """ds_ep.embed_grad as autodiff gives it: the embedding's VJP, a zero-filled table with the rows scattered in."""
  _, vjp = jax.vjp(lambda we: ds_ep.embed(we, tokens, scfg), {"token_embedder": w["token_embedder"]})
  return vjp(dh)[0]


def zipf_tokens(p, seed):
  """[gbs, seq] tokens of `p`'s vocabulary drawn under Zipf's law, as words in text: the k-th most frequent token with
  a probability proportional to 1/k (at a vocabulary of 2048 the first 12.2% of the tokens, the first four 25.4%), the
  ranks a random permutation of the ids."""
  V = p.scfg.model.vocab_size
  rng = np.random.default_rng(seed)
  prob = 1 / np.arange(1, V + 1)
  return rng.permutation(V)[rng.choice(V, size=(p.gbs, p.seq), p=prob / prob.sum())].astype(np.int32)


def embed_step(args, dense, repeats):
  """One loss-and-gradients pass (the fusion plan's) of the pipeline `args` builds, from init(key 1), its chunk-0
  backwards taking the embedding's gradient as rows (ds_ep.embed_grad) or, `dense`, as autodiff's table
  (dense_embed_grad) -> (the pipeline, the loss, the stage sumsqs, every reduced gradient leaf by chunk as (path, fp32
  host array)).  The batch: zipf_tokens(3) with repeats='zipf', synthetic_batch(3) with other `repeats`, else every
  token of a microbatch a different one (the vocabulary must hold a microbatch's tokens)."""
  with mock.patch.object(ds_ep, "embed_grad", dense_embed_grad) if dense else contextlib.nullcontext():
    p = dsp.build(dsp.make_parser().parse_args(args))
    params, _, wc0 = p.init(jax.random.key(1))
    if repeats == "zipf":
      tok = zipf_tokens(p, 3)
      batch = p.put_batch(tok, np.roll(tok, -1, axis=1))
    elif repeats:
      batch = p.synthetic_batch(3)
    else:
      rng = np.random.default_rng(3)
      tok = np.concatenate(
          [rng.permutation(p.scfg.model.vocab_size)[: p.mbs * p.seq].reshape(p.mbs, p.seq) for _ in range(p.M)]
      ).astype(np.int32)
      batch = p.put_batch(tok, np.roll(tok, -1, axis=1))
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
    flat = jax.tree_util.tree_flatten_with_path(p.chunk_trees(grads))[0]
    return p, float(jnp.sum(lacc)), [float(s) for s in sqs], [(jax.tree_util.keystr(k), leaves(v)[0]) for k, v in flat]


def table_ops(jaxpr, shapes):
  """{primitive: how many equations of `jaxpr` (sub-jaxprs included) output an array of one of `shapes`}, a
  broadcast_in_dim of a scalar (a constant table: zeros) counted as 'fill', of an array (at these shapes an axis of
  one added) as 'reshape', and a scatter-add with unique indices (TableRows.into's below fp32: each id once, its rows
  summed) as 'scatter-add unique'."""
  out = collections.Counter()
  for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
    name = eqn.primitive.name
    if name == "broadcast_in_dim":
      name = "fill" if not eqn.invars[0].aval.shape else "reshape"
    elif name == "scatter-add" and eqn.params["unique_indices"]:
      name = "scatter-add unique"
    out[name] += any(tuple(v.aval.shape) in shapes for v in eqn.outvars)
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          out.update(table_ops(sub, shapes))
  return +out


@needs_tokamax
@testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
class EmbedRows(unittest.TestCase):
  """The gathered embedding's weight gradient straight into the accumulator (--embed gather, the GPU default;
  ds_ep.embed_grad, ds_ep.TableRows): chunk 0's accumulating backward scatter-adds a microbatch's rows into the donated
  accumulator, its first backward into a fresh one, instead of adding autodiff's zero-filled [vocab, emb] table whole.
  Against that table, into an fp32 accumulator: the sums are its own where no token repeats on a device, so a step is
  bit for bit in fp32 -- with saved residuals and recomputed; where tokens repeat, a repeated token's rows add into the
  accumulator one after the other rather than summed first: that leaf to fp32 rounding, every other bit for bit.  A
  bf16 accumulator takes each token's rows summed in fp32, once a microbatch, as it took the table: bit for bit the
  table's with tokens repeating (fp32 compute).  Under grad_reduce 'microbatch' each backward reduce-scatters the
  table, which the rows make whole first: bit for bit either way (fp32 compute).  Those two with tokens repeating are
  bit for bit on CPU, whose scatter-adds take a repeated index's updates in their order; a GPU's are atomics, whose
  order changes from run to run, so off CPU a repeated token's leaf is held to rounding in both: in fp32 to 2^-20 of
  its largest magnitude, as above, in bf16 to 2^-7.  These pipelines run fp32 compute, whose rows are fp32, so that
  autodiff's table sums them in fp32 too.  The recipes run bf16 compute, whose rows are bf16 (the cotangent in the
  compute copy's dtype, with fp32 master weights too), and autodiff's scatter would add a repeated token's rows in
  bf16: the rows' sums in fp32, into a bf16 accumulator and in dense() at grad_reduce 'microbatch', are held on
  TableRows itself (test_rows_below_fp32, fp32 and bf16 rows).  One pipeline runs on every PR,
  test_saved_residuals; a bf16 accumulator's step is slow (test_bf16_accumulator, scheduled_only), and
  test_no_table_add holds its programs to TableRows' sums on every PR."""

  ARGS = DENSE_ARGS + ["--embed", "gather", "--vocab", "2048", "--layer-tree", "per_layer"]

  def check(self, args, repeats, near=()):
    p, *got = embed_step(args, False, repeats)
    _, *want = embed_step(args, True, repeats)
    self.assertFalse(p.scfg.iota_embed)
    self.assertEqual(got[0], want[0], "loss")
    np.testing.assert_allclose(got[1], want[1], rtol=1e-6 if near else 0, err_msg="stage sumsqs")
    for (path, a), (path_b, b) in zip(got[2], want[2], strict=True):
      self.assertEqual(path, path_b)
      if any(n in path for n in near):
        top = float(np.max(np.abs(b)))
        self.assertLessEqual(float(np.max(np.abs(a - b))), 2**-20 * top, path)
        self.assertGreater(float(np.max(np.abs(a))), 0, path)
      else:
        np.testing.assert_array_equal(a, b, err_msg=path)

  def test_rows(self):
    """TableRows in fp32 against numpy's np.add.at and autodiff's VJP of the gather.  Into an fp32 accumulator block
    ([1, V, D]), with fp32 rows or bf16, the program is the plain scatter-add's: the rows add as they come, a repeated
    id's one after the other -- on CPU np.add.at's sums bit for bit (XLA:CPU's scatter adds its updates in order, as
    np.add.at does), elsewhere to fp32 rounding (assert_same: a GPU's scatter-adds are atomics, whose order changes
    from run to run).  Summed first, as below fp32, a repeated id's rows put 44 of the 512 entries here 1 to 4 fp32
    ulps off.  The dense table the VJP's, bit for bit on CPU."""
    V, D, n = 64, 8, 48
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, n).astype(np.int32)
    self.assertLess(len(set(ids.tolist())), n)  # repeats
    vals = rng.standard_normal((n, D)).astype(np.float32)
    acc = rng.standard_normal((1, V, D)).astype(np.float32)
    rows = ds_ep.TableRows(jnp.asarray(ids), jnp.asarray(vals), (V, D))
    want = acc.copy()
    np.add.at(want[0], ids, vals)
    assert_same(rows.into(jnp.asarray(acc)), want, rtol=1e-6, atol=1e-6)
    for r in (rows, ds_ep.TableRows(rows.ids, rows.values.astype(jnp.bfloat16), (V, D))):
      with self.subTest(rows=r.values.dtype.name):
        plain = lambda a, r=r: a.reshape(V, D).at[r.ids].add(r.values.astype(a.dtype)).reshape(a.shape)
        self.assertEqual(str(jax.make_jaxpr(r.into)(jnp.asarray(acc))), str(jax.make_jaxpr(plain)(jnp.asarray(acc))))
    table = jnp.asarray(rng.standard_normal((V, D)).astype(np.float32))
    _, vjp = jax.vjp(lambda t: t.at[jnp.asarray(ids)].get(), table)
    assert_same(rows.dense(), vjp(jnp.asarray(vals))[0], rtol=1e-6, atol=1e-6)
    self.assertEqual(rows.dense(jnp.bfloat16).dtype, jnp.bfloat16)

  def test_rows_below_fp32(self):
    """Into a bf16 accumulator each id's rows -- fp32, or bf16 as the recipes' bf16 compute gives them -- are summed in
    fp32 and added once, as autodiff's table is added (whose scatter sums bf16 rows in bf16): the accumulator plus
    dense(bf16), bit for bit.  4096 rows of ones on one id give 4096 -- added one by one, or summed in bf16, a bf16 sum
    stops at 256, where a row is half its ulp and the tie rounds to even -- and the rows no id names keep their value;
    Zipf's law's ids, 4096 of them on 64 rows, give their exact sums rounded once to bf16, bit for bit.  The rows are
    multiples of 1/16 below 5 in magnitude, exact in bf16, so every partial sum (a multiple of 1/16 below 2^15) is
    exact in fp32 and the sums are the same in any order: the checks do not depend on the order a backend adds in (a
    GPU's segment sum adds with atomics, in an order that changes from run to run).  With bf16 rows dense(), the table
    grad_reduce 'microbatch' reduce-scatters, is dense(bf16)."""
    V, D, n = 64, 8, 4096
    rng = np.random.default_rng(0)
    prob = 1 / np.arange(1, V + 1)
    ids = rng.choice(V, n, p=prob / prob.sum()).astype(np.int32)
    vals = (np.round(rng.standard_normal((n, D)) * 16) / 16).astype(np.float32)
    self.assertLess(float(np.max(np.abs(vals))), 5)
    exact = np.zeros((V, D))
    np.add.at(exact, ids, vals.astype(np.float64))
    acc = jnp.asarray(rng.standard_normal((1, V, D)), jnp.bfloat16)
    for dt in (jnp.float32, jnp.bfloat16):
      with self.subTest(rows=jnp.dtype(dt).name):
        ones = ds_ep.TableRows(jnp.zeros(n, jnp.int32), jnp.ones((n, D), dt), (V, D))
        got = np.asarray(ones.into(jnp.zeros((1, V, D), jnp.bfloat16)).astype(jnp.float32))
        np.testing.assert_array_equal(got[0, 0], np.full(D, n, np.float32))
        np.testing.assert_array_equal(got[0, 1:], np.zeros((V - 1, D), np.float32))
        rows = ds_ep.TableRows(jnp.asarray(ids), jnp.asarray(vals, dt), (V, D))
        np.testing.assert_array_equal(np.asarray(rows.values.astype(jnp.float32)), vals)  # exact in bf16 too
        table = rows.dense(jnp.bfloat16)
        rounded = np.asarray(table.astype(jnp.float32))
        np.testing.assert_array_equal(rounded, exact.astype(np.float32).astype(jnp.bfloat16).astype(np.float32))
        self.assertTrue(np.any(rounded != exact))  # sums that bf16 rounds
        np.testing.assert_array_equal(
            np.asarray(rows.into(acc).astype(jnp.float32)), np.asarray((acc + table).astype(jnp.float32))
        )
        if dt == jnp.bfloat16:
          np.testing.assert_array_equal(np.asarray(rows.dense().astype(jnp.float32)), rounded)

  def test_no_table_add(self):
    """Chunk 0's accumulating backward holds no full [vocab, emb] table but the accumulator's, scattered into once
    (no zero table, no add of one, as autodiff's has); its first backward one zero table, the fresh accumulator -- a
    bf16 accumulator's too, whose fp32 sums of a token's rows take arrays of the rows' size, and whose scatter takes
    each token once (unique indices: TableRows.into's sums below fp32, not the rows one by one).  (The recipe's chunk
    0, the embedding and the dense layers alone, and a vocabulary no other array's dim is.)"""

    def ops(acc, dense=False, acc_dtype="float32"):
      with (mock.patch.object(ds_ep, "embed_grad", dense_embed_grad) if dense else contextlib.nullcontext(),):
        p = dsp.build(
            dsp.make_parser().parse_args(
                self.ARGS
                + ["--save-residuals", "on", "--layer-split", "0,2,1,1", "--vocab", "1536", "--acc-dtype", acc_dtype]
            )
        )
        desc = (ds_ep.pdesc("B", acc=acc),)
        jx = jax.make_jaxpr(p.progs[0].program(desc))(*p._desc_avals(0, desc))
      V, D = p.scfg.model.vocab_size, p.scfg.model.emb_dim
      return {k: n for k, n in table_ops(jx, {(V, D), (1, V, D)}).items() if k not in ("reshape", "squeeze")}

    for acc_dtype, scatter in (("float32", "scatter-add"), ("bfloat16", "scatter-add unique")):
      with self.subTest(acc_dtype=acc_dtype):
        self.assertEqual(ops(acc=True, acc_dtype=acc_dtype), {scatter: 1})
        self.assertEqual(ops(acc=False, acc_dtype=acc_dtype), {scatter: 1, "fill": 1})
    self.assertEqual(ops(acc=True, dense=True), {"scatter-add": 1, "fill": 1, "add": 1})

  @pytest.mark.scheduled_only
  def test_saved_residuals(self):
    self.check(self.ARGS + ["--save-residuals", "on"], repeats=False)
    self.check(self.ARGS + ["--save-residuals", "on"], repeats=True, near=("['token_embedder']",))

  @pytest.mark.scheduled_only
  def test_bf16_accumulator(self):
    """--acc-dtype bfloat16 (the GB200 commands', the 64-B200 recipe's bf16 column) on Zipf's law's tokens over 16
    microbatches of a sequence a device: a device's most frequent token comes ~500 times a step.  Each microbatch adds
    a token's rows summed in fp32, once, as it added autodiff's table, so with fp32 compute the loss, the stage sumsqs
    and every leaf are the table's: bit for bit on CPU, whose scatter-adds take a repeated token's rows in their order.
    A GPU's are atomics, whose order changes from run to run, so off CPU the embedding's gradient is held to the
    table's within 2^-7 of its largest magnitude and the sumsqs to 1e-3 (each microbatch's rows summed in another order
    on CPU moved at most one of the gradient's 524288 entries, by 1.0e-7 of its largest magnitude: three orders on each
    jax).  Against an fp32 accumulator the embedding's gradient is within 2^-7 of its largest magnitude (5.0e-3 of it
    on jax 0.11.1, 3.5e-3 on 0.12.0.1).  Added one by one into the bf16 accumulator, a token's rows rounded away once
    the sum was ~256 times a row: 6.1e-2 on both.  The GB200 commands and the 64-B200 recipe run bf16 compute, so
    their rows are bf16 (embed_grad casts the cotangent to the compute copy's dtype, bf16 with fp32 master weights
    too), which autodiff's table would sum in bf16; here fp32 compute gives fp32 rows, and test_rows_below_fp32 holds
    bf16 rows.  Slow (three pipelines, 16 microbatches each): scheduled_only, test_rows_below_fp32 and
    test_no_table_add holding the path on every PR."""
    args = self.ARGS + ["--save-residuals", "on", "--mb", "16", "--gbs", "64"]
    _, *ref = embed_step(args, False, "zipf")  # an fp32 accumulator
    p, *got = embed_step(args + ["--acc-dtype", "bfloat16"], False, "zipf")
    _, *want = embed_step(args + ["--acc-dtype", "bfloat16"], True, "zipf")
    self.assertEqual((p.M, p.scfg.acc_dtype), (16, jnp.bfloat16))
    emb = lambda run: next(a for path, a in run[2] if "['token_embedder']" in path)
    top = float(np.max(np.abs(emb(ref))))
    self.assertLess(float(np.max(np.abs(emb(got) - emb(ref)))), 2**-7 * top, "against an fp32 accumulator")
    exact = jax.default_backend() == "cpu"
    self.assertEqual(got[0], want[0], "loss")
    np.testing.assert_allclose(got[1], want[1], rtol=0 if exact else 1e-3, err_msg="stage sumsqs")
    for (path, a), (path_b, b) in zip(got[2], want[2], strict=True):
      self.assertEqual(path, path_b)
      if exact or "['token_embedder']" not in path:
        np.testing.assert_array_equal(a, b, err_msg=path)
      else:
        self.assertLess(float(np.max(np.abs(a - b))), 2**-7 * top, path)

  @pytest.mark.scheduled_only
  def test_recompute(self):
    self.check(self.ARGS, repeats=False)
    self.check(self.ARGS, repeats=True, near=("['token_embedder']",))

  @pytest.mark.scheduled_only
  def test_grad_reduce_microbatch(self):
    self.check(
        self.ARGS + ["--grad-reduce", "microbatch"],
        repeats=True,
        near=() if jax.default_backend() == "cpu" else ("['token_embedder']",),
    )


# ------------------------------------------------------------------ interleaved F&B pairs
# SHAPE_ARGS's model under the GPU defaults (--platform gpu) but for what does not run on CPU -- cuDNN's attention (the
# reference kernel instead) and tokamax's Mosaic backend (its xla one) -- on 2 stages of EP 2 at fsdp 1, two one-layer
# chunks a stage (V = 2), interleaved in groups of 2 over 4 microbatches, fp32 compute and weights (so the routed
# experts are read in place, --direct-experts).
PAIRS_ARGS = SHAPE_ARGS + [
    "--platform",
    "gpu",
    "--attn",
    "reference",
    "--tokamax-gmm",
    "xla",
    "--vv",
    "2",
    "--layers",
    "4",
    "--mb",
    "4",
    "--schedule",
    "interleaved",
    "--interleave-group",
    "2",
    "--dtype",
    "float32",
    "--rs-dtype",
    "float32",
    "--weight-dtype",
    "float32",
]


@needs_tokamax
@testing.needs_devices(4)
class InterleavedPairs(unittest.TestCase):
  """--schedule interleaved --pair-stages all: the steady state's F(k+W) and B(k) of each stage as one program (an F&B
  pair, Megatron-LM's combined 1F1B: one microbatch's EP collectives beside another's compute) against the order
  unpaired.  On CPU, PAIRS_ARGS (the GPU defaults: tokamax's grouped matmuls on the compact buffer, the triton
  combine path -- XLA's sum off GPU -- per-layer trees, direct experts).  Each stage runs the same parts in the same
  sequence, composed in the pairs, under the same fusion plan, so one step's loss partials, stage sumsqs and every
  reduced gradient leaf are the unpaired run's bit for bit, with saved residuals and without, batch split
  (ds_ep.split_layers, the GPU's) and not.  Six pairs: on stage 0 a chunk's forward beside the other chunk's
  backward, on stage 1 chunk 1's own two microbatches (its other chunk is the fused last one, which never pairs).
  Each pair's lowered program donates the backward's accumulator, to be added to in place, and nothing else
  (assert_pair_donations)."""

  def run_step(self, args):
    """run_pipe's pass (init key 1, synthetic batch 3) of the pipeline `args` builds -> (the pipeline, [the loss
    partials, each stage's sumsq, every reduced gradient leaf])."""
    p = dsp.build(dsp.make_parser().parse_args(args))
    grads, sqs, lacc = run_pipe(p, 1, 3)
    return p, [lacc, *sqs, *jax.tree.leaves(p.chunk_trees(grads))]

  def test_composed_donations(self):
    """A backward donates the saved set, the stash and the dy it reads, and its accumulator -- beside a forward, in an
    F&B pair, only its accumulator, and the pair returns the backward's outputs first: no buffer goes to a forward
    output of its shape and dtype, which XLA would copy while the interleaved backward still reads it
    (ds_ep.composed_donations, ds_ep.composed_order).  The fused last chunk's part ('A'), no forward, leaves the
    backward's donations and the parts' order as they are; and a pair, which folds in no gather, takes no prefetch
    pin."""
    part = lambda name, n, **kw: ds_ep.Part(name, None, (None,) * n, (), (None,) * n, (), **kw)
    fwd, fused = part("fwd_c0_st", 3, forward=True), part("fba_c3", 6, donate=(1, 3, 4))
    # w, x, saved set, stash, dy, gacc, seed
    bwd = part("bwda_c2_st", 7, donate=(2, 3, 4, 5), backward=True, fwd_like=(2, 3, 4))
    self.assertEqual(ds_ep.composed_donations([bwd]), (2, 3, 4, 5))
    self.assertEqual(ds_ep.composed_donations([fwd, bwd]), (8,))
    self.assertEqual(ds_ep.composed_donations([fused, bwd]), (1, 3, 4, 8, 9, 10, 11))
    self.assertEqual(
        [ds_ep.composed_order(x) for x in ([bwd], [fwd], [fwd, bwd], [fused, bwd])], [(0,), (0,), (1, 0), (0, 1)]
    )
    with self.assertRaisesRegex(ValueError, "an F&B pair folds in no gather, so it pins none"):
      ds_ep.compose(None, [part("fwd_c0", 3, forward=True, pins=(("hn", 0, 0, 0),)), bwd], "fwd_c0__bwda_c2_st")

  def pair_aliases(self, p, c, desc):
    """What jit made of each argument of the F&B pair `p.progs[c].program(desc)`, lowered: {(part, argument): [one
    alias a leaf]} -- part 0 the forward, 1 the backward, each argument at its ChunkPrograms.part position -- where an
    alias is (part, i) when the leaf's buffer becomes the program's output i, one of that part's (tf.aliasing_output;
    the pair returns the backward's outputs, then the forward's: ds_ep.composed_order), 'donor' when XLA may reuse it
    for an output (jax.buffer_donor), None when it is not donated."""
    avals = p._desc_avals(c, desc)
    low = p.progs[c].program(desc).lower(*avals)
    kept = getattr(getattr(low, "_lowering", None), "compile_args", {}).get("kept_var_idx")
    if kept is None:
      self.skipTest(f"jax {jax.__version__} keeps a lowering's kept arguments elsewhere")
    parts = []
    for kind, kw in desc:
      kw = dict(kw)
      parts.append(kw.pop("peer", p.progs[c]).part(kind, **kw))
    arg_of, i = [], 0  # each flat argument's (part, argument)
    for k, part in enumerate(parts):
      for a in range(len(part.in_specs)):
        arg_of += [(k, a)] * len(jax.tree.leaves(avals[i]))
        i += 1
    out_of, i = [], 0  # each flat output's part, the backward's first
    for k in (1, 0):
      for o in low.out_info[i : i + len(parts[k].out_specs)]:
        out_of += [k] * len(jax.tree.leaves(o))
      i += len(parts[k].out_specs)
    main = re.search(r"func\.func public @main\((.*?)\) -> ", low.as_text(), re.S).group(1)
    got = {}
    for j, arg in zip(sorted(kept), re.split(r", (?=%arg\d+:)", main), strict=True):
      m = re.search(r"tf\.aliasing_output = (\d+)", arg)
      alias = (out_of[int(m.group(1))], int(m.group(1))) if m else "donor" if "jax.buffer_donor" in arg else None
      got.setdefault(arg_of[j], []).append(alias)
    return got

  def assert_pair_donations(self, p):
    """Each pair's lowered program (ds_ep.compose of the forward's part and the backward's) donates every leaf of the
    backward's accumulator, where it takes one, and nothing else: no argument of the forward, and none of the
    backward's saved set, stash and dy, arrays of the forward's outputs' shapes (ds_ep.composed_donations).  And each
    accumulator leaf's buffer becomes the backward's gradient leaf (tf.aliasing_output), in place, not a forward
    output, which XLA would copy while the backward still reads it: jit hands each output, in order, the first
    donated buffer of its shape and dtype, and the pair returns the backward's outputs first (ds_ep.composed_order),
    ahead of any forward output of an accumulator leaf's shape and dtype -- here, in fp32 with seq = emb = moe-mlp =
    256, a saved activation or a stash leaf can have a weight's."""
    pairs = list(dict.fromkeys((op.c, op.pair) for op in p.ops(True) if op.pair))
    self.assertTrue(pairs)
    for c, desc in pairs:
      (fkind, _), (bkind, bkw) = desc
      self.assertEqual((fkind, bkind), ("F", "B"))
      bkw = dict(bkw)
      dy = 4 if "stash" in bkw else 3  # the backward's w, x, saved set, [stash,] dy, [gacc,] seed
      gacc = dy + 1 if bkw.get("acc") else None
      dx = int(not bkw.get("peer", p.progs[c]).ch.first)  # the backward's outputs: dx (but chunk 0's), gacc
      for (part, arg), aliases in self.pair_aliases(p, c, desc).items():
        what = f"{p.progs[c].program(desc).__name__}: part {part}, argument {arg}"
        if (part, arg) != (1, gacc):
          self.assertEqual(set(aliases), {None}, what)
        else:
          self.assertEqual(aliases, [(1, dx + i) for i in range(len(aliases))], what)

  def test_lowered_donations(self):
    """The pairs' donations without a step (assert_pair_donations): recompute; a stash, whose forwards return stash
    leaves of a weight's shape and dtype here; and saved residuals in bf16 with a bf16 accumulator (the GB200
    commands' --acc-dtype), whose saved activations do too."""
    for extra in (
        ["--gbs", "8"],
        ["--gbs", "8", "--stash", "lite"],
        ["--gbs", "8", "--save-residuals", "on", "--dtype", "bfloat16", "--acc-dtype", "bfloat16"],
    ):
      with self.subTest(extra=extra):
        p = dsp.build(dsp.make_parser().parse_args(PAIRS_ARGS + extra + ["--pair-stages", "all"]))
        self.assertEqual(len(p.fuse), 6)
        self.assert_pair_donations(p)

  def check(self, extra):
    p, got = self.run_step(PAIRS_ARGS + extra + ["--pair-stages", "all"])
    q, want = self.run_step(PAIRS_ARGS + extra)
    self.assertEqual(
        (p.scfg.gmm_impl, p.scfg.combine_impl, p.scfg.layer_tree, p.scfg.use_ragged_sort),
        ("tokamax", "triton", "per_layer", False),
    )
    self.assertEqual(p.direct, {ch.index: ds_ep.routed_leaf_ids(ch) for ch in p.chunks})
    self.assertEqual(
        sorted(p.op_name(j) + "&" + p.op_name(j + 1) for j in p.fuse),
        ["0F2&2B0", "0F3&2B1", "1F2&1B0", "1F3&1B1", "2F2&0B0", "2F3&0B1"],
    )
    self.assertFalse(q.fuse)
    self.assertEqual(stage_seqs(p), stage_seqs(q))
    by_op = lambda x, d: {x.order[j]: v for j, v in d.items()}
    plan = lambda x: (by_op(x, x.pf), by_op(x, x.rs_at), by_op(x, x.rs_self), x.final_slots, x.gather_slots)
    self.assertEqual(plan(p), plan(q))
    assert_equal_leaves(self, got, want, "paired vs unpaired")
    self.assert_pair_donations(p)

  def test_saved_residuals(self):
    """Megatron-LM's recipe: no recompute, one row a device (no batch split)."""
    self.check(["--save-residuals", "on", "--gbs", "8"])

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
  def test_recompute(self):
    self.check(["--gbs", "8"])

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
  def test_batch_split_saved_residuals(self):
    """Batch split wants an even number of rows a device: 4 a microbatch."""
    self.check(["--save-residuals", "on", "--gbs", "16", *SPLIT_ARGS])

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "E2E_ARGS's two stages of fsdp 2 x ep 2")
  def test_batch_split_recompute(self):
    self.check(["--gbs", "16", *SPLIT_ARGS])


# ------------------------------------------------------------------ lean residuals
def lean_and_full(f, args, cts):
  """f's output and its VJP of `cts` at `args`, each jitted as one program with the cotangents an argument, under
  lean residuals and without -> (lean, full)."""

  def run(lean):
    def g(args, cts):
      with dl.lean_residuals(lean):
        y, vjp = jax.vjp(f, *args)
      return y, vjp(cts)

    return jax.jit(g)(args, cts)

  return run(True), run(False)


def vjp_kept(f, args, lean):
  """(shape, dtype, i) of every array f's VJP at `args` keeps, traced abstractly (as the pipeline's forwards trace)
  under lean residuals or not: i is the flat index of the argument it is, None for an array f computes."""
  kept = []

  def trace(*args):
    with dl.lean_residuals(lean):
      _, vjp = jax.vjp(f, *args)
    flat = jax.tree.leaves(args)
    kept.extend(
        (tuple(a.shape), jnp.dtype(a.dtype), next((i for i, q in enumerate(flat) if a is q), None))
        for a in jax.tree.leaves(vjp)
        if hasattr(a, "shape")
    )

  jax.eval_shape(trace, *args)
  return kept


def shard_kept(f, mesh, in_specs, out_specs, args, lean):
  """The shape of each array f's VJP keeps inside a shard_map over `mesh` (`in_specs`: `args`'), a device's view:
  traced abstractly, as a saved-residual forward traces its layers' VJP, each array once, the primal inputs left out
  (ring_kept's, for any f), under lean residuals or not -> [shape]."""
  kept = []

  def body(*args):
    with dl.lean_residuals(lean):
      y, vjp = jax.vjp(f, *args)
    prim, seen = jax.tree.leaves(args), []
    for a in jax.tree.leaves(vjp):
      if hasattr(a, "shape") and not any(a is q for q in seen + prim):
        seen.append(a)
        kept.append(tuple(a.shape))
    return y

  aval = lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=NamedSharding(mesh, s))
  jax.eval_shape(
      jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False),
      *jax.tree.map(aval, args, in_specs),
  )
  return kept


def contracting_matmuls(jaxpr):
  """(lhs shape, rhs shape) of every dot_general of `jaxpr` and its sub-jaxprs that contracts a dim -- the
  projections, their gradients -- sorted; RMSNorm's scale product, whose one shared dim is a batch dim, is not one."""
  out = []
  for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
    if eqn.primitive.name == "dot_general" and eqn.params["dimension_numbers"][0][0]:
      out.append((tuple(eqn.invars[0].aval.shape), tuple(eqn.invars[1].aval.shape)))
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          out.extend(contracting_matmuls(sub))
  return sorted(out)


def mla_case(dtype, rope_factor=40.0, kernel="reference"):
  """DeepSeek-V3's attention at small dims (emb 128, 4 heads, q / kv LoRA 64 / 32, qk 32 + 16, v 24, 64 tokens):
  -> (the config, random MLA and pre-attention norm weights, the residual stream x [2, 64, 128], mla's other
  arguments: positions, segment ids, the rotation table).  `rope_factor` 1 is the plain rotation, 40 YaRN."""
  cfg = DeepSeekConfig(
      emb_dim=128,
      num_query_heads=4,
      num_kv_heads=4,
      q_lora_rank=64,
      kv_lora_rank=32,
      qk_nope_head_dim=32,
      qk_rope_head_dim=16,
      v_head_dim=24,
      attention_kernel=kernel,
      max_target_length=64,
      rope_factor=rope_factor,
      max_position_embeddings=163840 if rope_factor > 1 else 4096,
      dtype=dtype,
  )
  # every leaf of a MoE layer drawn in tree order, the MoE block's first, so that the attention's weights and x are
  # the draws test_mla's fp32 bound was measured on
  r = np.random.default_rng(0)
  w = jax.tree.map(
      lambda s: jnp.asarray((1.0 if len(s) == 1 else 0.0) + 0.08 * r.standard_normal(s), cfg.weight_dtype),
      ds_ep.layer_shapes(cfg),
      is_leaf=lambda s: isinstance(s, tuple),
  )
  x = jnp.asarray(r.standard_normal((2, 64, 128)), dtype)
  pos = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), (2, 64))
  return (
      cfg,
      w["self_attention"],
      w["pre_self_attention_layer_norm"],
      x,
      (pos, jnp.ones((2, 64), jnp.int32), rope.freqs_cis_for(cfg, 64)),
  )


def saved_chunk(lean, mbs=8, seq=128):
  """A middle chunk of a dense and a MoE layer (tiny_xla's, per-layer trees, the reference attention) on a
  one-stage 2 x 2 mesh, its residuals saved -- lean, or the full VJP's -- its programs run on the stage's init and
  random activations -> (its ChunkPrograms, its backward program's jaxpr, [y, dx, *the chunk's per-device gradient
  partials] as fp32 host arrays)."""
  scfg = tiny_xla(2, layer_tree="per_layer", save_residuals=True)
  scfg = dataclasses.replace(
      scfg, model=scfg.model.replace(attention_kernel="reference", first_num_dense_layers=1, mlp_dim=384, moe_mlp_dim=128)
  )
  mesh = dsp.stage_meshes(1, scfg.fsdp, scfg.ep)[0]
  ch = ds_ep.ChunkSpec(
      index=1,
      layers=(0,),
      first=False,
      last=False,
      stage=0,
      dense=1,
      kinds=ds_ep.layer_kinds(scfg.model, (0,), True),
      dense_kinds=ds_ep.dense_layer_kinds(1, True),
  )
  rng = np.random.default_rng(1)
  put = lambda a: jax.device_put(jnp.asarray(a, scfg.model.dtype), NamedSharding(mesh, P(ds_ep.FE, None, None)))
  x, dy = (put(rng.standard_normal((mbs, seq, scfg.model.emb_dim))) for _ in range(2))
  real = dl.lean_residuals
  with mock.patch.object(dl, "lean_residuals", lambda on=True: real(on and lean)):
    sp = ds_ep.StagePrograms(scfg, [ch], mesh)
    cp = ds_ep.ChunkPrograms(scfg, ch, mesh, mbs, seq, mbs * seq)
    wc = sp.gather(sp.init(jax.random.key(0)))[0]
    y, res = cp.fwd(wc, x, jnp.int32(5))
    bwd = jax.make_jaxpr(cp.bwd)(wc, x, res, dy, jnp.int32(5))
    dx, gacc = cp.bwd(wc, x, res, dy, jnp.int32(5))
  return cp, bwd, leaves([y, dx, gacc])


@testing.needs_devices(4)
class LeanResiduals(unittest.TestCase):
  """What lean residuals (deepseek/layers.lean_residuals, which a saved-residual forward traces under) keep of
  attention, RoPE and the RMSNorms, at DeepSeek-V3 dims 1.03 GiB less of a two-MoE-layer chunk's 7.55 a microbatch
  (one sequence a GPU, EP 8, buffer factor 2): RoPE keeps its rows [b, s, 1, half], not broadcast over the heads;
  RMSNorm its input and scale; a norm and the projections that read its output run in one checkpoint
  (layers.rms_norm_read: mla's pre-attention norm, its query's and, without context parallelism, its key / value
  latent's, ds_ep's post-attention norm with the router and the shared expert, a dense layer's with its MLP), so that
  no VJP keeps a norm's output (but wkv_b's, the gathered latent, under context parallelism:
  test_mla_context_parallel); and the attention output takes no name, so that the out projection keeps the array
  cuDNN's VJP keeps.  Each function alone, jitted with the cotangents as arguments, computes what it computes without
  them bit for bit -- fp32 and bf16, the plain rotation and YaRN, both pair layouts; but mla's fp32 cotangents, which
  XLA:CPU may round apart (jax 0.11.1's has on other hosts, and both versions do here under context parallelism), to
  4 fp32 ulps (test_mla, test_mla_context_parallel) -- and no backward runs a projection again: the same contracting
  matmuls with and without.  A whole chunk's programs fuse the recomputation apart and round apart in fp32, as the
  lean residuals before these did: there to fp32 rounding."""

  def test_rope(self):
    """The plain rotation and YaRN, adjacent pairs (DeepSeek's rope_interleave) and halves, fp32 and bf16: the
    rotation and its input cotangent bit for bit; the lean VJP keeps the rows as [b, s, 1, half], the full one
    their broadcast over the 4 heads."""
    x32, ct32 = (jax.random.normal(jax.random.key(i), (2, 64, 4, 16)) for i in range(2))
    pos = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), (2, 64))
    for factor in (1.0, 40.0):
      for interleave in (True, False):
        for dtype in (jnp.float32, jnp.bfloat16):
          cfg = DeepSeekConfig(
              qk_rope_head_dim=16,
              rope_factor=factor,
              rope_interleave=interleave,
              dtype=dtype,
              max_position_embeddings=163840 if factor > 1 else 4096,
          )
          table = rope.freqs_cis_for(cfg, 64)
          f = lambda x, cfg=cfg, table=table: rope.apply_yarn(x, pos, table, cfg)
          x = x32.astype(dtype)
          with self.subTest(factor=factor, interleave=interleave, dtype=jnp.dtype(dtype).name):
            lean, full = lean_and_full(f, (x,), ct32.astype(dtype))
            assert_equal_leaves(self, lean, full, "rotation, its input cotangent")
            heads = lambda lean, f=f, x=x: {s[-2] for s, _, _ in vjp_kept(f, (x,), lean) if len(s) == 4}
            self.assertEqual((heads(True), heads(False)), ({1}, {4}))

  def test_rms_norm(self):
    """RMSNorm keeps its input and its scale, nothing it computes; a norm and a projection then silu
    (rms_norm_read) keep the input, the weights and the projection's output, which silu's backward reads -- no
    array of x's shape but x."""
    ks = jax.random.split(jax.random.key(0), 5)
    for dtype in (jnp.float32, jnp.bfloat16):
      cfg = DeepSeekConfig(dtype=dtype)
      x = jax.random.normal(ks[0], (2, 64, 256)).astype(dtype)
      norm = {"scale": 1 + 0.1 * jax.random.normal(ks[1], (256,))}
      w = {"kernel": 0.06 * jax.random.normal(ks[2], (256, 96))}
      f = lambda norm, x, cfg=cfg: dl.rms_norm(norm, x, cfg)
      g = lambda norm, w, x, cfg=cfg: dl.rms_norm_read(norm, x, cfg, lambda y: jax.nn.silu(dl.dense_general(w, y, cfg)))
      with self.subTest(dtype=jnp.dtype(dtype).name):
        for fn, args, ct in (
            (f, (norm, x), jax.random.normal(ks[3], x.shape).astype(dtype)),
            (g, (norm, w, x), jax.random.normal(ks[4], (2, 64, 96)).astype(dtype)),
        ):
          lean, full = lean_and_full(fn, args, ct)
          assert_equal_leaves(self, lean, full, "output, cotangents")
        self.assertEqual(sorted(i for _, _, i in vjp_kept(f, (norm, x), True)), [0, 1])  # the scale, x
        kept = vjp_kept(g, (norm, w, x), True)
        self.assertEqual(
            sorted((s, i) for s, _, i in kept), sorted([((2, 64, 96), None), ((256,), 0), ((256, 96), 1), (x.shape, 2)])
        )
        self.assertGreater(len(vjp_kept(g, (norm, w, x), False)), len(kept))

  def test_mla(self):
    """mla taking the residual stream and its pre-attention norm (`norm`, ds_ep's call): output and every cotangent
    bit for bit the full residuals' and the plain model's (the norm applied before the call), plain rotation and
    YaRN, fp32 and bf16 -- but in fp32 the cotangents to 4 fp32 ulps of each leaf's largest magnitude: XLA:CPU may
    round the q path's and the norms' apart under the lean checkpoints, as jax 0.11.1's has on other hosts (bit for
    bit on jax 0.11.1 and 0.12.0.1 on an AMD EPYC 9B14).  Its lean VJP keeps no array of the stream's shape but the
    stream, of each LoRA latent's only the latent its norm reads (not the norm's output too), no complex rows over
    the heads, fewer bytes, and its backward runs the full one's contracting matmuls."""
    for factor in (1.0, 40.0):
      for dtype in (jnp.float32, jnp.bfloat16):
        cfg, w, norm, x, rest = mla_case(dtype, factor)
        f = lambda w, norm, x, cfg=cfg, rest=rest: attention.mla(w, x, *rest, cfg, norm=norm)
        outside = lambda w, norm, x, cfg=cfg, rest=rest: attention.mla(w, dl.rms_norm(norm, x, cfg), *rest, cfg)
        ct = jnp.asarray(np.random.default_rng(3).standard_normal(x.shape), dtype)
        with self.subTest(factor=factor, dtype=jnp.dtype(dtype).name):
          lean, full = lean_and_full(f, (w, norm, x), ct)
          for what, want in (
              ("lean vs full residuals", full),
              ("vs the norm outside", lean_and_full(outside, (w, norm, x), ct)[1]),
          ):
            if dtype == jnp.float32:
              assert_equal_leaves(self, lean[0], want[0], what + ": output")
              assert_near(self, lean, want, 4, what, eps=FP32_EPS)
            else:
              assert_equal_leaves(self, lean, want, what)
          kept, every = vjp_kept(f, (w, norm, x), True), vjp_kept(f, (w, norm, x), False)
          self.assertEqual([s for s, _, i in kept if s == x.shape and i is None], [])
          # each LoRA latent's norm runs in one checkpoint with its up-projection: of the latent's shape only the
          # norm's input is kept, not its output (wq_b's / wkv_b's lhs) beside it
          for rank in (cfg.q_lora_rank, cfg.kv_lora_rank):
            latent = (*x.shape[:2], rank)
            self.assertEqual([s for s, _, _ in kept if s == latent], [latent], f"the {rank}-wide latent")
          self.assertEqual({s[-2] for s, d, _ in kept if d == jnp.complex64 and len(s) == 4}, {1})
          nbytes = lambda kept: sum(np.prod(s) * d.itemsize for s, d, i in kept if i is None)
          self.assertLess(nbytes(kept), nbytes(every))
          bwd = [jax.make_jaxpr(lambda ct, lean=lean: self._vjp(f, (w, norm, x), lean)(ct))(ct) for lean in (True, False)]
          self.assertEqual(contracting_matmuls(bwd[0]), contracting_matmuls(bwd[1]))

  @staticmethod
  def _vjp(f, args, lean):
    with dl.lean_residuals(lean):
      return jax.vjp(f, *args)[1]

  @testing.needs_devices(8, "two sequences at cp 4: an expert axis of 8")
  def test_mla_context_parallel(self):
    """mla under context parallelism as ds_ep calls it (ds_ep.cp_kv_context, no segment ids, the pre-attention norm
    inside): mla_case's two sequences cut into cp = 2 and 4 rows, a row a device of the 'expert' axis, fp32 and bf16 --
    output and cotangents lean against full as test_mla holds them (here, in fp32, the cotangents through the key /
    value latent's norm, which the backward recomputes, round apart on both jax versions: up to 1.6 fp32 ulps of a
    leaf's largest magnitude on 0.11.1, 1.2 on 0.12.0.1).  The latent's norm stays out of wkv_b's checkpoint, the
    gathered latent coming between them, so of the latent's two shapes the lean VJP keeps, a device's view, one array
    each: the norm's input [1, s, kv_lora] and wkv_b's lhs, the gathered [1, cp x s, kv_lora] (the full one three of
    the first: the norm's intermediates too)."""
    for cp in (2, 4):
      for dtype in (jnp.float32, jnp.bfloat16):
        cfg, w, norm, x, _ = mla_case(dtype)
        (B, S, D), kv = x.shape, cfg.kv_lora_rank
        ep, s = B * cp, S // cp
        scfg = ds_ep.StageConfig(
            model=cfg,
            fsdp=1,
            ep=ep,
            wi_tiling=(128,) * 9,
            wo_tiling=(128,) * 9,
            gmm_impl="ragged_dot",
            use_ragged_sort=False,
            fsdp_collectives="xla",
            cp=cp,
        )
        mesh = Mesh(np.array(jax.devices()[:ep]).reshape(1, ep), ds_ep.FE)

        def local(w, norm, x, scfg=scfg, cfg=cfg, s=s):  # x: this device's row, chunk cp_chunk of its sequence
          pos = jnp.broadcast_to(ds_ep.cp_chunk(scfg) * s + jnp.arange(s, dtype=jnp.int32), (1, s))
          return attention.mla(
              w, x, pos, None, rope.freqs_cis_for(cfg, scfg.cp * s), cfg, kv_context=ds_ep.cp_kv_context(scfg), norm=norm
          )

        specs = (jax.tree.map(lambda _: P(), w), jax.tree.map(lambda _: P(), norm), P(ds_ep.FE))
        f = jax.shard_map(local, mesh=mesh, in_specs=specs, out_specs=P(ds_ep.FE), check_vma=False)
        rows = lambda a, mesh=mesh, ep=ep, s=s: jax.device_put(a.reshape(ep, s, D), NamedSharding(mesh, P(ds_ep.FE)))
        args, ct = (w, norm, rows(x)), rows(jnp.asarray(np.random.default_rng(3).standard_normal(x.shape), dtype))
        with self.subTest(cp=cp, dtype=jnp.dtype(dtype).name):
          lean, full = lean_and_full(f, args, ct)
          if dtype == jnp.float32:
            assert_equal_leaves(self, lean[0], full[0], "output")
            assert_near(self, lean, full, 4, "lean vs full residuals", eps=FP32_EPS)
          else:
            assert_equal_leaves(self, lean, full, "lean vs full residuals")
          shapes = [(1, s, kv), (1, cp * s, kv)]  # the latent, a device's chunk and gathered
          latents = {
              on: sorted(k for k in shard_kept(local, mesh, specs, P(ds_ep.FE), args, on) if k in shapes)
              for on in (True, False)
          }
          self.assertEqual(latents[True], shapes)
          self.assertGreater(len(latents[False]), len(latents[True]))

  def test_cudnn_output_kept_once(self):
    """Under a stand-in for cuDNN's attention whose VJP keeps its output, as cuDNN's does, the full residuals keep
    that output twice -- the kernel's own, and the out projection's lhs, the 'attention_out' name's array -- and the
    lean ones once (one leaf fewer of the output's shape: the value's is that shape too)."""
    from jax._src.cudnn import fused_attention_stablehlo as cudnn_fa

    attend = lambda q, k, v: attention.reference_attention(q, k, v, None)

    @jax.custom_vjp
    def kernel(q, k, v):
      return attend(q, k, v)

    def fwd(q, k, v):
      out = attend(q, k, v)
      return out, (q, k, v, out)

    def bwd(res, g):
      return jax.vjp(attend, *res[:3])[1](g)

    kernel.defvjp(fwd, bwd)
    cfg, w, norm, x, rest = mla_case(jnp.bfloat16, kernel="cudnn")
    out_shape = (2, 64, cfg.num_query_heads, cfg.v_head_dim)
    f = lambda w, norm, x: attention.mla(w, x, *rest, cfg, norm=norm)
    with mock.patch.object(cudnn_fa, "dot_product_attention", lambda q, k, v, **kw: kernel(q, k, v)):
      n = {lean: sum(s == out_shape for s, _, _ in vjp_kept(f, (w, norm, x), lean)) for lean in (True, False)}
    self.assertEqual(n, {True: 2, False: 3})

  def test_chunk(self):
    """A middle chunk of a dense and a MoE layer, its residuals saved: lean, its set keeps three activations (the
    MoE layer's input and each layer's post-attention stream) where the full one keeps more, and fewer bytes; its
    backward program runs the full one's contracting matmuls (no projection again); its output, dx and every
    weight gradient are the full residuals' to fp32 rounding (y bit for bit: the forward computes the same)."""
    (cp, bwd, got), (cp_full, bwd_full, want) = saved_chunk(True), saved_chunk(False)
    cfg = cp.scfg.model
    act = lambda c: [r for r in c._saved_meta["local"] if r.shape == (2, 128, cfg.emb_dim)]  # 8 rows, 4 devices
    self.assertEqual((len(act(cp)), cp.res_bytes < cp_full.res_bytes), (3, True))
    self.assertGreater(len(act(cp_full)), 3)
    self.assertEqual(contracting_matmuls(bwd), contracting_matmuls(bwd_full))
    np.testing.assert_array_equal(got[0], want[0])
    for i, (a, b) in enumerate(zip(got[1:], want[1:], strict=True)):
      self.assertLessEqual(float(np.max(np.abs(a - b))), 1e-5 * float(np.max(np.abs(b))), i)


@testing.needs_devices(4)
class OffloadResiduals(unittest.TestCase):
  """--offload-residuals on: each (chunk, microbatch)'s forward residuals -- its input, and chunk 0's layer inputs --
  go to the same sharding in pinned host memory right after its forward (Pipeline._offload) and come back to the
  device right before its backward (_restore).  Pure data movement, so a step's loss, stage sumsqs and every reduced
  gradient leaf are offload off's bit for bit; and every forward of a non-last chunk offloaded its whole set, each
  array in pinned host memory on its own sharding, which its backward brought back to the device sharding it left,
  once.  The CPU backend has the pinned_host memory kind (jax 0.11.1), so this runs here as on a GPU -- where XLA's
  scatter-adds are atomics unless --xla_gpu_deterministic_ops=true, and the token embedding's gradient moves in its
  last bits from run to run, offload or not (6e-8 of its largest magnitude between two runs without offload on 4
  A100s; bit for bit under the flag): without the flag a GPU run holds the step to 1e-6 of each leaf's largest
  magnitude."""

  def step(self, args):
    """run_pipe's pass (init key 1, synthetic batch 3) of the pipeline `args` builds over the first four devices,
    with Pipeline._offload and _restore watched -> (the loss, the stage sumsqs, every reduced gradient leaf as (path,
    fp32 host array), the pipeline, per offload (its source leaves, [(source memory kind, copy memory kind, whether
    the copy has the source's sharding in pinned host memory)] of every array it holds), per restore of an offloaded
    set [(memory kind, whether back on the source's sharding)])."""
    offloads, restores, sources = [], [], {}
    real_offload, real_restore = dsp.Pipeline._offload, dsp.Pipeline._restore

    def offload(tree):
      held = real_offload(tree)
      src = jax.tree.leaves(tree)
      copies = [x for x in jax.tree.leaves(held) if isinstance(x, jax.Array)]
      sources[id(held)] = [a.sharding for a in src]
      offloads.append(
          (
              len(src),
              [
                  (
                      a.sharding.memory_kind,
                      h.sharding.memory_kind,
                      h.sharding == a.sharding.with_memory_kind("pinned_host"),
                  )
                  for a, h in zip(src, copies)
              ]
              + [None] * (len(copies) - len(src)),
          )
      )
      return held

    def restore(held):
      out = real_restore(held)
      shardings = sources.pop(id(held), None)
      if shardings is not None:
        restores.append([(b.sharding.memory_kind, b.sharding == sh) for b, sh in zip(jax.tree.leaves(out), shardings)])
      return out

    with (
        mock.patch.object(dsp.Pipeline, "_offload", staticmethod(offload)),
        mock.patch.object(dsp.Pipeline, "_restore", staticmethod(restore)),
    ):
      p = dsp.build(dsp.make_parser().parse_args(args), meshes=dsp.stage_meshes(2, 1, 2, devices=jax.devices()[:4]))
      grads, sqs, lacc = run_pipe(p, 1, 3)
      flat = jax.tree_util.tree_flatten_with_path(p.chunk_trees(grads))[0]
      return (
          float(jnp.sum(lacc)),
          [float(s) for s in sqs],
          [(jax.tree_util.keystr(k), leaves(v)[0]) for k, v in flat],
          p,
          offloads,
          restores,
      )

  def check(self, args):
    *got, p, offloads, restores = self.step(args + ["--offload-residuals", "on"])
    *want, q, none, _ = self.step(args)
    self.assertEqual((p.offload_residuals, q.offload_residuals), (True, False))
    self.assertEqual(none, [])
    n = sum(1 for op in p.ops() if op.kind == "F" and op.c != p.C - 1)
    self.assertEqual(n, (p.C - 1) * p.M)
    self.assertEqual(len(offloads), n)
    self.assertEqual(len(restores), n)  # every set came back, once
    for n_src, kinds in offloads:
      self.assertGreater(n_src, 0)
      self.assertEqual(kinds, [("device", "pinned_host", True)] * n_src)
    for back in restores:
      self.assertTrue(back)
      self.assertEqual(back, [("device", True)] * len(back))
    rel = (
        0.0
        if (jax.default_backend() != "gpu" or "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""))
        else 1e-6
    )
    np.testing.assert_allclose(got[0], want[0], rtol=rel, atol=0, err_msg="loss")
    np.testing.assert_allclose(got[1], want[1], rtol=rel, atol=0, err_msg="stage sumsqs")
    for (path, a), (path_b, b) in zip(got[2], want[2], strict=True):
      self.assertEqual(path, path_b)
      err, top = float(np.max(np.abs(a - b), initial=0.0)), float(np.max(np.abs(b), initial=0.0))
      self.assertLessEqual(err, rel * top, f"gradient leaf {path}: {err:.3g} of {top:.3g}")

  def test_deepseek(self):
    self.check(OFFLOAD_ARGS)

  @pytest.mark.scheduled_only
  def test_qwen(self):
    """Qwen3.8 layers: three Gated DeltaNet chunks, then the attention one."""
    self.check(Q_OFFLOAD_ARGS)


if __name__ == "__main__":
  unittest.main()
