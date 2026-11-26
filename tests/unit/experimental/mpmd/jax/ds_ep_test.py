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

"""CPU check of the FSDP x EP x PP MPMD pipeline against a monolithic reference.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_test.py -q

(XLA_FLAGS="--xla_force_host_platform_device_count=8 --xla_cpu_experimental_ynn_fusion_type=" runs the
classes that need 8 devices too, and the checks XLA:CPU's YNNPACK fusions break: testing.needs_ynn_off.)

The reference is `jax.grad` of ONE shard_map on ONE stage's mesh that runs
every layer for every microbatch, taking the fp32 storage weights and doing
the bf16 cast + all-gather inside -- so its reduce-scatter comes from AD's
transpose of `all_gather`, not from `ds_ep._scatter_leaf`, and its gradient
accumulation across microbatches is AD's, not the pipeline's donated buffers.
Kernels (megablox, ragged sort, tokamax splash) run in interpret / fallback
mode, but they are the same kernels on both sides.

Besides the classes on MaxText's kernels: the configuration / schedule /
plan checks, the pure-JAX `ragged_dot` experts, tokamax splash against the
reference attention, and `PipelineXlaKernels` -- the whole pipeline against
the reference with `gmm_impl='ragged_dot'` and the argsort dispatch (XLA's
ops, no MaxText kernel).  Of one chunk's programs against each other: the
stash and composed programs (StashAndCompose) and the zero-bubble
schedules' split backward, I and W, against the backward it is cut from
(SplitBackward), bit for bit.  The tests of ten seconds or more, and
`PipelineVsReference` and `StashAndCompose` (the pipeline on MaxText's
kernels), are marked scheduled_only (of `PipelineXlaKernels` the fp32 check
against the reference and the layer-split and cost checks run on every PR);
`TraceNames`, which takes seconds on MaxText's megablox kernels, is not
marked scheduled_only and runs on every PR.  Those of several minutes each
skip unless MAXTEXT_MPMD_SLOW_TESTS=1, run by hand (testing.slow).  Each
class asks for the devices it needs: two pipeline stages take 4 CPU devices,
as in MaxText's CPU CI (testing.stage_axes: FSDP 1 x EP 2 a stage, FSDP 2 x
EP 2 on 8); the few classes that need 8 say why.
"""

import argparse
import collections
import contextlib
import dataclasses
import functools
import gc
import hashlib
import io
import itertools
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax.deepseek import attention
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_lineage
from maxtext.experimental.mpmd.jax import ds_ep_parity
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import schedule as sched
from maxtext.experimental.mpmd.jax import stage_ad
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer
from tests.unit.experimental.mpmd.jax.schedule_test import assert_legal

TILE = (128,) * 9
ROOT = testing.import_root()  # the children's working directory

try:  # tiny()'s attention kernel, as in the benchmark
  import tokamax  # noqa: F401

  HAVE_TOKAMAX = True
except Exception:  # pragma: no cover
  HAVE_TOKAMAX = False


def setUpModule():
  testing.require_cpu()


def tiny(layers=4, **kw):
  cfg = DeepSeekConfig(
      vocab_size=512,
      emb_dim=256,
      num_decoder_layers=layers,
      first_num_dense_layers=0,
      num_query_heads=2,
      num_kv_heads=2,
      q_lora_rank=128,
      kv_lora_rank=128,
      mlp_dim=256,
      moe_mlp_dim=256,
      num_experts=8,
      num_experts_per_tok=2,
      shared_experts=1,
      routed_bias=False,
      attention_kernel="tokamax_splash",
      sa_block_q=128,
      sa_block_kv=128,
      sa_block_kv_compute=128,
      max_target_length=256,
  )
  base = dict(model=cfg, wi_tiling=TILE, wo_tiling=TILE, rs_dtype=jnp.float32)
  base.update(kw)
  return ds_ep.StageConfig(**base)


def tiny_xla(layers=4, **kw):
  """tiny() on XLA's ops instead of MaxText's kernels: `ragged_dot` experts and
  the argsort dispatch instead of megablox and the SparseCore ragged sort.  fp32
  compute, so the pipeline and the reference agree to rounding.  Attention is
  tokamax splash, as in the benchmark, where tokamax is installed (else the
  reference attention)."""
  scfg = tiny(layers, **{**XLA_KERNELS, **kw})
  return dataclasses.replace(
      scfg,
      model=scfg.model.replace(dtype=jnp.float32, attention_kernel="tokamax_splash" if HAVE_TOKAMAX else "reference"),
  )


# The batch split on ds_ep.split_layers (StageConfig.batch_split_impl 'native', where Lineage's TPU kernels do
# not run): what the tests of that path pin.
NATIVE_SPLIT = dict(batch_split=True, batch_split_impl="native")


# The kernels of tiny_xla: XLA's ragged dot and the argsort dispatch.
XLA_KERNELS = dict(gmm_impl="ragged_dot", use_ragged_sort=False)


@contextlib.contextmanager
def _sys_modules(entries):
  """Set sys.modules entries for the duration and restore exactly those
  after (patch.dict would also drop every module first imported inside).
  ds_ep caches its megablox import, so drop that cache on both edges."""
  saved = {k: sys.modules[k] for k in entries if k in sys.modules}
  ds_ep._megablox.cache_clear()
  sys.modules.update(entries)
  try:
    yield
  finally:
    for k in entries:
      if k in saved:
        sys.modules[k] = saved[k]
      else:
        sys.modules.pop(k, None)
    ds_ep._megablox.cache_clear()


def _maxtext_stand_ins(modules):
  """`modules` ({name: module}, stand-ins down to a MaxText module) for
  _sys_modules, less what is not MaxText's own (testing.is_maxtext): in MaxText,
  maxtext itself, which holds the package."""
  return {k: v for k, v in modules.items() if testing.is_maxtext(k)}


# The monolith reference lives in ds_ep_parity (a library module must not import a test module).
reference = ds_ep_parity.monolith_loss_and_grads


def _rel(a, b):
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def compare_with_reference(t, p, overlap, loss_tol=2e-3, grad_tol=3e-2, sq_tol=5e-2, reference_of=None):
  """One loss-and-gradients pass of `p` against `reference` (of
  `reference_of`, default `p`): the loss, every gradient leaf and the
  gradient sum of squares, each to a relative tolerance."""
  params, _, wc0 = p.init(jax.random.key(1))
  batch = p.synthetic_batch(3)
  if overlap:
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
    red = [(grads[s], sqs[s]) for s in range(p.S)]
  else:
    gaccs, lacc = p.loss_and_grads(params, batch)
    red = [p.stages[s].reduce(gaccs[s]) for s in range(p.S)]
  loss = float(jnp.sum(lacc))
  ref_loss, ref_grads = reference(reference_of or p, params, batch)
  t.assertAlmostEqual(loss, float(ref_loss), delta=loss_tol * abs(float(ref_loss)))
  worst, worst_at = 0.0, ""
  for ch in p.chunks:
    g = red[ch.stage][0][p.slot[ch.index]]
    r = ref_grads[ch.index]
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(r), strict=True):
      t.assertEqual(a.shape, b.shape, path)
      e = _rel(a, b)
      if e > worst:
        worst_at = f"chunk {ch.index} {jax.tree_util.keystr(path)}"
      worst = max(worst, e)
      t.assertLess(e, grad_tol, f"chunk {ch.index} {jax.tree_util.keystr(path)}: rel err {e:.3g}")
  # sumsq matches the reference gradient norm
  ref_sq = sum(float(jnp.sum(jnp.square(x.astype(jnp.float32)))) for x in jax.tree.leaves(ref_grads))
  got_sq = sum(float(r[1]) for r in red)
  t.assertLess(abs(got_sq - ref_sq) / ref_sq, sq_tol)
  print(f"loss {loss:.6f} ref {float(ref_loss):.6f} worst grad rel err {worst:.3g} at {worst_at}")


# ------------------------------------------------ the driver's legacy loops
# The two microbatch loops `Pipeline` ran before its one step loop (`_run`,
# which dispatches every op through `progs[c].program(desc)`), kept verbatim
# as the reference test_unified_loop_bitwise_vs_legacy holds that loop to.
# They call the FIXED programs (fwd / bwd / bwda / fb / fba and the hand-built
# fwd_prefetch, fwd_self_prefetch, bwd_reduce, bwda_reduce_self) on the
# pipeline's own plan, and run circular placements only (no same-stage hop).
def legacy_loss_and_grads(p, params, batch, wc=None):
  toks, tgts = batch
  S, C = p.S, p.C
  if wc is None:
    wc = [p.stages[s].gather(params[s]) for s in range(S)]
  act, res, cotangents = {}, {}, {}
  gacc = [None] * C
  lacc = None
  place, slot, progs, chans, seeds = p.place, p.slot, p.progs, p.channels, p.seeds
  for c, kind, m in p.order:
    s = place[c]
    w = wc[s][slot[c]]
    pr = progs[c]
    if kind == "F":
      x = toks[m] if c == 0 else act.pop((c, m))
      if c == C - 1:
        if gacc[c] is None:
          out = pr.fb(w, x, tgts[m], seeds[s][m])
        else:
          out = pr.fba(w, x, tgts[m], gacc[c], lacc, seeds[s][m])
        if c == 0:
          gacc[c], lacc = out
        else:
          dx, gacc[c], lacc = out
          cotangents[(c - 1, m)] = chans[(s, place[c - 1])].send_flat([dx])[0]
      else:
        y, r = pr.fwd(w, x, seeds[s][m])
        res[(c, m)] = (x, r)
        act[(c + 1, m)] = chans[(s, place[c + 1])].send_flat([y])[0]
    else:
      if c == C - 1:
        continue
      x, r = res.pop((c, m))
      dy = cotangents.pop((c, m))
      if gacc[c] is None:
        out = pr.bwd(w, x, r, dy, seeds[s][m])
      else:
        out = pr.bwda(w, x, r, dy, gacc[c], seeds[s][m])
      if c == 0:
        (gacc[c],) = out
      else:
        dx, gacc[c] = out
        cotangents[(c - 1, m)] = chans[(s, place[c - 1])].send_flat([dx])[0]
  gaccs = [[gacc[ch.index] for ch in p.stage_chunks[s]] for s in range(S)]
  return gaccs, lacc


def legacy_loss_and_grads_overlap(p, params, wc0, batch):
  # the plan's fixed programs, built as the driver used to build them
  first_b = {}
  for j, (c, k, m) in enumerate(p.order):
    if k == "B":
      first_b.setdefault(c, j)
  pf = {
      j: (b, g, p.progs[p.order[j][0]].fwd_prefetch(p.chunks[b], g if p.pf_n[j] > 1 else None))
      for j, (b, g) in p.pf.items()
  }
  rs_at = {
      j: (
          o,
          g,
          p.progs[p.order[j][0]].bwd_reduce(
              p.chunks[o], accumulate=j != first_b[p.order[j][0]], leaves=g if p.rs_groups[o] > 1 else None
          ),
      )
      for j, (o, g) in p.rs_at.items()
  }
  rs_self = {j: p.progs[p.order[j][0]].bwda_reduce_self(n) for j, n in p.rs_self.items()}
  self_pf = p.progs[0].fwd_self_prefetch(p.routed_leaves) if p.self_pf_at is not None else None
  toks, tgts = batch
  S, C = p.S, p.C
  wc = [[None] * len(p.stage_chunks[s]) for s in range(S)]
  for s in range(S):
    for i, w in zip(p.gather_slots[s], wc0[s]):
      wc[s][i] = w
  self_pf_at = p.self_pf_at
  wparts = {}  # chunk -> [compute leaves], filled by prefetch groups
  pending = {b: p.pf_n[j] for j, (b, _) in p.pf.items()}  # the step scope's: a chunk's one gather a step
  gparts = {}  # chunk -> [grad leaves], filled by reduce groups
  gpending = dict(p.rs_groups)
  grads = [[None] * len(p.stage_chunks[s]) for s in range(S)]
  sq = [[] for s in range(S)]
  act, res, cotangents = {}, {}, {}
  gacc = [None] * C
  gacc_leaves = {}
  lacc = None
  place, slot, progs, chans, seeds = p.place, p.slot, p.progs, p.channels, p.seeds
  tdef = p.treedef
  stage_tot = [None] * S
  for j, (c, kind, m) in enumerate(p.order):
    s = place[c]
    w = wc[s][slot[c]]
    pr = progs[c]
    if kind == "F":
      x = toks[m] if c == 0 else act.pop((c, m))
      if c == C - 1:
        if gacc[c] is None:
          out = pr.fb(w, x, tgts[m], seeds[s][m])
        else:
          out = pr.fba(w, x, tgts[m], gacc[c], lacc, seeds[s][m])
        if c == 0:
          gacc[c], lacc = out
        else:
          dx, gacc[c], lacc = out
          cotangents[(c - 1, m)] = chans[(s, place[c - 1])].send_flat([dx])[0]
      else:
        if j == self_pf_at:
          st = jax.tree.leaves(params[s][slot[c]])
          y, r, got = self_pf(w, x, seeds[s][m], [st[i] for i in p.routed_leaves])
          full = [None] * p.n_leaves[c]
          for i, v in zip(p.first_leaves[s], w):
            full[i] = v
          for i, v in zip(p.routed_leaves, got):
            full[i] = v
          wc[s][slot[c]] = jax.tree.unflatten(tdef[c], full)
        elif j in pf:
          b, g, prog = pf[j]
          st = jax.tree.leaves(params[s][slot[b]])
          y, r, got = prog(w, x, seeds[s][m], [st[i] for i in g])
          parts = wparts.setdefault(b, [None] * p.n_leaves[b])
          for i, v in zip(g, got):
            parts[i] = v
          pending[b] -= 1
          if not pending[b]:
            wc[s][slot[b]] = jax.tree.unflatten(tdef[b], wparts.pop(b))
        else:
          y, r = pr.fwd(w, x, seeds[s][m])
        res[(c, m)] = (x, r)
        act[(c + 1, m)] = chans[(s, place[c + 1])].send_flat([y])[0]
    else:
      if c == C - 1:
        continue
      x, r = res.pop((c, m))
      dy = cotangents.pop((c, m))
      if j in rs_self:
        *out, grads[s][slot[c]], stage_tot[s] = rs_self[j](w, x, r, dy, gacc[c], seeds[s][m], sq[s])
        gacc[c] = None
        if c != 0:
          cotangents[(c - 1, m)] = chans[(s, place[c - 1])].send_flat([out[0]])[0]
        continue
      if j in rs_at:
        o, g, prog = rs_at[j]
        if o not in gacc_leaves:
          gacc_leaves[o] = jax.tree.leaves(gacc[o])
          gacc[o] = None
        gl = gacc_leaves[o]
        go = [gl[i] for i in g]
        for i in g:
          gl[i] = None
        if gacc[c] is None:
          out = prog(w, x, r, dy, seeds[s][m], go)
        else:
          out = prog(w, x, r, dy, gacc[c], seeds[s][m], go)
        *out, got, part_sq = out
        sq[s].append(part_sq)
        parts = gparts.setdefault(o, [None] * p.n_leaves[o])
        for i, v in zip(g, got):
          parts[i] = v
        gpending[o] -= 1
        if not gpending[o]:
          grads[s][slot[o]] = jax.tree.unflatten(tdef[o], gparts.pop(o))
          del gacc_leaves[o]
      elif gacc[c] is None:
        out = pr.bwd(w, x, r, dy, seeds[s][m])
      else:
        out = pr.bwda(w, x, r, dy, gacc[c], seeds[s][m])
      if c == 0:
        (gacc[c],) = out
      else:
        dx, gacc[c] = out
        cotangents[(c - 1, m)] = chans[(s, place[c - 1])].send_flat([dx])[0]
  stage_sq = []
  for s in range(S):
    fs = p.final_slots[s]
    if not fs:
      stage_sq.append(stage_tot[s])
      continue
    g, tot = p.stages[s].reduce_final(fs, p.n_partial_sq[s])([gacc[p.stage_chunks[s][i].index] for i in fs], sq[s])
    for i, gi in zip(fs, g):
      grads[s][i] = gi
    stage_sq.append(tot)
  return grads, stage_sq, lacc


def legacy_program(p, op):
  """The fixed program the legacy loops call for `op` (a remat op of p's plan)."""
  pr = p.progs[op.c]
  if op.c == p.C - 1:
    return pr.fba if op.acc else pr.fb
  if op.kind == "F":
    if op.self_pf:
      return pr.fwd_self_prefetch(p.routed_leaves)
    if op.pf:
      b, g = op.pf
      return pr.fwd_prefetch(p.chunks[b], g if p.pf_n[op.j] > 1 else None)
    return pr.fwd
  if op.rs_self is not None:
    return pr.bwda_reduce_self(op.rs_self)
  if op.rs:
    o, g = op.rs
    return pr.bwd_reduce(p.chunks[o], accumulate=op.acc, leaves=g if p.rs_groups[o] > 1 else None)
  return pr.bwda if op.acc else pr.bwd


def legacy_rs_plan(p):
  """The first planner's reduce-scatter plan on p's order (the driver's `_plan_overlap`
  before the one step loop, verbatim but for the programs it built) ->
  (rs_at {order index: (chunk, leaf ids)}, rs_self order indices,
  final_slots, n_partial_sq): what circular plans must still be."""
  C, order, place = p.C, p.order, p.place
  shp = lambda c: ds_ep.chunk_shapes(p.scfg, p.chunks[c])
  is_bwd = lambda c, k: (k == "B" and c != C - 1) or (k == "F" and c == C - 1)
  last, first_b = {}, {}
  for i, (c, k, m) in enumerate(order):
    if is_bwd(c, k):
      last[c] = i
      first_b.setdefault(c, i)
  rs_at, taken, rs_groups = {}, set(), {}
  final = [[] for _ in range(p.S)]
  for c in sorted(last, key=lambda c: last[c]):
    s = place[c]
    hosts = [
        j
        for j in range(last[c] + 1, len(order))
        if order[j][1] == "B" and order[j][0] != C - 1 and place[order[j][0]] == s and j not in taken
    ][: p.rs_split]
    if not hosts:
      final[s].append(p.slot[c])
      continue
    groups = ds_ep.leaf_groups(shp(c), len(hosts))
    rs_groups[c] = len(groups)
    for j, g in zip(hosts, groups):
      taken.add(j)
      rs_at[j] = (c, g)
  final_slots = [tuple(sorted(f)) for f in final]
  n_partial_sq = [sum(rs_groups.get(ch.index, 0) for ch in p.stage_chunks[s]) for s in range(p.S)]
  rs_self = []
  for s in range(p.S):
    f = final_slots[s]
    if p.rs_self_enabled and len(f) == 1:
      c = p.stage_chunks[s][f[0]].index
      j = last[c]
      if c != C - 1 and order[j][1] == "B" and j != first_b[c] and j not in rs_at:
        rs_self.append(j)
        final_slots[s] = ()
  return rs_at, sorted(rs_self), final_slots, n_partial_sq


def assert_bitwise(t, got, want, what):
  """Every leaf of two trees array_equal (and as many leaves)."""
  gl, wl = jax.tree.leaves(got), jax.tree.leaves(want)
  t.assertEqual(len(gl), len(wl), what)
  for i, (a, b) in enumerate(zip(gl, wl)):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{what} leaf {i}")


def pipe(scfg, **kw):
  """A two-stage pipeline of `scfg` at seq 256 (nothing is compiled)."""
  k = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256)
  k.update(kw)
  return dsp.Pipeline(scfg, **k)


def stage_rows(gbs):
  """`gbs`, a global batch written for two stages of fsdp 2 x ep 2 (8 devices), for the two stages of the devices
  there are (testing.stage_axes): the same rows a device either way -- on 4 devices, half of it.  The bf16 checks on
  MaxText's kernels hold at fsdp 1 x ep 2 with as many rows a device as at fsdp 2 x ep 2 (worst gradient leaf 2.3e-2
  to 2.9e-2 against their 3e-2, jax 0.12.0.1 on CPU, the YNNPACK fusions on); at twice as many rows the YNNPACK
  fusions' bf16 grouped-matmul gradients come out wrong (testing.YNN_BUG_JAXLIBS_BEFORE), the router's off by 100%."""
  axes = testing.stage_axes()
  return gbs * axes["fsdp"] * axes["ep"] // 4


def lite_gib(scfg, **kw):
  """GiB per device of one 'lite' stash set of chunk 0 (every one-layer
  chunk's is the same) in the pipeline `pipe(scfg, **kw)` builds."""
  return pipe(scfg, **kw).progs[0].build_stash("lite").bytes / 2**30


def run_pipe(p, key, seed, overlap=True):
  """One microbatch loop of `p` from init(key) on synthetic_batch(seed): with
  the fusion plan (grads, stage sumsqs, loss partials), else (gaccs, loss
  partials)."""
  params, _, wc0 = p.init(jax.random.key(key))
  batch = p.synthetic_batch(seed)
  return p.loss_and_grads_overlap(params, wc0, batch) if overlap else p.loss_and_grads(params, batch)


def stage_seqs(p):
  """Each stage's own op sequence: what its devices run, in order (the fused
  last chunk's no-op backward left out: it sits after a pair holding its F)."""
  return [[a for a in p.order if p.place[a[0]] == s and a[:2] != (p.C - 1, "B")] for s in range(p.S)]


def n_pairs(streams):
  return sum(isinstance(it[0], tuple) for st in streams for it in st)


# tiny()'s model on the command line on XLA's ops (multihost_test.TINY's
# kernels: ragged_dot experts, the argsort dispatch, the reference attention):
# two one-layer chunks on two stages of fsdp 1 x ep 2.
TINY_ARGS = (
    "--emb 256 --heads 2 --q-lora 128 --kv-lora 128 --moe-mlp 256 --experts 8 --topk 2 --vocab 512 "
    "--seq 256 --sa-block 128 --sa-block-kv-compute 128 --gmm-impl ragged_dot --ragged-sort off "
    "--attn reference --routing topk --stages 2 --vv 1 --layers 2 --fsdp 1 --ep 2 --gbs 4 --mb 2"
).split()


def host_weights(p, seed=0):
  """Random weights for every chunk of `p`, in chunk order: host arrays of
  the storage shapes and weight dtype, which `params_from_chunks` places
  without compiling anything (`init`'s programs take seconds on CPU)."""
  rng = np.random.default_rng(seed)
  dt = jnp.dtype(p.scfg.model.weight_dtype)
  return [
      jax.tree.map(lambda a: (0.05 * rng.standard_normal(a.shape)).astype(dt), p._avals(ch.stage, "storage", ch))
      for ch in p.chunks
  ]


def explicit(scfg):
  """`scfg` with typed stage programs (StageConfig.shard_mode='explicit')."""
  return dataclasses.replace(scfg, shard_mode="explicit")


def mode_run(p, key, seed, steps=2, lr=1e-3):
  """What the shard mode must not change, from init(key) on
  synthetic_batch(seed): one microbatch loop's reduced gradients (folded into
  the plan's programs with overlap, else by the stages' reduce programs) by
  tree path, the stage sumsqs and the loss (and its partials); then the
  params, losses and gradient norms of `steps` training steps from the same
  start.  Host values only: an unreduced accumulator never reaches the host
  (jax refuses)."""
  params, opts, wc0 = p.init(jax.random.key(key))
  batch = p.synthetic_batch(seed)
  if p.overlap:
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
  else:
    gaccs, lacc = p.loss_and_grads(params, batch)
    red = [p.stages[s].reduce(gaccs[s]) for s in range(p.S)]
    grads, sqs = [r[0] for r in red], [r[1] for r in red]
  flat = jax.tree_util.tree_flatten_with_path(grads)[0]
  out = dict(
      grads=[(jax.tree_util.keystr(k), np.asarray(v)) for k, v in flat],
      sumsq=[float(x) for x in sqs],
      loss=float(jnp.sum(lacc)),
      loss_partials=np.asarray(lacc),
      losses=[],
      gnorms=[],
  )
  lrs = p.lr_arrays(lr)
  for _ in range(steps):
    params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
    out["losses"].append(float(jnp.sum(lacc)))
    out["gnorms"].append(float(gn))
  out["params"] = [np.asarray(x) for x in jax.tree.leaves(params)]
  return out


def assert_same_run(t, got, want, what, near=(), regrouped=False):
  """Two `mode_run`s bit for bit: every gradient leaf, the stage sumsqs, the
  loss, and every step's loss, gradient norm and resulting params.  A gradient
  leaf whose path holds one of `near` may differ by up to 4 float32 ulps of its
  largest magnitude; where one does, what is computed from it -- the sumsqs,
  and through the clip every update -- is held to a relative 1e-6.  So are
  they under `regrouped`, even where every gradient leaf agrees: two plans of
  other reduce-scatter groups, where a stage's sumsq sums other partial sums.
  -> the paths of the leaves that differed."""
  off = []
  for (path, a), (path_b, b) in zip(got["grads"], want["grads"], strict=True):
    t.assertEqual(path, path_b, what)
    if np.array_equal(a, b):
      continue
    a, b = a.astype(np.float64), b.astype(np.float64)
    err = float(np.max(np.abs(a - b)))
    t.assertTrue(any(n in path for n in near), f"{what}: gradient {path} differs by {err:.3g}")
    t.assertLessEqual(err, 2.0**-22 * float(np.max(np.abs(b))), f"{what}: gradient {path}")
    off.append(path)
  t.assertEqual(got["loss"], want["loss"], what)
  if not off and not regrouped:
    t.assertEqual((got["sumsq"], got["losses"], got["gnorms"]), (want["sumsq"], want["losses"], want["gnorms"]), what)
    for i, (a, b) in enumerate(zip(got["params"], want["params"], strict=True)):
      np.testing.assert_array_equal(a, b, err_msg=f"{what}: params leaf {i}")
    return off
  for k in ("sumsq", "losses", "gnorms"):
    np.testing.assert_allclose(got[k], want[k], rtol=1e-6, err_msg=f"{what}: {k}")
  for i, (a, b) in enumerate(zip(got["params"], want["params"], strict=True)):
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, err_msg=f"{what}: params leaf {i}")
  return off


def spec_of(x):
  """The PartitionSpec of an array's or a ShapeDtypeStruct's type (reduced /
  unreduced axes included)."""
  return (x.sharding if isinstance(x, jax.ShapeDtypeStruct) else jax.typeof(x).sharding).spec


@contextlib.contextmanager
def compiles():
  """-> the names of the programs XLA compiles inside (jax's own monitoring
  event, which every jit and `Lowered.compile` records)."""
  names = []

  def on(event, secs, **kw):
    if event == "/jax/core/compile/backend_compile_duration":
      names.append(kw.get("fun_name"))

  jax.monitoring.register_event_duration_secs_listener(on)
  try:
    yield names
  finally:
    jax.monitoring.unregister_event_duration_listener(on)


# PipelineVsReference._port's bounds are for bf16 matmuls accumulated in fp32 (its comment), which XLA:CPU's YNNPACK
# dots are not from jaxlib 0.11.2 on: there the port checks run with those fusions off.
PORT_NEEDS_YNN_OFF = testing.needs_ynn_off(
    "the port check's bounds are for bf16 matmuls accumulated in fp32, which a YNNPACK dot is not (cosine 0.99954-"
    "0.99959 against its 0.9999 with them on: PipelineVsReference._port)"
)
# Two of PipelineVsReference's plans make the monolith's backward NaN under jaxlib 0.12.0.1's YNNPACK fusions: every
# chunk-0 gradient of `reference` (60 leaves in test_overlap_default_plan_paths), where the pipeline's are finite and
# the losses agree.  Compiling the reference alone without them (testing.ynn_off) would hold the pipeline's bf16
# YNNPACK dots to fp32-accumulated ones, which the port checks show part by more than these bounds.  jaxlib 0.12.1
# fails both with the fusions on as well (test_pp4_fsdp2_microbatch_grad_reduce by a chunk-0 gradient 45% off rather
# than NaN; 2 of 2 runs each), and test_overlap_default_plan_paths with them off too (a chunk-0 gradient NaN, 4 of 4
# runs; 2 of 2 passed on 0.12.0.1): a bug of that jaxlib's own, which this gate leaves alone (2026-11-26).
REFERENCE_NEEDS_YNN_OFF = testing.needs_ynn_off(
    "on jaxlib 0.12.0.1 and 0.12.1 these checks fail with them on: on 0.12.0.1 the monolith reference's chunk-0 "
    "gradients come out NaN, the pipeline's finite (test_overlap_default_plan_paths 2 of 2 runs, "
    "test_pp4_fsdp2_microbatch_grad_reduce 3 of 3); on 0.12.1 a chunk-0 gradient is NaN "
    "(test_overlap_default_plan_paths, 2 of 2) or 45% off (test_pp4_fsdp2_microbatch_grad_reduce, 2 of 2); on "
    "0.12.0.1 with them off, and on jaxlib 0.11.1 either way, they pass",
    jaxlib_before=testing.YNN_BUG_JAXLIBS_BEFORE,
)


@pytest.mark.scheduled_only
@testing.needs_devices(4)
class PipelineVsReference(unittest.TestCase):

  def _check(self, scfg, vv, mb, gbs, schedule, overlap=False, plan_check=None, **kw):
    p = dsp.Pipeline(
        scfg,
        n_stages=2,
        chunks_per_stage=vv,
        n_micro=mb,
        global_batch=gbs,
        seq=256,
        schedule=schedule,
        overlap=overlap,
        **kw,
    )
    if plan_check:
      plan_check(p)
    compare_with_reference(self, p, overlap)

  def test_one_layer_chunks_circular(self):
    self._check(tiny(layers=4, **testing.stage_axes()), vv=2, mb=2, gbs=stage_rows(8), schedule="search")

  def test_two_layer_chunks_1f1b(self):
    self._check(tiny(layers=4, **testing.stage_axes()), vv=1, mb=2, gbs=stage_rows(8), schedule="1f1b")

  def test_overlap_one_layer_chunks(self):
    self._check(tiny(layers=4, **testing.stage_axes()), vv=2, mb=2, gbs=stage_rows(8), schedule="search", overlap=True)

  def test_overlap_two_layer_chunks_1f1b(self):
    self._check(tiny(layers=4, **testing.stage_axes()), vv=1, mb=2, gbs=stage_rows(8), schedule="1f1b", overlap=True)

  def test_overlap_v_default(self):
    """V placement's default -- breadth, repeat -- against the reference."""

    def plan_check(p):
      self.assertEqual((p.placement, p.schedule_name, p.fsdp_scope), ("v", "breadth", "repeat"))

    self._check(
        tiny(layers=4, **testing.stage_axes()),
        vv=2,
        mb=2,
        gbs=stage_rows(8),
        schedule=None,
        overlap=True,
        placement="v",
        plan_check=plan_check,
    )

  @testing.slow
  def test_overlap_breadth(self):
    """The circular default -- no schedule, no scope -- against the
    reference: MaxText's chunk-by-chunk order with weights per visit, each
    visit's gather riding in the previous visit's forwards, backwards or
    fused last chunk."""

    def plan_check(p):
      self.assertEqual((p.schedule_name, p.fsdp_scope), ("breadth", "repeat"))
      self.assertEqual([c for c, k, m in p.order if k == "F" and p.place[c] == 0], [0] * 4 + [2] * 4)
      self.assertEqual({"A" if p.order[j][0] == p.C - 1 else p.order[j][1] for j in p.pf}, {"F", "B", "A"})

    self._check(
        tiny(layers=4, **testing.stage_axes()),
        vv=2,
        mb=4,
        gbs=stage_rows(16),
        schedule=None,
        overlap=True,
        plan_check=plan_check,
    )

  @REFERENCE_NEEDS_YNN_OFF
  def test_overlap_default_plan_paths(self):
    """The fusion paths the full-size default runs: a chunk's gather split over
    several forwards of another chunk, first-chunk self-prefetch, and a
    self reduce-scatter on a chunk that returns dx and sums partial sumsqs."""

    def plan_check(p):
      self.assertGreater(max(p.pf_n.values()), 1)
      self.assertIsNotNone(p.self_pf_at)
      self.assertTrue(any(p.order[j][0] != 0 for j in p.rs_self))
      self.assertTrue(any(p.n_partial_sq))

    self._check(
        tiny(layers=4, **testing.stage_axes()),
        vv=2,
        mb=6,
        gbs=stage_rows(24),
        schedule="search",
        overlap=True,
        plan_check=plan_check,
    )

  @testing.slow
  def test_overlap_ppermute_ep_collectives(self):
    """EP all-gather / reduce-scatter as collective-permutes (TPU default on a
    sub-mesh without SparseCore offload) against the lax-collective reference."""
    self._check(
        tiny(layers=4, ep_collectives="ppermute", **testing.stage_axes()),
        vv=2,
        mb=4,
        gbs=stage_rows(16),
        schedule="search",
        overlap=True,
    )

  @PORT_NEEDS_YNN_OFF
  def test_matches_independent_port(self):
    """Loss and every gradient against `maxtext.experimental.mpmd.jax.deepseek`, the MaxText-parity
    port, which shares none of the pipeline's hot path (ragged_dot experts, no
    ring of experts / ragged sort / megablox / tokamax, no shard_map, one
    device).  Every token takes all experts: top-k with k < E is discontinuous
    and flips on last-bit differences -- see ds_ep_parity's docstring."""
    self._port()

  @PORT_NEEDS_YNN_OFF
  def test_matches_independent_port_stash(self):
    """The same against the port with the stash schedule: some (chunk,
    microbatch)s run the 'lite' stash programs, some of them hosting the
    folded FSDP collectives."""

    def plan_check(p):
      self.assertTrue(p.modes)
      self.assertTrue(any(op.stash and (op.pf or op.rs or op.self_pf or op.rs_self is not None) for op in p.ops()))

    self._port(plan_check, schedule="stash", stash="lite", stash_budget_gib=1.0)

  @PORT_NEEDS_YNN_OFF
  def test_matches_independent_port_batch_split(self):
    """The same against the port with batch split (each device's two rows of
    a microbatch as halves in Lineage's phases): every token takes all
    experts, so the halves change no routing."""
    self._port(scfg_kw=NATIVE_SPLIT, global_batch=stage_rows(16))

  @PORT_NEEDS_YNN_OFF
  def test_matches_independent_port_dense(self):
    """The same with DeepSeek-V3's layer structure: dense layers (an MLP of
    width mlp_dim, the port's own dense branch) before the MoE layers, all in
    chunk 0, unsplit and under batch split."""
    for kw in ({}, dict(scfg_kw=NATIVE_SPLIT, global_batch=stage_rows(16))):
      with self.subTest(**{k: str(v) for k, v in kw.items()}):
        self._port(model_kw=dict(num_decoder_layers=6, first_num_dense_layers=2, mlp_dim=384), **kw)

  def _port(self, plan_check=None, scfg_kw=None, model_kw=None, **kw):
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    scfg = tiny(layers=4, random_routing=False, **testing.stage_axes(), **(scfg_kw or {}))
    scfg = dataclasses.replace(scfg, model=dataclasses.replace(scfg.model, num_experts_per_tok=8, **(model_kw or {})))
    kw = dict(dict(n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=stage_rows(8), seq=256), **kw)
    p = dsp.Pipeline(scfg, **kw)
    if plan_check:
      plan_check(p)
    params, _, wc0 = p.init(jax.random.key(0))
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
    targets = np.roll(tokens, -1, axis=1)
    loss, grads = par.mpmd_loss_and_grads(p, params, wc0, p.put_batch(tokens, targets))
    ref_loss, ref = par.port_loss_and_grads(p, [[par.to_host(t) for t in ps] for ps in params], tokens, targets)
    self.assertLess(abs(loss - ref_loss) / ref_loss, 1e-4)
    rows, cos, ratio = par.errors(p, grads, ref)
    # Bounds for bf16 matmuls accumulated in fp32, as jax 0.11.1's CPU backend computes every one.  With
    # its YNNPACK dot fusions on (the default) 0.11.2's no longer upcasts them, and a weight gradient's,
    # whose lhs contracts its leading axis (YNNPACK does not take it), runs as a bf16 Eigen contraction,
    # ~2e-2 off at K = 2048 (1.7e-3 in fp32): logits_dense's gradient, the pipeline's and the port's
    # alike, lands ~12% from the fp32 one, and over 8 seeds of this check and 6 of a DualPipeV one the
    # cosine falls to 0.99954 and other leaves to 0.048, varying with the thread count; logits_dense to
    # 0.087.  With those fusions off (--xla_cpu_experimental_ynn_fusion_type= in XLA_FLAGS) every bf16 matmul
    # accumulates in fp32 again and this check is back at 0.11.1's level, the same on
    # 1, 2 and 224 cores: cosine >= 0.99996 and worst leaf <= 0.0204 over seeds 0-7 of this check and 0-5 of
    # a DualPipeV one.  So on jaxlib >= 0.11.2 the port checks skip where the fusions are on
    # (PORT_NEEDS_YNN_OFF; only here: qwen35_maxtext_parity_test's fp32 checks need the fusions on).
    self.assertGreater(cos, 0.9999)
    self.assertLess(abs(ratio - 1), 1e-2)
    self.assertLess(rows[0][0], 3e-2, rows[:3])
    print(f"port: loss {loss:.6f} ref {ref_loss:.6f} cosine {cos:.7f} worst leaf {rows[0][0]:.3g} at {rows[0][1]}")

  @REFERENCE_NEEDS_YNN_OFF
  @testing.needs_devices(8, "PP=4 x FSDP=2")
  def test_pp4_fsdp2_microbatch_grad_reduce(self):
    """PP=4 x FSDP=2 x EP=1 with the weight gradient reduce-scattered after
    EVERY microbatch (MaxText's pipeline_fsdp_ag_per_repeat cadence), through
    the argsort dispatch MaxText uses at EP=1 (no ragged sort)."""
    scfg = tiny(layers=4, fsdp=2, ep=1, grad_reduce="microbatch", use_ragged_sort=False)
    p = dsp.Pipeline(
        scfg, n_stages=4, chunks_per_stage=1, n_micro=4, global_batch=16, seq=256, schedule="search", layout="snake"
    )
    self.assertEqual([m.devices.shape for m in p.meshes], [(2, 1)] * 4)
    self.assertEqual(p.rs_at, {})
    params, _, wc0 = p.init(jax.random.key(1))
    batch = p.synthetic_batch(3)
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
    loss = float(jnp.sum(lacc))
    ref_loss, ref_grads = reference(p, params, batch)
    self.assertAlmostEqual(loss, float(ref_loss), delta=2e-3 * abs(float(ref_loss)))
    worst = 0.0
    for ch in p.chunks:
      g = grads[ch.stage][p.slot[ch.index]]
      for (path, a), b in zip(
          jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(ref_grads[ch.index]), strict=True
      ):
        self.assertEqual(a.shape, b.shape, path)
        e = _rel(a, b)
        worst = max(worst, e)
        self.assertLess(e, 3e-2, f"chunk {ch.index} {jax.tree_util.keystr(path)}: rel err {e:.3g}")
    params, opts, wc0 = p.init(jax.random.key(1))
    lrs = p.lr_arrays(1e-3)
    ls = []
    for _ in range(3):
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      ls.append(float(jnp.sum(lacc)))
      self.assertTrue(np.isfinite(ls[-1]) and np.isfinite(float(gn)))
    self.assertLess(ls[-1], ls[0])
    print(f"pp4 fsdp2 microbatch-RS: loss {loss:.6f} ref {float(ref_loss):.6f} worst grad rel err {worst:.3g}")

  def test_bf16_master_weights(self):
    """weight_dtype=bf16 (the benchmark's): params and AdamW nu stay bf16
    across steps -- the update runs in fp32 and stores back -- and the loss
    goes down."""
    base = tiny(layers=4, **testing.stage_axes())
    scfg = dataclasses.replace(base, model=base.model.replace(weight_dtype=jnp.bfloat16))
    p = dsp.Pipeline(
        scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=stage_rows(8), seq=256, schedule="search"
    )
    params, opts, wc0 = p.init(jax.random.key(1))
    batch = p.synthetic_batch(3)
    lrs = p.lr_arrays(1e-3)
    ls = []
    for _ in range(3):
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      ls.append(float(jnp.sum(lacc)))
      self.assertTrue(np.isfinite(ls[-1]) and np.isfinite(float(gn)))
    bf16 = {jnp.dtype(jnp.bfloat16)}
    self.assertEqual({x.dtype for x in jax.tree.leaves(params)}, bf16)
    self.assertEqual({x.dtype for x in jax.tree.leaves([o["nu"] for o in opts])}, bf16)
    self.assertLess(ls[-1], ls[0])

  @testing.slow
  def test_overlap_timed_interleaved(self):
    self._check(tiny(layers=4, **testing.stage_axes()), vv=2, mb=4, gbs=stage_rows(16), schedule="timed", overlap=True)

  @testing.slow
  def test_step_runs_and_overlap_is_exact(self):
    """Fusing the FSDP collectives into other programs changes no gradient bit.

    (The clipping norm is summed in a different order -- per chunk, then per
    stage -- so post-update parameters agree to rounding, not bitwise.)
    """
    scfg = tiny(layers=4, **testing.stage_axes())
    grads, losses = [], []
    for overlap in (False, True):
      p = dsp.Pipeline(
          scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=stage_rows(8), seq=256, overlap=overlap
      )
      params, opts, wc0 = p.init(jax.random.key(0))
      batch = p.synthetic_batch()
      if overlap:
        g, _, _ = p.loss_and_grads_overlap(params, wc0, batch)
      else:
        ga, _ = p.loss_and_grads(params, batch)
        g = [p.stages[s].reduce(ga[s])[0] for s in range(p.S)]
      grads.append([np.asarray(x) for x in jax.tree.leaves(g)])
      lrs = p.lr_arrays(1e-3)
      ls = []
      for _ in range(3):
        params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
        ls.append(float(jnp.sum(lacc)))
        self.assertTrue(np.isfinite(ls[-1]) and np.isfinite(float(gn)))
      self.assertLess(ls[-1], ls[0])
      losses.append(ls)
    for a, b in zip(*grads):
      np.testing.assert_array_equal(a, b)
    self.assertEqual(losses[0][0], losses[1][0])
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-4)

  @testing.slow
  def test_unified_loop_bitwise_vs_legacy(self):
    """The one step loop (every op through `progs[c].program(desc)`, the
    composed programs) against the two loops it replaced (the fixed programs),
    on the same plan, weights and batch: the loss, every gradient and the
    stage sumsqs bit for bit -- circular search and timed orders, overlap off
    and on."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    for schedule in ("search", "timed"):
      for overlap in (False, True):
        p = dsp.Pipeline(
            scfg, n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, schedule=schedule, overlap=overlap
        )
        params, _, wc0 = p.init(jax.random.key(2))
        batch = p.synthetic_batch(4)
        what = f"{schedule} overlap={overlap}"
        if overlap:
          got, want = (p.loss_and_grads_overlap(params, wc0, batch), legacy_loss_and_grads_overlap(p, params, wc0, batch))
        else:
          got, want = p.loss_and_grads(params, batch), legacy_loss_and_grads(p, params, batch)
        assert_bitwise(self, got, want, what)
        print(f"{what}: {len(jax.tree.leaves(got))} leaves bitwise, loss {float(jnp.sum(got[-1])):.6f}")

  def test_fixed_programs_lower_as_before(self):
    """Every program the first planner's plans dispatch -- each distinct descriptor of a
    circular search and timed order with the fusion plan (forwards with a
    gather or their own routed experts, backwards with a reduce-scatter or
    their own (search at M=6 has rs_self), the fused last chunk) -- lowers to its fixed program's
    StableHLO text, byte for byte, under the same name: nothing to recompile
    or reschedule differently on the TPU.  (B + R is bwd_reduce itself; the
    composed parts would trace the reduce-scatter after the accumulate.)"""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    kinds = collections.Counter()
    for schedule, M in (("search", 6), ("timed", 4)):
      p = pipe(scfg, schedule=schedule, n_micro=M, global_batch=2 * M)
      seen = set()
      for op in p.ops():
        if op.desc is None or (op.c, op.desc) in seen:
          continue
        seen.add((op.c, op.desc))
        av = p._desc_avals(op.c, op.desc)
        got, want = p.program(op), legacy_program(p, op)
        self.assertEqual(got.__name__, want.__name__, op)
        self.assertTrue(got.lower(*av).as_text() == want.lower(*av).as_text(), (schedule, op))
        kinds[
            "".join(k for k, _ in op.desc)
            + ("s" if op.self_pf else "")
            + ("+" if op.rs_self is not None else "")
            + ("a" if op.acc else "")
        ] += 1
    self.assertTrue({"Fs", "FG", "F", "A", "Aa", "B", "Ba", "BR", "BRa", "B+a"} <= set(kinds), kinds)
    print(f"{sum(kinds.values())} programs lower as the fixed ones: {dict(kinds)}")

  @testing.slow
  def test_v_matches_reference(self):
    """V placement (stage s holds chunks s and 3-s): the head shares stage 0
    with the embedding, the turn 1 -> 2 is a same-stage hop, and stage 0's
    last backward folds its own reduce-scatter (rs_self) -- against the
    monolith, overlap off and on."""

    def plan_check(p):
      self.assertEqual(p.place, [0, 1, 1, 0])
      if p.overlap:
        self.assertIn(p.last_writer[0], p.rs_self)
        self.assertEqual(p.final_slots[0], ())

    for overlap in (False, True):
      self._check(
          tiny(layers=4, **testing.stage_axes()),
          vv=2,
          mb=4,
          gbs=stage_rows(16),
          schedule="search",
          overlap=overlap,
          plan_check=plan_check,
          placement="v",
      )

  @testing.slow
  def test_v_equals_circular_bitwise(self):
    """With init_by_chunk the two placements start from the same weights, and
    run each chunk's same programs with FIFO accumulation: the loss and every
    gradient (in chunk order) are bit for bit the same."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    out = []
    for placement in ("circular", "v"):
      p = dsp.Pipeline(
          scfg,
          n_stages=2,
          chunks_per_stage=2,
          n_micro=4,
          global_batch=8,
          seq=256,
          placement=placement,
          init_by_chunk=True,
      )
      params, _, wc0 = p.init(jax.random.key(5))
      grads, _, lacc = p.loss_and_grads_overlap(params, wc0, p.synthetic_batch(6))
      out.append((p.chunk_trees(params), p.chunk_trees(grads), float(jnp.sum(lacc))))
    assert_bitwise(self, out[1][0], out[0][0], "weights by chunk")
    assert_bitwise(self, out[1][1], out[0][1], "gradients by chunk")
    self.assertEqual(out[1][2], out[0][2])
    print(f"v == circular: loss {out[0][2]:.6f}, {len(jax.tree.leaves(out[0][1]))} leaves bitwise")

  def test_stash_schedule_matches_reference(self):
    """`schedule='stash'`: a budget of three 'lite' sets per stage, so some
    (chunk, microbatch)s stash and some do not -- against the monolith."""
    scfg = tiny(layers=4, fsdp=1, ep=2)

    def plan_check(p):
      self.assertTrue(p.modes)
      self.assertLess(len(p.modes), 3 * p.M)
      self.assertTrue(all(g <= 3 * gib + 1e-9 for g in p.peak_stash_gib()))

    gib = lite_gib(scfg)
    self._check(
        scfg,
        vv=2,
        mb=4,
        gbs=8,
        schedule="stash",
        overlap=True,
        plan_check=plan_check,
        stash="lite",
        stash_budget_gib=3 * gib,
    )

  @testing.slow
  def test_stash_vs_remat_same_order_bitwise(self):
    """The stash schedule's order run with its modes and with every (chunk,
    microbatch) in remat: one-layer chunks, so the stash programs are today's
    bit for bit (ds_ep.ChunkPrograms.build_stash), host variants included --
    the loss and every gradient, circular and V, overlap on and off.  Under V
    the stash hosts include stage 0's last backward with its own reduce-
    scatter and a forward of chunk 0 prefetching the head."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    for placement, overlap in (("circular", True), ("circular", False), ("v", True), ("v", False)):
      ps = dsp.Pipeline(
          scfg,
          n_stages=2,
          chunks_per_stage=2,
          n_micro=4,
          global_batch=8,
          seq=256,
          schedule="stash",
          stash="lite",
          stash_budget_gib=2 * gib,
          overlap=overlap,
          placement=placement,
      )
      pr = dsp.Pipeline(
          scfg,
          n_stages=2,
          chunks_per_stage=2,
          n_micro=4,
          global_batch=8,
          seq=256,
          schedule=ps.order,
          overlap=overlap,
          placement=placement,
      )
      self.assertTrue(ps.modes)
      self.assertEqual(pr.modes, {})
      if overlap:  # a stash-mode op hosts a folded collective
        self.assertTrue(any(op.stash and (op.pf or op.rs or op.self_pf or op.rs_self is not None) for op in ps.ops()))
      if overlap and placement == "v":
        self.assertTrue(any(op.stash and op.rs_self is not None and op.c == 0 for op in ps.ops()))
        self.assertTrue(any(op.stash and op.pf and op.pf[0] == 3 for op in ps.ops()))
      out = []
      for p in (ps, pr):
        params, _, wc0 = p.init(jax.random.key(7))
        batch = p.synthetic_batch(8)
        if overlap:
          out.append(p.loss_and_grads_overlap(params, wc0, batch))
        else:
          out.append(p.loss_and_grads(params, batch))
      assert_bitwise(self, out[0], out[1], f"stash vs remat, {placement} overlap={overlap}")
      print(f"stash vs remat {placement} overlap={overlap}: {len(ps.modes)} stash (chunk, microbatch)s, " "bitwise")

  # ------------------------------------------------- F&B pairs
  @testing.slow
  def test_paired_vs_unpaired_bitwise(self):
    """An F&B pair is its two programs composed into one: interleaved 1F1B paired on stage 1 and on both stages
    against the same per-stage sequences unpaired, at M = 4 and 6 -- the loss and every gradient bit for bit,
    overlap on (and off at M = 4).  A pair hosts no collective, so the reduce-scatter plan can move; the stage
    sumsqs are bitwise where it does not, and within rounding where it does.  The plan is keyed by op, not by order
    index: `merge` interleaves the stages of a paired and an unpaired order differently."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    k = dict(n_stages=2, chunks_per_stage=2, seq=256)

    def plan(p):
      by_op = lambda d: {p.order[j]: v for j, v in d.items()}
      self_pf = None if p.self_pf_at is None else p.order[p.self_pf_at]
      return by_op(p.pf), by_op(p.rs_at), by_op(p.rs_self), p.final_slots, self_pf

    def same(p, ref, ov_list, want, what):
      for ov in ov_list:
        got, w = run_pipe(p, 3, 4, ov), want[ov]
        tag = f"{what} overlap={ov}"
        if not ov:
          assert_bitwise(self, got, w, tag)
          continue
        assert_bitwise(self, (got[0], got[2]), (w[0], w[2]), tag)
        if plan(p) == plan(ref):
          assert_bitwise(self, got[1], w[1], tag + " sumsq")
        else:
          np.testing.assert_allclose(np.asarray(got[1]), np.asarray(w[1]), rtol=1e-6, err_msg=tag)
      print(f'{what}: {len(p.fuse)} pairs, bitwise (plan {"same" if plan(p) == plan(ref) else "moved"})')

    for M in (4, 6):
      k.update(n_micro=M, global_batch=2 * M)
      ovs = (True, False) if M == 4 else (True,)
      ref = dsp.Pipeline(scfg, schedule="interleaved", **k)
      self.assertEqual(ref.fuse, frozenset())
      want = {ov: run_pipe(ref, 3, 4, ov) for ov in ovs}
      for ps in ((1,), (0, 1)):
        p = dsp.Pipeline(scfg, schedule="interleaved", pair_stages=ps, **k)
        self.assertEqual(stage_seqs(p), stage_seqs(ref))
        self.assertEqual(len(p.fuse), n_pairs(dsp.sched.megatron_interleaved(2, 2, M, pair_stages=ps)))
        self.assertEqual({p.place[p.order[j][0]] for j in p.fuse}, set(ps))
        same(p, ref, ovs if ps == (0, 1) else (True,), want, f"interleaved M={M} pairs on {ps}")

  @testing.slow
  def test_interleaved_pairs_match_reference(self):
    """Interleaved 1F1B paired on both stages against the monolith, overlap off and on."""

    def plan_check(p):
      self.assertEqual(p.place, [0, 1, 0, 1])
      self.assertEqual(len(p.fuse), n_pairs(dsp.sched.megatron_interleaved(2, 2, 4, pair_stages=None)))
      self.assertEqual({p.place[p.order[j][0]] for j in p.fuse}, {0, 1})

    for overlap in (False, True):
      self._check(
          tiny(layers=4, **testing.stage_axes()),
          vv=2,
          mb=4,
          gbs=stage_rows(16),
          schedule="interleaved",
          pair_stages="all",
          overlap=overlap,
          plan_check=plan_check,
      )

  @testing.slow
  def test_interleaved_pairs_equal_search_bitwise(self):
    """init_by_chunk: interleaved 1F1B with pairs on both stages runs each chunk's programs (composed, for the
    pairs) with FIFO accumulation, as circular placement's search order does -- the loss and every gradient (in
    chunk order) bit for bit."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    out = []
    for kw in (dict(schedule="search"), dict(schedule="interleaved", pair_stages="all")):
      p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, init_by_chunk=True, **kw)
      grads, _, lacc = run_pipe(p, 5, 6)
      out.append((p.chunk_trees(grads), float(jnp.sum(lacc)), len(p.fuse)))
    self.assertEqual(out[0][2], 0)
    self.assertGreater(out[1][2], 0)
    assert_bitwise(self, out[1][0], out[0][0], "gradients by chunk")
    self.assertEqual(out[1][1], out[0][1])
    print(f"interleaved pairs == search: loss {out[0][1]:.6f}, {len(jax.tree.leaves(out[0][0]))} leaves bitwise")

  @testing.slow
  def test_interleaved_pairs_stash_bitwise_vs_remat(self):
    """Interleaved pairs + stash: `stash_modes` hands out 'lite' sets on the fixed order under a two-set budget, some
    inside F&B pairs (a stash forward, a stash backward, or both composed); against the same order and pairs all in
    remat -- one-layer chunks, so bit for bit -- overlap on and off."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    k = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256)
    ps = dsp.Pipeline(scfg, schedule="interleaved", pair_stages="all", stash="lite", stash_budget_gib=2 * gib, **k)
    pr = dsp.Pipeline(scfg, schedule=ps.order, fuse=ps.fuse, rs_reserve=ps.rs_reserve, **k)
    self.assertTrue(0 < len(ps.modes) < 3 * ps.M)
    self.assertEqual((pr.modes, pr.fuse), ({}, ps.fuse))
    self.assertTrue(all(g <= 2 * gib + 1e-9 for g in ps.peak_stash_gib()))
    ops = ps.ops()
    mixed = [(ops[j].stash, ops[j + 1].stash) for j in ps.fuse]
    self.assertTrue(any(a or b for a, b in mixed), mixed)
    for overlap in (True, False):
      assert_bitwise(
          self, run_pipe(ps, 7, 8, overlap), run_pipe(pr, 7, 8, overlap), f"interleaved stash vs remat overlap={overlap}"
      )
    print(f"interleaved stash vs remat: {len(ps.modes)} stash (chunk, microbatch)s, pairs (F, B) stash {mixed}, bitwise")

  @testing.slow
  def test_three_step_training(self):
    """Three AdamW steps, precompiled: interleaved 1F1B with pairs on both stages and 'lite' stash sets against
    circular search from the same weights (init_by_chunk) -- the losses finite and falling, the first bit for bit,
    all within 1e-4 (the clip's global norm sums the stage sumsqs in a plan-dependent order, so the updates agree
    to rounding)."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    k = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, init_by_chunk=True)
    losses = []
    for kw in (dict(), dict(schedule="interleaved", pair_stages="all", stash="lite", stash_budget_gib=2 * gib)):
      p = dsp.Pipeline(scfg, **k, **kw)
      if p.fuse:
        p.precompile(threads=16)
        self.assertEqual(len(p._exe), len(set(p.dispatched())))
        self.assertTrue(any((op.c, op.pair) in p._exe for op in p.ops() if op.pair))
      params, opts, wc0 = p.init(jax.random.key(9))
      batch, lrs = p.synthetic_batch(10), p.lr_arrays(1e-3)
      ls = []
      for _ in range(3):
        params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
        ls.append(float(jnp.sum(lacc)))
        self.assertTrue(np.isfinite(ls[-1]) and np.isfinite(float(gn)))
      self.assertLess(ls[-1], ls[0])
      losses.append(ls)
    self.assertEqual(losses[1][0], losses[0][0])
    np.testing.assert_allclose(losses[1], losses[0], rtol=1e-4)
    print(f"three steps: circular {losses[0]}, interleaved + pairs + stash {losses[1]}")

  @PORT_NEEDS_YNN_OFF
  def test_matches_independent_port_interleaved_pairs(self):
    """The port check with interleaved 1F1B: pairs on both stages and 'lite' stash sets, some of them inside
    pairs."""

    def plan_check(p):
      self.assertTrue(p.fuse and 0 < len(p.modes) < 3 * p.M)
      self.assertTrue(any(p.modes.get(p.order[i][::2]) for j in p.fuse for i in (j, j + 1)))

    scfg = tiny(layers=4, random_routing=False, **testing.stage_axes())
    scfg = dataclasses.replace(scfg, model=dataclasses.replace(scfg.model, num_experts_per_tok=8))
    gib = pipe(scfg, n_micro=4, global_batch=stage_rows(16)).progs[0].build_stash("lite").bytes / 2**30
    self._port(
        plan_check,
        schedule="interleaved",
        pair_stages="all",
        n_micro=4,
        global_batch=stage_rows(16),
        stash="lite",
        stash_budget_gib=2 * gib,
    )

  # ------------------------------------------------ shard_mode='explicit'
  def test_explicit_v_cap2_bitwise_vs_auto(self):
    """The stage programs of the DeepSeek-V3 runs across two v5p-128 slices,
    with MaxText's kernels (megablox, the
    ragged sort and tokamax splash, in interpret mode): V placement at M=2,
    cap2 dispatch buffers, lax FSDP collectives, the gather embedding, top-k
    routing and the remat'd last chunk.  shard_mode explicit gives auto's
    gradients, sumsqs and loss, and its params, loss and gradient norm after
    a step, bit for bit (`mode_run`)."""
    sc = tiny(
        layers=4,
        random_routing=False,
        fsdp_collectives="xla",
        iota_embed=False,
        remat_fused_last=True,
        ragged_buffer_factor=2.0,
        **testing.stage_axes(),
    )
    runs = []
    for scfg in (sc, explicit(sc)):
      p = pipe(scfg, placement="v", n_micro=2, global_batch=stage_rows(8), init_by_chunk=True)
      self.assertTrue(p.rs_self and p.pf)
      runs.append(mode_run(p, 2, 4, steps=1))
    assert_same_run(self, runs[1], runs[0], "V cap2")
    print(
        f'explicit == auto, V cap2 with MaxText\'s kernels: {len(runs[0]["grads"])} gradient leaves, loss '
        f'{runs[0]["loss"]:.6f}, gnorm {runs[0]["gnorms"][0]:.6f}'
    )

  def test_explicit_programs_lower_as_the_fixed_ones(self):
    """Under shard_mode explicit too, every program of a circular search and
    timed plan with the fusion plan lowers to its fixed program's StableHLO
    text (test_fixed_programs_lower_as_before's check): the composed parts and
    the hand-built prefetch / reduce programs take and return the same types."""
    scfg = explicit(tiny(layers=4, fsdp=1, ep=2))
    n = 0
    for schedule, M in (("search", 6), ("timed", 4)):
      p = pipe(scfg, schedule=schedule, n_micro=M, global_batch=2 * M)
      for c, desc in dict.fromkeys(p.dispatched()):
        op = next(o for o in p.ops() if (o.c, o.pair or o.desc) == (c, desc))
        av = p._desc_avals(c, desc)
        got, want = p.program(op), legacy_program(p, op)
        self.assertEqual(got.__name__, want.__name__, op)
        self.assertTrue(got.lower(*av).as_text() == want.lower(*av).as_text(), (schedule, op))
        n += 1
    print(f"explicit: {n} programs lower as the fixed ones")


@testing.needs_devices(4)
class ScheduleChoice(unittest.TestCase):
  """What the driver makes of `schedule` and `costs`.  Nothing is compiled, but for the runs of
  test_interleave_group_trains_the_same and test_interleave_group_matches_the_reference."""

  def test_custom_order_as_lists(self):
    # An order read from JSON has lists for ops.  `validate` accepts it, so the
    # driver must run it: the same order and fusion plan as the tuples give.
    sched = dsp.sched
    order = sched.search(4, 2, sched.placement(2, 2))[0]
    ps = [
        dsp.Pipeline(
            tiny(layers=4, **testing.stage_axes()),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=8,
            seq=256,
            schedule=o,
        )
        for o in (order, [list(a) for a in order])
    ]
    self.assertEqual(ps[1].order, order)
    plan = lambda p: (
        p.self_pf_at,
        {j: v[:2] for j, v in p.pf.items()},
        p.pf_n,
        {j: v[:2] for j, v in p.rs_at.items()},
        sorted(p.rs_self),
        p.final_slots,
    )
    self.assertEqual(plan(ps[1]), plan(ps[0]))

  def test_default_schedule_is_breadth(self):
    """No schedule -- none given to Pipeline, no --schedule -- runs
    DEFAULT_SCHEDULE, breadth, on either placement, with repeat, breadth's
    default scope.  search, the default before, picks 1F1B at one chunk a
    stage and M=4 and another order on V, so each case tells the two apart."""
    sched = dsp.sched
    scfg = tiny_xla(fsdp=1, ep=2)
    k = dict(n_stages=2, n_micro=4, global_batch=8, seq=256)
    for p in (dsp.Pipeline(scfg, chunks_per_stage=1, **k), dsp.Pipeline(scfg, chunks_per_stage=2, placement="v", **k)):
      with self.subTest(placement=p.placement, chunks=p.C):
        self.assertEqual(
            (p.schedule_name, p.order, p.fsdp_scope), ("breadth", sched.breadth_first(p.C, 4, p.place), "repeat")
        )
        self.assertNotEqual(p.order, sched.search(p.C, 4, p.place)[0])
    ap = dsp.make_parser()
    for more in ([], ["--vv", "2", "--layers", "4", "--placement", "v"]):
      a = ap.parse_args(TINY_ARGS + ["--mb", "4", "--gbs", "8"] + more)
      self.assertIsNone(a.schedule)
      self.assertEqual(dsp.build(a).schedule_name, "breadth", more)

  def check_plan(self, p):
    """Every host of the fusion plan is legal and every chunk is gathered and
    reduced exactly once (the rules `_plan_overlap` promises)."""
    C, order, place = p.C, p.order, p.place
    tag = f"{p.placement} {p.schedule_name} M={p.M}"
    paired = set(p.fuse) | {j + 1 for j in p.fuse}
    self.assertEqual(p.self_pf_at, order.index((0, "F", 0)), tag)
    self.assertNotIn(p.self_pf_at, paired)
    covered = collections.defaultdict(list)  # chunk -> leaf ids gathered in forwards
    for j, (b, g) in p.pf.items():
      c, k, m = order[j]
      self.assertTrue(k == "F" and c != C - 1 and place[c] == place[b], (tag, j))
      self.assertLess(j, order.index((b, "F", 0)), (tag, j))
      self.assertNotEqual(j, p.self_pf_at, tag)
      covered[b] += list(g)
    for s in range(p.S):
      for i, ch in enumerate(p.stage_chunks[s]):
        if i in p.gather_slots[s]:
          self.assertNotIn(ch.index, covered, tag)
        else:
          self.assertEqual(sorted(covered[ch.index]), list(range(p.n_leaves[ch.index])), tag)
          self.assertLessEqual(max(p.pf_n[j] for j, (b, _) in p.pf.items() if b == ch.index), p.pf_split)
    reduced = collections.defaultdict(list)
    for j, (o, g) in p.rs_at.items():
      c, k, m = order[j]
      self.assertTrue(k == "B" and c != C - 1 and place[c] == place[o], (tag, j))
      self.assertGreater(j, p.last_writer[o], (tag, j))
      reduced[o] += list(g)
    self.assertFalse(set(p.rs_at) & set(p.rs_self), tag)
    for j, n in p.rs_self.items():
      c, k, m = order[j]
      s = place[c]
      stage_last = max(p.last_writer[ch.index] for ch in p.stage_chunks[s])
      self.assertTrue(k == "B" and j == p.last_writer[c] == stage_last, (tag, j))
      self.assertNotEqual(j, p.first_writer[c], tag)
      self.assertEqual(n, p.n_partial_sq[s], tag)
      self.assertTrue(all(h < j for h, (o, _) in p.rs_at.items() if place[o] == s), tag)
    for ch in p.chunks:
      c, s = ch.index, ch.stage
      ways = [c in reduced, p.slot[c] in p.final_slots[s], p.last_writer[c] in p.rs_self]
      self.assertEqual(sum(ways), 1, (tag, c, ways))
      if c in reduced:
        self.assertEqual(sorted(reduced[c]), list(range(p.n_leaves[c])), tag)
        self.assertLessEqual(p.rs_groups[c], p.rs_split)
    # the op records agree with the plan
    for op in p.ops():
      self.assertEqual(op.pair is not None, op.j in p.fuse, tag)
      if op.desc is None:
        self.assertEqual((op.c, op.kind), (C - 1, "B"))
        continue
      self.assertEqual(op.stash, p.modes.get((op.c, op.m)), tag)
      writes = op.kind == "B" or op.c == C - 1
      self.assertEqual(op.acc, writes and op.j != p.first_writer[op.c], tag)
    # an F&B pair hosts no collective, and its program is its two ops' parts
    ops = p.ops()
    for j in p.fuse:
      f, b = ops[j], ops[j + 1]
      for op in (f, b):
        self.assertTrue(op.pf is None and op.rs is None and not op.self_pf and op.rs_self is None, (tag, op))
      self.assertEqual(f.pair[0], f.desc[0], tag)
      self.assertEqual(f.pair[1][0], "B", tag)
      self.assertEqual({k: v for k, v in f.pair[1][1] if k != "peer"}, dict(b.desc[0][1]), tag)
      self.assertIs(dict(f.pair[1][1]).get("peer"), p.progs[b.c] if b.c != f.c else None, tag)
    self.assertEqual(len(p.dispatched()), sum(op.desc is not None for op in ops) - len(p.fuse), tag)

  @pytest.mark.scheduled_only
  def test_plan_hosts_valid(self):
    """The fusion plan over placements, schedules (stash included: its hosts
    may be in stash mode) and microbatch counts.  Under V the last backward of
    stage 0 is chunk 0's own rs_self (the head finishes first and used to
    take it), and prefetching the head needs only one forward of chunk 0."""
    sch = dsp.sched
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    stash_hosts = 0
    for placement in ("circular", "v"):
      for schedule in ("search", "timed", "stash"):
        for M in (2, 4, 8):
          kw = dict(stash="lite", stash_budget_gib=3 * gib) if schedule == "stash" else {}
          p = pipe(scfg, n_micro=M, global_batch=2 * M, schedule=schedule, placement=placement, **kw)
          self.check_plan(p)
          stash_hosts += sum(
              bool(op.stash and (op.pf or op.rs or op.self_pf or op.rs_self is not None)) for op in p.ops()
          )
          if placement == "v":
            self.assertIn(p.last_writer[0], p.rs_self, p.schedule_name)
    self.assertGreater(stash_hosts, 0)
    # rs_self off: every stage's leftovers get a standalone program (check_plan's
    # rules are the step scope's; RepeatScope checks the default's, repeat)
    p = pipe(scfg, n_micro=4, rs_self=False, fsdp_scope="step")
    self.check_plan(p)
    self.assertEqual(p.rs_self, {})
    # 1 to pf_split prefetch hosts: V at M=2 in search's order has one forward of chunk 0
    # before the head's (breadth, the default, runs both first)
    p = pipe(scfg, n_micro=2, global_batch=4, placement="v", schedule="search")
    self.assertEqual([(b, p.pf_n[j]) for j, (b, _) in p.pf.items()], [(3, 1)])
    # every distinct program descriptor traces with the avals precompile gives it
    p = pipe(scfg, n_micro=4, schedule="stash", stash="lite", stash_budget_gib=3 * gib, placement="v")
    keys = list(dict.fromkeys((op.c, op.desc) for op in p.ops() if op.desc is not None))
    for c, desc in keys:
      jax.eval_shape(p.progs[c].program(desc), *p._desc_avals(c, desc))
    self.assertGreater(len(keys), 8)
    self.assertEqual(
        sch.peak_bytes(p.order, p.place, p.modes, {"lite": p.stash_bytes["lite"]}),
        [round(g * 2**30) for g in p.peak_stash_gib()],
    )

  @testing.needs_devices(8, "four stages of fsdp 1 x ep 2")
  def test_sync_stage_reduces_at_its_end(self):
    """A stage whose collectives run synchronously (`sync_collective_stages`:
    its strip holds chip (0,0,0)) other than chunk 0's reduce-scatters every
    chunk in one program after its last backward instead of in its backwards,
    and only that stage's plan changes.  rs_self='first' folds a
    reduce-scatter into its own backward on chunk 0's stage only: the others
    reduce after their last backward.  (Run:
    PipelineXlaKernels.test_sync_stage_run.)"""
    scfg = tiny(layers=8, fsdp=1, ep=2, fsdp_collectives="xla")
    for schedule in ("search", "timed"):
      kw = dict(n_stages=4, n_micro=8, global_batch=16, schedule=schedule)
      plain = pipe(scfg, **kw)
      self.assertEqual(plain.sync_stages, frozenset())  # CPU: no SparseCore to be denied
      with mock.patch.object(dsp, "sync_collective_stages", lambda meshes: frozenset({3})):
        p = pipe(scfg, **kw)
        first = pipe(scfg, rs_self="first", **kw)
        # collective-permute FSDP collectives are async DMAs on any sub-mesh: never synchronous
        self.assertEqual(pipe(dataclasses.replace(scfg, fsdp_collectives="ppermute"), **kw).sync_stages, frozenset())
      for q in (p, first):
        self.check_plan(q)
        self.assertEqual(q.final_slots[3], (0, 1), schedule)
        self.assertEqual(q.n_partial_sq[3], 0, schedule)
        self.assertFalse([j for j, (c, _) in q.rs_at.items() if q.place[c] == 3], schedule)
      # the synchronous stage's plan changes, no other's
      self.assertEqual(
          {j: v for j, v in p.rs_at.items() if p.place[v[0]] != 3},
          {j: v for j, v in plain.rs_at.items() if plain.place[v[0]] != 3},
          schedule,
      )
      # rs_self='first': chunk 0's stage folds its own reduce-scatter as before; on the others the
      # backward that emits the stage's last cotangent carries no reduce-scatter at all
      self.assertTrue(all(first.place[first.order[j][0]] == first.place[0] for j in first.rs_self), schedule)
      self.assertEqual(
          {j: n for j, n in p.rs_self.items() if p.place[p.order[j][0]] == p.place[0]}, first.rs_self, schedule
      )
      if schedule == "search":  # (timed's circular plan folds none at M=8: rs_self is {} under 'all' too)
        self.assertIn(first.last_writer[0], first.rs_self)
        self.assertTrue(set(p.rs_self) - set(first.rs_self), p.rs_self)  # 'all' folds on other stages
      self.assertEqual(
          {j: v for j, v in first.rs_at.items() if first.place[v[0]] == first.place[0]},
          {j: v for j, v in p.rs_at.items() if p.place[v[0]] == p.place[0]},
          schedule,
      )
      for s in (1, 2):
        stop = max(first.last_writer[ch.index] for ch in first.stage_chunks[s])
        self.assertNotIn(stop, first.rs_at, schedule)
        self.assertNotIn(stop, first.rs_self, schedule)
        self.assertTrue(first.final_slots[s], schedule)  # its last chunk reduces after it
    # V placement (rs_reserve on): 'first' keeps the other stages' last backwards free too, and
    # chunk 0's stage its own rs_self
    for schedule in ("search", "timed"):
      v = pipe(scfg, n_stages=4, n_micro=4, global_batch=16, schedule=schedule, placement="v", rs_self="first")
      self.check_plan(v)
      self.assertIn(v.last_writer[0], v.rs_self, schedule)
      for s in range(v.S):
        stop = max(v.last_writer[ch.index] for ch in v.stage_chunks[s])
        if s != v.place[0]:
          self.assertNotIn(stop, v.rs_at, schedule)
          self.assertNotIn(stop, v.rs_self, schedule)
    with self.assertRaisesRegex(ValueError, "rs_self"):
      pipe(scfg, rs_self="some")

  @testing.needs_devices(8, "four stages of fsdp 1 x ep 2")
  def test_channels_link_the_placements_stages(self):
    """An activation channel for every two stages the placement links (chunk
    c's and chunk c+1's, both ways) and none for the others -- under
    --transport ppermute each would be a program to compile, under xfer a
    uuid range -- at 4 stages: circular with one chunk per stage and V link
    neighbours only, circular with two also stage 3 to stage 0.  One process
    hops the gradient norm with device_put: no scalar channels."""
    scfg = tiny(layers=8, fsdp=1, ep=2)
    both = lambda *links: {x for a, b in links for x in ((a, b), (b, a))}
    chain = both((0, 1), (1, 2), (2, 3))
    for kw, want in (
        (dict(chunks_per_stage=1), chain),
        (dict(), chain | both((3, 0))),
        (dict(transport="ppermute"), chain | both((3, 0))),
        (dict(placement="v"), chain),
    ):
      p = pipe(scfg, n_stages=4, n_micro=4, global_batch=8, schedule="gpipe", **kw)
      self.assertEqual(set(p.channels), want, kw)
      self.assertEqual(p.sq_channels, {}, kw)
      kind = transfer.PermuteChannel if kw.get("transport") == "ppermute" else transfer.Channel
      self.assertTrue(all(type(ch) is kind for ch in p.channels.values()), kw)

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "plans of four stages of fsdp 1 x ep 2, and of two of fsdp 2 x ep 2")
  def test_circular_plans_are_the_first_planners(self):
    """Circular placement keeps the first planner's fusion plans: keeping a stage's last
    backward free for rs_self (`rs_reserve`) is V's rule only -- under
    circular it would move reduce-scatter hosts at config A (timed) and B
    (search and timed), and the stage sumsq's summation order with them.
    Pinned at config A / B dims with the v6e-tile costs (that planner's plans:
    {order index: (chunk, leaves)}, rs_self, final_slots), and held to the
    planner itself (legacy_rs_plan) over a sweep of tiny pipelines."""
    import argparse

    a = argparse.Namespace(
        vocab=32768,
        emb=7168,
        layers=4,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=16,
        topk=8,
        attn="tokamax_splash",
        sa_block=2048,
        sa_block_kv_compute=512,
        cost_estimate=5_000_000_000_000,
        seq=4096,
        weight_dtype="bfloat16",
    )
    cfg = dsp.model_config(a)
    wi, wo = ds_ep.TILES["v6e"](cfg)
    scfg = ds_ep.StageConfig(model=cfg, fsdp=1, ep=2, wi_tiling=wi, wo_tiling=wo)
    costs = dict(F0=24.75, F=22.44, B0=55.84, B=53.10, A=71.46, xfer=1.0)  # V6E_TILES_COSTS
    search_rs = lambda b: {b: (3, 2), b + 2: (3, 2), b + 4: (3, 10), b + 6: (3, 4)}
    want = {  # (schedule, M): (rs_at, rs_self, final_slots)
        ("search", 8): ({**search_rs(42), 55: (2, 1), 57: (2, 1), 58: (2, 1), 59: (2, 13)}, [56, 63], [(), ()]),
        ("timed", 8): ({59: (3, 3), 61: (2, 2), 62: (3, 15), 63: (2, 14)}, [], [(0,), (0,)]),
        ("search", 4): ({**search_rs(22), 27: (2, 1), 29: (2, 1), 30: (2, 1), 31: (2, 13)}, [], [(0,), (0,)]),
        ("timed", 4): ({27: (3, 3), 29: (2, 2), 30: (3, 15), 31: (2, 14)}, [], [(0,), (0,)]),
    }
    got = lambda p: ({j: (c, len(g)) for j, (c, g) in p.rs_at.items()}, sorted(p.rs_self), p.final_slots)
    for (schedule, M), plan in want.items():
      p = dsp.Pipeline(
          scfg,
          n_stages=2,
          chunks_per_stage=2,
          n_micro=M,
          global_batch=2 * M,
          seq=4096,
          schedule=schedule,
          layout="fsdp_on_chip",
          costs=costs,
      )
      self.assertEqual(got(p), plan, (schedule, M))
      self.check_plan(p)
      if schedule == "timed":
        # the rule itself, asked for: each stage's last backward folds its own
        # reduce-scatter, and the finishing chunk's rides in one host
        q = dsp.Pipeline(
            scfg,
            n_stages=2,
            chunks_per_stage=2,
            n_micro=M,
            global_batch=2 * M,
            seq=4096,
            schedule=schedule,
            layout="fsdp_on_chip",
            costs=costs,
            rs_reserve=True,
        )
        self.assertEqual(q.order, p.order)
        self.assertEqual(sorted(q.rs_self), sorted(plan[0])[-2:])
        self.assertEqual(q.final_slots, [(), ()])
        self.check_plan(q)
    # the first planner over tiny circular pipelines (98 of these plans moved
    # when the rule applied to circular too)
    n = 0
    for S, V, L, fsdp in ((2, 2, 4, 1), (2, 1, 4, 1), (4, 1, 4, 1), (2, 2, 4, 2), (4, 2, 8, 1)):
      scfg = tiny(layers=L, fsdp=fsdp, ep=2)
      for M in (1, 2, 3, 4, 6, 8):  # M = 1 at V > 1: the update gathers chunk S (pf_fallback)
        for schedule in ("breadth", "search", "timed", "1f1b", "gpipe"):
          for rs_self in (True, False):
            p = dsp.Pipeline(
                scfg,
                n_stages=S,
                chunks_per_stage=V,
                n_micro=M,
                global_batch=2 * fsdp * M,
                seq=256,
                schedule=schedule,
                rs_self=rs_self,
            )
            tag = (S, V, fsdp, M, schedule, rs_self)
            self.assertEqual((p.rs_at, sorted(p.rs_self), p.final_slots, p.n_partial_sq), legacy_rs_plan(p), tag)
            n += 1
    print(f"{n} circular plans are the first planner's")

  @pytest.mark.scheduled_only
  def test_stash_budget_respected(self):
    """The stash schedule's live stash never exceeds the budget on any stage
    (`peak_bytes` walking the order), nor a chunk its cap; no budget, no
    stash; a V placement's stage 0 stashes only chunk 0 (chunk 3 is the head)."""
    sch = dsp.sched
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg, n_micro=8, global_batch=16)
    for placement in ("circular", "v"):
      for sets in (0, 1, 2.5, 4, 16):
        p = pipe(
            scfg,
            n_micro=8,
            global_batch=16,
            schedule="stash",
            stash="lite",
            stash_budget_gib=sets * gib,
            placement=placement,
        )
        budget = int(sets * gib * 2**30)
        peak = sch.peak_bytes(p.order, p.place, p.modes, {"lite": p.stash_bytes["lite"]})
        self.assertTrue(all(b <= budget for b in peak), (placement, sets, peak, budget))
        if sets < 1 or sets >= 4:
          self.assertEqual(bool(p.modes), sets >= 1, (placement, sets))
        self.assertTrue(all(c != 3 for c, _ in p.modes))
        sch.validate(p.order, p.place, p.M, modes=p.modes)
        # the step's peak grows by more than the live sets: each stash
        # backward's extra temporaries (sched.STASH_BWD_TEMP) come on top
        live, grow = p.peak_stash_gib(), p.peak_stash_gib(temps=True)
        extra = sch.STASH_BWD_TEMP["lite"] * gib
        for g, t in zip(live, grow):
          self.assertTrue(g <= t <= g + extra + 1e-9 and (t > g) == (g > 0), (placement, sets, live, grow))
    p = pipe(
        scfg, n_micro=8, global_batch=16, schedule="stash", stash="lite", stash_budget_gib=16 * gib, stash_caps=[1, 0, 2]
    )
    live, peak = collections.Counter(), collections.Counter()
    for c, k, m in p.order:
      if (c, m) in p.modes:
        live[c] += 1 if k == "F" else -1
        peak[c] = max(peak[c], live[c])
    self.assertEqual(dict(peak), {0: 1, 2: 2})
    with self.assertRaisesRegex(ValueError, "needs schedule stash"):
      pipe(scfg, stash="lite")

  def test_parity_budget_from_the_command_line(self):
    """ds_ep_parity.build: a preset's stash budget in sets (small-v6e-stash's
    three) gives way to --stash-budget-gib, and giving both is an error."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    par.PRESETS["tiny-stash"] = dict(
        par.PRESETS["tiny"], schedule="stash", stash="lite", stash_sets=3, **testing.stage_axes()
    )
    try:
      p = par.build("tiny-stash", random_routing=True)
      b = p.stash_bytes["lite"][0] / 2**30
      self.assertAlmostEqual(p.stash_budget_gib, 3 * b)
      p = par.build("tiny-stash", random_routing=True, stash_budget_gib=0.5 * b)
      self.assertEqual((p.stash_budget_gib, p.modes), (0.5 * b, {}))
      p = par.build("tiny-stash", random_routing=True, stash_sets=1)
      self.assertAlmostEqual(p.stash_budget_gib, b)
      with self.assertRaisesRegex(ValueError, "stash budget once"):
        par.build("tiny-stash", random_routing=True, stash_budget_gib=1.0, stash_sets=2)
    finally:
      del par.PRESETS["tiny-stash"]

  def test_parity_budget_in_sets_with_pairs(self):
    """ds_ep_parity.build: a stash budget in sets under --schedule interleaved with F&B pairs (--pair-stages,
    --pair-gain) sizes a set on the same pipeline without stash under search, which runs no pairs -- and the pair
    flags with it, which build refuses outside interleaved -- and builds the paired run on that budget."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    kw = dict(schedule="interleaved", stash="lite", **testing.stage_axes())
    for pairs in (dict(pair_stages="all"), dict(pair_stages="all", pair_gain=1.5)):
      with self.subTest(**pairs):
        p = par.build("tiny4", random_routing=True, stash_sets=1, **kw, **pairs)
        self.assertEqual((p.schedule, p.pair_stages), ("interleaved", None))  # None: pairs on every stage
        self.assertTrue(p.fuse)
        self.assertEqual(p.pair_gain, pairs.get("pair_gain", p.pair_gain))
        self.assertAlmostEqual(p.stash_budget_gib, p.stash_bytes["lite"][0] / 2**30)

  def test_parity_ladder_skips_infeasible_interleaved_rungs(self):
    """ds_ep_parity drops an interleaved rung its preset cannot run -- one
    chunk per stage, V placement, M not a multiple of S -- and a rung that
    names pair stages for another schedule before anything runs, instead of
    aborting the ladder when the rung builds."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    spec = "circular:search:none,circular:interleaved:none,v:interleaved:none,circular:interleaved:lite"
    kept = lambda vv, mb, stages: [
        (r["placement"], r["schedule"], r["stash"]) for r in par.parse_rungs(spec, vv, mb, stages)
    ]
    self.assertEqual(kept(1, 2, 4), [("circular", "search", "none")])  # small-pp4
    self.assertEqual(kept(2, 6, 4), [("circular", "search", "none")])
    self.assertEqual(
        kept(2, 8, 4),
        [("circular", "search", "none"), ("circular", "interleaved", "none"), ("circular", "interleaved", "lite")],
    )  # small-pp4-v2
    paired = "circular:interleaved:none:1,circular:search:none:1,v:timed:none:all"
    self.assertEqual([r["schedule"] for r in par.parse_rungs(paired, 2, 8, 4)], ["interleaved"])
    self.assertEqual(par.parse_rungs(paired, 2, 8, 4)[0]["pair_stages"], "1")

  def test_default_costs_follow_the_device_and_the_tiles(self):
    """TPU7x costs off a v6e; on one the table measured with the tiles the
    pipeline runs (ds_ep.TILES, at the benchmark's emb 7168 / moe_mlp 2048):
    V6E_TILES_COSTS for 'v6e' and for 'clamped' (unmeasured; unpadded like
    'v6e', see default_costs), V6E_COSTS for any other (padded tiles' too)."""
    sched = dsp.sched
    cfg = DeepSeekConfig(emb_dim=7168, moe_mlp_dim=2048)
    per = {
        name: ds_ep.StageConfig(model=cfg, wi_tiling=t[0], wo_tiling=t[1])
        for name, t in ((n, ds_ep.TILES[n](cfg)) for n in ds_ep.TILES)
    }
    other = ds_ep.StageConfig(model=cfg, wi_tiling=TILE, wo_tiling=TILE)
    padded = (512, cfg.emb_dim, 512) * 3
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # padded tiles: StageConfig warns
      pad = ds_ep.StageConfig(model=cfg, wi_tiling=padded, wo_tiling=padded)
    self.assertIs(dsp.default_costs("cpu", per["v6e"]), sched.TPU7X_COSTS)
    self.assertIs(dsp.default_costs("TPU7x", per["clamped"]), sched.TPU7X_COSTS)
    self.assertIs(dsp.default_costs("TPU v6 lite", other), sched.V6E_COSTS)
    self.assertIs(dsp.default_costs("TPU v6 lite", pad), sched.V6E_COSTS)
    self.assertIs(dsp.default_costs("TPU v6 lite", per["v6e"]), sched.V6E_TILES_COSTS)
    self.assertIs(dsp.default_costs("TPU v6 lite", per["clamped"]), sched.V6E_TILES_COSTS)
    self.assertIs(dsp.default_costs("TPU v6 lite", ds_ep.StageConfig(model=cfg)), sched.V6E_TILES_COSTS)
    # batch split's split_dw 'acc' caps the presets' wi dW tm at 256: still the presets' table
    split = ds_ep.StageConfig(model=cfg, **NATIVE_SPLIT)
    self.assertEqual(split.wi_tiling[6], 256)
    self.assertIs(dsp.default_costs("TPU v6 lite", split), sched.V6E_TILES_COSTS)
    # on Lineage's layers no tiling is capped (nor used): the presets' table
    lin = ds_ep.StageConfig(model=cfg, batch_split=True, shard_mode="explicit", random_routing=False)
    self.assertEqual(lin.wi_tiling, per["clamped"].wi_tiling)
    self.assertIs(dsp.default_costs("TPU v6 lite", lin), sched.V6E_TILES_COSTS)

  @pytest.mark.scheduled_only
  def test_pair_plans_valid(self):
    """Interleaved 1F1B's pairs with stash sets on its fixed order at M = 4 and 8, paired on every stage and on
    stage 1 (`check_plan`: no pair hosts a collective, a pair's program is its two ops' parts): the stash modes are
    `stash_modes`' under the budget, and every distinct program a step dispatches, pairs included, traces with the
    avals precompile gives it.  A custom order with the same pairs (`fuse`, from JSON too) plans the same; and what
    the constructor refuses: pair stages out of range, or given to any schedule but interleaved."""
    sch = dsp.sched
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    for M in (4, 8):
      for ps in ("all", (1,)):
        p = pipe(
            scfg,
            n_micro=M,
            global_batch=2 * M,
            schedule="interleaved",
            pair_stages=ps,
            stash="lite",
            stash_budget_gib=2 * gib,
        )
        self.check_plan(p)
        self.assertTrue(p.fuse and p.modes)
        self.assertTrue(all(g <= 2 * gib + 1e-9 for g in p.peak_stash_gib()))
        self.assertEqual(
            p.modes, sch.stash_modes(p.order, p.place, {"lite": p.stash_bytes["lite"]}, int(2 * gib * 2**30))
        )
        if M == 4 and ps == "all":
          keys = list(dict.fromkeys(p.dispatched()))
          self.assertTrue([d for _, d in keys if any(k == "B" for k, _ in d[1:])])  # the pair programs
          for c, desc in keys:
            jax.eval_shape(p.progs[c].program(desc), *p._desc_avals(c, desc))
    # None, the flag left out, is interleaved's default: no pairs
    self.assertEqual(
        [dsp.parse_pair_stages(x) for x in (None, "all", "none", "", "1", "0,1")], [None, "all", (), (), (1,), (0, 1)]
    )
    with self.assertRaisesRegex(ValueError, r"pair_stages \(2,\): stages are 0\.\.1"):
      pipe(scfg, schedule="interleaved", pair_stages=(2,))
    for schedule in ("search", "stash", None):
      with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, "only schedule 'interleaved' pairs"):
        pipe(scfg, schedule=schedule, pair_stages="all")
    with self.assertRaisesRegex(ValueError, "custom order"):
      pipe(scfg, fuse={3})
    with self.assertRaisesRegex(ValueError, "needs schedule stash or interleaved"):
      pipe(scfg, schedule="timed", stash="lite")
    p = pipe(scfg, schedule="interleaved", pair_stages="all")
    j = min(p.fuse)
    with self.assertRaises(sch.ScheduleError):  # the pair's B is not the next op
      pipe(scfg, schedule=p.order, fuse={j - 1})
    q = pipe(scfg, schedule=[list(a) for a in p.order], fuse=p.fuse, rs_reserve=p.rs_reserve)  # as from JSON
    plan = lambda p: (
        p.order,
        p.fuse,
        p.pf,
        p.rs_at,
        p.rs_self,
        p.final_slots,
        [(op.j, op.desc, op.pair is not None) for op in p.ops()],
    )
    self.assertEqual(plan(q), plan(p))

  def test_interleaved_plans_valid(self):
    """Megatron's interleaved 1F1B as Pipeline builds it at M = 2, 4, 8 (M = S:
    every forward in the first rank's warmup), remat and with a budget of two
    stash sets: a legal plan (`check_plan`) in which every stage's last
    gradient write reduces its own gradient, so no stage ends on a standalone
    reduce-scatter under the per-step reduce (`rs_reserve`); the budget's stash
    modes on the fixed order, binding; and what the constructor refuses."""
    sch = dsp.sched
    scfg = tiny(layers=4, fsdp=1, ep=2)
    gib = lite_gib(scfg)
    for M in (2, 4, 8):
      for stash in (False, True):
        kw = dict(stash="lite", stash_budget_gib=2 * gib) if stash else {}
        p = pipe(scfg, n_micro=M, global_batch=2 * M, schedule="interleaved", **kw)
        tag = f"M={M} stash={stash}"
        self.check_plan(p)
        self.assertFalse(p.fuse, tag)
        self.assertTrue(p.rs_reserve, tag)
        self.assertEqual(p.final_slots, [(), ()], tag)
        for s in range(p.S):  # chunk s writes its stage's last gradient
          self.assertIn(p.last_writer[s], p.rs_self, tag)
        self.assertEqual(bool(p.modes), stash, tag)
        self.assertTrue(all(g <= 2 * gib + 1e-9 for g in p.peak_stash_gib()), tag)
        if stash:
          self.assertEqual(
              p.modes, sch.stash_modes(p.order, p.place, {"lite": p.stash_bytes["lite"]}, int(2 * gib * 2**30)), tag
          )
          self.assertLess(len(p.modes), (p.C - 1) * M, tag)  # the budget binds
    q = pipe(
        scfg,
        n_micro=8,
        global_batch=16,
        schedule="interleaved",
        stash="lite",
        stash_budget_gib=2 * gib,
        stash_caps=[1, 0],
    )
    self.assertEqual(
        q.modes,
        sch.stash_modes(q.order, q.place, {"lite": q.stash_bytes["lite"]}, int(2 * gib * 2**30), [1, 0], policy="lite"),
    )
    self.assertEqual({c for c, _ in q.modes}, {2})  # a chunk left off the caps is uncapped here
    with self.assertRaisesRegex(ValueError, "placement 'circular'"):
      pipe(scfg, schedule="interleaved", placement="v")
    with self.assertRaisesRegex(ValueError, "multiple of n_stages"):
      pipe(scfg, n_micro=3, global_batch=6, schedule="interleaved")
    with self.assertRaisesRegex(ValueError, "chunks_per_stage >= 2"):
      pipe(scfg, chunks_per_stage=1, schedule="interleaved")
    # Megatron's microbatch groups (interleave_group): groups of 4 at S = 2 are a legal plan, rank 0 opens on chunk
    # 0's first four microbatches, and the name says so; only the interleaved schedule takes a group, and M must be
    # a multiple of it
    g = pipe(scfg, n_micro=8, global_batch=16, schedule="interleaved", interleave_group=4)
    self.check_plan(g)
    self.assertIn("groups of 4", g.schedule_name)
    self.assertEqual([a for a in g.order if g.place[a[0]] == 0][:4], [(0, "F", i) for i in range(4)])
    self.assertNotEqual(g.order, pipe(scfg, n_micro=8, global_batch=16, schedule="interleaved").order)
    with self.assertRaisesRegex(ValueError, "interleave_group"):
      pipe(scfg, n_micro=8, global_batch=16, schedule="breadth", interleave_group=4)
    with self.assertRaisesRegex(ValueError, "multiple of"):
      pipe(scfg, n_micro=6, global_batch=12, schedule="interleaved", interleave_group=4)

  @pytest.mark.scheduled_only
  def test_interleaved_pair_plans_valid(self):
    """Interleaved 1F1B with its steady state's F&B pairs (pair_stages 'all' or one stage) as Pipeline builds it at
    M = 4 and 8, with recompute and with saved residuals: a legal plan (`check_plan`: no pair hosts a collective, a
    pair's program is its two ops' parts) holding the streams' pairs (sched.megatron_interleaved), each stage's op
    sequence and the fusion plan, op by op, the unpaired order's -- every gather, reduce-scatter and rs_self is on
    an unpaired op either way -- and a name that says so; pair_stages None (the default) and () are the unpaired
    order itself.  Every distinct program a paired step dispatches traces with the avals precompile gives it.  From
    the command line --pair-stages pairs it, its absence does not (under --platform gpu's defaults neither); any
    other schedule refuses --pair-stages and --pair-gain."""
    sch = dsp.sched
    by_op = lambda p, d: {p.order[j]: v for j, v in d.items()}
    plan = lambda p: (
        by_op(p, p.pf),
        by_op(p, p.rs_at),
        by_op(p, p.rs_self),
        p.final_slots,
        p.order[p.self_pf_at],
        p.gather_slots,
    )
    pairs = lambda p: sorted((p.order[j], p.order[j + 1]) for j in p.fuse)
    for save in (False, True):
      scfg = tiny(layers=4, fsdp=1, ep=2, **(dict(save_residuals=True, layer_tree="per_layer") if save else {}))
      for M in (4, 8):
        ref = pipe(scfg, n_micro=M, global_batch=2 * M, schedule="interleaved")
        self.assertEqual((ref.fuse, ref.pair_stages), (frozenset(), ()))
        self.assertNotIn("pairs", ref.schedule_name)
        none = pipe(scfg, n_micro=M, global_batch=2 * M, schedule="interleaved", pair_stages=())
        self.assertEqual((none.order, none.fuse, none.schedule_name), (ref.order, ref.fuse, ref.schedule_name))
        for ps in ("all", (1,), (0,)):
          tag = f"save={save} M={M} pair_stages={ps}"
          p = pipe(scfg, n_micro=M, global_batch=2 * M, schedule="interleaved", pair_stages=ps)
          self.check_plan(p)
          streams = sch.megatron_interleaved(2, 2, M, pair_stages=None if ps == "all" else ps)
          self.assertEqual(pairs(p), sorted(it for st in streams for it in st if isinstance(it[0], tuple)), tag)
          self.assertEqual(len(p.fuse), n_pairs(streams), tag)
          self.assertTrue(p.fuse, tag)
          self.assertEqual(stage_seqs(p), stage_seqs(ref), tag)
          self.assertEqual(plan(p), plan(ref), tag)
          on = "every stage" if ps == "all" else f"stages {list(ps)}"
          self.assertIn(f"pairs {len(p.fuse)} ({on}, gain 0 ms)", p.schedule_name, tag)
          progs, _ = p.dispatches()
          self.assertEqual(progs, ref.dispatches()[0] - len(p.fuse), tag)
          if save and M == 4 and ps == "all":
            for c, desc in dict.fromkeys(p.dispatched()):
              jax.eval_shape(p.progs[c].program(desc), *p._desc_avals(c, desc))
    ap = dsp.make_parser()
    base = TINY_ARGS + ["--vv", "2", "--layers", "4", "--mb", "4", "--gbs", "8", "--schedule", "interleaved"]
    for extra, want in (
        ([], 0),
        (["--platform", "gpu"], 0),
        (["--pair-stages", "none"], 0),
        (["--pair-stages", "all"], 6),
        (["--pair-stages", "1"], 2),
        (["--pair-stages", "all", "--pair-gain", "2"], 6),
    ):
      with self.subTest(extra=extra):
        a = ap.parse_args(base + extra)
        self.assertEqual(len(dsp.build(a).fuse), want)
    for extra in (["--pair-stages", "all"], ["--pair-gain", "2"]):
      for schedule in ([], ["--schedule", "search"]):
        with (
            self.subTest(extra=extra, schedule=schedule),
            self.assertRaisesRegex(ValueError, r"^--pair-stages / --pair-gain: .*no other schedule runs"),
        ):
          dsp.build(ap.parse_args(base[:-2] + schedule + extra))

  @pytest.mark.scheduled_only
  def test_interleave_group_trains_the_same(self):
    """An interleaved order with microbatch groups of 4 against groups of S = 2 (`mode_run`): every gradient leaf
    and the loss bit for bit -- a group moves when each chunk runs, never the order of a chunk's microbatches (FIFO,
    which `sched.validate` holds every order to), so every gradient adds in the same order -- and two training steps
    to rounding: the reduce-scatters regroup too, and a stage's sumsq, the sum of its groups' partial sums, moves in
    the last bit for some inits -- this one among them on jax 0.11.1 with XLA:CPU's YNNPACK fusions off (XLA_FLAGS).
    Against the monolith: test_interleave_group_matches_the_reference."""
    k = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, schedule="interleaved")
    p = dsp.Pipeline(tiny_xla(layers=4, fsdp=1, ep=2), interleave_group=4, **k)
    q = dsp.Pipeline(tiny_xla(layers=4, fsdp=1, ep=2), **k)
    self.assertNotEqual(p.order, q.order)
    self.assertNotEqual(sorted(p.rs_at.values()), sorted(q.rs_at.values()))  # the reduce-scatters regroup
    assert_same_run(self, mode_run(p, 1, 3), mode_run(q, 1, 3), "groups of 4 vs groups of S", regrouped=True)

  @pytest.mark.scheduled_only
  def test_interleave_group_matches_the_reference(self):
    """Megatron's interleaved order in microbatch groups of 4, two of them at M = 8, against the monolith: the loss
    and every gradient leaf to fp32 rounding.  test_interleave_group_trains_the_same holds groups of S to groups of 4
    bit for bit, which cannot show a fault the two orders share (say in the reduce-scatter each stage folds into
    chunk s's last backward, as both do); this can."""
    p = dsp.Pipeline(
        tiny_xla(layers=4, **testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=8,
        global_batch=32,
        seq=256,
        schedule="interleaved",
        interleave_group=4,
    )
    compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)

  def test_pair_schedules_at_config_a(self):
    """Config A's dims and v6e-tile costs (pair gain 1.9 ms, the table's):
    interleaved 1F1B runs 18 pair programs paired on every stage (12 on stage
    0, 6 on stage 1), 6 when only stage 1 pairs, and its simulated loop is
    `merge`'s (at 1.9 ms a pair does not pay: unpaired is faster); rs_reserve
    (its default: no earlier plan to keep) folds each stage's last
    reduce-scatter into its last backward instead of ending the step with a
    standalone one; with 'lite' stash at 4.5 GiB its live sets stay within
    the budget.  Nothing is compiled."""
    import argparse

    sch = dsp.sched
    a = argparse.Namespace(
        vocab=32768,
        emb=7168,
        layers=4,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=16,
        topk=8,
        attn="tokamax_splash",
        sa_block=2048,
        sa_block_kv_compute=512,
        cost_estimate=5_000_000_000_000,
        seq=4096,
        weight_dtype="bfloat16",
    )
    cfg = dsp.model_config(a)
    wi, wo = ds_ep.TILES["v6e"](cfg)
    scfg = ds_ep.StageConfig(model=cfg, fsdp=1, ep=2, wi_tiling=wi, wo_tiling=wo)
    tiles, circ = sch.V6E_TILES_COSTS, sch.placement(2, 2)
    k = dict(
        n_stages=2, chunks_per_stage=2, n_micro=8, global_batch=16, seq=4096, layout="fsdp_on_chip", costs=dict(tiles)
    )
    sim = lambda p: sch.timed_simulate(
        p.order, p.place, tiles, 1.0, fuse_last=True, modes=p.modes, fuse=p.fuse, pair_gain=p.pair_gain
    )[0]
    mk = {}
    for ps, want in (("all", (12, 6)), ((1,), (0, 6)), ((), (0, 0))):
      p = dsp.Pipeline(scfg, schedule="interleaved", pair_stages=ps, **k)
      self.check_plan(p)
      self.assertEqual(p.pair_gain, 1.9)
      per = collections.Counter(p.place[p.order[j][0]] for j in p.fuse)
      self.assertEqual((per[0], per[1]), want)
      mk[ps] = sim(p)
      streams = sch.megatron_interleaved(2, 2, 8, pair_stages=None if ps == "all" else ps)
      self.assertAlmostEqual(mk[ps], sch.merge(streams, circ, tiles, 1.0, 1.9, n_micro=8)[2], places=6)
      self.assertIn(f"sim={mk[ps]:.1f}", p.schedule_name)
    self.assertLess(mk[()], mk[(1,)])  # at 1.9 ms a pair does not pay
    p = dsp.Pipeline(scfg, schedule="interleaved", pair_stages="all", **k)
    self.assertTrue(p.rs_reserve)
    self.assertEqual((len(p.rs_self), p.final_slots), (2, [(), ()]))
    q = dsp.Pipeline(scfg, schedule="interleaved", pair_stages="all", rs_reserve=False, **k)
    self.check_plan(q)
    self.assertEqual((q.order, q.fuse), (p.order, p.fuse))
    self.assertEqual((len(q.rs_self), q.final_slots), (0, [(0,), (0,)]))  # two standalone RS
    self.assertEqual(p.dispatches()[0], q.dispatches()[0] - 2)
    p = dsp.Pipeline(scfg, schedule="interleaved", pair_stages=(1,), stash="lite", stash_budget_gib=4.5, **k)
    self.check_plan(p)
    self.assertTrue(p.modes)
    self.assertTrue(all(g <= 4.5 for g in p.peak_stash_gib()))
    self.assertLess(sim(p), mk[(1,)])
    print(f"config A: interleaved {mk}, + lite 4.5 GiB {sim(p):.1f}")

  def test_dry_run_config_and_summary(self):
    """`main --dry-run` at config A's dims: run.json's config records the
    placement `--placement` defaulted to (circular), the schedule and scope
    the pipeline resolved (`resolved_config`),
    and the summary says what a pair program holds -- no FSDP collective,
    unless --grad-reduce microbatch has every backward, a pair's too,
    reduce-scatter its own gradient.  Nothing is compiled."""
    base = [
        "--fsdp",
        "1",
        "--ep",
        "2",
        "--experts",
        "16",
        "--gbs",
        "16",
        "--mb",
        "8",
        "--layout",
        "fsdp_on_chip",
        "--tiles",
        "v6e",
        "--costs",
        "v6e_tiles",
        "--dry-run",
    ]
    for schedule in ("search", "interleaved"):
      out = dsp.main(base + ["--schedule", schedule])
      self.assertEqual(out["config"]["placement"], "circular", schedule)
      self.assertIn("placement circular;", out["summary"])
    self.assertEqual(dsp.main(base + ["--placement", "v"])["config"]["placement"], "v")
    for more, want in (
        ([], ("circular", "breadth", "repeat")),
        (["--schedule", "search"], ("circular", "search", "step")),
    ):
      rec = dsp.main(base + more)["config"]
      self.assertEqual((rec["placement"], rec["schedule"], rec["fsdp_scope"]), want, more)
    paired = ["--schedule", "interleaved", "--pair-stages", "all"]
    step = dsp.main(base + paired)["summary"]
    mb = dsp.main(base + paired + ["--grad-reduce", "microbatch"])["summary"]
    n = n_pairs(dsp.sched.megatron_interleaved(2, 2, 8, pair_stages=None))
    self.assertIn(f"F&B pairs: {n} programs", step)
    self.assertIn("no pair hosts an FSDP collective", step)
    self.assertIn(f"F&B pairs: {n} programs", mb)
    self.assertNotIn("no pair hosts an FSDP collective", mb)
    self.assertIn("every backward, a pair's too, reduce-scatters its own microbatch gradient", mb)


@testing.needs_devices(4)
class TraceNames(unittest.TestCase):
  """What makes the trace readable: named kernels and named FSDP scopes."""

  def test_named_gmm_is_megablox_bit_for_bit(self):
    from maxtext.kernels.megablox import ops as mblx

    E, M, K, N = 4, 512, 256, 128
    ks = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(ks[0], (M, K), jnp.bfloat16)
    w = (jax.random.normal(ks[1], (E, K, N)) * 0.05).astype(jnp.bfloat16)
    gs = jnp.array([100, 0, 250, 162], jnp.int32)
    ct = jax.random.normal(ks[2], (M, N), jnp.bfloat16)

    def ref(x, w):
      return mblx.gmm(
          x, w, gs, preferred_element_type=jnp.bfloat16, tiling=TILE, group_offset=jnp.int32(0), interpret=True
      )

    def mine(x, w):
      return ds_ep.named_gmm("gmm_wi_0", TILE, True, x, w, gs, jnp.int32(0))

    ya, va = jax.vjp(ref, x, w)
    yb, vb = jax.vjp(mine, x, w)
    np.testing.assert_array_equal(np.asarray(ya), np.asarray(yb))
    for a, b in zip(va(ct), vb(ct)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_kernel_and_fsdp_scopes_reach_the_lowered_program(self):
    """The names are in the op locations -- the HLO op_name / xplane tf_op.

    As in the pipeline, the gather runs outside AD and the reduce-scatter on
    the finished gradient.  Under AD a transform wraps the next scope, so the
    forward kernel reads `jvp(megablox)/gmm_wo` and the backward ones
    `transpose(jvp(megablox))/gmm_wo_dx`: `megablox` takes the wrapper and the
    kernel's own scope, which names its HLO instruction, stays clean.
    """
    mesh = jax.make_mesh((2, 1), ("fsdp", "expert"), devices=jax.devices()[:2])
    E, M, K, N = 2, 256, 128, 128
    gs = jnp.full((E,), M // E, jnp.int32)

    def body(x, wg):
      return jnp.sum(ds_ep.named_gmm("gmm_wo", TILE, True, x, wg, gs, jnp.int32(0)).astype(jnp.float32))

    def step(x, w):
      wg = ds_ep._gather_leaf(w, P(None, "fsdp"), P(None, None), jnp.bfloat16, "ppermute")
      loss, (dx, dwg) = jax.value_and_grad(body, argnums=(0, 1))(x, wg)
      g = ds_ep._scatter_leaf(dwg[None], P(None, "fsdp"), P(None, None), jnp.float32, "ppermute")
      return loss, dx, g

    f = jax.jit(
        jax.shard_map(
            step, mesh=mesh, in_specs=(P(), P(None, "fsdp")), out_specs=(P(), P(), P(None, "fsdp")), check_vma=False
        )
    )
    x = jax.device_put(jnp.ones((M, K), jnp.bfloat16), NamedSharding(mesh, P()))
    w = jax.device_put(jnp.ones((E, K, N), jnp.float32), NamedSharding(mesh, P(None, "fsdp")))
    txt = f.lower(x, w).as_text(debug_info=True)
    for name in (
        '"fsdp_all_gather/ppermute',
        '"fsdp_reduce_scatter/ppermute',
        '"jvp(megablox)/gmm_wo/',
        '"transpose(jvp(megablox))/gmm_wo_dx/',
        '"transpose(jvp(megablox))/gmm_wo_dw/',
    ):
      self.assertIn(name, txt)
    self.assertNotIn("(gmm_wo", txt)


def _rand(avals, seed):
  """Random arrays with `avals`' shapes, dtypes and shardings (ints: token ids)."""
  leaves, tdef = jax.tree.flatten(avals)
  out = []
  for i, a in enumerate(leaves):
    k = jax.random.fold_in(jax.random.key(seed), i)
    if jnp.issubdtype(a.dtype, jnp.integer):
      v = jax.random.randint(k, a.shape, 0, 512, a.dtype)
    else:
      v = (0.05 * jax.random.normal(k, a.shape, jnp.float32)).astype(a.dtype)
    out.append(jax.device_put(v, a.sharding))
  return jax.tree.unflatten(tdef, out)


def _copy(tree):
  return jax.tree.map(lambda a: a.copy(), tree)


@pytest.mark.scheduled_only
@testing.needs_devices(4)
class StashAndCompose(unittest.TestCase):
  """Stash programs and composed programs against the fixed ones, bit for bit,
  on one fsdp 1 x ep 2 stage mesh.

  Exact equality between two different XLA programs needs XLA to round each
  bf16 intermediate where the program says to (xla_allow_excess_precision
  off, EXACT): by default it may keep a fused intermediate in f32, and which
  ones it fuses differs between, say, the recompute and the stash backward.
  On this CPU backend that moved the last bf16 bit of a two-layer chunk's
  gradients; the pipeline's one-layer chunks came out exact either way (but
  under 'all', see check_stash).
  """

  EXACT = {"xla_allow_excess_precision": False}

  @classmethod
  def setUpClass(cls):
    cls.scfg = tiny(layers=4, fsdp=1, ep=2)
    cls.mesh = dsp.stage_meshes(1, fsdp=1, ep=2)[0]
    cls._prs, cls._exe = {}, {}

  def pr(self, index, layers=None, first=None, last=False):
    layers = (index,) if layers is None else tuple(layers)
    first = index == 0 if first is None else first
    key = (index, layers, first, last)
    if key not in self._prs:
      ch = ds_ep.ChunkSpec(index=index, layers=layers, first=first, last=last, stage=0)
      self._prs[key] = ds_ep.ChunkPrograms(self.scfg, ch, self.mesh, mbs=2, seq=256, denom=float(8 * 256))
    return self._prs[key]

  def call(self, fn, *args, opts=None):
    """fn(*args), compiled once per (program, compiler options)."""
    key = (fn, tuple(sorted((opts or {}).items())))
    if key not in self._exe:
      self._exe[key] = fn.lower(*args).compile(compiler_options=opts)
    return self._exe[key](*args)

  def inputs(self, pr, seed=0):
    """(w, x, dy, gacc, seed) for chunk `pr`, random."""
    mesh, ch, scfg = self.mesh, pr.ch, self.scfg
    sds = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))
    tree = lambda shapes, specs, dt: jax.tree.map(lambda s, sp: sds(s, dt, sp), shapes, specs, is_leaf=ds_ep._is_shape)
    act = sds(pr.act_aval.shape, pr.act_aval.dtype, pr.act_spec)
    x = sds(pr.tok_aval.shape, jnp.int32, pr.tok_spec) if ch.first else act
    av = (
        tree(ds_ep.chunk_shapes(scfg, ch), pr.cspec, scfg.model.dtype),
        x,
        act,
        tree(ds_ep.acc_shapes(scfg, ch), pr.gspec, scfg.acc_dtype),
    )
    return (*_rand(av, seed), jax.device_put(jnp.int32(3 + seed), NamedSharding(mesh, P())))

  def assert_same(self, got, want, what, near=False):
    """Every leaf equal -- or, `near`, within one bf16 rounding of its scale."""
    gl, _ = jax.tree_util.tree_flatten_with_path(got)
    wl = jax.tree.leaves(want)
    self.assertEqual(len(gl), len(wl), what)
    for (path, a), b in zip(gl, wl):
      a, b = np.asarray(a), np.asarray(b)
      msg = f"{what} {jax.tree_util.keystr(path)}"
      if near:
        a, b = a.astype(np.float32), b.astype(np.float32)
        self.assertLessEqual(np.max(np.abs(a - b)), 2.0**-8 * np.max(np.abs(b)), msg)
      else:
        np.testing.assert_array_equal(a, b, err_msg=msg)

  def stash_vs_remat(self, pr, policy, opts, near=False):
    """y, res, dx and every dW of the stash programs == the fixed ones'."""
    w, x, dy, g, seed = self.inputs(pr)
    what = f"{policy} chunk {pr.ch.index} layers {pr.ch.layers} opts {opts}:"
    y0, r0 = self.call(pr.fwd, w, x, seed, opts=opts)
    st = pr.build_stash(policy)
    y, r, s = self.call(st.fwd, w, x, seed, opts=opts)
    self.assert_same((y, r), (y0, r0), what + " forward")
    if pr.ch.first:  # the first microbatch's backward (chunk 0 donates no dy)
      want = self.call(pr.bwd, w, x, r0, dy, seed, opts=opts)
      got = self.call(st.bwd, w, x, r, s, dy, seed, opts=opts)
    else:  # an accumulating one
      want = self.call(pr.bwda, w, x, r0, dy.copy(), _copy(g), seed, opts=opts)
      got = self.call(st.bwda, w, x, r, s, dy.copy(), _copy(g), seed, opts=opts)
    self.assert_same(got, want, what + " backward", near)

  def check_stash(self, policy):
    # 'all' (diagnostics only) keeps even the plain vjp's own intermediates
    # -- the weight-gradient matmuls' inputs among them -- which today's
    # backward computes in-program; XLA compiles some of those matmuls with
    # another summation order, and a few gradient elements of chunk 0 and of
    # a two-layer chunk come out up to one bf16 rounding apart (0.93 of one
    # at the leaf's scale, measured on jax 0.11.1).  On 0.11.2, whose CPU
    # backend keeps some bf16 matmuls in bf16 (PipelineVsReference._port),
    # the one-layer middle chunk's do too (three leaves, 0.42 of one; bitwise
    # on 0.11.1).  So 'all''s backward is compared within one bf16 rounding
    # everywhere (test_stash_matches_remat_all, not bitwise); its y and res
    # are exact.
    near = policy == "all"
    # a first and a middle (two-layer) chunk, every bf16 rounding explicit ...
    for pr in (self.pr(0), self.pr(1, layers=(1, 2))):
      self.stash_vs_remat(pr, policy, self.EXACT, near)
    # ... and the pipeline's own one-layer chunks as compiled by default.  (A
    # two-layer chunk as compiled by default is not bitwise: see build_stash.)
    self.stash_vs_remat(self.pr(0), policy, None, near)
    self.stash_vs_remat(self.pr(1), policy, None, near)

  def test_stash_bytes_v6e_dims(self):
    """Stash bytes per device per (layer, microbatch) at the v6e config
    (PP2 x FSDP1 x EP2, 16 experts, mbs 2, seq 4096), from the eager
    eval_shape alone: nothing is compiled.  Measured (GiB, on top of the
    0.055 layer input every mode keeps): attn 0.713, attn_min 0.659, lite
    1.354, lite_y 2.229, all 4.973 -- the same for chunk 0, whose layer
    input (the embedding) is a residual already."""
    import argparse

    a = argparse.Namespace(
        vocab=32768,
        emb=7168,
        layers=4,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=16,
        topk=8,
        attn="tokamax_splash",
        sa_block=2048,
        sa_block_kv_compute=512,
        cost_estimate=5_000_000_000_000,
        seq=4096,
        weight_dtype="bfloat16",
    )
    cfg = dsp.model_config(a)
    tiling = (512, cfg.emb_dim, 512) * 3
    scfg = ds_ep.StageConfig(model=cfg, fsdp=1, ep=2, wi_tiling=tiling, wo_tiling=tiling)
    G = 2**30
    bounds = {
        "attn": (0.65, 0.80),
        "attn_min": (0.60, 0.75),
        "lite": (1.25, 1.45),
        "lite_y": (2.0, 2.4),
        "all": (4.8, 5.1),
    }
    got = {}
    for c in (0, 1):
      ch = ds_ep.ChunkSpec(index=c, layers=(c,), first=c == 0, last=False, stage=0)
      pr = ds_ep.ChunkPrograms(scfg, ch, self.mesh, mbs=2, seq=4096, denom=16.0 * 4096)
      for pol in bounds:
        got[c, pol] = pr.build_stash(pol).bytes / G
    print("stash GiB/device/(layer, microbatch) at v6e dims: " + ", ".join(f"{p} {got[1, p]:.3f}" for p in bounds))
    for pol, (lo, hi) in bounds.items():
      self.assertEqual(got[0, pol], got[1, pol], pol)
      self.assertTrue(lo <= got[1, pol] <= hi, f"{pol}: {got[1, pol]:.3f} GiB not in [{lo}, {hi}]")
    self.assertEqual(sorted(bounds, key=lambda p: got[1, p]), ["attn_min", "attn", "lite", "lite_y", "all"])

  def test_stash_recompute_set(self):
    """What each policy's backward still recomputes, read off the optimized
    (dead-code-eliminated) HLO: an op named under `rematted_computation` is
    a forward op re-run in the backward.  lite re-runs only the gmm_wo
    kernel of the experts (the unsort's weight gradient needs its output),
    lite_y none, and attn none of the five attention projections (wq_a,
    wq_b, wkv_a, wkv_b, out) -- only the gate and shared-expert matmuls.
    And none re-runs the splash forward kernel, as today's backward does:
    every policy keeps its residuals ('context')."""
    import collections
    import re

    pr = self.pr(1)
    w, x, dy, g, seed = self.inputs(pr)

    def rerun(hlo, marker):
      """(expert kernels, other dots, splash forward ops) under `marker`."""
      splash = sum(marker in n and "splash_mha_fwd" in n for n in re.findall(r'op_name="([^"]*)"', hlo))
      kern, dots = collections.Counter(), 0
      for m in re.finditer(r' (dot|while)\(.*op_name="([^"]*)"', hlo):
        name = m.group(2)
        if marker not in name:
          continue
        k = re.search(r"/(gmm_wi_0|gmm_wi_1|gmm_wo)/", name)
        if k:
          kern[k.group(1)] += 1
        elif m.group(1) == "dot" and "splash" not in name:
          dots += 1
      return set(kern), dots, splash

    remat = pr.bwda.lower(w, x, (), dy, g, seed).compile().as_text()
    # today's backward re-runs the whole forward, under jvp(...) and not transposed
    fwd_kern, fwd_dots, fwd_splash = rerun(re.sub(r'op_name="[^"]*transpose[^"]*"', 'op_name=""', remat), "jvp(")
    self.assertEqual(fwd_kern, {"gmm_wi_0", "gmm_wi_1", "gmm_wo"})
    self.assertGreater(fwd_splash, 0)
    got = {}
    for pol in ("attn", "attn_min", "lite", "lite_y", "all"):
      st = pr.build_stash(pol)
      hlo = st.bwda.lower(w, x, (), st.avals, dy, g, seed).compile().as_text()
      got[pol] = rerun(hlo, "rematted_computation")
    print(
        f"forward re-run in the backward (kernels, other dots, splash ops): remat {fwd_kern} {fwd_dots} "
        f"{fwd_splash}; " + "; ".join(f'{p} {k or "{}"} {d} {n}' for p, (k, d, n) in got.items())
    )
    for pol, (_, _, n) in got.items():
      self.assertEqual(n, 0, f"{pol} re-runs the splash forward")
    self.assertEqual(got["attn"][0], {"gmm_wi_0", "gmm_wi_1", "gmm_wo"})
    self.assertEqual(got["attn"][1], fwd_dots - 5)
    self.assertEqual(got["lite"][0], {"gmm_wo"})
    self.assertEqual(got["lite_y"][0], set())
    self.assertEqual(got["all"], (set(), 0, 0))
    self.assertLessEqual(got["lite"][1], got["attn"][1])

  def test_compose_matches_existing(self):
    """compose(F, G) == fwd_prefetch, compose(B, R) == bwd_reduce, and F&B
    pairs == the two programs run separately -- remat, stash and the fused
    last chunk -- all bit for bit (default compiler options).  An F&B pair
    returns its backward's outputs first, an A&B pair its A's
    (ds_ep.composed_order)."""
    pd = ds_ep.pdesc
    p0, p1, p2, p3 = self.pr(0), self.pr(1), self.pr(2), self.pr(3, last=True)
    ch1 = p1.ch
    # F + G: chunk 0's forward gathering half of chunk 1's leaves
    w0, t0, dy0, g0, s0 = self.inputs(p0, seed=0)
    grp = ds_ep.leaf_groups(ds_ep.chunk_shapes(self.scfg, ch1), 2)[0]
    sds = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(self.mesh, spec))
    storage = _rand(
        [
            sds(s, self.scfg.model.weight_dtype, sp)
            for s, sp in zip(
                jax.tree.leaves(ds_ep.chunk_shapes(self.scfg, ch1), is_leaf=ds_ep._is_shape),
                jax.tree.leaves(ds_ep.chunk_storage_specs(ch1), is_leaf=ds_ep._is_spec),
            )
        ],
        7,
    )
    st1 = [storage[i] for i in grp]
    want = p0.fwd_prefetch(ch1, grp)(w0, t0, s0, st1)
    prog = p0.program((pd("F"), pd("G", other=ch1, leaves=grp)))
    self.assertEqual(prog.__name__, f"fwd_c0_ag1p{grp[0]}")
    self.assert_same(prog(w0, t0, s0, st1), want, "F+G vs fwd_prefetch")
    # B + R: chunk 2's accumulating backward reduce-scattering all of chunk 1's gacc
    w2, x2, dy2, g2, s2 = self.inputs(p2, seed=2)
    _, _, _, go, _ = self.inputs(p1, seed=5)
    go = jax.tree.leaves(go)
    want = p2.bwd_reduce(ch1, accumulate=True)(w2, x2, (), dy2.copy(), _copy(g2), s2, _copy(go))
    # (`program` gives a remat B + R descriptor bwd_reduce itself, for its
    # byte-identical lowering; the composed parts give the same values)
    prog = ds_ep.compose(self.mesh, [p2.part("B", acc=True), p2.part("R", other=ch1)], "bwda_c2_rs1")
    self.assert_same(prog(w2, x2, (), dy2.copy(), _copy(g2), s2, _copy(go)), want, "B+R vs bwd_reduce")
    self.assertEqual(p2.program((pd("B", acc=True), pd("R", other=ch1))).__name__, "bwda_c2_rs1")
    # F(c2) & B(c0): an F&B pair on stage 0, against fwd + bwd
    y2, r2 = p2.fwd(w2, x2, s2)
    _, r0 = p0.fwd(w0, t0, s0)
    want = (*p0.bwd(w0, t0, r0, dy0, s0), y2, r2)
    pair = p0.program((pd("F", peer=p2), pd("B")))
    self.assert_same(pair(w2, x2, s2, w0, t0, r0, dy0, s0), want, "F(c2)&B(c0) vs fwd + bwd")
    # the same pair with chunk 0's backward in stash mode, accumulating
    st0 = p0.build_stash("lite")
    _, r0s, stash0 = st0.fwd(w0, t0, s0)
    want = (*st0.bwda(w0, t0, r0s, _copy(stash0), dy0, _copy(g0), s0), y2, r2)
    pair = p0.program((pd("F", peer=p2), pd("B", stash="lite", acc=True)))
    self.assert_same(pair(w2, x2, s2, w0, t0, r0s, stash0, dy0, _copy(g0), s0), want, "F(c2)&B_lite(c0) vs fwd + bwda_s")
    # A(c3) & B(c1): the fused last chunk paired with a backward, both accumulating
    w3, x3, _, g3, s3 = self.inputs(p3, seed=3)
    w1, x1, dy1, g1, s1 = self.inputs(p1, seed=1)
    tgt = _rand(sds(p3.tok_aval.shape, jnp.int32, p3.tok_spec), 9)
    lacc = jax.device_put(jnp.full((2,), 0.5, jnp.float32), NamedSharding(self.mesh, p3.loss_spec))
    want = (*p3.fba(w3, x3.copy(), tgt, _copy(g3), lacc.copy(), s3), *p1.bwda(w1, x1, (), dy1.copy(), _copy(g1), s1))
    pair = p3.program((pd("A", acc=True), pd("B", acc=True, peer=p1)))
    self.assert_same(pair(w3, x3, tgt, g3, lacc, s3, w1, x1, (), dy1, g1, s1), want, "A(c3)&B(c1) vs fba + bwda")

  def test_compose_host_variants(self):
    """The parts the driver's hosts compose, bit for bit (default compiler
    options): rs_self == bwda_reduce_self, self_leaves == fwd_self_prefetch,
    B + R without accumulating == bwd_reduce, and in stash mode F_s + G,
    B_s + R, F_s with self_leaves and B_s with rs_self == the stash programs
    with the gather / reduce-scatter's own outputs (from the fixed programs
    above; for rs_self, a one-layer stash backward is today's bit for bit)."""
    pd = ds_ep.pdesc
    mesh, scfg = self.mesh, self.scfg
    p0, p1, p2 = self.pr(0), self.pr(1), self.pr(2)
    sds = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))
    sqs = [jax.device_put(jnp.float32(1.5 + i), NamedSharding(mesh, P())) for i in range(2)]
    _, _, _, go2, _ = self.inputs(p2, seed=5)
    go2 = jax.tree.leaves(go2)
    for pr in (p0, p1):
      ch, c = pr.ch, pr.ch.index
      w, x, dy, g, seed = self.inputs(pr, seed=c)
      _, res = pr.fwd(w, x, seed)
      st = pr.build_stash("lite")
      _, res_s, stash = st.fwd(w, x, seed)
      # the chunk's last backward with its own reduce-scatter
      want = pr.bwda_reduce_self(2)(w, x, res, dy.copy(), _copy(g), seed, list(sqs))
      prog = pr.program((pd("B", acc=True, rs_self=2),))
      self.assert_same(prog(w, x, res, dy.copy(), _copy(g), seed, list(sqs)), want, f"c{c} B rs_self")
      prog = pr.program((pd("B", stash="lite", acc=True, rs_self=2),))
      self.assert_same(
          prog(w, x, res_s, _copy(stash), dy.copy(), _copy(g), seed, list(sqs)),
          want,
          f"c{c} B_lite rs_self vs bwda_reduce_self",
      )
      # a non-accumulating backward reduce-scattering chunk 2's gacc
      want = pr.bwd_reduce(p2.ch, accumulate=False)(w, x, res, dy.copy(), seed, _copy(go2))
      prog = ds_ep.compose(mesh, [pr.part("B"), pr.part("R", other=p2.ch)], f"bwd_c{c}_rs2")
      self.assert_same(prog(w, x, res, dy.copy(), seed, _copy(go2)), want, f"c{c} B+R vs bwd_reduce")
      # the first forward gathering its own routed experts
      specs = jax.tree.leaves(ds_ep.chunk_storage_specs(ch), is_leaf=ds_ep._is_spec)
      shapes = jax.tree.leaves(ds_ep.chunk_shapes(scfg, ch), is_leaf=ds_ep._is_shape)
      idx = tuple(i for i, sp in enumerate(specs) if len(sp) > 1 and sp[1] == "expert")
      storage = _rand([sds(shapes[i], scfg.model.weight_dtype, specs[i]) for i in idx], 11)
      wl, tdef = jax.tree.flatten(w)
      rest = [wl[i] for i in range(len(wl)) if i not in idx]
      want = pr.fwd_self_prefetch(idx)(rest, x, seed, storage)
      prog = pr.program((pd("F", self_leaves=idx),))
      self.assert_same(prog(rest, x, seed, storage), want, f"c{c} F self_leaves vs fwd_self_prefetch")
      full = list(wl)
      for i, v in zip(idx, want[-1]):
        full[i] = v
      want_s = (*st.fwd(jax.tree.unflatten(tdef, full), x, seed), want[-1])
      prog = pr.program((pd("F", stash="lite", self_leaves=idx),))
      self.assert_same(prog(rest, x, seed, storage), want_s, f"c{c} F_lite self_leaves")
    # F_s + G, B_s + R: chunk 0's stash forward gathering half of chunk 1's
    # leaves, and chunk 1's stash backward reduce-scattering chunk 2's gacc
    ch1 = p1.ch
    w0, t0, _, _, s0 = self.inputs(p0, seed=0)
    grp = ds_ep.leaf_groups(ds_ep.chunk_shapes(scfg, ch1), 2)[0]
    storage = _rand(
        [
            sds(s_, scfg.model.weight_dtype, sp)
            for s_, sp in zip(
                jax.tree.leaves(ds_ep.chunk_shapes(scfg, ch1), is_leaf=ds_ep._is_shape),
                jax.tree.leaves(ds_ep.chunk_storage_specs(ch1), is_leaf=ds_ep._is_spec),
            )
        ],
        7,
    )
    st1 = [storage[i] for i in grp]
    want = (*p0.build_stash("lite").fwd(w0, t0, s0), p0.fwd_prefetch(ch1, grp)(w0, t0, s0, st1)[-1])
    prog = p0.program((pd("F", stash="lite"), pd("G", other=ch1, leaves=grp)))
    self.assertEqual(prog.__name__, f"fwd_c0_st_ag1p{grp[0]}")
    self.assert_same(prog(w0, t0, s0, st1), want, "F_lite+G")
    w1, x1, dy1, g1, s1 = self.inputs(p1, seed=1)
    _, res1, stash1 = p1.build_stash("lite").fwd(w1, x1, s1)
    want = (
        *p1.build_stash("lite").bwda(w1, x1, res1, _copy(stash1), dy1.copy(), _copy(g1), s1),
        *p1.bwd_reduce(p2.ch, accumulate=True)(w1, x1, (), dy1.copy(), _copy(g1), s1, _copy(go2))[-2:],
    )
    prog = p1.program((pd("B", stash="lite", acc=True), pd("R", other=p2.ch)))
    self.assertEqual(prog.__name__, "bwda_c1_st_rs2")
    self.assert_same(prog(w1, x1, res1, _copy(stash1), dy1.copy(), _copy(g1), s1, _copy(go2)), want, "B_lite+R")

  def test_stash_bitwise_vs_remat_attn(self):
    """y, res, dx and every dW of the 'attn' stash programs == today's."""
    self.check_stash("attn")

  def test_stash_bitwise_vs_remat_attn_min(self):
    """y, res, dx and every dW of the 'attn_min' stash programs == today's."""
    self.check_stash("attn_min")

  def test_stash_bitwise_vs_remat_lite(self):
    """y, res, dx and every dW of the 'lite' stash programs == today's."""
    self.check_stash("lite")

  def test_stash_bitwise_vs_remat_lite_y(self):
    """y, res, dx and every dW of the 'lite_y' stash programs == today's."""
    self.check_stash("lite_y")

  def test_stash_matches_remat_all(self):
    """y, res, dx and every dW of the 'all' stash programs == today's (the backward within one bf16 rounding: see
    check_stash)."""
    self.check_stash("all")


# ------------------------------------------- the split backward (I and W)
# W's XLA flops over the backward it is cut from (bwda's): a remat backward recomputes the forward, so its weight
# gradients' matmuls are about a third of it (0.30-0.43 seen, a Qwen3.5-family chunk's the most); a saved-residual
# one does not, so about half (0.44-0.50).  At least SPLIT_W_SHARE_MIN: W holds the weight gradients' matmuls, which
# I would hold instead if the cut moved them (W then little but the accumulate).  I's and W's together: no equation
# runs twice, so the backward's flops and the little the cut costs XLA's fusion.
SPLIT_W_SHARE = 0.45
SPLIT_W_SHARE_SAVED = 0.55
SPLIT_W_SHARE_MIN = 0.2
SPLIT_FLOPS_SUM = 1.01
# The primitives of a collective: none in a W under grad_reduce 'step' (the input gradient's EP collectives are I's,
# the FSDP ones outside the backward), the microbatch's reduce-scatter there under 'microbatch'.
COLLECTIVES = frozenset({"all_gather", "all_to_all", "ppermute", "psum", "psum_scatter", "reduce_scatter"})


def _shards(a):
  """(device id, the shard as an array) per addressable shard of `a`, in device order."""
  return [(s.device.id, np.asarray(s.data)) for s in sorted(a.addressable_shards, key=lambda s: s.device.id)]


def shards_apart(a, b):
  """None if `a` is `b` bit for bit on every device (layout, shape and dtype too), else max |a - b| over max |b|
  across the shards (inf for another layout; 0.0 for bits apart at equal values: -0.0, a NaN's payload)."""
  sa, sb = _shards(a), _shards(b)
  if [(d, x.shape, x.dtype) for d, x in sa] != [(d, y.shape, y.dtype) for d, y in sb]:
    return float("inf")
  if all(x.tobytes() == y.tobytes() for (_, x), (_, y) in zip(sa, sb)):
    return None
  diff = max(
      float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)), initial=0.0)) for (_, x), (_, y) in zip(sa, sb)
  )
  scale = max(float(np.max(np.abs(y.astype(np.float64)), initial=0.0)) for _, y in sb)
  return diff / scale if scale else (0.0 if not diff else float("inf"))


def split_inputs(pr, seed=0):
  """Random arguments for chunk programs `pr`'s backward cut: (w, x, res, dy, seed); the accumulator bwda takes --
  bwd's on other random arguments (an accumulator typed unreduced, shard_mode 'explicit', takes no device_put); two
  partial sums of squares, rs_self's."""
  ns = NamedSharding(pr.mesh, P())
  av = pr._part_avals("B")
  w, x, res, dy, _ = _rand(av, seed)
  w1, x1, res1, dy1, _ = _rand(av, seed + 1)
  gacc = pr.bwd(w1, x1, res1, dy1, jax.device_put(jnp.int32(5 + seed), ns))[-1]
  sqs = [jax.device_put(jnp.float32(1.5 + i), ns) for i in range(2)]
  return (w, x, res, dy, jax.device_put(jnp.int32(3 + seed), ns)), gacc, sqs


def _flops(compiled):
  ca = compiled.cost_analysis()
  return float((ca[0] if isinstance(ca, (list, tuple)) else ca).get("flops", 0.0))


# An fp32 leaf of a split chunk against its fused program on CPU: bit for bit on the hosts these checks were written
# on, but MaxText's CPU CI runners gave fp32 chunks on XLA's ops -- the experts' ragged_dot weight gradients -- up to
# 1.8e-7 of the leaf's largest value apart, with the YNNPACK fusions off in both programs (jaxlib 0.12.0.1).  So on
# CPU an fp32 leaf is held to 4 float32 ulps of its largest magnitude (check_split's and check_split_last's default).
CPU_FP32_SPLIT_TOL = 2.0**-21


def cpu_fp32_split_tol(scfg):
  """The default `tol` of check_split / check_split_last: CPU_FP32_SPLIT_TOL for every leaf of an fp32 `scfg` on CPU,
  else None (bit for bit)."""
  if jax.default_backend() == "cpu" and jnp.dtype(scfg.model.dtype) == jnp.float32:
    return lambda name, mode: CPU_FP32_SPLIT_TOL
  return None


def check_split(t, scfg, ch, mbs=4, tol=None):
  """Chunk `ch`'s backward cut into I and W (ds_ep.ChunkPrograms.build_split) against the programs it is cut from,
  on one stage mesh of scfg's fsdp x ep: I then W against bwd (the chunk's first gradient write), against bwda
  (accumulating), and -- under grad_reduce 'step', where the pipeline folds a chunk's reduce-scatter into its last
  gradient write -- W with rs_self against the B part with rs_self (bwda_rs).  dx and every output leaf bit for bit,
  under XLA's default options and with its excess precision off (testing.NO_EXCESS_PRECISION), XLA:CPU's YNNPACK
  fusions off in both (testing.ynn_off): each program compiled with the options its fused counterpart is.
  `tol(name, mode) -> bound`: a leaf held to max |diff| over its max |value| <= bound instead (0: bit for bit) in
  `mode` ('default' or 'no excess precision'; pathways_test.tpu_split_tol: a TPU's contract); by default
  cpu_fp32_split_tol's.

  One trace of the backward and one I for both W's, the accumulating W adding the accumulator to the first write's
  (Split.w_adds) -- but under in-place dW, which traces bwda's backward as well (and shares I where its cut's is the
  same program, Split.i_acc).  And W is the weight gradients: its XLA flops at least SPLIT_W_SHARE_MIN and at most
  SPLIT_W_SHARE of bwda's (SPLIT_W_SHARE_SAVED under saved residuals), I's and W's together at most SPLIT_FLOPS_SUM
  of it; under grad_reduce 'step' W holds no collective, under 'microbatch' the reduce-scatter.  -> (the
  ChunkPrograms, {acc: Split}, {'W/B': ., '(I+W)/B': .})."""
  tol = cpu_fp32_split_tol(scfg) if tol is None else tol
  pd = ds_ep.pdesc
  mesh = dsp.typed_meshes(dsp.stage_meshes(1, fsdp=scfg.fsdp, ep=scfg.ep), scfg.shard_mode)[0]
  pr = ds_ep.ChunkPrograms(scfg, ch, mesh, mbs=mbs, seq=256, denom=float(8 * 256))
  with mock.patch.object(stage_ad, "trace_body", wraps=stage_ad.trace_body) as traced:
    sps = {acc: pr.build_split(acc) for acc in (False, True)}
  t.assertEqual(
      (traced.call_count, sps[True].w_adds, sps[False].i_acc), (2 if pr.inplace_dw else 1, not pr.inplace_dw, False)
  )
  if not pr.inplace_dw:
    t.assertFalse(sps[True].i_acc)
    t.assertIs(pr.part("I", acc=True), pr.part("I"))
  args, gacc, sqs = split_inputs(pr)

  def path_names(avals):
    return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(avals)[0]]

  acc_names = path_names(pr._part_avals("B", acc=True)[4])  # the accumulator's leaves
  grad_names = path_names(args[0])  # the weights': the reduce-scatter's grads are their storage layout's
  step = scfg.grad_reduce == "step"
  cases = [
      ("bwd", pr.bwd, args, False, (), ["dx"] + acc_names),
      ("bwda", pr.bwda, args[:4] + (gacc,) + args[4:], True, (gacc,), ["dx"] + acc_names),
  ]
  if step:
    rs = pr.program((pd("B", acc=True, rs_self=len(sqs)),))
    cases.append(
        ("bwda_rs", rs, args[:4] + (gacc,) + args[4:] + (sqs,), True, (gacc, sqs), ["dx"] + grad_names + ["sumsq"])
    )

  def copy(tree):  # every program donates something
    return jax.tree.map(lambda a: a.copy(), tree)

  flops = {}
  for mode, opts in (
      ("default", testing.ynn_off()),
      ("no excess precision", {**testing.ynn_off(), **testing.NO_EXCESS_PRECISION}),
  ):
    i_out = {}  # per I program: one, but where in-place dW cut the accumulating W its own
    for case, fused, fargs, acc, wtail, names in cases:
      fc = fused.lower(*fargs).compile(compiler_options=opts)
      want = jax.tree.leaves(fc(*copy(fargs)))
      ia = sps[acc].i_acc
      if ia not in i_out:
        ic = pr.program((pd("I", acc=ia),)).lower(*args).compile(compiler_options=opts)
        i_out[ia] = (ic, ic(*copy(args)))
      ic, (dx, cross) = i_out[ia]
      wdesc = (pd("W", acc=acc, rs_self=len(sqs) if case == "bwda_rs" else None),)
      wargs = (cross, sps[acc].kept_of(args)) + wtail
      wc = pr.program(wdesc).lower(*wargs).compile(compiler_options=opts)
      got = [dx] + jax.tree.leaves(wc(*copy(wargs)))
      t.assertEqual(len(got), len(want), f"{case}: outputs")
      for name, a, b in zip(names, got, want):
        rel, bound = shards_apart(a, b), (tol(name, mode) if tol else 0.0)
        if rel is not None:
          t.assertLessEqual(rel, bound, f"{case}, {mode}: {name} not bit for bit (max |diff| / max |value| {rel:.3g})")
      if case == "bwda" and mode == "default":
        fb, fi, fw = _flops(fc), _flops(ic), _flops(wc)
        flops = {"W/B": fw / fb, "(I+W)/B": (fi + fw) / fb}
  share = SPLIT_W_SHARE_SAVED if scfg.save_residuals else SPLIT_W_SHARE
  t.assertLessEqual(flops["W/B"], share, flops)
  t.assertGreaterEqual(flops["W/B"], SPLIT_W_SHARE_MIN, flops)
  t.assertLessEqual(flops["(I+W)/B"], SPLIT_FLOPS_SUM, flops)
  for acc, sp in sps.items():
    coll = {e.primitive.name for e in sp.w_jaxpr.eqns} & COLLECTIVES
    if step:
      t.assertFalse(coll, f"acc {acc}: W holds collectives {coll}")
    else:
      t.assertTrue(coll, f"acc {acc}: W holds no reduce-scatter under grad_reduce {scfg.grad_reduce!r}")
    # I donates what the B does (dy; the saved set under saved residuals) but an argument W reads
    groups = {g for g, _ in sp.kept}
    t.assertLessEqual(groups, {"w", "x", "res", "dy", "seed"})
    donate = tuple(d for d, g in ((2, "res"), (3, "dy")) if (g == "dy" or scfg.save_residuals) and g not in groups)
    t.assertEqual((sp.i_donate, pr.part("I", acc=acc).donate), (donate, donate), f"acc {acc}: kept {sp.kept}")
  print(
      f"split chunk {ch.index} ({scfg.gmm_impl}, layers {ch.layers}, dense {ch.dense}): W/B {flops['W/B']:.3f}, "
      f"(I+W)/B {flops['(I+W)/B']:.4f}; {sps[True].stats}; kept {sps[True].kept}; I programs {len(i_out)}"
  )
  return pr, sps, flops


@testing.needs_devices(4)
class SplitBackward(unittest.TestCase):
  """A middle chunk's backward cut into I and W for the zero-bubble schedules (ds_ep.ChunkPrograms.build_split,
  parts 'I' and 'W') against the programs it is cut from, bit for bit (check_split): I then W against bwd, bwda and
  the B part with the chunk's own reduce-scatter (rs_self), under XLA's default options and with excess precision
  off, on one FSDP 2 x EP 2 stage mesh -- and W's XLA flops the weight gradients' share; an fp32 chunk's leaves on CPU
  within 4 float32 ulps instead (CPU_FP32_SPLIT_TOL).  One configuration a test, for each the split supports,
  scheduled_only (a Qwen3.5-family chunk's in ds_ep_qwen_test.QwenSplitBackward); on every PR what build_split and the
  parts refuse (nothing traced: test_refusals) and that tolerance (test_cpu_fp32_tol).  Not here, because they change
  no middle chunk's backward: MTP, which rides the last chunk's fused program (unsplit), and offloaded residuals,
  which the step loop moves.

  Bit for bit: I and W run the fused backward's equations, each once, on the same operands, so they reproduce its
  values wherever XLA compiles each operation as it does in the fused program.  The one place seen where it does
  not: XLA's algebraic simplifier rewrites one of two divisions by the same broadcast -- top-k routing's weight
  normalization, forward and backward -- into a multiplication by the reciprocal, and which one depends on the order
  it visits the program in, so I can rewrite the other one than the fused program does (XLA:CPU, jax 0.12.0.1).  A
  value one fp32 ulp apart; under float32_gate_logits in a bf16 model on megablox it reaches the router's fp32
  gradient, and where it flips a bf16 rounding, dx and the layer's other gradients (test_float32_gate_logits: the
  measured bounds).  On XLA's ops (ragged_dot) and with the router in bf16 the two programs choose alike (bit for
  bit, below, on the hosts these were written on; MaxText's CPU CI runners gave fp32 ones 1-2 ulps apart)."""

  @staticmethod
  def chunk(scfg, layers=(1,), dense=0, per_layer=False, routed_bias=False):
    kinds = ds_ep.layer_kinds(scfg.model, layers, per_layer=True) if per_layer else ()
    dkinds = ds_ep.dense_layer_kinds(dense, per_layer=True) if per_layer and dense else ()
    return ds_ep.ChunkSpec(
        1, tuple(layers), False, False, 0, dense=dense, routed_bias=routed_bias, kinds=kinds, dense_kinds=dkinds
    )

  def test_cpu_fp32_tol(self):
    """check_split's default tol: an fp32 chunk's leaves on CPU within 4 float32 ulps of their largest magnitude
    (MaxText's CI runners' fp32 XLA ops), a bf16 chunk's bit for bit."""
    self.assertEqual(cpu_fp32_split_tol(tiny_xla(layers=3, fsdp=2, ep=2))("dx", "default"), 2.0**-21)
    self.assertIsNone(cpu_fp32_split_tol(tiny(layers=3, fsdp=2, ep=2)))

  @pytest.mark.scheduled_only
  def test_megablox(self):
    """The benchmark's kernels: MLA on tokamax splash, the routed experts on megablox and the ragged sort, the
    shared expert, bf16, one MoE layer.  And what the parts are: their names, one I for both W's, W reads dy alone
    besides the crossing set (so I donates nothing), I holds every EP collective, the crossing set's layout and
    bytes."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    pr, sps, _ = check_split(self, scfg, self.chunk(scfg))
    pd = ds_ep.pdesc
    names = [pr.program(d).__name__ for d in ((pd("I"),), (pd("W"),), (pd("W", acc=True),))]
    names.append(pr.program((pd("W", acc=True, rs_self=2),)).__name__)
    self.assertEqual(names, ["bwdi_c1", "bwdw_c1", "bwdwa_c1", "bwdwa_c1_rsself"])
    self.assertEqual([pr.part("W", acc=a).donate for a in (False, True)], [(), (2,)])
    self.assertEqual([(sp.i_acc, sp.w_adds) for sp in sps.values()], [(False, False), (False, True)])
    n_dev = scfg.fsdp * scfg.ep
    for acc, sp in sps.items():
      self.assertEqual(sp.acc, acc)
      self.assertEqual(sp.kept, (("dy", 0),))
      self.assertEqual((sp.i_donate, pr.part("I", acc=acc).donate), ((), ()))
      self.assertEqual(sp.kept_bytes, math.prod(pr.act_aval.shape) * 2 // n_dev)
      self.assertEqual(sp.bytes, sum(math.prod(a.shape) * a.dtype.itemsize for a in sp.crossing_local))
      self.assertEqual(sp.bytes, sp.stats["crossing_bytes"])
      for a, spec, g in zip(sp.crossing_local, sp.crossing_specs, sp.crossing_avals):
        self.assertEqual(spec, P(ds_ep.FE, *([None] * len(a.shape))))
        self.assertEqual((g.shape, g.dtype, g.sharding.spec), ((n_dev, *a.shape), a.dtype, spec))
      i_coll = {e.primitive.name for e in sp.i_jaxpr.eqns} & COLLECTIVES
      self.assertEqual(i_coll, {"all_gather", "reduce_scatter"})  # the ring of experts' gather and scatter
    self.assertIs(sps[True].i_jaxpr, sps[False].i_jaxpr)  # one I
    self.assertIs(pr.build_split(True), sps[True])  # cached

  def test_refusals(self):
    """build_split and the parts refuse what has no split -- chunk 0, the last chunk, a chunk without layers, a
    batch-split chunk on Lineage's layers -- and a stash mode, rs_self on an I or a first W, a gather's or a
    forward's arguments, before anything is traced."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    mesh = dsp.stage_meshes(1, fsdp=2, ep=2)[0]

    def mk(sc, ch, mbs=4):
      return ds_ep.ChunkPrograms(sc, ch, dsp.typed_meshes([mesh], sc.shard_mode)[0], mbs, 256, 2048.0)

    for ch, want in (
        (ds_ep.ChunkSpec(0, (0,), True, False, 0), "chunk 0 "),
        (ds_ep.ChunkSpec(2, (2,), False, True, 0), "the last chunk "),
        (ds_ep.ChunkSpec(1, (), False, False, 0), "a chunk without layers"),
    ):
      for acc in (False, True):
        with self.subTest(chunk=ch.index, acc=acc), self.assertRaisesRegex(ValueError, "no split backward .*" + want):
          mk(scfg, ch).build_split(acc)
      with self.assertRaisesRegex(ValueError, "no split backward"):
        mk(scfg, ch).part("I")
    with lineage_stand_in():
      lin = mk(lineage_scfg(layers=3, fsdp=2, ep=2), ds_ep.ChunkSpec(1, (1,), False, False, 0), mbs=8)
      self.assertTrue(lin.lineage)
      with self.assertRaisesRegex(ValueError, "no split backward .*Lineage's layers"):
        lin.build_split()
    pr = mk(scfg, ds_ep.ChunkSpec(1, (1,), False, False, 0))
    for kw, want in (
        ({"kind": "I", "stash": "lite"}, "cuts the recompute backward, not one in stash mode 'lite'"),
        ({"kind": "W", "acc": True, "stash": "lite"}, "stash mode"),
        ({"kind": "I", "acc": True, "rs_self": 2}, "rs_self folds into an accumulating W .*not an I"),
        ({"kind": "W", "rs_self": 2}, "rs_self folds into an accumulating W .*not a first W"),
        ({"kind": "W", "acc": True, "other": pr.ch}, r"takes acc \(and a W's rs_self\) only, not \['other'\]"),
        ({"kind": "I", "self_leaves": (0,)}, r"not \['self_leaves'\]"),
        ({"kind": "I", "pin": (("hn", 0, 1, 0),)}, "prefetch pins tie points of a forward"),
    ):
      kind = kw.pop("kind")
      with self.subTest(kind=kind, **{k: str(v) for k, v in kw.items()}), self.assertRaisesRegex(ValueError, want):
        pr.part(kind, **kw)
    self.assertFalse(pr._splits)  # nothing was built
    with self.assertRaisesRegex(ValueError, "no abstract arguments for part 'A' of chunk 1"):
      pr._part_avals("A")

  @pytest.mark.scheduled_only
  def test_two_moe_layers(self):
    """A chunk of two MoE layers (stacked), as compiled by default too -- which a stash backward of two layers is
    not (ds_ep.ChunkPrograms.build_stash)."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    check_split(self, scfg, self.chunk(scfg, (1, 2)))

  @pytest.mark.scheduled_only
  def test_xla_kernels(self):
    """XLA's ops: the experts on ragged_dot after the argsort dispatch, in bf16 and (tiny_xla) in fp32."""
    for scfg in (tiny(layers=3, fsdp=2, ep=2, **XLA_KERNELS), tiny_xla(layers=3, fsdp=2, ep=2)):
      with self.subTest(dtype=jnp.dtype(scfg.model.dtype).name):
        check_split(self, scfg, self.chunk(scfg))

  @pytest.mark.scheduled_only
  def test_tokamax(self):
    """gmm_impl 'tokamax' (its XLA backend), dW written apart and added (inplace_dw off)."""
    scfg = tiny(layers=3, fsdp=2, ep=2, gmm_impl="tokamax", tokamax_gmm="xla", use_ragged_sort=False, inplace_dw=False)
    pr, _, _ = check_split(self, scfg, self.chunk(scfg))
    self.assertFalse(pr.inplace_dw)

  @pytest.mark.scheduled_only
  def test_tokamax_inplace_dw(self):
    """gmm_impl 'tokamax' accumulating its routed experts' dW into the accumulator inside the dW kernel: into an
    fp32 one (inplace_dw_any_dtype: the starts are primals of the backward's VJP, dw32) and into one of the compute
    dtype.  The accumulating W is cut from bwda's own trace, its kernels taking the accumulator's blocks; I reads
    none of it, and is the first write's (one I program)."""
    tk = {"gmm_impl": "tokamax", "tokamax_gmm": "xla", "use_ragged_sort": False, "expert_weights": "sliced"}
    for name, kw in (("fp32", {"inplace_dw_any_dtype": True}), ("bf16", {"acc_dtype": jnp.bfloat16})):
      with self.subTest(acc=name):
        scfg = tiny(layers=3, fsdp=2, ep=2, **tk, **kw)
        pr, sps, _ = check_split(self, scfg, self.chunk(scfg))
        self.assertTrue(pr.inplace_dw)
        self.assertEqual((sps[True].w_adds, sps[True].i_acc), (False, False))
        self.assertTrue(any(not isinstance(s, stage_ad.Const) and s[0] == "gacc" for s in sps[True].w_src))

  @pytest.mark.scheduled_only
  def test_dense_layers(self):
    """A dense layer, then a MoE layer, in one chunk."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(first_num_dense_layers=1))
    check_split(self, scfg, self.chunk(scfg, dense=1))

  @pytest.mark.scheduled_only
  def test_routed_bias_aux_loss(self):
    """DeepSeek-V3's routing: top-k among the best groups on the scores plus the routed bias, updated from the
    expert counts (which ride the bias's gradient slot), the sequence-wise aux loss -- in fp32 (tiny_xla, the router
    in fp32 too: float32_gate_logits) and in bf16 on megablox (the router in bf16)."""
    feats = {
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "load_balance_loss_weight": 0.01,
        "moe_use_megatron_seq_aux_loss": True,
        "n_routing_groups": 4,
        "topk_routing_group": 2,
    }
    for scfg, f32 in ((tiny_xla(layers=3, fsdp=2, ep=2), True), (tiny(layers=3, fsdp=2, ep=2), False)):
      scfg = dataclasses.replace(scfg, random_routing=False, model=scfg.model.replace(**feats, float32_gate_logits=f32))
      with self.subTest(dtype=jnp.dtype(scfg.model.dtype).name):
        pr, _, _ = check_split(self, scfg, self.chunk(scfg, routed_bias=True))
        self.assertTrue(pr.counting and pr.aux_seqs)

  @pytest.mark.scheduled_only
  def test_float32_gate_logits(self):
    """Top-k routing with the router in fp32 (float32_gate_logits) in a bf16 model on megablox: not bit for bit
    (the class docstring) -- I computes one of the two divisions of the routing weights' normalization as a
    multiplication by the reciprocal where the fused backward computes the other so.  The router kernel's fp32
    gradient within 2^-20 of its scale (seen: 2.5e-7, 16 seeds, both XLA modes), and where that flips a bf16 rounding
    upstream -- about half the seeds -- dx and the layer's other gradients within 2^-6 of theirs (seen: 6.8e-3)."""
    scfg = tiny(layers=3, fsdp=2, ep=2, random_routing=False)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(float32_gate_logits=True))
    check_split(
        self, scfg, self.chunk(scfg), tol=lambda name, mode: 2.0**-20 if "['gate']['kernel']" in name else 2.0**-6
    )

  @pytest.mark.scheduled_only
  def test_save_residuals(self):
    """Saved residuals (per-layer trees): the B evaluates the VJP of the saved set, no recompute -- so W reads saved
    residuals too, which I then does not donate -- on megablox and on XLA's ops in fp32."""
    for scfg in (
        tiny(layers=3, fsdp=2, ep=2, save_residuals=True, layer_tree="per_layer"),
        tiny_xla(layers=3, fsdp=2, ep=2, save_residuals=True, layer_tree="per_layer"),
    ):
      with self.subTest(gmm_impl=scfg.gmm_impl):
        _, sps, _ = check_split(self, scfg, self.chunk(scfg, per_layer=True))
        for sp in sps.values():
          self.assertEqual({g for g, _ in sp.kept}, {"res"})
          self.assertEqual(sp.i_donate, (3,))  # dy, not the saved set W reads
          self.assertGreater(sp.kept_bytes, 0)

  @pytest.mark.scheduled_only
  def test_save_residuals_inplace_dw(self):
    """Saved residuals with tokamax's in-place dW into an fp32 accumulator (per-layer trees)."""
    scfg = tiny(
        layers=3,
        fsdp=2,
        ep=2,
        save_residuals=True,
        layer_tree="per_layer",
        gmm_impl="tokamax",
        tokamax_gmm="xla",
        use_ragged_sort=False,
        inplace_dw_any_dtype=True,
    )
    pr, _, _ = check_split(self, scfg, self.chunk(scfg, per_layer=True))
    self.assertTrue(pr.inplace_dw)

  @pytest.mark.scheduled_only
  def test_batch_split(self):
    """Native batch split (split_layers): two MoE layers, each device's rows as two halves in phases closed by
    optimization barriers -- every one in I -- and mb0's dW kernels continuing mb1's (split_dw 'acc')."""
    scfg = tiny(layers=3, fsdp=2, ep=2, **NATIVE_SPLIT)
    _, sps, _ = check_split(self, scfg, self.chunk(scfg, (1, 2)), mbs=8)
    for sp in sps.values():
      self.assertEqual(sum(e.primitive.name == "optimization_barrier" for e in sp.w_jaxpr.eqns), 0)
      self.assertGreater(sum(e.primitive.name == "optimization_barrier" for e in sp.i_jaxpr.eqns), 0)

  @pytest.mark.scheduled_only
  def test_batch_split_dense(self):
    """Native batch split of a chunk with a dense layer before its MoE layer."""
    scfg = tiny(layers=3, fsdp=2, ep=2, **NATIVE_SPLIT)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(first_num_dense_layers=1))
    check_split(self, scfg, self.chunk(scfg, dense=1), mbs=8)

  @pytest.mark.scheduled_only
  def test_grad_reduce_microbatch(self):
    """grad_reduce 'microbatch': the reduce-scatter in new_acc / add_acc, so in W (no rs_self: nothing is left to
    reduce at the step's end)."""
    scfg = tiny(layers=3, fsdp=2, ep=2, grad_reduce="microbatch")
    check_split(self, scfg, self.chunk(scfg))

  @pytest.mark.scheduled_only
  def test_explicit_sharding(self):
    """shard_mode 'explicit': the weights typed reduced, the accumulator unreduced, at the parts' boundaries."""
    scfg = tiny(layers=3, fsdp=2, ep=2, shard_mode="explicit")
    pr, _, _ = check_split(self, scfg, self.chunk(scfg))
    self.assertEqual(pr.part("W", acc=True).in_specs[2], pr.gspec)

  @pytest.mark.scheduled_only
  def test_context_parallel(self):
    """cp 2: attention collects the other rows' kv latents in I."""
    scfg = tiny_xla(layers=3, fsdp=2, ep=2, cp=2)
    check_split(self, scfg, self.chunk(scfg))

  @pytest.mark.scheduled_only
  def test_ppermute_ep(self):
    """ep_collectives 'ppermute': the ring of experts as collective-permutes, every one in I."""
    scfg = tiny(layers=3, fsdp=2, ep=2, ep_collectives="ppermute")
    _, sps, _ = check_split(self, scfg, self.chunk(scfg))
    self.assertEqual({e.primitive.name for e in sps[True].i_jaxpr.eqns} & COLLECTIVES, {"ppermute"})

  @pytest.mark.scheduled_only
  def test_sliced_experts(self):
    """expert_weights 'sliced': megablox reads each layer's experts from a slice of the stack (two MoE layers)."""
    scfg = tiny(layers=3, fsdp=2, ep=2, expert_weights="sliced")
    check_split(self, scfg, self.chunk(scfg, (1, 2)))

  @pytest.mark.scheduled_only
  def test_per_layer_tree(self):
    """layer_tree 'per_layer': a dense and two MoE layers, one subtree each."""
    scfg = tiny(layers=3, fsdp=2, ep=2, layer_tree="per_layer")
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(first_num_dense_layers=1))
    check_split(self, scfg, self.chunk(scfg, (1, 2), dense=1, per_layer=True))


# W's share of a split last chunk's fused program (check_split_last): its layers' weight gradients, 0.10 (beside an
# MTP layer's, which stay in AI) to 0.40 (a Qwen3.5-family chunk's) of A's XLA flops at TINY widths.
SPLIT_LAST_W_SHARE = (0.05, 0.45)


def last_inputs(pr, seed=0):
  """Random arguments for the fused last chunk's programs `pr`: (w, x, tgt[, tok], seed); the accumulator and the loss
  accumulator fba takes -- fb's outputs on other random arguments."""
  ns = NamedSharding(pr.mesh, P())
  args, args1 = (_rand(pr._part_avals("A"), k) for k in (seed, seed + 1))
  args = tuple(args[:-1]) + (jax.device_put(jnp.int32(3 + seed), ns),)
  args1 = tuple(args1[:-1]) + (jax.device_put(jnp.int32(5 + seed), ns),)
  out = pr.fb(*_copy(args1))
  return args, out[-2], out[-1]


def check_split_last(t, scfg, ch, mbs=4, tol=None):
  """The fused last chunk's program cut into AI and W (ds_ep.ChunkPrograms.build_split_last) against the programs it
  is cut from, on one stage mesh of scfg's fsdp x ep: AI then W against fb (the chunk's first gradient write) and
  against fba (accumulating, the accumulator's two parts and the loss accumulator) -- dx, the loss and every
  accumulator leaf bit for bit, under XLA's default options and with its excess precision off
  (testing.NO_EXCESS_PRECISION), XLA:CPU's YNNPACK fusions off in both (testing.ynn_off).  `tol(name, mode) ->
  bound`: a leaf held to max |diff| over its max |value| <= bound instead (0: bit for bit) in `mode`, as check_split's
  (by default cpu_fp32_split_tol's).

  One trace of the fused program, the accumulating halves adding the accumulator's parts and the loss accumulator to
  the first write's (Split.w_adds) -- but under in-place dW, which traces fba too.  W writes the layers' leaves
  (ds_ep.deferred_leaves), AI the rest -- the head's, the decoder norm's, MTP's, the routers' bias slots -- each
  once; W's XLA flops within SPLIT_LAST_W_SHARE of fba's, AI's and W's together at most SPLIT_FLOPS_SUM of it; and
  AI's temporaries no more than the fused program's and the crossing set's (an inlined remat keeps its barrier: no
  recompute merged with the forward).  -> (the ChunkPrograms, {acc: Split}, {'W/A': ., '(AI+W)/A': .})."""
  tol = cpu_fp32_split_tol(scfg) if tol is None else tol
  pd = ds_ep.pdesc
  mesh = dsp.typed_meshes(dsp.stage_meshes(1, fsdp=scfg.fsdp, ep=scfg.ep), scfg.shard_mode)[0]
  pr = ds_ep.ChunkPrograms(scfg, ch, mesh, mbs=mbs, seq=256, denom=float(8 * 256))
  with mock.patch.object(stage_ad, "trace_body", wraps=stage_ad.trace_body) as traced:
    sps = {acc: pr.build_split_last(acc) for acc in (False, True)}
  t.assertEqual((traced.call_count, sps[True].w_adds), (2 if pr.inplace_dw else 1, not pr.inplace_dw))
  defer = ds_ep.deferred_leaves(ch)
  for acc, sp in sps.items():
    t.assertTrue(sp.last and sp.acc == acc)
    t.assertEqual(sp.w_leaves, tuple(i for i, d in enumerate(defer) if d))
    t.assertEqual(sp.i_leaves, tuple(i for i, d in enumerate(defer) if not d))
    t.assertEqual(sp.groups, ("w", "x", "tgt") + (("tok",) if ch.mtp else ()) + ("seed",))
  args, gacc, lacc = last_inputs(pr)
  nt = int(bool(ch.mtp))
  names = [jax.tree_util.keystr(q) for q, _ in jax.tree_util.tree_flatten_with_path(pr._part_avals("A", True)[3 + nt])[0]]
  names = ([] if ch.first else ["dx"]) + names + ["loss"]

  def parts(sp, g):
    gl = jax.tree.leaves(g)
    return tuple(gl[i] for i in sp.i_leaves), tuple(gl[i] for i in sp.w_leaves)

  flops = {}
  for mode, opts in (
      ("default", testing.ynn_off()),
      ("no excess precision", {**testing.ynn_off(), **testing.NO_EXCESS_PRECISION}),
  ):
    for acc, sp in sps.items():
      case = "fba" if acc else "fb"
      gk, gw = parts(sp, gacc)
      fargs = args[: 3 + nt] + ((gacc, lacc) if acc else ()) + args[-1:]
      fc = (pr.fba if acc else pr.fb).lower(*fargs).compile(compiler_options=opts)
      want = jax.tree.leaves(fc(*_copy(fargs)))
      aargs = args[: 3 + nt] + ((gk, lacc) if acc else ()) + args[-1:]
      ic = pr.program((pd("AI", acc=acc),)).lower(*aargs).compile(compiler_options=opts)
      *dx, gk_out, loss, cross = ic(*_copy(aargs))
      wargs = (cross, sp.kept_of(args[: 3 + nt] + args[-1:])) + ((gw,) if acc else ())
      wc = pr.program((pd("W", acc=acc),)).lower(*wargs).compile(compiler_options=opts)
      (gw_out,) = wc(*_copy(wargs))
      leaves = [None] * len(defer)
      for idx, part in ((sp.i_leaves, gk_out), (sp.w_leaves, gw_out)):
        for i, v in zip(idx, part, strict=True):
          leaves[i] = v
      got = list(dx) + leaves + [loss]
      t.assertEqual(len(got), len(want), f"{case}: outputs")
      for name, a, b in zip(names, got, want):
        rel, bound = shards_apart(a, b), (tol(name, mode) if tol else 0.0)
        if rel is not None:
          t.assertLessEqual(rel, bound, f"{case}, {mode}: {name} not bit for bit (max |diff| / max |value| {rel:.3g})")
      temp = [c.memory_analysis().temp_size_in_bytes for c in (fc, ic)]
      t.assertLessEqual(temp[1], temp[0] + sp.bytes, f"{case}, {mode}: AI's temporaries (fused {temp[0]}, AI {temp[1]})")
      if acc and mode == "default":
        fa, fi, fw = _flops(fc), _flops(ic), _flops(wc)
        flops = {"W/A": fw / fa, "(AI+W)/A": (fi + fw) / fa}
  t.assertTrue(SPLIT_LAST_W_SHARE[0] <= flops["W/A"] <= SPLIT_LAST_W_SHARE[1], flops)
  t.assertLessEqual(flops["(AI+W)/A"], SPLIT_FLOPS_SUM, flops)
  for acc, sp in sps.items():
    # AI donates what A does (x; the accumulator's kept part and the loss accumulator, accumulating), but what W reads
    x = () if ch.first or any(g == "x" for g, _ in sp.kept) else (1,)
    donate = x + ((3 + nt, 4 + nt) if acc else ())
    t.assertEqual((sp.i_donate, pr.part("AI", acc=acc).donate), (donate, donate), f"acc {acc}: kept {sp.kept}")
    t.assertEqual(pr.part("W", acc=acc).donate, (2,) if acc else ())
    coll = {e.primitive.name for e in sp.w_jaxpr.eqns} & COLLECTIVES
    if scfg.grad_reduce == "step":
      t.assertFalse(coll, f"acc {acc}: W holds collectives {coll}")
  print(
      f"split last chunk {ch.index} ({scfg.gmm_impl}, layers {ch.layers}, mtp {ch.mtp}): W/A {flops['W/A']:.3f}, "
      f"(AI+W)/A {flops['(AI+W)/A']:.4f}; {sps[True].stats}; kept {sps[True].kept}; deferred {len(sps[True].w_leaves)}"
      f" of {len(defer)} leaves"
  )
  return pr, sps, flops


@testing.needs_devices(4)
class SplitLast(unittest.TestCase):
  """The fused last chunk's program cut into AI and W for the zero-bubble schedules with --zb-split-last
  (ds_ep.ChunkPrograms.build_split_last, parts 'AI' and the last chunk's 'W') against fb and fba, bit for bit
  (check_split_last), under XLA's default options and with excess precision off, on one FSDP 2 x EP 2 stage mesh --
  AI the forward, the head, the loss, the input gradient and the head's, the norm's and MTP's weight gradients (the
  head's T x vocab logits and their cotangent never cross), W the layers' weight gradients.  One configuration a test,
  scheduled_only; on every PR what build_split_last and the parts refuse (nothing traced: test_refusals)."""

  @staticmethod
  def chunk(layers=(2,), mtp=0, routed_bias=False):
    return ds_ep.ChunkSpec(2, tuple(layers), False, True, 0, mtp=mtp, routed_bias=routed_bias)

  @pytest.mark.scheduled_only
  def test_megablox(self):
    """The benchmark's kernels: MLA on tokamax splash, the routed experts on megablox and the ragged sort, bf16, one
    MoE layer.  And what the parts are: their names, the deferred leaves (the layer's), W reading nothing of AI's
    arguments (so AI donates x), the crossing set's layout and bytes."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    pr, sps, _ = check_split_last(self, scfg, self.chunk())
    pd = ds_ep.pdesc
    names = [pr.program((pd(k, acc=a),)).__name__ for k in ("AI", "W") for a in (False, True)]
    self.assertEqual(names, ["fbi_c2", "fbai_c2", "bwdw_c2", "bwdwa_c2"])
    n_dev = scfg.fsdp * scfg.ep
    paths = [jax.tree_util.keystr(q) for q, _ in jax.tree_util.tree_flatten_with_path(pr._part_avals("A", True)[3])[0]]
    for sp in sps.values():
      self.assertEqual(sp.kept, ())
      self.assertEqual([paths[i] for i in sp.i_leaves], ["['decoder_norm']['scale']", "['logits_dense']['kernel']"])
      self.assertTrue(all(paths[i].startswith("['layers']") for i in sp.w_leaves))
      self.assertEqual(sp.bytes, sum(math.prod(a.shape) * a.dtype.itemsize for a in sp.crossing_local))
      for a, spec, g in zip(sp.crossing_local, sp.crossing_specs, sp.crossing_avals):
        self.assertEqual((g.shape, g.dtype, g.sharding.spec), ((n_dev, *a.shape), a.dtype, spec))
      # the head's logits (T x vocab) do not cross
      vocab = scfg.model.vocab_size
      self.assertFalse([a.shape for a in sp.crossing_local if vocab in a.shape])
    self.assertIs(pr.build_split_last(True), sps[True])  # cached

  def test_refusals(self):
    """build_split_last and the parts refuse what has no split last chunk -- a chunk that is not the last, a last
    chunk without layers, a batch-split chunk on Lineage's layers -- and a stash mode, rs_self, a gather's arguments
    and an 'AI' of a chunk that is not the last, before anything is traced."""
    scfg = tiny(layers=3, fsdp=2, ep=2)
    mesh = dsp.stage_meshes(1, fsdp=2, ep=2)[0]

    def mk(sc, ch, mbs=4):
      return ds_ep.ChunkPrograms(sc, ch, dsp.typed_meshes([mesh], sc.shard_mode)[0], mbs, 256, 2048.0)

    for ch, want in (
        (ds_ep.ChunkSpec(1, (1,), False, False, 0), "not the last chunk"),
        (ds_ep.ChunkSpec(2, (), False, True, 0), "a chunk without layers"),
    ):
      for acc in (False, True):
        with self.subTest(chunk=ch.index, acc=acc), self.assertRaisesRegex(ValueError, "no split last chunk .*" + want):
          mk(scfg, ch).build_split_last(acc)
      with self.assertRaisesRegex(ValueError, "no split last chunk"):
        mk(scfg, ch).part("AI")
    with lineage_stand_in():
      lin = mk(lineage_scfg(layers=3, fsdp=2, ep=2), ds_ep.ChunkSpec(2, (2,), False, True, 0), mbs=8)
      with self.assertRaisesRegex(ValueError, "no split last chunk .*Lineage's layers"):
        lin.build_split_last()
    pr = mk(scfg, ds_ep.ChunkSpec(2, (2,), False, True, 0))
    for kw, want in (
        ({"kind": "AI", "stash": "lite"}, "the fused last chunk keeps nothing, so no stash mode"),
        ({"kind": "W", "acc": True, "rs_self": 2}, "folds no reduce-scatter"),
        ({"kind": "AI", "other": pr.ch}, r"takes acc only, not \['other'\]"),
        ({"kind": "I"}, "no split backward .*the last chunk"),
    ):
      kind = kw.pop("kind")
      with self.subTest(kind=kind, **{k: str(v) for k, v in kw.items()}), self.assertRaisesRegex(ValueError, want):
        pr.part(kind, **kw)
    self.assertFalse(pr._splits)  # nothing was built

  @pytest.mark.scheduled_only
  def test_remat_last(self):
    """remat_fused_last: each layer of the fused program a remat, inlined behind its optimization barrier -- in AI's
    program, none in W -- and AI's temporaries still within the fused program's and the crossing set's."""
    scfg = tiny(layers=3, fsdp=2, ep=2, remat_fused_last=True)
    pr, sps, _ = check_split_last(self, scfg, self.chunk())
    for sp in sps.values():
      self.assertGreater(sum(e.primitive.name == "optimization_barrier" for e in sp.i_jaxpr.eqns), 0)
      self.assertEqual(sum(e.primitive.name == "optimization_barrier" for e in sp.w_jaxpr.eqns), 0)
    self.assertIn("optimization_barrier", pr.program((ds_ep.pdesc("AI"),)).lower(*pr._part_avals("A")).as_text())

  @pytest.mark.scheduled_only
  def test_xla_kernels(self):
    """XLA's ops: the experts on ragged_dot after the argsort dispatch, in bf16 and (tiny_xla) in fp32; two MoE
    layers in the bf16 one."""
    for scfg, layers in ((tiny(layers=4, fsdp=2, ep=2, **XLA_KERNELS), (2, 3)), (tiny_xla(layers=3, fsdp=2, ep=2), (2,))):
      with self.subTest(dtype=jnp.dtype(scfg.model.dtype).name):
        check_split_last(self, scfg, self.chunk(layers))

  @pytest.mark.scheduled_only
  def test_mtp_routed_bias(self):
    """MTP after the head and DeepSeek-V3's routing -- top-k among the best groups on the scores plus the routed
    bias, updated from the expert counts, the sequence-wise aux loss: the MTP layer's weight gradients and every
    router's bias slot (its layer's expert counts) stay in AI."""
    feats = {
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "load_balance_loss_weight": 0.01,
        "moe_use_megatron_seq_aux_loss": True,
        "n_routing_groups": 4,
        "topk_routing_group": 2,
        "mtp_num_layers": 1,
    }
    scfg = tiny(layers=3, fsdp=2, ep=2, random_routing=False)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(**feats))
    pr, sps, _ = check_split_last(self, scfg, self.chunk(mtp=1, routed_bias=True))
    self.assertTrue(pr.counting and pr.aux_seqs and pr.mtp)
    paths = [jax.tree_util.keystr(q) for q, _ in jax.tree_util.tree_flatten_with_path(pr._part_avals("A", True)[4])[0]]
    kept = {paths[i] for i in sps[True].i_leaves}
    self.assertEqual(kept, {q for q in paths if not q.startswith("['layers']")} | {"['layers']['moe']['gate']['bias']"})

  @pytest.mark.scheduled_only
  def test_tokamax_inplace_dw(self):
    """gmm_impl 'tokamax' accumulating the routed experts' dW into a compute-dtype accumulator inside the dW kernel:
    the accumulating halves are cut from fba's own trace, W reading the accumulator's deferred part, AI its kept part
    and the loss accumulator."""
    tk = {"gmm_impl": "tokamax", "tokamax_gmm": "xla", "use_ragged_sort": False, "expert_weights": "sliced"}
    scfg = tiny(layers=3, fsdp=2, ep=2, acc_dtype=jnp.bfloat16, **tk)
    pr, sps, _ = check_split_last(self, scfg, self.chunk())
    self.assertTrue(pr.inplace_dw)
    self.assertTrue(any(not isinstance(s, stage_ad.Const) and s[0] == "gacc" for s in sps[True].w_src))

  @pytest.mark.scheduled_only
  def test_grad_reduce_microbatch(self):
    """grad_reduce 'microbatch': each half reduce-scatters the leaves it writes (W its layers')."""
    scfg = tiny(layers=3, fsdp=2, ep=2, grad_reduce="microbatch")
    _, sps, _ = check_split_last(self, scfg, self.chunk())
    self.assertTrue({e.primitive.name for e in sps[True].w_jaxpr.eqns} & COLLECTIVES)

  @pytest.mark.scheduled_only
  def test_batch_split(self):
    """Native batch split (split_layers): each device's rows as two halves in phases closed by optimization barriers,
    every one in AI."""
    scfg = tiny(layers=4, fsdp=2, ep=2, **NATIVE_SPLIT)
    _, sps, _ = check_split_last(self, scfg, self.chunk((2, 3)), mbs=8)
    for sp in sps.values():
      self.assertEqual(sum(e.primitive.name == "optimization_barrier" for e in sp.w_jaxpr.eqns), 0)

  @pytest.mark.scheduled_only
  def test_explicit_sharding(self):
    """shard_mode 'explicit': the weights typed reduced, the accumulator's parts unreduced, at the parts' boundaries."""
    scfg = tiny(layers=3, fsdp=2, ep=2, shard_mode="explicit")
    pr, sps, _ = check_split_last(self, scfg, self.chunk())
    gl = jax.tree.leaves(pr.gspec, is_leaf=ds_ep._is_spec)
    self.assertEqual(pr.part("W", acc=True).in_specs[2], tuple(gl[i] for i in sps[True].w_leaves))


# --------------------------------------------- the zero-bubble schedules
# zbh1 / zbh2 against 1F1B, bit for bit: the split backward's halves (SplitBackward) run by the step loop -- with
# XLA:CPU's YNNPACK fusions off.  With them on, as MaxText's CI runs (jaxlib's default), the fusions compile the halves
# otherwise than the fused backward in some configurations: fp32 on XLA's ops and batch split (SplitBackward's notes),
# and in bf16 an RMSNorm scale's gradient -- its sum over the tokens fused otherwise, up to 7.7e-5 of the leaf's
# largest value apart -- with fewer microbatches than stages, under the routed bias with MTP, under saved residuals,
# with the W-buffer caps, at three and at eight stages, and (on MaxText's CPU runners, not on the hosts these were
# written on) with dense layers first; and the split last chunk's AI compiles the fused program's input gradient
# otherwise, so chunk 0's gradients come out up to ~1e-5 apart (jaxlib 0.12.0.1).  On a host with many cores the
# fusions also corrupt some of these pipelines' results outright (NaN, inf; all 224 cores of a host, not 32).  So every
# bit-for-bit pipeline check skips with the fusions on (ZB_NEEDS_YNN_OFF), but ZeroBubblePipeline's three-stage ones
# in fp32 (test_three_stages, test_paced: the last chunk split too and not), which hold the gradients to the fusions'
# few ulps (`check_zero_bubble`'s `ynn`).
ZB_NEEDS_YNN_OFF = testing.needs_ynn_off(
    "they compile the split backward's halves otherwise than the fused backward, so a gradient comes out a few rounding "
    "steps apart (an RMSNorm scale's in bf16; fp32 on XLA's ops; the split last chunk's input gradient), and on many "
    "cores they corrupt some runs"
)


def zb_ep(n_stages):
  """EP of each of `n_stages` one-device-or-two stages on the devices jax has: 2 where there are two devices a stage
  (the ring of experts' collectives in every I), else 1 (MaxText's CPU CI's 4 devices)."""
  return 2 if jax.device_count() >= 2 * n_stages else 1


def zb_pipe(scfg, schedule, n_stages=4, n_micro=4, rows=2, **kw):
  """A pipeline of `scfg` at one chunk a stage, seq 256, `rows` rows a device a microbatch, under `schedule` (a
  name or a custom order), its weights seeded by chunk (init_by_chunk); nothing is compiled."""
  k = {
      "n_stages": n_stages,
      "chunks_per_stage": 1,
      "n_micro": n_micro,
      "global_batch": n_micro * rows * scfg.fsdp * scfg.ep,
      "seq": 256,
      "init_by_chunk": True,
  }
  return dsp.Pipeline(scfg, schedule=schedule, **{**k, **kw})


def check_zero_bubble(t, scfg, *, variants=None, base="1f1b", precompile=False, steps=2, ynn=False, **kw):
  """Each zero-bubble variant -- (schedule, extra Pipeline arguments); by default zbh1 and zbh2 as they are --
  against `base` on the pipeline `zb_pipe(scfg, ..., **kw)` otherwise, bit for bit (`mode_run`, `assert_same_run`):
  the loss partials, every reduced gradient, the stage sumsqs, then `steps` AdamW steps' losses, gradient norms and
  params.  Each splits every middle chunk (`split_chunks`) -- with zb_split_last the last one too -- its I's and W's
  in the order, and holds at most its cap of pending W buffers a stage; its run leaves none (`Pipeline._run`
  raises).  `precompile`: every pipeline runs
  its precompiled executables.  `ynn`: where XLA:CPU's YNNPACK fusions are on (testing.ynn_on), every gradient leaf
  within 4 float32 ulps of its scale and what follows from them to a relative 1e-6 (`assert_same_run`'s `near`), the
  loss partials -- the forward's, before any split -- still bit for bit; and not the steps' params (AdamW scales each
  element by its own gradient, so a gradient's few ulps come out magnified where it is near 0: up to 1e-3 relative
  in fp32 where the last chunk splits, 9.4e-6 of one element without on MaxText's CPU runners), but the steps'
  losses and gradient norms, which the updates feed.  -> [the variants' pipelines]."""
  variants = [(name, {}) for name in dsp.ZB_SCHEDULES] if variants is None else variants
  ref = zb_pipe(scfg, base, **kw)
  if precompile:
    ref.precompile()
  want = mode_run(ref, 0, 0, steps=steps)
  out = []
  for name, extra in variants:
    p = zb_pipe(scfg, name, **{**kw, **extra})
    what = f"{p.schedule_name} against {base}"
    t.assertEqual(p.split_chunks, list(range(1, p.S - (not p.zb_split_last))), what)
    t.assertEqual(p.split_last, p.zb_split_last, what)
    t.assertEqual({k for _, k, _ in p.order} >= {"I", "W"}, p.S > 2, what)
    if p.zb_wbuf_cap is not None:
      t.assertLessEqual(max(sched.peak_wbufs(p.order, p.place)), p.zb_wbuf_cap, what)
    if precompile:
      p.precompile()
    got = mode_run(p, 0, 0, steps=steps)
    np.testing.assert_array_equal(got["loss_partials"], want["loss_partials"], err_msg=what)
    if ynn and testing.ynn_on():
      got = dict(got, params=want["params"])  # the steps' losses and gradient norms held, their params not
    assert_same_run(t, got, want, what, near=("",) if ynn and testing.ynn_on() else ())
    out.append(p)
  return out


def split_order(order, chunks, defer=False):
  """`order` with each backward of `chunks` run as its I and then its W: at once, or (`defer`) every W of a chunk
  after the last of its I's."""
  out, M = [], 1 + max(m for _, _, m in order)
  for c, k, m in order:
    if k != "B" or c not in chunks:
      out.append((c, k, m))
    elif not defer:
      out += [(c, "I", m), (c, "W", m)]
    else:
      out.append((c, "I", m))
      out += [(c, "W", w) for w in range(M)] if m == M - 1 else []
  return out


@testing.needs_devices(4)
class ZeroBubblePipeline(unittest.TestCase):
  """The zero-bubble schedules end to end: zbh1 and zbh2 -- each middle chunk's backward run as I, the input
  gradient the stage upstream waits on, and later W, its weight gradients (ds_ep.ChunkPrograms.build_split) --
  against 1F1B on the same pipeline otherwise, bit for bit (`check_zero_bubble`): a chunk's W's write its gradient
  accumulator in microbatch order, as its B's would, and every other program is the same.  On every PR what one-device
  stages build, compiling nothing, the last chunk split too or not (test_plan); scheduled_only, three such stages run
  in fp32 on XLA's ops, with the last chunk split and without (also with the W buffers paced as on a TPU VM's client:
  test_paced) -- MaxText's CI's runs of the split step loop, its YNNPACK fusions on -- and four stages of EP 2 where
  there are 8 devices (else EP 1): MaxText's kernels in bf16 with and without the fusion plan, fp32 on XLA's ops,
  dense layers in a split chunk, batch split, saved and offloaded residuals, the reduce-scatter in every gradient
  write (grad_reduce 'microbatch'), the routed bias with its aux loss and MTP, explicit sharding, the last chunk's
  remat, fewer microbatches than stages, three and eight stages, precompiled programs, W-buffer caps 0 / 1 / 2, and
  a custom order of I and W ops at two chunks a stage against breadth's.  And with the last chunk split too
  (zb_split_last), bf16: at two stages, where it is the only split, with and without the fusion plan; at four,
  precompiled; and with the last chunk's remat, and with MTP and the routed bias, at three."""

  def test_plan(self):
    """What zbh1 and zbh2 build, compiling nothing, fp32 on XLA's ops (tiny_xla) on one device a stage: at three
    stages the middle chunk split -- its backward traced once, one I program for both its W's -- its last W folding
    the chunk's reduce-scatter (rs_self) and issued last, each stage's ops issued in the order's order, at most the
    cap of pending W buffers, the W buffer in the summary and the run record's block (unpaced: CPU devices), the
    simulated loop no slower than 1F1B's; at two stages, where no chunk splits, no cap in the name; 1F1B's ops issued
    as its order has them;
    with the last chunk split too, `plan_split_last`.  And under measured costs whose split costs more than the
    backward it cuts -- round 2's TPU7x L4, I + W = 1.036 B -- the warning where the order simulates slower than
    1F1B's (64 microbatches), none under the default costs (I + W = B)."""
    scfg = tiny_xla(layers=3, fsdp=1, ep=1)
    for name in dsp.ZB_SCHEDULES:
      with self.subTest(schedule=name), mock.patch.object(stage_ad, "trace_body", wraps=stage_ad.trace_body) as traced:
        p = zb_pipe(scfg, name, n_stages=3, n_micro=3)
        self.assertEqual((p.split_chunks, traced.call_count), ([1], 1))
        self.assertEqual({op.desc for op in p.ops() if op.kind == "I"}, {(ds_ep.pdesc("I"),)})
        self.assertEqual(sorted(p.order[j] for j in p.rs_self), [(0, "B", 2), (1, "W", 2)])
        issue, stage = p.issue_order(), [op.s for op in p.ops()]
        self.assertEqual(sorted(issue), list(range(len(p.order))))
        self.assertEqual(p.order[issue[-1]], (1, "W", 2))
        # (a stable sort by stage: each stage's ops in issue order, then in the order's)
        self.assertEqual(sorted(issue, key=stage.__getitem__), sorted(range(len(p.order)), key=stage.__getitem__))
        self.assertIn(" wbuf<=1 sim=", p.schedule_name)
        self.assertIn("split backward (I / W) on chunks [1]: W buffer", p.summary())
        self.assertGreater(p.wbuf_bytes[1], 0)
        rec = p.zb_record()
        self.assertEqual(
            (rec["split_chunks"], rec["wbuf_cap"], rec["paced"], rec["peak_wbufs"]), ([1], 1, False, [0, 1, 0])
        )
        self.assertLessEqual(rec["sim_ms"]["order"], rec["sim_ms"]["1f1b"])
    for name in dsp.ZB_SCHEDULES:
      order, what = dsp.build_schedule(name, 2, 4, [0, 1])
      self.assertEqual(({k for _, k, _ in order}, "wbuf" in what), ({"F", "B"}, False), what)
      self.assertIn(" (no split chunk) sim=", what)
    p = zb_pipe(scfg, "1f1b", n_stages=3, n_micro=3)
    self.assertEqual(p.issue_order(), list(range(len(p.order))))
    self.assertFalse(p.pace_i or zb_pipe(scfg, "zbh1", n_stages=3, n_micro=3).pace_i)  # McJAX's only (test_paced)
    with mock.patch.object(dsp, "allocates_at_issue", return_value=True):  # devices as a TPU VM's own client's
      self.assertTrue(zb_pipe(scfg, "zbh1", n_stages=3, n_micro=3).pace_i)
      # ds_ep_aot --pathways plans one Pathways client's run on compile-only TPU devices: unpaced, as that run is
      self.assertFalse(zb_pipe(scfg, "zbh1", n_stages=3, n_micro=3, aot=True, proxy=True).pace_i)
    self.plan_split_last()
    measured = {"F": 31.0, "B": 89.3, "I": 66.1, "W": 26.4, "A": 102.0}  # ms, round 2's L4 on McJAX
    for name in dsp.ZB_SCHEDULES:
      with self.subTest(schedule=name, costs="measured"):
        with self.assertWarnsRegex(RuntimeWarning, r"simulates slower than 1F1B's .* compare a --schedule 1f1b run"):
          dsp.build_schedule(name, 4, 64, list(range(4)), costs=measured)
        with warnings.catch_warnings(record=True) as seen:
          warnings.simplefilter("always")
          dsp.build_schedule(name, 4, 64, list(range(4)))
          dsp.build_schedule(name, 4, 8, list(range(4)), costs=measured)  # faster than 1F1B at 8
        self.assertEqual([str(w.message) for w in seen if "slower than 1F1B" in str(w.message)], [])

  def plan_split_last(self):
    """(test_plan) What zbh1 and zbh2 build at two stages with the last chunk split too (zb_split_last; opt-in,
    --zb-split-last on): the only split chunk the last -- its fused program traced once, its F an AI (the first
    write's, then accumulating), a W of its own after each, hosting nothing -- the order no longer 1F1B's and faster in
    simulation, at most the cap of W buffers on the last stage, the split in the name, the summary and the run
    record."""
    scfg = tiny_xla(layers=2, fsdp=1, ep=1)
    self.assertEqual(dsp.make_parser().parse_args([]).zb_split_last, "off")  # the command line's default
    for name in dsp.ZB_SCHEDULES:
      with self.subTest(schedule=name), mock.patch.object(stage_ad, "trace_body", wraps=stage_ad.trace_body) as traced:
        p = zb_pipe(scfg, name, n_stages=2, n_micro=4, zb_split_last=True)
        self.assertEqual((p.split_chunks, p.split_last, traced.call_count), ([1], True, 1))
        self.assertNotEqual(p.order, sched.one_f_one_b(2, 4, p.place))
        self.assertEqual({k for c, k, _ in p.order if c == 1}, {"F", "B", "W"})
        ops = [op for op in p.ops() if op.c == 1 and op.desc is not None]
        self.assertEqual(
            [op.desc for op in ops if op.kind == "F"], [(ds_ep.pdesc("AI"),)] + [(ds_ep.pdesc("AI", acc=True),)] * 3
        )
        self.assertEqual(
            [op.desc for op in ops if op.kind == "W"], [(ds_ep.pdesc("W"),)] + [(ds_ep.pdesc("W", acc=True),)] * 3
        )
        self.assertTrue(all(op.rs is None and op.rs_self is None for op in ops))
        self.assertEqual(p.final_slots[1], (0,))  # the last chunk's reduce-scatter: the stage's standalone reduce
        self.assertIn(" split-last wbuf<=1 sim=", p.schedule_name)
        self.assertEqual(sched.peak_wbufs(p.order, p.place), [0, 1])
        self.assertIn("split backward (I / W) on chunks [1]: W buffer", p.summary())
        rec = p.zb_record()
        self.assertEqual((rec["split_chunks"], rec["split_last"], rec["peak_wbufs"]), ([1], True, [0, 1]))
        self.assertEqual(set(rec["chunks"]["1"]["costs"]), {"A", "AI", "W"})
        self.assertLess(rec["sim_ms"]["order"], rec["sim_ms"]["1f1b"])

  @pytest.mark.scheduled_only
  def test_three_stages(self):
    """test_plan's pipelines run: three stages of one device, fp32 on XLA's ops (tiny_xla), the middle chunk split,
    and with the last chunk split too (zb_split_last: its AI and W, the accumulator's two parts merged after its last
    W).  Bit for bit with XLA:CPU's YNNPACK fusions off; with them on, as MaxText's CI runs, to their few ulps
    (`check_zero_bubble`: up to 0.6 of its bound on 4 devices, jaxlib 0.12.0.1)."""
    variants = [(name, kw) for kw in ({}, {"zb_split_last": True}) for name in dsp.ZB_SCHEDULES]
    check_zero_bubble(self, tiny_xla(layers=3, fsdp=1, ep=1), variants=variants, n_stages=3, n_micro=3, ynn=True)

  @pytest.mark.scheduled_only
  def test_paced(self):
    """The pacing of the W buffers a TPU VM's client gets (Pipeline.pace_i, forced on: on CPU it is off) waits and
    changes nothing: test_three_stages' pipelines, and with the last chunk split too (its AI's paced as I's are); the
    run record's block says paced."""
    scfg = tiny_xla(layers=3, fsdp=1, ep=1)
    with mock.patch.object(dsp, "allocates_at_issue", return_value=True):
      for kw in ({}, {"zb_split_last": True}):
        with self.subTest(**kw):
          variants = [(name, kw) for name in dsp.ZB_SCHEDULES]
          ps = check_zero_bubble(self, scfg, variants=variants, n_stages=3, n_micro=3, ynn=True)
          self.assertTrue(all(p.pace_i and p.zb_record()["paced"] for p in ps))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_megablox(self):
    """MaxText's kernels -- megablox and the ragged sort, tokamax splash -- in bf16, with the fusion plan (the
    gathers in forwards, every chunk's reduce-scatter folded into its last gradient write) and without it."""
    for overlap in (True, False):
      with self.subTest(overlap=overlap):
        check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4)), overlap=overlap)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_xla_kernels_fp32(self):
    """fp32 on XLA's ops (tiny_xla), with and without the fusion plan."""
    for overlap in (True, False):
      with self.subTest(overlap=overlap):
        check_zero_bubble(self, tiny_xla(layers=4, fsdp=1, ep=zb_ep(4)), overlap=overlap)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_dense_layers(self):
    """A layer split counting every layer, two dense ones first: chunk 1 holds a dense layer and a MoE one, and
    splits both."""
    scfg = tiny(layers=5, fsdp=1, ep=zb_ep(4))
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(first_num_dense_layers=2, mlp_dim=384))
    p = check_zero_bubble(self, scfg, layer_split=(1, 2, 1, 1))[0]
    self.assertEqual([(c.dense, c.layers) for c in p.chunks], [(1, ()), (1, (0,)), (0, (1,)), (0, (2,))])

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_batch_split(self):
    """Native batch split (split_layers), two MoE layers a chunk: each device's rows as two halves."""
    check_zero_bubble(self, tiny(layers=8, fsdp=1, ep=zb_ep(4), **NATIVE_SPLIT))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_saved_and_offloaded_residuals(self):
    """Saved residuals (per-layer trees: the backward is the VJP of the saved set, which a W reads too, so its W
    buffer holds those as well), and residuals offloaded to pinned host memory between a forward and its backward
    or I (a W's reads of them stay in HBM until it runs)."""
    for what, scfg, kw in (
        ("saved", tiny(layers=4, fsdp=1, ep=zb_ep(4), save_residuals=True, layer_tree="per_layer"), {}),
        ("offloaded", tiny(layers=4, fsdp=1, ep=zb_ep(4)), {"offload_residuals": True}),
    ):
      with self.subTest(what):
        p = check_zero_bubble(self, scfg, **kw)[0]
        kept = {g for g, _ in p.progs[1].build_split(True).kept}
        self.assertEqual(kept, {"res"} if what == "saved" else {"dy"})

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_microbatch_reduce(self):
    """grad_reduce 'microbatch': every gradient write reduce-scatters its microbatch's gradient -- a split chunk's
    in its W's."""
    check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4), grad_reduce="microbatch"))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_routed_bias_aux_loss_mtp(self):
    """DeepSeek-V3's routing -- top-k among the best groups on the scores plus the routed bias, updated from the
    expert counts, the sequence-wise aux loss -- and an MTP layer after the head (the last chunk's, unsplit); the
    router in bf16 (in fp32 on megablox the split is not bit for bit: SplitBackward.test_float32_gate_logits)."""
    feats = {
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "load_balance_loss_weight": 0.01,
        "moe_use_megatron_seq_aux_loss": True,
        "n_routing_groups": 4,
        "topk_routing_group": 2,
        "mtp_num_layers": 1,
    }
    scfg = tiny(layers=4, fsdp=1, ep=zb_ep(4), random_routing=False)
    p = check_zero_bubble(self, dataclasses.replace(scfg, model=scfg.model.replace(**feats)))[0]
    self.assertTrue(p.mtp and p.progs[1].counting and p.progs[1].aux_seqs)

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "four stages of fsdp 1 x ep 2: an explicit stage takes more than one device")
  @ZB_NEEDS_YNN_OFF
  def test_explicit_sharding(self):
    """shard_mode 'explicit': the weights typed reduced, the accumulators unreduced."""
    check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=2, shard_mode="explicit"))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_remat_last(self):
    """The fused last chunk recomputing its forward in its backward (remat_fused_last)."""
    check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4), remat_fused_last=True))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_fewer_microbatches_than_stages(self):
    """M = 3 < S = 4."""
    check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4)), n_micro=3)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_three_stages_ep(self):
    """Three stages, the middle one of EP 2 where there are 6 devices or more, M = 6."""
    check_zero_bubble(self, tiny(layers=3, fsdp=1, ep=zb_ep(3)), n_stages=3, n_micro=6)

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "eight stages of one device")
  @ZB_NEEDS_YNN_OFF
  def test_eight_stages(self):
    """Eight stages of one device, six chunks split, M = 8."""
    check_zero_bubble(self, tiny(layers=8, fsdp=1, ep=1, **XLA_KERNELS), n_stages=8, n_micro=8)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_precompile(self):
    """Every pipeline's programs precompiled (Pipeline.precompile: the I and W parts' abstract arguments,
    _desc_avals) and run as executables."""
    for p in check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4)), precompile=True):
      names = {n for n, _ in p.compiled}
      self.assertTrue({"bwdi_c1", "bwdw_c1", "bwdwa_c1", "bwdwa_c1_rsself", "bwdi_c2"} <= names, sorted(names))
      self.assertTrue(all(isinstance(p.program(op), jax.stages.Compiled) for op in p.ops() if op.desc is not None))

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_wbuf_caps(self):
    """zb_wbuf_cap 0 (no cap), 1 and 2: each order within its cap (or the schedules' own bound), every one 1F1B's
    gradients."""
    variants = [(name, {"zb_wbuf_cap": cap}) for name in dsp.ZB_SCHEDULES for cap in (0, 1, 2)]
    ps = check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4)), variants=variants, n_micro=8)
    caps = [(p.zb_wbuf_cap, max(sched.peak_wbufs(p.order, p.place))) for p in ps]
    self.assertEqual([c for c, _ in caps], [None, 1, 2] * 2)
    self.assertTrue(all(peak <= (c or 2 * 4 - 1) for c, peak in caps), caps)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_custom_order_two_chunks_a_stage(self):
    """A custom order of I and W ops at two chunks a stage (zbh1 / zbh2 build one chunk a stage; a custom order runs
    at any): breadth's, each middle chunk's backwards run as their I's and then, all deferred, their W's -- M pending
    W buffers on its stage, and the last chunk's reduce-scatter folded into chunk 1's W's -- against breadth's own,
    bit for bit, with the fusion plan and without."""
    S, M = 2, 4
    scfg = tiny(layers=4, fsdp=1, ep=zb_ep(S))
    base = sched.breadth_first(2 * S, M, sched.placement(S, 2))
    order = split_order(base, (1, 2), defer=True)
    kw = {"n_stages": S, "chunks_per_stage": 2, "n_micro": M, "global_batch": M * 2 * scfg.ep, "seq": 256}
    kw["init_by_chunk"] = True
    for overlap in (True, False):
      with self.subTest(overlap=overlap):
        want = mode_run(dsp.Pipeline(scfg, schedule=base, overlap=overlap, **kw), 0, 0)
        p = dsp.Pipeline(scfg, schedule=order, overlap=overlap, **kw)
        self.assertEqual((p.split_chunks, sched.peak_wbufs(p.order, p.place)), ([1, 2], [M, M]))
        if overlap:
          self.assertIn((1, "W", 0), [p.order[j] for j in p.rs_at])  # chunk 3's group, in a W
        got = mode_run(p, 0, 0)
        np.testing.assert_array_equal(got["loss_partials"], want["loss_partials"])
        assert_same_run(self, got, want, f"custom I/W order against breadth (overlap {overlap})")

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_split_last_two_stages(self):
    """zb_split_last at two stages, where the last chunk is the one chunk that splits: zbh1 and zbh2 against 1F1B --
    their orders other than 1F1B's, and faster in simulation -- at W-buffer caps 1 and 2 with the fusion plan (the
    last chunk's reduce-scatter the stage's standalone reduce, after its accumulator's two parts merge), at cap 1
    without it."""
    for overlap, caps in ((True, (1, 2)), (False, (1,))):
      variants = [(name, {"zb_split_last": True, "zb_wbuf_cap": cap}) for name in dsp.ZB_SCHEDULES for cap in caps]
      with self.subTest(overlap=overlap):
        ps = check_zero_bubble(self, tiny(layers=2, fsdp=1, ep=zb_ep(2)), variants=variants, n_stages=2, overlap=overlap)
        for p in ps:
          self.assertNotEqual(p.order, sched.one_f_one_b(2, p.M, p.place))
          self.assertLess(p.zb_sims()["order"], p.zb_sims()["1f1b"])

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_split_last_four_stages(self):
    """zb_split_last at four stages: every chunk but the first split, the last one's W buffers under the cap with
    the middle ones'; precompiled (Pipeline.precompile: the AI and last W parts' abstract arguments, _desc_avals)."""
    variants = [(name, {"zb_split_last": True}) for name in dsp.ZB_SCHEDULES]
    for p in check_zero_bubble(self, tiny(layers=4, fsdp=1, ep=zb_ep(4)), variants=variants, precompile=True):
      names = {n for n, _ in p.compiled}
      self.assertTrue({"fbi_c3", "fbai_c3", "bwdw_c3", "bwdwa_c3", "bwdi_c1"} <= names, sorted(names))
      self.assertLessEqual(max(sched.peak_wbufs(p.order, p.place)), 1)

  @pytest.mark.scheduled_only
  @ZB_NEEDS_YNN_OFF
  def test_split_last_remat_mtp_routed_bias(self):
    """zb_split_last with the last chunk's remat (remat_fused_last), and with MTP after the head and DeepSeek-V3's
    routing -- the routed bias updated from the expert counts, the aux loss -- whose MTP layer and bias slots stay in
    the last chunk's AI, at three stages."""
    feats = {
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "load_balance_loss_weight": 0.01,
        "moe_use_megatron_seq_aux_loss": True,
        "n_routing_groups": 4,
        "topk_routing_group": 2,
        "mtp_num_layers": 1,
    }
    variants = [(name, {"zb_split_last": True}) for name in dsp.ZB_SCHEDULES]
    remat = tiny(layers=3, fsdp=1, ep=zb_ep(3), remat_fused_last=True)
    routed = tiny(layers=3, fsdp=1, ep=zb_ep(3), random_routing=False)
    routed = dataclasses.replace(routed, model=routed.model.replace(**feats))
    for what, scfg in (("remat_fused_last", remat), ("mtp, routed bias", routed)):
      with self.subTest(what):
        check_zero_bubble(self, scfg, variants=variants, n_stages=3, n_micro=4)


@testing.needs_devices(4)
class ZeroBubbleRefusals(unittest.TestCase):
  """What the split backward cannot run, refused before anything is built (`dsp.zb_refusals`, sched.validate):
  zbh1 / zbh2 at more than one chunk a stage, batch split on Lineage's layers, fsdp_scope 'repeat', a stash mode on
  a split chunk, an F&B pair holding an I, an I or W of chunk 0 or of the fused last chunk, a W-buffer cap that is
  not a count.  Nothing is traced or compiled.  On every PR the orders' rules (test_illegal_orders), the rest
  scheduled_only."""

  def setUp(self):
    self.scfg = tiny_xla(layers=4, fsdp=1, ep=1)
    self.order = split_order(sched.one_f_one_b(4, 4, list(range(4))), (1, 2))

  def pipe(self, schedule, scfg=None, **over):
    """A pipeline of four one-device stages, M = 4, under `schedule`, with `over` for any of its arguments."""
    kw = {"n_stages": 4, "chunks_per_stage": 1, "n_micro": 4, "global_batch": 8, "seq": 256}
    return dsp.Pipeline(scfg or self.scfg, schedule=schedule, **{**kw, **over})

  @pytest.mark.scheduled_only
  def test_one_chunk_a_stage(self):
    """zbh1 / zbh2 at two chunks a stage: refused by Pipeline (either placement) and build_schedule alike."""
    for name in dsp.ZB_SCHEDULES:
      for placement in ("circular", "v"):
        with (
            self.subTest(schedule=name, placement=placement),
            self.assertRaisesRegex(ValueError, rf"schedule '{name}': ZB-H1 / ZB-H2 run one chunk a stage"),
        ):
          self.pipe(name, n_stages=2, chunks_per_stage=2, placement=placement)
      with self.assertRaisesRegex(ValueError, r"not placement \[0, 1, 0, 1\]; a custom order of I and W"):
        dsp.build_schedule(name, 4, 4, sched.placement(2, 2))

  @pytest.mark.scheduled_only
  def test_lineage(self):
    """Batch split on Lineage's layers: one custom VJP of Lineage's whole region, no input gradient to cut."""
    with lineage_stand_in():
      for schedule in ("zbh1", self.order):
        with (
            self.subTest(custom=not isinstance(schedule, str)),
            self.assertRaisesRegex(ValueError, "cannot run this: batch_split on Lineage's layers"),
        ):
          self.pipe(schedule, lineage_scfg(layers=4, fsdp_collectives="xla"), global_batch=16)

  @pytest.mark.scheduled_only
  def test_repeat_scope(self):
    for schedule in ("zbh2", self.order):
      with (
          self.subTest(custom=not isinstance(schedule, str)),
          self.assertRaisesRegex(ValueError, r"cannot run this: fsdp_scope 'repeat'.*\(fsdp_scope 'step'\)"),
      ):
        self.pipe(schedule, fsdp_scope="repeat")

  @pytest.mark.scheduled_only
  def test_stash(self):
    """A stash mode on a split chunk; --stash with zbh1 / zbh2 (no stash scheduler there)."""
    with self.assertRaisesRegex(ValueError, r"stash modes on split chunks \[\(1, 0\), \(2, 3\)\]: the split cuts"):
      self.pipe(self.order, modes={(0, 1): "lite", (1, 0): "lite", (2, 3): "attn"})
    with self.assertRaisesRegex(ValueError, "stash 'lite' needs schedule stash or interleaved"):
      self.pipe("zbh1", stash="lite")

  @pytest.mark.scheduled_only
  def test_pair_holding_an_i(self):
    """An F&B pair of a forward and an I (merge's pair of 1F1 and 1I0 on stage 1): no program pairs them."""
    place = list(range(3))
    streams = [
        [(0, "F", 0), (0, "F", 1), (0, "B", 0), (0, "B", 1)],
        [(1, "F", 0), ((1, "F", 1), (1, "I", 0)), (1, "W", 0), (1, "I", 1), (1, "W", 1)],
        [(2, "F", 0), (2, "B", 0), (2, "F", 1), (2, "B", 1)],
    ]
    order, fuse, _ = sched.merge(streams, place, sched.COST, n_micro=2)
    with self.assertRaisesRegex(ValueError, r"F&B pairs of a forward and an I \(1F1&1I0\): none is built"):
      self.pipe(order, tiny_xla(layers=3, fsdp=1, ep=1), n_stages=3, n_micro=2, global_batch=4, fuse=fuse)

  def test_illegal_orders(self):
    """sched.validate's rules for I and W, through validate_schedule: no split of chunk 0, no I of the fused last
    chunk, a W after its I, both or neither, the W's of a chunk in microbatch order -- and a split last chunk's W's
    (zb_split_last: its F runs the input gradient) one for every microbatch, each after its F and no-op B, in
    microbatch order."""
    C, M, good = 4, 4, self.order
    deferred = split_order(sched.one_f_one_b(C, M, list(range(C))), (1, 2), defer=True)
    last = [y for x in good for y in ([x, (3, "W", x[2])] if x[:2] == (3, "B") else [x])]  # the last chunk split too
    last_deferred = good + [(3, "W", m) for m in range(M)]
    for order in (good, deferred, last, last_deferred):
      dsp.validate_schedule(order, C, M, list(range(C)))

    def swap(o, a, b):
      return [b if x == a else a if x == b else x for x in o]

    cases = {
        "chunk 0": (split_order(good, (0,)), "chunk 0 has no input gradient to split off"),
        "last chunk": ([x if x != (3, "B", 0) else (3, "I", 0) for x in good], "fused last chunk: it has no I"),
        "W before I": (swap(good, (1, "I", 0), (1, "W", 0)), r"1W0 at order\[\d+\] runs before 1B0/1I0"),
        "no W": ([x for x in good if x != (2, "W", 1)], r"2W1 \(after 2I1\) is missing"),
        "both": (good + [(1, "B", 0)], r"\(1, 0\) has both a B and an I/W"),
        "W order": (swap(deferred, (1, "W", 0), (1, "W", 1)), r"chunk 1 accumulates .* not FIFO"),
        "a last W missing": ([x for x in last if x != (3, "W", 2)], "3W2 is missing: a split last chunk has a W for"),
        "last W before its F": (
            [y for x in last if x != (3, "W", 1) for y in ([(3, "W", 1), x] if x == (3, "F", 1) else [x])],
            r"3W1 at order\[\d+\] runs before 3B1",
        ),
        "last W order": (
            swap(last_deferred, (3, "W", 0), (3, "W", 1)),
            "chunk 3 accumulates its deferred weight gradients in microbatch order",
        ),
    }
    for name, (order, msg) in cases.items():
      with self.subTest(name), self.assertRaisesRegex(ValueError, msg):
        dsp.validate_schedule(order, C, M, list(range(C)))
    self.assertEqual(dsp.normalize_schedule([[1, "I", 0], [1, "W", 0]]), [(1, "I", 0), (1, "W", 0)])
    with self.assertRaisesRegex(ValueError, "kind must be 'F', 'B', 'I' or 'W'"):
      dsp.normalize_schedule([(1, "X", 0)])

  @pytest.mark.scheduled_only
  def test_wbuf_cap(self):
    for cap in (-1, True, 1.5):
      with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "zb_wbuf_cap must be None or an int >= 0"):
        self.pipe("zbh1", zb_wbuf_cap=cap)

  @pytest.mark.scheduled_only
  def test_split_last(self):
    """zb_split_last: with a schedule other than zbh1 / zbh2 (a custom order splits the last chunk by giving it W's),
    on a last chunk without layers (nothing for its W to defer), and a custom order's F&B pair of the split last
    chunk's AI and another chunk's backward -- refused before anything is built."""
    for schedule in ("1f1b", "breadth", self.order):
      with (
          self.subTest(schedule=schedule if isinstance(schedule, str) else "custom"),
          self.assertRaisesRegex(ValueError, "zb_split_last splits the fused last chunk of zbh1 / zbh2's orders"),
      ):
        self.pipe(schedule, zb_split_last=True)
    with self.assertRaisesRegex(ValueError, r"zb_split_last: the last chunk holds no layer \(layer_split \[2, 1, 1, 0\]"):
      self.pipe("zbh1", zb_split_last=True, layer_split=(2, 1, 1, 0))
    with self.assertRaisesRegex(ValueError, "zb_split_last must be a bool"):
      self.pipe("zbh2", zb_split_last="on")
    # two chunks a stage: the last chunk's F (3F1) paired with chunk 1's backward (1B0) on stage 1
    order = [(0, "F", 0), (1, "F", 0), (2, "F", 0), (3, "F", 0), (3, "B", 0), (0, "F", 1), (1, "F", 1), (2, "F", 1)]
    order += [(3, "F", 1), (1, "B", 0), (3, "B", 1), (3, "W", 0), (3, "W", 1)]
    with self.assertRaisesRegex(ValueError, r"F&B pairs of the split last chunk's forward \(AI.*3F1&1B0"):
      dsp.zb_refusals("a custom order of I and W ops", self.scfg, "step", order=order, fuse={8})


# --------------------------------------------------- devices and meshes
@dataclasses.dataclass(frozen=True)
class FakeTpu:
  """A TPU device as stage_meshes and sync_collective_stages see it (and
  mesh_utils' topology-aware mesh: coords, core_on_chip, device_kind)."""

  id: int
  coords: tuple
  process_index: int
  core_on_chip: int = 0
  platform: str = "tpu"
  device_kind: str = "TPU v6 lite"


def fake_v6e(x=16, y=16):
  """A v6e slice of x by y chips, one device per chip and a host per 2x2
  chips, hosts row-major and device ids host by host as on the slice:
  v6e-256 by default."""
  devs = []
  for hy in range(y // 2):
    for hx in range(x // 2):
      for cy in range(2):
        for cx in range(2):
          devs.append(FakeTpu(id=len(devs), coords=(2 * hx + cx, 2 * hy + cy, 0), process_index=hy * (x // 2) + hx))
  return devs


@testing.needs_devices(4)
class Validation(unittest.TestCase):
  """Options are checked where they are given: nothing compiles here."""

  @testing.needs_devices(4)
  def test_save_residuals_refusal_says_what_to_set(self):
    """--save-residuals on with what its saved-residual forward does not run -- stacked expert reads (the default
    --expert-weights stacked on a stacked layer tree), the routed-bias update, the sequence-wise aux loss -- is
    refused naming each with the setting to change (the message once read as if --expert-weights sliced were the
    trouble); --expert-weights sliced or --layer-tree per_layer builds."""
    ap = dsp.make_parser()
    base = TINY_ARGS + ["--save-residuals", "on"]
    stacked = (
        r"stacked expert reads -- the routed experts' weights read in place from the chunk's stacked layer leaves, "
        r"expert_weights 'stacked' \(the default\) on layer_tree 'stacked': set expert_weights 'sliced' "
        r"\(--expert-weights sliced\) or layer_tree 'per_layer' \(--layer-tree per_layer\)"
    )
    with self.assertRaisesRegex(
        ValueError, r"^chunk 0: save_residuals \(--save-residuals on\) does not support " + stacked + "$"
    ):
      dsp.build(ap.parse_args(base))
    bias = r"the routed-bias update's expert counts \(routed_bias_update_rate > 0\): set it to 0 \(--bias-update-rate 0\)"
    aux = r"the sequence-wise aux loss \(load_balance_loss_weight > 0\): set it to 0 \(--lb-loss-weight 0\)"
    for extra, want in (
        (["--routed-bias", "on", "--bias-update-rate", "0.001"], f"{stacked}; nor {bias}$"),
        (["--lb-loss-weight", "0.01"], f"{stacked}; nor {aux}$"),
        (["--layer-tree", "per_layer", "--lb-loss-weight", "0.01"], f"does not support {aux}$"),
    ):
      with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, want):
        dsp.build(ap.parse_args(base + extra))
    for fix in (["--expert-weights", "sliced"], ["--layer-tree", "per_layer"]):
      with self.subTest(fix=fix):
        self.assertTrue(dsp.build(ap.parse_args(base + fix)).scfg.save_residuals)

  def test_unknown_names_are_rejected(self):
    for kw in (
        dict(fsdp_collectives="all_gather"),
        dict(fsdp_collectives="sparsecore"),
        dict(ep_collectives="sparsecore"),
        dict(gmm_impl="tokamax"),
    ):
      with self.subTest(**kw), self.assertRaises(ValueError):
        tiny(**kw)

  def test_ppermute_needs_size_two_axes(self):
    with self.assertRaises(ValueError):
      tiny(layers=4, fsdp=4, ep=1)
    with self.assertRaises(ValueError):
      tiny(layers=4, fsdp=1, ep=4, fsdp_collectives="xla", ep_collectives="ppermute")

  @testing.needs_devices(4)
  def test_pipeline_level_choices(self):
    """'auto' (EP) is Pipeline's to resolve per stage: a stage program
    refuses it rather than silently running the lax collectives.  Off TPU
    nothing is offloaded, so it resolves to the collective-permutes."""
    scfg = tiny(ep_collectives="auto", **testing.stage_axes())
    with self.assertRaises(ValueError):
      ds_ep.ChunkPrograms(scfg, None, None, 4, 256, 1.0)
    with self.assertRaises(ValueError):
      ds_ep.StagePrograms(scfg, [], None)
    p = dsp.Pipeline(
        tiny_xla(ep_collectives="auto", **testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=1,
        n_micro=2,
        global_batch=8,
        seq=256,
        overlap=False,
    )
    self.assertEqual({(c.fsdp_collectives, c.ep_collectives) for c in p.stage_scfg}, {("ppermute", "ppermute")})
    # one EP flavour per stage, whatever scfg.ep_collectives says (--ep-collectives ppermute,xla)
    kw = dict(n_stages=2, chunks_per_stage=1, n_micro=2, global_batch=8, seq=256, overlap=False)
    for ep_coll in ("auto", "xla"):
      p = dsp.Pipeline(
          tiny_xla(ep_collectives=ep_coll, **testing.stage_axes()), stage_ep_collectives=["ppermute", "xla"], **kw
      )
      self.assertEqual([c.ep_collectives for c in p.stage_scfg], ["ppermute", "xla"])
    for bad in (["ppermute"], ["ppermute", "auto"], ["xla"] * 3):
      with self.subTest(stage_ep_collectives=bad), self.assertRaises(ValueError):
        dsp.Pipeline(tiny_xla(**testing.stage_axes()), stage_ep_collectives=bad, **kw)
    with self.assertRaisesRegex(ValueError, "ep == 2"):
      dsp.Pipeline(tiny_xla(fsdp=2, ep=1, fsdp_collectives="xla"), stage_ep_collectives=["ppermute"] * 2, **kw)
    for arg in ("auto", "xla", "ppermute", "ppermute,xla", "xla,xla,ppermute,xla"):
      self.assertEqual(dsp.ep_collectives_arg(arg), arg)
    for arg in ("sparsecore", "auto,xla", "ppermute,", ""):
      with self.subTest(arg=arg), self.assertRaises(argparse.ArgumentTypeError):
        dsp.ep_collectives_arg(arg)

  def test_shard_mode(self):
    """shard_mode is auto or explicit, explicit only with grad_reduce='step'
    (its unreduced accumulator holds a step's per-device partials);
    --shard-mode reaches the StageConfig through `build`, explicit gives the
    stage meshes Explicit axes (auto leaves them as they are), and the plan
    fingerprint and the summary tell the modes apart -- the plan itself is
    the same."""
    with self.assertRaisesRegex(ValueError, "shard_mode must be one of"):
      tiny_xla(shard_mode="manual")
    with self.assertRaisesRegex(ValueError, "needs grad_reduce='step'"):
      tiny_xla(shard_mode="explicit", grad_reduce="microbatch")
    # a one-device stage: jax lowers its shard_maps inline, which refuses unreduced outputs
    with self.assertRaisesRegex(ValueError, "more than one device"):
      tiny_xla(shard_mode="explicit", fsdp=1, ep=1)
    for fsdp, ep in ((1, 2), (2, 1)):
      self.assertEqual(tiny_xla(shard_mode="explicit", fsdp=fsdp, ep=ep).shard_mode, "explicit")
    ap = dsp.make_parser()
    with self.assertRaisesRegex(ValueError, "more than one device"):  # TINY_ARGS' fsdp 1, and ep 1
      dsp.build(ap.parse_args(TINY_ARGS + ["--ep", "1", "--shard-mode", "explicit"]))
    self.assertEqual(ap.parse_args(TINY_ARGS).shard_mode, "auto")
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(TINY_ARGS + ["--shard-mode", "manual"])
    p, q = (dsp.build(ap.parse_args(TINY_ARGS + ["--shard-mode", m])) for m in ("auto", "explicit"))
    self.assertEqual((p.scfg.shard_mode, q.scfg.shard_mode), ("auto", "explicit"))
    Ex, Au = jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto
    self.assertEqual(({m.axis_types for m in p.meshes}, {m.axis_types for m in q.meshes}), ({(Au, Au)}, {(Ex, Ex)}))
    self.assertEqual([m.devices.tolist() for m in p.meshes], [m.devices.tolist() for m in q.meshes])
    plan = lambda r: (r.order, r.pf, r.rs_at, r.rs_self, r.final_slots)
    self.assertEqual(plan(p), plan(q))
    self.assertNotEqual(dsp.plan_fingerprint(p), dsp.plan_fingerprint(q))
    self.assertNotIn("shard_mode", p.summary())
    self.assertIn("shard_mode=explicit", q.summary())

  def test_adam_math_optax_is_the_ports_chain(self):
    """adam_math='optax': the stage update is maxtext.experimental.mpmd.jax.deepseek.train's
    clip_by_global_norm / adamw_update / apply_updates -- the chain held
    bit-exact against MaxText -- bit for bit on bf16 weights, moments and
    gradients, clipped or not, over three steps."""
    from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

    scfg = tiny_xla(layers=4, fsdp=1, ep=2, adam_math="optax")
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16))
    p = pipe(scfg, n_micro=2, global_batch=4)
    st, s = p.stages[0], 0
    params, opts, _ = p.init(jax.random.key(1))
    tc = ds_train.TrainConfig(
        steps=10, learning_rate=1e-3, warmup_steps_fraction=0.0, learning_rate_final_fraction=1.0, mu_dtype=jnp.bfloat16
    )
    sched = ds_train.learning_rate_schedule(tc)
    ref_p = jax.tree.map(np.asarray, params[s])
    ref_o = ds_train.adamw_init(ref_p, tc)

    @jax.jit  # one program, as MaxText's train step (XLA fuses the bf16 chain the same way)
    def chain(g, o, w):
      updates, o = ds_train.adamw_update(ds_train.clip_by_global_norm(g, 1.0), o, w, tc, sched)
      return ds_train.apply_updates(w, updates), o

    got_p, got_o = params[s], opts[s]
    for t, mag in enumerate((3.0, 0.01, 3.0)):  # global norm above, below, above the clip
      keys = iter(jax.random.split(jax.random.key(10 + t), 64))
      g = jax.tree.map(lambda x: (mag * jax.random.normal(next(keys), x.shape)).astype(x.dtype), ref_p)
      norm = ds_train.global_norm(g)  # bf16, as MaxText's
      self.assertEqual(norm.dtype, jnp.bfloat16)
      ref_p, ref_o = chain(g, ref_o, ref_p)
      sq = [jnp.float32(norm) ** 2] + [jnp.float32(0)] * (p.S - 1)  # sqrt gives back that norm exactly
      gs = jax.tree.map(lambda x, sh: jax.device_put(x, sh), g, st.s_sh)
      got_p, got_o, gn = st.update(got_p, got_o, gs, sq, jnp.float32(sched(t)))
      self.assertEqual(float(gn), float(norm))
      assert_bitwise(self, got_p, ref_p, f"params step {t}")
      assert_bitwise(self, got_o["mu"], ref_o["mu"], f"mu step {t}")
      assert_bitwise(self, got_o["nu"], ref_o["nu"], f"nu step {t}")
    with self.assertRaisesRegex(ValueError, "adam_math"):
      tiny_xla(layers=4, adam_math="sgd")

  def test_grad_dtype_casts_the_reduced_gradient(self):
    """grad_dtype (--grad-dtype, MaxText grad_dtype): the reduced gradient enters AdamW in it, and without one
    (None) under adam_math 'optax' in its weight's dtype.  fp32 weights and a bf16 reduce-scatter -- MaxText's
    weight_dtype=float32 grad_dtype=float32, which reduce-scatters in bf16 and casts after -- is
    maxtext.experimental.mpmd.jax.deepseek.train's chain on the bf16 gradient cast to fp32, bit for bit, clipped or
    not, over three
    steps, under grad_dtype float32 or None: the update of an fp32 reduce-scatter handed that gradient, not
    grad_dtype bfloat16's (the optax chain in the bf16 gradient's dtype).  Under adam_math 'fp32' grad_dtype
    bfloat16 rounds an fp32 reduce-scatter's gradient to bf16 first.  Every program here compiles without XLA's
    excess precision (testing.NO_EXCESS_PRECISION): with it, jax 0.12.0.1's XLA:CPU kept the bf16 product b1 * mu
    of the bf16 moments unrounded in the stage update, whose bf16 gradient's cast sits in the same fusion, and not in
    the chain -- the router kernels' weights an fp32 ulp of the update apart in 5% of their elements."""
    from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

    base = tiny_xla(layers=4, fsdp=1, ep=2, adam_math="optax", rs_dtype=jnp.bfloat16)
    self.assertEqual(jnp.dtype(base.model.weight_dtype), jnp.float32)
    stage = lambda **kw: pipe(dataclasses.replace(base, **kw), n_micro=2, global_batch=4)
    p = stage(grad_dtype=jnp.float32)
    sts = {
        "cast": p.stages[0],
        "none": stage().stages[0],
        "rs32": stage(rs_dtype=jnp.float32).stages[0],
        "bf16": stage(grad_dtype=jnp.bfloat16).stages[0],
    }
    self.assertEqual((sts["cast"].scfg.grad_dtype, sts["none"].scfg.grad_dtype), (jnp.float32, None))
    params, opts, _ = p.init(jax.random.key(1))
    host = lambda t: jax.tree.map(np.asarray, t)
    put = lambda t, sh: jax.tree.map(jax.device_put, t, sh)  # a fresh copy: the updates donate theirs
    tc = ds_train.TrainConfig(
        steps=10, learning_rate=1e-3, warmup_steps_fraction=0.0, learning_rate_final_fraction=1.0, mu_dtype=jnp.bfloat16
    )
    sched = ds_train.learning_rate_schedule(tc)
    ref_p = host(params[0])
    ref_o = ds_train.adamw_init(ref_p, tc)
    state = {k: (host(params[0]), host(opts[0])) for k in sts}

    @functools.partial(jax.jit, compiler_options=testing.NO_EXCESS_PRECISION)
    def chain(g, o, w):
      updates, o = ds_train.adamw_update(ds_train.clip_by_global_norm(g, 1.0), o, w, tc, sched)
      return ds_train.apply_updates(w, updates), o

    exact = {}

    def update(st, *args):  # st.update compiled as `chain` is, once a stage
      if id(st) not in exact:
        exact[id(st)] = st.update.lower(*args).compile(compiler_options=testing.NO_EXCESS_PRECISION)
      return exact[id(st)](*args)

    for t, mag in enumerate((3.0, 1e-4, 3.0)):  # global norm above (6e3), below (0.2), above the clip
      keys = iter(jax.random.split(jax.random.key(20 + t), 64))
      g16 = jax.tree.map(lambda x: (mag * jax.random.normal(next(keys), x.shape)).astype(jnp.bfloat16), ref_p)
      g32 = jax.tree.map(lambda x: x.astype(jnp.float32), g16)
      norm = ds_train.global_norm(g32)  # fp32, as MaxText's under grad_dtype=float32
      self.assertEqual(bool(norm < 1.0), t == 1)
      ref_p, ref_o = chain(g32, ref_o, ref_p)
      sq = [norm**2] + [jnp.float32(0)] * (p.S - 1)
      for k, st in sts.items():
        w, o = state[k]
        w, o, gn = update(
            st, put(w, st.s_sh), put(o, st.opt_sh), put(g32 if k == "rs32" else g16, st.s_sh), sq, jnp.float32(sched(t))
        )
        self.assertEqual(float(gn), float(norm), k)
        state[k] = (host(w), host(o))
      for k in ("cast", "none", "rs32"):
        assert_bitwise(self, state[k][0], ref_p, f"{k} params step {t}")
        assert_bitwise(self, state[k][1]["mu"], ref_o["mu"], f"{k} mu step {t}")
        assert_bitwise(self, state[k][1]["nu"], ref_o["nu"], f"{k} nu step {t}")
    self.assertFalse(all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state["bf16"][0]), jax.tree.leaves(ref_p))))
    # adam_math 'fp32': grad_dtype bfloat16 is the update of the gradient rounded to bf16
    fp = {
        k: stage(adam_math="fp32", rs_dtype=jnp.float32, **kw).stages[0]
        for k, kw in (("round", dict(grad_dtype=jnp.bfloat16)), ("none", {}))
    }
    keys = iter(jax.random.split(jax.random.key(30), 64))
    g = jax.tree.map(lambda x: 0.01 * jax.random.normal(next(keys), x.shape, jnp.float32), ref_p)
    rounded = jax.tree.map(lambda x: x.astype(jnp.bfloat16).astype(jnp.float32), g)
    sq = [ds_train.global_norm(g) ** 2] + [jnp.float32(0)] * (p.S - 1)
    run = lambda st, grads: host(
        update(
            st, put(host(params[0]), st.s_sh), put(host(opts[0]), st.opt_sh), put(grads, st.s_sh), sq, jnp.float32(1e-3)
        )[:2]
    )
    got = run(fp["round"], g)
    assert_bitwise(self, got, run(fp["round"], rounded), "fp32 adam_math, grad_dtype bfloat16: only bf16(g) counts")
    # ... and is what the stage without grad_dtype does with bf16(g): to 2 fp32 ulps of each leaf's largest weight,
    # not bit for bit -- the two programs differ by the cast's fusion, and on jax 0.12.0.1's XLA:CPU (AVX-512 with
    # its default 256-bit vectors, excess precision or not) the router kernels' updated weights came out up to a
    # quarter of the bound apart in 10% of their elements: none with --xla_cpu_max_isa=AVX2, or on jax 0.11.1.  g
    # itself, unrounded, moves the weight matrices 4-32x this bound (3.9-31.6x measured) but the 1-D leaves at most
    # half of it: there the bitwise check above, only bf16(g) counts, is what catches a rounding left out.
    want = run(fp["none"], rounded)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
      np.testing.assert_allclose(a, b, rtol=0, atol=2 * np.finfo(np.float32).eps * float(np.max(np.abs(b))))
    self.assertFalse(
        all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(run(fp["none"], g)), jax.tree.leaves(want)))
    )
    with self.assertRaisesRegex(ValueError, "grad_dtype must be"):
      tiny_xla(layers=4, grad_dtype=jnp.int32)

  def test_adam_math_optax_casts_gradients_to_the_weights_dtype(self):
    """adam_math='optax' with fp32 weights: a bf16 gradient (--rs-dtype bfloat16) enters the chain cast to fp32, as
    MaxText's does -- its bf16 reduce-scatter's result takes the fp32 parameter's dtype before optax."""
    from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

    scfg = tiny_xla(layers=4, fsdp=1, ep=2, adam_math="optax", grad_clip=0.0)
    p = pipe(scfg, n_micro=2, global_batch=4)
    st, s = p.stages[0], 0
    params, opts, _ = p.init(jax.random.key(1))
    self.assertEqual({x.dtype for x in jax.tree.leaves(params[s])}, {np.dtype(jnp.float32)})
    tc = ds_train.TrainConfig(
        steps=10, learning_rate=1e-3, warmup_steps_fraction=0.0, learning_rate_final_fraction=1.0, mu_dtype=jnp.bfloat16
    )
    sched = ds_train.learning_rate_schedule(tc)
    ref_p = jax.tree.map(np.asarray, params[s])
    keys = iter(jax.random.split(jax.random.key(10), 64))
    g = jax.tree.map(lambda x: (3.0 * jax.random.normal(next(keys), x.shape)).astype(jnp.bfloat16), ref_p)
    updates, _ = jax.jit(lambda g, o, w: ds_train.adamw_update(g, o, w, tc, sched))(
        jax.tree.map(lambda x: x.astype(jnp.float32), g), ds_train.adamw_init(ref_p, tc), ref_p
    )
    want = ds_train.apply_updates(ref_p, updates)
    sq = [jnp.float32(1.0)] + [jnp.float32(0)] * (p.S - 1)
    gs = jax.tree.map(lambda x, sh: jax.device_put(x, sh), g, st.s_sh)
    got, _, _ = st.update(params[s], opts[s], gs, sq, jnp.float32(sched(0)))
    # Not bit for bit in fp32: the two programs fuse the chain differently (last-bit differences, 3.7e-9 at most
    # here); without the cast the chain would run in bf16 and miss by about the bf16 rounding of the updates, 1e-6.
    for i, (x, y) in enumerate(zip(jax.tree.leaves(got), jax.tree.leaves(want))):
      np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-8, err_msg=f"params leaf {i}")

  def test_grad_clip(self):
    """grad_clip: MaxText's gradient_clipping_threshold.  0 does not clip -- a norm-3 gradient updates as the
    unclipped chain does -- and a threshold of 2 clips it as clip_by_global_norm(2) does."""
    from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

    for clip in (0.0, 2.0):
      scfg = tiny_xla(layers=4, fsdp=1, ep=2, adam_math="optax", grad_clip=clip)
      scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16))
      p = pipe(scfg, n_micro=2, global_batch=4)
      st, s = p.stages[0], 0
      params, opts, _ = p.init(jax.random.key(1))
      tc = ds_train.TrainConfig(
          steps=10, learning_rate=1e-3, warmup_steps_fraction=0.0, learning_rate_final_fraction=1.0, mu_dtype=jnp.bfloat16
      )
      sched = ds_train.learning_rate_schedule(tc)
      ref_p = jax.tree.map(np.asarray, params[s])
      keys = iter(jax.random.split(jax.random.key(10), 64))
      g = jax.tree.map(lambda x: (3.0 * jax.random.normal(next(keys), x.shape)).astype(x.dtype), ref_p)
      norm = ds_train.global_norm(g)
      clipped = ds_train.clip_by_global_norm(g, clip) if clip else g
      updates, _ = jax.jit(lambda g, o, w: ds_train.adamw_update(g, o, w, tc, sched))(
          clipped, ds_train.adamw_init(ref_p, tc), ref_p
      )
      want = ds_train.apply_updates(ref_p, updates)
      sq = [jnp.float32(norm) ** 2] + [jnp.float32(0)] * (p.S - 1)
      gs = jax.tree.map(lambda x, sh: jax.device_put(x, sh), g, st.s_sh)
      got, _, _ = st.update(params[s], opts[s], gs, sq, jnp.float32(sched(0)))
      assert_bitwise(self, got, want, f"params, clip {clip}")
    with self.assertRaisesRegex(ValueError, "grad_clip"):
      tiny_xla(layers=4, grad_clip=-1.0)

  def test_init_from_chunk_files_and_token_batches(self):
    """--init-from: chunk files as `write_chunk_files` writes them (and
    maxtext_parity: '.'-joined keys, bfloat16 as uint16 bits) load back to the
    same weights bit for bit, in another placement too.  --data: a [gbs,
    seq + 1] token array becomes inputs [:, :-1] on the first stage and
    targets [:, 1:] on the last."""
    scfg = tiny_xla(layers=4, fsdp=1, ep=2)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16))
    p = pipe(scfg, n_micro=2, global_batch=4)
    params, _, _ = p.init(jax.random.key(3))
    want = p.chunk_trees(params)
    self.assertEqual({v.dtype for v in jax.tree.leaves(want)}, {jnp.dtype(jnp.bfloat16)})
    with tempfile.TemporaryDirectory() as d:
      dsp.write_chunk_files(d, want)
      with np.load(f"{d}/chunk3.npz") as z:  # the format the files promise maxtext_parity
        self.assertEqual(
            sorted(z.files),
            sorted(".".join(k.key for k in path) for path, _ in jax.tree_util.tree_flatten_with_path(want[3])[0]),
        )
        self.assertEqual({z[k].dtype for k in z.files}, {np.dtype(np.uint16)})
      for q in (p, pipe(scfg, n_micro=2, global_batch=4, placement="v")):
        got = q.chunk_trees(q.params_from_chunks(q.chunk_trees_from(d)))
        assert_bitwise(self, got, want, q.placement)
        self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    toks = np.arange(p.gbs * (p.seq + 1), dtype=np.int32).reshape(p.gbs, p.seq + 1) % 512
    x, y = p.token_batch(toks)
    for m in range(p.M):
      np.testing.assert_array_equal(np.asarray(x[m]), toks[m * p.mbs : (m + 1) * p.mbs, :-1])
      np.testing.assert_array_equal(np.asarray(y[m]), toks[m * p.mbs : (m + 1) * p.mbs, 1:])
    with self.assertRaisesRegex(ValueError, "seq \\+ 1"):
      p.token_batch(toks[:, :-1])
    # the MaxText batch: [gbs, seq + 1] int32 in [0, vocab), drawn under the PRNG MaxText uses
    t = p.maxtext_tokens()
    self.assertEqual((t.shape, t.dtype), ((p.gbs, p.seq + 1), np.int32))
    self.assertTrue(0 <= t.min() and t.max() < p.scfg.model.vocab_size)
    np.testing.assert_array_equal(t, p.maxtext_tokens(dsp.MAXTEXT_BATCH_PRNG))
    self.assertFalse(np.array_equal(p.maxtext_tokens("threefry2x32"), p.maxtext_tokens("unsafe_rbg")))

  def test_chunk_files_per_process(self):
    """Under multi-controller JAX a process places only its own stages
    (`placed`), and reads only their chunk files: with only stage 1's files
    on disk, a process of stage 1 loads them -- bit for bit, bfloat16 too --
    and None for the rest, which also come back None from `chunk_trees`;
    one holding every stage needs every file.  A tree key with a '.' cannot
    be written (it would not read back)."""
    scfg = tiny_xla(layers=4, fsdp=1, ep=2)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16))
    p = pipe(scfg, n_micro=2, global_batch=4)
    want = host_weights(p)
    mine = [ch.stage == 1 for ch in p.chunks]
    self.assertEqual(mine, [False, True, False, True])
    d = self.enterContext(tempfile.TemporaryDirectory())
    dsp.write_chunk_files(d, want)
    for ch in p.chunks:
      if not mine[ch.index]:
        os.remove(f"{d}/chunk{ch.index}.npz")
    # a process of a multi-controller run holding stage 1 (in one controller `placed` is not looked at)
    with mock.patch.object(p, "multi", True), mock.patch.object(p, "placed", frozenset({1})):
      got = p.chunk_trees_from(d)
      self.assertEqual([t is not None for t in got], mine)
      assert_bitwise(self, [t for t in got if t is not None], [t for t, m in zip(want, mine) if m], "stage 1")
      placed = p.chunk_trees(p.params_from_chunks(got))
    self.assertEqual([t is not None for t in placed], mine)
    assert_bitwise(self, [t for t in placed if t is not None], [t for t, m in zip(want, mine) if m], "placed")
    with self.assertRaises(FileNotFoundError):
      p.chunk_trees_from(d)
    for bad in ({"layers.moe": np.zeros(2)}, {"layers": {3: np.zeros(2)}}):
      with self.subTest(tree=bad), self.assertRaisesRegex(ValueError, "want a str without"):
        dsp.write_chunk_files(d, [bad])

  def test_weights_and_tokens_are_checked(self):
    """What --init-from and --data hand the pipeline is checked before it is
    placed: every chunk's weights must be the model's leaves at the model's
    shapes (device_put would take any divisible shape), naming the chunk and
    the leaf; tokens must be integers in [0, vocab) of shape [gbs, seq + 1]."""
    p = pipe(tiny_xla(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4)
    trees = host_weights(p)
    wo, extra = "['layers']['moe']['wo']", "['extra']"
    for change, want in (
        (lambda t: t["layers"]["moe"].pop("wo"), f"missing {[wo]}"),
        (lambda t: t.update(extra=np.zeros(3)), f"unexpected {[extra]}"),
        (
            lambda t: t["layers"]["moe"].update(wo=t["layers"]["moe"]["wo"][:, :4]),
            f"{wo} (1, 4, 256, 256) (model (1, 8, 256, 256))",
        ),
    ):
      bad = jax.tree.map(lambda x: x, trees)  # new containers, the same leaves
      change(bad[1])
      with self.subTest(want):
        with self.assertRaises(ValueError) as cm:
          p.params_from_chunks(bad)
        self.assertTrue(str(cm.exception).startswith("chunk 1: "), cm.exception)
        self.assertIn(want, str(cm.exception))
    toks = np.arange(p.gbs * (p.seq + 1)).reshape(p.gbs, p.seq + 1) % p.scfg.model.vocab_size
    for bad, msg in (
        (toks.astype(np.float32), "integer tokens"),
        (toks - 1, r"outside \[0, 512\): min -1"),
        (toks + 1, r"outside \[0, 512\).* max 512"),
        (toks[:, 1:], r"\[gbs, seq \+ 1\]"),
    ):
      with self.subTest(msg), self.assertRaisesRegex(ValueError, msg):
        p.token_batch(bad)
    x, y = p.token_batch(toks.astype(np.int64))  # any integer dtype: int32 on the devices
    self.assertEqual({a.dtype for a in x + y}, {jnp.dtype(jnp.int32)})

  def test_unsupported_options_are_refused(self):
    """What the pipeline does not run -- --schedule greedy (with a --cap), the F&B-pair schedules dualpipev and
    1f1b_pairs, --fsdp-collectives sparsecore, --sync-rs, --tiles published, a --batch-split-impl other than
    lineage and native -- is refused: by argparse, and by build_schedule, Pipeline and StageConfig.  --pair-stages
    / --pair-gain are interleaved's only (test_interleaved_pair_plans_valid), --batch-split-impl native is
    DeepSeek-V3's off TPU only (gpu_port_test's test_batch_split_impl)."""
    ap = dsp.make_parser()
    for bad in (
        ["--schedule", "greedy"],
        ["--cap", "2,1"],
        ["--fsdp-collectives", "sparsecore"],
        ["--schedule", "dualpipev"],
        ["--schedule", "1f1b_pairs"],
        ["--sync-rs", "fold"],
        ["--tiles", "published"],
        ["--batch-split-impl", "other"],
    ):
      err = io.StringIO()
      with self.subTest(bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
        ap.parse_args(TINY_ARGS + bad)
      self.assertRegex(err.getvalue(), "invalid choice|unrecognized arguments")
    for name in ("greedy", "dualpipev", "1f1b_pairs"):
      with self.assertRaisesRegex(ValueError, f"unknown schedule '{name}'"):
        dsp.build_schedule(name, 4, 4, sched.placement(2, 2))
    with self.assertRaisesRegex(ValueError, "unknown schedule 'dualpipev'"):
      pipe(tiny(layers=4, fsdp=1, ep=2), schedule="dualpipev", placement="v")
    with self.assertRaises(TypeError):  # 'end', the plan that stayed, is the only one
      pipe(tiny(layers=4, fsdp=1, ep=2), sync_rs="end")
    self.assertEqual(sorted(ds_ep.TILES), ["clamped", "v6e"])
    with self.assertRaisesRegex(ValueError, "fsdp_collectives must be 'ppermute' or 'xla'"):
      tiny(fsdp_collectives="sparsecore")
    with self.assertRaisesRegex(ValueError, r"batch_split_impl must be one of \('lineage', 'native'\), got 'other'"):
      tiny_xla(batch_split=True, batch_split_impl="other")

  def test_native_transport_warning_kinds(self):
    """The --transport native warning's device kinds: the v6e and v5p
    spellings libtpu and Pathways report, not v5e or TPU7x."""
    for kind, want in (
        ("TPU v6 lite", True),
        ("TPU v6e", True),
        ("TPU v5", True),
        ("TPU v5p", True),
        ("TPU v5 lite", False),
        ("TPU7x", False),
        ("cpu", False),
    ):
      with self.subTest(kind=kind):
        self.assertEqual(dsp.native_transport_aborts(kind), want)

  def test_cli_checks_its_flags_first(self):
    """main refuses what it cannot run before compiling anything: a negative
    --max-ahead, an unknown --data, and --bench or --profile with no step to
    time or trace (--bench would divide by the count after the compile and
    warm-up) before it builds; float32 with the ragged sort where no
    SparseCore runs it (`float32_ragged_sort_problem`), and an --init-from
    chunk file or --data npy: file that is not there (`missing_input`),
    before the precompile.  --init-from weights that are not the model's stop
    it with a message, not a traceback."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    for bad, msg in (
        (["--max-ahead", "-1"], "--max-ahead must be >= 0"),
        (["--data", "bogus"], "--data must be"),
        (["--bench", "--steps", "0"], r"--steps must be >= 1 with --bench .*, not 0"),
        (["--bench", "--steps", "-2"], r"--steps must be >= 1 with --bench .*, not -2"),
        (["--profile", f"{d}/xprof", "--profile-steps", "0"], r"--profile-steps must be >= 1 with --profile .*, not 0"),
    ):
      with self.subTest(bad=bad), self.assertRaisesRegex(SystemExit, msg):
        dsp.main(TINY_ARGS + bad)
    for bad, msg in (
        (
            ["--dtype", "float32", "--ragged-sort", "on"],
            r"^--dtype float32 with the ragged sort .* add --ragged-sort off",
        ),
        (["--init-from", f"{d}/"], f"^--init-from {d}/: no such file {d}/chunk0.npz$"),
        (["--data", f"npy:{d}/tokens.npy"], f"^--data npy:{d}/tokens.npy: no such file$"),
    ):
      with self.subTest(bad=bad), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, msg):
        dsp.main(TINY_ARGS + bad)
    trees = host_weights(dsp.build(dsp.make_parser().parse_args(TINY_ARGS)))
    trees[0]["token_embedder"]["embedding"] = trees[0]["token_embedder"]["embedding"][:256]
    dsp.write_chunk_files(d, trees)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        self.assertRaisesRegex(SystemExit, f"--init-from {d}: chunk 0: weight shapes .*'embedding'"),
    ):
      dsp.main(TINY_ARGS + ["--no-precompile", "--init-from", d])

  def test_float32_ragged_sort_needs_sparsecore(self):
    """float32 with the ragged sort runs where MaxText's ragged_gather_reduce
    takes its SparseCore kernel.  Its JAX fallback -- off TPU, or on a TPU
    without SparseCores (v5e) -- returns bfloat16, so the first backward would
    fail to lower: a message naming --ragged-sort off instead.  bfloat16, or
    the argsort dispatch, runs anywhere."""
    from jax.experimental.pallas import tpu as pltpu

    parse = lambda *extra: dsp.make_parser().parse_args(TINY_ARGS + list(extra))
    problem = lambda a: dsp.float32_ragged_sort_problem(a, dsp.build(a).scfg)
    fp32 = parse("--dtype", "float32", "--ragged-sort", "auto")  # auto: on at TINY_ARGS' --ep 2
    self.assertRegex(problem(fp32), r"^--dtype float32 with the ragged sort \(--ragged-sort auto at --ep 2\) .* not cpu")
    self.assertIsNone(problem(parse("--ragged-sort", "on")))
    self.assertIsNone(problem(parse("--dtype", "float32")))  # TINY_ARGS' --ragged-sort off
    scfg = dsp.build(fp32).scfg
    for sparse_core, refused in ((types.SimpleNamespace(num_cores=2), False), (None, True)):
      with (
          self.subTest(sparse_core=sparse_core),
          mock.patch.object(pltpu, "is_tpu_device", lambda: True),
          mock.patch.object(pltpu, "get_tpu_info", lambda sc=sparse_core: types.SimpleNamespace(sparse_core=sc)),
      ):
        self.assertEqual(dsp.float32_ragged_sort_problem(fp32, scfg) is not None, refused)

  def test_storage_dtype_flags_are_floating(self):
    """--rs-dtype, --acc-dtype and --weight-dtype take any name jnp.dtype reads of a floating dtype, kept as given
    (the run record's config), and refuse at argparse an integer one, which trains nothing (every gradient,
    accumulator or weight truncated to 0), and a name that is no dtype."""
    ap = dsp.make_parser()
    for flag, field in (("--rs-dtype", "rs_dtype"), ("--acc-dtype", "acc_dtype"), ("--weight-dtype", "weight_dtype")):
      for name, want in (("bfloat16", jnp.bfloat16), ("float32", jnp.float32), ("f4", jnp.float32)):
        with self.subTest(flag=flag, name=name):
          a = ap.parse_args([*TINY_ARGS, flag, name])
          self.assertEqual(getattr(a, field), name)
          got = dsp.model_config(a).weight_dtype if field == "weight_dtype" else getattr(dsp.build(a).scfg, field)
          self.assertEqual(got, want)
      for bad, msg in (("int32", "want a floating dtype"), ("bool", "want a floating dtype"), ("bf16", "is not a dtype")):
        err = io.StringIO()
        with self.subTest(flag=flag, bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
          ap.parse_args([*TINY_ARGS, flag, bad])
        self.assertIn(f"argument {flag}: ", err.getvalue())
        self.assertIn(msg, err.getvalue())

  def test_costs_names_its_tables(self):
    """--costs is a table's name, a JSON object or @ a file holding one; anything else -- a table name misspelt,
    say -- stops with the flag and the tables named, not a JSON decoder's error."""
    self.assertIsNone(dsp.load_costs(""))
    self.assertEqual(dsp.load_costs("v6e"), dict(dsp.COST_TABLES["v6e"]))
    self.assertEqual(dsp.load_costs(' {"F0": 1.5, "_source": "notes"}'), {"F0": 1.5})
    d = self.enterContext(tempfile.TemporaryDirectory())
    with open(f"{d}/costs.json", "w") as f:
      f.write('{"B0": 2.5}')
    self.assertEqual(dsp.load_costs(f"@{d}/costs.json"), {"B0": 2.5})
    for bad in ("v6e_tile", "tpu7x,v6e"):
      with (
          self.subTest(bad=bad),
          self.assertRaisesRegex(
              ValueError, f"^--costs '{bad}': not a JSON object, @file or one of the tables tpu7x, v6e, v6e_tiles$"
          ),
      ):
        dsp.load_costs(bad)

  def test_precompile_names_the_failing_program(self):
    """A program that fails to lower or compile stops `precompile` at the
    first one, and the error says which program of which stage."""
    p = pipe(tiny_xla(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4)
    c, desc = p.dispatched()[0]
    with mock.patch.object(p, "_desc_avals", lambda c, desc: ()), self.assertRaises(Exception) as cm:
      p.precompile(threads=1)  # no arguments: every lowering fails at once
    self.assertEqual(
        [n for n in cm.exception.__notes__ if n.startswith("precompile:")],
        [f"precompile: lowering {p.progs[c].program(desc).__name__} (stage {p.place[c]}) failed"],
    )
    self.assertEqual(p.compiled, [])

  @testing.needs_devices(8, "two stages of fsdp 2 x expert 2: the layouts differ in where each axis's pairs go")
  def test_layouts(self):
    # eight devices, whatever the host has: the physical layouts cut the whole device set they get, and
    # 'slices' runs a stage on each of two fake slices of it (slices_layout_test.py has the rest)
    devs = jax.devices()[:8]
    with mock.patch.dict(os.environ, {"MPMD_FAKE_SLICES": "2"}):
      meshes = {layout: dsp.stage_meshes(2, 2, 2, devices=devs, layout=layout) for layout in dsp.LAYOUTS}
    for layout, ms in meshes.items():
      with self.subTest(layout=layout):
        self.assertEqual([dict(m.shape) for m in ms], [{"fsdp": 2, "expert": 2}] * 2)
        ids = [d.id for m in ms for d in m.devices.flat]
        self.assertEqual(sorted(ids), list(range(8)))
    # fsdp_on_chip is ep_on_chip transposed: the pair on one chip is the other axis
    for a, b in zip(meshes["fsdp_on_chip"], meshes["ep_on_chip"]):
      np.testing.assert_array_equal(a.devices, b.devices.T)
    # physical_origin_last: physical's strips rotated until the one holding chip (0,0,0) -- on CPU
    # devices, which have no coordinates, device 0 -- runs the last stage
    phys, rot = meshes["physical"], meshes["physical_origin_last"]
    self.assertIn(0, [d.id for d in phys[0].devices.flat])
    for a, b in zip(rot, phys[1:] + phys[:1]):
      np.testing.assert_array_equal(a.devices, b.devices)
    four = dsp.stage_meshes(4, 2, 1, devices=devs, layout="physical_origin_last")
    self.assertEqual(
        [[d.id for d in m.devices.flat] for m in four][-1],
        [d.id for d in dsp.stage_meshes(4, 2, 1, devices=devs, layout="physical")[0].devices.flat],
    )
    self.assertEqual(dsp.sync_collective_stages(rot), frozenset())  # CPU devices
    for bad in ("fsdp-on-chip", "EP_ON_CHIP", ""):
      with self.subTest(layout=bad), self.assertRaises(ValueError):
        dsp.stage_meshes(2, 2, 2, devices=devs, layout=bad)

  def test_physical_layouts_on_a_v6e_256(self):
    """The TPU side of --layout physical / physical_origin_last and
    `sync_collective_stages`, on fake v6e-256 devices (16x16 chips, a host
    per 2x2): at 4 x 64 'physical' cuts four solid 16x4 strips of 16 whole
    hosts, chip (0,0,0)'s first, and libtpu keeps that stage's collectives
    synchronous; 'physical_origin_last' runs the same strips in the same
    cyclic order with that one last.  A device set without chip (0,0,0) stays
    in 'physical' order with no synchronous stage, and a stage of one device
    has no collective to keep synchronous."""
    devs = fake_v6e()

    def strip(m):  # (x range, y range) of a stage's chips if they are a solid box, its hosts
      xs, ys = sorted({d.coords[0] for d in m.devices.flat}), sorted({d.coords[1] for d in m.devices.flat})
      self.assertEqual(
          (len(xs) * len(ys), xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1),
          (m.devices.size, len(xs), len(ys)),
          "not a solid box",
      )
      return (xs[0], xs[-1]), (ys[0], ys[-1]), len({d.process_index for d in m.devices.flat})

    phys = dsp.stage_meshes(4, 64, 1, devices=devs, layout="physical")
    rot = dsp.stage_meshes(4, 64, 1, devices=devs, layout="physical_origin_last")
    strips = [((0, 15), (4 * s, 4 * s + 3), 16) for s in range(4)]
    self.assertEqual([strip(m) for m in phys], strips)
    self.assertEqual([any(map(dsp.on_origin_chip, m.devices.flat)) for m in phys], [True, False, False, False])
    for a, b in zip(rot, phys[1:] + phys[:1]):  # the same meshes, device order and all
      self.assertEqual(a.device_ids.tolist(), b.device_ids.tolist())
    self.assertEqual(dsp.sync_collective_stages(phys), frozenset({0}))
    self.assertEqual(dsp.sync_collective_stages(rot), frozenset({3}))
    self.assertEqual((dsp.shared_stage_hosts(phys), dsp.shared_stage_hosts(rot)), ([], []))
    # the y >= 8 half holds no chip (0,0,0): not rotated (by its lowest device id, say), nothing synchronous
    upper = [d for d in devs if d.coords[1] >= 8]
    half = {
        layout: dsp.stage_meshes(2, 64, 1, devices=upper, layout=layout)
        for layout in ("physical", "physical_origin_last")
    }
    self.assertEqual([strip(m) for m in half["physical"]], [((0, 15), (8, 11), 16), ((0, 15), (12, 15), 16)])
    self.assertEqual(
        [m.device_ids.tolist() for m in half["physical_origin_last"]], [m.device_ids.tolist() for m in half["physical"]]
    )
    self.assertEqual(dsp.sync_collective_stages(half["physical_origin_last"]), frozenset())
    # host 0's four chips: as two 2-device stages the first holds chip (0,0,0) and is synchronous (both
    # on host 0: shared); as four 1-device stages none is
    self.assertEqual(dsp.sync_collective_stages(dsp.stage_meshes(2, 2, 1, devices=devs[:4])), frozenset({0}))
    self.assertEqual(dsp.shared_stage_hosts(dsp.stage_meshes(2, 2, 1, devices=devs[:4])), [(0, 1, [0])])
    self.assertEqual(dsp.sync_collective_stages(dsp.stage_meshes(4, 1, 1, devices=devs[:4])), frozenset())


class FakeCoordinationClient:
  """The coordination service's key-value store and barrier, as
  check_plan_agreement uses them, for one process: `published` is what the
  other processes wrote before this one got to it."""

  def __init__(self, published=()):
    self.kv, self.calls = dict(published), []

  def key_value_set(self, key, value):
    self.calls.append(("set", key))
    self.kv[key] = value

  def wait_at_barrier(self, name, timeout_ms):
    self.calls.append(("barrier", name))

  def key_value_dir_get(self, prefix):
    self.calls.append(("dir_get", prefix))
    return [(k, v) for k, v in sorted(self.kv.items()) if k.startswith(prefix)]

  def blocking_key_value_get(self, key, timeout_ms):
    self.calls.append(("get", key))
    return self.kv[key]


@testing.needs_devices(4)
class MultiController(unittest.TestCase):
  """What a multi-controller run decides before it compiles, checked in one
  process: the plan agreement (over a fake coordination service), the
  process sets of --profile-procs / --json-procs, the summary's marks for
  another process's stages and init_distributed's environment.  Nothing is
  compiled."""

  def test_plan_agreement(self):
    """Every process publishes its plan's hash, waits for all the others and
    compares every hash -- process 0 too -- so all stop together, each naming
    which processes built which plan.  The hash covers the order and plan,
    the transport, the placement, the stage meshes' devices, the prefetch
    pins, the gathers' and reduce-scatters' axes and what main adds to it
    (every flag)."""
    from jax._src import distributed

    scfg = tiny(layers=4, fsdp=1, ep=2)
    p = pipe(scfg)
    h = lambda q: hashlib.sha256(repr(dsp.plan_fingerprint(q)).encode()).hexdigest()

    def agree(q, published, pid):
      client = FakeCoordinationClient(published)
      with (
          mock.patch.object(distributed.global_state, "client", client),
          mock.patch.object(jax, "process_index", lambda: pid),
      ):
        return dsp.check_plan_agreement(q), client

    got, client = agree(p, {"mpmd/plan/0": h(p), "mpmd/plan/2": h(p)}, 1)
    self.assertEqual(got, h(p))
    self.assertEqual(client.kv["mpmd/plan/1"], h(p))
    self.assertEqual(
        client.calls, [("set", "mpmd/plan/1"), ("barrier", "mpmd_plan_agreement"), ("dir_get", "mpmd/plan/")]
    )
    other = pipe(scfg, schedule="gpipe")
    self.assertNotEqual(h(other), h(p))
    for pid, rest in ((1, [0, 2]), (0, [1, 2])):
      with self.subTest(process=pid):
        with self.assertRaises(RuntimeError) as cm:
          agree(p, {f"mpmd/plan/{q}": h(other) for q in rest}, pid)
        msg = str(cm.exception)
        self.assertIn("the processes built 2 different plans", msg)
        self.assertIn(f"{h(p)[:12]}: processes [{pid}]", msg)
        self.assertIn(f"{h(other)[:12]}: processes {rest}", msg)
    # what the hash sees: the same flags give the same plan; each of these changes it
    fp = dsp.plan_fingerprint
    self.assertEqual(fp(pipe(scfg)), fp(p))
    for what, q in (
        ("transport", pipe(scfg, transport="ppermute")),
        ("placement", pipe(scfg, placement="v")),
        ("stage devices", pipe(scfg, meshes=p.meshes[::-1])),
        ("prefetch pins", pipe(scfg, prefetch_pin="hn=moe.wi_0")),
        ("gather axes", pipe(dataclasses.replace(scfg, gather_axes="each"))),
        ("reduce-scatter axes", pipe(dataclasses.replace(scfg, rs_axes="each"))),
    ):
      self.assertNotEqual(fp(q), fp(p), what)
      if what != "placement":
        self.assertEqual((q.order, q.rs_at, q.rs_self), (p.order, p.rs_at, p.rs_self), what)  # that alone moved
    self.assertNotEqual(fp(p, [("steps", 10)]), fp(p, [("steps", 11)]))

  def test_missing_input_stops_every_process(self):
    """A file only some processes lack -- each checks the files it reads, its
    stages' chunks, maybe in a directory local to its host -- stops every
    process, not only those: each lacking one publishes its message, waits for
    all the others and reads every one, so all of them stop together, saying
    the same and naming the processes lacking each file.  No process lacking
    anything: nothing published, nothing to say."""
    from jax._src import distributed

    chunk, data = "--init-from D: no such file D/chunk1.npz", "--data npy:T: no such file"

    def check(pid, lacks):  # the store after the barrier: every other process's message, then this one's
      client = FakeCoordinationClient({f"mpmd/inputs/{q}": m for q, m in lacks.items() if m and q != pid})
      with (
          mock.patch.object(distributed.global_state, "client", client),
          mock.patch.object(jax, "process_index", lambda: pid),
          mock.patch.object(jax, "process_count", lambda: 4),
      ):
        return dsp.missing_input_anywhere(lacks[pid]), client

    got, client = check(0, dict.fromkeys(range(4)))
    self.assertIsNone(got)
    self.assertEqual(client.calls, [("barrier", "mpmd_inputs"), ("dir_get", "mpmd/inputs/")])
    lacks = {0: None, 1: data, 2: chunk, 3: chunk}
    said = set()
    for pid in range(4):
      got, client = check(pid, lacks)
      said.add(got)
      self.assertEqual(client.calls[0] == ("set", f"mpmd/inputs/{pid}"), lacks[pid] is not None, pid)
    self.assertEqual(
        said,
        {
            "input files missing on 3 of 4 processes, so every process stops: "
            f"{data} (process 1); {chunk} (processes 2-3)"
        },
    )

  @staticmethod
  def _program_agreement(prints, stage_procs, what="lowered"):
    """check_program_agreement as each process of a run sees it: every
    process's writes to the key-value store first (its barrier stops it
    there), then each one's check over all of them -> {process: its digests
    or its RuntimeError}, and the store."""
    from jax._src import distributed

    p = types.SimpleNamespace(stage_procs=[frozenset(s) for s in stage_procs])

    class Published(Exception):
      pass

    def run(pid, client):
      with (
          mock.patch.object(distributed.global_state, "client", client),
          mock.patch.object(jax, "process_index", lambda: pid),
          mock.patch.object(dsp, "_PROGRAM_KEYS", itertools.count()),
      ):
        return dsp.check_program_agreement(p, prints[pid], what)

    kv = {}
    for pid in prints:
      client = FakeCoordinationClient()
      client.wait_at_barrier = mock.Mock(side_effect=Published)
      try:
        run(pid, client)
      except Published:
        pass
      kv.update(client.kv)
    out = {}
    for pid in prints:
      try:
        out[pid] = run(pid, FakeCoordinationClient(kv))
      except RuntimeError as e:
        out[pid] = e
    return out, kv

  def test_program_agreement(self):
    """Before the first launch every process publishes a digest of its
    programs per stage, and its prints; after a barrier all compare every
    stage's digests, so where two processes of a stage lowered a program
    differently -- what halts the TPU runtime at that program's first launch
    -- all of them stop together, each naming the program, its two prints
    (with the counts that tell how they differ) and the processes holding
    each.  A boundary copy's program (stage (0, 1)) is compared over both
    stages' processes; a process that lowered nothing for its stage is
    named too."""

    def lw(sha, consts=336):
      return {"sha": sha, "bytes": 1000 + consts, "consts": consts, "funcs": 17}

    def prints(stage, fb="aaaa", fb_consts=336):
      own = (
          [("jit_fwd_c0", 0, "f0f0", lw("f0f0")), ("jit_bwd_c0", 0, "b0b0", lw("b0b0"))]
          if stage == 0
          else [("jit_fb_c1", 1, fb, lw(fb, fb_consts)), ("jit_update_s1", 1, "u1u1", lw("u1u1"))]
      )
      return own + [("xfer_s0_to_s1", (0, 1), "x01x", lw("x01x"))]

    procs = [{0, 1}, {2, 3}]
    same = {q: prints(0 if q < 2 else 1) for q in range(4)}
    out, kv = self._program_agreement(same, procs)
    self.assertEqual(sorted(out[0]), ["0", "0-1", "1"])
    self.assertTrue(all(out[q] == out[0] for q in range(4)), out)
    self.assertEqual(json.loads(kv["mpmd/programs/lowered/0/prints/2"])[0], ["jit_fb_c1", "1", "aaaa", lw("aaaa")])
    # process 3 lowered one table copy more into jit_fb_c1
    out, _ = self._program_agreement({**same, 3: prints(1, fb="bbbb", fb_consts=337)}, procs)
    msgs = {q: str(out[q]) for q in range(4)}
    self.assertTrue(all(isinstance(out[q], RuntimeError) for q in range(4)), out)
    self.assertEqual(len(set(msgs.values())), 1)  # every process stops, saying the same
    msg = msgs[0]
    self.assertIn("the processes hold different lowered programs", msg)
    self.assertIn("stage 1: 2 program sets (", msg)
    self.assertNotIn("stage 0", msg)
    self.assertIn(
        "  jit_fb_c1: aaaa (1336 bytes, 336 constants, 17 funcs) on processes 2; "
        "bbbb (1337 bytes, 337 constants, 17 funcs) on processes 3",
        msg,
    )
    self.assertNotIn("jit_update_s1", msg)
    # process 1 lowered no stage-0 program
    out, _ = self._program_agreement({**same, 1: [x for x in prints(0) if x[1] != 0]}, procs)
    self.assertIn("stage 0: 1 program set (", str(out[0]))
    self.assertIn("none from processes 1)", str(out[0]))
    # the executables' prints carry no stats
    exe = {q: [(n, s, fp + "e", None) for n, s, fp, _ in same[q]] for q in range(4)}
    exe[0] = [(n, s, "zzzz" if n == "jit_fwd_c0" else fp, None) for n, s, fp, _ in exe[0]]
    out, _ = self._program_agreement(exe, procs, "executables")
    self.assertIn("  jit_fwd_c0: zzzz on processes 0; f0f0e on processes 1", str(out[2]))
    self.assertEqual(dsp._ranges([0, 1, 2, 5, 7, 8]), "0-2,5,7-8")

  def test_executable_print(self):
    """executable_print: the sha256 of a TPU executable's fingerprint (what
    jaxlib seeds the program's launch ids with); None off TPU, where a CPU
    fingerprint is not even stable, for none, and where the backend refuses."""

    def compiled(platform, fp=b"abc", fail=False):
      def exe():
        if fail:
          raise RuntimeError("UNIMPLEMENTED")
        return types.SimpleNamespace(local_devices=lambda: [types.SimpleNamespace(platform=platform)], fingerprint=fp)

      return types.SimpleNamespace(runtime_executable=exe)

    self.assertEqual(dsp.executable_print(compiled("tpu")), hashlib.sha256(b"abc").hexdigest()[:16])
    self.assertIsNone(dsp.executable_print(compiled("cpu")))
    self.assertIsNone(dsp.executable_print(compiled("tpu", fp=None)))
    self.assertIsNone(dsp.executable_print(compiled("tpu", fail=True)))
    self.assertIsNone(dsp.executable_print(jax.jit(lambda x: x + 1).lower(jnp.zeros(4)).compile()))

  def test_process_sets(self):
    """--profile-procs / --json-procs: leaders (each stage's first process),
    all, or a comma list of process ids; anything else, or a process the run
    does not have, stops every process (SystemExit) before anything compiles."""
    p = pipe(tiny(layers=4, fsdp=1, ep=2))
    self.assertEqual([dsp.proc_set(spec, p) for spec in ("leaders", "all", "0")], [{0}] * 3)
    with (
        mock.patch.object(jax, "process_count", lambda: 4),
        mock.patch.object(p, "stage_procs", [frozenset({1, 0}), frozenset({3, 2})]),
    ):
      self.assertEqual(dsp.proc_set("leaders", p), {0, 2})
      self.assertEqual(dsp.proc_set("all", p), {0, 1, 2, 3})
      self.assertEqual(dsp.proc_set("3,1", p, "--json-procs"), {1, 3})
      for bad, msg in (
          ("2,4", r"--json-procs: no process \[4\] \(there are 4\)"),
          ("1,x", "--json-procs: leaders, all or a comma list"),
          ("", "comma list"),
      ):
        with self.subTest(spec=bad), self.assertRaisesRegex(SystemExit, msg):
          dsp.proc_set(bad, p, "--json-procs")

  def test_profile_chips_default_is_left_to_profiling(self):
    """--profile-chips unset is profiling.CHIPS (one TPU a traced process:
    one a slice under --layout slices); fewer than one chip stops at
    argparse."""
    self.assertIsNone(dsp.make_parser().parse_args([]).profile_chips)
    self.assertEqual(dsp.make_parser().parse_args(["--profile-chips", "2"]).profile_chips, 2)
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      dsp.make_parser().parse_args(["--profile-chips", "0"])

  def test_summary_marks_other_processes_stages(self):
    """A process resolves the collective flavours of its own stages; another
    process's stage it never probes (they are libtpu's rule from the
    devices), and the summary marks those '*'.  In one process: no mark (one
    controller holds every stage: `local` is not looked at)."""
    p = pipe(tiny_xla(layers=4, fsdp=1, ep=2, ep_collectives="auto"))
    self.assertNotIn("*", p.summary())
    with mock.patch.object(p, "local", frozenset({0})):
      self.assertNotIn("*", p.summary())
      with mock.patch.object(p, "multi", True):  # a process of a multi-controller run holding stage 0
        head = p.summary().splitlines()[0]
    self.assertEqual(
        head,
        "EP collectives per stage: ['ppermute', 'ppermute*']; FSDP collectives per stage: "
        "['ppermute', 'ppermute*'] (* another process's stage: inferred)",
    )

  def test_distributed_environment(self):
    """init_distributed before jax.distributed.initialize: MPMD_COORD_PORT
    becomes jax's coordinator port unless JAX_COORDINATOR_PORT is set (then
    with a warning), the cross-host copy flags come from the environment --
    booleans in jax's own spellings, anything else refused -- and the
    processes wait 30 minutes for each other by default."""
    read = (
        "MPMD_COORD_PORT",
        "MPMD_INIT_TIMEOUT",
        "JAX_COORDINATOR_PORT",
        "JAX_NUM_PROCESSES",
        "JAX_PROCESS_ID",
        "JAX_CROSS_HOST_TRANSFER_SOCKET_ADDRESS",
        "JAX_CROSS_HOST_TRANSPORT_ADDRESSES",
        "JAX_FORCE_DCN_CROSS_HOST_TRANSFERS",
        "JAX_CROSS_HOST_TRANSFER_TRANSFER_SIZE",
        "JAX_CROSS_HOST_TRANSFER_TIMEOUT_SECONDS",
    )

    def run(**env):
      calls, out = [], io.StringIO()
      with (
          mock.patch.dict(os.environ, env),
          mock.patch.object(jax.distributed, "initialize", lambda **kw: calls.append(("initialize", kw))),
          mock.patch.object(jax.config, "update", lambda name, v: calls.append((name, v))),
          contextlib.redirect_stdout(out),
      ):
        for k in set(read) - set(env):  # only what the case sets (patch.dict restores the rest)
          os.environ.pop(k, None)
        dsp.init_distributed()
        return calls, os.environ.get("JAX_COORDINATOR_PORT"), out.getvalue()

    calls, port, out = run(
        MPMD_COORD_PORT="8478",
        JAX_FORCE_DCN_CROSS_HOST_TRANSFERS="Yes",
        JAX_CROSS_HOST_TRANSFER_TIMEOUT_SECONDS="60",
        JAX_NUM_PROCESSES="4",
        JAX_PROCESS_ID="2",
    )
    self.assertEqual((port, out), ("8478", ""))
    self.assertEqual(
        calls,
        [
            ("jax_force_dcn_cross_host_transfers", True),
            ("jax_cross_host_transfer_timeout_seconds", 60),
            ("initialize", dict(num_processes=4, process_id=2, initialization_timeout=1800)),
        ],
    )
    calls, port, out = run(MPMD_COORD_PORT="8478", JAX_COORDINATOR_PORT="1234", JAX_FORCE_DCN_CROSS_HOST_TRANSFERS="off")
    self.assertEqual((port, calls[0]), ("1234", ("jax_force_dcn_cross_host_transfers", False)))
    self.assertEqual(
        out,
        "init_distributed: warning: JAX_COORDINATOR_PORT=1234 takes the place of this step's "
        "MPMD_COORD_PORT=8478: runs on one port can join each other's coordinator\n",
    )
    self.assertEqual(run(MPMD_COORD_PORT="8478", JAX_COORDINATOR_PORT="8478")[1:], ("8478", ""))
    with self.assertRaisesRegex(ValueError, "JAX_FORCE_DCN_CROSS_HOST_TRANSFERS='maybe'"):
      run(JAX_FORCE_DCN_CROSS_HOST_TRANSFERS="maybe")

  def test_multislice_coordinator(self):
    """On several slices, what jax.distributed's own cluster detection makes of init_distributed's
    environment (jax/_src/clusters/cloud_tpu_cluster.py, as on a GKE pod): the coordinator is
    MEGASCALE_COORDINATOR_ADDRESS's host (its port dropped) at MPMD_COORD_PORT, so the processes of one
    agent step meet at a port of their own on every slice, and the processes are numbered slice x hosts
    per slice + worker of slices x hosts: process 7 of 8 for worker 3 of slice 1's 4 hosts, the id the
    launchers derive from the same environment to name the job's host logs."""
    from jax._src import clusters
    from jax._src.clusters import cloud_tpu_cluster as ctc
    # a GKE pod's (TPU_SKIP_MDS_QUERY keeps jax's GCE TPU VM detection, tried first, off the metadata server)
    env = dict(
        MPMD_COORD_PORT="8503",
        MEGASCALE_COORDINATOR_ADDRESS="job-slice-job-0-0.job:8080",
        MEGASCALE_NUM_SLICES="2",
        MEGASCALE_SLICE_ID="1",
        TPU_WORKER_ID="3",
        TPU_SKIP_MDS_QUERY="true",
        TPU_WORKER_HOSTNAMES=",".join(f"job-slice-job-1-{w}.job" for w in range(4)),
    )
    seen = []

    def no_metadata(*a):
      raise AssertionError(f"asked the metadata server for {a}")

    def initialize(**kw):  # jax.distributed.initialize up to the point it would connect
      seen.append(
          clusters.ClusterEnv.auto_detect_unset_distributed_params(
              None, kw.get("num_processes"), kw.get("process_id"), None, None, kw["initialization_timeout"]
          )
      )

    keep = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAX_", "MEGASCALE_", "TPU_", "MAXTEXT_MPMD_", "MPMD_", "SLURM_", "OMPI_"))
        and k not in ("KUBERNETES_SERVICE_HOST", "JOB_COMPLETION_INDEX")
    }
    waited = []
    with (
        mock.patch.dict(os.environ, {**keep, **env}, clear=True),
        mock.patch.object(jax.distributed, "initialize", initialize),
        mock.patch.object(ctc, "running_in_cloud_tpu_vm", True),
        mock.patch.object(ctc, "get_metadata", no_metadata),
        mock.patch.object(
            ctc.BaseTpuCluster,
            "wait_for_coordinator",
            classmethod(lambda cls, address, timeout: waited.append((address, timeout))),
        ),
    ):
      dsp.init_distributed()
    self.assertEqual(seen, [("job-slice-job-0-0.job:8503", 8, 7, None)])
    self.assertEqual(waited, [("job-slice-job-0-0.job", 1800)])


@testing.needs_devices(4)
class Tilings(unittest.TestCase):
  """megablox tilings: tk never exceeds the k it tiles unless asked to."""

  cfg = DeepSeekConfig(emb_dim=7168, moe_mlp_dim=2048)
  padded = (512, 7168, 512) * 3  # tk = D in every kernel, for wi and wo: the one tuple the TPU7x runs used

  def test_presets(self):
    D, F = self.cfg.emb_dim, self.cfg.moe_mlp_dim
    wi, wo = ds_ep.TILES["clamped"](self.cfg)
    self.assertEqual([wi[i] for i in (1, 4, 7)], [D, F, D])  # fwd, dlhs (_dx), drhs (_dw) tk
    self.assertEqual([wo[i] for i in (1, 4, 7)], [F, D, F])
    self.assertEqual(ds_ep.tiling_problems(self.cfg, wi, wo), [])
    self.assertEqual(ds_ep.tiling_problems(self.cfg, *ds_ep.TILES["v6e"](self.cfg)), [])
    bad = ds_ep.tiling_problems(self.cfg, self.padded, self.padded)
    self.assertEqual(sorted(b.split(":")[0] for b in bad), ["gmm_wi_{0,1}_dx", "gmm_wo", "gmm_wo_dw"])
    small = DeepSeekConfig(emb_dim=256, moe_mlp_dim=128)
    self.assertEqual(ds_ep.tiling_problems(small, *ds_ep.TILES["clamped"](small)), [])

  def test_stage_config(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      scfg = ds_ep.StageConfig(model=self.cfg)
      self.assertEqual((scfg.wi_tiling, scfg.wo_tiling), ds_ep.TILES["clamped"](self.cfg))
      # ragged_dot runs no tiles, so padded ones are no problem there
      ds_ep.StageConfig(
          model=self.cfg, wi_tiling=self.padded, wo_tiling=self.padded, gmm_impl="ragged_dot", strict_tiling=True
      )
    with testing.assert_warns_regex(self, UserWarning, "gmm_wo_dw"):
      ds_ep.StageConfig(model=self.cfg, wi_tiling=self.padded, wo_tiling=self.padded)
    with self.assertRaisesRegex(ValueError, "gmm_wo"):
      ds_ep.StageConfig(model=self.cfg, wi_tiling=self.padded, wo_tiling=self.padded, strict_tiling=True)
    for bad in ((512,) * 8, (512, 0, 512) * 3, (512.0,) * 9):
      with self.subTest(tiling=bad), self.assertRaises(ValueError):
        ds_ep.StageConfig(model=self.cfg, wi_tiling=bad)

  @testing.needs_devices(4)
  def test_padded_tiling_warns_once(self):
    """Once for the StageConfig, not again when Pipeline re-runs its
    __post_init__ to resolve each stage's collectives."""
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      scfg = dataclasses.replace(
          tiny_xla(ep_collectives="auto", **testing.stage_axes()), gmm_impl="megablox", wi_tiling=(128, 512, 128) * 3
      )
      dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=1, n_micro=2, global_batch=8, seq=256, overlap=False)
    self.assertEqual(
        [str(w.message).split(" --")[0] for w in caught if ds_ep.TILING_WARNING in str(w.message)], [ds_ep.TILING_WARNING]
    )

  def test_cli_precedence(self):
    """--tiles, then --wi-tiling / --wo-tiling; a hand-built namespace
    without them gets the default."""
    ns, T = argparse.Namespace, ds_ep.TILES
    t, u = tuple(range(1, 10)), tuple(range(11, 20))
    self.assertEqual(dsp.resolve_tilings(ns(), self.cfg), T["clamped"](self.cfg))
    self.assertEqual(dsp.resolve_tilings(ns(tiles="v6e", wo_tiling=list(u)), self.cfg), (T["v6e"](self.cfg)[0], u))
    self.assertEqual(dsp.resolve_tilings(ns(tiles="v6e", wi_tiling=t, wo_tiling=u), self.cfg), (t, u))
    self.assertEqual(dsp.resolve_tilings(ns(tiles="v6e", wi_tiling=u), self.cfg), (u, T["v6e"](self.cfg)[1]))


@testing.needs_devices(4)
class Schedules(unittest.TestCase):
  """Every order the pipeline runs -- named or custom -- is normalized and
  checked before anything is built."""

  C, M = 4, 3
  place = sched.placement(2, 2)

  def test_named_schedules_are_legal(self):
    for name in dsp.ORDER_SCHEDULES:
      with self.subTest(schedule=name):
        if name in dsp.ZB_SCHEDULES:  # one chunk a stage: refused at two, legal at one, the middle chunks split
          with self.assertRaisesRegex(ValueError, "run one chunk a stage"):
            dsp.build_schedule(name, self.C, self.M, self.place)
          order, _ = dsp.build_schedule(name, self.C, self.M, list(range(self.C)))
          dsp.validate_schedule(order, self.C, self.M, list(range(self.C)))
          self.assertEqual(sorted({c for c, k, _ in order if k in ("I", "W")}), [1, 2])
          continue
        order, _ = dsp.build_schedule(name, self.C, self.M, self.place)
        assert_legal(self, order, self.C, self.M)
        self.assertTrue(all(type(a) is tuple for a in order))
    for name in set(dsp.SCHEDULES) - set(dsp.ORDER_SCHEDULES):  # Pipeline's own
      with self.subTest(schedule=name), self.assertRaisesRegex(ValueError, "built by Pipeline"):
        dsp.build_schedule(name, self.C, self.M, self.place)
    with self.assertRaises(ValueError):
      dsp.build_schedule("zb_v", self.C, self.M, self.place)

  def test_pipeline_schedules_are_legal(self):
    """Every named schedule as Pipeline builds it -- stash and interleaved's
    pairs included, remat and with stash, on both placements -- passes
    validate_schedule too (with the placement and the pairs), the check a
    custom order gets; nothing is compiled."""
    scfg = tiny_xla(fsdp=1, ep=2)
    kw = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256)
    for name in dsp.SCHEDULES:
      for placement in ("circular", "v"):
        if (name, placement) == ("interleaved", "v"):
          continue
        if name in dsp.ZB_SCHEDULES:  # one chunk a stage: four of a device, the two middle chunks split (traced only)
          if placement == "circular":
            with self.subTest(schedule=name):
              p = dsp.Pipeline(
                  tiny_xla(fsdp=1, ep=1),
                  schedule=name,
                  n_stages=4,
                  chunks_per_stage=1,
                  n_micro=4,
                  global_batch=8,
                  seq=256,
              )
              dsp.validate_schedule(p.order, p.C, p.M, p.place, fuse=p.fuse)
              self.assertEqual(p.split_chunks, [1, 2])
          with (
              self.subTest(schedule=name, placement=placement),
              self.assertRaisesRegex(ValueError, "run one chunk a stage"),
          ):
            dsp.Pipeline(scfg, schedule=name, placement=placement, **kw)
          continue
        extra = [{}] + ([dict(stash="lite", stash_budget_gib=1.0)] if name in ("stash", "interleaved") else [])
        if name == "interleaved":
          extra += [dict(e, pair_stages="all") for e in extra]
        for more in extra:
          with self.subTest(schedule=name, placement=placement, **more):
            p = dsp.Pipeline(scfg, schedule=name, placement=placement, **kw, **more)
            dsp.validate_schedule(p.order, p.C, p.M, p.place, fuse=p.fuse)
            self.assertTrue(all(type(a) is tuple for a in p.order))
            if "stash" in more:
              self.assertTrue(p.modes)
            if "pair_stages" in more:
              self.assertTrue(p.fuse)

  def test_normalize(self):
    order = sched.gpipe(self.C, self.M, self.place)
    self.assertEqual(dsp.normalize_schedule([list(a) for a in order]), order)
    self.assertEqual(dsp.normalize_schedule([(np.int64(0), "F", np.int32(1))]), [(0, "F", 1)])
    for bad in ([(0, "F")], [(0, "X", 0)], [(True, "F", 0)], [(0.0, "F", 0)], [0]):
      with self.subTest(schedule=bad), self.assertRaises(ValueError):
        dsp.normalize_schedule(bad)

  def test_illegal_orders_are_rejected(self):
    C, M = self.C, self.M
    good = sched.one_f_one_b(C, M, self.place)
    dsp.validate_schedule(good, C, M)
    f2 = good.index((2, "F", 0))
    b1 = good.index((1, "B", 0))
    cases = {
        "incomplete": (good[:-1], "lacks 1"),
        "duplicate": (good + [good[0]], "twice"),
        "out of range": (good + [(C, "F", 0)], "out of range"),
        "forward before upstream": ([(1, "F", 0)] + [a for a in good if a != (1, "F", 0)], "before F\\(0,0\\)"),
        "backward before forward": ([(0, "B", 0)] + [a for a in good if a != (0, "B", 0)], "before F\\(0,0\\)"),
        "backward before downstream": (
            good[: f2 + 1] + [good[b1]] + [a for a in good[f2 + 1 :] if a != good[b1]],
            "before B\\(2,0\\)",
        ),
    }
    for name, (order, msg) in cases.items():
      with self.subTest(name), self.assertRaisesRegex(ValueError, msg):
        dsp.validate_schedule(order, C, M)

  @testing.needs_devices(4)
  def test_custom_schedule_in_pipeline(self):
    """A custom order as a list of lists (e.g. from JSON) builds the same plan
    as the tuple order; an illegal one is refused before anything is built."""
    kw = dict(n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=16, seq=256, overlap=True)
    scfg = tiny_xla(**testing.stage_axes())
    base = dsp.Pipeline(scfg, schedule="search", **kw)
    lists = [list(a) for a in base.order]
    p = dsp.Pipeline(scfg, schedule=lists, **kw)
    self.assertEqual(p.schedule_name, "custom")
    self.assertEqual(p.order, base.order)
    self.assertEqual(p.pf_plan, base.pf_plan)
    self.assertEqual({j: v[:2] for j, v in p.rs_at.items()}, {j: v[:2] for j, v in base.rs_at.items()})
    self.assertEqual(p.plan_summary(), base.plan_summary())
    for bad in (lists[:-1], lists[::-1]):
      with self.assertRaises(ValueError):
        dsp.Pipeline(scfg, schedule=bad, **kw)

  @testing.needs_devices(4)
  def test_one_microbatch_plan(self):
    """--mb 1: the first stage's only forward before its second chunk is 0F0,
    which self-prefetches chunk 0's experts, so chunk 2 is gathered by the
    update program instead of raising; without self-prefetch 0F0 carries it.
    The step scope's plan (RepeatScope.test_one_microbatch has repeat's)."""
    kw = dict(n_stages=2, chunks_per_stage=2, n_micro=1, global_batch=4, seq=256, overlap=True, fsdp_scope="step")
    p = dsp.Pipeline(tiny_xla(**testing.stage_axes()), **kw)
    self.assertEqual(p.pf_fallback, [2])
    self.assertNotIn(2, p.pf_plan)
    self.assertIn(p.slot[2], p.gather_slots[p.place[2]])
    self.assertIn("no earlier forward", p.plan_summary())
    q = dsp.Pipeline(tiny_xla(**testing.stage_axes()), self_prefetch=False, **kw)
    self.assertEqual(q.pf_fallback, [])
    self.assertEqual([j for j, _ in q.pf_plan[2]], [q.order.index((0, "F", 0))])
    # V placement: stage 0 holds chunk 0 and the head (3), and 0F0 is again
    # its only forward before 3F0 -- under the stash schedule too
    for more in ({}, dict(schedule="stash", stash="lite", stash_budget_gib=1.0)):
      v = dsp.Pipeline(tiny_xla(**testing.stage_axes()), placement="v", **kw, **more)
      self.assertEqual(v.pf_fallback, [3], more)
      self.assertEqual(v.gather_slots[0], (0, 1), more)
      self.assertEqual(v.pf, {}, more)
      self.assertEqual(bool(v.modes), bool(more), more)


class Kernels(unittest.TestCase):
  """The pure-JAX expert matmuls, and how the megablox ones are reached."""

  def test_ragged_dot_gmm_matches_dense(self):
    """named_gmm(impl='ragged_dot') against a per-row dense product: rows of
    this shard's groups [off, off + E_local) use their group's weight, and
    every other row (other shards' groups, the padding past sum(gs)) is zero
    -- forward, dx and dW, with the offset traced as in the pipeline."""
    E, El, M, K, N = 8, 4, 64, 16, 8
    gs = np.array([5, 0, 9, 7, 3, 11, 0, 6], np.int32)  # 41 rows, 23 of padding
    gid = np.concatenate([np.repeat(np.arange(E), gs), np.full(M - gs.sum(), -1)])
    ks = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(ks[0], (M, K), jnp.float32)
    w = jax.random.normal(ks[1], (El, K, N), jnp.float32)
    ct = jax.random.normal(ks[2], (M, N), jnp.float32)

    @jax.jit
    def mine(x, w, off, ct):
      y, vjp = jax.vjp(
          lambda x, w: ds_ep.named_gmm("gmm_wi_0", TILE, True, x, w, jnp.asarray(gs), off, impl="ragged_dot"), x, w
      )
      return y, vjp(ct)

    for off in (0, 2, 4):
      local = (gid >= off) & (gid < off + El)

      def ref(x, w, off=off, local=local):
        y = jnp.einsum("mk,mkn->mn", x, w[np.clip(gid - off, 0, El - 1)], precision=lax.Precision.HIGHEST)
        return jnp.where(local[:, None], y, 0)

      ya, ga = mine(x, w, jnp.int32(off), ct)
      yb, vb = jax.vjp(ref, x, w)
      with self.subTest(off=off):
        self.assertFalse(np.any(np.asarray(ya)[~local]))
        self.assertLess(_rel(ya, yb), 1e-5)
        for a, b in zip(ga, vb(ct)):
          self.assertLess(_rel(a, b), 1e-5)
    with self.assertRaises(ValueError):
      ds_ep.named_gmm("gmm_wi_0", TILE, True, x, w, jnp.asarray(gs), jnp.int32(0), impl="tokamax")

  def test_gmm_acc(self):
    """named_gmm(acc=True) hands its rhs on, and a second grouped matmul of
    the same weight run on it adds its dW into this call's inside the dW
    kernel (megablox: tgmm's existing_out, interpreted here; ragged_dot: in
    fp32 before the rounding): the pair's gradients are two plain calls'
    (fp32: to rounding), with no add_any over the weight where those have
    one."""
    G, K, N, M = 4, 128, 256, 512
    ks = jax.random.split(jax.random.key(1), 5)
    w = jax.random.normal(ks[0], (G, K, N), jnp.float32) * 0.1
    x0, x1 = (jax.random.normal(k, (M, K), jnp.float32) for k in ks[1:3])
    c0, c1 = (jax.random.normal(k, (M, N), jnp.float32) for k in ks[3:5])
    gs0, gs1 = jnp.array([100, 200, 0, 212], jnp.int32), jnp.array([50, 0, 300, 162], jnp.int32)
    for impl in ("ragged_dot", "megablox"):

      def loss(w, x0, x1, acc, impl=impl):
        def gmm(name, lhs, rhs, gs, acc=False):
          return ds_ep.named_gmm(name, TILE, True, lhs, rhs, gs, jnp.int32(0), impl=impl, acc=acc)

        y0, wp = gmm("a", x0, w, gs0, acc=True) if acc else (gmm("a", x0, w, gs0), w)
        y1 = gmm("b", x1, wp, gs1)
        return jnp.sum(y0 * c0) + jnp.sum(y1 * c1)

      with self.subTest(impl=impl):
        acc, plain = (jax.grad(loss, argnums=(0, 1, 2))(w, x0, x1, a) for a in (True, False))
        for a, b in zip(acc, plain):
          self.assertLess(_rel(a, b), 1e-6)
        adds = [
            _eqn_out_shapes(jax.make_jaxpr(jax.grad(lambda w, a=a: loss(w, x0, x1, a)))(w), "add_any")
            for a in (True, False)
        ]
        self.assertEqual([s.count(w.shape) for s in adds], [0, 1])

  def test_gmm_stacked(self):
    """named_gmm(stacked=(w, l)) reads block l of the stacked weight `w` in
    place of its slice `rhs` -- megablox, interpreted: the output, dlhs (on
    every row of this shard's groups) and dW bit for bit the slice's, at
    each layer, with and without acc, for the truncated ring buffer (`gs` this
    shard's groups only) and for global group sizes at a shard offset.
    ragged_dot reads the slice either way."""
    L, G, K, N, M = 3, 4, 128, 256, 512
    ks = jax.random.split(jax.random.key(2), 3)
    ws = jax.random.normal(ks[0], (L, G, K, N), jnp.float32) * 0.1
    x = jax.random.normal(ks[1], (M, K), jnp.float32)
    c = jax.random.normal(ks[2], (M, N), jnp.float32)
    cases = (
        ("truncated", jnp.array([100, 200, 0, 150], jnp.int32), 0),
        ("global, offset 4", jnp.array([30, 20, 10, 0, 100, 0, 120, 90, 50, 40, 0, 0], jnp.int32), 4),
    )
    for impl in ("ragged_dot", "megablox"):
      for name, gs, off in cases:
        ends = np.cumsum(np.asarray(gs))
        lo, hi = int(ends[off] - gs[off]), int(ends[off + G - 1])
        for layer, acc in ((0, False), (1, True), (2, False), (2, True)):

          def loss(wl, x, stacked, layer=layer, acc=acc, gs=gs, off=off, impl=impl):
            st = (ws, layer) if stacked else None

            def gmm(nm, lhs, rhs, acc=False):
              return ds_ep.named_gmm(nm, TILE, True, lhs, rhs, gs, jnp.int32(off), impl=impl, acc=acc, stacked=st)

            if acc:
              y, wp = gmm("a", x, wl, acc=True)
              y = y + gmm("b", x * 0.5, wp)
            else:
              y = gmm("a", x, wl)
            return jnp.sum(jnp.where(jnp.arange(M)[:, None] < hi, y, 0) * c), y

          with self.subTest(impl=impl, case=name, layer=layer, acc=acc):
            (_, ya), (gwa, gxa) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(ws[layer], x, True)
            (_, yb), (gwb, gxb) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(ws[layer], x, False)
            np.testing.assert_array_equal(np.asarray(ya[lo:hi]), np.asarray(yb[lo:hi]))
            np.testing.assert_array_equal(np.asarray(gxa[lo:hi]), np.asarray(gxb[lo:hi]))
            np.testing.assert_array_equal(np.asarray(gwa), np.asarray(gwb))
    with self.assertRaisesRegex(ValueError, "is not the"):
      ds_ep.named_gmm("a", TILE, True, x, ws[0][:, :64], cases[0][1], jnp.int32(0), impl="ragged_dot", stacked=(ws, 0))

  def test_gmm_keep(self):
    """named_gmm(keep=Recompute(fn, args)), lean residuals': the VJP keeps `args` in the lhs's place and recomputes
    the lhs for the dW kernel -- for ragged_dot and megablox, plain, acc (a second product on the handed-on weight)
    and stacked reads, at a shard's group offset: the output and every gradient bit for bit the plain call's, and no
    lhs-shaped residual left (without keep there is one)."""
    L, G, K, D, N, M, T = 3, 4, 2, 128, 256, 512, 200
    r = np.random.default_rng(4)
    ws = jnp.asarray(0.1 * r.standard_normal((L, G, D, N)), jnp.float32)
    xt = jnp.asarray(r.standard_normal((T, D)), jnp.float32)
    order = jnp.asarray(r.integers(0, T * K, M), jnp.int32)  # each row's (token, choice): fewer tokens than rows
    ct = jnp.asarray(r.standard_normal((M, N)), jnp.float32)
    gs, off = jnp.array([30, 20, 10, 0, 100, 0, 120, 90, 50, 40, 0, 0], jnp.int32), jnp.int32(4)
    rows = functools.partial(ds_ep._argsort_rows, order=order, K=K, pad=0)
    impls = [("ragged_dot", contextlib.nullcontext)]
    impls.append(("megablox", contextlib.nullcontext))

    def loss(xt, w, impl, keep, acc=False, stacked=False):
      def gmm(nm, scale, rhs, acc=False):
        f = lambda xt: rows(xt) * scale
        return ds_ep.named_gmm(
            nm,
            TILE,
            True,
            f(xt),
            rhs,
            gs,
            off,
            impl=impl,
            acc=acc,
            stacked=(ws, 1) if stacked else None,
            keep=ds_ep.Recompute(f, (xt,)) if keep else None,
        )

      if acc:
        y, wp = gmm("a", 1.0, w, acc=True)
        return y + gmm("b", 0.5, wp)
      return gmm("a", 1.0, w)

    for impl, ctx in impls:
      for kw in ({}, dict(acc=True), dict(stacked=True)):
        with self.subTest(impl=impl, **kw), ctx():
          runs = {}
          for keep in (False, True):
            f = functools.partial(loss, impl=impl, keep=keep, **kw)
            y, vjp = jax.vjp(f, xt, ws[1])
            res = jax.eval_shape(lambda *a, f=f: jax.tree_util.tree_leaves(jax.vjp(f, *a)[1]), xt, ws[1])
            runs[keep] = ([y, *vjp(ct)], [tuple(a.shape) for a in res])
          for a, b in zip(runs[True][0], runs[False][0], strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
          self.assertIn((M, D), runs[False][1])
          self.assertNotIn((M, D), runs[True][1])
          self.assertIn((T, D), runs[True][1])

  def test_megablox_signature_check(self):
    """named_gmm calls MaxText's private megablox bodies by keyword and checks
    their signatures first, so a MaxText that renames one fails naming the
    function and keyword rather than deep in a kernel call."""

    def gmm(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset, transpose_rhs=False, interpret=False):
      pass

    def tgmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        num_actual_groups,
        existing_out=None,
        interpret=False,
    ):
      pass

    def gmm_renamed(
        lhs, rhs, group_sizes, preferred_element_type, tiling, group_start, transpose_rhs=False, interpret=False
    ):
      pass

    def fake(**fns):
      backend = types.ModuleType("maxtext.kernels.megablox.backend")
      backend.__file__ = "<fake>"
      for k, f in fns.items():
        setattr(backend, k, types.SimpleNamespace(__wrapped__=f) if f else (lambda: None))
      pkg = types.ModuleType("maxtext.kernels.megablox")
      pkg.backend = backend
      return _sys_modules(
          _maxtext_stand_ins(
              {
                  "maxtext": types.ModuleType("maxtext"),
                  "maxtext.kernels": types.ModuleType("maxtext.kernels"),
                  "maxtext.kernels.megablox": pkg,
                  "maxtext.kernels.megablox.backend": backend,
              }
          )
      )

    with fake(gmm=gmm, tgmm=tgmm):
      self.assertEqual(ds_ep._megablox(), (gmm, tgmm))
    with fake(gmm=gmm_renamed, tgmm=tgmm), self.assertRaisesRegex(RuntimeError, r"gmm.*\['group_offset'\]"):
      ds_ep._megablox()
    with fake(gmm=gmm, tgmm=None), self.assertRaisesRegex(RuntimeError, "tgmm.*__wrapped__"):
      ds_ep._megablox()


@unittest.skipUnless(HAVE_TOKAMAX, "needs tokamax")
class TokamaxSplashAttention(unittest.TestCase):
  """deepseek.attention's 'tokamax_splash' kernel (the pipeline's and the
  MaxText baseline's; interpret mode off TPU) against reference_attention:
  the output and all three input gradients, with MLA's unequal qk / v head
  dims, over several blocks, with and without segment ids."""

  def test_matches_reference_attention(self):
    for base2 in (False, True):
      with self.subTest(sa_use_base2_exp=base2):
        self._matches_reference_attention(base2)

  def _matches_reference_attention(self, base2):
    cfg = DeepSeekConfig(
        attention_kernel="tokamax_splash",
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_use_base2_exp=base2,
    )
    b, t, n, dqk, dv = 2, 256, 2, 192, 128
    ks = jax.random.split(jax.random.key(0), 4)
    segs = jnp.concatenate([jnp.zeros((b, 100), jnp.int32), jnp.ones((b, t - 100), jnp.int32)], axis=1)
    for dt, tol in ((jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)):
      q = (jax.random.normal(ks[0], (b, t, n, dqk)) * 0.3).astype(dt)
      k = (jax.random.normal(ks[1], (b, t, n, dqk)) * 0.3).astype(dt)
      v = jax.random.normal(ks[2], (b, t, n, dv)).astype(dt)
      ct = jax.random.normal(ks[3], (b, t, n, dv)).astype(dt)
      for seg in (None, segs):
        ya, va = jax.vjp(lambda q, k, v, seg=seg: attention.attention_op(q, k, v, seg, cfg), q, k, v)
        yb, vb = jax.vjp(lambda q, k, v, seg=seg: attention.reference_attention(q, k, v, seg), q, k, v)
        with self.subTest(dtype=jnp.dtype(dt).name, segment_ids=seg is not None):
          self.assertEqual(ya.dtype, yb.dtype)
          self.assertLess(_rel(ya, yb), tol)
          for name, a, c in zip("qkv", va(ct), vb(ct)):
            self.assertLess(_rel(a, c), tol, f"d{name}")

  def test_base2_exp_reaches_the_kernel(self):
    """DeepSeekConfig.sa_use_base2_exp is tokamax's SplashConfig.use_base2_exp, on by default as MaxText's (the
    kernels scale a bf16 query by log2(e) rounded to bf16, see config.py); off, the exact softmax."""
    tsk = attention.tokamax_splash_module("splash_attention_kernel")
    seen, real = [], tsk.SplashConfig

    def spy(**kw):
      seen.append(kw["use_base2_exp"])
      return real(**kw)

    q, v = jnp.zeros((1, 128, 1, 192), jnp.float32), jnp.zeros((1, 128, 1, 128), jnp.float32)
    blocks = dict(attention_kernel="tokamax_splash", sa_block_q=128, sa_block_kv=128, sa_block_kv_compute=128)
    with mock.patch.object(tsk, "SplashConfig", spy):
      for cfg in (DeepSeekConfig(**blocks, sa_use_base2_exp=False), DeepSeekConfig(**blocks)):
        attention.tokamax_splash_attention(q, q, v, None, cfg)
    self.assertEqual(seen, [False, True])


def fresh_kernel_caches():
  """Empty what a new process starts without: jax's caches (jits traced, a
  jitted kernel's trace among them) and the splash kernels' mask-table
  lru_caches (jax's, tokamax's)."""
  from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as jmi

  jax.clear_caches()
  jmi._process_mask.cache_clear()
  if HAVE_TOKAMAX:
    attention.tokamax_splash_module("splash_attention_mask_info")._process_mask.cache_clear()


@testing.needs_devices(4)
class PrecompileIsDeterministic(unittest.TestCase):
  """`precompile` with its thread pool lowers a pipeline to the same HLO run
  after run: it lowers one program at a time and only compiles in threads.

  Lowered in its compile threads, as they once were, the programs of a
  pipeline with tokamax splash attention came out as one of several HLOs:
  tokamax keeps its mask tables in a functools.lru_cache, two threads that
  miss it together each build a copy of their own, and a program traced
  meanwhile can hold two equal copies of a table where one traced alone holds
  one -- a constant more (CPU, 16 processes of a 4-layer pipeline with MTP:
  25 of its 31 programs in two to six versions).  Across the processes of a
  stage that is a different program at one launch, which halts the TPU
  runtime.  Each run here starts from what a new process has (cleared
  caches) and stubs the compiles: the lowered text is the point.  jax's
  splash kernel, a jit, was never affected: jax traces it once per process."""

  # two one-layer chunks, the last with the MTP layer: its fused programs run two attention layers
  ARGS = TINY_ARGS + ["--mtp", "1", "--weight-dtype", "float32"]

  def _lowered(self, attn, runs=3, threads=16):
    """[Counter of (module name, sha256 of the lowered text)] of `runs`
    precompiles of ARGS, each from fresh caches, as the compile stub saw the
    texts; each checked against `program_prints`."""
    out = []
    for _ in range(runs):
      fresh_kernel_caches()
      p = dsp.build(dsp.make_parser().parse_args(self.ARGS + ["--attn", attn]))
      seen = []

      def compile_(lowered, *a, **k):  # no XLA compile: an executable without a fingerprint
        txt = lowered.as_text()
        seen.append((re.match(r"module @(\S+)", txt).group(1), hashlib.sha256(txt.encode()).hexdigest()[:16]))
        return types.SimpleNamespace(runtime_executable=lambda: None)

      with mock.patch.object(jax.stages.Lowered, "compile", compile_), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, _ = p.precompile(threads=threads)
      self.assertEqual(len(seen), n)
      self.assertEqual(sorted(sha for _, sha in seen), sorted(lw["sha"] for _, _, lw, _ in p.program_prints))
      self.assertTrue(all(x is None for _, _, _, x in p.program_prints))
      out.append(collections.Counter(seen))
    return out

  def _check(self, attn):
    runs = self._lowered(attn)
    differ = sorted({nm for r in runs[1:] for nm, _ in (r - runs[0]) + (runs[0] - r)})
    self.assertEqual(differ, [], f"{attn}: programs lowered differently from run to run")
    self.assertTrue(any("fb_c1" in nm for nm, _ in runs[0]), sorted(runs[0]))

  @unittest.skipUnless(HAVE_TOKAMAX, "needs tokamax")
  def test_tokamax_splash(self):
    self._check("tokamax_splash")

  def test_jax_splash(self):
    self._check("splash")

  def test_cli_prints_programs_and_grad_digests(self):
    """main logs each stage's programs as precompile printed them (the
    lowered text's sha256, size and constants, the executable's sha256) and
    records them in the run record; --grad-digest N logs, in the first N
    warmup steps, a sha256 of each gradient leaf of every local chunk: two
    runs from scratch print the same programs and the same digests."""
    out = []
    for _ in range(2):
      fresh_kernel_caches()
      path = os.path.join(self.enterContext(tempfile.TemporaryDirectory()), "run.json")
      with contextlib.redirect_stdout(io.StringIO()) as log, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = dsp.main(TINY_ARGS + ["--warmup", "2", "--grad-digest", "1", "--json", path])
      with open(path) as f:
        self.assertEqual(json.load(f)["programs"], [list(x) for x in rec["programs"]])
      out.append((log.getvalue(), rec))
    (log, rec), (log2, rec2) = out
    n = int(re.search(r"precompiled (\d+) programs", log).group(1))
    self.assertEqual(len(rec["programs"]), n)
    self.assertEqual(rec["programs"], rec2["programs"])
    # CPU executables print nothing (executable_print: their fingerprints are not stable)
    self.assertTrue(all(len(sha) == 16 and x is None for _, _, sha, x in rec["programs"]), rec["programs"])
    prints = re.findall(r"^stage (\d): (\d+) programs, digest ([0-9a-f]{16})", log, re.M)
    self.assertEqual([s for s, _, _ in prints], ["0", "1"])
    self.assertEqual(sum(int(k) for _, k, _ in prints), n)
    for name, stage, sha, x in rec["programs"]:
      self.assertRegex(log, rf"\n  {name} {sha} \d+B \d+c -\n")
    digests = re.findall(r"^grad digest step (\d) c(\d): (.*)$", log, re.M)
    self.assertEqual([(k, c) for k, c, _ in digests], [("0", "0"), ("0", "1")])
    self.assertIn("token_embedder.embedding=", digests[0][2])
    self.assertIn("logits_dense.kernel=", digests[1][2])
    self.assertEqual(digests, re.findall(r"^grad digest step (\d) c(\d): (.*)$", log2, re.M))
    self.assertEqual(rec["warmup_losses"], rec2["warmup_losses"])


@testing.needs_devices(4)
class PipelineXlaKernels(unittest.TestCase):
  """The whole pipeline against the reference on XLA's ops (XLA_KERNELS):
  `ragged_dot` experts and the argsort dispatch.  fp32 compute, so the
  tolerances are rounding-level (measured: identical loss, worst gradient
  leaf ~1e-7)."""

  def _check(self, overlap, mb, **kw):
    p = dsp.Pipeline(
        tiny_xla(**testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=mb,
        global_batch=4 * mb,
        seq=256,
        overlap=overlap,
        **kw,
    )
    compare_with_reference(self, p, overlap, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)
    return p

  def test_matches_reference(self):
    self._check(overlap=False, mb=2)

  @pytest.mark.scheduled_only
  def test_overlap_matches_reference(self):
    p = self._check(overlap=True, mb=2)
    self.assertTrue(p.pf and p.rs_at)

  @pytest.mark.scheduled_only
  def test_route_weights_matches_independent_port(self):
    """Both route_weights against `maxtext.experimental.mpmd.jax.deepseek`, the MaxText-parity port, which shares
    none of the pipeline's
    hot path and weighs the experts' outputs (MaxText's order): 'combine', and 'experts', which weighs wo's input rows
    instead (Megatron-LM's order).  fp32, every token taking every expert (top-k's choices would flip on last-bit
    differences): the same loss and gradients but for fp32 reassociation (both: loss 7.6e-8, worst leaf 1.9e-6
    measured)."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    for route in ds_ep.ROUTE_WEIGHTS:
      with self.subTest(route_weights=route):
        scfg = tiny_xla(layers=4, random_routing=False, route_weights=route, **testing.stage_axes())
        scfg = dataclasses.replace(scfg, model=dataclasses.replace(scfg.model, num_experts_per_tok=8))
        p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=8, seq=256)
        params, _, wc0 = p.init(jax.random.key(0))
        tokens = np.random.default_rng(0).integers(0, scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
        targets = np.roll(tokens, -1, axis=1)
        loss, grads = par.mpmd_loss_and_grads(p, params, wc0, p.put_batch(tokens, targets))
        ref_loss, ref = par.port_loss_and_grads(p, [[par.to_host(t) for t in ps] for ps in params], tokens, targets)
        rows, cos, _ = par.errors(p, grads, ref)
        self.assertLess(abs(loss - ref_loss) / ref_loss, 1e-6)
        self.assertLess(rows[0][0], 2e-5, rows[:3])
        self.assertGreater(cos, 1 - 1e-9)

  @pytest.mark.scheduled_only
  def test_overlap_one_microbatch_repeat(self):
    """--mb 1 under the default, breadth with weights per visit: the update
    gathers chunk 2 too, which its visit then releases; the gradients are
    still exact."""
    p = self._check(overlap=True, mb=1)
    self.assertEqual((p.fsdp_scope, p.pf_fallback, p.gather_slots), ("repeat", [2], [(0, 1), (0,)]))

  @pytest.mark.scheduled_only
  def test_overlap_one_microbatch(self):
    """--mb 1: chunk 2's gather moves to the update program (the step scope's
    plan the Schedules test checks), and the gradients are still exact."""
    p = self._check(overlap=True, mb=1, fsdp_scope="step")
    self.assertEqual(p.pf_fallback, [2])

  @testing.slow
  def test_uneven_layer_split(self):
    """layer_split: chunks of different depths (the head's chunk holding fewer layers) -- 1F1B on (3, 1), V placement
    on (1, 2, 2, 1) and circular on (2, 1, 2, 1) of 6 layers -- against the reference, with and without overlap."""
    cases = [
        (4, 1, "circular", "1f1b", (3, 1)),
        (6, 2, "v", "search", (1, 2, 2, 1)),
        (6, 2, "circular", "search", (2, 1, 2, 1)),
    ]
    for layers, vv, placement, schedule, split in cases:
      for overlap in (False, True):
        with self.subTest(split=split, placement=placement, overlap=overlap):
          p = dsp.Pipeline(
              tiny_xla(layers=layers, **testing.stage_axes()),
              n_stages=2,
              chunks_per_stage=vv,
              n_micro=2,
              global_batch=8,
              seq=256,
              overlap=overlap,
              placement=placement,
              schedule=schedule,
              layer_split=split,
          )
          self.assertEqual([len(ch.layers) for ch in p.chunks], list(split))
          self.assertEqual([l for ch in p.chunks for l in ch.layers], list(range(layers)))
          compare_with_reference(self, p, overlap, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)

  def test_layer_split_checks(self):
    """Wrong counts, sums, types and signs are refused; a 0 is a chunk of no layer, which only the first (the
    embedding's alone) and the last (the head's alone) may be (ds_ep_qwen_test's test_empty_end_chunks runs them)."""
    for split in ((2, 1), (5, -1), (2, 2, 0), (2.0, 2)):
      with self.subTest(split=split), self.assertRaisesRegex(ValueError, "layer_split"):
        dsp.Pipeline(
            tiny_xla(layers=4, **testing.stage_axes()),
            n_stages=2,
            chunks_per_stage=1,
            n_micro=2,
            global_batch=8,
            seq=256,
            schedule="1f1b",
            layer_split=split,
        )
    with self.assertRaisesRegex(ValueError, "the last may hold none"):
      dsp.Pipeline(
          tiny_xla(layers=4, **testing.stage_axes()),
          n_stages=2,
          chunks_per_stage=2,
          n_micro=2,
          global_batch=8,
          seq=256,
          layer_split=(0, 4, 0, 0),
      )
    p = dsp.Pipeline(
        tiny_xla(layers=4, **testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=1,
        n_micro=2,
        global_batch=8,
        seq=256,
        schedule="1f1b",
        layer_split=(4, 0),
    )
    self.assertEqual([ch.layers for ch in p.chunks], [(0, 1, 2, 3), ()])

  @pytest.mark.scheduled_only
  def test_empty_first_chunk_under_stash(self):
    """An empty chunk 0 (the embedding alone) in stash mode -- its stash set is empty, so the scheduler stashes
    every one of its microbatches -- trains as the reference: its stash backward is the embedding's VJP alone."""
    p = dsp.Pipeline(
        tiny_xla(layers=3, fsdp=1, ep=2),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=4,
        global_batch=8,
        seq=256,
        schedule="stash",
        stash="lite",
        stash_budget_gib=1.0,
        layer_split=(0, 1, 1, 1),
    )
    self.assertEqual([ch.layers for ch in p.chunks], [(), (0,), (1,), (2,)])
    self.assertTrue(any(c == 0 for c, _ in p.modes))
    compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)
    self.assertEqual(dsp.parse_layer_split("3,1"), (3, 1))
    self.assertIsNone(dsp.parse_layer_split(""))
    with self.assertRaisesRegex(ValueError, "comma-separated"):
      dsp.parse_layer_split("3;1")

  def test_split_costs(self):
    t = dict(F0=18.5, F=16.5, B0=51.3, B=48.6, A=58.3, xfer=1.0)
    self.assertIs(dsp.split_costs(t, (2, 2)), t)
    c = dsp.split_costs(t, (3, 1))  # even would be 2 a chunk
    self.assertAlmostEqual(c["F0"], 18.5 * 1.5)
    self.assertAlmostEqual(c["B0"], 51.3 * 1.5)
    # The fused last chunk recomputes nothing by default: a layer there costs B's share, not F's and B's.
    self.assertAlmostEqual(c["A"], 58.3 - 0.5 * 48.6)
    self.assertAlmostEqual(dsp.split_costs(dict(t, A=80.0), (3, 1), remat_last=True)["A"], 80.0 - 0.5 * (16.5 + 48.6))
    c = dsp.split_costs(t, (1, 2, 2, 1))  # even 1.5
    self.assertAlmostEqual(c["F1"], 16.5 * 2 / 1.5)
    self.assertAlmostEqual(c["F0"], 18.5 / 1.5)
    # A table whose A undercuts its layers (TPU7x's under remat: 58.3 < F + B) prices the head at 0, not below:
    # 58 layers as (55, 3) would have given A = -0.07 ms.
    self.assertAlmostEqual(dsp.split_costs(t, (55, 3))["A"], 58.3 - 48.6 + 48.6 * 3 / 29)
    self.assertAlmostEqual(dsp.split_costs(t, (55, 3), remat_last=True)["A"], (16.5 + 48.6) * 3 / 29)
    # A per-chunk key is that chunk's (even-split) cost: it wins over F / B, and scales too.
    self.assertAlmostEqual(dsp.split_costs(dict(t, F1=30.0), (1, 2, 2, 1))["F1"], 30.0 * 2 / 1.5)
    # Stash keys scale with their chunk's layers, each the key cost_table reads for that chunk: the v6e table's
    # d-keys (chunk 0's own, and the rest's) become per-chunk d-keys over the scaled F{c} / B{c}.
    v = dsp.split_costs(dict(sched.V6E_TILES_COSTS), (3, 1, 3, 1))
    self.assertAlmostEqual(v["dBs0:lite"], -9.16 * 1.5)
    self.assertAlmostEqual(v["dBs1:lite"], -9.58 * 0.5)
    self.assertAlmostEqual(v["dFs2:lite_y"], 4.90 * 1.5)
    cost = sched.cost_table(v, 4)
    self.assertAlmostEqual(cost(1, "B", "lite"), (53.10 - 9.58) * 0.5)
    self.assertAlmostEqual(cost(0, "B", "lite"), (55.84 - 9.16) * 1.5)
    # an absolute stash key stays absolute; a policy's own key wins over a policy-less one
    a = dsp.split_costs(dict(t, **{"Bs:attn": 40.0, "Bs": 45.0, "dBs0:attn": -2.0}), (3, 1))
    self.assertNotIn("Bs0:attn", a)  # chunk 0 reads its own d-key, not Bs:attn
    self.assertEqual((a["dBs0:attn"], a["Bs0"]), (-3.0, 67.5))
    self.assertAlmostEqual(sched.cost_table(a, 2)(0, "B", "attn"), (51.3 - 2.0) * 1.5)
    # A split backward's measured I / W (round 2's TPU7x xprof costs) scale as their chunk's B does -- a middle
    # chunk's own keys, else the generic ones -- and so does the split last chunk's W (W{C-1}, else WA, else A - AI
    # from a measured AI), its AI moving as A does less what W takes of the move; zb_costs then prices the rest.
    z = dict(t, A=102.0, I1=66.1, W1=26.4, I=60.0, W=24.0, AI=80.2, WA=26.4)
    c = dsp.split_costs(z, (1, 3, 2, 2))  # even 2: chunk 1 1.5 x, chunk 2 as measured, the last 1 x
    for key, want in (("I1", 66.1 * 1.5), ("W1", 26.4 * 1.5), ("I2", 60.0), ("W2", 24.0), ("AI", 80.2), ("WA", 26.4)):
      self.assertAlmostEqual(c[key], want, msg=key)
    self.assertNotIn("I0", c)  # chunk 0's backward never splits
    c = dsp.split_costs(z, (2, 3, 2, 1))  # the last chunk at half: its W halves, A loses half a chunk's layers
    self.assertAlmostEqual(c["WA"], 13.2)
    self.assertAlmostEqual(c["A"], 102.0 - 0.5 * 48.6)
    self.assertAlmostEqual(c["AI"], 80.2 - 0.5 * 48.6 + 13.2)
    self.assertAlmostEqual(dsp.split_costs(dict(z, W3=20.0), (2, 3, 2, 1))["W3"], 10.0)  # W{C-1} before WA
    c = dsp.split_costs(dict(t, AI=40.0), (2, 3, 2, 1))  # AI alone: its W is A - AI
    self.assertNotIn("WA", c)
    self.assertAlmostEqual(c["AI"], 40.0 - 0.5 * 48.6 + 0.5 * (58.3 - 40.0))
    k = sched.zb_costs(c, 4, last=True)
    self.assertAlmostEqual(k["WA"], 0.5 * (58.3 - 40.0), places=3)  # = A' - AI'

  def test_layer_split_scales_the_costs_given(self):
    """--costs merge over the defaults before the split, so their F / B / A are scaled too: under the
    defaults' per-chunk keys a user's F / B used to price nothing (review F5)."""
    p = dsp.Pipeline(
        tiny_xla(layers=6, **testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=2,
        global_batch=8,
        seq=256,
        schedule="1f1b",
        layer_split=(1, 2, 2, 1),
        costs={"F": 100.0, "B": 200.0, "A": 300.0},
    )
    self.assertAlmostEqual(p.costs["F1"], 100.0 * 2 / 1.5)
    self.assertAlmostEqual(p.costs["B2"], 200.0 * 2 / 1.5)
    self.assertAlmostEqual(p.costs["F0"], sched.TPU7X_COSTS["F0"] / 1.5)  # chunk 0: the default's F0
    self.assertAlmostEqual(p.costs["A"], 300.0 - 200.0 + 200.0 / 1.5)

  @pytest.mark.scheduled_only
  def test_overlap_one_microbatch_v_stash(self):
    """--mb 1 under V placement and the stash schedule: the head's gather
    moves to stage 0's update program, stash programs host the rest."""
    p = self._check(overlap=True, mb=1, placement="v", schedule="stash", stash="lite", stash_budget_gib=1.0)
    self.assertEqual(p.pf_fallback, [3])
    self.assertTrue(p.modes)

  @pytest.mark.scheduled_only
  def test_interleaved_pairs_stash_matches_reference(self):
    """Interleaved 1F1B's F&B pair programs and 'lite' stash programs (the
    composer's, ds_ep.compose) with `ragged_dot` experts: the
    grouped matmuls inside every stash forward / backward and pair go
    through named_gmm's pure-JAX path."""
    p = self._check(overlap=True, mb=4, schedule="interleaved", pair_stages="all", stash="lite", stash_budget_gib=1.0)
    self.assertTrue(p.fuse and p.modes)
    self.assertTrue(any(op.pair and any(dict(kw).get("stash") for _, kw in op.pair) for op in p.ops()))

  @pytest.mark.scheduled_only
  def test_sync_stage_run(self):
    """A synchronous stage's run: stage 1, synchronous (`sync_collective_stages`
    patched: on CPU no stage is) and not chunk 0's, reduce-scatters both its
    chunks in one program after its last backward, which sums no partial
    sumsq.  Precompiled -- that program too -- against the same pipeline with
    no synchronous stage, which reduce-scatters in stage 1's backwards, on the
    same weights and batch: the loss and every gradient bit for bit, the stage
    sumsqs within rounding (summed in another order).  One microbatch and the
    reference attention keep it to seconds."""
    scfg = tiny_xla(layers=4, fsdp=1, ep=2, fsdp_collectives="xla")
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(attention_kernel="reference"))
    with mock.patch.object(dsp, "sync_collective_stages", lambda meshes: frozenset({1})):
      end = pipe(scfg, n_micro=1, global_batch=2)
    plain = pipe(scfg, n_micro=1, global_batch=2)
    ps = {"end": end, "plain": plain}
    self.assertEqual((end.sync_stages, end.final_slots[1], end.n_partial_sq[1]), (frozenset({1}), (0, 1), 0))
    on_1 = lambda q: [j for j, (c, _) in q.rs_at.items() if q.place[c] == 1]
    self.assertEqual((on_1(end), bool(on_1(plain))), ([], True))  # plain: in stage 1's backwards
    trees, out = host_weights(end), {}
    for rs, p in ps.items():
      p.precompile(threads=16)
      params = p.params_from_chunks(trees)
      wc0 = [st.gather_first(w) for st, w in zip(p.stages, params)]
      batch = p.synthetic_batch(4)
      with compiles() as names:
        out[rs] = p.loss_and_grads_overlap(params, wc0, batch)
        jax.block_until_ready(out[rs])
      self.assertEqual(names, [], rs)  # precompile left nothing for the step to compile
    self.assertIsInstance(end.stages[1]._final[((0, 1), 0)], jax.stages.Compiled)
    (ge, se, le), (gf, sf, lf) = out["end"], out["plain"]
    assert_bitwise(self, (ge, le), (gf, lf), "synchronous stage vs none")
    np.testing.assert_allclose(np.asarray(se, np.float64), np.asarray(sf, np.float64), rtol=1e-6)

  @pytest.mark.scheduled_only
  def test_precompile_covers_every_program(self):
    """`precompile` compiles every program a step runs, so the step compiles
    nothing -- under --no-overlap too, where each step gathers all of a
    stage's weights (`gather`; `init_state` its first chunk's) -- and with
    --transport ppermute each boundary copy's program, recorded against both
    its stages.  Built ahead of time (`build(aot=True)`, as ds_ep_aot builds
    it, here over this host's devices: nothing placed) it compiles the same
    programs plus the gradient norm's scalar copies, which only a
    multi-controller run makes programs (one process hops with device_put)."""
    a = dsp.make_parser().parse_args(TINY_ARGS + ["--no-overlap", "--transport", "ppermute"])
    p = dsp.build(a)
    aot = dsp.build(a, meshes=dsp.stage_meshes(a.stages, a.fsdp, a.ep, layout=a.layout), aot=True)
    self.assertEqual((aot.placed, aot.local, aot.seeds), (frozenset(), frozenset({0, 1}), [None, None]))
    self.assertEqual((sorted(p.channels), p.sq_channels), ([(0, 1), (1, 0)], {}))
    self.assertEqual((sorted(aot.channels), sorted(aot.sq_channels)), ([(0, 1), (1, 0)], [(0, 1), (1, 0)]))
    n, _ = p.precompile(threads=16)
    n_aot, _ = aot.precompile(threads=16)
    stage_programs = ("gather", "gather_first", "reduce", "update")
    self.assertEqual(n, len(set(p.dispatched())) + len(stage_programs) * p.S + len(p.channels))
    names = lambda q: sorted(name for name, _ in q.compiled)
    self.assertEqual(n_aot, n + 2)
    self.assertEqual(names(aot), sorted(names(p) + ["xfer_sumsq_s0_to_s1", "xfer_sumsq_s1_to_s0"]))
    self.assertEqual([s for s in p.compiled_stage if isinstance(s, tuple)], list(p.channels))
    self.assertEqual([s for s in aot.compiled_stage if isinstance(s, tuple)], [*aot.channels, *aot.sq_channels])
    self.assertEqual({s for s in p.compiled_stage if not isinstance(s, tuple)}, {0, 1})
    for st in p.stages:
      for name in stage_programs:
        self.assertIsInstance(getattr(st, name), jax.stages.Compiled, name)
    self.assertTrue(all(isinstance(ch.prog, jax.stages.Compiled) for ch in p.channels.values()))
    params, opts, wc0 = p.init_state(p.params_from_chunks(host_weights(p)))
    batch, lrs = p.synthetic_batch(1), p.lr_arrays(1e-3)
    jax.block_until_ready((params, opts, batch, lrs))
    with compiles() as names:
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      jax.block_until_ready((params, opts, lacc))
    self.assertEqual(names, [])
    self.assertTrue(np.isfinite(float(jnp.sum(lacc))) and np.isfinite(float(gn)))
    with compiles() as names:  # and the count is no dead listener: a new program counts
      jax.jit(lambda x: x + 1)(np.float32(1))
    self.assertTrue(names)

  @pytest.mark.scheduled_only
  def test_precompile_explicit(self):
    """--shard-mode explicit with the fusion plan: `precompile` lowers every
    program from typed avals (`_avals`: the weights reduced, the accumulators
    unreduced) and compiles all a step runs -- the folded gathers and
    reduce-scatters, gather_first, update_gather, the standalone reduce -- so
    the step compiles nothing, and built ahead of time it compiles the same
    programs.  The step's loss and gradient norm are auto's, bit for bit."""
    args = TINY_ARGS + "--vv 2 --layers 4 --gbs 8 --mb 4".split()
    a = dsp.make_parser().parse_args(args + ["--shard-mode", "explicit"])
    p = dsp.build(a)
    aot = dsp.build(a, meshes=dsp.stage_meshes(a.stages, a.fsdp, a.ep, layout=a.layout), aot=True)
    self.assertTrue(p.pf and p.rs_at and p.self_pf_at is not None)
    n, _ = p.precompile(threads=16)
    n_aot, _ = aot.precompile(threads=16)
    self.assertEqual(n, len(set(p.dispatched())) + sum(2 + bool(f) for f in p.final_slots))
    self.assertEqual(sorted(name for name, _ in aot.compiled), sorted(name for name, _ in p.compiled))
    trees, out = host_weights(p), []
    for q in (p, dsp.build(dsp.make_parser().parse_args(args))):
      params, opts, wc0 = q.init_state(q.params_from_chunks(trees))
      batch, lrs = q.synthetic_batch(1), q.lr_arrays(1e-3)
      jax.block_until_ready((params, opts, wc0, batch, lrs))
      with compiles() as names:
        params, opts, wc0, lacc, gn = q.step(params, opts, wc0, batch, lrs)
        jax.block_until_ready((params, opts, lacc))
      out.append((float(jnp.sum(lacc)), float(gn)))
      if q is p:
        self.assertEqual(names, [])
    self.assertEqual(out[0], out[1])

  def test_run_steps_keeps_no_earlier_step_state(self):
    """run_steps empties the caller's [params, opts, wc0], so no step's inputs
    outlive it: the gathered weights wc0 are not donated, and main() keeping
    the first step's for the whole run-ahead loop held a second copy, which
    on 2 x v6e-128 fragmented stage 0's HBM."""

    class Box:  # a pytree of arrays, as far as the loop cares
      pass

    class Fake:
      step_gnorms = ()

      def __init__(self):
        self.refs, self.alive = [], []

      def step(self, params, opts, wc0, batch, lrs):
        del params, opts, batch, lrs
        gc.collect()
        self.alive.append(sum(r() is not None for r in self.refs))  # the earlier steps' wc0
        self.refs.append(weakref.ref(wc0))
        return Box(), Box(), Box(), 0.0, 1.0

    for max_ahead in (None, 0, 1):
      p, state = Fake(), [Box(), Box(), Box()]
      out = dsp.run_steps(p, state, 3, None, None, max_ahead)
      self.assertEqual(state, [], max_ahead)
      self.assertEqual(p.alive, [0, 0, 0], max_ahead)
      self.assertEqual(len(out), 5)

  def test_run_steps_waits_beside_the_dispatch(self):
    """run_steps with a bound dispatches step k once step k - max_ahead - 1 has finished, never sooner, and waits for
    the steps to finish on a thread of its own, in step order: a wait for a step that has finished is a round trip
    through the IFRT proxy, which in the dispatch loop slowed short Pathways steps by as much.  A step whose wait
    raises (its error) stops the loop with that error."""
    finished, failing, threads = [], set(), set()

    class Outs:
      """A step's never-donated outputs, finished when the test says so (at once if `at_once`)."""

      at_once = False

      def __init__(self, k):
        self.k, self.ready = k, threading.Event()
        if self.at_once:
          self.ready.set()

      def block_until_ready(self):  # jax.block_until_ready's call for a leaf that is no jax array
        threads.add(threading.current_thread())
        self.ready.wait(60)
        if self.k in failing:
          raise RuntimeError(f"step {self.k} failed")
        finished.append(self.k)
        return self

    class Fake:
      step_gnorms = ()

      def __init__(self):
        self.outs, self.seen = [], []

      def step(self, params, opts, wc0, batch, lrs):
        self.seen.append((len(self.outs), len(finished)))  # (this step, the steps finished when it was dispatched)
        self.outs.append(Outs(len(self.outs)))
        return params, opts, wc0, self.outs[-1], 1.0

    def until(cond):
      deadline = time.monotonic() + 60
      while not cond():
        self.assertLess(time.monotonic(), deadline)
        time.sleep(0.001)

    for max_ahead in (0, 1, 3):
      with self.subTest(max_ahead=max_ahead):
        finished.clear()
        threads.clear()
        p, n = Fake(), 6
        loop = threading.Thread(target=dsp.run_steps, args=(p, [None] * 3, n, None, None, max_ahead))
        loop.start()
        for k in range(n):  # the steps finish one by one, each once the loop has dispatched what it may before it
          until(lambda k=k, p=p, n=n, ahead=max_ahead: len(p.outs) >= min(k + ahead + 1, n))
          p.outs[k].ready.set()
        loop.join(60)
        self.assertFalse(loop.is_alive())
        self.assertEqual(len(threads), 1)
        self.assertNotIn(loop, threads)  # beside the dispatch
        until(lambda n=n: len(finished) == n)  # the last steps' waits go on after the loop: its caller blocks at the end
        self.assertEqual(finished, list(range(n)))
        for k, done in p.seen:
          self.assertGreaterEqual(done, k - max_ahead, p.seen)  # never more than max_ahead ahead
    p, Outs.at_once = Fake(), True
    failing.add(1)
    with self.assertRaisesRegex(RuntimeError, "step 1 failed"):
      dsp.run_steps(p, [None] * 3, 5, None, None, 0)
    self.assertEqual([k for k, _ in p.seen], [0, 1])  # stopped before step 2: its wait for step 1 raised

  @pytest.mark.scheduled_only
  def test_cli_bench_without_running_ahead(self):
    """`main --bench --max-ahead 0 --data maxtext --json`: the bench loop
    dispatches a step only once this process has seen the one before it
    finish (every local stage's update, `Pipeline.step_gnorms`, and the loss
    partials), and the run record holds the loop's timings.  Its warmup on
    MaxText's batch is a run-ahead one's (no --max-ahead; the same pipeline,
    so no second compile), and every stage holds the same global norm."""
    runs, real = [], dsp.build

    def build(a, **kw):
      if runs:  # the second run: the first one's pipeline, precompiled
        return runs[0]
      p = real(a, **kw)
      step, p.ready = p.step, []

      def step_after_the_last(*args):  # were the last step's outputs ready when this one was dispatched?
        last = getattr(step_after_the_last, "last", None)
        p.ready.append(last is None or all(x.is_ready() for x in jax.tree.leaves(last)))
        out = step(*args)
        step_after_the_last.last = (p.step_gnorms, out[3])
        return out

      p.step = step_after_the_last
      runs.append(p)
      return p

    path = os.path.join(self.enterContext(tempfile.TemporaryDirectory()), "run.json")
    args = TINY_ARGS + ["--warmup", "1", "--data", "maxtext", "--bench", "--steps", "2"]
    with mock.patch.object(dsp, "build", build), contextlib.redirect_stdout(io.StringIO()) as log:
      dsp.main(args + ["--max-ahead", "0", "--json", path])
      p = runs[0]
      ready = list(p.ready)
      ref = dsp.main(args + ["--no-precompile"])
    with open(path) as f:
      rec = json.load(f)
    for r in (rec, ref):
      for k in ("step_ms", "blocked_step_ms", "blocked_step_median_ms", "blocked_step_mean_ms"):
        self.assertTrue(np.isfinite(r[k]) and r[k] > 0, k)
    self.assertEqual((rec["config"]["max_ahead"], ref["config"]["max_ahead"]), (0, None))
    self.assertEqual(len(rec["blocked_steps_ms"]), 3)  # max(3, --steps)
    self.assertEqual(rec["blocked_step_median_ms"], statistics.median(rec["blocked_steps_ms"]))
    self.assertEqual(ready, [True] * 6)  # the warmup step, 2 bench steps, 3 blocked ones
    self.assertTrue(np.isfinite(rec["warmup_losses"]).all() and np.isfinite(rec["warmup_gnorms"]).all())
    self.assertEqual((rec["warmup_losses"], rec["warmup_gnorms"]), (ref["warmup_losses"], ref["warmup_gnorms"]))
    self.assertEqual(len(p.step_gnorms), p.S)
    self.assertEqual(len({float(g) for g in p.step_gnorms}), 1)
    self.assertIn(f"data: maxtext (under {dsp.MAXTEXT_BATCH_PRNG}), inputs sha256", log.getvalue())
    self.assertIn("(mean of 2, at most 0 step(s) ahead)", log.getvalue())
    self.assertIn("(mean of 2, run-ahead)", log.getvalue())


@pytest.mark.scheduled_only
@testing.needs_devices(4)
class ExplicitSharding(unittest.TestCase):
  """shard_mode='explicit' (ds_ep.StageConfig): Explicit stage-mesh axes, the
  gathered weight copies typed reduced and the gradient accumulators unreduced
  over the axes their gradients are partial sums over -- the programs'
  arguments and outputs typed, their bodies, collectives and folding 'auto''s.
  No MaxText kernel here (`ragged_dot` experts, the argsort dispatch); with
  MaxText's kernels see PipelineVsReference.test_explicit_v_cap2_bitwise_vs_auto,
  with context parallelism cp_test.CPPipeline.test_cp2_explicit_vs_auto."""

  def test_types(self):
    """The stage meshes are Explicit.  Every gathered copy -- gather_first's,
    the update's gather of the next step, a forward's prefetch ('G') and its
    self-prefetch, --no-overlap's stage gather -- is the compute layout typed
    reduced over its leaf's partial axes (both mesh axes; the routed experts
    'fsdp' alone), and every program takes its weights so; every accumulator
    is the compute layout typed unreduced over them, without the leading axis,
    in the same bytes a device as 'auto''s.  And jax's own sum of an
    accumulator -- a jit-level reshard to the storage layout, which adds the
    devices' partials over the unreduced axes -- is the stage's reduce program's
    gradient: an unreduced claim over an axis the partials were already summed
    over would multiply the gradient by that axis's size (an n x bug this check was written for)."""
    ps = {
        ov: pipe(explicit(tiny_xla(**testing.stage_axes())), n_micro=2, global_batch=8, overlap=ov)
        for ov in (True, False)
    }
    auto = pipe(tiny_xla(**testing.stage_axes()), n_micro=2, global_batch=8, overlap=False)
    for p in ps.values():
      self.assertEqual({t for m in p.meshes for t in m.axis_types}, {jax.sharding.AxisType.Explicit})
    self.assertEqual({t for m in auto.meshes for t in m.axis_types}, {jax.sharding.AxisType.Auto})
    self.assertTrue(all(a is b for a, b in zip(dsp.typed_meshes(auto.meshes, "auto"), auto.meshes)))
    lv = lambda t: jax.tree.leaves(t, is_leaf=ds_ep._is_spec)
    cspec = lambda ch: lv(ds_ep.chunk_compute_specs(ch))

    def typed(tree, ch, kind, idx=None, what=""):
      """Every leaf of `tree` (arrays or avals; chunk ch's leaves `idx`) is its compute spec typed `kind`."""
      cs, xs = cspec(ch), jax.tree.leaves(tree)
      idx = range(len(cs)) if idx is None else idx
      self.assertEqual(len(xs), len(idx), what)
      for x, i in zip(xs, idx):
        self.assertEqual(spec_of(x), ds_ep.typed_spec(cs[i], kind), f"{what} chunk {ch.index} leaf {i}")
        self.assertEqual(tuple(x.shape), jax.tree.leaves(ds_ep.chunk_shapes(p.scfg, ch), is_leaf=ds_ep._is_shape)[i])
      return len(xs)

    # every per-microbatch program: its weights in, its gathered leaves and accumulator out
    p = ps[True]
    seen = collections.Counter()
    for c, desc in dict.fromkeys(p.dispatched()):
      av = p._desc_avals(c, desc)
      out = jax.eval_shape(p.progs[c].program(desc), *av)
      kinds = [k for k, _ in desc]
      ch = p.chunks[c]
      if "self_leaves" in dict(desc[0][1]):
        idx = dict(desc[0][1])["self_leaves"]
        rest = [i for i in range(len(cspec(ch))) if i not in idx]
        seen["w"] += typed(av[0], ch, "reduced", rest, f"{desc} weights")
        seen["self-prefetch"] += typed(out[-1], ch, "reduced", idx, f"{desc} self-prefetch")
      elif kinds[0] in "FBA":
        seen["w"] += typed(av[0], ch, "reduced", what=f"{desc} weights")
      if kinds[-1] in "GR":
        kw = dict(desc[-1][1])
        if kinds[-1] == "G":
          seen["prefetch"] += typed(out[-1], kw["other"], "reduced", kw.get("leaves"), f"{desc} prefetch")
        else:  # the leaves of another chunk's accumulator it reduces
          seen["reduced gacc"] += typed(av[-1], kw["other"], "unreduced", kw.get("leaves"), f"{desc} reduce")
      if kinds[0] in "BA" and len(kinds) == 1 + (kinds[-1] == "R") and "rs_self" not in dict(desc[0][1]):
        g = out[-1 - (kinds[-1] == "R") * 2 - (kinds[0] == "A")]  # dx?, gacc[, loss][, grads, sumsq]
        seen["gacc"] += typed(g, ch, "unreduced", what=f"{desc} accumulator")
    self.assertTrue(all(seen[k] for k in ("w", "self-prefetch", "prefetch", "reduced gacc", "gacc")), seen)
    # the stage programs' gathers, and the arrays themselves
    for p in ps.values():
      params, _, wc0 = p.init(jax.random.key(1))
      for s, st in enumerate(p.stages):
        chs = p.stage_chunks[s]
        params_av = [p._avals(s, "storage", c) for c in chs]
        for i, w in zip(p.gather_slots[s], wc0[s]):
          fl = st.first_leaves if i == 0 else None
          typed(w, chs[i], "reduced", fl, f"wc0 s{s}")
        if p.overlap:
          grads_av = [p._avals(s, "grads", c) for c in chs]
          opt = {"mu": params_av, "nu": params_av, "count": p._avals(s, "scalar_i")}
          nxt = jax.eval_shape(
              st.update_gather, params_av, opt, grads_av, [p._avals(s, "scalar_f")] * p.S, p._avals(s, "scalar_f")
          )[-1]
          for i, w in zip(p.gather_slots[s], nxt):
            typed(w, chs[i], "reduced", st.first_leaves if i == 0 else None, f"update_gather s{s}")
        else:
          for c, w in zip(chs, jax.eval_shape(st.gather, params_av)):
            typed(w, c, "reduced", what=f"gather s{s}")
    # the accumulators, against 'auto''s bytes and the stage reduce
    p = ps[False]
    params, _, _ = p.init(jax.random.key(1))
    batch = p.synthetic_batch(2)
    gaccs, _ = p.loss_and_grads(params, batch)
    ga_auto, _ = auto.loss_and_grads(auto.params_from_chunks(p.chunk_trees(params)), batch)
    per_device = lambda t: sorted(sh.data.nbytes for x in jax.tree.leaves(t) for sh in x.addressable_shards)
    for s in range(p.S):
      for c, g in zip(p.stage_chunks[s], gaccs[s]):
        typed(g, c, "unreduced", what=f"gacc s{s}")
      self.assertEqual(per_device(gaccs[s]), per_device(ga_auto[s]))
      storage = jax.tree.map(
          lambda sp: NamedSharding(p.meshes[s], sp),
          [ds_ep.chunk_storage_specs(c) for c in p.stage_chunks[s]],
          is_leaf=ds_ep._is_spec,
      )
      summed = jax.jit(lambda g: jax.tree.map(jax.sharding.reshard, g, storage))(gaccs[s])
      grads, _ = p.stages[s].reduce(gaccs[s])
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(summed)[0], jax.tree.leaves(grads)):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        np.testing.assert_allclose(
            a,
            b,
            rtol=1e-5,
            atol=1e-6 * np.abs(b).max(),
            err_msg=f"s{s} {jax.tree_util.keystr(path)}: jax's sum of the unreduced "
            f"accumulator is {np.sum(a * b) / np.sum(b * b):.4g} x the reduce's",
        )
    print(
        f"explicit types: {dict(seen)} leaves checked; accumulators as many bytes as auto's, their jax sum "
        "the stage reduce's"
    )

  @testing.slow
  def test_bitwise_vs_auto_circular(self):
    """Circular placement, overlap on (the fusion plan's gathers, reduce-
    scatters and rs_self) and off (the stage gather and reduce): explicit and
    auto give the same gradients, sumsqs and loss, and the same params, losses
    and gradient norms over two steps, bit for bit."""
    for overlap in (True, False):
      runs = [
          mode_run(pipe(sc, n_micro=2, global_batch=8, overlap=overlap), 2, 4)
          for sc in (tiny_xla(**testing.stage_axes()), explicit(tiny_xla(**testing.stage_axes())))
      ]
      assert_same_run(self, runs[1], runs[0], f"circular overlap={overlap}")
      print(
          f'explicit == auto, circular overlap={overlap}: {len(runs[0]["grads"])} gradient leaves, loss '
          f'{runs[0]["loss"]:.6f}, two steps\' params bitwise'
      )

  @testing.slow
  def test_bitwise_vs_auto_pairs_stash(self):
    """V placement under the stash schedule -- 'lite' stash programs, the head's gather and the rs_self of stage 0
    -- and interleaved 1F1B's F&B pair programs with 'lite' stash programs (some inside pairs), with lax FSDP
    collectives: explicit and auto bit for bit (`mode_run`, one step)."""
    sc = tiny_xla(fsdp_collectives="xla", **testing.stage_axes())
    for what, kw in (
        ("v + stash", dict(placement="v", schedule="stash")),
        ("interleaved pairs + stash", dict(schedule="interleaved", pair_stages="all")),
    ):
      runs = []
      for scfg in (sc, explicit(sc)):
        p = pipe(scfg, stash="lite", stash_budget_gib=1.0, n_micro=4, global_batch=16, **kw)
        self.assertTrue(p.modes and (p.fuse or kw["schedule"] == "stash"), what)
        runs.append(mode_run(p, 3, 5, steps=1))
      assert_same_run(self, runs[1], runs[0], what)


@testing.needs_devices(8, "fsdp 2 x expert 2 stages: a reduce-scatter one axis at a time against both at once takes both")
class ReduceAxes(unittest.TestCase):
  """StageConfig.rs_axes under lax FSDP collectives: a gradient leaf stored
  over both mesh axes reduce-scatters in one psum_scatter over both ('joint',
  the default) or one axis at a time ('each': on a v5p-128 stage the joint
  one compiles to a full-size all-reduce, slices_layout_test.
  CompileOnlyTopology.test_reduce_scatters_one_axis_at_a_time)."""

  def test_programs(self):
    """Read off the stage reduce program's jaxpr (nothing compiles): 'each'
    gives every leaf stored over both axes a psum_scatter over 'fsdp' and one
    over 'expert' where 'joint' gives it one over both; the leaves stored over
    one axis reduce alike either way."""

    def scatters(rs_axes):
      p = pipe(tiny_xla(fsdp_collectives="xla", rs_axes=rs_axes), n_micro=2, global_batch=8, overlap=False)
      out = collections.Counter()

      def walk(jaxpr):
        for eqn in jaxpr.eqns:
          if eqn.primitive.name == "reduce_scatter":
            ax = eqn.params["axis_name"]
            out[tuple(ax) if isinstance(ax, (tuple, list)) else (ax,)] += 1
          for v in eqn.params.values():
            for sub in v if isinstance(v, (tuple, list)) else (v,):
              if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
                walk(getattr(sub, "jaxpr", sub))

      walk(jax.make_jaxpr(p.stages[0].reduce)([p._avals(0, "gacc", c) for c in p.stage_chunks[0]]).jaxpr)
      return out, p

    each, p = scatters("each")
    joint, _ = scatters("joint")
    both = sum(
        any(isinstance(d, tuple) and len(d) == 2 for d in sp)
        for c in p.stage_chunks[0]
        for sp in jax.tree.leaves(ds_ep.chunk_storage_specs(c), is_leaf=ds_ep._is_spec)
    )
    self.assertGreater(both, 0)
    self.assertEqual((joint[ds_ep.FE], each[ds_ep.FE]), (both, 0))
    self.assertEqual((each[("fsdp",)], each[("expert",)]), (joint[("fsdp",)] + both, joint[("expert",)] + both))
    print(f"psum_scatters by axes: each {dict(each)}, joint {dict(joint)} ({both} leaves stored over both axes)")

  @pytest.mark.scheduled_only
  def test_each_vs_joint(self):
    """'each' against the monolithic reference (a shard put in the wrong
    place would be far off), its reduces folded into the plan's programs and
    in the stages' reduce programs; and against 'joint': with fp32
    reduce-scatters to float32 rounding, with bf16 ones (the study's rs_dtype)
    within 4 bf16 ulps of each leaf's largest value -- 'each' rounds the sum
    of one axis to bf16 before adding the other's (measured: 2 ulps at most,
    here and on a v6e-4)."""
    p = pipe(tiny_xla(fsdp_collectives="xla", rs_axes="each"), n_micro=2, global_batch=8)
    self.assertEqual({c.rs_axes for c in p.stage_scfg}, {"each"})
    for overlap in (True, False):
      compare_with_reference(self, p, overlap, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)
    for rs, tol in ((jnp.float32, 2.0**-21), (jnp.bfloat16, 2.0**-6)):
      got = {}
      for axes in ("each", "joint"):
        q = pipe(tiny_xla(fsdp_collectives="xla", rs_dtype=rs, rs_axes=axes), n_micro=2, global_batch=8)
        params, _, wc0 = q.init(jax.random.key(1))
        grads, _, _ = q.loss_and_grads_overlap(params, wc0, q.synthetic_batch(3))
        got[axes] = [np.asarray(x, np.float64) for x in jax.tree.leaves(grads)]
      worst = 0.0
      for a, b in zip(got["each"], got["joint"], strict=True):
        scale = float(np.abs(b).max())
        worst = max(worst, float(np.abs(a - b).max()) / scale)
        self.assertLessEqual(float(np.abs(a - b).max()), tol * scale, jnp.dtype(rs).name)
      print(f"rs_dtype {jnp.dtype(rs).name}: max |each - joint| / max |joint| {worst:.3g}")

  def test_vocab_leaves_stay_joint(self):
    """'each' reduces a leaf with an unsharded dim of 16384 or more that is
    no multiple of 512 (a vocab: libtpu 0.0.46 runs its per-axis pair on the
    TensorCore, ds_ep._RS_ALIGN) in one psum_scatter over both axes, in the
    embedding's orientation and the head's, from either shard mode's
    accumulator block; an aligned one, and a smaller one, one axis at a
    time; the sums are numpy's either way."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:8]).reshape(2, 4), ds_ep.FE)
    rng = np.random.default_rng(0)

    def scatters(jaxpr, out):
      for eqn in jaxpr.eqns:
        if eqn.primitive.name == "reduce_scatter":
          ax = eqn.params["axis_name"]
          out.append(tuple(ax) if isinstance(ax, (tuple, list)) else (ax,))
        for v in eqn.params.values():
          for sub in v if isinstance(v, (tuple, list)) else (v,):
            if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
              scatters(getattr(sub, "jaxpr", sub), out)
      return out

    for rows, joint in ((16400, True), (16896, False), (1000, False)):
      for shape, spec in (((rows, 64), P(None, ds_ep.FE)), ((64, rows), P(ds_ep.FE, None))):
        host = rng.standard_normal((8,) + shape).astype(np.float32)
        g = jax.device_put(host, NamedSharding(mesh, P(ds_ep.FE, None, None)))
        for per_axis, lead in ((True, True), (True, False), (False, True)):

          def body(x, lead=lead, per_axis=per_axis, spec=spec):
            return ds_ep._scatter_leaf(
                x if lead else x[0], spec, P(None, None), jnp.float32, "xla", lead=lead, per_axis=per_axis
            )

          f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P(ds_ep.FE, None, None), out_specs=spec, check_vma=False))
          want = [ds_ep.FE] if joint or not per_axis else [("fsdp",), ("expert",)]
          with self.subTest(shape=shape, per_axis=per_axis, lead=lead):
            self.assertEqual(scatters(jax.make_jaxpr(f)(g).jaxpr, []), want)
            np.testing.assert_allclose(np.asarray(f(g)), host.sum(0), rtol=1e-5, atol=1e-5)

  def test_flag(self):
    """--rs-axes reaches the StageConfig through `build`, 'joint' by default,
    and the summary names 'each'."""
    ap = dsp.make_parser()
    self.assertEqual(ap.parse_args(TINY_ARGS).rs_axes, "joint")
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(TINY_ARGS + ["--rs-axes", "both"])
    p, q = (dsp.build(ap.parse_args(TINY_ARGS + ["--rs-axes", r])) for r in ("joint", "each"))
    self.assertEqual((p.scfg.rs_axes, q.scfg.rs_axes), ("joint", "each"))
    self.assertNotIn("rs_axes", p.summary())
    self.assertIn("rs_axes=each", q.summary())


def check_repeat_plan(t, p):
  """fsdp_scope 'repeat''s plan, walked op by op on every stage: each op finds
  its chunk's weights whole; a gather delivers every leaf of its chunk once,
  inside a visit of another chunk; the weights go after their visit's last
  op, and by the stage's last op all of them -> per stage, the most chunks
  whose weights (whole or arriving) it holds at once."""
  C, order, place = p.C, p.order, p.place
  tag = f"{p.schedule_name} S={p.S} V={p.V} M={p.M}"
  last_of_visit = {js[-1] for vs in p.visits for _, js in vs}
  t.assertLessEqual(set(p.release_at), last_of_visit, tag)
  peaks = []
  for s in range(p.S):
    have = {p.stage_chunks[s][i].index: set(range(p.n_leaves[p.stage_chunks[s][i].index])) for i in p.gather_slots[s]}
    peak = len(have)
    for j, (c, k, m) in enumerate(order):
      if place[c] != s or (c == C - 1 and k == "B"):
        continue
      t.assertEqual(have.get(c), set(range(p.n_leaves[c])), (tag, p.op_name(j), "weights not whole"))
      if j in p.pf:
        b, g = p.pf[j]
        t.assertNotEqual(b, c, (tag, p.op_name(j)))
        t.assertTrue(set(g).isdisjoint(have.get(b, ())), (tag, p.op_name(j), "a leaf gathered twice"))
        t.assertTrue(set(g).isdisjoint(getattr(p, "direct", {}).get(b, ())), (tag, p.op_name(j), "read in place"))
        t.assertLessEqual(p.pf_n[j], p.pf_split, tag)
        have.setdefault(b, set(getattr(p, "direct", {}).get(b, ()))).update(g)
      peak = max(peak, len(have))
      for b in p.release_at.get(j, ()):
        del have[b]
    t.assertEqual(have, {}, (tag, s))
    peaks.append(peak)
  return peaks


@testing.needs_devices(4)
class RepeatScope(unittest.TestCase):
  """--fsdp-scope repeat: a chunk's gathered weights live one visit at a time
  (the circular pipeline's per-repeat FSDP), under --schedule breadth."""

  def tearDown(self):
    # The executables of a CPU pipeline map their code into the process, against vm.max_map_count's 65530, and a dead
    # pipeline is cyclic garbage: free them before the next test, not whenever Python's own full collection comes
    # (BatchSplitLineage.tearDown's case; test_bitwise_vs_step_more frees each case's too).
    gc.collect()

  def plan(self, S, V, M, **kw):
    return pipe(
        tiny(layers=S * V, fsdp=1, ep=2),
        n_stages=S,
        chunks_per_stage=V,
        n_micro=M,
        global_batch=2 * M,
        schedule="breadth",
        fsdp_scope="repeat",
        **kw,
    )

  def test_breadth_plans(self):
    """At S=2-4, V=1-4, M=2-8: the plan holds (`check_repeat_plan`), with at
    most two chunks' weights on a stage at once (V=1: one); the update
    gathers each stage's first chunk only; the weights of every chunk but a
    stage's turnaround one are gathered twice a step, the second time inside
    backwards or the fused last chunk; and the order is `search`'s at S=4.
    (S=4, two devices a stage, where there are 8 devices.)"""
    four = jax.device_count() >= 8
    for S, V, M in ((2, 1, 4), (2, 2, 2), (2, 2, 4), (2, 3, 4), (2, 4, 8), (4, 2, 4), (4, 2, 8)):
      if S == 4 and not four:
        continue
      with self.subTest(S=S, V=V, M=M):
        p = self.plan(S, V, M)
        self.assertEqual(max(check_repeat_plan(self, p)), min(V, 2))
        self.assertEqual(p.gather_slots, [(0,)] * S)
        gathers = collections.Counter(b for b, _ in p.pf_gathers)
        for s in range(S):
          turn = p.stage_chunks[s][-1].index
          for ch in p.stage_chunks[s][1:]:
            self.assertEqual(gathers[ch.index], 1 if ch.index == turn else 2, (S, V, M, ch.index))
          self.assertEqual(gathers[p.stage_chunks[s][0].index], 0 if V == 1 else 1)
        kinds = {("A" if p.order[j][0] == p.C - 1 else p.order[j][1]) for j in p.pf}
        self.assertEqual(kinds, set() if V == 1 else {"F", "B", "A"}, (S, V, M))
        # the chunks' accumulators and reduce-scatters are the step scope's: the same plan
        q = pipe(
            tiny(layers=S * V, fsdp=1, ep=2),
            n_stages=S,
            chunks_per_stage=V,
            n_micro=M,
            global_batch=2 * M,
            schedule="breadth",
            fsdp_scope="step",
        )
        self.assertEqual((p.order, p.rs_at, p.rs_self, p.final_slots), (q.order, q.rs_at, q.rs_self, q.final_slots))
    if four:
      self.assertEqual(
          self.plan(4, 2, 8).order,
          pipe(tiny(layers=8, fsdp=1, ep=2), n_stages=4, n_micro=8, global_batch=16, schedule="search").order,
      )

  @pytest.mark.scheduled_only
  def test_v_placement(self):
    """V placement, the default's order and scope there too: stage s holds
    chunks s and 2S-1-s, visited forward, turnaround (stage 0's is the fused
    last chunk), backward.  The plan holds at S=2-4, M=1-8, with two chunks'
    weights at most; one microbatch loop, two steps and the params are the
    step scope's bit for bit.  (S stages of two devices each, as many as the devices hold.)"""
    for S, M in ((2, 1), (2, 2), (2, 8), (3, 4), (4, 4), (4, 8)):
      if 2 * S > jax.device_count():
        continue
      with self.subTest(S=S, M=M):
        p = pipe(
            tiny(layers=2 * S, fsdp=1, ep=2),
            n_stages=S,
            chunks_per_stage=2,
            n_micro=M,
            global_batch=2 * M,
            placement="v",
        )
        self.assertEqual((p.schedule_name, p.fsdp_scope), ("breadth", "repeat"))
        self.assertEqual([len(v) for v in p.visits], [3] * S)
        self.assertEqual(max(check_repeat_plan(self, p)), 2)
    self.bitwise("V placement, M=2", tiny_xla(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4, placement="v")

  @pytest.mark.scheduled_only
  def test_one_microbatch(self):
    """At M=1 a stage's second visit has no free op before it on stage 0 --
    0F0 gathers its own routed experts -- so the update gathers that chunk
    too (pf_fallback, which the summary explains), and it is held until its
    first visit; the step's numbers are the step scope's, bit for bit."""
    p = self.plan(2, 2, 1)
    check_repeat_plan(self, p)
    self.assertEqual((p.gather_slots, p.pf_fallback), ([(0, 1), (0,)], [2]))
    self.assertIn(
        "gathered in update: s0: c[0, 2], s1: c[1] (c[2]: no op of an earlier visit on the stage to " "gather in)",
        p.plan_summary(),
    )
    q = self.bitwise("M=1", tiny_xla(layers=4, fsdp=1, ep=2), n_micro=1, global_batch=2)
    self.assertEqual((q.gather_slots, q.pf_fallback), ([(0, 1), (0,)], [2]))
    self.assertIn(2, {c for cs in q.release_at.values() for c in cs})  # released after its visit

  def test_refusals(self):
    """An order that visits a chunk more than twice (gpipe and 1f1b at V > 1
    and M > S are depth-first), no fusion plan, F&B pairs, an unknown scope:
    refused before anything is built; pf_split 0 leaves every visit without
    a host, as under 'step': the update gathers each stage's chunks and the
    visits keep them."""
    scfg = tiny(layers=4, fsdp=1, ep=2)
    for schedule in ("gpipe", "1f1b"):
      with self.assertRaisesRegex(ValueError, "more than twice.*schedule 'breadth'"):
        pipe(scfg, schedule=schedule, fsdp_scope="repeat")
    with self.assertRaisesRegex(ValueError, "needs the fusion plan"):
      pipe(scfg, schedule="breadth", fsdp_scope="repeat", overlap=False)
    with self.assertRaisesRegex(ValueError, "F&B pairs"):
      pipe(scfg, schedule="interleaved", pair_stages="all", fsdp_scope="repeat")
    with self.assertRaisesRegex(ValueError, "fsdp_scope must be 'step' or 'repeat'"):
      pipe(scfg, fsdp_scope="layer")
    # one chunk per stage: gpipe's single visit a stage is fine
    p = pipe(tiny(layers=2, fsdp=1, ep=2), chunks_per_stage=1, schedule="gpipe", fsdp_scope="repeat")
    self.assertEqual(max(check_repeat_plan(self, p)), 1)
    self.assertEqual(p.pf, {})
    p = pipe(scfg, n_micro=8, global_batch=16, schedule="breadth", fsdp_scope="repeat", pf_split=0)
    self.assertEqual((p.pf, p.gather_slots), ({}, [(0, 1), (0, 1)]))
    self.assertEqual(max(check_repeat_plan(self, p)), 2)

  def bitwise(self, what, scfg, **kw):
    """`mode_run` of the repeat scope against the step scope's, under the
    breadth-first order, bit for bit (`assert_same_run`) -> the repeat
    pipeline."""
    p = pipe(scfg, schedule="breadth", fsdp_scope="repeat", **kw)
    q = pipe(scfg, schedule="breadth", fsdp_scope="step", **kw)
    self.assertEqual((p.fsdp_scope, q.fsdp_scope), ("repeat", "step"))
    assert_same_run(self, mode_run(p, 1, 2), mode_run(q, 1, 2), what)
    return p

  @pytest.mark.scheduled_only
  def test_bitwise_vs_step(self):
    """The step scope's numbers bit for bit -- the same gathered values and the
    same accumulation, only held for less long: one microbatch loop's reduced
    gradients, the stage sumsqs and the loss, then two training steps' losses,
    gradient norms and params -- at V=2, M=2, where the turnaround visit's
    backwards and the fused last chunk host the re-gathers."""
    p = self.bitwise("V=2 M=2", tiny_xla(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4)
    self.assertEqual({p.order[j][1] + ("A" if p.order[j][0] == p.C - 1 else "") for j in p.pf}, {"F", "B", "FA"})

  @testing.slow
  def test_bitwise_vs_step_more(self):
    """As test_bitwise_vs_step, at M=4; with three chunks a stage, where
    backwards host a reduce-scatter and a gather both (B + R + G); typed
    stage programs (explicit); context parallelism; and per-microbatch
    reduce-scatters."""
    base = tiny_xla(layers=4, fsdp=1, ep=2)
    cases = [
        ("V=2 M=4", base, dict(n_micro=4, global_batch=8)),
        ("V=3 M=4", tiny_xla(layers=6, fsdp=1, ep=2), dict(chunks_per_stage=3, n_micro=4, global_batch=8)),
        ("explicit", explicit(base), dict(n_micro=4, global_batch=8)),
        ("cp=2", base, dict(n_micro=4, global_batch=4, cp=2)),
        ("grad_reduce=microbatch", dataclasses.replace(base, grad_reduce="microbatch"), dict(n_micro=4, global_batch=8)),
    ]
    for what, scfg, kw in cases:
      with self.subTest(what):
        p = self.bitwise(what, scfg, **kw)
        if what == "V=3 M=4":
          self.assertTrue(any(op.pf and op.rs for op in p.ops()), what)
      # A case's two pipelines map 27k to 44k regions of compiled code (jaxlib 0.12.0.1, YNN fusions off): two cases'
      # worth, left to Python's own collection, ran the process up to vm.max_map_count, and XLA:CPU's compile aborted
      # or segfaulted (testing.CPU_HEAP_BUG_JAXLIBS's note: how to tell that from the heap bug).
      p = None
      gc.collect()

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "fsdp 2 x expert 2 stages: --gather-axes each against joint takes both axes")
  def test_pins_and_gather_axes(self):
    """The fusion plan's pin and gather-axes options under the repeat scope.
    --prefetch-pin: the gathering forwards are pinned -- 'auto' all of them,
    a SPEC those holding a leaf it names -- and their descriptors carry the
    pins; the backwards and the fused last chunk that host a re-gather take
    none, and the summary lists them unpinned.  --gather-axes each: those
    re-gathers go one axis at a time too, where 'joint' gathers a leaf stored
    over both axes over both at once.  Pinned, and pinned gathering one axis
    at a time, the step trains as the unpinned joint one, bit for bit."""
    scfg = tiny_xla(layers=4, fsdp_collectives="xla", **testing.stage_axes())
    kw = dict(n_micro=4, global_batch=16, schedule="breadth", fsdp_scope="repeat")
    p = pipe(scfg, prefetch_pin="auto", **kw)
    check_repeat_plan(self, p)
    fwd = {j for j in p.pf if p.order[j][1] == "F" and p.order[j][0] != p.C - 1} | {p.self_pf_at}
    self.assertEqual(sorted(p.pins), sorted(fwd))
    self.assertEqual(sorted(p.gathering_forwards()), sorted(fwd))
    rest = {j: "the fused last chunk" if p.order[j][0] == p.C - 1 else "a backward" for j in p.pf if j not in fwd}
    self.assertEqual(set(rest.values()), {"the fused last chunk", "a backward"})
    self.assertEqual(p.unpinned, rest)
    for op in p.ops():
      self.assertEqual(op.pin, p.pins.get(op.j), op)
      if op.desc:
        self.assertEqual(dict(op.desc[0][1]).get("pin"), p.pins.get(op.j), op)
    self.assertIn(
        "; unpinned: " + ", ".join(f"{p.op_name(j)} ({why})" for j, why in sorted(rest.items())), p.pin_summary()
    )
    spec = pipe(scfg, prefetch_pin="hn=moe.wo", **kw)
    self.assertTrue(spec.pins and set(spec.pins) < fwd)
    self.assertEqual({j: w for j, w in spec.unpinned.items() if j not in fwd}, rest)
    off = pipe(scfg, **kw)
    each = pipe(dataclasses.replace(scfg, gather_axes="each"), prefetch_pin="auto", **kw)
    joint_ags = 0
    for q in (off, each):
      for j in rest:
        ag = _collective_axes(_program_jaxpr(q, q.ops()[j]), "all_gather")
        joint_ags += ag.count(ds_ep.FE)
        if q is each:
          self.assertNotIn(ds_ep.FE, ag, q.op_name(j))
    self.assertGreater(joint_ags, 0)  # all of them 'joint''s
    want = mode_run(off, 1, 2)
    assert_same_run(self, mode_run(p, 1, 2), want, "repeat: auto vs unpinned")
    assert_same_run(self, mode_run(each, 1, 2), want, "repeat: auto, each vs unpinned, joint")

  @pytest.mark.scheduled_only
  def test_loop_without_fusion_plan(self):
    """`loss_and_grads`, the loop without the fusion plan, on a repeat pipeline:
    it gathers every chunk itself, so it must not release any -- its
    accumulators are the step scope's bit for bit."""
    scfg = tiny_xla(layers=4, fsdp=1, ep=2)
    got, want = (pipe(scfg, n_micro=2, global_batch=4, schedule="breadth", fsdp_scope=f) for f in ("repeat", "step"))
    params, _, _ = got.init(jax.random.key(1))
    batch = got.synthetic_batch(2)
    (ga, la), (gb, lb) = got.loss_and_grads(params, batch), want.loss_and_grads(params, batch)
    assert_bitwise(self, (ga, la), (gb, lb), "loss_and_grads")

  @pytest.mark.scheduled_only
  def test_step_consumes_wc0(self):
    """`step` takes the wc0 it is given: its entries are None after it, and
    the one it returns is whole; `loss_and_grads_overlap` leaves the
    caller's alone."""
    p = pipe(tiny_xla(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4, schedule="breadth", fsdp_scope="repeat")
    params, opts, wc0 = p.init(jax.random.key(1))
    batch = p.synthetic_batch(2)
    p.loss_and_grads_overlap(params, wc0, batch)
    self.assertTrue(all(w is not None for w in wc0))
    params, opts, new_wc0, _, _ = p.step(params, opts, wc0, batch, p.lr_arrays(1e-3))
    self.assertEqual(wc0, [None] * p.S)
    self.assertTrue(all(w is not None for w in new_wc0))

  def test_progbench_hosts(self):
    """ds_ep_progbench's --prog fwd_pf and bwd_rs (`host_op`) on the repeat
    scope's plan at V=3, where backwards carry reduce-scatters and gathers
    both: fwd_pf takes forwards only -- not a backward, nor the fused last
    chunk, whose every program hosts a re-gather -- and bwd_rs each backward
    with all it carries (B + R + G); every host traces with the arguments
    host_op gives its program."""
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")  # its import raises PJRT's queue: a warning off TPU
      from maxtext.experimental.mpmd.jax import ds_ep_progbench as pb
    p = pipe(
        tiny_xla(layers=6, fsdp=1, ep=2),
        chunks_per_stage=3,
        n_micro=4,
        global_batch=8,
        schedule="breadth",
        fsdp_scope="repeat",
    )
    with self.assertRaisesRegex(SystemExit, f"chunk {p.C - 1} hosts 0 prefetch groups in its forwards"):
      pb.host_op(p, p.C - 1, 0, "fwd_pf")
    traced = collections.Counter()
    for prog, table, kind in (("fwd_pf", p.pf, "F"), ("bwd_rs", p.rs_at, "B")):
      for c in sorted({p.order[j][0] for j in table} - {p.C - 1}):
        js = [j for j in sorted(table) if p.order[j][:2] == (c, kind)]
        for g, j in enumerate(js):
          op, fn, av = pb.host_op(p, c, g, prog)
          self.assertEqual(op.j, j, (prog, c, g))
          fn.trace(*av)
          traced[prog, "+".join(k for k, _ in op.desc)] += 1
        with self.assertRaisesRegex(SystemExit, f"chunk {c} hosts {len(js)} "):
          pb.host_op(p, c, len(js), prog)
    self.assertTrue(traced["fwd_pf", "F+G"] and traced["bwd_rs", "B+R"] and traced["bwd_rs", "B+R+G"], traced)

  @pytest.mark.scheduled_only
  def test_released_weights_are_freed(self):
    """The weights a visit releases are freed then: nothing outlives
    `_release` -- the step state, the loop's locals -- so once each stage's
    update-gathered chunk ends its first visit its arrays are gone, and its
    backward visit runs on a re-gather."""
    p = pipe(tiny_xla(layers=4, fsdp=1, ep=2), n_micro=4, global_batch=8, schedule="breadth", fsdp_scope="repeat")
    params, _, wc0 = p.init(jax.random.key(1))
    refs = {
        (s, i): [weakref.ref(x) for x in jax.tree.leaves(w)]
        for s in range(p.S)
        for i, w in zip(p.gather_slots[s], wc0[s])
    }
    self.assertEqual(sorted(refs), [(s, 0) for s in range(p.S)])
    release, freed = p._release, []

    def spy(st, op):
      release(st, op)
      for b in p.release_at.get(op.j, ()):
        if (p.place[b], p.slot[b]) in refs:
          gc.collect()
          freed.append((p.op_name(op.j), b, sum(r() is not None for r in refs.pop((p.place[b], p.slot[b])))))

    p._release = spy
    p.loss_and_grads_overlap(params, wc0, p.synthetic_batch(2), consume=True)
    self.assertEqual(freed, [("0F3", 0, 0), ("1F3", 1, 0)])  # (release, chunk, its arrays still alive)

  @testing.slow
  def test_held_weights(self):
    """In the step loop a stage holds at most two chunks' weights, whole or
    arriving, under 'repeat' -- all three by its last forward under
    'step'."""
    scfg = tiny_xla(layers=6, fsdp=1, ep=2)
    for scope, most in (("repeat", 2), ("step", 3)):
      p = pipe(scfg, chunks_per_stage=3, n_micro=4, global_batch=8, schedule="breadth", fsdp_scope=scope)
      seen = collections.defaultdict(int)
      release = p._release

      def spy(st, op, p=p, release=release, seen=seen):
        held = {p.slot[b] for b in st.wparts if p.place[b] == op.s}  # arriving: some hosts still to come
        held |= {i for i, w in enumerate(st.wc[op.s]) if w is not None}  # whole
        seen[op.s] = max(seen[op.s], len(held))
        release(st, op)

      p._release = spy
      params, _, wc0 = p.init(jax.random.key(1))
      p.loss_and_grads_overlap(params, wc0, p.synthetic_batch(2))
      self.assertEqual(dict(seen), {0: most, 1: most}, scope)

  def test_flags(self):
    """--fsdp-scope and --schedule breadth reach the Pipeline through `build`;
    the summary names the scope and its releases."""
    ap = dsp.make_parser()
    self.assertIsNone(ap.parse_args(TINY_ARGS).fsdp_scope)
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(TINY_ARGS + ["--fsdp-scope", "layer"])
    args = TINY_ARGS + ["--vv", "2", "--layers", "4", "--schedule", "breadth"]
    p, q = (dsp.build(ap.parse_args(args + ["--fsdp-scope", f])) for f in ("step", "repeat"))
    self.assertEqual((p.fsdp_scope, q.fsdp_scope, q.schedule_name), ("step", "repeat", "breadth"))
    self.assertNotIn("fsdp_scope", p.summary())
    self.assertIn("fsdp_scope=repeat", q.summary())
    self.assertIn("released after", q.summary())
    self.assertNotEqual(dsp.plan_fingerprint(p), dsp.plan_fingerprint(q))

  def test_default_scope(self):
    """No --fsdp-scope: repeat under breadth -- named, or the default, on
    either placement -- with the fusion plan; step under any other schedule
    and without the fusion plan, which repeat needs.  The resolved scope is
    the one plan_fingerprint and run.json record."""
    ap = dsp.make_parser()
    base = TINY_ARGS + ["--vv", "2", "--layers", "4"]
    for more, want in (
        ([], "repeat"),
        (["--schedule", "breadth"], "repeat"),
        (["--schedule", "search"], "step"),
        (["--placement", "v"], "repeat"),
        (["--placement", "v", "--schedule", "search"], "step"),
        (["--no-overlap"], "step"),
        (["--fsdp-scope", "step"], "step"),
    ):
      with self.subTest(more=more):
        self.assertEqual(dsp.build(ap.parse_args(base + more)).fsdp_scope, want)
    self.assertEqual(
        dsp.Pipeline(
            tiny(layers=4, fsdp=1, ep=2), n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256
        ).fsdp_scope,
        "repeat",
    )
    for schedule, want in (("breadth", "repeat"), ("search", "step"), ("gpipe", "step")):
      self.assertEqual(dsp.default_fsdp_scope(schedule, True), want, schedule)
    self.assertEqual(dsp.default_fsdp_scope("breadth", False), "step")


def _collective_axes(jaxpr, prim, out=None):
  """The mesh axes of every `prim` (all_gather, ...) in `jaxpr` and its
  sub-jaxprs, in order, each as a tuple."""
  out = [] if out is None else out
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == prim:
      ax = eqn.params["axis_name"]
      out.append(tuple(ax) if isinstance(ax, (tuple, list)) else (ax,))
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          _collective_axes(getattr(sub, "jaxpr", sub), prim, out)
  return out


def _count_eqns(jaxpr, prim):
  """How many `prim` equations `jaxpr` holds, its sub-jaxprs' included."""
  n = 0
  for eqn in jaxpr.eqns:
    n += eqn.primitive.name == prim
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          n += _count_eqns(getattr(sub, "jaxpr", sub), prim)
  return n


def _eqn_out_shapes(jaxpr, prim):
  """The output shape of every `prim` equation in `jaxpr`, its sub-jaxprs'
  included, in order."""
  out = []
  for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
    if eqn.primitive.name == prim:
      out.extend(tuple(v.aval.shape) for v in eqn.outvars)
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          out.extend(_eqn_out_shapes(sub, prim))
  return out


def _barrier_firsts_used(jaxpr):
  """For every optimization_barrier in `jaxpr` and its sub-jaxprs, in order:
  whether its first output -- a pin's forward value (ds_ep._tie) -- is an
  input of a later equation of its jaxpr or one of that jaxpr's outputs."""
  jaxpr, out = getattr(jaxpr, "jaxpr", jaxpr), []
  used = {id(v) for e in jaxpr.eqns for v in e.invars} | {id(v) for v in jaxpr.outvars}
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "optimization_barrier":
      out.append(id(eqn.outvars[0]) in used)
    for v in eqn.params.values():
      for sub in v if isinstance(v, (tuple, list)) else (v,):
        if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
          out += _barrier_firsts_used(getattr(sub, "jaxpr", sub))
  return out


# A first, a middle (two layers) and a last chunk of tiny's model: every weight
# layout the pipeline gathers (the embedding, the head and the layer leaves).
_ALL_LAYOUTS = (
    ds_ep.ChunkSpec(index=0, layers=(0,), first=True, last=False, stage=0),
    ds_ep.ChunkSpec(index=1, layers=(1, 2), first=False, last=False, stage=0),
    ds_ep.ChunkSpec(index=2, layers=(3,), first=False, last=True, stage=0),
)


@testing.needs_devices(8, "fsdp 2 x expert 2 stages: a gather one axis at a time against both at once takes both")
class GatherAxes(unittest.TestCase):
  """StageConfig.gather_axes under lax FSDP collectives: a weight stored over
  both mesh axes gathers in one all_gather over both ('joint', the default)
  or one axis at a time, the minor first ('each': on a stage of a
  multi-slice v5p program the joint one compiles to a zero buffer and a
  full-size all-reduce, each to native SparseCore all-gathers)."""

  @pytest.mark.scheduled_only
  def test_each_is_joint_bit_for_bit(self):
    """Every leaf layout the pipeline gathers, from random fp32 shards on a
    2 x 4, a 4 x 2 and a 1 x 8 ('fsdp' of one device) stage mesh: 'each'
    gives the joint gather's bf16 leaf bit for bit -- the whole leaf, cast,
    as the compute layout holds it -- with one all_gather a mesh axis of more
    than one device, 'expert' (the minor) first, where 'joint' gathers over
    both at once; a leaf stored over one axis gathers alike either way."""
    scfg = tiny_xla(layers=4, fsdp_collectives="xla")
    rng = np.random.default_rng(0)
    n = 0
    for grid in ((2, 4), (4, 2), (1, 8)):
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:8]).reshape(grid), ds_ep.FE)
      for ch in _ALL_LAYOUTS:
        lv = lambda t, leaf: jax.tree.leaves(t, is_leaf=leaf)
        for name, shape, ss, cs in zip(
            ds_ep.leaf_names(ch),
            lv(ds_ep.chunk_shapes(scfg, ch), ds_ep._is_shape),
            lv(ds_ep.chunk_storage_specs(ch), ds_ep._is_spec),
            lv(ds_ep.chunk_compute_specs(ch), ds_ep._is_spec),
        ):
          x = rng.standard_normal(shape).astype(np.float32)
          xd = jax.device_put(x, NamedSharding(mesh, ss))
          got = {}
          for per_axis in (False, True):
            f = jax.jit(
                jax.shard_map(
                    lambda a, ss=ss, cs=cs, per_axis=per_axis: ds_ep._gather_leaf(
                        a, ss, cs, jnp.bfloat16, "xla", per_axis
                    ),
                    mesh=mesh,
                    in_specs=ss,
                    out_specs=cs,
                    check_vma=False,
                )
            )
            got[per_axis] = (np.asarray(f(xd)), _collective_axes(jax.make_jaxpr(f)(xd).jaxpr, "all_gather"))
          what = f"{grid} chunk {ch.index} {name}"
          np.testing.assert_array_equal(got[True][0], got[False][0], err_msg=what)
          np.testing.assert_array_equal(got[False][0], np.asarray(jnp.asarray(x, jnp.bfloat16)), err_msg=what)
          joint, each = got[False][1], got[True][1]
          if ds_ep.FE in [tuple(ds_ep._axes_of(s)) for s in ss]:
            self.assertEqual(joint, [ds_ep.FE], what)
            self.assertEqual(each, [(a,) for a in reversed(ds_ep.FE) if dict(zip(ds_ep.FE, grid))[a] > 1], what)
            n += 1
          else:
            self.assertEqual(each, joint, what)
    self.assertEqual(n, 3 * (7 + 6 + 7))  # chunk 0: 6 layer leaves + the embedding; chunk 2: + the head

  @pytest.mark.scheduled_only
  def test_pipeline_bitwise(self):
    """A pipeline gathering one axis at a time trains as the joint one does,
    bit for bit -- its gradients, sums of squares, losses and the params of
    two steps -- with the fusion plan (the prefetch parts, the self-prefetch,
    gather_first and update_gather) and without it (the stages' gather
    programs); and its programs gather over one axis at a time: the stage's
    gather_first / gather, and each forward that gathers (ChunkPrograms' own
    flag), whose joint gathers under 'joint' are one a leaf stored over both
    axes -- six of chunk 2's sixteen in its prefetch part (wq_a, wkv_a, out,
    the shared experts), none in the self-prefetch (routed experts, stored
    over one axis a dim) -- and none under 'each'."""
    for overlap in (True, False):
      runs = {}
      for axes in ("joint", "each"):
        p = pipe(
            tiny_xla(fsdp_collectives="xla", gather_axes=axes),
            n_micro=2,
            global_batch=8,
            fsdp_scope="step",
            overlap=overlap,
        )
        self.assertEqual({c.gather_axes for c in p.stage_scfg}, {axes})
        st = p.stages[0]
        params = [p._avals(0, "storage", c) for c in p.stage_chunks[0]]
        prog = st.gather_first if overlap else st.gather
        ag = _collective_axes(jax.make_jaxpr(prog)(params).jaxpr, "all_gather")
        self.assertEqual(ds_ep.FE in ag, axes == "joint", (overlap, axes, ag))
        both = {}
        for op in p.ops():
          if op.pf or op.self_pf:
            b, idx = op.pf or (op.c, p.routed_leaves)
            ss = jax.tree.leaves(ds_ep.chunk_storage_specs(p.chunks[b]), is_leaf=ds_ep._is_spec)
            both[p.op_name(op.j)] = n = sum(ds_ep.FE in [tuple(ds_ep._axes_of(s)) for s in ss[i]] for i in idx)
            ag = _collective_axes(_program_jaxpr(p, op), "all_gather")
            self.assertEqual(ag.count(ds_ep.FE), n if axes == "joint" else 0, (axes, p.op_name(op.j), ag))
        self.assertEqual(both, {"0F0": 0, "0F1": 6} if overlap else {}, axes)
        runs[axes] = mode_run(p, 1, 3)
      assert_same_run(self, runs["each"], runs["joint"], f"overlap {overlap}: each vs joint")

  def test_flag(self):
    """--gather-axes reaches the StageConfig through `build`, 'joint' by
    default; the summary names 'each'; anything else is refused."""
    ap = dsp.make_parser()
    self.assertEqual(ap.parse_args(TINY_ARGS).gather_axes, "joint")
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(TINY_ARGS + ["--gather-axes", "both"])
    with self.assertRaisesRegex(ValueError, "gather_axes"):
      tiny(gather_axes="both")
    p, q = (dsp.build(ap.parse_args(TINY_ARGS + ["--gather-axes", g])) for g in ("joint", "each"))
    self.assertEqual((p.scfg.gather_axes, q.scfg.gather_axes), ("joint", "each"))
    self.assertNotIn("gather_axes", p.summary())
    self.assertIn("gather_axes=each", q.summary())


def _program_jaxpr(p, op):
  """The jaxpr of `op`'s program in pipeline `p`, traced as the top-level jit
  it is (`.trace`): a pinned program carries compiler options on TPU, and jax
  refuses to stage such a jit inside jax.make_jaxpr or another jit."""
  return p.progs[op.c].program(op.desc).trace(*p._desc_avals(op.c, op.desc)).jaxpr


def _dcn_scfg(experts, cp, gather_axes="joint"):
  """The stage of the DeepSeek-V3 runs across two v5p-128 slices: DeepSeek-V3
  layers with `experts` routed experts on fsdp 4 x expert 16, the head's vocab
  129280, context parallelism `cp`."""
  a = dsp.make_parser().parse_args(
      f"--layers 4 --experts {experts} --topk 8 --vocab 129280 --seq 4096 --attn splash "
      f"--sa-block 2048 --sa-block-kv-compute 2048".split()
  )
  return ds_ep.StageConfig(
      model=dsp.model_config(a), fsdp=4, ep=16, cp=cp, fsdp_collectives="xla", gather_axes=gather_axes
  )


def _v6e_scfg(experts, cp, fsdp, ep, gather_axes="joint"):
  """The FSDP-overlap A/B's stage on 2 x v6e-128 (ds_ep.PIN_MODEL_V6E): the
  DeepSeek-V3 layers of `_dcn_scfg` with `experts` routed experts on fsdp `fsdp`
  x expert `ep` (R2v: E384, cp 2, 8 x 16; R1v: E128, 16 x 8)."""
  a = dsp.make_parser().parse_args(
      f"--layers 4 --experts {experts} --topk 8 --vocab 129280 --seq 4096 --attn splash "
      f"--sa-block 2048 --sa-block-kv-compute 2048".split()
  )
  return ds_ep.StageConfig(
      model=dsp.model_config(a), fsdp=fsdp, ep=ep, cp=cp, fsdp_collectives="xla", gather_axes=gather_axes
  )


@testing.needs_devices(8, "fsdp 2 x expert 2 stages: the pins tie each axis's gathers")
class PrefetchPins(unittest.TestCase):
  """--prefetch-pin: the gathers folded into a forward tied to points of its
  layers by optimization barriers -- identities, so a pinned step's values
  are the unpinned one's bit for bit -- by a SPEC of leaf names or the
  'auto' planner."""

  def test_names_and_spec(self):
    """leaf_names, the SPEC's form, which leaves a name selects (the longest
    wins; a chunk prefix narrows it to one chunk; a leaf's path and a
    chunk-qualified name of one length must agree)."""
    last = _ALL_LAYOUTS[2]
    self.assertEqual(
        ds_ep.leaf_names(last),
        [
            "decoder_norm",
            "moe.gate",
            "moe.shared_experts.wi_0",
            "moe.shared_experts.wi_1",
            "moe.shared_experts.wo",
            "moe.wi_0",
            "moe.wi_1",
            "moe.wo",
            "post_self_attention_layer_norm",
            "pre_self_attention_layer_norm",
            "self_attention.kv_norm",
            "self_attention.out",
            "self_attention.q_norm",
            "self_attention.wkv_a",
            "self_attention.wkv_b",
            "self_attention.wq_a",
            "self_attention.wq_b",
            "logits_dense",
        ],
    )
    self.assertEqual(ds_ep.leaf_names(_ALL_LAYOUTS[0])[-1], "token_embedder.embedding")
    self.assertEqual(
        (ds_ep.parse_prefetch_pin("off"), ds_ep.parse_prefetch_pin(None), ds_ep.parse_prefetch_pin("auto")),
        ("off", "off", "auto"),
    )
    entries = ds_ep.parse_prefetch_pin(" hn=moe,c3.moe.wi_0 ; xs@1=self_attention.out;y=c3 ;")
    self.assertEqual(
        entries, (("hn", 0, "moe"), ("hn", 0, "c3.moe.wi_0"), ("xs", 1, "self_attention.out"), ("y", 0, "c3"))
    )
    self.assertEqual(ds_ep.pin_of(entries, "moe.wo", 0), ("hn", 0))
    self.assertEqual(ds_ep.pin_of(entries, "moe.wo", 3), ("hn", 0))  # 'moe' (3) is longer than 'c3' (2)
    self.assertEqual(ds_ep.pin_of(entries, "logits_dense", 3), ("y", 0))
    self.assertEqual(ds_ep.pin_of(entries, "self_attention.out", 3), ("xs", 1))
    self.assertIsNone(ds_ep.pin_of(entries, "self_attention.wq_a", 1))
    # a leaf's path and a chunk-qualified name of one length: refused where they differ (whichever comes first),
    # taken where they agree, and only for the chunk the qualified one names
    for spec, generic in (("hn=c3.moe;y=moe.wo", "y"), ("y=c3.moe;hn=moe.wo", "hn")):
      with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, r"\['c3.moe', 'moe.wo'\], names of one length"):
        ds_ep.pin_of(ds_ep.parse_prefetch_pin(spec), "moe.wo", 3)
      self.assertEqual(ds_ep.pin_of(ds_ep.parse_prefetch_pin(spec), "moe.wo", 2), (generic, 0))
    self.assertEqual(ds_ep.pin_of(ds_ep.parse_prefetch_pin("xs=c3.moe,moe.wo"), "moe.wo", 3), ("xs", 0))
    self.assertEqual(ds_ep.pin_of(ds_ep.parse_prefetch_pin("hn=c3.moe;y=c3.moe.wo"), "moe.wo", 3), ("y", 0))
    self.assertTrue(ds_ep.pin_selects("moe.shared_experts", "moe.shared_experts.wo", 2))
    self.assertFalse(ds_ep.pin_selects("moe.w", "moe.wo", 2))
    for bad in ("hn", "hn=", "zz=moe.wo", "hn@x=moe.wo", "hn=moe.wo,", "hn=moe.wo;y=moe.wo", ";"):
      with self.subTest(spec=bad), self.assertRaises(ValueError):
        ds_ep.parse_prefetch_pin(bad)
    self.assertEqual(dsp.prefetch_pin_arg(" auto "), "auto")
    with self.assertRaises(argparse.ArgumentTypeError):
      dsp.prefetch_pin_arg("hn=moe.wo;y=moe.wo")
    self.assertTrue(ds_ep.has_gather(last, 1) and ds_ep.has_gather(last, 17))
    self.assertFalse(ds_ep.has_gather(last, 0) or ds_ep.has_gather(last, 12))  # decoder_norm, q_norm

  def test_plans(self):
    """The pipeline's pins: a SPEC's points for the gathered leaves it
    names -- in the self-prefetch (0F0: chunk 0's routed experts) and in the
    prefetch host (0F1: chunk 2's leaves), never a norm scale -- in the
    forward descriptors and the summary; 'off' none, and no 'pin' key in any
    descriptor; a name no leaf has, a layer the host lacks, or two names of
    one length pinning a leaf at different points, is refused; a forward in
    stash mode is left unpinned, and every part but a forward refuses a pin."""
    scfg = tiny_xla(fsdp_collectives="xla")
    kw = dict(n_micro=2, global_batch=8, fsdp_scope="step")
    p = pipe(scfg, prefetch_pin="hn=c0.moe.wi_0,c2.moe.wi_0,self_attention;xs=moe.wo;y=c2.moe.shared_experts", **kw)
    names = p.leaf_names
    got = {p.op_name(j): {(pt, names[c][i]) for pt, _, c, i in plan} for j, plan in p.pins.items()}
    self.assertEqual(got["0F0"], {("hn", "moe.wi_0"), ("xs", "moe.wo")})
    att = {f"self_attention.{n}" for n in ("out", "wkv_a", "wkv_b", "wq_a", "wq_b")}
    self.assertEqual(
        got["0F1"],
        {("hn", "moe.wi_0"), ("xs", "moe.wo")}
        | {("hn", n) for n in att}
        | {("y", f"moe.shared_experts.{n}") for n in ("wi_0", "wi_1", "wo")},
    )
    self.assertEqual(sorted(p.pins), sorted(p.gathering_forwards()))
    for op in p.ops():
      if op.desc:  # (the fused last chunk's no-op backward has none)
        self.assertEqual(dict(op.desc[0][1]).get("pin"), p.pins.get(op.j), op)
      self.assertEqual(op.pin, p.pins.get(op.j), op)
    self.assertIn("prefetch pins (spec): 0F0 hn=c0.moe.wi_0 xs=c0.moe.wo; 0F1 hn=c2.moe.wi_0,", p.summary())
    off = pipe(scfg, **kw)
    self.assertEqual((off.pins, off.pin_summary()), ({}, ""))
    self.assertFalse(any("pin" in dict(kw_) for op in off.ops() if op.desc for _, kw_ in op.desc))
    with self.assertRaisesRegex(ValueError, r"\['moe.wq_a'\] select"):
      pipe(scfg, prefetch_pin="hn=moe.wq_a", **kw)
    with self.assertRaisesRegex(ValueError, "pinned in layer 1"):
      pipe(scfg, prefetch_pin="hn@1=moe.wo", **kw)
    with self.assertRaisesRegex(ValueError, "names of one length"):  # chunk 1's leaves, which no forward gathers
      pipe(scfg, prefetch_pin="hn=c1.moe;y=moe.wo", **kw)
    st = pipe(
        tiny_xla(),
        placement="v",
        schedule="stash",
        stash="lite",
        stash_budget_gib=1.0,
        n_micro=1,
        global_batch=4,
        overlap=True,
        prefetch_pin="hn=moe.wi_0",
    )
    self.assertTrue(st.modes and st.self_pf_at is not None)
    self.assertNotIn(st.self_pf_at, st.pins)  # 0F0 stashes: unpinned
    with self.assertRaisesRegex(ValueError, "remat forward"):
      st.progs[0].part("F", stash="lite", pin=(("hn", 0, 0, 4),))
    # every other kind of part refuses a pin too (the gather, reduce-scatter and fused last chunk ones dropped it)
    for c, kind, other in ((3, "A", None), (0, "B", None), (0, "G", off.chunks[2]), (0, "R", off.chunks[2])):
      with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"forward \\('F'\\), not of a '{kind}' part"):
        off.progs[c].part(kind, other=other, pin=(("hn", 0, 2, 5),))

  def test_summary_names_the_unpinned(self):
    """The summary's pins line names every gathering forward left unpinned and
    why, and says 'none' when no forward took a pin -- a run asked for pins
    that got none says so: every gathering forward in stash mode, 'auto' on
    collective-permute stages, a SPEC naming no gathered leaf.  Under 'off'
    there is no line; without the fusion plan the overlap line says there is
    nothing to pin."""
    scfg = tiny_xla(fsdp_collectives="xla")
    kw = dict(n_micro=2, global_batch=8, fsdp_scope="step")
    line = "           prefetch pins "
    p = pipe(scfg, prefetch_pin="hn=c0.moe.wi_0", **kw)  # 0F1 gathers chunk 2's leaves
    self.assertEqual(
        p.pin_summary(), line + "(spec): 0F0 hn=c0.moe.wi_0; unpinned: 0F1 (none of its gathered leaves " "named)"
    )
    st = pipe(
        tiny(fsdp_collectives="xla"),
        placement="v",
        schedule="stash",
        stash="lite",
        stash_budget_gib=100.0,
        prefetch_pin="auto",
        **kw,
    )
    hosts = [st.op_name(j) for j in st.gathering_forwards()]
    self.assertTrue(hosts and all(h.endswith("s") for h in hosts), hosts)  # every one in stash mode
    self.assertEqual(
        (st.pins, st.pin_summary()),
        ({}, line + "(auto): none; unpinned: " + ", ".join(f"{h} (stash mode)" for h in hosts)),
    )
    self.assertIn(st.pin_summary(), st.summary())
    q = pipe(tiny_xla(), prefetch_pin="auto", **kw)  # collective-permute FSDP collectives
    self.assertEqual(
        q.pin_summary(),
        line + "(auto): none; unpinned: 0F0 (collective-permute FSDP collectives), "
        "0F1 (collective-permute FSDP collectives)",
    )
    self.assertEqual(pipe(scfg, **kw).pin_summary(), "")
    flat = pipe(scfg, prefetch_pin="auto", overlap=False, **kw)
    self.assertEqual((flat.pins, flat.pin_summary()), ({}, ""))
    self.assertIn(
        "overlap: off (stage-level gather / reduce programs); --prefetch-pin: no forward gathers, nothing " "to pin",
        flat.summary(),
    )
    self.assertNotIn("--prefetch-pin", pipe(scfg, overlap=False, **kw).summary())

  def test_auto_skips_synchronous_stages(self):
    """'auto' plans no pin on a stage whose lax FSDP collectives libtpu runs
    synchronously (`sync_stages`), where a gather blocks the TensorCore
    wherever it sits; the other stage plans as before, and a SPEC still pins
    there."""
    scfg = tiny_xla(fsdp_collectives="xla")
    kw = dict(n_micro=2, global_batch=8, fsdp_scope="step")
    p = pipe(scfg, prefetch_pin="auto", **kw)
    pinned = lambda q: sorted(q.op_name(j) for j in q.pins)
    self.assertEqual(pinned(p), ["0F0", "0F1"])  # chunk 0's forwards: stage 0
    for sync in (frozenset({0}), frozenset({1})):
      with mock.patch.object(dsp, "sync_collective_stages", lambda meshes, sync=sync: sync):
        auto, spec = (pipe(scfg, prefetch_pin=pin, **kw) for pin in ("auto", "hn=moe.wi_0"))
      self.assertEqual((auto.sync_stages, spec.sync_stages), (sync, sync))
      self.assertEqual(auto.pins, {} if 0 in sync else p.pins, sync)
      self.assertEqual(pinned(spec), ["0F0", "0F1"], sync)
      self.assertEqual(
          "unpinned: 0F0 (synchronous FSDP collectives), 0F1 (synchronous FSDP collectives)" in auto.pin_summary(),
          0 in sync,
          auto.pin_summary(),
      )

  def test_auto_warns_off_v5p(self):
    """'auto' plans with costs measured on v5p and v6e (ds_ep.PIN_MODELS): a
    plan on another TPU warns; on v5p (whose devices report 'TPU v5'), on v6e
    ('TPU v6 lite'), off TPU, with a SPEC, or with nothing planned, nothing
    does."""
    tpus = lambda *kinds: [FakeTpu(id=i, coords=(i, 0, 0), process_index=0, device_kind=k) for i, k in enumerate(kinds)]
    self.assertEqual(dsp.pin_model_mismatch(tpus("TPU v5", "TPU v5p", "TPU v6 lite")), [])
    self.assertEqual(dsp.pin_model_mismatch(tpus("TPU v5", "TPU7x", "TPU v6 lite", "TPU7x")), ["TPU7x"])
    self.assertEqual(dsp.pin_model_mismatch(jax.devices()[:2]), [])
    kw = dict(n_micro=2, global_batch=8)
    lax_fsdp = tiny_xla(fsdp_collectives="xla")
    with mock.patch.object(dsp, "pin_model_mismatch", lambda devices: ["TPU7x"]):
      with testing.assert_warns_regex(self, RuntimeWarning, r"calibrated on v5p and v6e, not TPU7x"):
        self.assertTrue(pipe(lax_fsdp, prefetch_pin="auto", **kw).pins)
      for scfg, pin in ((lax_fsdp, "hn=moe.wi_0"), (lax_fsdp, "off"), (tiny_xla(), "auto")):
        with self.subTest(pin=pin, fsdp=scfg.fsdp_collectives), warnings.catch_warnings(record=True) as seen:
          warnings.simplefilter("always")
          pipe(scfg, prefetch_pin=pin, **kw)
        self.assertEqual([w for w in seen if "calibrated on v5p" in str(w.message)], [])
    with warnings.catch_warnings(record=True) as seen:
      warnings.simplefilter("always")
      self.assertTrue(pipe(lax_fsdp, prefetch_pin="auto", **kw).pins)  # CPU devices
    self.assertEqual([w for w in seen if "calibrated on v5p" in str(w.message)], [])

  def test_pin_model_per_kind(self):
    """'auto' plans a stage with its devices' kind's PinModel (ds_ep.PIN_MODELS:
    v5p's for 'TPU v5', v6e's for 'TPU v6 lite') and with v5p's off TPU, on
    a kind none was measured on, or on a stage of mixed kinds; the pipeline's
    plan is plan_pins' under that model."""

    def stage(*kinds):
      devs = [types.SimpleNamespace(device_kind=k) for k in kinds]
      return types.SimpleNamespace(devices=np.array(devs, dtype=object))

    p = types.SimpleNamespace(
        meshes=[
            stage("TPU v6 lite", "TPU v6 lite"),
            stage("TPU v5", "TPU v5"),
            stage("TPU7x"),
            stage("TPU v6 lite", "TPU v5"),
            stage("cpu"),
        ]
    )
    got = [dsp.Pipeline.pin_model(p, k) for k in range(5)]
    self.assertEqual(got, [ds_ep.PIN_MODEL_V6E, ds_ep.PIN_MODEL, ds_ep.PIN_MODEL, ds_ep.PIN_MODEL, ds_ep.PIN_MODEL])
    self.assertEqual((ds_ep.PIN_MODEL.ring_max, ds_ep.PIN_MODEL_V6E.ring_max), (4, 8))
    # the pipeline plans with the host stage's model: every model reaches plan_pins
    lax_fsdp = tiny_xla(fsdp_collectives="xla")
    seen = []
    real = ds_ep.plan_pins

    def spy(*a, **k):
      seen.append(k["model"])
      return real(*a, **k)

    for model in (ds_ep.PIN_MODEL, ds_ep.PIN_MODEL_V6E):
      seen.clear()
      with (
          mock.patch.object(dsp.Pipeline, "pin_model", lambda self, stage, m=model: m),
          mock.patch.object(ds_ep, "plan_pins", spy),
      ):
        q = pipe(lax_fsdp, prefetch_pin="auto", n_micro=2, global_batch=8)
      self.assertTrue(q.pins)
      self.assertEqual(set(seen), {model})

  def test_v6e_windows_and_gathers(self):
    """PIN_MODEL_V6E on the FSDP-overlap A/B's two points on 2 x v6e-128
    (R2v = E384, cp 2, fsdp 8 x expert 16; R1v = E128, one 4096-token row a
    device, fsdp 16 x expert 8) against what their traces measured: attention
    6.30 / 9.66 ms up to 'hn', the TopK 0.374 / 0.072 (XLA's sort: one
    estimate, 0.37, for both), the gmms 4.91-4.99 / 7.02-7.03; a routed
    expert's 8-way gather 6.97-7.05 ms (R2v) and 16-way 2.40-2.58 (R1v); the
    head's native 128-way gather 10.11-10.12 (both), per axis (each) 1.21 +
    18.27 (R2v: 16-way, then 8-way) and 1.14 + 9.51 (R1v: 8-way, then
    16-way).  v5p's costs priced the R2v ones at 1.99 and 5.93."""
    last = _ALL_LAYOUTS[2]
    for name, experts, cp, fsdp, ep, tokens, win, routed in (
        ("R2v", 384, 2, 8, 16, 2048, (6.30, 0.37, 4.93), 7.01),
        ("R1v", 128, 1, 16, 8, 4096, (9.67, 0.37, 7.05), 2.47),
    ):
      s = _v6e_scfg(experts, cp, fsdp, ep)
      got = ds_ep.pin_windows(s, _ALL_LAYOUTS[0], tokens, model=ds_ep.PIN_MODEL_V6E)
      np.testing.assert_allclose([ms for _, _, ms in got], win, atol=0.01, err_msg=name)
      for axes, head in (("joint", 10.51), ("each", 19.65 if name == "R2v" else 10.91)):
        s = _v6e_scfg(experts, cp, fsdp, ep, axes)
        leaf = lambda i: (
            jax.tree.leaves(t, is_leaf=f)[i]
            for t, f in (
                (ds_ep.chunk_shapes(s, last), ds_ep._is_shape),
                (ds_ep.chunk_storage_specs(last), ds_ep._is_spec),
                (ds_ep.chunk_compute_specs(last), ds_ep._is_spec),
            )
        )
        ms = lambda i, model=ds_ep.PIN_MODEL_V6E: ds_ep.pin_gather_ms(s, *leaf(i), joint_allreduce=False, model=model)
        self.assertAlmostEqual(ms(5), routed, places=2, msg=(name, axes))  # moe.wi_0
        self.assertAlmostEqual(ms(17), head, places=2, msg=(name, axes))  # logits_dense
        self.assertEqual(ms(0), 0.0)  # decoder_norm
        if name == "R2v":
          self.assertAlmostEqual(ms(5, ds_ep.PIN_MODEL), 1.99, places=2)
          self.assertAlmostEqual(ms(17, ds_ep.PIN_MODEL), 5.93, places=2)

  def test_v6e_auto_plans(self):
    """'auto' with PIN_MODEL_V6E on the A/B's points: the self-prefetch pins
    chunk 0's routed wi_0 at 'hn', as v5p's plans; R1v's one prefetch host
    pins the head in attention and the three routed experts at 'y', as v5p's;
    R2v's 8-way routed gathers (7.0 ms) fill a window each and outgrow the
    gmms' 4.9 ms, and its head (10.5 ms native, 19.6 per axis) outgrows every
    window: with overflow 'last' one routed expert of the first group and the
    head of the second go to 'y' -- where v5p's costs (2.0 and 5.9 ms) put
    them, and the run's traces show them overrunning the gmms by 1.8 and 10.4
    ms, against 5.9 and 17.4 ms exposed when XLA places them -- and with
    'xla' they would be left out.  Every leaf with a gather is pinned."""
    chunks = [
        ds_ep.ChunkSpec(index=c, layers=(c,), first=c == 0, last=c == 3, stage=0 if c in (0, 3) else 1) for c in range(4)
    ]
    names = ds_ep.leaf_names(chunks[3])
    routed = (4, 5, 6)  # chunk 0's moe.wi_0, wi_1, wo
    xla = dataclasses.replace(ds_ep.PIN_MODEL_V6E, overflow="xla")
    for axes in ("joint", "each"):
      for run, experts, cp, fsdp, ep, tokens in (("R2v", 384, 2, 8, 16, 2048), ("R1v", 128, 1, 16, 8, 4096)):
        s = _v6e_scfg(experts, cp, fsdp, ep, axes)
        plan = lambda host_leaves, own=(), model=ds_ep.PIN_MODEL_V6E: ds_ep.plan_pins(
            s, chunks[0], host_leaves, tokens, joint_allreduce=False, own=own, model=model
        )
        self.assertEqual(plan([(chunks[0], i) for i in routed], own=tuple((0, i) for i in routed)), (("hn", 0, 0, 4),))
        groups = ds_ep.leaf_groups(ds_ep.chunk_shapes(s, chunks[3]), 1 if run == "R1v" else 2)
        at, at5, atx = {}, {}, {}
        for g in groups:
          got = plan([(chunks[3], i) for i in g])
          self.assertEqual(sorted(i for *_, i in got), [i for i in g if ds_ep.has_gather(chunks[3], i)], (axes, g))
          at.update({names[i]: pt for pt, _, _, i in got})
          at5.update({names[i]: pt for pt, _, _, i in plan([(chunks[3], i) for i in g], model=ds_ep.PIN_MODEL)})
          atx.update({names[i]: pt for pt, _, _, i in plan([(chunks[3], i) for i in g], model=xla)})
        what = (axes, run, at)
        if run == "R1v":
          for a in (at, at5, atx):
            self.assertEqual((a["logits_dense"], a["moe.wi_0"], a["moe.wi_1"], a["moe.wo"]), ("hn", "y", "y", "y"), what)
        else:
          self.assertEqual(groups[0], (5, 7))  # routed wi_0, wo
          for a in (at, at5):
            self.assertEqual(sorted((a["moe.wi_0"], a["moe.wo"])), ["hn", "y"], what)
            self.assertEqual(a["logits_dense"], "y", what)
          self.assertEqual(at["moe.wi_1"], "hn", what)
          self.assertEqual(sorted(atx.get(n) or "-" for n in ("moe.wi_0", "moe.wo")), ["-", "hn"], what)
          self.assertNotIn("logits_dense", atx, what)
    with self.assertRaisesRegex(ValueError, "overflow must be 'xla' or 'last'"):
      dataclasses.replace(xla, overflow="end")

  @pytest.mark.scheduled_only
  def test_pinned_runs_bitwise(self):
    """Pinned steps -- 'auto' and a SPEC at all three points, with joint and
    per-axis gathers, the auto one precompiled in threads -- give the
    unpinned step's gradients, sums of squares, losses and params bit for
    bit, in bf16 with MaxText's kernels; each pinned program holds one
    optimization barrier a pinned point,
    whose first output the forward goes on from (ds_ep._tie: else the barrier
    orders nothing, with the same values), an unpinned one none."""
    scfg = tiny(fsdp_collectives="xla")
    kw = dict(n_micro=2, global_batch=8, fsdp_scope="step")
    off = pipe(scfg, **kw)
    for op in off.ops():
      if op.pf or op.self_pf:
        self.assertEqual(_count_eqns(_program_jaxpr(off, op), "optimization_barrier"), 0)
    want = mode_run(off, 1, 3)
    spec = "hn=c0.moe.wi_0,c2.moe.gate,c2.self_attention;xs=c0.moe.wi_1,c2.moe.wo;y=c2.moe.wi_0,c2.moe.shared_experts"
    for pin, axes in (("auto", "joint"), (spec, "joint"), (spec, "each")):
      p = pipe(dataclasses.replace(scfg, gather_axes=axes), prefetch_pin=pin, **kw)
      self.assertEqual(sorted(p.pins), sorted(p.gathering_forwards()), pin)
      for j, plan in p.pins.items():
        jx = _program_jaxpr(p, p.ops()[j])
        n = _count_eqns(jx, "optimization_barrier")
        self.assertEqual(n, len({(pt, layer) for pt, layer, _, _ in plan}), (pin, p.op_name(j)))
        self.assertEqual(_barrier_firsts_used(jx), [True] * n, (pin, p.op_name(j)))
      if pin == "auto":
        p.precompile(threads=8)
      assert_same_run(self, mode_run(p, 1, 3), want, f"{pin} {axes} vs unpinned")

  @pytest.mark.scheduled_only
  def test_pinned_explicit_cp2_bitwise(self):
    """Under the arguments of an A/B of the two-v5p-128 runs -- --shard-mode explicit, cp 2, the
    V placement in search's order (its vshape arm's), M=4 microbatches of one
    row a device, lax FSDP collectives, the gather embedding and top-k
    routing -- 'auto' pins all three gathering forwards (the self-prefetch
    and the two prefetch groups of the last chunk's leaves) and the step
    trains as the unpinned one, bit for bit."""
    sc = explicit(
        tiny_xla(layers=4, random_routing=False, fsdp_collectives="xla", iota_embed=False, remat_fused_last=True)
    )
    mk = lambda pin: dsp.Pipeline(
        sc,
        n_stages=2,
        chunks_per_stage=2,
        n_micro=4,
        global_batch=8,
        seq=256,
        cp=2,
        placement="v",
        schedule="search",
        init_by_chunk=True,
        prefetch_pin=pin,
    )
    p = mk("auto")
    self.assertEqual(sorted(p.op_name(j) for j in p.pins), ["0F0", "0F1", "0F2"])
    self.assertEqual({c for plan in p.pins.values() for _, _, c, _ in plan}, {0, 3})
    assert_same_run(self, mode_run(p, 1, 3, steps=1), mode_run(mk("off"), 1, 3, steps=1), "explicit cp2 V: auto")

  def test_pin_trace_state(self):
    """The tracing state behind the pins: a point the forward never reached
    and a leaf no part gathers are errors; pinned forwards traced at once in
    eight threads (precompile's way) each get their own barriers."""
    with self.assertRaisesRegex(ValueError, "never reached"):
      with ds_ep._pinned({("hn", 0): [[(3, 1), jnp.ones(2)]]}):
        pass
    with self.assertRaisesRegex(ValueError, "no part of the program gathers"):
      with ds_ep._pinned({}, plan=(("hn", 0, 3, 1),)):
        ds_ep._tie("hn", jnp.ones(2))
    self.assertIsNone(ds_ep._PINS.regs)
    p = pipe(
        tiny_xla(fsdp_collectives="xla"),
        n_micro=2,
        global_batch=8,
        fsdp_scope="step",
        prefetch_pin="hn=c0.moe.wi_0,c2.moe.wi_0;xs=c2.moe.wo;y=c2.moe.wi_1",
    )
    jobs = [(op, len({(pt, l) for pt, l, _, _ in op.pin})) for op in p.ops() if op.pin]
    self.assertEqual(sorted(n for _, n in jobs), [1, 3])

    def trace(job):
      op, _ = job
      return _count_eqns(_program_jaxpr(p, op), "optimization_barrier")

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(8) as ex:
      got = list(ex.map(trace, jobs * 8))
    self.assertEqual(got, [n for _, n in jobs] * 8)

  def test_compiler_options(self):
    """A program with pins compiles with libtpu's barrier removal off on TPU
    devices (a compile-only topology's too), with no option elsewhere; a
    program without pins with none anywhere.  With an option a program is a
    top-level jit only, which `.trace` takes and jax.make_jaxpr refuses."""
    tpu = types.SimpleNamespace(devices=np.array([FakeTpu(id=0, coords=(0, 0, 0), process_index=0)]))
    self.assertEqual(ds_ep.pin_compiler_options(tpu), {"xla_tpu_aggressive_opt_barrier_removal": "DISABLED"})
    self.assertIsNone(ds_ep.pin_compiler_options(dsp.stage_meshes(1, 1, 2)[0]))
    seen = []
    real = jax.jit

    def spy(*a, **k):
      seen.append(k.get("compiler_options"))
      return real(*a, **k)

    with (
        mock.patch.object(ds_ep, "pin_compiler_options", lambda mesh: {"opt": "x"}),
        mock.patch.object(ds_ep.jax, "jit", spy),
    ):
      p = pipe(
          tiny_xla(fsdp_collectives="xla"),
          n_micro=2,
          global_batch=8,
          fsdp_scope="step",
          prefetch_pin="hn=moe.wi_0",
      )
      for op in p.ops():
        if op.desc:
          p.progs[op.c].program(op.desc)
    self.assertEqual(seen.count({"opt": "x"}), len(p.pins))
    self.assertEqual(len(p.pins), 2)
    # A program with compiler options (on TPU, every pinned one) is a top-level jit only: jax.make_jaxpr refuses
    # it, `.trace` -- as these tests trace programs (_program_jaxpr) -- takes it.  A CPU option stands in here.
    with mock.patch.object(ds_ep, "pin_compiler_options", lambda mesh: {"xla_cpu_enable_fast_math": False}):
      q = pipe(
          tiny_xla(fsdp_collectives="xla"),
          n_micro=2,
          global_batch=8,
          fsdp_scope="step",
          prefetch_pin="hn=moe.wi_0",
      )
      op = q.ops()[min(q.pins)]
      self.assertEqual(_count_eqns(_program_jaxpr(q, op), "optimization_barrier"), 1)
      with self.assertRaisesRegex(ValueError, "only be passed to top-level"):
        jax.make_jaxpr(q.progs[op.c].program(op.desc))(*q._desc_avals(op.c, op.desc))

  def test_windows_and_gathers(self):
    """The 'auto' estimates on two profiled runs across two v5p-128 slices (R1 = E576,
    one 4096-token row a device; R2 = E1024, cp 2): attention 9.7 / 5.9 ms
    up to 'hn', the router's TopK 4.3 / 2.2, the gmms 11.0 / 10.6 (measured
    9.7 / 5.9, 4.22 / 2.18, 11.5 / 10.2); a routed expert's 4-way gather 4.28 /
    7.62 ms (measured 4.3 / 7.6), the head's joint 64-way gather as a
    multi-slice all-reduce 12.58 (12.6), one axis at a time 8.9, natively over
    the 64 at once 5.9 (a one-slice program, or prefag: at PinModel.wide_gbps,
    the 16-way rate standing in for an unmeasured one); nothing for a norm
    scale."""
    for experts, cp, tokens, want in ((576, 1, 4096, (9.7, 4.31, 11.05)), (1024, 2, 2048, (5.9, 2.16, 10.61))):
      s = _dcn_scfg(experts, cp)
      win = ds_ep.pin_windows(s, _ALL_LAYOUTS[0], tokens)
      self.assertEqual([pt for pt, _, _ in win], ["hn", "xs", "y"])
      np.testing.assert_allclose([ms for _, _, ms in win], want, atol=0.01)
      self.assertEqual(len(ds_ep.pin_windows(s, _ALL_LAYOUTS[1], tokens)), 6)  # two layers
    for experts, routed in ((576, 4.28), (1024, 7.62)):
      for axes in ("joint", "each"):
        s, last = _dcn_scfg(experts, 1, axes), _ALL_LAYOUTS[2]
        ms = lambda i, joint=True: ds_ep.pin_gather_ms(
            s,
            *(
                jax.tree.leaves(t, is_leaf=f)[i]
                for t, f in (
                    (ds_ep.chunk_shapes(s, last), ds_ep._is_shape),
                    (ds_ep.chunk_storage_specs(last), ds_ep._is_spec),
                    (ds_ep.chunk_compute_specs(last), ds_ep._is_spec),
                )
            ),
            joint_allreduce=joint,
        )
        self.assertAlmostEqual(ms(5), routed, places=2)  # moe.wi_0
        self.assertEqual(ms(0), 0.0)  # decoder_norm
        self.assertAlmostEqual(ms(17), 12.58 if axes == "joint" else 8.91, places=2)
        self.assertAlmostEqual(ms(17, joint=False), 5.89 if axes == "joint" else 8.91, places=2)
        # the native 64-way gather (prefag, one slice) takes PinModel.wide_gbps, which no run has measured; the 4-
        # and 16-way gathers keep their measured rates
        half = dataclasses.replace(ds_ep.PIN_MODEL, wide_gbps=ds_ep.PIN_MODEL.wide_gbps / 2)
        wide = lambda i, joint: ds_ep.pin_gather_ms(
            s,
            *(
                jax.tree.leaves(t, is_leaf=f)[i]
                for t, f in (
                    (ds_ep.chunk_shapes(s, last), ds_ep._is_shape),
                    (ds_ep.chunk_storage_specs(last), ds_ep._is_spec),
                    (ds_ep.chunk_compute_specs(last), ds_ep._is_spec),
                )
            ),
            joint_allreduce=joint,
            model=half,
        )
        self.assertAlmostEqual(wide(17, False), 11.77 if axes == "joint" else 8.91, places=2)
        self.assertAlmostEqual(wide(17, True), 12.58 if axes == "joint" else 8.91, places=2)
        self.assertAlmostEqual(wide(5, False), routed, places=2)

  def test_windows_per_sequence(self):
    """Several rows a device (cp 1, a microbatch of b x fsdp x ep sequences)
    each attend within their own sequence: b rows take b times one row's
    attention, not one causal span of b x seq tokens' (b^2 times) -- R1's
    stage at 2 / 4 rows 17.31 / 32.51 ms up to 'hn' (one span: 22.01 /
    60.75), and at 2 rows 'auto' places R1's prefetch host's head and routed
    experts as at one row (the head at 'y', wi_0 / wi_1 at 'hn', wo at 'xs';
    one span: the head in attention, routed wi_0 / wi_1 at 'y').  The pipeline
    plans with its sequence length, at cp 1 and at cp 2 (where a row is half
    of it)."""
    s, last = _dcn_scfg(576, 1), _ALL_LAYOUTS[2]
    attn = lambda b: ds_ep.pin_windows(s, _ALL_LAYOUTS[0], b * 4096, seq=4096)[0][2] - ds_ep.PIN_MODEL.attn_ms
    for b, hn in ((2, 17.31), (4, 32.51)):
      self.assertAlmostEqual(attn(b) + ds_ep.PIN_MODEL.attn_ms, hn, places=2)
      self.assertAlmostEqual(attn(b), b * attn(1))
    names = ds_ep.leaf_names(last)
    plan = ds_ep.plan_pins(
        s, _ALL_LAYOUTS[0], [(last, i) for i in range(len(names))], 2 * 4096, joint_allreduce=True, seq=4096
    )
    at = {names[i]: pt for pt, _, _, i in plan}
    self.assertEqual((at["logits_dense"], at["moe.wi_0"], at["moe.wi_1"], at["moe.wo"]), ("y", "hn", "hn", "xs"))
    seen, real = [], ds_ep.plan_pins

    def spy(*a, **k):
      seen.append((a[3], k.get("seq")))
      return real(*a, **k)

    with mock.patch.object(ds_ep, "plan_pins", spy):  # 2 rows of 256 tokens a device: 8 sequences on 2 x 2
      self.assertTrue(pipe(tiny_xla(fsdp_collectives="xla"), prefetch_pin="auto", n_micro=2, global_batch=16).pins)
    self.assertEqual(set(seen), {(512, 256)})
    seen.clear()
    with mock.patch.object(ds_ep, "plan_pins", spy):  # cp 2: one row of 128 tokens a device, seq 256
      sc = explicit(tiny_xla(layers=4, random_routing=False, fsdp_collectives="xla", iota_embed=False))
      self.assertTrue(
          dsp.Pipeline(
              sc,
              n_stages=2,
              chunks_per_stage=2,
              n_micro=4,
              global_batch=8,
              seq=256,
              cp=2,
              placement="v",
              prefetch_pin="auto",
          ).pins
      )
    self.assertEqual(set(seen), {(128, 256)})

  def test_auto_plans_the_runs_best(self):
    """'auto' on those two runs (R1, R2: test_windows_and_gathers) gives the
    plans an AOT replay of the runs' programs found best: the
    self-prefetch pins chunk 0's routed wi_0 alone at 'hn'; R1's one prefetch
    host (all 18 of chunk 3's leaves) pins both routed wi_0 and wi_1 and the
    small leaves in attention, routed wo at 'xs' and the head at 'y'; R2's two
    -- the routed wi_0 + wo group and the other 16 -- one routed expert in
    attention and the other, or the head, at 'y'.  Every leaf with a gather is
    pinned somewhere, and so with per-axis gathers."""
    chunks = [
        ds_ep.ChunkSpec(index=c, layers=(c,), first=c == 0, last=c == 3, stage=0 if c in (0, 3) else 1) for c in range(4)
    ]
    names = [ds_ep.leaf_names(c) for c in chunks]
    routed = (4, 5, 6)  # chunk 0's moe.wi_0, wi_1, wo
    for axes in ("joint", "each"):
      for run, experts, cp, tokens in (("R1", 576, 1, 4096), ("R2", 1024, 2, 2048)):
        s = _dcn_scfg(experts, cp, axes)
        plan = ds_ep.plan_pins(
            s,
            chunks[0],
            [(chunks[0], i) for i in routed],
            tokens,
            joint_allreduce=True,
            own=tuple((0, i) for i in routed),
        )
        self.assertEqual(plan, (("hn", 0, 0, 4),), (axes, run))
        groups = ds_ep.leaf_groups(ds_ep.chunk_shapes(s, chunks[3]), 1 if run == "R1" else 2)
        at = {}
        for g in groups:
          got = ds_ep.plan_pins(s, chunks[0], [(chunks[3], i) for i in g], tokens, joint_allreduce=True)
          self.assertEqual(sorted(i for *_, i in got), [i for i in g if ds_ep.has_gather(chunks[3], i)], (axes, g))
          at.update({names[3][i]: pt for pt, _, _, i in got})
        what = (axes, run, at)
        if run == "R1":
          self.assertEqual(
              (at["moe.wi_0"], at["moe.wi_1"], at["moe.wo"], at["logits_dense"]), ("hn", "hn", "xs", "y"), what
          )
          if axes == "joint":
            self.assertEqual(at["self_attention.out"], "hn", what)
        else:
          self.assertEqual(groups[0], (5, 7))  # routed wi_0, wo
          self.assertEqual(sorted((at["moe.wi_0"], at["moe.wo"])), ["hn", "y"], what)
          self.assertEqual((at["moe.wi_1"], at["logits_dense"]), ("hn", "y"), what)

  def test_joint_gather_allreduce(self):
    """The planner prices a joint gather as an all-reduce in a program
    compiled for several slices -- counted among the devices' client's
    devices where they have one (a compile-only topology's: ds_ep_aot
    compiles slice 1's stage on slice 0's devices), else the meshes' --
    unless LIBTPU_INIT_ARGS prefers async all-gathers, by its last mention in
    any absl spelling; CPU devices are one slice."""

    def stage(k, client=None):  # a mesh-like stage of two devices of slice k
      return types.SimpleNamespace(
          devices=np.array([types.SimpleNamespace(slice_index=k, client=client)] * 2, dtype=object)
      )

    topology = types.SimpleNamespace(devices=lambda: [types.SimpleNamespace(slice_index=k) for k in (0, 0, 1, 1)])
    one = types.SimpleNamespace(meshes=[stage(0), stage(0)])
    two = types.SimpleNamespace(meshes=[stage(0), stage(1)])
    moved = types.SimpleNamespace(meshes=[stage(0, topology), stage(0, topology)])
    plan = dsp.Pipeline.joint_gather_allreduce
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_foo=1"}):
      self.assertEqual((plan(one), plan(two), plan(moved)), (False, True, True))
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_prefer_async_allgather_to_allreduce=true"}):
      self.assertEqual((plan(two), plan(moved)), (False, False))
    # read as libtpu reads it: the last mention wins, in any absl spelling (ds_ep_pipeline.libtpu_flag)
    f = "xla_tpu_prefer_async_allgather_to_allreduce"
    for args, prefer in (
        (f"--{f}=True", True),
        (f"--{f}=1", True),
        (f"--{f}", True),
        (f"-{f}=yes", True),
        (f"--{f}=true --xla_foo=1 --{f}=false", False),
        (f"--{f}=true --no{f}", False),
        (f"--{f}=false --{f}=TRUE", True),
        (f"--{f}_x=true", False),
    ):
      with self.subTest(args=args), mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": args}):
        self.assertEqual((plan(two), plan(moved), plan(one)), (not prefer, not prefer, False))
    self.assertFalse(plan(pipe(tiny_xla(), n_micro=2, global_batch=8)))

    # v6e's multi-slice programs gather a leaf stored over both axes natively (ds_ep.NATIVE_JOINT_GATHER_KINDS)
    def v6e(k):
      dev = types.SimpleNamespace(slice_index=k, device_kind="TPU v6 lite")
      return types.SimpleNamespace(devices=np.array([dev] * 2, dtype=object))

    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_foo=1"}):
      self.assertFalse(plan(types.SimpleNamespace(meshes=[v6e(0), v6e(1)])))
      self.assertTrue(plan(types.SimpleNamespace(meshes=[v6e(0), stage(1)])))  # a device of another kind

  def test_flag(self):
    """--prefetch-pin reaches the pipeline through `build`, 'off' by
    default; a SPEC of the wrong form is refused by the parser."""
    ap = dsp.make_parser()
    self.assertEqual(ap.parse_args(TINY_ARGS).prefetch_pin, "off")
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(TINY_ARGS + ["--prefetch-pin", "hn:moe.wo"])
    p = dsp.build(
        ap.parse_args(
            TINY_ARGS
            + [
                "--vv",
                "2",
                "--layers",
                "4",
                "--gbs",
                "8",
                "--mb",
                "4",
                "--fsdp-collectives",
                "xla",
                "--prefetch-pin",
                "auto",
            ]
        )
    )
    self.assertEqual(p.prefetch_pin, "auto")
    self.assertTrue(p.pins)
    self.assertIn("prefetch pins (auto)", p.summary())
    # 'auto' prices SparseCore collectives: on collective-permute stages it pins nothing (a SPEC still does)
    args = TINY_ARGS + ["--vv", "2", "--layers", "4", "--gbs", "8", "--mb", "4", "--fsdp-collectives", "ppermute"]
    q = dsp.build(ap.parse_args(args + ["--prefetch-pin", "auto"]))
    self.assertEqual(({c.fsdp_collectives for c in q.stage_scfg}, q.pins), ({"ppermute"}, {}))
    self.assertTrue(dsp.build(ap.parse_args(args + ["--prefetch-pin", "hn=moe.wi_0"])).pins)


@testing.needs_devices(4)
class BatchSplit(unittest.TestCase):
  """--batch-split --batch-split-impl native (StageConfig.batch_split on
  ds_ep.split_layers, off TPU; Lineage's: BatchSplitLineage): each device's rows of a
  microbatch run as two halves through a chunk's layers in Lineage's phases
  (ds_ep.split_layers), and a backward is the VJP of the whole split chunk.
  Top-k routing routes a token alike in either half, so there a split step
  is the unsplit step's to rounding; random routing draws each half's
  experts with a key of its own, which `reference` follows."""

  def _check(self, layers=4, overlap=True, unsplit_reference=False, **kw):
    scfg = tiny_xla(layers=layers, **NATIVE_SPLIT, **kw, **testing.stage_axes())
    p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=16, seq=256, overlap=overlap)
    ref = None
    if unsplit_reference:
      ref = types.SimpleNamespace(
          **{k: getattr(p, k) for k in ("meshes", "chunks", "slot", "gbs", "seq", "M")},
          scfg=dataclasses.replace(scfg, batch_split=False),
      )
    compare_with_reference(self, p, overlap, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4, reference_of=ref)
    return p

  def test_matches_unsplit_reference(self):
    """Top-k routing, one-layer chunks, the default plan (breadth, weights
    per visit), fp32 on XLA's ops: the split step is the unsplit
    reference's to rounding."""
    p = self._check(random_routing=False, unsplit_reference=True)
    self.assertEqual((p.schedule, p.fsdp_scope), ("breadth", "repeat"))

  @testing.slow
  def test_matches_reference_variants(self):
    """Random routing: two-layer chunks (mb0's attention of layer 1 beside
    mb1's combine of layer 0) with the fused last chunk recomputed, the
    plain plan, and the halves' expert weight gradients summed after their
    kernels (split_dw 'sum')."""
    self._check(layers=8, remat_fused_last=True)
    self._check(overlap=False)
    self._check(layers=8, split_dw="sum")

  def test_split_dw(self):
    """split_dw 'acc', the default: mb0's grouped matmuls hand their weights
    on to mb1's, so a backward adds the halves' routed-expert weight
    gradients inside the dW kernel and sums no [E / EP, D, F] leaf, where
    'sum' sums each of a layer's three; every other leaf's halves are summed
    either way.  The forward is the same program."""
    counts = {}
    for split_dw in ds_ep.SPLIT_DW:
      scfg = tiny_xla(layers=8, split_dw=split_dw, **NATIVE_SPLIT, **testing.stage_axes())
      scfg = dataclasses.replace(scfg, model=scfg.model.replace(moe_mlp_dim=128))  # F != D
      p = pipe(scfg, n_micro=2, global_batch=16, fsdp_scope="step")
      cfg = scfg.model
      expert = {
          (cfg.num_experts // scfg.ep, cfg.emb_dim, cfg.moe_mlp_dim),
          (cfg.num_experts // scfg.ep, cfg.moe_mlp_dim, cfg.emb_dim),
      }
      for kind, c, op_kind in (("F", 1, "F"), ("B", 1, "B"), ("A", 3, "F")):  # A: the fused last chunk
        op = next(o for o in p.ops() if (o.c, o.kind) == (c, op_kind))
        shapes = _eqn_out_shapes(_program_jaxpr(p, op), "add_any")
        counts[split_dw, kind] = (sum(s in expert for s in shapes), sum(s not in expert for s in shapes))
    n = len(p.chunks[1].layers)
    self.assertEqual([counts["acc", k][0] for k in "FBA"], [0, 0, 0])
    self.assertEqual([counts["sum", k][0] for k in "FBA"], [0, 3 * n, 3 * n])
    for k in "FBA":
      self.assertEqual(counts["acc", k][1], counts["sum", k][1], k)
    self.assertGreater(counts["acc", "B"][1], 0)

  @testing.slow
  def test_maxtext_kernels(self):
    """bf16 with MaxText's ragged sort and megablox, random routing, under
    lax and collective-permute EP collectives, at PipelineVsReference's
    tolerances.  (Top-k routing is not held to the reference in bf16, split
    or not: rounding flips near-tied experts, and the router's gradient
    leaves its tolerance.)"""
    for epc in ("xla", "ppermute"):
      with self.subTest(ep_collectives=epc):
        p = dsp.Pipeline(
            tiny(ep_collectives=epc, **NATIVE_SPLIT, **testing.stage_axes()),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=stage_rows(16),
            seq=256,
            overlap=True,
        )
        compare_with_reference(self, p, True)

  def test_phases(self):
    """A split forward closes four phases a layer with an optimization
    barrier each, and names each half's EP collectives (a scope inside the
    collective's, which a transform then still wraps); a backward reads x
    and, in chunk 0, the embedding only."""
    p = pipe(tiny_xla(layers=8, **NATIVE_SPLIT, **testing.stage_axes()), n_micro=2, global_batch=16, fsdp_scope="step")
    self.assertEqual([pr.n_res for pr in p.progs], [1, 0, 0, 0])
    op = next(o for o in p.ops() if (o.c, o.kind) == (1, "F"))
    self.assertEqual(_count_eqns(_program_jaxpr(p, op), "optimization_barrier"), 4 * 2)
    text = p.progs[1].program(op.desc).trace(*p._desc_avals(1, op.desc)).lower().as_text(debug_info=True)
    for scope in ("ep_all_gather/mb0", "ep_all_gather/mb1", "ep_reduce_scatter/mb0", "ep_reduce_scatter/mb1"):
      self.assertIn(scope, text)
    unsplit = pipe(tiny_xla(layers=8, **testing.stage_axes()), n_micro=2, global_batch=16, fsdp_scope="step")
    self.assertEqual([pr.n_res for pr in unsplit.progs], [2, 1, 1, 1])
    op = next(o for o in unsplit.ops() if (o.c, o.kind) == (1, "F"))
    self.assertEqual(_count_eqns(_program_jaxpr(unsplit, op), "optimization_barrier"), 0)

  def test_pins_tie_mb0(self):
    """Prefetch pins tie mb0's points, each once a layer (mb1 traces
    untied): a pinned split forward holds its phase barriers and one a
    pinned point."""
    p = pipe(
        tiny_xla(fsdp_collectives="xla", **NATIVE_SPLIT, **testing.stage_axes()),
        n_micro=2,
        global_batch=16,
        fsdp_scope="step",
        prefetch_pin="hn=moe.wi_0;xs=moe.wo",
    )
    self.assertTrue(p.pins)
    for j, plan in p.pins.items():
      op = p.ops()[j]
      n = len(p.chunks[op.c].layers)
      self.assertEqual(
          _count_eqns(_program_jaxpr(p, op), "optimization_barrier"),
          4 * n + len({(pt, layer) for pt, layer, _, _ in plan}),
          p.op_name(j),
      )

  def test_refusals(self):
    """An odd number of rows a device, context parallelism and a stash mode
    are refused."""
    with self.assertRaisesRegex(ValueError, "want an even number"):
      pipe(tiny_xla(**NATIVE_SPLIT, **testing.stage_axes()), n_micro=2, global_batch=stage_rows(8))  # 1 row a device
    with self.assertRaisesRegex(ValueError, "context parallelism"):
      tiny_xla(cp=2, **NATIVE_SPLIT)
    with self.assertRaisesRegex(ValueError, "recompute programs only"):
      pipe(tiny_xla(**NATIVE_SPLIT, **testing.stage_axes()), n_micro=2, global_batch=16).progs[0].build_stash("lite")

  def test_flag(self):
    """--batch-split --batch-split-impl native and --split-dw reach the
    stage config through `build`, and the summary says so.  Under 'acc' the
    preset's wi dW tm is capped at 256 (the accumulating kernel's VMEM:
    split_acc_wi_tiling), in `build` -- ds_ep_parity's --batch-split too,
    which runs this path -- as in StageConfig's own default; a given
    --wi-tiling stands."""
    ap = dsp.make_parser()
    self.assertFalse(ap.parse_args(TINY_ARGS).batch_split)
    old = ["--gbs", "8", "--batch-split", "--batch-split-impl", "native"]
    p = dsp.build(ap.parse_args(TINY_ARGS + old))
    clamped = ds_ep.TILES["clamped"](p.scfg.model)[0]
    self.assertEqual((p.scfg.batch_split, p.scfg.batch_split_impl, p.scfg.lineage_split), (True, "native", False))
    self.assertEqual(p.scfg.split_dw, "acc")
    self.assertEqual((clamped[6], p.scfg.wi_tiling), (512, clamped[:6] + (256,) + clamped[7:]))
    self.assertIn(
        "batch_split=native (ds_ep.split_layers; each device's 2 rows of a microbatch as two halves",
        p.summary(),
    )
    self.assertIn("expert weight gradients accumulated in the dW kernel", p.summary())
    self.assertEqual((p.scfg.random_routing, p.scfg.shard_mode), (False, "auto"))  # TINY_ARGS: --routing topk
    self.assertEqual(dsp.provenance(p.scfg)["batch_split_impl"], "native")
    self.assertNotIn("lineage", dsp.provenance(p.scfg))
    p = dsp.build(ap.parse_args(TINY_ARGS + old + ["--split-dw", "sum"]))
    self.assertEqual((p.scfg.split_dw, p.scfg.wi_tiling), ("sum", clamped))
    self.assertIn("expert weight gradients summed after their kernels", p.summary())
    given = (512,) * 9
    p = dsp.build(ap.parse_args(TINY_ARGS + old + ["--wi-tiling", *map(str, given)]))
    self.assertEqual(p.scfg.wi_tiling, given)
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par  # its preset's tiles: tm 512

    self.assertEqual(
        [
            par.build(
                "small",
                random_routing=True,
                batch_split=split,
                batch_split_impl="native",
                gbs=16,
                **testing.stage_axes(),
            ).scfg.wi_tiling[6]
            for split in (False, True)
        ],
        [512, 256],
    )
    cfg = p.scfg.model
    self.assertEqual(
        [
            ds_ep.StageConfig(model=cfg, **kw).wi_tiling[6]
            for kw in ({}, NATIVE_SPLIT, dict(NATIVE_SPLIT, split_dw="sum"))
        ],
        [512, 256, 512],
    )
    with self.assertRaisesRegex(ValueError, "split_dw must be one of"):
      tiny_xla(split_dw="add", **testing.stage_axes())


class LineageStandIn:
  """A stand-in for ds_ep_lineage.Region that honours its contract with ds_ep's own unsplit layer math, for the CPU
  plumbing checks (Lineage's kernels are TPU-only): `region(w_layers, w_dense, h) -> (y, aux)` at global view on
  the Explicit stage mesh -- the weights typed ds_ep.chunk_wc_specs and h P(FE, None, None), checked when it traces
  -- a jax.custom_vjp whose rules are nested jits (`traces` counts them: every program of one chunk structure
  shares them) and whose weight cotangents come out typed chunk_gacc_specs(explicit=True), each device's partial,
  unreduced.  It computes dense_layer_fwd / layer_fwd per device in a shard_map, unsplit, as an unsplit chunk does:
  so a step through it is the unsplit pipeline's to fp32 rounding, whatever the programs around it do differently.
  `aux` is None (ds_ep's programs read no routing).  It keeps what `region` was handed for the rest of a Region's
  identity, `capacity_factor` and `sparsecore` (which only Lineage's kernels would use)."""

  def __init__(self, scfg, ch, mesh, capacity_factor=None, sparsecore=None):
    nd, n = ch.dense, ch.n_moe
    self.traces = collections.Counter()
    self.mesh, self.capacity_factor, self.sparsecore = mesh, capacity_factor, sparsecore
    spec = dataclasses.replace(ch, index=0, first=False, last=False, mtp=0)  # its layers only
    wc, ga = ds_ep.chunk_wc_specs(scfg, spec), ds_ep.chunk_gacc_specs(spec, explicit=True)
    act = P(ds_ep.FE, None, None)
    keys = ("layers",) + (("dense_layers",) if nd else ())

    def layers(w, h):
      for l in range(nd):
        h = ds_ep.dense_layer_fwd(ds_ep._take(w["dense_layers"], l), h, scfg)
      for l in range(n):
        h = ds_ep.layer_fwd(ds_ep._take(w["layers"], l), h, None, scfg)
      return h

    def layers_vjp(w, h, dy):
      return jax.vjp(layers, w, h)[1](dy)

    sm = lambda f, i, o: jax.shard_map(f, mesh=mesh, in_specs=i, out_specs=o, check_vma=False)
    fwd = sm(layers, ({k: wc[k] for k in keys}, act), act)
    bwd = sm(layers_vjp, ({k: wc[k] for k in keys}, act, act), ({k: ga[k] for k in keys}, act))

    def fwd_rule(w, h):
      self.traces["fwd"] += 1
      for k in keys:
        jax.tree.map(lambda x, s, k=k: self._same(k, x, s), w[k], wc[k], is_leaf=ds_ep._is_spec)
      self._same("h", h, act)
      return fwd(w, h), (w, h)

    def bwd_rule(res, dy):
      self.traces["bwd"] += 1
      return bwd(*res, dy)

    fwd_jit, bwd_jit = jax.jit(fwd_rule), jax.jit(bwd_rule)
    pack = lambda wl, wd: {"layers": wl, **({"dense_layers": wd} if nd else {})}

    @jax.custom_vjp
    def call(wl, wd, h):
      return fwd_jit(pack(wl, wd), h)[0], None

    def call_fwd(wl, wd, h):
      y, res = fwd_jit(pack(wl, wd), h)
      return (y, None), res

    def call_bwd(res, cts):
      dw, dh = bwd_jit(res, cts[0])
      return dw["layers"], dw.get("dense_layers"), dh

    call.defvjp(call_fwd, call_bwd)
    self._call = call

  @staticmethod
  def _same(name, x, spec):
    got = jax.typeof(x).sharding.spec
    if got != spec:
      raise AssertionError(f"the region got {name} typed {got}, want {spec}")

  def __call__(self, w_layers, w_dense, h):
    return self._call(w_layers, w_dense, h)


@contextlib.contextmanager
def lineage_stand_in():
  """ds_ep_lineage.region -> a LineageStandIn per chunk structure, stage mesh, capacity factor and SparseCore answer
  (as region caches its Regions), -> {key: stand-in}; and ds_ep_lineage.check, the model part of StageConfig's
  validation, a no-op: the stand-in computes the tiny fp32 models the reference tests use, which Lineage's bf16
  kernels would refuse."""
  from maxtext.experimental.mpmd.jax import ds_ep_lineage

  made = {}

  def region(scfg, ch, mesh, capacity_factor=ds_ep_lineage.CAPACITY_FACTOR, sparsecore=None):
    key = (scfg, ch.n_moe, ch.dense, ch.routed_bias, mesh, capacity_factor, sparsecore)
    if key not in made:
      made[key] = LineageStandIn(scfg, ch, mesh, capacity_factor, sparsecore)
    return made[key]

  with (
      mock.patch.object(ds_ep_lineage, "region", region),
      mock.patch.object(ds_ep_lineage, "check", lambda cfg=None, scfg=None: None),
  ):
    yield made


def lineage_scfg(**kw):
  """tiny() as batch_split_impl 'lineage' takes it: explicit shard mode, top-k routing (and Lineage-valid: bf16,
  D 256)."""
  return tiny(**{**dict(batch_split=True, shard_mode="explicit", random_routing=False), **kw})


def lineage_split_scfg(**kw):
  """tiny_xla() batch-split on Lineage's layers, as the stand-in region runs it (fp32, ragged_dot: the model
  part of StageConfig's check would refuse it -- make it inside lineage_stand_in, which mutes that)."""
  return dataclasses.replace(tiny_xla(shard_mode="explicit", random_routing=False), batch_split=True, **kw)


@testing.needs_devices(4)
class BatchSplitLineage(unittest.TestCase):
  """--batch-split with batch_split_impl 'lineage' (the default): a batch-split chunk's decoder layers are
  ds_ep_lineage's Region -- Lineage's layers -- and its programs global-view jits (ds_ep.ChunkPrograms), the
  embedding, the head and the folded gathers / reduce-scatters in shard_maps of their own.  Lineage's kernels run
  on TPU only (ds_ep_lineage_test.OnTPU); here the configuration, and the programs around a stand-in region
  (LineageStandIn) with ds_ep's own unsplit math, against the unsplit pipeline.  Of those pipeline runs only the
  default plan's (test_programs_match_unsplit) runs on every PR, the rest are scheduled_only."""

  def tearDown(self):
    # A test compiles up to three CPU pipelines, whose executables map thousands of regions into the process against
    # vm.max_map_count's 65530, and a dead one is cyclic garbage (LineageStandIn's custom_vjp closes over itself):
    # free them before the next test, not whenever Python's own full collection comes (after ds_ep_lineage_test in
    # one process it came too late: "Cannot allocate memory").
    gc.collect()

  def test_config(self):
    """StageConfig: 'lineage' by default; it refuses what Lineage's layers cannot run (shard_mode auto, grad_reduce
    microbatch, random routing, a token-dropping buffer, cp, experts not divisible by ep, a model ds_ep_lineage.check
    refuses), every problem at once, without importing Lineage (no MaxText needed); and neither caps the wi tiling
    nor warns of padded megablox tiles (none run).  Impl 'native' is today's batch split."""
    self.assertEqual(ds_ep.BATCH_SPLIT_IMPLS, ("lineage", "native"))
    from maxtext.experimental.mpmd.jax import ds_ep_lineage

    base = lineage_scfg()
    self.assertEqual((base.batch_split_impl, base.lineage_split), ("lineage", True))
    self.assertEqual(base.lineage_capacity_factor, ds_ep_lineage.CAPACITY_FACTOR)
    self.assertFalse(dataclasses.replace(base, batch_split=False).lineage_split)
    self.assertFalse(dataclasses.replace(base, batch_split_impl="native").lineage_split)
    for change, want in (
        (dict(shard_mode="auto"), "shard_mode 'auto', want 'explicit'"),
        (dict(grad_reduce="microbatch"), "grad_reduce 'microbatch', want 'step'"),
        (dict(random_routing=True), "random routing: Lineage routes top-k only"),
        (dict(ragged_buffer_factor=2.0), "ragged_buffer_factor 2.0: Lineage is dropless"),
        (dict(cp=2), "cp 2: the batch split halves the rows"),
        (dict(fsdp=1, ep=4, model=base.model.replace(num_experts=6)), "num_experts 6 is not a " "multiple of ep 4"),
        (dict(model=base.model.replace(dtype=jnp.float32)), "bf16 activations only"),
        (dict(model=base.model.replace(routed_score_func="softmax")), "sigmoid only"),
        (dict(batch_split_impl="megablox"), "batch_split_impl must be one of"),
        (dict(lineage_capacity_factor=0.0), "lineage_capacity_factor must be a number > 0"),
    ):
      with self.subTest(change=str(change)), self.assertRaisesRegex(ValueError, re.escape(want)):
        dataclasses.replace(base, **change)
    with self.assertRaises(ValueError) as e:
      dataclasses.replace(base, shard_mode="auto", random_routing=True, model=base.model.replace(dtype=jnp.float32))
    for want in ("want 'explicit'", "random routing", "bf16 activations", "batch_split_impl 'native'"):
      self.assertIn(want, str(e.exception))
    # the same settings under the old path, or unsplit, are fine
    for ok in (
        dict(batch_split_impl="native", shard_mode="auto", random_routing=True),
        dict(batch_split=False, shard_mode="auto", random_routing=True),
    ):
      dataclasses.replace(base, **ok)
    # megablox's tilings: no split_acc cap and no padding warning under Lineage (none of its layers runs megablox)
    cfg = base.model.replace(moe_mlp_dim=128)  # F < D: tk = D pads wo's kernels
    clamped, padded = ds_ep.TILES["clamped"](cfg), ((512, cfg.emb_dim, 512) * 3,) * 2
    self.assertEqual(
        ds_ep.StageConfig(model=cfg, batch_split=True, shard_mode="explicit", random_routing=False).wi_tiling, clamped[0]
    )
    self.assertEqual(ds_ep.StageConfig(model=cfg, **NATIVE_SPLIT).wi_tiling[6], 256)
    self.assertTrue(ds_ep.tiling_problems(cfg, *padded))
    with warnings.catch_warnings(record=True) as got:
      warnings.simplefilter("always")
      ds_ep.StageConfig(
          model=cfg,
          wi_tiling=padded[0],
          wo_tiling=padded[1],
          strict_tiling=True,
          batch_split=True,
          shard_mode="explicit",
          random_routing=False,
      )
      self.assertFalse([w for w in got if ds_ep.TILING_WARNING in str(w.message)])
      ds_ep.StageConfig(model=cfg, wi_tiling=padded[0], wo_tiling=padded[1], **NATIVE_SPLIT)
      self.assertTrue([w for w in got if ds_ep.TILING_WARNING in str(w.message)])

  def test_flag(self):
    """--batch-split runs Lineage's layers unless --batch-split-impl native.  The parser resolves the defaults it
    decides (split_defaults): --routing topk and --shard-mode explicit on Lineage's layers, random and auto elsewhere
    (unsplit, or the native impl); built -- no Lineage needed to build -- the stage meshes are Explicit, the impl
    reaches the stage config with --lineage-capacity-factor, the summary and provenance name it."""
    ap = dsp.make_parser()
    no_routing = [x for x in TINY_ARGS if x not in ("--routing", "topk")]
    self.assertEqual(len(no_routing), len(TINY_ARGS) - 2)
    for extra, want in (
        ([], ("random", "auto")),
        (["--batch-split"], ("topk", "explicit")),
        (["--batch-split", "--batch-split-impl", "native"], ("random", "auto")),
        (["--batch-split-impl", "lineage"], ("random", "auto")),  # no batch split
        (["--batch-split", "--routing", "random"], ("random", "explicit")),
        (["--batch-split", "--shard-mode", "auto"], ("topk", "auto")),
    ):
      with self.subTest(extra=extra):
        a = ap.parse_args(no_routing + extra)
        self.assertEqual((a.routing, a.shard_mode), want)
        self.assertEqual(a.batch_split_impl, "native" if "native" in extra else "lineage")
    a = ap.parse_args(no_routing)
    self.assertEqual(a.lineage_capacity_factor, ds_ep_lineage.CAPACITY_FACTOR)
    # a namespace made by hand: build fills in what is missing alike, and leaves the caller's as it is
    hand = argparse.Namespace(
        **{**{k: v for k, v in vars(a).items() if k not in ("routing", "shard_mode")}, "batch_split": True, "gbs": 8}
    )
    p = dsp.build(hand)
    self.assertEqual((p.scfg.random_routing, p.scfg.shard_mode), (False, "explicit"))
    self.assertFalse(hasattr(hand, "routing") or hasattr(hand, "shard_mode"))
    args = no_routing + ["--gbs", "8", "--batch-split"]
    p = dsp.build(ap.parse_args(args + ["--lineage-capacity-factor", "3"]))
    self.assertTrue(p.scfg.lineage_split)
    self.assertEqual((p.scfg.lineage_capacity_factor, p.scfg.random_routing), (3.0, False))
    self.assertTrue(all(t == AxisType.Explicit for m in p.meshes for t in m.axis_types))
    self.assertEqual(p.scfg.wi_tiling, ds_ep.TILES["clamped"](p.scfg.model)[0])  # no split_acc cap
    text = p.summary()
    self.assertIn(
        "batch_split=lineage (each device's 2 rows of a microbatch as two halves through MaxText's " "Lineage layers",
        text,
    )
    self.assertIn("capacity factor 3)", text)
    # CPU devices offload nothing: both stages' Regions would trace Lineage's SparseCore regions as plain code
    self.assertEqual(p.lineage_sparsecore(), {0: False, 1: False})
    self.assertIn(
        "EP collectives: Lineage's own (as TensorCore code on stage [0, 1], where libtpu offloads no " "collective)",
        text,
    )
    self.assertTrue(all(pr.lineage for pr in p.progs))
    pv = dsp.provenance(p.scfg)
    self.assertEqual((pv["batch_split_impl"], pv["lineage_capacity_factor"]), ("lineage", 3.0))
    self.assertIn("lineage", pv)  # its path and git state, or None and the error without Lineage
    self.assertIn("batch split on Lineage", dsp.provenance_line(pv))
    with self.assertRaisesRegex(ValueError, "lineage_capacity_factor must be a number > 0"):
      dsp.build(ap.parse_args(args + ["--lineage-capacity-factor", "0"]))
    # the same flags unsplit, or on split_layers (native, off TPU): the defaults as ever
    p = dsp.build(ap.parse_args(args + ["--batch-split-impl", "native"]))
    self.assertEqual((p.scfg.lineage_split, p.scfg.random_routing, p.scfg.shard_mode), (False, True, "auto"))
    self.assertTrue(all(t == AxisType.Auto for m in p.meshes for t in m.axis_types))

  def test_flag_refusals(self):
    """What Lineage's layers cannot run, refused by `build` before anything is built, each named by its flag and
    all at once (lineage_problems), the model's too (ds_ep_lineage.check); the same with --batch-split-impl
    native, or unsplit, builds.  What only picks among ds_ep's own layer kernels is accepted, and named
    (lineage_ignored) for main to warn of."""
    ap = dsp.make_parser()
    args = TINY_ARGS + ["--gbs", "8", "--batch-split"]
    cases = (
        (["--routing", "random"], "--routing random: Lineage routes top-k only"),
        (["--shard-mode", "auto"], "--shard-mode auto: Lineage's code reads its layouts from sharding types"),
        (["--ep", "1"], "--fsdp 1 --ep 1: a one-device stage, which the Explicit stage meshes "),
        (["--grad-reduce", "microbatch"], "--grad-reduce microbatch: "),
        (["--ragged-buffer-factor", "2", "--ragged-sort", "on"], "--ragged-buffer-factor 2: Lineage is dropless"),
        (["--cp", "2", "--gbs", "4"], "--cp 2: the batch split halves the rows"),
        (["--stash", "lite", "--schedule", "stash"], "--stash lite / --schedule stash: "),
        (["--schedule", "stash"], "--schedule stash: "),
        (["--prefetch-pin", "auto"], "--prefetch-pin auto: pins tie gathers"),
        (["--prefetch-pin", "hn=moe.wi_0"], "--prefetch-pin hn=moe.wi_0: "),
        (["--dtype", "float32"], "dtype float32: Lineage's expert kernels take bf16 activations only"),
        (["--experts", "6", "--ep", "4", "--gbs", "16"], "--experts 6 is not a multiple of --ep 4: "),
        (["--gbs", "4"], "--gbs 4 / --mb 2: 1 row(s) a device a microbatch (fsdp x ep 2): "),
        (
            ["--vv", "2", "--layers", "5", "--dense-layers", "2", "--mlp", "512", "--layer-split", "2,1,1,1"],
            "--layer-split 2,1,1,1: chunk(s) [0] hold dense layers only",
        ),
    )
    for extra, want in cases:
      with self.subTest(extra=extra):
        with self.assertRaisesRegex(
            ValueError, "--batch-split runs Lineage's layers .*cannot run this: .*" + re.escape(want)
        ):
          dsp.build(ap.parse_args(args + extra))
        if {"--cp", "--stash", "--experts", "--gbs"} & set(extra):
          continue  # (split_layers takes no cp, stash policy, experts % ep or odd rows a device either)
        p = dsp.build(ap.parse_args(args + extra + ["--batch-split-impl", "native"]))
        if "--layer-split" in extra:  # its chunk of dense layers alone runs unsplit: nothing to split
          self.assertEqual([pr.split for pr in p.progs], [False, True, True, True])
    # (the two ChunkPrograms would refuse, here before anything is built: nothing probed, no program made)
    with (
        mock.patch.object(dsp, "offload_answers", side_effect=AssertionError("probed")),
        mock.patch.object(ds_ep, "ChunkPrograms", side_effect=AssertionError("built")),
    ):
      for extra, _ in cases[-2:]:
        self.assertRaises(ValueError, dsp.build, ap.parse_args(args + extra))
    with self.assertRaises(ValueError) as e:
      dsp.build(
          ap.parse_args(
              args
              + [
                  "--routing",
                  "random",
                  "--shard-mode",
                  "auto",
                  "--prefetch-pin",
                  "auto",
                  "--dtype",
                  "float32",
                  "--gbs",
                  "4",
              ]
          )
      )
    for want in (
        "--routing random",
        "--shard-mode auto",
        "--prefetch-pin auto",
        "bf16 activations",
        "--gbs 4 / --mb 2",
        "(off TPU, --batch-split-impl native runs ds_ep.split_layers)",
    ):
      self.assertIn(want, str(e.exception))
    with self.assertRaises(ValueError) as e:  # a one-device stage in the same message (not StageConfig's, after)
      dsp.build(ap.parse_args(args + ["--ep", "1", "--routing", "random"]))
    for want in ("--routing random", "--fsdp 1 --ep 1: a one-device stage"):
      self.assertIn(want, str(e.exception))
    # A hand-built namespace's prefetch_pin None is off, as Pipeline reads it.
    hand = ap.parse_args(args)
    hand.prefetch_pin = None
    self.assertTrue(dsp.build(hand).scfg.lineage_split)
    # What picks among ds_ep's own layer kernels is accepted and named (main warns of each): Lineage's splash is
    # base-2 always, so --sa-base2-exp off is one (ds_ep_lineage.IGNORED_MODEL_FIELDS), on is not.
    a = ap.parse_args(args + ["--sa-base2-exp", "off", "--expert-weights", "sliced", "--wi-tiling", *["128"] * 9])
    p = dsp.build(a)
    self.assertEqual((p.scfg.lineage_split, p.scfg.model.sa_use_base2_exp), (True, False))
    self.assertNotIn("expert_weights=sliced", p.summary())  # no megablox reads anything under Lineage
    self.assertEqual(
        dsp.lineage_ignored(a),
        [
            "--gmm-impl ragged_dot",
            f"--wi-tiling {' '.join(['128'] * 9)}",
            "--ragged-sort off",
            "--expert-weights sliced",
            "--attn reference",
            "--sa-base2-exp off",
        ],
    )
    self.assertNotIn(
        "--sa-base2-exp on",
        dsp.lineage_ignored(ap.parse_args(args + ["--sa-base2-exp", "on", "--attn", "tokamax_splash"])),
    )
    for other in (TINY_ARGS, args + ["--batch-split-impl", "native"]):
      self.assertEqual(dsp.lineage_ignored(ap.parse_args(list(other) + ["--sa-base2-exp", "off"])), [])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      dsp.main(args + ["--sa-base2-exp", "off", "--dry-run"])
    self.assertIn("warning: --sa-base2-exp off: not used by --batch-split on Lineage's layers", out.getvalue())
    self.assertIn("warning: --gmm-impl ragged_dot: not used by --batch-split on Lineage's layers", out.getvalue())

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_parity_build(self):
    """ds_ep_parity's --batch-split runs Lineage's layers by default: its build routes top-k on Explicit stage meshes
    there (the port check's pipeline; the monolith's random routing is refused, so main skips it) and passes
    --lineage-capacity-factor on; --batch-split-impl native (off TPU) keeps ds_ep.split_layers' random routing and
    auto shard mode.  Its --ladder leaves out the stash rungs under batch split, and builds each rung with
    --pair-stages only where it is interleaved's (any other schedule refuses it); two rungs whose stages' Lineage
    SparseCore answers differ do not run the same programs (programs_of)."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    names = lambda **kw: [":".join(r.values()) for r in par.parse_rungs(par.RUNGS, 2, 8, 2, **kw)]
    self.assertEqual(len(names()), 6)
    self.assertEqual(
        names(batch_split=True), ["circular:search:none", "circular:timed:none", "v:search:none", "v:timed:none"]
    )
    # ... as main hands them to ladder: --pair-stages reaches the interleaved rung's build only
    built = []

    def build(preset, random_routing, **kw):
      built.append((kw["schedule"], kw.get("pair_stages")))
      raise RuntimeError("built")

    with (
        mock.patch.object(dsp, "lineage_missing", lambda a: None),
        mock.patch.object(dsp, "lineage_off_tpu", lambda a, devices=None: None),
        mock.patch.object(par, "build", build),
    ):
      for rungs, want in (
          ("circular:search:none,circular:interleaved:none", ("search", None)),
          ("circular:interleaved:none,circular:search:none", ("interleaved", "1")),
      ):
        built.clear()
        with self.subTest(rungs=rungs), contextlib.redirect_stdout(io.StringIO()):
          with self.assertRaisesRegex(RuntimeError, "built"):
            par.main(
                ["--preset", "tiny4", "--gbs", "32", "--batch-split", "--ladder", "--skip-port", "--pair-stages", "1"]
                + ["--rungs", rungs]
            )
          self.assertEqual(built, [want])
    with lineage_stand_in():
      ps = [
          pipe(lineage_split_scfg(), n_micro=2, global_batch=16, stage_lineage_sparsecore=sc)
          for sc in ((False, True), (False, True), (False, False))
      ]
      self.assertEqual([par.programs_of(q) == par.programs_of(ps[0]) for q in ps], [True, True, False])
    # (tiny is a model Lineage builds: bf16, D 256, top-k of all 8 experts)
    p = par.build("tiny", random_routing=False, topk=8, batch_split=True, gbs=16, lineage_capacity_factor=3.0)
    self.assertEqual(
        (p.scfg.lineage_split, p.scfg.random_routing, p.scfg.shard_mode, p.scfg.lineage_capacity_factor),
        (True, False, "explicit", 3.0),
    )
    self.assertEqual(
        par.build("tiny", random_routing=False, topk=8, batch_split=True, gbs=16).scfg.lineage_capacity_factor,
        ds_ep_lineage.CAPACITY_FACTOR,
    )
    with self.assertRaisesRegex(ValueError, "--routing random: Lineage routes top-k only"):
      par.build("tiny", random_routing=True, batch_split=True, gbs=16)
    q = par.build("tiny", random_routing=True, batch_split=True, batch_split_impl="native", gbs=16)
    self.assertEqual((q.scfg.lineage_split, q.scfg.random_routing, q.scfg.shard_mode), (False, True, "auto"))
    r = par.build("tiny", random_routing=False, topk=8)  # unsplit: as ever
    self.assertEqual((r.scfg.batch_split, r.scfg.random_routing, r.scfg.shard_mode), (False, False, "auto"))

  def test_parity_main_needs_a_check(self):
    """ds_ep_parity's main fails, not ALL PASS, when its flags leave nothing to compare: --skip-port under
    --batch-split on Lineage's layers (no monolith phase: it routes at random), or with --skip-monolith; a --ladder
    whose rungs are all skipped (pair stages on a schedule other than interleaved; a stash rung under --batch-split,
    which main hands to the ladder), or of one rung with no reference.  Nothing is built (Lineage's presence
    stubbed: the check comes before), and as nothing would run Lineage's layers there, these CPU devices are not
    refused either (test_main_needs_a_tpu)."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    ladder = ["--preset", "tiny", "--ladder", "--rungs"]
    cases = (
        (["--preset", "tiny", "--gbs", "16", "--batch-split", "--skip-port"], None),
        (["--preset", "tiny", "--skip-port", "--skip-monolith"], None),
        (ladder + ["circular:search:none:1"], "skipping rung circular:search:none:1: only interleaved pairs"),
        (
            ladder + ["circular:stash:lite", "--gbs", "16", "--batch-split", "--batch-split-impl", "native"],
            "skipping rung circular:stash:lite: batch split runs the recompute programs only",
        ),
        (ladder + ["circular:search:none", "--skip-port", "--skip-monolith"], None),
    )
    with (
        mock.patch.object(dsp, "lineage_missing", lambda a: None),
        mock.patch.object(par, "build", side_effect=AssertionError("built")),
    ):
      for argv, note in cases:
        with self.subTest(argv=argv):
          out = io.StringIO()
          with contextlib.redirect_stdout(out), self.assertRaisesRegex(SystemExit, "^ds_ep_parity: nothing checked"):
            par.main(argv)
          self.assertNotIn("ALL PASS", out.getvalue())
          if note:
            self.assertIn(note, out.getvalue())
    with contextlib.redirect_stdout(io.StringIO()):
      for results, code in (([True, True], 0), ([True, False], 1)):
        with self.assertRaises(SystemExit) as e:
          par.finish(results)
        self.assertEqual(e.exception.code, code)

  def test_main_needs_lineage(self):
    """main stops before anything is built when Lineage does not import (naming what it misses), but for a dry run,
    which compiles nothing and only warns; on the native impl it says nothing."""
    args = TINY_ARGS + ["--gbs", "8", "--batch-split"]

    def missing():
      raise ImportError(
          "ds_ep_lineage could not import MaxText's Lineage stack (maxtext.experimental.lineage): No module named "
          "'typeguard' -- a module it imports is missing"
      )

    with mock.patch.object(ds_ep_lineage, "import_lineage", missing):
      self.assertIn("'typeguard'", dsp.lineage_missing(dsp.make_parser().parse_args(args)))
      self.assertIsNone(dsp.lineage_missing(dsp.make_parser().parse_args(TINY_ARGS)))
      with self.assertRaisesRegex(
          SystemExit,
          r"--batch-split: ds_ep_lineage could not import MaxText's Lineage stack .*'typeguard'.*"
          r"\(off TPU, --batch-split-impl native runs without it\)$",
      ):
        dsp.main(args)
      out = io.StringIO()
      with contextlib.redirect_stdout(out):
        dsp.main(args + ["--dry-run"])
      self.assertIn("warning: --batch-split: ds_ep_lineage could not import MaxText's Lineage stack", out.getvalue())
      self.assertIn("batch_split=lineage", out.getvalue())
      out = io.StringIO()
      with contextlib.redirect_stdout(out):
        dsp.main(args + ["--batch-split-impl", "native", "--dry-run"])
      self.assertNotIn("warning:", out.getvalue())
      self.assertNotIn("Lineage stack", out.getvalue())

  def test_main_needs_a_tpu(self):
    """Lineage's kernels have no CPU path: on this process's CPU devices main stops --batch-split on Lineage's layers
    before anything is built, naming the devices and --batch-split-impl native, but for a dry run, which only warns
    (lineage_off_tpu).  It asks only after init_distributed, never in lineage_missing's gate before it: asking
    initializes the backend, which jax.distributed.initialize refuses to follow (every --distributed run on Lineage's
    layers would stop there).  ds_ep_parity stops where something would run them -- its port phase, --ladder -- and
    --skip-port alone still checks nothing (test_parity_main_needs_a_check).  Lineage's presence stubbed (else
    lineage_missing stops them first)."""
    if {d.platform for d in jax.devices()} != {"cpu"}:
      self.skipTest("on CPU devices (JAX_PLATFORMS=cpu)")
    ap = dsp.make_parser()
    args = TINY_ARGS + ["--gbs", "8", "--batch-split"]
    self.assertRegex(
        dsp.lineage_off_tpu(ap.parse_args(args)),
        r"^Lineage's layers run TPU kernels only .*, and this "
        r"process's devices are cpu ones \(--batch-split-impl "
        r"native runs here\)$",
    )
    tpu = types.SimpleNamespace(platform="tpu", device_kind="TPU7x")
    self.assertIsNone(dsp.lineage_off_tpu(ap.parse_args(args), [tpu] * 8))
    self.assertIn("cpu ones", dsp.lineage_off_tpu(ap.parse_args(args), [tpu] * 7 + jax.devices()[:1]))
    for other in (TINY_ARGS, args + ["--batch-split-impl", "native"]):  # Lineage's layers do not run
      self.assertIsNone(dsp.lineage_off_tpu(ap.parse_args(list(other))))
    want = r"^--batch-split: Lineage's layers run TPU kernels only .*\(--batch-split-impl native runs here\)$"
    with mock.patch.object(dsp, "lineage_missing", lambda a: None):
      with mock.patch.object(dsp, "build", side_effect=AssertionError("built")):
        with self.assertRaisesRegex(SystemExit, want):
          dsp.main(args)
      out = io.StringIO()
      with contextlib.redirect_stdout(out):
        dsp.main(args + ["--dry-run"])
      self.assertIn("warning: --batch-split: Lineage's layers run TPU kernels only", out.getvalue())
      self.assertIn("batch_split=lineage", out.getvalue())  # (the plan, as ever)
      out = io.StringIO()
      with contextlib.redirect_stdout(out):
        dsp.main(args + ["--batch-split-impl", "native", "--dry-run"])
      self.assertNotIn("TPU kernels only", out.getvalue())
      calls = []
      with (
          mock.patch.object(dsp, "init_distributed", lambda: calls.append("init")),
          mock.patch.object(dsp, "lineage_off_tpu", lambda a, devices=None: calls.append("tpu")),
          mock.patch.object(dsp, "build", side_effect=AssertionError("built")),
      ):
        with self.assertRaisesRegex(AssertionError, "built"):
          dsp.main(args + ["--distributed"])
      self.assertEqual(calls, ["init", "tpu"])  # (asked after init_distributed: asking initializes the backend)
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    with (
        mock.patch.object(dsp, "lineage_missing", lambda a: None),
        mock.patch.object(par, "build", side_effect=AssertionError("built")),
    ):
      for argv in (
          ["--preset", "tiny", "--gbs", "16", "--batch-split"],
          ["--preset", "tiny", "--gbs", "16", "--batch-split", "--ladder", "--skip-port"],
      ):
        with (
            self.subTest(argv=argv),
            self.assertRaisesRegex(
                SystemExit, r"^ds_ep_parity --batch-split: Lineage's " r"layers run TPU kernels only .*cpu ones"
            ),
        ):
          par.main(argv)

  def test_flags_where_lineage_does_not_run(self):
    """lineage_ignored's other direction: --lineage-capacity-factor where Lineage's layers do not run (unsplit, or
    --batch-split-impl native) and --batch-split-impl native without --batch-split are named (lineage_unused), at
    ds_ep_pipeline's defaults only where the caller knows them given (ds_ep_parity's --lineage-capacity-factor,
    whose default is None), and main and ds_ep_parity's main warn of each; and --float32-gate-logits on, which
    Lineage's layers apply in part only (fp32 selection, the router kernel rounded to bf16), is named on them alone
    (lineage_partial)."""
    ap = dsp.make_parser()
    split = TINY_ARGS + ["--gbs", "8", "--batch-split"]
    cf, native = ["--lineage-capacity-factor", "5"], ["--batch-split-impl", "native"]
    cf2, impl = ["--lineage-capacity-factor", "2"], ["--batch-split-impl", "lineage"]  # (the defaults)
    for argv, want in (
        (TINY_ARGS + cf, ["--lineage-capacity-factor 5: not used without --batch-split"]),
        (split + native + cf, ["--lineage-capacity-factor 5: not used by --batch-split-impl native"]),
        (TINY_ARGS + native, ["--batch-split-impl native: not used without --batch-split"]),
        (
            TINY_ARGS + native + cf,
            [
                "--lineage-capacity-factor 5: not used without --batch-split",
                "--batch-split-impl native: not used without --batch-split",
            ],
        ),
        (split + cf, []),
        (split, []),
        (split + native, []),
        (TINY_ARGS, []),
        (TINY_ARGS + cf2, []),
        (TINY_ARGS + impl, []),
    ):  # (main's defaults: not told apart)
      with self.subTest(argv=argv[len(TINY_ARGS) :]):
        self.assertEqual([line.split(" (")[0] for line in dsp.lineage_unused(ap.parse_args(argv))], want)
    for argv, given, want in (
        (TINY_ARGS + cf2, {"lineage_capacity_factor"}, ["--lineage-capacity-factor 2: not used without --batch-split"]),
        (
            split + native + cf2,
            {"lineage_capacity_factor"},
            ["--lineage-capacity-factor 2: not used by --batch-split-impl native"],
        ),
        (TINY_ARGS + impl, {"batch_split_impl"}, ["--batch-split-impl lineage: not used without --batch-split"]),
        (split + cf2, {"lineage_capacity_factor"}, []),
    ):  # (read there)
      with self.subTest(argv=argv[len(TINY_ARGS) :], given=given):
        self.assertEqual([line.split(" (")[0] for line in dsp.lineage_unused(ap.parse_args(argv), given)], want)
    f32 = ["--float32-gate-logits", "on"]
    self.assertEqual(
        [line.split(":")[0] for line in dsp.lineage_partial(ap.parse_args(split + f32))], ["--float32-gate-logits on"]
    )
    self.assertIn("rounded to bf16", dsp.lineage_partial(ap.parse_args(split + f32))[0])
    for argv in (split, TINY_ARGS + f32, split + native + f32):
      self.assertEqual(dsp.lineage_partial(ap.parse_args(argv)), [])
    with mock.patch.object(dsp, "lineage_missing", lambda a: None):
      for argv, want in (
          (TINY_ARGS + cf, "warning: --lineage-capacity-factor 5: not used without --batch-split"),
          (TINY_ARGS + native, "warning: --batch-split-impl native: not used without --batch-split"),
          (
              split + f32,
              "warning: --float32-gate-logits on: on Lineage's layers only the expert " "selection runs in fp32",
          ),
      ):
        with self.subTest(main=argv[len(TINY_ARGS) :]):
          out = io.StringIO()
          with contextlib.redirect_stdout(out):
            dsp.main(argv + ["--dry-run"])
          self.assertIn(want, out.getvalue())
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    with mock.patch.object(par, "build", side_effect=RuntimeError("built")):
      for argv, want in (
          (cf, "--lineage-capacity-factor 5: not used without --batch-split"),
          (cf2, "--lineage-capacity-factor 2: not used without --batch-split"),  # (given: told apart)
          (
              ["--batch-split", "--gbs", "16"] + native + cf,
              "--lineage-capacity-factor 5: not used by --batch-split-impl native",
          ),
          (native, "--batch-split-impl native: not used without --batch-split"),
      ):
        with self.subTest(parity=argv):
          out = io.StringIO()
          with contextlib.redirect_stdout(out), self.assertRaisesRegex(RuntimeError, "built"):
            par.main(["--preset", "tiny"] + argv)
          self.assertIn(f"warning: {want}", out.getvalue())

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_parity_ladder_sparsecore_switch(self):
    """ds_ep_parity --ladder on Lineage's layers: a rung pair whose chunks' stages' Lineage SparseCore answers differ
    -- circular:timed -> v:search on a slice whose stage 0 offloads nothing, as a TPU7x-8's, moves chunks 2 and 3
    across -- runs those chunks' SparseCore regions as TensorCore code on one side only: held to TOL_SC (the cosine
    and worst leaf ds_ep_lineage_test holds that switch to), not TOL_RUNG, as ladder hands compare_rungs the moved
    chunks (sparsecore_moves).  A leaf 1.05e-3 apart, the switch's measured worst, passes there and fails at
    TOL_RUNG.  The ladder runs on stand-in pipelines with their loss and gradients stubbed (nothing compiled but the
    weight gathers)."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    rng = np.random.default_rng(0)
    prev = [{"w": rng.standard_normal(256).astype(np.float32)} for _ in range(4)]
    noise = rng.standard_normal(256)
    got = [
        prev[0],
        prev[1],
        {"w": (prev[2]["w"] + 1.05e-3 * np.linalg.norm(prev[2]["w"]) * noise / np.linalg.norm(noise)).astype(np.float32)},
        prev[3],
    ]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      self.assertFalse(par.compare_rungs("rungs", 2.0, 2.0, got, prev, same=False, strict=False))
      self.assertTrue(par.compare_rungs("rungs", 2.0, 2.0, got, prev, same=False, strict=False, moved=[2, 3]))
      self.assertFalse(
          par.compare_rungs(
              "rungs",
              2.0,
              2.0,
              [prev[0], prev[1], {"w": 1.05 * prev[2]["w"]}, prev[3]],
              prev,
              same=False,
              strict=False,
              moved=[2, 3],
          )
      )  # 5e-2: outside it too
    self.assertIn("PASS (within the SparseCore switch's tolerance, TOL_SC: chunk(s) [2, 3]", out.getvalue())
    self.assertEqual(out.getvalue().count("FAIL"), 2)
    with lineage_stand_in():

      def build(preset, random_routing, **kw):
        p = pipe(
            lineage_split_scfg(),
            n_micro=2,
            global_batch=16,
            placement=kw["placement"],
            schedule=kw["schedule"],
            stage_lineage_sparsecore=(False, True),
        )
        p.init = lambda key: (p.params_from_chunks(host_weights(p)), None, None)
        return p

      grads = lambda p, params, wc0, batch: (2.0, [[par.to_host(t) for t in ps] for ps in params])
      with (
          mock.patch.object(dsp, "lineage_missing", lambda a: None),
          mock.patch.object(dsp, "lineage_off_tpu", lambda a, devices=None: None),
          mock.patch.object(par, "build", build),
          mock.patch.object(par, "mpmd_loss_and_grads", grads),
          mock.patch.object(par, "compare_rungs", wraps=par.compare_rungs) as cmp,
      ):
        for rungs, moved in (("circular:timed:none,v:search:none", [2, 3]), ("v:search:none,v:timed:none", [])):
          with self.subTest(rungs=rungs):
            cmp.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
              par.main(["--preset", "tiny", "--gbs", "16", "--batch-split", "--ladder", "--skip-port", "--rungs", rungs])
            self.assertEqual(e.exception.code, 0)
            self.assertEqual(cmp.call_count, 1)
            self.assertEqual(list(cmp.call_args.kwargs.get("moved", ())), moved)

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_refusals(self):
    """Pipeline refuses stash modes, the stash scheduler and prefetch pins under Lineage's batch split before it
    builds anything (Lineage's layers keep their own residuals and have no point to tie a gather to); a Lineage
    chunk refuses a stash policy, a pinned forward and a chunk without a MoE layer itself, naming Lineage."""
    with lineage_stand_in():
      for kw, want in (
          (dict(stash="lite", schedule="stash"), "stash 'lite' / schedule 'stash': a chunk on "),
          (dict(schedule="stash"), "schedule 'stash': "),
          (dict(schedule="interleaved", stash="lite"), "stash 'lite': "),
          (dict(prefetch_pin="hn=moe.wi_0", fsdp_scope="step"), "prefetch_pin 'hn=moe.wi_0': pins tie"),
          (dict(prefetch_pin="auto"), "prefetch_pin 'auto'"),
          (dict(modes={(0, 0): "lite"}), "modes: a chunk on "),
      ):
        with (
            self.subTest(**kw),
            self.assertRaisesRegex(
                ValueError, "batch_split on Lineage's layers .*cannot run " "this: " + re.escape(want)
            ),
        ):
          pipe(lineage_scfg(fsdp_collectives="xla"), n_micro=4, global_batch=32, **kw)
      p = pipe(lineage_scfg(fsdp_collectives="xla"), n_micro=2, global_batch=16, fsdp_scope="step")
      self.assertEqual(p.pins, {})
      with self.assertRaisesRegex(ValueError, "on Lineage's layers .*recompute programs only"):
        p.progs[0].build_stash("lite")
      with self.assertRaisesRegex(ValueError, "prefetch pins tie points of ds_ep's own layers .* Lineage's layers"):
        p.progs[0].part("F", pin=(("hn", 0, 0, 0),))
      sc = lineage_scfg(layers=5)
      sc = dataclasses.replace(sc, model=sc.model.replace(first_num_dense_layers=2))
      with self.assertRaises(NotImplementedError) as e:
        pipe(sc, n_micro=2, global_batch=16, layer_split=(2, 1, 1, 1))
      self.assertRegex(
          str(e.exception),
          r"^chunk 0: batch_split on Lineage's layers \(batch_split_impl 'lineage'\) "
          r"cannot run a chunk without a MoE layer \(",
      )
      self.assertNotIn("routed-bias", str(e.exception))  # it names what applies only
      # An F&B pair of the last chunk's fused program with a backward (a custom order's, under V placement: A&B on
      # the stage holding chunk 0 and the last) runs two of Lineage's backwards in one program: refused
      vee = sched.placement(2, 2, "v")
      streams = [
          [(0, "F", 0), (0, "F", 1), (3, "F", 0), ((3, "F", 1), (0, "B", 0)), (0, "B", 1)],
          [(1, "F", 0), (2, "F", 0), (1, "F", 1), (2, "F", 1), (2, "B", 0), (1, "B", 0), (2, "B", 1), (1, "B", 1)],
      ]
      order, fuse, _ = sched.merge(streams, vee, sched.COST, n_micro=2)
      with self.assertRaisesRegex(
          ValueError,
          r"F&B pairs of the last chunk's fused program with another chunk's "
          r"backward \(3F1&0B0\).* leave them out of `fuse`$",
      ):
        pipe(lineage_scfg(fsdp_collectives="xla"), n_micro=2, global_batch=16, placement="v", schedule=order, fuse=fuse)
      # interleaved's steady-state pairs never hold the last chunk's program: every stage pairs
      q = pipe(
          lineage_scfg(fsdp_collectives="xla"), n_micro=4, global_batch=32, schedule="interleaved", pair_stages="all"
      )
      self.assertEqual({q.place[q.order[j][0]] for j in q.fuse}, {0, 1})
      self.assertFalse([j for j in q.fuse if q.C - 1 in (q.order[j][0], q.order[j + 1][0])])
    # and compose, a global-view part beside a pinned one
    part = lambda name, **kw: ds_ep.Part(name, None, (), (), (), (), **kw)
    with self.assertRaisesRegex(ValueError, "prefetch pins tie points of ds_ep's own layers, not of a global-view"):
      ds_ep.compose(p.meshes[0], [part("fwd_c1", global_view=True), part("fwd_c0", pins=(("hn", 0, 0, 0),))], "x")

  def _compare(self, kw, scfg_kw=(), model_kw=()):
    """The PP=2 x V=2 pipeline `kw` with Lineage's batch split (the stand-in region) against the same pipeline
    unsplit, from one init and batch: one microbatch loop's reduced gradients, stage sumsqs and loss, and a training
    step's loss, gradient norm and params (`mode_run`) -> (the split pipeline, the stand-ins)."""
    sc = tiny_xla(shard_mode="explicit", random_routing=False, **dict(scfg_kw))
    if model_kw:
      sc = dataclasses.replace(sc, model=sc.model.replace(**dict(model_kw)))
    kw = dict(dict(n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=16, seq=256, overlap=True), **kw)
    with lineage_stand_in() as made:
      split = dsp.Pipeline(dataclasses.replace(sc, batch_split=True), **kw)
      self.assertTrue(all(pr.lineage for pr in split.progs))
      got = mode_run(split, 3, 5, steps=1)
    want = mode_run(dsp.Pipeline(sc, **kw), 3, 5, steps=1)
    worst = 0.0
    for (path, a), (path_b, b) in zip(got["grads"], want["grads"], strict=True):
      self.assertEqual(path, path_b)
      e = _rel(a, b)
      worst = max(worst, e)
      self.assertLess(e, 1e-5, path)
    self.assertAlmostEqual(got["loss"], want["loss"], delta=1e-6 * abs(want["loss"]))
    for k in ("sumsq", "losses", "gnorms"):
      np.testing.assert_allclose(got[k], want[k], rtol=1e-5, err_msg=k)
    # AdamW's first step moves each weight by about lr x g / (|g| + eps): a gradient element near eps carries its
    # fp32 rounding into its step at full size (measured: one of 196608 elements 2.5e-5 apart at lr 1e-3), so the
    # params are held by the RMS of their difference, 1e-3 of a step's (a lost microbatch moves it by ~lr)
    for i, (a, b) in enumerate(zip(got["params"], want["params"], strict=True)):
      rms = float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b))))
      self.assertLess(rms, 1e-3 * 1e-3, f"params leaf {i}")
    print(
        f'lineage stand-in vs unsplit ({kw.get("placement", "circular")}): loss {got["loss"]:.6f} vs '
        f'{want["loss"]:.6f}, {len(got["grads"])} gradient leaves, worst rel {worst:.3g}'
    )
    # each stage's Regions got its capacity factor and its SparseCore answer (no offload on CPU)
    sc = split.lineage_sparsecore()
    self.assertEqual(sc, dict.fromkeys(range(split.S), False))
    self.assertEqual(
        {(split.meshes.index(r.mesh), r.capacity_factor, r.sparsecore) for r in made.values()},
        {(s, split.scfg.lineage_capacity_factor, sc[s]) for s in range(split.S)},
    )
    return split, made

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_programs_match_unsplit(self):
    """The default plan (breadth, circular, weights per visit) on 8 CPU devices: chunk 0's first forward gathering
    its own routed experts (self_leaves), forwards / fused last chunks / backwards with another chunk's gathers
    ('G') and backwards with reduce-scatters ('R') -- mixed programs, the chunk's part at global view and the rest
    in shard_maps of their own -- chunk 0's embedding, the head, accumulation over the microbatches: with the
    stand-in region the step is the unsplit pipeline's to fp32 rounding.  Each chunk structure's region traces its
    two rules once for all its programs."""
    p, made = self._compare({})
    descs = [desc for _, desc in dict.fromkeys(p.dispatched())]
    shape = lambda d: tuple((k, tuple(sorted(dict(kw)))) for k, kw in d)
    seen = {shape(d) for d in descs}
    for want in (
        (("F", ("self_leaves",)),),
        (("F", ()), ("G", ("leaves", "other"))),
        (("A", ()), ("G", ("leaves", "other"))),
        (("A", ("acc",)), ("G", ("leaves", "other"))),
        (("B", ()), ("R", ("leaves", "other"))),
        (("B", ("acc",)), ("R", ("leaves", "other"))),
        (("B", ("acc",)), ("G", ("leaves", "other"))),
    ):
      self.assertIn(want, seen)
    for c, desc in dict.fromkeys(p.dispatched()):
      parts = [p.progs[c].part(k, **dict(kw)) for k, kw in desc]
      self.assertEqual([pt.global_view for pt in parts], [k in "FBA" for k, _ in desc], desc)
    self.assertEqual([dict(r.traces) for r in made.values()], [{"fwd": 1, "bwd": 1}] * 2)  # one a stage mesh
    # a mixed program: the region's rules (nested jits) and the gather's shard_map side by side at its top level
    c, desc = next((c, d) for c, d in dict.fromkeys(p.dispatched()) if [k for k, _ in d] == ["B", "R"])
    top = collections.Counter(
        e.primitive.name for e in p.progs[c].program(desc).trace(*p._desc_avals(c, desc)).jaxpr.eqns
    )
    self.assertGreaterEqual(top["shard_map"], 1)
    self.assertGreaterEqual(top["jit"] + top["pjit"], 2)
    # the fixed fused programs of a Lineage chunk are the composed ones
    pr, other = p.progs[1], p.chunks[2]
    self.assertIs(
        pr.bwd_reduce(other, accumulate=True), pr.program((ds_ep.pdesc("B", acc=True), ds_ep.pdesc("R", other=other)))
    )
    self.assertIs(
        pr.fwd_prefetch(other, (0, 1)), pr.program((ds_ep.pdesc("F"), ds_ep.pdesc("G", other=other, leaves=(0, 1))))
    )
    self.assertIs(
        p.progs[0].fwd_self_prefetch(p.routed_leaves),
        p.progs[0].program((ds_ep.pdesc("F", self_leaves=p.routed_leaves),)),
    )
    self.assertIs(pr.bwda_reduce_self(2), pr.program((ds_ep.pdesc("B", acc=True, rs_self=2),)))

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_programs_match_unsplit_v_dense(self):
    """V placement (the last backward of each stage folds its own reduce-scatter: rs_self), two dense layers in
    chunk 0 (the region runs them first) and the fused last chunk recomputed (remat_fused_last): the unsplit step
    to fp32 rounding."""
    p, _ = self._compare(
        dict(placement="v"),
        scfg_kw=dict(layers=6, remat_fused_last=True),
        model_kw=dict(first_num_dense_layers=2, mlp_dim=384),
    )
    self.assertTrue(p.rs_self)
    self.assertEqual(p.chunks[0].dense, 2)
    rs = [d for _, d in dict.fromkeys(p.dispatched()) if "rs_self" in dict(d[0][1])]
    self.assertTrue(rs)
    # the recompute is there (the values are the same without it): the fused last chunk's programs hold the region
    # under jax.checkpoint, the other chunks' do not
    with lineage_stand_in():
      for op in [o for o in p.ops() if o.desc is not None]:
        j = _program_jaxpr(p, op)
        n = _count_eqns(j, "remat2") + _count_eqns(j, "checkpoint")  # (jax 0.11 names it remat2 in a VJP)
        self.assertEqual(n > 0, op.c == p.C - 1, (op.c, op.desc, n))

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_programs_match_unsplit_pairs(self):
    """Interleaved 1F1B with F&B pairs on every stage: one program runs one chunk's forward and another's
    backward, two global-view parts side by side in one jit (no pair holds the last chunk's fused program:
    test_refusals) -- the unsplit step to fp32 rounding."""
    p, _ = self._compare(dict(schedule="interleaved", n_micro=4, global_batch=32, pair_stages="all"))
    self.assertTrue(p.fuse)
    pairs = [d for _, d in dict.fromkeys(p.dispatched()) if [k for k, _ in d][:2] in (["F", "B"], ["B", "F"])]
    self.assertTrue(pairs, list(dict.fromkeys(p.dispatched())))

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_legacy_loops(self):
    """The fixed programs of a Lineage chunk -- fwd / bwd / bwda / fb / fba, global-view jits of their own (the step
    loop composes parts instead), and fwd_prefetch / fwd_self_prefetch / bwd_reduce / bwda_reduce_self, the composed
    ones -- under the driver's two legacy loops (circular search plan, weights a step) give the step loop's loss,
    reduced gradients and stage sumsqs bit for bit: without the fusion plan (each op one part) and with it.  Without
    it the step is also the unsplit pipeline's to fp32 rounding (with it: _compare)."""
    kw = dict(n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=16, seq=256, schedule="search")
    # (unreduced accumulators never reach the host: reduced by the stages' own programs)
    red = lambda p, out: ([p.stages[s].reduce(out[0][s]) for s in range(p.S)], out[1])
    for overlap in (False, True):
      with self.subTest(overlap=overlap), lineage_stand_in():
        p = dsp.Pipeline(lineage_split_scfg(), overlap=overlap, **kw)
        params, _, wc0 = p.init(jax.random.key(2))
        batch = p.synthetic_batch(4)
        if overlap:
          got, want = (
              p.loss_and_grads_overlap(params, wc0, batch),
              legacy_loss_and_grads_overlap(p, params, wc0, batch),
          )
          self.assertTrue(p.rs_at and p.pf)  # the plan folds gathers and reduce-scatters
        else:
          got, want = red(p, p.loss_and_grads(params, batch)), red(p, legacy_loss_and_grads(p, params, batch))
        assert_bitwise(self, got, want, f"overlap={overlap}")
        print(
            f"lineage stand-in, legacy loops, overlap={overlap}: {len(jax.tree.leaves(got))} leaves bitwise, "
            f"loss {float(jnp.sum(got[-1])):.6f}"
        )
      if not overlap:
        up = dsp.Pipeline(tiny_xla(shard_mode="explicit", random_routing=False), overlap=False, **kw)
        ref = red(up, up.loss_and_grads(up.init(jax.random.key(2))[0], up.synthetic_batch(4)))  # the same init
        loss, ref_loss = float(jnp.sum(got[1])), float(jnp.sum(ref[1]))
        self.assertAlmostEqual(loss, ref_loss, delta=1e-6 * abs(ref_loss))
        for s in range(p.S):
          for a, b in zip(jax.tree.leaves(got[0][s][0]), jax.tree.leaves(ref[0][s][0]), strict=True):
            self.assertLess(_rel(a, b), 1e-5)
          np.testing.assert_allclose(float(got[0][s][1]), float(ref[0][s][1]), rtol=1e-5)

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_sparsecore_answers(self):
    """Each stage's answer for Lineage's SparseCore regions is the pipeline's, asked as the FSDP question is (over
    the whole stage, offload_answers: agreed across processes, the device rule where a probe fails) unless
    stage_lineage_sparsecore gives it: the summary reads it (probing nothing), each chunk's ChunkPrograms carries
    its stage's, and its Region gets it with the capacity factor when a program traces.  libtpu's answer is mocked
    here: stage 1 (devices 4-7) offloads."""
    stage1 = lambda mesh, axes=None: 4 in {d.id for d in mesh.devices.flat}
    with lineage_stand_in() as made:
      with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=stage1) as probe:
        p = pipe(lineage_split_scfg(lineage_capacity_factor=3.0), n_micro=2, global_batch=16)
        self.assertEqual(
            sorted((p.meshes.index(c.args[0]), c.args[1]) for c in probe.call_args_list), [(0, ds_ep.FE), (1, ds_ep.FE)]
        )
        self.assertEqual(p.lineage_sparsecore(), {0: False, 1: True})
        self.assertEqual([pr.lineage_sc for pr in p.progs], [False, True, False, True])
        self.assertIn(
            "EP collectives: Lineage's own (on SparseCore on stage [1]; as TensorCore code on stage [0], "
            "where libtpu offloads no collective)",
            p.summary(),
        )
        for c in (0, 1):
          desc = (ds_ep.pdesc("F"),)
          p.progs[c].program(desc).trace(*p._desc_avals(c, desc))
        self.assertEqual(probe.call_count, 2)  # neither the summary nor a program probes again
      self.assertEqual(
          sorted((p.meshes.index(r.mesh), r.capacity_factor, r.sparsecore) for r in made.values()),
          [(0, 3.0, False), (1, 3.0, True)],
      )
      with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=AssertionError("probed")):
        q = pipe(lineage_split_scfg(), n_micro=2, global_batch=16, stage_lineage_sparsecore=(True, False))
        self.assertEqual(q.lineage_sparsecore(), {0: True, 1: False})
        with self.assertRaisesRegex(ValueError, r"stage_lineage_sparsecore: a bool per stage \(2\)"):
          pipe(lineage_split_scfg(), n_micro=2, global_batch=16, stage_lineage_sparsecore=(True,))
      # a probe that fails answers by the device rule (CPU devices: no offload), saying why
      with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=RuntimeError("no probe here")):
        q = pipe(lineage_split_scfg(), n_micro=2, global_batch=16)
      self.assertEqual(q.lineage_sparsecore(), {0: False, 1: False})
      self.assertIn("its libtpu probe failed (RuntimeError: no probe here)", q.offloads[(1, ds_ep.FE)][1])
      # EP collectives 'auto' at EP 2: on Lineage's layers, which run their own, ds_ep's flavour is not asked
      # ('xla', unused), whatever libtpu would say
      with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=stage1) as probe:
        q = pipe(lineage_split_scfg(ep_collectives="auto"), n_micro=2, global_batch=16)
      self.assertEqual(sorted({c.args[1] for c in probe.call_args_list}), [ds_ep.FE])
      self.assertEqual([c.ep_collectives for c in q.stage_scfg], ["xla", "xla"])
    with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=stage1) as probe:
      r = pipe(tiny_xla(ep_collectives="auto"), n_micro=2, global_batch=16)  # unsplit: asked as ever
    self.assertIn(ds_ep.EP_AXIS, {c.args[1] for c in probe.call_args_list})
    self.assertEqual([c.ep_collectives for c in r.stage_scfg], ["ppermute", "xla"])
    # not on Lineage's layers: nothing asked or kept
    with mock.patch.object(ds_ep, "sparsecore_offload_available", side_effect=AssertionError):
      for sc in (tiny_xla(**NATIVE_SPLIT), tiny_xla()):
        r = pipe(sc, n_micro=2, global_batch=16)
        self.assertEqual((r.lineage_sparsecore(), {pr.lineage_sc for pr in r.progs}), ({}, {None}))

  @testing.needs_devices(8, "the Lineage batch split's stand-in pipelines: written for two stages of fsdp 2 x ep 2")
  def test_compiler_options(self):
    """Every program of a Lineage chunk compiles with both option sets it needs -- ds_ep's barrier-keeping one
    (pin_compiler_options: Lineage orders its phases with program_order's optimization barriers) and Lineage's
    (ds_ep_lineage.compiler_options: its splash dkv kernel's scoped VMEM) -- the dispatched ones, mixed (its part
    beside another chunk's gather or reduce-scatter) or not, and the fixed fwd / bwd / bwda / fb / fba.  Both are
    None off TPU, so mocked here."""
    keep, lin = {"xla_keep_barriers": "true"}, {"xla_keep_barriers": "true", "xla_scoped_vmem": "65536"}
    with (
        lineage_stand_in(),
        mock.patch.object(ds_ep, "pin_compiler_options", lambda mesh: dict(keep)),
        mock.patch.object(ds_ep_lineage, "compiler_options", lambda mesh: dict(lin)),
    ):
      p = pipe(lineage_split_scfg(), n_micro=2, global_batch=16)
      lowering = lambda prog, c, desc: prog.trace(*p._desc_avals(c, desc)).lower()._lowering
      if not hasattr(lowering(p.progs[0].fwd, 0, (ds_ep.pdesc("F"),)), "_compiler_options_kvs"):
        self.skipTest(f"jax {jax.__version__} keeps a lowering's compiler options elsewhere")
      progs = [(p.progs[c].program(desc), c, desc) for c, desc in dict.fromkeys(p.dispatched())]
      self.assertTrue([desc for _, _, desc in progs if len(desc) > 1])  # mixed ones among them
      for c, pr in enumerate(p.progs):
        fixed = (
            (("fb", "A", {}), ("fba", "A", dict(acc=True)))
            if pr.ch.last
            else (("fwd", "F", {}), ("bwd", "B", {}), ("bwda", "B", dict(acc=True)))
        )
        progs += [(getattr(pr, name), c, (ds_ep.pdesc(kind, **kw),)) for name, kind, kw in fixed]
      for prog, c, desc in progs:
        self.assertEqual(dict(lowering(prog, c, desc)._compiler_options_kvs), lin, (c, desc))


@testing.needs_devices(4)
class DenseLayers(unittest.TestCase):
  """--dense-layers (DeepSeekConfig.first_num_dense_layers): DeepSeek-V3's
  first decoder layers have a dense MLP of width --mlp in place of the
  experts.  They run first in chunk 0 (ds_ep.ChunkSpec.dense), stacked as
  their own leaves ('dense_layers') -- or one subtree each under layer_tree
  'per_layer' (ChunkSpec.dense_kinds) -- unsplit even under batch split."""

  def _scfg(self, **kw):
    scfg = tiny_xla(layers=6, **testing.stage_axes(), **kw)
    return dataclasses.replace(scfg, model=scfg.model.replace(first_num_dense_layers=2, mlp_dim=384))

  @pytest.mark.scheduled_only
  def test_matches_reference(self):
    """fp32 on XLA's ops: a step with two dense layers is the monolithic
    reference's to rounding, unsplit and batch split (top-k routing)."""
    for split in (False, True):
      with self.subTest(batch_split=split):
        p = dsp.Pipeline(
            self._scfg(random_routing=False, batch_split=split, batch_split_impl="native"),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=16,
            seq=256,
            overlap=True,
        )
        compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)

  @pytest.mark.scheduled_only
  def test_dense_only_chunk_zero(self):
    """A layer split's 0 for chunk 0 leaves it the embedding and the dense layers alone -- a lighter first stage,
    whose chunk-0 backward would otherwise pair with a later chunk's forward into the slowest program pair of the
    interleaved order (DeepSeek-V3 on 16 GB200 stages) -- and under batch split it runs the plain path (no MoE layer
    to split) beside the split chunks: the monolith's numbers."""
    p = dsp.Pipeline(
        self._scfg(random_routing=False, **NATIVE_SPLIT),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=2,
        global_batch=16,
        seq=256,
        overlap=True,
        layer_split=(0, 2, 1, 1),
    )
    self.assertEqual([(c.layers, c.dense) for c in p.chunks], [((), 2), ((0, 1), 0), ((2,), 0), ((3,), 0)])
    self.assertEqual([pr.split for pr in p.progs], [False, True, True, True])
    compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)

  @testing.slow
  def test_per_layer(self):
    """layer_tree 'per_layer' (the GPU default), each dense layer a subtree as each MoE layer is: the monolithic
    reference's numbers, unsplit, and batch split with a chunk 0 of dense layers alone; and against
    maxtext.experimental.mpmd.jax.deepseek,
    the independent port (every token taking all experts: ds_ep_parity's docstring), whose stacked tree ds_ep_parity
    maps the per-layer one to and from: the same loss and gradients but for fp32 reassociation, to
    test_route_weights_matches_independent_port's bounds (loss 1.5e-7, worst leaf 1.9e-6 measured)."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity as par

    for split, layer_split in ((False, None), (True, (0, 2, 1, 1))):
      with self.subTest(batch_split=split):
        p = dsp.Pipeline(
            self._scfg(random_routing=False, batch_split=split, batch_split_impl="native", layer_tree="per_layer"),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=16,
            seq=256,
            overlap=True,
            layer_split=layer_split,
        )
        self.assertEqual([c.dense_kinds for c in p.chunks], [("mla", "mla"), (), (), ()])
        compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)
    scfg = self._scfg(random_routing=False, layer_tree="per_layer")
    p = dsp.Pipeline(
        dataclasses.replace(scfg, model=scfg.model.replace(num_experts_per_tok=8)),
        n_stages=2,
        chunks_per_stage=2,
        n_micro=2,
        global_batch=8,
        seq=256,
    )
    params, _, wc0 = p.init(jax.random.key(0))
    tokens = np.random.default_rng(0).integers(0, scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
    targets = np.roll(tokens, -1, axis=1)
    loss, grads = par.mpmd_loss_and_grads(p, params, wc0, p.put_batch(tokens, targets))
    ref_loss, ref = par.port_loss_and_grads(p, [[par.to_host(t) for t in ps] for ps in params], tokens, targets)
    rows, cos, ratio = par.errors(p, grads, ref)
    self.assertLess(abs(loss - ref_loss) / ref_loss, 1e-6)
    self.assertLess(rows[0][0], 2e-5, rows[:3])
    self.assertGreater(cos, 1 - 1e-9)

  def test_chunks_and_leaves(self):
    """Chunk 0 holds the dense layers (MoE layer ids stay the routing keys'),
    their leaves are named 'dense.*' and gathered like the shared expert's;
    a stash mode (up front, in Pipeline) and a chunk without an MoE layer are
    refused -- but under a layer split that counts every layer, where the
    dense layers may span the first chunks (exact_dsv3_test), a later chunk's
    dense layers build."""
    p = pipe(self._scfg(), n_micro=2, global_batch=16)
    self.assertEqual([(c.layers, c.dense) for c in p.chunks], [((0,), 2), ((1,), 0), ((2,), 0), ((3,), 0)])
    names = ds_ep.leaf_names(p.chunks[0])
    self.assertIn("dense.mlp.wi_0", names)
    self.assertNotIn("dense.mlp.wi_0", ds_ep.leaf_names(p.chunks[1]))
    self.assertEqual(len(names), len(set(names)))
    shapes = ds_ep.chunk_shapes(p.scfg, p.chunks[0])["dense_layers"]
    self.assertEqual(shapes["mlp"]["wi_0"]["kernel"], (2, 256, 384))
    self.assertEqual(
        ds_ep.chunk_storage_specs(p.chunks[0])["dense_layers"]["mlp"]["wo"]["kernel"], P(None, None, ds_ep.FE)
    )
    self.assertEqual(p.progs[0].n_res, 2 + 1)  # the embedding's and each dense layer's output
    with self.assertRaisesRegex(ValueError, "holds dense layers"):
      p.progs[0].build_stash("lite")
    later = ds_ep.ChunkPrograms(
        p.scfg, dataclasses.replace(p.chunks[1], dense=1), p.meshes[1], p.mbs, p.seq, float(p.gbs * p.seq)
    )
    self.assertEqual((later.nd, later.n_res), (1, 1))  # the MoE layer's input: its dense layer's output
    for kw in (
        dict(schedule="stash", stash="lite"),
        dict(schedule="interleaved", stash="lite", n_micro=4),
    ):
      with self.subTest(**kw), self.assertRaisesRegex(ValueError, "no stash policy with dense layers"):
        pipe(self._scfg(), **{"n_micro": 2, "global_batch": 16, **kw})
    scfg = self._scfg()
    for layers, dense in ((2, 2), (6, -1)):  # every layer dense (0 MoE layers a chunk); a negative count
      bad = dataclasses.replace(scfg, model=scfg.model.replace(num_decoder_layers=layers, first_num_dense_layers=dense))
      with self.subTest(layers=layers, dense=dense), self.assertRaisesRegex(ValueError, ">= 1 MoE layer each"):
        pipe(bad, n_micro=2, global_batch=16)

  def test_flag(self):
    """--dense-layers / --mlp reach the model through `build`; the summary
    names them, and MaxText's FLOP accounting counts their MLP."""
    ap = dsp.make_parser()
    p = dsp.build(
        ap.parse_args(TINY_ARGS + ["--layers", "4", "--dense-layers", "2", "--mlp", "384", "--vv", "1", "--gbs", "8"])
    )
    self.assertEqual((p.scfg.model.first_num_dense_layers, p.scfg.model.mlp_dim), (2, 384))
    self.assertEqual([c.dense for c in p.chunks], [2, 0])
    self.assertIn("2 MoE layers after 2 dense (mlp 384, in chunk 0)", p.summary())
    w_dense, _ = dsp.maxtext_training_flops(p.scfg.model, 1, 256)
    w_moe, _ = dsp.maxtext_training_flops(p.scfg.model.replace(first_num_dense_layers=0), 1, 256)
    self.assertNotEqual(w_dense, w_moe)


@testing.needs_devices(4)
class ExpertWeights(unittest.TestCase):
  """--expert-weights (StageConfig.expert_weights): the routed experts'
  grouped matmuls read a layer's weights in place in the chunk's stacked
  leaves ('stacked', the default) or from a per-layer slice ('sliced')."""

  def _calls(self, scfg, **kw):
    """Every named_gmm call of the forward, backward and fused last-chunk
    programs as they trace: (program, name, (stacked weight's shape, layer)
    or None)."""
    calls, real = [], ds_ep.named_gmm

    def spy(name, *a, stacked=None, **k):
      calls.append((prog, name, None if stacked is None else (tuple(stacked[0].shape), stacked[1])))
      return real(name, *a, stacked=stacked, **k)

    p = pipe(scfg, n_micro=2, global_batch=16, fsdp_scope="step", **kw)
    out = {}
    with mock.patch.object(ds_ep, "named_gmm", spy):
      for prog, c, kind in (("F", 1, "F"), ("B", 1, "B"), ("A", 3, "F")):
        op = next(o for o in p.ops() if (o.c, o.kind) == (c, kind))
        _program_jaxpr(p, op)
    for prog, name, st in calls:
      out.setdefault(prog, []).append((name, st))
    return p, out

  @testing.needs_devices(4)
  def test_programs_read_stacked(self):
    """Under 'stacked' every expert grouped matmul of every program, split or
    not, gets its layer's block of the chunk's stacked leaf -- each layer's
    three weights at their layer index -- and under 'sliced' none does."""
    for split in (False, True):
      scfg = tiny_xla(layers=8, batch_split=split, batch_split_impl="native", **testing.stage_axes())
      p, out = self._calls(scfg)
      cfg, n = scfg.model, len(p.chunks[1].layers)
      el = cfg.num_experts // scfg.ep
      want = {
          "gmm_wi_0": (n, el, cfg.emb_dim, cfg.moe_mlp_dim),
          "gmm_wi_1": (n, el, cfg.emb_dim, cfg.moe_mlp_dim),
          "gmm_wo": (n, el, cfg.moe_mlp_dim, cfg.emb_dim),
      }
      for prog, calls in out.items():
        with self.subTest(split=split, program=prog):
          self.assertTrue(calls)
          self.assertEqual(
              sorted({(name, st[1]) for name, st in calls}), sorted((name, l) for name in want for l in range(n))
          )
          for name, st in calls:
            self.assertEqual(st[0], want[name])
      _, out = self._calls(dataclasses.replace(scfg, expert_weights="sliced"))
      self.assertTrue(all(st is None for calls in out.values() for _, st in calls), split)

  def test_flag(self):
    """--expert-weights reaches the stage config through `build`; the summary
    names 'sliced'."""
    ap = dsp.make_parser()
    p = dsp.build(ap.parse_args(TINY_ARGS))
    self.assertEqual(p.scfg.expert_weights, "stacked")
    self.assertNotIn("expert_weights", p.summary())
    p = dsp.build(ap.parse_args(TINY_ARGS + ["--expert-weights", "sliced"]))
    self.assertEqual(p.scfg.expert_weights, "sliced")
    self.assertIn("expert_weights=sliced", p.summary())
    with self.assertRaisesRegex(ValueError, "expert_weights must be one of"):
      tiny_xla(expert_weights="views")

  @pytest.mark.scheduled_only
  @testing.needs_devices(8, "two stages of fsdp 2 x ep 2")
  @testing.needs_ynn_off(
      "on jaxlib 0.12.0.1 with them on both unsplit runs' gradients come out wrong, up to 1e4x the values with them "
      "off (the losses agree), and the two part in 57 of 67 leaves: 3 of 3 runs (on 0.12.1 the two part too, 2 of "
      "2); with them off, and on jaxlib 0.11.1 either way, the two agree bit for bit",
      jaxlib_before=testing.YNN_BUG_JAXLIBS_BEFORE,
  )
  def test_stacked_matches_sliced_bitwise(self):
    """MaxText's kernels in bf16, unsplit and split: a step reading the
    stacked leaves is the sliced step's, loss and every gradient, bit for
    bit."""
    for split in (False, True):
      out = []
      for ew in EXPERT_WEIGHT_MODES:
        p = dsp.Pipeline(
            tiny(layers=4, batch_split=split, batch_split_impl="native", expert_weights=ew),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=16,
            seq=256,
            init_by_chunk=True,
        )
        grads, _, lacc = run_pipe(p, 5, 6)
        out.append((p.chunk_trees(grads), float(jnp.sum(lacc))))
      with self.subTest(split=split):
        assert_bitwise(self, out[0][0], out[1][0], "gradients by chunk")
        self.assertEqual(out[0][1], out[1][1])


EXPERT_WEIGHT_MODES = ("stacked", "sliced")


if __name__ == "__main__":
  unittest.main()
