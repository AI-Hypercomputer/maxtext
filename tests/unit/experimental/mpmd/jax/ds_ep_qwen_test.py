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

"""CPU checks of the MPMD pipeline's Qwen3.5-family layers (Qwen3.8-2.4T-A95B, Qwen3.5-397B-A17B).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_qwen_test.py -q

`ds_ep` builds the Qwen3.5 family's layers when `StageConfig.model` is a
`maxtext.experimental.mpmd.jax.qwen35.config.Qwen35Config`
(MaxText's `decoder_block: qwen3_5`: three Gated DeltaNet layers then one gated full-attention layer, every layer
MoE with a sigmoid-gated shared expert).  What `ds_ep_test` holds for DeepSeek-V3, this file holds for them:

  * the chunk trees, specs and layer kinds, the refusals, the `--model` flags and MaxText's FLOPs -- nothing
    compiled;
  * one chunk's programs against `maxtext.experimental.mpmd.jax.qwen35.model.decoder_layer`, and the MoE half against
    `maxtext.experimental.mpmd.jax.qwen35.moe.sparse_moe_block` (the port, MaxText's numerics), forward and backward;
  * the whole pipeline against `ds_ep_test.reference` (the monolith) and against
    `maxtext.experimental.mpmd.jax.qwen35.train`'s loss
    (the port), in fp32 on XLA's ops -- `ragged_dot` experts, the argsort dispatch -- at rounding-level bounds;
    the repeat scope bit for bit against the step scope;
  * batch split (QwenBatchSplit): the GDN or attention mixer in the phase where DeepSeek's attention runs, against
    the unsplit monolith (top-k) and the halved one (random routing), under breadth and interleaved 1F1B's F&B
    pairs;
  * a chunk's backward cut into the zero-bubble schedules' I and W against the backward it is cut from, bit for bit
    (QwenSplitBackward, ds_ep_test.check_split), and zbh1 / zbh2 against 1F1B, bit for bit (QwenZeroBubble,
    ds_ep_test.check_zero_bubble);
  * StagePrograms.init's Qwen rules (and the port's copy of them, `qwen35.init`), the gathered compute copies' dtypes
    (fp32 where MaxText reads a stored value in fp32), and the CLI;
  * the benchmark's kernels in bf16 -- megablox, the ragged sort, tokamax splash -- against the monolith and the
    port, batch split too, at bounds calibrated from the port's own bf16 floor (QwenMaxTextKernels), and the ragged
    sort's dx against MaxText's direct gather, bit for bit, what the refusals let through at ep 1
    (QwenDispatchVsMaxText).

The port shares its Gated DeltaNet, gated attention and norms with ds_ep's layer, as
`maxtext.experimental.mpmd.jax.deepseek` shares
MLA with DeepSeek's, so a bug in those would cancel here: `qwen35_model_test` (against a float64 recurrence) and
`qwen35_maxtext_parity_test` (against MaxText's qwen3_5) hold them.  Against the port every token takes all
experts (top-k = E), for the reason ds_ep_parity's docstring gives: top-k with k < E flips on last-bit
differences.  The MoE half alone (QwenMoeVsPort) is where top-k < E is checked, at router logits with no ties.
"""

import argparse
import collections
import contextlib
import dataclasses
import io
import math
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

try:
  import yaml
except ImportError:  # the MaxText yml check skips without it
  yaml = None
from jax.sharding import NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import maxtext_parity as mp
from maxtext.experimental.mpmd.jax import testing
# Functions and constants only: a TestCase imported here would be collected, and run, a second time.
from tests.unit.experimental.mpmd.jax.ds_ep_test import (
    HAVE_TOKAMAX,
    TILE,
    XLA_KERNELS,
    _count_eqns,
    _program_jaxpr,
    assert_bitwise,
    assert_same_run,
    check_repeat_plan,
    check_split,
    check_split_last,
    check_zero_bubble,
    host_weights,
    mode_run,
    pipe,
    reference,
    tiny,
    zb_ep,
)
from maxtext.experimental.mpmd.jax.qwen35 import _ops as q_ops
from maxtext.experimental.mpmd.jax.qwen35 import gdn as q_gdn
from maxtext.experimental.mpmd.jax.qwen35 import layers as q_layers
from maxtext.experimental.mpmd.jax.qwen35 import model as qm
from maxtext.experimental.mpmd.jax.qwen35 import moe as q_moe
from maxtext.experimental.mpmd.jax.qwen35 import train as qt
from maxtext.experimental.mpmd.jax.qwen35.config import Qwen35Config, qwen35_35b


def setUpModule():
  testing.require_cpu()


# ds_ep_test.tiny()'s widths for the Qwen3.5 family: vocab 512, emb 256, MoE width 256, 8 experts, top-2, seq 256;
# gated attention of 4 query / 2 kv heads of 64 (GQA 2:1; rotary dim 16, so mRoPE (3, 3, 2)); a Gated DeltaNet of
# 2 key / 16 value heads of 32 -- 8 value heads a key head, as Qwen3.8-2.4T has (Qwen3.5-397B: 4) -- in chunks of
# 64 tokens.
Q_TINY = dict(
    vocab_size=512,
    emb_dim=256,
    num_query_heads=4,
    num_kv_heads=2,
    head_dim=64,
    mrope_section=(3, 3, 2),
    gdn_num_key_heads=2,
    gdn_num_value_heads=16,
    gdn_key_head_dim=32,
    gdn_value_head_dim=32,
    gdn_chunk_size=64,
    moe_mlp_dim=256,
    num_experts=8,
    num_experts_per_tok=2,
    shared_experts=1,
    max_target_length=256,
    attention_kernel="tokamax_splash",
    sa_block_q=128,
    sa_block_kv=128,
    sa_block_kv_compute=128,
)

# MaxText's configs/models/qwen3.8-2.4t-a95b.yml (decoder_block qwen3_5) as Qwen35Config fields, the yml's
# numbers: what `--model qwen3.8-2.4t-a95b` must give, and what the parameter and FLOP counts below are of.  The
# yml sets no use_mrope: base.yml's false, the partial RoPE.
QWEN38 = dict(
    vocab_size=248320,
    emb_dim=8192,
    num_decoder_layers=92,
    num_query_heads=64,
    num_kv_heads=4,
    head_dim=256,
    gdn_num_key_heads=16,
    gdn_num_value_heads=128,
    gdn_key_head_dim=128,
    gdn_value_head_dim=128,
    gdn_conv_kernel_dim=4,
    gdn_chunk_size=64,
    moe_mlp_dim=2048,
    num_experts=512,
    num_experts_per_tok=10,
    shared_experts=1,
    norm_topk_prob=True,
    inhomogeneous_layer_cycle_interval=4,
    partial_rotary_factor=0.25,
    rope_max_timescale=10_000_000,
    normalization_layer_epsilon=1e-6,
    use_mrope=False,
)

# Q_TINY's dims on the command line, on XLA's ops: --model qwen3.8-2.4t-a95b with every dim given (it brings its
# partial RoPE, where Q_TINY takes Qwen35Config's default text mRoPE), two stages of fsdp 1 x ep 2, the 4 layers cut
# inside their block (1 + 3: the head's chunk holds two GDN layers and the attention one).
Q_TINY_ARGS = (
    "--model qwen3.8-2.4t-a95b --layers 4 --emb 256 --heads 4 --kv-heads 2 --head-dim 64 "
    "--gdn-key-heads 2 --gdn-value-heads 16 --gdn-key-head-dim 32 --gdn-value-head-dim 32 --moe-mlp 256 "
    "--experts 8 --topk 2 --vocab 512 --seq 256 --sa-block 128 --sa-block-kv-compute 128 "
    "--gmm-impl ragged_dot --ragged-sort off --stages 2 --vv 1 --layer-split 1,3 --fsdp 1 --ep 2 "
    "--gbs 4 --mb 2"
).split()

# fp32 on XLA's ops, the pipeline against the monolith and the port: ds_ep_test.PipelineXlaKernels's bounds
# (loss 1e-5, every gradient leaf 1e-4 L2-rel, the gradient's sum of squares 1e-4) -- but for the Gated DeltaNet's
# decay parameters, A_log and dt_bias: one value a value head, each gradient a sum over every token with heavy
# cancellation, they agree to only ~1e-4 in fp32 (the (1, 3, 3, 1) split against the monolith over six seeds:
# median 3e-5, worst 1.2e-4; every other leaf to 4e-6), and are held to 5e-4.
DECAY_LEAVES = ("['A_log']", "['dt_bias']")
FP32_LEAF, FP32_DECAY = 1e-4, 5e-4


def qtiny(layers=4, **kw):
  """ds_ep_test.tiny() for the Qwen3.5 family (Q_TINY): bf16 compute, megablox experts and the ragged sort,
  tokamax splash attention, the reduce-scatters in fp32."""
  base = dict(
      model=Qwen35Config(num_decoder_layers=layers, **Q_TINY), wi_tiling=TILE, wo_tiling=TILE, rs_dtype=jnp.float32
  )
  base.update(kw)
  return ds_ep.StageConfig(**base)


def qtiny_xla(layers=4, **kw):
  """qtiny() on XLA's ops instead of MaxText's kernels (ds_ep_test.tiny_xla): `ragged_dot` experts and the argsort
  dispatch, fp32 compute, tokamax splash where tokamax is installed (else the reference attention)."""
  scfg = qtiny(layers, **{**XLA_KERNELS, **kw})
  return dataclasses.replace(
      scfg,
      model=scfg.model.replace(dtype=jnp.float32, attention_kernel="tokamax_splash" if HAVE_TOKAMAX else "reference"),
  )


def all_experts(scfg):
  """`scfg` routing every token to every expert (top-k = E, no random routing): the port checks' routing."""
  return dataclasses.replace(
      scfg, random_routing=False, model=scfg.model.replace(num_experts_per_tok=scfg.model.num_experts)
  )


def is_attention(cfg, layer):
  """Global layer `layer` is gated full attention: the last of each cycle of layers, written out here rather
  than taken from Qwen35Config or ds_ep."""
  return layer % cfg.inhomogeneous_layer_cycle_interval == cfg.inhomogeneous_layer_cycle_interval - 1


def leaf_name(path):
  return str(getattr(path[-1], "key", path[-1]))


def perturb_norms(tree, seed, scale=0.1):
  """Every norm scale of `tree` (`ds_ep` arrays or host ones) moved off its init by scale x N(0, 1): a zero-centred
  norm at init applies 1, which is also what the plain norm in its place would apply at a scale of 1, and a
  dropped `+ 1` would apply 0 -- off init each of those differs."""
  flat, tdef = jax.tree_util.tree_flatten_with_path(tree)
  keys = jax.random.split(jax.random.key(seed), len(flat))
  return jax.tree.unflatten(
      tdef,
      [
          x + scale * jax.random.normal(k, x.shape, x.dtype) if leaf_name(path) == "scale" else x
          for k, (path, x) in zip(keys, flat)
      ],
  )


def max_rel(a, b):
  """max |a - b| / max |b|."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def l2_rel(a, b):
  """|a - b| / |b|, in the L2 norm."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def port_layer(lw):
  """One layer's weights as `ds_ep.layer_weights` gives them -> as
  `maxtext.experimental.mpmd.jax.qwen35.model.decoder_layer` takes
  them (the same leaves under the port's names: the mixer under 'attention')."""
  return {
      "input_layernorm": lw["input_layernorm"],
      "post_attention_layernorm": lw["post_attention_layernorm"],
      "mlp": lw["mlp"],
      "attention": {"attention": lw["attn"]} if "attn" in lw else lw["gdn"],
  }


def port_tree(p, trees):
  """Chunk trees (chunk order; weights or gradients) -> the `maxtext.experimental.mpmd.jax.qwen35` parameter tree:
  global layer l is
  position l % cycle of block l // cycle, `decoder/layers/layer_{i}` stacked over the blocks (`qm.param_avals`)."""
  cfg = p.scfg.model
  cyc = cfg.inhomogeneous_layer_cycle_interval
  per = {
      lid: port_layer(ds_ep.layer_weights(t["layers"], l, ch.kinds))
      for ch, t in zip(p.chunks, trees)
      for l, lid in enumerate(ch.layers)
  }
  layers = {
      f"layer_{i}": jax.tree.map(lambda *a: np.stack(a), *[per[b * cyc + i] for b in range(cfg.num_blocks)])
      for i in range(cyc)
  }
  return {
      "token_embedder": trees[0]["token_embedder"],
      "decoder": {"layers": layers, "decoder_norm": trees[-1]["decoder_norm"], "logits_dense": trees[-1]["logits_dense"]},
  }


def port_cfg(cfg):
  """The port's config for a pipeline's model: the fp32 reference attention (independent of tokamax splash),
  MaxText's sparse_matmul experts (`moe_impl='ragged'`, `ragged_dot` with fp32 accumulation), and the pipeline's
  Gated DeltaNet: `gdn_packing=False`, MaxText's packing=false, under which the all-one segment ids of `port_batch`
  only zero padding, as the pipeline's GDN takes no segment ids (`ds_ep.qwen_mixer_part`;
  test_port_runs_the_pipelines_gdn).  Under gdn_packing they would run the packed GDN, the same math in other
  roundings (in bf16 its conv keeps the fp32 kernel): an offset from the pipeline in every comparison, as
  maxtext_parity_test runs MaxText with packing=false for the same reason."""
  return cfg.replace(attention_kernel="reference", moe_impl="ragged", gmm_fp32_accum=True, gdn_packing=False)


def port_batch(tokens, targets):
  """The batch `maxtext.experimental.mpmd.jax.qwen35.train.compute_loss` takes for `tokens` / `targets` `[B, L]`:
  positions 0..L-1, every segment id one (a document a row)."""
  B, L = tokens.shape
  return {
      "inputs": tokens,
      "targets": targets,
      "inputs_position": np.broadcast_to(np.arange(L, dtype=np.int32), (B, L)).copy(),
      "inputs_segmentation": np.ones((B, L), np.int32),
      "targets_segmentation": np.ones((B, L), np.int32),
  }


def port_loss_and_grads(cfg, params, tokens, targets):
  """`maxtext.experimental.mpmd.jax.qwen35.train.compute_loss` (MaxText's masked-mean cross entropy) and its
  gradients, on one device
  -> (loss, gradient tree)."""
  dev = jax.devices()[0]
  f = jax.jit(jax.value_and_grad(lambda pr, b: qt.compute_loss(pr, b, cfg)[0]))
  loss, g = f(jax.device_put(params, dev), jax.device_put(port_batch(tokens, targets), dev))
  return float(loss), jax.tree.map(np.asarray, g)


def tree_errors(got, want):
  """Two trees of one structure -> ([(L2-rel error, path)] worst first, whole-tree cosine, norm ratio)."""
  rows, dot, na, nb = [], 0.0, 0.0, 0.0
  for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(got)[0], jax.tree.leaves(want), strict=True):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (jax.tree_util.keystr(path), a.shape, b.shape)
    rows.append((l2_rel(a, b), jax.tree_util.keystr(path)))
    dot, na, nb = dot + float(np.vdot(a, b)), na + float(np.vdot(a, a)), nb + float(np.vdot(b, b))
  return sorted(rows, reverse=True), dot / math.sqrt(na * nb), math.sqrt(na / nb)


def stage_chunk(scfg, layers, mbs=4, seq=128):
  """`layers` as one middle chunk on a one-stage mesh of scfg's fsdp x ep -> (ChunkSpec, StagePrograms,
  ChunkPrograms)."""
  mesh = dsp.stage_meshes(1, scfg.fsdp, scfg.ep)[0]
  layers = tuple(layers)
  ch = ds_ep.ChunkSpec(
      index=1, layers=layers, first=False, last=False, stage=0, kinds=ds_ep.layer_kinds(scfg.model, layers)
  )
  return ch, ds_ep.StagePrograms(scfg, [ch], mesh), ds_ep.ChunkPrograms(scfg, ch, mesh, mbs, seq, mbs * seq)


def act(mesh, x):
  """An activation [b, S, D] laid out as the chunk programs take it: rows over ('fsdp', 'expert')."""
  return jax.device_put(x, NamedSharding(mesh, P(ds_ep.FE, None, None)))


def summed_gacc(gacc):
  """A chunk program's per-device partial gradients (a leading axis over the axes still to reduce) -> the
  gradient: their sum."""
  return jax.tree.map(lambda a: np.asarray(a, np.float64).sum(0), gacc)


def leaf_bound(path):
  """FP32_DECAY for a decay parameter's leaf (A_log, dt_bias), FP32_LEAF for any other."""
  return FP32_DECAY if path.endswith(DECAY_LEAVES) else FP32_LEAF


def against_monolith(t, p, overlap, reference_of=None):
  """`ds_ep_test.compare_with_reference` at the fp32 bounds, the decay leaves at theirs (`leaf_bound`): one
  loss-and-gradients pass of `p` from init(key 1) on synthetic_batch(3) -- with the fusion plan, or the plain
  loop and the stages' reduce programs -- against `reference` (of `reference_of`, default `p`): the loss to
  1e-5, every gradient leaf (L2-rel), the stage sumsqs' total to 1e-4 of the gradient's."""
  params, _, wc0 = p.init(jax.random.key(1))
  batch = p.synthetic_batch(3)
  if overlap:
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, batch)
  else:
    gaccs, lacc = p.loss_and_grads(params, batch)
    grads, sqs = zip(*(p.stages[s].reduce(gaccs[s]) for s in range(p.S)))
  loss = float(jnp.sum(lacc))
  ref_loss, ref = reference(reference_of or p, params, batch)
  ref = [jax.tree.map(np.asarray, r) for r in ref]
  rows, _, _ = tree_errors([jax.tree.map(np.asarray, c) for c in p.chunk_trees(list(grads))], ref)
  ref_sq = sum(float(np.sum(np.square(x.astype(np.float64)))) for x in jax.tree.leaves(ref))
  sq_err = abs(sum(float(x) for x in sqs) - ref_sq) / ref_sq
  decay = [r for r in rows if r[1].endswith(DECAY_LEAVES)]
  rest = [r for r in rows if not r[1].endswith(DECAY_LEAVES)]
  print(
      f"loss {loss:.6f} ref {float(ref_loss):.6f}; worst gradient leaf {rest[0][0]:.3g} at chunk {rest[0][1]}, "
      f"of the decay leaves {decay[0][0]:.3g} at chunk {decay[0][1]}; sumsq rel {sq_err:.2g}"
  )
  t.assertLess(abs(loss - float(ref_loss)), 1e-5 * abs(float(ref_loss)))
  for e, path in rows:
    t.assertLess(e, leaf_bound(path), f"chunk {path}: rel err {e:.3g}")
  t.assertLess(sq_err, 1e-4)


# ------------------------------------------------------------ trees, specs, kinds
@testing.needs_devices(4)
class QwenTree(unittest.TestCase):
  """A Qwen chunk's 'layers' tree: one subtree a layer, keyed by its place in the chunk (`qwen_layer_key`), with the
  leaves every layer has and its own mixer's, each with a leading layer axis of one.  Nothing compiles here."""

  def test_shapes_and_specs_for_every_kind_mix(self):
    """Every run of 1 to 5 layers of a 12-layer model -- all Gated DeltaNet, GDN and attention, attention alone or
    first, across a block boundary -- as a first, middle, last and only chunk: its shapes, storage specs and
    compute specs are one tree with a spec of each leaf's rank; each layer's subtree, in the chunk's order, holds
    the common leaves (two norms, router, routed experts, shared expert and its gate) and its own mixer ('gdn' /
    'attn'), every leaf with a layer axis of one; and every dim a storage spec shards divides by its mesh axes
    (at Qwen3.8's dims, over one tpu7x-4x4x4 slice too)."""
    scfg = qtiny(layers=12)
    cfg = scfg.model
    common = {"input_layernorm", "post_attention_layernorm", "mlp"}
    mixes = set()
    for start in range(12):
      for n in range(1, 6):
        layers = tuple(range(start, min(start + n, 12)))
        kinds = ds_ep.layer_kinds(cfg, layers)
        mixes.add(tuple(sorted(set(kinds))) + (kinds[0],))
        for first, last in ((True, False), (False, False), (False, True), (True, True)):
          with self.subTest(layers=layers, first=first, last=last):
            ch = ds_ep.ChunkSpec(index=0, layers=layers, first=first, last=last, stage=0, kinds=kinds)
            self.check_chunk(scfg, ch, {"fsdp": 2, "expert": 2})
            lay = ds_ep.chunk_shapes(scfg, ch)["layers"]
            self.assertEqual(sorted(lay), [ds_ep.qwen_layer_key(l) for l in range(len(layers))])
            for l, kind in enumerate(kinds):
              sub = lay[ds_ep.qwen_layer_key(l)]
              self.assertEqual(set(sub), common | {kind})
              self.assertEqual({s[0] for s in jax.tree.leaves(sub, is_leaf=ds_ep._is_shape)}, {1})
    # all GDN, GDN then attention, attention alone or first
    self.assertEqual(mixes, {("gdn", "gdn"), ("attn", "gdn", "gdn"), ("attn", "attn"), ("attn", "gdn", "attn")})
    big = ds_ep.StageConfig(model=Qwen35Config(**QWEN38), fsdp=16, ep=8, fsdp_collectives="xla")
    for layers in ((0, 1, 2, 3), (3,)):
      ch = ds_ep.ChunkSpec(
          index=0, layers=layers, first=True, last=True, stage=0, kinds=ds_ep.layer_kinds(big.model, layers)
      )
      self.check_chunk(big, ch, {"fsdp": 16, "expert": 8})

  def check_chunk(self, scfg, ch, sizes):
    shapes = ds_ep.chunk_shapes(scfg, ch)
    st, cs = ds_ep.chunk_storage_specs(ch), ds_ep.chunk_compute_specs(ch)
    tdef = jax.tree.structure(shapes, is_leaf=ds_ep._is_shape)
    self.assertEqual(jax.tree.structure(st, is_leaf=ds_ep._is_spec), tdef)
    self.assertEqual(jax.tree.structure(cs, is_leaf=ds_ep._is_spec), tdef)
    for (path, shape), s, c in zip(
        jax.tree_util.tree_flatten_with_path(shapes, is_leaf=ds_ep._is_shape)[0],
        jax.tree.leaves(st, is_leaf=ds_ep._is_spec),
        jax.tree.leaves(cs, is_leaf=ds_ep._is_spec),
    ):
      where = jax.tree_util.keystr(path)
      self.assertEqual((len(s), len(c)), (len(shape), len(shape)), where)
      for d, ax in zip(shape, s):
        axes = () if ax is None else (ax,) if isinstance(ax, str) else ax
        self.assertEqual(d % math.prod(sizes[a] for a in axes), 0, (where, shape, s))

  def test_kinds_follow_the_global_layer_id(self):
    """A chunk's kinds are its GLOBAL layer ids' -- attention on the last layer of every cycle of the model,
    wherever the chunk starts -- for even splits and splits that cut inside a block, on circular and V
    placements, and for a 2-layer cycle too; the chunks cover the layers in order; DeepSeek's have no kinds."""
    cases = [
        (8, 2, "circular", None, 4),
        (8, 2, "circular", (3, 1, 3, 1), 4),
        (8, 2, "circular", (1, 3, 3, 1), 4),
        (8, 2, "v", (1, 2, 2, 3), 4),
        (8, 1, "circular", (5, 3), 4),
        (12, 2, "v", (5, 1, 3, 3), 4),
        (4, 1, "circular", (1, 3), 4),
        (8, 2, "circular", (3, 1, 1, 3), 2),
    ]
    for layers, vv, placement, split, cyc in cases:
      with self.subTest(layers=layers, placement=placement, split=split, cycle=cyc):
        scfg = qtiny(layers=layers, fsdp=1, ep=2)
        scfg = dataclasses.replace(scfg, model=scfg.model.replace(inhomogeneous_layer_cycle_interval=cyc))
        p = pipe(scfg, chunks_per_stage=vv, placement=placement, layer_split=split)
        self.assertEqual([l for ch in p.chunks for l in ch.layers], list(range(layers)))
        for ch in p.chunks:
          want = tuple("attn" if is_attention(scfg.model, l) else "gdn" for l in ch.layers)
          self.assertEqual(ch.kinds, want, ch)
          self.assertEqual(ch.kinds, ds_ep.layer_kinds(scfg.model, ch.layers))
          self.assertEqual(ch.kinds, tuple("attn" if scfg.model.is_attention_layer(l) else "gdn" for l in ch.layers))
    self.assertEqual({ch.kinds for ch in pipe(tiny(layers=4, fsdp=1, ep=2)).chunks}, {()})

  def test_layer_weights_round_trip(self):
    """`ds_ep.layer_weights` takes layer l out of a chunk's 'layers' tree -- its subtree, without the layer axis of
    one: the common leaves and its own mixer's -- and `stack_layers` puts the layers back, bit for bit, for every
    kind mix; DeepSeek's (no kinds) is every leaf of its stacks at l."""
    scfg = qtiny(layers=8)
    rng = np.random.default_rng(0)
    for layers in ((0,), (3,), (0, 1, 2), (2, 3, 4), (3, 4), (1, 2, 3, 4, 5), (3, 4, 5, 6, 7)):
      kinds = ds_ep.layer_kinds(scfg.model, layers)
      ch = ds_ep.ChunkSpec(index=0, layers=layers, first=False, last=False, stage=0, kinds=kinds)
      wl = jax.tree.map(
          lambda s: rng.standard_normal(s).astype(np.float32),
          ds_ep.chunk_shapes(scfg, ch)["layers"],
          is_leaf=ds_ep._is_shape,
      )
      per = [ds_ep.layer_weights(wl, l, kinds) for l in range(len(layers))]
      with self.subTest(layers=layers):
        for l, (lw, kind) in enumerate(zip(per, kinds)):
          self.assertEqual(set(lw), {"input_layernorm", "post_attention_layernorm", "mlp", kind})
          assert_bitwise(self, lw, jax.tree.map(lambda a: a[0], wl[ds_ep.qwen_layer_key(l)]), f"layer {l} {kind}")
        back = ds_ep.stack_layers(per, kinds)
        self.assertEqual(jax.tree.structure(back), jax.tree.structure(wl))
        assert_bitwise(self, back, wl, f"{layers} round trip")
    ds = tiny(layers=2)
    ch = ds_ep.ChunkSpec(index=0, layers=(0, 1), first=False, last=False, stage=0)
    wl = jax.tree.map(
        lambda s: rng.standard_normal(s).astype(np.float32), ds_ep.chunk_shapes(ds, ch)["layers"], is_leaf=ds_ep._is_shape
    )
    per = [ds_ep.layer_weights(wl, l) for l in range(2)]
    for l in range(2):
      assert_bitwise(self, per[l], jax.tree.map(lambda a: a[l], wl), f"DeepSeek layer {l}")
    assert_bitwise(self, ds_ep.stack_layers(per), wl, "DeepSeek round trip")

  def test_routed_leaves_and_names(self):
    """The routed experts are stored as DeepSeek's: expert axis over 'expert', embed axis over 'fsdp' -- wi_0 /
    wi_1 P(None, 'expert', 'fsdp', None), wo P(None, 'expert', None, 'fsdp') -- so `Pipeline.routed_leaves`, the
    leaves chunk 0's first forward gathers itself (self-prefetch), is exactly chunk 0's three; what is used
    elementwise (the norm scales, the conv kernel, A_log, dt_bias) is replicated, nothing to gather
    (`has_gather`); and `leaf_names` reads the port's names, each behind its layer's key in the chunk."""
    p = pipe(qtiny(layers=4, fsdp=1, ep=2), chunks_per_stage=1, layer_split=(1, 3))
    ch0, ch1 = p.chunks
    names = ds_ep.leaf_names(ch0)
    st = jax.tree.leaves(ds_ep.chunk_storage_specs(ch0), is_leaf=ds_ep._is_spec)
    routed = {n: i for i, n in enumerate(names) if n.startswith("00.mlp.routed_experts.w")}
    self.assertEqual(
        sorted(routed), ["00.mlp.routed_experts.wi_0", "00.mlp.routed_experts.wi_1", "00.mlp.routed_experts.wo"]
    )
    self.assertEqual(p.routed_leaves, tuple(sorted(routed.values())))
    for n, i in routed.items():
      self.assertEqual(st[i], P(None, "expert", None, "fsdp") if n.endswith("wo") else P(None, "expert", "fsdp", None))
    # one chunk holding both mixers, the embedding and the head
    both = ds_ep.ChunkSpec(index=0, layers=(2, 3), first=True, last=True, stage=0, kinds=("gdn", "attn"))
    gdn = [
        "gdn.A_log",
        "gdn.conv1d",
        "gdn.dt_bias",
        "gdn.in_proj_ba",
        "gdn.in_proj_qkvz",
        "gdn.norm.rms_norm",
        "gdn.out_proj",
    ]
    attn = ["attn.key", "attn.key_norm", "attn.out", "attn.query", "attn.query_norm", "attn.value"]
    rest = [
        "input_layernorm",
        "mlp.routed_experts.gate",
        "mlp.routed_experts.wi_0",
        "mlp.routed_experts.wi_1",
        "mlp.routed_experts.wo",
        "mlp.shared_expert.wi_0",
        "mlp.shared_expert.wi_1",
        "mlp.shared_expert.wo",
        "mlp.shared_expert_gate",
        "post_attention_layernorm",
    ]
    self.assertEqual(
        ds_ep.leaf_names(both),
        ["decoder_norm"]
        + [f"00.{n}" for n in gdn + rest]
        + [f"01.{n}" for n in attn + rest]
        + ["logits_dense", "token_embedder.embedding"],
    )
    elementwise = {
        "decoder_norm",
        "attn.key_norm",
        "attn.query_norm",
        "gdn.A_log",
        "gdn.conv1d",
        "gdn.dt_bias",
        "gdn.norm.rms_norm",
        "input_layernorm",
        "post_attention_layernorm",
    }
    for i, n in enumerate(ds_ep.leaf_names(both)):
      unkeyed = n.split(".", 1)[1] if n[:2].isdigit() else n  # without a layer leaf's key
      self.assertEqual(ds_ep.has_gather(both, i), unkeyed not in elementwise, n)
    self.assertEqual(ds_ep.leaf_names(ch1)[:3], ["decoder_norm", "00.gdn.A_log", "00.gdn.conv1d"])
    self.assertIn("02.attn.query", ds_ep.leaf_names(ch1))

  @testing.needs_devices(4)
  def test_qwen38_parameter_counts(self):
    """Qwen3.8-2.4T-A95B at its full dims, from the chunk shapes alone (nothing allocated): 2,419,804,697,984
    parameters over the chunks of the 92 layers however they are cut -- 26,262,741,376 a Gated DeltaNet layer,
    26,243,785,216 a gated-attention layer, 2,034,237,440 the embedding and as many the head -- MaxText's
    qwen3.8-2.4t-a95b.yml counted (2.42T, as its name says)."""
    scfg = ds_ep.StageConfig(
        model=Qwen35Config(**QWEN38), gmm_impl="ragged_dot", use_ragged_sort=False, **testing.stage_axes()
    )
    count = lambda tree: sum(math.prod(s) for s in jax.tree.leaves(tree, is_leaf=ds_ep._is_shape))
    one = lambda layer: ds_ep.ChunkSpec(
        index=1, layers=(layer,), first=False, last=False, stage=0, kinds=ds_ep.layer_kinds(scfg.model, (layer,))
    )
    gdn, attn = count(ds_ep.chunk_shapes(scfg, one(0))), count(ds_ep.chunk_shapes(scfg, one(3)))
    self.assertEqual((gdn, attn), (26_262_741_376, 26_243_785_216))
    ends = ds_ep.chunk_shapes(scfg, ds_ep.ChunkSpec(index=0, layers=(0,), first=True, last=True, stage=0, kinds=("gdn",)))
    self.assertEqual((count(ends["token_embedder"]), count(ends["logits_dense"])), (2_034_237_440,) * 2)
    for vv, split in ((2, None), (1, (47, 45)), (2, (1, 30, 30, 31))):
      with self.subTest(vv=vv, split=split):
        p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=vv, n_micro=4, global_batch=16, seq=4096, layer_split=split)
        self.assertEqual(ds_ep.num_params(scfg, p.chunks), 2_419_804_697_984)
        self.assertEqual(sum(ch.kinds.count("attn") for ch in p.chunks), 23)
        for ch in p.chunks:
          self.assertEqual(
              ds_ep.num_params(scfg, [ch]),
              ch.kinds.count("gdn") * gdn
              + ch.kinds.count("attn") * attn
              + ch.first * 2_034_237_440
              + ch.last * (2_034_237_440 + 8192),
          )
    self.assertEqual(69 * gdn + 23 * attn + 2 * 2_034_237_440 + 8192, 2_419_804_697_984)

  def test_chunk_files_round_trip(self):
    """--init-from's chunk files (`write_chunk_files` / `chunk_trees_from`) of Qwen chunks -- the port's leaf names,
    bfloat16 too -- load back bit for bit, into another placement too, and a DeepSeek chunk's do not load."""
    scfg = qtiny(layers=4, fsdp=1, ep=2)
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16))
    p = pipe(scfg, n_micro=2, global_batch=4)
    want = host_weights(p)
    with tempfile.TemporaryDirectory() as d:
      dsp.write_chunk_files(d, want)
      for q in (p, pipe(scfg, n_micro=2, global_batch=4, placement="v")):
        got = q.chunk_trees(q.params_from_chunks(q.chunk_trees_from(d)))
        assert_bitwise(self, got, want, q.placement)
        self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
      ds = pipe(tiny(layers=4, fsdp=1, ep=2), n_micro=2, global_batch=4)
      with self.assertRaisesRegex(ValueError, "not this model's"):
        ds.params_from_chunks(ds.chunk_trees_from(d))


# ------------------------------------------------------------------ refusals
@testing.needs_devices(4)
class QwenRefusals(unittest.TestCase):
  """What the Qwen3.5-family stage programs do not implement is refused where it is asked for, with a message
  that says why.  Nothing compiles here."""

  def test_stage_config(self):
    """What the stage programs do not build is refused, and so is a setting of a port flag whose choice they fix
    another way: the argsort dispatch (use_ragged_sort off) is MaxText's float32_weight_sum=true
    moe_use_direct_token_gather=false, and the ragged sort at ep 1 (where MaxText's dispatch is the argsort one
    too) its float32_weight_sum=true moe_use_direct_token_gather=true -- its backward sums a token's k copies in
    fp32, as the direct gather does (QwenDispatchVsMaxText) -- while MaxText's ring + ragged-sort path, like this
    one, reads neither; both gmm_impl settings accumulate the grouped matmuls in fp32, only megablox computes dW as
    gmm_exact_drhs reproduces it, and the head casts the logits to fp32.  fp32 router logits and the manual GDN
    core (an identity under the stage's manual context) build."""
    base = qtiny()
    m = base.model
    ring, argsort = dict(use_ragged_sort=True, ep=2), dict(use_ragged_sort=False)
    ep1 = dict(use_ragged_sort=True, fsdp=4, ep=1, fsdp_collectives="xla")
    ep1_argsort = dict(ep1, use_ragged_sort=False)
    cases = [
        (dict(cp=2), "context parallelism is not implemented for the Qwen3.5 family"),
        (dict(model=m.replace(shared_experts=0)), "shared_experts=0: .*sigmoid-gated shared expert"),
        (dict(model=m.replace(moe_impl="dense")), "moe_impl='dense': .*no dense MoE"),
        (
            dict(model=m.replace(float32_weight_sum=False), **argsort),
            "float32_weight_sum=False: MaxText's argsort dispatch .* compute dtype; the stage programs sum them "
            "in fp32",
        ),
        (dict(model=m.replace(float32_weight_sum=False), **ep1), "float32_weight_sum=False"),
        (
            dict(model=m.replace(moe_use_direct_token_gather=True), **argsort),
            "moe_use_direct_token_gather: MaxText's argsort dispatch \\(the counterpart with use_ragged_sort off\\) "
            "then sums dx over a token's k copies in fp32",
        ),
        (dict(model=m.replace(moe_use_direct_token_gather=True), **ep1_argsort), "moe_use_direct_token_gather"),
        (
            dict(model=m.replace(gmm_fp32_accum=False), gmm_impl="megablox"),
            "gmm_fp32_accum=False .*accumulate the grouped matmuls in fp32",
        ),
        (dict(model=m.replace(gmm_fp32_accum=False), gmm_impl="ragged_dot"), "gmm_fp32_accum=False"),
        (
            dict(model=m.replace(gmm_exact_drhs=True), gmm_impl="ragged_dot"),
            "gmm_exact_drhs .*gmm_impl='ragged_dot' computes dW with an fp32 ragged_dot_general",
        ),
        (dict(model=m.replace(cast_logits_to_fp32=False)), "cast_logits_to_fp32=False: the head casts"),
        (dict(model=m.replace(emb_dim=258)), "emb_dim=258 must divide by fsdp\\*ep=4"),
        (
            dict(model=m.replace(emb_dim=324), fsdp=2, ep=4, fsdp_collectives="xla"),
            "emb_dim=324 must divide by fsdp\\*ep=8",
        ),
        (dict(model=m.replace(num_experts=6), ep=4, fsdp=1, fsdp_collectives="xla"), "num_experts=6 by ep=4"),
    ]
    for kw, msg in cases:
      with self.subTest(**{k: str(v)[:40] for k, v in kw.items()}), self.assertRaisesRegex(ValueError, msg):
        dataclasses.replace(base, **kw)
    # what they do take: the flags the ring + ragged-sort path ignores (as MaxText's does), the direct gather the
    # ragged sort computes at ep 1, megablox's own dW, fp32 router logits, the manual GDN core
    for kw in (
        dict(model=m.replace(float32_weight_sum=False), **ring),
        dict(model=m.replace(moe_use_direct_token_gather=True), **ring),
        dict(model=m.replace(moe_use_direct_token_gather=True), **ep1),
        dict(model=m.replace(gmm_exact_drhs=True), gmm_impl="megablox"),
        dict(model=m.replace(float32_gate_logits=True)),
        dict(model=m.replace(gdn_core="manual")),
    ):
      with self.subTest(**{k: str(v)[:40] for k, v in kw.items()}):
        dataclasses.replace(base, **kw)
    # and one tpu7x-4x4x4 slice a stage at Qwen3.8's dims
    for fsdp, ep in ((16, 8), (4, 32), (1, 128)):
      ds_ep.StageConfig(model=Qwen35Config(**QWEN38), fsdp=fsdp, ep=ep, fsdp_collectives="xla")

  def test_pipeline(self):
    """No stash modes (STASH_POLICIES name MLA's residuals) -- no stash policy, under any schedule -- and no 'auto'
    prefetch pins (`pin_windows` prices MLA's FLOPs), before anything is built; the stash schedule without a policy
    (--stash none), a search of remat-only orders, builds, as on DeepSeek; pins by SPEC take the Qwen leaf names,
    and refuse DeepSeek's."""
    scfg = qtiny(layers=4, fsdp=1, ep=2)
    for kw in (
        dict(stash="lite"),
        dict(schedule="stash", stash="lite"),
        dict(schedule="interleaved", stash="attn"),
    ):
      with self.subTest(**kw), self.assertRaisesRegex(ValueError, "stash modes are not implemented for the Qwen3.5"):
        pipe(scfg, **kw)
    p = pipe(scfg, schedule="stash")
    self.assertEqual((p.modes, p.stash_bytes), ({}, {}))
    self.assertTrue(p.schedule_name.startswith("stash none"), p.schedule_name)
    with self.assertRaisesRegex(ValueError, "prefetch_pin 'auto' .* for the Qwen3.5 family pin by SPEC"):
      pipe(scfg, prefetch_pin="auto")
    ch = pipe(scfg).chunks[0]
    with self.assertRaisesRegex(ValueError, "a Qwen3.5-family stage has no window model"):
      ds_ep.pin_windows(scfg, ch, 4096)
    p = pipe(scfg, prefetch_pin="hn=00.mlp.routed_experts.wi_0,00.mlp.routed_experts.wo")
    self.assertTrue(p.pins)
    with self.assertRaisesRegex(ValueError, "moe.wi_0"):
      pipe(scfg, prefetch_pin="hn=moe.wi_0")
    # the same options on DeepSeek build
    pipe(tiny(layers=4, fsdp=1, ep=2), prefetch_pin="auto")

  def test_cli(self):
    """The model flags only the DeepSeek-V3 layers implement (MLA's --q-lora, --kv-lora; dense layers, the routed
    bias, its update rate and groups, the aux loss, MTP) are refused under a Qwen --model, the Qwen-only ones
    (--kv-heads, --head-dim, --gdn-*) under deepseek3 -- by `build`, so by `main` before anything runs.
    --float32-gate-logits, which MaxText's qwen3_5 takes too, reaches the Qwen model's config."""
    ap = dsp.make_parser()
    q38 = ["--model", "qwen3.8-2.4t-a95b"]
    ds_only = "only the DeepSeek-V3 layers implement these, but --model is"
    for args, msg in (
        (q38 + ["--q-lora", "64"], f"\\['--q-lora'\\]: {ds_only} qwen3.8-2.4t-a95b"),
        (["--model", "qwen3.5-397b-a17b", "--kv-lora", "64"], f"\\['--kv-lora'\\]: {ds_only} qwen3.5-397b-a17b"),
        (q38 + ["--dense-layers", "1", "--mlp", "512"], f"\\['--dense-layers', '--mlp'\\]: {ds_only}"),
        (
            q38 + ["--routing", "topk", "--routed-bias", "on", "--bias-update-rate", "0.001"],
            f"\\['--routed-bias', '--bias-update-rate'\\]: {ds_only}",
        ),
        (
            q38 + ["--routing", "topk", "--routing-groups", "8", "--topk-groups", "4"],
            f"\\['--routing-groups', '--topk-groups'\\]: {ds_only}",
        ),
        (
            q38 + ["--float32-gate-logits", "on", "--lb-loss-weight", "0.01", "--mtp", "1"],
            f"\\['--lb-loss-weight', '--mtp'\\]: {ds_only}",
        ),
        (["--kv-heads", "2"], "\\['--kv-heads'\\]: Qwen3.5-family flags, but --model is deepseek3"),
        (
            ["--model", "deepseek3", "--head-dim", "64", "--gdn-value-head-dim", "32"],
            "\\['--head-dim', '--gdn-value-head-dim'\\]: Qwen3.5-family flags",
        ),
    ):
      with self.subTest(args=args):
        with self.assertRaisesRegex(ValueError, msg):
          dsp.model_config(ap.parse_args(args))
        with self.assertRaisesRegex(ValueError, msg):
          dsp.main(args + ["--dry-run"])
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args(["--model", "qwen3_5"])
    for flag, f32 in (("on", True), ("off", False)):
      self.assertIs(dsp.model_config(ap.parse_args(q38 + ["--float32-gate-logits", flag])).float32_gate_logits, f32)
      self.assertIs(
          dsp.build(ap.parse_args(Q_TINY_ARGS + ["--float32-gate-logits", flag])).scfg.model.float32_gate_logits, f32
      )

  def test_lineage_batch_split(self):
    """--batch-split-impl lineage (batch_split_impl 'lineage') runs Lineage's layers, which are DeepSeek-V3's: a
    Qwen3.5-family model is refused alone, as a ValueError naming the model (not an AttributeError on a DeepSeekConfig
    field), by the StageConfig, by ds_ep_lineage.check, and by `build` before anything is probed or built; and by
    `main` so too, whatever MaxText is installed and whatever the backend -- no Lineage asked for first, no device
    refused, no flag named as one Lineage's layers ignore or apply in part."""
    from maxtext.experimental.mpmd.jax import ds_ep_lineage

    base = qtiny()
    for kw in (
        dict(batch_split=True, batch_split_impl="lineage"),
        dict(batch_split=True, batch_split_impl="lineage", shard_mode="explicit", random_routing=False),
    ):
      with self.subTest(**kw):
        with self.assertRaises(ValueError) as e:
          dataclasses.replace(base, **kw)
        self.assertIn("Lineage builds DeepSeek-V3's layers only", str(e.exception))
        self.assertNotIn("shard_mode", str(e.exception))  # the stage's settings are moot
        self.assertIn(
            "(batch_split_impl 'native' -- the Qwen3.5 family's default, None -- runs its split_layers)", str(e.exception)
        )
    self.assertFalse(dataclasses.replace(base, batch_split=False, batch_split_impl="lineage").lineage_split)
    with self.assertRaisesRegex(ValueError, "a Qwen3.5-family model -- Lineage builds DeepSeek-V3's layers only"):
      ds_ep_lineage.check(base.model)
    # the flags as asked (Lineage's defaults), every Lineage-only check silent: build gives the one reason
    args = Q_TINY_ARGS + ["--gbs", "8", "--batch-split", "--batch-split-impl", "lineage", "--float32-gate-logits", "on"]
    a = dsp.make_parser().parse_args(args)
    self.assertEqual(
        (a.batch_split_impl, a.routing, a.shard_mode, dsp.uses_lineage(a)), ("lineage", "topk", "explicit", True)
    )

    def missing():
      raise ImportError("ds_ep_lineage could not import MaxText's Lineage stack: No module named 'typeguard'")

    cpu = [types.SimpleNamespace(platform="cpu", device_kind="cpu")]
    with mock.patch.object(ds_ep_lineage, "import_lineage", missing):
      self.assertIsNone(dsp.lineage_missing(a))
    self.assertIsNone(dsp.lineage_off_tpu(a, cpu))
    self.assertEqual((dsp.lineage_ignored(a), dsp.lineage_partial(a)), ([], []))  # (--gmm-impl, --ragged-sort given)
    want = (
        r"^--batch-split runs Lineage's layers \(--batch-split-impl lineage\), which cannot run this: --model "
        r"qwen3.8-2.4t-a95b: Lineage builds DeepSeek-V3's layers only, [^;]* \(leave --batch-split-impl out: a "
        r"Qwen3.5-family model's default, native, runs ds_ep.split_layers\)$"
    )
    with (
        mock.patch.object(ds_ep_lineage, "import_lineage", missing),
        mock.patch.object(dsp, "offload_answers", side_effect=AssertionError("probed")),
        mock.patch.object(ds_ep, "ChunkPrograms", side_effect=AssertionError("built")),
    ):
      with self.assertRaisesRegex(ValueError, want):
        dsp.build(a)
      for extra in ([], ["--dry-run"]):
        with self.subTest(main=extra):
          with self.assertRaisesRegex(ValueError, want), contextlib.redirect_stdout(io.StringIO()) as log:
            dsp.main(args + extra)
          self.assertNotIn("warning", log.getvalue())


# ---------------------------------------------------------------------- flags
@testing.needs_devices(4)
class QwenFlags(unittest.TestCase):
  """`--model`: a Qwen3.5-family model by its MaxText name makes its dims the shape flags' defaults."""

  CLI = dict(
      layers="num_decoder_layers",
      emb="emb_dim",
      heads="num_query_heads",
      kv_heads="num_kv_heads",
      head_dim="head_dim",
      moe_mlp="moe_mlp_dim",
      experts="num_experts",
      topk="num_experts_per_tok",
      vocab="vocab_size",
      gdn_key_heads="gdn_num_key_heads",
      gdn_value_heads="gdn_num_value_heads",
      gdn_key_head_dim="gdn_key_head_dim",
      gdn_value_head_dim="gdn_value_head_dim",
  )

  def test_model_defaults_are_the_yml(self):
    """`--model qwen3.8-2.4t-a95b` alone is the yml's model: 92 layers, emb 8192, 64 / 4 heads of 256, a GDN of
    16 / 128 heads of 128, 512 experts of 2048, top-10, vocab 248320; the MLA flags None.  `model_config` builds
    the Qwen35Config of those dims, with MaxText's qwen3_5 rest (the 4-layer cycle, partial rotary 0.25, rope
    10M, a 4-tap conv, 64-token GDN chunks, one shared expert, renormalised top-k) and the yml's rotary embedding:
    the partial RoPE (MaxText's use_mrope is base.yml's false there), where the Qwen3.5 models take the text
    mRoPE their ymls set; --qwen-rope overrides it."""
    a = dsp.make_parser().parse_args(["--model", "qwen3.8-2.4t-a95b"])
    self.assertEqual({k: getattr(a, k) for k in self.CLI}, {k: QWEN38[f] for k, f in self.CLI.items()})
    self.assertEqual((a.q_lora, a.kv_lora), (None, None))
    cfg = dsp.model_config(a)
    self.assertIsInstance(cfg, Qwen35Config)
    self.assertTrue(ds_ep.is_qwen(cfg))
    for f, v in QWEN38.items():
      self.assertEqual(getattr(cfg, f), v, f)
    self.assertEqual(cfg.mrope_section, (11, 11, 10))  # qwen3.5's, at head_dim 256 (unread here)
    self.assertEqual((a.qwen_rope, cfg.use_mrope), ("partial", False))
    for name, rope in (("qwen3.5-397b-a17b", True), ("qwen3.5-35b-a3b", True)):
      self.assertIs(dsp.model_config(dsp.make_parser().parse_args(["--model", name])).use_mrope, rope, name)
    over = dsp.make_parser().parse_args(["--model", "qwen3.8-2.4t-a95b", "--qwen-rope", "mrope"])
    self.assertIs(dsp.model_config(over).use_mrope, True)
    self.assertEqual(
        (cfg.max_target_length, cfg.weight_dtype, cfg.attention_kernel), (4096, jnp.bfloat16, "tokamax_splash")
    )
    # every model in the table builds, with an mRoPE split summing to rotary_dim // 2
    for name in dsp.QWEN_MODELS:
      c = dsp.model_config(dsp.make_parser().parse_args(["--model", name]))
      self.assertEqual(sum(c.mrope_section), c.rotary_dim // 2, name)
      self.assertEqual(c.num_decoder_layers % c.inhomogeneous_layer_cycle_interval, 0, name)
    c = dsp.model_config(dsp.make_parser().parse_args(["--model", "qwen3.5-397b-a17b"]))
    q397 = Qwen35Config()  # the class defaults are 397B-A17B's
    self.assertEqual({f: getattr(c, f) for f in self.CLI.values()}, {f: getattr(q397, f) for f in self.CLI.values()})
    c = dsp.model_config(dsp.make_parser().parse_args(["--model", "qwen3.5-35b-a3b"]))
    q35 = qwen35_35b()  # the port's preset of MaxText's yml
    self.assertEqual({f: getattr(c, f) for f in self.CLI.values()}, {f: getattr(q35, f) for f in self.CLI.values()})

  def test_models_are_maxtexts_ymls(self):
    """Every QWEN_MODELS entry is MaxText's configs/models/<name>.yml, field for field: a qwen3_5 decoder of the
    4-layer cycle with one shared expert, and every dim the flags take.  The ymls are MaxText's configs/models/
    (maxtext_parity.maxtext_models_dir)."""
    models = mp.maxtext_models_dir()
    if yaml is None:
      self.skipTest("no yaml")
    self.assertTrue(os.path.isfile(os.path.join(models, "deepseek3-671b.yml")), models)
    root = pathlib.Path(models)
    keys = dict(
        layers="base_num_decoder_layers",
        emb="base_emb_dim",
        heads="base_num_query_heads",
        kv_heads="base_num_kv_heads",
        head_dim="head_dim",
        moe_mlp="base_moe_mlp_dim",
        experts="num_experts",
        topk="num_experts_per_tok",
        vocab="vocab_size",
        gdn_key_heads="gdn_num_key_heads",
        gdn_value_heads="gdn_num_value_heads",
        gdn_key_head_dim="gdn_key_head_dim",
        gdn_value_head_dim="gdn_value_head_dim",
    )
    for name, dims in dsp.QWEN_MODELS.items():
      f = root / f"{name}.yml"
      y = yaml.safe_load(f.read_text())
      with self.subTest(name):
        self.assertEqual({k: dims[k] for k in keys}, {k: y[yk] for k, yk in keys.items()})
        self.assertEqual(
            (y["decoder_block"], y["inhomogeneous_layer_cycle_interval"], y["shared_experts"]), ("qwen3_5", 4, 1)
        )
        # no use_mrope in the yml: base.yml's false, the partial RoPE
        self.assertEqual(dims["qwen_rope"], "mrope" if y.get("use_mrope", False) else "partial")

  def test_given_flags_win(self):
    """A shape flag given wins over the model's default (fewer layers and experts at the per-layer dims), and the
    tiny dims all given build Q_TINY's dims, with qwen3.8's partial RoPE."""
    a = dsp.make_parser().parse_args(
        ["--model", "qwen3.8-2.4t-a95b", "--layers", "8", "--experts", "64", "--weight-dtype", "float32"]
    )
    self.assertEqual((a.layers, a.experts, a.emb, a.gdn_value_heads, a.topk), (8, 64, 8192, 128, 10))
    cfg = dsp.model_config(a)
    self.assertEqual((cfg.num_decoder_layers, cfg.num_experts, cfg.emb_dim, cfg.weight_dtype), (8, 64, 8192, jnp.float32))
    cfg = dsp.model_config(dsp.make_parser().parse_args(Q_TINY_ARGS))
    for f, v in Q_TINY.items():
      if f not in ("max_target_length", "attention_kernel"):
        self.assertEqual(getattr(cfg, f), v, f)
    self.assertIs(cfg.use_mrope, False)  # --model qwen3.8's partial RoPE, not Q_TINY's default mRoPE

  def test_no_defaults_leak_between_parses(self):
    """One parser parses a Qwen --model, another Qwen one, then deepseek3: each parse gets its own model's
    defaults -- the third exactly what a fresh parser gives with nothing set."""
    fresh = vars(dsp.make_parser().parse_args([]))
    ap = dsp.make_parser()
    self.assertEqual(ap.parse_args(["--model", "qwen3.8-2.4t-a95b"]).emb, 8192)
    a = ap.parse_args(["--model", "qwen3.5-35b-a3b"])
    self.assertEqual((a.layers, a.emb, a.experts, a.topk, a.gdn_value_heads), (40, 2048, 256, 8, 32))
    self.assertEqual(vars(ap.parse_args([])), fresh)
    self.assertEqual(vars(ap.parse_args(["--model", "deepseek3"])), fresh)
    self.assertEqual(ap.parse_args(["--model", "qwen3.8-2.4t-a95b"]).emb, 8192)

  def test_parser_reads_its_arguments_once(self):
    """The model is peeked at, then parsed for real, from one list: a one-shot iterator of arguments keeps every
    flag (as a plain parser's does); a model in a given namespace picks the defaults; and a parser's own
    set_defaults, before or after a Qwen parse, still holds for the parses after it."""
    a = dsp.make_parser().parse_args(iter(["--model", "qwen3.8-2.4t-a95b", "--layers", "8", "--mb", "4"]))
    self.assertEqual((a.model, a.layers, a.mb, a.emb), ("qwen3.8-2.4t-a95b", 8, 4, 8192))
    a = dsp.make_parser().parse_args(iter(["--layers", "6"]))
    self.assertEqual((a.model, a.layers, a.emb), ("deepseek3", 6, 7168))
    a = dsp.make_parser().parse_args([], namespace=argparse.Namespace(model="qwen3.8-2.4t-a95b"))
    self.assertEqual((a.layers, a.emb, a.q_lora), (92, 8192, None))
    ap = dsp.make_parser()
    ap.parse_args(["--model", "qwen3.8-2.4t-a95b"])
    ap.set_defaults(mb=4, emb=1024)
    self.assertEqual((ap.parse_args([]).mb, ap.parse_args([]).emb), (4, 1024))
    a = ap.parse_args(["--model", "qwen3.8-2.4t-a95b"])
    self.assertEqual((a.mb, a.emb), (4, 8192))  # the model's dim wins over set_defaults' emb
    self.assertEqual(ap.parse_args([]).emb, 1024)

  def test_a_namespace_model_yields_to_the_arguments(self):
    """The model whose defaults a parse takes is argparse's own pick: the arguments' --model, then a given
    namespace's, then the default.  With another model in the namespace -- Qwen3.8 there and DeepSeek-V3 in the
    arguments, the reverse, or Qwen3.5-35B there and Qwen3.8 in the arguments -- each parse is its arguments' alone
    and builds their model, where the namespace's model used to pick the dims (and then refuse flags nobody gave,
    or build Qwen3.8 at Qwen3.5-35B's dims)."""
    ds3 = ("deepseek3", 4, 7168, 1536, 512, None, None)
    q38 = ("qwen3.8-2.4t-a95b", 92, 8192, None, None, 4, "partial")
    for theirs, dims in (("qwen3.8-2.4t-a95b", ds3), ("deepseek3", q38), ("qwen3.5-35b-a3b", q38)):
      args = ["--model", dims[0]]
      with self.subTest(args=args, namespace=theirs):
        a = dsp.make_parser().parse_args(args, namespace=argparse.Namespace(model=theirs))
        self.assertEqual((a.model, a.layers, a.emb, a.q_lora, a.kv_lora, a.kv_heads, a.qwen_rope), dims)
        self.assertEqual(vars(a), vars(dsp.make_parser().parse_args(args)))
        cfg = dsp.model_config(a)
        self.assertEqual((ds_ep.is_qwen(cfg), cfg.num_decoder_layers, cfg.emb_dim), (dims is q38, *dims[1:3]))
    # with no --model among the arguments, the namespace's picks them
    a = dsp.make_parser().parse_args(["--layers", "8"], namespace=argparse.Namespace(model="qwen3.5-35b-a3b"))
    self.assertEqual((a.model, a.layers, a.emb, a.kv_heads, a.qwen_rope), ("qwen3.5-35b-a3b", 8, 2048, 2, "mrope"))

  def test_mrope_sections_split_evenly(self):
    """--qwen-rope mrope at any head dim: the three mRoPE sections are rotary_dim // 2 (rotary_dim = head_dim x
    Qwen35Config.partial_rotary_factor) split as evenly as it goes, none negative -- (11, 11, 10) at head_dim 256,
    as qwen3.5's yml -- where the ceil split went (1, 1, -1) at head dims 8 and 10 and (6, 6, 4) at 128."""
    for head_dim in (8, 10, 16, 32, 64, 128, 256):
      with self.subTest(head_dim=head_dim):
        cfg = dsp.model_config(
            dsp.make_parser().parse_args(
                ["--model", "qwen3.8-2.4t-a95b", "--qwen-rope", "mrope", "--head-dim", str(head_dim)]
            )
        )
        sec = cfg.mrope_section
        self.assertEqual(cfg.rotary_dim, int(head_dim * Qwen35Config.partial_rotary_factor))
        self.assertEqual(sum(sec), cfg.rotary_dim // 2)
        self.assertGreaterEqual(min(sec), 0)
        self.assertLessEqual(max(sec) - min(sec), 1)
        self.assertEqual(list(sec), sorted(sec, reverse=True))
        if head_dim == 256:
          self.assertEqual(sec, (11, 11, 10))

  def test_deepseek_unchanged(self):
    """Under deepseek3 (the default) every flag keeps the default it declares -- those --batch-split decides as
    split_defaults fills them in (--batch-split-impl lineage, --routing random, --shard-mode auto) --, the new ones
    are --model and the Qwen-only shape flags (None), and `model_config` is the DeepSeekConfig the flags always
    made."""
    ap = dsp.make_parser()
    declared = vars(
        dsp.split_defaults(
            argparse.Namespace(**{act.dest: act.default for act in dsp.make_parser()._actions if act.dest != "help"})
        )
    )
    self.assertEqual(
        {k: declared[k] for k in ("batch_split_impl", "routing", "shard_mode")},
        {"batch_split_impl": "lineage", "routing": "random", "shard_mode": "auto"},
    )
    a = ap.parse_args([])
    self.assertEqual(vars(a), declared)
    new = {
        "model": "deepseek3",
        "kv_heads": None,
        "head_dim": None,
        "gdn_key_heads": None,
        "gdn_value_heads": None,
        "gdn_key_head_dim": None,
        "gdn_value_head_dim": None,
    }
    self.assertEqual({k: vars(a)[k] for k in new}, new)
    self.assertEqual(
        (a.layers, a.emb, a.heads, a.q_lora, a.kv_lora, a.moe_mlp, a.experts, a.topk, a.vocab),
        (4, 7168, 128, 1536, 512, 2048, 128, 8, 32768),
    )
    cfg = dsp.model_config(a)
    self.assertFalse(ds_ep.is_qwen(cfg))
    self.assertEqual(
        (
            cfg.emb_dim,
            cfg.num_query_heads,
            cfg.num_kv_heads,
            cfg.q_lora_rank,
            cfg.kv_lora_rank,
            cfg.first_num_dense_layers,
            cfg.vocab_size,
        ),
        (7168, 128, 128, 1536, 512, 0, 32768),
    )

  def test_batch_split_is_split_layers(self):
    """--batch-split under a Qwen3.5-family --model runs ds_ep.split_layers (--batch-split-impl native), the
    family's only batch split -- Lineage builds DeepSeek-V3's layers only -- whatever MaxText is installed: the
    parser resolves the impl by the model (split_defaults), and with it --routing and --shard-mode as unsplit
    (random, auto); none of the Lineage-only checks speaks (no Lineage asked for, no backend refused, nothing named
    as ignored, applied in part or unused but what the flags gave), nothing refused on TPU either (native is
    refused there for DeepSeek-V3 only: native_split_refusal), and the summary names split_layers.  A StageConfig
    resolves batch_split_impl None alike, and first, so that its default wi tiling caps a Qwen batch split's dW tm
    at 256 under split_dw 'acc' (split_acc_wi_tiling); deepseek3 keeps lineage."""
    ap = dsp.make_parser()
    cpu = [types.SimpleNamespace(platform="cpu", device_kind="cpu")]
    for extra in (["--batch-split"], ["--batch-split", "--batch-split-impl", "native"], []):
      with self.subTest(extra=extra):
        a = ap.parse_args(Q_TINY_ARGS + ["--gbs", "8", "--float32-gate-logits", "on"] + extra)
        self.assertEqual((a.batch_split_impl, a.routing, a.shard_mode), ("native", "random", "auto"))
        self.assertFalse(dsp.uses_lineage(a))
        self.assertEqual((dsp.lineage_missing(a), dsp.lineage_off_tpu(a, cpu)), (None, None))
        self.assertEqual((dsp.lineage_ignored(a), dsp.lineage_partial(a), dsp.lineage_unused(a)), ([], [], []))
        self.assertIsNone(dsp.native_split_refusal(ap.parse_args(Q_TINY_ARGS + ["--platform", "tpu"] + extra)))
    # Lineage's own flags given where its layers do not run are named, but never an impl the flags did not give
    cf = ["--lineage-capacity-factor", "5"]
    for extra, want in (
        (
            ["--batch-split"] + cf,
            ["--lineage-capacity-factor 5: not used by --model " "qwen3.8-2.4t-a95b's batch split, ds_ep.split_layers"],
        ),
        (cf, ["--lineage-capacity-factor 5: not used without --batch-split"]),
        (["--batch-split-impl", "lineage"], ["--batch-split-impl lineage: not used without --batch-split"]),
        (["--batch-split-impl", "native"], []),
    ):  # (the model's default: not told apart)
      with self.subTest(extra=extra):
        a = ap.parse_args(Q_TINY_ARGS + ["--gbs", "8"] + extra)
        self.assertEqual([line.split(" (")[0] for line in dsp.lineage_unused(a)], want)
    self.assertEqual(ap.parse_args(["--batch-split"]).batch_split_impl, "lineage")
    # a namespace made by hand, without the flags split_defaults fills in: build resolves them by the model too
    hand = argparse.Namespace(
        **{
            k: v
            for k, v in vars(ap.parse_args(Q_TINY_ARGS + ["--gbs", "8", "--batch-split"])).items()
            if k not in ("batch_split_impl", "routing", "shard_mode")
        }
    )
    self.assertEqual(dsp.batch_split_impl(hand), "native")
    p = dsp.build(hand)
    self.assertFalse(hasattr(hand, "batch_split_impl"))
    self.assertEqual(
        (p.scfg.batch_split, p.scfg.batch_split_impl, p.scfg.lineage_split, p.scfg.random_routing, p.scfg.shard_mode),
        (True, "native", False, True, "auto"),
    )
    self.assertIn("batch_split=native (ds_ep.split_layers; each device's 2 rows", p.summary())
    self.assertNotIn("deprecated", p.summary())
    self.assertEqual(dsp.provenance(p.scfg)["batch_split_impl"], "native")
    with contextlib.redirect_stdout(io.StringIO()) as log:
      out = dsp.main(Q_TINY_ARGS + ["--gbs", "8", "--batch-split", "--dry-run"])
    self.assertNotIn("warning", log.getvalue())
    self.assertEqual(out["config"]["batch_split_impl"], "native")
    self.assertEqual((qtiny(batch_split=True).batch_split_impl, qtiny().batch_split_impl), ("native", "native"))
    self.assertFalse(qtiny(batch_split=True).lineage_split)
    # the default wi tiling (none given; qtiny's is TILE) reads the resolved impl: dW tm 256 under split_dw 'acc'
    cfg = qtiny().model
    self.assertEqual(
        [
            ds_ep.StageConfig(model=cfg, fsdp=1, ep=2, **kw).wi_tiling[6]
            for kw in ({}, dict(batch_split=True), dict(batch_split=True, split_dw="sum"))
        ],
        [512, 256, 512],
    )
    self.assertEqual(
        (tiny().batch_split_impl, tiny(batch_split=True, shard_mode="explicit", random_routing=False).batch_split_impl),
        ("lineage", "lineage"),
    )


# ---------------------------------------------------------------------- FLOPs
class MaxTextConfig(types.SimpleNamespace):
  """The fields MaxText's calculate_tflops_training_per_device reads of a qwen3_5 MoE model, from a Qwen35Config at
  per_device_batch_size 1: ce608b918's, which b33d0f687 and 1f1e0935a read too, all but use_m3_model (MaxText reads
  it first since bacbda5ac, 2026-10-26)."""

  def __init__(self, cfg, seq):
    from maxtext.common.common_types import DecoderBlockType

    super().__init__(
        decoder_block=DecoderBlockType.QWEN3_5,
        model_name="qwen3_5",
        per_device_batch_size=1,
        max_target_length=seq,
        emb_dim=cfg.emb_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        shared_experts=cfg.shared_experts,
        moe_mlp_dim=cfg.moe_mlp_dim,
        mlp_dim=cfg.moe_mlp_dim,
        mlp_activations=list(cfg.mlp_activations),
        inhomogeneous_layer_cycle_interval=cfg.inhomogeneous_layer_cycle_interval,
        gdn_num_key_heads=cfg.gdn_num_key_heads,
        gdn_num_value_heads=cfg.gdn_num_value_heads,
        gdn_key_head_dim=cfg.gdn_key_head_dim,
        gdn_value_head_dim=cfg.gdn_value_head_dim,
        gdn_conv_kernel_dim=cfg.gdn_conv_kernel_dim,
        attention_type="global",
        share_kv_projections=False,
        mtp_num_layers=0,
        engram_layers=None,
        use_multimodal=False,
        gradient_accumulation_steps=1,
        use_dpo=False,
        use_m3_model=False,
    )  # base.yml's default


def flop_cases():
  """(name, config, seq): Qwen3.8 and Qwen3.5-397B at full dims, at two sequence lengths, and Q_TINY."""
  return [
      ("qwen3.8", Qwen35Config(**QWEN38), 4096),
      ("qwen3.8 seq 8192", Qwen35Config(**QWEN38), 8192),
      ("qwen3.5-397b", Qwen35Config(), 4096),
      ("tiny", Qwen35Config(num_decoder_layers=8, **Q_TINY), 256),
  ]


@testing.needs_devices(4)
class QwenFlops(unittest.TestCase):
  """MaxText's own FLOP accounting for decoder_block qwen3_5 (`dsp.qwen_training_flops`), the numerator of every
  TFLOP/s the pipeline prints."""

  def test_qwen38_is_maxtexts(self):
    """Qwen3.8-2.4T at seq 4096: 2268.036 TFLOP a sequence, 2215.878 of learnable weights + 52.158 of attention
    (causal attention and the delta rule) -- what MaxText's calculate_tflops_training_per_device reports at
    per_device_batch_size 1."""
    w, a = dsp.qwen_training_flops(Qwen35Config(**QWEN38), 4096, 4096)
    self.assertAlmostEqual(w / 1e12, 2215.878, delta=1e-3)
    self.assertAlmostEqual(a / 1e12, 52.158, delta=1e-3)
    self.assertLess(abs((w + a) / 1e12 / 2268.036 - 1), 1e-3)
    self.assertEqual(dsp.maxtext_training_flops(Qwen35Config(**QWEN38), 4096, 4096), (w, a))

  @testing.needs_devices(4)
  def test_maxtext_tflops_per_device(self):
    """A Qwen pipeline's per-device TFLOP a step: MaxText's for its gbs x seq tokens over S x fsdp x ep devices."""
    axes = testing.stage_axes()
    p = pipe(qtiny(layers=8, **axes), n_micro=2, global_batch=8)
    w, a = dsp.qwen_training_flops(p.scfg.model, 8 * 256, 256)
    devices = 2 * axes["fsdp"] * axes["ep"]
    self.assertAlmostEqual(dsp.maxtext_tflops_per_device(p), (w + a) / devices / 1e12, places=15)
    self.assertGreater(a, 0)

  def test_against_maxtexts_function(self):
    """Bit-identical (to 1e-12) to MaxText's calculate_tflops_training_per_device itself, given the qwen3_5 fields
    it reads (MaxTextConfig)."""
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

    for name, cfg, seq in flop_cases():
      with self.subTest(name):
        total, weights, attention = maxtext_utils.calculate_tflops_training_per_device(MaxTextConfig(cfg, seq), log=False)
        w, a = dsp.qwen_training_flops(cfg, seq, seq)
        self.assertAlmostEqual(w / 1e12 / weights, 1.0, places=12)
        self.assertAlmostEqual(a / 1e12 / attention, 1.0, places=12)
        self.assertAlmostEqual((w + a) / 1e12 / total, 1.0, places=12)


# ---------------------------------------------------------------------- init
@testing.needs_devices(4)
class QwenInit(unittest.TestCase):
  """StagePrograms.init: MaxText's init rules where they shape the math (`ds_ep.qwen_init_rule`)."""

  def test_qwen_rules(self):
    """A chunk of 12 layers with the embedding and the head: the zero-centred norm scales (input,
    post-attention, final, query, key) start at 0 -- applied as 1 + scale --, the GDN's gated output norm and
    dt_bias at 1, A_log at log A with A ~ U(1e-9, 16), every kernel, routed expert and the embedding ~N(0,
    0.02^2), each leaf from its own key."""
    scfg = qtiny(layers=12)
    layers = tuple(range(12))
    ch = ds_ep.ChunkSpec(
        index=0, layers=layers, first=True, last=True, stage=0, kinds=ds_ep.layer_kinds(scfg.model, layers)
    )
    params = ds_ep.StagePrograms(scfg, [ch], dsp.stage_meshes(1, 2, 2)[0]).init(jax.random.key(0))[0]
    names = ds_ep.leaf_names(ch)
    self.assertEqual({x.dtype for x in jax.tree.leaves(params)}, {jnp.dtype(jnp.float32)})
    # A layer's leaves are its own: each name's leaves over the chunk's layers, concatenated on the layer axis
    groups = collections.defaultdict(list)
    for n, x in zip(names, jax.tree.leaves(params)):
      groups[n.split(".", 1)[1] if n[:2].isdigit() else n].append(np.asarray(x, np.float64))
    zeros = {"input_layernorm", "post_attention_layernorm", "decoder_norm", "attn.query_norm", "attn.key_norm"}
    ones = {"gdn.norm.rms_norm", "gdn.dt_bias"}
    normal = []
    for n, xs in groups.items():
      x = np.concatenate(xs)
      with self.subTest(n):
        if n in zeros:
          np.testing.assert_array_equal(x, 0)
        elif n in ones:
          np.testing.assert_array_equal(x, 1)
        elif n == "gdn.A_log":
          # 9 GDN layers x 16 heads: inside (log 1e-9, log 16], and A = exp(A_log) spread over (0, 16] about 8
          self.assertEqual(x.shape, (9, 16))
          self.assertGreaterEqual(x.min(), math.log(1e-9) - 1e-6)
          self.assertLessEqual(x.max(), math.log(16.0) + 1e-6)
          a = np.exp(x)
          self.assertLess(abs(a.mean() - 8.0), 4 * 16 / math.sqrt(12 * a.size))
          self.assertTrue(a.min() < 2 and a.max() > 14, (a.min(), a.max()))
        else:
          self.assertGreater(x.size, 1000)
          self.assertLess(abs(x.std() / 0.02 - 1), 0.05)
          self.assertLess(abs(x.mean()), 5 * 0.02 / math.sqrt(x.size))
          normal.extend(v.ravel()[:64] for v in xs)
    self.assertEqual(len(normal), sum(len(xs) for n, xs in groups.items() if n not in zeros | ones | {"gdn.A_log"}))
    self.assertEqual(len({v.tobytes() for v in normal}), len(normal))  # no two leaves share a draw

  def test_rules_are_the_ports(self):
    """`ds_ep.qwen_init_rule` is the port's init rule, `maxtext.experimental.mpmd.jax.qwen35.init.leaf_rule` -- its
    zeros / ones
    / a_log / normal choice on every leaf of a 12-layer chunk with the embedding and the head, each kind there --
    and ds_ep's A_LOG_RANGE its range: the two copies a new rule has to reach (its 'normal' draws by fan_in, ds_ep's
    as N(0, 0.02^2), on purpose)."""
    from maxtext.experimental.mpmd.jax.qwen35 import init as qs

    scfg = qtiny(layers=12)
    layers = tuple(range(12))
    ch = ds_ep.ChunkSpec(
        index=0, layers=layers, first=True, last=True, stage=0, kinds=ds_ep.layer_kinds(scfg.model, layers)
    )
    seen = collections.Counter()
    for path, shape in jax.tree_util.tree_flatten_with_path(ds_ep.chunk_shapes(scfg, ch), is_leaf=ds_ep._is_shape)[0]:
      keys = tuple(str(getattr(k, "key", k)) for k in path)
      want = qs.leaf_rule(keys, shape[1:] if keys[0] in ds_ep.STACKS else shape)[0]
      self.assertEqual(ds_ep.qwen_init_rule(path), want, "/".join(keys))
      seen[want] += 1
    self.assertEqual(set(seen), set(qs.LEAF_KINDS))
    self.assertEqual(ds_ep.A_LOG_RANGE, qs.A_LOG_RANGE)

  def test_deepseek_unchanged(self):
    """DeepSeek's init is what it always was: every 'scale' leaf exactly 1, every other leaf 0.02 x N(0, 1) from
    its own key of split(fold_in(key, chunk), leaves) -- the same draw, to the last bits the jitted program's
    fused erfinv rounds differently from the one here (measured: 1.2e-7)."""
    scfg = tiny(layers=2, fsdp=1, ep=2)
    ch = ds_ep.ChunkSpec(index=3, layers=(6, 7), first=False, last=True, stage=0)
    key = jax.random.key(5)
    got = ds_ep.StagePrograms(scfg, [ch], dsp.stage_meshes(1, 1, 2)[0]).init(key)[0]
    flat, _ = jax.tree_util.tree_flatten_with_path(ds_ep.chunk_shapes(scfg, ch), is_leaf=ds_ep._is_shape)
    ks = jax.random.split(jax.random.fold_in(key, ch.index), len(flat))
    self.assertEqual(sum(leaf_name(path) == "scale" for path, _ in flat), 5)  # 4 norms a layer + the final one
    for k, (path, s), x in zip(ks, flat, jax.tree.leaves(got), strict=True):
      where = jax.tree_util.keystr(path)
      self.assertEqual((x.shape, x.dtype), (s, jnp.dtype(jnp.float32)), where)
      if leaf_name(path) == "scale":
        np.testing.assert_array_equal(np.asarray(x), 1, err_msg=where)
      else:
        np.testing.assert_allclose(
            np.asarray(x), np.asarray(0.02 * jax.random.normal(k, s, jnp.float32)), rtol=1e-6, atol=1e-9, err_msg=where
        )


# ------------------------------------------------------------ compute dtypes
@testing.needs_devices(4)
class QwenComputeDtypes(unittest.TestCase):
  """A Qwen chunk's compute copies (`ds_ep.leaf_dtypes`) with fp32 weights and bf16 compute -- MaxText's own
  defaults for qwen3.8, and the TPU parity runs' setup: bf16 copies, but an fp32 one, bit for bit the master,
  of each leaf MaxText reads at the stored value in fp32 -- the norm scales under norm_numerics 'fp32' (its
  RMSNorm casts the stored scale, in the zero-centred norms and in the GDN gated norm's alike), the routers'
  kernels under float32_gate_logits (GateLogit casts the stored kernel).  The monolith gathers alike.  The (1, 3)
  cut: 14 norm scales (3 in chunk 0's GDN layer; 11 in chunk 1's two GDN layers, its attention layer and the
  final norm) and 4 routers."""

  def pipeline(self, **model):
    scfg = qtiny(layers=4, **testing.stage_axes())
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(**model))
    self.assertEqual((scfg.model.weight_dtype, scfg.model.dtype), (jnp.float32, jnp.bfloat16))
    return dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=1, n_micro=2, global_batch=8, seq=256, layer_split=(1, 3))

  def test_fp32_compute_copies(self):
    """Each stage's gather of the masters (norm scales off their init: 0 and 1 are exact in bf16), leaf by leaf:
    the leaves MaxText reads in fp32 come out fp32 and equal to their masters, every other leaf the master rounded
    to bf16 -- under 'bf16' norm numerics (MaxText before 9a343b3d8) the scales too, as that MaxText rounds them."""
    f32, bf16 = jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)
    for numerics, gate in (("fp32", False), ("bf16", False), ("fp32", True), ("bf16", True)):
      with self.subTest(norm_numerics=numerics, float32_gate_logits=gate):
        p = self.pipeline(norm_numerics=numerics, float32_gate_logits=gate)
        params = perturb_norms(p.init(jax.random.key(0))[0], seed=7)
        n = collections.Counter()
        for ch in p.chunks:
          master = params[ch.stage][p.slot[ch.index]]
          copy = p.stages[ch.stage].gather(params[ch.stage])[p.slot[ch.index]]
          for (path, m), c in zip(jax.tree_util.tree_flatten_with_path(master)[0], jax.tree.leaves(copy), strict=True):
            where = f"chunk {ch.index} {jax.tree_util.keystr(path)}"
            kind = (
                "scale"
                if leaf_name(path) == "scale"
                else "router"
                if where.endswith("['routed_experts']['gate']['kernel']")
                else "other"
            )
            n[kind] += 1
            want = f32 if (kind, numerics) == ("scale", "fp32") or (kind, gate) == ("router", True) else bf16
            self.assertEqual((jnp.dtype(m.dtype), jnp.dtype(c.dtype)), (f32, want), where)
            np.testing.assert_array_equal(np.asarray(c), np.asarray(m).astype(want), err_msg=where)
            if kind == "scale" and want == bf16:
              self.assertFalse(np.array_equal(np.asarray(c, np.float32), np.asarray(m)), where)  # off init
        self.assertEqual((n["scale"], n["router"]), (14, 4))

  def test_reference_gathers_as_the_pipeline(self):
    """`ds_ep_test.reference`, the monolith the pipeline is held to, gathers every leaf in its `leaf_dtypes`
    compute dtype -- the norm scales and the routers' kernels in fp32 here -- so in bf16 it computes what the
    pipeline does.  Traced up to its embedding only."""
    p = self.pipeline(float32_gate_logits=True)
    params = p.init(jax.random.key(0))[0]
    seen, gather = [], ds_ep._gather_leaf

    class Gathered(Exception):
      pass

    def record(x, s, c, d, *a, **k):
      seen.append(jnp.dtype(d))
      return gather(x, s, c, d, *a, **k)

    def stop(*a, **k):
      raise Gathered

    with mock.patch.object(ds_ep, "_gather_leaf", record), mock.patch.object(ds_ep, "embed", stop):
      with self.assertRaises(Gathered):
        reference(p, params, p.synthetic_batch(3))
    want = [d for ch in p.chunks for d in ds_ep.leaf_dtypes(p.scfg, ch, "compute")]
    self.assertEqual(seen, want)
    self.assertEqual(sum(d == jnp.float32 for d in want), 14 + 4)


# ------------------------------------------------------------ one chunk against the port
@testing.needs_devices(4)
class QwenLayerVsPort(unittest.TestCase):
  """One chunk's fixed programs -- `ChunkPrograms.fwd` / `bwd`, its layers in a 2 x 2 shard_map with the ring of
  experts over 'expert' -- against `maxtext.experimental.mpmd.jax.qwen35.model.decoder_layer` run on the whole batch,
  unsharded: the
  output, the input gradient and every weight gradient, each to 1e-4 in max-rel (to its largest magnitude) and in
  L2-rel.  fp32 on XLA's ops, every token on every expert, the norm scales off their init."""

  def check(self, layers):
    scfg = all_experts(qtiny_xla(layers=8))
    cfg = scfg.model
    ch, sp, cp = stage_chunk(scfg, layers)
    self.assertEqual(ch.kinds, tuple("attn" if is_attention(cfg, l) else "gdn" for l in layers))
    params = perturb_norms(sp.init(jax.random.key(0)), seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 128, cfg.emb_dim)).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    wc = sp.gather(params)[0]
    y, res = cp.fwd(wc, act(sp.mesh, x), jnp.int32(0))
    dx, gacc = cp.bwd(wc, act(sp.mesh, x), res, act(sp.mesh, dy), jnp.int32(0))
    pcfg = port_cfg(cfg)

    def port(wl, h):
      pos = jnp.broadcast_to(jnp.arange(h.shape[1], dtype=jnp.int32), h.shape[:2])
      for l, lid in enumerate(layers):
        h = qm.decoder_layer(
            port_layer(ds_ep.layer_weights(wl, l, ch.kinds)), h, pos, None, None, pcfg, is_attention(cfg, lid)
        )
      return h

    want, vjp = jax.vjp(jax.jit(port), jax.tree.map(np.asarray, params[0]["layers"]), x)
    dw_want, dx_want = vjp(dy)
    self.assertEqual(set(gacc), {"layers"})
    errs = {"out": (max_rel(y, want), l2_rel(y, want)), "dx": (max_rel(dx, dx_want), l2_rel(dx, dx_want))}
    for (path, a), b in zip(
        jax.tree_util.tree_flatten_with_path(summed_gacc(gacc)["layers"])[0], jax.tree.leaves(dw_want), strict=True
    ):
      errs["dW" + jax.tree_util.keystr(path)] = (max_rel(a, b), l2_rel(a, b))
    worst = max(errs, key=lambda k: max(errs[k]))
    print(
        f'{ch.kinds}: out {errs["out"][0]:.2g}, dx {errs["dx"][0]:.2g}, worst {worst} {max(errs[worst]):.2g} '
        f"(max-rel, L2-rel)"
    )
    for k, (m, l2) in errs.items():
      self.assertLess(m, 1e-4, k)
      self.assertLess(l2, 1e-4, k)

  def test_gdn_attn_gdn_chunk(self):
    """Layers 2-4: a Gated DeltaNet layer, the block's attention layer, the next block's first GDN layer."""
    self.check((2, 3, 4))

  def test_attention_first_chunk(self):
    """Layers 7-8: a chunk that starts with the attention layer (the second block's), then a GDN layer."""
    self.check((7, 8))


# ------------------------------------------------------------ the MoE half against the port
# Its MoE block's backward trips jaxlib 0.12.0.x's XLA:CPU heap corruption (testing.CPU_HEAP_BUG_JAXLIBS): the file's
# fast tests, 16 processes at once on 8 CPU devices, crashed in 14 of 16 runs with this class (in it each time), in 0
# of 56 without it (8 and 4 devices, 2026-11-25).  The workers earlier rounds lost in QwenPipelineXlaKernels and
# QwenRepeatScope had run this class first.
@testing.trips_cpu_heap_bug
@testing.needs_devices(4)
class QwenMoeVsPort(unittest.TestCase):
  """The MoE half of ds_ep's Qwen layer -- the ring of experts with Qwen's router, plus the sigmoid-gated shared
  expert -- against `maxtext.experimental.mpmd.jax.qwen35.moe.sparse_moe_block` at top-k < E (2 of 8): a one-layer
  Gated DeltaNet chunk
  whose out_proj kernel is zero, so its mixer adds exactly nothing and the layer is x + moe(post_attention_
  layernorm(x)), forward and backward, on a 2 x 2 mesh (ep 2), fp32 on XLA's ops.  The router logits are spread
  wide (std ~3) so that no token's k-th and (k+1)-th logits are near a tie (checked): both sides pick the same
  experts.  x is small, so the residual add rounds at the MoE output's scale, not x's.  fp32 router logits are
  checked in bf16, where the flag changes something."""

  def run_moe(self, **model):
    """-> (scfg, layer weights on the host, x, dy, y, dx, dW) of ds_ep's layer."""
    scfg = qtiny_xla(layers=4, random_routing=False)
    scfg = dataclasses.replace(scfg, random_routing=model.pop("random_routing", False), model=scfg.model.replace(**model))
    ch, sp, cp = stage_chunk(scfg, (0,))

    def edit(path, w):
      keys = [str(getattr(k, "key", k)) for k in path]
      if keys[-2:] == ["out_proj", "kernel"]:
        return jnp.zeros_like(w)
      return 10 * w if keys[-3:] == ["routed_experts", "gate", "kernel"] else w

    params = jax.tree_util.tree_map_with_path(edit, perturb_norms(sp.init(jax.random.key(0)), seed=1))
    rng = np.random.default_rng(3)
    x = (0.01 * rng.standard_normal((4, 128, scfg.model.emb_dim))).astype(scfg.model.dtype)
    dy = rng.standard_normal(x.shape).astype(scfg.model.dtype)
    wc = sp.gather(params)[0]
    y, res = cp.fwd(wc, act(sp.mesh, x), jnp.int32(5))
    dx, gacc = cp.bwd(wc, act(sp.mesh, x), res, act(sp.mesh, dy), jnp.int32(5))
    lw = ds_ep.layer_weights(jax.tree.map(np.asarray, params[0]["layers"]), 0, ch.kinds)
    dw = ds_ep.layer_weights(summed_gacc(gacc)["layers"], 0, ch.kinds)
    return scfg, lw, x, dy, np.asarray(y), np.asarray(dx), dw

  def assert_matches(self, scfg, lw, x, dy, y, dx, dw, moe, what, tol=1e-5):
    """ds_ep's layer (y, dx, dW) against x + moe(lw['mlp'], rms_norm(lw['post_attention_layernorm'], x)) and its
    VJP; the mixer's weights (but out_proj, whose gradient is its input's) and its input norm get none."""
    cfg = port_cfg(scfg.model)

    def port(mlp, post, x):
      return x + moe(mlp, q_layers.rms_norm(post, x, cfg))

    want, vjp = jax.vjp(jax.jit(port), lw["mlp"], lw["post_attention_layernorm"], x)
    d_mlp, d_post, d_x = vjp(dy)
    errs = {"moe": max_rel(y - x, np.asarray(want) - x), "dx": max_rel(dx, d_x)}
    self.assertLess(errs["moe"], tol, f"{what}: moe(x)")
    self.assertLess(errs["dx"], 10 * tol, f"{what}: dx")
    for (path, a), b in zip(
        jax.tree_util.tree_flatten_with_path({"mlp": dw["mlp"], "post": dw["post_attention_layernorm"]})[0],
        jax.tree.leaves({"mlp": d_mlp, "post": d_post}),
        strict=True,
    ):
      errs["dW" + jax.tree_util.keystr(path)] = max_rel(a, b)
      self.assertLess(errs["dW" + jax.tree_util.keystr(path)], 10 * tol, f"{what}: dW{jax.tree_util.keystr(path)}")
    zero = {**{k: v for k, v in dw["gdn"].items() if k != "out_proj"}, "input": dw["input_layernorm"]}
    for path, a in jax.tree_util.tree_flatten_with_path(zero)[0]:
      np.testing.assert_array_equal(a, 0, err_msg=f"{what}: dW{jax.tree_util.keystr(path)}")
    self.assertGreater(np.abs(dw["gdn"]["out_proj"]["kernel"]).max(), 0)
    worst = max((k for k in errs if k.startswith("dW")), key=errs.get)
    print(f'{what}: moe {errs["moe"]:.2g}, dx {errs["dx"]:.2g}, worst {worst} {errs[worst]:.2g} (max-rel)')

  def logit_gaps(self, scfg, lw, x):
    """The smallest gap between a token's k-th and (k+1)-th router logits (float64 on the host)."""
    cfg = scfg.model.replace(dtype=jnp.float32)
    hn = np.asarray(q_layers.rms_norm(lw["post_attention_layernorm"], x, cfg), np.float64).reshape(-1, cfg.emb_dim)
    logits = np.sort(hn @ np.asarray(lw["mlp"]["routed_experts"]["gate"]["kernel"], np.float64), axis=-1)
    k = cfg.num_experts_per_tok
    return float(np.min(logits[:, -k] - logits[:, -k - 1])), float(logits.std())

  def test_topk_matches_port(self):
    """Top-2 of 8 on the raw logits, softmax over the two in fp32, renormalised (norm_topk_prob, the yml's):
    the output to 1e-5 of the MoE's largest magnitude, the input and every MoE weight gradient (router
    included) to 1e-4."""
    scfg, lw, x, dy, y, dx, dw = self.run_moe()
    gap, std = self.logit_gaps(scfg, lw, x)
    self.assertGreater(gap, 1e-4, "a near-tie at the top-k boundary: choose another seed")
    print(f"top-2 of 8: smallest k / k+1 logit gap {gap:.2g} (logit std {std:.2g})")
    cfg = port_cfg(scfg.model)
    self.assert_matches(scfg, lw, x, dy, y, dx, dw, lambda m, h: q_moe.sparse_moe_block(m, h, cfg), "top-2")

  def test_norm_topk_prob_off(self):
    """norm_topk_prob=False matches the port too.  It is the setting that tells where the softmax goes: MaxText's
    (and the port's) softmax over the k selected logits already sums to one, so renormalising is the identity
    up to rounding and both sides give what they give with it on, to fp32 rounding -- where HF's form, a softmax
    over all E then top-k, would leave the selected weights summing to less than one: the port's output moves
    by more than 1e-2 under it, a thousand times the bound."""
    scfg, lw, x, dy, y, dx, dw = self.run_moe(norm_topk_prob=False)
    self.assertGreater(self.logit_gaps(scfg, lw, x)[0], 1e-4)
    pcfg = port_cfg(scfg.model)
    self.assert_matches(scfg, lw, x, dy, y, dx, dw, lambda m, h: q_moe.sparse_moe_block(m, h, pcfg), "norm_topk_prob off")
    *_, y_on, _, _ = self.run_moe(norm_topk_prob=True)
    self.assertLess(max_rel(y - x, y_on - x), 1e-6)
    hn = q_layers.rms_norm(lw["post_attention_layernorm"], x, pcfg)
    on, off = (
        jax.jit(lambda m, h, c=c: q_moe.sparse_moe_block(m, h, c))(lw["mlp"], hn)
        for c in (pcfg.replace(norm_topk_prob=True), pcfg)
    )
    self.assertLess(max_rel(off, on), 1e-6)
    # ... and the flag is read: the renormalisation re-rounds part of the fp32 top-k weights by an ulp or two, so
    # the outputs with it on and off differ in their last bits (25,327 of 131,072 measured); always or never
    # renormalising makes them bitwise equal.
    self.assertGreater(int(np.sum(np.asarray(y) != np.asarray(y_on))), 0, "norm_topk_prob has no effect")

    def hf_route(params, h, config):  # softmax over every expert, then top-k, not renormalised
      logits = h @ params["kernel"].astype(h.dtype)
      w, idx = jax.lax.top_k(jax.nn.softmax(logits.astype(jnp.float32), axis=-1), config.num_experts_per_tok)
      return w.astype(config.dtype), idx

    with mock.patch.object(q_moe, "route", hf_route):
      hf = jax.jit(lambda m, h: q_moe.sparse_moe_block(m, h, pcfg))(lw["mlp"], hn)
    self.assertGreater(max_rel(hf, off), 1e-2)

  def test_float32_gate_logits_in_bf16(self):
    """float32_gate_logits with fp32 weights and bf16 compute (MaxText's qwen3_5 takes it: GateLogit's fp32 logits
    from the stored kernel, top-k and the softmax over the k on them -- the port's `route`, which is MaxText
    ce608b918's qwen3_5 and qwen3.8 bit for bit in bf16, measured at the parity test's shape): ds_ep's layer
    against the port with the flag on is within the flag-off floor (the two EP shards' bf16 reduce of the output),
    and the router kernel's gradient agrees to 1e-4 -- both sides take it through an fp32 matmul, where the
    flag-off copies agree to bf16's ~3e-3; against the port with the flag off the output is a routing flip off."""
    f64 = lambda a: np.asarray(a, np.float64)
    runs = {f32: self.run_moe(dtype=jnp.bfloat16, float32_gate_logits=f32) for f32 in (True, False)}
    scfg, lw, x, dy = runs[True][:4]
    gate = lambda t: t["routed_experts"]["gate"]["kernel"]
    np.testing.assert_array_equal(gate(lw["mlp"]), gate(runs[False][1]["mlp"]))  # one model, two flags
    port = {}
    for f32 in (True, False):
      cfg = port_cfg(scfg.model).replace(float32_gate_logits=f32)

      def layer(mlp, post, h, cfg=cfg):
        return h + q_moe.sparse_moe_block(mlp, q_layers.rms_norm(post, h, cfg), cfg)

      want, vjp = jax.vjp(jax.jit(layer), lw["mlp"], lw["post_attention_layernorm"], x)
      port[f32] = (f64(want), f64(gate(vjp(dy)[0])))
    moe = lambda y, f32: max_rel(f64(y) - f64(x), port[f32][0] - f64(x))
    on, off, cross = moe(runs[True][4], True), moe(runs[False][4], False), moe(runs[True][4], False)
    g_on, g_off = (l2_rel(gate(runs[f32][6]["mlp"]), port[f32][1]) for f32 in (True, False))
    print(
        f"float32_gate_logits, bf16: moe(x) {on:.3g} (flag off: {off:.3g}; on against the port off: {cross:.3g}), "
        f"router gradient {g_on:.3g} (flag off: {g_off:.3g})"
    )
    self.assertLess(on, 2 * off + 1e-3)
    self.assertGreater(cross, 10 * on)
    self.assertLess(g_on, 1e-4)
    self.assertGreater(g_off, 10 * g_on)

  def test_random_routing_takes_the_raw_logits(self):
    """--routing random under a Qwen model: each token's k experts drawn uniformly with replacement from the
    layer's key (`_layer_key`, over its EP group's tokens) and weighted by their RAW router logits, as MaxText's
    random_routing returns before any softmax -- against the port's experts (qwen35 `_ops.routed_experts`) given
    those ids and weights, plus its gated shared expert."""
    scfg, lw, x, dy, y, dx, dw = self.run_moe(random_routing=True)
    cfg = port_cfg(scfg.model)
    K, E, S = cfg.num_experts_per_tok, cfg.num_experts, x.shape[1]
    rows = scfg.ep * x.shape[0] // (scfg.fsdp * scfg.ep)  # an EP group's rows: it gathers them over 'expert'
    idx = jax.random.randint(ds_ep._layer_key(0, 5), (rows * S * K,), 0, E, dtype=jnp.int32).reshape(rows, S, K)

    def moe(mlp, h):
      out = []
      for g in range(scfg.fsdp):  # every group draws the same ids
        hg = h[g * rows : (g + 1) * rows]
        logits = jnp.einsum("bsd,de->bse", hg, mlp["routed_experts"]["gate"]["kernel"])
        routed = q_ops.routed_experts(mlp["routed_experts"], hg, jnp.take_along_axis(logits, idx, -1), idx, cfg)
        gate = q_ops.dense(mlp["shared_expert_gate"], hg, cfg)
        out.append(routed + jax.nn.sigmoid(gate) * q_ops.mlp(mlp["shared_expert"], hg, cfg))
      return jnp.concatenate(out)

    self.assert_matches(scfg, lw, x, dy, y, dx, dw, moe, "random routing")


# ------------------------------------------------------------ the pipeline against the monolith
@testing.needs_devices(4)
class QwenPipelineXlaKernels(unittest.TestCase):
  """The whole pipeline of Qwen3.5-family chunks against `ds_ep_test.reference`, the monolith (one shard_map
  running every layer for every microbatch on one stage mesh, AD's accumulation, lax collectives), on XLA's ops:
  `ragged_dot` experts, the argsort dispatch.  The monolith's layers are
  ds_ep's own, so this holds the pipeline's machinery for chunks of either mixer -- a subtree a layer, the
  plans, the folded gathers and reduce-scatters, the boundary copies, the donated accumulation -- and the layer
  math is QwenLayerVsPort's and QwenAgainstPort's.  fp32 (`against_monolith`): loss 1e-5, every gradient leaf and
  the sum of squares 1e-4, the decay leaves 5e-4."""

  def check(self, overlap, mb, layers=4, vv=2, **kw):
    p = dsp.Pipeline(
        qtiny_xla(layers=layers, **testing.stage_axes()),
        n_stages=2,
        chunks_per_stage=vv,
        n_micro=mb,
        global_batch=4 * mb,
        seq=256,
        overlap=overlap,
        **kw,
    )
    against_monolith(self, p, overlap)
    return p

  @pytest.mark.scheduled_only
  def test_matches_reference(self):
    """Without the fusion plan: four one-layer chunks, three Gated DeltaNet and the attention one."""
    p = self.check(overlap=False, mb=2)
    self.assertEqual([ch.kinds for ch in p.chunks], [("gdn",), ("gdn",), ("gdn",), ("attn",)])

  @pytest.mark.scheduled_only
  def test_overlap_matches_reference(self):
    """With the fusion plan (the default breadth order, weights per visit): gathers folded into forwards and
    backwards, reduce-scatters into backwards."""
    p = self.check(overlap=True, mb=2)
    self.assertTrue(p.pf and p.rs_at)
    self.assertEqual((p.schedule_name, p.fsdp_scope), ("breadth", "repeat"))

  @pytest.mark.scheduled_only
  def test_v_placement(self):
    """V placement's default, breadth with weights per visit: stage 0 holds the first chunk and the head's."""
    p = self.check(overlap=True, mb=2, placement="v")
    self.assertEqual((p.placement, p.schedule_name, p.fsdp_scope, p.place), ("v", "breadth", "repeat", [0, 1, 1, 0]))

  def test_one_microbatch(self):
    """--mb 1: the update gathers chunk 2 too (no op of an earlier visit to gather it in), which its visit then
    releases."""
    p = self.check(overlap=True, mb=1)
    self.assertEqual((p.fsdp_scope, p.pf_fallback, p.gather_slots), ("repeat", [2], [(0, 1), (0,)]))

  @testing.slow
  def test_empty_end_chunks(self):
    """A layer split's 0 at an end: chunk 0 the embedding alone, the last chunk the head alone -- a stage of its own
    for a head that would otherwise make its stage the pipeline's slowest (Qwen3.8's 248320-word head on a GB200) --
    without and with the fusion plan, and under batch split, where an empty chunk runs the plain path (no layer to
    split: ChunkPrograms.split off): the monolith's numbers, and the loss bit for bit the all-layers split's on the
    same weights (`same_loss_as_all_layers`).  A 0 inside the split is refused."""
    for overlap in (False, True):
      with self.subTest(overlap=overlap):
        p = self.check(overlap=overlap, mb=2, vv=3, layer_split=(0, 1, 1, 1, 1, 0))
        self.assertEqual([ch.layers for ch in p.chunks], [(), (0,), (1,), (2,), (3,), ()])
        trees = p.chunk_trees(p.init(jax.random.key(1))[0])
        self.assertEqual((sorted(trees[0]), sorted(trees[-1])), (["token_embedder"], ["decoder_norm", "logits_dense"]))
        self.same_loss_as_all_layers(p)
    with self.subTest("batch split"):
      p = dsp.Pipeline(
          qtiny_xla(layers=4, batch_split=True, **testing.stage_axes()),
          n_stages=2,
          chunks_per_stage=3,
          n_micro=2,
          global_batch=16,
          seq=256,
          layer_split=(0, 1, 1, 1, 1, 0),
      )
      self.assertTrue(p.progs[0].empty and p.progs[-1].empty and not p.progs[1].empty)
      self.assertEqual([pr.split for pr in p.progs], [False, True, True, True, True, False])
      against_monolith(self, p, True)
      self.same_loss_as_all_layers(p)
    for bad in ((1, 1, 0, 1, 1, 0), (0, 2, 0, 1, 1, 0), (0, 1, 1, 1, 2, -1)):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "the last may hold none"):
        dsp.Pipeline(
            qtiny_xla(layers=4, **testing.stage_axes()),
            n_stages=2,
            chunks_per_stage=3,
            n_micro=2,
            global_batch=8,
            seq=256,
            layer_split=bad,
        )

  def same_loss_as_all_layers(self, p):
    """p's loss as `against_monolith` takes it -- init(key 1), synthetic_batch(3), p's loop (the fusion plan's if
    p.overlap) -- bit for bit that of the same weights split a layer a chunk, (1, 1, 1, 1) at V = 2 (`rechunk`: the
    embedding moves into layer 0's chunk, the head into layer 3's).  The loss alone: the gradients agree to rounding
    only.  Over six inits on jax 0.11.1 and 0.12, YNNPACK's fusions on and off, every loss was bit for bit but two
    batch-split ones on 0.12 with them off (1 and 2 ulps; not this init's)."""
    q = dsp.Pipeline(
        p.scfg,
        n_stages=p.S,
        chunks_per_stage=2,
        n_micro=p.M,
        global_batch=p.gbs,
        seq=p.seq,
        overlap=p.overlap,
        layer_split=(1, 1, 1, 1),
    )
    params = p.init(jax.random.key(1))[0]
    losses = []
    for r, w in ((p, params), (q, q.params_from_chunks(rechunk(p, host_trees(p.chunk_trees(params)), q)))):
      batch = r.synthetic_batch(3)
      lacc = (
          r.loss_and_grads_overlap(w, [st.gather_first(x) for st, x in zip(r.stages, w)], batch)[2]
          if r.overlap
          else r.loss_and_grads(w, batch)[1]
      )
      losses.append(float(jnp.sum(lacc)))
    self.assertEqual(losses[0], losses[1], "the all-layers split (1, 1, 1, 1) on the same weights")

  @testing.slow
  def test_cut_inside_a_block(self):
    """Two chunks cut inside the 4-layer block, with and without the fusion plan: (1, 3) -- the head's chunk holds
    two GDN layers and the attention one, its 'layers' tree three one-layer subtrees, '00' and '01' with 'gdn' and
    '02' with 'attn' -- and (3, 1)."""
    for split, kinds in (((1, 3), [("gdn",), ("gdn", "gdn", "attn")]), ((3, 1), [("gdn",) * 3, ("attn",)])):
      for overlap in (False, True):
        with self.subTest(split=split, overlap=overlap):
          p = self.check(overlap=overlap, mb=2, vv=1, layer_split=split)
          self.assertEqual([ch.kinds for ch in p.chunks], kinds)

  @testing.slow
  def test_collective_and_typing_modes(self):
    """Qwen chunks through the pipeline's other program variants: typed stage programs (shard_mode explicit),
    per-microbatch reduce-scatters, collective-permute EP collectives, gathers and reduce-scatters one mesh axis
    at a time, folded gathers pinned by a SPEC of Qwen leaf names, the recomputing fused last chunk, and ep 1 (the
    argsort dispatch with no ring) at as many devices a stage: fsdp 4 x ep 1 (fsdp 2 x ep 1 on 4 devices)."""
    axes = testing.stage_axes()
    cases = [
        ("explicit", dict(shard_mode="explicit"), {}),
        ("microbatch", dict(grad_reduce="microbatch"), {}),
        ("ppermute", dict(ep_collectives="ppermute"), {}),
        ("each", dict(gather_axes="each", rs_axes="each", fsdp_collectives="xla"), {}),
        (
            "pins",
            dict(fsdp_collectives="xla"),
            dict(prefetch_pin="hn=00.mlp.routed_experts.wi_0,00.mlp.routed_experts.wo;xs=00.gdn.out_proj"),
        ),
        ("remat_last", dict(remat_fused_last=True), {}),
        ("ep1", dict(fsdp=axes["fsdp"] * axes["ep"], ep=1, fsdp_collectives="xla"), {}),
    ]
    for what, scfg_kw, kw in cases:
      with self.subTest(what):
        p = dsp.Pipeline(
            qtiny_xla(layers=4, **{**axes, **scfg_kw}),
            n_stages=2,
            chunks_per_stage=2,
            n_micro=2,
            global_batch=8,
            seq=256,
            **kw,
        )
        if what == "pins":
          self.assertTrue(p.pins)
        against_monolith(self, p, True)

  @testing.slow
  def test_layer_splits_of_two_blocks(self):
    """8 layers in four chunks cut inside the blocks -- (3, 1, 3, 1): a GDN run and an attention layer per stage;
    (1, 3, 3, 1): chunks spanning a block boundary -- circular and V, with and without the fusion plan."""
    for split, placement in (((3, 1, 3, 1), "circular"), ((1, 3, 3, 1), "circular"), ((1, 3, 3, 1), "v")):
      for overlap in (False, True):
        with self.subTest(split=split, placement=placement, overlap=overlap):
          p = self.check(overlap=overlap, mb=2, layers=8, layer_split=split, placement=placement)
          self.assertEqual([len(ch.layers) for ch in p.chunks], list(split))


# ------------------------------------------------------------------ batch split
@testing.needs_devices(4)
class QwenBatchSplit(unittest.TestCase):
  """--batch-split on Qwen3.5-family chunks (StageConfig.batch_split): each device's rows of a microbatch run as
  two halves through a chunk's layers in Lineage's phases (ds_ep.split_layers), with the layer's mixer -- Gated
  DeltaNet or gated attention, `qwen_mixer_part` -- where DeepSeek's attention runs, so one half's EP all-gather or
  reduce-scatter has the other half's mixer or experts beside it; a backward is the VJP of the whole split chunk.
  fp32 on XLA's ops, at `against_monolith`'s bounds.  Top-k routing routes a token alike in either half, so
  there the split step is the unsplit step's to rounding; random routing draws each half's experts with a key of
  its own, which `reference` follows."""

  def check(self, overlap=True, unsplit_reference=False, layers=4, vv=2, mb=2, **kw):
    pkw = {k: kw.pop(k) for k in ("schedule", "placement", "layer_split", "pair_stages") if k in kw}
    scfg = qtiny_xla(layers=layers, batch_split=True, **testing.stage_axes(), **kw)
    # 8 rows a microbatch: over fsdp 2 x ep 2 2 rows a device, a row a half; over fsdp 1 x ep 2 4, two a half
    p = dsp.Pipeline(
        scfg, n_stages=2, chunks_per_stage=vv, n_micro=mb, global_batch=8 * mb, seq=256, overlap=overlap, **pkw
    )
    ref = None
    if unsplit_reference:
      ref = types.SimpleNamespace(
          **{k: getattr(p, k) for k in ("meshes", "chunks", "slot", "gbs", "seq", "M")},
          scfg=dataclasses.replace(scfg, batch_split=False),
      )
    against_monolith(self, p, overlap, reference_of=ref)
    return p

  def test_matches_unsplit_reference(self):
    """Top-k routing, one-layer chunks (three Gated DeltaNet, then the attention one), the default plan (breadth,
    weights per visit): the split step is the unsplit monolith's to rounding."""
    p = self.check(random_routing=False, unsplit_reference=True)
    self.assertEqual([ch.kinds for ch in p.chunks], [("gdn",), ("gdn",), ("gdn",), ("attn",)])
    self.assertEqual((p.schedule_name, p.fsdp_scope), ("breadth", "repeat"))

  @pytest.mark.scheduled_only
  def test_random_routing(self):
    """Random routing, each half's experts drawn with a key of its own: two chunks cut inside the block (1 + 3:
    the head's chunk holds two GDN layers and the attention one, so mb0's mixer of layer l + 1 runs beside mb1's
    combine of layer l for either mixer), with and without the fusion plan."""
    for overlap in (True, False):
      with self.subTest(overlap=overlap):
        p = self.check(overlap=overlap, vv=1, layer_split=(1, 3))
        self.assertEqual([ch.kinds for ch in p.chunks], [("gdn",), ("gdn", "gdn", "attn")])

  @testing.slow
  def test_interleaved_pairs(self):
    """Interleaved 1F1B with its steady state's F&B pairs (each one program of two batch-split chunks' parts) and
    the split step: the same numbers as the monolith's."""
    p = self.check(mb=4, schedule="interleaved", pair_stages="all")
    self.assertTrue(p.fuse)
    self.assertTrue(p.schedule_name.startswith("interleaved"))

  def test_phases(self):
    """A split forward closes four phases a layer with an optimization barrier each, for either mixer, and names
    each half's EP collectives; a backward reads x and, in chunk 0, the embedding only."""
    p = pipe(qtiny_xla(layers=8, batch_split=True, **testing.stage_axes()), n_micro=2, global_batch=16, fsdp_scope="step")
    self.assertEqual([ch.kinds for ch in p.chunks[:2]], [("gdn", "gdn"), ("gdn", "attn")])
    self.assertEqual([pr.n_res for pr in p.progs], [1, 0, 0, 0])
    op = next(o for o in p.ops() if (o.c, o.kind) == (1, "F"))
    self.assertEqual(_count_eqns(_program_jaxpr(p, op), "optimization_barrier"), 4 * 2)
    text = p.progs[1].program(op.desc).trace(*p._desc_avals(1, op.desc)).lower().as_text(debug_info=True)
    for scope in ("ep_all_gather/mb0", "ep_all_gather/mb1", "ep_reduce_scatter/mb0", "ep_reduce_scatter/mb1"):
      self.assertIn(scope, text)


# ------------------------------------------------------------ the split backward (I and W)
@pytest.mark.scheduled_only
@testing.needs_devices(4)
class QwenSplitBackward(unittest.TestCase):
  """ds_ep_test.SplitBackward for a Qwen3.5-family chunk (ds_ep_test.check_split): a Gated DeltaNet layer, then the
  gated attention one, on one FSDP 2 x EP 2 stage mesh -- I then W against bwd, bwda and the B part with rs_self,
  bit for bit under XLA's default options and with excess precision off, and W the weight gradients' flops.  And the
  fused last chunk cut into AI and W (ds_ep_test.SplitLast: check_split_last)."""

  @staticmethod
  def chunk(scfg, layers=(2, 3)):
    return ds_ep.ChunkSpec(1, layers, False, False, 0, kinds=ds_ep.layer_kinds(scfg.model, layers))

  def test_gdn_attention(self):
    """The benchmark's kernels: megablox, the ragged sort, tokamax splash; bf16."""
    scfg = qtiny(layers=4, fsdp=2, ep=2)
    pr, _, _ = check_split(self, scfg, self.chunk(scfg))
    self.assertEqual(pr.ch.kinds, ("gdn", "attn"))

  def test_xla_kernels(self):
    """XLA's ops in fp32 (qtiny_xla): ragged_dot experts after the argsort dispatch."""
    scfg = qtiny_xla(layers=4, fsdp=2, ep=2)
    check_split(self, scfg, self.chunk(scfg))

  def test_top_k_routing(self):
    """Top-k routing: a softmax over each token's k logits in fp32, renormalised in bf16 (norm_topk_prob)."""
    scfg = qtiny(layers=4, fsdp=2, ep=2, random_routing=False)
    check_split(self, scfg, self.chunk(scfg))

  def test_batch_split(self):
    """Batch split (the family's own, split_layers): each mixer in the phase DeepSeek's attention runs in."""
    scfg = qtiny(layers=4, fsdp=2, ep=2, batch_split=True)
    check_split(self, scfg, self.chunk(scfg), mbs=8)

  def test_last_chunk(self):
    """The fused last chunk (a Gated DeltaNet layer, then the gated attention one, and the zero-centred final norm's
    head) cut into AI and W for --zb-split-last (ds_ep_test.check_split_last), in bf16 on the benchmark's kernels and
    in fp32 on XLA's ops."""
    for scfg in (qtiny(layers=4, fsdp=2, ep=2), qtiny_xla(layers=4, fsdp=2, ep=2)):
      with self.subTest(gmm_impl=scfg.gmm_impl):
        ch = ds_ep.ChunkSpec(2, (2, 3), False, True, 0, kinds=ds_ep.layer_kinds(scfg.model, (2, 3)))
        check_split_last(self, scfg, ch)


@pytest.mark.scheduled_only
@testing.needs_devices(4)
@testing.needs_ynn_off(
    "XLA:CPU's YNNPACK fusions compile a Qwen3.5-family chunk's split backward otherwise than its fused one, a "
    "gradient one rounding apart (QwenSplitBackward compiles its programs with them off)"
)
class QwenZeroBubble(unittest.TestCase):
  """zbh1 and zbh2 on Qwen3.5-family chunks against 1F1B, bit for bit (ds_ep_test.check_zero_bubble: the loss
  partials, every reduced gradient, the stage sumsqs, two AdamW steps): four stages of two layers -- the middle
  chunks, both split, a Gated DeltaNet layer then the gated attention one, and two Gated DeltaNet layers -- of EP 2
  where there are 8 devices (else EP 1), on the benchmark's kernels in bf16 (megablox, the ragged sort, tokamax
  splash), with batch split (the family's own, split_layers), and with the last chunk split too (zb_split_last)."""

  def test_gdn_attention(self):
    p = check_zero_bubble(self, qtiny(layers=8, fsdp=1, ep=zb_ep(4)))[0]
    self.assertEqual([ch.kinds for ch in p.chunks[1:3]], [("gdn", "attn"), ("gdn", "gdn")])

  def test_batch_split(self):
    check_zero_bubble(self, qtiny(layers=8, fsdp=1, ep=zb_ep(4), batch_split=True))

  def test_split_last(self):
    """The last chunk split too (zb_split_last): at four stages, every chunk but the first split."""
    variants = [(name, {"zb_split_last": True}) for name in dsp.ZB_SCHEDULES]
    check_zero_bubble(self, qtiny(layers=8, fsdp=1, ep=zb_ep(4)), variants=variants)


# ------------------------------------------------------------ the pipeline against the port
@testing.needs_devices(4)
class QwenAgainstPort(unittest.TestCase):
  """The whole pipeline against `maxtext.experimental.mpmd.jax.qwen35` -- the MaxText-parity port, which shares none
  of the pipeline's
  hot path: MaxText's sparse_matmul experts (`ragged_dot`) instead of the ring of experts, the reference
  attention instead of tokamax splash, a gather embedding, one device, no shard_map, no pipeline -- for the loss
  and every gradient.  fp32 on XLA's ops, every token on every expert, the norm scales off their init."""

  def test_matches_qwen35_port(self):
    """Two chunks cut inside the block (1, 3), two microbatches: loss to 1e-5, every gradient leaf to 1e-4
    (L2-rel; A_log and dt_bias, `leaf_bound`, to 5e-4), whole-model cosine 0.999999."""
    self.check(layers=4, chunks_per_stage=1, layer_split=(1, 3))

  @pytest.mark.scheduled_only
  def test_two_blocks(self):
    """Two blocks in four chunks cut inside them, (3, 1, 3, 1) -- the port's scan over two blocks, its second
    block's layers at global ids 4-7 -- to the same bounds."""
    self.check(layers=8, chunks_per_stage=2, layer_split=(3, 1, 3, 1))

  def test_port_runs_the_pipelines_gdn(self):
    """The port runs the Gated DeltaNet the pipeline runs: `port_loss_and_grads` passes all-one segment ids
    (`port_batch`) and `port_cfg` keeps them out of the GDN, whose pipeline counterpart takes none
    (`ds_ep.qwen_mixer_part`) -- the plain conv and delta rule, not the packed ones (`segmented_causal_conv`,
    `_segment_masks`), a program of other roundings."""
    cfg = port_cfg(qtiny_xla().model)
    tokens = np.zeros((2, cfg.max_target_length), np.int32)
    traced = collections.Counter()

    def spy(name):
      real = getattr(q_gdn, name)

      def counted(*a, **k):
        traced[name] += 1
        return real(*a, **k)

      return mock.patch.object(q_gdn, name, counted)

    with spy("causal_depthwise_conv"), spy("segmented_causal_conv"), spy("_segment_masks"):
      jax.eval_shape(lambda pr, b: qt.compute_loss(pr, b, cfg)[0], qm.param_avals(cfg), port_batch(tokens, tokens))
    self.assertGreater(traced["causal_depthwise_conv"], 0, traced)
    self.assertEqual((traced["segmented_causal_conv"], traced["_segment_masks"]), (0, 0), traced)

  def check(self, layers, **kw):
    scfg = all_experts(qtiny_xla(layers=layers, **testing.stage_axes()))
    p = dsp.Pipeline(scfg, n_stages=2, n_micro=2, global_batch=8, seq=256, **kw)
    params, _, _ = p.init(jax.random.key(0))
    params = perturb_norms(params, seed=7)
    wc0 = [st.gather_first(w) for st, w in zip(p.stages, params)]
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
    targets = np.roll(tokens, -1, axis=1)
    grads, _, lacc = p.loss_and_grads_overlap(params, wc0, p.put_batch(tokens, targets))
    loss = float(jnp.sum(lacc))
    host = lambda trees: [jax.tree.map(np.asarray, t) for t in trees]
    cfg = port_cfg(scfg.model)
    ref_loss, ref = port_loss_and_grads(cfg, port_tree(p, host(p.chunk_trees(params))), tokens, targets)
    got = port_tree(p, host(p.chunk_trees(grads)))
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(qm.param_avals(cfg)))
    rows, cos, ratio = tree_errors(got, ref)
    rest = [r for r in rows if not r[1].endswith(DECAY_LEAVES)]
    print(
        f"port: loss {loss:.7f} ref {ref_loss:.7f} cosine {cos:.10f} norm ratio {ratio:.7f} worst leaf "
        f"{rows[0][0]:.3g} at {rows[0][1]}, of the rest {rest[0][0]:.3g} at {rest[0][1]}"
    )
    self.assertLess(abs(loss - ref_loss) / abs(ref_loss), 1e-5)
    for e, path in rows:
      self.assertLess(e, leaf_bound(path), path)
    self.assertGreater(cos, 0.999999)


# ------------------------------------------------------------ the repeat scope, bit for bit
@testing.needs_devices(4)
class QwenRepeatScope(unittest.TestCase):
  """--fsdp-scope repeat on Qwen chunks: a chunk's gathered weights live one visit at a time."""

  def test_bitwise_vs_step(self):
    """The step scope's numbers bit for bit (`ds_ep_test.RepeatScope.bitwise`): one microbatch loop's reduced
    gradients, the stage sumsqs and the loss, then two training steps' losses, gradient norms and params -- at
    V=2, M=2, where the turnaround visit's backwards and the fused last chunk host the re-gathers."""
    scfg = qtiny_xla(layers=4, fsdp=1, ep=2)
    p = pipe(scfg, n_micro=2, global_batch=4, schedule="breadth", fsdp_scope="repeat")
    q = pipe(scfg, n_micro=2, global_batch=4, schedule="breadth", fsdp_scope="step")
    self.assertEqual(max(check_repeat_plan(self, p)), 2)
    self.assertEqual({p.order[j][1] + ("A" if p.order[j][0] == p.C - 1 else "") for j in p.pf}, {"F", "B", "FA"})
    assert_same_run(self, mode_run(p, 1, 2), mode_run(q, 1, 2), "Qwen V=2 M=2")


# ------------------------------------------------------------------------ CLI
@testing.needs_devices(4)
class QwenCli(unittest.TestCase):
  """`--model qwen3.8-2.4t-a95b` on the command line: the plan alone (--dry-run), and a whole run that cannot
  import MaxText."""

  @testing.needs_devices(4)
  def test_dry_run(self):
    """`main --dry-run` names the family and its layer kinds in the summary and records --model in the config --
    at Q_TINY_ARGS, at Qwen3.8's full dims (92 layers, 23 of them attention; nothing but the plan is built), and
    DeepSeek's line as it was."""
    with contextlib.redirect_stdout(io.StringIO()) as log:
      out = dsp.main(Q_TINY_ARGS + ["--dry-run"])
    line = "MPMD Qwen3.5 family: 4 MoE layers (3 Gated DeltaNet, 1 gated attention), 2 chunks (V=1, layers [1, 3])"
    self.assertIn(line, out["summary"])
    self.assertIn(line, log.getvalue())
    self.assertEqual(
        (out["config"]["model"], out["config"]["kv_heads"], out["config"]["q_lora"]), ("qwen3.8-2.4t-a95b", 2, None)
    )
    fsdp = testing.stage_axes()["fsdp"]  # two stages of the devices there are
    stages = ["--fsdp", str(fsdp), "--ep", "2"]
    with contextlib.redirect_stdout(io.StringIO()):
      full = dsp.main(["--model", "qwen3.8-2.4t-a95b", "--gmm-impl", "ragged_dot", "--dry-run", *stages])
      ds = dsp.main(["--gmm-impl", "ragged_dot", "--dry-run", *stages])
    self.assertIn(
        "MPMD Qwen3.5 family: 92 MoE layers (69 Gated DeltaNet, 23 gated attention), 4 chunks (V=2), 2 "
        f"stages x (fsdp={fsdp}, ep=2), M=8 x mbs=4 x seq=4096",
        full["summary"],
    )
    self.assertIn("MPMD DeepSeek sparse: 4 MoE layers, 4 chunks (V=2), 2 stages", ds["summary"])
    self.assertEqual(ds["config"]["model"], "deepseek3")


# ------------------------------------------------------------ MaxText's kernels, bf16
def port_floor(cfg, params, tokens, targets, n_micro, whole=None):
  """The port against itself: its gradients over the whole batch (`whole`, if computed already) against the mean
  of its per-microbatch ones -- the same numbers summed in another order, which is all a correct pipeline may
  change (ds_ep_parity.port_floor) -> [(L2-rel error, path)], worst first."""
  if whole is None:
    _, whole = port_loss_and_grads(cfg, params, tokens, targets)
  mbs = len(tokens) // n_micro
  parts = [
      port_loss_and_grads(cfg, params, tokens[m * mbs : (m + 1) * mbs], targets[m * mbs : (m + 1) * mbs])[1]
      for m in range(n_micro)
  ]
  return tree_errors(jax.tree.map(lambda *g: sum(np.asarray(x, np.float64) for x in g) / n_micro, *parts), whole)[0]


def rechunk(p, trees, q):
  """Chunk trees of pipeline `p` -> the same model's chunk trees for pipeline `q` (another layer split or
  placement): each layer's weights (`ds_ep.layer_weights`) restacked for q's chunks (`ds_ep.stack_layers`)."""
  per = {
      lid: ds_ep.layer_weights(t["layers"], l, ch.kinds)
      for ch, t in zip(p.chunks, trees)
      for l, lid in enumerate(ch.layers)
  }
  out = []
  for ch in q.chunks:
    t = {"layers": jax.tree.map(np.asarray, ds_ep.stack_layers([per[l] for l in ch.layers], ch.kinds))}
    if ch.first:
      t["token_embedder"] = trees[0]["token_embedder"]
    if ch.last:
      t["decoder_norm"], t["logits_dense"] = trees[-1]["decoder_norm"], trees[-1]["logits_dense"]
    out.append(t)
  return out


def host_trees(trees):
  return [jax.tree.map(np.asarray, t) for t in trees]


@pytest.mark.scheduled_only
@testing.needs_devices(4)
class QwenMaxTextKernels(unittest.TestCase):
  """The benchmark's kernels in bf16 -- MaxText's megablox experts and SparseCore ragged sort (interpret mode),
  tokamax splash -- against the port and against the monolith, on one model: the same weights (norm scales off
  init), batch and routing (every token on every expert).

  bf16 rounds the Gated DeltaNet's decay gradients far more than anything of DeepSeek's: A_log's and dt_bias's
  each sum a gated term over every token, with heavy cancellation, so two correct bf16 implementations differ
  there by up to 0.34 L2-rel (seeds 0-7 of these weights and batch, jax 0.11.1 and 0.11.2), eleven times
  DeepSeek's 3e-2 bound, and the other leaves by up to 3.4e-2 (norm scales, the shared expert's gate).  So the
  bounds are calibrated here, from floors: the worst leaf any of three changes that keep the math -- the port's
  batch in two microbatches, the port's attention as tokamax splash, the monolith's two microbatches as one --
  moves, taken apart for the two decay leaves (A_log, dt_bias) and for every other one; a decay leaf is held to
  max(3e-2, 3 x the decay floor), any other leaf to max(3e-2, 3 x the worst other leaf's floor), so a leaf of
  the experts, the router or attention is not let off by the decay leaves' rounding.
  The loss, sum-of-squares and whole-model cosine bounds the floor does not loosen (the cosine came to 0.99974
  or more against the port, 0.99983 or more against the monolith)."""

  @classmethod
  def setUpClass(cls):
    scfg = all_experts(qtiny(layers=4, **testing.stage_axes()))
    cls.p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=1, n_micro=2, global_batch=8, seq=256, layer_split=(1, 3))
    params, _, _ = cls.p.init(jax.random.key(0))
    cls.params = perturb_norms(params, seed=7)
    rng = np.random.default_rng(0)
    cls.tokens = rng.integers(0, scfg.model.vocab_size, size=(cls.p.gbs, cls.p.seq), dtype=np.int32)
    cls.targets = np.roll(cls.tokens, -1, axis=1)
    cls.trees = host_trees(cls.p.chunk_trees(cls.params))
    cfg, port_params, toks = port_cfg(scfg.model), port_tree(cls.p, cls.trees), (cls.tokens, cls.targets)
    cls.port_params = port_params
    cls.port_loss, cls.port = port_loss_and_grads(cfg, port_params, *toks)
    floors = {
        "the port in two microbatches": port_floor(cfg, port_params, *toks, 2, whole=cls.port),
        "the port with tokamax splash": tree_errors(
            port_loss_and_grads(cfg.replace(attention_kernel="tokamax_splash"), port_params, *toks)[1], cls.port
        )[0],
    }
    # the monolith of the default circular pipeline, at two microbatches (test_matches_reference's) and at one
    cls.q = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=8, seq=256)
    cls.q_params = cls.q.params_from_chunks(rechunk(cls.p, cls.trees, cls.q))
    cls.mono_loss, mono = reference(cls.q, cls.q_params, cls.q.put_batch(*toks))
    cls.mono = host_trees(mono)
    one = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=1, global_batch=8, seq=256)
    _, mono1 = reference(one, one.params_from_chunks(rechunk(cls.p, cls.trees, one)), one.put_batch(*toks))
    floors["the monolith in one microbatch"] = tree_errors(host_trees(mono1), cls.mono)[0]
    worst = lambda decay: max(e for rows in floors.values() for e, path in rows if path.endswith(DECAY_LEAVES) == decay)
    cls.tol_decay, cls.tol_other = max(3e-2, 3 * worst(True)), max(3e-2, 3 * worst(False))
    print(
        "bf16 floors: "
        + "; ".join(f"{what}: {rows[0][0]:.3g} at {rows[0][1]}" for what, rows in floors.items())
        + f" -> leaf bounds {cls.tol_decay:.3g} (A_log, dt_bias), {cls.tol_other:.3g} (every other leaf)"
    )

  def assert_leaves(self, rows):
    """Every leaf within its bound: the decay leaves' or the others' (setUpClass)."""
    for e, path in rows:
      self.assertLess(e, self.tol_decay if path.endswith(DECAY_LEAVES) else self.tol_other, path)

  def test_matches_port(self):
    """The pipeline (1 + 3 layers, two microbatches) against the port in bf16 (reference attention, `ragged_dot`
    experts): loss 1e-4, every leaf within its floor bound (`assert_leaves`), cosine 0.999, norm ratio 1 +- 1e-2."""
    p = self.p
    wc0 = [st.gather_first(w) for st, w in zip(p.stages, self.params)]
    grads, _, lacc = p.loss_and_grads_overlap(self.params, wc0, p.put_batch(self.tokens, self.targets))
    loss = float(jnp.sum(lacc))
    rows, cos, ratio = tree_errors(port_tree(p, host_trees(p.chunk_trees(grads))), self.port)
    print(
        f"bf16 port: loss {loss:.6f} ref {self.port_loss:.6f} cosine {cos:.7f} norm ratio {ratio:.5f} worst leaf "
        f"{rows[0][0]:.3g} at {rows[0][1]}"
    )
    self.assertLess(abs(loss - self.port_loss) / abs(self.port_loss), 1e-4)
    self.assert_leaves(rows)
    self.assertGreater(cos, 0.999)
    self.assertLess(abs(ratio - 1), 1e-2)

  def test_matches_reference(self):
    """The default circular pipeline -- four one-layer chunks; breadth, weights per visit, gathers and
    reduce-scatters folded in; two microbatches -- against the monolith on the same kernels: loss 2e-4, every
    leaf within its floor bound (`assert_leaves`), the stage sumsqs' total 3e-3 of the gradient's, cosine 0.999."""
    q = self.q
    self.assertTrue(q.pf and q.rs_at)
    self.assertEqual((q.schedule_name, q.fsdp_scope), ("breadth", "repeat"))
    wc0 = [st.gather_first(w) for st, w in zip(q.stages, self.q_params)]
    grads, sqs, lacc = q.loss_and_grads_overlap(self.q_params, wc0, q.put_batch(self.tokens, self.targets))
    rows, cos, ratio = tree_errors(host_trees(q.chunk_trees(grads)), self.mono)
    loss, ref_loss = float(jnp.sum(lacc)), float(self.mono_loss)
    ref_sq = sum(float(np.sum(np.square(np.asarray(x, np.float64)))) for x in jax.tree.leaves(self.mono))
    sq_err = abs(sum(float(s) for s in sqs) - ref_sq) / ref_sq
    print(
        f"bf16 monolith: loss {loss:.6f} ref {ref_loss:.6f} cosine {cos:.7f} sumsq rel {sq_err:.2g} worst leaf "
        f"{rows[0][0]:.3g} at chunk {rows[0][1]}"
    )
    self.assertLess(abs(loss - ref_loss) / abs(ref_loss), 2e-4)
    self.assert_leaves(rows)
    self.assertLess(sq_err, 3e-3)
    self.assertGreater(cos, 0.999)

  @testing.slow
  def test_batch_split_matches_reference(self):
    """--batch-split on the benchmark's kernels (megablox, the SparseCore ragged sort, tokamax splash; bf16) and
    test_matches_reference's plan, on the same weights and the batch twice over (two rows a device, a row a half):
    against the monolith running each layer on each half in turn (`reference` under batch split), at
    test_matches_reference's bounds -- the leaves within their floor bounds (`assert_leaves`)."""
    q = self.q
    split = dsp.Pipeline(
        dataclasses.replace(q.scfg, batch_split=True), n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=16, seq=256
    )
    self.assertEqual((split.schedule_name, split.fsdp_scope), ("breadth", "repeat"))
    params = split.params_from_chunks(rechunk(self.p, self.trees, split))
    toks = (np.concatenate([self.tokens] * 2), np.concatenate([self.targets] * 2))
    batch = split.put_batch(*toks)
    wc0 = [st.gather_first(w) for st, w in zip(split.stages, params)]
    grads, sqs, lacc = split.loss_and_grads_overlap(params, wc0, batch)
    ref_loss, ref = reference(split, params, batch)
    ref = host_trees(ref)
    rows, cos, _ = tree_errors(host_trees(split.chunk_trees(grads)), ref)
    loss, ref_loss = float(jnp.sum(lacc)), float(ref_loss)
    ref_sq = sum(float(np.sum(np.square(np.asarray(x, np.float64)))) for x in jax.tree.leaves(ref))
    sq_err = abs(sum(float(x) for x in sqs) - ref_sq) / ref_sq
    print(
        f"bf16 batch split: loss {loss:.6f} ref {ref_loss:.6f} cosine {cos:.7f} sumsq rel {sq_err:.2g} worst leaf "
        f"{rows[0][0]:.3g} at chunk {rows[0][1]}"
    )
    self.assertLess(abs(loss - ref_loss) / abs(ref_loss), 2e-4)
    self.assert_leaves(rows)
    self.assertLess(sq_err, 3e-3)
    self.assertGreater(cos, 0.999)


class QwenDispatchVsMaxText(unittest.TestCase):
  """What `_check_qwen`'s moe_use_direct_token_gather refusal rests on, on MaxText's own kernels in bf16: the
  ragged sort's backward sums dx over a token's k copies as MaxText's direct gather does, and the argsort dispatch
  here (repeat + sort, MaxText's without the flag) does not -- so the flag is refused with use_ragged_sort off only,
  and at ep 1 the ragged sort is MaxText's moe_use_direct_token_gather=true."""

  def test_ragged_sort_is_the_direct_gather(self):
    """ring_ragged_sort at ep 1 in a one-device shard_map, as moe_experts calls it, against MaxText's direct gather
    (sort_activations.route, which its argsort dispatch runs under moe_use_direct_token_gather): the same rows, and
    dx bit for bit -- both sum a token's k copies in fp32 and round once.  moe_experts' argsort dispatch (at ep 1:
    the ids not rolled) gives the same rows, but its dx, summed in bf16, differs (42,942 of 131,072 elements, by up
    to 0.0625, on jax 0.11.1).  Top-4 of 8: at top-2 the two agree, as two bf16 terms add exactly in fp32 and
    either sum rounds that once."""
    from maxtext.kernels import sort_activations
    from maxtext.kernels.ragged import ragged_sort
    from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe

    T, D, E, K = 512, 256, 8, 4
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((T, D)), jnp.bfloat16)
    idx = jnp.asarray(np.stack([rng.permutation(E)[:K] for _ in range(T)]), jnp.int32)
    g = jnp.asarray(rng.standard_normal((T * K, D)), jnp.bfloat16)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ds_ep.FE)
    ragged = jax.shard_map(
        lambda x, i: ragged_sort.ring_ragged_sort(x, i, E, K, ds_ep.EP_AXIS, 1)[0],
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    dispatch = {
        "ragged sort": lambda x: ragged(x, idx),
        "direct gather": lambda x: sort_activations.route(x, idx, False),
        "argsort": lambda x: ds_moe._sort_activations(jnp.repeat(x, K, axis=0), jnp.argsort(idx.reshape(-1))),
    }
    y, dx = {}, {}
    for name, f in dispatch.items():
      out, vjp = jax.vjp(jax.jit(f), x)
      d = vjp(g)[0]
      self.assertEqual((out.dtype, d.dtype), (jnp.bfloat16, jnp.bfloat16), name)
      y[name], dx[name] = np.asarray(out, np.float32), np.asarray(d, np.float32)
    for name in ("ragged sort", "argsort"):
      np.testing.assert_array_equal(y[name], y["direct gather"], err_msg=f"{name}: the rows")
    np.testing.assert_array_equal(dx["ragged sort"], dx["direct gather"], err_msg="ragged sort: dx")
    differ = int(np.sum(dx["argsort"] != dx["direct gather"]))
    print(
        f"the argsort dispatch's dx against the direct gather's: {differ} of {dx['argsort'].size} elements differ, "
        f"by up to {np.abs(dx['argsort'] - dx['direct gather']).max():.3g}"
    )
    self.assertGreater(differ, dx["argsort"].size // 10)
