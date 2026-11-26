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

"""CPU tests of the MPMD pipeline on DeepSeek-V3's whole architecture: dense layers, the routed bias and its update,
group-limited routing, the fp32 router, the sequence-wise aux loss and multi-token prediction (MaxText's
deepseek3-671b with the MLPerf recipe's routing, aux loss and MTP).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/exact_dsv3_test.py -q

A tiny model with every feature (2 dense layers, then MoE layers with a routed bias and an fp32 router, 4 routing
groups of 2 experts, the sequence-wise aux loss, one MTP layer) runs through the pipeline -- dense, mixed and MoE
chunks, the fused last chunk with MTP -- and is held to
one monolithic program over the same weights (`reference`): the loss, every gradient (the tied token embedding's, the
sum of chunk 0's table's and MTP's copy's), the routers' expert counts in their bias slots, the gradient norm.  fp32
compute and the pure-JAX kernels (ragged_dot experts, the argsort dispatch), so the two agree to rounding.
"""

import contextlib
import dataclasses
import io
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from tests.unit.experimental.mpmd.jax import ds_ep_test as dt
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer

RATE = 0.01


def setUpModule():
  testing.require_cpu()


def exact_scfg(layers=5, dense=2, **kw):
  """tiny_xla with DeepSeek-V3's features: `dense` dense layers (a wider MLP), a routed bias updated at RATE,
  4 groups of 2 of the 8 experts with the best 2 groups kept, one MTP layer; top-k routing.  Two stages of the
  devices there are (testing.stage_axes)."""
  scfg = dt.tiny_xla(layers, random_routing=False, **testing.stage_axes())
  feats = dict(
      first_num_dense_layers=dense,
      mlp_dim=384,
      routed_bias=True,
      routed_bias_update_rate=RATE,
      n_routing_groups=4,
      topk_routing_group=2,
      mtp_num_layers=1,
      float32_gate_logits=True,
      load_balance_loss_weight=0.01,
      moe_use_megatron_seq_aux_loss=True,
  )
  return dataclasses.replace(scfg, model=scfg.model.replace(**{**feats, **kw}))


def pipeline(scfg, schedule="breadth", layer_split=(1, 2, 1, 1), **kw):
  return dsp.Pipeline(
      scfg,
      **{
          **dict(
              n_stages=2,
              chunks_per_stage=2,
              n_micro=4,
              global_batch=16,
              seq=256,
              schedule=schedule,
              layer_split=layer_split,
          ),
          **kw,
      },
  )


def reference(p, params, batch):
  """(loss, gradients in the storage layout per chunk, expert counts per chunk) of one monolithic shard_map over
  stage 0's mesh, from ds_ep's layer functions: every microbatch through every layer in order (dense first), the
  head, and MTP after it; each MoE layer's aux loss rides its output (aux_seqs: the global batch).  The counts:
  each MoE layer's (and the MTP layer's), summed over the microbatches and psum'd over the stage, as the pipeline's
  reduce leaves its bias slots."""
  scfg, chunks, mesh, M = p.scfg, p.chunks, p.meshes[0], p.M
  cfg = scfg.model
  storage = [params[ch.stage][p.slot[ch.index]] for ch in chunks]
  storage = [jax.device_put(w, jax.tree.map(lambda x: NamedSharding(mesh, x.sharding.spec), w)) for w in storage]
  toks, tgts = batch[:2]
  tok_sh = NamedSharding(mesh, P(ds_ep.FE, None))
  toks, tgts = ([jax.device_put(t, tok_sh) for t in a] for a in (toks, tgts))
  sspec = [ds_ep.chunk_storage_specs(c) for c in chunks]
  cspec = [ds_ep.chunk_compute_specs(c) for c in chunks]
  denom = float(p.gbs * p.seq)
  mtp_denom = denom * (p.seq - 1) / p.seq
  aux = p.gbs if cfg.load_balance_loss_weight > 0 else 0
  cdt = [ds_ep.leaf_dtype_tree(scfg, c, "compute") for c in chunks]

  def body(storage, toks, tgts):
    wc = jax.tree.map(
        lambda x, s, c, d: ds_ep._gather_leaf(x, s, c, d),
        storage,
        sspec,
        cspec,
        cdt,
        is_leaf=lambda x: not isinstance(x, (dict, list)),
    )
    total = jnp.zeros((), jnp.float32)
    counts = [[jnp.zeros((cfg.num_experts,), jnp.int32) for _ in range(ch.n_moe + ch.mtp)] for ch in chunks]
    for m in range(M):
      h = ds_ep.embed(wc[0], toks[m], scfg)
      for ch, w in zip(chunks, wc):
        for l in range(ch.dense):  # a chunk's dense layers first, then its MoE layers
          h = ds_ep.dense_layer_fwd(ds_ep.layer_weights(w["dense_layers"], l, ch.dense_kinds), h, scfg)
        for l, lid in enumerate(ch.layers):
          h, cnt = ds_ep.layer_fwd_counts(
              ds_ep.layer_weights(w["layers"], l, ch.kinds), h, ds_ep._layer_key(lid, m), scfg, aux_seqs=aux
          )
          counts[ch.index][l] += cnt
      total = total + ds_ep.head_loss(wc[-1], h, tgts[m], denom, scfg)
      extra, cnt = ds_ep.mtp_loss(
          wc[-1], h, toks[m], tgts[m], mtp_denom, ds_ep._layer_key(cfg.num_decoder_layers, m), scfg, aux
      )
      total = total + extra
      counts[-1][-1] += cnt
    counts = [[jax.lax.psum(c, ds_ep.FE) for c in cs] for cs in counts]
    return total[None], counts

  f = jax.shard_map(
      body,
      mesh=mesh,
      in_specs=(sspec, [P(ds_ep.FE, None)] * M, [P(ds_ep.FE, None)] * M),
      out_specs=(P(ds_ep.FE), P()),
      check_vma=False,
  )

  def loss(s, a, b):
    tot, counts = f(s, a, b)
    return jnp.sum(tot), counts

  (val, counts), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(storage, toks, tgts)
  return float(val), grads, counts


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


class Routing(unittest.TestCase):

  def test_group_mask_and_selection(self):
    """Each token's experts lie in its best groups (by the sum of a group's two best scores) and are the top-k of
    what those groups hold; without groups the selection is plain top-k."""
    cfg = DeepSeekConfig(num_experts=16, num_experts_per_tok=4, n_routing_groups=4, topk_routing_group=2)
    x = jax.random.uniform(jax.random.key(0), (3, 7, 16))
    mask = np.asarray(ds_moe.expert_group_mask(x, cfg))
    xs = np.asarray(x).reshape(3, 7, 4, 4)
    best2 = np.sort(xs, axis=-1)[..., -2:].sum(-1)
    want_groups = np.argsort(-best2, axis=-1)[..., :2]
    want = np.zeros((3, 7, 4))
    np.put_along_axis(want, want_groups, 1.0, axis=-1)
    np.testing.assert_array_equal(mask, np.repeat(want, 4, axis=-1))
    idx = np.asarray(ds_moe.select_experts(x, cfg))
    np.testing.assert_array_equal(np.sort(idx, -1), np.sort(np.argsort(-np.where(mask > 0, x, -np.inf), -1)[..., :4], -1))
    plain = cfg.replace(n_routing_groups=-1, topk_routing_group=-1)
    np.testing.assert_array_equal(ds_moe.select_experts(x, plain), jax.lax.top_k(x, 4)[1])

  def test_bias_picks_weights_do_not_see_it(self):
    """The routed bias decides which experts a token takes; their weights come from the scores without it."""
    scfg = exact_scfg(layers=1, dense=0, mtp_num_layers=0)
    cfg = scfg.model.replace(n_routing_groups=-1, topk_routing_group=-1)
    scfg = dataclasses.replace(scfg, model=cfg, ep=1, fsdp=1)
    w = {
        "gate": {
            "kernel": jax.random.normal(jax.random.key(1), (256, 8)) * 0.1,
            "bias": jnp.array([0, 0, 0, 0, 0, 0, 5.0, 5.0]),
        },
        "wi_0": jnp.ones((8, 256, 256)) * 0.01,
        "wi_1": jnp.ones((8, 256, 256)) * 0.01,
        "wo": jnp.ones((8, 256, 256)) * 0.01,
    }
    x = jax.random.normal(jax.random.key(2), (1, 4, 256))
    mesh = jax.make_mesh((1, 1), ds_ep.FE, devices=jax.devices()[:1])
    f = jax.shard_map(
        lambda w, x: ds_ep._ring_moe(w, x, jax.random.key(0), scfg)[1],
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    counts = np.asarray(jax.jit(f)(w, x))
    self.assertEqual(counts.tolist(), [0, 0, 0, 0, 0, 0, 4, 4])  # every token took the two biased experts


@testing.needs_devices(4)
class ShardedEmbed(unittest.TestCase):

  def test_matches_the_gathered_lookup(self):
    """MTP's lookup in the table's column blocks: the gathered table's values and gradient, gather and iota alike."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ds_ep.FE)  # Auto axes
    table = jax.random.normal(jax.random.key(0), (64, 32))
    ids = jax.random.randint(jax.random.key(1), (8, 5), 0, 64)
    probe = jax.random.normal(jax.random.key(2), (8, 5, 32))
    for iota in (False, True):
      scfg = dataclasses.replace(dt.tiny_xla(), iota_embed=iota)
      scfg = dataclasses.replace(scfg, model=scfg.model.replace(vocab_size=64, emb_dim=32))
      sharded = jax.shard_map(
          lambda t, i: ds_ep._sharded_embed(t, i, scfg),
          mesh=mesh,
          in_specs=(P(None, ds_ep.FE), P(ds_ep.FE, None)),
          out_specs=P(ds_ep.FE, None, None),
          check_vma=False,
      )
      full = lambda t, i: ds_ep.embed({"token_embedder": {"embedding": t}}, i, scfg)
      got, want = jax.jit(sharded)(table, ids), full(table, ids)
      np.testing.assert_array_equal(got, want)
      gg = jax.jit(jax.grad(lambda t: jnp.sum(sharded(t, ids) * probe)))(table)
      gw = jax.grad(lambda t: jnp.sum(full(t, ids) * probe))(table)
      np.testing.assert_allclose(gg, gw, rtol=1e-6, atol=1e-6)


class AuxLoss(unittest.TestCase):

  def test_seq_aux_loss(self):
    """Per sequence: the fp32 scores normalized per token, the plain top-k's token fractions (no bias, no groups),
    w x E x sum_e fraction x mean probability -- moe.megatron_aux_loss_inputs + load_balance_loss in numpy."""
    cfg = DeepSeekConfig(num_experts=8, num_experts_per_tok=2, load_balance_loss_weight=0.01)
    scores = jax.nn.sigmoid(jax.random.normal(jax.random.key(3), (2, 16, 8)))
    sc = np.asarray(scores, np.float64)
    probs = sc / (sc.sum(-1, keepdims=True) + 1e-20)
    top = np.argsort(-sc, axis=-1)[..., :2]
    frac = np.zeros((2, 8))
    for b in range(2):
      for t in range(16):
        frac[b, top[b, t]] += 1.0 / (16 * 2)
    want = 0.01 * 8 * np.sum(frac * probs.mean(1), axis=-1)
    np.testing.assert_allclose(ds_ep.seq_aux_loss(scores, cfg), want, rtol=1e-5)

  def test_aux_rides_the_backward(self):
    """A layer whose output carries its aux loss differentiates as if the aux loss were added to the objective,
    while its value leaves the output alone."""
    y = jnp.arange(6.0).reshape(2, 3)
    a = jnp.array(2.0)
    f = lambda y, a: jnp.sum(ds_ep._with_aux_loss(y * a, a**2) * 3.0)
    g = lambda y, a: jnp.sum(y * a * 3.0) + a**2
    self.assertEqual(float(f(y, a)), float(jnp.sum(y * a * 3.0)))
    for got, want in zip(jax.grad(f, argnums=(0, 1))(y, a), jax.grad(g, argnums=(0, 1))(y, a)):
      np.testing.assert_allclose(got, want)


@testing.needs_devices(4)
class Pipeline(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.p = pipeline(exact_scfg(), fsdp_scope="repeat")
    cls.params, _, cls.wc0 = cls.p.init(jax.random.key(1))
    cls.batch = cls.p.synthetic_batch(3)
    cls._ref = None

  @classmethod
  def ref(cls):
    """`reference` of the class's pipeline, params and batch -- computed once: it compiles the whole model as one
    program."""
    if cls._ref is None:
      cls._ref = reference(cls.p, cls.params, cls.batch)
    return cls._ref

  def test_chunks(self):
    """A layer split that counts every layer (1, 2, 1, 1 of 5, 2 of them dense): the dense layers first, spanning
    chunks 0 and 1, stacked apart from the MoE layers (their ids the MoE layers' own); a chunk of both; MTP and
    the embedding's copy on the last chunk only."""
    p = self.p
    self.assertTrue(p.every_layer)
    self.assertEqual(
        [(ch.layers, ch.dense, ch.mtp) for ch in p.chunks], [((), 1, 0), ((0,), 1, 0), ((1,), 0, 0), ((2,), 0, 1)]
    )
    keys = [sorted(ds_ep.chunk_shapes(p.scfg, ch)) for ch in p.chunks]
    self.assertEqual(
        keys,
        [
            ["dense_layers", "token_embedder"],
            ["dense_layers", "layers"],
            ["layers"],
            ["decoder_norm", "layers", "logits_dense", "mtp"],
        ],
    )
    shapes = ds_ep.chunk_shapes(p.scfg, p.chunks[1])
    self.assertEqual(shapes["dense_layers"]["mlp"]["wi_0"]["kernel"], (1, 256, 384))
    self.assertEqual(shapes["layers"]["moe"]["gate"]["bias"], (1, 8))
    self.assertEqual(sum(ds_ep.count_leaves(p.chunks[3])), 2)  # the last chunk's layer and MTP's
    self.assertFalse(p.self_prefetch)  # chunk 0 has no routed experts
    self.assertEqual(p.progs[2].aux_seqs, 16)  # the global batch's sequences
    # MTP's embedding copy computes in its storage blocks: no gather
    cs = ds_ep.chunk_compute_specs(p.chunks[3])
    self.assertEqual(cs["mtp"]["token_embedder"]["embedding"], P(None, ds_ep.FE))
    self.assertEqual(cs["logits_dense"]["kernel"], P(None, None))

  def test_routed_bias_stays_fp32(self):
    """Under float32_gate_logits the routed biases are fp32 in storage and compute, whatever the weights' dtype
    (MaxText's bias_dtype), and the routers' kernels compute in fp32 (GateLogit casts the stored kernel); every
    other leaf follows weight_dtype / dtype."""
    scfg = exact_scfg()
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.bfloat16, dtype=jnp.bfloat16))
    p = pipeline(scfg, fsdp_scope="repeat")
    ch = p.chunks[3]
    biases = {"moe.gate.bias", "mtp.layer.moe.gate.bias"}
    for kind, fp32 in (("storage", biases), ("compute", biases | {"moe.gate", "mtp.layer.moe.gate"})):
      got = dict(zip(ds_ep.leaf_names(ch), ds_ep.leaf_dtypes(scfg, ch, kind)))
      self.assertEqual({n for n, d in got.items() if d == jnp.float32}, fp32, kind)
      self.assertTrue(all(d == jnp.bfloat16 for n, d in got.items() if n not in fp32), kind)
    params, _, wc0 = p.init(jax.random.key(0))
    w = params[ch.stage][p.slot[ch.index]]
    self.assertEqual(
        (w["layers"]["moe"]["gate"]["bias"].dtype, w["layers"]["moe"]["gate"]["kernel"].dtype),
        (jnp.float32, jnp.bfloat16),
    )
    wc = p.stages[ch.stage].gather(params[ch.stage])[p.slot[ch.index]]
    self.assertEqual(
        (wc["layers"]["moe"]["gate"]["bias"].dtype, wc["mtp"]["layer"]["moe"]["gate"]["bias"].dtype),
        (jnp.float32, jnp.float32),
    )

  @pytest.mark.scheduled_only
  def test_router_reads_the_fp32_kernel(self):
    """fp32 weights and bf16 compute under float32_gate_logits (the DeepSeek-V3 parity runs'): every router -- each
    MoE layer's, MTP's -- reads its kernel's fp32 master values, bit for bit, as GateLogit and moe.route cast them,
    not a bf16 copy's; without float32_gate_logits it gets the bf16 copy every other kernel does."""
    for f32 in (True, False):
      scfg = exact_scfg(float32_gate_logits=f32)
      scfg = dataclasses.replace(scfg, model=scfg.model.replace(weight_dtype=jnp.float32, dtype=jnp.bfloat16))
      p = pipeline(scfg, fsdp_scope="repeat")
      ch = p.chunks[3]
      params, _, _ = p.init(jax.random.key(0))
      w = params[ch.stage][p.slot[ch.index]]
      wc = p.stages[ch.stage].gather(params[ch.stage])[p.slot[ch.index]]
      for master, got in (
          (w["layers"]["moe"]["gate"]["kernel"], wc["layers"]["moe"]["gate"]["kernel"]),
          (w["mtp"]["layer"]["moe"]["gate"]["kernel"], wc["mtp"]["layer"]["moe"]["gate"]["kernel"]),
      ):
        self.assertEqual(got.dtype, jnp.float32 if f32 else jnp.bfloat16, f32)
        if f32:
          np.testing.assert_array_equal(np.asarray(got), np.asarray(master))
      self.assertEqual(wc["layers"]["moe"]["wi_0"].dtype, jnp.bfloat16)

  @pytest.mark.scheduled_only
  @unittest.skipUnless(
      dt.HAVE_TOKAMAX,
      "needs tokamax: tiny_xla's fallback, the reference attention, is "
      "not bit for bit between stash and remat (in about 70% of the leaves, before this change too)",
  )
  def test_stash_with_an_fp32_router_matches_recompute(self):
    """Stash mode traces its layers on the compute weights' own dtypes (the fp32 router under float32_gate_logits,
    bf16 elsewhere): the stash schedule's order run with its modes and without, one-layer MoE chunks with a routed
    bias (no update) and the fp32 router, bf16 compute, tokamax attention -- the loss and every gradient bit for
    bit."""
    scfg = exact_scfg(
        layers=4,
        dense=0,
        mtp_num_layers=0,
        routed_bias_update_rate=0.0,
        load_balance_loss_weight=0.0,
        moe_use_megatron_seq_aux_loss=False,
        dtype=jnp.bfloat16,
    )
    gib = dt.lite_gib(scfg, global_batch=16)
    ps = pipeline(scfg, schedule="stash", layer_split=(1, 1, 1, 1), stash="lite", stash_budget_gib=2 * gib)
    pr = pipeline(scfg, schedule=ps.order, layer_split=(1, 1, 1, 1))
    self.assertTrue(ps.modes)
    self.assertEqual(pr.modes, {})
    params, _, wc0 = pr.init(jax.random.key(4))
    batch = pr.synthetic_batch(5)
    out = []
    for q in (ps, pr):
      grads, _, lacc = q.loss_and_grads_overlap(params, [list(w) for w in wc0], batch)
      out.append((q.chunk_trees(grads), float(jnp.sum(lacc))))
    dt.assert_bitwise(self, out[0][0], out[1][0], "stash vs remat gradients")
    self.assertEqual(out[0][1], out[1][1])

  def test_stash_refuses_the_aux_loss(self):
    """Stash mode runs its layers without the aux loss riding their outputs: a stash schedule with the
    sequence-wise aux loss on is refused while the pipeline is built, not trained without the loss."""
    scfg = exact_scfg(layers=4, dense=0, mtp_num_layers=0, routed_bias_update_rate=0.0)
    kw = dict(schedule="stash", layer_split=(1, 1, 1, 1), stash="lite")
    with self.assertRaisesRegex(NotImplementedError, r"chunk 0: stash mode .* aux loss True"):
      pipeline(scfg, **kw)
    self.assertTrue(
        pipeline(dataclasses.replace(scfg, model=scfg.model.replace(load_balance_loss_weight=0.0)), **kw).modes
    )

  def test_prefetch_pin_in_a_dense_chunk_is_refused(self):
    """A --prefetch-pin SPEC whose host forward runs dense layers only (chunk 0 here, hosting chunk 2's gathers) is
    refused while the pipeline is built: the pin points are in MoE layers, a dense layer has none."""
    with self.assertRaisesRegex(
        ValueError, r"--prefetch-pin: moe.wo \(chunk 2\) is pinned in layer 0 of 0F2, whose " r"chunk 0 has 0 MoE layer"
    ):
      pipeline(exact_scfg(), fsdp_scope="step", prefetch_pin="hn=moe.wo")

  def test_batch_split_refuses_what_it_does_not_run(self):
    """Batch split (split_layers: batch_split_impl 'native') runs its MoE layers without counts or an aux loss
    riding their outputs, and its fused last chunk without MTP: each of the routed-bias update, the aux loss and MTP
    is refused; routed biases, groups and the fp32 router alone run.  A chunk without a MoE layer -- chunk 0 of a
    layer split that counts every layer, one dense layer alone (chunk 1 the other and a MoE layer) -- has nothing to
    split and runs the plain path (ds_ep.ChunkPrograms.split; ds_ep_test.DenseLayers.test_dense_only_chunk_zero holds
    its numbers)."""
    base = dict(
        dense=0,
        mtp_num_layers=0,
        routed_bias_update_rate=0.0,
        load_balance_loss_weight=0.0,
        moe_use_megatron_seq_aux_loss=False,
    )
    kw = dict(layer_split=(1, 1, 1, 1), global_batch=32)  # an even number of rows a device
    split = lambda scfg: dataclasses.replace(scfg, batch_split=True, batch_split_impl="native")
    self.assertTrue(pipeline(split(exact_scfg(layers=4, **base)), **kw).scfg.batch_split)
    for extra, what in (
        (dict(routed_bias_update_rate=RATE), "counting True"),
        (dict(load_balance_loss_weight=0.01, moe_use_megatron_seq_aux_loss=True), "aux loss True"),
        (dict(mtp_num_layers=1), "mtp 1"),
    ):
      with self.subTest(what=what), self.assertRaisesRegex(NotImplementedError, f"batch_split runs .*{what}"):
        pipeline(split(exact_scfg(layers=4, **{**base, **extra})), **kw)
    p = pipeline(split(exact_scfg(layers=5, **{**base, "dense": 2})), global_batch=32)
    self.assertEqual([(c.layers, c.dense) for c in p.chunks], [((), 1), ((0,), 1), ((1,), 0), ((2,), 0)])
    self.assertEqual([pr.split for pr in p.progs], [False, True, True, True])

  def test_parity_port_maps_dense_layers_across_chunks(self):
    """ds_ep_parity's port tree concatenates the dense layers of every chunk that holds some (a layer split that
    counts every layer: chunks 0 and 1 here), and its gradients go back to those chunks -- a chunk of dense layers
    alone without a 'layers' stack.  Under layer_tree 'per_layer' (without the routed-bias update, which files counts
    into the stacks) the port tree of the same weights cut a layer a subtree is the same, and its gradients go back
    as those subtrees."""
    from maxtext.experimental.mpmd.jax import ds_ep_parity

    p = pipeline(exact_scfg(mtp_num_layers=0), fsdp_scope="repeat")
    trees = p.chunk_trees(p.init(jax.random.key(3))[0])
    self.assertEqual([sorted(t) for t in trees][:2], [["dense_layers", "token_embedder"], ["dense_layers", "layers"]])
    port = ds_ep_parity.port_tree(p, trees)
    self.assertEqual(port["decoder"]["dense_layers"]["mlp"]["wi_0"]["kernel"].shape[0], 2)
    self.assertEqual(port["decoder"]["moe_layers"]["self_attention"]["wq_a"]["kernel"].shape[0], 3)
    back = ds_ep_parity.port_to_chunks(p, port)
    self.assertEqual(jax.tree.structure(back), jax.tree.structure(trees))
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(trees)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    q = pipeline(
        dataclasses.replace(exact_scfg(mtp_num_layers=0, routed_bias_update_rate=0.0), layer_tree="per_layer"),
        fsdp_scope="repeat",
    )
    self.assertEqual([(c.kinds, c.dense_kinds) for c in q.chunks][:2], [((), ("mla",)), (("mla",), ("mla",))])
    per = [ds_ep_parity.relayout(jax.tree.map(np.asarray, t), a, b) for t, a, b in zip(trees, p.chunks, q.chunks)]
    self.assertEqual(sorted(per[1]["dense_layers"]), ["00"])
    qport = ds_ep_parity.port_tree(q, per)
    self.assertEqual(jax.tree.structure(qport), jax.tree.structure(port))
    for a, b in zip(jax.tree.leaves(qport), jax.tree.leaves(port)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    qback = ds_ep_parity.port_to_chunks(q, qport)
    self.assertEqual(jax.tree.structure(qback), jax.tree.structure(per))
    for a, b in zip(jax.tree.leaves(qback), jax.tree.leaves(per)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_auto_pins_skip_a_dense_chunk(self):
    """--prefetch-pin auto leaves a forward of dense layers alone, saying why, under either PinModel: its windows
    are its MoE layers' (none), and v6e's overflow 'last' pinned a gather no window held at the last of them."""
    scfg = dataclasses.replace(exact_scfg(), fsdp_collectives="xla")
    for model in (ds_ep.PIN_MODEL, ds_ep.PIN_MODEL_V6E):
      with mock.patch.object(dsp.Pipeline, "pin_model", lambda self, stage, model=model: model):
        p = pipeline(scfg, fsdp_scope="step", prefetch_pin="auto")
      dense = [j for j in p.unpinned if not p.chunks[p.order[j][0]].n_moe]
      self.assertTrue(dense, p.unpinned)
      self.assertEqual({p.unpinned[j] for j in dense}, {"dense layers only: no pin points"})
      wo = ds_ep.leaf_names(p.chunks[2]).index("moe.wo")
      self.assertEqual(ds_ep.plan_pins(scfg, p.chunks[0], [(p.chunks[2], wo)], 4096, model=model), ())

  def test_pin_costs_price_the_fp32_router(self):
    """'auto' pins price a gather at its compute copy's bytes: 4 an element for the fp32 router's kernel and bias
    under float32_gate_logits (bf16 compute), 2 for every other leaf."""
    scfg = exact_scfg(dtype=jnp.bfloat16)
    p = pipeline(scfg, fsdp_scope="repeat")
    names, calls = ds_ep.leaf_names(p.chunks[2]), []
    with mock.patch.object(ds_ep, "pin_gather_ms", lambda *a, itemsize=2, **k: calls.append(itemsize) or 0.0):
      ds_ep.plan_pins(scfg, p.chunks[1], [(p.chunks[2], i) for i in range(len(names))], 4096)
    got = dict(zip(names, calls, strict=True))
    self.assertEqual({n for n, b in got.items() if b == 4}, {"moe.gate", "moe.gate.bias"})
    self.assertEqual({b for n, b in got.items() if n not in ("moe.gate", "moe.gate.bias")}, {2})

  @pytest.mark.scheduled_only
  def test_precompile_covers_the_tied_embedding(self):
    """precompile compiles MTP's tied-embedding programs too -- the sum of the two gradients on chunk 0's stage and
    on the last chunk's (`_tie`), and under --transport ppermute the two copy channels -- so none is traced and
    compiled inside the first step, outside check_program_agreement; a step then runs on them.  Ahead of time the
    ppermute channels exist too (ds_ep_aot compiles what precompile does)."""
    p = pipeline(exact_scfg(), fsdp_scope="repeat", transport="ppermute")
    a, b = p.place[0], p.place[p.C - 1]
    self.assertNotEqual(a, b)
    self.assertEqual(sorted(p.tie_channels), sorted([(a, b), (b, a)]))
    p.precompile()
    for key in (("tie", a, True), ("tie", b, False)):
      self.assertIsInstance(p._exe[key], jax.stages.Compiled, key)
    for ch in p.tie_channels.values():
      self.assertIsInstance(ch.prog, jax.stages.Compiled)
    names = [n for n, _, _, _ in p.program_prints]
    self.assertIn(f"tie_s{a}", names)
    self.assertIn(f"tie_mtp_s{b}", names)
    params, opts, wc0 = p.init(jax.random.key(1))
    params, _, _, _, gn = p.step(params, opts, wc0, self.batch, p.lr_arrays(1e-3))
    self.assertTrue(np.isfinite(float(gn)))
    np.testing.assert_array_equal(
        np.asarray(params[a][p.slot[0]]["token_embedder"]["embedding"]),
        np.asarray(params[b][p.slot[p.C - 1]]["mtp"]["token_embedder"]["embedding"]),
    )
    # ahead of time (ds_ep_aot) the ppermute tie channels are programs to compile and price, as the activations' are
    aot = {
        t: pipeline(exact_scfg(), fsdp_scope="repeat", transport=t, aot=True).tie_channels for t in ("ppermute", "native")
    }
    self.assertEqual(sorted(aot["ppermute"]), sorted([(a, b), (b, a)]))
    self.assertTrue(all(isinstance(ch, transfer.PermuteChannel) for ch in aot["ppermute"].values()))
    self.assertEqual(aot["native"], {})

  @pytest.mark.scheduled_only
  def test_matches_reference(self):
    """Loss, every gradient (the tied embedding's: chunk 0's and MTP's summed, the same for both), the expert counts
    in the bias slots and the global norm -- the tied gradient once, no counts -- against `reference`."""
    p = self.p
    grads, sq, lacc = p.loss_and_grads_overlap(self.params, self.wc0, self.batch)
    grads, sq = p._tie(grads, sq)
    loss, ref, counts = self.ref()
    self.assertAlmostEqual(float(jnp.sum(lacc)), loss, delta=1e-5 * abs(loss))
    tied = ref[0]["token_embedder"]["embedding"] + ref[-1]["mtp"]["token_embedder"]["embedding"]
    ref_sq = 0.0
    for ch in p.chunks:
      g = grads[ch.stage][p.slot[ch.index]]
      flat_g = jax.tree_util.tree_flatten_with_path(g)[0]
      flat_r = jax.tree.leaves(ref[ch.index])
      flags = ds_ep.count_leaves(ch)
      for (path, a), b, k in zip(flat_g, flat_r, flags, strict=True):
        name = f"chunk {ch.index} {jax.tree_util.keystr(path)}"
        if k:
          continue
        if path[-1].key == "embedding":
          b = tied
        self.assertLess(rel(a, b), 1e-4, name)
        ref_sq += float(jnp.sum(jnp.square(b)))
      # the bias slots: each MoE layer's expert counts (the MTP layer's last)
      got = []
      if ch.n_moe:
        got += list(np.asarray(g["layers"]["moe"]["gate"]["bias"]))
      if ch.mtp:
        got.append(np.asarray(g["mtp"]["layer"]["moe"]["gate"]["bias"]))
      want = [np.asarray(c, np.float32) for c in counts[ch.index]]
      self.assertEqual(len(got), len(want))
      for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)
    ref_sq -= float(jnp.sum(jnp.square(tied)))  # counted twice above (both copies): once
    self.assertLess(abs(sum(float(x) for x in sq) - ref_sq) / ref_sq, 1e-4)

  @testing.slow
  def test_saved_residuals_match_too(self):
    """--save-residuals on, whose forwards keep lean residuals (attention's and the post-attention norms in one
    checkpoint with the projections that read them: the fp32 router's, the shared expert's, a dense layer's MLP),
    on every feature but the two the saved path refuses (the routed bias's update counts, the sequence-wise aux
    loss): the same loss and gradients.  So also as a GPU runs it: every layer a subtree (layer_tree 'per_layer', the
    dense layers over chunks 0 and 1 too) and the gathered embedding, whose rows chunk 0's backwards scatter into the
    accumulator (ds_ep.embed_grad)."""
    for tree, iota in (("stacked", True), ("per_layer", False)):
      with self.subTest(layer_tree=tree, iota_embed=iota):
        scfg = dataclasses.replace(
            exact_scfg(routed_bias_update_rate=0.0, load_balance_loss_weight=0.0),
            save_residuals=True,
            expert_weights="sliced",
            layer_tree=tree,
            iota_embed=iota,
        )
        p = pipeline(scfg, fsdp_scope="repeat")
        params, _, wc0 = p.init(jax.random.key(1))
        grads, sq, lacc = p.loss_and_grads_overlap(params, wc0, self.batch)
        grads, sq = p._tie(grads, sq)
        loss, ref, _ = reference(p, params, self.batch)
        self.assertAlmostEqual(float(jnp.sum(lacc)), loss, delta=1e-5 * abs(loss))
        tied = ref[0]["token_embedder"]["embedding"] + ref[-1]["mtp"]["token_embedder"]["embedding"]
        for ch in p.chunks:
          g = grads[ch.stage][p.slot[ch.index]]
          for (path, a), b, k in zip(
              jax.tree_util.tree_flatten_with_path(g)[0],
              jax.tree.leaves(ref[ch.index]),
              ds_ep.count_leaves(ch),
              strict=True,
          ):
            if not k:
              self.assertLess(
                  rel(a, tied if path[-1].key == "embedding" else b),
                  1e-4,
                  f"chunk {ch.index} {jax.tree_util.keystr(path)}",
              )

  @pytest.mark.scheduled_only
  def test_remat_last_matches_too(self):
    """The fused last chunk with its layers recomputed (--remat-last, which the TPU7x run needs for MTP's memory):
    the same loss and gradients."""
    scfg = dataclasses.replace(exact_scfg(), remat_fused_last=True)
    p = pipeline(scfg, fsdp_scope="repeat")
    grads, sq, lacc = p.loss_and_grads_overlap(self.params, p.init(jax.random.key(1))[2], self.batch)
    grads, sq = p._tie(grads, sq)
    loss, ref, _ = reference(p, self.params, self.batch)
    self.assertAlmostEqual(float(jnp.sum(lacc)), loss, delta=1e-5 * abs(loss))
    ch = p.chunks[-1]
    g = grads[ch.stage][p.slot[ch.index]]
    for (path, a), b, k in zip(
        jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(ref[-1]), ds_ep.count_leaves(ch), strict=True
    ):
      if not k and path[-1].key != "embedding":
        self.assertLess(rel(a, b), 1e-4, jax.tree_util.keystr(path))

  @pytest.mark.scheduled_only
  def test_route_weights_experts_matches_too(self):
    """route_weights 'experts' (the weights on wo's input rows, the combine an unweighted sum) against `reference`
    under 'combine': fp32 weights from the fp32 router, a routed bias that picks the experts but not their weights,
    group-limited routing, the aux loss and the MTP layer -- the same loss and gradients to fp32 rounding, the same
    expert counts in the bias slots exactly (the routing is the same)."""
    p = pipeline(dataclasses.replace(exact_scfg(), route_weights="experts"), fsdp_scope="repeat")
    grads, sq, lacc = p.loss_and_grads_overlap(self.params, p.init(jax.random.key(1))[2], self.batch)
    grads, _ = p._tie(grads, sq)
    loss, ref, counts = self.ref()
    self.assertAlmostEqual(float(jnp.sum(lacc)), loss, delta=1e-5 * abs(loss))
    tied = ref[0]["token_embedder"]["embedding"] + ref[-1]["mtp"]["token_embedder"]["embedding"]
    for ch in p.chunks:
      g = grads[ch.stage][p.slot[ch.index]]
      for (path, a), b, k in zip(
          jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(ref[ch.index]), ds_ep.count_leaves(ch), strict=True
      ):
        if not k:
          self.assertLess(
              rel(a, tied if path[-1].key == "embedding" else b), 1e-4, f"chunk {ch.index} {jax.tree_util.keystr(path)}"
          )
      got = list(np.asarray(g["layers"]["moe"]["gate"]["bias"])) if ch.n_moe else []
      got += [np.asarray(g["mtp"]["layer"]["moe"]["gate"]["bias"])] if ch.mtp else []
      for a, b in zip(got, counts[ch.index], strict=True):
        np.testing.assert_array_equal(a, np.asarray(b, np.float32))

  @testing.slow
  def test_step_updates_bias_and_keeps_the_copies_equal(self):
    """One step: each routed bias moves by RATE x sign(mean load - load) of its layer's counts (from zero), AdamW
    leaves it alone; chunk 0's embedding and MTP's copy start and stay equal."""
    p = self.p
    params, opts, wc0 = p.init(jax.random.key(1))
    _, _, counts = reference(p, params, self.batch)
    emb = lambda ps: (
        np.asarray(ps[p.place[0]][p.slot[0]]["token_embedder"]["embedding"]),
        np.asarray(ps[p.place[-1]][p.slot[p.C - 1]]["mtp"]["token_embedder"]["embedding"]),
    )
    a, b = emb(params)
    np.testing.assert_array_equal(a, b)
    params, opts, wc0, lacc, gn = p.step(params, opts, wc0, self.batch, p.lr_arrays(1e-3))
    a2, b2 = emb(params)
    np.testing.assert_array_equal(a2, b2)
    self.assertGreater(np.abs(a2 - a).max(), 0)
    for ch in p.chunks:
      w = params[ch.stage][p.slot[ch.index]]
      got = list(np.asarray(w["layers"]["moe"]["gate"]["bias"])) if ch.n_moe else []
      if ch.mtp:
        got.append(np.asarray(w["mtp"]["layer"]["moe"]["gate"]["bias"]))
      for g, c in zip(got, counts[ch.index]):
        c = np.asarray(c, np.float32)
        np.testing.assert_allclose(g, RATE * np.sign(c.mean() - c), atol=1e-7)

  def test_grad_dtype_leaves_the_counts_alone(self):
    """grad_dtype casts the reduced gradients only: the expert counts in the bias slots keep their fp32, under
    either adam_math.  Rounded to bf16, loads of 257 / 256 x 6 / 255 (mean 256: six balanced experts, whose biases
    stay) would become 256 x 7 / 255, and the balanced experts' biases would move by -RATE."""
    E = self.p.scfg.model.num_experts
    counts = np.array([257] + [256] * (E - 2) + [255], np.float32)
    self.assertEqual(counts.mean(), 256)
    host = lambda t: jax.tree.map(np.asarray, t)
    put = lambda t, sh: jax.tree.map(jax.device_put, t, sh)
    for math in ("fp32", "optax"):
      p = pipeline(dataclasses.replace(exact_scfg(), grad_dtype=jnp.bfloat16, adam_math=math), fsdp_scope="repeat")
      s = p.place[p.C - 1]  # MTP's chunk's stage: a layer's and MTP's bias slots
      st, chs = p.stages[s], p.stage_chunks[s]
      w = host(self.params[s])
      leaves, tdef = jax.tree.flatten(w)
      flags = [k for ch in chs for k in ds_ep.count_leaves(ch)]
      grads = jax.tree.unflatten(
          tdef,
          [
              np.broadcast_to(counts, x.shape).copy() if k else np.zeros(x.shape, np.float32)
              for x, k in zip(leaves, flags, strict=True)
          ],
      )
      new, _, _ = st.update(
          put(w, st.s_sh), st.init_opt(put(w, st.s_sh)), put(grads, st.s_sh), [jnp.float32(0)] * p.S, jnp.float32(1e-3)
      )
      got = []
      for ch in chs:
        b = new[p.slot[ch.index]]
        got += list(np.asarray(b["layers"]["moe"]["gate"]["bias"])) if ch.n_moe else []
        got += [np.asarray(b["mtp"]["layer"]["moe"]["gate"]["bias"])] if ch.mtp else []
      self.assertEqual(len(got), 3, math)  # chunk 1's layer, chunk 3's layer and MTP's
      for g in got:
        np.testing.assert_allclose(g, RATE * np.sign(256 - counts), atol=1e-7, err_msg=math)

  def test_init_without_mtp_keeps_its_embedding_key(self):
    """A model without MTP draws chunk 0's embedding as before (its chunk's key), not from the shared one."""
    scfg = exact_scfg(mtp_num_layers=0)
    p = pipeline(scfg, fsdp_scope="repeat")
    self.assertEqual(p.mtp, 0)
    self.assertNotIn("mtp", ds_ep.chunk_shapes(scfg, p.chunks[-1]))
    self.assertEqual(len(p.put_batch(np.zeros((16, 256), np.int32), np.zeros((16, 256), np.int32))), 2)


@testing.needs_devices(4)
class Flags(unittest.TestCase):

  def args(self, *extra):
    return dsp.make_parser().parse_args(["--layers", "61", "--experts", "256", "--vocab", "129280", *extra])

  def test_deepseek_v3(self):
    a = self.args(
        "--dense-layers",
        "3",
        "--mlp",
        "18432",
        "--routed-bias",
        "on",
        "--bias-update-rate",
        "0.001",
        "--routing-groups",
        "8",
        "--topk-groups",
        "4",
        "--mtp",
        "1",
        "--routing",
        "topk",
        "--float32-gate-logits",
        "on",
        "--lb-loss-weight",
        "0.01",
    )
    cfg = dsp.model_config(a)
    self.assertEqual(
        (
            cfg.first_num_dense_layers,
            cfg.mlp_dim,
            cfg.moe_mlp_dim,
            cfg.routed_bias,
            cfg.routed_bias_update_rate,
            cfg.n_routing_groups,
            cfg.topk_routing_group,
            cfg.mtp_num_layers,
            cfg.mtp_loss_scaling_factor,
            cfg.float32_gate_logits,
            cfg.load_balance_loss_weight,
            cfg.moe_use_megatron_seq_aux_loss,
        ),
        (3, 18432, 2048, True, 0.001, 8, 4, 1, 0.1, True, 0.01, True),
    )
    # the defaults stay the sparse-layer model
    cfg = dsp.model_config(self.args())
    self.assertEqual(
        (
            cfg.first_num_dense_layers,
            cfg.mlp_dim,
            cfg.routed_bias,
            cfg.n_routing_groups,
            cfg.mtp_num_layers,
            cfg.float32_gate_logits,
            cfg.load_balance_loss_weight,
        ),
        (0, 2048, False, -1, 0, False, 0.0),
    )

  def test_attention_flags(self):
    """The tokamax kernel runs its base-2 softmax unless told off (MaxText's sa_use_base2_exp, its default on as
    here, rounds log2(e) to bf16 in the kernels): every tokamax run before the flag did; --attn splash takes no
    flag (it is exact); its fused backward's blocks default to --sa-block."""
    cfg = dsp.model_config(self.args("--sa-block", "1024"))
    self.assertEqual(
        (
            cfg.attention_kernel,
            cfg.sa_use_base2_exp,
            cfg.sa_block_q_dkv,
            cfg.sa_block_kv_dkv,
            cfg.sa_block_kv_dkv_compute,
        ),
        ("tokamax_splash", True, 1024, 1024, 1024),
    )
    cfg = dsp.model_config(
        self.args("--sa-base2-exp", "off", "--sa-block-dkv", "512", "--sa-block-kv-dkv-compute", "256")
    )
    self.assertEqual(
        (cfg.sa_use_base2_exp, cfg.sa_block_q_dkv, cfg.sa_block_kv_dkv, cfg.sa_block_kv_dkv_compute),
        (False, 512, 512, 256),
    )
    self.assertEqual(dsp.model_config(self.args("--attn", "splash")).attention_kernel, "splash")
    # the kv compute block follows --sa-block-dkv: a compute block wider than its kv block fails tokamax's backward
    cfg = dsp.model_config(self.args("--sa-block-dkv", "512"))
    self.assertEqual((cfg.sa_block_q_dkv, cfg.sa_block_kv_dkv, cfg.sa_block_kv_dkv_compute), (512, 512, 512))

  def test_compute_dtype_flag(self):
    """--dtype sets the compute dtype (parity checks in fp32); bf16 by default."""
    self.assertEqual(dsp.model_config(self.args()).dtype, jnp.bfloat16)
    self.assertEqual(dsp.model_config(self.args("--dtype", "float32")).dtype, jnp.float32)

  def test_grad_dtype_flag(self):
    """--grad-dtype (MaxText grad_dtype, the dtype the reduced gradient enters AdamW in): auto keeps --rs-dtype's
    (StageConfig.grad_dtype None); float32 / bfloat16 reach the StageConfig through `build`."""
    ap = dsp.make_parser()
    self.assertEqual(self.args().grad_dtype, "auto")
    for flag, want in (("auto", None), ("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
      scfg = dsp.build(ap.parse_args([*dt.TINY_ARGS, "--rs-dtype", "bfloat16", "--grad-dtype", flag])).scfg
      self.assertEqual((scfg.rs_dtype, scfg.grad_dtype), (jnp.bfloat16, want), flag)
    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
      ap.parse_args([*dt.TINY_ARGS, "--grad-dtype", "float16"])

  def test_refusals(self):
    for extra, msg in (
        (("--routing-groups", "8"), "come together"),
        (("--routing-groups", "7", "--topk-groups", "4", "--routing", "topk"), "dividing"),
        (("--bias-update-rate", "0.001"), "needs --routed-bias on"),
        (("--routed-bias", "on"), "need --routing topk"),
        (("--attn", "splash", "--sa-base2-exp", "on"), "tokamax_splash's base-2 softmax"),
        (("--v-head-dim", "0"), "want a value head dim of at least 1"),
        (("--qk-nope-head-dim", "0", "--qk-rope-head-dim", "0"), "summing to at least 1"),
        (("--qk-rope-head-dim", "-1"), "non-negative query / key head dims"),
    ):
      with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, msg):
        dsp.model_config(self.args(*extra))

  def test_head_dims(self):
    """MLA's head dims: DeepSeek-V3's 128 / 64 / 128 unless given (also for a namespace without the flags, or with
    them None); an explicit 0 stays 0 -- --qk-rope-head-dim 0 is MLA without RoPE, --qk-nope-head-dim 0 keys from
    the RoPE channels alone -- while the dims that leave no attention are refused (test_refusals: a value head dim
    of 0, whose attention output is all zeros, and no query / key channel)."""
    keys = ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")

    def dims(a):
      cfg = dsp.model_config(a)
      return tuple(getattr(cfg, k) for k in keys)

    self.assertEqual(dims(self.args()), (128, 64, 128))
    self.assertEqual(
        dims(types.SimpleNamespace(**{k: v for k, v in vars(self.args()).items() if k not in keys})), (128, 64, 128)
    )
    self.assertEqual(dims(types.SimpleNamespace(**{**vars(self.args()), **dict.fromkeys(keys)})), (128, 64, 128))
    self.assertEqual(dims(self.args("--qk-nope-head-dim", "64", "--qk-rope-head-dim", "64")), (64, 64, 128))
    self.assertEqual(dims(self.args("--qk-rope-head-dim", "0")), (128, 0, 128))
    self.assertEqual(dims(self.args("--qk-nope-head-dim", "0")), (0, 64, 128))


def fake_maxtext_cfg(**kw):
  """The attributes deepseek.config.from_maxtext reads, as a MaxText pyconfig names them (DeepSeekConfig's
  defaults)."""
  base = {f.name: getattr(DeepSeekConfig(), f.name) for f in dataclasses.fields(DeepSeekConfig)}
  base.update(attention="dot_product", sparse_matmul=True, megablox=False, use_tokamax_gmm=False, **kw)
  return types.SimpleNamespace(**base)


class FromMaxText(unittest.TestCase):

  def test_carries_the_deepseek_v3_keys(self):
    """from_maxtext carries the routing groups, MTP and the sequence-wise aux loss."""
    from maxtext.experimental.mpmd.jax.deepseek import config as ds_config

    keys = dict(
        n_routing_groups=8,
        topk_routing_group=4,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.3,
        moe_use_megatron_seq_aux_loss=True,
    )
    got = ds_config.from_maxtext(fake_maxtext_cfg(**keys))
    self.assertEqual({k: getattr(got, k) for k in keys}, keys)


class ReferenceTrainer(unittest.TestCase):

  def test_refuses_mtp(self):
    """maxtext.experimental.mpmd.jax.deepseek.train builds no MTP layer: a config with one is refused, not trained on
    the main loss
    alone (from_maxtext carries MaxText's mtp_num_layers, e.g. the MLPerf recipe's 1)."""
    from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

    cfg = DeepSeekConfig(mtp_num_layers=1)
    with self.assertRaisesRegex(NotImplementedError, "mtp_num_layers=1"):
      ds_train.compute_loss(None, None, None, cfg, ds_train.TrainConfig())


class MaxTextFlops(unittest.TestCase):

  def test_mtp_counts_one_moe_layer(self):
    """An MTP layer adds one MoE layer's FFN, attention projections and attention, as MaxText counts it."""
    cfg = DeepSeekConfig(
        vocab_size=129280,
        emb_dim=7168,
        num_decoder_layers=61,
        first_num_dense_layers=3,
        num_query_heads=128,
        num_kv_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        mlp_dim=18432,
        moe_mlp_dim=2048,
        num_experts=256,
        num_experts_per_tok=8,
        shared_experts=1,
    )
    w0, a0 = dsp.maxtext_training_flops(cfg, 4096, 4096)
    w1, a1 = dsp.maxtext_training_flops(cfg.replace(mtp_num_layers=1), 4096, 4096)
    w2, _ = dsp.maxtext_training_flops(cfg.replace(num_decoder_layers=62), 4096, 4096)
    self.assertAlmostEqual(w1 - w0, w2 - w0)  # one more MoE layer's weights
    self.assertAlmostEqual(a1 / a0, 62 / 61)


if __name__ == "__main__":
  unittest.main()
