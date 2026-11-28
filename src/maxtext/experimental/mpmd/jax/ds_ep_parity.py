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

"""Loss and gradient parity of the MPMD PP x FSDP x EP pipeline, on TPU.

    python3 -m maxtext.experimental.mpmd.jax.ds_ep_parity --preset small [--floor]
    python3 -m maxtext.experimental.mpmd.jax.ds_ep_parity --preset full-dims
    python3 -m maxtext.experimental.mpmd.jax.ds_ep_parity --preset small --gbs 16 --batch-split [--batch-split-impl native]

This runs directly, not through ds_ep_run.sh (which runs the benchmark), so
it sees whatever LIBTPU_INIT_ARGS the environment sets.  The presets need 8
devices (a TPU7x-8; `tiny` runs on 8 CPU devices, but not `--batch-split` on
Lineage's layers, whose kernels run on TPU only: refused there, and
`--batch-split-impl native` runs).  On v6e, emb 1024 (`small`, `small-pp4`)
trips an assert in MaxText's SparseCore gather-reduce; use emb >= 2048 there.

Two independent references, same weights and batch, loss and EVERY gradient:

  port       `maxtext.experimental.mpmd.jax.deepseek` -- the plain-JAX DeepSeek-V3 port that
             deepseek_maxtext_parity_test holds bit-exact against MaxText's
             own DeepSeek-V3 on one device.  The pipeline reuses the
             port's layer code (`mla`, `mlp_block`, `rms_norm`,
             `dense_general`, rope; at EP=1 also its argsort dispatch), which
             that parity covers, but not its expert matmuls, attention kernel
             or anything distributed: the port has `ragged_dot` experts
             instead of ring-of-experts + ragged sort + megablox, jax splash
             instead of tokamax splash, a gather embedding instead of the
             iota one, no shard_map, no pipeline, no FSDP / EP collectives,
             one device.
  monolith   `monolith_loss_and_grads`: one shard_map over one stage mesh running
             every layer for every microbatch with the pipeline's own layer
             math, lax collectives and AD's accumulation -- on TPU, so the
             real kernels (SparseCore ragged sort, megablox, tokamax splash)
             run instead of their CPU fallbacks.  The benchmark's random routing.

Routing against the port: every token takes ALL experts (top-k = num_experts).
Top-k with k < E is discontinuous, and at init the bf16 sigmoid scores sit
near 0.5 where ~3-6% of tokens have an exact tie at the top-k boundary: any
last-bit difference upstream of the router (a different attention kernel, a
differently shaped matmul) flips some of them, and causal attention spreads
each flip to every later token, so two correct implementations diverge
chaotically (measured: top-2 of 8 gives gradient cosine 0.978 against the port
while top-8 of 8 gives 0.99996; the ring-of-experts block alone matches a
custom-VJP-free dense MoE to 2e-3 even with those ties).  Random routing does
not depend on the activations, so the monolith check keeps the benchmark's.

Gradients are compared in the storage layout, reduced in fp32
(`rs_dtype=float32`) so the check is not blurred by the bf16 reduce-scatter
the benchmark uses.  The presets also keep fp32 master weights and no
`--remat-last`, not the benchmark's bf16 weights; the TPU7x presets keep
MaxText's `ep_on_chip` placement, the v6e ones (`small-v6e*`, `full-dims-v6e`,
`v6e-a`) the benchmark's `fsdp_on_chip`.  Parity at every benchmark setting
is a follow-up.  Megablox tiles: `--tiles clamped` (ds_ep.TILES; the tiles do
not change the outputs) unless a preset says otherwise -- the CPU presets 128
everywhere, the full-dims v6e ones `v6e`.

`--batch-split` checks the pipeline with each device's rows of a microbatch
run as two halves (ds_ep_pipeline --batch-split).  A microbatch then needs an
even number of rows a device: `--gbs` doubles a preset's batch (`small` and
`full-dims` hold one row a device a microbatch).  Its layers:

  lineage   (the default `--batch-split-impl`) MaxText's Lineage layers
            (ds_ep_lineage), which route top-k only, on Explicit stage
            meshes.  Checked: the port, at its top-k (every expert unless
            `--topk`), at its tolerances (loss 2e-3, worst leaf 5e-2, cosine
            0.9999) -- loose here, as Lineage's numerics are not the port's
            (its splash softmax is base-2 only, 1.0018x sharper, whatever a
            preset's sa_use_base2_exp says: ignored on Lineage's layers;
            bf16 rope cos / sin; a bf16 router kernel; the routing coefficient
            applied before wo), and each costs the match a little.  Not
            checked: the monolith (its random routing has no Lineage
            counterpart: skipped, and so is its phase of `--ladder`, whose
            rung-to-rung phase routes top-k instead), nor anything at k < E
            against the port (near-tied experts flip, as above).  So only a
            preset whose one-device port holds the doubled batch is checked
            here: `small --gbs 16`, not `full-dims --gbs 16` (that port does
            not fit on one device, and `--skip-port` leaves
            nothing to check).  The tight checks of
            Lineage's layers are ds_ep_lineage_test's (against Lineage's own
            entry points, bitwise at one layer, and ds_ep's own math with
            routing pinned) and ds_ep_lineage_pipeline_test's (the pipeline
            against the unsplit one at pinned top-k routing, TPU7x).
  native    ds_ep.split_layers over the kernels above, in Lineage's phases,
            against both references: the port routes every token to all
            experts, so the halves change no routing, and the monolith routes
            each half with the key the pipeline gives it.  Off TPU only
            (`tiny`): on TPU ds_ep_pipeline refuses it for DeepSeek-V3
            (native_split_refusal).

`--ladder` leaves out its stash rungs under either (batch split runs the
recompute programs only).

    python3 -m maxtext.experimental.mpmd.jax.ds_ep_parity --preset small-v6e-stash
    python3 -m maxtext.experimental.mpmd.jax.ds_ep_parity --preset v6e-a --ladder --skip-port

`--ladder` runs several (placement, schedule, stash[, pair stages]) rungs --
`--rungs circular:search:none,circular:stash:lite,circular:interleaved:none:1,...`,
interleaved's pair stages 'all', 'none' or a '+' list such as 0+1 -- on the SAME
weights (made once per chunk index, `init_by_chunk`, and transplanted into
each rung's layout) and the SAME batch, and compares each rung's loss and
every gradient leaf with the rung before it: bit for bit where the two run
the same programs (same modes, same per-stage collective flavours, the same
F&B pairs), else within
loss rel 1e-5, cosine 0.999999 and max leaf rel err 1e-3 (a stash or
composed program may fuse differently on the TPU).  On Lineage's layers a
rung that moves a chunk between a stage libtpu offloads collectives on and
one it does not (stage 0, which holds chip (0,0,0), on a TPU7x-8) runs that
chunk's SparseCore regions as TensorCore code on one side: such a pair is
held to loss rel 1e-5, cosine 0.9999 and max leaf rel err 2e-2 instead
(TOL_SC; the switch alone moved a leaf by 1.05e-3 in ds_ep_lineage_test,
its forward bit for bit).  Pre-clip gradients are compared: the clip's
global norm sums its squares in a plan-dependent order.
Every rung is also checked against the port / the monolith as above.  On CPU
(`--preset tiny`, or `tiny4` at M = 4) every rung of the default ladder is bit
for bit the one before it; F&B pair
rungs are bitwise on CPU too (ds_ep_test.test_paired_vs_unpaired_bitwise),
but count as other programs (a pair is one program: the TPU may fuse it
differently).  Keep the splash kernel for stash rungs: under `--attn reference`
(which names none of its residuals) a stash backward recomputes the softmax
in other fusions than remat does, and on CPU its gradients moved by up to 9e-3
of a leaf (cosine 0.999994) -- outside the rung tolerance.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys

from maxtext.experimental.mpmd.jax import transfer

if __name__ == "__main__" and os.environ.get("JAX_PLATFORMS", "") != "cpu":
  transfer.set_max_inflight(1024)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from maxtext.experimental.mpmd.jax.deepseek import train as ds_train  # noqa: E402
from maxtext.experimental.mpmd.jax import ds_ep  # noqa: E402
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp  # noqa: E402

VERBOSE = False

PRESETS = {
    # CPU sanity (interpret-mode kernels): the ds_ep_test shapes
    "tiny": dict(
        layers=4,
        emb=256,
        heads=2,
        q_lora=128,
        kv_lora=128,
        moe_mlp=256,
        experts=8,
        topk=2,
        vocab=512,
        seq=256,
        gbs=8,
        mb=2,
        vv=2,
        wi_tiling=[128] * 9,
        wo_tiling=[128] * 9,
        sa_block=128,
    ),
    # ... at M = 4
    "tiny4": dict(
        layers=4,
        emb=256,
        heads=2,
        q_lora=128,
        kv_lora=128,
        moe_mlp=256,
        experts=8,
        topk=2,
        vocab=512,
        seq=256,
        gbs=16,
        mb=4,
        vv=2,
        wi_tiling=[128] * 9,
        wo_tiling=[128] * 9,
        sa_block=128,
    ),
    # layer dims shrunk so the single-device port holds the whole global batch
    "small": dict(
        layers=4,
        emb=1024,
        heads=16,
        q_lora=512,
        kv_lora=512,
        moe_mlp=512,
        experts=8,
        topk=2,
        vocab=4096,
        seq=1024,
        gbs=8,
        mb=2,
        vv=2,
    ),
    # PP=4 x FSDP=2 (over each chip's cores) x EP=1, gradients reduce-scattered every
    # microbatch, argsort dispatch (what MaxText requires at EP=1)
    "small-pp4": dict(
        layers=4,
        emb=1024,
        heads=16,
        q_lora=512,
        kv_lora=512,
        moe_mlp=512,
        experts=8,
        topk=2,
        vocab=4096,
        seq=1024,
        gbs=8,
        mb=2,
        vv=1,
        stages=4,
        fsdp=2,
        ep=1,
        layout="snake",
        grad_reduce="microbatch",
        ragged_sort="off",
    ),
    # DeepSeek-V3's own per-layer dims (the benchmark's), 2 layers, 2 microbatches
    "full-dims": dict(
        layers=2,
        emb=7168,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=32,
        topk=8,
        vocab=32768,
        seq=4096,
        gbs=8,
        mb=2,
        vv=1,
    ),
    # v6e-4 (PP2 x FSDP1 x EP2, FSDP over each chip's cores): small's layer dims but emb 2048
    # (v6e's SparseCore gather-reduce asserts below 2048: its 32 subcores need >= 4 column
    # partitions of >= 4 x 128 lanes to keep num_row_partitions <= its 8 SIMD lanes), 4
    # microbatches; -stash puts some (chunk, microbatch)s in 'lite' stash mode (a budget of
    # three sets per stage, so the scheduler mixes the modes)
    "small-v6e": dict(
        layers=4,
        emb=2048,
        heads=16,
        q_lora=512,
        kv_lora=512,
        moe_mlp=512,
        experts=8,
        topk=2,
        vocab=4096,
        seq=1024,
        gbs=8,
        mb=4,
        vv=2,
        fsdp=1,
        ep=2,
        layout="fsdp_on_chip",
    ),
    "small-v6e-stash": dict(
        layers=4,
        emb=2048,
        heads=16,
        q_lora=512,
        kv_lora=512,
        moe_mlp=512,
        experts=8,
        topk=2,
        vocab=4096,
        seq=1024,
        gbs=8,
        mb=4,
        vv=2,
        fsdp=1,
        ep=2,
        layout="fsdp_on_chip",
        schedule="stash",
        stash="lite",
        stash_sets=3,
    ),
    # the benchmark's layer dims on v6e-4: 2 layers, 16 experts, the v6e megablox tiles
    "full-dims-v6e": dict(
        layers=2,
        emb=7168,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=16,
        topk=8,
        vocab=32768,
        seq=4096,
        gbs=4,
        mb=2,
        vv=1,
        fsdp=1,
        ep=2,
        layout="fsdp_on_chip",
        tiles="v6e",
    ),
    # config A itself (4 layers, gbs 16, M 8, V 2), for the full-dims ladder (--skip-port:
    # the one-device port does not hold it)
    "v6e-a": dict(
        layers=4,
        emb=7168,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=16,
        topk=8,
        vocab=32768,
        seq=4096,
        gbs=16,
        mb=8,
        vv=2,
        fsdp=1,
        ep=2,
        layout="fsdp_on_chip",
        tiles="v6e",
    ),
    # small-pp4 at 2 chunks per stage: 8 layers and M=8, so Megatron's interleaved order (M a
    # multiple of 4) and V placement run against the port and the monolith
    "small-pp4-v2": dict(
        layers=8,
        emb=1024,
        heads=16,
        q_lora=512,
        kv_lora=512,
        moe_mlp=512,
        experts=8,
        topk=2,
        vocab=4096,
        seq=1024,
        gbs=16,
        mb=8,
        vv=2,
        stages=4,
        fsdp=2,
        ep=1,
        layout="snake",
        grad_reduce="microbatch",
        ragged_sort="off",
    ),
    # TPU7x-8 PP=4 x FSDP=2 x EP=1 at full size, the PP=4 schedule sweep's model (8 layers, 32
    # experts, fp32 weights): rung against rung only (--skip-port --skip-monolith)
    "pp4-e32": dict(
        layers=8,
        emb=7168,
        heads=128,
        q_lora=1536,
        kv_lora=512,
        moe_mlp=2048,
        experts=32,
        topk=8,
        vocab=32768,
        seq=4096,
        gbs=32,
        mb=8,
        vv=2,
        stages=4,
        fsdp=2,
        ep=1,
        layout="snake",
        grad_reduce="microbatch",
        ragged_sort="off",
    ),
}
# What a preset (or the command line) may set about the pipeline, and the defaults
# (placement None: circular; schedule None: ds_ep_pipeline.DEFAULT_SCHEDULE,
# breadth; pair_stages None: interleaved's, none).
PIPE = dict(
    placement=None,
    schedule=None,
    stash="none",
    stash_budget_gib=4.5,
    stash_caps="",
    init_by_chunk=False,
    stash_sets=None,
    pair_stages=None,
    pair_gain=None,
)


def monolith_loss_and_grads(p, params, batch):
  """(loss, grads in the storage layout) from one monolithic shard_map.

  Always uses lax collectives for EP (and the FSDP gather), so a pipeline built
  with the collective-permute flavours is checked against an independent path.
  Under batch split each layer runs on each device's two halves of the rows in
  turn, with the halves' routing keys (ds_ep.split_layers' phases and its
  whole-chunk backward are what it checks).
  """
  scfg = dataclasses.replace(p.scfg, ep_collectives="xla")
  mesh = p.meshes[0]
  chunks = p.chunks
  storage = [params[ch.stage][p.slot[ch.index]] for ch in chunks]
  storage = [jax.device_put(w, jax.tree.map(lambda x: NamedSharding(mesh, x.sharding.spec), w)) for w in storage]
  toks, tgts = batch
  tok_sh = NamedSharding(mesh, P(ds_ep.FE, None))
  toks = [jax.device_put(t, tok_sh) for t in toks]
  tgts = [jax.device_put(t, tok_sh) for t in tgts]
  sspec = [ds_ep.chunk_storage_specs(c) for c in chunks]
  cspec = [ds_ep.chunk_compute_specs(c) for c in chunks]
  denom = float(p.gbs * p.seq)
  cdt = [ds_ep.leaf_dtype_tree(scfg, c, "compute") for c in chunks]  # the pipeline's compute copies' dtypes
  M = p.M

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
    for m in range(M):
      h = ds_ep.embed(wc[0], toks[m], scfg)
      for ch, w in zip(chunks, wc):
        for l in range(ch.dense):  # chunk 0's dense layers, on all of h
          h = ds_ep.dense_layer_fwd(ds_ep.layer_weights(w["dense_layers"], l, ch.dense_kinds), h, scfg)
        for l, lid in enumerate(ch.layers):
          wl, key = ds_ep.layer_weights(w["layers"], l, ch.kinds), ds_ep._layer_key(lid, m)
          if scfg.batch_split:  # each device's rows as two halves, routed with keys of their own, one layer at a time
            r = h.shape[0] // 2
            h = jnp.concatenate(
                [
                    ds_ep.layer_fwd(wl, h[:r], jax.random.fold_in(key, 0), scfg),
                    ds_ep.layer_fwd(wl, h[r:], jax.random.fold_in(key, 1), scfg),
                ]
            )
          else:
            h = ds_ep.layer_fwd(wl, h, key, scfg)
      total = total + ds_ep.head_loss(wc[-1], h, tgts[m], denom, scfg)
    return total[None]

  f = jax.shard_map(
      body,
      mesh=mesh,
      in_specs=(sspec, [P(ds_ep.FE, None)] * M, [P(ds_ep.FE, None)] * M),
      out_specs=P(ds_ep.FE),
      check_vma=False,
  )
  loss, grads = jax.jit(jax.value_and_grad(lambda s, a, b: jnp.sum(f(s, a, b))))(storage, toks, tgts)
  return loss, grads


def build(
    preset,
    random_routing,
    attn="tokamax_splash",
    topk=None,
    batch_split=False,
    batch_split_impl="lineage",
    lineage_capacity_factor=None,
    **pipe,
):
  """The preset's pipeline.  `pipe` overrides the preset's PIPE keys
  (placement, schedule, stash, stash_budget_gib, stash_caps, init_by_chunk,
  stash_sets: a stash budget of that many sets per stage, pair_stages as
  `--pair-stages` has them, pair_gain) and fsdp / ep / layout / tiles / gbs.
  A budget given in GiB replaces the preset's in sets.  `batch_split`: the
  stage programs run each device's rows as two halves
  (StageConfig.batch_split), on `batch_split_impl`'s layers: Lineage's
  (top-k routing only, Explicit stage meshes: ds_ep_pipeline resolves the
  shard mode) or ds_ep.split_layers ('native', off TPU only)."""
  k = dict(PIPE, **PRESETS[preset])
  pipe = {n: v for n, v in pipe.items() if v is not None}
  if "stash_budget_gib" in pipe and "stash_sets" in pipe:
    raise ValueError("give the stash budget once: stash_budget_gib or stash_sets")
  if "stash_budget_gib" in pipe:
    k["stash_sets"] = None
  k.update(pipe)
  if topk:
    k["topk"] = topk
  a = argparse.Namespace(
      stages=k.get("stages", 2),
      vv=k["vv"],
      mb=k["mb"],
      gbs=k["gbs"],
      seq=k["seq"],
      fsdp=k.get("fsdp", 2),
      ep=k.get("ep", 2),
      layers=k["layers"],
      emb=k["emb"],
      heads=k["heads"],
      q_lora=k["q_lora"],
      kv_lora=k["kv_lora"],
      moe_mlp=k["moe_mlp"],
      experts=k["experts"],
      topk=k["topk"],
      vocab=k["vocab"],
      attn=attn,
      sa_block=k.get("sa_block", 2048),
      sa_block_kv_compute=min(512, k.get("sa_block", 2048)),
      cost_estimate=5_000_000_000_000,
      tiles=k.get("tiles", "clamped"),
      wi_tiling=k.get("wi_tiling"),
      wo_tiling=k.get("wo_tiling"),
      rs_dtype="float32",
      acc_dtype="float32",
      remat_last=False,
      schedule=k["schedule"],
      layout=k.get("layout", "ep_on_chip"),
      no_overlap=False,
      fsdp_collectives="auto",
      ep_collectives="auto",
      pf_split=4,
      rs_split=4,
      costs="",
      grad_reduce=k.get("grad_reduce", "step"),
      routing="random" if random_routing else "topk",
      shard_mode=None,
      ragged_sort=k.get("ragged_sort", "auto"),
      placement=k["placement"],
      stash=k["stash"],
      stash_budget_gib=k["stash_budget_gib"],
      stash_caps=k["stash_caps"],
      init_by_chunk=k["init_by_chunk"],
      pair_stages=k["pair_stages"],
      pair_gain=k["pair_gain"],
      batch_split=batch_split,
      batch_split_impl=batch_split_impl,
  )
  if lineage_capacity_factor is not None:
    a.lineage_capacity_factor = lineage_capacity_factor
  if k["stash_sets"] is not None and k["stash"] != "none":
    # a budget in whole sets: the set's size from the same pipeline without stash, under search, which runs no
    # F&B pairs (build refuses the pair flags outside interleaved)
    plain = {**vars(a), "schedule": "search", "stash": "none", "pair_stages": None, "pair_gain": None}
    b = dsp.build(argparse.Namespace(**plain), random_routing=random_routing).progs[0].build_stash(k["stash"]).bytes
    a.stash_budget_gib = k["stash_sets"] * b / 2**30
  # batch_split rides in the namespace, as --batch-split does, so resolve_tilings caps the wi dW tile (split_dw 'acc')
  return dsp.build(a, random_routing=random_routing)


def scale_gate(p, params, factor):
  """Scale every router kernel, then re-gather the pre-gathered chunks."""
  out = []
  for s_, ps in enumerate(params):
    new = []
    for t in ps:
      t = dict(t)
      lay = dict(t["layers"])
      moe = dict(lay["moe"])
      moe["gate"] = {"kernel": moe["gate"]["kernel"] * factor}
      lay["moe"] = moe
      t["layers"] = lay
      new.append(t)
    out.append(new)
  return out, [st.gather_first(q) for st, q in zip(p.stages, out)]


def to_host(tree):
  return jax.tree.map(lambda x: np.asarray(jax.device_get(x), np.float32), tree)


def mpmd_loss_and_grads(p, params, wc0, batch):
  grads, _, lacc = p.loss_and_grads_overlap(params, wc0, batch)
  return float(jnp.sum(lacc)), [[to_host(g) for g in gs] for gs in grads]


def stacked(ch):
  """Chunk `ch` with its layers stacked (layer_tree 'stacked'), whatever its own layer tree."""
  return dataclasses.replace(ch, kinds=(), dense_kinds=())


def relayout(t, src, dst):
  """A DeepSeek chunk tree `t` (weights or gradients) in chunk `src`'s layer tree -> in chunk `dst`'s: its 'layers'
  and 'dense_layers' stacked ([n, ...] leaves) or one subtree a layer with a layer axis of one (layer_tree
  'per_layer': non-empty `ChunkSpec.kinds` / `dense_kinds`).  Host arrays out where a stack is made; a cut is a
  view."""
  t = dict(t)
  for key, a, b in (("layers", src.kinds, dst.kinds), ("dense_layers", src.dense_kinds, dst.dense_kinds)):
    if key not in t or bool(a) == bool(b):
      continue
    if a:
      t[key] = jax.tree.map(
          lambda *xs: np.concatenate(xs, axis=0), *(t[key][ds_ep.qwen_layer_key(l)] for l in range(len(a)))
      )
    else:
      t[key] = {ds_ep.qwen_layer_key(l): jax.tree.map(lambda x, l=l: x[l : l + 1], t[key]) for l in range(len(b))}
  return t


def port_tree(p, chunk_trees):
  """ds_ep chunk trees (in chunk order, in the pipeline's layer tree) -> the param tree of the DeepSeek-V3 port
  (maxtext.experimental.mpmd.jax.deepseek)."""
  chunk_trees = [relayout(t, ch, stacked(ch)) for ch, t in zip(p.chunks, chunk_trees, strict=True)]

  def moe_to_port(layer):
    moe = layer["moe"]
    return {
        **{k: v for k, v in layer.items() if k != "moe"},
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {"gate": moe["gate"], "wi_0": moe["wi_0"], "wi_1": moe["wi_1"], "wo": moe["wo"]},
            "shared_experts": moe["shared_experts"],
        },
    }

  cat = lambda trees: jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *trees)
  decoder = {
      "moe_layers": cat([moe_to_port(t["layers"]) for t in chunk_trees if "layers" in t]),
      "decoder_norm": chunk_trees[-1]["decoder_norm"],
      "logits_dense": chunk_trees[-1]["logits_dense"],
  }
  dense = [t["dense_layers"] for t in chunk_trees if "dense_layers" in t]
  if dense:  # the dense layers, chunk 0's (or the first chunks', in order)
    decoder["dense_layers"] = cat(dense)
  return {"token_embedder": chunk_trees[0]["token_embedder"], "decoder": decoder}


def port_to_chunks(p, g):
  """Inverse of port_tree for gradients: -> list of ds_ep chunk trees, in the pipeline's layer tree."""
  out, start, dstart = [], 0, 0
  for ch in p.chunks:
    n = len(ch.layers)
    t = {}
    if n:
      lay = jax.tree.map(lambda x: x[start : start + n], g["decoder"]["moe_layers"])
      start += n
      blk = lay.pop("DeepSeekMoeBlock_0")
      lay["moe"] = {**blk["MoeBlock_0"], "shared_experts": blk["shared_experts"]}
      t["layers"] = lay
    if ch.dense:
      t["dense_layers"] = jax.tree.map(lambda x: x[dstart : dstart + ch.dense], g["decoder"]["dense_layers"])
      dstart += ch.dense
    if ch.first:
      t["token_embedder"] = g["token_embedder"]
    if ch.last:
      t["decoder_norm"] = g["decoder"]["decoder_norm"]
      t["logits_dense"] = g["decoder"]["logits_dense"]
    out.append(relayout(t, stacked(ch), ch))
  return out


def port_loss_and_grads(p, params_host, tokens, targets):
  cfg = dataclasses.replace(
      p.scfg.model,
      attention_kernel="splash" if jax.default_backend() == "tpu" else "reference",
      moe_impl="ragged",
      gmm_fp32_accum=True,
  )
  dev = jax.devices()[0]
  chunk_trees = [params_host[ch.stage][p.slot[ch.index]] for ch in p.chunks]
  params = jax.device_put(port_tree(p, chunk_trees), dev)
  B, L = tokens.shape
  batch = jax.device_put(
      {
          "inputs": tokens,
          "targets": targets,
          "inputs_position": np.broadcast_to(np.arange(L, dtype=np.int32), (B, L)).copy(),
          "inputs_segmentation": np.ones((B, L), np.int32),
          "targets_segmentation": np.ones((B, L), np.int32),
      },
      dev,
  )
  tcfg = ds_train.TrainConfig()
  f = jax.jit(jax.value_and_grad(lambda pr: ds_train.compute_loss(pr, {}, batch, cfg, tcfg)[0]))
  loss, g = f(params)
  return float(loss), port_to_chunks(p, to_host(g))


def port_floor(p, params_host, tokens, targets):
  """The reassociation floor: the port against ITSELF, whole batch vs the sum of
  per-microbatch gradients -- the same numbers in a different summation order,
  which is all a correct pipeline is allowed to change."""
  _, whole = port_loss_and_grads(p, params_host, tokens, targets)
  parts = None
  for m in range(p.M):
    sl = slice(m * p.mbs, (m + 1) * p.mbs)
    _, g = port_loss_and_grads(p, params_host, tokens[sl], targets[sl])
    g = jax.tree.map(lambda x: x * (p.mbs / p.gbs), g)  # per-microbatch mean -> share of the batch mean
    parts = g if parts is None else jax.tree.map(np.add, parts, g)
  as_grads = [[None] * len(p.stage_chunks[s]) for s in range(p.S)]
  for ch in p.chunks:
    as_grads[ch.stage][p.slot[ch.index]] = parts[ch.index]
  return errors(p, as_grads, whole)


def errors(p, grads, ref_chunks):
  """-> (sorted [(rel_err, leaf name)], whole-model cosine, norm ratio)."""
  return chunk_errors([grads[ch.stage][p.slot[ch.index]] for ch in p.chunks], ref_chunks)


def chunk_errors(got_chunks, ref_chunks):
  """errors() of two lists of chunk trees, in chunk order."""
  rows, dot, na, nb = [], 0.0, 0.0, 0.0
  for c, (g, r) in enumerate(zip(got_chunks, ref_chunks, strict=True)):
    ga = jax.tree_util.tree_flatten_with_path(g)[0]
    gb = jax.tree.leaves(r)
    for (path, a), b in zip(ga, gb, strict=True):
      assert a.shape == b.shape, (jax.tree_util.keystr(path), a.shape, b.shape)
      a64, b64 = a.astype(np.float64), b.astype(np.float64)
      rows.append((float(np.linalg.norm(a64 - b64) / (np.linalg.norm(b64) + 1e-30)), f"c{c}{jax.tree_util.keystr(path)}"))
      dot += float(np.vdot(a64, b64))
      na += float(np.vdot(a64, a64))
      nb += float(np.vdot(b64, b64))
  rows.sort(reverse=True)
  return rows, dot / np.sqrt(na * nb), np.sqrt(na / nb)


def compare(label, p, loss, ref_loss, grads, ref_chunks, tol_loss, tol_leaf):
  print(f"\n=== {label}")
  rl = abs(loss - ref_loss) / abs(ref_loss)
  print(f"  loss  mpmd {loss:.7f}  ref {ref_loss:.7f}  rel err {rl:.2e}  (tol {tol_loss:.0e})")
  rows, cos, gn = errors(p, grads, ref_chunks)
  print(f"  gradients: {len(rows)} leaves, whole-model cosine {cos:.7f}, norm ratio {gn:.6f}")
  print(
      f"  per-leaf relative error: median {np.median([e for e, _ in rows]):.2e}, "
      f"worst {rows[0][0]:.2e}  (tol {tol_leaf:.0e})"
  )
  for e, n in sorted(rows, key=lambda r: r[1]) if VERBOSE else rows[:5]:
    print(f"      {e:.2e}  {n}")
  ok = rl < tol_loss and rows[0][0] < tol_leaf and cos > 0.9999
  print(f'  {"PASS" if ok else "FAIL"}')
  return ok


# ------------------------------------------------------------------ ladder
# (placement, schedule, stash[, pair stages]) rungs; each is compared with the
# one before it.
RUNGS = "circular:search:none,circular:timed:none,circular:stash:lite,v:search:none,v:timed:none,v:stash:lite"
# rung to rung where the programs differ (stash, or another collective
# flavour): what a TPU fusing a program differently may move, and no more
TOL_RUNG = dict(loss=1e-5, cosine=0.999999, leaf=1e-3)
# ... where a chunk's stage's Lineage SparseCore answer differs (on Lineage's
# layers, sparsecore_moves): its Region runs those regions as TensorCore code
# on one side, an engine change, not a fusion one -- the cosine and worst leaf
# ds_ep_lineage_test holds that switch alone to (test_sparsecore_off_vs_on;
# measured: the forward bit for bit, a leaf 1.05e-3 apart), the loss as above
TOL_SC = dict(loss=1e-5, cosine=0.9999, leaf=2e-2)


def parse_rungs(spec, vv, mb=None, stages=2, batch_split=False):
  """--rungs -> [the rung's pipeline keys], leaving out with a note the rungs the preset cannot run, a rung naming
  pair stages for a schedule other than interleaved (the only one that pairs on stages), and under `batch_split`
  the stash rungs (batch split runs the recompute programs only)."""
  out = []
  for r in spec.split(","):
    placement, schedule, stash, *pairs = r.split(":")
    if placement == "v" and vv != 2:
      print(f"  skipping rung {r}: V placement needs vv 2 (the preset has {vv})")
      continue
    if batch_split and stash != "none":
      print(f"  skipping rung {r}: batch split runs the recompute programs only (no stash)")
      continue
    if pairs and schedule != "interleaved":
      print(f"  skipping rung {r}: only interleaved pairs F and B on stages (pair stages {pairs[0]})")
      continue
    if schedule == "interleaved" and (placement != "circular" or vv < 2 or (mb is not None and mb % stages)):
      print(
          f"  skipping rung {r}: interleaved 1F1B needs circular placement, vv >= 2 and M a "
          f"multiple of S = {stages} (the preset has vv {vv}, M {mb})"
      )
      continue
    rung = dict(placement=placement, schedule=schedule, stash=stash)
    if pairs:
      rung["pair_stages"] = pairs[0].replace("+", ",")
    out.append(rung)
  return out


def programs_of(p):
  """What decides whether two rungs run the same arithmetic: every (chunk,
  microbatch)'s mode, the collective flavours of each chunk's stage --
  under batch split on Lineage's layers also whether it runs their
  SparseCore regions (ds_ep_lineage_test measured 1.05e-3 a leaf between
  the two: such rungs are held to TOL_SC, sparsecore_moves) -- and which
  ops run together as one F&B pair program."""
  sc = p.lineage_sparsecore()
  return (
      tuple(sorted(p.modes.items())),
      tuple(
          (p.stage_scfg[ch.stage].ep_collectives, p.stage_scfg[ch.stage].fsdp_collectives, sc.get(ch.stage))
          for ch in p.chunks
      ),
      tuple(sorted((p.order[j], p.order[j + 1]) for j in p.fuse)),
  )


def sparsecore_moves(progs, prev):
  """The chunks whose stage's Lineage SparseCore answer differs between two rungs' `programs_of` -> [chunk]; [] off
  Lineage's layers, where every answer is None."""
  return [c for c, (x, y) in enumerate(zip(progs[1], prev[1], strict=True)) if x[2] != y[2]]


def compare_rungs(label, loss, prev_loss, grads, prev, same, strict, moved=()):
  """One rung against the rung before it (chunk-order host trees).  `same`:
  they run the same programs, so the gradients should be bit for bit equal
  (a miss within TOL_RUNG fails under `strict`, else only warns).  `moved`:
  the chunks whose Lineage SparseCore answer differs between the two
  (sparsecore_moves), which holds them to TOL_SC instead of TOL_RUNG."""
  tol = TOL_SC if moved else TOL_RUNG
  print(f"\n=== {label}")
  bitwise = loss == prev_loss and all(
      np.array_equal(x, y) for x, y in zip(jax.tree.leaves(grads), jax.tree.leaves(prev), strict=True)
  )
  rl = abs(loss - prev_loss) / abs(prev_loss)
  rows, cos, gn = chunk_errors(grads, prev)
  print(
      f"  loss {loss:.7f} vs {prev_loss:.7f} (rel {rl:.2e}); {len(rows)} leaves, cosine {cos:.9f}, "
      f"norm ratio {gn:.9f}, worst leaf rel err {rows[0][0]:.2e} ({rows[0][1]})"
  )
  near = rl <= tol["loss"] and cos >= tol["cosine"] and rows[0][0] < tol["leaf"]
  within = (
      f"the SparseCore switch's tolerance, TOL_SC: chunk(s) {list(moved)} run Lineage's SparseCore regions "
      "on one side only"
      if moved
      else "tolerance"
  )
  if bitwise:
    verdict, ok = "PASS (bitwise)", True
  elif same:
    ok = near and not strict
    verdict = ("WARN" if ok else "FAIL") + ": the same programs, not bitwise" + (f" (within {within})" if near else "")
  else:
    verdict, ok = (
        f"PASS (within {within}; the programs differ)" if near else "FAIL" + (f" (outside {within})" if moved else "")
    ), near
  print(f"  {verdict}")
  return ok


def ladder(a, pipe):
  """Every rung on one set of weights and one batch: against the port / the
  monolith (as `main` checks one pipeline), and against the rung before it."""
  pre = PRESETS[a.preset]
  lineage = a.batch_split and a.batch_split_impl == "lineage"  # top-k routing only: no monolith phase
  rungs = parse_rungs(a.rungs, pre["vv"], pre["mb"], pre.get("stages", 2), batch_split=a.batch_split)
  phases = []
  if not a.skip_port:
    phases.append(("port", False, PRESETS[a.preset]["experts"] if a.topk == -1 else (a.topk or None), 0))
  if not a.skip_monolith and not lineage:
    phases.append(("monolith", True, None, 1))
  if not phases:
    # neither reference fits (config A: the one-device port, the one-program monolith):
    # rung against rung only, with the benchmark's random routing (Lineage's layers: the preset's top-k)
    phases.append(("rungs", not lineage, None, 1))
  if not rungs or (len(rungs) == 1 and phases[0][0] == "rungs"):
    return []  # nothing to compare with anything: `finish` fails the run before a rung is built
  results = []
  for phase, random_routing, topk, seed in phases:
    prev = None
    for i, r in enumerate(rungs):
      name = ":".join(r.values()).replace(",", "+")
      keys = {**pipe, **r, "init_by_chunk": True}
      if r["schedule"] != "interleaved":  # --pair-stages / --pair-gain are its pairs' (ds_ep_pipeline refuses them else)
        keys = {k: v for k, v in keys.items() if k not in ("pair_stages", "pair_gain")}
      p = build(
          a.preset,
          random_routing,
          attn=a.attn if phase == "port" else "tokamax_splash",
          topk=topk,
          batch_split=a.batch_split,
          **keys,
      )
      print(f"\n##### rung {name} ({phase})\n" + p.summary(), flush=True)
      if i == 0:
        # the weights and the batch every rung of this phase shares
        params, _, _ = p.init(jax.random.key(seed))
        trees = [to_host(t) for t in p.chunk_trees(params)]
        rng = np.random.default_rng(0 if phase == "port" else 3)
        tokens = rng.integers(0, p.scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
        targets = np.roll(tokens, -1, axis=1)
        if phase == "port":
          host = [[trees[ch.index] for ch in p.stage_chunks[s]] for s in range(p.S)]
          ref_loss, ref = port_loss_and_grads(p, host, tokens, targets)
        elif phase == "rungs":
          ref_loss = ref = None
        else:
          ref_loss, ref_grads = monolith_loss_and_grads(p, params, p.put_batch(tokens, targets))
          ref_loss, ref = float(ref_loss), [to_host(g) for g in ref_grads]
          del ref_grads
        del params
        gc.collect()
      params = p.params_from_chunks(trees)
      wc0 = [st.gather_first(q) for st, q in zip(p.stages, params)]
      loss, grads = mpmd_loss_and_grads(p, params, wc0, p.put_batch(tokens, targets))
      if ref is not None:
        results.append(compare(f"rung {name} vs {phase}", p, loss, ref_loss, grads, ref, 2e-3, 5e-2))
      else:
        print(f"  loss {loss:.7f}", flush=True)
      chunks, progs = p.chunk_trees(grads), programs_of(p)
      if prev is not None:
        results.append(
            compare_rungs(
                f"rung {name} vs rung {prev[0]} ({phase})",
                loss,
                prev[1],
                chunks,
                prev[2],
                progs == prev[3],
                a.strict_bitwise,
                moved=sparsecore_moves(progs, prev[3]),
            )
        )
      prev = (name, loss, chunks, progs)
      del p, params, wc0, grads
      gc.collect()
  return results


def finish(results):
  """Exit with the verdict of the checks `results` (each True for a pass): 0 only when something was compared and
  every comparison passed.  Flags that leave nothing to compare -- --skip-port with the monolith phase skipped too
  (--skip-monolith, or --batch-split on Lineage's layers: the monolith routes at random, which they cannot), a
  --ladder whose rungs are all skipped or of one rung with no reference -- fail, rather than report ALL PASS over no
  check."""
  if not results:
    sys.stdout.flush()  # the notes on skipped rungs before this, where a log takes stdout and stderr together
    sys.exit(
        "ds_ep_parity: nothing checked -- every reference phase was skipped (--skip-port; the monolith by "
        "--skip-monolith, or by --batch-split on Lineage's layers), or a --ladder had no rung to run (each "
        "skipped, as printed above) or one rung and no reference"
    )
  print("\nALL PASS" if all(results) else "\nSOME CHECK FAILED")
  sys.exit(0 if all(results) else 1)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--preset", default="small", choices=list(PRESETS))
  ap.add_argument("--skip-port", action="store_true")
  ap.add_argument("--verbose", action="store_true")
  ap.add_argument("--floor", action="store_true", help="also measure the summation-order floor")
  ap.add_argument(
      "--attn", default="tokamax_splash", help="pipeline attention kernel ('reference' = the port's CPU kernel)"
  )
  ap.add_argument(
      "--topk", type=int, default=-1, help="experts per token for the port check (-1: all experts, see the docstring)"
  )
  ap.add_argument(
      "--gate-scale", type=float, default=1.0, help="multiply the router weights at init (margins between top-k choices)"
  )
  ap.add_argument("--skip-monolith", action="store_true")
  # the pipeline (default: the preset's)
  ap.add_argument("--placement", default=None, choices=["circular", "v"])
  ap.add_argument("--schedule", default=None)
  ap.add_argument("--stash", default=None)
  ap.add_argument("--stash-budget-gib", type=float, default=None)
  ap.add_argument("--stash-sets", type=int, default=None, help="a stash budget of this many sets per stage")
  ap.add_argument("--stash-caps", default=None)
  ap.add_argument("--init-by-chunk", action="store_true", default=None)
  ap.add_argument("--pair-stages", default=None, help="all | none | 1 | 0,1 (see ds_ep_pipeline)")
  ap.add_argument("--pair-gain", type=float, default=None)
  ap.add_argument(
      "--ladder",
      action="store_true",
      help="run --rungs on the same weights and batch, each against the one before and " "the references",
  )
  ap.add_argument(
      "--rungs", default=RUNGS, help="placement:schedule:stash[:pair stages],... (pair stages: all, none, 1, 0+1)"
  )
  ap.add_argument(
      "--strict-bitwise",
      action="store_true",
      help="--ladder: two rungs running the same programs must be bitwise equal (else a "
      "miss within tolerance only warns)",
  )
  ap.add_argument(
      "--batch-split",
      action="store_true",
      help="each device's rows of a microbatch as two halves in Lineage's phases (ds_ep_pipeline "
      "--batch-split): wants an even number of rows a device, see --gbs",
  )
  ap.add_argument(
      "--batch-split-impl",
      default="lineage",
      choices=list(ds_ep.BATCH_SPLIT_IMPLS),
      help="--batch-split's layers, as ds_ep_pipeline's flag: lineage (MaxText's Lineage layers, TPU "
      "only: the port check only, at top-k, so a preset whose one-device port holds the doubled "
      "batch; see the docstring) or native (ds_ep.split_layers, both references; off TPU only)",
  )
  ap.add_argument(
      "--lineage-capacity-factor",
      type=float,
      default=None,
      help="--batch-split on Lineage's layers: ds_ep_pipeline's flag (default its default)",
  )
  ap.add_argument("--gbs", type=int, default=None, help="the preset's global batch, overridden")
  a = ap.parse_args(argv)
  global VERBOSE
  VERBOSE = a.verbose
  pipe = {k: getattr(a, k) for k in PIPE if getattr(a, k, None) is not None}
  if a.gbs is not None:
    pipe["gbs"] = a.gbs
  lineage = a.batch_split and a.batch_split_impl == "lineage"
  if lineage:
    ns = argparse.Namespace(batch_split=True, batch_split_impl="lineage")
    why = dsp.lineage_missing(ns)
    if why is None and (a.ladder or not a.skip_port):  # (else nothing runs Lineage's layers: `finish` says so)
      why = dsp.lineage_off_tpu(ns)
    if why is not None:
      sys.exit(f"ds_ep_parity --batch-split: {why}")
    pipe.update(batch_split_impl="lineage", lineage_capacity_factor=a.lineage_capacity_factor)
  elif a.batch_split:
    pipe["batch_split_impl"] = "native"
  # the flags only Lineage's layers read, given where they do not run (ds_ep_pipeline's warning); a given
  # --lineage-capacity-factor is told apart here even at ds_ep_pipeline's default (ours is None)
  cf = {} if a.lineage_capacity_factor is None else {"lineage_capacity_factor": a.lineage_capacity_factor}
  for line in dsp.lineage_unused(
      argparse.Namespace(batch_split=a.batch_split, batch_split_impl=a.batch_split_impl, **cf), given=set(cf)
  ):
    print(f"warning: {line}", flush=True)
  split = (", batch split on Lineage's layers" if lineage else ", batch split (native)") if a.batch_split else ""
  results = []

  if a.ladder:
    finish(ladder(a, pipe))

  if not a.skip_port:
    p = build(
        a.preset,
        random_routing=False,
        attn=a.attn,
        topk=PRESETS[a.preset]["experts"] if a.topk == -1 else (a.topk or None),
        batch_split=a.batch_split,
        **pipe,
    )
    print(p.summary(), flush=True)
    params, _, wc0 = p.init(jax.random.key(0))
    if a.gate_scale != 1.0:
      params, wc0 = scale_gate(p, params, a.gate_scale)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, p.scfg.model.vocab_size, size=(p.gbs, p.seq), dtype=np.int32)
    targets = np.roll(tokens, -1, axis=1)
    batch = p.put_batch(tokens, targets)
    loss, grads = mpmd_loss_and_grads(p, params, wc0, batch)
    params_host = [[to_host(t) for t in ps] for ps in params]
    del params, wc0, batch
    gc.collect()
    ref_loss, ref = port_loss_and_grads(p, params_host, tokens, targets)
    if a.floor:
      rows, cos, _ = port_floor(p, params_host, tokens, targets)
      print(f"\n=== summation-order floor (port whole batch vs port summed over {p.M} microbatches)")
      print(
          f"  cosine {cos:.7f}; per-leaf relative error median {np.median([e for e, _ in rows]):.2e}, "
          f"worst {rows[0][0]:.2e} ({rows[0][1]})"
      )
    results.append(
        compare(
            f"MPMD pipeline vs maxtext.experimental.mpmd.jax.deepseek port (top-{p.scfg.model.num_experts_per_tok}"
            f" of {p.scfg.model.num_experts} routing, {a.preset}{split})",
            p,
            loss,
            ref_loss,
            grads,
            ref,
            2e-3,
            5e-2,
        )
    )
    del p
    gc.collect()

  if not a.skip_monolith and lineage:
    print(
        "\n=== monolith: skipped -- it routes at random, which Lineage's layers cannot (off TPU, --batch-split-impl "
        "native checks ds_ep's own split layers against it)",
        flush=True,
    )
  elif not a.skip_monolith:
    p = build(a.preset, random_routing=True, batch_split=a.batch_split, **pipe)
    params, _, wc0 = p.init(jax.random.key(1))
    batch = p.synthetic_batch(3)
    loss, grads = mpmd_loss_and_grads(p, params, wc0, batch)
    ref_loss, ref_grads = monolith_loss_and_grads(p, params, batch)
    results.append(
        compare(
            f"MPMD pipeline vs monolithic single-program reference (random routing, {a.preset}{split})",
            p,
            loss,
            float(ref_loss),
            grads,
            [to_host(r) for r in ref_grads],
            2e-3,
            5e-2,
        )
    )
  finish(results)


if __name__ == "__main__":
  main()
