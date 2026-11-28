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

"""ds_ep_pipeline --batch-split on Lineage's layers (ds_ep_lineage), end to end on a TPU7x of 8 devices (a tpu7x-8).

    JAX_PLATFORMS=tpu MAXTEXT_MPMD_REQUIRE_TPU=1 MAXTEXT_MPMD_REQUIRE_LINEAGE=1 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_lineage_pipeline_test.py -s [-k ...]

Each test builds a pipeline from ds_ep_pipeline's own flags (`build`), PP=2 x (fsdp 2 x ep 2) on the 8 devices:
stage 0 is devices 0-3, which hold chip (0,0,0), so libtpu offloads no collective there and its Regions trace
Lineage's SparseCore regions as TensorCore code (ds_ep_lineage.no_sparsecore) of their own accord; stage 1 is
devices 4-7, with SparseCore.  It runs the pipeline with --batch-split (Lineage's layers: --routing and
--shard-mode at the defaults --batch-split gives them, topk and explicit) and the same pipeline unsplit (ds_ep's
own layers: MaxText's megablox and ragged-sort kernels, tokamax splash) from one set of weights and one batch, and
holds the split one to the unsplit one at ds_ep_parity's port tolerances: loss rel < 2e-3, cosine of all the
reduced gradients > 0.9999, worst gradient leaf rel < 5e-2 -- Lineage's numerics are not ds_ep's (base-2 softmax
in both here, but bf16 rope cos / sin, a bf16 router kernel, the routing coefficient before wo).

Routing is pinned (ROUTING_PIN), as ds_ep_lineage_test pins it: a routed bias of 8 on two experts of each of four
groups, every token picks those eight whatever its sigmoid scores.  Unpinned, top-k routing on bf16-rounded scores
flips the tokens near a tie on any last-bit difference between two programs, more in each later layer, and the
router kernel's gradient moves by several percent with them (ds_ep_lineage_test, layer level: cosine 0.99954, worst
leaf 9.4e-2 against ds_ep's math unpinned; 0.999976 / 8.7e-3 pinned) -- below this cosine bound for reasons that are
no bug.  Everything but the choice of experts stays live: the routing coefficients (sigmoid scores, normalized and
scaled), the router kernel's gradient through them, both EP shards' experts (four of the eight on each).

Gradients are reduced in fp32 (--rs-dtype float32) so the check is not blurred by a bf16 reduce-scatter, and the
master weights are fp32 for the training steps.  Skipped unless Lineage imports (typeguard and jaxtyping
installed) and -- asked only then, as the class sets up: it starts jax's backend --
this is a TPU7x with 8 devices: MaxText's TPU CI (a v6e-4, Lineage's typeguard and jaxtyping not installed) skips
it.  MAXTEXT_MPMD_REQUIRE_LINEAGE=1 makes a Lineage that does not import a failure instead (ds_ep_lineage_test, which
this module imports, raises): the command above sets it, as every run meant for Lineage should, where a dependency
left out would otherwise skip all five.  So does MAXTEXT_MPMD_REQUIRE_TPU=1 a missing TPU7x (testing.require_tpu): a
run left on another backend would otherwise pass as five skips.  On Pathways' devices the class skips
(testing.skip_on_pathways): Pathways' servers offload stage 0's collectives to SparseCore too, so the checks of a
stage 0 without SparseCore are a TPU VM's own client's (McJAX).
Each test compiles two pipelines (the split pipeline's programs lower in ~15 s each, Lineage's tracing): 21.8 min for
the five in one process on a tpu7x-8, libtpu 0.0.46, so the class is scheduled_only.  In one process, pipelines built
from the same flags share their Regions' traces (ds_ep_lineage.region: test_v1 reuses test_training_steps'), so a
test run alone (-k) takes longer than its share.  Numbers go to stdout (-s).
"""

import collections
import re
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import testing
# (raises under MAXTEXT_MPMD_REQUIRE_LINEAGE=1 where Lineage does not import: module doc)
from tests.unit.experimental.mpmd.jax.ds_ep_lineage_test import NO_LINEAGE, SC_STAGE0, _on_tpu7x

# The model: Lineage's DeepSeek-V3 layer at small widths (D / 128 even, E % EP == 0, groups 8 / 4 as DeepSeek-V3's),
# two MoE layers a chunk; 2 rows a device a microbatch (the halves are one row each), M = 2 (an accumulating
# backward and fused last chunk per chunk).  Blocks of 1024: ds_ep's tokamax splash dkv at 2048 wants libtpu's
# scoped-VMEM flag, which Lineage's programs get from ds_ep_lineage.compiler_options but the unsplit ones would not.
BASE = (
    "--emb 1024 --heads 16 --q-lora 512 --kv-lora 512 --moe-mlp 512 --experts 32 --topk 8 --routing-groups 8 "
    "--topk-groups 4 --routed-bias on --vocab 4096 --seq 2048 --sa-block 1024 --sa-block-kv-compute 512 "
    "--stages 2 --fsdp 2 --ep 2 --mb 2 --gbs 16 --rs-dtype float32 --acc-dtype float32 "
    "--weight-dtype float32"
).split()
# What the unsplit pipeline is given that --batch-split defaults to (top-k routing, Explicit stage meshes).
UNSPLIT = ["--routing", "topk", "--shard-mode", "explicit"]
TOL = dict(loss=2e-3, cosine=0.9999, leaf=5e-2)  # ds_ep_parity's port tolerances


def routing_pin(cfg):
  """[E] routed bias: 8 on experts 0, 1 of groups 0, 1, G/2, G/2 + 1 (four on each of the two EP shards), 0 elsewhere.
  Scores are sigmoids in (0, 1): a pinned group scores >= 16 by its top two, an unpinned one < 2, so every token
  takes the four pinned groups and, in each, its two pinned experts -- k = 8 of 32, whatever the scores."""
  g = cfg.num_experts // cfg.n_routing_groups
  pin = np.zeros(cfg.num_experts, np.float32)
  for grp in (0, 1, cfg.n_routing_groups // 2, cfg.n_routing_groups // 2 + 1):
    pin[grp * g : grp * g + 2] = 8.0
  return pin


def pinned(trees, pin):
  """Host chunk trees with every MoE layer's routed bias set to `pin`."""
  out = []
  for t in trees:
    t = jax.tree.map(np.asarray, t)
    if "layers" in t:
      b = t["layers"]["moe"]["gate"]["bias"]
      t["layers"]["moe"]["gate"]["bias"] = np.broadcast_to(pin.astype(b.dtype), b.shape).copy()
    out.append(t)
  return out


def chunk_leaves(p, grads):
  """Per-stage reduced gradients -> [(chunk-qualified path, host fp64 array)] in chunk order."""
  out = []
  for c, t in enumerate(p.chunk_trees(grads)):
    for path, x in jax.tree_util.tree_flatten_with_path(t)[0]:
      out.append((f"c{c}{jax.tree_util.keystr(path)}", np.asarray(jax.device_get(x), np.float64)))
  return out


def sparsecore_regions(p):
  """{stage: Lineage's SparseCore regions (compute_on('tpu_sparsecore'): "sparseoffload" in the lowered text) in its
  first chunk's forward (the fused program of the last chunk)} -- what each stage's Regions traced under their
  SparseCore answer: none where libtpu offloads no collective (ds_ep_lineage.no_sparsecore), some elsewhere.  Only
  the lowered program tells: compiled, a stage that offloads runs Lineage's lax collectives on SparseCore either way
  (traced with sparsecore False on stage 1, the counts below and the loss came out the same)."""
  out = {}
  for s in range(p.S):
    c = p.stage_chunks[s][0].index
    desc = (ds_ep.pdesc("A" if p.chunks[c].last else "F"),)
    out[s] = p.progs[c].program(desc).trace(*p._desc_avals(c, desc)).lower().as_text().count("sparseoffload")
  return out


def sparsecore_collectives(p):
  """{stage: collectives inside the computations its compiled programs run on SparseCore} (`execution_thread=
  "sparsecore"`), from `precompile`'s executables: Lineage's compute_on('tpu_sparsecore') regions', and the
  collectives libtpu offloads there itself -- the FSDP side-cars' all-gathers / reduce-scatters where the stage
  offloads, as many in the unsplit pipeline (6-18 a program on stage 1 here), and on such a stage Lineage's own lax
  collectives, regions or not (sparsecore_regions) -- so `check` holds the split pipeline's count above the unsplit
  one's: Lineage's EP collectives run on SparseCore there.  (libtpu's own copy and gather offloads,
  async_execution_thread, are not counted.)"""
  coll = re.compile(r"\b(all-gather|reduce-scatter|all-reduce|all-to-all|collective-permute)(-start)?\(")
  out = collections.Counter({s: 0 for s in range(p.S)})
  for (_, c), s in zip(p.compiled, p.compiled_stage):
    if isinstance(s, int):
      blocks = re.split(r"\n(?=(?:ENTRY )?%[\w.\-]+ \()", c.as_text())
      out[s] += sum(len(coll.findall(b)) for b in blocks if re.search(r'^}, execution_thread="sparsecore"', b, re.M))
  return dict(out)


def float_all_reduces(p, min_bytes=1 << 20):
  """[(program, MiB)] of `p`'s compiled programs' all-reduces of at least `min_bytes` of floats outside ds_ep's
  side-car scopes (an xla reduce-scatter may compile to one: ds_ep._RS_ALIGN).  None belongs in a split program:
  the weight gradients stay each device's partial until the reduce-scatter, the loss and sums of squares are
  scalars, and MLA's collectives over the size-1 'attn' axis compile away.  (Two of Lineage's backwards in one
  program -- an F&B pair of the last chunk's fused program with a backward, which the pipeline refuses -- compiled
  with ~257 MiB of them over 'fsdp' here.)"""
  size = {"bf16": 2, "f16": 2, "f32": 4}
  out = []
  for name, c in p.compiled:
    for line in c.as_text().splitlines():
      lhs, hit, _ = line.partition(" all-reduce(")
      if not hit:
        lhs, hit, _ = line.partition(" all-reduce-start(")
      if hit and not re.search(r'op_name="[^"]*(fsdp_reduce_scatter|fsdp_all_gather|grad_norm_all_reduce)', line):
        n = sum(
            size[t] * int(np.prod([int(d) for d in dims.split(",") if d] or [1]))
            for t, dims in re.findall(r"\b(bf16|f16|f32)\[([\d,]*)\]", lhs.partition("=")[2])
        )
        if n >= min_bytes:
          out.append((name, round(n / 2**20, 1)))
  return out


def errors(got, want):
  """(worst [(rel err, path)] first, whole-model cosine, norm ratio) of two chunk_leaves lists."""
  rows, dot, na, nb = [], 0.0, 0.0, 0.0
  for (path, a), (path_b, b) in zip(got, want, strict=True):
    assert path == path_b and a.shape == b.shape, (path, path_b, a.shape, b.shape)
    rows.append((float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)), path))
    dot, na, nb = dot + float(np.vdot(a, b)), na + float(np.vdot(a, a)), nb + float(np.vdot(b, b))
  rows.sort(reverse=True)
  return rows, dot / np.sqrt(na * nb), np.sqrt(na / nb)


@pytest.mark.tpu_only
@pytest.mark.scheduled_only
class LineagePipeline(unittest.TestCase):
  """The pipeline with --batch-split on Lineage's layers against the same pipeline unsplit (module docstring)."""

  @classmethod
  def setUpClass(cls):
    if NO_LINEAGE is not None:
      raise unittest.SkipTest(f"needs Lineage: {NO_LINEAGE}")
    testing.require_tpu(_on_tpu7x(), "a TPU7x with 8 devices (a tpu7x-8: the test's stage meshes are its halves)")
    testing.skip_on_pathways(SC_STAGE0)

  def build(self, extra, split):
    a = dsp.make_parser().parse_args(BASE + list(extra) + (["--batch-split"] if split else UNSPLIT))
    p = dsp.build(a)
    t0 = time.perf_counter()
    n, _ = p.precompile()
    lower = sum(x for x, _ in p.compile_times)
    print(
        f"  {'split' if split else 'unsplit'}: {n} programs precompiled in {time.perf_counter() - t0:.1f} s "
        f"(lowering {lower:.0f} s)",
        flush=True,
    )
    return p

  def trees(self, p, seed=0):
    """One init, the routed biases pinned: host chunk trees both pipelines load (params_from_chunks)."""
    params, _, _ = p.init(jax.random.key(seed))
    return pinned(p.chunk_trees(params), routing_pin(p.scfg.model))

  def run_loop(self, p, trees, seed=3):
    """One microbatch loop with the fusion plan -> (loss, chunk_leaves of the reduced gradients, stage sumsqs)."""
    params, _, wc0 = p.init_state(p.params_from_chunks(trees))
    grads, sqs, lacc = p.loss_and_grads_overlap(params, wc0, p.synthetic_batch(seed))
    return float(jnp.sum(lacc)), chunk_leaves(p, grads), [float(x) for x in sqs]

  def check(self, extra, label):
    """The split pipeline of `extra` against the unsplit one -> (the split pipeline, its loss, the errors)."""
    print(f"\n=== {label}", flush=True)
    sp = self.build(extra, split=True)
    self.assertTrue(sp.scfg.lineage_split)
    self.assertEqual((sp.scfg.random_routing, sp.scfg.shard_mode), (False, "explicit"))
    self.assertTrue(all(pr.lineage for pr in sp.progs))
    self.assertEqual(sp.lineage_sparsecore(), {0: False, 1: True})  # devices 0-3 hold chip (0,0,0)
    regions = sparsecore_regions(sp)
    print(f"  Lineage SparseCore regions in each stage's first program (lowered): {regions}", flush=True)
    self.assertEqual(regions[0], 0)
    self.assertGreater(regions[1], 0)
    self.assertEqual(float_all_reduces(sp), [])
    trees = self.trees(sp)
    loss, got, sq = self.run_loop(sp, trees)
    up = self.build(extra, split=False)
    self.assertFalse(up.scfg.batch_split)
    # Stage 0 runs no collective on SparseCore (libtpu offloads none there: Lineage's regions traced as TensorCore
    # code, the compiler aborts on them otherwise); on stage 1 Lineage's EP collectives join the side-cars there.
    sc, sc_up = sparsecore_collectives(sp), sparsecore_collectives(up)
    print(f"  SparseCore-region collectives in the compiled programs, by stage: split {sc}, unsplit {sc_up}", flush=True)
    self.assertEqual((sc[0], sc_up[0]), (0, 0))
    self.assertGreater(sc[1], sc_up[1])
    ref_loss, want, ref_sq = self.run_loop(up, trees)
    rl = abs(loss - ref_loss) / abs(ref_loss)
    rows, cos, ratio = errors(got, want)
    print(
        f"  loss split {loss:.7f} unsplit {ref_loss:.7f} rel {rl:.2e}; {len(rows)} gradient leaves, cosine "
        f"{cos:.7f}, norm ratio {ratio:.6f}, median leaf rel {np.median([e for e, _ in rows]):.2e}; worst:",
        flush=True,
    )
    for e, path in rows[:4]:
      print(f"      {e:.2e}  {path}", flush=True)
    print(f"  stage sumsq split {sq} unsplit {ref_sq}", flush=True)
    self.assertTrue(np.isfinite(loss) and all(np.isfinite(a).all() for _, a in got))
    self.assertLess(rl, TOL["loss"])
    self.assertGreater(cos, TOL["cosine"])
    self.assertLess(rows[0][0], TOL["leaf"], rows[:3])
    np.testing.assert_allclose(sq, ref_sq, rtol=2 * TOL["leaf"])
    return sp, loss, rows

  def test_v1(self):
    """V=1: chunk 0 (embedding + 2 MoE layers) on stage 0 without SparseCore, chunk 1 (2 MoE layers + the head,
    one fused program) on stage 1 with it."""
    self.check(["--vv", "1", "--layers", "4"], "V=1, 2 MoE layers a chunk")

  def test_v2_breadth(self):
    """V=2, circular placement, breadth-first (the default schedule), weights gathered per visit: chunks 0 and 2 on
    stage 0, 1 and 3 on stage 1; mixed programs folding other chunks' gathers and reduce-scatters."""
    sp, _, _ = self.check(["--vv", "2", "--layers", "8"], "V=2 breadth-first circular, 2 MoE layers a chunk")
    self.assertEqual((sp.schedule, sp.placement, sp.fsdp_scope), ("breadth", "circular", "repeat"))

  def test_dense(self):
    """Two dense layers in chunk 0, before its two MoE layers (the region runs them first, Lineage's dense layer
    each, on the whole microbatch)."""
    sp, _, _ = self.check(["--vv", "1", "--layers", "6", "--dense-layers", "2"], "V=1, 2 dense + 2 MoE in chunk 0")
    self.assertEqual([c.dense for c in sp.chunks], [2, 0])

  def test_remat_last(self):
    """--remat-last: the fused last chunk recomputes its layers in its backward (the region under jax.checkpoint)."""
    sp, _, _ = self.check(["--vv", "1", "--layers", "4", "--remat-last"], "V=1, --remat-last")
    self.assertTrue(sp.scfg.remat_fused_last)

  def test_training_steps(self):
    """Four AdamW steps on one batch from one init: both losses fall every step and stay within the loss bound of
    each other, and so do the gradient norms."""
    extra = ["--vv", "1", "--layers", "4"]
    lr, steps, out = 1e-3, 4, {}
    print("\n=== V=1, 4 AdamW steps", flush=True)
    for split in (True, False):
      p = self.build(extra, split)
      if split:
        trees = self.trees(p)
      params, opts, wc0 = p.init_state(p.params_from_chunks(trees))
      batch, lrs, losses, gnorms = p.synthetic_batch(3), p.lr_arrays(lr), [], []
      for _ in range(steps):
        params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
        losses.append(float(jnp.sum(lacc)))
        gnorms.append(float(gn))
      out[split] = (losses, gnorms)
      print(
          f"  {'split' if split else 'unsplit'}: losses {[round(x, 6) for x in losses]} gnorms "
          f"{[round(x, 5) for x in gnorms]}",
          flush=True,
      )
      del p, params, opts, wc0
    (ls, gs), (lu, gu) = out[True], out[False]
    for losses in (ls, lu):
      self.assertTrue(all(b < a for a, b in zip(losses, losses[1:])), losses)
    rel = [abs(a - b) / abs(b) for a, b in zip(ls, lu)]
    print(
        f'  loss rel per step {[f"{x:.2e}" for x in rel]}; gnorm rel '
        f'{[f"{abs(a - b) / abs(b):.2e}" for a, b in zip(gs, gu)]}',
        flush=True,
    )
    self.assertLess(max(rel), TOL["loss"])
    np.testing.assert_allclose(gs, gu, rtol=TOL["leaf"])


if __name__ == "__main__":
  unittest.main()
