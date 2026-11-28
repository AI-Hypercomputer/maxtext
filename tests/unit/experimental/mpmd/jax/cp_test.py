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

"""CPU check of ds_ep's context parallelism (StageConfig.cp): a sequence cut into cp chunks, one row a device.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/cp_test.py -q

MLA over the chunks (each device's queries against the kv latents its 'expert' ring neighbours hold) must be MLA over
the whole sequence, output and gradients; and a pipeline of cp=2 / cp=4 microbatches must train like one of whole
sequences on the same tokens (cp=1 at fewer, longer microbatches), in fp32.
"""

import dataclasses
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from packaging.version import Version
import pytest

from maxtext.experimental.mpmd.jax.deepseek import attention
from maxtext.experimental.mpmd.jax.deepseek import rope
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import testing

try:
  import tokamax

  HAVE_TOKAMAX = True
except Exception:  # pragma: no cover
  HAVE_TOKAMAX = False

SEQ, D, H = 256, 128, 2


def setUpModule():
  testing.require_cpu()


def attn_cfg(kernel, seq=SEQ):
  # splash blocks are multiples of 128 (its kv lanes): a cp=4 chunk of 128 needs seq 512
  return DeepSeekConfig(
      vocab_size=128,
      emb_dim=D,
      num_decoder_layers=1,
      first_num_dense_layers=0,
      num_query_heads=H,
      num_kv_heads=H,
      q_lora_rank=64,
      kv_lora_rank=64,
      mlp_dim=64,
      moe_mlp_dim=64,
      num_experts=4,
      num_experts_per_tok=2,
      shared_experts=1,
      routed_bias=False,
      attention_kernel=kernel,
      sa_block_q=128,
      sa_block_kv=128,
      sa_block_kv_compute=128,
      max_target_length=seq,
      dtype=jnp.float32,
  )


def mla_params(cfg, seed=0):
  shapes = ds_ep.layer_shapes(cfg)["self_attention"]
  keys = iter(jax.random.split(jax.random.key(seed), 16))
  out = {}
  for name, leaf in shapes.items():
    k, shape = next(iter(leaf.items()))
    if k == "scale":
      out[name] = {k: 1.0 + 0.1 * jax.random.normal(next(keys), shape, jnp.float32)}
    else:
      out[name] = {k: jax.random.normal(next(keys), shape, jnp.float32) * (shape[0] ** -0.5)}
  return out


def _rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


@testing.needs_devices(4)
class CPAttention(unittest.TestCase):
  """attention.mla with ds_ep.cp_kv_context on cp devices of an 'expert' axis (each holding one chunk of every
  sequence of its ring) against attention.mla over the whole sequences: output, input and every weight gradient."""

  def _check(self, cp, ep, kernel, SEQ=SEQ, segments="ones"):
    """`segments`: the sequences' segment ids -- 'ones' (one document each, as layer_fwd's synthetic data), None
    (no ids: the chunk mask is the keys' ids alone) or 'packed' (documents of 96 tokens, numbered on from 1 and
    starting at another offset in each sequence, so some cross a chunk boundary)."""
    cfg = attn_cfg(kernel, SEQ)
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
    B = ep // cp  # sequences: one row a device
    w = mla_params(cfg)
    x = jax.random.normal(jax.random.key(1), (B, SEQ, D), jnp.float32)
    probe = jax.random.normal(jax.random.key(2), (B, SEQ, D), jnp.float32)
    freqs = rope.freqs_cis_for(cfg, SEQ)
    seg = {
        "ones": jnp.ones((B, SEQ), jnp.int32),
        None: None,
        "packed": 1 + (jnp.arange(SEQ)[None] + 40 * jnp.arange(B)[:, None]) // 96,
    }[segments]

    def whole(w, x, seg):
      pos = jnp.broadcast_to(jnp.arange(SEQ, dtype=jnp.int32), (B, SEQ))
      return attention.mla(w, x, pos, seg, freqs, cfg)

    mesh = Mesh(np.array(jax.devices()[:ep]).reshape(1, ep), ds_ep.FE)
    s = SEQ // cp

    def local(w, x, seg):  # x: this device's row, [1, s, D]; seg its ids, [1, s]
      pos = ds_ep.cp_chunk(scfg) * s + jnp.arange(s, dtype=jnp.int32)[None]
      return attention.mla(w, x, pos, seg, freqs, cfg, kv_context=ds_ep.cp_kv_context(scfg))

    wspec = jax.tree.map(lambda _: P(), w)
    sspec = None if seg is None else P(ds_ep.FE)
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=(wspec, P(ds_ep.FE), sspec), out_specs=P(ds_ep.FE), check_vma=False
    )

    def chunked(w, x, seg):  # [B, SEQ, D] -> rows [B * cp, s, D] (chunk-major per sequence)
      put = lambda a: jax.device_put(a.reshape(B * cp, s, *a.shape[2:]), NamedSharding(mesh, P(ds_ep.FE)))
      return sharded(w, put(x), None if seg is None else put(seg)).reshape(B, SEQ, D)

    want = whole(w, x, seg)
    got = jax.jit(chunked)(w, x, seg)
    loss = lambda f: lambda w, x: jnp.sum(f(w, x, seg) * probe)
    gw_want, gx_want = jax.grad(loss(whole), argnums=(0, 1))(w, x)
    gw_got, gx_got = jax.jit(jax.grad(loss(chunked), argnums=(0, 1)))(w, x)
    errs = {"out": _rel(got, want), "dx": _rel(gx_got, gx_want)}
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(gw_got)[0], jax.tree.leaves(gw_want)):
      errs["d" + jax.tree_util.keystr(path)] = _rel(a, b)
    worst = max(errs, key=errs.get)
    print(f"cp={cp} ep={ep} {kernel} segments={segments}: worst rel err {errs[worst]:.2e} ({worst})")
    for k, e in errs.items():
      self.assertLess(e, 2e-5, f"cp={cp} {kernel} segments={segments} {k}: {e:.3g}")
    return errs

  @testing.needs_devices(8, "cp 2 and 4 over expert axes of up to 8")
  def test_reference_kernel(self):
    for cp, ep in ((2, 2), (2, 4), (4, 4), (4, 8)):
      with self.subTest(cp=cp, ep=ep):
        self._check(cp, ep, "reference")

  def test_without_segment_ids(self):
    """mla's segment_ids may be None: the later chunks are still masked (the keys' ids alone), on every kernel."""
    for kernel, seq in (("reference", SEQ), ("splash", 512)):
      for cp, ep in ((2, 2), (4, 4)):
        with self.subTest(kernel=kernel, cp=cp, ep=ep):
          self._check(cp, ep, kernel, SEQ=seq, segments=None)

  def test_packed_segment_ids(self):
    """Packed sequences: each slot's keys carry their chunk's own document ids, the later chunks' taken out."""
    for kernel, seq in (("reference", SEQ), ("splash", 512)):
      for cp, ep in ((2, 2), (2, 4), (4, 4)):
        with self.subTest(kernel=kernel, cp=cp, ep=ep):
          self._check(cp, ep, kernel, SEQ=seq, segments="packed")

  def test_splash_kernel(self):
    """The jax splash kernel (--attn splash, the runs'), interpret mode off TPU."""
    for cp, ep in ((2, 2), (4, 4)):
      with self.subTest(cp=cp, ep=ep):
        self._check(cp, ep, "splash", SEQ=512)

  @unittest.skipUnless(HAVE_TOKAMAX, "needs tokamax")
  def test_tokamax_splash_kernel(self):
    # tokamax 0.0.16 began pruning splash tiles by segment ids (_refine_mask_info_with_segments), and in interpret
    # mode on jax < 0.11.2 the backward pass then goes wrong on both sides of this check, the whole sequences and the
    # context-parallel chunks: with the pruning patched out, jax 0.11.1 passes.  On a TPU v6e the compiled kernel's
    # gradients were right on jax 0.11.1 (whole sequences of 512 with segment ids, tokamax 0.0.17).
    if (
        jax.default_backend() != "tpu"
        and jax.__version_info__[:3] < (0, 11, 2)
        and Version(tokamax.__version__) >= Version("0.0.16")
    ):
      self.skipTest(
          "on jax < 0.11.2, tokamax >= 0.0.16's splash in interpret mode can give wrong gradients with "
          "segment ids, context parallel or not: dx off by 0.40 here on jax 0.11.0 and 0.11.1 with tokamax "
          "0.0.16 to 0.0.18.dev, where tokamax 0.0.12 to 0.0.15 and jax 0.11.2 pass"
      )
    for cp, ep in ((2, 2), (4, 4)):
      with self.subTest(cp=cp, ep=ep):
        self._check(cp, ep, "tokamax_splash", SEQ=512)

  def test_offset_and_kv_segments_reference(self):
    """attention_op's offset / kv_segment_ids on the reference kernel against an explicit mask."""
    cfg = attn_cfg("reference")
    b, t, s, n, d = 1, 64, 128, 2, 32
    ks = jax.random.split(jax.random.key(3), 3)
    q, k, v = (jax.random.normal(kk, (b, l, n, d)) for kk, l in zip(ks, (t, s, s)))
    qseg = jnp.ones((b, t), jnp.int32)
    kseg = jnp.concatenate([jnp.zeros((b, 32), jnp.int32), jnp.ones((b, s - 32), jnp.int32)], axis=1)
    out = attention.attention_op(q, k, v, qseg, cfg, kseg, s - t)
    keep = (jnp.arange(s)[None, :] <= jnp.arange(t)[:, None] + (s - t)) & (kseg[0] == 1)[None, :]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k)
    p = jax.nn.softmax(jnp.where(keep[None, None], logits, -1e30), axis=-1)
    self.assertLess(_rel(out, jnp.einsum("bhqk,bkhd->bqhd", p, v)), 1e-5)


def _pipeline_grads(p, params, wc0, toks, tgts, overlap):
  """(loss, grads per chunk in chunk order) of one pass of `p` on host tokens."""
  batch = p.put_batch(toks, tgts)
  if overlap:
    grads, _, lacc = p.loss_and_grads_overlap(params, wc0, batch)
  else:
    gaccs, lacc = p.loss_and_grads(params, batch)
    grads = [p.stages[s].reduce(gaccs[s])[0] for s in range(p.S)]
  return float(jnp.sum(lacc)), [grads[ch.stage][p.slot[ch.index]] for ch in p.chunks]


@testing.needs_devices(
    8, "context-parallel rows, one a stage device: written for two stages of fsdp 2 x ep 2 (cp 4: ep 4)"
)
class CPPipeline(unittest.TestCase):
  """A pipeline of cp-chunked microbatches against one of whole sequences on the same tokens and weights (top-k
  routing: random routing draws over the EP group's tokens, which context parallelism lays out differently): the
  loss and every gradient leaf, fp32, ragged_dot experts and the argsort dispatch (no MaxText kernel)."""

  def _check(self, cp, fsdp, ep, gbs, kernel, overlap, **kw):
    from tests.unit.experimental.mpmd.jax.ds_ep_test import tiny_xla

    seq = 256
    scfg = tiny_xla(
        layers=4, fsdp=fsdp, ep=ep, random_routing=False, fsdp_collectives="xla" if max(fsdp, ep) > 2 else "ppermute"
    )
    scfg = dataclasses.replace(scfg, model=scfg.model.replace(attention_kernel=kernel))
    per_stage = fsdp * ep
    # init_by_chunk: chunk c's weights whatever the placement, so the reference may take the default schedule
    # (a named schedule may want more microbatches than the whole-sequence pipeline, at 1/cp of them, has)
    mk = lambda c, m, **k: dsp.Pipeline(
        scfg,
        n_stages=2,
        chunks_per_stage=2,
        n_micro=m,
        global_batch=gbs,
        seq=seq,
        overlap=overlap,
        cp=c,
        init_by_chunk=True,
        **k,
    )
    p = mk(cp, gbs * cp // per_stage, **kw)  # one row a stage device
    q = mk(1, gbs // per_stage)  # whole sequences: fewer microbatches
    self.assertEqual((p.rows, p.row_len, q.rows, q.row_len), (per_stage, seq // cp, per_stage, seq))
    params, _, wc0 = p.init(jax.random.key(1))
    params_q, _, wc0_q = q.init(jax.random.key(1))
    for a, b in zip(jax.tree.leaves(p.chunk_trees(params)), jax.tree.leaves(q.chunk_trees(params_q))):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    rng = np.random.default_rng(0)
    toks = rng.integers(0, scfg.model.vocab_size, size=(gbs, seq), dtype=np.int32)
    tgts = np.roll(toks, -1, axis=1)
    loss, g = _pipeline_grads(p, params, wc0, toks, tgts, overlap)
    loss_q, g_q = _pipeline_grads(q, params_q, wc0_q, toks, tgts, overlap)
    worst, at = 0.0, ""
    for c, (a, b) in enumerate(zip(g, g_q)):
      for (path, x), y in zip(jax.tree_util.tree_flatten_with_path(a)[0], jax.tree.leaves(b), strict=True):
        e = _rel(x, y)
        if e > worst:
          worst, at = e, f"chunk {c} {jax.tree_util.keystr(path)}"
    print(
        f"cp={cp} (M={p.M}) vs whole sequences (M={q.M}), {kernel}, overlap={overlap}: loss {loss:.7f} vs "
        f"{loss_q:.7f}, worst grad rel err {worst:.2e} at {at}"
    )
    self.assertLess(abs(loss - loss_q) / abs(loss_q), 1e-5)
    self.assertLess(worst, 1e-4, at)
    return p

  @pytest.mark.scheduled_only
  def test_cp2(self):
    kernel = "tokamax_splash" if HAVE_TOKAMAX else "reference"
    for overlap in (False, True):
      with self.subTest(overlap=overlap):
        p = self._check(2, fsdp=2, ep=2, gbs=8, kernel=kernel, overlap=overlap)
        self.assertEqual(p.M, 4)

  @pytest.mark.scheduled_only
  def test_cp4(self):
    for overlap in (False, True):
      with self.subTest(overlap=overlap):
        p = self._check(4, fsdp=1, ep=4, gbs=4, kernel="reference", overlap=overlap)
        self.assertEqual(p.M, 4)

  @pytest.mark.scheduled_only
  def test_cp2_pairs_and_stash(self):
    """cp through interleaved 1F1B's F&B pairs, and through the V placement with the stash schedule's programs."""
    kernel = "tokamax_splash" if HAVE_TOKAMAX else "reference"
    self._check(2, fsdp=2, ep=2, gbs=8, kernel=kernel, overlap=True, schedule="interleaved", pair_stages="all")
    self._check(
        2,
        fsdp=2,
        ep=2,
        gbs=8,
        kernel=kernel,
        overlap=True,
        placement="v",
        schedule="stash",
        stash="lite",
        stash_budget_gib=1.0,
    )

  @pytest.mark.scheduled_only
  def test_cp2_pipeline_matches_monolithic_cp2(self):
    """compare_with_reference -- one monolithic shard_map running the same cp=2 layers -- holds at cp=2: the
    pipeline's order, boundary copies and overlap plan with context-parallel rows.  A consistency check, not one
    of CP: a bug in cp_kv_context, cp_chunk or the positions is on both sides and cancels.  CPAttention and
    test_cp2 / test_cp4, against whole sequences, check CP itself."""
    from tests.unit.experimental.mpmd.jax.ds_ep_test import compare_with_reference, tiny_xla

    p = dsp.Pipeline(
        tiny_xla(layers=4), n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, overlap=True, cp=2
    )
    compare_with_reference(self, p, True, loss_tol=1e-5, grad_tol=1e-4, sq_tol=1e-4)

  @pytest.mark.scheduled_only
  def test_cp2_explicit_vs_auto(self):
    """shard_mode explicit against auto at cp=2 under V placement (M=4
    microbatches of one row a device), lax FSDP collectives,
    the gather embedding and top-k routing: every gradient leaf, the stage
    sumsqs and the loss, and the step after them, bit for bit -- the embedding
    table's too: either mode's backward scatter-adds the tokens' rows into the
    accumulator itself (ds_ep.embed_grad), a row hit by several tokens one
    token at a time.  (Autodiff's zero table with the rows scattered in, added
    to the accumulator, had XLA fold the add into the scatter under explicit
    alone, whose accumulator has no leading partial axis: a few of the table's
    elements a float32 ulp apart.)"""
    from tests.unit.experimental.mpmd.jax.ds_ep_test import assert_same_run, explicit, mode_run, tiny_xla

    sc = tiny_xla(layers=4, random_routing=False, fsdp_collectives="xla", iota_embed=False, remat_fused_last=True)
    runs = [
        mode_run(
            dsp.Pipeline(
                scfg,
                n_stages=2,
                chunks_per_stage=2,
                n_micro=4,
                global_batch=8,
                seq=256,
                cp=2,
                placement="v",
                init_by_chunk=True,
            ),
            2,
            4,
            steps=1,
        )
        for scfg in (sc, explicit(sc))
    ]
    assert_same_run(self, runs[1], runs[0], "cp2 V")
    print(
        f'cp=2 V: explicit vs auto, all {len(runs[0]["grads"])} gradient leaves bitwise; loss ' f'{runs[0]["loss"]:.7f}'
    )

  def test_build_keeps_a_cp_override(self):
    """build puts --cp in its StageConfig, so a cp passed to build as a StageConfig override stays (Pipeline's
    own cp keyword overrides scfg.cp only when given); and that keyword re-validates the config without
    repeating the padded-tiling warning the config gave (review F29, F36)."""
    from tests.unit.experimental.mpmd.jax.ds_ep_test import TINY_ARGS, tiny_xla

    ap = dsp.make_parser()
    a = ap.parse_args(TINY_ARGS + ["--gbs", "2"])  # a sequence a microbatch: cp=2 is a row a device
    self.assertEqual(a.cp, 1)
    self.assertEqual(dsp.build(a, cp=2).cp, 2)
    self.assertEqual(dsp.build(ap.parse_args(TINY_ARGS + ["--gbs", "2", "--cp", "2"])).scfg.cp, 2)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      padded = dataclasses.replace(tiny_xla(layers=4), gmm_impl="megablox", wi_tiling=(512, 1024, 512) * 3)
      self.assertEqual(sum(ds_ep.TILING_WARNING in str(x.message) for x in w), 1)  # tk 1024 > emb 256
      p = dsp.Pipeline(padded, n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, cp=2, schedule="1f1b")
    self.assertEqual(p.scfg.cp, 2)
    self.assertEqual(sum(ds_ep.TILING_WARNING in str(x.message) for x in w), 1)

  def test_rows_and_checks(self):
    from tests.unit.experimental.mpmd.jax.ds_ep_test import tiny_xla

    scfg = tiny_xla(layers=4)
    p = dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=4, global_batch=8, seq=256, cp=2)
    self.assertEqual((p.cp, p.scfg.cp, p.mbs, p.rows, p.row_len), (2, 2, 2, 4, 128))
    a = np.arange(2 * 256).reshape(2, 256)
    np.testing.assert_array_equal(p.cp_rows(a), a.reshape(4, 128))
    self.assertIn("cp=2: 4 rows of 128", p.summary())
    with self.assertRaisesRegex(ValueError, "exactly one a stage device"):
      dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=2, global_batch=8, seq=256, cp=2)
    with self.assertRaisesRegex(ValueError, "dividing ep=2"):
      dsp.Pipeline(scfg, n_stages=2, chunks_per_stage=2, n_micro=8, global_batch=8, seq=256, cp=4)


class CPConfig(unittest.TestCase):

  def test_checks(self):
    cfg = attn_cfg("reference")
    base = dict(
        model=cfg,
        fsdp=1,
        ep=4,
        wi_tiling=(128,) * 9,
        wo_tiling=(128,) * 9,
        gmm_impl="ragged_dot",
        fsdp_collectives="xla",
        use_ragged_sort=False,
    )
    for cp in (1, 2, 4):
      self.assertEqual(ds_ep.StageConfig(**base, cp=cp).cp, cp)
    for cp in (3, 8, 0):
      with self.subTest(cp=cp), self.assertRaisesRegex(ValueError, "cp"):
        ds_ep.StageConfig(**base, cp=cp)
    with self.assertRaisesRegex(ValueError, "dividing ep=2"):
      ds_ep.StageConfig(**dict(base, ep=2), cp=4)


if __name__ == "__main__":
  unittest.main()
