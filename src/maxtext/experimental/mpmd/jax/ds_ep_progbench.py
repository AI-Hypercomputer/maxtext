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

"""Time ONE pipeline program in isolation under different compiler options.

    python3 -m maxtext.experimental.mpmd.jax.ds_ep_progbench --prog fwd_pf --chunk 0 --group 1 \
        --opts '{}' --opts '{"xla_tpu_enable_latency_hiding_layer_scheduler": false}'

    # v6e-4 (PP2 x FSDP1 x EP2, 16 experts): a stash backward
    python3 -m maxtext.experimental.mpmd.jax.ds_ep_progbench --preset v6e --prog bwda_s --stash-policy lite

Builds the full-size Pipeline (no step compile) -- by default the 128-expert
TPU7x benchmark with the `clamped` megablox tiles (`--tiles`, ds_ep.TILES),
`--preset v6e` the v6e-4 one with the `v6e` tiles --
materializes random inputs with the exact shardings the step uses, compiles
the program once per --opts, and prints the median and min of --reps runs
for each, and the program's compile-time temporaries.  At the default config
chunk 0's forwards carry chunk 2's gather in 4 leaf groups (--group 0..3); a
chunk that carries none exits with the list of chunks that do.  Unlike
ds_ep_pipeline's default the default preset uses MaxText's placement
(`ep_on_chip`).  FSDP_COLL / EP_COLL (environment) pick the FSDP and EP
collectives (default ppermute / auto).  Used to tune the scheduling of the
programs that fold FSDP collectives in, and to measure what the schedules
assume: the stash backward's cost (`Bs`), and a split backward's I and W
(`--prog bwdi` / `bwdwa`: the zero-bubble schedules' I / W costs).

Stash programs (`--stash-policy`, ds_ep.STASH_POLICIES) take the stash their
own forward produces from the random inputs, so the backward sees real
routing (group sizes), not zeros.  The inputs are random activations, and for
chunk 0 random token ids: under top-k routing identical tokens would make
every token's hidden state the same after the first layer and send them all
to the same experts.
"""

import argparse
import json
import os
import time
import warnings

from maxtext.experimental.mpmd.jax import transfer

transfer.set_max_inflight(1024)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from maxtext.experimental.mpmd.jax import ds_ep  # noqa: E402
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp  # noqa: E402

# The full-size config, a TPU7x benchmark shape, and what each preset changes.
BASE = dict(
    stages=2,
    vv=2,
    mb=8,
    gbs=32,
    seq=4096,
    fsdp=2,
    ep=2,
    layers=4,
    emb=7168,
    heads=128,
    q_lora=1536,
    kv_lora=512,
    moe_mlp=2048,
    experts=128,
    topk=8,
    weight_dtype="bfloat16",
    vocab=32768,
    attn="tokamax_splash",
    sa_block=2048,
    sa_block_kv_compute=512,
    cost_estimate=5_000_000_000_000,
    tiles="clamped",
    wi_tiling=None,
    wo_tiling=None,
    rs_dtype="bfloat16",
    acc_dtype="float32",
    remat_last=False,
    schedule=None,
    layout="ep_on_chip",
    no_overlap=False,
    fsdp_collectives=os.environ.get("FSDP_COLL", "ppermute"),
    ep_collectives=os.environ.get("EP_COLL", "auto"),
    pf_split=4,
    rs_split=4,
    costs="",
    grad_reduce="step",
    routing="random",
    ragged_sort="auto",
)
PRESETS = {
    "default": {},
    # v6e-4: 2 stages x (fsdp 1 x ep 2), 16 experts, gbs 16 in 8 microbatches, and the
    # megablox tiles swept on v6e (ds_ep.TILES['v6e']), as the v6e baseline runs
    "v6e": dict(stages=2, fsdp=1, ep=2, experts=16, gbs=16, mb=8, layout="fsdp_on_chip", tiles="v6e"),
    # TPU7x-8: PP=4 x FSDP=2 x EP=1 over each chip's cores, the shape of a PP=4 comparison with MaxText
    # (32 experts, 8 layers, fp32 weights, top-k routing, per-microbatch reduce-scatter)
    "pp4": dict(
        stages=4,
        fsdp=2,
        ep=1,
        layers=8,
        experts=32,
        weight_dtype="float32",
        layout="snake",
        routing="topk",
        ragged_sort="off",
        grad_reduce="microbatch",
    ),
}
# (flag, type): each overrides the preset
OVERRIDES = [
    ("stages", int),
    ("vv", int),
    ("mb", int),
    ("gbs", int),
    ("seq", int),
    ("fsdp", int),
    ("ep", int),
    ("layers", int),
    ("emb", int),
    ("heads", int),
    ("q_lora", int),
    ("kv_lora", int),
    ("moe_mlp", int),
    ("experts", int),
    ("topk", int),
    ("vocab", int),
    ("sa_block", int),
    ("sa_block_kv_compute", int),
    ("layout", str),
    ("schedule", str),
    ("tiles", str),
]


def materialize(avals, key, vocab=None):
  """Random arrays with `avals`' shapes, dtypes and shardings: floats
  ~N(0, 0.02^2); integer arrays (token ids) uniform over [0, vocab), and zero
  without a `vocab`; integer scalars (seeds, counters) zero."""
  leaves, tdef = jax.tree.flatten(avals)
  out = []
  for i, a in enumerate(leaves):
    k = jax.random.fold_in(key, i)
    if jnp.issubdtype(a.dtype, jnp.integer) and vocab and a.ndim:
      f = lambda k, a=a: jax.random.randint(k, a.shape, 0, vocab, a.dtype)
    elif jnp.issubdtype(a.dtype, jnp.integer):
      f = lambda k, a=a: jnp.zeros(a.shape, a.dtype)
    else:
      f = lambda k, a=a: (0.02 * jax.random.normal(k, a.shape, jnp.float32)).astype(a.dtype)
    out.append(jax.jit(f, out_shardings=a.sharding)(k))
  return jax.tree.unflatten(tdef, out)


def bench(label, fn, args_, donated, opts, reps):
  """Compile `fn` for `args_` under compiler options `opts` (a JSON string),
  run it `reps` times and print one line; returns the median in ms."""
  o = json.loads(opts)
  t0 = time.perf_counter()
  comp = fn.lower(*args_).compile(compiler_options=o or None)
  tc = time.perf_counter() - t0
  mem = comp.memory_analysis()
  tmp = f"temp {mem.temp_size_in_bytes / 2**30:6.2f} GiB" if mem is not None else ""
  ts = []
  for _ in range(reps):
    x = [jax.tree.map(lambda v: v.copy(), args_)] if donated else [args_]
    jax.block_until_ready(x)
    t1 = time.perf_counter()
    out = comp(*x[0])
    jax.block_until_ready(out)
    ts.append(time.perf_counter() - t1)
    del out
  ts.sort()
  med = ts[len(ts) // 2] * 1e3
  print(f"{label} {opts:80s} median {med:7.2f} ms  min {ts[0] * 1e3:7.2f}  {tmp}  (compile {tc:.0f}s)", flush=True)
  return med


def forward_host(p, j):
  """Is order[j] a forward that can host a prefetch group -- not the fused
  last chunk's (A), which under --fsdp-scope repeat hosts re-gathers too?"""
  c, k, _ = p.order[j]
  return k == "F" and c != p.C - 1


def host_op(p, c, group, prog):
  """Chunk c's `group`-th host op for --prog fwd_pf (a forward's prefetch
  group) or bwd_rs (a backward's reduce-scatter group, with whatever else
  it carries: under --fsdp-scope repeat a gather, B + R + G) -> (op, its
  program, the program's abstract arguments); SystemExit naming the chunks
  that host one."""
  if prog == "fwd_pf":
    js = sorted(j for j in p.pf if forward_host(p, j) and p.order[j][0] == c)
    what, hosts = "prefetch groups in its forwards", sorted({p.order[j][0] for j in p.pf if forward_host(p, j)})
  else:
    js = sorted(j for j in p.rs_at if p.order[j][0] == c)
    what, hosts = "reduce-scatter groups", sorted({p.order[j][0] for j in p.rs_at})
  if group >= len(js):
    raise SystemExit(f"chunk {c} hosts {len(js)} {what}; hosts are {hosts}")
  op = p.ops()[js[group]]
  return op, p.program(op), p._desc_avals(op.c, op.desc)


def main():
  warnings.filterwarnings("ignore")
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--prog",
      default="fwd_pf",
      choices=[
          "fwd",
          "fwd_pf",
          "bwda",
          "bwd_rs",
          "update_ag",
          "fwd_s",
          "bwd_s",
          "bwda_s",
          "fwd_pf_s",
          "bwd_rs_s",
          "bwdi",
          "bwdwa",
      ],
      help="the program to time; bwdi / bwdwa are a middle chunk's split backward (the zero-bubble schedules' I and "
      "accumulating W: ds_ep.ChunkPrograms.build_split), W on the crossing set I makes of the random inputs",
  )
  ap.add_argument("--chunk", type=int, default=None, help="default: the first chunk hosting --prog")
  ap.add_argument("--group", type=int, default=0, help="which prefetch/reduce leaf group")
  ap.add_argument("--opts", action="append", default=[], help="JSON compiler options; repeatable")
  ap.add_argument("--reps", type=int, default=10)
  ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
  ap.add_argument(
      "--stash-policy",
      default="lite",
      choices=[k for k in ds_ep.STASH_POLICIES if k != "none"],
      help="for the *_s programs",
  )
  ap.add_argument("--wi-tiling", type=int, nargs=9, default=None, help="wi_0 / wi_1 tiling (overrides --tiles)")
  ap.add_argument("--wo-tiling", type=int, nargs=9, default=None, help="wo tiling (overrides --tiles)")
  for name, typ in OVERRIDES:
    ap.add_argument("--" + name.replace("_", "-"), type=typ, default=None, help="overrides --preset")
  a = ap.parse_args()
  cfg = dict(BASE, **PRESETS[a.preset])
  cfg.update({k: getattr(a, k) for k, _ in OVERRIDES if getattr(a, k) is not None})
  if cfg["tiles"] not in ds_ep.TILES:
    raise SystemExit(f"--tiles {cfg['tiles']!r}: want one of {sorted(ds_ep.TILES)}")
  for k in ("wi_tiling", "wo_tiling"):
    if getattr(a, k):
      cfg[k] = getattr(a, k)
  args = argparse.Namespace(**cfg)
  p = dsp.build(args)
  sc = p.progs[0].scfg
  print(
      f"preset {a.preset}: "
      + " ".join(f"{k}={cfg[k]}" for k, _ in OVERRIDES)
      + f" wi_tiling={sc.wi_tiling} wo_tiling={sc.wo_tiling}",
      flush=True,
  )
  A = p._avals

  def mat(avals, key_):
    """`materialize` with this model's vocabulary for the token ids."""
    return materialize(avals, key_, vocab=p.scfg.model.vocab_size)

  pd = ds_ep.pdesc
  hosts = {
      "fwd_pf": sorted({p.order[j][0] for j in p.pf if forward_host(p, j)}),
      "bwd_rs": sorted({p.order[j][0] for j in p.rs_at}),
  }
  hosts["fwd_pf_s"], hosts["bwd_rs_s"] = hosts["fwd_pf"], hosts["bwd_rs"]
  if a.prog in hosts and not hosts[a.prog]:
    what = "forward carries a gather" if a.prog.startswith("fwd") else "backward carries another chunk's reduce-scatter"
    why = ""
    if a.prog.startswith("bwd") and p.scfg.grad_reduce == "microbatch":
      why = ": this preset reduce-scatters every microbatch's gradient in its own backward " "(time --prog bwda)"
    raise SystemExit(f"--prog {a.prog}: no {what} in this configuration{why}")
  c = a.chunk
  if c is None:
    c = hosts[a.prog][0] if a.prog in hosts else 2
  ch = p.chunks[c]
  s = ch.stage
  pr = p.progs[c]
  key = jax.random.key(0)
  pol = a.stash_policy
  stash_progs = ("fwd_s", "bwd_s", "bwda_s", "fwd_pf_s", "bwd_rs_s")
  opts = a.opts or ["{}"]

  def x_of(ch_):
    return A(ch_.stage, "tok" if ch_.first else "act")

  def stash_of(pr_):
    st = pr_.build_stash(pol)
    print(
        f"stash {pol} c{pr_.ch.index}: {len(st.avals)} arrays, {st.bytes / 2**30:.3f} GiB/device/microbatch "
        f'({", ".join(f"{b / 2**30:.3f}" for b in st.layer_bytes)} per layer)',
        flush=True,
    )
    return st

  def forward(pr_, stash):
    """(w, x, seed) of chunk pr_, random, and what its forward makes of them:
    res (the layer inputs), and the stash in stash mode."""
    ch_ = pr_.ch
    w, x, seed = mat((A(ch_.stage, "compute", ch_), x_of(ch_), A(ch_.stage, "scalar_i")), key)
    if stash:
      _, res, stash_ = stash_of(pr_).fwd(w, x, seed)
      return w, x, seed, res, stash_
    _, res = pr_.fwd(w, x, seed)
    return w, x, seed, res, None

  if a.prog in stash_progs and ch.last:
    raise SystemExit(f"chunk {c} is the last: its forward and backward are one fused program (A)")
  if a.prog in ("bwdi", "bwdwa"):
    if ch.first or ch.last:
      raise SystemExit(f"--prog {a.prog}: chunk {c} has no split backward (a middle chunk's: 1 to {p.C - 2})")
    w, x, seed, res, _ = forward(pr, False)
    dy, gacc = mat((A(s, "act"), A(s, "gacc", ch)), key)
    i_args = [w, x, res, dy, seed]
    i_prog = pr.program((pd("I", acc=pr.build_split(True).i_acc),))  # the I the accumulating W reads
    if a.prog == "bwdi":
      fn, args_ = i_prog, i_args
    else:  # W's inputs: the crossing set the I makes of these, and the arguments of I it reads (no copy)
      _, cross = i_prog(*jax.tree.map(lambda v: v.copy(), i_args))  # (I donates dy where W does not read it)
      fn, args_ = pr.program((pd("W", acc=True),)), [cross, pr.build_split(True).kept_of(i_args), gacc]
    for o in opts:
      bench(f"{a.prog} c{c}", fn, args_, True, o, a.reps)
    return
  if a.prog == "fwd":
    fn, av = pr.fwd, (A(s, "compute", ch), A(s, "tok" if ch.first else "act"), A(s, "scalar_i"))
  elif a.prog in ("fwd_pf", "bwd_rs"):
    op, fn, av = host_op(p, c, a.group, a.prog)
  elif a.prog == "bwda":
    fn = pr.bwda
    av = (
        A(s, "compute", ch),
        A(s, "tok" if ch.first else "act"),
        tuple(A(s, "act") for _ in range(pr.n_res)),
        A(s, "act"),
        A(s, "gacc", ch),
        A(s, "scalar_i"),
    )
  elif a.prog == "update_ag":
    st = p.stages[s]
    chs = p.stage_chunks[s]
    params = [A(s, "storage", cc) for cc in chs]
    opt = {
        "mu": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.bfloat16, sharding=x.sharding), params),
        "nu": params,
        "count": A(s, "scalar_i"),
    }
    fn = st.update_gather
    av = (params, opt, [A(s, "grads", cc) for cc in chs], [A(s, "scalar_f")] * p.S, A(s, "scalar_f"))
  if a.prog in ("fwd", "fwd_pf", "bwda", "bwd_rs", "update_ag"):
    args_ = mat(av, key)
    donated = a.prog in ("bwda", "bwd_rs", "update_ag")
    label = f"{a.prog} c{c} g{a.group}"
    if a.prog in ("fwd_pf", "bwd_rs"):
      label += f' ({p.op_name(op.j)}: {"+".join(k for k, _ in op.desc)})'
    for o in opts:
      bench(label, fn, args_, donated, o, a.reps)
    return
  # stash programs: inputs from the chunk's own stash forward
  w, x, seed, res, stash = forward(pr, True)
  dy, gacc = mat((A(s, "act"), A(s, "gacc", ch)), key)
  if a.prog == "fwd_s":
    fn, args_ = pr.build_stash(pol).fwd, [w, x, seed]
  elif a.prog in ("bwd_s", "bwda_s"):
    fn = pr.program((pd("B", stash=pol, acc=a.prog == "bwda_s"),))
    args_ = [w, x, res, stash, dy] + ([gacc] if a.prog == "bwda_s" else []) + [seed]
  elif a.prog == "fwd_pf_s":
    hop, _, _ = host_op(p, c, a.group, "fwd_pf")
    b, g = hop.pf
    fn = pr.program((pd("F", stash=pol), pd("G", other=p.chunks[b], leaves=g if p.pf_n[hop.j] > 1 else None)))
    st = jax.tree.leaves(mat(A(s, "storage", p.chunks[b]), key))
    args_ = [w, x, seed, [st[i] for i in g]]
  else:  # bwd_rs_s
    js = sorted(j for j in p.rs_at if p.order[j][0] == c)
    if a.group >= len(js):
      raise SystemExit(
          f"chunk {c} hosts {len(js)} reduce-scatter groups; hosts are " f"{sorted({p.order[j][0] for j in p.rs_at})}"
      )
    j = js[a.group]
    o_, g = p.rs_at[j]
    acc = j != min(jj for jj, (cc, k, m) in enumerate(p.order) if cc == c and k == "B")
    fn = pr.program((pd("B", stash=pol, acc=acc), pd("R", other=p.chunks[o_], leaves=g if p.rs_groups[o_] > 1 else None)))
    gl = jax.tree.leaves(mat(A(s, "gacc", p.chunks[o_]), key))
    args_ = [w, x, res, stash, dy] + ([gacc] if acc else []) + [seed, [gl[i] for i in g]]
  for o in opts:
    bench(f"{a.prog} {pol} c{c} g{a.group}", fn, args_, a.prog not in ("fwd_s", "fwd_pf_s"), o, a.reps)


if __name__ == "__main__":
  main()
