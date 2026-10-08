"""Time and MFU breakdown of Gated Delta Net, stage by stage, against the other
components of the same layer stack.

Why this exists. On qwen3.5-35b-a3b the step is GDN-bound: the whole delta-rule
path lands in one opaque shard_map custom-call, so an xplane profile reports it
as a single 5.69 s block and says nothing about which stage inside it is
expensive. This benchmark runs the stages as separate jitted functions at the
per-device shapes the real run uses, so each one gets its own time and its own
arithmetic intensity.

It is deliberately pure JAX and uses no Pallas or Mosaic, exactly like
`jax_chunk_gated_delta_rule` in models/qwen3.py, so it runs on any TPU
generation including the v4 dev box.

FLOP counting is conventional (2*M*N*K for a matmul). MFU is against the bf16
peak of one device, passed with --peak-tflops.

  python3 scripts/gdn_component_bench.py --seq 8192 --peak-tflops 137.5
"""

import argparse
import functools
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax

# qwen3.5-35b-a3b, per device at per_device_batch_size 1.
GEOM = dict(
    emb=2048,
    layers=40,
    cycle=4,  # every 4th layer is full attention, so 30 GDN + 10 attention
    # GDN
    h_k=16,
    h_v=32,
    d_k=128,
    d_v=128,
    conv_kernel=4,
    chunk=64,
    # full attention
    q_heads=16,
    kv_heads=2,
    attn_head_dim=256,
    # MoE
    experts=256,
    top_k=8,
    moe_hidden=512,
)


def timeit(fn, *args, iters=20, warmup=3):
  """Median wall time of a jitted call, in milliseconds."""
  f = jax.jit(fn)
  out = f(*args)
  jax.block_until_ready(out)
  for _ in range(warmup):
    jax.block_until_ready(f(*args))
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    ts.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(ts)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq", type=int, default=8192)
  ap.add_argument("--batch", type=int, default=1)
  ap.add_argument("--chunk", type=int, default=GEOM["chunk"])
  ap.add_argument("--peak-tflops", type=float, default=137.5, help="bf16 peak of ONE device")
  ap.add_argument("--dtype", default="bfloat16")
  args = ap.parse_args()

  g = dict(GEOM, chunk=args.chunk)
  B, T = args.batch, args.seq
  H, Dk, Dv, C = g["h_v"], g["d_k"], g["d_v"], g["chunk"]
  NC = T // C
  dt = jnp.dtype(args.dtype)
  k = jax.random.key(0)

  def rnd(shape, d=dt):
    nonlocal k
    k, sub = jax.random.split(k)
    return jax.random.normal(sub, shape, dtype=jnp.float32).astype(d)

  rows = []  # (component, stage, ms, flops)

  def add(component, stage, ms, flops):
    rows.append((component, stage, ms, flops))

  # ---------------------------------------------------------------- GDN inputs
  x = rnd((B, T, g["emb"]))
  key_dim, value_dim = g["d_k"] * g["h_k"], g["d_v"] * g["h_v"]
  conv_dim = key_dim * 2 + value_dim

  # Stage A: the two input projections.
  w_qkvz = rnd((g["emb"], key_dim * 2 + value_dim * 2))
  w_ba = rnd((g["emb"], g["h_v"] * 2))
  add("GDN", "in_proj_qkvz", timeit(lambda a, w: a @ w, x, w_qkvz),
      2 * B * T * g["emb"] * (key_dim * 2 + value_dim * 2))
  add("GDN", "in_proj_ba", timeit(lambda a, w: a @ w, x, w_ba), 2 * B * T * g["emb"] * g["h_v"] * 2)

  # Stage B: depthwise causal conv1d over the qkv stream, then silu.
  xc = rnd((B, T, conv_dim))
  wc = rnd((g["conv_kernel"], conv_dim))

  def dwconv(a, w):
    acc = jnp.zeros_like(a)
    for i in range(g["conv_kernel"]):
      acc = acc + jnp.pad(a, ((0, 0), (g["conv_kernel"] - 1 - i, 0), (0, 0)))[:, : a.shape[1]] * w[i]
    return jax.nn.silu(acc)

  add("GDN", "conv1d_depthwise+silu", timeit(dwconv, xc, wc), 2 * B * T * conv_dim * g["conv_kernel"])

  # Stage C: the gates. Elementwise, so ~0 FLOPs but real HBM traffic.
  b_raw, a_raw = rnd((B, T, g["h_v"]), jnp.float32), rnd((B, T, g["h_v"]), jnp.float32)
  a_log, dt_bias = rnd((g["h_v"],), jnp.float32), rnd((g["h_v"],), jnp.float32)
  add("GDN", "gates(sigmoid,softplus,exp)",
      timeit(lambda b, a, al, db: (jax.nn.sigmoid(b), -jnp.exp(al) * jax.nn.softplus(a + db)),
             b_raw, a_raw, a_log, dt_bias), 0)

  # ------------------------------------------------- chunked delta rule stages
  q_c = rnd((B, NC, H, C, Dk))
  k_c = rnd((B, NC, H, C, Dk))
  v_c = rnd((B, NC, H, C, Dv))
  g_cs = rnd((B, NC, H, C), jnp.float32)
  beta_c = rnd((B, NC, H, C), jnp.float32)
  HI = lax.Precision.HIGHEST

  # C1: the S matrix, k_beta @ k^T per chunk, at HIGHEST precision as shipped.
  def s_matrix(kb, kk):
    return jnp.matmul(kb, kk.swapaxes(-1, -2), precision=HI).astype(jnp.float32)

  k_beta = (k_c * beta_c[..., None].astype(dt))
  add("GDN", "S = k_beta @ k^T", timeit(s_matrix, k_beta, k_c), 2 * B * NC * H * C * C * Dk)

  # C2: decay mask. cumsum + a CxC outer difference + exp, no matmul at all.
  def decay(gc):
    cs = jnp.cumsum(gc, axis=-1)
    d = cs[..., :, None] - cs[..., None, :]
    m = jnp.tril(jnp.ones((C, C), dtype=bool), k=-1)
    return jnp.where(m, jnp.exp(jnp.where(m, d, -1e30)), 0.0)

  add("GDN", "cumsum + exp(g_diff) mask", timeit(decay, g_cs), 0)

  # C3: the triangular inverse. Sequential in C, and the reason chunk size is a
  # first-class perf knob.
  S = rnd((B, NC, H, C, C), jnp.float32)

  def tri_solve(s):
    eye = jnp.eye(C, dtype=jnp.float32)
    return jax.scipy.linalg.solve_triangular(
        eye + jnp.tril(s, -1), jnp.broadcast_to(eye, s.shape), lower=True, unit_diagonal=True
    )

  add("GDN", "solve_triangular (A)", timeit(tri_solve, S), B * NC * H * C * C * C // 3)

  # C4: the WY factors, two matmuls against A.
  A = rnd((B, NC, H, C, C), jnp.float32)
  add("GDN", "WY: u = A@v_beta", timeit(lambda a, v: jnp.matmul(a, v, precision=HI), A, v_c.astype(jnp.float32)),
      2 * B * NC * H * C * C * Dv)
  add("GDN", "WY: w = A@k_beta_g", timeit(lambda a, kk: jnp.matmul(a, kk, precision=HI), A, k_beta.astype(jnp.float32)),
      2 * B * NC * H * C * C * Dk)

  # C5: the inter-chunk scan. Sequential over NC, four matmuls per chunk.
  w_s = rnd((NC, B, H, C, Dk), jnp.float32)
  u_s = rnd((NC, B, H, C, Dv), jnp.float32)
  q_s = rnd((NC, B, H, C, Dk))
  k_s = rnd((NC, B, H, C, Dk))
  g_s = rnd((NC, B, H, C), jnp.float32)

  def scan_all(w_, u_, q_, k_, g_):
    h0 = jnp.zeros((B, H, Dk, Dv), jnp.float32)

    def body(h, xs):
      w, u, q, kk, gg = xs
      qg = q.astype(jnp.float32) * jnp.exp(gg)[..., None]
      attn_inter = jnp.matmul(qg, h, precision=HI)
      v_new = u - jnp.matmul(w, h, precision=HI)
      at = jnp.matmul(q, kk.swapaxes(-1, -2), precision=HI).astype(jnp.float32)
      gd = gg[..., :, None] - gg[..., None, :]
      m = jnp.tril(jnp.ones((C, C), dtype=bool))
      at = jnp.where(m, at * jnp.exp(jnp.where(m, gd, -1e30)), 0.0)
      o = attn_inter + jnp.matmul(at, v_new, precision=HI)
      hn = h * jnp.exp(gg[..., -1, None, None])
      kg = kk.astype(jnp.float32) * jnp.exp(gg[..., -1, None] - gg)[..., None]
      return hn + jnp.matmul(kg.swapaxes(-1, -2), v_new, precision=HI), o

    return lax.scan(body, h0, (w_, u_, q_, k_, g_))

  scan_flops = B * NC * H * 2 * (C * Dk * Dv + C * Dk * Dv + C * C * Dk + C * C * Dv + Dk * C * Dv)
  add("GDN", "inter-chunk scan (5 matmul)", timeit(scan_all, w_s, u_s, q_s, k_s, g_s), scan_flops)

  # Stage D: gated RMSNorm and the output projection.
  o_flat = rnd((B, T, value_dim))
  w_o = rnd((value_dim, g["emb"]))
  add("GDN", "out_proj", timeit(lambda a, w: a @ w, o_flat, w_o), 2 * B * T * value_dim * g["emb"])

  # ------------------------------------------------------- the other components
  # Full attention layer, 16 q heads over 2 kv heads at head_dim 256.
  hd, qh, kvh = g["attn_head_dim"], g["q_heads"], g["kv_heads"]
  qa = rnd((B, qh, T, hd))
  ka = rnd((B, qh, T, hd))
  va = rnd((B, qh, T, hd))

  def attn(qq, kk, vv):
    s = jnp.einsum("bhqd,bhkd->bhqk", qq, kk) / jnp.sqrt(hd)
    m = jnp.tril(jnp.ones((T, T), dtype=bool))
    s = jnp.where(m, s, -1e30)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s.astype(jnp.float32), -1).astype(dt), vv)

  # Causal, so about half the dense score matrix is useful work.
  add("Attention", "scores+softmax+AV (causal)", timeit(attn, qa, ka, va), 2 * 2 * B * qh * T * T * hd // 2)

  # MoE expert GEMMs. rows per expert per device = T*top_k/experts.
  rows_per_expert = T * g["top_k"] // g["experts"]
  xe = rnd((g["experts"], rows_per_expert, g["emb"]))
  wi = rnd((g["experts"], g["emb"], g["moe_hidden"] * 2))
  wo = rnd((g["experts"], g["moe_hidden"], g["emb"]))

  def moe(xx, w1, w2):
    h = jnp.einsum("erd,edh->erh", xx, w1)
    a, bb = jnp.split(h, 2, axis=-1)
    return jnp.einsum("erh,ehd->erd", jax.nn.silu(a) * bb, w2)

  moe_flops = 2 * g["experts"] * rows_per_expert * g["emb"] * g["moe_hidden"] * 3
  add("MoE", f"expert GEMMs ({rows_per_expert} rows/expert)", timeit(moe, xe, wi, wo), moe_flops)

  # Router.
  wr = rnd((g["emb"], g["experts"]))
  add("MoE", "router + top-k", timeit(lambda a, w: lax.top_k(a @ w, g["top_k"]), x, wr),
      2 * B * T * g["emb"] * g["experts"])

  # -------------------------------------------------------------------- report
  peak = args.peak_tflops
  print(f"\nqwen3.5-35b-a3b geometry, per device, B={B} T={T} chunk={C} dtype={args.dtype}")
  print(f"device {jax.devices()[0].device_kind}, bf16 peak {peak} TF/s, {NC} chunks, H={H} Dk={Dk} Dv={Dv}\n")
  print(f"{'component':<11} {'stage':<34} {'ms':>9} {'GFLOP':>9} {'TF/s':>8} {'MFU%':>7}")
  print("-" * 82)
  for comp, stage, ms, fl in rows:
    tfs = (fl / 1e12) / (ms / 1e3) if ms > 0 and fl > 0 else 0.0
    print(f"{comp:<11} {stage:<34} {ms:9.3f} {fl/1e9:9.2f} {tfs:8.1f} {100*tfs/peak:7.2f}")

  gdn_ms = sum(m for c, _, m, _ in rows if c == "GDN")
  att_ms = sum(m for c, _, m, _ in rows if c == "Attention")
  moe_ms = sum(m for c, _, m, _ in rows if c == "MoE")
  gdn_fl = sum(f for c, _, _, f in rows if c == "GDN")
  att_fl = sum(f for c, _, _, f in rows if c == "Attention")
  moe_fl = sum(f for c, _, _, f in rows if c == "MoE")

  print("\nper-layer totals (one layer of each kind)")
  print(f"{'kind':<12} {'ms':>9} {'GFLOP':>9} {'TF/s':>8} {'MFU%':>7}")
  for name, ms, fl in (("GDN", gdn_ms, gdn_fl), ("Attention", att_ms, att_fl), ("MoE", moe_ms, moe_fl)):
    tfs = (fl / 1e12) / (ms / 1e3) if ms else 0
    print(f"{name:<12} {ms:9.3f} {fl/1e9:9.2f} {tfs:8.1f} {100*tfs/peak:7.2f}")

  # Whole stack: cycle=4 means 3 GDN + 1 attention per cycle, MoE in every layer.
  n_gdn = g["layers"] - g["layers"] // g["cycle"]
  n_att = g["layers"] // g["cycle"]
  tot_ms = n_gdn * gdn_ms + n_att * att_ms + g["layers"] * moe_ms
  tot_fl = n_gdn * gdn_fl + n_att * att_fl + g["layers"] * moe_fl
  print(f"\nfull stack, {n_gdn} GDN + {n_att} attention + {g['layers']} MoE layers (forward only)")
  print(f"{'':<12} {'ms':>9} {'share':>8} {'GFLOP':>10} {'share':>8}")
  for name, ms, fl in (
      ("GDN", n_gdn * gdn_ms, n_gdn * gdn_fl),
      ("Attention", n_att * att_ms, n_att * att_fl),
      ("MoE", g["layers"] * moe_ms, g["layers"] * moe_fl),
  ):
    print(f"{name:<12} {ms:9.1f} {100*ms/tot_ms:7.1f}% {fl/1e9:10.1f} {100*fl/tot_fl:7.1f}%")
  print(f"{'TOTAL':<12} {tot_ms:9.1f} {'':8} {tot_fl/1e9:10.1f}")
  print(f"\nforward TF/s {(tot_fl/1e12)/(tot_ms/1e3):.1f}  = {100*((tot_fl/1e12)/(tot_ms/1e3))/peak:.2f}% MFU")


if __name__ == "__main__":
  main()
