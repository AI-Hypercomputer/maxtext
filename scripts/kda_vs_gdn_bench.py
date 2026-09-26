"""KDA vs GDN (Gated DeltaNet) kernels: analytic roofline and measured time, one device.

KDA and GDN run the same chunked delta rule. The only difference is the decay:
GDN has one scalar per head and token, KDA one per head, token and key channel.
That changes how well each one maps onto the MXU. For GDN the intra-chunk decay
is a C x C factor applied after the K K^T matmul. For KDA it sits inside the
contraction (sum_d k_id k_jd exp(G_id - G_jd)), so it is only a matmul if the
decay is folded into the operands, which overflows unless the chunk is split.

Implementations timed, all at the OLMo 3.5 tiny per-device shape:

  kda_tokamax      tokamax fused Pallas KDA kernel (what OLMo 3.5 trains with)
  gdn_tokamax      the same kernel with the gate broadcast over the key dim (GDN math)
  gdn_maxtext      maxtext.models.qwen3.jax_chunk_gated_delta_rule (pure JAX, what qwen3.5 trains with)
  gdn_naive        maxtext.models.qwen3.naive_jax_chunk_gated_delta_rule (HIGHEST precision)
  kda_jnp          maxtext.models.olmoe3._delta_rule_chunked (pure JAX KDA, fp32 state)
  kda_jnp_bf16     the same with bf16 state

The tokamax precision switches (TOKAMAX_KDA_BF16_FWD/BWD, DENSE_PAIRS) are read at
import, so run the script once per setting:

  TOKAMAX_KDA_BF16_FWD=1 TOKAMAX_KDA_BF16_BWD=1 TOKAMAX_KDA_DENSE_PAIRS=1 \
    python3 scripts/kda_vs_gdn_bench.py --tag bf16
"""

import argparse
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp

PEAK_TFLOPS = 1153.5  # bf16, per v7x device
HBM_TBS = 3.69  # per v7x device


def roofline(b, t, h, dk, dv, c, kda):
  """Chunked delta rule (WY form), per device. Returns (mxu_flop, min_bytes) for fwd and bwd.

  MXU work per chunk: K K^T and Q K^T (2 C^2 dk each), the WY solve applied to
  k and v (2 C^2 dk + 2 C^2 dv), the intra-chunk output (2 C^2 dv), and three
  C x dk x dv state products (q S, w S, k^T v). The bwd is taken as 2x the fwd.
  Minimum bytes is one read of q, k, v, gate, beta and one write of o in bf16,
  plus the per-chunk fp32 state the bwd needs (saved or recomputed, it costs one
  write and one read of dk x dv per chunk either way unless kept in VMEM).
  """
  n_chunks = t // c
  per_chunk = 2 * c * c * dk * 3 + 2 * c * c * dv * 2 + 6 * c * dk * dv + c**3
  fwd_flop = per_chunk * n_chunks * b * h
  gate = dk if kda else 1
  io_fwd = 2 * b * t * h * (dk + dk + dv + gate + 1 + dv)
  state = 4 * b * h * n_chunks * dk * dv
  # bwd reads the fwd inputs and do, writes dq dk dv dgate dbeta, reads the states.
  io_bwd = 2 * b * t * h * (2 * (dk + dk + dv + gate + 1) + dv) + state
  return {"fwd_flop": fwd_flop, "bwd_flop": 2 * fwd_flop, "fwd_bytes": io_fwd + state, "bwd_bytes": io_bwd}


def make_inputs(b, t, h, dk, dv, seed=0, dtype=jnp.bfloat16):
  ks = jax.random.split(jax.random.key(seed), 6)
  q = jax.random.normal(ks[0], (b, t, h, dk), dtype)
  k = jax.random.normal(ks[1], (b, t, h, dk), dtype)
  v = jax.random.normal(ks[2], (b, t, h, dv), dtype)
  # Log decays in the trained range: KDA per channel, GDN one per head.
  g_kda = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, h, dk), jnp.float32) - 2.0)
  g_gdn = g_kda.mean(axis=-1)
  beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, t, h), jnp.float32)).astype(jnp.bfloat16)
  return q, k, v, g_kda, g_gdn, beta


def l2n(x):
  x = x.astype(jnp.float32)
  return (x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)).astype(jnp.bfloat16)


def build(name, chunk):
  """Returns f(q, k, v, g, beta) -> o [B, T, H, dv], and which gate it takes."""
  if name in ("kda_tokamax", "gdn_tokamax"):
    from tokamax._src.ops.experimental.kda import api as kda_api  # pylint: disable=import-outside-toplevel

    def hf(x):
      return jnp.moveaxis(x, 2, 0)

    def f(q, k, v, g, beta):
      if g.ndim == 3:  # GDN: one decay per head, broadcast over the key channels
        g = jnp.broadcast_to(g[..., None], q.shape).astype(jnp.float32)
      o, _ = kda_api.kimi_delta_attention(
          hf(q), hf(k), hf(v), hf(g), jnp.moveaxis(beta, 2, 0), use_qk_l2norm=True, implementation="mosaic"
      )
      return jnp.moveaxis(o, 0, 2)

    return f, ("kda" if name == "kda_tokamax" else "gdn")

  if name in ("gdn_maxtext", "gdn_naive"):
    from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel

    fn = qwen3.jax_chunk_gated_delta_rule if name == "gdn_maxtext" else qwen3.naive_jax_chunk_gated_delta_rule

    def f(q, k, v, g, beta):
      o, _ = fn(q, k, v, g, beta, chunk_size=chunk, use_qk_norm_in_gdn=True)
      return o

    return f, "gdn"

  if name in ("kda_jnp", "kda_jnp_bf16"):
    from maxtext.models import olmoe3  # pylint: disable=import-outside-toplevel

    state_dtype = "bfloat16" if name.endswith("bf16") else "float32"

    def f(q, k, v, g, beta):
      scale = q.shape[-1] ** -0.5
      resets = jnp.zeros(q.shape[:2], bool)
      return olmoe3._delta_rule_chunked(  # pylint: disable=protected-access
          l2n(q) * scale, l2n(k), v, g, beta.astype(jnp.float32), resets, chunk, state_dtype
      )

    return f, "kda"

  raise ValueError(name)


def timeit(fn, args, iters):
  out = fn(*args)
  jax.block_until_ready(out)
  jax.block_until_ready(fn(*args))
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    times.append(time.perf_counter() - t0)
  return statistics.median(times), out


def xla_cost(jitted, args):
  try:
    ca = jitted.lower(*args).compile().cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    return ca.get("flops", 0.0), ca.get("bytes accessed", 0.0)
  except Exception:  # pylint: disable=broad-except
    return 0.0, 0.0


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--impls", default="kda_tokamax,gdn_tokamax,gdn_maxtext,gdn_naive,kda_jnp,kda_jnp_bf16")
  p.add_argument("--batch", type=int, default=3)
  p.add_argument("--seq", type=int, default=8192)
  p.add_argument("--heads", type=int, default=8)
  p.add_argument("--dk", type=int, default=128)
  p.add_argument("--dv", type=int, default=256)
  p.add_argument("--chunk", type=int, default=64)
  p.add_argument("--iters", type=int, default=10)
  # float32 reproduces the pre-fix model, where the fp32 conv weight promoted q/k/v.
  p.add_argument("--in-dtype", default="bfloat16")
  p.add_argument("--tag", default=os.environ.get("TOKAMAX_KDA_BF16_FWD", "0") == "1" and "bf16" or "fp32")
  a = p.parse_args()
  dev = jax.devices()[0]
  print(f"# device {dev.device_kind} x{jax.device_count()}, jax {jax.__version__}, tag {a.tag}", flush=True)

  with jax.default_device(dev):
    q, k, v, g_kda, g_gdn, beta = make_inputs(a.batch, a.seq, a.heads, a.dk, a.dv, dtype=jnp.dtype(a.in_dtype))
    ct = jax.random.normal(jax.random.key(9), (a.batch, a.seq, a.heads, a.dv), jnp.float32)
    ref = {}
    for name in a.impls.split(","):
      row = {"impl": name, "tag": a.tag, "B": a.batch, "T": a.seq, "H": a.heads, "dk": a.dk, "dv": a.dv, "C": a.chunk}
      try:
        f, kind = build(name, a.chunk)
        g = g_kda if kind == "kda" else g_gdn
        rl = roofline(a.batch, a.seq, a.heads, a.dk, a.dv, a.chunk, kind == "kda")
        fwd = jax.jit(f)

        def loss(q_, k_, v_, g_, b_, f=f):
          return jnp.sum(f(q_, k_, v_, g_, b_).astype(jnp.float32) * ct)

        fb = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))
        args = (q, k, v, g, beta)
        t_f, o = timeit(fwd, args, a.iters)
        t_fb, _ = timeit(fb, args, a.iters)
        xf, xb = xla_cost(fwd, args)
        o = o.astype(jnp.float32)
        ref.setdefault(kind, o)
        row.update(
            fwd_ms=1e3 * t_f,
            fwdbwd_ms=1e3 * t_fb,
            bwd_ms=1e3 * (t_fb - t_f),
            roof_fwd_ms=1e3 * max(rl["fwd_flop"] / (PEAK_TFLOPS * 1e12), rl["fwd_bytes"] / (HBM_TBS * 1e12)),
            roof_fwdbwd_ms=1e3
            * max(
                (rl["fwd_flop"] + rl["bwd_flop"]) / (PEAK_TFLOPS * 1e12),
                (rl["fwd_bytes"] + rl["bwd_bytes"]) / (HBM_TBS * 1e12),
            ),
            mxu_tflops_fwdbwd=(rl["fwd_flop"] + rl["bwd_flop"]) / t_fb / 1e12,
            xla_fwd_gflop=xf / 1e9,
            xla_fwd_gb=xb / 1e9,
            max_abs_vs_first_same_kind=float(jnp.max(jnp.abs(o - ref[kind]))),
            finite=bool(jnp.isfinite(o).all()),
        )
      except Exception as e:  # pylint: disable=broad-except
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
      print(json.dumps(row), flush=True)

    rl_k = roofline(a.batch, a.seq, a.heads, a.dk, a.dv, a.chunk, True)
    rl_g = roofline(a.batch, a.seq, a.heads, a.dk, a.dv, a.chunk, False)
    print("# roofline kda " + json.dumps(rl_k))
    print("# roofline gdn " + json.dumps(rl_g))


if __name__ == "__main__":
  main()
