# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU check + benchmark for the fused MoE combine kernel (run on one chip).

  PYTHONPATH=<pkg root> python -m maxtext.kernels.moe_combine_bench [--tokens T] [--top_k K] \
      [--num_experts G] [--emb E] [--blocks 128,256] [--weight_dtype bf16|f32] [--interpret] [--strict]

Checks the compiled kernel against the jnp reference at production shape
(default T=32768 tokens/chip, K=8, G=128 experts, E=2816) and times:
  * baseline fwd: x[argsort(sort_idx)] + f32 einsum (XLA; SparseCore offload if flags set)
  * baseline fwd+bwd (value_and_grad of sum(y * r))
  * fused fwd / fused fwd+bwd / bwd-only (vjp pullback; fwd kernel is DCE'd) per block size
  * kernel-only fwd / bwd (precomputed metadata and row info) and the XLA-side metadata.
Each block size is tried independently (a compile failure is reported and the
next size still runs). With --strict the exit code is 1 if any block failed or dx was not exact.
"""

import argparse
import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import moe_combine_tc as mc


def _timeit(f, *args, iters=20):
  out = f(*args)
  jax.block_until_ready(out)
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    ts.append(time.perf_counter() - t0)
  return 1e3 * float(np.median(ts)), out


def _problem(t, k, g, e, seed=0, wdtype=jnp.bfloat16):
  """Mildly imbalanced routing: gumbel top-k over logits with a per-expert bias."""
  key = jax.random.key(seed)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  bias = 0.5 * jax.random.normal(k1, (g,))
  logits = bias[None, :] + jax.random.gumbel(k2, (t, g))
  topw, topi = jax.lax.top_k(logits, k)
  w = jax.nn.softmax(topw, axis=-1).astype(wdtype)
  flat = topi.reshape(-1).astype(jnp.int32)
  sort_idx = jnp.argsort(flat)
  gs = jnp.bincount(flat, length=g).astype(jnp.int32)
  x = jax.random.normal(k3, (t * k, e), jnp.float32).astype(jnp.bfloat16)
  dy = jax.random.normal(k4, (t, e), jnp.float32).astype(jnp.bfloat16)
  return x, sort_idx, w, gs, dy


def main(t, k, g, e, blocks, wdtype, interpret, align):
  print(jax.devices()[0].device_kind, f"T={t} K={k} G={g} E={e}")
  x, sort_idx, w, gs, dy = _problem(t, k, g, e, wdtype=wdtype)
  print("weights dtype:", w.dtype)
  print("group sizes min/max:", int(gs.min()), int(gs.max()))

  def vg(fn):
    def loss(a, b):
      return jnp.sum(fn(a, b).astype(jnp.float32) * dy.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

  ms, y_ref = _timeit(jax.jit(mc.combine_reference), x, sort_idx, w)
  print(f"baseline fwd          : {ms:8.3f} ms")
  ref_vg = vg(lambda a, b: mc.combine_reference(a, sort_idx, b))
  ms, (l_ref, (dx_ref, dw_ref)) = _timeit(ref_vg, x, w)
  print(f"baseline fwd+bwd      : {ms:8.3f} ms")

  def _bwd_only(c):
    def pull(a, b, gg):
      _, vjp = jax.vjp(lambda aa, bb: mc.combine(aa, sort_idx, bb, gs, block_tokens=c, interpret=interpret), a, b)
      return vjp(gg)

    ms, _ = _timeit(jax.jit(pull), x, w, dy)
    return ms

  failures = 0
  for c in blocks:
    print(f"--- C={c} ALIGN={align} J={mc._buffer_rows(c, k, g, align)}", flush=True)  # pylint: disable=protected-access
    try:
      f = jax.jit(functools.partial(mc.combine, block_tokens=c, interpret=interpret, align=align))
      ms, y = _timeit(f, x, sort_idx, w, gs)
      d = np.abs(np.asarray(y, np.float32) - np.asarray(y_ref, np.float32))
      print(f"fused fwd  C={c:4d}     : {ms:8.3f} ms   max|y-y_ref|={d.max():.3e}", flush=True)
      kvg = vg(lambda a, b, c=c: mc.combine(a, sort_idx, b, gs, block_tokens=c, interpret=interpret, align=align))
      ms, (l, (dx, dw)) = _timeit(kvg, x, w)
      dx_eq = bool(np.array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32)))
      dwd = np.abs(np.asarray(dw, np.float32) - np.asarray(dw_ref, np.float32)).max()
      print(
          f"fused fwd+bwd C={c:4d}  : {ms:8.3f} ms   loss {float(l):.6e} vs {float(l_ref):.6e}"
          f"  dx exact={dx_eq}  max|dw-dw_ref|={dwd:.3e}",
          flush=True,
      )
      if not dx_eq:
        failures += 1
      dwr = np.abs(np.asarray(dw_ref, np.float32))
      print(
          f"  |dw_ref| max {dwr.max():.3e} mean {dwr.mean():.3e}; rel max|dw-dw_ref|/max|dw_ref| = {dwd / dwr.max():.3e}"
      )
      bwd_ms = _bwd_only(c)
      print(f"fused bwd-only C={c:4d} : {bwd_ms:8.3f} ms   (metadata + row info + bwd kernel)", flush=True)
      # Kernels alone (precomputed metadata / row info / padded weights).
      # pylint: disable=protected-access
      cfgk = dict(block_tokens=c, num_groups=g, split=w.dtype != jnp.bfloat16, interpret=interpret, align=align)
      tab_c = jax.jit(lambda si, gg, c=c: mc._metadata(si, gg, t, k, c, align))(sort_idx, gs)
      info_c = jax.jit(lambda si: mc._row_info(si, k))(sort_idx)
      wp = mc._padded_weights(w)
      kf = jax.jit(lambda a, b, tb, inf, cfgk=cfgk: mc._fwd_call(a, b, tb, inf, **cfgk))
      kb = jax.jit(lambda a, b, tb, inf, d, cfgk=cfgk: mc._bwd_call(a, b, tb, inf, d, **cfgk))
      fk, _ = _timeit(kf, x, wp, tab_c, info_c)
      bk, _ = _timeit(kb, x, wp, tab_c, info_c, dy)
      ri, _ = _timeit(jax.jit(lambda si: mc._row_info(si, k)), sort_idx)
      print(f"  kernel-only C={c:4d}: fwd {fk:8.3f} ms   bwd {bk:8.3f} ms   (row info {ri:.3f} ms)", flush=True)
      meta = jax.jit(lambda si, gg, c=c: mc._metadata(si, gg, t, k, c, align))
      # pylint: enable=protected-access
      ms, _ = _timeit(meta, sort_idx, gs)
      print(f"  metadata C={c:4d}: {ms:8.3f} ms", flush=True)
    except Exception as ex:  # pylint: disable=broad-exception-caught
      failures += 1
      msg = str(ex).strip().splitlines()
      print(f"FAILED C={c}: {type(ex).__name__}: " + " | ".join(msg[:6])[:2000], flush=True)
  print("BENCH RESULT:", "OK" if failures == 0 else f"{failures} FAILURE(S)", flush=True)
  return failures


def _parse_args(argv):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--tokens", type=int, default=32768, help="tokens per chip (T)")
  p.add_argument("--top_k", type=int, default=8, help="experts per token (K)")
  p.add_argument("--num_experts", type=int, default=128, help="number of experts (G)")
  p.add_argument("--emb", type=int, default=2816, help="model dimension (E)")
  p.add_argument("--blocks", default="128,256", help="comma-separated block_tokens values to try")
  p.add_argument("--align", type=int, default=8, help="DMA / VMEM row alignment")
  p.add_argument("--weight_dtype", choices=("bf16", "f32"), default="bf16", help="routing weight dtype")
  p.add_argument("--interpret", action="store_true", help="Pallas interpret mode (CPU smoke test at tiny shapes)")
  p.add_argument("--strict", action="store_true", help="exit 1 on any failure")
  return p.parse_args(argv)


if __name__ == "__main__":
  args = _parse_args(sys.argv[1:])
  n_fail = main(
      args.tokens,
      args.top_k,
      args.num_experts,
      args.emb,
      [int(s) for s in args.blocks.split(",") if s],
      {"bf16": jnp.bfloat16, "f32": jnp.float32}[args.weight_dtype],
      args.interpret,
      args.align,
  )
  sys.exit(1 if (n_fail and args.strict) else 0)
