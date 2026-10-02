# Copyright 2023–2026 Google LLC
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

"""TPU check + benchmark for the fused MoE combine (run on one v6e chip).

  PYTHONPATH=<pkg root> python -m maxtext.kernels.moe_combine_bench [T] [K] [G] [E]

Checks the compiled kernels against the jnp reference at production shape
(default T=32768 tokens/chip, K=8, G=128 experts, E=2816) and times:
  * baseline fwd: x[argsort(sort_idx)] + f32 einsum (XLA; SC offload if flags set)
  * baseline fwd+bwd (value_and_grad of sum(y * r))
  * fused fwd / fused fwd+bwd / bwd-only (vjp pullback; fwd kernel is DCE'd)
    for block_tokens in MAXTEXT_G4_COMBINE_BENCH_BLOCKS (default "128,256");
    all MAXTEXT_G4_COMBINE_* kernel env vars are honoured.
  * microbench breakdown: the same timings with MAXTEXT_G4_COMBINE_MODE set to
    each of MAXTEXT_G4_COMBINE_BENCH_MODES (default "dma,compute,nomerge"; ""
    disables). dma = DMAs + waits only; compute = MXU/VPU only (no DMAs);
    nomerge = full bwd minus partial-granule staging (results wrong on purpose).
  * backward variants: MAXTEXT_G4_COMBINE_BENCH_BWDS (default "v4,v5:512,v5:1024",
    name[:row chunk]) -> kernel-only bwd, bwd-only and correctness per variant.
  * MAXTEXT_G4_COMBINE_BENCH_WDTYPE=bf16|f32 (default bf16): routing weight dtype
    (f32 exercises the exact hi/lo split, i.e. 2 matmuls for S).
Each block size is tried independently (a compile failure is reported and the
next size still runs). Exit code != 0 if any config failed or dx was not exact.
"""

import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import moe_combine_tc as mc

# CPU smoke test of the bench script itself (tiny shapes): ..._BENCH_INTERPRET=1.
_INTERP = os.environ.get("MAXTEXT_G4_COMBINE_BENCH_INTERPRET", "0") == "1"


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
  key = jax.random.key(seed)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)
  # Mildly imbalanced routing: gumbel top-k over logits with per-expert bias.
  bias = 0.5 * jax.random.normal(k1, (g,))
  logits = bias[None, :] + jax.random.gumbel(k2, (t, g))
  topw, topi = jax.lax.top_k(logits, k)
  w = jax.nn.softmax(topw, axis=-1).astype(wdtype)
  flat = topi.reshape(-1).astype(jnp.int32)
  sort_idx = jnp.argsort(flat)
  gs = jnp.bincount(flat, length=g).astype(jnp.int32)
  x = jax.random.normal(k3, (t * k, e), jnp.float32).astype(jnp.bfloat16)
  dy = jax.random.normal(k4, (t, e), jnp.float32).astype(jnp.bfloat16)
  del k5
  return x, sort_idx, w, gs, dy


def main(t=32768, k=8, g=128, e=2816):
  print(jax.devices()[0].device_kind, f"T={t} K={k} G={g} E={e}")
  wdt = {"bf16": jnp.bfloat16, "f32": jnp.float32}[os.environ.get("MAXTEXT_G4_COMBINE_BENCH_WDTYPE", "bf16")]
  x, sort_idx, w, gs, dy = _problem(t, k, g, e, wdtype=wdt)
  print("weights dtype:", w.dtype)
  print("group sizes min/max:", int(gs.min()), int(gs.max()))

  ref_f = jax.jit(mc.combine_reference)

  def vg(fn):
    def loss(a, b):
      return jnp.sum(fn(a, b).astype(jnp.float32) * dy.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

  ms, y_ref = _timeit(ref_f, x, sort_idx, w)
  print(f"baseline fwd          : {ms:8.3f} ms")
  ref_vg = vg(lambda a, b: mc.combine_reference(a, sort_idx, b))
  ms, (l_ref, (dx_ref, dw_ref)) = _timeit(ref_vg, x, w)
  print(f"baseline fwd+bwd      : {ms:8.3f} ms")

  blocks = [int(s) for s in os.environ.get("MAXTEXT_G4_COMBINE_BENCH_BLOCKS", "128,256").split(",") if s]
  align = mc.align_from_env()
  modes = [m for m in os.environ.get("MAXTEXT_G4_COMBINE_BENCH_MODES", "dma,compute,nomerge").split(",") if m]
  os.environ.pop("MAXTEXT_G4_COMBINE_MODE", None)

  def _bwd_only(c):
    def pull(a, b, gg):
      _, vjp = jax.vjp(lambda aa, bb: mc.combine(aa, sort_idx, bb, gs, block_tokens=c, interpret=_INTERP), a, b)
      return vjp(gg)

    ms, _ = _timeit(jax.jit(pull), x, w, dy)
    return ms

  failures = 0
  for c in blocks:
    print(f"--- C={c} ALIGN={align} J={mc._buffer_rows(c, k, g, align)}", flush=True)  # pylint: disable=protected-access
    try:
      f = jax.jit(functools.partial(mc.combine, block_tokens=c, interpret=_INTERP))
      ms, y = _timeit(f, x, sort_idx, w, gs)
      d = np.abs(np.asarray(y, np.float32) - np.asarray(y_ref, np.float32))
      print(f"fused fwd  C={c:4d}     : {ms:8.3f} ms   max|y-y_ref|={d.max():.3e}", flush=True)
      kvg = vg(lambda a, b, c=c: mc.combine(a, sort_idx, b, gs, block_tokens=c, interpret=_INTERP))
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
      print(f"  |dw_ref| max {dwr.max():.3e} mean {dwr.mean():.3e}; rel max|dw-dw_ref|/max|dw_ref| = {dwd / dwr.max():.3e}")
      bwd_ms = _bwd_only(c)
      print(f"fused bwd-only C={c:4d} : {bwd_ms:8.3f} ms   (metadata + row info + bwd kernel)", flush=True)
      # kernels alone (precomputed metadata / row info / weights)
      split = w.dtype != jnp.bfloat16
      cfgk = dict(block_tokens=c, num_groups=g, split=bool(split), interpret=_INTERP, align=align, opts=mc._kernel_opts() + (("upcast", _INTERP),))  # pylint: disable=protected-access
      tab_c = jax.jit(lambda si, gg, c=c: mc._metadata(si, gg, t, k, c, align))(sort_idx, gs)  # pylint: disable=protected-access
      info_c = jax.jit(lambda si: mc._row_info(si, k))(sort_idx)  # pylint: disable=protected-access
      wp = mc._padded_weights(w)  # pylint: disable=protected-access
      kf = jax.jit(lambda a, b, tb, inf, cfgk=cfgk: mc._fwd_call(a, b, tb, inf, **cfgk))  # pylint: disable=protected-access
      kb = jax.jit(lambda a, b, tb, inf, d, cfgk=cfgk: mc._bwd_call(a, b, tb, inf, d, **cfgk))  # pylint: disable=protected-access
      fk, _ = _timeit(kf, x, wp, tab_c, info_c)
      bk, _ = _timeit(kb, x, wp, tab_c, info_c, dy)
      ri, _ = _timeit(jax.jit(lambda si: mc._row_info(si, k)), sort_idx)  # pylint: disable=protected-access
      print(f"  kernel-only C={c:4d}: fwd {fk:8.3f} ms   bwd {bk:8.3f} ms   (row info {ri:.3f} ms)", flush=True)
      # backward variants (MAXTEXT_G4_COMBINE_BENCH_BWDS, e.g. "v4,v5:512,v5:1024"):
      # kernel-only bwd, bwd-only pullback and correctness of each.
      for bv in [v for v in os.environ.get("MAXTEXT_G4_COMBINE_BENCH_BWDS", "v4,v5:512,v5:1024").split(",") if v]:
        saved = {n: os.environ.get(n) for n in ("MAXTEXT_G4_COMBINE_BWD", "MAXTEXT_G4_COMBINE_BWD_RQ")}
        name, _, rq = bv.partition(":")
        os.environ["MAXTEXT_G4_COMBINE_BWD"] = name
        if rq:
          os.environ["MAXTEXT_G4_COMBINE_BWD_RQ"] = rq
        try:
          cfgv = dict(cfgk, opts=mc._kernel_opts() + (("upcast", _INTERP),))  # pylint: disable=protected-access
          bkv, _ = _timeit(jax.jit(lambda a, b, tb, inf, d, cfgv=cfgv: mc._bwd_call(a, b, tb, inf, d, **cfgv)), x, wp, tab_c, info_c, dy)  # pylint: disable=protected-access
          bov = _bwd_only(c)
          _, vjp_v = jax.vjp(lambda aa, bb, c=c: mc.combine(aa, sort_idx, bb, gs, block_tokens=c, interpret=_INTERP), x, w)
          dxv, dwv = jax.jit(vjp_v)(dy)
          dxv_eq = bool(np.array_equal(np.asarray(dxv, np.float32), np.asarray(dx_ref, np.float32)))
          dwv_err = np.abs(np.asarray(dwv, np.float32) - np.asarray(dw_ref, np.float32)).max() / dwr.max()
          print(f"  bwd={bv:9s} C={c:4d}: kernel-only bwd {bkv:8.3f} ms   bwd-only {bov:8.3f} ms   dx exact={dxv_eq}  rel dw err={dwv_err:.2e}", flush=True)
          if not dxv_eq:
            failures += 1
        except Exception as ex:  # pylint: disable=broad-exception-caught
          failures += 1
          print(f"  bwd={bv} FAILED: {type(ex).__name__}: " + " | ".join(str(ex).strip().splitlines()[:6])[:1500], flush=True)
        finally:
          for n_, v_ in saved.items():
            if v_ is None:
              os.environ.pop(n_, None)
            else:
              os.environ[n_] = v_
      # metadata alone (XLA side)
      meta = jax.jit(lambda si, gg, c=c: mc._metadata(si, gg, t, k, c, align))  # pylint: disable=protected-access
      ms, _ = _timeit(meta, sort_idx, gs)
      print(f"  metadata C={c:4d}: {ms:8.3f} ms", flush=True)
      for mode in modes:
        os.environ["MAXTEXT_G4_COMBINE_MODE"] = mode
        try:
          f = jax.jit(functools.partial(mc.combine, block_tokens=c, interpret=_INTERP))
          fms, _ = _timeit(f, x, sort_idx, w, gs)
          bms = _bwd_only(c)
          cfgm = dict(cfgk, opts=mc._kernel_opts() + (("upcast", _INTERP),))  # pylint: disable=protected-access
          fkm, _ = _timeit(jax.jit(lambda a, b, tb, inf, cfgm=cfgm: mc._fwd_call(a, b, tb, inf, **cfgm)), x, wp, tab_c, info_c)  # pylint: disable=protected-access
          bkm, _ = _timeit(jax.jit(lambda a, b, tb, inf, d, cfgm=cfgm: mc._bwd_call(a, b, tb, inf, d, **cfgm)), x, wp, tab_c, info_c, dy)  # pylint: disable=protected-access
          print(
              f"  mode={mode:8s} C={c:4d}: fwd {fms:8.3f} ms   bwd-only {bms:8.3f} ms   kernel-only fwd {fkm:8.3f} bwd {bkm:8.3f}",
              flush=True,
          )
        except Exception as ex:  # pylint: disable=broad-exception-caught
          print(f"  mode={mode} FAILED: {type(ex).__name__}: {str(ex).strip().splitlines()[:3]}", flush=True)
        finally:
          os.environ.pop("MAXTEXT_G4_COMBINE_MODE", None)
    except Exception as ex:  # pylint: disable=broad-exception-caught
      failures += 1
      msg = str(ex).strip().splitlines()
      print(f"FAILED C={c}: {type(ex).__name__}: " + " | ".join(msg[:6])[:2000], flush=True)
  print("BENCH RESULT:", "OK" if failures == 0 else f"{failures} FAILURE(S)", flush=True)
  return failures


if __name__ == "__main__":
  n_fail = main(*[int(a) for a in sys.argv[1:]])
  sys.exit(1 if (n_fail and os.environ.get("MAXTEXT_G4_COMBINE_BENCH_STRICT", "0") == "1") else 0)
