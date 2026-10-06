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

"""TPU correctness + speed gate for the transposed-RHS gmm_v2 kernel (`use_gmm_v2_transposed_rhs_dlhs`).

  python3 -m maxtext.kernels.megablox.gmm_trhs_bench [--wi_tiles t1,...,t9]
      [--wo_tiles t1,...,t9] [--iters 10] [--tol 2e-2]

--wi_tiles / --wo_tiles: the 9 config tile values in `ops.gmm` order
(fwd m,k,n, dlhs m,k,n, drhs m,k,n) i.e. wi_tile_fwd_batch_seq, wi_tile_fwd_embed_dim,
wi_tile_fwd_mlp_dim, wi_tile_dlhs_batch_seq, wi_tile_dlhs_mlp_dim, wi_tile_dlhs_embed_dim,
wi_tile_drhs_batch_seq, wi_tile_drhs_embed_dim, wi_tile_drhs_mlp_dim (same for wo_*:
fwd m,mlp,embed / dlhs m,embed,mlp / drhs m,mlp,embed). Empty = gmm_v2 heuristic
tiling (what use_gmm_v2_heuristic_tiling=True does).

Runs on the first local TPU device (no jax.distributed needed). For each
production dlhs shape it compares

  NT   : gmm_v2_trhs.gmm_v2(dout, W, transpose_rhs=True)          (new, no copy)
  REF  : gmm_v2.gmm_v2(dout, W.swapaxes(1, 2))                    (current path)

(plain and with group_offset), then the full custom-VJP dlhs of `ops.gmm` (wi and wo
shapes) with use_gmm_v2_transposed_rhs_dlhs True vs False.
Prints max-abs-diff (and relative to max|REF|) and ms for both; exits 1 on any
mismatch (rel > --tol or non-finite), 2 on compile/runtime error.
"""

import argparse
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

M = 262144
EMB = 2816
WI_N = 1408  # 2 * moe_mlp_dim (gate|up) of a prefused wi
MLP = 704
E = 128


def _tiles9(s):
  if not s:
    return None
  t = tuple(int(v) for v in s.split(","))
  assert len(t) == 9, s
  return t


def _timeit(fn, args, iters):
  out = fn(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
  jax.block_until_ready(out)
  return out, (time.perf_counter() - t0) * 1e3 / iters


def _cmp(name, a, b, tol):
  a = np.asarray(jax.device_get(a)).astype(np.float32)
  b = np.asarray(jax.device_get(b)).astype(np.float32)
  finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
  d = float(np.max(np.abs(a - b))) if finite else float("inf")
  scale = float(np.max(np.abs(b))) or 1.0
  rel = d / scale
  ok = finite and rel <= tol
  print(
      f"  [{name}] max|NT-REF|={d:.4g} max|REF|={scale:.4g} rel={rel:.3g} finite={finite} -> {'OK' if ok else 'MISMATCH'}"
  )
  return ok


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--wi_tiles", default="", help="9 wi tile values (see module doc); empty = heuristic")
  ap.add_argument("--wo_tiles", default="", help="9 wo tile values (see module doc); empty = heuristic")
  ap.add_argument("--iters", type=int, default=10)
  ap.add_argument("--tol", type=float, default=2e-2)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  from maxtext.kernels.megablox import gmm_v2_trhs  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.megablox import ops  # pylint: disable=import-outside-toplevel

  dev = jax.local_devices()[0]
  print(f"device={dev} jax={jax.__version__}")
  if dev.platform != "tpu":
    print("gmm_trhs_bench needs a TPU device")
    return 2

  rng = np.random.default_rng(args.seed)
  gs = rng.multinomial(M, np.ones(E) / E).astype(np.int32)
  k0, k1, k2, k3 = jax.random.split(jax.random.key(args.seed), 4)
  put = lambda x: jax.device_put(x, dev)
  group_sizes = put(jnp.asarray(gs))
  wi9 = _tiles9(args.wi_tiles)
  wo9 = _tiles9(args.wo_tiles)
  wi_tile = None if wi9 is None else wi9[3:6]
  wo_tile = None if wo9 is None else wo9[3:6]

  def ti(mod, t):
    return mod.calculate_tiling if t is None else mod.TileSizes(tile_m=t[0], tile_k=t[1], tile_n=t[2])

  ok = True
  saved_ms = 0.0

  # ---- kernel level ----
  # (name, dout [M, K], W [G, N, K] (untransposed fwd weight), tile, group_offset)
  dout_wi = put(jax.random.normal(k0, (M, WI_N), jnp.bfloat16))
  w_wi = put((jax.random.normal(k1, (E // 2, EMB, WI_N), jnp.float32) * 0.02).astype(jnp.bfloat16))
  dout_wo = put(jax.random.normal(k2, (M, EMB), jnp.bfloat16))
  w_wo = put((jax.random.normal(k3, (E, MLP, EMB), jnp.float32) * 0.02).astype(jnp.bfloat16))
  cases = [
      ("wi_dlhs g64 k1408 n2816", dout_wi, w_wi, wi_tile, 0),
      ("wi_dlhs g64 off64 k1408 n2816", dout_wi, w_wi, wi_tile, E // 2),
      ("wo_dlhs g128 k2816 n704", dout_wo, w_wo, wo_tile, 0),
  ]
  for name, dout, w, t, off in cases:
    print(f"== {name}  tile={t or 'heuristic'}")
    try:
      goff = jnp.asarray(off, jnp.int32)

      def nt(d, w, goff=goff, t=t):
        return gmm_v2_trhs.gmm_v2(
            d,
            w,
            group_sizes,
            group_offset=goff,
            tile_info=ti(gmm_v2_trhs, t),
            preferred_element_type=jnp.bfloat16,
            transpose_rhs=True,
        )

      def ref_t(d, wt, goff=goff, t=t):
        return gmm_v2.gmm_v2(
            d, wt, group_sizes, group_offset=goff, tile_info=ti(gmm_v2, t), preferred_element_type=jnp.bfloat16
        )

      def ref(d, w):
        return ref_t(d, w.swapaxes(1, 2))

      o_nt, ms_nt = _timeit(jax.jit(nt), (dout, w), args.iters)
      wt = jax.block_until_ready(jax.jit(lambda w: w.swapaxes(1, 2))(w))
      o_rt, ms_rt = _timeit(jax.jit(ref_t), (dout, wt), args.iters)
      o_ref, ms_ref = _timeit(jax.jit(ref), (dout, w), args.iters)
      print(f"  ms: NT={ms_nt:.3f}  REF gmm only (pre-transposed)={ms_rt:.3f}  REF swapaxes+gmm={ms_ref:.3f}")
      saved_ms += ms_ref - ms_nt
      ok &= _cmp(name, o_nt, o_rt, args.tol)
      del o_nt, o_rt, o_ref
    except Exception:  # pylint: disable=broad-except
      traceback.print_exc()
      print(f"  [{name}] ERROR")
      return 2

  # ---- custom-VJP level (what the model runs) ----
  x_wi = put(jax.random.normal(k2, (M, EMB), jnp.bfloat16))
  x_wo = put(jax.random.normal(k1, (M, MLP), jnp.bfloat16))

  def ops_dx(tiles):
    def mk(trhs):
      def f(x, w, g):
        _, vjp = jax.vjp(
            lambda x: ops.gmm(
                x,
                w,
                group_sizes,
                preferred_element_type=jnp.bfloat16,
                tiling=tiles,
                group_offset=jnp.int32(0),
                use_tokamax_backend=True,
                use_gmm_v2=True,
                use_gmm_v2_heuristic_tiling=tiles is None,
                use_gmm_v2_transposed_rhs_dlhs=trhs,
            ),
            x,
        )
        return vjp(g)[0]

      return jax.jit(f)

    return mk

  for name, mk, a in [
      ("ops.gmm wi dlhs (vjp)", ops_dx(wi9), (x_wi, w_wi, dout_wi)),
      ("ops.gmm wo dlhs (vjp)", ops_dx(wo9), (x_wo, w_wo, dout_wo)),
  ]:
    print(f"== {name}")
    try:
      d1, ms1 = _timeit(mk(True), a, args.iters)
      d0, ms0 = _timeit(mk(False), a, args.iters)
      print(f"  ms (vjp dx): trhs={ms1:.3f}  swapaxes={ms0:.3f}")
      ok &= _cmp(name, d1, d0, args.tol)
    except Exception:  # pylint: disable=broad-except
      traceback.print_exc()
      print(f"  [{name}] ERROR")
      return 2

  print(f"kernel-level ms saved over the three dlhs cases: {saved_ms:.3f}")
  print("GMM_TRHS_BENCH", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
