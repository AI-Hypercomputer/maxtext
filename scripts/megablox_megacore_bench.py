"""megablox 3-GMM expert block at the EP 64 per-device shape on one v4 megacore device, any tiling.

  python scripts/megablox_megacore_bench.py ROWS_PER_EXPERT WI_TILES WO_TILES
  e.g. 9216 512,768,896,512,1792,384,512,768,896 512,1792,384,512,768,896,512,896,384  (megacore split)
       9216 512,768,1792,512,1792,768,512,768,896 512,1792,768,512,768,1792,512,896,768  (one n-tile)

Each tuple is (fwd m,k,n, dlhs m,k,n, drhs m,k,n) as moe.py builds it. Megablox marks only its n-tile grid
axis parallel, so with one n-tile a GMM runs on one of v4's two megacore TensorCores."""
import sys, time, statistics, jax, jax.numpy as jnp
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
from maxtext.kernels.megablox import ops as mblx
E, rpe, K, N = 8, int(sys.argv[1]), 768, 1792
wi_t = tuple(int(v) for v in sys.argv[2].split(","))
wo_t = tuple(int(v) for v in sys.argv[3].split(","))
R = E * rpe
k = jax.random.split(jax.random.key(0), 5)
x = (jax.random.normal(k[0], (R, K)) * 0.05).astype(jnp.bfloat16)
wg = (jax.random.normal(k[1], (E, K, N)) * 0.05).astype(jnp.bfloat16)
wu = (jax.random.normal(k[2], (E, K, N)) * 0.05).astype(jnp.bfloat16)
wd = (jax.random.normal(k[3], (E, N, K)) * 0.05).astype(jnp.bfloat16)
gs = jnp.full((E,), rpe, jnp.int32)
def block(x, wg, wu, wd):
  g = mblx.gmm(x, wg, gs, preferred_element_type=jnp.bfloat16, tiling=wi_t)
  u = mblx.gmm(x, wu, gs, preferred_element_type=jnp.bfloat16, tiling=wi_t)
  return mblx.gmm(jax.nn.silu(g) * u, wd, gs, preferred_element_type=jnp.bfloat16, tiling=wo_t)
w = jax.random.normal(k[4], (R, K)).astype(jnp.bfloat16)
fwd = jax.jit(block)
fb = jax.jit(jax.grad(lambda *a: jnp.sum(block(*a).astype(jnp.float32) * w), argnums=(0, 1, 2, 3)))
def t(f):
  jax.block_until_ready(f(x, wg, wu, wd)); ts = []
  for _ in range(10):
    t0 = time.perf_counter(); jax.block_until_ready(f(x, wg, wu, wd)); ts.append(time.perf_counter() - t0)
  return statistics.median(ts) * 1e3
fl = 2 * R * K * N * 3
a, b = t(fwd), t(fb)
print(f"wi={sys.argv[2]} wo={sys.argv[3]}: fwd {a:.3f} ms ({fl/a/1e9:.1f} TF/s)  fwd+bwd {b:.3f} ms ({3*fl/b/1e9:.1f} TF/s)")
