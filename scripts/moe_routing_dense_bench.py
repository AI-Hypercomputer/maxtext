import time, statistics, jax, jax.numpy as jnp, numpy as np
T, K, E = int(__import__('sys').argv[1]), 16, 512
k = jax.random.split(jax.random.key(0), 3)
logits = jax.random.normal(k[0], (T, E), jnp.float32)
idx = jax.lax.top_k(logits, K)[1].astype(jnp.int32)
g = jax.random.normal(k[1], (T, K), jnp.float32)
def gather(x, i): return jnp.take_along_axis(x, i, axis=-1)
def dense(x, i):
  hit = i[..., None] == jax.lax.broadcasted_iota(i.dtype, (1, 1, E), 2)
  return jnp.sum(jnp.where(hit, x[:, None, :], 0.0), axis=-1)
def bc(i): return jnp.bincount(i.reshape(-1), length=E)
def bc_dense(i): return jnp.sum((i.reshape(-1)[:, None] == jnp.arange(E, dtype=i.dtype)).astype(jnp.int32), axis=0)
def rep(gs, n): return jnp.repeat(jnp.arange(gs.shape[0], dtype=jnp.int32), gs, total_repeat_length=n)
def rep_dense(gs, n): return jnp.sum((jnp.arange(n, dtype=jnp.int32)[:, None] >= jnp.cumsum(gs)[None, :-1]).astype(jnp.int32), axis=1)
def t(f, *a):
  f = jax.jit(f); jax.block_until_ready(f(*a)); ts = []
  for _ in range(20):
    t0 = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t0)
  return statistics.median(ts) * 1e3
fb = lambda h: (lambda x, i: jax.grad(lambda x: jnp.sum(h(x, i) * g))(x))
assert np.array_equal(np.asarray(gather(logits, idx)), np.asarray(dense(logits, idx)))
assert np.array_equal(np.asarray(fb(gather)(logits, idx)), np.asarray(fb(dense)(logits, idx)))
assert np.array_equal(np.asarray(bc(idx)), np.asarray(bc_dense(idx)))
gs = jnp.asarray(np.random.default_rng(0).multinomial(T * K, np.ones(8) / 8), jnp.int32); n = int(T * K * 1.125)
assert np.array_equal(np.asarray(rep(gs, n)), np.asarray(rep_dense(gs, n)))
print(f"T={T}: take_along fwd {t(gather, logits, idx):.3f} fwd+bwd {t(fb(gather), logits, idx):.3f} | dense fwd {t(dense, logits, idx):.3f} fwd+bwd {t(fb(dense), logits, idx):.3f} ms")
print(f"T={T}: bincount {t(bc, idx):.3f} | compare-sum {t(bc_dense, idx):.3f} ms;  repeat {t(lambda g_: rep(g_, n), gs):.3f} | compare-sum {t(lambda g_: rep_dense(g_, n), gs):.3f} ms")
