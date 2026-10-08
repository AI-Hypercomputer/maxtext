"""EP dispatch + combine (fwd+bwd) on the local v4: sender-major a2a + local permute vs expert-major a2a."""
import sys, time, statistics
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from maxtext.layers.moe import RoutedMoE
D = len(jax.devices()); E = 512; J = E // D; ROWS = int(sys.argv[1]); H = 768; BUF = int(ROWS * 1.125)
mesh = Mesh(np.array(jax.devices()), ("x",))
rng = np.random.default_rng(0)
gs = rng.multinomial(ROWS, np.ones(E) / E, size=D).astype(np.int32)  # [D, E] balanced-ish routing
x = jnp.asarray(rng.standard_normal((D * ROWS, H)), jnp.bfloat16)
w = jnp.asarray(rng.standard_normal((D * ROWS, H)), jnp.bfloat16)

def sender_major(x, ggs):
  sid = jax.lax.axis_index("x")
  g = ggs[sid]
  red = jnp.sum(ggs.reshape(D, D, J), axis=2)  # [s, d]
  io, ss, oo, rs = RoutedMoE.get_all_to_all_params(red, sid, D, ragged_buffer_factor=1.125, buffer_size=BUF)
  y = jax.lax.ragged_all_to_all(x, jnp.zeros((BUF, H), x.dtype), io, ss, oo, rs, axis_name="x")
  y, idx, lgs, _ = RoutedMoE.local_permute(y, ggs, J, sid, ragged_buffer_factor=1.125)
  y = y * 2  # stand-in for the experts
  y = y[jnp.argsort(idx)]
  io, ss, oo, rs = RoutedMoE.get_all_to_all_params(red, sid, D, ragged_buffer_factor=1.125, buffer_size=BUF, is_dispatch=False)
  return jax.lax.ragged_all_to_all(y, jnp.zeros_like(x), io, ss, oo, rs, axis_name="x")

def expert_major(x, ggs):
  sid = jax.lax.axis_index("x")
  io, ss, oo, rs, _ = RoutedMoE.get_expert_major_all_to_all_params(ggs, sid, J, BUF)
  y = jax.lax.ragged_all_to_all(x, jnp.zeros((BUF, H), x.dtype), io, ss, oo, rs, axis_name="x")
  y = y * 2
  io, ss, oo, rs, _ = RoutedMoE.get_expert_major_all_to_all_params(ggs, sid, J, BUF, is_dispatch=False)
  return jax.lax.ragged_all_to_all(y, jnp.zeros_like(x), io, ss, oo, rs, axis_name="x")

for name, fn in (("sender-major + permute", sender_major), ("expert-major", expert_major)):
  f = jax.shard_map(fn, mesh=mesh, in_specs=(P("x"), P()), out_specs=P("x"))
  loss = lambda x, g: jnp.sum(f(x, g).astype(jnp.float32) * w)
  fwd = jax.jit(f); fb = jax.jit(jax.grad(loss))
  def t(h):
    jax.block_until_ready(h(x, jnp.asarray(gs))); ts = []
    for _ in range(10):
      t0 = time.perf_counter(); jax.block_until_ready(h(x, jnp.asarray(gs))); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3
  out = np.asarray(fwd(x, jnp.asarray(gs)).astype(jnp.float32))
  print(f"rows/dev {ROWS} {name:24s} fwd {t(fwd):7.3f} ms  fwd+bwd {t(fb):7.3f} ms  checksum {out.sum():.1f}")
