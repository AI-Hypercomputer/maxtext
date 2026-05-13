"""Minimal repro for Pathways reshard hang with compound tuple sharding spec.

Root cause: reshard_pytree hangs when the target PartitionSpec uses a compound
tuple axis, e.g. P(('dp', 'tp'), None), even with simple meshes.

Usage:
  JAX_PLATFORMS=proxy,cpu \
  JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
  python3 -u repro_reshard_hang.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "proxy,cpu")

import time
import numpy as np
import pathwaysutils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def log(msg):
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_mesh(devices, shape, axes):
  return Mesh(np.array(devices).reshape(shape), axes)


def reshard(label, src, dst):
  from tunix.rl.reshard import reshard_pytree
  log(f"{label}: dispatching reshard_pytree (use_experimental_pre_reshard=False) ...")
  t = time.time()
  result = reshard_pytree(source=src, target=dst, use_experimental_pre_reshard=False)
  log(f"{label}: dispatch returned in {time.time()-t:.3f}s, block_until_ready ...")
  jax.block_until_ready(result)
  log(f"{label}: DONE in {time.time()-t:.3f}s")
  return result


def main():
  pathwaysutils.initialize()
  log(f"JAX {jax.__version__}, devices: {len(jax.devices())}")

  devices = jax.devices()
  # Split: first 32 = src, remaining 96 = dst (matches 0.25/0.75 unequal split)
  src_devices = devices[:32]
  dst_devices = devices[32:]

  src_mesh = make_mesh(src_devices, (32,), ("fsdp",))
  dst_mesh = make_mesh(dst_devices, (24, 4), ("dp", "tp"))

  # Array shape: divisible by fsdp=32 on dim1, by dp*tp=96 on dim0
  shape = (9600, 2048)

  src_sharding = NamedSharding(src_mesh, P(None, "fsdp"))

  log(f"Array shape: {shape}")
  log(f"Src mesh: {dict(src_mesh.shape)}, spec: P(None, 'fsdp')")
  log(f"Dst mesh: {dict(dst_mesh.shape)}")

  # Control: single axis spec P('dp', None) — expected to work
  log("--- Control: dst spec P('dp', None) [should pass] ---")
  dst_ctrl = NamedSharding(dst_mesh, P("dp", None))
  with src_mesh:
    src_ctrl = {"w": jax.device_put(jnp.ones(shape, jnp.bfloat16), src_sharding)}
  with dst_mesh:
    spec_ctrl = {"w": jax.device_put(jnp.ones(shape, jnp.bfloat16), dst_ctrl)}
  jax.block_until_ready([src_ctrl["w"], spec_ctrl["w"]])
  reshard("control P('dp',None)", src_ctrl, spec_ctrl)

  # Repro: compound tuple spec P(('dp', 'tp'), None) — hangs
  log("--- Repro: dst spec P(('dp','tp'), None) [expected to hang] ---")
  dst_compound = NamedSharding(dst_mesh, P(("dp", "tp"), None))
  with src_mesh:
    src_repro = {"w": jax.device_put(jnp.ones(shape, jnp.bfloat16), src_sharding)}
  with dst_mesh:
    spec_repro = {"w": jax.device_put(jnp.ones(shape, jnp.bfloat16), dst_compound)}
  jax.block_until_ready([src_repro["w"], spec_repro["w"]])
  reshard("repro P(('dp','tp'),None)", src_repro, spec_repro)

  log("Done.")


if __name__ == "__main__":
  main()
