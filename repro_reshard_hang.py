"""Standalone repro for Pathways reshard hang.

Source mesh: trainer devices (one z-plane), FSDP sharding.
Target mesh: sampler devices (remaining z-planes), DP x TP sharding.

Matches train_rl.py + mk-next-unequal-mesh.yaml setup:
  trainer_devices_fraction=0.25 (32/128), trainer_z_index=1
  sampler: rollout_data_parallelism=24, rollout_tensor_parallelism=4

Usage:
  JAX_PLATFORMS=proxy,cpu \
  JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
  python3 repro_reshard_hang.py [--mode=pathways|tunix|tunix_chunked]

  --mode=pathways         (default) calls pathwaysutils.experimental.reshard directly
  --mode=tunix            calls tunix.rl.reshard.reshard_pytree
  --mode=tunix_chunked    mirrors maxtext_vllm_rollout.py: _reshard_in_chunks with reshard_pytree
  --chunk_size=N          chunk size for tunix_chunked mode (default: 4)
  --num_arrays=N          number of weight tensors to simulate (default: 8)
  --trainer_z_index=N     z-plane to use for trainer devices; "top" for max z (default: top)
  --trainer_fsdp=N        trainer mesh FSDP size (default: inferred from trainer device count)
  --sampler_dp=N          sampler mesh DP size (default: 24)
  --sampler_tp=N          sampler mesh TP size (default: 4)
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "proxy,cpu")

import sys
import time
import numpy as np
import pathwaysutils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def parse_args():
  args = {"mode": "pathways", "chunk_size": 4, "num_arrays": 8,
          "trainer_z_index": "top", "trainer_fsdp": None, "sampler_dp": 24, "sampler_tp": 4}
  for arg in sys.argv[1:]:
    if "=" in arg:
      k, v = arg.lstrip("-").split("=", 1)
      args[k] = v
  args["chunk_size"] = int(args["chunk_size"])
  args["num_arrays"] = int(args["num_arrays"])
  args["sampler_dp"] = int(args["sampler_dp"])
  args["sampler_tp"] = int(args["sampler_tp"])
  if args["trainer_fsdp"] is not None:
    args["trainer_fsdp"] = int(args["trainer_fsdp"])
  return args


def build_mesh(devices_list, shape, axis_names):
  arr = np.array(devices_list).reshape(shape)
  return Mesh(arr, axis_names)


def make_trainer_sampler_devices(devices, trainer_z_index="top"):
  devices_by_z = {}
  for d in devices:
    z = d.coords[2] if hasattr(d, "coords") and d.coords is not None else 0
    devices_by_z.setdefault(z, []).append(d)

  print(f"Devices by Z: { {z: len(devs) for z, devs in sorted(devices_by_z.items())} }")

  max_z = max(devices_by_z.keys())
  if str(trainer_z_index).lower() == "top":
    target_z = max_z
  elif str(trainer_z_index).lower() == "bottom":
    target_z = min(devices_by_z.keys())
  else:
    target_z = int(trainer_z_index)

  if target_z not in devices_by_z:
    raise ValueError(f"trainer_z_index={trainer_z_index} (z={target_z}) not found. Available: {sorted(devices_by_z.keys())}")

  trainer_devices = sorted(devices_by_z[target_z], key=lambda d: tuple(d.coords))
  sampler_devices = []
  for z in sorted(k for k in devices_by_z if k != target_z):
    sampler_devices.extend(sorted(devices_by_z[z], key=lambda d: tuple(d.coords)))

  print(f"Trainer z={target_z}, sampler z={sorted(k for k in devices_by_z if k != target_z)}")
  return trainer_devices, sampler_devices


def make_source_arrays(source_mesh, source_sharding, num_arrays, array_shape):
  """Create flat dict of source arrays mirroring weight tensors."""
  arrays = {}
  with source_mesh:
    for i in range(num_arrays):
      key = (f"layer_{i}", "weight")
      arrays[key] = jax.device_put(
          jnp.ones(array_shape, dtype=jnp.bfloat16), source_sharding
      )
  jax.block_until_ready(list(arrays.values()))
  return arrays


def make_spec_arrays(target_mesh, target_sharding, num_arrays, array_shape):
  """Create flat dict of target spec arrays (defines target sharding)."""
  arrays = {}
  with target_mesh:
    for i in range(num_arrays):
      key = (f"layer_{i}", "weight")
      arrays[key] = jax.device_put(
          jnp.ones(array_shape, dtype=jnp.bfloat16), target_sharding
      )
  jax.block_until_ready(list(arrays.values()))
  return arrays


def reshard_pathways(x, target_sharding):
  from pathwaysutils.experimental import reshard as experimental_reshard
  print("Calling pathwaysutils.experimental.reshard ...")
  result = experimental_reshard.reshard(
      x,
      target_sharding,
      donate=False,
      cache_resharding_plans=True,
  )
  result = jax.block_until_ready(result)
  return result


def reshard_tunix(x, target_sharding):
  from tunix.rl.reshard import reshard_pytree
  print("Calling tunix.rl.reshard.reshard_pytree ...")
  result = reshard_pytree(
      source=x,
      target=target_sharding,
      cache_plan=True,
      donate_input=False,
      use_experimental_pre_reshard=True,
  )
  result = jax.block_until_ready(result)
  return result


def reshard_tunix_chunked(src_flat, spec_flat, chunk_size):
  """Mirrors maxtext_vllm_rollout.py update_params exactly."""
  from tunix.generate.utils import _reshard_in_chunks
  from tunix.rl.reshard import reshard_pytree

  print(f"Calling _reshard_in_chunks (chunk_size={chunk_size}, n={len(src_flat)}) ...")
  start = time.time()
  resharded = _reshard_in_chunks(
      src_flat=src_flat,
      spec_flat=spec_flat,
      reshard_fn=reshard_pytree,
      chunk_size=chunk_size,
      delete_spec_buffers=True,
  )
  jax.block_until_ready(list(resharded.values()))
  print(f"_reshard_in_chunks done in {time.time() - start:.3f}s")
  return resharded


def main():
  args = parse_args()
  pathwaysutils.initialize()
  mode = args["mode"]
  chunk_size = args["chunk_size"]
  num_arrays = args["num_arrays"]

  trainer_z_index = args["trainer_z_index"]
  sampler_dp = args["sampler_dp"]
  sampler_tp = args["sampler_tp"]

  print(f"JAX version: {jax.__version__}")
  print(f"Reshard mode: {mode}, chunk_size={chunk_size}, num_arrays={num_arrays}")
  print(f"trainer_z_index={trainer_z_index}, sampler_dp={sampler_dp}, sampler_tp={sampler_tp}")

  devices = jax.devices()
  print(f"Total devices: {len(devices)}")

  trainer_devices, sampler_devices = make_trainer_sampler_devices(devices, trainer_z_index)
  print(f"Trainer devices: {len(trainer_devices)}, Sampler devices: {len(sampler_devices)}")

  trainer_fsdp = args["trainer_fsdp"] or len(trainer_devices)
  source_mesh = build_mesh(trainer_devices, (trainer_fsdp,), ("fsdp",))
  print(f"Source mesh shape: {source_mesh.shape}")

  target_mesh = build_mesh(sampler_devices, (sampler_dp, sampler_tp), ("dp", "tp"))
  print(f"Target mesh shape: {target_mesh.shape}")

  array_shape = (4096, 4096)
  source_sharding = NamedSharding(source_mesh, P("fsdp", None))
  target_sharding = NamedSharding(target_mesh, P(None, "tp"))

  if mode == "pathways":
    with source_mesh:
      x = jax.device_put(jnp.ones(array_shape, dtype=jnp.bfloat16), source_sharding)
    print(f"Source sharding: {x.sharding}")
    result = reshard_pathways(x, target_sharding)
    print(f"Result sharding: {result.sharding}")

  elif mode == "tunix":
    with source_mesh:
      x = jax.device_put(jnp.ones(array_shape, dtype=jnp.bfloat16), source_sharding)
    print(f"Source sharding: {x.sharding}")
    result = reshard_tunix(x, target_sharding)
    print(f"Result sharding: {result.sharding}")

  elif mode == "tunix_chunked":
    src_flat = make_source_arrays(source_mesh, source_sharding, num_arrays, array_shape)
    spec_flat = make_spec_arrays(target_mesh, target_sharding, num_arrays, array_shape)
    print(f"Source arrays: {len(src_flat)}, Source sharding: {next(iter(src_flat.values())).sharding}")
    print(f"Spec arrays: {len(spec_flat)}, Target sharding: {next(iter(spec_flat.values())).sharding}")
    result = reshard_tunix_chunked(src_flat, spec_flat, chunk_size)
    print(f"Resharded {len(result)} arrays. First result sharding: {next(iter(result.values())).sharding}")

  else:
    raise ValueError(f"Unknown mode: {mode!r}. Use pathways, tunix, or tunix_chunked")

  print("Done.")


if __name__ == "__main__":
  main()
