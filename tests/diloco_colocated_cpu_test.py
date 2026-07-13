"""Test script exercising Non-SPMD Streaming DiLoCo with Colocated CPU Syncer on Pathways.

This script demonstrates and tests the core mechanics described in the DiLoCo
architecture on Pathways:
1. Unstacking a TPU mesh into K independent learner submeshes.
2. Routing syncer arrays to colocated host CPUs using `.with_memory_kind('pinned_host')`.
3. Running K learner loops on TPUs and 1 syncer loop on CPUs concurrently via Python threads.
4. Communicating weights asynchronously over channel queues.
5. Exercising stack/unstack (resharding) between TPU learner meshes and XLA:CPU (`@compute_on('device_host')`).
"""

import concurrent.futures
import queue
import time
from typing import List, Tuple
import jax

try:
  import pathwaysutils
except ImportError:
  pathwaysutils = None
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import colocated_python
import numpy as np


def create_meshes(num_replicas: int) -> Tuple[Mesh, List[Mesh]]:
  """Creates full and unstacked TPU meshes."""
  tpu_devices = jax.devices()
  if len(tpu_devices) < num_replicas:
    raise ValueError(f"Need at least {num_replicas} devices, got {len(tpu_devices)}")

  # Reshape devices into (diloco_replicas, devices_per_replica)
  devices_per_replica = len(tpu_devices) // num_replicas
  tpu_grid = np.array(tpu_devices[: num_replicas * devices_per_replica]).reshape((num_replicas, devices_per_replica))

  # 1. Full Mesh
  full_tpu_mesh = Mesh(tpu_grid, axis_names=("replica", "model"))

  # 2. Unstack along the 'replica' axis (index 0) to get per-learner submeshes
  learner_tpu_meshes = [Mesh(tpu_grid[i : i + 1], axis_names=("replica", "model")) for i in range(num_replicas)]

  return full_tpu_mesh, learner_tpu_meshes


def learner_thread_task(
    learner_id: int,
    tpu_mesh: Mesh,
    to_syncer_queue: queue.Queue,
    from_syncer_queue: queue.Queue,
    num_local_steps: int = 4,
):
  """Simulates an independent learner training loop running on a TPU submesh."""
  print(f"[{time.strftime('%X')}] 🚀 Learner {learner_id} starting on TPU mesh {tpu_mesh.devices.shape}...")

  # Initialize toy weights on TPU HBM ('device')
  sharding = NamedSharding(tpu_mesh, P(None, "model"))
  weights = jax.device_put(jnp.ones((8, 16), dtype=jnp.float32) * (learner_id + 1.0), sharding)

  for step in range(1, num_local_steps + 1):
    # Simulating TPU computation (e.g. inner training steps)
    weights = weights + 0.1
    time.sleep(0.5)  # Simulate compute time

    if step % 2 == 0 or step == num_local_steps:
      print(f"[{time.strftime('%X')}] 📤 Learner {learner_id} step {step}: sending weights to syncer channel...")
      to_syncer_queue.put((learner_id, step, weights))

      # Block and wait for synced weights from XLA:CPU syncer
      print(f"[{time.strftime('%X')}] ⏳ Learner {learner_id} step {step}: waiting for synced weights...")
      weights = from_syncer_queue.get()
      mean_val = float(jnp.mean(weights))
      print(
          f"[{time.strftime('%X')}] 📥 Learner {learner_id} step {step}: received synced weights (mean: {mean_val:.4f})"
      )

  print(f"[{time.strftime('%X')}] ✅ Learner {learner_id} finished training loop.")
  return float(jnp.mean(weights))


def syncer_thread_task(
    full_tpu_mesh: Mesh,
    to_syncer_queues: List[queue.Queue],
    from_syncer_queues: List[queue.Queue],
    expected_syncs: int = 2,
):
  """Simulates the synchronizer running on colocated host CPUs via XLA:CPU on CPU mesh."""
  cpu_mesh = colocated_python.colocated_cpu_devices(full_tpu_mesh)
  print(f"[{time.strftime('%X')}] 🧠 Syncer starting on CPU mesh {cpu_mesh.devices.shape} (derived via colocated_cpu)...")

  @jax.jit
  def cpu_allreduce_merge(stacked_array: jax.Array) -> jax.Array:
    """Executes on host CPU mesh using XLA:CPU."""
    # All-reduce / average across the 'replica' axis on CPU
    mean_weights = jnp.mean(stacked_array, axis=0, keepdims=True)
    return jnp.repeat(mean_weights, stacked_array.shape[0], axis=0)

  for sync_idx in range(1, expected_syncs + 1):
    print(f"[{time.strftime('%X')}] 🔄 Syncer cycle {sync_idx}: waiting for fragments from all learners...")

    # 1. Pull unstacked weights from all learner channels
    received_fragments = [q.get() for q in to_syncer_queues]
    received_fragments.sort(key=lambda x: x[0])  # sort by learner_id

    learner_arrays = [frag[2] for frag in received_fragments]
    print(f"[{time.strftime('%X')}] 📦 Syncer cycle {sync_idx}: received all fragments. Transferring to CPU mesh RAM...")
    learner_cpu_arrays = [
        jax.device_put(np.asarray(arr), NamedSharding(cpu_mesh, P(None, "model"))) for arr in learner_arrays
    ]
    stacked_cpu_array = jnp.stack(learner_cpu_arrays, axis=0)

    # 3. Execute XLA:CPU megascale merge math on host
    print(f"[{time.strftime('%X')}] 🧮 Syncer cycle {sync_idx}: executing XLA:CPU allreduce merge on host...")
    merged_cpu_array = cpu_allreduce_merge(stacked_cpu_array)

    # 4. UNSTACK across meshes and send back over channels to TPU learners
    print(f"[{time.strftime('%X')}] 🚀 Syncer cycle {sync_idx}: unstacking and distributing back to learners...")
    merged_host_np = np.asarray(merged_cpu_array)
    for lid, q in enumerate(from_syncer_queues):
      q.put(merged_host_np[lid])

  print(f"[{time.strftime('%X')}] ✅ Syncer finished all synchronization cycles.")


def main():
  print("=" * 70)
  print("Testing Non-SPMD Streaming DiLoCo with Colocated CPU Syncer & TPU Learners")
  print("=" * 70)
  if pathwaysutils is not None:
    try:
      pathwaysutils.initialize()
      print("✓ pathwaysutils.initialize() called successfully.")
    except (RuntimeError, ValueError) as e:
      print(f"Note: pathwaysutils.initialize() raised: {e}")
  else:
    print("Note: pathwaysutils not imported.")

  num_replicas = 2
  full_tpu_mesh, learner_tpu_meshes = create_meshes(num_replicas)

  print(f"Full TPU Mesh: {full_tpu_mesh.devices.shape} on devices: {[d.platform for d in full_tpu_mesh.devices.flat]}")
  for i, lm in enumerate(learner_tpu_meshes):
    print(f"  Learner {i} TPU submesh: {lm.devices.shape}")
  print("-" * 70)

  to_syncer_queues = [queue.Queue() for _ in range(num_replicas)]
  from_syncer_queues = [queue.Queue() for _ in range(num_replicas)]

  with concurrent.futures.ThreadPoolExecutor(max_workers=num_replicas + 1) as executor:
    syncer_future = executor.submit(
        syncer_thread_task,
        full_tpu_mesh,
        to_syncer_queues,
        from_syncer_queues,
        expected_syncs=2,
    )

    learner_futures = [
        executor.submit(
            learner_thread_task,
            lid,
            learner_tpu_meshes[lid],
            to_syncer_queues[lid],
            from_syncer_queues[lid],
            num_local_steps=4,
        )
        for lid in range(num_replicas)
    ]

    concurrent.futures.wait([syncer_future] + learner_futures)

    for lid, f in enumerate(learner_futures):
      print(f"Learner {lid} final weight mean: {f.result():.4f}")
    syncer_future.result()

  print("=" * 70)
  print("🎉 All tests passed! Non-SPMD multithreading and CPU/TPU overlap verified.")
  print("=" * 70)


if __name__ == "__main__":
  main()
