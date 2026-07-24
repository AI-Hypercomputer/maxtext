"""Manages asynchronous backups of JAX array states to pinned host memory."""

import contextlib
import logging
import threading
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter as BaseSnapshotter

_logger = logging.getLogger(__name__)

_identity_jit = jax.jit(lambda x: x)
_original_block_until_ready = jax.block_until_ready
_thread_local = threading.local()

def shard_single_device_arrays(pytree: Any) -> Any:
  """Converts single device arrays in a PyTree to NamedSharding fully replicated on the global mesh."""
  import jax
  from jax.sharding import NamedSharding, PartitionSpec, Mesh
  from jax.core import ShapedArray

  # 1. Find the global mesh by looking at the first Array with NamedSharding
  global_mesh = None
  for leaf in jax.tree.leaves(pytree):
    if isinstance(leaf, (jax.Array, ShapedArray)):
      sharding = getattr(leaf, 'sharding', None)
      if sharding is not None and isinstance(sharding, NamedSharding):
        global_mesh = sharding.mesh
        break
        
  if global_mesh is None:
    return pytree # No NamedSharding found to infer the global mesh
    
  new_sharding = NamedSharding(global_mesh, PartitionSpec())

  def _ensure_mesh_sharding(x: Any) -> Any:
    if isinstance(x, (jax.Array, ShapedArray, jax.ShapeDtypeStruct)):
      sharding = getattr(x, 'sharding', None)
      if sharding is not None and not isinstance(sharding, NamedSharding):
        if isinstance(x, jax.Array):
            return jax.device_put(x, new_sharding)
        else:
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=new_sharding)
    return x

  return jax.tree.map(_ensure_mesh_sharding, pytree)


def _custom_block_until_ready(x: Any) -> Any:
  """Thread-local override calling identity_jit(x).block_until_ready() during load."""
  if hasattr(_thread_local, "use_identity_jit") and _thread_local.use_identity_jit:
    try:
      return _identity_jit(x).block_until_ready()
    except Exception:
      return _original_block_until_ready(x)
  return _original_block_until_ready(x)


# Install thread-safe wrapper once at module import
jax.block_until_ready = _custom_block_until_ready


@contextlib.contextmanager
def _identity_jit_block_until_ready_context():
  """Context manager enabling thread-local identity_jit for block_until_ready."""
  _thread_local.use_identity_jit = True
  try:
    yield
  finally:
    _thread_local.use_identity_jit = False


class Snapshotter(BaseSnapshotter):
  """Extends Orbax Snapshotter using thread-safe identity_jit during load and supports non-uniform meshes."""

  def save(self, step: int, state: Any, **kwargs) -> bool:
    """Saves state after ensuring all single device arrays are sharded onto one-device-per-live-slice mesh."""
    state = shard_single_device_arrays(state)
    return super().save(step, state, **kwargs)

  def load(self, abstract_state: Any, *, reset_snapshot_state: bool = True) -> Any:
    """Move arrays from workers onto TPU devices supporting non-uniform meshes."""
    import jax
    from pathwaysutils.experimental import concatenate_by_mesh_axis, split_by_mesh_axis
    
    abstract_state = shard_single_device_arrays(abstract_state)
    
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def is_replica_active(arr, idx):
      try:
        import jax.random
        data = jax.random.key_data(arr) if (hasattr(arr, "dtype") and hasattr(arr, "shape") and jax.dtypes.issubdtype(arr.dtype, jax.dtypes.prng_key)) else arr
        jax.block_until_ready(data)
        return True
      except Exception as e:
        import sys
        print(f"[REPLICA DEAD] Replica {idx} failed with Exception: {e}", file=sys.stderr, flush=True)
        return False

    def get_active_pytree(x):
      mesh_axis_name = x.sharding.mesh.axis_names[self.replica_axis_index]
      try:
          import sys
          import jax.random
          data = jax.random.key_data(x) if (hasattr(x, "dtype") and hasattr(x, "shape") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)) else x
          all_replicas = split_by_mesh_axis.split_by_mesh_axis(data, mesh_axis_name)
      except Exception as e:
          print(f"FAILED SPLIT! x.shape={x.shape}, x.dtype={x.dtype}, type(x.sharding)={type(x.sharding)}, x.sharding={x.sharding}", flush=True)
          raise e
        
      import sys
      print(f"[DEBUG] get_active_pytree for axis={mesh_axis_name}: len(all_replicas)={len(all_replicas)}", file=sys.stderr, flush=True)
      active_replicas = []
      for i, replica in enumerate(all_replicas):
        if is_replica_active(replica, i):
          active_replicas.append(replica)
      if not active_replicas:
        print(f"[DEBUG] No active replicas for x! shape={x.shape}", file=sys.stderr, flush=True)
        raise RuntimeError("No active replicas found.")
      reconstructed = concatenate_by_mesh_axis.concatenate_by_mesh_axis(active_replicas, mesh_axis_name)
      if hasattr(x, "dtype") and hasattr(x, "shape") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
          return jax.random.wrap_key_data(reconstructed, dtype=x.dtype)
      return reconstructed

    def is_shardable_array(x):
      return isinstance(x, jax.Array)

    with _identity_jit_block_until_ready_context():
      pinned_state_for_map = pinned_state
      popped_keys = {}
      if isinstance(pinned_state, dict) and isinstance(abstract_state, dict):
        pinned_state_for_map = {}
        for k in pinned_state.keys():
          if k in abstract_state:
            pinned_state_for_map[k] = pinned_state[k]
          else:
            popped_keys[k] = pinned_state[k]

      # Map over pinned state to get active parts
      def _process_array(kp, x):
        if not is_shardable_array(x):
            return x
        try:
            return get_active_pytree(x)
        except RuntimeError as e:
            if "No active replicas" in str(e):
                import sys
                print(f"[FATAL] Array failed at path {jax.tree_util.keystr(kp)}: shape={x.shape}, sharding={x.sharding}", file=sys.stderr, flush=True)
            raise e

      active_state = jax.tree_util.tree_map_with_path(
          _process_array,
          pinned_state_for_map,
      )

      def _device_put_pinned(x, abs_x):
        if is_shardable_array(x):
          sharding = getattr(abs_x, 'sharding', None)
          if sharding is not None:
             sharding = sharding.with_memory_kind("pinned_host")
          return jax.device_put(x, sharding)
        return x

      host_target_state = jax.tree.map(_device_put_pinned, active_state, abstract_state)

      def _device_put_to_device(x, abs_x):
        if is_shardable_array(x):
          sharding = getattr(abs_x, 'sharding', None)
          if sharding is not None:
             sharding = sharding.with_memory_kind(None)
          return jax.device_put(x, sharding)
        return x

      restored_state = jax.tree.map(_device_put_to_device, host_target_state, abstract_state)
      jax.block_until_ready(restored_state)
      
      if popped_keys and isinstance(restored_state, dict):
        for k, v in popped_keys.items():
          restored_state[k] = v

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    self._queue.join()
