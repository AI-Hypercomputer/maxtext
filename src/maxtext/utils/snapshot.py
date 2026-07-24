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

def _ensure_mesh_sharding(x: Any) -> Any:
  """Shards single device arrays onto a mesh constructed with one device per live slice."""
  import jax
  from jax.sharding import NamedSharding, PartitionSpec, Mesh
  from jax.core import ShapedArray
  if isinstance(x, (jax.Array, ShapedArray, jax.ShapeDtypeStruct)):
    sharding = getattr(x, 'sharding', None)
    if sharding is not None and type(sharding).__name__ in ("SingleDeviceSharding", "PmapSharding"):
      from maxtext.utils import elastic_utils
      import numpy as np
      
      first_devices = []
      if elastic_utils.elastic_manager is not None:
          for slice_idx in elastic_utils.elastic_manager.active_slice_indices:
              devices = elastic_utils.elastic_manager.slice_to_devices.get(slice_idx, None)
              if devices:
                  first_devices.append(devices[0])
      
      if not first_devices:
          first_devices = [jax.devices()[0]]
          
      mesh = Mesh(np.array(first_devices), ("replica",))
      new_sharding = NamedSharding(mesh, PartitionSpec())
      
      if isinstance(x, jax.Array):
          return jax.device_put(x, new_sharding)
      else:
          return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=new_sharding)
  return x


def shard_single_device_arrays(pytree: Any) -> Any:
  """Converts single device arrays in a PyTree to NamedSharding on a one-device-per-live-slice mesh."""
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
    
    abstract_state = shard_single_device_arrays(abstract_state)
    
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def _rebuild_pinned_array(x, abs_x):
      if not isinstance(x, jax.Array):
        return x
        
      # Target sharding for this array in pinned host memory
      target_sharding = getattr(abs_x, 'sharding', None)
      if target_sharding is None:
        return x
        
      target_sharding = target_sharding.with_memory_kind("pinned_host")
      target_devices = target_sharding.mesh.devices.flat
      
      # Build dictionary from logically matched devices
      x_shards_by_device = {s.device.id: s for s in x.global_shards}
      
      buffers = []
      for d in target_devices:
        shard = x_shards_by_device.get(d.id, None)
        if shard is None:
            raise RuntimeError(f"Target device {d.id} not found in x's old shards!")
            
        try:
            data = shard.data
            jax.block_until_ready(data)
            buffers.append(data)
        except Exception as e:
            raise RuntimeError(f"Data loss on target device {d.id} during array reconstruction: {e}")
            
      return jax.make_array_from_single_device_arrays(x.shape, target_sharding, buffers)

    def is_shardable_array(x):
      return isinstance(x, jax.Array)

    with _identity_jit_block_until_ready_context():
      # Re-shard on host directly from live shards matching the generic abstract_state
      host_target_state = jax.tree.map(_rebuild_pinned_array, pinned_state, abstract_state)

      def _device_put_to_device(x, abs_x):
        if is_shardable_array(x):
          sharding = getattr(abs_x, 'sharding', None)
          if sharding is not None:
             sharding = sharding.with_memory_kind(None)
          return jax.device_put(x, sharding)
        return x

      restored_state = jax.tree.map(_device_put_to_device, host_target_state, abstract_state)
      jax.block_until_ready(restored_state)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    self._queue.join()
