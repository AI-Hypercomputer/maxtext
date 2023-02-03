"""Create an Orbax async CheckpointManager."""

from etils import epath
import jax
import portpicker
from jax.experimental import multihost_utils
try: #TODO(migrate to updated API fully once it is universally available)
  from jax._src.cloud_tpu_init import get_metadata
except ImportError:
  from jax._src.clusters.cloud_tpu_cluster import get_metadata
from orbax import checkpoint
from orbax.checkpoint.async_checkpointer import AsyncCheckpointer
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions

def _multislice_distribute_initialize():
  """Calls jax.distribute.initialize() with appropriate multislice arguments."""

  def gen_local_ip():
    return get_metadata('worker-network-endpoints').split(',')[0]

  def gen_local_ip_nums():
    return [int(num) for num in gen_local_ip().split(':')[-1].split('.')]

  def get_coordinator_ip():
    local_ip_nums = jax.numpy.array(gen_local_ip_nums())
    coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
    coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
    return '.'.join(coordinator_ip_strings)

  port = multihost_utils.broadcast_one_to_all(jax.numpy.array(portpicker.pick_unused_port()))
  coordinator_address = get_coordinator_ip() + ':' + str(port)
  jax.distributed.initialize(coordinator_address=coordinator_address,
                             num_processes=jax.process_count(),
                             process_id=jax.process_index())

def create_orbax_checkpoint_manager(checkpoint_dir: str, enable_checkpointing: bool):
  """Returns an Orbax async CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    print("Checkpointing disabled, not creating checkpoint manager.")
    return None
  _multislice_distribute_initialize()
  p = epath.Path(checkpoint_dir)
  return CheckpointManager(p,
                           AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler()),
                           options = CheckpointManagerOptions(create=True))
