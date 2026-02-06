import jax
import pathwaysutils
from pathwaysutils.elastic import manager


elastic_manager: manager.Manager | None = None


def devices():
  device_list = jax.devices()

  if pathwaysutils.is_pathways_backend_used() and elastic_manager is not None:
      return [
          d for d in device_list
          if d.slice_index in elastic_manager.active_slice_indices
      ]
  else:
    return device_list
