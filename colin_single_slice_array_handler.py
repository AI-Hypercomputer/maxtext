import asyncio
import socket
from typing import Any, Optional, Sequence
from typing import cast
from etils import epath
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization
import numpy as np
import orbax.checkpoint as ocp
import portpicker
import time

def _is_host_for_slice(idx: int) -> bool:
  #return True
  return jax.local_devices()[0].slice_index == idx

def get_jax_slice_zero_devices(devices=jax.devices()):
    slice_zero_devices = []
    for device in devices:
        if device.slice_index == 0:
            slice_zero_devices.append(device)
    return slice_zero_devices

class SingleSliceArrayHandler(ocp.type_handlers.ArrayHandler):

  async def deserialize(
      self,
      infos: Sequence[ocp.type_handlers.ParamInfo],
      args: Optional[Sequence[ocp.RestoreArgs]] = None,
  ) -> Sequence[jax.Array]:
    """See superclass documentation.

    Args:
      infos: ParamInfo.
      args: must be of type `ArrayRestoreArgs`.

    Returns:
      The deserialized parameter.

    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.sharding` is not provided or `args.mesh` and
      `args.mesh_axes` are not provided.
    """)
    if args is None:
      raise ValueError('Must provide ArrayRestoreArgs to restore as jax.Array.')
    ocp.type_handlers.check_input_arguments(infos, args)
    deserialize_ops = []
    for info, arg in zip(infos, args):
      arg = cast(ocp.ArrayRestoreArgs, arg)
      if isinstance(arg, ocp.ArrayRestoreArgs) and arg.sharding is not None:
        sharding = arg.sharding
        if not isinstance(sharding, jax.sharding.NamedSharding):
          raise ValueError(
              "`sharding` must be of type `jax.sharding.NamedSharding`."
          )
      else:
        raise ValueError(
            'Sharding of jax.Array cannot be None. Provide `mesh`'
            ' and `mesh_axes` OR `sharding`'
        )
      if not info.is_ocdbt_checkpoint:
        await ocp.type_handlers._assert_parameter_files_exist(  # pylint: disable=protected-access
            info.path, self._metadata_key
        )
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = ocp.type_handlers._use_ocdbt_for_restore(  # pylint: disable=protected-access
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = ocp.type_handlers._get_cast_tspec_deserialize(tspec, arg)  # pylint: disable=protected-access

      if _is_host_for_slice(0):
        slice_zero_devices = np.asarray(get_jax_slice_zero_devices()).reshape(
            sharding.mesh.devices.shape
        )
        single_slice_mesh = jax.sharding.Mesh(
            slice_zero_devices, sharding.mesh.axis_names
        )
        single_slice_sharding = jax.sharding.NamedSharding(
            single_slice_mesh, sharding.spec
        )
        deserialize_ops += [
            serialization.async_deserialize(
                single_slice_sharding,
                tspec,
                global_shape=arg.global_shape
                if hasattr(arg, "global_shape")
                else None,
                byte_limiter=info.byte_limiter,
                context=self._ts_context,
            )
        ]
        print("Finished slice 0 tasks", flush=True)
    print("Starting deserialization", flush=True)
    deserialized = await asyncio.gather(*deserialize_ops)
    print("Finished deserialization", flush=True)
    # deserialized = _broadcast_one_to_all(deserialized)
    return deserialized