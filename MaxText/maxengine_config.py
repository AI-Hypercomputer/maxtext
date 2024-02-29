"""Config for mock."""

from typing import Type
import functools

import dataclasses
import jax
from typing import Any, Callable, Tuple, Type, Union, cast

from jetstream.core import config_lib
from jetstream.engine import engine_api
import myengine # change the path to maxtext myengine

def create_myengine(devices: config_lib.Devices, config: Any) -> engine_api.Engine:
  del devices
  return myengine.TestEngine(config)


def get_server_config(config_str: str, config: Any) -> Type[config_lib.ServerConfig]:
  match config_str:
    case 'MaxtextInterleavedServer':
      server_config = config_lib.ServerConfig(
        prefill_slices = (),
        generate_slices = (),
        interleaved_slices = ('tpu='+str(jax.device_count()),),
        prefill_engine_create_fns = (),
        generate_engine_create_fns = (),
        interleaved_engine_create_fns = (functools.partial(
            create_myengine,
            config=config
          ),
        )
      )
    case 'InterleavedCPUTestServer':
      server_config = config_lib.InterleavedCPUTestServer
    case 'CPUTestServer':
      server_config = config_lib.CPUTestServer
    case _:
      raise NotImplementedError
  return server_config
