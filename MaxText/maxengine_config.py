
from typing import Type
import functools

import dataclasses
import jax
from typing import Any, Callable, Tuple, Type, Union, cast

from jetstream.core import config_lib
from jetstream.engine import engine_api
import maxengine

def create_maxengine(devices: config_lib.Devices, config: Any) -> engine_api.Engine:
  del devices
  return maxengine.MaxEngine(config)


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
            create_maxengine,
            config=config
          ),
        )
      )
    case _:
      raise NotImplementedError
  return server_config
