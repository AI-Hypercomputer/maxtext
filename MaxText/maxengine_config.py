# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Configure MaxText For JetStream'''
import functools

import jax
from typing import Any, Type

from jetstream.core import config_lib
from jetstream.engine import engine_api
import maxengine

def create_maxengine(devices: config_lib.Devices, config: Any) -> engine_api.Engine:
  del devices
  return maxengine.MaxEngine(config)


def get_server_config(config_str: str, config: Any) -> Type[config_lib.ServerConfig]:
  ''' Gets the Server Config Required by JetStream '''
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
