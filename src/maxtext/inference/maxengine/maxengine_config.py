# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configure MaxText For JetStream"""
import functools
from typing import Any

import jax

from maxtext.common.gcloud_stub import jetstream, is_decoupled
from maxtext.inference.maxengine import maxengine

config_lib, engine_api, _token_utils, _tokenizer_api, _token_params_ns = jetstream()



# TODO: merge it with the above create_maxengine().
def create_exp_maxengine(devices: Any, config: Any):
  if is_decoupled():
    return maxengine.MaxEngine(config)
  return maxengine.MaxEngine(config=config, devices=devices)


def create_maxengine(devices: Any, config: Any) -> engine_api.Engine:
  del devices
  return maxengine.MaxEngine(config)


def get_server_config(config_str: str, config: Any):
  """Gets the Server Config Required by JetStream."""
  # If Jetstream is stub and decoupled, return a minimal stub server config and log the no-op.
  config_lib_is_stub = getattr(config_lib, "_IS_STUB", False)
  engine_api_is_stub = getattr(engine_api, "_IS_STUB", False)
  if is_decoupled() and (config_lib_is_stub or engine_api_is_stub):
    raise RuntimeError("[DECOUPLED NO-OP] jetstream.config_lib is stubbed; returning minimal server config.")
  # Not decoupled and no Jetstream found -> allow the later code to raise.
  match config_str:
    case "MaxtextInterleavedServer":
      server_config = config_lib.ServerConfig(
          prefill_slices=(),
          generate_slices=(),
          interleaved_slices=("tpu=" + str(jax.device_count()),),
          prefill_engine_create_fns=(),
          generate_engine_create_fns=(),
          interleaved_engine_create_fns=(functools.partial(create_maxengine, config=config),),
      )
    case "ExperimentalMaxtextDisaggregatedServer":
      # ExperimentalMaxtextDisaggregatedServer is still under development.
      # Its dependencies IFRT Proxy and other components are not publicly available
      # either.
      server_config = config_lib.ServerConfig(
          prefill_slices=(config.prefill_slice,),
          generate_slices=(config.generate_slice,),
          interleaved_slices=(),
          prefill_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          generate_engine_create_fns=(functools.partial(create_exp_maxengine, config=config),),
          interleaved_engine_create_fns=(),
      )
    case _:
      raise NotImplementedError
  return server_config
