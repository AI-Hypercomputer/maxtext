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

"""Runs a server with maxtext."""

from __future__ import annotations

import os
import sys
from typing import Any

import jax

from MaxText import pyconfig
from maxtext.common import gcloud_stub
from maxtext.inference.maxengine import maxengine_config

# _PORT = flags.DEFINE_integer('port', 9000, 'port to listen on')
# _THREADS = flags.DEFINE_integer(
#     'threads', 64, 'number of worker threads in thread pool'
# )
# _CONFIG = flags.DEFINE_string(
#     'config',
#     'MaxtextInterleavedServer',
#     'available servers',
# )


def _create_prefix_caching_config(config, config_lib_module):
  if not config.enable_prefix_caching:
    return None

  if not config.use_chunked_prefill:
    raise ValueError("Prefix caching requires chunked prefill.")

  return config_lib_module.PrefixCachingConfig(
      max_hbm_byte=config.prefix_caching_hbm_byte,
      max_dram_byte=config.prefix_caching_dram_byte,
  )


def main(config):
  # Obtain the jetstream helper modules (or stubs if appropriate).
  config_lib, _engine_api, *_ = gcloud_stub.jetstream()

  # If running decoupled and gcloud_stub returned lightweight stubs, skip
  # starting the real server. Use the explicit _IS_STUB marker when present.
  config_lib_is_stub = getattr(config_lib, "_IS_STUB", False)
  engine_api_is_stub = getattr(_engine_api, "_IS_STUB", False)
  if gcloud_stub.is_decoupled() and (config_lib_is_stub or engine_api_is_stub):
    raise RuntimeError(
        "JetStream helper modules are stubbed or DECOUPLE_GCLOUD=TRUE; server cannot be started in decoupled mode. "
        "Unset DECOUPLE_GCLOUD or install JetStream to run the server."
    )

  # Import the real server_lib now that it's known present.
  from jetstream.core import server_lib  # type: ignore  # pylint: disable=import-outside-toplevel
  import pathwaysutils  # pylint: disable=unused-import,import-outside-toplevel

  pathwaysutils.initialize()

  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = maxengine_config.get_server_config(config.inference_server, config)

  metrics_server_config: Any | None = None
  if config.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(port=config.prometheus_port)

  # We separate credential from run so that we can unit test it with
  # local credentials.
  # TODO: Add grpc credentials for OSS.
  # pylint: disable=unexpected-keyword-arg
  jetstream_server = server_lib.run(
      threads=256,
      port=9000,
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
      enable_jax_profiler=config.enable_jax_profiler if config.enable_jax_profiler else False,
      jax_profiler_port=config.jax_profiler_port if config.jax_profiler_port else 9999,
      enable_model_warmup=config.enable_model_warmup if config.enable_model_warmup else False,
      lora_input_adapters_path=config.lora_input_adapters_path,
      multi_sampling=config.multi_sampling if config.multi_sampling else False,
      prefix_caching_config=_create_prefix_caching_config(config, config_lib),
  )
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  cfg = pyconfig.initialize(sys.argv)
  main(cfg)
