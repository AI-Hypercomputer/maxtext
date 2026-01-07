# Copyright 2023–2026 Google LLC
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

import os
import sys

import pathwaysutils  # pylint: disable=unused-import

from jetstream.core import server_lib, config_lib

import jax

from MaxText import pyconfig
from MaxText import maxengine_config

# _PORT = flags.DEFINE_integer('port', 9000, 'port to listen on')
# _THREADS = flags.DEFINE_integer(
#     'threads', 64, 'number of worker threads in thread pool'
# )
# _CONFIG = flags.DEFINE_string(
#     'config',
#     'MaxtextInterleavedServer',
#     'available servers',
# )


def _create_prefix_caching_config(config) -> config_lib.PrefixCachingConfig | None:
  if not config.enable_prefix_caching:
    return None

  if not config.use_chunked_prefill:
    raise ValueError("Prefix caching requires chunked prefill.")

  return config_lib.PrefixCachingConfig(
      max_hbm_byte=config.prefix_caching_hbm_byte,
      max_dram_byte=config.prefix_caching_dram_byte,
  )


def main(config):
  pathwaysutils.initialize()

  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = maxengine_config.get_server_config(config.inference_server, config)

  metrics_server_config: config_lib.MetricsServerConfig | None = None
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
      prefix_caching_config=_create_prefix_caching_config(config),
  )
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  cfg = pyconfig.initialize(sys.argv)
  main(cfg)
