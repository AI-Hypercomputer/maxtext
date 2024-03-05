"""Runs a server with maxtext."""

from typing import Sequence

from absl import app
from absl import flags
import jax
import os
import sys
import pyconfig # change the path to maxtext pyconfig

from jetstream.core.implementations.maxtext import config as maxtext_config
from jetstream.core import server_lib
from jax.experimental.compilation_cache import compilation_cache as cc

# _PORT = flags.DEFINE_integer('port', 9000, 'port to listen on')
# _THREADS = flags.DEFINE_integer(
#     'threads', 64, 'number of worker threads in thread pool'
# )
# _CONFIG = flags.DEFINE_string(
#     'config',
#     'MaxtextInterleavedServer',
#     'available servers',
# )

#  python -m maxtext.server configs/base.yml assets_path=gs://maxtext-gamma/gamma per_device_batch_size=1 run_name=serving max_prefill_predict_length=8 max_target_length=16 dataset_path=gs://maxtext-dataset async_checkpointing=false scan_layers=false model_name=gamma-2b attention=recommended

def main(config):
  print("==========")
  print(config)
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = maxtext_config.get_server_config('MaxtextInterleavedServer', config)
  # We separate credential from run so that we can unit test it with
  # local credentials.
  # TODO: Add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=64,
      port=9000,
      config=server_config,
      devices=devices,
  )
  jetstream_server.wait_for_termination()


if __name__ == '__main__':
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  cc.set_cache_dir(os.path.expanduser(config.jax_cache_dir))
  main(config)
