"""Script to verify Qwix quantization interception via eval_shape.

python check_qwix_interception.py
"""

from absl import app
from absl import logging as absl_logging
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils

absl_logging.set_verbosity(absl_logging.DEBUG)


def main(argv):

  print("DEBUG 1: quantize_moe_gate=true (default: gate quantized)")

  config_args = [
      "",
      "src/maxtext/configs/base.yml",
      "model_name=deepseek3-671b",
      "quantization=fp8_full",
      "use_qwix_quantization=true",
      "scan_layers=true",
      "per_device_batch_size=1",
      "max_target_length=128",
      "quantize_moe_gate=true",
      "skip_jax_distributed_system=True",
  ] + (argv[1:] if len(argv) > 1 else [])
  config = pyconfig.initialize(config_args)
  model_creation_utils.create_nnx_abstract_model(config)

  print("DEBUG 2: quantize_moe_gate=false (gate unquantized)")

  config_args = [
      "",
      "src/maxtext/configs/base.yml",
      "model_name=deepseek3-671b",
      "quantization=fp8_full",
      "use_qwix_quantization=true",
      "scan_layers=true",
      "per_device_batch_size=1",
      "max_target_length=128",
      "quantize_moe_gate=false",
      "skip_jax_distributed_system=True",
  ] + (argv[1:] if len(argv) > 1 else [])
  config = pyconfig.initialize(config_args)
  model_creation_utils.create_nnx_abstract_model(config)


if __name__ == "__main__":
  app.run(main)
