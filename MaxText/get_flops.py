# SPDX-License-Identifier: Apache-2.0

""" A wrapper file for easily calculating training TFLOPs. """

from MaxText.maxtext_utils import calculate_tflops_training_per_device
from MaxText import pyconfig
from MaxText.globals import PKG_DIR
import os
from typing import Sequence
from absl import app

def main(argv: Sequence[str]):
  """
  Calculates and prints TFLOPs using command
  Example invocation:
  python3 -m MaxText.get_flops model_name=llama2-7b
  """
  pyconfig_argv = [argv[0], os.path.join(PKG_DIR, "configs", "base.yml")] + argv[1:]
  config = pyconfig.initialize(pyconfig_argv)
  tflops, _, _ = calculate_tflops_training_per_device(config, log=False)
  print(f"Total TFLOPs per device per step: {tflops}")

if __name__ == "__main__":
  app.run(main)
