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

""" A wrapper file for easily calculating training TFLOPs. """

from maxtext.configs import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from maxtext.utils.maxtext_utils import calculate_tflops_training_per_device
import os
from typing import Sequence, cast
from absl import app


def main(argv: Sequence[str]):
  """
  Calculates and prints TFLOPs using command
  Example invocation:
  python3 -m MaxText.get_flops model_name=llama2-7b
  """
  pyconfig_argv = [argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")] + cast(list[str], argv[1:])
  config = pyconfig.initialize(pyconfig_argv)
  tflops, _, _ = calculate_tflops_training_per_device(config, log=False)
  print(f"Total TFLOPs per device per step: {tflops}")


if __name__ == "__main__":
  app.run(main)
