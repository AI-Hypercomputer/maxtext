# Copyright 2026 Google LLC
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

"""Experimental SFT training runner for Omni (Gemma 3 + Qwen 3).

Executes native MaxText SFT training with multimodal data processing and
trainable parameter masking to fine-tune the custom MLP vision projector.

Example usage:
  python3 -m maxtext.experimental.omni_poc.train_sft_omni \
    src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
    load_parameters_path=gs://YOUR_BUCKET/path/to/checkpoint/items \
    base_output_directory=gs://YOUR_BUCKET/output_directory \
    run_name=my_omni_sft_run
"""

import maxtext
# Eagerly initialize core MaxText C++ and model dependencies before train.py
_ = (maxtext.Mesh, maxtext.pyconfig, maxtext.models, maxtext.model_creation_utils)

from typing import Sequence
from absl import app
from maxtext.trainers.pre_train.train import get_train_func, initialize

from maxtext.common.goodput import (
    RECORD_JOB_START_TIME,
    maybe_monitor_goodput,
    record_goodput,
)


def main(argv: Sequence[str]) -> None:
  """Runs the native SFT training loop for Omni."""
  argv = list(argv)
  if "use_sft=True" not in argv and not any(a.startswith("use_sft=") for a in argv):
    argv.append("use_sft=True")
  if "use_tunix_gradient_accumulation=False" not in argv and not any(
      a.startswith("use_tunix_gradient_accumulation=") for a in argv
  ):
    argv.append("use_tunix_gradient_accumulation=False")
  if "override_model_config=True" not in argv and not any(a.startswith("override_model_config=") for a in argv):
    argv.append("override_model_config=True")

  # Initialize config and goodput recorder using native SFT initialization
  mt_config, goodput_recorder = initialize(argv)

  # Run native SFT training loop (code derived from train_sft_native.py)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(mt_config, goodput_recorder, argv)
  with maybe_monitor_goodput(mt_config):
    train_func()


if __name__ == "__main__":
  app.run(main)
