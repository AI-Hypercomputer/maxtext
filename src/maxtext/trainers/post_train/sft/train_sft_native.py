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

"Training loop for Supervised Fine-Tuning (SFT)."

from typing import Sequence

from absl import app

from maxtext.common.goodput import (
    RECORD_JOB_START_TIME,
    maybe_monitor_goodput,
    record_goodput,
)
from maxtext.trainers.pre_train.train import get_train_func, initialize


def main(argv: Sequence[str]) -> None:
  argv = list(argv)
  argv.append("use_sft=True")
  argv.append("use_tunix_gradient_accumulation=False")
  config, recorder = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
