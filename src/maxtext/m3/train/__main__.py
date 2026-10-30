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

"""YAML-free CLI entrypoint (`python3 -m maxtext.m3.train --preset <model> ...`)."""

from collections.abc import Sequence
import sys

from maxtext.m3.configs.cli_parser import M3ArgumentParser
from maxtext.m3.train.trainer import M3Trainer


def main(argv: Sequence[str] | None = None) -> None:
  """Parses `ModelConfig` + `TrainingConfig` CLI arguments and runs training or AOT compilation."""
  parser = M3ArgumentParser()
  model_cfg, train_cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)
  trainer = M3Trainer(model=model_cfg, args=train_cfg)
  if train_cfg.aot.compile_topology:
    trainer.compile_aot()
  else:
    trainer.train()


if __name__ == "__main__":
  main()
