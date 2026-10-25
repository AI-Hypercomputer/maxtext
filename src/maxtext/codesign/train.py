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

"""Pre-Training / General Training entrypoint with Multi-Pass Precision Emulation.

Supports dtype-agnostic multi-pass residual decomposition (`emulate_multipass_enabled`,
`emulate_multipass_preset`, `emulate_src_dtype`, `emulate_num_passes`,
`emulate_pass_dtypes`, `emulate_scale_mode`, `emulate_cross_term_mode`,
`emulate_acc_dtype`, `emulate_dlhs`, `emulate_drhs`) and custom registered
`EmulationFeature` plugins.

Example usage:
  python3 train.py src/maxtext/configs/base.yml \\
    run_name=pretrain_fp8_x3 \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \\
    emulate_scale_mode=per_block_pow2 \\
    emulate_block_size=128
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

from absl import app

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
  sys.path.insert(0, _CUR_DIR)

if __package__:
  from .hooks import install_emulation_hooks
else:
  from hooks import install_emulation_hooks  # type: ignore


def main(argv: Sequence[str]) -> None:
  install_emulation_hooks(workflow="pretrain")
  from maxtext.trainers.pre_train import train  # pylint: disable=g-import-not-at-top

  train.main(argv)


def run() -> None:
  install_emulation_hooks(workflow="pretrain")
  app.run(main)


if __name__ == "__main__":
  run()
