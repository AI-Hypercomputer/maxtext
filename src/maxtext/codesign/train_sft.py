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

"""SFT (Supervised Fine-Tuning) entrypoint with Multi-Pass Precision Emulation.

Supports dtype-agnostic multi-pass residual decomposition (`emulate_multipass_enabled`,
`emulate_multipass_preset`, `emulate_src_dtype`, `emulate_num_passes`,
`emulate_pass_dtypes`, `emulate_scale_mode`, `emulate_cross_term_mode`,
`emulate_acc_dtype`, `emulate_dlhs`, `emulate_drhs`) and custom registered
`EmulationFeature` plugins.

Example usage:
  python3 train_sft.py src/maxtext/configs/post_train/sft.yml \\
    run_name=sft_fp8_x3 \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \\
    emulate_target_preset=all_linears
"""

from __future__ import annotations

from typing import Sequence

from absl import app
from maxtext.codesign.hooks import install_emulation_hooks


def main(argv: Sequence[str]) -> None:
  """Installs multi-pass emulation hooks and delegates to MaxText SFT training."""
  install_emulation_hooks(workflow="sft")
  from maxtext.trainers.post_train.sft import train_sft  # pylint: disable=import-outside-toplevel

  train_sft.main(argv)


def run() -> None:
  """CLI entrypoint for SFT with multi-pass emulation."""
  install_emulation_hooks(workflow="sft")
  app.run(main)


if __name__ == "__main__":
  run()
