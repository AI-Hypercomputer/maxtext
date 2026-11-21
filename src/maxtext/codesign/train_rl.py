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

"""RL (GRPO / GSPO) post-training entrypoint with Multi-Pass Precision Emulation.

Automatically:
1. Applies all active multi-pass emulation rules to the Actor policy model during
   GRPO/GSPO updates.
2. Keeps the Reference model un-emulated by default (unless `emulate_reference_model=True`).
3. Materializes weight transforms when syncing Actor weights to the vLLM rollout engine
   when `emulate_bake_weight_multipass_on_save=True`.

Example usage:
  python3 train_rl.py src/maxtext/configs/post_train/rl.yml \\
    run_name=grpo_fp8_x3 \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \\
    emulate_target_preset=all_linears
"""

from __future__ import annotations

from typing import Sequence

from absl import app
from maxtext.codesign.hooks import install_emulation_hooks


def main(argv: Sequence[str]) -> None:
  """Installs multi-pass emulation hooks and delegates to MaxText RL training."""
  install_emulation_hooks(workflow="rl")
  from maxtext.trainers.post_train.rl import train_rl  # pylint: disable=import-outside-toplevel

  train_rl.main(argv)


def run() -> None:
  """CLI entrypoint for RL training with multi-pass emulation."""
  install_emulation_hooks(workflow="rl")
  app.run(main)


if __name__ == "__main__":
  run()
