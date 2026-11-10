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

"""Checks that DeepSeek SFT batches keep loss weight on completion EOS tokens.

Builds the production SFT input pipeline on CPU (1-device mesh, no model) from the
same config flags as the SFT stage and fails if any EOS target has zero loss weight,
e.g. when the tokenizer's pad_id equals eos_id and EOS is masked out of the loss.

Usage:
  JAX_PLATFORMS=cpu python3 sft_eos_probe.py <sft.yml> key=value ...
"""

from absl import app
import numpy as np
import transformers

from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils import maxtext_utils

# Single-device CPU mesh; appended after the caller's flags.
_CPU_FLAGS = (
    "hardware=cpu",
    "skip_jax_distributed_system=True",
    "enable_checkpointing=False",
    "per_device_batch_size=1",
    "ici_fsdp_parallelism=1",
    "ici_expert_parallelism=1",
    "dcn_fsdp_parallelism=1",
    "dcn_data_parallelism=1",
)
_MAX_BATCHES = 8


def count_eos_targets(batch, eos_id):
  """Returns (number of EOS targets, number of EOS targets with nonzero loss weight)."""
  eos = np.asarray(batch["targets"]) == eos_id
  weighted = eos & (np.asarray(batch["targets_segmentation"]) != 0)
  return int(eos.sum()), int(weighted.sum())


def main(argv):
  config = pyconfig.initialize(list(argv) + list(_CPU_FLAGS))
  eos_id = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path).eos_token_id
  mesh = maxtext_utils.get_mesh_from_config(config)
  train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)

  n_eos = n_weighted = 0
  for _ in range(_MAX_BATCHES):
    batch_eos, batch_weighted = count_eos_targets(next(train_iter), eos_id)
    n_eos += batch_eos
    n_weighted += batch_weighted
    if n_eos:
      break
  print(f"EOS_PROBE: eos_id={eos_id} n_eos={n_eos} n_weighted={n_weighted}")
  if n_eos == 0 or n_weighted != n_eos:
    print("EOS_PROBE: RESULT FAIL")
    return 1
  print("EOS_PROBE: RESULT PASS")
  return 0


if __name__ == "__main__":
  app.run(main)
