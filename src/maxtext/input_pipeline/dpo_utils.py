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

"""DPO specific input pipeline utilities."""

import dataclasses
import grain.python as grain
import numpy as np


@dataclasses.dataclass
class DPOTunixPrep(grain.MapTransform):
  """Prepares DPO data for Tunix.
  Renames input columns, extracts common prefix if needed, generates masks, and performs
  DPO-aware padding (left-padded prompts, right-padded responses).
  """

  pad_id: int
  max_target_length: int
  data_column_names: tuple[str, ...]
  max_prompt_length: int | None = None

  def map(self, element):
    "Apply the dataset transformations for Tunix-based DPO."
    # 1. Reformat/Extract Columns
    try:
      if len(self.data_column_names) == 3:
        input_ids = element[self.data_column_names[0]]
        chosen_ids = element[self.data_column_names[1]]
        rejected_ids = element[self.data_column_names[2]]
      elif len(self.data_column_names) == 2:
        # Support for datasets like Anthropic/hh-rlhf where prompt is a common prefix
        full_chosen = element[self.data_column_names[0]]
        full_rejected = element[self.data_column_names[1]]

        # Find common prefix length
        prefix_len = 0
        for c, r in zip(full_chosen, full_rejected):
          if c != r:
            break
          prefix_len += 1
        input_ids = full_chosen[:prefix_len]
        chosen_ids = full_chosen[prefix_len:]
        rejected_ids = full_rejected[prefix_len:]
      else:
        raise ValueError(f"DPOTunixPrep expects 2 or 3 columns, got {len(self.data_column_names)}")
    except KeyError as e:
      raise KeyError(
          f"Column '{e.args[0]}' not found in the dataset. "
          f"Expected columns: {self.data_column_names}. "
          f"Available columns: {list(element.keys())}. "
          "Please verify that 'train_data_columns' and 'eval_data_columns' match your dataset."
      ) from e

    # 2. Padding and Masking
    max_prompt_length = self.max_prompt_length or (self.max_target_length // 2)
    max_response_length = self.max_target_length - max_prompt_length

    assert max_prompt_length > 0, (
        "max_prompt_length must be positive. " "Check the configs for 'max_prompt_length' and 'max_target_length'."
    )
    assert max_response_length > 0, (
        "max_response_length must be positive. " "Check the configs for 'max_prompt_length' and 'max_target_length'."
    )

    prompt_ids = self._pad(input_ids, max_prompt_length, left=True)
    chosen_ids = self._pad(chosen_ids, max_response_length, left=False)
    rejected_ids = self._pad(rejected_ids, max_response_length, left=False)

    # Remove old columns if they exist
    for key in self.data_column_names:
      if key in element:
        del element[key]

    element["prompt_ids"] = prompt_ids
    element["chosen_ids"] = chosen_ids
    element["rejected_ids"] = rejected_ids
    element["prompt_mask"] = (prompt_ids != self.pad_id).astype(np.int32)
    element["chosen_mask"] = (chosen_ids != self.pad_id).astype(np.int32)
    element["rejected_mask"] = (rejected_ids != self.pad_id).astype(np.int32)
    return element

  def _pad(self, x, length, left=False):
    """Pads or trims an array to a specific length.

    When left=True (for prompts), trims from the left to keep the suffix (closest context).
    When left=False (for responses), trims from the right to keep the prefix.
    """
    x = np.asarray(x)
    pad_amount = max(length - x.shape[0], 0)
    if left:
      pad_width = ((pad_amount, 0),)
      x_trimmed = x[-length:]
    else:
      pad_width = ((0, pad_amount),)
      x_trimmed = x[:length]
    return np.pad(x_trimmed, pad_width, constant_values=self.pad_id).astype(np.int32)
