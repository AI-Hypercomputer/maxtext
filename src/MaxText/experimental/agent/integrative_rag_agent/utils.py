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

"""Utility functions for the Integrative RAG Agent.

This module provides helper functions for handling data, such as reading
and sampling code blocks from files.
"""

import json
import random


def read_code_blocks(file_path, number_of_blocks):
  """
  Reads and returns a random selection of code blocks from a JSON file.

  Args:
    file_path (str): The path to the JSON file containing the scraped code blocks.
    number_of_blocks (int): The number of code blocks to randomly select.

  Returns:
    str: A string containing the randomly selected code blocks, separated by
      three newlines.
  """
  with open(file_path, "rt", encoding="utf-8") as f:
    all_blocks = json.load(f)
  return "\n\n\n".join(
      random.choices(sum([v for k, v in all_blocks["scraped_blocks"].items() if len(v) > 0], []), k=number_of_blocks)
  )
