# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the markdown table that build_tpu7x_sheet_table.py writes."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from build_tpu7x_sheet_table import HEADER, markdown  # pylint: disable=wrong-import-position

# The alignment of the 10 columns that the table had before the `log` column, the same as in the tabs.
FIRST_TEN_ALIGNMENTS = [":---", ":---:", ":---:", ":---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]


def _cells(line: str) -> list[str]:
  """The cells of a markdown table row `| a | b | c |`."""
  return [cell.strip() for cell in line.removeprefix("|").removesuffix("|").split("|")]


class MarkdownTest(unittest.TestCase):
  """GitHub renders no table at all when the separator row and the header row have different numbers of cells."""

  def setUp(self):
    super().setUp()
    row = [f"cell {i}" for i in range(len(HEADER))]
    table = [line for line in markdown("p1", [row]).splitlines() if line.startswith("|")]
    self.header, self.separator = _cells(table[0]), _cells(table[1])

  def test_separator_has_one_cell_per_header_column(self):
    self.assertEqual(len(self.header), len(HEADER))
    self.assertEqual(len(self.separator), len(HEADER))

  def test_separator_keeps_the_alignment_of_the_first_ten_columns(self):
    self.assertEqual(self.separator[:10], FIRST_TEN_ALIGNMENTS)

  def test_log_column_is_left_aligned(self):
    alignment = dict(zip(self.header, self.separator))
    self.assertEqual(alignment.get("log"), ":---")


if __name__ == "__main__":
  unittest.main()
