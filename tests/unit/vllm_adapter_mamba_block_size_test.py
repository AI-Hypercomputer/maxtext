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

"""Tests the gdn_mamba_block_size config field used for vLLM mamba prefix caching."""

import unittest

from maxtext.configs import types as config_types


class GdnMambaBlockSizeFieldTest(unittest.TestCase):
  """`gdn_mamba_block_size` must be a declared config field."""

  def test_field_is_declared(self):
    # pyconfig._prepare_for_pydantic validates every YAML/CLI key against
    # model_fields, so the adapter's override raises
    # "'gdn_mamba_block_size' not in ..." unless the field is declared here,
    # and the server never starts.
    self.assertIn("gdn_mamba_block_size", config_types.Qwen3Next.model_fields)

  def test_default_means_no_prefix_caching(self):
    # Training, MaxEngine and vLLM non-align serving never set it.
    self.assertEqual(config_types.Qwen3Next.model_fields["gdn_mamba_block_size"].default, 0)


if __name__ == "__main__":
  unittest.main()
