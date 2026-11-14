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

"""`tokenizer_trust_remote_code` reaches `transformers.AutoTokenizer.from_pretrained` (no download needed)."""

import types
import unittest
from unittest import mock

from absl.testing import parameterized

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import tokenizer


def _fake_hf_tokenizer():
  return mock.Mock(pad_token_id=0, unk_token_id=None, bos_token_id=1, eos_token_id=2)


class TokenizerTrustRemoteCodeTest(parameterized.TestCase):
  """The flag defaults to False and is passed through every pretraining tokenizer entry point."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(
        tokenizer.transformers.AutoTokenizer, "from_pretrained", return_value=_fake_hf_tokenizer()
    )
    self.from_pretrained = patcher.start()
    self.addCleanup(patcher.stop)

  def _passed_flag(self):
    self.from_pretrained.assert_called_once()
    return self.from_pretrained.call_args.kwargs["trust_remote_code"]

  def test_build_tokenizer_defaults_to_false(self):
    tokenizer.build_tokenizer("some/path", "huggingface", True, False, None)
    self.assertFalse(self._passed_flag())

  @parameterized.parameters(True, False)
  def test_build_tokenizer_passes_flag(self, trust):
    tok = tokenizer.build_tokenizer("some/path", "huggingface", True, False, None, trust_remote_code=trust)
    self.assertIsInstance(tok, tokenizer.HFTokenizer)
    self.assertEqual(self._passed_flag(), trust)

  @parameterized.parameters(True, False)
  def test_tfds_get_tokenizer_passes_flag(self, trust):
    input_pipeline_utils.get_tokenizer("some/path", "huggingface", True, False, None, trust_remote_code=trust)
    self.assertEqual(self._passed_flag(), trust)

  @parameterized.parameters(True, False)
  def test_grain_reads_flag_from_config(self, trust):
    config = types.SimpleNamespace(
        tokenizer_path="some/path",
        tokenizer_type="huggingface",
        add_bos=True,
        add_eos=False,
        hf_access_token=None,
        tokenizer_trust_remote_code=trust,
    )
    _, pad_id = data_processing_utils.get_tokenizer_and_pad_id(config)
    self.assertEqual(self._passed_flag(), trust)
    self.assertEqual(pad_id, 0)


if __name__ == "__main__":
  unittest.main()
