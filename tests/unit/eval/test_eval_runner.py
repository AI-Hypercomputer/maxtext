# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for maxtext.eval.runner.eval_runner."""

import os
import tempfile
import textwrap
import unittest

from maxtext.eval.runner.eval_runner import (
    _build_results_path,
    _derive_from_maxtext_config,
    _merge_config,
)


class TestMergeConfig(unittest.TestCase):

  def test_override_takes_precedence(self):
    base = {"concurrency": 64, "temperature": 0.0}
    overrides = {"concurrency": 128}
    merged = _merge_config(base, overrides)
    self.assertEqual(merged["concurrency"], 128)
    self.assertEqual(merged["temperature"], 0.0)

  def test_none_override_does_not_overwrite(self):
    base = {"concurrency": 64}
    overrides = {"concurrency": None}
    merged = _merge_config(base, overrides)
    self.assertEqual(merged["concurrency"], 64)

  def test_new_key_from_override(self):
    merged = _merge_config({}, {"benchmark": "mmlu"})
    self.assertEqual(merged["benchmark"], "mmlu")

  def test_base_preserved_when_no_override(self):
    base = {"a": 1, "b": 2}
    merged = _merge_config(base, {})
    self.assertEqual(merged, base)


class TestBuildResultsPath(unittest.TestCase):

  def test_standard_path(self):
    cfg = {"base_output_directory": "gs://bucket/", "run_name": "run1"}
    self.assertEqual(_build_results_path(cfg), "gs://bucket/run1/eval_results")

  def test_trailing_slash_stripped(self):
    cfg = {"base_output_directory": "gs://bucket///", "run_name": "run1"}
    self.assertEqual(_build_results_path(cfg), "gs://bucket/run1/eval_results")

  def test_local_path(self):
    cfg = {"base_output_directory": "/tmp/out", "run_name": "test"}
    self.assertEqual(_build_results_path(cfg), "/tmp/out/test/eval_results")

  def test_missing_run_name_raises(self):
    cfg = {"base_output_directory": "gs://bucket/"}
    with self.assertRaises(ValueError):
      _build_results_path(cfg)

  def test_missing_base_output_directory_raises(self):
    cfg = {"run_name": "run1"}
    with self.assertRaises(ValueError):
      _build_results_path(cfg)

  def test_empty_strings_raise(self):
    cfg = {"base_output_directory": "", "run_name": ""}
    with self.assertRaises(ValueError):
      _build_results_path(cfg)


class TestDeriveFromMaxtextConfig(unittest.TestCase):

  def _write_yaml(self, content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name

  def test_derives_max_model_len(self):
    path = self._write_yaml("max_target_length: 1024\n")
    derived = _derive_from_maxtext_config(path)
    self.assertEqual(derived["max_model_len"], 1024)
    os.unlink(path)

  def test_derives_max_tokens_default(self):
    path = self._write_yaml(
        "max_target_length: 1024\nmax_prefill_predict_length: 256\n"
    )
    derived = _derive_from_maxtext_config(path)
    self.assertEqual(derived["max_tokens_default"], 768)
    os.unlink(path)

  def test_no_max_tokens_default_without_both_fields(self):
    path = self._write_yaml("max_target_length: 1024\n")
    derived = _derive_from_maxtext_config(path)
    self.assertNotIn("max_tokens_default", derived)
    os.unlink(path)

  def test_derives_run_name_and_base_output_directory(self):
    path = self._write_yaml(
        "base_output_directory: gs://b/\nrun_name: myrun\n"
    )
    derived = _derive_from_maxtext_config(path)
    self.assertEqual(derived["base_output_directory"], "gs://b/")
    self.assertEqual(derived["run_name"], "myrun")
    os.unlink(path)

  def test_empty_run_name_not_derived(self):
    # base.yml has run_name: '' by default; empty string should not be derived.
    path = self._write_yaml("run_name: ''\n")
    derived = _derive_from_maxtext_config(path)
    self.assertNotIn("run_name", derived)
    os.unlink(path)

  def test_empty_yaml_returns_empty_dict(self):
    path = self._write_yaml("")
    derived = _derive_from_maxtext_config(path)
    self.assertEqual(derived, {})
    os.unlink(path)


if __name__ == "__main__":
  unittest.main()
