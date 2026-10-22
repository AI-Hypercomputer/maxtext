# Copyright 2023–2026 Google LLC
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

"""Tests for the Wan model utilities (`utils_wan.py`)."""

import os
import tempfile
import types
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
import jax

from maxtext.m3.models.wan import utils_wan

# pylint: enable=wrong-import-position


class UtilsWanTest(unittest.TestCase):

  def test_get_activation(self):
    self.assertIs(utils_wan.get_activation("silu"), jax.nn.silu)
    self.assertIs(utils_wan.get_activation("gelu"), jax.nn.gelu)
    with self.assertRaises(ValueError):
      utils_wan.get_activation("unknown")

  def test_safe_getattr(self):
    obj = types.SimpleNamespace(a=1)
    self.assertEqual(utils_wan.safe_getattr(obj, "a"), 1)
    self.assertEqual(utils_wan.safe_getattr(obj, "b", 2), 2)

  def test_get_precision(self):
    self.assertIsNone(utils_wan.get_precision(types.SimpleNamespace(precision="")))
    self.assertIs(utils_wan.get_precision(types.SimpleNamespace(precision="HIGHEST")), jax.lax.Precision.HIGHEST)

  def test_get_gcs_output_path(self):
    self.assertIsNone(utils_wan.get_gcs_output_path(types.SimpleNamespace(output_dir="/tmp/out", run_name="r")))
    self.assertEqual(
        utils_wan.get_gcs_output_path(types.SimpleNamespace(output_dir="gs://bucket/out", run_name="r")),
        "gs://bucket/out/r",
    )

  def test_load_prompts(self):
    self.assertEqual(utils_wan.load_prompts("", default_prompt="a cat"), ["a cat"])
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "prompts.txt")
      with open(path, "w", encoding="utf-8") as f:
        f.write("# comment\nfirst prompt\n\n  second prompt  \n")
      self.assertEqual(utils_wan.load_prompts(path), ["first prompt", "second prompt"])
    with self.assertRaises(ValueError):
      utils_wan.load_prompts("")

  def test_chunk_and_pad(self):
    chunks = list(utils_wan.chunk_and_pad(["a", "b", "c"], batch_size=2))
    self.assertEqual(chunks, [(0, ["a", "b"], 2), (2, ["c", "c"], 1)])
    with self.assertRaises(ValueError):
      list(utils_wan.chunk_and_pad(["a"], batch_size=0))


if __name__ == "__main__":
  unittest.main()
