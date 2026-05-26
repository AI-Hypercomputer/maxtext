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

"""Tests for _load_custom_callable (used by dataset_processor_path knob)."""

import os
import sys
import tempfile
import textwrap
import unittest

import pytest

from maxtext.trainers.post_train.rl.train_rl import _load_custom_callable


pytestmark = [pytest.mark.post_training]


_USER_PROCESS_DATA_SOURCE = textwrap.dedent(
    """
    # Simulated user-provided dataset processor file.
    def process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x):
      # Minimal stand-in for utils_rl.process_data: returns a dict shaped like
      # what the RL data pipeline expects, with a marker so the test can verify
      # that THIS function (not the built-in) was actually invoked.
      return {
          "prompts": f"USER_PROCESSOR<{x.get('question', '')}>",
          "question": x.get("question", ""),
          "answer": x.get("answer", ""),
          "_marker": "loaded_from_user_file",
      }


    def another_helper(x):
      return x * 2
    """
).strip()


def _write_user_file(tmpdir):
  """Write the user processor file inside tmpdir and return its absolute path."""
  path = os.path.join(tmpdir, "user_processor.py")
  with open(path, "w", encoding="utf-8") as f:
    f.write(_USER_PROCESS_DATA_SOURCE)
  return path


class LoadCustomCallableTest(unittest.TestCase):
  """Verify _load_custom_callable loads a function from a user .py file."""

  @pytest.mark.cpu_only
  def test_loads_function_from_user_file(self):
    """Returns a callable that behaves like the function in the user file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      user_file = _write_user_file(tmpdir)
      fn = _load_custom_callable(user_file, "process_data")

      self.assertTrue(callable(fn))
      # pylint: disable-next=not-callable
      result = fn(
          "dataset_name",
          model_tokenizer=None,
          template_config=None,
          tmvp_config=None,
          x={"question": "2+2?", "answer": "4"},
      )
      self.assertEqual(result["_marker"], "loaded_from_user_file")
      self.assertEqual(result["prompts"], "USER_PROCESSOR<2+2?>")
      self.assertEqual(result["question"], "2+2?")
      self.assertEqual(result["answer"], "4")

  @pytest.mark.cpu_only
  def test_loads_any_named_function(self):
    """function_name argument selects which symbol to return."""
    with tempfile.TemporaryDirectory() as tmpdir:
      user_file = _write_user_file(tmpdir)
      fn = _load_custom_callable(user_file, "another_helper")
      self.assertEqual(fn(5), 10)  # pylint: disable=not-callable

  @pytest.mark.cpu_only
  def test_raises_when_file_does_not_exist(self):
    """Nonexistent path -> ValueError, not a cryptic ImportError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      bogus = os.path.join(tmpdir, "does_not_exist.py")
      with self.assertRaises(ValueError):
        _load_custom_callable(bogus, "process_data")

  @pytest.mark.cpu_only
  def test_raises_when_function_not_defined(self):
    """File exists but doesn't define the named function -> ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      user_file = _write_user_file(tmpdir)
      with self.assertRaises(ValueError):
        _load_custom_callable(user_file, "no_such_function")

  @pytest.mark.cpu_only
  def test_does_not_pollute_sys_path(self):
    """Loading the file must not append its directory to sys.path."""
    sys_path_before = list(sys.path)
    with tempfile.TemporaryDirectory() as tmpdir:
      user_file = _write_user_file(tmpdir)
      _load_custom_callable(user_file, "process_data")
    self.assertEqual(sys.path, sys_path_before)

  @pytest.mark.cpu_only
  def test_does_not_pollute_sys_modules_globally(self):
    """The loaded module gets a unique synthetic name; it should not shadow
    other modules with a generic name like 'user_processor'."""
    with tempfile.TemporaryDirectory() as tmpdir:
      user_file = _write_user_file(tmpdir)
      _load_custom_callable(user_file, "process_data")
      # The helper uses '_user_module_<function_name>' as the synthetic module
      # name, not the file's basename - so 'user_processor' should NOT exist.
      self.assertNotIn("user_processor", sys.modules)


if __name__ == "__main__":
  unittest.main()
