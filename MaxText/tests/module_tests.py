"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for train.py with various configs"""

import os
import sys
import subprocess
import unittest
import pytest
from absl.testing import absltest

from MaxText.globals import PKG_DIR

class ModuleTests(unittest.TestCase):
  """Tests train.py with various invokation methods"""

  def test_get_informative_error(self):
    # cwd is MaxText so step up a level
    os.chdir("..")

    command = [
        sys.executable,  # use the same interpreter instance
        "MaxText/train.py",
        "MaxText/configs/base.yml",
        "run_name=maxtext-module-test",
        "base_output_directory=gs://does-not-exist",
        "dataset_type=synthetic"
    ]

    with self.assertRaises(subprocess.CalledProcessError) as context:
      subprocess.run(command, check=True, text=True, capture_output=True)

    ex = context.exception
    self.assertEqual(ex.returncode, 64)
    self.assertIn("The MaxText API has changed", ex.stderr)

if __name__ == "__main__":
  absltest.main()

