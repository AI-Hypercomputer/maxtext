# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""fp32_master_storage, the config key Tunix's Fp32MasterOptimizer reads.

MaxText itself does not read the key; Tunix (utils/maxtext_utils.py, create_maxtext_engine) builds its float32-master
optimizer with master_storage=config.fp32_master_storage. So this file checks only what MaxText owns: the key exists
with exactly the two values Tunix accepts ("f32", "residual16"), defaults to "f32" in types.py and base.yml (the
optimizer then builds exactly as before), is resolved by pyconfig, and any other value is refused when the config is
built, before Tunix sees it.
"""

import os
import sys
import typing

import pytest
import yaml

from maxtext.configs import types as mt_types

_KEY = "fp32_master_storage"
# The key this file tests. On a MaxText without it this line raises, so the file cannot be collected there.
_KEY_FIELD = mt_types.RematAndOffload.model_fields[_KEY]


def _pyconfig(**overrides):
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  common = {
      "run_name": "fp32_master_storage_key",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **common, **overrides)


def test_key_takes_exactly_the_two_storages_tunix_accepts():
  assert typing.get_origin(_KEY_FIELD.annotation) is typing.Literal
  assert set(typing.get_args(_KEY_FIELD.annotation)) == {"f32", "residual16"}


def test_default_is_f32_in_types_and_base_yml():
  assert _KEY_FIELD.default == "f32"
  base_yml = os.path.join(os.path.dirname(mt_types.__file__), "base.yml")
  with open(base_yml, encoding="utf-8") as f:
    assert yaml.safe_load(f)[_KEY] == "f32"
  assert getattr(_pyconfig(), _KEY) == "f32"


def test_residual16_is_resolved():
  assert getattr(_pyconfig(**{_KEY: "residual16"}), _KEY) == "residual16"


@pytest.mark.parametrize("value", ("bf16", "residual8", "F32", ""))
def test_other_values_are_refused(value):
  with pytest.raises(ValueError, match=_KEY):
    _pyconfig(**{_KEY: value})
