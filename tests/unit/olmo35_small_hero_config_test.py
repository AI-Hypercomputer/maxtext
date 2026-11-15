# Copyright 2023-2026 Google LLC
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

"""olmo35_small_hero.yml resolves to what it says.

A model config overrides the keys of the config file that names it (only command-line arguments beat it), so a hero
setting the model config also sets would be silently replaced (gdn_chunk_size was: 128 in the file, 64 in the model's).

    JAX_PLATFORMS=cpu pytest tests/unit/olmo35_small_hero_config_test.py
"""

import os
import unittest

import yaml

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR

HERO = os.path.join(MAXTEXT_PKG_DIR, "configs", "olmo", "olmo35_small_hero.yml")


def _norm(value):
  value = getattr(value, "value", value)  # enums
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return float(value)
  if isinstance(value, (list, tuple)):
    return [_norm(v) for v in value]
  return value


class OLMo35SmallHeroConfigTest(unittest.TestCase):
  """Every key of the hero config file survives config resolution."""

  def test_every_setting_resolves_to_the_files_value(self):
    with open(HERO, encoding="utf-8") as f:
      want = yaml.safe_load(f)
    cfg = pyconfig.initialize(["", HERO, "run_name=olmo35_small_hero_config_test", "skip_jax_distributed_system=True"])
    shadowed = {
        k: (v, getattr(cfg, k)) for k, v in want.items() if k != "base_config" and _norm(getattr(cfg, k)) != _norm(v)
    }
    self.assertEqual(shadowed, {}, "settings replaced at resolution (by the model config?): {key: (file, resolved)}")


if __name__ == "__main__":
  unittest.main()
