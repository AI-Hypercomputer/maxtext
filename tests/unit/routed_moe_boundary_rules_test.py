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
"""Checks that routed-MoE boundary rules mirror the residual rules unless a rule set opts out.

`activation_norm_length_routed` / `activation_embed_routed` lay out the routed MoE's shard_map inputs and
output. A config that overrides the residual rules but forgets the routed ones would silently change the
MoE boundary layout, so every shipped config must keep them equal, except the rule sets listed below.
"""

import os
import unittest

import yaml
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

_CONFIGS = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs")
_CUSTOM_RULES = os.path.join(_CONFIGS, "custom_mesh_and_rule")

_MIRRORED = (
    ("activation_norm_length", "activation_norm_length_routed"),
    ("activation_embed", "activation_embed_routed"),
)

# Custom rule sets that intentionally lay out the MoE boundary differently from the residual stream.
_INTENTIONAL_DIFFERENCES = {
    "tp-as-ep": "splits tokens over tensor on the sequence dim at the MoE boundary",
}


def _axes(rules, name):
  for rule in rules:
    if rule[0] == name:
      value = rule[1]
      return [] if value is None else ([value] if isinstance(value, str) else list(value))
  return []  # an undefined logical name resolves to replicated


def _load(path):
  with open(path, encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _effective_rule_sets():
  """Yields (config path, logical_axis_rules as pyconfig / MaxTextConfig would apply them)."""
  base_rules = _load(os.path.join(_CONFIGS, "base.yml"))["logical_axis_rules"]
  yield "base.yml", base_rules

  # Custom rule sets replace the rules wholesale (MaxTextConfig.set_derived_and_validate_values).
  for file_name in sorted(os.listdir(_CUSTOM_RULES)):
    name, ext = os.path.splitext(file_name)
    if ext == ".yml" and name not in _INTENTIONAL_DIFFERENCES:
      mesh_config = types.MaxTextConfig._load_mesh_config_from_yaml(name)  # pylint: disable=protected-access
      yield os.path.join("custom_mesh_and_rule", file_name), mesh_config.get("logical_axis_rules", [])

  # Other configs are merged into base.yml by logical name, or replace it with override_logical_axis_rules.
  for root, _, files in os.walk(_CONFIGS):
    if os.path.abspath(root).startswith(os.path.abspath(_CUSTOM_RULES)):
      continue
    for file_name in sorted(files):
      path = os.path.join(root, file_name)
      rel = os.path.relpath(path, _CONFIGS)
      if not file_name.endswith(".yml") or rel == "base.yml":
        continue
      config = _load(path)
      if isinstance(config, dict) and config.get("logical_axis_rules"):
        yield rel, pyconfig._apply_rules(base_rules, config["logical_axis_rules"], config)  # pylint: disable=protected-access


class RoutedMoeBoundaryRulesTest(unittest.TestCase):

  def test_routed_rules_mirror_residual_rules(self):
    mismatches = [
        f"{rel}: {residual}={_axes(rules, residual)} but {routed}={_axes(rules, routed)}"
        for rel, rules in _effective_rule_sets()
        for residual, routed in _MIRRORED
        if _axes(rules, residual) != _axes(rules, routed)
    ]
    self.assertFalse(
        mismatches,
        "routed-MoE boundary rules must mirror the residual rules (or be listed in _INTENTIONAL_DIFFERENCES):\n"
        + "\n".join(mismatches),
    )


if __name__ == "__main__":
  unittest.main()
