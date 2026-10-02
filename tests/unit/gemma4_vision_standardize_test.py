# Copyright 2026 Google LLC
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

"""Unit tests for the Gemma-4 vision standardization (std_bias/std_scale) contract.

HF `modeling_gemma4` registers vision `std_bias`/`std_scale` via `register_buffer` (non-trainable) and ONLY when
`vision_config.standardize=True`. Per-variant HF truth: Gemma-4 26B/31B ship standardize=True (2 std tensors);
E2B/E4B ship standardize=False (0 std tensors). This test locks the MaxText contract:

  * The `standardize_for_vit` config field is a per-model-family semantic distinction and must be set explicitly
    by every Gemma-4 model config (no silent global default); an unset Gemma-4 config is a hard config error.
  * standardize_for_vit=False  -> NO std state at all (the standardize op is an exact identity).
  * standardize_for_vit=True   -> exactly two std leaves, constructed as VisionStdVar (a non-trainable
    `nnx.Variable`, NOT `nnx.Param`), so they are checkpoint-resident but excluded from the optimizer/gradient.
"""

import unittest

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.models.gemma4_vision import Gemma4VisionEncoderLayer, VisionStdVar


def _init(model_name):
  base_config_path = f"{MAXTEXT_REPO_ROOT}/src/maxtext/configs/base.yml"
  argv = [
      "",
      base_config_path,
      f"model_name={model_name}",
      "attention=dot_product",
      "matmul_precision=highest",
      "dtype=float32",
      "dtype_mm=float32",
      "weight_dtype=float32",
      "skip_jax_distributed_system=true",
      "enable_checkpointing=false",
      "run_name=std_test",
      "base_output_directory=/tmp/std_test",
  ]
  if model_name in ("gemma4-e2b", "gemma4-e4b"):
    argv.append("scan_layers=false")
  return pyconfig.initialize(argv)


class TestGemma4VisionStandardizeContract(unittest.TestCase):
  """standardize_for_vit gates whether/how std_bias/std_scale exist, per HF vision_config.standardize."""

  def setUp(self):
    self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

  def _std_leaves(self, model, filt):
    names = set()
    for path, _ in jax.tree_util.tree_flatten_with_path(nnx.state(model, filt))[0]:
      ps = "/".join(str(getattr(p, "key", p)) for p in path)
      if "std_bias" in ps or "std_scale" in ps:
        names.add(ps)
    return names

  def test_e2b_standardize_false_has_no_std_state(self):
    """E2B (HF standardize=false): no std state at all; standardize is an exact identity."""
    cfg = _init("gemma4-e2b")
    self.assertFalse(cfg.standardize_for_vit)
    model = Gemma4VisionEncoderLayer(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertFalse(hasattr(model, "std_bias"))
    self.assertFalse(hasattr(model, "std_scale"))
    self.assertEqual(self._std_leaves(model, nnx.Param), set())
    self.assertEqual(self._std_leaves(model, nnx.Variable), set())

  def test_26b_standardize_true_has_nontrainable_std_buffers(self):
    """26B (HF standardize=true): exactly two std leaves as VisionStdVar, NOT in nnx.Param."""
    cfg = _init("gemma4-26b")
    self.assertTrue(cfg.standardize_for_vit)
    model = Gemma4VisionEncoderLayer(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(model.std_bias, VisionStdVar)
    self.assertIsInstance(model.std_scale, VisionStdVar)
    # Present as (non-Param) Variables, absent from the Param (trainable) collection.
    self.assertEqual(self._std_leaves(model, nnx.Param), set())
    self.assertEqual(len(self._std_leaves(model, nnx.Variable)), 2)

  def test_e2b_e4b_false_and_26b_31b_true(self):
    """All four Gemma-4 configs resolve standardize_for_vit to their HF truth from the real YAMLs."""
    self.assertFalse(_init("gemma4-e2b").standardize_for_vit)
    self.assertFalse(_init("gemma4-e4b").standardize_for_vit)
    self.assertTrue(_init("gemma4-26b").standardize_for_vit)
    self.assertTrue(_init("gemma4-31b").standardize_for_vit)


if __name__ == "__main__":
  unittest.main()
