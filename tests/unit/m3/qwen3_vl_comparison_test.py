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

"""Unit tests for self-contained m3 Qwen3-VL multimodal model family.

Verifies:
  1. Factory routing through model_creation_utils with use_m3_model=True.
  2. 100% parameter path and shape parity against the reference model.
  3. Qwen3VLModel forward pass execution with multimodal visual embeddings and 3D MRoPE.
"""

import os
import unittest
import numpy as np
import pytest
from flax import nnx

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    create_qwen3_vl_model,
)
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


def _get_qwen3_vl_config_args(use_m3: bool = True, reduced: bool = True):
  """Returns Qwen3-VL config arguments with optional m3 routing and reduced layers."""
  args = [
      "",
      BASE_CONFIG_PATH,
      "model_name=qwen3-vl-2b",
      f"use_m3_model={'true' if use_m3 else 'false'}",
      "use_multimodal=true",
      "scan_layers=false",
      "per_device_batch_size=1",
      "dtype=float32",
      "weight_dtype=float32",
      "skip_jax_distributed_system=true",
  ]
  if reduced:
    args.extend(
        [
            "override_model_config=true",
            "base_num_decoder_layers=2",
            "num_hidden_layers_for_vit=2",
            "deepstack_visual_indexes_for_vit=[1]",
        ]
    )
  return args


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class Qwen3VLComparisonTest(unittest.TestCase):
  """Unit tests for self-contained m3 Qwen3-VL implementation."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()[:1]
    self.mesh = Mesh(np.array(devices), ("data",))

  def test_model_creation_utils_routing(self):
    """Verifies that model_creation_utils routes to Qwen3VLModel when use_m3_model=True."""
    cfg = pyconfig.initialize(_get_qwen3_vl_config_args(use_m3=True, reduced=True))
    with jax.set_mesh(self.mesh):
      model = model_creation_utils.create_model(cfg, self.mesh)
      self.assertIsInstance(model, Qwen3VLModel)

  def test_parameter_shapes_and_paths_match(self):
    """Verifies that all parameter paths and shapes match 100% between legacy and m3 Qwen3-VL."""
    cfg_old = pyconfig.initialize(_get_qwen3_vl_config_args(use_m3=False, reduced=False))
    cfg_new = pyconfig.initialize(_get_qwen3_vl_config_args(use_m3=True, reduced=False))

    _, abs_old = model_creation_utils.create_nnx_abstract_model(cfg_old, self.mesh)
    _, abs_new = model_creation_utils.create_nnx_abstract_model(cfg_new, self.mesh)

    p_old = nnx.state(abs_old, nnx.Param)
    p_new = nnx.state(abs_new, nnx.Param)

    flat_old = dict(p_old.flat_state())
    flat_new = dict(p_new.flat_state())

    self.assertEqual(
        len(flat_old), len(flat_new), f"Parameter counts differ: old has {len(flat_old)}, new has {len(flat_new)}"
    )
    self.assertEqual(set(flat_old.keys()), set(flat_new.keys()), "Parameter keys / paths do not match")

    for key in sorted(flat_old.keys()):
      val_old = flat_old[key]
      val_new = flat_new[key]
      self.assertEqual(val_old.shape, val_new.shape, f"Shape mismatch for {key}: {val_old.shape} vs {val_new.shape}")

  def test_forward_pass_multimodal(self):
    """Verifies end-to-end forward pass produces correct logits shape."""
    cfg = pyconfig.initialize(_get_qwen3_vl_config_args(use_m3=True, reduced=True))
    with jax.set_mesh(self.mesh):
      model = create_qwen3_vl_model(cfg, self.mesh)
      batch_size, num_frames, height, width = 1, 2, 32, 32
      channels = 3
      images = jnp.zeros((batch_size, channels, num_frames, height, width), dtype=jnp.float32)
      grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)
      tokens = jnp.array(
          [[151644, 872, 198, 151652, 151655, 151653, 151645]],
          dtype=jnp.int32,
      )
      logits = model(
          decoder_input_tokens=tokens,
          encoder_images=images,
          encoder_video_grid_thw=grid_thw,
      )
      self.assertEqual(logits.shape, (1, tokens.shape[1], cfg.vocab_size))


if __name__ == "__main__":
  unittest.main()
