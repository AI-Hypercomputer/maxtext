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

"""Unit tests for Workstream 4: Orbax save/restore and unscanned `layers_<i>` translation."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.configs.types import DType
from maxtext.m3.core.checkpoint import (
    from_flat_layer_names,
    latest_checkpoint_step,
    match_checkpoint_layer_names,
    restore_m3_checkpoint,
    save_m3_checkpoint,
    to_flat_layer_names,
)
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra.scan import apply_scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config


class CheckpointTest(absltest.TestCase):
  """Tests scanned and unscanned Orbax roundtrips and zero-copy `layers_<i>` translation."""

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir().full_path
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.cfg = Qwen3Config.mini_config(compute_dtype=DType.FLOAT32, param_dtype=DType.FLOAT32)

  def test_layer_name_translation_reuses_array_leaves(self):
    leaf = jnp.ones((4, 4), dtype=jnp.float32)
    seq_tree = {"decoder": {"layers": {"layers": {0: {"w": leaf}, 1: {"w": leaf}}}, "decoder_norm": {"scale": leaf}}}
    flat_tree = to_flat_layer_names(seq_tree)
    self.assertIn("layers_0", flat_tree["decoder"])
    self.assertIn("layers_1", flat_tree["decoder"])
    self.assertIs(flat_tree["decoder"]["layers_0"]["w"], leaf)

    matched = match_checkpoint_layer_names(seq_tree, flat_tree)
    self.assertIn("layers_0", matched["decoder"])
    roundtrip = from_flat_layer_names(flat_tree, model_state=seq_tree)
    self.assertIs(roundtrip["decoder"]["layers"]["layers"][0]["w"], leaf)

  def test_scanned_model_and_optimizer_orbax_roundtrip(self):
    with jax.set_mesh(self.mesh):
      src_model = apply_scan(create_qwen3_model(self.cfg, self.mesh, rngs=nnx.Rngs(1)), param_scan_axis=1)
      src_opt = nnx.Optimizer(src_model, optax.adamw(1e-3), wrt=nnx.Param)
      tokens = jnp.ones((2, 4), dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None, :], (2, 4))
      _, grads = nnx.value_and_grad(lambda m: jnp.mean(m(tokens, pos)))(src_model)
      src_opt.update(src_model, grads)

      save_m3_checkpoint(self.ckpt_dir, 5, src_model, src_opt)
      self.assertEqual(latest_checkpoint_step(self.ckpt_dir), 5)

      dst_model = apply_scan(create_qwen3_model(self.cfg, self.mesh, rngs=nnx.Rngs(99)), param_scan_axis=1)
      dst_opt = nnx.Optimizer(dst_model, optax.adamw(1e-3), wrt=nnx.Param)
      step = restore_m3_checkpoint(self.ckpt_dir, dst_model, dst_opt)
      self.assertEqual(step, 5)
      self.assertEqual(int(dst_opt.step[...]), 1)
      np.testing.assert_allclose(
          np.asarray(dst_model.decoder.layers.mlp.wi_0.kernel[...]),
          np.asarray(src_model.decoder.layers.mlp.wi_0.kernel[...]),
          atol=0.0,
      )

  def test_flat_unscanned_checkpoint_restores_into_nnx_sequential_model(self):
    flat_dir = self.create_tempdir().full_path
    with jax.set_mesh(self.mesh):
      src_model = create_qwen3_model(self.cfg, self.mesh, rngs=nnx.Rngs(3))
      save_m3_checkpoint(flat_dir, 2, src_model, flat_layer_names=True)

      dst_model = create_qwen3_model(self.cfg, self.mesh, rngs=nnx.Rngs(77))
      step = restore_m3_checkpoint(flat_dir, dst_model)
      self.assertEqual(step, 2)
      np.testing.assert_allclose(
          np.asarray(dst_model.decoder.layers.layers[0].mlp.wi_0.kernel[...]),
          np.asarray(src_model.decoder.layers.layers[0].mlp.wi_0.kernel[...]),
          atol=0.0,
      )


if __name__ == "__main__":
  absltest.main()
