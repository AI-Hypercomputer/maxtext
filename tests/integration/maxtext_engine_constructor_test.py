# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for what MaxTextTrainingEngine's own constructor builds.

Nothing else covers this. `compare_training_engine.py` reassigns
`engine.model`/`optimizer`/`state` right after construction -- it has to, because its
"default" path compares against a hand-written TinyDecoder that `from_pretrained` would
never produce -- and `maxtext_engine_e2e_test.py` mocks `from_pretrained` and
`create_training_optimizer` outright. So no existing test runs a step against the model
and optimizer that `__init__` actually builds.

That gap let a real bug ship: `__init__` handed the raw optax GradientTransformation from
`create_training_optimizer` straight to `TrainStateNNX`, whose `apply_gradients` calls
`optimizer.update(model, grads)` -- the nnx.Optimizer signature, not optax's
`update(updates, state, params)`. Weight updates and `save_checkpoint` were both broken on
the path production code takes.

Every test here uses the engine exactly as constructed.
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest
from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]


def _tiny_config(**overrides) -> pyconfig.HyperParameters:
  """A deliberately tiny real config: no checkpoint to load, no HF fetch."""
  argv = [
      "maxtext_engine_constructor_test.py",
      get_test_config_path("base.yml"),
      "model_name=default",
      "run_name=engine_constructor_test",
      "enable_checkpointing=False",
      # Never reach out to HuggingFace from a test.
      "convert_checkpoint_if_possible=False",
      "init_weights_seed=42",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "skip_jax_distributed_system=True",
      "gradient_accumulation_steps=1",
      # The LR schedule warms up linearly from zero, so lr(step=0) == 0 and a single
      # optimizer step would be a silent no-op. Disable warmup so one step is a valid
      # assertion, and use a large LR so the delta is unmistakable.
      "warmup_steps_fraction=0.0",
      "learning_rate=1e-2",
      # Tiny model, matching the sizing the parity suite uses for model_name=default.
      "vocab_size=8",
      "emb_dim=4",
      "mlp_dim=8",
      "micro_batch_size_to_train_on=2",
      "max_target_length=4",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


def _mesh(cfg: pyconfig.HyperParameters) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)


def _dummy_batch(cfg: pyconfig.HyperParameters) -> dict[str, jax.Array]:
  """Minimal batch shaped for `maxtext.trainers.pre_train.train.loss_fn`."""
  batch, seq = cfg.micro_batch_size_to_train_on, cfg.max_target_length
  rng = np.random.default_rng(0)
  tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(batch, seq)), dtype=jnp.int32)
  targets = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(batch, seq)), dtype=jnp.int32)
  positions = jnp.arange(seq, dtype=jnp.int32)[None, :].repeat(batch, axis=0)
  segmentation = jnp.ones((batch, seq), dtype=jnp.int32)
  return {
      "inputs": tokens,
      "targets": targets,
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets_segmentation": segmentation,
      "decoder_input_tokens": tokens,
      "decoder_target_tokens": targets,
      "decoder_loss_weights": jnp.ones((batch, seq), dtype=jnp.float32),
      "decoder_positions": positions,
  }


def _param_leaves(model) -> list[jax.Array]:
  return jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))


@pytest.mark.integration_test
class MaxTextTrainingEngineConstructorTest(absltest.TestCase):
  """Uses the engine exactly as constructed -- nothing is reassigned."""

  def test_constructor_builds_an_nnx_optimizer(self):
    """Regression: __init__ used to leave a raw optax GradientTransformation here."""
    cfg = _tiny_config()
    mesh = _mesh(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)

    self.assertIsInstance(
        engine.optimizer,
        nnx.Optimizer,
        msg=f"engine.optimizer must be an nnx.Optimizer, got {type(engine.optimizer).__name__}",
    )
    self.assertIsInstance(engine.model, nnx.Module)
    # TrainStateNNX.apply_gradients and CheckpointState both need real OptState; a raw
    # optax transform carries none.
    self.assertNotEmpty(jax.tree.leaves(nnx.state(engine.optimizer, nnx.optimizer.OptState)))

  def test_single_step_updates_weights(self):
    """One fwd_bwd + update on the as-constructed engine must move the parameters."""
    cfg = _tiny_config()
    mesh = _mesh(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      before = [jnp.asarray(p) for p in _param_leaves(engine.model)]
      step_before = engine.train_step

      engine.fwd_bwd(_dummy_batch(cfg))
      self.assertTrue(engine.has_accumulated_grads, "fwd_bwd accumulated no gradients")
      grads = jax.tree.leaves(engine._accumulated_grads)  # pylint: disable=protected-access
      grad_norm = float(jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grads)))
      self.assertGreater(grad_norm, 0.0, "gradients are identically zero")

      engine.update()
      after = [jnp.asarray(p) for p in _param_leaves(engine.model)]

    delta = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(after, before))
    self.assertGreater(delta, 0.0, "update() did not change any parameter")
    self.assertEqual(engine.train_step, step_before + 1)

  def test_save_and_restore_checkpoint_round_trip(self):
    """save_checkpoint needs a real nnx.Optimizer: CheckpointState reads its OptState."""
    output_dir = self.create_tempdir().full_path
    cfg = _tiny_config(
        enable_checkpointing=True,
        base_output_directory=output_dir,
        async_checkpointing=False,
        checkpoint_period=1,
    )
    mesh = _mesh(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.fwd_bwd(_dummy_batch(cfg))
      engine.update()

      engine.save_checkpoint(metadata={"marker": 7}, force=True)
      engine._checkpoint_manager.wait_until_finished()  # pylint: disable=protected-access

      self.assertIsNotNone(
          engine._checkpoint_manager.get_latest_step(),  # pylint: disable=protected-access
          "save_checkpoint wrote nothing",
      )
      restored = engine.restore_checkpoint()

    self.assertEqual(restored, {"marker": 7}, "checkpoint metadata did not round trip")

  def test_construction_without_a_mesh(self):
    """`from_pretrained` returns (model, mesh) when it derives the mesh itself.

    That tuple used to land in `self._model` and surface much later as
    "requires an NNX model ..., got tuple".
    """
    cfg = _tiny_config()
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=None)

    self.assertIsInstance(engine.model, nnx.Module)
    self.assertIsNotNone(engine._mesh, "engine did not adopt the derived mesh")  # pylint: disable=protected-access

  def test_adapter_wrap_requires_pad_id_and_mesh(self):
    """Without a pad id the adapter silently corrupts log-probs, so it must be enforced."""
    cfg = _tiny_config()
    mesh = _mesh(cfg)

    with self.assertRaisesRegex(ValueError, "tokenizer_pad_id"):
      maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=True)

    with self.assertRaisesRegex(ValueError, "mesh"):
      maxtext_engine.MaxTextTrainingEngine(cfg, mesh=None, wrap_with_tunix_adapter=True, tokenizer_pad_id=0)


if __name__ == "__main__":
  absltest.main()
