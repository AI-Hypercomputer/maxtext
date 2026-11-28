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

"""Tests for the m3 diffusion training loop (`loop_diffusers.py`) on a tiny Wan 2.1 transformer."""

import math
import os

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext_diffusers import restore_diffusers_checkpoint
from maxtext.m3.configs import config as config_lib
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.m3.diffuser_pipelines.pipeline_utils import _to_int_path
from maxtext.m3.diffuser_pipelines.wan_pipeline import _wan_model_kwargs, model_init_config_for_wan
from maxtext.m3.models.diffuser_schedulers.scheduling_flow_match import FlaxFlowMatchScheduler
from maxtext.m3.models.wan import WanModel, utils_wan
from maxtext.m3.models.wan.utils_wan import WAN_MESH_AXES, WanSharding
from maxtext.m3.train import loop_diffusers

# pylint: enable=wrong-import-position

# Tiny Wan 2.1 transformer (real 14B: 40 layers, 40 heads x 128, in_channels 16, text_dim 4096).
_CONFIG = {
    "attention_head_dim": 8,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "ffn_dim": 32,
    "freq_dim": 16,
    "in_channels": 4,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 4,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 64,
    "text_dim": 16,
}
_BATCH, _FRAMES, _HEIGHT, _WIDTH, _TEXT_LEN = 2, 2, 4, 4, 8
_MESH = Mesh(np.array(jax.devices()[:1]).reshape((1, 1, 1, 1)), WAN_MESH_AXES)  # the yml's mesh axes, one device
_SHARDING = WanSharding.from_rules(utils_wan.yml_rules("logical_axis_rules"))


def _make_model(seed=0, **overrides):
  kwargs = {**_CONFIG, "attention": "dot_product", "sharding": _SHARDING, "scan_layers": False, **overrides}
  return WanModel(rngs=nnx.Rngs(seed), **kwargs)


def _make_batch(seed=0):
  k1, k2 = jax.random.split(jax.random.key(seed))
  latents = jax.random.normal(k1, (_BATCH, _CONFIG["in_channels"], _FRAMES, _HEIGHT, _WIDTH), jnp.float32)
  text = jax.random.normal(k2, (_BATCH, _TEXT_LEN, _CONFIG["text_dim"]), jnp.float32)
  return {"latents": latents, "encoder_hidden_states": text}


def _loss_and_grads(model, batch, scheduler, seed=1):
  return nnx.value_and_grad(loop_diffusers.loss_fn)(model, batch, jax.random.key(seed), scheduler)


def _tiny_config(output_dir: str, **overrides):
  """The Wan yml with the tiny transformer, a 4-step synthetic run on the CPU devices (FSDP over all of them)."""
  args = {
      "attention": "dot_product",
      "weights_dtype": "float32",
      "activations_dtype": "float32",
      "scan_layers": "false",
      "skip_jax_distributed_system": "true",
      "ici_fsdp_parallelism": str(len(jax.devices())),
      "ici_context_parallelism": "1",
      "per_device_batch_size": "1",
      "height": "32",
      "width": "32",
      "num_frames": "5",
      "max_sequence_length": "8",
      "max_train_steps": "4",
      "log_period": "2",
      "learning_rate": "1e-3",
      "output_dir": output_dir,
  }
  args.update({k: str(v) for k, v in overrides.items()})
  yml = os.path.join(os.path.dirname(utils_wan.__file__), "configs", "wan2.1-t2v-14b.yml")
  # A VAE with the tiny transformer's latent channels (Wan 2.1 compression: 4x in time, 8x in space).
  return config_lib.initialize(
      ["", yml] + [f"{k}={v}" for k, v in args.items()],
      transformer_config=dict(_CONFIG),
      vae_config={"z_dim": _CONFIG["in_channels"]},
  )


class FlowMatchTrainingTest(parameterized.TestCase):
  """The loss and train step on the tiny transformer (one device)."""

  def setUp(self):
    super().setUp()
    # The model's sharding constraints resolve against the mesh in scope (as in WanPipeline).
    self.enter_context(jax.set_mesh(_MESH))
    self.scheduler, _ = FlaxFlowMatchScheduler.from_config({"shift": 5.0})
    self.batch = _make_batch()

  def test_grads_finite_and_nonzero_for_every_param(self):
    loss, grads = _loss_and_grads(_make_model(), self.batch, self.scheduler)
    self.assertTrue(np.isfinite(float(loss)))
    zero = []
    for path, leaf in nnx.to_flat_state(grads):
      g = np.asarray(leaf[...])
      self.assertTrue(np.all(np.isfinite(g)), path)
      if not np.any(g):
        zero.append("/".join(map(str, path)))
    self.assertEmpty(zero, f"params with all-zero grads: {zero}")

  @parameterized.named_parameters(("full", "FULL"), ("matmul_without_batch", "MATMUL_WITHOUT_BATCH"))
  def test_remat_matches_no_remat(self, policy):
    loss_ref, grads_ref = _loss_and_grads(_make_model(remat_policy=None), self.batch, self.scheduler)
    loss, grads = _loss_and_grads(
        _make_model(remat_policy=loop_diffusers.remat_policy(policy)), self.batch, self.scheduler
    )
    np.testing.assert_allclose(float(loss), float(loss_ref), rtol=1e-6)
    for (path, g), (_, g_ref) in zip(nnx.to_flat_state(grads), nnx.to_flat_state(grads_ref)):
      np.testing.assert_allclose(np.asarray(g[...]), np.asarray(g_ref[...]), rtol=1e-5, atol=1e-6, err_msg=str(path))

  def test_remat_policy_names(self):
    self.assertIsNone(loop_diffusers.remat_policy("none"))
    self.assertIs(loop_diffusers.remat_policy("full"), jax.checkpoint_policies.nothing_saveable)
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      loop_diffusers.remat_policy("custom")

  def test_dropout_is_stochastic(self):
    model = _make_model(dropout=0.5)
    rng = jax.random.key(3)
    first = loop_diffusers.loss_fn(model, self.batch, rng, self.scheduler)
    second = loop_diffusers.loss_fn(model, self.batch, jax.random.key(4), self.scheduler)
    self.assertNotEqual(float(first), float(second))

  def test_train_step_reduces_loss_on_fixed_batch(self):
    model = _make_model()
    trainer = config_lib.TrainerConfig(learning_rate=1e-2, max_train_steps=20)
    tx, schedule = loop_diffusers.make_optimizer(trainer, config_lib.OptimizerConfig())
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_step = loop_diffusers.make_train_step(
        self.scheduler,
        schedule,
        loop_diffusers.place_state(model, (nnx.Param,), _MESH),
        loop_diffusers.place_state(optimizer, (), _MESH),
    )
    rng = jax.random.key(7)
    first = float(train_step(model, optimizer, self.batch, rng)["loss"])
    for _ in range(20):
      last = float(train_step(model, optimizer, self.batch, rng)["loss"])
    self.assertLess(last, first)

  def test_optimizer_warmup_then_constant(self):
    trainer = config_lib.TrainerConfig(learning_rate=1e-3, max_train_steps=100)
    # Default warmup_steps_fraction 0.1: 10 warmup steps.
    _, schedule = loop_diffusers.make_optimizer(trainer, config_lib.OptimizerConfig())
    self.assertEqual(float(schedule(0)), 0.0)
    np.testing.assert_allclose(float(schedule(5)), 5e-4)
    np.testing.assert_allclose(float(schedule(50)), 1e-3)


class TrainKeysTest(absltest.TestCase):
  """Command-line overrides reach the typed fields; bad values fail fast."""

  def test_overrides_are_typed(self):
    config = _tiny_config("/tmp/unused", adam_b1=0.5, warmup_steps_fraction=0.0, dataset_type="synthetic_repeat")
    keys = loop_diffusers.TrainKeys.from_config(config)
    self.assertEqual((keys.adam_b1, keys.warmup_steps_fraction, keys.dataset_type), (0.5, 0.0, "synthetic_repeat"))
    self.assertEqual(keys.transformer_config["num_layers"], _CONFIG["num_layers"])
    self.assertEqual((config.trainer.learning_rate, config.trainer.max_train_steps), (1e-3, 4))

  def test_rejects_unknown_dataset_type(self):
    with self.assertRaisesRegex(ValueError, "dataset_type"):
      loop_diffusers.TrainKeys.from_config(_tiny_config("/tmp/unused", dataset_type="parquet"))


class TrainLoopTest(absltest.TestCase):
  """Runs `train_loop` end to end on the tiny transformer over the CPU devices."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_from_scratch_then_from_its_export(self):
    """From scratch: finite loss and an export; then restoring that export continues from the trained params."""
    scratch = _tiny_config(self.output_dir)
    loss = loop_diffusers.train_loop(scratch)
    self.assertTrue(math.isfinite(loss), f"loss={loss}")
    export_dir = os.path.join(self.output_dir, "export")
    self.assertEqual(loop_diffusers.restore_config(export_dir, "transformer")["num_layers"], _CONFIG["num_layers"])

    resumed_dir = self.create_tempdir().full_path
    resumed = _tiny_config(resumed_dir, checkpoint_directory=export_dir, max_train_steps=1)
    mesh = mesh_from_config(resumed)
    kwargs = _wan_model_kwargs(resumed, model_init_config_for_wan(_CONFIG))
    with jax.set_mesh(mesh):
      model = loop_diffusers.create_model(resumed, WanModel, kwargs, mesh, export_dir)
    # The restored params equal the exported ones, read back by the converter's (independent, replicated) reader.
    exported = flatten_dict(restore_diffusers_checkpoint(export_dir)["transformer_state"])
    exported = {_to_int_path(path[:-1] if path[-1] == "value" else path): v for path, v in exported.items()}
    for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
      np.testing.assert_array_equal(np.asarray(var[...]), np.asarray(exported[path]), err_msg=str(path))
    # Training in bfloat16 from that export still stores every param in the source export's dtype (float32 here).
    resumed_bf16 = _tiny_config(resumed_dir, checkpoint_directory=export_dir, max_train_steps=1, weights_dtype="bfloat16")
    self.assertTrue(math.isfinite(loop_diffusers.train_loop(resumed_bf16)))
    re_exported = flatten_dict(restore_diffusers_checkpoint(os.path.join(resumed_dir, "export"))["transformer_state"])
    self.assertEqual({v.dtype for v in re_exported.values()}, {np.dtype(np.float32)})

  def test_synthetic_repeat_overfits(self):
    """One repeated batch: the loss drops (the convergence smoke test of the real run, at tiny scale)."""
    config = _tiny_config(self.output_dir, dataset_type="synthetic_repeat", max_train_steps=30, log_period=30)
    first = _tiny_config(self.create_tempdir().full_path, dataset_type="synthetic_repeat", max_train_steps=1)
    self.assertLess(loop_diffusers.train_loop(config), loop_diffusers.train_loop(first))

  def test_rejects_gradient_accumulation(self):
    with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=2"):
      loop_diffusers.train_loop(_tiny_config(self.output_dir, gradient_accumulation_steps=2))


if __name__ == "__main__":
  absltest.main()
