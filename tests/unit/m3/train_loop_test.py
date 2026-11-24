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

"""Tests for the minimal m3 training loop."""

import glob
import math
import os
import re
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.configs.types import DType, ProfilerType, Profiling, QuantizationType
from maxtext.m3.configs.pyconfig_bridge import from_pyconfig
from maxtext.m3.configs.training_config import (
    MeshConfig,
    OptimizerConfig,
    TrainingConfig,
    TransformConfig,
)
from maxtext.m3.core.sharding import DefaultSharding, create_mesh
from maxtext.m3.infra import quantization
from maxtext.m3.models.granitemoe.granitemoe_config import GraniteMoeModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.train import loop
from tests.utils.test_helpers import get_test_config_path


def _tiny_config(output_dir: str, **overrides):
  """A 2-layer qwen3-0.6b variant small enough to train on CPU in seconds.

  The `base_*` keys are what the model YAML sets; overriding the derived names
  (`emb_dim`, `mlp_dim`, ...) is silently ignored.
  """
  args = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "scan_layers": "false",
      "override_model_config": "true",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "2",
      "max_target_length": "16",
      "steps": "3",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_loop_test",
  }
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


class TrainLoopTest(absltest.TestCase):
  """Runs the loop end to end on a tiny model."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_tiny_config_takes_effect(self):
    """Guards the `base_*` overrides; a full-size model here would make the suite slow."""
    config = _tiny_config(self.output_dir)
    self.assertEqual(config.emb_dim, 64)
    self.assertEqual(config.num_decoder_layers, 2)

  def test_synthetic_batch_targets_are_shifted_inputs(self):
    config = _tiny_config(self.output_dir)
    batch = loop.make_synthetic_batch(config, create_mesh(), loop.batch_spec(DefaultSharding()), jax.random.key(0))
    inputs, targets = np.asarray(batch["inputs"]), np.asarray(batch["targets"])
    self.assertEqual(inputs.shape, (config.global_batch_size_to_train_on, config.max_target_length))
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])

  def test_train_loop_returns_finite_loss(self):
    loss = loop.train_loop(_tiny_config(self.output_dir))
    self.assertTrue(math.isfinite(loss), f"loss={loss}")

  def test_train_loop_writes_xplane_profile(self):
    config = _tiny_config(self.output_dir, profiler="xplane", skip_first_n_steps_for_profiler=1, profiler_steps=1)
    loop.train_loop(config)
    traces = glob.glob(os.path.join(config.tensorboard_dir, "**", "*.xplane.pb"), recursive=True)
    self.assertNotEmpty(traces, f"no .xplane.pb under {config.tensorboard_dir}")

  def test_train_step_ops_carry_named_scopes(self):
    for overrides in ({}, {"quantization": "int8", "use_qwix_quantization": "true"}):
      with self.subTest(**overrides):
        config = _tiny_config(self.output_dir, **overrides)
        mesh = create_mesh()
        with jax.set_mesh(mesh):
          model, optimizer = loop.setup_model(config, mesh)
          batch = loop.make_synthetic_batch(config, mesh, loop.batch_spec(DefaultSharding()), jax.random.key(0))
          hlo = loop.train_step.lower(model, optimizer, batch).compile().as_text()
          if overrides:
            # Scopes go on after qwix; its matmuls must stay quantized.
            example = quantization.example_inputs(config)
            self.assertGreater(quantization.quantized_matmuls(model, jnp.int8, *example), 0)
        # Scopes entered under `value_and_grad` read `jvp(decoder)/...`; drop the transform wrappers.
        names = {re.sub(r"\w+\(|\)", "", n) for n in re.findall(r'op_name="([^"]*)"', hlo)}
        for scope in ("decoder/layers/0/mlp/wi_0/", "cross_entropy/", "optimizer/"):
          self.assertTrue(any(scope in n for n in names), f"no op under {scope}")

  def test_train_loop_avoids_per_step_graph_traversal(self):
    with (
        mock.patch.object(nnx, "state", wraps=nnx.state) as state_spy,
        mock.patch.object(nnx.graphlib, "flatten", wraps=nnx.graphlib.flatten) as flatten_spy,
    ):
      loop.train_loop(_tiny_config(self.output_dir, steps=2))
      state_calls_2, flatten_calls_2 = state_spy.call_count, flatten_spy.call_count

      state_spy.reset_mock()
      flatten_spy.reset_mock()
      loop.train_loop(_tiny_config(self.output_dir, steps=5))
      state_calls_5, flatten_calls_5 = state_spy.call_count, flatten_spy.call_count

    self.assertEqual(state_calls_2, state_calls_5)
    self.assertEqual(flatten_calls_2, flatten_calls_5)

  def test_matmul_precision_reaches_the_step(self):
    """`matmul_precision` is baked into every `dot_general` of the step at trace time."""
    seen = {}
    original = nnx.cached_partial

    def precisions(jaxpr):
      for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
          yield str(eqn.params["precision"])
        for value in eqn.params.values():
          if isinstance(value, jex.core.ClosedJaxpr):
            yield from precisions(value.jaxpr)

    def spy(fn, model, optimizer):
      # Trace the forward pass functionally (the cached step mutates state in place, which a bare
      # `make_jaxpr` rejects); it runs inside the loop's `default_matmul_precision` like the step does.
      graphdef, state = nnx.split(model)
      step_fn = original(fn, model, optimizer)

      def forward(state, batch):
        return loop.loss_fn(nnx.merge(graphdef, state), batch)

      def traced(batch):
        seen["precisions"] = set(precisions(jax.make_jaxpr(forward)(state, batch).jaxpr))
        return step_fn(batch)

      return traced

    for setting, expected in (("highest", "HIGHEST"), ("default", "DEFAULT")):
      with mock.patch.object(nnx, "cached_partial", spy):
        loop.train_loop(_tiny_config(self.output_dir, dtype="float32", matmul_precision=setting, steps=1))
      self.assertNotEmpty(seen["precisions"])
      self.assertTrue(all(expected in p for p in seen["precisions"]), (setting, seen["precisions"]))

  def test_rejects_grad_dtype_other_than_float32(self):
    with self.assertRaisesRegex(ValueError, "grad_dtype=bfloat16"):
      loop.train_loop(_tiny_config(self.output_dir, grad_dtype="bfloat16"))

  def test_rejects_gradient_accumulation(self):
    with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=2"):
      loop.train_loop(_tiny_config(self.output_dir, gradient_accumulation_steps=2))

  def test_rejects_batch_that_does_not_split_over_mesh(self):
    """`per_device_batch_size=0.5` never splits: batch 0 on one device, n // 2 on n devices."""
    with self.assertRaisesRegex(ValueError, "does not split over"):
      loop.train_loop(_tiny_config(self.output_dir, per_device_batch_size=0.5))

  def _tiny_typed_configs(
      self,
      *,
      mesh: MeshConfig | None = None,
      transforms: TransformConfig | None = None,
      profiling: Profiling | None = None,
      **train_overrides,
  ) -> tuple[Qwen3Config, TrainingConfig]:
    """Builds a tiny typed `(Qwen3Config, TrainingConfig)` pair for CPU tests."""
    model_cfg = Qwen3Config(
        model_name="qwen3-0.6b",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        float32_logits=False,
    )
    train_kwargs = {
        "run_name": "m3_loop_typed_test",
        "base_output_directory": self.output_dir,
        "per_device_batch_size": 2.0,
        "max_target_length": 16,
        "steps": 3,
        "mesh": mesh or MeshConfig(),
        "transforms": transforms or TransformConfig(scan_layers=False),
        "optimizer": OptimizerConfig(),
        "profiling": profiling or Profiling(),
    }
    train_kwargs.update(train_overrides)
    train_cfg = TrainingConfig(**train_kwargs)
    return model_cfg, train_cfg

  def test_typed_train_loop_matches_pyconfig_loss(self):
    # `base.yml` defaults `float32_logits=false`, whereas `Qwen3Config` defaults to `true`;
    # pass it explicitly so both paths run the same attention softmax precision.
    legacy_cfg = _tiny_config(self.output_dir, steps=2, float32_logits="true")
    bridged_model, bridged_train = from_pyconfig(legacy_cfg)
    legacy_loss = loop.train_loop(legacy_cfg)
    typed_loss = loop.train_loop(bridged_model, bridged_train)
    self.assertTrue(math.isfinite(typed_loss))
    self.assertAlmostEqual(typed_loss, legacy_loss, places=6)

    # Explicit sharding executes cleanly with zero-YAML typed configs.
    model_cfg, train_cfg = self._tiny_typed_configs(mesh=MeshConfig(shard_mode="explicit"), steps=2)
    self.assertTrue(math.isfinite(loop.train_loop(model_cfg, train_cfg)))

  def test_typed_xplane_profile_quantization_and_named_scopes_toggle(self):
    model_cfg, prof_train = self._tiny_typed_configs(
        run_name="typed_prof",
        steps=2,
        profiling=Profiling(profiler=ProfilerType.XPLANE, skip_first_n_steps_for_profiler=0, profiler_steps=1),
    )
    loop.train_loop(model_cfg, prof_train)
    tb_dir = os.path.join(self.output_dir, "typed_prof", "tensorboard")
    traces = glob.glob(os.path.join(tb_dir, "**", "*.xplane.pb"), recursive=True)
    self.assertNotEmpty(traces, f"no .xplane.pb under {tb_dir}")

    # `quantization=int8` works via `TransformConfig` and `named_scopes` toggles wrapping.
    mesh = create_mesh()
    with jax.set_mesh(mesh):
      _, tx_on = self._tiny_typed_configs(
          transforms=TransformConfig(scan_layers=False, quantization=QuantizationType.INT8, named_scopes=True)
      )
      model_q, opt_q = loop.setup_model(model_cfg, mesh, train_cfg=tx_on)
      batch = loop.make_synthetic_batch(
          model_cfg, mesh, loop.batch_spec(DefaultSharding()), jax.random.key(0), train_cfg=tx_on
      )
      hlo_on = loop.train_step.lower(model_q, opt_q, batch).compile().as_text()
      self.assertGreater(quantization.quantized_matmuls(model_q, jnp.int8, batch["inputs"], batch["positions"]), 0)
      names_on = {re.sub(r"\w+\(|\)", "", n) for n in re.findall(r'op_name="([^"]*)"', hlo_on)}
      self.assertTrue(any("decoder/layers/0/mlp/wi_0/" in n for n in names_on))

      _, tx_off = self._tiny_typed_configs(transforms=TransformConfig(scan_layers=False, named_scopes=False))
      model_off, opt_off = loop.setup_model(model_cfg, mesh, train_cfg=tx_off)
      hlo_off = loop.train_step.lower(model_off, opt_off, batch).compile().as_text()
      self.assertNotIn("decoder/layers/0/mlp/wi_0/", hlo_off)

  def test_typed_validation_errors(self):
    model_cfg, train_cfg = self._tiny_typed_configs()
    with self.assertRaises(TypeError):
      loop.train_loop(model_cfg)
    with self.assertRaises(TypeError):
      loop.train_loop(_tiny_config(self.output_dir), train_cfg)
    with self.assertRaises(TypeError):
      loop.train_loop(model_cfg, "not_a_training_config")  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, "grad_dtype=bfloat16"):
      loop.train_loop(model_cfg, self._tiny_typed_configs(grad_dtype=DType.BFLOAT16)[1])
    with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=2"):
      loop.train_loop(model_cfg, self._tiny_typed_configs(gradient_accumulation_steps=2)[1])
    with self.assertRaisesRegex(ValueError, "does not split over"):
      loop.train_loop(model_cfg, self._tiny_typed_configs(per_device_batch_size=0.5)[1])
    with self.assertRaisesRegex(ValueError, "shard_mode='explicit'"):
      loop.train_loop(
          model_cfg,
          self._tiny_typed_configs(
              mesh=MeshConfig(shard_mode="explicit"),
              transforms=TransformConfig(scan_layers=False, quantization=QuantizationType.INT8),
          )[1],
      )
    with self.assertRaisesRegex(ValueError, "shard_mode"):
      loop.train_loop(
          GraniteMoeModelConfig(),
          self._tiny_typed_configs(mesh=MeshConfig(shard_mode="explicit"))[1],
      )

  def test_typed_granitemoe_runs_end_to_end(self):
    gm_cfg = GraniteMoeModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        compute_dtype=DType.FLOAT32,
    )
    train_cfg = TrainingConfig(
        run_name="gm_typed_test",
        base_output_directory=self.output_dir,
        per_device_batch_size=1.0,
        max_target_length=8,
        steps=1,
        transforms=TransformConfig(scan_layers=False),
    )
    self.assertTrue(math.isfinite(loop.train_loop(gm_cfg, train_cfg)))

  def test_parse_argv_dispatches_zero_yaml_and_legacy_yaml(self):
    cli_argv = [
        "loop.py",
        "--model_name=qwen3-0.6b",
        "--model.vocab_size=256",
        "--model.hidden_size=64",
        "--model.intermediate_size=128",
        "--model.num_hidden_layers=2",
        "--model.num_attention_heads=4",
        "--model.num_key_value_heads=2",
        "--model.head_dim=16",
        "--transforms.scan_layers=false",
        "--per_device_batch_size=2",
        "--max_target_length=16",
        "--steps=1",
        f"--base_output_directory={self.output_dir}",
        "--run_name=cli_test",
    ]
    model_cfg, train_cfg = loop.parse_argv(cli_argv)
    self.assertIsInstance(model_cfg, Qwen3Config)
    self.assertEqual(model_cfg.hidden_size, 64)
    self.assertFalse(train_cfg.transforms.scan_layers)
    self.assertEqual(train_cfg.steps, 1)
    loop.main(cli_argv)

    yml_argv = [
        "loop.py",
        get_test_config_path(),
        "model_name=qwen3-0.6b",
        "override_model_config=true",
        "base_num_decoder_layers=2",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
        "scan_layers=false",
        "steps=2",
        f"base_output_directory={self.output_dir}",
        "run_name=yml_test",
        "skip_jax_distributed_system=true",
    ]
    yml_model, yml_train = loop.parse_argv(yml_argv)
    self.assertIsInstance(yml_model, Qwen3Config)
    self.assertEqual(yml_model.hidden_size, 64)
    self.assertEqual(yml_train.steps, 2)

    # Non-`.py` program tokens and flag-only argv sequences also route to `ArgumentParser`.
    bin_model, bin_train = loop.parse_argv(["maxtext-loop", "--model_name=qwen3-0.6b", "--steps=4"])
    self.assertIsInstance(bin_model, Qwen3Config)
    self.assertEqual(bin_train.steps, 4)
    bare_model, bare_train = loop.parse_argv(["--model_name=qwen3-0.6b", "--steps=5"])
    self.assertIsInstance(bare_model, Qwen3Config)
    self.assertEqual(bare_train.steps, 5)


if __name__ == "__main__":
  absltest.main()
