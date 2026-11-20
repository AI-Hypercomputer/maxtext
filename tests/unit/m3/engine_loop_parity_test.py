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

"""Unit tests for 3-step numerical parity between `MaxTextTrainingEngine` and `m3/train/loop.py` (`M4E-14`)."""

import contextlib
import sys
import types
from typing import Any
import unittest

from absl.testing import absltest
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.m3.models.granitemoe import utils_granitemoe
from maxtext.m3.train import loop
import numpy as np
from tests.utils.test_helpers import get_test_config_path

_TE_PKG_NAME = f"{pyconfig.__name__.rsplit('.configs.', 1)[0]}.training_engine"

try:
  from maxtext.training_engine import abstract_engine
  from maxtext.training_engine import maxtext_engine

  HAS_TUNIX = True
except ImportError:
  abstract_engine = None
  maxtext_engine = None
  HAS_TUNIX = False


@contextlib.contextmanager
def _with_tunix_stub_if_needed():
  """Provides minimal Tunix dataclass stubs when running in `cpu-unit` without `tunix` installed."""
  if HAS_TUNIX:
    yield maxtext_engine, abstract_engine
    return

  tunix_mod = types.ModuleType("tunix")
  dist_rl = types.ModuleType("tunix.dist_rl")
  common = types.ModuleType("tunix.dist_rl.common")
  datatypes = types.ModuleType("tunix.dist_rl.common.datatypes")
  sft = types.ModuleType("tunix.sft")
  sft_utils = types.ModuleType("tunix.sft.utils")

  class _TrainerPayload:
    pass

  @flax.struct.dataclass
  class _RLTrainerPayload(_TrainerPayload):
    pass

  @flax.struct.dataclass
  class _LossOutput:
    primary_loss: object = None
    aux_metrics: object = None

  @flax.struct.dataclass
  class _WeightedMetric:
    """Minimal WeightedMetric stub for torch-free MaxTextTrainingEngine tests."""

    unreduced_sum: object
    denominator: object
    eps: float = flax.struct.field(pytree_node=False, default=1e-8)
    min_denom: float = flax.struct.field(pytree_node=False, default=0.0)

    @classmethod
    def from_mean_and_weight(cls, mean, weight):
      return cls(unreduced_sum=mean * weight, denominator=weight)

    def compute(self):
      return self.unreduced_sum / self.denominator

    def compute_scale(self):
      return self.denominator

  datatypes.TrainerPayload = _TrainerPayload
  datatypes.RLTrainerPayload = _RLTrainerPayload
  sft_utils.LossOutput = _LossOutput
  sft_utils.WeightedMetric = _WeightedMetric
  stub_modules = {
      "tunix": tunix_mod,
      "tunix.dist_rl": dist_rl,
      "tunix.dist_rl.common": common,
      "tunix.dist_rl.common.datatypes": datatypes,
      "tunix.sft": sft,
      "tunix.sft.utils": sft_utils,
  }
  engine_submodules = ("abstract_engine", "maxtext_engine", "metrics")
  tracked_keys = (*stub_modules, *(f"{_TE_PKG_NAME}.{attr}" for attr in engine_submodules))
  saved_sys = {key: sys.modules[key] for key in tracked_keys if key in sys.modules}
  pkg = sys.modules.get(_TE_PKG_NAME)
  saved_pkg_attrs = {attr: getattr(pkg, attr) for attr in engine_submodules if pkg is not None and hasattr(pkg, attr)}
  try:
    sys.modules.update(stub_modules)
    for attr in engine_submodules:
      sys.modules.pop(f"{_TE_PKG_NAME}.{attr}", None)
      if pkg is not None and hasattr(pkg, attr):
        delattr(pkg, attr)
    from maxtext.training_engine import abstract_engine as stubbed_abs  # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import maxtext_engine as stubbed_eng  # pylint: disable=import-outside-toplevel

    yield stubbed_eng, stubbed_abs
  finally:
    for key in tracked_keys:
      if key in saved_sys:
        sys.modules[key] = saved_sys[key]
      else:
        sys.modules.pop(key, None)
    pkg_after = sys.modules.get(_TE_PKG_NAME)
    if pkg_after is not None:
      for attr in engine_submodules:
        if attr in saved_pkg_attrs:
          setattr(pkg_after, attr, saved_pkg_attrs[attr])
        elif hasattr(pkg_after, attr):
          delattr(pkg_after, attr)


def _make_parity_cfg(output_dir: str, model_name: str = "qwen3-0.6b", **overrides: Any):
  """Builds a small 4-CPU m3 config for engine vs loop parity tests."""
  argv = [
      "",
      get_test_config_path("base.yml"),
      f"base_output_directory={output_dir}",
      "run_name=m3_engine_loop_parity_test",
      f"model_name={model_name}",
      "use_m3_model=true",
      "override_model_config=true",
      "base_emb_dim=64",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=16",
      "vocab_size=256",
      "ici_fsdp_parallelism=2",
      "ici_tensor_parallelism=2",
      "per_device_batch_size=1.0",
      "max_target_length=8",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "opt_type=adamw",
      "learning_rate=1e-3",
      "warmup_steps_fraction=0.0",
      "learning_rate_final_fraction=1.0",
      "adam_weight_decay=0.01",
      "gradient_clipping_threshold=1.0",
      "enable_checkpointing=false",
      "use_weight_converter=false",
      "skip_jax_distributed_system=true",
  ]
  if model_name == "granite-3.0-1b-a400m":
    argv.extend(
        [
            "base_mlp_dim=64",
            "base_moe_mlp_dim=64",
            "num_experts=4",
            "num_experts_per_tok=2",
        ]
    )
  else:
    argv.append("base_mlp_dim=128")
  for key, value in overrides.items():
    argv.append(f"{key}={value}")
  return pyconfig.initialize(argv)


def _clone_params(model: nnx.Module) -> nnx.State:
  """Returns an independent copy of `model`'s `nnx.Param` state so buffer donation cannot alias."""
  return jax.tree.map(
      lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else x,
      nnx.state(model, nnx.Param),
  )


def _run_three_step_comparison(
    tc: absltest.TestCase,
    eng_mod: Any,
    abs_mod: Any,
    loop_cfg: Any,
    eng_cfg: Any,
) -> None:
  """Runs 3 synthetic train steps comparing `MaxTextTrainingEngine` vs `m3/train/loop.py`."""
  mesh = mesh_from_config(loop_cfg)
  with jax.set_mesh(mesh):
    loop_model, loop_opt = loop.setup_model(loop_cfg, mesh)
    engine = eng_mod.MaxTextTrainingEngine(eng_cfg, mesh=mesh, wrap_with_tunix_adapter=False)
    nnx.update(engine.model, _clone_params(loop_model))
    _ = engine.state

    def _engine_loss_fn(model, config, data, dropout_rng=None, params=None, is_train=True):
      del config, dropout_rng, params, is_train
      batch = {
          "inputs": data["inputs"],
          "positions": data["inputs_position"],
          "targets": data["targets"],
      }
      mean_loss = loop.loss_fn(model, batch)
      denom = jnp.asarray(data["inputs"].size, dtype=jnp.float32)
      return abs_mod.WeightedMetric(unreduced_sum=mean_loss * denom, denominator=denom), {}

    engine.with_loss_fn(_engine_loss_fn, has_aux=True)
    engine.with_gen_model_input_fn(None)

    for step in range(3):
      toks = jax.random.randint(jax.random.PRNGKey(100 + step), (4, 8), 1, 256, dtype=jnp.int32)
      targets = jax.random.randint(jax.random.PRNGKey(200 + step), (4, 8), 1, 256, dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (4, 8))
      seg = jnp.ones((4, 8), dtype=jnp.int32)

      loop_batch = {"inputs": toks, "positions": pos, "targets": targets}
      eng_batch = {
          "inputs": toks,
          "targets": targets,
          "inputs_position": pos,
          "targets_position": pos,
          "inputs_segmentation": seg,
          "targets_segmentation": seg,
      }

      # Verify single-step gradient parity before applying the optimizer update on step 0.
      if step == 0:
        _, loop_grads = nnx.value_and_grad(loop.loss_fn)(loop_model, loop_batch)
      loop_loss = float(loop.train_step(loop_model, loop_opt, loop_batch))

      engine.fwd_bwd(eng_batch)
      eng_loss = float(engine._cached_losses[-1].compute())  # pylint: disable=protected-access
      if step == 0:
        eng_denom = float(engine._accumulated_denominator)  # pylint: disable=protected-access
        for (p1, g1), (p2, g2) in zip(
            nnx.to_flat_state(loop_grads),
            nnx.to_flat_state(engine._accumulated_grads),  # pylint: disable=protected-access
        ):
          tc.assertEqual(p1, p2)
          np.testing.assert_allclose(
              np.asarray(g2[...]) / eng_denom,
              np.asarray(g1[...]),
              rtol=1e-5,
              atol=1e-5,
              err_msg=f"Step 0 gradient mismatch at {p1}",
          )
      engine.update()

      np.testing.assert_allclose(eng_loss, loop_loss, rtol=1e-5, atol=1e-5, err_msg=f"Loss mismatch at step {step}")
      for (p1, v1), (p2, v2) in zip(
          nnx.to_flat_state(nnx.state(loop_model, nnx.Param)),
          nnx.to_flat_state(nnx.state(engine.model, nnx.Param)),
      ):
        tc.assertEqual(p1, p2)
        np.testing.assert_allclose(
            np.asarray(v2[...]),
            np.asarray(v1[...]),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Updated parameter mismatch at step {step}, path {p1}",
        )


class EngineLoopParityTest(absltest.TestCase):
  """Verifies 3-step loss, gradient, and parameter parity between `MaxTextTrainingEngine` and `loop.py`."""

  @unittest.skipUnless(HAS_TUNIX, "tunix not installed")
  def test_three_step_parity_with_native_tunix(self):
    out_dir = self.create_tempdir().full_path
    cfg = _make_parity_cfg(out_dir)
    _run_three_step_comparison(self, maxtext_engine, abstract_engine, cfg, cfg)

  def test_three_step_parity_tunix_free_and_negative_controls(self):
    out_dir = self.create_tempdir().full_path
    cfg = _make_parity_cfg(out_dir)
    with _with_tunix_stub_if_needed() as (eng_mod, abs_mod):
      _run_three_step_comparison(self, eng_mod, abs_mod, cfg, cfg)

      # Negative control 1: perturbed learning rate (`2e-3` vs `1e-3`) breaks updated parameter parity.
      perturbed_lr_cfg = _make_parity_cfg(out_dir, learning_rate="2e-3")
      with self.assertRaises(AssertionError):
        _run_three_step_comparison(self, eng_mod, abs_mod, cfg, perturbed_lr_cfg)

  def test_negative_control_dropped_moe_aux_loss_breaks_engine_step_parity(self):
    out_dir = self.create_tempdir().full_path
    gm_cfg = _make_parity_cfg(out_dir, model_name="granite-3.0-1b-a400m", load_balance_loss_weight="0.5")
    mesh = mesh_from_config(gm_cfg)
    with _with_tunix_stub_if_needed() as (eng_mod, _), jax.set_mesh(mesh):
      engine_with_aux = eng_mod.MaxTextTrainingEngine(gm_cfg, mesh=mesh, wrap_with_tunix_adapter=False)
      engine_dropped_aux = eng_mod.MaxTextTrainingEngine(gm_cfg, mesh=mesh, wrap_with_tunix_adapter=False)
      nnx.update(engine_dropped_aux.model, _clone_params(engine_with_aux.model))

      # `engine_with_aux` uses the registered GraniteMoe loss adapter (which includes `moe_lb_loss`).
      # Override `engine_dropped_aux` with a loss that drops `moe_lb_loss` (`router_aux_loss_coef=0.0`).
      def _dropped_aux_loss_fn(model, config, data, dropout_rng=None, params=None, *, is_train=True):
        del config
        no_aux_cfg = types.SimpleNamespace(router_aux_loss_coef=0.0)
        return utils_granitemoe.granitemoe_engine_loss_fn(model, no_aux_cfg, data, dropout_rng, params, is_train=is_train)

      engine_dropped_aux.with_loss_fn(_dropped_aux_loss_fn, has_aux=True)

      toks = jax.random.randint(jax.random.PRNGKey(7), (4, 8), 1, 256, dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (4, 8))
      seg = jnp.ones((4, 8), dtype=jnp.int32)
      batch = {
          "inputs": toks,
          "targets": toks,
          "inputs_position": pos,
          "targets_position": pos,
          "inputs_segmentation": seg,
          "targets_segmentation": seg,
      }
      engine_with_aux.fwd_bwd(batch)
      loss_with_aux = float(engine_with_aux._cached_losses[-1].compute())  # pylint: disable=protected-access
      engine_with_aux.update()

      engine_dropped_aux.fwd_bwd(batch)
      loss_dropped_aux = float(engine_dropped_aux._cached_losses[-1].compute())  # pylint: disable=protected-access
      engine_dropped_aux.update()

      self.assertGreater(abs(loss_with_aux - loss_dropped_aux), 1e-4)
      router_with_aux = np.asarray(engine_with_aux.model.layers.block_sparse_moe.router.weight.get_raw_value())
      router_dropped_aux = np.asarray(engine_dropped_aux.model.layers.block_sparse_moe.router.weight.get_raw_value())
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(router_dropped_aux, router_with_aux, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
