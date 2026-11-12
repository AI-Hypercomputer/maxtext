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

"""Integration tests for canonical m3 transform composition (`scan -> remat -> quantize -> named scopes -> offload`)."""

from typing import Any
from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.m3.infra import offload as m3_offload
from maxtext.m3.infra import quantization as m3_quant
from maxtext.m3.infra import remat as m3_remat
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.train import loop
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _make_config(output_dir: str, model_name: str, **overrides: Any):
  """Builds a small 4-CPU m3 config for `qwen3-0.6b` or `qwen3.8-flash-next-mini`."""
  argv = [
      "",
      get_test_config_path("base.yml"),
      f"base_output_directory={output_dir}",
      "run_name=m3_transform_composition_test",
      f"model_name={model_name}",
      "use_m3_model=true",
      "ici_fsdp_parallelism=2",
      "ici_tensor_parallelism=2",
      "per_device_batch_size=1.0",
      "max_target_length=16",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "enable_checkpointing=false",
      "use_weight_converter=false",
      "skip_jax_distributed_system=true",
  ]
  if model_name == "qwen3-0.6b":
    argv.extend(
        [
            "override_model_config=true",
            "base_emb_dim=64",
            "base_mlp_dim=128",
            "base_num_decoder_layers=2",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
        ]
    )
  for key, value in overrides.items():
    argv.append(f"{key}={value}")
  return pyconfig.initialize(argv)


def _param_paths(model: nnx.Module) -> tuple[tuple[Any, ...], ...]:
  """Returns sorted tuple of flat `nnx.Param` key paths for `model`."""
  return tuple(path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param)))


def _non_param_paths(model: nnx.Module) -> tuple[tuple[Any, ...], ...]:
  """Returns sorted tuple of flat non-`nnx.Param` key paths for `model`."""
  return tuple(path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Not(nnx.Param))))


def _assert_composed_invariants(
    tc: absltest.TestCase,
    baseline_paths: tuple[tuple[Any, ...], ...],
    model: nnx.Module,
    initial_non_param_paths: tuple[tuple[Any, ...], ...],
    mesh: jax.sharding.Mesh,
) -> None:
  """Verifies path transparency, non-Param stability, Qwix wrappers, and sharding."""
  composed_paths = _param_paths(model)
  tc.assertEqual(composed_paths, baseline_paths)
  for path in composed_paths:
    tc.assertNotIn("module", path, f"Wrapper leaked '.module.' segment into path {path}")
    tc.assertNotIn("_module", path, f"Wrapper leaked '._module.' segment into path {path}")

  tc.assertEqual(_non_param_paths(model), initial_non_param_paths)
  for path, var in nnx.to_flat_state(nnx.state(model)):
    tc.assertNotIsInstance(var, nnx.Intermediate, f"Unexpected sowed Intermediate at {path}")

  tc.assertIsInstance(model, m3_offload.HostOffloadedParameters)
  tc.assertTrue(hasattr(type(model), "_unquantized_type"))

  param_vars = [v for _, v in nnx.to_flat_state(nnx.state(model, nnx.Param)) if isinstance(v, nnx.Variable)]
  tc.assertNotEmpty(param_vars)
  sharded_multi_d = 0
  for var in param_vars:
    raw = var.get_raw_value()
    tc.assertIsInstance(raw.sharding, jax.sharding.NamedSharding)
    tc.assertEqual(raw.sharding.mesh, mesh)
    tc.assertEqual(raw.sharding.memory_kind, "pinned_host")
    if raw.ndim >= 2 and any(axis is not None for axis in raw.sharding.spec):
      sharded_multi_d += 1
  tc.assertGreater(sharded_multi_d, 0)


class TransformCompositionTest(parameterized.TestCase):
  """Tests `scan -> quantize -> remat -> offload` composition and negative controls (`M3-08`)."""

  @parameterized.named_parameters(
      ("qwen3_scanned", "qwen3-0.6b", True),
      ("qwen3_unscanned", "qwen3-0.6b", False),
      ("qwen3_8_flash_next_mini_scanned", "qwen3.8-flash-next-mini", True),
  )
  def test_canonical_transform_composition_and_multi_step_stability(self, model_name: str, scan_layers: bool):
    out_dir = self.create_tempdir().full_path
    cfg = _make_config(
        out_dir,
        model_name,
        scan_layers=str(scan_layers).lower(),
        quantization="int8",
        use_qwix_quantization="true",
        remat_policy="full",
        parameter_memory_host_offload="true",
        optimizer_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      # Baseline after model creation + optional layer scan, before quantize -> remat -> offload.
      baseline_model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      baseline_model = m3_scan.maybe_apply_scan(baseline_model, cfg)
      baseline_paths = _param_paths(baseline_model)
      initial_non_param_paths = _non_param_paths(baseline_model)
      del baseline_model

      # Canonical composition via model_creation_utils.create_model:
      # create_model -> maybe_apply_scan -> maybe_quantize_model -> maybe_apply_remat -> maybe_offload_parameters.
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      _assert_composed_invariants(self, baseline_paths, model, initial_non_param_paths, mesh)

      toks = jnp.ones((4, 16), dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (4, 16))
      self.assertGreater(m3_quant.quantized_matmuls(model, jnp.int8, toks, pos), 0)

      # Verify nnx.eval_shape preserves "pinned_host" memory kind on abstract parameter shardings.
      abs_model = nnx.eval_shape(
          lambda: model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      )
      self.assertEqual(_param_paths(abs_model), baseline_paths)
      for _, var in nnx.to_flat_state(nnx.state(abs_model, nnx.Param)):
        self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")

      # Run multiple train steps and verify zero non-Param state widening and preserved paths.
      optimizer = m3_offload.make_optimizer(model, loop.make_optimizer(cfg), cfg)
      batch = {"inputs": toks, "positions": pos, "targets": toks}
      losses = []
      for _ in range(2):
        loss = loop.train_step(model, optimizer, batch)
        losses.append(float(loss))
        _assert_composed_invariants(self, baseline_paths, model, initial_non_param_paths, mesh)
      self.assertTrue(all(jnp.isfinite(l) for l in losses))

  def test_negative_control_wrapper_inserting_module_segment_or_widening_state_is_caught(self):
    out_dir = self.create_tempdir().full_path
    cfg = _make_config(
        out_dir,
        "qwen3-0.6b",
        scan_layers="true",
        quantization="int8",
        use_qwix_quantization="true",
        remat_policy="full",
        parameter_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      base_model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      base_model = m3_scan.maybe_apply_scan(base_model, cfg)
      baseline_paths = _param_paths(base_model)
      initial_non_param_paths = _non_param_paths(base_model)

      # Negative control 1: a naive wrapper that stores the inner model under `self.module`
      # prefixes every parameter path with `('module', ...)` and must fail the path check.
      class _OpaqueWrapper(nnx.Module):

        def __init__(self, inner: nnx.Module):
          self.module = inner

      wrapped_opaque = _OpaqueWrapper(base_model)
      with self.assertRaises(AssertionError):
        _assert_composed_invariants(self, baseline_paths, wrapped_opaque, initial_non_param_paths, mesh)

      # Negative control 2: a transform or forward pass that sows non-Param state widens
      # `nnx.state(model, nnx.Not(nnx.Param))` and must fail the state stability check.
      composed = quantizations.maybe_quantize_model(base_model, cfg)
      composed = m3_remat.maybe_apply_remat(composed, cfg, model_mode=MODEL_MODE_TRAIN)
      composed = m3_offload.maybe_offload_parameters(composed, cfg)
      composed.leaked_activation = nnx.Intermediate(jnp.ones((2, 2), dtype=jnp.float32))
      with self.assertRaises(AssertionError):
        _assert_composed_invariants(self, baseline_paths, composed, initial_non_param_paths, mesh)


if __name__ == "__main__":
  absltest.main()
