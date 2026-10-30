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

"""Sharding of m3 models end to end: annotations, placement, the mesh from config, and the loop.

`sharding_test.py` covers the rules and the mesh in isolation. This file checks
that the reference model (`qwen3`) actually uses them: every parameter carries
a resolved `out_sharding`, parameters land on the mesh as the rules say, the
loop builds its mesh from the config, and both scan and explicit sharding keep
the placement. Multi-device cases size themselves from `jax.device_count()`
(CI runs the CPU suite with `--xla_force_host_platform_device_count=4`).
"""

import math
from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.core.sharding import DefaultSharding, create_mesh, mesh_from_config
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.train import loop
from tests.utils import m3_test_utils


def _params(model):
  """`{path: variable}` for every `nnx.Param`."""
  return {"/".join(map(str, path)): var for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _shard_fraction(value: jax.Array) -> float:
  return math.prod(value.sharding.shard_shape(value.shape)) / value.size


def _devices_or_skip(test, minimum: int = 4):
  n = jax.device_count()
  if n < minimum or n % 2:
    test.skipTest(f"needs an even device count of at least {minimum}, found {n}")
  return jax.devices()


def _layer(i: int) -> str:
  """Path of decoder block `i`; `nnx.Sequential` contributes the second `layers` segment."""
  return f"decoder/layers/layers/{i}"


class AnnotationTest(absltest.TestCase):
  """Construction on one device: every tensor resolves, paths are unchanged."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.enter_context(jax.set_mesh(self.mesh))

  def test_every_parameter_carries_a_resolved_sharding(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    params = _params(model)
    self.assertNotEmpty(params)
    for path, var in params.items():
      spec = var.get_metadata().get("out_sharding")
      self.assertIsInstance(spec, P, f"{path} has no resolved out_sharding: {var.get_metadata()}")
      self.assertLen(spec, var[...].ndim, path)
      self.assertTrue(var.get_metadata().get("eager_sharding"), f"{path} is not placed at construction")

  def test_weights_name_the_strategy_axes(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    params = _params(m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0)))

    def spec(path):
      return params[path].get_metadata()["out_sharding"]

    self.assertEqual(spec(f"{_layer(0)}/mlp/wi_0/kernel"), P(("fsdp", "expert"), "tensor"))
    self.assertEqual(spec(f"{_layer(0)}/mlp/wo/kernel"), P("tensor", ("fsdp", "expert")))
    self.assertEqual(spec(f"{_layer(0)}/self_attention/query/kernel"), P(("fsdp", "expert"), "tensor", None))
    self.assertEqual(spec(f"{_layer(0)}/self_attention/out/kernel"), P("tensor", None, ("fsdp", "expert")))
    self.assertEqual(spec(f"{_layer(0)}/pre_self_attention_layer_norm/scale"), P(None))
    self.assertEqual(spec("token_embedder/embedding"), P("tensor", ("fsdp", "expert")))

  def test_parameter_paths_are_unchanged(self):
    """Annotations add metadata, not path segments; these are the paths checkpoints and HF mappings see."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    paths = set(_params(m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))))
    expected = {"token_embedder/embedding", "decoder/decoder_norm/scale"}
    for i in range(config.num_decoder_layers):
      expected |= {f"{_layer(i)}/self_attention/{name}/kernel" for name in ("query", "key", "value", "out")}
      expected |= {f"{_layer(i)}/self_attention/{name}/scale" for name in ("query_norm", "key_norm")}
      expected |= {f"{_layer(i)}/mlp/{name}/kernel" for name in ("wi_0", "wi_1", "wo")}
      expected |= {f"{_layer(i)}/{n}/scale" for n in ("pre_self_attention_layer_norm", "post_self_attention_layer_norm")}
    self.assertEqual(paths, expected)

  def test_a_strategy_can_be_injected(self):
    class _Replicated(DefaultSharding):

      def map_axis(self, axis, tensor_name, tensor_type):
        return None

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0), sharding=_Replicated())
    self.assertTrue(all(all(a is None for a in v.get_metadata()["out_sharding"]) for v in _params(model).values()))

  def test_model_built_under_a_mesh_runs_outside_one(self):
    """Eager use after construction (parity checks, notebooks): activation constraints become no-ops."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    tokens = jax.random.randint(jax.random.key(0), (2, 8), 0, config.vocab_size)
    positions = jax.numpy.broadcast_to(jax.numpy.arange(8), (2, 8))
    with jax.set_mesh(self.mesh):
      inside = model(tokens, positions)
    with jax.sharding.set_mesh(None):
      outside = model(tokens, positions)
    np.testing.assert_array_equal(np.asarray(inside), np.asarray(outside))


class MeshFromConfigTest(parameterized.TestCase):
  """`mesh_from_config`: legacy field names in, the m3 mesh out, the rest refused."""

  def test_defaults_match_create_mesh(self):
    mesh = mesh_from_config(SimpleNamespace(), devices=jax.devices()[:1])
    self.assertEqual(mesh.axis_names, m3_sharding.MESH_AXIS_NAMES)
    self.assertEqual(dict(mesh.shape), dict(create_mesh(devices=jax.devices()[:1]).shape))
    self.assertFalse(any(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types))

  def test_real_config_fields_are_read(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = mesh_from_config(config, devices=jax.devices()[:1])
    self.assertEqual(dict(mesh.shape), {"data": 1, "fsdp": 1, "tensor": 1, "expert": 1})

  def test_shard_mode_explicit_types_the_axes(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b", shard_mode="explicit")
    mesh = mesh_from_config(config, devices=jax.devices()[:1])
    self.assertTrue(all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types))

  def test_rejects_unknown_shard_mode(self):
    with self.assertRaisesRegex(ValueError, "shard_mode"):
      mesh_from_config(SimpleNamespace(shard_mode="manual"), devices=jax.devices()[:1])

  @parameterized.parameters(
      ("ici_context_parallelism", 2),
      ("dcn_pipeline_parallelism", 2),
      ("ici_fsdp_transpose_parallelism", 4),
      ("ici_sequence_parallelism", 2),
      ("custom_mesh", "hybrid_ring_64x4"),
      ("allow_split_physical_axes", True),
      ("optimize_mesh_for_tpu_v6e", True),
  )
  def test_unsupported_legacy_setting_is_refused(self, field, value):
    config = SimpleNamespace(**{field: value})
    self.assertEqual(m3_sharding.unsupported_mesh_settings(config), {field: value})
    with self.assertRaisesRegex(ValueError, f"does not implement.*{field}"):
      mesh_from_config(config, devices=jax.devices()[:1])

  def test_refusal_against_a_real_config(self):
    """The field names are checked against pyconfig, so a rename there surfaces here."""
    config = m3_test_utils.tiny_config("qwen3-0.6b", ici_context_parallelism=2)
    with self.assertRaisesRegex(ValueError, "ici_context_parallelism"):
      mesh_from_config(config, devices=jax.devices()[:1])

  def test_legacy_defaults_are_not_refused(self):
    self.assertEqual(m3_sharding.unsupported_mesh_settings(m3_test_utils.tiny_config("qwen3-0.6b")), {})


class PlacementTest(parameterized.TestCase):
  """On several devices the rules decide where each parameter lives."""

  def setUp(self):
    super().setUp()
    self.devices = _devices_or_skip(self)

  def _model(self, mesh, **overrides):
    config = m3_test_utils.tiny_config("qwen3-0.6b", **overrides)
    with jax.set_mesh(mesh):
      return config, m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))

  def test_fsdp_splits_the_model_dimension_and_replicates_norms(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    _, model = self._model(mesh)
    params = _params(model)
    n = len(self.devices)
    for path, var in params.items():
      value = var[...]
      if path.endswith("/scale"):
        self.assertEqual(_shard_fraction(value), 1.0, path)
      else:
        self.assertAlmostEqual(_shard_fraction(value), 1 / n, msg=path)
    kernel = params[f"{_layer(0)}/mlp/wi_0/kernel"][...]
    self.assertEqual(kernel.sharding.shard_shape(kernel.shape), (kernel.shape[0] // n, kernel.shape[1]))
    self.assertLess(loop.params_held_per_device(model), 1 / n + 0.05)

  def test_tensor_parallelism_splits_heads_and_mlp(self):
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    _, model = self._model(mesh)
    params = _params(model)
    wi = params[f"{_layer(0)}/mlp/wi_0/kernel"][...]
    query = params[f"{_layer(0)}/self_attention/query/kernel"][...]
    fsdp = mesh.shape["fsdp"]
    self.assertEqual(wi.sharding.shard_shape(wi.shape), (wi.shape[0] // fsdp, wi.shape[1] // 2))
    self.assertEqual(
        query.sharding.shard_shape(query.shape), (query.shape[0] // fsdp, query.shape[1] // 2, query.shape[2])
    )

  def test_scan_keeps_the_placement(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    config, model = self._model(mesh, scan_layers="true")
    with jax.set_mesh(mesh):
      scanned = m3_scan.maybe_apply_scan(model, config)
    kernel = _params(scanned)["decoder/layers/mlp/wi_0/kernel"][...]
    self.assertEqual(kernel.ndim, 3)
    self.assertAlmostEqual(_shard_fraction(kernel), 1 / len(self.devices))

  def test_construction_under_an_explicit_mesh_places_parameters(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices, explicit=True)
    _, model = self._model(mesh)
    kernel = _params(model)[f"{_layer(0)}/mlp/wi_0/kernel"][...]
    self.assertAlmostEqual(_shard_fraction(kernel), 1 / len(self.devices))

  def test_legacy_driver_sees_the_annotations(self):
    """`create_nnx_abstract_model` (the `from_pretrained` / training-engine path) traces construction
    under `nnx.eval_shape` without entering the mesh; the abstract state must still carry the
    strategy's specs rather than the driver's one-axis fallback."""
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    _, abstract = model_creation_utils.create_nnx_abstract_model(config, mesh)
    state = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abstract, nnx.Param))}
    kernel = state[f"{_layer(0)}/mlp/wi_0/kernel"]
    self.assertEqual(kernel.get_value().sharding.spec, P(("fsdp", "expert"), "tensor"))
    self.assertEqual(state["token_embedder/embedding"].get_value().sharding.spec, P("tensor", ("fsdp", "expert")))
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(("fsdp", "expert"), "tensor"))


class ShardedLoopTest(parameterized.TestCase):
  """The loop builds its mesh from the config and trains on it."""

  def setUp(self):
    super().setUp()
    self.devices = _devices_or_skip(self)

  def _config(self, **overrides):
    return m3_test_utils.tiny_config(
        "qwen3-0.6b",
        steps=2,
        base_output_directory=self.create_tempdir().full_path,
        run_name="m3_sharding_test",
        **overrides,
    )

  @parameterized.named_parameters(
      ("fsdp", {}),
      ("fsdp_and_tensor", {"ici_tensor_parallelism": 2}),
      ("scanned_fsdp", {"scan_layers": "true"}),
  )
  def test_train_loop_runs(self, overrides):
    self.assertTrue(np.isfinite(loop.train_loop(self._config(**overrides))))

  @pytest.mark.xfail(
      strict=True,
      reason="nnx.Embed's gather has no out_sharding under an explicit mesh (jax ShardingTypeError); "
      "the mesh builds and parameters place, the forward pass does not run yet.",
  )
  def test_train_loop_runs_with_explicit_sharding(self):
    self.assertTrue(np.isfinite(loop.train_loop(self._config(shard_mode="explicit"))))

  def test_batch_spec_follows_the_strategy(self):
    self.assertEqual(loop.batch_spec(DefaultSharding()), P(("data", "fsdp", "expert"), None))
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    shards = loop._batch_shards(mesh, loop.batch_spec(DefaultSharding()))  # pylint: disable=protected-access
    self.assertEqual(shards, len(self.devices) // 2)


@pytest.mark.tpu_only
class ShardedTrainingTpuTest(absltest.TestCase):
  """What only a real multi-chip run shows: the reduce-scatter of FSDP, and 8-way placement."""

  def test_fsdp_step_reduce_scatters_gradients(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = mesh_from_config(config)
    sharding = DefaultSharding()
    with jax.set_mesh(mesh):
      model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0), sharding=sharding)
      optimizer = nnx.Optimizer(model, loop.make_optimizer(config), wrt=nnx.Param)
      batch = loop.make_synthetic_batch(config, mesh, sharding, jax.random.key(0))
      model_def, model_state = nnx.split(model)
      opt_def, opt_state = nnx.split(optimizer)

      def step(model_state, opt_state, batch):
        m, o = nnx.merge(model_def, model_state), nnx.merge(opt_def, opt_state)
        loss, grads = nnx.value_and_grad(loop.loss_fn)(m, batch)
        o.update(m, grads)
        return loss, nnx.state(m), nnx.state(o)

      hlo = jax.jit(step).lower(model_state, opt_state, batch).compile().as_text()
    self.assertGreater(mesh.shape["fsdp"], 1)
    self.assertIn("reduce-scatter", hlo)
    self.assertLess(loop.params_held_per_device(model), 1 / mesh.shape["fsdp"] + 0.05)

  def test_train_loop_runs_on_all_chips(self):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b", steps=3, base_output_directory=self.create_tempdir().full_path, run_name="m3_sharding_tpu"
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))


if __name__ == "__main__":
  absltest.main()
