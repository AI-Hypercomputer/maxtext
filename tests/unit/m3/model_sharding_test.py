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
import sys
import types
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import optax
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.core.sharding import DefaultSharding, create_mesh, mesh_from_config, sharding_from_config
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
      self.assertEqual(var[...].sharding.spec, spec, f"{path} is not placed as its out_sharding says")
      # Placement ends with construction, so variables rebuilt from this metadata need no mesh.
      self.assertFalse(var.get_metadata().get("eager_sharding"), f"{path} still re-places on rebuild")

  def test_two_models_from_one_config_share_a_graphdef(self):
    """The strategy is a static attribute; equal graphdefs mean one `nnx.jit` trace serves both."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    first = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    second = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(1))
    self.assertEqual(nnx.graphdef(first), nnx.graphdef(second))

  def test_output_head_takes_vocab_tiles(self):
    """Vocab tiling calls the output head on `(tile, emb)` chunks, so its constraint must not assume rank 3."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    hidden = jax.numpy.ones((4, config.emb_dim), config.dtype)
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic=True)
    self.assertEqual(logits.shape, (4, config.vocab_size))

  def test_weights_name_the_strategy_axes(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    params = _params(m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0)))

    def spec(path):
      return params[path].get_metadata()["out_sharding"]

    dense_fsdp = ("fsdp", "fsdp_transpose", "context", "expert")
    tp_sp = ("tensor", "tensor_sequence")
    self.assertEqual(spec(f"{_layer(0)}/mlp/wi_0/kernel"), P(dense_fsdp, tp_sp))
    self.assertEqual(spec(f"{_layer(0)}/mlp/wo/kernel"), P(tp_sp, dense_fsdp))
    self.assertEqual(spec(f"{_layer(0)}/self_attention/query/kernel"), P(dense_fsdp, tp_sp, None))
    self.assertEqual(spec(f"{_layer(0)}/self_attention/out/kernel"), P(tp_sp, None, dense_fsdp))
    self.assertEqual(spec(f"{_layer(0)}/pre_self_attention_layer_norm/scale"), P(None))
    self.assertEqual(spec("token_embedder/embedding"), P(tp_sp, dense_fsdp))

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

  def test_a_strategy_reaches_the_qwen3_vl_decoder(self):
    """`sharding=` goes through `Qwen3VLModel` and `Qwen3VLDecoder` to every annotated text parameter."""

    class _Replicated(DefaultSharding):

      def map_axis(self, axis, tensor_name, tensor_type):
        return None

    config = m3_test_utils.tiny_config("qwen3-vl-2b")
    abstract = nnx.eval_shape(lambda: m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0), sharding=_Replicated()))
    params = _params(abstract)
    annotated = {
        "token_embedder/embedding",
        "decoder/decoder_norm/scale",
        f"{_layer(0)}/mlp/wi_0/kernel",
        f"{_layer(0)}/mlp/wo/kernel",
        f"{_layer(0)}/pre_self_attention_layer_norm/scale",
        f"{_layer(0)}/post_self_attention_layer_norm/scale",
    }
    for path in annotated:
      spec = params[path].get_metadata().get("out_sharding")
      self.assertIsInstance(spec, P, path)
      self.assertTrue(all(a is None for a in spec), f"{path} ignored the injected strategy: {spec}")

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

  def test_no_mesh_construction_raises_unless_explicitly_opted_in(self):
    from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    with jax.sharding.set_mesh(None):
      with self.assertRaisesRegex(ValueError, "mesh"):
        m3_models.create_model(config, mesh=None, rngs=nnx.Rngs(0))
      with self.assertRaisesRegex(ValueError, "mesh"):
        modeling_qwen3.Qwen3Model(config, None, rngs=nnx.Rngs(0))
      direct_model = modeling_qwen3.Qwen3Model(config, self.mesh, rngs=nnx.Rngs(0))
      self.assertIs(direct_model.mesh, self.mesh)
      with self.assertWarnsRegex(UserWarning, "allow_unsharded_construction"):
        with m3_sharding.allow_unsharded_construction():
          unsharded_model = m3_models.create_model(config, mesh=None, rngs=nnx.Rngs(0))
      tokens = jnp.ones((2, 8), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8))
      self.assertEqual(unsharded_model(tokens, positions).shape, (2, 8, config.vocab_size))

  def test_params_held_per_device_without_parameters(self):
    self.assertEqual(loop.params_held_per_device(nnx.Module()), 1.0)


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
    self.assertEqual(
        dict(mesh.shape),
        {
            "data": 1,
            "stage": 1,
            "fsdp": 1,
            "fsdp_transpose": 1,
            "context": 1,
            "tensor": 1,
            "tensor_sequence": 1,
            "expert": 1,
        },
    )

  def test_supported_multi_axis_fields_map_into_mesh(self):
    devices = _devices_or_skip(self)[:4]
    cfg = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        ici_fsdp_parallelism=1,
        ici_fsdp_transpose_parallelism=2,
        ici_context_parallelism=2,
        context_parallel_load_balance=False,
        allow_split_physical_axes=True,
    )
    mesh = mesh_from_config(cfg, devices=devices)
    self.assertEqual(mesh.shape["stage"], 1)
    self.assertEqual(mesh.shape["fsdp_transpose"], 2)
    self.assertEqual(mesh.shape["context"], 2)

    tsp_cfg = m3_test_utils.tiny_config(
        "qwen3-0.6b", ici_fsdp_parallelism=2, ici_tensor_sequence_parallelism=2, allow_split_physical_axes=True
    )
    tsp_mesh = mesh_from_config(tsp_cfg, devices=devices)
    self.assertEqual(tsp_mesh.shape["tensor_sequence"], 2)
    self.assertEqual(tsp_mesh.shape["fsdp"], 2)

  def test_shard_mode_explicit_types_the_axes(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b", shard_mode="explicit")
    mesh = mesh_from_config(config, devices=jax.devices()[:1])
    self.assertTrue(all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types))

  def test_every_field_read_or_refused_exists_on_the_config(self):
    """`getattr` defaults would silently disable a refusal if pyconfig renamed or removed the field."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    # pylint: disable=protected-access
    fields = [f"{net}_{axis}_parallelism" for net in ("ici", "dcn") for axis in m3_sharding._UNSUPPORTED_PARALLELISM_AXES]
    fields += list(m3_sharding._UNSUPPORTED_MESH_FIELDS)
    # pylint: enable=protected-access
    cfg_axes = ["pipeline" if axis == "stage" else axis for axis in m3_sharding.MESH_AXIS_NAMES]
    fields += [f"{net}_{axis}_parallelism" for net in ("ici", "dcn") for axis in cfg_axes]
    fields += ["shard_mode", "custom_mesh", "allow_split_physical_axes", "optimize_mesh_for_tpu_v6e"]
    missing = [field for field in fields if not hasattr(config, field)]
    self.assertEqual(missing, [])

  @parameterized.parameters(
      ("ici_pipeline_parallelism", 2),
      ("dcn_pipeline_parallelism", 2),
      ("ici_sequence_parallelism", 2),
      ("ici_diloco_parallelism", 2),
      ("ici_context_usp_ulysses_parallelism", 2),
      ("ici_autoregressive_parallelism", 2),
      ("custom_mesh_and_rule", "ep-as-cp"),
      ("custom_mesh_and_rule_for_eval", "pure-fsdp"),
  )
  def test_unsupported_legacy_setting_is_refused(self, field, value):
    config = SimpleNamespace(**{field: value})
    self.assertEqual(m3_sharding.unsupported_mesh_settings(config), {field: value})
    with self.assertRaisesRegex(ValueError, f"does not implement.*{field}"):
      mesh_from_config(config, devices=jax.devices()[:1])

  @parameterized.parameters(
      ("ici_sequence_parallelism", 2),
      ("custom_mesh_and_rule", "ep-as-cp"),  # pyconfig turns this one into a `CustomRule` Enum
  )
  def test_refusal_against_a_real_config(self, field, value):
    """The field names and types are checked against pyconfig, so a change there surfaces here."""
    config = m3_test_utils.tiny_config("qwen3-0.6b", **{field: value})
    self.assertEqual(m3_sharding.unsupported_mesh_settings(config).get(field), value)
    with self.assertRaisesRegex(ValueError, field):
      mesh_from_config(config, devices=jax.devices()[:1])

  def test_legacy_defaults_are_not_refused(self):
    self.assertEqual(m3_sharding.unsupported_mesh_settings(m3_test_utils.tiny_config("qwen3-0.6b")), {})

  @parameterized.parameters(
      ("qwen3-vl-2b", {"shard_mode": "explicit"}),
      ("granite-3.0-1b-a400m", {"shard_mode": "explicit"}),
      ("glm5.3-flash", {"shard_mode": "explicit"}),
      ("qwen3.8-flash-next-mini", {"shard_mode": "explicit"}),
      ("granite-3.0-1b-a400m", {"ici_fsdp_transpose_parallelism": 2}),
      ("granite-3.0-1b-a400m", {"ici_tensor_sequence_parallelism": 2}),
      ("granite-3.0-1b-a400m", {"ici_context_parallelism": 2, "context_parallel_load_balance": False}),
      ("qwen3-0.6b", {"ici_context_parallelism": 2, "context_parallel_load_balance": True}),
      (
          "qwen3-0.6b",
          {"ici_context_parallelism": 2, "context_parallel_load_balance": False, "context_parallel_strategy": "ring"},
      ),
  )
  def test_per_model_and_cp_unsupported_settings_are_refused(self, model_name, overrides):
    cfg = SimpleNamespace(model_name=model_name, **overrides)
    self.assertNotEmpty(m3_sharding.unsupported_mesh_settings(cfg))
    with self.assertRaises(ValueError):
      mesh_from_config(cfg, devices=jax.devices()[:1])


class CheckMeshTest(parameterized.TestCase):
  """Meshes other drivers build reach the model, which refuses the ones its strategy cannot use."""

  def test_the_m3_mesh_passes(self):
    m3_sharding.check_mesh(create_mesh(devices=jax.devices()[:1]))

  def test_size_one_extra_axes_pass(self):
    """Legacy's default mesh carries axes m3 does not use; at size 1 they shard nothing and change nothing."""
    shape = {"data": 1, "stage": 1, "fsdp": 8, "context": 1, "tensor": 1, "expert": 1}
    m3_sharding.check_mesh(SimpleNamespace(axis_names=tuple(shape), shape=shape))

  def test_a_missing_axis_is_refused(self):
    with self.assertRaisesRegex(ValueError, r"lacks mesh axes \['data', 'tensor', 'expert'\]"):
      m3_sharding.check_mesh(SimpleNamespace(axis_names=("fsdp",), shape={"fsdp": 8}))

  def test_a_split_unused_axis_is_refused(self):
    shape = {"data": 1, "fsdp": 4, "sequence": 2, "tensor": 1, "expert": 1}
    with self.assertRaisesRegex(ValueError, r"splits mesh axes \{'sequence': 2\}"):
      m3_sharding.check_mesh(SimpleNamespace(axis_names=tuple(shape), shape=shape))
    stage_shape = {"data": 1, "stage": 2, "fsdp": 4, "tensor": 1, "expert": 1}
    with self.assertRaisesRegex(ValueError, r"splits mesh axes \{'stage': 2\}"):
      m3_sharding.check_mesh(SimpleNamespace(axis_names=tuple(stage_shape), shape=stage_shape))

  def test_granitemoe_sharding_accepts_four_axis_mesh_and_refuses_split_new_axes(self):
    from maxtext.m3.models.granitemoe import modeling_granitemoe  # pylint: disable=import-outside-toplevel

    mesh_4ax = create_mesh(ici={"fsdp": 1}, devices=jax.devices()[:1], axis_names=("data", "fsdp", "tensor", "expert"))
    s_4ax = modeling_granitemoe.GraniteMoeSharding.from_mesh(mesh_4ax)
    self.assertEqual(s_4ax.fsdp_size, 1)

    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, got {len(devices)}")
    mesh_bad = create_mesh(ici={"fsdp": 2, "fsdp_transpose": 2}, devices=devices[:4])
    with self.assertRaises((ValueError, NotImplementedError)):
      modeling_granitemoe.GraniteMoeSharding.from_mesh(mesh_bad)

  def test_legacy_driver_with_a_mesh_lacking_tensor_is_refused(self):
    """`custom_mesh_and_rule=pure-fsdp` builds a legacy mesh over `fsdp` alone."""
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b", custom_mesh_and_rule="pure-fsdp")
    mesh = maxtext_utils.get_mesh_from_config(config, jax.devices()[:1])
    with self.assertRaisesRegex(ValueError, "lacks mesh axes"):
      model_creation_utils.create_nnx_abstract_model(config, mesh)


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

  def test_create_model_enters_the_mesh_it_is_passed(self):
    """With no mesh in scope, `create_model(config, mesh)` enters `mesh` itself and leaves none behind."""
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    kernel = _params(model)[f"{_layer(0)}/mlp/wi_0/kernel"][...]
    self.assertAlmostEqual(_shard_fraction(kernel), 1 / len(self.devices))

  def test_create_model_under_eval_shape_without_a_mesh_in_scope(self):
    """Under a trace `jax.set_mesh` is refused, so `create_model` enters only the abstract mesh."""
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    abstract = nnx.eval_shape(lambda: m3_models.create_model(config, mesh, rngs=nnx.Rngs(0)))
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    kernel = _params(abstract)[f"{_layer(0)}/mlp/wi_0/kernel"]
    self.assertEqual(
        kernel.get_metadata()["out_sharding"],
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
    )

  def test_create_model_refuses_a_different_mesh_in_scope(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    other = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    with jax.set_mesh(other), self.assertRaisesRegex(ValueError, "different mesh is in scope"):
      m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))

  def test_construction_under_an_explicit_mesh_places_parameters(self):
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices, explicit=True)
    _, model = self._model(mesh)
    kernel = _params(model)[f"{_layer(0)}/mlp/wi_0/kernel"][...]
    self.assertAlmostEqual(_shard_fraction(kernel), 1 / len(self.devices))

  def test_context_parallelism_shards_dense_weights_and_direct_factory_enters_mesh(self):
    from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=import-outside-toplevel

    n = len(self.devices)
    mesh = create_mesh(ici={"fsdp": n // 2, "context": 2}, devices=self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b", context_parallel_load_balance=False)
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    model = modeling_qwen3.create_qwen3_model(config, mesh, rngs=nnx.Rngs(0))
    self.assertLess(loop.params_held_per_device(model), 1 / n + 0.05)

  def test_explicit_data_context_scanned_preserves_metadata_and_scan_axis(self):
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    n = len(self.devices)
    cfg_exp = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        scan_layers="true",
        shard_mode="explicit",
        context_parallel_load_balance=False,
    )
    cfg_auto = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        scan_layers="true",
        shard_mode="auto",
        context_parallel_load_balance=False,
    )
    mesh_exp = create_mesh(ici={"data": n // 2, "fsdp": 1, "context": 2}, devices=self.devices, explicit=True)
    mesh_auto = create_mesh(ici={"data": n // 2, "fsdp": 1, "context": 2}, devices=self.devices, explicit=False)
    create_fn, abs_exp = model_creation_utils.create_nnx_abstract_model(cfg_exp, mesh_exp)
    _, abs_auto = model_creation_utils.create_nnx_abstract_model(cfg_auto, mesh_auto)
    st_exp = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abs_exp, nnx.Param))}
    st_auto = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abs_auto, nnx.Param))}
    wi_exp = st_exp["decoder/layers/mlp/wi_0/kernel"]
    wi_auto = st_auto["decoder/layers/mlp/wi_0/kernel"]
    self.assertEqual(wi_exp.get_metadata()["out_sharding"], wi_auto.get_metadata()["out_sharding"])
    # Scanned layer axis (dim 1) must stay replicated (`None`), while dim 0 (`embed`) shards over `context`.
    self.assertIsNone(wi_exp.get_value().sharding.spec[1])
    eager_model = create_fn()
    batch = loop.make_synthetic_batch(cfg_exp, mesh_exp, DefaultSharding(), jax.random.key(0))
    with jax.set_mesh(mesh_exp):
      logits = nnx.jit(lambda m, t, p: m(t, p))(eager_model, batch["inputs"], batch["positions"])
    self.assertEqual(logits.shape, (cfg_exp.global_batch_size_to_train_on, cfg_exp.max_target_length, cfg_exp.vocab_size))

  def test_model_creation_utils_propagates_value_error_without_rerunning_rngs(self):
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    cfg = m3_test_utils.tiny_config("qwen3-0.6b")
    calls = []

    def boom(*_args, **_kwargs):
      calls.append(1)
      raise ValueError("synthetic init failure")

    with mock.patch.object(m3_models, "create_model", side_effect=boom):
      with self.assertRaisesRegex(ValueError, "synthetic init failure"):
        model_creation_utils.create_model(cfg, mesh, rngs=nnx.Rngs(0))
    self.assertLen(calls, 1)

  def test_legacy_driver_sees_the_annotations(self):
    """`create_nnx_abstract_model` (the `from_pretrained` / training-engine path) traces construction
    under `nnx.eval_shape` inside the mesh; the abstract state must carry the strategy's specs,
    replicated norm scales included, rather than the driver's one-axis fallback."""
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    _, abstract = model_creation_utils.create_nnx_abstract_model(config, mesh)
    state = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abstract, nnx.Param))}
    dense_fsdp = ("fsdp", "fsdp_transpose", "context", "expert")
    kernel = state[f"{_layer(0)}/mlp/wi_0/kernel"]
    tp_sp = ("tensor", "tensor_sequence")
    self.assertEqual(kernel.get_value().sharding.spec, P(dense_fsdp, tp_sp))
    self.assertEqual(state["token_embedder/embedding"].get_value().sharding.spec, P(tp_sp, dense_fsdp))
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(dense_fsdp, tp_sp))
    for path in (f"{_layer(0)}/pre_self_attention_layer_norm/scale", "decoder/decoder_norm/scale"):
      self.assertEqual(state[path].get_value().sharding.spec, P(None), path)

  def test_optimizer_built_after_create_model_needs_no_mesh(self):
    """The training engine's `__init__` and `from_pretrained` callers build the optimizer outside any mesh."""
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    kernel = model.decoder.layers.layers[0].mlp.wi_0.kernel[...]
    mu = optimizer.opt_state[0].mu.decoder.layers.layers[0].mlp.wi_0.kernel[...]
    self.assertEqual(mu.sharding, kernel.sharding)
    self.assertAlmostEqual(_shard_fraction(mu), 1 / len(self.devices))

  def test_legacy_train_state_traces_for_an_m3_model(self):
    """`setup_train_loop` with `use_m3_model=true`: model and optimizer traced under a mesh-less `eval_shape`."""
    from maxtext.common import train_state_nnx  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    create_model, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

    def create_train_state():
      model = create_model()
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param))

    state, _, _ = maxtext_utils.get_abstract_state_nnx(config, mesh, create_train_state)
    self.assertNotEmpty(jax.tree.leaves(state))

  def test_legacy_fallback_spares_only_rank_one_annotated_leaves(self):
    """A rank-1 `P(None)` (a norm scale) stays replicated; a larger all-`None` leaf still gets FSDP."""
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    n = len(self.devices)
    state = nnx.State(
        {
            "scale": nnx.Param(jax.numpy.ones((2 * n,)), out_sharding=P(None), eager_sharding=False),
            # The fallback never shards dim 0 of an all-`None` rank>=2 leaf (it may be a scan axis),
            # so the divisible axis goes on dim 1.
            "kernel": nnx.Param(jax.numpy.ones((3, 2 * n)), out_sharding=P(None, None), eager_sharding=False),
            "plain": nnx.Param(jax.numpy.ones((3, 2 * n)), eager_sharding=False),
        }
    )
    is_variable = lambda x: isinstance(x, nnx.Variable)  # pylint: disable=unnecessary-lambda-assignment
    replicated = jax.tree.map(lambda v: v.replace(NamedSharding(mesh, P(*([None] * v.ndim)))), state, is_leaf=is_variable)
    # pylint: disable-next=protected-access
    placed = model_creation_utils._apply_m3_default_named_sharding(state, replicated, mesh)
    self.assertEqual(placed["scale"].get_value().spec, P(None))
    self.assertEqual(placed["kernel"].get_value().spec, P(None, "fsdp"))
    self.assertEqual(placed["plain"].get_value().spec, P(None, "fsdp"))

  def test_four_axis_test_mesh_backwards_compatibility(self):
    mesh_4ax = create_mesh(
        ici={"fsdp": -1, "tensor": 2},
        devices=self.devices,
        axis_names=("data", "fsdp", "tensor", "expert"),
    )
    config, model = self._model(mesh_4ax)
    with jax.set_mesh(mesh_4ax):
      batch = loop.make_synthetic_batch(config, mesh_4ax, DefaultSharding(), jax.random.key(0))
      loss, grads = nnx.jit(nnx.value_and_grad(loop.loss_fn))(model, batch)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertNotEmpty(_params(grads))

  def test_multi_axis_fsdp_transpose_and_expert_never_replicate_dense_weights(self):
    n = len(self.devices)
    cases = [
        {"fsdp": 1, "fsdp_transpose": n // 2, "tensor": 2},
        {"fsdp": 1, "expert": n // 2, "tensor": 2},
    ]
    if n >= 8 and n % 8 == 0:
      cases.append({"fsdp": 2, "fsdp_transpose": 2, "expert": n // 8, "tensor": 2})
    for ici in cases:
      mesh = create_mesh(ici=ici, devices=self.devices)
      _, model = self._model(mesh)
      for path, var in _params(model).items():
        if not path.endswith("/scale"):
          self.assertAlmostEqual(_shard_fraction(var[...]), 1 / n, msg=f"{ici}: {path}")

  def test_pure_tp_compiled_hlo_has_zero_weight_all_gather(self):
    n = len(self.devices)
    config = m3_test_utils.tiny_config("qwen3-0.6b", max_target_length=8)
    sharding = DefaultSharding()

    def compile_hlo(ici):
      mesh = create_mesh(ici=ici, devices=self.devices)
      with jax.set_mesh(mesh):
        model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0), sharding=sharding)
        batch = loop.make_synthetic_batch(config, mesh, sharding, jax.random.key(0))
        graphdef, state = nnx.split(model)

        def fwd_bwd(st, b):
          m = nnx.merge(graphdef, st)
          loss, grads = nnx.value_and_grad(loop.loss_fn)(m, b)
          return loss, nnx.state(grads, nnx.Param)

        return jax.jit(fwd_bwd).lower(state, batch).compile().as_text()

    hlo_pure_tp = compile_hlo({"data": n // 2, "fsdp": 1, "tensor": 2})
    self.assertNotIn("all-gather(", hlo_pure_tp)

    # Positive control: FSDP > 1 on the same model emits weight all-gathers.
    hlo_fsdp_tp = compile_hlo({"data": 1, "fsdp": n // 2, "tensor": 2})
    self.assertIn("all-gather(", hlo_fsdp_tp)


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

  @parameterized.named_parameters(
      ("explicit_fsdp", {"shard_mode": "explicit"}),
      ("explicit_fsdp_and_tensor", {"shard_mode": "explicit", "ici_tensor_parallelism": 2}),
      (
          "explicit_context_tensor_sp_scanned",
          {
              "shard_mode": "explicit",
              "scan_layers": "true",
              "ici_fsdp_parallelism": 1,
              "ici_context_parallelism": 2,
              "context_parallel_load_balance": False,
              "ici_tensor_parallelism": 1,
              "ici_tensor_sequence_parallelism": 2,
          },
      ),
      (
          "auto_int8_quantized_sp",
          {
              "shard_mode": "auto",
              "quantization": "int8",
              "use_qwix_quantization": "true",
              "ici_tensor_parallelism": 1,
              "ici_tensor_sequence_parallelism": 2,
          },
      ),
  )
  def test_train_loop_runs_with_explicit_sharding(self, overrides):
    self.assertTrue(np.isfinite(loop.train_loop(self._config(**overrides))))

  def test_explicit_sharding_rejects_quantization(self):
    cfg = self._config(shard_mode="explicit", quantization="int8", use_qwix_quantization="true")
    with self.assertRaisesRegex(ValueError, "shard_mode=.explicit.*qwix"):
      loop.train_loop(cfg)

  def test_setup_model_leaves_the_strategy_to_the_model(self):
    """Constructors differ on `sharding=` (GLM rejects unknown keywords), so the loop passes none."""
    config = self._config()
    mesh = mesh_from_config(config)
    with jax.set_mesh(mesh), mock.patch.object(m3_models, "create_model", wraps=m3_models.create_model) as create:
      loop.setup_model(config, mesh)
    self.assertNotIn("sharding", create.call_args.kwargs)

  def test_batch_spec_follows_the_strategy(self):
    self.assertEqual(
        loop.batch_spec(DefaultSharding()),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context"),
    )
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    shards = loop._batch_shards(mesh, loop.batch_spec(DefaultSharding()))  # pylint: disable=protected-access
    self.assertEqual(shards, len(self.devices) // 2)

  def test_maxtext_training_engine_runs_explicit_zero1_step(self):
    """Verifies `MaxTextTrainingEngine` ZeRO-1 + `ShardMode.EXPLICIT` + `sequence_parallel` on `m3`."""
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    stub_modules = {}
    try:
      import tunix  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
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

    n = len(self.devices)
    with mock.patch.dict(sys.modules, stub_modules):
      from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

      eng_cfg = pyconfig.initialize(
          [
              "",
              get_test_config_path("base.yml"),
              "model_name=qwen3-0.6b",
              "run_name=m3_engine_explicit_zero1",
              f"base_output_directory={self.create_tempdir().full_path}",
              "use_m3_model=true",
              "shard_mode=explicit",
              "shard_optimizer_over_data=true",
              "ici_tensor_sequence_parallelism=2",
              "scan_layers=true",
              "dtype=float32",
              "weight_dtype=float32",
              "grad_dtype=float32",
              "warmup_steps_fraction=0.0",
              "learning_rate=1e-2",
              "gradient_accumulation_steps=1",
              f"ici_data_parallelism={n // 2}",
              "ici_fsdp_parallelism=1",
              "ici_tensor_parallelism=1",
              "per_device_batch_size=1",
              "max_target_length=16",
              "max_prefill_predict_length=8",
              "override_model_config=true",
              "base_emb_dim=64",
              "base_mlp_dim=128",
              "base_num_query_heads=4",
              "base_num_kv_heads=2",
              "base_num_decoder_layers=2",
              "head_dim=16",
              "vocab_size=256",
              "enable_checkpointing=false",
              "convert_checkpoint_if_possible=false",
              "enable_tensorboard=false",
              "record_internal_nn_metrics=false",
              "use_weight_converter=false",
              "skip_jax_distributed_system=true",
          ]
      )
      eng_mesh = mesh_from_config(eng_cfg, devices=self.devices)
      with jax.set_mesh(eng_mesh):
        engine = maxtext_engine.MaxTextTrainingEngine(eng_cfg, mesh=eng_mesh)
        self.assertIsNone(maxtext_engine._zero1_active(eng_cfg, eng_mesh))  # pylint: disable=protected-access
        before = [np.asarray(p) for p in jax.tree.leaves(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))]
        bsz, seq = int(eng_cfg.micro_batch_size_to_train_on), eng_cfg.max_target_length
        rng = np.random.default_rng(0)
        toks = rng.integers(1, eng_cfg.vocab_size, size=(bsz, seq)).astype(np.int32)
        eng_batch = {
            "inputs": toks,
            "targets": np.roll(toks, -1, axis=-1),
            "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (bsz, 1)),
            "inputs_segmentation": np.ones((bsz, seq), dtype=np.int32),
            "targets_segmentation": np.ones((bsz, seq), dtype=np.int32),
        }
        engine.compile(eng_batch)
        engine.fwd_bwd(eng_batch)
        engine.update()
        after = [np.asarray(p) for p in jax.tree.leaves(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))]
        opt_leaves = [
            x
            for x in jax.tree.leaves(nnx.state(engine.optimizer, nnx.optimizer.OptState))
            if hasattr(x, "sharding") and getattr(x, "ndim", 0) >= 1
        ]
        sharded_over_data = [
            x for x in opt_leaves if any(a == "data" or (isinstance(a, tuple) and "data" in a) for a in x.sharding.spec)
        ]
        self.assertNotEmpty(sharded_over_data)
        delta = max(float(np.max(np.abs(a - b))) for a, b in zip(after, before))
        self.assertGreater(delta, 0.0)
        self.assertEqual(engine.train_step, 1)
        from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
        from maxtext.trainers.pre_train import train as maxtext_train  # pylint: disable=import-outside-toplevel
        from maxtext.utils import sharding as mt_sharding  # pylint: disable=import-outside-toplevel

        with nn_partitioning.axis_rules(eng_cfg.logical_axis_rules):
          ds = mt_sharding.get_input_data_sharding(eng_cfg, eng_mesh)
          traced_batch = jax.tree.map(lambda a: jax.device_put(jnp.asarray(a), ds), eng_batch)
          explicit_loss, _ = nnx.jit(lambda m, b: maxtext_train.loss_fn(m, eng_cfg, b, None, None))(
              engine.model, traced_batch
          )
          self.assertTrue(np.isfinite(float(explicit_loss)))

      # Pure-DP explicit mesh (`data=n`, all other axes 1): verifies engine ZeRO-1 compiles cleanly on m3.
      dp_cfg = pyconfig.initialize(
          [
              "",
              get_test_config_path("base.yml"),
              "model_name=qwen3-0.6b",
              "run_name=m3_engine_explicit_pure_dp_zero1",
              f"base_output_directory={self.create_tempdir().full_path}",
              "use_m3_model=true",
              "shard_mode=explicit",
              "shard_optimizer_over_data=true",
              "scan_layers=true",
              "dtype=float32",
              "weight_dtype=float32",
              "grad_dtype=float32",
              "warmup_steps_fraction=0.0",
              "learning_rate=1e-2",
              "gradient_accumulation_steps=1",
              f"ici_data_parallelism={n}",
              "ici_fsdp_parallelism=1",
              "ici_tensor_parallelism=1",
              "per_device_batch_size=1",
              "max_target_length=16",
              "max_prefill_predict_length=8",
              "override_model_config=true",
              "base_emb_dim=64",
              "base_mlp_dim=128",
              "base_num_query_heads=4",
              "base_num_kv_heads=2",
              "base_num_decoder_layers=2",
              "head_dim=16",
              "vocab_size=256",
              "enable_checkpointing=false",
              "convert_checkpoint_if_possible=false",
              "enable_tensorboard=false",
              "record_internal_nn_metrics=false",
              "use_weight_converter=false",
              "skip_jax_distributed_system=true",
          ]
      )
      dp_mesh = mesh_from_config(dp_cfg, devices=self.devices)
      with jax.set_mesh(dp_mesh):
        dp_engine = maxtext_engine.MaxTextTrainingEngine(dp_cfg, mesh=dp_mesh)
        dp_bsz = int(dp_cfg.micro_batch_size_to_train_on)
        dp_toks = np.ones((dp_bsz, 16), dtype=np.int32)
        dp_batch = {
            "inputs": dp_toks,
            "targets": dp_toks,
            "inputs_position": np.tile(np.arange(16, dtype=np.int32), (dp_bsz, 1)),
            "inputs_segmentation": np.ones((dp_bsz, 16), dtype=np.int32),
            "targets_segmentation": np.ones((dp_bsz, 16), dtype=np.int32),
        }
        dp_engine.compile(dp_batch)
        dp_engine.fwd_bwd(dp_batch)
        dp_engine.update()
        self.assertEqual(dp_engine.train_step, 1)


class SequenceParallelismTest(parameterized.TestCase):
  """Megatron sequence parallelism (`tensor_sequence` axis): shard shapes, HLO, and parity."""

  def setUp(self):
    super().setUp()
    self.devices = _devices_or_skip(self)

  def test_norm_and_residual_shard_over_tensor_and_lower_reduce_scatter(self):
    n = len(self.devices)
    mesh_no = create_mesh(ici={"data": n // 2, "fsdp": 1, "tensor": 2, "tensor_sequence": 1}, devices=self.devices)
    mesh_sp = create_mesh(ici={"data": n // 2, "fsdp": 1, "tensor": 1, "tensor_sequence": 2}, devices=self.devices)
    sharding = DefaultSharding()
    x = jnp.zeros((n // 2, 16, 64), dtype=jnp.float32)

    with jax.set_mesh(mesh_no):
      out_no = jax.jit(lambda v: sharding.constrain(v, "res", ("batch", "norm_length", "embed")))(x)
      self.assertEqual(out_no.sharding.shard_shape(x.shape), (1, 16, 64))

      cfg_no = m3_test_utils.tiny_config(
          "qwen3-0.6b",
          max_target_length=16,
          ici_data_parallelism=n // 2,
          ici_fsdp_parallelism=1,
          ici_tensor_parallelism=2,
          ici_tensor_sequence_parallelism=1,
      )
      model_no = m3_models.create_model(cfg_no, mesh_no, rngs=nnx.Rngs(0), sharding=sharding)
      batch_no = loop.make_synthetic_batch(cfg_no, mesh_no, sharding, jax.random.key(0))
      g_no, st_no = nnx.split(model_no)
      hlo_no = (
          jax.jit(lambda st, b: nnx.value_and_grad(loop.loss_fn)(nnx.merge(g_no, st), b)[0])
          .lower(st_no, batch_no)
          .compile()
          .as_text()
      )

    with jax.set_mesh(mesh_sp):
      out_sp = jax.jit(lambda v: sharding.constrain(v, "res", ("batch", "norm_length", "embed")))(x)
      self.assertEqual(out_sp.sharding.shard_shape(x.shape), (1, 8, 64))

      cfg_sp = m3_test_utils.tiny_config(
          "qwen3-0.6b",
          max_target_length=16,
          ici_data_parallelism=n // 2,
          ici_fsdp_parallelism=1,
          ici_tensor_parallelism=1,
          ici_tensor_sequence_parallelism=2,
      )
      model_sp = m3_models.create_model(cfg_sp, mesh_sp, rngs=nnx.Rngs(0), sharding=sharding)
      batch_sp = loop.make_synthetic_batch(cfg_sp, mesh_sp, sharding, jax.random.key(0))
      graphdef, state = nnx.split(model_sp)
      hlo_sp = (
          jax.jit(lambda st, b: nnx.value_and_grad(loop.loss_fn)(nnx.merge(graphdef, st), b)[0])
          .lower(state, batch_sp)
          .compile()
          .as_text()
      )

    self.assertNotIn("all-gather(", hlo_no)
    self.assertIn("all-gather(", hlo_sp)

    # Isolate `Qwen3DecoderLayer` directly so stripping `norm_length` from decoder layers cannot be masked by embeddings.
    from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=import-outside-toplevel

    pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (n // 2, 16))
    with jax.set_mesh(mesh_no):
      lyr_no = modeling_qwen3.Qwen3DecoderLayer(cfg_no, rngs=nnx.Rngs(0), sharding=sharding)
      gl_no, sl_no = nnx.split(lyr_no)
      x_no = jax.device_put(x, NamedSharding(mesh_no, P("data", None, None)))
      p_no = jax.device_put(pos, NamedSharding(mesh_no, P("data", None)))
      hlo_lyr_no = jax.jit(lambda st, inp, p: nnx.merge(gl_no, st)(inp, p)).lower(sl_no, x_no, p_no).compile().as_text()

    with jax.set_mesh(mesh_sp):
      lyr_sp = modeling_qwen3.Qwen3DecoderLayer(cfg_sp, rngs=nnx.Rngs(0), sharding=sharding)
      gl_sp, sl_sp = nnx.split(lyr_sp)
      x_sp = jax.device_put(x, NamedSharding(mesh_sp, P("data", "tensor_sequence", None)))
      p_sp = jax.device_put(pos, NamedSharding(mesh_sp, P("data", None)))
      lyr_fn_sp = jax.jit(lambda st, inp, p: nnx.merge(gl_sp, st)(inp, p))
      out_lyr_sp = lyr_fn_sp(sl_sp, x_sp, p_sp)
      hlo_lyr_sp = lyr_fn_sp.lower(sl_sp, x_sp, p_sp).compile().as_text()

    self.assertEqual(out_lyr_sp.sharding.shard_shape(x.shape), (1, 8, 64))
    self.assertNotIn("all-gather(", hlo_lyr_no)
    self.assertIn("all-gather(", hlo_lyr_sp)
    # XLA:CPU decomposes reduce-scatter into all-reduce + dynamic-slice inside each SP decoder layer.
    self.assertGreaterEqual(hlo_lyr_sp.count("dynamic-slice("), hlo_lyr_no.count("dynamic-slice(") + 2)

  @parameterized.named_parameters(
      ("auto_unscanned", "auto", "false"),
      ("auto_scanned", "auto", "true"),
      ("explicit_unscanned", "explicit", "false"),
      ("explicit_scanned", "explicit", "true"),
  )
  def test_forward_and_all_leaf_grad_parity_with_negative_control(self, shard_mode, scan_layers):
    common_kw = {
        "dtype": "float32",
        "weight_dtype": "float32",
        "float32_logits": "true",
        "scan_layers": scan_layers,
        "shard_mode": shard_mode,
        "ici_fsdp_parallelism": -1,
        "ici_context_parallelism": 1,
        "max_target_length": 16,
        "per_device_batch_size": 1,
    }
    cfg_no = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        ici_tensor_parallelism=2,
        ici_tensor_sequence_parallelism=1,
        **common_kw,
    )
    cfg_sp = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        ici_tensor_parallelism=1,
        ici_tensor_sequence_parallelism=2,
        **common_kw,
    )
    mesh_no = mesh_from_config(cfg_no)
    mesh_sp = mesh_from_config(cfg_sp)
    s_no = sharding_from_config(cfg_no)
    s_sp = sharding_from_config(cfg_sp)

    @nnx.jit
    def fwd_bwd(model, b):
      logits = model(b["inputs"], b["positions"])
      loss, grads = nnx.value_and_grad(loop.loss_fn)(model, b)
      return logits, loss, grads

    with jax.set_mesh(mesh_no):
      m_no = m3_scan.maybe_apply_scan(m3_models.create_model(cfg_no, mesh_no, rngs=nnx.Rngs(0), sharding=s_no), cfg_no)
      batch_no = loop.make_synthetic_batch(cfg_no, mesh_no, s_no, jax.random.key(7))
      logits_no, loss_no, grads_no = fwd_bwd(m_no, batch_no)

    with jax.set_mesh(mesh_sp):
      m_sp = m3_scan.maybe_apply_scan(m3_models.create_model(cfg_sp, mesh_sp, rngs=nnx.Rngs(0), sharding=s_sp), cfg_sp)
      batch_sp = loop.make_synthetic_batch(cfg_sp, mesh_sp, s_sp, jax.random.key(7))
      logits_sp, loss_sp, grads_sp = fwd_bwd(m_sp, batch_sp)

    np.testing.assert_allclose(np.asarray(logits_sp), np.asarray(logits_no), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(loss_sp), float(loss_no), rtol=1e-5, atol=1e-5)
    g_no = {p: np.asarray(v[...]) for p, v in _params(grads_no).items()}
    g_sp = {p: np.asarray(v[...]) for p, v in _params(grads_sp).items()}
    self.assertEqual(set(g_no), set(g_sp))
    self.assertNotEmpty(g_sp)
    for path in g_no:
      np.testing.assert_allclose(g_sp[path], g_no[path], rtol=1e-5, atol=1e-5, err_msg=path)

    # Perturbation negative control: scaling final norm weight MUST break both logit and grad parity.
    with jax.set_mesh(mesh_sp):
      m_sp.decoder.decoder_norm.scale[...] = m_sp.decoder.decoder_norm.scale[...] * 1.05
      logits_bad, _, grads_bad = fwd_bwd(m_sp, batch_sp)
    self.assertFalse(np.allclose(np.asarray(logits_bad), np.asarray(logits_no), rtol=1e-5, atol=1e-5))
    g_bad = {p: np.asarray(v[...]) for p, v in _params(grads_bad).items()}
    self.assertTrue(
        any(not np.allclose(g_bad[p], g_no[p], rtol=1e-5, atol=1e-5) for p in g_no),
        "Perturbation negative control failed to fire on leaf parameter gradients!",
    )

  def test_rejects_target_length_indivisible_by_sequence_shards(self):
    cfg = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        steps=1,
        ici_fsdp_parallelism=-1,
        ici_context_parallelism=1,
        ici_tensor_sequence_parallelism=2,
        max_target_length=5,
    )
    with self.assertRaisesRegex(ValueError, r"not divisible by the \d+ sequence shards"):
      loop.train_loop(cfg)

  def test_omitted_decoder_positions_and_submodule_sharding_from_config(self):
    """Verifies default `decoder_positions=None` and submodule `sharding_from_config` under Explicit + SP."""
    from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=import-outside-toplevel

    cfg = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        shard_mode="explicit",
        ici_fsdp_parallelism=1,
        ici_data_parallelism=len(self.devices) // 2,
        ici_tensor_sequence_parallelism=2,
    )
    mesh = create_mesh(
        ici={"data": len(self.devices) // 2, "tensor_sequence": 2},
        devices=self.devices,
        explicit=True,
    )
    with jax.set_mesh(mesh):
      layer_only = modeling_qwen3.Qwen3DecoderLayer(cfg, rngs=nnx.Rngs(0))
      self.assertIsInstance(layer_only.sharding, DefaultSharding)

      model = m3_models.create_model(cfg, mesh, rngs=nnx.Rngs(0))
      batch = loop.make_synthetic_batch(cfg, mesh, model.sharding, jax.random.key(11))
      logits_explicit_pos = nnx.jit(lambda m, t, p: m(t, p))(model, batch["inputs"], batch["positions"])
      logits_default_pos = nnx.jit(lambda m, t: m(t))(model, batch["inputs"])
      logits_none_pos = nnx.jit(lambda m, t: m(t, None))(model, batch["inputs"])

    np.testing.assert_allclose(np.asarray(logits_default_pos), np.asarray(logits_explicit_pos), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(logits_none_pos), np.asarray(logits_explicit_pos), rtol=1e-5, atol=1e-5)


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

  def test_sequence_parallel_decoder_layer_lowers_reduce_scatter_on_tpu(self):
    from maxtext.m3.models.qwen3 import modeling_qwen3  # pylint: disable=import-outside-toplevel

    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 TPU chips, got {len(devices)}")
    n = len(devices)
    seq_len = 512
    mesh_tp = create_mesh(ici={"data": n // 2, "fsdp": 1, "tensor": 2, "tensor_sequence": 1}, devices=devices)
    mesh_sp = create_mesh(ici={"data": n // 2, "fsdp": 1, "tensor": 1, "tensor_sequence": 2}, devices=devices)
    cfg_tp = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        max_target_length=seq_len,
        ici_data_parallelism=n // 2,
        ici_fsdp_parallelism=1,
        ici_tensor_parallelism=2,
        ici_tensor_sequence_parallelism=1,
    )
    cfg_sp = m3_test_utils.tiny_config(
        "qwen3-0.6b",
        max_target_length=seq_len,
        ici_data_parallelism=n // 2,
        ici_fsdp_parallelism=1,
        ici_tensor_parallelism=1,
        ici_tensor_sequence_parallelism=2,
    )
    sharding = DefaultSharding()
    x = jnp.zeros((n // 2, seq_len, cfg_sp.emb_dim), dtype=jnp.bfloat16)
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (n // 2, seq_len))
    with jax.set_mesh(mesh_tp):
      lyr_tp = modeling_qwen3.Qwen3DecoderLayer(cfg_tp, rngs=nnx.Rngs(0), sharding=sharding)
      gl_tp, sl_tp = nnx.split(lyr_tp)
      x_tp = jax.device_put(x, NamedSharding(mesh_tp, P("data", None, None)))
      p_tp = jax.device_put(pos, NamedSharding(mesh_tp, P("data", None)))
      hlo_tp = jax.jit(lambda st, inp, p: nnx.merge(gl_tp, st)(inp, p)).lower(sl_tp, x_tp, p_tp).compile().as_text()
    with jax.set_mesh(mesh_sp):
      lyr_sp = modeling_qwen3.Qwen3DecoderLayer(cfg_sp, rngs=nnx.Rngs(0), sharding=sharding)
      gl_sp, sl_sp = nnx.split(lyr_sp)
      x_sp = jax.device_put(x, NamedSharding(mesh_sp, P("data", "tensor_sequence", None)))
      p_sp = jax.device_put(pos, NamedSharding(mesh_sp, P("data", None)))
      fn_sp = jax.jit(lambda st, inp, p: nnx.merge(gl_sp, st)(inp, p))
      out_sp = fn_sp(sl_sp, x_sp, p_sp)
      hlo_sp = fn_sp.lower(sl_sp, x_sp, p_sp).compile().as_text()
    self.assertEqual(out_sp.sharding.shard_shape(x.shape), (1, seq_len // 2, cfg_sp.emb_dim))
    self.assertNotIn("reduce-scatter", hlo_tp)
    self.assertNotIn("all-gather(", hlo_tp)
    # XLA:TPU emits either standalone `reduce-scatter(` or `kCustom` `calls=%all-reduce-scatter`.
    self.assertIn("reduce-scatter", hlo_sp)
    self.assertIn("all-gather(", hlo_sp)

  def test_train_loop_runs_on_all_chips(self):
    config = m3_test_utils.tiny_config(
        "qwen3-0.6b", steps=3, base_output_directory=self.create_tempdir().full_path, run_name="m3_sharding_tpu"
    )
    self.assertTrue(np.isfinite(loop.train_loop(config)))


if __name__ == "__main__":
  absltest.main()
