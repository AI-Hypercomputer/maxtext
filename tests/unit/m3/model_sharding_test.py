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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax._src import core as jax_core  # pylint: disable=protected-access
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import optax
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
    self.assertEqual(dict(mesh.shape), {"data": 1, "fsdp": 1, "tensor": 1, "expert": 1})

  def test_shard_mode_explicit_is_refused(self):
    """The forward pass does not run under an explicit mesh yet, so the loop must not start on one."""
    config = m3_test_utils.tiny_config("qwen3-0.6b", shard_mode="explicit")
    with self.assertRaisesRegex(ValueError, "does not implement.*shard_mode"):
      mesh_from_config(config, devices=jax.devices()[:1])

  def test_every_field_read_or_refused_exists_on_the_config(self):
    """`getattr` defaults would silently disable a refusal if pyconfig renamed or removed the field."""
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    # pylint: disable=protected-access
    fields = [f"{net}_{axis}_parallelism" for net in ("ici", "dcn") for axis in m3_sharding._UNSUPPORTED_PARALLELISM_AXES]
    fields += list(m3_sharding._UNSUPPORTED_MESH_FIELDS)
    # pylint: enable=protected-access
    fields += [f"{net}_{axis}_parallelism" for net in ("ici", "dcn") for axis in m3_sharding.MESH_AXIS_NAMES]
    fields += ["shard_mode"]
    missing = [field for field in fields if not hasattr(config, field)]
    self.assertEqual(missing, [])

  @parameterized.parameters(
      ("ici_context_parallelism", 2),
      ("dcn_pipeline_parallelism", 2),
      ("ici_fsdp_transpose_parallelism", 4),
      ("ici_sequence_parallelism", 2),
      ("custom_mesh", "hybrid_ring_64x4"),
      ("custom_mesh_and_rule", "ep-as-cp"),
      ("custom_mesh_and_rule_for_eval", "pure-fsdp"),
      ("allow_split_physical_axes", True),
      ("optimize_mesh_for_tpu_v6e", True),
  )
  def test_unsupported_legacy_setting_is_refused(self, field, value):
    config = SimpleNamespace(**{field: value})
    self.assertEqual(m3_sharding.unsupported_mesh_settings(config), {field: value})
    with self.assertRaisesRegex(ValueError, f"does not implement.*{field}"):
      mesh_from_config(config, devices=jax.devices()[:1])

  @parameterized.parameters(
      ("ici_context_parallelism", 2),
      ("custom_mesh_and_rule", "ep-as-cp"),  # pyconfig turns this one into a `CustomRule` Enum
  )
  def test_refusal_against_a_real_config(self, field, value):
    """The field names and types are checked against pyconfig, so a change there surfaces here."""
    config = m3_test_utils.tiny_config("qwen3-0.6b", **{field: value})
    # pyconfig may derive related fields (custom_mesh_and_rule_for_eval defaults to custom_mesh_and_rule).
    self.assertEqual(m3_sharding.unsupported_mesh_settings(config).get(field), value)
    with self.assertRaisesRegex(ValueError, field):
      mesh_from_config(config, devices=jax.devices()[:1])

  def test_legacy_defaults_are_not_refused(self):
    self.assertEqual(m3_sharding.unsupported_mesh_settings(m3_test_utils.tiny_config("qwen3-0.6b")), {})


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
    shape = {"data": 1, "fsdp": 4, "fsdp_transpose": 2, "tensor": 1, "expert": 1}
    with self.assertRaisesRegex(ValueError, r"splits mesh axes \{'fsdp_transpose': 2\}"):
      m3_sharding.check_mesh(SimpleNamespace(axis_names=tuple(shape), shape=shape))

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
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(("fsdp", "expert"), "tensor"))

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

  def test_legacy_driver_sees_the_annotations(self):
    """`create_nnx_abstract_model` (the `from_pretrained` / training-engine path) traces construction
    under `nnx.eval_shape` inside the mesh; the abstract state must carry the strategy's specs,
    replicated norm scales included, rather than the driver's one-axis fallback."""
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    config = m3_test_utils.tiny_config("qwen3-0.6b")
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    _, abstract = model_creation_utils.create_nnx_abstract_model(config, mesh)
    state = {"/".join(map(str, p)): v for p, v in nnx.to_flat_state(nnx.state(abstract, nnx.Param))}
    kernel = state[f"{_layer(0)}/mlp/wi_0/kernel"]
    self.assertEqual(kernel.get_value().sharding.spec, P(("fsdp", "expert"), "tensor"))
    self.assertEqual(state["token_embedder/embedding"].get_value().sharding.spec, P("tensor", ("fsdp", "expert")))
    self.assertEqual(kernel.get_metadata()["out_sharding"], P(("fsdp", "expert"), "tensor"))
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
      raises=jax_core.ShardingTypeError,
      reason="Under an explicit mesh m3 does not yet pass the out_sharding its ops need, starting with the "
      "embedding gather; parameters place, the forward pass does not run. mesh_from_config refuses explicit.",
  )
  def test_forward_pass_under_an_explicit_mesh(self):
    config = self._config()
    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices, explicit=True)
    with jax.set_mesh(mesh):
      model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(0))
      batch = loop.make_synthetic_batch(config, mesh, loop.batch_spec(DefaultSharding()), jax.random.key(0))
      self.assertTrue(np.isfinite(loop.loss_fn(model, batch)))

  def test_setup_model_leaves_the_strategy_to_the_model(self):
    """Constructors differ on `sharding=` (GLM rejects unknown keywords), so the loop passes none."""
    config = self._config()
    mesh = mesh_from_config(config)
    with jax.set_mesh(mesh), mock.patch.object(m3_models, "create_model", wraps=m3_models.create_model) as create:
      loop.setup_model(config, mesh)
    self.assertNotIn("sharding", create.call_args.kwargs)

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
      batch = loop.make_synthetic_batch(config, mesh, loop.batch_spec(sharding), jax.random.key(0))
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
