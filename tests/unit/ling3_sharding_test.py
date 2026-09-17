# Copyright 2023–2026 Google LLC
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

"""Ling expert placement and numerical parity on a multi-device mesh.

CPU example: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
PYTHONPATH=src ../venv1/bin/python -m unittest tests.unit.ling3_sharding_test
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from flax import linen as nn, nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.models.ling3 import Ling3MoE
from maxtext.utils import model_creation_utils
from maxtext.utils.maxtext_utils_nnx import create_nnx_sharded_model
from maxtext.utils.sharding import nnx_construct_named_sharding
from tests.unit.ling3_layers_test import small_config


class Ling3ShardingTest(unittest.TestCase):

  @unittest.skipIf(jax.device_count() < 4, "Requires four devices")
  def test_restore_without_materializing_weights(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("expert", "tensor"))
    rules = (("exp", "expert"), ("embed_moe", None), ("mlp_moe", "tensor"))
    cfg = small_config()

    def initialize():
      model = Ling3MoE(cfg, mesh, "prefill", 2, rngs=nnx.Rngs(42))
      model.cache = nnx.Cache(jnp.zeros((2, 4)))
      model.rngs = nnx.Rngs(17)
      return model

    abstract = nnx.eval_shape(initialize)
    graph, state = nnx.split(abstract)
    with nn.logical_axis_rules(rules):
      shardings = nnx_construct_named_sharding(state, mesh)
    state = jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), state, shardings)
    abstract = nnx.merge(graph, state)
    source = initialize()
    weights = nnx.to_pure_dict(nnx.state(source, nnx.Param))

    for linen_format in (False, True):
      with self.subTest(linen_format=linen_format), tempfile.TemporaryDirectory() as directory:
        path = directory + "/checkpoint"
        payload = {"params": {"params": weights}} if linen_format else jax.tree.map(lambda x: {"value": x}, weights)
        with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as writer:
          writer.save(path, payload)
        config = SimpleNamespace(
            convert_checkpoint_if_possible=False,
            load_parameters_path=path,
            checkpoint_storage_concurrent_gb=1,
            checkpoint_storage_use_ocdbt=True,
            checkpoint_storage_use_zarr3=False,
        )
        restore = ocp.Checkpointer.restore

        def checked_restore(checkpointer, *args, **kwargs):
          leaves = jax.tree.leaves(kwargs["item"])
          self.assertTrue(leaves)
          self.assertTrue(all(isinstance(x, jax.ShapeDtypeStruct) for x in leaves))
          return restore(checkpointer, *args, **kwargs)

        with (
            mock.patch.object(model_creation_utils, "verify_and_sync_scan_layers", side_effect=lambda c: c),
            mock.patch.object(model_creation_utils, "create_nnx_abstract_model", return_value=(initialize, abstract)),
            mock.patch.object(ocp.Checkpointer, "restore", checked_restore),
        ):
          loaded = model_creation_utils.from_pretrained(config, mesh=mesh)
        self.assertTrue(all(isinstance(x, jax.Array) for x in jax.tree.leaves(nnx.state(loaded))))
        loaded_weights = nnx.to_pure_dict(nnx.state(loaded, nnx.Param))
        for expected, actual in zip(jax.tree.leaves(weights), jax.tree.leaves(loaded_weights)):
          np.testing.assert_array_equal(expected, actual)
        self.assertEqual(loaded.wi_0.get_value().sharding.spec, P("expert", None, "tensor"))
        np.testing.assert_array_equal(loaded.cache.get_value(), source.cache.get_value())

  @unittest.skipIf(jax.device_count() < 4, "Requires four devices")
  def test_decode_initialization_sharding(self):
    """Exercise the production state creator with decode's RNG implementation."""
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("expert", "tensor"))
    rules = (("exp", "expert"), ("embed_moe", None), ("mlp_moe", "tensor"))
    cfg = small_config()

    def initialize():
      return Ling3MoE(cfg, mesh, "prefill", 2, rngs=nnx.Rngs(jax.random.key(42, impl="unsafe_rbg")))

    abstract = nnx.eval_shape(initialize)
    with nn.logical_axis_rules(rules):
      shardings = nnx_construct_named_sharding(nnx.state(abstract), mesh)
      model = create_nnx_sharded_model(abstract, initialize, mesh, shardings)
    for name, spec in (
        ("wi_0", P("expert", None, "tensor")),
        ("wi_1", P("expert", None, "tensor")),
        ("wo", P("expert", "tensor", None)),
    ):
      arr = getattr(model, name).get_value()
      self.assertEqual(arr.sharding.spec, spec)
      self.assertEqual(arr.addressable_shards[0].data.size, arr.size // 4)

  @unittest.skipIf(jax.device_count() < 4, "Requires four devices")
  def test_expert_tensor_sharding_and_forward(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("expert", "tensor"))
    rules = (("exp", "expert"), ("embed_moe", None), ("mlp_moe", "tensor"))
    cfg = small_config()
    for index in (2, 35, 40):
      with self.subTest(layer=index):
        model = Ling3MoE(cfg, mesh, "prefill", index, rngs=nnx.Rngs(42))
        graph, state = nnx.split(model)
        with nn.logical_axis_rules(rules):
          shardings = nnx_construct_named_sharding(state, mesh)
        self.assertEqual(shardings.wi_0.get_value().spec, P("expert", None, "tensor"))
        self.assertEqual(shardings.wo.get_value().spec, P("expert", "tensor", None))

        def initialize():
          return nnx.state(Ling3MoE(cfg, mesh, "prefill", index, rngs=nnx.Rngs(42)))

        placed = jax.jit(initialize, out_shardings=shardings)()
        for name in ("wi_0", "wi_1", "wo"):
          arr = placed[name].get_value()
          self.assertEqual(arr.addressable_shards[0].data.size, arr.size // 4)
        x = jax.random.normal(jax.random.key(7), (2, 3, cfg.emb_dim))
        # Every token selects experts on both halves of the expert mesh.
        forced = jnp.array([[[0, 7], [1, 6], [2, 5]], [[3, 4], [7, 0], [6, 1]]])
        for routing in (None, forced):
          expected = model(x, forced_routed_experts=routing)

          def forward(s, inputs):
            return nnx.merge(graph, s)(inputs, forced_routed_experts=routing)

          run = jax.jit(forward, in_shardings=(shardings, NamedSharding(mesh, P())))
          actual = run(placed, x)
          np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

        # Existing checkpoints contain global arrays; restore into the new
        # placement without changing tensor names, shapes, or conversion hooks.
        if index == 2:
          original = nnx.to_pure_dict(state)
          target = jax.tree.map(
              lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
              nnx.to_pure_dict(placed),
          )
          with tempfile.TemporaryDirectory() as directory:
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(directory + "/checkpoint", original)
            checkpointer.wait_until_finished()
            restored = checkpointer.restore(directory + "/checkpoint", target=target)
            checkpointer.close()
          self.assertEqual(restored["wi_0"].sharding.spec, P("expert", None, "tensor"))
          nnx.update(placed, restored)
          np.testing.assert_allclose(run(placed, x), expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
  unittest.main()
