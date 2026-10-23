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

"""Checkpoint compatibility for container decoder layers."""

import unittest
import pathlib
import tempfile
from unittest import mock
from collections import namedtuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils

from maxtext.common import checkpointing, train_state_nnx
from maxtext.m3.core.checkpoint import from_legacy_layer_names, to_legacy_layer_names
from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco.utils import spmd_diloco_checkpointing


class Layer(nnx.Module):

  def __init__(self, value):
    self.weight = nnx.Param(jnp.array([value], dtype=jnp.float32))
    self.stat = nnx.BatchStat(jnp.array([value + 1], dtype=jnp.float32))


class Model(nnx.Module):

  def __init__(self):
    self.decoder = nnx.Dict(layers=nnx.List([Layer(1), Layer(2)]))


class CheckpointTest(unittest.TestCase):

  def test_from_pretrained_legacy_qwen3_checkpoint(self):
    with tempfile.TemporaryDirectory(prefix="m3-checkpoint-") as directory:
      args = [
          "",
          "src/maxtext/configs/base.yml",
          "model_name=qwen3-0.6b",
          "use_m3_model=true",
          "scan_layers=false",
          "override_model_config=true",
          "base_num_decoder_layers=2",
          "base_emb_dim=32",
          "base_mlp_dim=64",
          "base_num_query_heads=2",
          "base_num_kv_heads=1",
          "head_dim=16",
          "vocab_size=64",
          "per_device_batch_size=1",
          "max_target_length=8",
          "max_prefill_predict_length=4",
          "skip_jax_distributed_system=true",
          "enable_checkpointing=false",
          f"base_output_directory={directory}/output",
          "run_name=checkpoint-test",
      ]
      config = pyconfig.initialize_pydantic(args)
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(config.mesh_axes)), config.mesh_axes)
      with jax.set_mesh(mesh):
        model = model_creation_utils.create_model(config, mesh, rngs=nnx.Rngs(42))
        params = nnx.state(model, nnx.Param).to_pure_dict()
        self.assertEqual(set(params["decoder"]["layers"]), {0, 1})
        legacy = {"params": {"params": to_legacy_layer_names(params)}}
        path = pathlib.Path(directory) / "weights"
        with ocp.PyTreeCheckpointer() as checkpointer:
          checkpointer.save(path, legacy)
        restore_config = config.model_copy(update={"load_parameters_path": str(path)})
        restored = model_creation_utils.from_pretrained(restore_config, mesh=mesh, rng_key=jax.random.key(7))
        restored_params = nnx.state(restored, nnx.Param).to_pure_dict()
        self.assertEqual(set(restored_params["decoder"]["layers"]), {0, 1})
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        positions = jnp.arange(4, dtype=jnp.int32)[None, :]
        expected = model(tokens, positions, enable_dropout=False)
        actual = restored(tokens, positions, enable_dropout=False)
        np.testing.assert_array_equal(expected, actual)

  def test_weights_only_legacy_checkpoint(self):
    state = nnx.state(train_state_nnx.TrainStateNNX(Model(), None))
    expected = state["model"].filter(nnx.Param)
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    with tempfile.TemporaryDirectory(prefix="m3-weights-") as directory:
      path = pathlib.Path(directory) / "weights"
      with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, train_state_nnx.to_checkpoint_dict(state))
      _, restored = checkpointing.load_state_if_possible(
          None,
          None,
          str(path),
          "",
          1,
          abstract,
      )
    self.assertEqual(set(restored["decoder"]["layers"]), {0, 1})
    self.assertEqual(jax.tree.structure(expected), jax.tree.structure(restored))
    for original, actual in zip(jax.tree.leaves(expected), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(original, actual)

  def test_dynamic_safetensors_uses_legacy_mapping_paths(self):
    state = nnx.state(train_state_nnx.TrainStateNNX(Model(), None))
    expected = state["model"].filter(nnx.Param)
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    legacy = to_legacy_layer_names(expected.to_pure_dict())

    def load_weights(path, target, config):
      self.assertEqual(set(target["decoder"]), {"layers_0", "layers_1"})
      return None, {"params": legacy}

    with mock.patch.object(checkpointing, "load_safetensors_dynamic_state", side_effect=load_weights):
      _, restored = checkpointing.load_state_if_possible(
          None,
          None,
          "/tmp/hf-weights",
          "",
          1,
          abstract,
          source_checkpoint_layout="safetensors_dynamic",
      )
    self.assertEqual(jax.tree.structure(expected), jax.tree.structure(restored))
    for original, actual in zip(jax.tree.leaves(expected), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(original, actual)

  def test_diloco_checkpoint_layer_names_roundtrip(self):
    inner = nnx.state(train_state_nnx.TrainStateNNX(Model(), None))
    params = inner["model"].filter(nnx.Param).to_pure_dict()
    outer_optimizer = optax.sgd(0.1, momentum=0.9)
    state = diloco.DiLoCoTrainState(
        inner_state=inner,
        params=params,
        outer_opt_state=outer_optimizer.init(params),
        step=jnp.array(1),
    )
    saved = spmd_diloco_checkpointing.to_diloco_checkpoint_dict(state)
    self.assertEqual(set(saved["params"]["params"]["decoder"]), {"layers_0", "layers_1"})
    restored = spmd_diloco_checkpointing.from_diloco_checkpoint_dict(saved, state)
    self.assertEqual(jax.tree.structure(state), jax.tree.structure(restored))
    for original, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(original, actual)

  def test_full_train_state_roundtrip(self):
    model = Model()
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    grads = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))
    optimizer.update(model, grads)
    state = nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))
    saved = train_state_nnx.to_checkpoint_dict(state)
    direct = train_state_nnx.to_linen_checkpoint_dict(state.to_pure_dict())
    self.assertEqual(set(direct["params"]["params"]["decoder"]), {"layers_0", "layers_1"})
    self.assertEqual(set(saved["params"]["params"]["decoder"]), {"layers_0", "layers_1"})
    self.assertEqual(set(saved["nnx_aux"]["model"]["decoder"]), {"layers_0", "layers_1"})
    moments = saved["opt_state"][0]["mu"]["params"]["decoder"]
    self.assertEqual(set(moments), {"layers_0", "layers_1"})
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    restored = checkpointing.restored_linen_to_nnx(saved, abstract)
    self.assertEqual(jax.tree.structure(state), jax.tree.structure(restored))
    for original, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(original, actual)

  def test_nested_optimizer_containers_and_leaf_identity(self):
    moment = namedtuple("Moment", ["mu", "nu"])
    leaf = jnp.ones((2,))
    weights = {"decoder": {"layers": {0: {"weight": leaf}, 1: {"weight": leaf}}}, "other": {"layers_0": leaf}}
    tree = {"params": weights, "optimizer": [moment(weights, (weights,))]}
    saved = to_legacy_layer_names(tree)
    self.assertIs(saved["params"]["decoder"]["layers_0"]["weight"], leaf)
    self.assertIs(saved["params"]["other"]["layers_0"], leaf)
    restored = from_legacy_layer_names(saved)
    self.assertIsInstance(restored["optimizer"], list)
    self.assertIsInstance(restored["optimizer"][0], moment)
    self.assertIsInstance(restored["optimizer"][0].nu, tuple)
    self.assertEqual(jax.tree.structure(tree), jax.tree.structure(restored))
    self.assertTrue(all(a is b for a, b in zip(jax.tree.leaves(tree), jax.tree.leaves(restored))))

  def test_scanned_and_legacy_targets_passthrough(self):
    leaf = jnp.ones((2,))
    scanned = {"decoder": {"layers": {"attention": {"weight": leaf}}}}
    self.assertIs(to_legacy_layer_names(scanned)["decoder"]["layers"]["attention"]["weight"], leaf)
    legacy = {"decoder": {"layers_0": {"weight": leaf}}}
    self.assertIs(from_legacy_layer_names(legacy, scanned), legacy)
    self.assertIs(from_legacy_layer_names(legacy, {"model": legacy}), legacy)
    target = {"model": {"decoder": {"layers": {0: {"weight": leaf}}}}}
    self.assertIs(from_legacy_layer_names(legacy, target)["decoder"]["layers"][0]["weight"], leaf)


if __name__ == "__main__":
  unittest.main()
