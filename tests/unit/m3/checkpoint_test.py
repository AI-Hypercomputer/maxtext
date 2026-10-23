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

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    self.assertEqual(set(saved["params"]["params"]["decoder"]), {"layers_0", "layers_1"})
    self.assertEqual(set(saved["nnx_aux"]["model"]["decoder"]), {"layers_0", "layers_1"})
    moments = saved["opt_state"][0]["mu"]["params"]["decoder"]
    self.assertEqual(set(moments), {"layers_0", "layers_1"})
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    restored = checkpointing.restored_linen_to_nnx(saved, abstract)
    self.assertEqual(jax.tree.structure(state), jax.tree.structure(restored))
    for original, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(original, actual)

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
