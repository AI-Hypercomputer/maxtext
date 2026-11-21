# Copyright 2025-2026 Google LLC
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

"""UntrainableParam is stored and loaded like nnx.Param, but never trained. Drives the real save and restore on CPU."""
# This suite unit-tests private restore helpers directly, by design.
# pylint: disable=protected-access

import shutil
import tempfile
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.common.common_types import UntrainableParam, variable_collection_name
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.optimizers import optimizers

_BOUND = np.asarray([1.5, 2.5, 3.5], np.float32)


class _Model(nnx.Module):
  """A Param, an untrainable weight, and RNG state."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.bound = UntrainableParam(jnp.asarray(_BOUND))
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)


class _Clipped(nnx.Module):
  """A Param and an untrainable weight, callable so the Linen bridge can wrap it."""

  def __init__(self):
    self.kernel = nnx.Param(jnp.ones(3))
    self.bound = UntrainableParam(jnp.asarray(_BOUND))

  def __call__(self, x):
    return jnp.minimum(x * self.kernel[...], self.bound[...])


def _train_state(model):
  tx = optimizers.freeze_params(optax.adamw(1e-2, weight_decay=0.5), optimizers.untrainable_paths(model))
  return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param)))


def _abstract_train_state():
  return nnx.eval_shape(lambda: _train_state(_Model(nnx.Rngs(9))))


def _checkpoint():
  return train_state_nnx.to_checkpoint_dict(_train_state(_Model(nnx.Rngs(0))))


class UntrainableParamGradientTest(unittest.TestCase):

  def test_gradient_is_zero_however_the_value_is_read(self):
    reads = {
        "index": lambda v: v[...],
        "get_value": lambda v: v.get_value(),
        "value": lambda v: v.value,
        "asarray": jnp.asarray,
        "arithmetic": lambda v: v + 0.0,
    }
    for name, read in reads.items():
      with self.subTest(name):
        self.assertEqual(float(jax.grad(lambda x, r=read: r(UntrainableParam(x)) ** 2)(2.0)), 0.0)
        self.assertEqual(float(jax.grad(lambda x, r=read: r(nnx.Param(x)) ** 2)(2.0)), 4.0)

  def test_reads_an_abstract_value_without_failing(self):
    abstract = nnx.eval_shape(lambda: UntrainableParam(jnp.ones(2)))
    self.assertEqual(abstract.get_value().shape, (2,))


class UntrainableParamCheckpointTest(unittest.TestCase):

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  def _save(self, checkpoint):
    root = self._dir + "/ckpt"
    checkpointing.ocp.save(root, checkpoint, checkpointable_name="items")
    return root + "/items"

  def _resume(self, path):
    return checkpointing._load_linen_checkpoint_into_nnx(path, _abstract_train_state(), 8, True, True)

  def test_saved_with_the_regular_params(self):
    checkpoint = _checkpoint()

    self.assertEqual(set(checkpoint["params"]), {"params"})
    self.assertEqual(set(checkpoint["params"]["params"]), {"linear", "bound"})
    self.assertEqual(set(checkpoint["nnx_aux"]["model"]), {"dropout"})

  def test_resume_restores_type_and_value(self):
    checkpoint = _checkpoint()

    restored = self._resume(self._save(checkpoint))

    self.assertIsInstance(restored.model.bound, UntrainableParam)
    np.testing.assert_array_equal(np.asarray(restored.model.bound[...]), _BOUND)
    np.testing.assert_array_equal(
        np.asarray(restored.model.linear.kernel[...]), np.asarray(checkpoint["params"]["params"]["linear"]["kernel"])
    )

  def test_resume_raises_when_weight_is_missing(self):
    checkpoint = _checkpoint()
    del checkpoint["params"]["params"]["bound"]

    with self.assertRaisesRegex(ValueError, "bound"):
      self._resume(self._save(checkpoint))

  def test_weights_only_load_reads_the_params_collection(self):
    path = self._save(_checkpoint())
    abstract = checkpointing._abstract_params(_abstract_train_state())

    with mock.patch.object(checkpointing.max_logging, "log") as log:
      restored = checkpointing.load_params_from_path(path, abstract, 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["bound"]), _BOUND)
    self.assertNotIn("predates", " ".join(str(c) for c in log.call_args_list))


class UntrainableParamNamingTest(unittest.TestCase):

  def test_collection_name_matches_regular_params(self):
    class _Custom(nnx.Variable):  # pylint: disable=abstract-method
      pass

    self.assertEqual(variable_collection_name(UntrainableParam), "params")
    self.assertEqual(variable_collection_name(nnx.Param), "params")
    self.assertEqual(variable_collection_name(_Custom), "_Custom")

  def test_scanned_sharding_matches_regular_params(self):
    metadata = {"out_sharding": ("x",), "param_scan_axis": 1, nnx.PARTITION_NAME: "layers"}
    regular = variable_to_logically_partitioned(nnx.Param(jnp.ones((2, 3)), **metadata))
    untrainable = variable_to_logically_partitioned(UntrainableParam(jnp.ones((2, 3)), **metadata))

    self.assertEqual(untrainable.names, ("x", "layers"))
    self.assertEqual(untrainable.names, regular.names)


class UntrainableParamLinenBridgeTest(unittest.TestCase):

  def test_bridge_keeps_it_in_the_params_collection(self):
    model = nnx_wrappers.to_linen(_Clipped, skip_rng=True)
    x = jnp.full(3, 10.0)
    variables = model.init(jax.random.key(0), x)
    self.assertEqual(set(variables), {"params"})

    loaded = {"params": {**variables["params"], "bound": jnp.asarray(_BOUND + 1)}}
    np.testing.assert_array_equal(model.apply(loaded, x), _BOUND + 1)
    grads = jax.grad(lambda variables: model.apply(variables, x).sum())(loaded)
    np.testing.assert_array_equal(grads["params"]["bound"], np.zeros(3))


if __name__ == "__main__":
  unittest.main()
