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

"""Unit tests for the NNX branches of load_state_if_possible."""

import os
import tempfile
import unittest
from unittest import mock

from etils import epath
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from maxtext.common import checkpointing
from maxtext.layers import train_state_nnx


class _Model(nnx.Module):
  """Tiny single-linear NNX model for restore tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


def _abstract_nnx_state():
  """Build an nnx.State from a TrainStateNNX — same shape that pre_train passes in."""
  model = _Model(rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  return nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))


class TestLoadStateIfPossibleNNX(unittest.TestCase):
  """Cover the NNX branches in load_state_if_possible."""

  def test_load_parameters_from_path_splits_nnx_state_for_param_view(self):
    """When abstract_unboxed_pre_state is an nnx.State, the function must call
    nnx.split(model, nnx.Param, ...) to get the params and forward them to load_params_from_path."""
    abstract = _abstract_nnx_state()
    sentinel_restored = {"linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.zeros((1,))}}

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel_restored) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=abstract,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel_restored)
    m.assert_called_once()
    forwarded_params = m.call_args[0][1]  # second positional arg = abstract_unboxed_params
    # The forwarded params come from nnx.split(..., nnx.Param, ...) — same key shape as the model.
    leaves = jax.tree.leaves(forwarded_params)
    self.assertEqual(len(leaves), 2)  # linear.kernel + linear.bias

  def test_load_parameters_from_path_uses_state_params_for_linen(self):
    """For Linen TrainState, the function must use state.params (not nnx.split)."""
    fake_state = mock.Mock(spec=["params"])
    fake_state.params = {"layer": {"kernel": jnp.ones((2, 2))}}
    sentinel = object()

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=fake_state,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel)
    forwarded_params = m.call_args[0][1]
    self.assertIs(forwarded_params, fake_state.params)

  def test_no_paths_returns_none_none(self):
    """Sanity: with no checkpoint manager and no load paths, the function returns (None, None)."""
    full, params = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_nnx_state(),
    )
    self.assertIsNone(full)
    self.assertIsNone(params)


class TestLoadParamsIntoNNX(unittest.TestCase):
  """Weight-only load (load_parameters_path) of a Linen-layout checkpoint into NNX."""

  def test_linen_layout_params_restore_into_nnx_state(self):
    """load_params_from_path reshapes an on-disk Linen-layout checkpoint into the NNX params state."""
    model = _Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)
    weights = {"linear": {"kernel": jnp.arange(2, dtype=jnp.float32).reshape(2, 1), "bias": jnp.array([5.0])}}

    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      path = os.path.join(d, "ckpt")
      # On-disk Linen layout: params/params/<weights> plus an unrelated `step`.
      ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
          epath.Path(path), {"params": {"params": weights}, "step": jnp.array(3)}, force=True
      )
      restored = checkpointing.load_params_from_path(path, params_abstract, 8)

    self.assertIsInstance(restored, nnx.State)
    pure = restored.to_pure_dict()
    self.assertTrue(jnp.array_equal(pure["linear"]["kernel"], weights["linear"]["kernel"]))
    self.assertTrue(jnp.array_equal(pure["linear"]["bias"], weights["linear"]["bias"]))


if __name__ == "__main__":
  unittest.main()
