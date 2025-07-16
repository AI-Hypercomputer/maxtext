#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the DiLoCo implementation in diloco.py"""

from collections.abc import Mapping
import dataclasses
import unittest
from typing import Any

import chex
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32, PyTree
import optax

import diloco
import pyconfig

_ParamsType = PyTree
_InputsType = Float32[Array, "5 2"]
_LabelsType = Float32[Array, "5"]
_BatchType = tuple[_InputsType, _LabelsType]


def _test_train_step(state: train_state.TrainState, batch: _BatchType, prng_key: diloco.PRNGKey):
  """A simple MSE loss train step to enable numerics testing."""
  del prng_key  # Unused.

  def predict(params: _ParamsType, inputs: _InputsType) -> Float32[Array, "5 1"]:
    # Map the model across the batch (0-th) dimension.
    return jax.vmap(state.apply_fn, in_axes=(None, 0))(params, inputs)

  def loss_fn(params: _ParamsType, batch: _BatchType) -> Float32[Array, ""]:
    inputs, labels = batch
    logits = predict(params, inputs)
    residual = logits - labels
    sq_residual = jnp.square(residual)
    msq_residual = jnp.mean(sq_residual)
    return msq_residual

  loss, grad = jax.value_and_grad(loss_fn)(state.params, batch)
  return state.apply_gradients(grads=grad), loss


class _SimpleModel(nn.Module):
  """A simple state for testing a minimal model."""

  @nn.compact
  def __call__(self, x):
    return nn.Dense(
        # 2 * x1 + 1 * x2 + 1 = y
        features=1,
        kernel_init=nn.with_partitioning(nn.initializers.constant(jnp.asarray([[2.0], [1.0]])), names="mlp"),
        bias_init=nn.with_partitioning(nn.initializers.ones_init(), names="mlp"),
    )(x)


@dataclasses.dataclass
class _TestConfig:
  """A fake config for test."""

  keys: Mapping[str, Any]


class DiLoCoTest(unittest.TestCase):

  def test_simple_model_2replicas_no_mesh(self):
    num_steps = 4
    test_config = pyconfig.HyperParameters(
        config=_TestConfig(
            keys={
                "num_diloco_replicas": 2,
                "diloco_outer_momentum": 0.9,
                "diloco_outer_lr": 1.0,
                "diloco_sync_period": num_steps - 1,
            }
        )
    )

    model = _SimpleModel()
    tx = optax.sgd(learning_rate=0.1)
    initial_params = model.init(jax.random.key(seed=42), jnp.ones((1, 2)))
    initial_test_state = train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=initial_params,
        tx=tx,
        opt_state=tx.init(initial_params),
    )

    diloco_test_state = diloco.build_diloco_state(test_config, lambda: initial_test_state)
    chex.assert_equal(diloco_test_state.step, 0)
    chex.assert_trees_all_equal(diloco_test_state.outer_params, initial_test_state.params)

    diloco_train_step = diloco.build_diloco_train_step(test_config, _test_train_step)
    inputs = jnp.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],  # First replica inputs.
            [[1.0, 0.0], [0.0, 1.0]],  # Second replica inputs.
        ]
    )
    labels = jnp.array(
        [
            [[1.0], [2.0]],  # First replica labels.
            [[2.0], [3.0]],  # Second replica labels.
        ]
    )

    # Run the first step (no synchronization).
    # Replica 0:
    #   Data: [[0, 1], [1, 0]]
    #   Labels: [[1], [2]]
    #   Weights: w = [[2], [1]]
    #   Bias: b = [1]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[2], [1]] + [1])) ^ 2 )
    #   = mean( ([[1], [2]] - [[2], [3]]) ^ 2 )
    #   = mean( ([-1, 1]) ^ 2 ) = mean( [1, 1] )
    #   = 1.0
    #
    # Replica 1:
    #   Data: [[1, 0], [0, 1]]
    #   Labels: [[2], [3]]
    #   Weights: w = [[2], [1]]
    #   Bias: b = [1]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[2], [1]] + [1])) ^ 2 )
    #   = mean( ([[2], [3]] - [[3], [2]]) ^ 2 )
    #   = mean( ([-1, 1]) ^ 2 ) = mean( [1, 1] )
    #   = 1.0
    diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
    chex.assert_equal(diloco_test_state.step, 1.0)
    chex.assert_equal(loss, 1.0)
    # Assert no updates to the global model yet (no synchronization)
    chex.assert_trees_all_equal(diloco_test_state.outer_params, initial_test_state.params)

    # Run the second step (no synchronization).
    # Replica 0:
    #   Data: [[0, 1], [1, 0]]
    #   Labels: [[1], [2]]
    #   Weights: w = [[1.9], [0.9]]
    #   Bias: b = [0.8]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.9], [0.9]] + [0.8])) ^ 2 )
    #   = mean( ([[1], [2]] - [[1.7], [2.7]]) ^ 2 )
    #   = mean( ([-0.7, 0.7]) ^ 2 ) = mean( [0.49, 0.49] )
    #   = 0.49
    #
    # Replica 1:
    #   Data: [[1, 0], [0, 1]]
    #   Labels: [[2], [3]]
    #   Weights: w = [[1.9], [1.1]]
    #   Bias: b = [1]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.9], [1.1]] + [1])) ^ 2 )
    #   = mean( ([[2], [3]] - [[2.9], [2.1]]) ^ 2 )
    #   = mean( ([-0.9, 0.9]) ^ 2 ) = mean( [0.81, 0.81] )
    #   = 0.81
    diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
    chex.assert_equal(diloco_test_state.step, 2.0)
    chex.assert_trees_all_close(loss, 0.65)
    # Assert no updates to the global model yet (no synchronization)
    chex.assert_trees_all_equal(diloco_test_state.outer_params, initial_test_state.params)

    # Run the third step, which synchronizes afterwards.
    # Replica 0:
    #   Data: [[0, 1], [1, 0]]
    #   Labels: [[1], [2]]
    #   Weights: w = [[1.83], [0.83]]
    #   Bias: b = [0.66]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.83], [0.83]] + [0.66])) ^ 2 )
    #   = mean( ([[1], [2]] - [[1.49], [2.49]]) ^ 2 )
    #   = mean( ([-0.49, 0.49]) ^ 2 ) = mean( [0.2401, 0.2401] )
    #   = 0.2401
    #
    # Replica 1:
    #   Data: [[1, 0], [0, 1]]
    #   Labels: [[2], [3]]
    #   Weights: w = [[1.81], [1.19]]
    #   Bias: b = [1.]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.81], [1.19]] + [1])) ^ 2 )
    #   = mean( ([[2], [3]] - [[2.81], [2.19]]) ^ 2 )
    #   = mean( ([-0.81, 0.81]) ^ 2 ) = mean( [0.6561, 0.6561] )
    #   = 0.6561
    #
    # After these are averaged, the model differences are computed to create a
    # pseudo-gradient update to the outer_params and applied via a momentum
    # based outer optimizer.
    diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
    chex.assert_equal(diloco_test_state.step, 3.0)
    chex.assert_trees_all_close(loss, 0.4481)
    # Assert that inner and outer parameters are all equal now that
    # synchronization has happened.
    chex.assert_trees_all_equal(
        diloco_test_state.outer_params,
        jax.tree.map(lambda arr: arr[0, ...], diloco_test_state.inner_state.params),
        diloco_test_state.outer_params,
    )
    chex.assert_trees_all_equal(
        diloco_test_state.outer_params,
        jax.tree.map(lambda arr: arr[1, ...], diloco_test_state.inner_state.params),
        diloco_test_state.outer_params,
    )

    # Run the fourth step (no synchronization).
    # Replica 0:
    #   Data: [[0, 1], [1, 0]]
    #   Labels: [[1], [2]]
    #   Weights: w = [[1.5345], [1.0494]]
    #   Bias: b = [0.5839]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[1], [2]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[1], [2]] - ([[0, 1], [1, 0]] . [[1.5345], [1.0494]]] + [0.5839])) ^ 2 )
    #   = mean( ([[1], [2]] - [[1.6333], [2.1184]]) ^ 2 )
    #   = mean( ([-0.6333, 0.1184]) ^ 2 ) = mean( [0.4010, 0.0140] )
    #   ~ 0.2075
    #
    # Replica 1:
    #   Data: [[1, 0], [0, 1]]
    #   Labels: [[2], [3]]
    #   Weights: w = [[1.5345], [1.0494]]
    #   Bias: b = [0.5839]
    #   Loss = mean((y - pred)^2) =
    #   = mean( ([[2], [3]] - (x . w + b)) ^ 2 ) )
    #   = mean( ([[2], [3]] - ([[1, 0], [0, 1]] . [[1.5345], [1.0494]] + [0.5839])) ^ 2 )
    #   = mean( ([[2], [3]] - [[2.1184], [1.6333]]) ^ 2 )
    #   = mean( ([-0.1184, 1.3667]) ^ 2 ) = mean( [0.0140, 1.8678] )
    #   ~ 0.94
    step_three_outer_params = diloco_test_state.outer_params
    diloco_test_state, loss = diloco_train_step(diloco_test_state, (inputs, labels), jax.random.key(seed=42))
    chex.assert_equal(diloco_test_state.step, 4.0)
    chex.assert_trees_all_close(loss, 0.574244)
    # Assert no updates to the global model since previous step (no
    # synchronization).
    chex.assert_trees_all_equal(diloco_test_state.outer_params, step_three_outer_params)
