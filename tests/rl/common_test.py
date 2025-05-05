# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)


class CommonTest(absltest.TestCase):

  def test_selective_log_softmax(self):
    rng = jax.random.PRNGKey(0)
    logits = jax.random.uniform(rng, shape=(2, 4, 8))
    input_ids = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    per_token_logps = common.selective_log_softmax(logits, input_ids)
    jitted_per_token_logps = jax.jit(common.selective_log_softmax)(
        logits, input_ids
    )
    expected_value = jnp.array([
        [-2.242679, -2.2733693, -2.1024966, -1.9994389],
        [-2.0603075, -2.4863663, -1.9176172, -2.0206313],
    ])
    np.testing.assert_allclose(per_token_logps, expected_value)
    np.testing.assert_allclose(per_token_logps, jitted_per_token_logps)

  def test_get_per_token_logps(self):
    rng = jax.random.PRNGKey(0)
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    input_tokens = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    positions = jnp.ones((2, 4))
    attn_mask = common.make_causal_attn_mask(positions)
    per_token_logps = common.get_per_token_logps(
        model, input_tokens, positions, attn_mask, logits_to_keep=2
    )
    np.testing.assert_allclose(
        per_token_logps,
        np.array([[-5.7448483, -5.937829], [-4.222273, -4.41953]]),
    )

  def test_make_completion_mask(self):
    completion_ids = jnp.array([
        [1, 2, 3, 4],
        [1, 2, 3, 0],
        [1, 2, 0, 0],
        [1, 0, 0, 0],
    ])
    completion_mask = common.make_completion_mask(completion_ids)
    expected_value = jnp.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    np.testing.assert_allclose(completion_mask, expected_value)

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array([
        [True, True, True, True],
        [True, True, True, False],
        [False, True, True, False],
    ])
    attn_mask = common.make_causal_attn_mask(input_mask)
    expected_value = jnp.array([
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, False],
        ],
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
            [False, True, True, False],
        ],
    ])
    np.testing.assert_allclose(attn_mask, expected_value)

  def test_pad_to_length(self):
    x = jnp.ones((2, 4))
    padded_x = common.pad_to_length(x, target_length=5)
    self.assertEqual(padded_x.shape, (5, 4))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, pad_value=1, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 10)
    padded_x = common.pad_to_length(x, target_length=3, axis=-1)
    np.testing.assert_array_equal(padded_x, x)
    padded_x = common.pad_to_length(x, target_length=5, left=True, axis=-1)
    np.testing.assert_array_equal(
        padded_x, jnp.array([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    )

  def test_build_positions_from_mask(self):
    input_mask = jnp.array(
        [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]]
    )
    positions = common.build_positions_from_mask(input_mask)
    expected_value = jnp.array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 1, 2, 2],
        [0, 0, 1, 1],
    ])
    np.testing.assert_array_equal(positions, expected_value)

if __name__ == "__main__":
  absltest.main()
