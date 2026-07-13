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

"""Unit tests for Gemma 4 decoder blocks."""

from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import AttentionType, MODEL_MODE_AUTOREGRESSIVE
from maxtext.models import gemma4


class _StatefulDecoderLayer(nnx.Module):
  """Small stand-in that exposes cache ordering and mutable-state updates."""

  def __init__(self, *, attention_type, **unused_kwargs):
    self.increment = 10 if attention_type == AttentionType.GLOBAL else 1
    self.call_count = nnx.Intermediate(jnp.array(0, dtype=jnp.int32))
    self.received_attention_metadata = nnx.Intermediate(jnp.array(False))

  def __call__(
      self,
      inputs,
      *unused_args,
      kv_cache=None,
      attention_metadata=None,
      **unused_kwargs,
  ):
    self.call_count.value += 1
    self.received_attention_metadata.value = attention_metadata is not None
    output = inputs + self.increment
    if kv_cache is None:
      return output
    return output, kv_cache + self.increment


def test_scannable_block_updates_state_through_global_single_iteration_scan():
  config = SimpleNamespace(
      dtype=jnp.float32,
      param_scan_axis=1,
      remat_policy="none",
      scan_layers=True,
  )

  with mock.patch.object(gemma4, "Gemma4DecoderLayer", _StatefulDecoderLayer):
    block = gemma4.Gemma4ScannableBlock(
        config=config,
        mesh=None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        rngs=nnx.Rngs(0),
    )
    output, updated_kvs = block(
        jnp.zeros((1, 1, 1)),
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )

  np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 15))
  assert updated_kvs is None
  np.testing.assert_array_equal(
      block.local_layers.call_count.value, jnp.ones(5, dtype=jnp.int32)
  )
  np.testing.assert_array_equal(block.global_layer.call_count.value, 1)


def test_scannable_block_restores_local_state_and_preserves_kv_order():
  config = SimpleNamespace(
      dtype=jnp.float32,
      param_scan_axis=1,
      remat_policy="none",
      scan_layers=True,
  )
  attention_metadata = object()

  with mock.patch.object(gemma4, "Gemma4DecoderLayer", _StatefulDecoderLayer):
    block = gemma4.Gemma4ScannableBlock(
        config=config,
        mesh=None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        rngs=nnx.Rngs(0),
    )
    output, updated_kvs = block(
        jnp.zeros((1, 1, 1)),
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        kv_cache=tuple(jnp.array(i) for i in range(6)),
        attention_metadata=attention_metadata,
    )

  np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 15))
  np.testing.assert_array_equal(jnp.stack(updated_kvs), jnp.array([1, 2, 3, 4, 5, 15]))
  np.testing.assert_array_equal(block.local_layers.call_count.value, jnp.ones(5, dtype=jnp.int32))
  np.testing.assert_array_equal(block.local_layers.received_attention_metadata.value, jnp.ones(5, dtype=jnp.bool_))
  np.testing.assert_array_equal(block.global_layer.call_count.value, 1)
  np.testing.assert_array_equal(block.global_layer.received_attention_metadata.value, True)
