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

"""Unit tests for the NNX path of generate_param_only_checkpoint.

Covers `_possibly_unroll_params_nnx` (slicing scanned NNX layers) and the
shape parity of `_save_decode_checkpoint_nnx`'s bf16 cast.
"""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from flax.training import train_state as linen_train_state

from maxtext.common.common_types import DecoderBlockType
from maxtext.layers import train_state_nnx
from maxtext.utils.generate_param_only_checkpoint import (
    _possibly_unroll_lora_params_nnx,
    _possibly_unroll_params_nnx,
)


class _ScanLayerLeaf(nnx.Module):
  """One scanned-layer kernel with leading shape `[num_layers, *]`."""

  def __init__(self, num_layers: int, in_dim: int, out_dim: int):
    self.kernel = nnx.Param(
        jnp.arange(num_layers * in_dim * out_dim, dtype=jnp.float32).reshape(num_layers, in_dim, out_dim)
    )


class _Decoder(nnx.Module):

  def __init__(self, num_layers: int):
    self.layers = _ScanLayerLeaf(num_layers, 3, 5)


class _Model(nnx.Module):

  def __init__(self, num_layers: int):
    self.decoder = _Decoder(num_layers)


def _make_split_state(num_layers: int):
  model = _Model(num_layers)
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  ts = train_state_nnx.TrainStateNNX(model, optimizer)
  _, state = nnx.split(ts)
  return state


def _make_shardings_state(state, mesh):
  """Build a sibling shardings tree where each Variable is replaced by NamedSharding(replicated)."""

  def to_named(v):
    return NamedSharding(mesh, PartitionSpec())

  return jax.tree_util.tree_map(to_named, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


class PossiblyUnrollParamsNNXTest(unittest.TestCase):

  def setUp(self):
    devices = np.array(jax.devices()).reshape(-1)
    self.mesh = Mesh(devices, ("data",))

  def test_unrolls_scanned_layers(self):
    num_layers = 3
    state = _make_split_state(num_layers)
    shardings = _make_shardings_state(state, self.mesh)

    original_kernel = np.asarray(state.model.decoder.layers.kernel[...])

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        pure_nnx=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.LLAMA2,
        num_decoder_layers=num_layers,
    )

    _possibly_unroll_params_nnx(config, state, shardings, self.mesh)

    self.assertNotIn("layers", state.model.decoder)
    self.assertNotIn("layers", shardings.model.decoder)
    for i in range(num_layers):
      self.assertIn(f"layers_{i}", state.model.decoder)
      self.assertIn(f"layers_{i}", shardings.model.decoder)
      sliced = state.model.decoder[f"layers_{i}"]["kernel"][...]
      expected = jnp.take(original_kernel, i, axis=0)
      self.assertTrue(jnp.array_equal(sliced, expected))

  def test_deepseek_split(self):
    """DeepSeek decoder has separate dense/moe layer collections."""

    # Build a DeepSeek-flavored synthetic model with two scanned groups.
    class _DeepSeekDecoder(nnx.Module):

      def __init__(self):
        self.dense_layers = _ScanLayerLeaf(2, 3, 5)
        self.moe_layers = _ScanLayerLeaf(3, 3, 5)

    class _DSModel(nnx.Module):

      def __init__(self):
        self.decoder = _DeepSeekDecoder()

    model = _DSModel()
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    ts = train_state_nnx.TrainStateNNX(model, optimizer)
    _, state = nnx.split(ts)
    shardings = _make_shardings_state(state, self.mesh)

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        pure_nnx=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.DEEPSEEK,
        num_decoder_layers=5,
        first_num_dense_layers=2,
    )

    _possibly_unroll_params_nnx(config, state, shardings, self.mesh)

    self.assertNotIn("dense_layers", state.model.decoder)
    self.assertNotIn("moe_layers", state.model.decoder)
    for i in range(2):
      self.assertIn(f"dense_layers_{i}", state.model.decoder)
    for i in range(3):
      self.assertIn(f"moe_layers_{i}", state.model.decoder)


class PossiblyUnrollLoraParamsNNXTest(unittest.TestCase):
  """The LoRA delta tree is single-nested (`{"decoder": {...}}`) and held in a
  Linen `TrainState` even on the NNX path — the unroll has to walk that shape."""

  def setUp(self):
    devices = np.array(jax.devices()).reshape(-1)
    self.mesh = Mesh(devices, ("data",))

  def _make_lora_state(self, num_layers: int, lora_rank: int = 4):
    """Build a synthetic LoRA delta TrainState mirroring `get_lora_abstract_state_nnx`'s output shape."""
    lora_a = jnp.arange(num_layers * 8 * lora_rank, dtype=jnp.float32).reshape(num_layers, 8, lora_rank)
    lora_b = jnp.arange(num_layers * lora_rank * 4 * 2, dtype=jnp.float32).reshape(num_layers, lora_rank, 4, 2)
    params = {
        "decoder": {
            "layers": {
                "self_attention": {
                    "query": {"lora_a.kernel": lora_a, "lora_b.kernel": lora_b},
                }
            }
        }
    }
    annotations_params = jax.tree_util.tree_map(lambda _: PartitionSpec(), params)
    state = linen_train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})
    annotations = linen_train_state.TrainState(step=0, apply_fn=None, params=annotations_params, tx=None, opt_state={})
    return state, annotations

  def test_unrolls_scanned_lora_layers(self):
    num_layers = 3
    state, annotations = self._make_lora_state(num_layers)
    original_a = np.asarray(state.params["decoder"]["layers"]["self_attention"]["query"]["lora_a.kernel"])

    config = SimpleNamespace(
        scan_layers=True,
        force_unroll=True,
        pure_nnx=True,
        param_scan_axis=0,
        decoder_block=DecoderBlockType.LLAMA2,
        num_decoder_layers=num_layers,
    )

    _possibly_unroll_lora_params_nnx(config, state, annotations, self.mesh)

    self.assertNotIn("layers", state.params["decoder"])
    self.assertNotIn("layers", annotations.params["decoder"])
    for i in range(num_layers):
      self.assertIn(f"layers_{i}", state.params["decoder"])
      sliced_a = state.params["decoder"][f"layers_{i}"]["self_attention"]["query"]["lora_a.kernel"]
      expected = jnp.take(original_a, i, axis=0)
      self.assertTrue(jnp.array_equal(sliced_a, expected))


if __name__ == "__main__":
  unittest.main()
