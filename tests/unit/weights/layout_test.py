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

"""Tests for weight layout metadata: name flattening and leaf specs."""

import unittest

from flax import nnx
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.weights import layout

EMB, VOCAB, EXPERTS, MOE_DIM, BLOCKS = 8, 16, 4, 6, 2

# Names every fixture below must flatten to, written out by hand.
MODEL_NAMES = [
    "decoder.decoder_norm.scale",
    "decoder.layers.local_layers.mlp.routed_experts.wi_0",
    "decoder.layers.local_layers.mlp.routed_experts.wo",
    "token_embedder.embedding",
]


def _model_tree(make):
  """A Qwen3.5-shaped tree with no two leaves of the same shape or dtype pair."""
  return {
      "token_embedder": {"embedding": make((VOCAB, EMB), jnp.float32)},
      "decoder": {
          "decoder_norm": {"scale": make((EMB,), jnp.float32)},
          "layers": {
              "local_layers": {
                  "mlp": {
                      "routed_experts": {
                          "wi_0": make((EXPERTS, BLOCKS, 3, EMB, MOE_DIM), jnp.bfloat16),
                          "wo": make((EXPERTS, BLOCKS, 3, MOE_DIM, EMB), jnp.bfloat16),
                      }
                  }
              }
          },
      },
  }


def _zeros(shape, dtype):
  return jnp.zeros(shape, dtype)


class _Norm(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.ones((EMB,), jnp.float32))


class _TinyModel(nnx.Module):

  def __init__(self):
    self.decoder_norm = _Norm()
    self.embedding = nnx.Param(jnp.ones((VOCAB, EMB), jnp.bfloat16))


class FlattenLeavesTest(unittest.TestCase):

  def test_trainer_and_rollout_wrappers_yield_the_same_names(self):
    trainer = {"base": _model_tree(_zeros)}
    rollout = {"model": {"model": _model_tree(_zeros)}}
    self.assertEqual(sorted(layout.flatten_leaves(trainer)), MODEL_NAMES)
    self.assertEqual(sorted(layout.flatten_leaves(rollout)), MODEL_NAMES)
    self.assertEqual(sorted(layout.flatten_leaves(_model_tree(_zeros))), MODEL_NAMES)

  def test_non_weight_state_is_dropped_before_wrappers_are_stripped(self):
    rollout = {"model": _model_tree(_zeros)}
    rollout["model"]["decoder"]["cache"] = {"key": _zeros((2, 3), jnp.float32)}
    rollout["model"]["decoder"]["_intermediates"] = {"x": _zeros((2,), jnp.float32)}
    rollout["rngs"] = {"default": {"count": _zeros((), jnp.uint32)}}
    self.assertEqual(sorted(layout.flatten_leaves(rollout)), MODEL_NAMES)

  def test_a_root_not_shared_by_every_weight_is_kept(self):
    tree = {"base": {"a": _zeros((1,), jnp.float32)}, "vision": {"b": _zeros((2,), jnp.float32)}}
    self.assertEqual(sorted(layout.flatten_leaves(tree)), ["base.a", "vision.b"])

  def test_two_different_wrapper_roots_side_by_side_are_kept(self):
    tree = {"base": {"a": _zeros((1,), jnp.float32)}, "model": {"b": _zeros((2,), jnp.float32)}}
    self.assertEqual(sorted(layout.flatten_leaves(tree)), ["base.a", "model.b"])

  def test_a_weight_named_like_a_wrapper_is_kept(self):
    self.assertEqual(list(layout.flatten_leaves({"model": _zeros((1,), jnp.float32)})), ["model"])

  def test_integer_keys_become_name_segments(self):
    tree = {"layers": {0: {"w": _zeros((1,), jnp.float32)}, 1: {"w": _zeros((2,), jnp.float32)}}}
    self.assertEqual(sorted(layout.flatten_leaves(tree)), ["layers.0.w", "layers.1.w"])

  def test_nnx_state_and_frozen_dict_are_unwrapped_to_arrays(self):
    state = nnx.state(_TinyModel(), nnx.Param)
    flat = layout.flatten_leaves(state)
    self.assertEqual(sorted(flat), ["decoder_norm.scale", "embedding"])
    self.assertIsInstance(flat["embedding"], jax.Array)
    frozen = FrozenDict({"base": {"decoder_norm": {"scale": np.ones((EMB,), np.float32)}}})
    self.assertEqual(list(layout.flatten_leaves(frozen)), ["decoder_norm.scale"])

  def test_param_leaves_in_a_plain_dict_are_unwrapped_to_arrays(self):
    flat = layout.flatten_leaves({"decoder": {"w": nnx.Param(jnp.ones((2, 3), jnp.bfloat16))}})
    self.assertEqual(list(flat), ["decoder.w"])
    self.assertIsInstance(flat["decoder.w"], jax.Array)


class LeafSpecsTest(unittest.TestCase):

  def test_specs_carry_exact_shape_and_dtype(self):
    specs = layout.leaf_specs({"base": _model_tree(_zeros)})
    self.assertEqual(len(specs), 4)
    self.assertEqual(
        {name: (spec.shape, spec.dtype) for name, spec in specs.items()},
        {
            "token_embedder.embedding": ((16, 8), np.dtype("float32")),
            "decoder.decoder_norm.scale": ((8,), np.dtype("float32")),
            "decoder.layers.local_layers.mlp.routed_experts.wi_0": ((4, 2, 3, 8, 6), np.dtype(jnp.bfloat16)),
            "decoder.layers.local_layers.mlp.routed_experts.wo": ((4, 2, 3, 6, 8), np.dtype(jnp.bfloat16)),
        },
    )

  def test_specs_need_no_tensor_values(self):
    abstract = {"model": _model_tree(jax.ShapeDtypeStruct)}
    self.assertEqual(layout.leaf_specs(abstract), layout.leaf_specs({"base": _model_tree(np.zeros)}))

  def test_sharding_is_recorded_for_device_arrays_and_none_for_host_arrays(self):
    device = layout.leaf_specs({"w": jnp.zeros((2,), jnp.float32)})["w"]
    host = layout.leaf_specs({"w": np.zeros((2,), np.float32)})["w"]
    self.assertEqual(device.sharding, jnp.zeros((2,), jnp.float32).sharding)
    self.assertIsNone(host.sharding)


if __name__ == "__main__":
  unittest.main()
