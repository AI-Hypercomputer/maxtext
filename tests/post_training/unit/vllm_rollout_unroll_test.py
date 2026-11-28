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

"""Unit tests for MaxText scanned-weight unrolling workarounds."""

import gc
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.integration.vllm.maxtext_vllm_rollout import (
    _create_model_converter,
    MaxTextVllmRollout,
    MaxTextVllmSampler,
    prepare_direct_sync_additional_config,
    requires_maxtext_scanned_weight_unroll,
    unroll_layers_n_blocks,
    unroll_nested_block_scan,
    unroll_scanned_block_weights,
    uses_maxtext_vllm_adapter,
    validate_direct_sync_layer_coverage,
)
from maxtext.integration.vllm.weight_converter import WeightConverter

from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout

pytestmark = pytest.mark.post_training


class MockWeights:
  """A mock weight container that implements to_pure_dict."""

  def __init__(self, pure_dict):
    self._pure_dict = pure_dict

  def to_pure_dict(self):
    return self._pure_dict


class GemmaScannedWeightsUnrollTest(unittest.TestCase):
  """Verify the correctness of the unroll_layers_n_blocks utility."""

  def test_bypasses_non_pytree_weights(self):
    """If the weights object doesn't have `to_pure_dict`, it should be returned unchanged."""
    raw_weights = {"dummy": np.ones(5)}
    result = unroll_layers_n_blocks(raw_weights)
    self.assertIs(result, raw_weights)

  def test_bypasses_non_scanned_checkpoints(self):
    """If the checkpoint is not scanned (no 'layers_0' inside 'decoder/layers/'), return unchanged."""
    pure_dict = {
        "decoder": {
            "layers": {
                "0": {"attn": {"wq": np.ones(10)}},
                "1": {"attn": {"wq": np.ones(10)}},
            }
        }
    }
    weights = MockWeights(pure_dict)
    result = unroll_layers_n_blocks(weights)
    self.assertIs(result, weights)

  def test_correctly_unrolls_gemma_scanned_weights(self):
    """Verify that scanned layers are properly interleaved and mapped, and remainder layers are appended."""
    # Pattern length = 2 (layers_0 and layers_1)
    # Scan length = 3. In MaxText, param_scan_axis=1, so shape is (feature_dim, scan_length, ...)

    # We want an array where axis 1 has length 3. Let's make it (2, 3, 1)
    # For layers_0, values should be 0, 2, 4
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    # Shared attention weight (broadcast across scan length)
    attn0 = np.zeros((2, 1))
    attn0[:, :] = 9

    # For layers_1, values should be 1, 3, 5
    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "layers": {
                "layers_0": {
                    "mlp": {"wi_0": arr0},
                    "attn": {"wq": attn0},
                    "dropout": {"rngs": {"params": {"count": np.ones((5,))}}},
                },
                "layers_1": {
                    "mlp": {"wi_0": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "mlp": {"wi_0": np.array([[6, 6]]).transpose()},  # shape (2, 1)
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_layers_n_blocks(weights)

    # Check unrolled structure
    decoder_dict = unrolled["decoder"]

    # Should contain keys layers_0 to layers_6
    self.assertIn("layers_0", decoder_dict)
    self.assertIn("layers_1", decoder_dict)
    self.assertIn("layers_2", decoder_dict)
    self.assertIn("layers_3", decoder_dict)
    self.assertIn("layers_4", decoder_dict)
    self.assertIn("layers_5", decoder_dict)
    self.assertIn("layers_6", decoder_dict)

    self.assertIsInstance(list(decoder_dict.keys())[0], str)

    # Verify dropout was dropped
    self.assertNotIn("dropout", decoder_dict["layers_0"])

    # Check that values are correctly sliced
    np.testing.assert_array_equal(decoder_dict["layers_0"]["mlp"]["wi_0"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers_1"]["mlp"]["wi_0"], np.array([[1], [1]]))
    np.testing.assert_array_equal(decoder_dict["layers_2"]["mlp"]["wi_0"], np.array([[2], [2]]))
    np.testing.assert_array_equal(decoder_dict["layers_3"]["mlp"]["wi_0"], np.array([[3], [3]]))
    np.testing.assert_array_equal(decoder_dict["layers_4"]["mlp"]["wi_0"], np.array([[4], [4]]))
    np.testing.assert_array_equal(decoder_dict["layers_5"]["mlp"]["wi_0"], np.array([[5], [5]]))
    np.testing.assert_array_equal(decoder_dict["layers_6"]["mlp"]["wi_0"], np.array([[6], [6]]))

    # Check that shared attention weight was broadcasted
    np.testing.assert_array_equal(decoder_dict["layers_0"]["attn"]["wq"], np.array([[9], [9]]))
    np.testing.assert_array_equal(decoder_dict["layers_2"]["attn"]["wq"], np.array([[9], [9]]))

  def test_correctly_unrolls_gemma3_gemma4_scanned_blocks(self):
    """Verify that scanned layers under scanned_blocks are properly interleaved and mapped."""
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "scanned_blocks": {
                "layers_0": {
                    "mlp": {"wi_0": arr0},
                },
                "layers_1": {
                    "mlp": {"wi_0": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "mlp": {"wi_0": np.array([[6, 6]]).transpose()},
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_layers_n_blocks(weights)

    decoder_dict = unrolled["decoder"]
    self.assertIn("layers_0", decoder_dict)
    self.assertIn("layers_6", decoder_dict)
    self.assertIsInstance(list(decoder_dict.keys())[0], str)
    np.testing.assert_array_equal(decoder_dict["layers_0"]["mlp"]["wi_0"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers_6"]["mlp"]["wi_0"], np.array([[6], [6]]))


class QwenScannedWeightsUnrollTest(unittest.TestCase):
  """Verify heterogeneous Qwen blocks map to unscanned decoder attributes."""

  @pytest.mark.cpu_only
  def test_interleaves_slots_and_repetitions(self):
    slot_0 = np.zeros((2, 2, 1), dtype=np.float32)
    slot_0[:, 0, :] = 0
    slot_0[:, 1, :] = 2
    slot_1 = np.zeros((2, 2, 1), dtype=np.float32)
    slot_1[:, 0, :] = 1
    slot_1[:, 1, :] = 3
    weights = MockWeights(
        {
            "base": {
                "decoder": {
                    "layers": {
                        "layer_0": {"probe": slot_0},
                        "layer_1": {"probe": slot_1, "rngs": {"key": np.ones(2, dtype=np.uint32)}},
                    },
                    "decoder_norm": {"scale": np.ones(2)},
                }
            }
        }
    )

    unrolled = unroll_nested_block_scan(weights)

    decoder = unrolled["base"]["decoder"]
    for layer_idx in range(4):
      self.assertIn(f"layers_{layer_idx}", decoder)
      np.testing.assert_array_equal(
          decoder[f"layers_{layer_idx}"]["probe"],
          np.full((2, 1), layer_idx, dtype=np.float32),
      )
    np.testing.assert_array_equal(decoder["decoder_norm"]["scale"], np.ones(2))
    self.assertNotIn("probe", decoder["layers"]["layer_1"])
    np.testing.assert_array_equal(decoder["layers"]["layer_1"]["rngs"]["key"], np.ones(2, dtype=np.uint32))
    target = {
        "model": {
            "decoder": {f"layers_{layer_idx}": {"probe": np.zeros((2, 1), dtype=np.float32)} for layer_idx in range(4)}
        }
    }
    self.assertEqual(validate_direct_sync_layer_coverage(unrolled, target), 4)

  @pytest.mark.cpu_only
  def test_unrolls_the_nested_local_and_global_block_scan(self):
    """Nested `local_layers` plus `global_layer` unroll to a dense `layers_{global_idx}` run."""
    num_blocks, num_local = 2, 3
    cycle = num_local + 1
    local_probe = np.zeros((2, num_blocks, num_local, 1), dtype=np.float32)
    global_probe = np.zeros((2, num_blocks, 1), dtype=np.float32)
    for block in range(num_blocks):
      for slot in range(num_local):
        local_probe[:, block, slot, :] = block * cycle + slot
      global_probe[:, block, :] = block * cycle + num_local
    weights = MockWeights(
        {
            "base": {
                "decoder": {
                    "layers": {
                        "local_layers": {"probe": local_probe},
                        "global_layer": {"probe": global_probe},
                    },
                    "decoder_norm": {"scale": np.ones(2)},
                }
            }
        }
    )

    unrolled = unroll_nested_block_scan(weights)

    decoder = unrolled["base"]["decoder"]
    for layer_idx in range(num_blocks * cycle):
      self.assertIn(f"layers_{layer_idx}", decoder)
      np.testing.assert_array_equal(
          decoder[f"layers_{layer_idx}"]["probe"],
          np.full((2, 1), layer_idx, dtype=np.float32),
      )
    np.testing.assert_array_equal(decoder["decoder_norm"]["scale"], np.ones(2))
    target = {
        "model": {
            "decoder": {
                f"layers_{layer_idx}": {"probe": np.zeros((2, 1), dtype=np.float32)}
                for layer_idx in range(num_blocks * cycle)
            }
        }
    }
    self.assertEqual(validate_direct_sync_layer_coverage(unrolled, target), num_blocks * cycle)

  @pytest.mark.cpu_only
  def test_nested_block_scan_honors_an_explicit_pattern_length(self):
    """The global layer is the last slot of `pattern_length`, which `local_layers` must fill up to."""
    num_blocks, num_local = 2, 3
    weights = MockWeights(
        {
            "decoder": {
                "layers": {
                    "local_layers": {"probe": np.zeros((2, num_blocks, num_local, 1), dtype=np.float32)},
                    "global_layer": {"probe": np.ones((2, num_blocks, 1), dtype=np.float32)},
                }
            }
        }
    )

    unrolled = unroll_nested_block_scan(weights, pattern_length=num_local + 1)
    for block in range(num_blocks):
      np.testing.assert_array_equal(unrolled["decoder"][f"layers_{block * 4 + 3}"]["probe"], np.ones((2, 1)))

    with self.assertRaisesRegex(ValueError, "hold 3 cycle slots, but pattern_length 5"):
      unroll_nested_block_scan(weights, pattern_length=num_local + 2)

  @pytest.mark.cpu_only
  def test_rejects_the_nested_block_scan_remainder(self):
    """Layers past the last whole cycle live in `layers_remainder`, which must not be dropped silently."""
    weights = MockWeights(
        {
            "decoder": {
                "layers": {
                    "local_layers": {"probe": np.zeros((2, 2, 3, 1), dtype=np.float32)},
                    "global_layer": {"probe": np.zeros((2, 2, 1), dtype=np.float32)},
                },
                "layers_remainder": {"local_layers": {"probe": np.zeros((2, 2, 1), dtype=np.float32)}},
            }
        }
    )

    with self.assertRaisesRegex(ValueError, "layers_remainder"):
      unroll_nested_block_scan(weights)

  @pytest.mark.cpu_only
  def test_rejects_nested_local_layers_without_a_cycle_slot_axis(self):
    """A `local_layers` tensor missing its slot axis must not be read as flat."""
    weights = MockWeights({"decoder": {"layers": {"local_layers": {"probe": np.ones((2, 2))}}}})

    with self.assertRaisesRegex(ValueError, "has no scan axis"):
      unroll_nested_block_scan(weights)

  @pytest.mark.cpu_only
  def test_correctly_unrolls_homogeneous_scanned_weights(self):
    """Verify homogeneous scanned layers (LLaMA, Qwen Base) unroll to layers_{global_idx}."""
    # 4 layers stacked on scan_axis=1 (feature_dim=2, num_layers=4, hidden=1)
    probe_arr = np.zeros((2, 4, 1), dtype=np.float32)
    for i in range(4):
      probe_arr[:, i, :] = i
    weights = MockWeights(
        {
            "base": {
                "decoder": {
                    "layers": {
                        "self_attention": {"probe": probe_arr},
                        "dropout": {"rngs": {"key": np.ones(2, dtype=np.uint32)}},
                    },
                    "decoder_norm": {"scale": np.ones(2)},
                }
            }
        }
    )

    unrolled = unroll_nested_block_scan(weights)

    decoder = unrolled["base"]["decoder"]
    for layer_idx in range(4):
      self.assertIn(f"layers_{layer_idx}", decoder)
      np.testing.assert_array_equal(
          decoder[f"layers_{layer_idx}"]["self_attention"]["probe"],
          np.full((2, 1), layer_idx, dtype=np.float32),
      )
    np.testing.assert_array_equal(decoder["decoder_norm"]["scale"], np.ones(2))
    target = {
        "model": {
            "decoder": {
                f"layers_{layer_idx}": {"self_attention": {"probe": np.zeros((2, 1), dtype=np.float32)}}
                for layer_idx in range(4)
            }
        }
    }
    self.assertEqual(validate_direct_sync_layer_coverage(unrolled, target), 4)

  @pytest.mark.cpu_only
  def test_rejects_inconsistent_scan_lengths(self):
    weights = MockWeights(
        {
            "decoder": {
                "layers": {
                    "layer_0": {"probe": np.ones((2, 2, 1))},
                    "layer_1": {"probe": np.ones((2, 3, 1))},
                }
            }
        }
    )

    with self.assertRaisesRegex(ValueError, "disagree on scan length"):
      unroll_nested_block_scan(weights)

  @pytest.mark.cpu_only
  def test_invalid_scan_axis_formats_non_string_parameter_path(self):
    weights = MockWeights({"decoder": {"layers": {"layer_0": {0: np.ones((2, 1))}}}})

    with self.assertRaisesRegex(ValueError, r"decoder\.layers\.layer_0\.0"):
      unroll_nested_block_scan(weights, scan_axis=2, pattern_length=1)

  @pytest.mark.cpu_only
  def test_supports_nondefault_axis_and_sparse_slots(self):
    slot_1 = np.stack(
        [np.full((2, 1), 1, dtype=np.float32), np.full((2, 1), 3, dtype=np.float32)],
        axis=0,
    )
    weights = MockWeights({"decoder": {"layers": {"layer_1": {"probe": slot_1}}}})

    unrolled = unroll_nested_block_scan(weights, scan_axis=0, pattern_length=2)

    self.assertNotIn("layers_0", unrolled["decoder"])
    np.testing.assert_array_equal(unrolled["decoder"]["layers_1"]["probe"], np.full((2, 1), 1))
    np.testing.assert_array_equal(unrolled["decoder"]["layers_3"]["probe"], np.full((2, 1), 3))

  @pytest.mark.cpu_only
  def test_rejects_missing_target_layer_parameters(self):
    source = {"base": {"decoder": {"layers_0": {"probe": np.ones((2, 1))}}}}
    target = {
        "model": {
            "decoder": {
                "layers_0": {"probe": np.zeros((2, 1))},
                "layers_1": {"probe": np.zeros((2, 1))},
            }
        }
    }

    with self.assertRaisesRegex(ValueError, "leave rollout transformer parameters at random initialization"):
      validate_direct_sync_layer_coverage(source, target)

  @pytest.mark.cpu_only
  def test_rejects_source_without_unrolled_layers(self):
    source = {"base": {"decoder": {"layers": {"layer_0": {"probe": np.ones((2, 1))}}}}}
    target = {"model": {"decoder": {"layers_0": {"probe": np.zeros((2, 1))}}}}

    with self.assertRaisesRegex(ValueError, "matched 0/1 target layer parameters"):
      validate_direct_sync_layer_coverage(source, target)

  @pytest.mark.cpu_only
  def test_accepts_split_moe_weights_for_prefused_target(self):
    source = {
        "base": {
            "decoder": {
                "layers_0": {
                    "mlp": {
                        "routed_experts": {
                            "wi_0": np.ones((2, 3, 4)),
                            "wi_1": np.ones((2, 3, 4)),
                        }
                    }
                }
            }
        }
    }
    target = {"model": {"decoder": {"layers_0": {"mlp": {"routed_experts": {"wi": np.zeros((2, 3, 8))}}}}}}

    self.assertEqual(validate_direct_sync_layer_coverage(source, target), 1)


class ScannedBlockUnrollDispatchTest(unittest.TestCase):
  """Verify `unroll_scanned_block_weights` picks the unroll from the tree structure, not the model name."""

  @pytest.mark.cpu_only
  def test_unrolls_moe_blocks_without_an_mlp_wi_0(self):
    """b/584148200: GPT-OSS and Llama 4 blocks have no `mlp.wi_0`, which the scan length was read from."""
    num_blocks, block_length = 3, 2
    # `wo[j][:, i, :]` is layer `j` of block `i`, i.e. physical layer `i * block_length + j`.
    wo = [
        np.stack([np.full((4, 2), i * block_length + j) for i in range(num_blocks)], axis=1) for j in range(block_length)
    ]
    weights = {"decoder": {"layers": {f"layers_{j}": {"GptOssMlp": {"wo": wo[j]}} for j in range(block_length)}}}

    decoder = unroll_scanned_block_weights(weights)["decoder"]

    self.assertCountEqual(decoder, [f"layers_{n}" for n in range(num_blocks * block_length)])
    for n in range(num_blocks * block_length):
      np.testing.assert_array_equal(decoder[f"layers_{n}"]["GptOssMlp"]["wo"], np.full((4, 2), n))

  @pytest.mark.cpu_only
  def test_unrolls_the_nested_block_scan_under_scanned_blocks(self):
    """b/584148367: Gemma 4 nests Qwen 3.5's `local_layers`/`global_layer` scan under `scanned_blocks`."""
    num_blocks, num_local = 2, 2
    cycle = num_local + 1
    local_probe = np.zeros((2, num_blocks, num_local, 1), dtype=np.float32)
    global_probe = np.zeros((2, num_blocks, 1), dtype=np.float32)
    for block in range(num_blocks):
      for slot in range(num_local):
        local_probe[:, block, slot, :] = block * cycle + slot
      global_probe[:, block, :] = block * cycle + num_local
    weights = {
        "decoder": {
            "scanned_blocks": {
                "local_layers": {"probe": local_probe},
                "global_layer": {"probe": global_probe},
            }
        }
    }

    decoder = unroll_scanned_block_weights(weights)["decoder"]

    self.assertCountEqual(decoder, [f"layers_{n}" for n in range(num_blocks * cycle)])
    for n in range(num_blocks * cycle):
      np.testing.assert_array_equal(decoder[f"layers_{n}"]["probe"], np.full((2, 1), n, dtype=np.float32))

  @pytest.mark.cpu_only
  def test_leaves_a_homogeneous_stack_for_tunix(self):
    weights = {"decoder": {"layers": {"mlp": {"wo": np.zeros((4, 3, 2))}}}}
    self.assertIs(unroll_scanned_block_weights(weights), weights)

  @pytest.mark.cpu_only
  def test_rejects_scanned_blocks_after_unscanned_layers(self):
    """Ming and HY4 keep unscanned layers before their scanned blocks, which must not overwrite them."""
    weights = {
        "decoder": {
            "layers_0": {"mlp": {"wo": np.zeros((4, 2))}},
            "scanned_blocks": {"layers_0": {"mlp": {"wo": np.ones((4, 3, 2))}}},
        }
    }
    with self.assertRaisesRegex(ValueError, "would overwrite unscanned layers"):
      unroll_scanned_block_weights(weights)


class ScannedSourceLayerCoverageTest(unittest.TestCase):
  """Verify the coverage check accepts scanned sources that Tunix unrolls itself.

  MaxText unrolls scanned blocks before the direct sync; a homogeneous scan
  (e.g. Llama) keeps `decoder.layers.*` stacked on the scan axis and Tunix's
  `transfer_state_directly` slices layer N out for `layers_N`.
  """

  _PER_LAYER_PARAMS = (
      ("mlp", "wi_0", "kernel"),
      ("mlp", "wo", "kernel"),
      ("pre_self_attention_layer_norm", "scale"),
      ("self_attention", "query", "kernel"),
  )

  def _scanned_llama_trees(self, num_layers, drop_from_source=()):
    """Builds a Llama-style scanned source and the matching unscanned rollout target."""
    source_flat = {("base", "decoder", "decoder_norm", "scale"): np.ones((2,))}
    target_flat = {("model", "decoder", "decoder_norm", "scale"): np.zeros((2,))}
    for param in self._PER_LAYER_PARAMS:
      if param not in drop_from_source:
        # param_scan_axis=1: the layer axis sits after the leading feature dim.
        source_flat[("base", "decoder", "layers") + param] = np.ones((2, num_layers, 3))
      for i in range(num_layers):
        target_flat[("model", "decoder", f"layers_{i}") + param] = np.zeros((2, 3))
    return unflatten_dict(source_flat), unflatten_dict(target_flat)

  @pytest.mark.cpu_only
  def test_accepts_homogeneous_scanned_source(self):
    source, target = self._scanned_llama_trees(num_layers=3)

    self.assertEqual(validate_direct_sync_layer_coverage(source, target), 3 * len(self._PER_LAYER_PARAMS))

  @pytest.mark.cpu_only
  def test_rejects_scanned_source_missing_a_target_parameter(self):
    source, target = self._scanned_llama_trees(num_layers=2, drop_from_source=(("mlp", "wo", "kernel"),))

    with self.assertRaisesRegex(ValueError, r"matched 6/8 target layer parameters.*decoder\.layers_0\.mlp\.wo\.kernel"):
      validate_direct_sync_layer_coverage(source, target)

  @pytest.mark.cpu_only
  def test_accepts_implicitly_stacked_source(self):
    source = {"base": {"decoder": {"probe": np.ones((2, 2, 3))}}}
    target = {"model": {"decoder": {"layers_0": {"probe": np.zeros((2, 3))}, "layers_1": {"probe": np.zeros((2, 3))}}}}

    self.assertEqual(validate_direct_sync_layer_coverage(source, target), 2)

  @pytest.mark.cpu_only
  def test_accepts_scanned_split_moe_weights_for_prefused_target(self):
    experts = {"wi_0": np.ones((4, 2, 3, 5)), "wi_1": np.ones((4, 2, 3, 5))}
    source = {"base": {"decoder": {"layers": {"mlp": {"routed_experts": experts}}}}}
    target = {
        "model": {"decoder": {f"layers_{i}": {"mlp": {"routed_experts": {"wi": np.zeros((4, 3, 10))}}} for i in range(2)}}
    }

    self.assertEqual(validate_direct_sync_layer_coverage(source, target), 2)

  @pytest.mark.cpu_only
  def test_sampler_leaves_scanned_llama_weights_for_tunix(self):
    """Regression: llama3.1-70b direct sync failed with `matched 0/720 target layer parameters`."""
    sampler = object.__new__(MaxTextVllmSampler)
    sampler._direct_maxtext_sync = True  # pylint: disable=protected-access
    sampler._is_qwen = False  # pylint: disable=protected-access
    source, target = self._scanned_llama_trees(num_layers=2)

    with (
        mock.patch.object(MaxTextVllmSampler, "transformer_state", new_callable=mock.PropertyMock, return_value=target),
        mock.patch.object(VllmSampler, "update_params") as tunix_update_params,
    ):
      sampler.update_params(source, filter_types=None)

    tunix_update_params.assert_called_once_with(source, None)

  @pytest.mark.cpu_only
  def test_sampler_update_params_skips_redundant_sync_safely(self):
    """Identical live jax.Arrays skip reshard; GC/mutable/KV-deleted do not."""
    sampler = object.__new__(MaxTextVllmSampler)
    sampler._direct_maxtext_sync = True  # pylint: disable=protected-access
    sampler._is_gemma = False  # pylint: disable=protected-access
    sampler._is_qwen = False  # pylint: disable=protected-access

    w0 = jnp.ones((2, 2), dtype=jnp.float32)
    w1 = jnp.ones((2, 2), dtype=jnp.float32) * 2.0
    state_holder = {"model": {"layers_0": {"w": jnp.zeros((2, 2), dtype=jnp.float32)}}}
    calls = []

    def _fake_super_update(self_arg, updated_weights, filter_types=None):
      del self_arg, filter_types
      calls.append(updated_weights)
      src_w = updated_weights["base"]["layers_0"]["w"]
      state_holder["model"]["layers_0"]["w"] = jnp.array(src_w)

    with (
        mock.patch.object(
            VllmSampler,
            "transformer_state",
            new_callable=mock.PropertyMock,
            return_value=state_holder,
        ),
        mock.patch.object(
            VllmSampler,
            "update_params",
            _fake_super_update,
        ),
    ):
      sampler.update_params({"base": {"layers_0": {"w": w0}}})
      self.assertEqual(len(calls), 1)
      self.assertFalse(sampler.last_update_skipped)

      # Identical live source/target buffers skip redundant reshard even when
      # filter_types differs (e.g. load_checkpoint None -> RLCluster nnx.Param).
      sampler.update_params({"base": {"layers_0": {"w": w0}}}, filter_types=(object,))
      self.assertEqual(len(calls), 1)
      self.assertTrue(sampler.last_update_skipped)

      # Never skip when KV cache was deleted and needs reinitialize_cache().
      sampler._kv_cache_deleted = True  # pylint: disable=protected-access
      sampler.update_params({"base": {"layers_0": {"w": w0}}})
      self.assertEqual(len(calls), 2)
      self.assertFalse(sampler.last_update_skipped)
      sampler._kv_cache_deleted = False  # pylint: disable=protected-access

      # Target buffers freed (e.g. release_for_training w/ delete_dst_buffers) -> must re-sync.
      state_holder["model"]["layers_0"]["w"].delete()
      sampler.update_params({"base": {"layers_0": {"w": w0}}})
      self.assertEqual(len(calls), 3)
      self.assertFalse(sampler.last_update_skipped)

      # Target leaf replaced out-of-band while old target array stays live -> must re-sync.
      prev_tgt = state_holder["model"]["layers_0"]["w"]
      state_holder["model"]["layers_0"]["w"] = jnp.zeros((2, 2), dtype=jnp.float32)
      sampler.update_params({"base": {"layers_0": {"w": w0}}})
      self.assertEqual(len(calls), 4)
      self.assertFalse(sampler.last_update_skipped)
      del prev_tgt

      # Re-syncing after freeing/GC'ing the previous source array must not skip.
      del w0
      calls.clear()
      gc.collect()
      w0_replacement = jnp.ones((2, 2), dtype=jnp.float32) * 3.0
      sampler.update_params({"base": {"layers_0": {"w": w0_replacement}}})
      self.assertEqual(len(calls), 1)

      # Mutable np.ndarray leaves must never be skipped.
      np_w = np.ones((2, 2), dtype=np.float32)
      sampler.update_params({"base": {"layers_0": {"w": np_w}}})
      self.assertEqual(len(calls), 2)
      sampler.update_params({"base": {"layers_0": {"w": np_w}}})
      self.assertEqual(len(calls), 3)

      # New jax.Array leaf triggers a sync.
      sampler.update_params({"base": {"layers_0": {"w": w1}}})
      self.assertEqual(len(calls), 4)

      # Traced inputs (e.g. under jax.eval_shape / jax.jit) must not call
      # is_deleted() on abstract tracers and must never skip the sync.
      def _traced_update(weight):
        sampler.update_params({"base": {"layers_0": {"w": weight}}})
        return weight

      jax.eval_shape(_traced_update, jax.ShapeDtypeStruct((2, 2), jnp.float32))
      self.assertEqual(len(calls), 5)


class DirectSyncRolloutConfigTest(unittest.TestCase):
  """Verify MoE rollout targets request the safe/fast fused layout."""

  @pytest.mark.cpu_only
  def test_enables_prefusion_for_direct_moe_tp(self):
    original = {"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}}

    prepared = prepare_direct_sync_additional_config(
        original,
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=4,
    )

    self.assertTrue(prepared["maxtext_config"]["prefuse_moe_weights"])
    self.assertNotIn("prefuse_moe_weights", original["maxtext_config"])

    forced = prepare_direct_sync_additional_config(
        {"maxtext_config": {"prefuse_moe_weights": False}},
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=4,
    )
    self.assertTrue(forced["maxtext_config"]["prefuse_moe_weights"])

  @pytest.mark.cpu_only
  def test_defaults_prefusion_for_tp1_moe_and_honors_explicit_false(self):
    original = {"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}}
    prepared = prepare_direct_sync_additional_config(
        original,
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=1,
    )
    self.assertTrue(prepared["maxtext_config"]["prefuse_moe_weights"])
    self.assertNotIn("prefuse_moe_weights", original["maxtext_config"])

    opt_out = prepare_direct_sync_additional_config(
        {"maxtext_config": {"prefuse_moe_weights": False}},
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=1,
    )
    self.assertFalse(opt_out["maxtext_config"]["prefuse_moe_weights"])

  @pytest.mark.cpu_only
  def test_leaves_dense_or_non_direct_config_unchanged(self):
    original = {"maxtext_config": {"model_name": "qwen3-0.6b"}}

    self.assertIs(
        prepare_direct_sync_additional_config(
            original,
            direct_maxtext_sync=True,
            num_experts=1,
            tensor_parallel_size=4,
        ),
        original,
    )
    self.assertIs(
        prepare_direct_sync_additional_config(
            original,
            direct_maxtext_sync=False,
            num_experts=256,
            tensor_parallel_size=1,
        ),
        original,
    )

  @pytest.mark.cpu_only
  def test_leaves_tp1_prefusion_unset_with_custom_mappings_or_restore_path(self):
    with_restore = {"maxtext_config": {"model_name": "qwen3.5-35b-a3b", "load_parameters_path": "/ckpt/dir"}}
    prepared_restore = prepare_direct_sync_additional_config(
        with_restore,
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=1,
    )
    self.assertNotIn("prefuse_moe_weights", prepared_restore["maxtext_config"])

    custom_mapped = prepare_direct_sync_additional_config(
        {"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}},
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=1,
        has_custom_mappings=True,
    )
    self.assertNotIn("prefuse_moe_weights", custom_mapped["maxtext_config"])

    forced_tp = prepare_direct_sync_additional_config(
        with_restore,
        direct_maxtext_sync=True,
        num_experts=256,
        tensor_parallel_size=4,
        has_custom_mappings=True,
    )
    self.assertTrue(forced_tp["maxtext_config"]["prefuse_moe_weights"])


class MaxTextVllmRolloutConfigForwardingTest(unittest.TestCase):
  """Verify the custom rollout preserves Tunix rollout options."""

  _SAMPLING_KWARGS = {
      "stop": ["</answer>"],
      "detokenize": True,
      "include_stop_str_in_output": True,
  }

  def _build_vllm_config(self, vllm_config_cls=None, **rollout_overrides):
    """Constructs the rollout with fakes and returns (VllmConfig it built, fake sampler)."""
    sampling_kwargs = self._SAMPLING_KWARGS
    rollout_config = SimpleNamespace(
        kv_cache_size=1280,
        rollout_mapping_config=None,
        rollout_vllm_model_version="Qwen/Qwen3.5-35B-A3B",
        rollout_vllm_swap_space_size_gb=2,
        rollout_vllm_async_scheduling=False,
        rollout_vllm_max_num_batched_tokens=16384,
        rollout_vllm_max_num_seqs=32,
        rollout_vllm_hf_config_path=None,
        rollout_vllm_logprobs_mode="raw_logprobs",
        rollout_vllm_kwargs={"dtype": "bfloat16"},
        rollout_vllm_additional_config={"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}},
        rollout_vllm_hbm_utilization=0.6,
        rollout_vllm_init_with_random_weights=True,
        rollout_vllm_tpu_backend_type="jax",
        rollout_vllm_lora_config=None,
        rollout_vllm_server_mode=False,
        rollout_vllm_server_mode_submission_threshold=7,
        rollout_vllm_server_mode_submission_timeout_s=3.0,
        return_logprobs=True,
        tensor_parallel_size=4,
        data_parallel_size=2,
        expert_parallel_size=1,
        rollout_vllm_enable_dp_attention=False,
        rollout_vllm_delete_dst_buffers=True,
        rollout_vllm_reshard_chunk_size=8,
        rollout_vllm_sampling_kwargs=sampling_kwargs,
        **rollout_overrides,
    )
    maxtext_config = SimpleNamespace(
        model_name="qwen3.5-35b-a3b",
        num_experts=256,
        param_scan_axis=1,
        inhomogeneous_layer_cycle_interval=4,
        swap_space_vllm_gb=2,
        vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
        rollout_tensor_parallelism=4,
    )
    fake_sampler = mock.MagicMock()

    with (
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.mappings.MappingConfig.build",
            return_value=object(),
        ),
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.VllmConfig",
            **({"new": vllm_config_cls} if vllm_config_cls else {"side_effect": SimpleNamespace}),
        ),
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.MaxTextVllmSampler",
            return_value=fake_sampler,
        ) as sampler_cls,
        mock.patch("maxtext.integration.vllm.maxtext_vllm_rollout.nnx.state", return_value={"base": {}}),
    ):
      MaxTextVllmRollout(
          rollout_actor=object(),
          tokenizer=object(),
          mesh=object(),
          rollout_config=rollout_config,
          maxtext_config=maxtext_config,
      )

    return sampler_cls.call_args.kwargs["config"], fake_sampler

  @pytest.mark.cpu_only
  def test_forwards_sampling_parallelism_and_capacity_options(self):
    config, fake_sampler = self._build_vllm_config()

    self.assertEqual(config.sampling_kwargs, self._SAMPLING_KWARGS)
    self.assertEqual(config.expert_parallel_size, 1)
    self.assertEqual(config.return_logprobs, True)
    self.assertEqual(config.reshard_chunk_size, 8)
    self.assertEqual(config.server_mode_submission_threshold, 7)
    self.assertEqual(config.server_mode_submission_timeout_s, 3.0)
    self.assertEqual(config.engine_kwargs["max_num_batched_tokens"], 16384)
    self.assertEqual(config.engine_kwargs["max_num_seqs"], 32)
    self.assertEqual(config.engine_kwargs["logprobs_mode"], "raw_logprobs")
    self.assertNotIn("swap_space", config.engine_kwargs)
    fake_sampler.load_checkpoint.assert_called_once_with({"base": {}})

  @pytest.mark.cpu_only
  def test_forwards_free_kv_cache_during_weight_sync(self):
    for value in (True, False):
      config, _ = self._build_vllm_config(rollout_vllm_free_kv_cache_during_weight_sync=value)
      self.assertIs(config.free_kv_cache_during_weight_sync, value)

  @pytest.mark.cpu_only
  def test_omits_free_kv_cache_option_when_vllm_config_does_not_accept_it(self):
    """A rollout config carrying the field must not break a VllmConfig that predates it."""

    class StrictVllmConfig:
      """Rejects the option, like the VllmConfig of a tunix that predates it."""

      def __init__(self, *, mesh=None, **fields):
        if "free_kv_cache_during_weight_sync" in fields:
          raise TypeError("unexpected keyword argument 'free_kv_cache_during_weight_sync'")
        vars(self).update(fields, mesh=mesh)

    StrictVllmConfig.__signature__ = inspect.Signature(
        [inspect.Parameter("mesh", inspect.Parameter.KEYWORD_ONLY, default=None)]
    )

    with self.assertLogs(level="WARNING") as logs:
      config, _ = self._build_vllm_config(
          vllm_config_cls=StrictVllmConfig, rollout_vllm_free_kv_cache_during_weight_sync=False
      )

    self.assertFalse(hasattr(config, "free_kv_cache_during_weight_sync"))
    self.assertTrue(any("free_kv_cache_during_weight_sync" in line for line in logs.output), logs.output)
    self.assertEqual(config.reshard_chunk_size, 8)

  @pytest.mark.cpu_only
  def test_forwards_free_kv_cache_option_when_vllm_config_signature_cannot_be_inspected(self):
    """An uninspectable VllmConfig is treated as accepting the option, like `_kwargs_supported_by` does."""
    with mock.patch.object(inspect, "signature", side_effect=ValueError("no signature found")):
      config, _ = self._build_vllm_config(rollout_vllm_free_kv_cache_during_weight_sync=False)

    self.assertFalse(config.free_kv_cache_during_weight_sync)

  @pytest.mark.cpu_only
  def test_free_kv_cache_option_names_match_the_installed_tunix(self):
    """A misspelt or renamed tunix field would be dropped without an error, so pin both names."""

    def fields_mentioning_free_kv_cache(cls):
      return [name for name in inspect.signature(cls).parameters if "free_kv_cache" in name]

    rollout_fields = fields_mentioning_free_kv_cache(base_rollout.RolloutConfig)
    if not rollout_fields:
      self.skipTest("installed tunix predates free_kv_cache_during_weight_sync")

    self.assertEqual(rollout_fields, ["rollout_vllm_free_kv_cache_during_weight_sync"])
    self.assertEqual(fields_mentioning_free_kv_cache(VllmConfig), ["free_kv_cache_during_weight_sync"])

  @pytest.mark.cpu_only
  def test_omits_free_kv_cache_option_for_a_tunix_that_predates_it(self):
    config, _ = self._build_vllm_config()

    self.assertFalse(hasattr(config, "free_kv_cache_during_weight_sync"))

  @pytest.mark.cpu_only
  def test_leaves_tp1_prefusion_unset_when_mapping_config_has_custom_mappings(self):
    mapping_with_rules = SimpleNamespace(to_hf_mappings={"a": ("b", None)})
    rollout_config = SimpleNamespace(
        kv_cache_size=1280,
        rollout_mapping_config=None,
        rollout_vllm_model_version="Qwen/Qwen3.5-35B-A3B",
        rollout_vllm_swap_space_size_gb=2,
        rollout_vllm_async_scheduling=False,
        rollout_vllm_max_num_batched_tokens=16384,
        rollout_vllm_max_num_seqs=32,
        rollout_vllm_hf_config_path=None,
        rollout_vllm_logprobs_mode="raw_logprobs",
        rollout_vllm_kwargs={"dtype": "bfloat16"},
        rollout_vllm_additional_config={"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}},
        rollout_vllm_hbm_utilization=0.6,
        rollout_vllm_init_with_random_weights=True,
        rollout_vllm_tpu_backend_type="jax",
        rollout_vllm_lora_config=None,
        rollout_vllm_server_mode=False,
        rollout_vllm_server_mode_submission_threshold=7,
        rollout_vllm_server_mode_submission_timeout_s=3.0,
        return_logprobs=True,
        tensor_parallel_size=1,
        data_parallel_size=2,
        expert_parallel_size=1,
        rollout_vllm_enable_dp_attention=False,
        rollout_vllm_delete_dst_buffers=True,
        rollout_vllm_reshard_chunk_size=8,
        rollout_vllm_sampling_kwargs=self._SAMPLING_KWARGS,
    )
    maxtext_config = SimpleNamespace(
        model_name="qwen3.5-35b-a3b",
        num_experts=256,
        param_scan_axis=1,
        inhomogeneous_layer_cycle_interval=4,
        swap_space_vllm_gb=2,
        vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
        rollout_tensor_parallelism=1,
    )
    with (
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.mappings.MappingConfig.build",
            return_value=mapping_with_rules,
        ),
        mock.patch("maxtext.integration.vllm.maxtext_vllm_rollout.VllmConfig", side_effect=SimpleNamespace),
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.MaxTextVllmSampler",
            return_value=mock.MagicMock(),
        ) as sampler_cls,
        mock.patch("maxtext.integration.vllm.maxtext_vllm_rollout.nnx.state", return_value={"base": {}}),
    ):
      MaxTextVllmRollout(
          rollout_actor=object(),
          tokenizer=object(),
          mesh=object(),
          rollout_config=rollout_config,
          maxtext_config=maxtext_config,
      )

    cfg = sampler_cls.call_args.kwargs["config"]
    self.assertNotIn("prefuse_moe_weights", cfg.additional_config["maxtext_config"])

  @pytest.mark.cpu_only
  def test_timed_weight_sync_skips_count_barrier_and_timing_on_noop_skip(self):
    """Skipped no-op syncs must not increment count, block, or log false 0.000s timings."""
    rollout = object.__new__(MaxTextVllmRollout)
    rollout._weight_sync_count = 1  # pylint: disable=protected-access
    rollout._weight_sync_debug = False  # pylint: disable=protected-access
    rollout._maxtext_config = SimpleNamespace(  # pylint: disable=protected-access
        log_weight_sync_time=True, log_weight_sync_norm=False
    )
    rollout._sampler = SimpleNamespace(last_update_skipped=False, transformer_state={})  # pylint: disable=protected-access

    def _fake_rollout_update(self_arg, params, filter_types=None):
      del params, filter_types
      self_arg._sampler.last_update_skipped = True  # pylint: disable=protected-access

    with (
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.vllm_rollout.VllmRollout.update_params",
            _fake_rollout_update,
        ),
        mock.patch.object(rollout, "_block_on_rollout_state") as block_mock,
    ):
      with self.assertLogs(level="INFO") as logs:
        rollout.update_params({})

      block_mock.assert_not_called()
      self.assertEqual(rollout._weight_sync_count, 1)  # pylint: disable=protected-access
      self.assertTrue(any("sync 1 (sync_weights) SKIPPED (buffers unchanged)" in line for line in logs.output))

    def _fake_real_rollout_update(self_arg, params, filter_types=None):
      del self_arg, params, filter_types

    with (
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.vllm_rollout.VllmRollout.update_params",
            _fake_real_rollout_update,
        ),
        mock.patch.object(rollout, "_block_on_rollout_state") as block_mock,
    ):
      with self.assertLogs(level="INFO") as logs:
        rollout.update_params({})

      block_mock.assert_called_once_with(1)
      self.assertEqual(rollout._weight_sync_count, 2)  # pylint: disable=protected-access
      self.assertTrue(any("weight_sync_time: sync 1 (sync_weights) via" in line for line in logs.output))

    rollout._maxtext_config = SimpleNamespace(  # pylint: disable=protected-access
        log_weight_sync_time=False, log_weight_sync_norm=False
    )
    with mock.patch(
        "maxtext.integration.vllm.maxtext_vllm_rollout.vllm_rollout.VllmRollout.update_params",
        _fake_rollout_update,
    ):
      rollout.update_params({})
    self.assertEqual(rollout._weight_sync_count, 2)  # pylint: disable=protected-access

    with mock.patch(
        "maxtext.integration.vllm.maxtext_vllm_rollout.vllm_rollout.VllmRollout.update_params",
        _fake_real_rollout_update,
    ):
      rollout.update_params({})
    self.assertEqual(rollout._weight_sync_count, 3)  # pylint: disable=protected-access


class MaxTextAdapterSelectionTest(unittest.TestCase):
  """Verify only scanned MaxText-adapter rollouts take the custom sync path."""

  @pytest.mark.cpu_only
  def test_detects_dict_and_string_overrides(self):
    dict_config = SimpleNamespace(vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}, scan_layers=True)
    string_config = SimpleNamespace(vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}', scan_layers=True)

    self.assertTrue(uses_maxtext_vllm_adapter(dict_config))
    self.assertTrue(uses_maxtext_vllm_adapter(string_config))
    self.assertTrue(requires_maxtext_scanned_weight_unroll(dict_config))

  @pytest.mark.cpu_only
  def test_bypasses_native_or_unscanned_rollouts(self):
    native_config = SimpleNamespace(
        vllm_hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]}, scan_layers=True
    )
    unscanned_config = SimpleNamespace(vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}, scan_layers=False)

    self.assertFalse(requires_maxtext_scanned_weight_unroll(native_config))
    self.assertFalse(requires_maxtext_scanned_weight_unroll(unscanned_config))

  @pytest.mark.cpu_only
  def test_qwen35_direct_weight_converter_selection(self):
    cfg = SimpleNamespace(
        model_name="qwen3.5-35b-a3b",
        rollout_tensor_parallelism=8,
        num_decoder_layers=8,
        inhomogeneous_layer_cycle_interval=4,
        param_scan_axis=1,
        kv_tp_size=1,
        moe_mlp_tp_size=1,
        fp8_moe=False,
        rollout_fp8_moe=False,
        dtype="bfloat16",
    )
    conv = _create_model_converter(
        "qwen3.5-35b-a3b",
        config=cfg,
        mesh=None,
        use_hf_mapping=False,
        use_weight_converter=True,
    )
    self.assertIsInstance(conv, WeightConverter)
    self.assertIsNone(conv.rules)

  @pytest.mark.cpu_only
  def test_sampler_preserves_user_sharding_strategy(self):
    sampler = object.__new__(MaxTextVllmSampler)
    mesh = SimpleNamespace(
        shape={"data": 1, "model": 8},
        device_ids=np.arange(8).reshape(1, 8),
    )
    config = VllmConfig(
        init_with_random_weights=True,
        hbm_utilization=0.35,
        lora_config=None,
        mesh=mesh,
        tensor_parallel_size=8,
        data_parallel_size=1,
        expert_parallel_size=1,
        enable_dp_attention=False,
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": False,
                    "tensor_parallelism": 4,
                    "expert_parallelism": 2,
                }
            }
        },
    )
    args = sampler._vllm_config(config)  # pylint: disable=protected-access
    strategy = args["additional_config"]["sharding"]["sharding_strategy"]
    self.assertEqual(strategy["tensor_parallelism"], 4)
    self.assertEqual(strategy["expert_parallelism"], 2)
    self.assertFalse(strategy["enable_dp_attention"])
    self.assertEqual(strategy["device_indexes"], list(range(8)))
