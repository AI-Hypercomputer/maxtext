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

from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import pytest

from maxtext.integration.vllm.maxtext_vllm_rollout import (
    MaxTextVllmRollout,
    prepare_direct_sync_additional_config,
    requires_maxtext_scanned_weight_unroll,
    unroll_gemma_scanned_weights,
    unroll_qwen_scanned_weights,
    uses_maxtext_vllm_adapter,
    validate_direct_sync_layer_coverage,
)

pytestmark = pytest.mark.post_training


class MockWeights:
  """A mock weight container that implements to_pure_dict."""

  def __init__(self, pure_dict):
    self._pure_dict = pure_dict

  def to_pure_dict(self):
    return self._pure_dict


class GemmaScannedWeightsUnrollTest(unittest.TestCase):
  """Verify the correctness of the unroll_gemma_scanned_weights utility."""

  def test_bypasses_non_pytree_weights(self):
    """If the weights object doesn't have `to_pure_dict`, it should be returned unchanged."""
    raw_weights = {"dummy": np.ones(5)}
    result = unroll_gemma_scanned_weights(raw_weights)
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
    result = unroll_gemma_scanned_weights(weights)
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
    unrolled = unroll_gemma_scanned_weights(weights)

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
    unrolled = unroll_gemma_scanned_weights(weights)

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

    unrolled = unroll_qwen_scanned_weights(weights)

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
      unroll_qwen_scanned_weights(weights)

  @pytest.mark.cpu_only
  def test_invalid_scan_axis_formats_non_string_parameter_path(self):
    weights = MockWeights({"decoder": {"layers": {"layer_0": {0: np.ones((2, 1))}}}})

    with self.assertRaisesRegex(ValueError, r"decoder\.layers\.layer_0\.0"):
      unroll_qwen_scanned_weights(weights, scan_axis=2, pattern_length=1)

  @pytest.mark.cpu_only
  def test_supports_nondefault_axis_and_sparse_slots(self):
    slot_1 = np.stack(
        [np.full((2, 1), 1, dtype=np.float32), np.full((2, 1), 3, dtype=np.float32)],
        axis=0,
    )
    weights = MockWeights({"decoder": {"layers": {"layer_1": {"probe": slot_1}}}})

    unrolled = unroll_qwen_scanned_weights(weights, scan_axis=0, pattern_length=2)

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


class DirectSyncRolloutConfigTest(unittest.TestCase):
  """Verify TP-sharded MoE rollout targets request the safe fused layout."""

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

  @pytest.mark.cpu_only
  def test_leaves_dense_or_single_tp_config_unchanged(self):
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
            direct_maxtext_sync=True,
            num_experts=256,
            tensor_parallel_size=1,
        ),
        original,
    )


class MaxTextVllmRolloutConfigForwardingTest(unittest.TestCase):
  """Verify the custom rollout preserves Tunix rollout options."""

  @pytest.mark.cpu_only
  def test_forwards_sampling_parallelism_and_capacity_options(self):
    sampling_kwargs = {
        "stop": ["</answer>"],
        "detokenize": True,
        "include_stop_str_in_output": True,
    }
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
    )
    maxtext_config = SimpleNamespace(
        model_name="qwen3.5-35b-a3b",
        num_experts=256,
        param_scan_axis=1,
        inhomogeneous_layer_cycle_interval=4,
        swap_space_vllm_gb=2,
        vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
    )
    fake_sampler = mock.MagicMock()

    with (
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.mappings.MappingConfig.build",
            return_value=object(),
        ),
        mock.patch(
            "maxtext.integration.vllm.maxtext_vllm_rollout.VllmConfig",
            side_effect=SimpleNamespace,
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

    config = sampler_cls.call_args.kwargs["config"]
    self.assertEqual(config.sampling_kwargs, sampling_kwargs)
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
