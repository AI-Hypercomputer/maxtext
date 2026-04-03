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

""" Unit tests for all optimizers. """
import re
import unittest
from unittest.mock import patch, MagicMock
import jax
import jax.numpy as jnp

import pytest
from absl.testing import parameterized
from flax import nnx
from optax.contrib import MuonDimensionNumbers as mdn

from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils, muon_utils
from tests.utils.test_helpers import get_test_config_path
from typing import NamedTuple


# deepseek2, specific: q_lora_rank=0
# applicable: deepseek2-16, but not deepseek2-236b (q_lora_rank=1536)
_DEEPSEEK2_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "query": {"kernel": mdn((0,), (-2, -1))},  # ds2
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK2_ATTENTION,
            },
            "logits_dense": {"kernel": None},
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,))},  # ds2
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                **_DEEPSEEK2_ATTENTION,
            },
        },
        "token_embedder": {"embedding": None},
    }
}


# deepseek3
_DEEPSEEK3_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "q_norm": {"scale": None},  # ds3
        "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
        "wq_b": {"kernel": mdn((0,), (-2, -1))},  # ds3
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK3_ATTENTION,
            },
            "logits_dense": {"kernel": None},
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,)), "bias": None},  # ds3
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                **_DEEPSEEK3_ATTENTION,
            },
        },
        "token_embedder": {"embedding": None},
    }
}

# gemma3
_GEMMA3_LAYER = {
    "mlp": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
    "post_ffw_norm": {"scale": None},
    "pre_ffw_norm": {"scale": None},
    "self_attention": {
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
        "key_norm": {"scale": None},
        "query_norm": {"scale": None},
    },
    "post_self_attention_norm": {"scale": None},
    "pre_self_attention_norm": {"scale": None},
}

GEMMA3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {f"layers_{i}": _GEMMA3_LAYER for i in range(6)},
            "layers_remainder": {f"layers_{i}": _GEMMA3_LAYER for i in range(4)},
        },
        "token_embedder": {"embedding": None},
    }
}


# llama2 (also llama3)
LLAMA2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3, specific: logits_via_embedding=True
# applicable: qwen3-0.6b, qwen3-4b, but not: qwen3-8b, qwen3-14b (logits_via_embedding=False)
QWEN3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "key_norm": {"scale": None},
                    "query_norm": {"scale": None},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
        },
        "token_embedder": {"embedding": None},
    }
}


class MuonDimensionTest(parameterized.TestCase):
  """Unit tests for Muon dimension number generation.

  This suite verifies that the automatically generated Muon dimension numbers
  for various models match their hardcoded reference values.
  """

  @parameterized.named_parameters(
      ("deepseek2-16b", "deepseek2-16b", DEEPSEEK2_DIMENSION_NUMBER),
      ("deepseek3-671b", "deepseek3-671b", DEEPSEEK3_DIMENSION_NUMBER),
      ("kimi-k2-1t", "kimi-k2-1t", DEEPSEEK3_DIMENSION_NUMBER),
      ("llama2-7b", "llama2-7b", LLAMA2_DIMENSION_NUMBER),
      ("llama3-8b", "llama3-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.1-8b", "llama3.1-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.3-70b", "llama3.3-70b", LLAMA2_DIMENSION_NUMBER),
      ("gemma3-4b", "gemma3-4b", GEMMA3_DIMENSION_NUMBER),
      ("qwen3-0.6b", "qwen3-0.6b", QWEN3_DIMENSION_NUMBER),
  )
  @pytest.mark.tpu_only
  def test_model_integration(self, model_name, expected_output):
    """
    Initializes the specified MaxText model and asserts that the generated
    Muon dimension numbers match the hardcoded reference.
    """
    actual_output = muon_utils.get_model_mdn(model_name, scan_layers=True)
    self.assertEqual(actual_output, expected_output)


class AdamWMaskTest(parameterized.TestCase):
  """Tests for the AdamW mask functionality"""

  def test_get_adamw_mask_with_empty_mask(self):
    """Directly test the get_adamw_mask function with empty list"""
    # Case 1: No mask in config (empty list)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=[]"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertIsNone(mask_fn)

  def test_get_adamw_mask_with_valid_mask(self):
    """Directly test the get_adamw_mask function with valid mask"""
    # Case 2: Mask in config
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias', '.*norm', '.*ln.*']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertTrue(callable(mask_fn))

    params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": {"ln": {"scale": 4}}}
    mask = mask_fn(params)
    self.assertTrue(mask["layer1"]["kernel"])
    self.assertFalse(mask["layer1"]["bias"])
    self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
    self.assertFalse(mask["layer3"]["ln"]["scale"])

  def test_get_adamw_mask_with_invalid_mask(self):
    """Test that an invalid regex in the mask config raises an error when used"""
    # Create a config with an invalid regex (unbalanced bracket)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['[']"]
    config = pyconfig.initialize(argv)

    # Applying the mask should raise re.error due to the invalid regex
    with self.assertRaises(re.error):
      optimizers.get_adamw_mask(config)

  def test_get_adamw_mask_with_getattrkey(self):
    """Test that get_adamw_mask correctly handles GetAttrKey (e.g. from NamedTuples)"""

    class MyParams(NamedTuple):
      kernel: jax.Array
      bias: jax.Array

    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)

    params = MyParams(kernel=jax.numpy.ones((2, 2)), bias=jax.numpy.zeros((2,)))
    mask = mask_fn(params)

    self.assertTrue(mask.kernel)
    self.assertFalse(mask.bias)

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_with_mask(self, opt_type, mock_path):
    """Test that optimizer receives the mask function from config and it works as expected"""
    # Create a config with a mask list including regex
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "adamw_mask=['bias', 'layer_norm', 'layer1/.*kernel']",
        f"opt_type={opt_type}",
    ]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with a mask function
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      mask_fn = kwargs["mask"]

      # Verify that mask_fn is not None
      self.assertIsNotNone(mask_fn)

      # Test the behavior of mask_fn
      params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": [4, 5]}

      mask = mask_fn(params)

      # kernel in layer1 should be False because of 'layer1/.*kernel'
      self.assertFalse(mask["layer1"]["kernel"])
      # bias in layer1 should be False because of 'bias'
      self.assertFalse(mask["layer1"]["bias"])
      # layer_norm should be False because of 'layer_norm'
      self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
      # layer3 elements should be True
      self.assertTrue(mask["layer3"][0])
      self.assertTrue(mask["layer3"][1])

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_without_mask(self, opt_type, mock_path):
    """Test that optimizer receives None for mask when config is empty"""
    argv = ["", get_test_config_path(), "run_name=test", f"opt_type={opt_type}"]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with mask=None
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      self.assertIsNone(kwargs["mask"])


class TrainableParametersMaskTest(parameterized.TestCase):
  """Tests for the trainable parameters mask functionality via get_optimizer"""

  def test_get_optimizer_with_trainable_mask(self):
    """Test get_optimizer with a valid trainable_parameters_mask."""
    argv = [
        "",
        get_test_config_path(),
        "run_name=test_with_trainable_mask",
        "trainable_parameters_mask=['.*indexer.*', 'layer_norm']",
    ]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    # We can test the optimizer by creating some dummy params and gradients
    # and checking if the updates are zeroed out for non-trainable parameters.
    params = {
        "layer1": {"kernel": jax.numpy.ones((2, 2)), "indexer": jax.numpy.ones((2, 2))},
        "layer2": {"layer_norm": {"scale": jax.numpy.ones((2, 2))}},
        "layer3": {"ln": {"scale": jax.numpy.ones((2, 2))}},
    }

    # Give some non-zero gradients
    grads = jax.tree_util.tree_map(lambda x: jax.numpy.ones_like(x) * 0.5, params)

    # Initialize optimizer state
    opt_state = opt.init(params)

    # Compute updates
    updates, _ = opt.update(grads, opt_state, params)

    # 'layer1/kernel' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer1"]["kernel"] == 0))
    # 'layer3/ln/scale' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer3"]["ln"]["scale"] == 0))
    # 'layer1/indexer' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer1"]["indexer"] == 0))
    # 'layer2/layer_norm/scale' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer2"]["layer_norm"]["scale"] == 0))

  def test_get_optimizer_without_trainable_mask(self):
    """Test get_optimizer when trainable_parameters_mask is empty."""
    argv = ["", get_test_config_path(), "run_name=test", "trainable_parameters_mask=[]"]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    params = {"layer1": {"kernel": jax.numpy.ones((2, 2))}}
    grads = {"layer1": {"kernel": jax.numpy.ones((2, 2)) * 0.5}}

    opt_state = opt.init(params)
    updates, _ = opt.update(grads, opt_state, params)

    # When no trainable mask is provided, nothing is frozen by this mechanism
    self.assertFalse(jax.numpy.all(updates["layer1"]["kernel"] == 0))


class TestMuonLogic(unittest.TestCase):
  """Tests the granular path transformation functions."""

  def test_is_path_contain_any(self):
    # pylint: disable=protected-access
    self.assertTrue(muon_utils._is_path_contain_any(("a", "b"), ("x", "a", "z")))
    self.assertFalse(muon_utils._is_path_contain_any(("a", "b"), ("x", "y", "z")))

  def test_transform_logic_exclusions(self):
    self.assertIsNone(muon_utils.transform_logic(("layer_0", "bias")))
    self.assertIsNone(muon_utils.transform_logic(("layer_0", "scale")))
    self.assertIsNone(muon_utils.transform_logic(("embedding", "kernel")))

  def test_transform_logic_moe(self):
    path = ("layers_0", "MoeBlock_0", "wi_0")
    result = muon_utils.transform_logic(path)
    self.assertEqual(result.reduction_axis, (-2,))
    self.assertEqual(result.output_axis, (-1,))

  def test_transform_logic_attention(self):
    path_out = ("layers_0", "self_attention", "out", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_out), mdn((0, -2), (-1,)))

    path_q = ("layers_0", "self_attention", "query", "kernel")
    self.assertEqual(muon_utils.transform_logic(path_q), mdn((0,), (-2, -1)))

  def test_get_transform_tree(self):
    fake_tree = {"params": {"layer_0": {"kernel": "leaf", "bias": "leaf"}, "MoeBlock_0": {"wi_0": "leaf"}}}
    result = muon_utils.get_transform_tree(fake_tree)
    self.assertEqual(result["params"]["layer_0"]["kernel"], mdn((0,), (-1,)))
    self.assertIsNone(result["params"]["layer_0"]["bias"])

  def test_get_muon_weight_dimension_numbers_nnx(self):
    """Verifies dimension extraction for stateful NNX modules."""

    class MockNNXModel(nnx.Module):
      """Mock NNX Module."""

      def __init__(self, rngs: nnx.Rngs):
        # 1. Standard layer
        self.layer1 = nnx.Linear(2, 4, rngs=rngs)

        # 2. MoE specific naming to trigger transform logic.
        # The logic expects "MoeBlock_0" AND "wi_0"/"wi_1"/"wo" in the path.
        # We nest the linear layer to create the path: ('MoeBlock_0', 'wi_0', 'kernel')
        self.MoeBlock_0 = nnx.Module()
        self.MoeBlock_0.wi_0 = nnx.Linear(4, 2, rngs=rngs)

        # 3. Exclusion case (scaler/scale)
        self.scale = nnx.Param(jnp.ones((1,)))

    # Use eval_shape to create an abstract version of the model.
    model = nnx.eval_shape(lambda: MockNNXModel(rngs=nnx.Rngs(0)))
    config = MagicMock()

    # Extract dimension numbers using the NNX path in muon_utils
    result = muon_utils.get_muon_weight_dimension_numbers(model, config)

    # Verify standard weight path: ('layer1', 'kernel') -> default (0,)
    self.assertEqual(result.layer1.kernel.value, mdn((0,), (-1,)))

    # Verify MoE weight path: ('MoeBlock_0', 'wi_0', 'kernel') -> (-2,)
    self.assertEqual(result.MoeBlock_0.wi_0.kernel.value, mdn((-2,), (-1,)))

    # Verify exclusion (scalar/scale)
    self.assertIsNone(result.scale.value)

  def test_verbose_output_nnx(self):
    """Covers lines 128 and 135-154: _print_structure_debug via verbose=True with NNX model."""

    class SimpleNNXModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)

    model = nnx.eval_shape(lambda: SimpleNNXModel(rngs=nnx.Rngs(0)))
    config = MagicMock()
    muon_utils.get_muon_weight_dimension_numbers(model, config, verbose=True)

  def test_nnx_deepseek_attention_logic(self):
    """Simulates a DeepSeek-like attention structure in NNX."""

    class DeepSeekAttention(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.self_attention = nnx.Module()
        self.self_attention.query = nnx.Linear(8, 8, rngs=rngs)
        self.self_attention.out = nnx.Linear(8, 8, rngs=rngs)

    # Use eval_shape to create an abstract version of the model.
    model = nnx.eval_shape(lambda: DeepSeekAttention(nnx.Rngs(0)))
    config = MagicMock()
    result = muon_utils.get_muon_weight_dimension_numbers(model, config)

    # Check attention query: [0] -> [-2, -1]
    self.assertEqual(result.self_attention.query.kernel.value, mdn((0,), (-2, -1)))
    # Check attention out: [0, -2] -> [-1]
    self.assertEqual(result.self_attention.out.kernel.value, mdn((0, -2), (-1,)))


if __name__ == "__main__":
  unittest.main()
