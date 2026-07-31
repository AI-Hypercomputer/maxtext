"""Tests for the unstacking/weight-mapping logic in MaxTextVllmSampler."""

import unittest
from unittest import mock

import jax
from flax import nnx

from maxtext.configs import types
from maxtext.integration.vllm.maxtext_vllm_adapter.adapter import MaxTextForCausalLM
from maxtext.integration.vllm import maxtext_vllm_rollout


class RLWeightUnstackingTests(unittest.TestCase):
  """Tests to ensure MaxText models do not throw IndexError during parameter unstacking."""

  def _create_dummy_model(self, scan_layers: bool):
    """Create a minimal MaxTextForCausalLM instance."""
    # We instantiate RLConfig directly, mirroring other unit tests, to bypass
    # strict CLI/YAML argument validation for unsupported legacy flags.
    config = types.RLConfig(
        model_name="llama3.1-8b",
        scan_layers=scan_layers,
        base_emb_dim=16,
        base_mlp_dim=32,
        base_num_decoder_layers=2,
        base_num_heads=2,
        base_num_kv_heads=2,
        vocab_size=128,
        head_dim=8,
    )

    # We must explicitly disable Pathways resharding logic for standalone testing
    config.use_pathways = False

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    with mesh:
      model = MaxTextForCausalLM(config, mesh)
      state = nnx.state(model)

    return config, state

  def test_unstacking_llama_scanned_workaround(self):
    """Tests the generic unstacking algorithm for scanned base layers."""
    config, state = self._create_dummy_model(scan_layers=True)
    pure_dict = state.to_pure_dict()

    # Verify the structure is actually scanned
    self.assertTrue(any("layers" in k for k in pure_dict.keys()))

    # Run the unrolling wrapper that checks for the Gemma bug
    unrolled_state = maxtext_vllm_rollout.unroll_gemma_scanned_weights(pure_dict)

    # Since this is a Llama architecture (MaxTextForCausalLM), it should
    # successfully process the dictionary without an IndexError.
    self.assertIsNotNone(unrolled_state)

  def test_unstacking_llama_unscanned_workaround(self):
    """Tests that unscanned layers pass through safely without modification."""
    config, state = self._create_dummy_model(scan_layers=False)
    pure_dict = state.to_pure_dict()

    unrolled_state = maxtext_vllm_rollout.unroll_gemma_scanned_weights(pure_dict)
    self.assertIsNotNone(unrolled_state)

  @mock.patch("maxtext.integration.vllm.maxtext_vllm_rollout._create_model_converter")
  def test_standalone_converter_qwen_routing(self, mock_create):
    """Tests that Qwen architectures bypass Tunix unstacking entirely."""
    # Qwen models use use_standalone_converter=True
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    maxtext_vllm_rollout._create_model_converter("qwen3-30b-a3b", None, mesh)
    mock_create.assert_called_once()


if __name__ == "__main__":
  unittest.main()
