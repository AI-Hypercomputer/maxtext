"""
Copyright 2023 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=line-too-long
"""
Tests for convert_gemma_chkpt.py
"""
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from MaxText.checkpoint_conversion_utils import convert_gemma_weights, nest_params

class TestConvertGemmaCheckpoint(unittest.TestCase):
  """Tests for Gemma checkpoint conversion."""

  def _assert_is_bfloat16_array(self, arr):
    self.assertIsInstance(arr, jnp.ndarray)
    self.assertEqual(arr.dtype, jnp.bfloat16)

  def test_convert_gemma_2b_structure_and_names(self):
    """Tests the basic structure and naming of the converted 2b model weights."""
    # Minimal mock data for a 1-layer "2b" model
    # Dimensions (can be arbitrary small numbers for structure test)
    embed_dim = 128
    num_heads = 4
    head_dim = 32
    hidden_dim = 256 # mlp intermediate dim

    # Using np arrays as source_params are usually numpy arrays when loaded by orbax
    source_params = {
        "transformer/final_norm/scale": np.ones((embed_dim,)),
        "transformer/embedder/input_embedding": np.ones((embed_dim, hidden_dim)), # Corrected: vocab_size, embed_dim
        "transformer/layer_0/attn/q_einsum/w": np.ones((num_heads, head_dim, embed_dim)),
        "transformer/layer_0/attn/kv_einsum/w": np.ones((2, num_heads, head_dim, embed_dim)), # K and V for MQA
        "transformer/layer_0/attn/attn_vec_einsum/w": np.ones((embed_dim, num_heads, head_dim)),
        "transformer/layer_0/mlp/gating_einsum/w": np.ones((2, hidden_dim, embed_dim)), # wi_0 and wi_1
        "transformer/layer_0/mlp/linear/w": np.ones((embed_dim, hidden_dim)),
        "transformer/layer_0/pre_attention_norm/scale": np.ones((embed_dim,)),
        "transformer/layer_0/pre_ffw_norm/scale": np.ones((embed_dim,)),
    }
    # Correct embedder shape based on usage: params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)
    # And typical shape is (vocab_size, embed_dim). For this test, make hidden_dim act as vocab_size for simplicity.
    source_params["transformer/embedder/input_embedding"] = np.ones((hidden_dim, embed_dim))


    # The script expects non-nested params for convert_gemma_weights, as it calls nest_params internally.
    converted_weights = convert_gemma_weights(source_params, model_size="2b")

    self.assertIn("decoder", converted_weights)
    self.assertIn("token_embedder", converted_weights)

    self.assertIn("decoder_norm", converted_weights["decoder"])
    self.assertIn("scale", converted_weights["decoder"]["decoder_norm"])
    self._assert_is_bfloat16_array(converted_weights["decoder"]["decoder_norm"]["scale"])

    self.assertIn("embedding", converted_weights["token_embedder"])
    self._assert_is_bfloat16_array(converted_weights["token_embedder"]["embedding"])

    self.assertIn("layers", converted_weights["decoder"])
    self.assertEqual(len(converted_weights["decoder"]["layers"]["mlp"]["wi_0"]["kernel"]), 1) # num_layers = 1

    layer_0 = converted_weights["decoder"]["layers"]
    self.assertIn("self_attention", layer_0)
    self.assertIn("query", layer_0["self_attention"])
    self.assertIn("kernel", layer_0["self_attention"]["query"])
    self._assert_is_bfloat16_array(layer_0["self_attention"]["query"]["kernel"])

    self.assertIn("key", layer_0["self_attention"])
    self.assertIn("kernel", layer_0["self_attention"]["key"])
    self._assert_is_bfloat16_array(layer_0["self_attention"]["key"]["kernel"])

    self.assertIn("value", layer_0["self_attention"])
    self.assertIn("kernel", layer_0["self_attention"]["value"])
    self._assert_is_bfloat16_array(layer_0["self_attention"]["value"]["kernel"])

    self.assertIn("out", layer_0["self_attention"])
    self.assertIn("kernel", layer_0["self_attention"]["out"])
    self._assert_is_bfloat16_array(layer_0["self_attention"]["out"]["kernel"])

    self.assertIn("mlp", layer_0)
    self.assertIn("wi_0", layer_0["mlp"])
    self.assertIn("kernel", layer_0["mlp"]["wi_0"])
    self._assert_is_bfloat16_array(layer_0["mlp"]["wi_0"]["kernel"])
    self.assertIn("wi_1", layer_0["mlp"])
    self.assertIn("kernel", layer_0["mlp"]["wi_1"])
    self._assert_is_bfloat16_array(layer_0["mlp"]["wi_1"]["kernel"])
    self.assertIn("wo", layer_0["mlp"])
    self.assertIn("kernel", layer_0["mlp"]["wo"])
    self._assert_is_bfloat16_array(layer_0["mlp"]["wo"]["kernel"])

    self.assertIn("pre_self_attention_norm", layer_0)
    self.assertIn("scale", layer_0["pre_self_attention_norm"])
    self._assert_is_bfloat16_array(layer_0["pre_self_attention_norm"]["scale"])
    self.assertIn("pre_ffw_norm", layer_0)
    self.assertIn("scale", layer_0["pre_ffw_norm"])
    self._assert_is_bfloat16_array(layer_0["pre_ffw_norm"]["scale"])


  def test_convert_gemma_2b_values_and_transformations(self):
    """Tests the specific values, shapes, dtypes, and transformations for 2b."""
    embed_dim = 8 # Must be divisible by num_heads for q_einsum shape
    num_heads = 2 # For MQA, K and V heads are 1
    head_dim = 4
    hidden_dim = 16 # mlp intermediate dim (e.g. feedforward_dim)
    vocab_size = 10 # arbitrary for token_embedder

    # Using easily identifiable np arrays
    source_params = {
        "transformer/final_norm/scale": np.arange(embed_dim, dtype=np.float32),
        "transformer/embedder/input_embedding": np.arange(vocab_size * embed_dim, dtype=np.float32).reshape((vocab_size, embed_dim)),
        # q_einsum.w original shape (num_heads, head_dim, embed_dim) -> MaxText (num_layers, num_heads, embed_dim, head_dim)
        "transformer/layer_0/attn/q_einsum/w": np.arange(num_heads * head_dim * embed_dim, dtype=np.float32).reshape((num_heads, head_dim, embed_dim)),
        # kv_einsum.w original shape (2, num_kv_heads, head_dim, embed_dim) -> MaxText (num_layers, num_kv_heads, embed_dim, head_dim)
        # For 2b, num_kv_heads is num_heads (MQA implies num_kv_heads is not 1, but tied to main heads)
        # The code seems to assume kv_einsum[0] is K and kv_einsum[1] is V, and that their num_heads dim matches q_einsum's num_heads for MQA
        "transformer/layer_0/attn/kv_einsum/w": np.arange(2 * num_heads * head_dim * embed_dim, dtype=np.float32).reshape((2, num_heads, head_dim, embed_dim)),
        # attn_vec_einsum.w original (embed_dim, num_heads, head_dim) -> MaxText (num_layers, num_heads, head_dim, embed_dim)
        "transformer/layer_0/attn/attn_vec_einsum/w": np.arange(embed_dim * num_heads * head_dim, dtype=np.float32).reshape((embed_dim, num_heads, head_dim)),
        # gating_einsum.w original (2, hidden_dim, embed_dim) -> MaxText (num_layers, embed_dim, hidden_dim)
        "transformer/layer_0/mlp/gating_einsum/w": np.arange(2 * hidden_dim * embed_dim, dtype=np.float32).reshape((2, hidden_dim, embed_dim)),
        # linear.w original (embed_dim, hidden_dim) -> MaxText (num_layers, hidden_dim, embed_dim)
        "transformer/layer_0/mlp/linear/w": np.arange(embed_dim * hidden_dim, dtype=np.float32).reshape((embed_dim, hidden_dim)),
        "transformer/layer_0/pre_attention_norm/scale": np.arange(embed_dim, dtype=np.float32) + 10.0, # Add offset to distinguish
        "transformer/layer_0/pre_ffw_norm/scale": np.arange(embed_dim, dtype=np.float32) + 20.0, # Add offset
    }

    converted_weights = convert_gemma_weights(source_params, model_size="2b")

    # Check dtypes (all should be bfloat16)
    # Use tree_leaves_with_path to get the path (k) for better error messages
    for k_path, v in jax.tree_util.tree_leaves_with_path(converted_weights):
      path_str = '/'.join(map(str, jax.tree_util.keystr(k_path)))
      if isinstance(v, jnp.ndarray):
        self.assertEqual(v.dtype, jnp.bfloat16, f"Key path {path_str} is not bfloat16")


    # Check shapes and transformations
    # 1. Decoder norm scale
    expected_final_norm_scale = jnp.array(source_params["transformer/final_norm/scale"] + 1, dtype=jnp.bfloat16)
    np.testing.assert_array_equal(converted_weights["decoder"]["decoder_norm"]["scale"], expected_final_norm_scale)

    # 2. Token embedder
    expected_embedding_val = source_params["transformer/embedder/input_embedding"] * jnp.sqrt(embed_dim)
    np.testing.assert_allclose(
        converted_weights["token_embedder"]["embedding"],
        jnp.array(expected_embedding_val, dtype=jnp.bfloat16),
        rtol=1e-2, atol=1e-2 # bfloat16 comparison
    )
    self.assertEqual(converted_weights["token_embedder"]["embedding"].shape, (vocab_size, embed_dim))

    # Layer specific checks (num_layers = 1)
    layer_0_weights = converted_weights["decoder"]["layers"]

    # 3. Attention QKV Out
    # Q: params["transformer"][in_layer_name]["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * head_dim**-0.5
    # Transpose from (num_heads, head_dim, embed_dim) to (head_dim, num_heads, embed_dim) then scale
    # Final MaxText shape: (1, num_heads, embed_dim, head_dim)
    q_original = source_params["transformer/layer_0/attn/q_einsum/w"]
    expected_q_kernel_single_layer = (q_original.transpose(1,0,2) * (head_dim**-0.5))
    expected_q_kernel = jnp.array(np.expand_dims(expected_q_kernel_single_layer, axis=0).transpose(0,2,3,1), dtype=jnp.bfloat16) # (1, N, D, H)
    self.assertEqual(layer_0_weights["self_attention"]["query"]["kernel"].shape, (1, num_heads, embed_dim, head_dim))
    np.testing.assert_allclose(layer_0_weights["self_attention"]["query"]["kernel"], expected_q_kernel, rtol=1e-2, atol=1e-2)

    # K (MQA): params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
    # Transpose from (num_heads, head_dim, embed_dim) to (head_dim, num_heads, embed_dim)
    # Final MaxText shape: (1, num_heads, embed_dim, head_dim) (num_kv_heads is num_heads for MQA in this setup)
    k_original = source_params["transformer/layer_0/attn/kv_einsum/w"][0]
    expected_k_kernel_single_layer = k_original.transpose(1,0,2)
    expected_k_kernel = jnp.array(np.expand_dims(expected_k_kernel_single_layer, axis=0).transpose(0,2,3,1), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["self_attention"]["key"]["kernel"].shape, (1, num_heads, embed_dim, head_dim))
    np.testing.assert_allclose(layer_0_weights["self_attention"]["key"]["kernel"], expected_k_kernel, rtol=1e-2, atol=1e-2)

    # V (MQA): params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
    v_original = source_params["transformer/layer_0/attn/kv_einsum/w"][1]
    expected_v_kernel_single_layer = v_original.transpose(1,0,2)
    expected_v_kernel = jnp.array(np.expand_dims(expected_v_kernel_single_layer, axis=0).transpose(0,2,3,1), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["self_attention"]["value"]["kernel"].shape, (1, num_heads, embed_dim, head_dim))
    np.testing.assert_allclose(layer_0_weights["self_attention"]["value"]["kernel"], expected_v_kernel, rtol=1e-2, atol=1e-2)

    # Out: params["transformer"][in_layer_name]["attn"]["attn_vec_einsum"]["w"]
    # Original (embed_dim, num_heads, head_dim) -> MaxText (1, num_heads, head_dim, embed_dim)
    out_original = source_params["transformer/layer_0/attn/attn_vec_einsum/w"]
    expected_out_kernel = jnp.array(np.expand_dims(out_original, axis=0).transpose(0,2,3,1), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["self_attention"]["out"]["kernel"].shape, (1, num_heads, head_dim, embed_dim))
    np.testing.assert_allclose(layer_0_weights["self_attention"]["out"]["kernel"], expected_out_kernel, rtol=1e-2, atol=1e-2)


    # 4. MLP weights
    # wi_0: params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][0]
    # Original (hidden_dim, embed_dim) -> MaxText (1, embed_dim, hidden_dim)
    wi0_original = source_params["transformer/layer_0/mlp/gating_einsum/w"][0]
    expected_wi0_kernel = jnp.array(np.expand_dims(wi0_original.T, axis=0), dtype=jnp.bfloat16) # Transpose then expand
    self.assertEqual(layer_0_weights["mlp"]["wi_0"]["kernel"].shape, (1, embed_dim, hidden_dim))
    np.testing.assert_allclose(layer_0_weights["mlp"]["wi_0"]["kernel"], expected_wi0_kernel, rtol=1e-2, atol=1e-2)

    # wi_1: params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][1]
    # Original (hidden_dim, embed_dim) -> MaxText (1, embed_dim, hidden_dim)
    wi1_original = source_params["transformer/layer_0/mlp/gating_einsum/w"][1]
    expected_wi1_kernel = jnp.array(np.expand_dims(wi1_original.T, axis=0), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["mlp"]["wi_1"]["kernel"].shape, (1, embed_dim, hidden_dim))
    np.testing.assert_allclose(layer_0_weights["mlp"]["wi_1"]["kernel"], expected_wi1_kernel, rtol=1e-2, atol=1e-2)

    # wo: params["transformer"][in_layer_name]["mlp"]["linear"]["w"]
    # Original (embed_dim, hidden_dim) -> MaxText (1, hidden_dim, embed_dim)
    wo_original = source_params["transformer/layer_0/mlp/linear/w"]
    expected_wo_kernel = jnp.array(np.expand_dims(wo_original.T, axis=0), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["mlp"]["wo"]["kernel"].shape, (1, hidden_dim, embed_dim))
    np.testing.assert_allclose(layer_0_weights["mlp"]["wo"]["kernel"], expected_wo_kernel, rtol=1e-2, atol=1e-2)

    # 5. Norm scales
    # pre_self_attention_norm: params["transformer"][in_layer_name]["pre_attention_norm"]["scale"] + 1
    # Original (embed_dim,) -> MaxText (1, embed_dim)
    pre_attn_norm_orig = source_params["transformer/layer_0/pre_attention_norm/scale"]
    expected_pre_attn_norm_scale = jnp.array(np.expand_dims(pre_attn_norm_orig + 1, axis=0), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["pre_self_attention_norm"]["scale"].shape, (1, embed_dim))
    np.testing.assert_allclose(layer_0_weights["pre_self_attention_norm"]["scale"], expected_pre_attn_norm_scale, rtol=1e-2, atol=1e-2)

    # pre_ffw_norm: params["transformer"][in_layer_name]["pre_ffw_norm"]["scale"] + 1
    pre_ffw_norm_orig = source_params["transformer/layer_0/pre_ffw_norm/scale"]
    expected_pre_ffw_norm_scale = jnp.array(np.expand_dims(pre_ffw_norm_orig + 1, axis=0), dtype=jnp.bfloat16)
    self.assertEqual(layer_0_weights["pre_ffw_norm"]["scale"].shape, (1, embed_dim))
    np.testing.assert_allclose(layer_0_weights["pre_ffw_norm"]["scale"], expected_pre_ffw_norm_scale, rtol=1e-2, atol=1e-2)


  def test_convert_gemma_7b_attention_structure(self):
    """Tests the attention structure for 7b (non-MQA path)."""
    embed_dim = 128
    num_heads = 4 # For 7b (non-MQA), K and V heads are also num_heads
    head_dim = 32
    hidden_dim = 256

    source_params = {
        "transformer/final_norm/scale": np.ones((embed_dim,)),
        "transformer/embedder/input_embedding": np.ones((hidden_dim, embed_dim)), # vocab_size, embed_dim
        # For 7b, qkv_einsum.w provides Q, K, V
        # Original shape (3, num_heads, head_dim, embed_dim)
        "transformer/layer_0/attn/qkv_einsum/w": np.ones((3, num_heads, head_dim, embed_dim)),
        "transformer/layer_0/attn/attn_vec_einsum/w": np.ones((embed_dim, num_heads, head_dim)),
        "transformer/layer_0/mlp/gating_einsum/w": np.ones((2, hidden_dim, embed_dim)),
        "transformer/layer_0/mlp/linear/w": np.ones((embed_dim, hidden_dim)),
        "transformer/layer_0/pre_attention_norm/scale": np.ones((embed_dim,)),
        "transformer/layer_0/pre_ffw_norm/scale": np.ones((embed_dim,)),
    }

    converted_weights = convert_gemma_weights(source_params, model_size="7b")

    self.assertIn("decoder", converted_weights)
    self.assertIn("layers", converted_weights["decoder"])
    layer_0 = converted_weights["decoder"]["layers"]

    self.assertIn("self_attention", layer_0)
    # Check that Q, K, V kernels exist and have the expected shapes after processing
    # Q: params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][0].transpose((1, 0, 2)) * head_dim**-0.5
    # K: params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][1].transpose((1, 0, 2))
    # V: params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][2].transpose((1, 0, 2))
    # All should end up as (1, num_heads, embed_dim, head_dim) for MaxText
    expected_attn_shape = (1, num_heads, embed_dim, head_dim)

    self._assert_is_bfloat16_array(layer_0["self_attention"]["query"]["kernel"])
    self.assertEqual(layer_0["self_attention"]["query"]["kernel"].shape, expected_attn_shape)

    self._assert_is_bfloat16_array(layer_0["self_attention"]["key"]["kernel"])
    self.assertEqual(layer_0["self_attention"]["key"]["kernel"].shape, expected_attn_shape)

    self._assert_is_bfloat16_array(layer_0["self_attention"]["value"]["kernel"])
    self.assertEqual(layer_0["self_attention"]["value"]["kernel"].shape, expected_attn_shape)

    # A quick check on values for Q (scaled) vs K/V (not scaled initially from qkv_einsum)
    # This is a bit implicit, but the scaling is only on Q in the 7b path.
    # We can't directly compare values without replicating the exact transpose and scaling,
    # but we can ensure they are different if scaling factor is not 1.
    # If head_dim is 1, scaling is by 1, so this check isn't useful.
    if head_dim != 1:
        # Create a small distinct array for q,k,v part of qkv_einsum
        qkv_w_distinct = np.array([
            np.full((num_heads, head_dim, embed_dim), 1.0), # Q part
            np.full((num_heads, head_dim, embed_dim), 2.0), # K part
            np.full((num_heads, head_dim, embed_dim), 3.0)  # V part
        ])
        source_params_distinct_qkv = source_params.copy()
        source_params_distinct_qkv["transformer/layer_0/attn/qkv_einsum/w"] = qkv_w_distinct

        # Also need attn_vec_einsum for num_heads, head_dim to be inferred
        source_params_distinct_qkv["transformer/layer_0/attn/attn_vec_einsum/w"] = np.ones((embed_dim, num_heads, head_dim))


        converted_distinct = convert_gemma_weights(source_params_distinct_qkv, model_size="7b")

        q_kernel = converted_distinct["decoder"]["layers"]["self_attention"]["query"]["kernel"]
        k_kernel = converted_distinct["decoder"]["layers"]["self_attention"]["key"]["kernel"]
        v_kernel = converted_distinct["decoder"]["layers"]["self_attention"]["value"]["kernel"]

        # After transpose and scaling, Q values should be different from K/V if scaling factor is not 1.0
        # Q is scaled by head_dim**-0.5. K and V are not.
        # Original Q values were 1.0. Original K values were 2.0.
        # Expected Q value after scaling: 1.0 * head_dim**-0.5
        # Expected K value (after transpose but no scaling): 2.0
        # This checks that Q path is scaled and K/V path is not (for this step)
        self.assertNotAlmostEqual(q_kernel.mean(), k_kernel.mean())
        self.assertAlmostEqual(k_kernel.mean(), 2.0 * (head_dim**-0.0), delta=1e-2) # Effectively 2.0
        self.assertAlmostEqual(v_kernel.mean(), 3.0 * (head_dim**-0.0), delta=1e-2) # Effectively 3.0
        self.assertAlmostEqual(q_kernel.mean(), 1.0 * (head_dim**-0.5), delta=1e-2)


if __name__ == '__main__':
  unittest.main()
