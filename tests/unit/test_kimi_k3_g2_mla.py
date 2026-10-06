#  Copyright 2023-2026 Google LLC
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

"""Unit tests for Kimi-K3 Group 2: Gated Multi-Head Latent Attention (MLA)."""

import os
import sys
import unittest
from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch

from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING, HOOK_FNS
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference


def convert_pytorch_module_to_maxtext_params(pt_state_dict, hf_config, maxtext_config, prefix_filter=None):
  """Converts PyTorch state dict tensors to MaxText param arrays using Kimi-K3 mappings and hooks."""
  mapping_fn = PARAM_MAPPING["kimi-k3"]
  hook_fn_factory = HOOK_FNS["kimi-k3"]

  param_map = mapping_fn(hf_config, maxtext_config, scan_layers=False)
  hook_map = hook_fn_factory(hf_config, maxtext_config, scan_layers=False, saving_to_hf=False)

  converted_params = {}
  for mt_key, hf_target in param_map.items():
    if prefix_filter and not mt_key.startswith(prefix_filter):
      continue

    if isinstance(hf_target, str):
      matching_key = None
      for k in pt_state_dict.keys():
        if hf_target.endswith(k):
          matching_key = k
          break

      if matching_key in pt_state_dict:
        tensor_np = pt_state_dict[matching_key].detach().cpu().float().numpy()
        if mt_key in hook_map:
          tensor_np = hook_map[mt_key](tensor_np)
        converted_params[mt_key] = tensor_np

  return converted_params


class KimiK3G2MLATest(unittest.TestCase):
  """Unit tests for Kimi-K3 Gated MLA Attention layer."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 16
    self.hidden_size = 7168
    self.num_heads = 96
    self.head_dim = 128
    self.qk_rope_head_dim = 64
    self.q_lora_rank = 1536
    self.kv_lora_rank = 512
    self.rms_norm_eps = 1e-5

    self.hf_config = {
        "text_config": {
            "num_hidden_layers": 4,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "mla_use_output_gate": True,
            "linear_attn_config": {
                "num_layers": 4,
                "layer_indices": [0, 1, 2],
            },
        }
    }

  @requires_kimi_k3_reference
  def test_gated_mla_numerical_parity(self):
    """Tests numerical parity between MaxText MLA and PyTorch reference KimiMLAAttention."""
    ref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kimi-k3-hf-reference"))
    if ref_dir not in sys.path:
      sys.path.insert(0, ref_dir)
    from configuration_kimi_k3 import KimiLinearConfig  # pylint: disable=import-outside-toplevel
    from modeling_kimi_linear import KimiMLAAttention  # pylint: disable=import-outside-toplevel

    # 1. PyTorch Reference Setup
    pt_config = KimiLinearConfig(
        hidden_size=self.hidden_size,
        num_attention_heads=self.num_heads,
        num_key_value_heads=self.num_heads,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        qk_nope_head_dim=self.head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.head_dim,
        mla_use_nope=True,
        mla_use_output_gate=True,
        rms_norm_eps=self.rms_norm_eps,
    )
    pt_config._attn_implementation = "eager"  # pylint: disable=protected-access
    pt_mla = KimiMLAAttention(pt_config, layer_idx=3)
    pt_mla.eval()

    # 2. PyTorch Forward Pass
    torch.manual_seed(42)
    pt_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    causal_mask = torch.triu(
        torch.full((self.batch_size, 1, self.seq_len, self.seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    with torch.no_grad():
      pt_out = pt_mla(pt_input, attention_mask=causal_mask)
      if isinstance(pt_out, tuple):
        pt_out = pt_out[0]

    # 3. Convert PyTorch weights to MaxText params
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_mla.state_dict(),
        self.hf_config,
        None,
        prefix_filter="params-decoder-layers_3-self_attention",
    )

    # 4. MaxText MLA Setup
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="kimi_k3_mla_test",
        model_name="default",
        attention="dot_product",
        attention_type="mla",
        base_emb_dim=self.hidden_size,
        emb_dim=self.hidden_size,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        head_dim=self.head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        qk_nope_head_dim=self.head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.head_dim,
        mla_use_output_gate=True,
        mla_naive_kvcache=True,
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        normalization_layer_epsilon=self.rms_norm_eps,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        global_batch_size_to_train_on=self.batch_size,
        per_device_batch_size=self.batch_size,
        rope_max_timescale=500000.0,
        max_position_embeddings=1048576,
        original_max_position_embeddings=1048576,
        enable_checkpointing=False,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    nnx_rng = nnx.Rngs(params=0, dropout=0)

    jax_mla = attention_mla.MLA(
        config=cfg,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        head_dim=self.head_dim,
        inputs_q_shape=(self.batch_size, self.seq_len, self.hidden_size),
        inputs_kv_shape=(self.batch_size, self.seq_len, self.hidden_size),
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        attention_type="mla",
        mla_use_output_gate=True,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        qk_nope_head_dim=self.head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.head_dim,
        max_position_embeddings=1048576,
        original_max_position_embeddings=1048576,
        mscale=1.0,
        rope_factor=40.0,
        rngs=nnx_rng,
    )

    # 5. Populate MaxText MLA weights with converted parameters
    jax_mla.wq_a.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-wq_a-kernel"])
    jax_mla.q_norm.scale.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-q_norm-scale"])
    jax_mla.wq_b.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-wq_b-kernel"])
    jax_mla.wkv_a.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-wkv_a-kernel"])
    jax_mla.kv_norm.scale.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-kv_norm-scale"])
    jax_mla.wkv_b.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-wkv_b-kernel"])
    jax_mla.g_proj.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-g_proj-kernel"])
    jax_mla.out.kernel.value = jnp.asarray(converted_params["params-decoder-layers_3-self_attention-out-kernel"])

    # 6. MaxText Forward Pass
    x_jax = jnp.asarray(pt_input.detach().cpu().numpy())
    positions = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)
    decoder_segment_ids = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)

    jax_out, _ = jax_mla(
        inputs_q=x_jax,
        inputs_kv=x_jax,
        inputs_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
        model_mode="train",
    )

    # 7. Numerical Parity Assertion
    np.testing.assert_allclose(
        np.asarray(jax_out),
        pt_out.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_param_mapping_shapes(self):
    """Verifies that Group 2 parameter mappings and transformation hooks produce correct MaxText shapes."""
    hooks = HOOK_FNS["kimi-k3"](self.hf_config, None, scan_layers=False, saving_to_hf=False)

    # Test wq_a: [1536, 7168] -> [7168, 1536]
    pt_wq_a = np.zeros((self.q_lora_rank, self.hidden_size), dtype=np.float32)
    mt_wq_a = hooks["params-decoder-layers_3-self_attention-wq_a-kernel"](pt_wq_a)
    self.assertEqual(mt_wq_a.shape, (self.hidden_size, self.q_lora_rank))

    # Test wq_b: [96 * 192, 1536] -> [1536, 96, 192]
    pt_wq_b = np.zeros((self.num_heads * (self.head_dim + self.qk_rope_head_dim), self.q_lora_rank), dtype=np.float32)
    mt_wq_b = hooks["params-decoder-layers_3-self_attention-wq_b-kernel"](pt_wq_b)
    self.assertEqual(mt_wq_b.shape, (self.q_lora_rank, self.num_heads, self.head_dim + self.qk_rope_head_dim))

    # Test wkv_a: [512 + 64, 7168] -> [7168, 576]
    pt_wkv_a = np.zeros((self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size), dtype=np.float32)
    mt_wkv_a = hooks["params-decoder-layers_3-self_attention-wkv_a-kernel"](pt_wkv_a)
    self.assertEqual(mt_wkv_a.shape, (self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim))

    # Test wkv_b: [96 * (128 + 128), 512] -> [512, 96, 256]
    pt_wkv_b = np.zeros((self.num_heads * (self.head_dim + self.head_dim), self.kv_lora_rank), dtype=np.float32)
    mt_wkv_b = hooks["params-decoder-layers_3-self_attention-wkv_b-kernel"](pt_wkv_b)
    self.assertEqual(mt_wkv_b.shape, (self.kv_lora_rank, self.num_heads, self.head_dim + self.head_dim))

    # Test g_proj: [96 * 128, 7168] -> [7168, 96, 128]
    pt_g_proj = np.zeros((self.num_heads * self.head_dim, self.hidden_size), dtype=np.float32)
    mt_g_proj = hooks["params-decoder-layers_3-self_attention-g_proj-kernel"](pt_g_proj)
    self.assertEqual(mt_g_proj.shape, (self.hidden_size, self.num_heads, self.head_dim))

    # Test out: [7168, 96 * 128] -> [96, 128, 7168]
    pt_out = np.zeros((self.hidden_size, self.num_heads * self.head_dim), dtype=np.float32)
    mt_out = hooks["params-decoder-layers_3-self_attention-out-kernel"](pt_out)
    self.assertEqual(mt_out.shape, (self.num_heads, self.head_dim, self.hidden_size))


if __name__ == "__main__":
  unittest.main()
