# Copyright 2026 Google LLC
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

"""Unit tests for Kimi-K3 Group 3: Latent MoE."""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

_ORIG_MATMUL_PRECISION = None


def setUpModule():
  """Sets up high precision matmul and mocks reference modules."""
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")

  ref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kimi-k3-hf-reference"))
  if os.path.exists(ref_dir) and ref_dir not in sys.path:
    sys.path.insert(0, ref_dir)

  import types  # pylint: disable=import-outside-toplevel
  import transformers.utils.generic  # pylint: disable=import-outside-toplevel

  if not hasattr(transformers.utils.generic, "OutputRecorder"):

    class OutputRecorder:

      def __init__(self, *args, **kwargs):
        pass

    transformers.utils.generic.OutputRecorder = OutputRecorder
  if not hasattr(transformers.utils.generic, "check_model_inputs"):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None

  for mod in ["fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils", "fla.ops.utils.index", "fla.utils"]:
    if mod not in sys.modules:
      m = types.ModuleType(mod)
      sys.modules[mod] = m
      m.FusedRMSNormGated = None
      m.ShortConvolution = None
      m.chunk_kda = None
      m.fused_recurrent_kda = None
      m.prepare_cu_seqlens_from_mask = None
      m.prepare_lens_from_mask = None
      m.tensor_cache = lambda f: f


def tearDownModule():
  if _ORIG_MATMUL_PRECISION is not None:
    jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


from maxtext.layers.latent_moe import KimiDenseMLP, KimiMoERouter, KimiRoutedExperts, KimiLatentMoEBlock
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference


class KimiK3Group3MoETest(unittest.TestCase):
  """Tests validating KimiMoERouter, KimiDenseMLP, and KimiLatentMoEBlock."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 64
    self.routed_expert_hidden_size = 32
    self.moe_intermediate_size = 16
    self.shared_intermediate_size = 32
    self.num_experts = 8
    self.top_k = 2
    self.num_shared_experts = 2
    self.rms_norm_eps = 1e-5

    self.hf_config = {
        "hidden_size": self.hidden_size,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.top_k,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "shared_intermediate_size": self.shared_intermediate_size,
        "num_shared_experts": self.num_shared_experts,
        "rms_norm_eps": self.rms_norm_eps,
        "first_k_dense_replace": 1,
        "full_attn_layers": [1],
    }

  def test_kimi_moe_router(self):
    """Verifies router scoring, top-k selection, and weight normalization."""
    rngs = nnx.Rngs(0)
    router = KimiMoERouter(
        hidden_size=self.hidden_size,
        num_experts=self.num_experts,
        top_k=self.top_k,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.seq_len, self.hidden_size))
    topk_idx, topk_weight = router(x)

    self.assertEqual(topk_idx.shape, (self.batch_size, self.seq_len, self.top_k))
    self.assertEqual(topk_weight.shape, (self.batch_size, self.seq_len, self.top_k))

    # Check that weights sum to 1 for each token
    weight_sums = np.sum(np.array(topk_weight), axis=-1)
    np.testing.assert_allclose(weight_sums, np.ones_like(weight_sums), rtol=1e-5, atol=1e-5)

  def test_routed_expert_params_have_sharding_axes(self):
    """Router and routed-expert params carry logical sharding axes (else they replicate per chip)."""
    expected = {
        "bf16": {
            "gate/kernel": ("embed", None),
            "gate/e_score_correction_bias": (None,),
            "wi_0": ("exp", "embed_moe", "mlp_moe"),
            "wi_1": ("exp", "embed_moe", "mlp_moe"),
            "wo": ("exp", "mlp_moe", "embed_moe"),
        },
    }
    for fmt, want in expected.items():
      with self.subTest(weight_format=fmt):
        experts = KimiRoutedExperts(
            num_experts=4,
            in_features=64,
            intermediate_dim=32,
            hidden_size=self.hidden_size,
            top_k=2,
            rngs=nnx.Rngs(0),
        )
        got = {
            "/".join(map(str, path)): var.get_metadata().get("out_sharding")
            for path, var in nnx.state(experts, nnx.Param).flat_state()
        }
        self.assertEqual(got, want)

  @requires_kimi_k3_reference
  def test_kimi_dense_mlp_parity(self):
    """Verifies Layer 0 Dense MLP parity against PyTorch KimiMLP with 4x expansion."""
    import modeling_kimi_linear  # pylint: disable=import-outside-toplevel
    from configuration_kimi_k3 import KimiLinearConfig  # pylint: disable=import-outside-toplevel

    torch.manual_seed(42)
    test_hidden_size = 128
    test_intermediate_size = 512
    seq_len = 16
    pt_config = KimiLinearConfig(
        hidden_size=test_hidden_size,
        intermediate_size=test_intermediate_size,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    pt_mlp = modeling_kimi_linear.KimiMLP(pt_config)
    pt_mlp.eval()

    test_hf_config = dict(self.hf_config)
    test_hf_config["hidden_size"] = test_hidden_size
    test_hf_config["intermediate_size"] = test_intermediate_size

    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_mlp.state_dict(),
        test_hf_config,
        maxtext_config=None,
        prefix_filter="params-decoder-layers_0-mlp-",
    )

    rngs = nnx.Rngs(0)
    jax_mlp = KimiDenseMLP(
        in_features=test_hidden_size,
        intermediate_dim=test_intermediate_size,
        rngs=rngs,
    )
    jax_mlp.wi_0.kernel.value = jnp.array(converted_params["params-decoder-layers_0-mlp-wi_0-kernel"])
    jax_mlp.wi_1.kernel.value = jnp.array(converted_params["params-decoder-layers_0-mlp-wi_1-kernel"])
    jax_mlp.wo.kernel.value = jnp.array(converted_params["params-decoder-layers_0-mlp-wo-kernel"])

    x_torch = torch.randn(self.batch_size, seq_len, test_hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_mlp(x_torch).numpy()

    x_jax = jnp.array(x_torch.numpy())
    y_jax = np.array(jax_mlp(x_jax))

    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

  @requires_kimi_k3_reference
  def test_kimi_latent_moe_block_parity(self):
    """Verifies Latent MoE block parity against PyTorch KimiSparseMoeBlock."""
    import modeling_kimi_linear  # pylint: disable=import-outside-toplevel
    from configuration_kimi_k3 import KimiLinearConfig  # pylint: disable=import-outside-toplevel

    torch.manual_seed(42)
    pt_config = KimiLinearConfig(
        hidden_size=self.hidden_size,
        intermediate_size=128,
        num_experts=self.num_experts,
        num_experts_per_token=self.top_k,
        routed_expert_hidden_size=self.routed_expert_hidden_size,
        moe_intermediate_size=self.moe_intermediate_size,
        num_shared_experts=self.num_shared_experts,
        shared_intermediate_size=self.shared_intermediate_size,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        latent_moe_use_norm=True,
        rms_norm_eps=self.rms_norm_eps,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    pt_moe = modeling_kimi_linear.KimiSparseMoeBlock(pt_config)
    pt_moe.eval()

    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_moe.state_dict(),
        self.hf_config,
        maxtext_config=None,
        prefix_filter="params-decoder-layers_1-mlp-",
    )

    rngs = nnx.Rngs(0)
    jax_moe = KimiLatentMoEBlock(
        hidden_size=self.hidden_size,
        num_experts=self.num_experts,
        top_k=self.top_k,
        routed_expert_hidden_size=self.routed_expert_hidden_size,
        moe_intermediate_size=self.moe_intermediate_size,
        num_shared_experts=self.num_shared_experts,
        shared_intermediate_size=self.shared_intermediate_size,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        latent_moe_use_norm=True,
        rms_norm_eps=self.rms_norm_eps,
        rngs=rngs,
    )

    jax_moe.routed_experts.gate.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-routed_experts-gate-kernel"]
    )
    jax_moe.routed_experts.gate.e_score_correction_bias.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-routed_experts-gate-e_score_correction_bias"]
    )
    jax_moe.routed_expert_down_proj.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-routed_expert_down_proj-kernel"]
    )
    jax_moe.routed_expert_norm.scale.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-routed_expert_norm-scale"]
    )
    jax_moe.routed_expert_up_proj.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-routed_expert_up_proj-kernel"]
    )
    jax_moe.shared_expert.wi_0.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-shared_expert-wi_0-kernel"]
    )
    jax_moe.shared_expert.wi_1.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-shared_expert-wi_1-kernel"]
    )
    jax_moe.shared_expert.wo.kernel.value = jnp.array(
        converted_params["params-decoder-layers_1-mlp-shared_expert-wo-kernel"]
    )
    jax_moe.routed_experts.wi_0.value = jnp.array(converted_params["params-decoder-layers_1-mlp-routed_experts-wi_0"])
    jax_moe.routed_experts.wi_1.value = jnp.array(converted_params["params-decoder-layers_1-mlp-routed_experts-wi_1"])
    jax_moe.routed_experts.wo.value = jnp.array(converted_params["params-decoder-layers_1-mlp-routed_experts-wo"])

    x_torch = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_moe(x_torch).numpy()

    x_jax = jnp.array(x_torch.numpy())
    y_jax = np.array(jax_moe(x_jax))

    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
