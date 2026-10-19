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


from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.latent_moe import KimiDenseMLP, KimiMoERouter, KimiRoutedExperts, KimiLatentMoEBlock
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params, load_params_into_nnx
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference
from tests.utils.test_helpers import get_test_config_path


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

  def _make_config(self, **overrides):
    """A real kimi-k3 HyperParameters at this test's toy MoE dims (RoutedMoE needs one)."""
    kwargs = {
        "override_model_config": True,
        "base_emb_dim": self.hidden_size,
        "base_num_decoder_layers": 2,
        "full_attn_layers": [1],
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "head_dim": 16,
        "vocab_size": 256,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "base_mlp_dim": 128,
        "base_moe_mlp_dim": self.moe_intermediate_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "shared_intermediate_size": self.shared_intermediate_size,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.top_k,
        "shared_experts": self.num_shared_experts,
        "num_shared_experts": self.num_shared_experts,
        "routed_scaling_factor": 1.0,
        "normalization_layer_epsilon": self.rms_norm_eps,
        # Experts on the latent, router on the hidden state. Megablox is a TPU Pallas kernel;
        # on CPU the sparse path uses `jax.lax.ragged_dot`.
        "moe_expert_input_dim": self.routed_expert_hidden_size,
        "moe_router_input_dim": self.hidden_size,
        "megablox": False,
        "max_target_length": self.seq_len,
        "max_position_embeddings": self.seq_len,
        "original_max_position_embeddings": self.seq_len,
        "per_device_batch_size": self.batch_size,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "scan_layers": False,
        "enable_checkpointing": False,
    }
    kwargs.update(overrides)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **kwargs)

  def _make_block(self, **config_overrides):
    """A KimiLatentMoEBlock built from a real toy kimi-k3 config (bf16 => moe.RoutedMoE)."""
    cfg = self._make_config(**config_overrides)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    block = KimiLatentMoEBlock(
        hidden_size=self.hidden_size,
        num_experts=self.num_experts,
        top_k=self.top_k,
        routed_expert_hidden_size=self.routed_expert_hidden_size,
        moe_intermediate_size=self.moe_intermediate_size,
        num_shared_experts=self.num_shared_experts,
        shared_intermediate_size=self.shared_intermediate_size,
        routed_scaling_factor=1.0,
        rms_norm_eps=self.rms_norm_eps,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        mesh=mesh,
        routed_experts_weight_format=cfg.routed_experts_weight_format,
        config=cfg,
        rngs=nnx.Rngs(0),
    )
    return block, cfg, mesh

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
    """Router and routed-expert params carry logical sharding axes (else they replicate per chip).

    MXFP4 packed codes / scales leave the quantized contraction axis (-2) unsharded.
    """

    def sharding_of(module):
      return {
          "/".join(map(str, path)): var.get_metadata().get("out_sharding")
          for path, var in nnx.state(module, nnx.Param).flat_state()
      }

    with self.subTest(weight_format="bf16"):
      block, _, _ = self._make_block()
      self.assertIsInstance(block.routed_experts, moe.RoutedMoE)
      self.assertEqual(
          sharding_of(block.routed_experts),
          {
              "gate/kernel": ("embed", None),
              "gate/bias": (None,),
              "wi_0": ("exp", "embed_moe", "mlp_moe"),
              "wi_1": ("exp", "embed_moe", "mlp_moe"),
              "wo": ("exp", "mlp_moe", "embed_moe"),
          },
      )
      # Router reads the full hidden state; experts run on the latent.
      self.assertEqual(block.routed_experts.gate.kernel.shape, (self.hidden_size, self.num_experts))
      self.assertEqual(
          block.routed_experts.wi_0.shape, (self.num_experts, self.routed_expert_hidden_size, self.moe_intermediate_size)
      )

    with self.subTest(weight_format="mxfp4"):
      experts = KimiRoutedExperts(
          num_experts=4,
          in_features=64,
          intermediate_dim=32,
          hidden_size=self.hidden_size,
          top_k=2,
          weight_format="mxfp4",
          rngs=nnx.Rngs(0),
      )
      self.assertEqual(
          sharding_of(experts),
          {
              "gate/kernel": ("embed", None),
              "gate/e_score_correction_bias": (None,),
              **{f"{n}_{s}": ("exp", None, "mlp_moe") for n in ("wi_0", "wi_1") for s in ("packed", "scale")},
              **{f"wo_{s}": ("exp", None, "embed_moe") for s in ("packed", "scale")},
          },
      )

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

    x_torch = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_moe(x_torch).numpy()

    x_jax = jnp.array(x_torch.numpy())
    # Both RoutedMoE dispatch paths: sorted/grouped (`ragged_dot` on CPU) and dense einsum.
    for dispatch, overrides in (("sparse_ragged_dot", {"sparse_matmul": True}), ("dense", {"sparse_matmul": False})):
      with self.subTest(dispatch=dispatch):
        jax_moe, _, mesh = self._make_block(**overrides)
        # Strict: every module param gets a converted weight and every converted key is used,
        # so this also checks the HF -> MaxText mapping (incl. the router `gate/bias`).
        load_params_into_nnx(jax_moe, converted_params, "params-decoder-layers_1-mlp")
        with mesh:
          y_jax = np.array(jax_moe(x_jax))
        np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
