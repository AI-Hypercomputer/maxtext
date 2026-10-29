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

"""Tests for the Kimi-K3 latent MoE block and SiTU dense MLP (`layers/latent_moe.py`).

The sharding/shape/guard tests run everywhere. The parity tests compare against Moonshot's PyTorch
modeling code and skip unless torch, transformers and einops are installed and the HF reference is
checked out at `kimi-k3-hf-reference/` (see `tests/utils/kimi_k3_reference.py`).
"""

import contextlib
import importlib
import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.latent_moe import KimiDenseMLP, KimiLatentMoEBlock
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, reference_available
from tests.utils.test_helpers import get_test_config_path

try:
  import einops  # pylint: disable=unused-import
  import torch
  import transformers  # pylint: disable=unused-import

  from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params, load_params_into_nnx

  _HAVE_TORCH_DEPS = True
except ImportError:
  _HAVE_TORCH_DEPS = False

requires_reference = unittest.skipUnless(
    _HAVE_TORCH_DEPS and reference_available(),
    "needs torch, transformers, einops and the Kimi-K3 HF reference checkout at " + KIMI_K3_REFERENCE_DIR,
)

_REFERENCE_MODULES = ("modeling_kimi_linear", "configuration_kimi_k3")


@contextlib.contextmanager
def _hf_reference():
  """Imports the HF reference modules, then removes them from `sys.modules`/`sys.path` again.

  The reference has its own fallbacks for the optional `fla` kernels and transformers helpers,
  so no stubs are needed.
  """
  saved_path = list(sys.path)
  preexisting = {name: sys.modules[name] for name in _REFERENCE_MODULES if name in sys.modules}
  sys.path.insert(0, KIMI_K3_REFERENCE_DIR)
  try:
    yield importlib.import_module("modeling_kimi_linear"), importlib.import_module("configuration_kimi_k3")
  finally:
    sys.path[:] = saved_path
    for name in _REFERENCE_MODULES:
      sys.modules.pop(name, None)
    sys.modules.update(preexisting)


class KimiK3LatentMoETest(unittest.TestCase):
  """Tests validating KimiDenseMLP and KimiLatentMoEBlock."""

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
    # Non-trivial so the DeepSeek-V3 `routed_scaling_factor` path is exercised.
    self.routed_scaling_factor = 2.5

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
        "routed_scaling_factor": self.routed_scaling_factor,
        "normalization_layer_epsilon": self.rms_norm_eps,
        "mlp_activations": ["situ_gate", "situ_linear"],
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
    """A KimiLatentMoEBlock built from a real toy kimi-k3 config."""
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
        routed_scaling_factor=self.routed_scaling_factor,
        rms_norm_eps=self.rms_norm_eps,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        mesh=mesh,
        config=cfg,
        rngs=nnx.Rngs(0),
    )
    return block, cfg, mesh

  def test_routed_expert_params_have_sharding_axes(self):
    """Router and routed-expert params carry logical sharding axes (else they replicate per chip)."""

    def sharding_of(module, var_type):
      return {
          "/".join(map(str, path)): var.get_metadata().get("out_sharding")
          for path, var in nnx.state(module, var_type).flat_state()
      }

    block, _, _ = self._make_block()
    self.assertIsInstance(block.routed_experts, moe.RoutedMoE)
    self.assertEqual(
        sharding_of(block.routed_experts, nnx.Param),
        {
            "gate/kernel": ("embed_router", None),
            "wi_0": ("exp", "embed_moe", "mlp_moe"),
            "wi_1": ("exp", "embed_moe", "mlp_moe"),
            "wo": ("exp", "mlp_moe", "embed_moe"),
        },
    )
    # The router bias is a `MoEBiasVar` (as for DeepSeek-V3 / Kimi-K2): same `gate/bias` path, but outside
    # `nnx.Param`, so the optimizer (and AdamW weight decay) never touches it.
    self.assertIsInstance(block.routed_experts.gate.bias, moe.MoEBiasVar)
    self.assertEqual(sharding_of(block.routed_experts, moe.MoEBiasVar), {"gate/bias": (None,)})
    # Router reads the full hidden state; experts run on the latent.
    self.assertEqual(block.routed_experts.gate.kernel.shape, (self.hidden_size, self.num_experts))
    self.assertEqual(
        block.routed_experts.wi_0.shape, (self.num_experts, self.routed_expert_hidden_size, self.moe_intermediate_size)
    )

  def test_rejects_load_balance_loss_it_would_drop(self):
    """The block returns only the MoE output, so a config that needs its aux loss fails fast."""
    with self.assertRaisesRegex(NotImplementedError, "load-balance loss"):
      self._make_block(load_balance_loss_weight=0.01)

  def test_forwards_out_sharding(self):
    """`out_sharding` reaches both `routed_expert_up_proj` and `shared_expert.wo`."""
    block, _, mesh = self._make_block()
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
    dense_cls = type(block.routed_expert_up_proj)
    orig_call = dense_cls.__call__
    seen = {}

    def spy_call(mod, x, *args, **kwargs):
      if mod is block.routed_expert_up_proj:
        seen["up"] = kwargs.get("out_sharding")
      elif mod is block.shared_expert.wo:
        seen["shared_wo"] = kwargs.get("out_sharding")
      return orig_call(mod, x, *args, **kwargs)

    x = jnp.ones((self.batch_size, self.seq_len, self.hidden_size), dtype=jnp.float32)
    with mesh, mock.patch.object(dense_cls, "__call__", spy_call):
      _ = block(x, out_sharding=sharding)
    self.assertEqual(seen, {"up": sharding, "shared_wo": sharding})

  def _load_block(self, block, converted, prefix):
    """Strictly loads `converted` into `block`, including the `MoEBiasVar` router bias."""
    loaded = load_params_into_nnx(block, converted, prefix)
    self.assertIn(f"{prefix}-routed_experts-gate-bias", loaded)
    self.assertEqual(set(loaded), {k for k in converted if k.startswith(prefix + "-")})
    self.assertEqual(len(loaded), len(nnx.state(block, (nnx.Param, moe.MoEBiasVar)).flat_state()))

  @requires_reference
  def test_kimi_dense_mlp_parity(self):
    """Verifies Layer 0 Dense MLP parity against PyTorch KimiMLP with 4x expansion."""
    test_hidden_size = 128
    test_intermediate_size = 512
    seq_len = 16
    with _hf_reference() as (modeling_kimi_linear, configuration_kimi_k3):
      torch.manual_seed(42)
      pt_config = configuration_kimi_k3.KimiLinearConfig(
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
    prefix = "params-decoder-layers_0-mlp"
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_mlp.state_dict(), test_hf_config, maxtext_config=None, prefix_filter=prefix + "-"
    )

    jax_mlp = KimiDenseMLP(
        in_features=test_hidden_size,
        intermediate_dim=test_intermediate_size,
        matmul_precision="highest",
        rngs=nnx.Rngs(0),
    )
    # Strict: also checks the layer-0 HF -> MaxText mapping.
    load_params_into_nnx(jax_mlp, converted_params, prefix)

    x_torch = torch.randn(self.batch_size, seq_len, test_hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_mlp(x_torch).numpy()

    y_jax = np.array(jax_mlp(jnp.array(x_torch.numpy())))
    np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)

  @requires_reference
  def test_kimi_latent_moe_block_parity(self):
    """Verifies Latent MoE block parity against PyTorch KimiSparseMoeBlock."""
    with _hf_reference() as (modeling_kimi_linear, configuration_kimi_k3):
      torch.manual_seed(42)
      pt_config = configuration_kimi_k3.KimiLinearConfig(
          hidden_size=self.hidden_size,
          intermediate_size=128,
          num_experts=self.num_experts,
          num_experts_per_token=self.top_k,
          routed_expert_hidden_size=self.routed_expert_hidden_size,
          moe_intermediate_size=self.moe_intermediate_size,
          num_shared_experts=self.num_shared_experts,
          shared_intermediate_size=self.shared_intermediate_size,
          moe_renormalize=True,
          routed_scaling_factor=self.routed_scaling_factor,
          latent_moe_use_norm=True,
          rms_norm_eps=self.rms_norm_eps,
          hidden_act="situ",
          activation_situ_beta=4.0,
          activation_situ_linear_beta=25.0,
      )
      pt_moe = modeling_kimi_linear.KimiSparseMoeBlock(pt_config)
      pt_moe.eval()
    # A non-zero correction bias, so a mis-mapped or unloaded router bias changes expert selection.
    with torch.no_grad():
      pt_moe.gate.e_score_correction_bias.normal_(std=0.5)

    prefix = "params-decoder-layers_1-mlp"
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_moe.state_dict(), self.hf_config, maxtext_config=None, prefix_filter=prefix + "-"
    )

    x_torch = torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
    with torch.no_grad():
      y_torch = pt_moe(x_torch).numpy()

    x_jax = jnp.array(x_torch.numpy())
    # Both RoutedMoE dispatch paths: sorted/grouped (`ragged_dot` on CPU) and dense einsum.
    for dispatch, overrides in (("sparse_ragged_dot", {"sparse_matmul": True}), ("dense", {"sparse_matmul": False})):
      with self.subTest(dispatch=dispatch):
        jax_moe, _, mesh = self._make_block(**overrides)
        self._load_block(jax_moe, converted_params, prefix)
        with mesh, jax.default_matmul_precision("highest"):
          y_jax = np.array(jax_moe(x_jax))
        np.testing.assert_allclose(y_jax, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
