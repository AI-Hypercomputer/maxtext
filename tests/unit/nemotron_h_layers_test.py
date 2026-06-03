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

"""Tests for Nemotron-H layers comparing MaxText vs PyTorch."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch

from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHMamba2Mixer as PtMamba,
    NemotronHMoE as PtMoE,
)
from transformers.models.zamba2.modeling_zamba2 import Zamba2RMSNormGated as PtNorm

from maxtext.models import nemotron_h as jax_nemotron
from maxtext.common.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_linear_weights,
    create_random_jax_torch,
)

class DummyConfig:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

def copy_zamba2_rms_norm_gated_weights(torch_norm, jax_norm):
  jax_norm.scale.value = jnp.array(torch_norm.weight.detach().cpu().numpy())

def copy_mamba2_mixer_weights(torch_mamba, jax_mamba):
  jax_mamba.in_proj.kernel.value = jnp.array(torch_mamba.in_proj.weight.detach().cpu().numpy().T)
  if torch_mamba.in_proj.bias is not None and jax_mamba.in_proj.bias is not None:
    jax_mamba.in_proj.bias.value = jnp.array(torch_mamba.in_proj.bias.detach().cpu().numpy())

  torch_conv_weight = torch_mamba.conv1d.weight.detach().cpu().numpy()
  # Pt: (dim, 1, kernel) -> Jax: (kernel, 1, dim)
  jax_conv_weight = np.transpose(torch_conv_weight, (2, 1, 0))
  jax_mamba.conv1d.conv.kernel.value = jnp.array(jax_conv_weight)
  if torch_mamba.conv1d.bias is not None and jax_mamba.conv1d.conv.bias is not None:
    jax_mamba.conv1d.conv.bias.value = jnp.array(torch_mamba.conv1d.bias.detach().cpu().numpy())

  jax_mamba.dt_bias.value = jnp.array(torch_mamba.dt_bias.detach().cpu().numpy())
  jax_mamba.A_log.value = jnp.array(torch_mamba.A_log.detach().cpu().numpy())
  copy_zamba2_rms_norm_gated_weights(torch_mamba.norm, jax_mamba.norm)
  jax_mamba.D.value = jnp.array(torch_mamba.D.detach().cpu().numpy())

  jax_mamba.out_proj.kernel.value = jnp.array(torch_mamba.out_proj.weight.detach().cpu().numpy().T)
  if torch_mamba.out_proj.bias is not None and jax_mamba.out_proj.bias is not None:
    jax_mamba.out_proj.bias.value = jnp.array(torch_mamba.out_proj.bias.detach().cpu().numpy())

def copy_moe_weights(torch_moe, jax_moe):
  jax_moe.gate.weight.value = jnp.array(torch_moe.gate.weight.detach().cpu().numpy())
  jax_moe.gate.e_score_correction_bias.value = jnp.array(torch_moe.gate.e_score_correction_bias.detach().cpu().numpy())

  jax_moe.experts.up_proj.value = jnp.array(torch_moe.experts.up_proj.detach().cpu().numpy())
  jax_moe.experts.down_proj.value = jnp.array(torch_moe.experts.down_proj.detach().cpu().numpy())

  copy_linear_weights(torch_moe.shared_experts.up_proj, jax_moe.shared_experts.up_proj)
  copy_linear_weights(torch_moe.shared_experts.down_proj, jax_moe.shared_experts.down_proj)

  if jax_moe.use_latent_proj:
    copy_linear_weights(torch_moe.fc1_latent_proj, jax_moe.fc1_latent_proj)
    copy_linear_weights(torch_moe.fc2_latent_proj, jax_moe.fc2_latent_proj)


class NemotronHLayersTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_default_matmul_precision", "highest")
    self.mesh = Mesh(jax.devices(), "data")
    torch.set_grad_enabled(False)

  def test_zamba2_rms_norm_gated(self):
    hidden_size = 64
    group_size = 16
    eps = 1e-5
    batch_size = 2
    seq_len = 8

    # Init layers
    rngs = nnx.Rngs(0)
    jax_layer = jax_nemotron.Zamba2RMSNormGated(
        num_features=hidden_size,
        group_size=group_size,
        epsilon=eps,
        rngs=rngs,
    )

    pt_layer = PtNorm(
        hidden_size=hidden_size,
        group_size=group_size,
        eps=eps,
    )

    # Randomize weights
    pt_layer.weight.data.normal_(mean=1.0, std=0.1)
    copy_zamba2_rms_norm_gated_weights(pt_layer, jax_layer)

    # Inputs
    x_jax, x_pt = create_random_jax_torch(batch_size, seq_len, hidden_size)
    gate_jax, gate_pt = create_random_jax_torch(batch_size, seq_len, hidden_size)

    # Run
    y_pt = pt_layer(x_pt, gate_pt)
    y_jax = jax_layer(x_jax, gate_jax)

    assert_all_close_jax_torch(y_jax, y_pt, rtol=1e-5, atol=1e-5)

    # Test without gate
    y_pt_nogate = pt_layer(x_pt)
    y_jax_nogate = jax_layer(x_jax)
    assert_all_close_jax_torch(y_jax_nogate, y_pt_nogate, rtol=1e-5, atol=1e-5)

  def test_mamba2_mixer_prefill(self):
    hidden_size = 64
    ssm_state_size = 16
    conv_kernel = 4
    n_groups = 2
    mamba_head_dim = 8
    mamba_num_heads = 8
    chunk_size = 16
    batch_size = 2
    seq_len = 32

    # Pt Config
    pt_config = NemotronHConfig(
        hidden_size=hidden_size,
        ssm_state_size=ssm_state_size,
        conv_kernel=conv_kernel,
        n_groups=n_groups,
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        chunk_size=chunk_size,
        use_bias=False,
        use_conv_bias=False,
        layer_norm_epsilon=1e-5,
        mamba_hidden_act="silu",
        use_mamba_kernels=False,
    )

    # JAX Config
    jax_cfg = DummyConfig(
        emb_dim=hidden_size,
        ssm_state_size=ssm_state_size,
        conv_kernel=conv_kernel,
        n_groups=n_groups,
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        mamba_chunk_size=chunk_size,
        use_bias=False,
        use_conv_bias=False,
        normalization_layer_epsilon=1e-5,
        per_device_batch_size=batch_size,
        time_step_min=0.001,
        time_step_max=float('inf'),
    )

    rngs = nnx.Rngs(0)
    jax_layer = jax_nemotron.NemotronHMamba2Mixer(
        config=jax_cfg,
        model_mode=MODEL_MODE_TRAIN,
        dtype=jnp.float32,
        rngs=rngs,
    )

    pt_layer = PtMamba(pt_config)

    # Randomize weights
    for p in pt_layer.parameters():
      p.data.normal_(std=0.1)
    pt_layer.A_log.data.uniform_(-1.0, 1.0)
    pt_layer.dt_bias.data.uniform_(-1.0, 1.0)
    pt_layer.D.data.uniform_(-1.0, 1.0)

    print("DEBUG: pt_config.ssm_state_size:", pt_config.ssm_state_size)
    print("DEBUG: pt_layer.ssm_state_size:", pt_layer.ssm_state_size)
    print("DEBUG: pt_layer.conv_dim:", pt_layer.conv_dim)
    print("DEBUG: pt_layer.intermediate_size:", pt_layer.intermediate_size)
    print("DEBUG: pt_layer.conv1d:", pt_layer.conv1d)
    print("DEBUG: jax_cfg.ssm_state_size:", jax_cfg.ssm_state_size)
    print("DEBUG: jax_layer.ssm_state_size:", jax_layer.ssm_state_size)
    print("DEBUG: jax_layer.conv_dim:", jax_layer.conv_dim)

    copy_mamba2_mixer_weights(pt_layer, jax_layer)

    # Inputs
    x_jax, x_pt = create_random_jax_torch(batch_size, seq_len, hidden_size)

    # Run
    y_pt = pt_layer(x_pt)
    y_jax = jax_layer(x_jax)

    assert_all_close_jax_torch(y_jax, y_pt, rtol=1e-3, atol=1e-3)

  def test_mamba2_mixer_decode(self):
    hidden_size = 64
    ssm_state_size = 16
    conv_kernel = 4
    n_groups = 2
    mamba_head_dim = 8
    mamba_num_heads = 8
    chunk_size = 16
    batch_size = 2
    prefill_len = 32

    # Pt Config
    pt_config = NemotronHConfig(
        hidden_size=hidden_size,
        ssm_state_size=ssm_state_size,
        conv_kernel=conv_kernel,
        n_groups=n_groups,
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        chunk_size=chunk_size,
        use_bias=False,
        use_conv_bias=False,
        layer_norm_epsilon=1e-5,
        mamba_hidden_act="silu",
        use_mamba_kernels=False,
    )

    # JAX Config
    jax_cfg = DummyConfig(
        emb_dim=hidden_size,
        ssm_state_size=ssm_state_size,
        conv_kernel=conv_kernel,
        n_groups=n_groups,
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        mamba_chunk_size=chunk_size,
        use_bias=False,
        use_conv_bias=False,
        normalization_layer_epsilon=1e-5,
        per_device_batch_size=batch_size,
        micro_batch_size_to_train_on=batch_size,
        time_step_min=0.001,
        time_step_max=float('inf'),
    )

    rngs = nnx.Rngs(0)
    jax_layer = jax_nemotron.NemotronHMamba2Mixer(
        config=jax_cfg,
        model_mode=MODEL_MODE_PREFILL,
        dtype=jnp.float32,
        rngs=rngs,
    )

    pt_layer = PtMamba(pt_config)

    # Randomize weights and copy
    for p in pt_layer.parameters():
      p.data.normal_(std=0.1)
    pt_layer.A_log.data.uniform_(-1.0, 1.0)
    pt_layer.dt_bias.data.uniform_(-1.0, 1.0)
    pt_layer.D.data.uniform_(-1.0, 1.0)
    copy_mamba2_mixer_weights(pt_layer, jax_layer)

    # Prefill Inputs
    x_jax, x_pt = create_random_jax_torch(batch_size, prefill_len, hidden_size)

    # Init PyTorch Cache
    from transformers import DynamicCache
    pt_cache = DynamicCache(config=pt_config)

    # Run Prefill on PyTorch
    pt_layer.layer_idx = 0
    y_pt_prefill = pt_layer(x_pt, cache_params=pt_cache)

    # Run Prefill on JAX
    y_jax_prefill = jax_layer(x_jax)

    # Compare prefill outputs
    assert_all_close_jax_torch(y_jax_prefill, y_pt_prefill, rtol=5e-3, atol=5e-3)

    # Compare caches after prefill
    pt_conv_state = pt_cache.layers[0].conv_states
    pt_recurrent_state = pt_cache.layers[0].recurrent_states

    jax_conv_state = jax_layer.cache.conv_state[...]
    jax_recurrent_state = jax_layer.cache.recurrent_state[...]

    assert_all_close_jax_torch(jax_conv_state, pt_conv_state[:, :, 1:], rtol=1e-5, atol=1e-5)
    assert_all_close_jax_torch(jax_recurrent_state, pt_recurrent_state, rtol=1e-2, atol=1e-2)

    jax_layer.model_mode = MODEL_MODE_AUTOREGRESSIVE

    # Decode Inputs
    x_jax_dec, x_pt_dec = create_random_jax_torch(batch_size, 1, hidden_size)

    # Run Decode Pt
    y_pt_dec = pt_layer(x_pt_dec, cache_params=pt_cache)

    # Run Decode JAX
    y_jax_dec = jax_layer(x_jax_dec)

    # Compare decode outputs
    assert_all_close_jax_torch(y_jax_dec, y_pt_dec, rtol=1e-2, atol=1e-2)

    # Compare caches after decode
    jax_conv_state_post = jax_layer.cache.conv_state[...]
    jax_recurrent_state_post = jax_layer.cache.recurrent_state[...]

    pt_conv_state_post = pt_cache.layers[0].conv_states
    pt_recurrent_state_post = pt_cache.layers[0].recurrent_states

    assert_all_close_jax_torch(jax_conv_state_post, pt_conv_state_post[:, :, 1:], rtol=1e-5, atol=1e-5)
    assert_all_close_jax_torch(jax_recurrent_state_post, pt_recurrent_state_post, rtol=1e-2, atol=1e-2)

  def test_moe_no_latent(self):
    self._run_moe_test(use_latent=False)

  def test_moe_with_latent(self):
    self._run_moe_test(use_latent=True)

  def _run_moe_test(self, use_latent):
    hidden_size = 64
    n_routed_experts = 4
    moe_intermediate_size = 32
    moe_shared_expert_intermediate_size = 16
    n_group = 2
    topk_group = 1
    norm_topk_prob = True
    routed_scaling_factor = 1.0
    num_experts_per_tok = 2
    moe_latent_size = 32 if use_latent else None
    mlp_bias = False
    batch_size = 2
    seq_len = 8

    # Pt Config
    pt_config = NemotronHConfig(
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        moe_intermediate_size=moe_intermediate_size,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        num_experts_per_tok=num_experts_per_tok,
        moe_latent_size=moe_latent_size,
        mlp_bias=mlp_bias,
        mlp_hidden_act="relu2",
    )

    # JAX Config
    jax_cfg = DummyConfig(
        emb_dim=hidden_size,
        num_experts=n_routed_experts,
        moe_mlp_dim=moe_intermediate_size,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        n_routing_groups=n_group,
        topk_routing_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        num_experts_per_tok=num_experts_per_tok,
        moe_latent_size=moe_latent_size,
        mlp_bias=mlp_bias,
    )

    rngs = nnx.Rngs(0)
    jax_layer = jax_nemotron.NemotronHMoE(
        config=jax_cfg,
        rngs=rngs,
    )

    pt_layer = PtMoE(pt_config)

    # Randomize weights
    for p in pt_layer.parameters():
      p.data.normal_(std=0.1)

    copy_moe_weights(pt_layer, jax_layer)

    # Inputs
    x_jax, x_pt = create_random_jax_torch(batch_size, seq_len, hidden_size)

    # Run
    y_pt = pt_layer(x_pt)
    y_jax = jax_layer(x_jax)

    assert_all_close_jax_torch(y_jax, y_pt, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  unittest.main()
