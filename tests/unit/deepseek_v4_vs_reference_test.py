# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-8.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests validating DeepSeek-V4 MaxText components against PyTorch references."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock pathwaysutils to avoid missing dependency errors on local CPU runs
sys.modules['pathwaysutils'] = MagicMock()
sys.modules['pathwaysutils.elastic'] = MagicMock()
sys.modules['pathwaysutils.experimental'] = MagicMock()

# Dynamic PYTHONPATH mapping for MaxText source imports
sys.path.insert(0, os.path.abspath("third_party/maxtext/src"))

# Force CPU execution for testing and clear polluted TPU cluster env vars
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.pop("TPU_WORKER_HOSTNAMES", None)
os.environ.pop("TPU_ACCELERATOR_TYPE", None)

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from flax import nnx

from maxtext.layers.embeddings import PartialRotaryEmbedding
from maxtext.layers.linears import DeepSeekV4GroupedLinear
from maxtext.layers.moe import RoutedMoE
from maxtext.common import common_types as ctypes
from maxtext.common.common_types import ShardMode
from maxtext.layers.attention_op import AttentionOp
from maxtext.common.common_types import AttentionType, DEFAULT_MASK_VALUE
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding as MTRope
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# Import PyTorch references locally
import sys
import os
sys.path.insert(0, os.path.abspath("third_party/deepseek_v4"))
from modeling_deepseek_v4 import DeepseekV4Attention
from modeling_deepseek_v4 import DeepseekV4RotaryEmbedding as PTRope
from modeling_deepseek_v4 import apply_rotary_pos_emb


# ==============================================================================
# 1. Embedded PyTorch Reference Classes (Bypassing HF relative import bugs)
# ==============================================================================

class DeepseekV4Config:
  """Mock configuration containing keys needed by PyTorch reference classes."""
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    # Ensure default properties for backward compatibility
    if not hasattr(self, "_attn_implementation"):
      self._attn_implementation = "eager"
    if not hasattr(self, "rms_norm_eps"):
      self.rms_norm_eps = 1e-6

class DeepseekV4RotaryEmbedding_PT(nn.Module):
  """PyTorch reference Rotary Embedding class."""
  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
    for layer_type in self.layer_types:
      base = config.rope_parameters[layer_type]["rope_theta"]
      partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
      head_dim = config.hidden_size // config.num_attention_heads
      dim = int(head_dim * partial_rotary_factor)
      inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
      self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

  def forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half_PT(x):
  """Interleaved rotate half in PyTorch."""
  x1 = x[..., 0::2]
  x2 = x[..., 1::2]
  return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_pos_emb_PT(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
  """PyTorch reference apply rotary pos emb for trailing slice."""
  cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = ((rope.float() * cos) + (rotate_half_PT(rope).float() * sin)).to(x.dtype)
  return torch.cat([nope, rotated], dim=-1)

class DeepseekV4GroupedLinear_PT(nn.Linear):
  """PyTorch reference Grouped Linear class."""
  def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
    super().__init__(in_features_per_group, out_features, bias=bias)
    self.n_groups = n_groups

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)


def sqrt_softplus_pt(x):
  return torch.sqrt(torch.nn.functional.softplus(x))


class DeepseekV4TopKRouter_PT(nn.Module):
  def __init__(self, vocab_size=100, num_experts=16, num_experts_per_tok=4, hidden_size=128, routed_scaling_factor=1.0):
    super().__init__()
    self.top_k = num_experts_per_tok
    self.num_experts = num_experts
    self.hidden_dim = hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    nn.init.orthogonal_(self.weight)
    self.routed_scaling_factor = routed_scaling_factor
    self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

  def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = torch.nn.functional.linear(flat, self.weight)
    scores = sqrt_softplus_pt(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter_PT(nn.Module):
  def __init__(self, vocab_size=100, num_experts=16, num_experts_per_tok=4, hidden_size=128, routed_scaling_factor=1.0):
    super().__init__()
    self.top_k = num_experts_per_tok
    self.num_experts = num_experts
    self.hidden_dim = hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    nn.init.orthogonal_(self.weight)
    self.routed_scaling_factor = routed_scaling_factor
    self.register_buffer("tid2eid", torch.zeros(vocab_size, self.top_k, dtype=torch.long), persistent=True)

  def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = torch.nn.functional.linear(flat, self.weight)
    scores = sqrt_softplus_pt(logits)
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


class MoeConfig:
  """Mock Config class for MoE tests."""
  def __init__(self):
    self.vocab_size = 100
    self.mlp_activations = ["silu", "linear"]
    self.padded_base_moe_mlp_dim = None
    self.debug_sharding = False
    self.using_pipeline_parallelism = False
    self.logical_axis_rules = None
    self.mlp_bias = False
    self.sparse_matmul = False
    self.model_name = "deepseek3"
    self.float32_gate_logits = True
    self.routed_bias = False
    self.routed_score_func = "sqrtsoftplus"
    self.matmul_precision = "default"
    self.shard_mode = ShardMode.AUTO
    self.moe_expert_input_dim = 128
    self.shard_exp_on_fsdp = False
    self.use_batch_split_schedule = False
    self.attention = "default"
    self.enable_dp_attention = False
    self.custom_mesh_and_rule = None
    self.load_balance_loss_weight = 0.0
    self.routed_bias_update_rate = 0.0
    self.capacity_factor = 0.0
    self.emb_dim = 128
    self.prefuse_moe_weights = False
    self.routed_scaling_factor = 1.0
    self.decoder_block = ctypes.DecoderBlockType.DEEPSEEK
    self.norm_topk_prob = True
    self.use_random_routing = False
    self.n_routing_groups = -1
    self.topk_routing_group = 1
    self.routed_bias_update_rate = 0.0
    self.model_call_mode = "training"
    self.fuse_expert_scales = False
    self.wi_tile_fwd_batch_seq = 1
    self.use_tokamax_gmm = False
    self.megablox = False
    self.use_ring_of_experts = False
    self.ragged_buffer_factor = 1.0
    self.use_ragged_sort = False
    self.use_custom_sort_vjp = False

# ==============================================================================
# Tests
# ==============================================================================


class DeepSeekV4RotaryEmbeddingTest(unittest.TestCase):
  """Tests to validate MaxText RoPE implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 16
    self.head_dim = 128
    self.num_heads = 4
    self.main_rope_theta = 10000.0
    self.compress_rope_theta = 160000.0
    self.partial_rotary_factor = 64.0 / 128.0

    # Build a mock mesh
    devices = jax.devices()
    self.mesh = jax.sharding.Mesh(np.array(devices).reshape(1, len(devices)), ("x", "y"))

    self.config = DeepseekV4Config(
        hidden_size=self.num_heads * self.head_dim,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        rope_theta=self.main_rope_theta,
        rope_parameters={
            "main": {
                "rope_type": "default",
                "rope_theta": self.main_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "compress": {
                "rope_type": "default",
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
        },
    )

  def test_rotary_embedding_main(self):
    self._run_rotary_test(layer_type="main", expected_theta=self.main_rope_theta)

  def test_rotary_embedding_compress(self):
    self._run_rotary_test(layer_type="compress", expected_theta=self.compress_rope_theta)

  def _run_rotary_test(self, layer_type, expected_theta):
    """
    Validates that the MaxText RoPE implementation is mathematically identical to
    the PyTorch reference up to 1e-5 tolerance.
    """
    # --------------------------------------------------------------------------
    # 1. Initialization
    # --------------------------------------------------------------------------
    ref_rope = DeepseekV4RotaryEmbedding_PT(self.config)
    
    # Initialize the newly updated PartialRotaryEmbedding with interleaved=True and trailing=True
    mt_rope = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=int(expected_theta),
        mesh=self.mesh,
        embedding_dims=self.head_dim,
        partial_rotary_factor=self.partial_rotary_factor,
        interleaved=True,
        trailing=True,
        cast_as_fprop_dtype=False,  # Keep in float32 for exact parity check
    )

    # --------------------------------------------------------------------------
    # 2. Input Generation
    # --------------------------------------------------------------------------
    np.random.seed(42)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)
    position_ids_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)

    x_pt = torch.tensor(x_np)
    position_ids_pt = torch.tensor(position_ids_np, dtype=torch.long)

    x_mt = jnp.array(x_np)
    position_ids_mt = jnp.array(position_ids_np)

    # --------------------------------------------------------------------------
    # 3. Apply PyTorch Reference Rotation
    # --------------------------------------------------------------------------
    # PyTorch reference expects flattened hidden dim for frequency calculation
    ref_cos, ref_sin = ref_rope(
        x_pt.view(self.batch_size, self.seq_len, -1), position_ids=position_ids_pt, layer_type=layer_type
    )

    # PyTorch reference apply_rotary_pos_emb expects head dimension to be before sequence length:
    # Expected PyTorch Shape: [Batch, NumHeads, SeqLen, HeadDim]
    x_pt_transpose = x_pt.transpose(1, 2)
    ref_rotated = apply_rotary_pos_emb_PT(x_pt_transpose, ref_cos, ref_sin)
    ref_rotated_np = ref_rotated.transpose(1, 2).numpy()

    # --------------------------------------------------------------------------
    # 4. Apply MaxText Rotary Rotation
    # --------------------------------------------------------------------------
    # MaxText PartialRotaryEmbedding operates natively on [B, S, H, D]
    mt_rotated = mt_rope(x_mt, position_ids_mt)
    mt_rotated_np = np.array(mt_rotated)

    # --------------------------------------------------------------------------
    # 5. Final Validation
    # --------------------------------------------------------------------------
    np.testing.assert_allclose(mt_rotated_np, ref_rotated_np, rtol=1e-5, atol=1e-5)
    print(f"Rotary Embedding test ({layer_type}) passed successfully with 100% parity!")


class DeepSeekV4GroupedLinearTest(unittest.TestCase):
  """Tests to validate MaxText GroupedLinear implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 8
    self.n_groups = 4
    self.in_features_per_group = 128
    self.out_features = 256  # 64 per group

    self.rngs = nnx.Rngs(0)

  def test_grouped_linear_forward(self):
    """
    Validates that the MaxText GroupedLinear projection is mathematically identical to
    the PyTorch bmm logic up to 1e-5 tolerance.
    """
    # --------------------------------------------------------------------------
    # 1. Initialization
    # --------------------------------------------------------------------------
    ref_linear = DeepseekV4GroupedLinear_PT(
        in_features_per_group=self.in_features_per_group,
        out_features=self.out_features,
        n_groups=self.n_groups,
        bias=False,
    )

    # --------------------------------------------------------------------------
    # 2. Extract and Reshape Weights
    # --------------------------------------------------------------------------
    pt_weight = ref_linear.weight.data.numpy()
    out_features_per_group = self.out_features // self.n_groups
    mt_weight_np = pt_weight.reshape(self.n_groups, out_features_per_group, self.in_features_per_group).transpose(0, 2, 1)

    # --------------------------------------------------------------------------
    # 3. Initialize MaxText Implementation
    # --------------------------------------------------------------------------
    mt_linear = DeepSeekV4GroupedLinear(
        in_features_per_group=self.in_features_per_group,
        out_features=self.out_features,
        n_groups=self.n_groups,
        rngs=self.rngs,
    )
    mt_linear.kernel[...] = jnp.array(mt_weight_np)

    # --------------------------------------------------------------------------
    # 4. Input Generation
    # --------------------------------------------------------------------------
    np.random.seed(42)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.n_groups, self.in_features_per_group)).astype(
        np.float32
    )

    x_pt = torch.tensor(x_np)
    x_mt = jnp.array(x_np)

    # --------------------------------------------------------------------------
    # 5. Execute Forward Pass
    # --------------------------------------------------------------------------
    ref_out = ref_linear(x_pt)
    mt_out = mt_linear(x_mt)

    # --------------------------------------------------------------------------
    # 6. Final Validation
    # --------------------------------------------------------------------------
    np.testing.assert_allclose(np.array(mt_out), ref_out.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Grouped Linear test passed successfully with 100% parity!")


class DeepSeekV4MoeRoutingTest(unittest.TestCase):
  """Tests to validate MaxText MoE routing (TopK learned and Hash static) against PyTorch."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 8
    self.hidden_size = 128
    self.num_experts = 16
    self.num_experts_per_tok = 4
    self.vocab_size = 100
    
    # Mesh setup for JAX
    self.mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    self.rngs = nnx.Rngs(42)

  def test_learned_moe_routing(self):
    # 1. Initialize PyTorch Reference Router
    ref_router = DeepseekV4TopKRouter_PT(
        vocab_size=self.vocab_size,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        hidden_size=self.hidden_size,
    )
    
    # 2. Extract weights and convert to JAX
    pt_weight = ref_router.weight.data.numpy()
    
    # 3. Initialize MaxText implementation
    cfg = MoeConfig()
    cfg.vocab_size = self.vocab_size
    cfg.emb_dim = self.hidden_size
    cfg.moe_expert_input_dim = self.hidden_size
    
    # GateLogit in JAX
    from maxtext.layers.moe import GateLogit
    mt_gate = GateLogit(
        in_features_shape=self.hidden_size,
        out_features_shape=self.num_experts,
        model_name="deepseek3",
        mesh=self.mesh,
        rngs=self.rngs,
        score_func="sqrtsoftplus",
    )
    mt_gate.kernel[...] = jnp.array(pt_weight.T)

    # 4. Generate random inputs
    np.random.seed(42)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    
    x_pt = torch.tensor(x_np)
    x_mt = jnp.array(x_np)

    # 5. Run forward passes
    ref_logits, ref_weights, ref_indices = ref_router(x_pt)
    
    mt_gate_out, mt_pre_bias = mt_gate(x_mt)
    mt_moe = RoutedMoE(
        config=cfg,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=lambda rng, shape, dtype, *args, **kwargs: jnp.zeros(shape, dtype=dtype),
        kernel_axes=(),
        rngs=self.rngs,
        intermediate_dim=256,
        use_hash_routing=False,
    )
    mt_moe.gate.kernel[...] = jnp.array(pt_weight.T)
    
    mt_weights, mt_indices = mt_moe.get_topk(mt_gate_out, mt_pre_bias)

    # 6. Validate Parity
    ref_scores = sqrt_softplus_pt(ref_logits)
    np.testing.assert_allclose(np.array(mt_gate_out).reshape(-1, self.num_experts), ref_scores.detach().numpy(), rtol=1e-5, atol=1e-5)
    mt_indices_sorted = np.sort(np.array(mt_indices), axis=-1)
    ref_indices_sorted = np.sort(ref_indices.numpy().reshape(self.batch_size, self.seq_len, -1), axis=-1)
    np.testing.assert_array_equal(mt_indices_sorted, ref_indices_sorted)
    
    mt_weights_sorted = np.sort(np.array(mt_weights), axis=-1)
    ref_weights_sorted = np.sort(ref_weights.detach().numpy().reshape(self.batch_size, self.seq_len, -1), axis=-1)
    np.testing.assert_allclose(mt_weights_sorted, ref_weights_sorted, rtol=1e-5, atol=1e-5)
    print("MoE Learned Routing (TopK + SqrtSoftplus) test passed successfully with 100% parity!")

  def test_hash_moe_routing(self):
    # 1. Initialize PyTorch Reference Hash Router
    ref_router = DeepseekV4HashRouter_PT(
        vocab_size=self.vocab_size,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        hidden_size=self.hidden_size,
    )
    np.random.seed(42)
    tid2eid_np = np.random.randint(0, self.num_experts, size=(self.vocab_size, self.num_experts_per_tok)).astype(np.int32)
    ref_router.tid2eid.copy_(torch.tensor(tid2eid_np, dtype=torch.long))
    
    # 2. Extract weights
    pt_weight = ref_router.weight.data.numpy()
    
    # 3. Initialize MaxText implementation
    cfg = MoeConfig()
    cfg.vocab_size = self.vocab_size
    cfg.emb_dim = self.hidden_size
    cfg.moe_expert_input_dim = self.hidden_size
    
    mt_moe = RoutedMoE(
        config=cfg,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=lambda rng, shape, dtype, *args, **kwargs: jnp.zeros(shape, dtype=dtype),
        kernel_axes=(),
        rngs=self.rngs,
        intermediate_dim=256,
        use_hash_routing=True,
        tid2eid=jnp.array(tid2eid_np),
    )
    mt_moe.gate.kernel[...] = jnp.array(pt_weight.T)

    # 4. Generate random inputs
    input_ids_np = np.random.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len)).astype(np.int32)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    
    x_pt = torch.tensor(x_np)
    input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
    
    x_mt = jnp.array(x_np)
    input_ids_mt = jnp.array(input_ids_np)

    # 5. Run forward passes
    ref_logits, ref_weights, ref_indices = ref_router(x_pt, input_ids_pt)
    
    mt_gate_out, mt_pre_bias = mt_moe.gate(x_mt)
    mt_weights, mt_indices = mt_moe.get_topk(mt_gate_out, mt_pre_bias, input_ids=input_ids_mt)

    # 6. Validate Parity
    np.testing.assert_array_equal(np.array(mt_indices), ref_indices.numpy().reshape(self.batch_size, self.seq_len, -1))
    np.testing.assert_allclose(np.array(mt_weights), ref_weights.detach().numpy().reshape(self.batch_size, self.seq_len, -1), rtol=1e-5, atol=1e-5)
    print("MoE Static Hash-MoE Routing test passed successfully with 100% parity!")




class DeepSeekV4AttentionMaskingTest(unittest.TestCase):
  """Tests to validate AttentionOp masking logic for DeepSeek-V4 attention patterns."""

  def setUp(self):
    self.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="test",
        enable_checkpointing=False,
    )

  def test_generate_attention_mask_local_sliding(self):
    """Verifies AttentionType.LOCAL_SLIDING enforces both causal and sliding window constraints."""

    # Test with multiple heads and different sequence lengths
    for s_len in [1, 8, 128]:
      op = AttentionOp(
          config=self.config,
          num_query_heads=4,
          num_kv_heads=1,
          max_target_length=256,
          mesh=None,
          attention_kernel="dot_product",
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=3,
      )

      batch_size = 1
      q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
      k_dummy = jnp.zeros((batch_size, s_len, 1, 128))

      mask = op.generate_attention_mask(
          query=q_dummy,
          key=k_dummy,
          decoder_segment_ids=None,
          model_mode="train",
      )

      self.assertEqual(mask.shape, (1, 1, 1, s_len, s_len))
      mask_np = np.array(mask)[0, 0, 0]

      # Expected float mask for window_size=3
      # Row 0: [0.0, INF, INF, INF, INF, ...]
      # Row 1: [0.0, 0.0, INF, INF, INF, ...]
      # Row 2: [0.0, 0.0, 0.0, INF, INF, ...]
      # Row 3: [INF, 0.0, 0.0, 0.0, INF, ...]
      if s_len > 1:
        self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)  # strict causal
      self.assertEqual(mask_np[0, 0], 0.0)

      if s_len >= 4:
        self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)  # sliding window size=3
        self.assertEqual(mask_np[3, 1], 0.0)

  def test_generate_attention_mask_compressed(self):
    """Verifies AttentionType.COMPRESSED stitches sliding window and float compressed_mask."""

    batch_size = 1
    s_len = 8
    c_len = 2
    kv_len = s_len + c_len

    op = AttentionOp(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        max_target_length=128,
        mesh=None,
        attention_kernel="dot_product",
        attention_type=AttentionType.COMPRESSED,
        sliding_window_size=3,
    )

    q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
    k_dummy = jnp.zeros((batch_size, kv_len, 1, 128))

    # Simulate a compressed float mask [batch, 1, s_len, c_len]
    compressed_mask = np.zeros((batch_size, 1, s_len, c_len), dtype=np.float32)
    compressed_mask[:, :, :, 0] = DEFAULT_MASK_VALUE
    compressed_mask = jnp.array(compressed_mask)

    mask = op.generate_attention_mask(
        query=q_dummy,
        key=k_dummy,
        decoder_segment_ids=None,
        model_mode="train",
        compressed_mask=compressed_mask,
    )

    # Returned float mask should dynamically inherit the dimensionality of compressed_mask
    self.assertEqual(mask.shape, (batch_size, 1, s_len, kv_len))
    mask_np = np.array(mask)[0, 0]

    # Uncompressed block (first s_len cols) follows sliding window float mask
    self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[0, 0], 0.0)
    self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[3, 1], 0.0)

    # Compressed block (last c_len cols) follows compressed_mask strictly
    np.testing.assert_allclose(mask_np[:, s_len], DEFAULT_MASK_VALUE)
    np.testing.assert_allclose(mask_np[:, s_len + 1], 0.0)
    print("Mask logic for uncompressed & compressed attention passed perfectly.")


class DeepSeekV4CompressedAttentionTest(unittest.TestCase):
  """Tests to validate MaxText CompressedAttention implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 512
    self.num_heads = 4
    self.head_dim = 128
    self.hidden_size = 256
    self.q_lora_rank = 32
    self.o_groups = 2
    self.o_lora_rank = 64

    self.rngs = nnx.Rngs(0)

    self.pt_config = DeepseekV4Config(
        hidden_size=self.hidden_size,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.head_dim,
        o_groups=self.o_groups,
        o_lora_rank=self.o_lora_rank,
        rope_theta=10000.0,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        index_n_heads=2,
        index_head_dim=self.head_dim,
        index_topk=2,
        layer_types=["sliding_attention"],
        num_hidden_layers=1,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            "compress": {"rope_type": "default", "rope_theta": 160000.0, "partial_rotary_factor": 1.0},
        },
        sliding_window=2048,
        attention_dropout=0.0,
        max_position_embeddings=2048,
    )

  def _build_maxtext_config(self, layer_type):
    """Builds a MaxText pyconfig for a specific layer_type."""

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "base_emb_dim": self.pt_config.hidden_size,
        "head_dim": self.pt_config.head_dim,
        "base_num_query_heads": self.pt_config.num_attention_heads,
        "base_num_kv_heads": 1,
        "dtype": "float32",
        "weight_dtype": "float32",
        "sliding_window_size": self.pt_config.sliding_window,
        "q_lora_rank": self.pt_config.q_lora_rank,
        "o_groups": self.pt_config.o_groups,
        "o_lora_rank": self.pt_config.o_lora_rank,
        "compress_ratios": [0, 4, 128],  # Dummy list for the test
        "compressed_rope_max_timescale": self.pt_config.rope_parameters["compress"]["rope_theta"],
        "indexer_n_heads": self.pt_config.index_n_heads,
        "indexer_head_dim": self.pt_config.index_head_dim,
        "indexer_topk": self.pt_config.index_topk,
        "normalization_layer_epsilon": self.pt_config.rms_norm_eps,
        "partial_rotary_factor": self.pt_config.rope_parameters["compress" if layer_type != "sliding_attention" else "main"]["partial_rotary_factor"],
    }

    argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
    mt_config = pyconfig.initialize(argv, **config_arguments)

    return mt_config

  def _copy_linear(self, mt_linear, pt_linear):
    if pt_linear is None or mt_linear is None:
      return
    mt_linear.kernel.value = jnp.array(pt_linear.weight.data.numpy().T)
    if hasattr(pt_linear, "bias") and pt_linear.bias is not None:
      mt_linear.bias.value = jnp.array(pt_linear.bias.data.numpy())

  def _copy_norm(self, mt_norm, pt_norm):
    if pt_norm is None or mt_norm is None:
      return
    if hasattr(pt_norm, "weight") and pt_norm.weight is not None:
      mt_norm.scale.value = jnp.array(pt_norm.weight.data.numpy())

  def _run_e2e_test(self, layer_type, is_packed=False):
    self.pt_config.layer_types = [layer_type]

    torch.manual_seed(42)
    ref_attn = DeepseekV4Attention(self.pt_config, layer_idx=0)
    self.ref_attn = ref_attn

    if layer_type == "compressed_sparse_attention" and self.pt_config.index_topk == 2:
      for p in ref_attn.parameters():
        p.data = torch.abs(p.data) + 0.1

    rope_main = PTRope(self.pt_config)
    rope_compress = PTRope(self.pt_config)

    mt_config = self._build_maxtext_config(layer_type)

    mesh = Mesh(mesh_utils.create_device_mesh((1,)), axis_names=("fsdp",))

    compress_ratio_map = {
        "sliding_attention": 0,
        "compressed_sparse_attention": self.pt_config.compress_rates["compressed_sparse_attention"],
        "heavily_compressed_attention": self.pt_config.compress_rates["heavily_compressed_attention"],
    }
    mt_attn = CompressedAttention(
        config=mt_config,
        compress_ratio=compress_ratio_map[layer_type],
        num_query_heads=self.num_heads,
        num_kv_heads=1,
        head_dim=self.head_dim,
        max_target_length=128,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.batch_size, self.seq_len, self.hidden_size),
        inputs_kv_shape=(self.batch_size, self.seq_len, self.hidden_size),
        q_lora_rank=self.q_lora_rank,
        sliding_window_size=mt_config.sliding_window_size,
        rngs=self.rngs,
    )
    self.mt_attn = mt_attn
    if layer_type == "sliding_attention":
      rope_factor = self.pt_config.rope_parameters["main"]["partial_rotary_factor"]
      mt_rope = MTRope(head_dim=self.head_dim, partial_rotary_factor=rope_factor, rope_theta=10000.0)
    else:
      rope_factor = self.pt_config.rope_parameters["compress"]["partial_rotary_factor"]
      mt_rope = MTRope(head_dim=self.head_dim, partial_rotary_factor=rope_factor, rope_theta=160000.0)

    mt_attn.rotary_embedding = mt_rope
    mt_attn.rotary_emb = mt_rope
    if hasattr(mt_attn, "csa_compressor"):
      mt_attn.csa_compressor.rotary_emb = mt_rope
      mt_attn.csa_compressor.indexer.rotary_emb = mt_rope
    if hasattr(mt_attn, "hca_compressor"):
      mt_attn.hca_compressor.rotary_emb = mt_rope

    # 3. Copy Weights
    self._copy_linear(mt_attn.wq_a, ref_attn.q_a_proj)
    mt_attn.wq_b.kernel.value = jnp.array(
        ref_attn.q_b_proj.weight.data.numpy().T.reshape(self.q_lora_rank, self.num_heads, self.head_dim)
    )
    mt_attn.wkv.kernel.value = jnp.array(
        ref_attn.kv_proj.weight.data.numpy().T.reshape(
            self.hidden_size, self.pt_config.num_key_value_heads, self.head_dim
        )
    )
    self._copy_norm(mt_attn.q_norm, ref_attn.q_a_norm)
    self._copy_norm(mt_attn.kv_norm, ref_attn.kv_norm)
    mt_attn.sinks.value = jnp.array(ref_attn.sinks.data.numpy().reshape(-1))

    pt_oa_weight = ref_attn.o_a_proj.weight.data.numpy()
    mt_oa_weight = pt_oa_weight.reshape(self.o_groups, -1, (self.num_heads * self.head_dim) // self.o_groups).transpose(
        0, 2, 1
    )
    mt_attn.o_a_proj.kernel.value = jnp.array(mt_oa_weight)
    self._copy_linear(mt_attn.o_b_proj, ref_attn.o_b_proj)

    if layer_type == "heavily_compressed_attention":
      self._copy_linear(mt_attn.hca_compressor.kv_proj, ref_attn.compressor.kv_proj)
      self._copy_linear(mt_attn.hca_compressor.gate_proj, ref_attn.compressor.gate_proj)
      mt_attn.hca_compressor.position_bias.value = jnp.array(ref_attn.compressor.position_bias.data.numpy())
      self._copy_norm(mt_attn.hca_compressor.kv_norm, ref_attn.compressor.kv_norm)

    if layer_type == "compressed_sparse_attention":
      self._copy_linear(mt_attn.csa_compressor.kv_proj, ref_attn.compressor.kv_proj)
      self._copy_linear(mt_attn.csa_compressor.gate_proj, ref_attn.compressor.gate_proj)
      mt_attn.csa_compressor.position_bias.value = jnp.array(ref_attn.compressor.position_bias.data.numpy())
      self._copy_norm(mt_attn.csa_compressor.kv_norm, ref_attn.compressor.kv_norm)

      self._copy_linear(mt_attn.csa_compressor.indexer.q_proj, ref_attn.compressor.indexer.q_b_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.kv_proj, ref_attn.compressor.indexer.kv_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.gate_proj, ref_attn.compressor.indexer.gate_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.weights_proj, ref_attn.compressor.indexer.weights_proj)
      mt_attn.csa_compressor.indexer.position_bias.value = jnp.array(
          ref_attn.compressor.indexer.position_bias.data.numpy()
      )
      self._copy_norm(mt_attn.csa_compressor.indexer.kv_norm, ref_attn.compressor.indexer.kv_norm)

    # 4. Inputs
    np.random.seed(42)
    if layer_type == "compressed_sparse_attention" and self.pt_config.index_topk == 2:
      x_np = np.random.uniform(0.1, 1.0, size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    else:
      x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    pos_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)

    x_pt = torch.tensor(x_np)
    pos_pt = torch.tensor(pos_np, dtype=torch.long)

    x_mt = jnp.array(x_np)
    pos_mt = jnp.array(pos_np)

    if is_packed:
      half = self.seq_len // 2
      segs_np = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
      segs_np[:, half:] = 2
      segs_mt = jnp.array(segs_np)
    else:
      segs_mt = jnp.ones_like(pos_mt, dtype=jnp.int32)

    # 5. Execute PyTorch
    dummy_x_main = torch.zeros(self.batch_size, self.seq_len, 1)
    cos_main, sin_main = rope_main(dummy_x_main, pos_pt, "main")
    cos_comp, sin_comp = rope_compress(dummy_x_main, pos_pt, "compress")

    pt_positions = {"main": (cos_main, sin_main), "compress": (cos_comp, sin_comp)}

    if is_packed:
      pt_mask = torch.full((self.batch_size, 1, self.seq_len, self.seq_len), float("-inf"))
      pt_mask[:, :, :half, :half] = _prepare_4d_causal_attention_mask(None, (self.batch_size, half), x_pt, 0, 2048)
      pt_mask[:, :, half:, half:] = _prepare_4d_causal_attention_mask(
          None, (self.batch_size, self.seq_len - half), x_pt, 0, 2048
      )
    else:
      pt_mask = _prepare_4d_causal_attention_mask(None, (self.batch_size, self.seq_len), x_pt, 0, 2048)

    pt_out, _ = ref_attn(x_pt, pt_positions, pos_pt, attention_mask=pt_mask)

    # Extract indexer top_k from PyTorch
    if layer_type == "compressed_sparse_attention":
      pt_q_residual = ref_attn.q_a_norm(ref_attn.q_a_proj(x_pt))
      pt_top_k_indices = ref_attn.compressor.indexer(x_pt, pt_q_residual, pos_pt, None, 0)
      print(f"PyTorch top_k_indices:\n{pt_top_k_indices[0]}")

      mt_q_latent = mt_attn.wq_a(x_mt)
      mt_q_residual = mt_attn.q_norm(mt_q_latent)
      mt_top_k_indices = mt_attn.csa_compressor.indexer(x_mt, mt_q_residual, pos_mt)
      print(f"MaxText top_k_indices:\n{mt_top_k_indices[0]}")

      num_mismatches = np.sum(pt_top_k_indices.detach().numpy() != np.array(mt_top_k_indices))
      print(f"top_k_indices mismatches: {num_mismatches}")

    # 6. Execute MaxText
    mt_out = mt_attn(x_mt, x_mt, segs_mt, pos_mt, deterministic=True, model_mode=MODEL_MODE_TRAIN)

    # 7. Asserts
    if not is_packed:
      print("Comparing MaxText vs PyTorch:")
      if hasattr(mt_attn, "hca_compressor"):
        mt_comp = mt_attn.hca_compressor
        pt_comp = ref_attn.compressor

        pt_kv = pt_comp.kv_proj(x_pt)
        mt_kv = mt_comp.kv_proj(x_mt)
        print(f"kv_proj error: {np.max(np.abs(pt_kv.detach().numpy() - np.array(mt_kv)))}")

        pt_gate = pt_comp.gate_proj(x_pt)
        mt_gate = mt_comp.gate_proj(x_mt)
        print(f"gate_proj error: {np.max(np.abs(pt_gate.detach().numpy() - np.array(mt_gate)))}")

        batch, seq_len, _ = x_pt.shape
        n_windows = seq_len // pt_comp.compress_rate
        pt_chunk_kv = pt_kv.view(batch, n_windows, pt_comp.compress_rate, -1)
        pt_chunk_gate = pt_gate.view(batch, n_windows, pt_comp.compress_rate, -1) + pt_comp.position_bias

        mt_chunk_kv = mt_kv.reshape((batch, n_windows, mt_comp.compress_rate, -1))
        mt_chunk_gate = mt_gate.reshape((batch, n_windows, mt_comp.compress_rate, -1)) + mt_comp.position_bias.value
        print(f"chunk_gate error: {np.max(np.abs(pt_chunk_gate.detach().numpy() - np.array(mt_chunk_gate)))}")

        pt_gate_weights = pt_chunk_gate.softmax(dim=2, dtype=torch.float32).to(pt_chunk_kv.dtype)
        mt_gate_weights = jax.nn.softmax(mt_chunk_gate, axis=2).astype(mt_chunk_kv.dtype)
        print(f"gate_weights error: {np.max(np.abs(pt_gate_weights.detach().numpy() - np.array(mt_gate_weights)))}")

        pt_compressed = pt_comp.kv_norm((pt_chunk_kv * pt_gate_weights).sum(dim=2))
        mt_compressed = mt_comp.kv_norm(jnp.sum(mt_chunk_kv * mt_gate_weights, axis=2))
        print(f"compressed before rope error: {np.max(np.abs(pt_compressed.detach().numpy() - np.array(mt_compressed)))}")

        pt_positions = torch.arange(n_windows) * pt_comp.compress_rate
        pt_positions = pt_positions.unsqueeze(0).expand(batch, -1)
        pt_cos, pt_sin = pt_comp.rotary_emb(pt_compressed, position_ids=pt_positions, layer_type=pt_comp.rope_layer_type)

        mt_positions = jnp.arange(n_windows) * mt_comp.compress_rate
        mt_positions = jnp.broadcast_to(mt_positions[None, :], (batch, n_windows))
        mt_cos, mt_sin = mt_comp.rotary_emb.get_freqs(mt_positions)
        print(f"cos error: {np.max(np.abs(pt_cos.detach().numpy() - np.array(mt_cos)))}")
        print(f"sin error: {np.max(np.abs(pt_sin.detach().numpy() - np.array(mt_sin)))}")

        pt_compressed_rot = apply_rotary_pos_emb(pt_compressed.unsqueeze(1), pt_cos, pt_sin).squeeze(1)
        mt_compressed_rot = mt_comp.rotary_emb(mt_compressed, mt_positions, unsqueeze_dim=None)
        error = np.max(np.abs(pt_compressed_rot.detach().numpy() - np.array(mt_compressed_rot)))
        print(f"compressed after rope error: {error}")

      if layer_type == "compressed_sparse_attention":
        pt_comp = ref_attn.compressor
        mt_comp = mt_attn.csa_compressor
        kv_error = np.max(np.abs(pt_comp.kv_proj(x_pt).detach().numpy() - np.array(mt_comp.kv_proj(x_mt))))
        print(f"csa kv_proj error: {kv_error}")
        gate_error = np.max(np.abs(pt_comp.gate_proj(x_pt).detach().numpy() - np.array(mt_comp.gate_proj(x_mt))))
        print(f"csa gate_proj error: {gate_error}")

      np.testing.assert_allclose(np.array(mt_out), pt_out.detach().numpy(), rtol=1e-5, atol=1e-5)
    else:
      self.assertFalse(np.allclose(np.array(mt_out), pt_out.detach().numpy(), rtol=1e-3, atol=1e-3))
      print(f"Document packing test ({layer_type}) successfully confirmed PyTorch bug and MaxText firewall.")

  def test_forward_uncompressed(self):
    self._run_e2e_test("sliding_attention")

  def test_forward_hca(self):
    self._run_e2e_test("heavily_compressed_attention")

  def test_forward_csa(self):
    self._run_e2e_test("compressed_sparse_attention")

  def test_document_packing_masking(self):
    self._run_e2e_test("heavily_compressed_attention", is_packed=True)
if __name__ == "__main__":
  unittest.main()
