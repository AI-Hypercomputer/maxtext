import unittest
import numpy as np
import scipy.linalg

import torch
from torch import nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from flax import nnx
import flax.traverse_util
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models import deepseek
from maxtext.layers import embeddings, linears, normalizations
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# -----------------------------------------------------------------------------
# 1. PyTorch Reference Implementation
# -----------------------------------------------------------------------------

class DeepseekV32Config:
    def __init__(self, cfg):
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.emb_dim
        self.num_hidden_layers = cfg.num_decoder_layers
        
        self.num_attention_heads = cfg.num_query_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.max_seq_len = cfg.max_target_length
        self.rope_theta = getattr(cfg, 'rope_max_timescale', 10000.0)
        
        self.index_n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_topk = cfg.index_topk
        
        self.n_routed_experts = cfg.num_experts
        self.n_shared_experts = cfg.shared_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.moe_intermediate_size = cfg.moe_mlp_dim
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.n_group = cfg.n_routing_groups
        self.topk_group = cfg.topk_routing_group
        self.layer_norm_eps = cfg.normalization_layer_epsilon

class RMSNorm_PT(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))

def precompute_freqs_cis_pt(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb_pt(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True):
    orig_dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_complex = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(orig_dtype)

class Indexer_PT(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        
        # FIX: MaxText uses LayerNorm for the Indexer, not RMSNorm!
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps) 
        
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(self, x, qr, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb_pt(q_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb_pt(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        
        weights = self.weights_proj(x) * (self.n_heads ** -0.5) * self.softmax_scale
        logits = F.relu(torch.einsum("bthd, bsd -> btsh", q, k))
        index_score = torch.einsum("btsh, bth -> bts", logits, weights)

        if mask is not None:
            index_score += mask
            
        topk_indices = index_score.topk(min(self.index_topk, seqlen), dim=-1)[1]
        index_mask = torch.full((bsz, seqlen, seqlen), -1e9, device=x.device).scatter_(-1, topk_indices, 0.0)
        return index_mask

class MLA_PT(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.kv_lora_rank = config.kv_lora_rank

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm_PT(config.q_lora_rank, eps=config.layer_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        
        self.wkv_a = nn.Linear(config.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm_PT(config.kv_lora_rank, eps=config.layer_norm_eps)
        self.wkv_b = nn.Linear(config.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.indexer = Indexer_PT(config)

    def forward(self, x, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb_pt(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb_pt(k_pe.unsqueeze(2), freqs_cis)
        
        kv_expanded = self.wkv_b(kv).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale

        index_mask = self.indexer(x, qr, freqs_cis, mask)
        scores += index_mask.unsqueeze(2) 
        
        if mask is not None:
            scores += mask.unsqueeze(0).unsqueeze(2)

        scores = F.softmax(scores, dim=-1)
        out = torch.einsum("bsht,bthd->bshd", scores, v)
        return self.wo(out.flatten(2))

class DeepseekMLP_PT(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DeepseekMoE_PT(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.experts = nn.ModuleList([DeepseekMLP_PT(config, config.moe_intermediate_size) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts))
        self.shared_experts = DeepseekMLP_PT(config, config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        router_logits = self.gate(x_flat)
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        
        if self.n_group > 1:
            group_scores = scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group).reshape(-1, self.num_experts)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), -float('inf'))

        _, topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        topk_weights = scores.gather(1, topk_indices)
        
        final_h = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            expert_mask = (topk_indices == i)
            if not expert_mask.any(): continue
            idx, top_x = torch.where(expert_mask)
            current_state = x_flat[idx]
            current_h = self.experts[i](current_state) * topk_weights[idx, top_x, None]
            final_h.index_add_(0, idx, current_h)
            
        shared_output = self.shared_experts(x_flat)
        return (final_h + shared_output).view(bsz, seq_len, dim)

class DeepseekV32DecoderLayer_PT(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.input_layernorm = RMSNorm_PT(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MLA_PT(config)
        self.post_attention_layernorm = RMSNorm_PT(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DeepseekMoE_PT(config)

    def forward(self, x, freqs_cis, mask):
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DeepseekV32ForCausalLM_PT(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV32DecoderLayer_PT(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm_PT(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = precompute_freqs_cis_pt(self.config.qk_rope_head_dim, seqlen, self.config.rope_theta).to(x.device)
        causal_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis, causal_mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------------------------------------------------------
# 2. MaxText JAX Wrapper (Mirrors the PyTorch Structure)
# -----------------------------------------------------------------------------
class MaxTextDeepseekV32ForCausalLM(nnx.Module):
    """Wraps authentic MaxText components to mirror the CausalLM structure."""
    def __init__(self, config, mesh, rngs):
        self.embed_tokens = embeddings.Embed(
            num_embeddings=config.vocab_size, num_features=config.emb_dim, 
            config=config, mesh=mesh, rngs=rngs
        )
        self.layers = nnx.List([
            deepseek.DeepSeekMoELayer(config=config, model_mode="train", mesh=mesh, rngs=rngs, layer_idx=i)
            for i in range(config.num_decoder_layers)
        ])
        self.norm = normalizations.RMSNorm(
            num_features=config.emb_dim, dtype=config.dtype, weight_dtype=config.weight_dtype, 
            kernel_axes=("norm",), epsilon=config.normalization_layer_epsilon, rngs=rngs
        )
        self.lm_head = linears.DenseGeneral(
            in_features_shape=config.emb_dim, out_features_shape=config.vocab_size, 
            axis=-1, kernel_axes=("embed", "vocab"), dtype=config.dtype, 
            weight_dtype=config.weight_dtype, use_bias=False, rngs=rngs
        )

    def __call__(self, input_ids, decoder_segment_ids, decoder_positions):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x, _ = layer(
                x, decoder_segment_ids=decoder_segment_ids, 
                decoder_positions=decoder_positions, 
                deterministic=True, model_mode="train"
            )
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------------------------------------------------------
# 3. Execution / Testing Block
# -----------------------------------------------------------------------------

class TestDeepSeekV32Parity(unittest.TestCase):
    def setUp(self):
        # Using exact official DeepSeek V3.2 - 671B dimensions
        self.cfg = pyconfig.initialize([
            None,
            get_test_config_path(),
            "run_name=deepseek32_e2e_constant_test",
            "model_name=deepseek3.2-671b",
            "decoder_block=deepseek",
            "dtype=float32",
            "weight_dtype=float32",
            "matmul_precision=highest",
            "attention=dot_product", 
            "attention_type=mla",
            
            # --- OFFICIAL DEEPSEEK V3.2 671B CONFIG ---
            "base_emb_dim=7168",
            "base_num_query_heads=128",
            "base_num_kv_heads=128",
            "head_dim=128",  # Corresponds to v_head_dim
            "q_lora_rank=1536",
            "kv_lora_rank=512",
            "qk_nope_head_dim=128",
            "qk_rope_head_dim=64",
            "v_head_dim=128",
            
            # RoPE Parameters
            "max_target_length=163840",
            "max_prefill_predict_length=163840",
            "rope_max_timescale=10000.0",
            
            # Sparse Indexer Parameters
            "use_sparse_indexer=True",
            "index_n_heads=64",
            "index_head_dim=128",
            "index_topk=2048",
            
            # MoE Parameters
            "num_experts=256",
            "num_experts_per_tok=8",
            "shared_experts=1",
            "base_mlp_dim=18432",
            "base_moe_mlp_dim=2048",
            "routed_scaling_factor=2.5",
            
            # DeepSeek V3/V3.2 standard group routing (8 groups, top 4)
            "n_routing_groups=8",
            "topk_routing_group=4",
            
            # Test Environment Overrides
            "num_decoder_layers=1",  # Change to 61 if you want to test the full ~400GB model!
            "routed_bias=True",
            "routed_score_func=sigmoid",
            "capacity_factor=0", # Disable token dropping for strict parity
            "use_random_routing=False",
            "mlp_activations=['silu', 'linear']",
            "fused_mlp=False",
            "sparse_matmul=False", # Use dense matmul path for deterministic unit testing
            "norm_topk_prob=False",
            "engram_layers=[]", 
            "skip_jax_distributed_system=True",
        ])
        
        self.batch_size = 2
        # Set sequence length slightly higher so the indexer has actual tokens to sort through
        self.seq_len = 128 
        
        devices_array = maxtext_utils.create_device_mesh(self.cfg)
        self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
        
    def test_e2e_parity_constant_init(self):
        print("\nRunning End-to-End Parity Test (No Param Mapping / Constant Weights)...")

        # ==========================================
        # 1. Initialize PyTorch Model with Constants
        # ==========================================
        pt_config = DeepseekV32Config(self.cfg)
        pt_model = DeepseekV32ForCausalLM_PT(pt_config)
        
        with torch.no_grad():
            for name, param in pt_model.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name and "norm" not in name:
                    nn.init.constant_(param, 0.01)
                # Norm weights remain 1.0 (PyTorch default)
                    
        pt_model.eval()

        # ==========================================
        # 2. Initialize MaxText JAX Model with Constants
        # ==========================================
        jax_model = MaxTextDeepseekV32ForCausalLM(config=self.cfg, mesh=self.mesh, rngs=nnx.Rngs(0))
        
        state = nnx.state(jax_model, nnx.Param)
        
        # FIX: Call .to_dict() to convert the NNX State object to a normal dictionary
        pure_dict_state = nnx.to_pure_dict(state)
        flat_state = flax.traverse_util.flatten_dict(pure_dict_state)
        new_flat_state = {}
        
        for k, v in flat_state.items():
            # Keys in the tuple might be strings or ints, cast to str to be safe
            k_str = "/".join(str(x) for x in k)
            
            # Depending on the Flax version, v might be an nnx.Param object or the raw array.
            arr = v.value if hasattr(v, 'value') else v
            
            if "bias" in k_str or "e_score_correction_bias" in k_str:
                new_flat_state[k] = nnx.Param(jnp.zeros_like(arr))
            elif "scale" in k_str or "norm" in k_str:
                new_flat_state[k] = nnx.Param(jnp.ones_like(arr))
            else:
                new_flat_state[k] = nnx.Param(jnp.full_like(arr, 0.01))
                
        new_state = flax.traverse_util.unflatten_dict(new_flat_state)
        nnx.update(jax_model, new_state)

        # ==========================================
        # 3. Create Shared Dummy Inputs
        # ==========================================
        # Use simple arange input IDs to easily track token behavior
        input_ids_np = np.arange(self.batch_size * self.seq_len, dtype=np.int32).reshape(self.batch_size, self.seq_len)
        
        input_ids_pt = torch.from_numpy(input_ids_np)
        input_ids_jax = jnp.array(input_ids_np)

        decoder_positions_jax = jnp.broadcast_to(jnp.arange(self.seq_len, dtype=jnp.int32), (self.batch_size, self.seq_len))
        decoder_segment_ids_jax = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

        # ==========================================
        # 4. Run Forward Passes
        # ==========================================
        with torch.no_grad():
            expected_logits_pt = pt_model(input_ids_pt)

        # @jax.jit
        # def run_jax(inputs, seg_ids, pos):
        #     return jax_model(inputs, seg_ids, pos)

        # actual_logits_jax = run_jax(input_ids_jax, decoder_segment_ids_jax, decoder_positions_jax)

        actual_logits_jax = jax_model(
            input_ids_jax, 
            decoder_segment_ids_jax, 
            decoder_positions_jax
        )

        # ==========================================
        # 5. Compare Outputs
        # ==========================================
        np.testing.assert_allclose(
            expected_logits_pt.numpy(),
            np.asarray(actual_logits_jax),
            rtol=1e-3,
            atol=1e-3,
            err_msg="MaxText and PyTorch do not output the same logits!"
        )
        print(f"Logits shape: {expected_logits_pt.shape}")
        print("Success! MaxText and PyTorch generated identical outputs without explicit param mapping.")

if __name__ == '__main__':
    unittest.main()