import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class DeepseekV32Config:
    def __init__(self):
        # Model Dimensions (Scaled down for easy CPU testing)
        self.vocab_size = 32000
        self.hidden_size = 256
        self.num_hidden_layers = 2
        
        # MLA (Multi-Head Latent Attention)
        self.num_attention_heads = 8
        self.q_lora_rank = 128
        self.kv_lora_rank = 128
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        self.max_seq_len = 128
        
        # RoPE Parameters
        self.rope_theta = 10000.0
        
        # DSA Indexer (Sparse Attention)
        self.index_n_heads = 4
        self.index_head_dim = 32
        self.index_topk = 16 # How many tokens to attend to
        
        # MoE Parameters
        self.n_routed_experts = 8
        self.n_shared_experts = 2
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 128
        self.routed_scaling_factor = 1.0
        self.n_group = 2
        self.topk_group = 1

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True):
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(x.dtype)


# -----------------------------------------------------------------------------
# Core Modules
# -----------------------------------------------------------------------------
class Indexer(nn.Module):
    """Dynamic Sparse Attention (DSA) Indexer to select Top-K tokens."""
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(self, x, qr, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        
        # Process Queries
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        
        # Process Keys
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        
        # --- NO HADAMARD ROTATION NEEDED FOR FP32/BF16 ---
        # We compute the logits directly on the un-rotated space
        
        # Scoring
        weights = self.weights_proj(x) * (self.n_heads ** -0.5) * self.softmax_scale
        logits = F.relu(torch.einsum("bthd, bsd -> btsh", q, k))
        index_score = torch.einsum("btsh, bth -> bts", logits, weights)

        # Apply causal mask and select Top-K
        if mask is not None:
            index_score += mask
            
        topk_indices = index_score.topk(min(self.index_topk, seqlen), dim=-1)[1]
        
        # Generate Sparse Mask (-inf for dropped tokens)
        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0.0)
        return index_mask

class MLA(nn.Module):
    """Multi-Head Latent Attention with DSA Indexing."""
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        
        self.wkv_a = nn.Linear(config.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank)
        self.wkv_b = nn.Linear(config.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.indexer = Indexer(config)

    def forward(self, x, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        
        # Query Latent Projection
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # KV Latent Projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        kv_expanded = self.wkv_b(kv).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # Attention Math
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale

        # Get Sparse Mask from Indexer and add Causal Mask
        index_mask = self.indexer(x, qr, freqs_cis, mask)
        scores += index_mask.unsqueeze(2) # [B, S, 1, T]
        if mask is not None:
            scores += mask.unsqueeze(1).unsqueeze(2)

        scores = F.softmax(scores, dim=-1)
        out = torch.einsum("bsht,bthd->bshd", scores, v)
        return self.wo(out.flatten(2))

class DeepseekMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DeepseekMoE(nn.Module):
    """Auxiliary-Loss-Free MoE with Shared Experts."""
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.experts = nn.ModuleList([DeepseekMLP(config, config.moe_intermediate_size) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts))
        
        self.shared_experts = DeepseekMLP(config, config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Routing Logic
        router_logits = self.gate(x_flat)
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        
        # Group Restricted Routing
        if self.n_group > 1:
            group_scores = scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group).reshape(-1, self.num_experts)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), -float('inf'))

        _, topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        topk_weights = scores.gather(1, topk_indices) # No scaling factor applied as routed_scaling_factor=1.0
        
        # Expert Execution
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

# -----------------------------------------------------------------------------
# DeepSeek V3.2 Model
# -----------------------------------------------------------------------------
class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = MLA(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = DeepseekMoE(config)

    def forward(self, x, freqs_cis, mask):
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DeepseekV32ForCausalLM(nn.Module):
    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV32DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        
        # Precompute RoPE frequencies and Causal Mask
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, seqlen, self.config.rope_theta).to(x.device)
        
        causal_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=x.device), diagonal=1)

        # Forward pass through layers
        for layer in self.layers:
            x = layer(x, freqs_cis, causal_mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# -----------------------------------------------------------------------------
# Execution (Test Forward Pass)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure float32 is used natively
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)

    print("Initializing DeepSeek V3.2 Model (CPU | Float32)...")
    
    # 1. Comment out the standard config initialization
    # config = DeepseekV32Config()
    
    # 2. Super Small Dummy Config for Fast Iteration
    config = DeepseekV32Config()
    
    # Tiny Embeddings & Layers
    config.vocab_size = 128
    config.hidden_size = 32
    config.num_hidden_layers = 1
    
    # Tiny MLA 
    config.num_attention_heads = 2
    config.q_lora_rank = 16
    config.kv_lora_rank = 16
    config.qk_nope_head_dim = 8
    config.qk_rope_head_dim = 4
    config.v_head_dim = 8
    config.max_seq_len = 16
    
    # Tiny DSA Indexer
    config.index_n_heads = 2
    config.index_head_dim = 8
    config.index_topk = 4
    
    # Tiny MoE
    config.n_routed_experts = 4
    config.n_shared_experts = 1
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 16
    
    # Initialize the micro-model
    model = DeepseekV32ForCausalLM(config)
    model.eval()

    # Generate Dummy Input
    batch_size = 2
    seq_length = 8
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    print(f"Running Forward Pass with input shape: {dummy_input_ids.shape}")
    with torch.no_grad():
        logits = model(dummy_input_ids)

    print(f"Forward Pass Successful!")
    print(f"Output Logits Shape: {logits.shape} (Expected: {batch_size}, {seq_length}, {config.vocab_size})")
    print(f"Sample Logit Values (First Token): {logits[0, 0, :5].tolist()}")