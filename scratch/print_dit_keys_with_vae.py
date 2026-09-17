import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
import sys
import os

# Append this directory to path
sys.path.append('src/')

from maxtext.models.dit import DiT

class DummyConfig:
    def __init__(self):
        self.base_emb_dim = 128
        self.patch_size_for_vit = 2
        self.num_channels_for_vit = 4
        self.max_target_length = 64
        self.dtype = jnp.float32
        self.weight_dtype = jnp.float32
        self.vocab_size = 10
        self.base_num_query_heads = 4
        self.base_num_decoder_layers = 2
        self.attention = 'dot_product'
        self.shard_mode = 'auto'
        self.fused_qkv = False
        self.use_bias_in_projections = True
        self.share_kv_projections = False
        self.attention_bias = False
        self.fused_mlp = False
        self.activations_in_float32 = False
        self.logical_axis_rules = None
        self.use_iota_embed = False
        self.debug_sharding = False
        self.moba = False
        self.context_parallel_load_balance = False
        self.context_sharding = 'data'
        self.attn_logits_soft_cap = 0.0
        self.sliding_window_size = None
        self.use_ragged_attention = False
        self.ragged_block_size = 16
        self.use_qk_norm = False
        self.query_pre_attn_scalar = None
        self.is_nope_layer = True
        self.is_vision = False
        self.model_mode = 'train'
        self.use_mrope = False
        self.mrope_section = None
        self.rope_type = 'default'
        self.use_v_norm = False
        self.rope_max_timescale = 10000.0
        self.partial_rotary_factor = 1.0
        self.share_kv_layer = False
        self.decoder_block = 'dit'
        self.attention_type = 'full'
        self.rope_linear_scaling_factor = 1.0
        self.rope_use_scale = False
        self.rope_min_timescale = 1.0
        self.chunk_attn_window_size = 0
        self.model_name = 'dit-xl-2-256'

config = DummyConfig()
devices = jax.devices()
mesh = Mesh(devices, ('data',))

def init_dit():
    return DiT(config, mesh, rngs=nnx.Rngs(0))

print("Starting eval_shape...", flush=True)
abs_model = nnx.eval_shape(init_dit)
print("eval_shape done.", flush=True)
_, abstract_params_tree, _ = nnx.split(abs_model, nnx.Param, ...)

pure_dict = abstract_params_tree.to_pure_dict()
print("\nPure Dict Structure:", flush=True)
def print_keys(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print_keys(v, prefix + str(k) + ".")
    else:
        print(f"{prefix} -> Shape: {d.shape}")

print_keys(pure_dict)


