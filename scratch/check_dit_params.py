import jax
from flax import nnx
from maxtext.models.dit import DiT
from maxtext.common.common_types import Config
import sys
import os

# Mock config
class MockConfig:
  def __init__(self):
    self.base_emb_dim = 1152
    self.patch_size_for_vit = 2
    self.num_channels_for_vit = 4
    self.vocab_size = 1000
    self.base_num_decoder_layers = 1 # Keep it small
    self.base_num_query_heads = 16
    self.max_target_length = 256
    self.dtype = 'float32'
    self.weight_dtype = 'float32'
    self.attention = 'dot_product'
    self.shard_mode = 'auto'
    self.debug_sharding = False
    self.ici_context_autoregressive_parallelism = 1
    self.fused_qkv = False
    self.attention_sink = False
    self.qk_norm_with_scale = True
    self.v_norm_with_scale = True
    self.normalization_layer_epsilon = 1e-6
    self.decoder_block = 'dit'
    self.fused_mlp = False
    self.dense_fsdp_use_two_stage_all_gather = False
    self.activations_in_float32 = False
    self.mesh_axes = None
    self.logical_axis_rules = None
    self.using_pipeline_parallelism = False
    self.context_parallel_strategy = 'none'
    self.rope_type = 'none'
    self.rope_max_timescale = 10000.0
    self.attention_type = 'global'
    self.chunk_attn_window_size = 0
    self.rope_use_scale = False
    self.model_name = 'dit-xl-2-256'
    self.rope_linear_scaling_factor = 1.0
    self.rope_min_timescale = 1.0
    self.matmul_precision = 'default'








config = MockConfig()
# We need a real mesh or mock it.
# Let's try without mesh first if possible, or simple CPU mesh.
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ('data',)) # Dummy mesh

# Set required attributes for mesh sharding if needed
# For now let's hope DiTBlock doesn't fail on initialization if mesh is simple.

try:
    model = DiT(config, mesh, rngs=nnx.Rngs(0))
    _, params, _ = nnx.split(model, nnx.Param, ...)
    print("Parameters found:")
    # Use jax.tree_util to flatten state
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, val in flat_params:
        # path is a tuple of DictKey, SequenceKey, etc.
        path_str = "/".join(str(p) for p in path)
        print(f"{path_str}: {type(val)}")
        if hasattr(val, 'value'):
            print(f"  value type: {type(val.value)}")
            if hasattr(val.value, 'shape'):
                print(f"  shape: {val.value.shape}")
        elif hasattr(val, 'shape'): # It might be a direct array
            print(f"  shape: {val.shape}")

except Exception as e:

    print(f"Error initializing model: {e}")
    import traceback
    traceback.print_exc()

