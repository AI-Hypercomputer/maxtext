import re

hybrid_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/models/hybrid_gdn.py'
with open(hybrid_path, 'r') as f:
    hybrid_code = f.read()

hybrid_code = hybrid_code.replace(
    'core_attn_out, next_recurrent_state = jax_chunk_gated_delta_rule(',
    'core_attn_out, next_recurrent_state, pure_jax_tap = jax_chunk_gated_delta_rule('
)

hybrid_code = hybrid_code.replace(
    'return core_attn_out.astype(qkv.dtype), (next_conv_state.astype(qkv.dtype), next_recurrent_state.astype(qkv.dtype))',
    'return core_attn_out.astype(qkv.dtype), (next_conv_state.astype(qkv.dtype), next_recurrent_state.astype(qkv.dtype)), pure_jax_tap'
)

hybrid_code = hybrid_code.replace(
    'return core_attn_out.astype(qkv.dtype), (new_conv_state[1:].astype(qkv.dtype), new_recurrent_state[1:].astype(qkv.dtype)), tap_out',
    'tap_out = tap_out.reshape(batch_size, -1, num_v_heads, chunk_size, chunk_size)\n    return core_attn_out.astype(qkv.dtype), (new_conv_state[1:].astype(qkv.dtype), new_recurrent_state[1:].astype(qkv.dtype)), tap_out'
)


with open(hybrid_path, 'w') as f:
    f.write(hybrid_code)
