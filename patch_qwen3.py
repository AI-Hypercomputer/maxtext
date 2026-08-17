import re

qwen3_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/models/qwen3.py'
with open(qwen3_path, 'r') as f:
    qwen3_code = f.read()

qwen3_code = qwen3_code.replace(
    'def jax_chunk_gated_delta_rule(\n    query: Array,\n    key: Array,\n    value: Array,\n    g: Array,\n    beta: Array,\n    chunk_size: int = 64,\n    initial_state: None | Array = None,\n    use_qk_norm_in_gdn: bool = False,\n    compute_dtype: jnp.dtype = jnp.bfloat16,\n) -> tuple[Array, None | Array]:',
    'def jax_chunk_gated_delta_rule(\n    query: Array,\n    key: Array,\n    value: Array,\n    g: Array,\n    beta: Array,\n    chunk_size: int = 64,\n    initial_state: None | Array = None,\n    use_qk_norm_in_gdn: bool = False,\n    compute_dtype: jnp.dtype = jnp.bfloat16,\n) -> tuple[Array, None | Array, Array]:'
)

qwen3_code = qwen3_code.replace(
    'return o, (final_h if initial_state is not None else None)',
    'return o, (final_h if initial_state is not None else None), A'
)

with open(qwen3_path, 'w') as f:
    f.write(qwen3_code)
