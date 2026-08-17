import re

wrapper_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/kernels/causal_conv1d_gated_delta_rule/wrapper.py'
with open(wrapper_path, 'r') as f:
    wrapper_code = f.read()

# 1. inner_kernel modifications
wrapper_code = wrapper_code.replace(
    'out_slot_ref,\n    # Scratches.',
    'out_slot_ref,\n    tap_slot_ref,\n    # Scratches.'
)
wrapper_code = wrapper_code.replace(
    'out, new_recurrent_state = compute_gdn.chunked_gdn(',
    'out, new_recurrent_state, tap_val = compute_gdn.chunked_gdn('
)
wrapper_code = wrapper_code.replace(
    'out_slot_ref[...] = out.astype(out_slot_ref.dtype)',
    'out_slot_ref[...] = out.astype(out_slot_ref.dtype)\n  tap_slot_ref[...] = tap_val.astype(tap_slot_ref.dtype)'
)
wrapper_code = wrapper_code.replace(
    'recurrent_alloc.spec,\n      ),\n      out_specs=(out_alloc.spec,),',
    'recurrent_alloc.spec,\n      ),\n      out_specs=(out_alloc.spec, out_alloc.spec,),'
)
wrapper_code = wrapper_code.replace(
    'out_alloc,\n      ),',
    'out_alloc,\n          out_alloc,\n      ),'
)
wrapper_code = wrapper_code.replace(
    'out_ref,\n        scratches=(',
    'out_ref,\n        tap_ref,\n        scratches=('
)
wrapper_code = wrapper_code.replace(
    'out_ref: jax.Array,\n    conv_state_out_ref: jax.Array,',
    'out_ref: jax.Array,\n    tap_ref: jax.Array,\n    conv_state_out_ref: jax.Array,'
)


# 2. fused_conv1d_gdn modifications
wrapper_code = wrapper_code.replace(
    'out_shape=(out_shape, in_conv_state, in_recurrent_state),',
    'out_shape=(out_shape, in_conv_state, in_recurrent_state, out_shape),' # tap shape is same as out_shape for chunked_gdn
)
wrapper_code = wrapper_code.replace(
    'out_specs=(hbm_spec, hbm_spec, hbm_spec),',
    'out_specs=(hbm_spec, hbm_spec, hbm_spec, hbm_spec),'
)
wrapper_code = wrapper_code.replace(
    '    )(',
    '        input_output_aliases=input_output_aliases,\n    )('
)
wrapper_code = wrapper_code.replace(
    'input_output_aliases=input_output_aliases,\n        compiler_params=',
    'compiler_params='
)

wrapper_code = wrapper_code.replace(
    'return pl.pallas_call(',
    'res = pl.pallas_call('
)

wrapper_code = wrapper_code.replace(
    '        weights,\n    )',
    '        weights,\n    )\n    out_act, out_conv, out_rec, out_tap = res\n    return (out_conv, out_rec), out_act, out_tap'
)


with open(wrapper_path, 'w') as f:
    f.write(wrapper_code)


compute_gdn_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/kernels/causal_conv1d_gated_delta_rule/compute_gdn.py'
with open(compute_gdn_path, 'r') as f:
    compute_gdn_code = f.read()

compute_gdn_code = compute_gdn_code.replace(
    ') -> tuple[jax.Array, jax.Array]:\n  """Perform chunked GDN over input [num_heads, chunk, head_dim]."""',
    ') -> tuple[jax.Array, jax.Array, jax.Array]:\n  """Perform chunked GDN over input [num_heads, chunk, head_dim]."""'
)
compute_gdn_code = compute_gdn_code.replace(
    'return out, state\n\n\ndef chunked_gdn(',
    'return out, state, t_inv\n\n\ndef chunked_gdn('
)
compute_gdn_code = compute_gdn_code.replace(
    'out, state = chunked_gdn_per_seq(',
    'out, state, t_inv = chunked_gdn_per_seq('
)
compute_gdn_code = compute_gdn_code.replace(
    '    state_list.append(state)\n  out = jnp.stack(out_list, axis=0)\n  state = jnp.stack(state_list, axis=0)\n  return out, state',
    '    state_list.append(state)\n    t_inv_list.append(t_inv)\n  out = jnp.stack(out_list, axis=0)\n  state = jnp.stack(state_list, axis=0)\n  t_inv_out = jnp.stack(t_inv_list, axis=0)\n  return out, state, t_inv_out'
)
compute_gdn_code = compute_gdn_code.replace(
    '  state_list = []\n',
    '  state_list = []\n  t_inv_list = []\n'
)

with open(compute_gdn_path, 'w') as f:
    f.write(compute_gdn_code)

