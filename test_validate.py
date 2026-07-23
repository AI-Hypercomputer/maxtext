import jax.numpy as jnp
import numpy as np

def _interleave_qkv(qkv_weight, tp):
    # Monolithic has shape [Q_size + K_size + V_size, hidden_dim]
    pass

def chunk_gate_up(gate_up, tp):
    # Monolithic gate_up has shape [experts, 2 * d_inner, d_model] or [2 * d_inner, d_model]
    pass
