"""Pallas-accelerated implementation of the Gated Delta Rule."""

import functools

import jax
from jax import lax
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from maxtext.layers.normalizations import l2norm

# ==========================================
# 1. OPTIMAL TRIANGLE_SOLVE DECOMPOSE PALLAS KERNEL
# ==========================================

def local_forward_substitution(a_matrix, b_matrix):
  """Performs forward substitution for a batch of lower triangular systems.

  Solves systems of the form A * x = b, where A is a batch of lower
  triangular matrices.

  Args:
      a_matrix: A JAX array of shape (B, N, N) representing a batch of lower
        triangular matrices.
      b_matrix: A JAX array of shape (B, N, K) representing the right-hand side
        of the linear systems.

  Returns:
      A JAX array of shape (B, N, K) containing the solutions x.
  """
  _, n_size, _ = b_matrix.shape
  x_list = []
  for i in range(n_size):
    b_i = b_matrix[:, i, :]
    if i == 0:
      x_i = b_i
    else:
      stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
      all_prev_a = a_matrix[:, i, :i]  # (B, i)
      prev_sum = jnp.sum(all_prev_a[..., None] * stacked_x, axis=1)  # (B, K)
      x_i = b_i - prev_sum  # (B, K) for the row i
    x_list.append(x_i)
  x = jnp.stack(x_list, axis=1)  # (B, N, K)
  return x


def optimal_decompose_kernel(a_ref, x_ref, *, block_size=16):
  """Pallas kernel to compute the inverse of a lower triangular matrix block-wise.

  This kernel solves for X in AX = I, where A is a batch of lower triangular
  matrices. It iterates through blocks of A and uses
  `local_forward_substitution` to solve for each block of X. The result is
  written back to `x_ref`.
  """
  a = a_ref[...]
  batch_size, n, _ = a.shape
  num_blocks = n // block_size

  # AX = I, solve for X block wise. X = I - sum(AX_prev)
  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = jnp.eye(n, dtype=a.dtype)[start:end, :]
    e_block = jnp.broadcast_to(e_block, (batch_size, block_size, n))

    if i == 0:
      target_b = e_block
    else:
      interaction_a = a[:, start:end, :start]
      solved_x = x_ref[:, :start, :]
      prev_sum = jnp.matmul(
          interaction_a, solved_x, precision=jax.lax.Precision.HIGHEST
      )
      target_b = e_block - prev_sum

    local_a = a[:, start:end, start:end]
    x_block = local_forward_substitution(local_a, target_b)
    x_ref[..., start:end, :] = x_block


@functools.partial(
    jax.custom_vjp, nondiff_argnums=(1, 2)
)
def run_optimal_decompose(a, n_block_size=8, block_size=16):
  """Differentiable wrapper for the Pallas exact decompose kernel."""
  orig_shape = a.shape
  n = orig_shape[-1]

  # Flatten all leading dimensions to handle (batch, chunks, heads, N, N)
  a_flat = a.reshape(-1, n, n)
  b_total = a_flat.shape[0]

  # Pad batch dimension if it's not cleanly divisible by n_block_size
  pad_b = (n_block_size - (b_total % n_block_size)) % n_block_size
  if pad_b > 0:
    a_flat = jnp.pad(a_flat, ((0, pad_b), (0, 0), (0, 0)))

  grid_size = a_flat.shape[0] // n_block_size
  kernel = functools.partial(optimal_decompose_kernel, block_size=block_size)

  x_flat = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(a_flat.shape, a_flat.dtype),
      grid=(grid_size,),
      in_specs=[pl.BlockSpec((n_block_size, n, n), lambda idx: (idx, 0, 0))],
      out_specs=pl.BlockSpec((n_block_size, n, n), lambda idx: (idx, 0, 0)),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100663296),
  )(a_flat)

  # Strip padding and restore original shape
  if pad_b > 0:
    x_flat = x_flat[:b_total]

  return x_flat.reshape(orig_shape)


def _run_optimal_decompose_fwd(a, n_block_size, block_size):
  x = run_optimal_decompose(a, n_block_size, block_size)
  return x, x  # Save x as residual for the backward pass


def _run_optimal_decompose_bwd(n_block_size, block_size, res, g):
  """Backward pass for `run_optimal_decompose`.

  Args:
      n_block_size: Non-differentiable param from forward pass.
      block_size: Non-differentiable param from forward pass.
      res: The residuals from the forward pass, which is the computed inverse `x`.
      g: The gradient of the output `x` (i.e., `d(A^{-1})`).
  """
  x = res
  # d(A^{-1}) = -x^T @ dA @ x^T
  x_t = x.swapaxes(-1, -2)
  d_a = -jnp.matmul(
      x_t,
      jnp.matmul(g, x_t, precision=jax.lax.Precision.HIGHEST),
      precision=jax.lax.Precision.HIGHEST,
  )

  # Original A = I - S is strictly lower triangular (or lower triangular).
  # Mask gradients to prevent upper-triangle bleeding.
  mask = jnp.tril(jnp.ones(d_a.shape[-2:], dtype=bool))
  d_a = jnp.where(mask, d_a, 0.0)
  return (d_a,)


run_optimal_decompose.defvjp(
    _run_optimal_decompose_fwd, _run_optimal_decompose_bwd
)

# ==============================================================================
# 1. Pallas Kernel Implementation (Forward Pass Logic)
# ==============================================================================
def gdn_scan_kernel_tpu(
    w_ref, u_ref, q_ref, k_ref, v_ref, g_ref, beta_ref, h_init_ref, 
    o_ref, h_final_ref,
    # Hyperparameters
    num_chunks: int, chunk_size: int, key_dim: int, val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16
):
  """Forward kernel for the WY-Represented Gated Delta Network."""
  h = h_init_ref[0, 0].astype(jnp.float32)

  mask_val = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  large_neg = -1e30

  ones_1xk = jnp.ones((1, key_dim), dtype=jnp.float32)
  ones_1xc = jnp.ones((1, chunk_size), dtype=jnp.float32)
  ones_cx1 = jnp.ones((chunk_size, 1), dtype=jnp.float32)

  for i in range(num_chunks):
    w = w_ref[0, 0, i].astype(jnp.float32) 
    u = u_ref[0, 0, i].astype(jnp.float32) 
    q = q_ref[0, 0, i].astype(jnp.float32) 
    k = k_ref[0, 0, i].astype(jnp.float32) 
    g = g_ref[0, 0, i].astype(jnp.float32)

    g_exp = jnp.exp(g).reshape((chunk_size, 1))
    g_exp_2d = jnp.dot(g_exp, ones_1xk)
    q_g = q * g_exp_2d

    term1 = jnp.dot(q_g, h) 

    v_prime = jnp.dot(w, h)
    v_new = u - v_prime

    attn = jnp.dot(q, k.T)

    g_col = jnp.dot(g.reshape((chunk_size, 1)), ones_1xc)
    g_row = jnp.dot(ones_cx1, g.reshape((1, chunk_size)))
    g_diff = g_col - g_row

    g_diff_masked = g_diff * mask_val + (1.0 - mask_val) * large_neg
    attn_decay = jnp.exp(g_diff_masked)

    attn_i = attn * attn_decay * mask_val
    term2 = jnp.dot(attn_i, v_new)

    o_chunk = term1 + term2
    o_ref[0, 0, i] = o_chunk.astype(dtype)

    chunk_decay = jnp.exp(g[chunk_size - 1]) 

    vec = jnp.exp(g[chunk_size - 1] - g).reshape((chunk_size, 1))
    vec_2d = jnp.dot(vec, ones_1xk)
    k_decayed = k * vec_2d

    update = jnp.dot(k_decayed.T, v_new)
    h = h * chunk_decay + update

  h_final_ref[0, 0] = h.astype(dtype)


# ==============================================================================
# 2. Pallas Kernel Implementation (Backward Pass Logic)
# ==============================================================================
def gdn_backward_kernel_tpu(
    w_ref, u_ref, q_ref, k_ref, v_ref, g_ref, beta_ref, h_init_ref,
    grad_o_ref, grad_h_final_ref,
    grad_w_ref, grad_u_ref, grad_q_ref, grad_k_ref, grad_v_ref, grad_g_ref, grad_beta_ref, grad_h_init_ref,
    h_buffer_ref,
    # Hyperparameters
    num_chunks: int, chunk_size: int, key_dim: int, val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
):
  """Corrected Backward kernel for the WY-Represented Gated Delta Network."""
  h = h_init_ref[0, 0].astype(jnp.float32)
  ones_1xk = jnp.ones((1, key_dim), dtype=jnp.float32)

  # Phase 1: Forward Recompute
  for i in range(num_chunks):
    h_buffer_ref[i] = h 

    w = w_ref[0, 0, i].astype(jnp.float32) 
    u = u_ref[0, 0, i].astype(jnp.float32) 
    k = k_ref[0, 0, i].astype(jnp.float32) 
    g = g_ref[0, 0, i].astype(jnp.float32)

    v_prime = jnp.dot(w, h)
    v_new = u - v_prime

    chunk_decay = jnp.exp(g[chunk_size - 1])
    vec = jnp.exp(g[chunk_size - 1] - g).reshape((chunk_size, 1))
    vec_2d = jnp.dot(vec, ones_1xk)
    k_decayed = k * vec_2d

    update = jnp.dot(k_decayed.T, v_new)
    h = h * chunk_decay + update

  grad_h = grad_h_final_ref[0, 0].astype(jnp.float32)

  mask_val = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  large_neg = -1e30

  ones_kx1 = jnp.ones((key_dim, 1), dtype=jnp.float32)
  ones_1xc = jnp.ones((1, chunk_size), dtype=jnp.float32)
  ones_cx1 = jnp.ones((chunk_size, 1), dtype=jnp.float32)
  ones_vx1 = jnp.ones((val_dim, 1), dtype=jnp.float32)

  # Phase 2: Backward Scan
  for i in range(num_chunks - 1, -1, -1):
    w = w_ref[0, 0, i].astype(jnp.float32)
    u = u_ref[0, 0, i].astype(jnp.float32)
    q = q_ref[0, 0, i].astype(jnp.float32)
    k = k_ref[0, 0, i].astype(jnp.float32)
    g = g_ref[0, 0, i].astype(jnp.float32)
    v = v_ref[0, 0, i].astype(jnp.float32)
    grad_o = grad_o_ref[0, 0, i].astype(jnp.float32)

    h = h_buffer_ref[i] 

    # 1. Recompute Forward Vars
    g_exp = jnp.exp(g).reshape((chunk_size, 1))
    g_exp_2d = jnp.dot(g_exp, ones_1xk)
    q_g = q * g_exp_2d

    v_prime = jnp.dot(w, h)
    v_new = u - v_prime

    attn = jnp.dot(q, k.T)

    g_col = jnp.dot(g.reshape((chunk_size, 1)), ones_1xc)
    g_row = jnp.dot(ones_cx1, g.reshape((1, chunk_size)))
    g_diff = g_col - g_row

    g_diff_masked = g_diff * mask_val + (1.0 - mask_val) * large_neg
    attn_decay = jnp.exp(g_diff_masked)

    # Apply mask explicitly after decay
    attn_i = attn * attn_decay * mask_val 

    chunk_decay = jnp.exp(g[chunk_size - 1])
    vec = jnp.exp(g[chunk_size - 1] - g).reshape((chunk_size, 1))
    vec_2d = jnp.dot(vec, ones_1xk)
    k_decayed = k * vec_2d

    # 2. Output Gradients
    grad_term2 = grad_o
    grad_attn_inter = grad_o

    grad_q_g = jnp.dot(grad_attn_inter, h.T)
    grad_h_from_inter = jnp.dot(q_g.T, grad_attn_inter)

    grad_attn_i = jnp.dot(grad_term2, v_new.T)
    grad_v_new_from_term2 = jnp.dot(attn_i.T, grad_term2)

    # 3. State Gradients
    grad_h_prev_from_decay = grad_h * chunk_decay
    grad_chunk_decay = jnp.dot(ones_1xk, jnp.dot(grad_h * h, ones_vx1))[0, 0]

    grad_update_term = grad_h
    grad_k_decayed = jnp.dot(v_new, grad_update_term.T)
    grad_v_new_from_update = jnp.dot(k_decayed, grad_update_term)

    # 4. Delta Gradients
    grad_v_new = grad_v_new_from_term2 + grad_v_new_from_update
    grad_u = grad_v_new
    grad_v_prime = -grad_v_new

    grad_w = jnp.dot(grad_v_prime, h.T)
    grad_h_from_v_prime = jnp.dot(w.T, grad_v_prime)

    # 5. Accumulate grad_h for previous chunk
    grad_h_prev = (
        grad_h_prev_from_decay + grad_h_from_inter + grad_h_from_v_prime
    )

    # 6. q, k, g intra-chunk gradients
    grad_q = grad_q_g * g_exp_2d
    grad_g_from_q_g = jnp.dot(grad_q_g * q_g, ones_kx1).reshape((chunk_size,))

    grad_attn_i = grad_attn_i * mask_val
    grad_attn = grad_attn_i * attn_decay
    grad_attn_decay = grad_attn_i * attn

    grad_q += jnp.dot(grad_attn, k)
    grad_k = jnp.dot(grad_attn.T, q)

    # Fix: Remove redundant mask_val multiplication here
    grad_g_diff = grad_attn_decay * attn_decay

    grad_g_from_diff_0 = jnp.dot(ones_1xc, grad_g_diff).reshape((chunk_size,))
    grad_g_from_diff_1 = jnp.dot(grad_g_diff, ones_cx1).reshape((chunk_size,))
    grad_g_from_diff = grad_g_from_diff_1 - grad_g_from_diff_0

    grad_k += grad_k_decayed * vec_2d

    grad_g_diff_state = jnp.dot(grad_k_decayed * k_decayed, ones_kx1).reshape((
        chunk_size,
    ))
    grad_g_from_state_decay = -grad_g_diff_state

    grad_g_last_from_state_decay = jnp.dot(
        ones_1xc, grad_g_diff_state.reshape((chunk_size, 1))
    )[0, 0]

    mask_last = (jnp.arange(chunk_size) == (chunk_size - 1)).astype(jnp.float32)
    grad_g_last = mask_last * (
        grad_chunk_decay * chunk_decay + grad_g_last_from_state_decay
    )

    grad_g = (
        grad_g_from_q_g
        + grad_g_from_diff
        + grad_g_from_state_decay
        + grad_g_last
    )

    # --- Store Gradients ---
    grad_w_ref[0, 0, i] = grad_w.astype(dtype)
    grad_u_ref[0, 0, i] = grad_u.astype(dtype)
    grad_q_ref[0, 0, i] = grad_q.astype(dtype)
    grad_k_ref[0, 0, i] = grad_k.astype(dtype)
    grad_g_ref[0, 0, i] = grad_g.astype(jnp.float32)

    grad_v_ref[0, 0, i] = jnp.zeros_like(v).astype(dtype)
    grad_beta_ref[0, 0, i] = jnp.zeros_like(g).astype(dtype)

    grad_h = grad_h_prev

  grad_h_init_ref[0, 0] = grad_h.astype(dtype)


# ==============================================================================
# 3. Custom VJP Registration & Wrappers
# ==============================================================================
def _gdn_pallas_forward(w, u, q, k, v, g, beta, h_init):
  """Performs the forward pass of the Gated Delta Network using a Pallas kernel."""
  batch_size, num_heads, num_chunks, chunk_size, k_dim = k.shape
  _, _, _, _, dv = v.shape

  in_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, k_dim),
  )
  val_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, dv),
  )
  scalar_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size),
  )
  out_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, dv),
  )
  state_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, k_dim, dv)
  )

  kernel_fn = functools.partial(
      gdn_scan_kernel_tpu,
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      key_dim=k_dim,
      val_dim=dv,
      dtype=v.dtype,
  )

  out, h_final = pl.pallas_call(
      kernel_fn,
      out_shape=[
          jax.ShapeDtypeStruct(
              (batch_size, num_heads, num_chunks, chunk_size, dv), v.dtype
          ),
          jax.ShapeDtypeStruct((batch_size, num_heads, k_dim, dv), v.dtype)
      ],
      grid=(batch_size, num_heads),
      in_specs=[
          in_specs,
          val_specs,
          in_specs,
          in_specs,
          val_specs,
          scalar_specs,
          scalar_specs,
          state_spec,
      ],
      out_specs=[out_spec, state_spec],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'parallel')
      ),
  )(w, u, q, k, v, g, beta, h_init)

  return (out, h_final), (w, u, q, k, v, g, beta, h_init)


def _gdn_pallas_backward(residuals, grad_out_tuple):
  """Performs the backward pass for the Gated Delta Network using a Pallas kernel."""
  grad_out, grad_h_final = grad_out_tuple
  w, u, q, k, v, g, beta, h_init = residuals

  batch_size, num_heads, num_chunks, chunk_size, k_dim = k.shape
  _, _, _, _, dv = v.shape

  in_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, k_dim),
  )
  val_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, dv),
  )
  scalar_specs = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size),
  )
  state_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, k_dim, dv)
  )

  grad_out_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0, 0),
      block_shape=(1, 1, num_chunks, chunk_size, dv)
  )
  grad_state_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, k_dim, dv)
  )

  scratch_spec = pltpu.VMEM((num_chunks, k_dim, dv), jnp.float32)

  kernel_fn = functools.partial(
      gdn_backward_kernel_tpu,
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      key_dim=k_dim,
      val_dim=dv,
      dtype=v.dtype,
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(batch_size, num_heads),
      in_specs=[
          in_specs,
          val_specs,
          in_specs,
          in_specs,
          val_specs,
          scalar_specs,
          scalar_specs,
          state_spec,
          grad_out_spec,
          grad_state_spec,
      ],
      out_specs=[
          in_specs,
          val_specs,
          in_specs,
          in_specs,
          val_specs,
          scalar_specs,
          scalar_specs,
          state_spec,
      ],
      scratch_shapes=[scratch_spec],
  )

  grads = pl.pallas_call(
      kernel_fn,
      out_shape=[
          jax.ShapeDtypeStruct(w.shape, w.dtype),
          jax.ShapeDtypeStruct(u.shape, u.dtype),
          jax.ShapeDtypeStruct(q.shape, q.dtype),
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(v.shape, v.dtype),
          jax.ShapeDtypeStruct(g.shape, g.dtype),
          jax.ShapeDtypeStruct(beta.shape, beta.dtype),
          jax.ShapeDtypeStruct(h_init.shape, h_init.dtype),
      ],
      grid_spec=grid_spec,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'parallel')
      ),
  )(w, u, q, k, v, g, beta, h_init, grad_out, grad_h_final)

  return grads


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def gdn_pallas_layer(w, u, q, k, v, g, beta, h_init):
  res, _ = _gdn_pallas_forward(w, u, q, k, v, g, beta, h_init)
  return res

gdn_pallas_layer.defvjp(_gdn_pallas_forward, _gdn_pallas_backward)


# =========================================================================
# 4. High-Level Pallas Wrapper
# =========================================================================
def pallas_chunk_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
    initial_state: None | jax.Array = None,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, None | jax.Array]:
  """Pallas-accelerated version of Gated Delta Rule."""

  # STAGE 1: PREPARATION & PADDING
  initial_dtype = query.dtype
  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(
      compute_dtype
  )
  query = query * scale

  batch_size, seq_len, num_heads, k_dim = key.shape
  v_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:
    pad_fn = lambda x, val=0.0: jnp.pad(
        x,
        ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2),
        constant_values=val,
    )
    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  def to_chunk(x):
    return x.reshape(
        batch_size, num_chunks, chunk_size, num_heads, -1
    ).transpose(0, 1, 3, 2, 4)

  def to_chunk_scalar(x):
    return x.reshape(batch_size, num_chunks, chunk_size, num_heads).transpose(
        0, 1, 3, 2
    )

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  s_matrix = jnp.matmul(
      k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST
  )
  s_matrix = s_matrix.astype(jnp.float32)
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)
  s_matrix = s_matrix * jnp.exp(g_diff)
  s_matrix = jnp.where(mask, s_matrix, 0.0)

  # --- Pallas Exact Decomposition ---
  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  a_matrix = identity - s_matrix

  # Call the custom VJP-wrapped Pallas Kernel
  if mesh is not None:
    # Extract mesh axis names for partitioning
    axis_names = mesh.axis_names
    batch_axes = [ax for ax in ('data', 'fsdp', 'fsdp_transpose', 'expert') if ax in axis_names]
    batch_spec = tuple(batch_axes) if batch_axes else None
    head_axes = [ax for ax in ('tensor', 'model') if ax in axis_names]
    head_spec = tuple(head_axes) if head_axes else None

    # a_matrix shape: (batch_size, num_chunks, num_heads, chunk_size, chunk_size)
    a_matrix_spec = P(batch_spec, None, head_spec, None, None)

    sharded_decompose = shard_map(
        functools.partial(run_optimal_decompose, n_block_size=8, block_size=16),
        mesh=mesh,
        in_specs=a_matrix_spec,
        out_specs=a_matrix_spec,
        check_rep=False,
    )
    matrix_a = sharded_decompose(a_matrix)
  else:
    matrix_a = run_optimal_decompose(
        a_matrix,
        n_block_size=8,
        block_size=16
    )

  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(
      matrix_a, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
  )
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(matrix_a, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # STAGE 3: INTER-CHUNK RECURRENCE (Pallas Kernel + shard_map)
  w_p = w_chunks.transpose(0, 2, 1, 3, 4)
  u_p = u_chunks.transpose(0, 2, 1, 3, 4)
  q_p = q_c.transpose(0, 2, 1, 3, 4)
  k_p = k_c.transpose(0, 2, 1, 3, 4)
  v_p = v_c.transpose(0, 2, 1, 3, 4)
  g_p = g_cumsum.transpose(0, 2, 1, 3)
  beta_p = beta_c.transpose(0, 2, 1, 3)

  # Handle initial state
  if initial_state is None:
    h_init = jnp.zeros(
        (batch_size, num_heads, k_dim, v_dim), dtype=compute_dtype
    )
  else:
    h_init = initial_state.astype(compute_dtype)

  kernel_to_use = gdn_pallas_layer 

  # Invoke Kernel
  if mesh is not None:
    # Mesh Partitioning
    axis_names = mesh.axis_names
    batch_axes = [
        ax
        for ax in ('data', 'fsdp', 'fsdp_transpose', 'expert')
        if ax in axis_names
    ]
    batch_spec = tuple(batch_axes) if batch_axes else None
    head_axes = [ax for ax in ('tensor', 'model') if ax in axis_names]
    head_spec = tuple(head_axes) if head_axes else None

    in_specs = P(batch_spec, head_spec, None, None, None)
    scalar_specs = P(batch_spec, head_spec, None, None)
    state_spec = P(batch_spec, head_spec, None, None)

    sharded_gdn = shard_map(
        kernel_to_use,
        mesh=mesh,
        in_specs=(
            in_specs,
            in_specs,
            in_specs,
            in_specs,
            in_specs,
            scalar_specs,
            scalar_specs,
            state_spec,
        ),
        out_specs=(in_specs, state_spec),
        check_rep=False,
    )

    o_pallas, h_final = sharded_gdn(
        w_p, u_p, q_p, k_p, v_p, g_p, beta_p, h_init
    )
  else:
    # Single Device
    o_pallas, h_final = kernel_to_use(
        w_p, u_p, q_p, k_p, v_p, g_p, beta_p, h_init
    )

  o_chunks = o_pallas.transpose(0, 2, 3, 1, 4)

  # STAGE 4: FINALIZATION
  o = o_chunks.reshape(batch_size, -1, num_heads, v_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, h_final