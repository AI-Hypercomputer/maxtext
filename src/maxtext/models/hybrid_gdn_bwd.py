
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def _safe_dot(lhs, rhs, *args, **kwargs):
    kwargs['preferred_element_type'] = jnp.float32
    if 'dimension_numbers' in kwargs:
        return jax.lax.dot_general(lhs, rhs, *args, **kwargs).astype(lhs.dtype)
    else:
        return _safe_dot(lhs, rhs, *args, **kwargs).astype(lhs.dtype)

def invert_triangular_matrix(t: jax.Array, block_size: int = 16) -> jax.Array:
    n_v, chunk_size, _ = t.shape
    iota_r = jax.lax.broadcasted_iota(jnp.int32, (n_v, chunk_size, chunk_size), 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, (n_v, chunk_size, chunk_size), 2)
    inv_t = jnp.where(iota_r == iota_c, 1.0, 0.0).astype(t.dtype)
    def body_fun(i, inv_t_acc):
        row_mask = jnp.where(jnp.arange(chunk_size) == i, 1.0, 0.0).astype(t.dtype).reshape(1, chunk_size, 1)
        t_row = jnp.sum(t * row_mask, axis=1, keepdims=True)
        col_mask = (jnp.arange(chunk_size) < i).reshape(1, 1, chunk_size)
        t_row = jnp.where(col_mask, t_row, 0.0)
        new_row = -_safe_dot(
            t_row, inv_t_acc,
            dimension_numbers=(((2,), (1,)), ((0,), (0,)))
        )
        one_hot = jnp.where(jnp.arange(chunk_size) == i, 1.0, 0.0).astype(t.dtype)
        new_row = new_row + one_hot.reshape(1, 1, chunk_size)
        inv_t_acc = inv_t_acc * (1.0 - row_mask) + new_row * row_mask
        return inv_t_acc
    return jax.lax.fori_loop(1, chunk_size, body_fun, inv_t)
def fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int) -> jax.Array:
    dtype = x.dtype
    mask_dtype = jnp.int32
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim]
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)
def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm
def l2_norm_bwd(d_out: jax.Array, x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm_sq = jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps
    norm = jnp.sqrt(norm_sq)
    x_normed = x / norm
    dot = jnp.sum(d_out * x_normed, axis=-1, keepdims=True)
    d_x = (d_out - x_normed * dot) / norm
    return d_x

def get_kernel(chunk_size, n_kq, n_v, d_k, d_v):
    def kernel(
        qkv_padded_ref, b_ref, a_ref, d_out_ref,
        conv_weight_ref, conv_bias_ref, a_log_ref, dt_bias_ref,
        state_init_ref, d_state_final_ref,
        d_qkv_ref, d_b_ref, d_a_ref, d_state_out_ref,
        d_conv_weight_ref, d_conv_bias_ref, d_a_log_ref, d_dt_bias_ref, all_states_hbm_ref,
        qkv_db, b_db, a_db, d_out_db,
        d_qkv_db, d_b_db, d_a_db, d_state_out_scratch,
        state_vmem_db, state_read_db, init_state_scratch,
        sem_qkv_fwd, sem_b_fwd, sem_a_fwd, sem_state_fwd,
        sem_qkv_bwd, sem_b_bwd, sem_a_bwd, sem_dout_bwd,
        sem_dqkv_bwd, sem_db_bwd, sem_da_bwd, sem_state_bwd
    ):
        batch_idx = pl.program_id(0)

        dim_size = n_kq * d_k * 2 + n_v * d_v
        num_chunks = b_ref.shape[1] // chunk_size
        kernel_size = conv_weight_ref.shape[0]
        prev_kernel_size = kernel_size - 1
        v_per_kq_head = n_v // n_kq
        W = conv_weight_ref[...]
        conv_bias = conv_bias_ref[...]
        a_log = a_log_ref[...]
        dt_bias = dt_bias_ref[...]
        d_state_final = d_state_final_ref[batch_idx, ...]
        init_state = (
            d_state_final,
            jnp.zeros((prev_kernel_size, dim_size), dtype=jnp.float32),
            jnp.zeros_like(W),
            jnp.zeros_like(conv_bias),
            jnp.zeros_like(a_log),
            jnp.zeros_like(dt_bias)
        )
        # --- FWD Pass Prologue ---
        def fwd_prologue(_):
            pltpu.make_async_copy(qkv_padded_ref.at[batch_idx, pl.ds(0, chunk_size + prev_kernel_size), :], qkv_db.at[0, ...], sem_qkv_fwd.at[0]).start()
            pltpu.make_async_copy(b_ref.at[batch_idx, pl.ds(0, chunk_size), :], b_db.at[0, ...], sem_b_fwd.at[0]).start()
            pltpu.make_async_copy(a_ref.at[batch_idx, pl.ds(0, chunk_size), :], a_db.at[0, ...], sem_a_fwd.at[0]).start()
            return None
        jax.lax.cond(num_chunks > 0, fwd_prologue, lambda _: None, None)
        pltpu.sync_copy(state_init_ref.at[batch_idx, ...], init_state_scratch)
        pltpu.sync_copy(init_state_scratch, all_states_hbm_ref.at[batch_idx, 0, ...])
        state_0 = init_state_scratch[...]
        def fwd_body_fun(i, state_prev):
            chunk_idx = i
            db_idx = i % 2
            next_db_idx = (i + 1) % 2
            pl.semaphore_wait(sem_qkv_fwd.at[db_idx], 1)
            pl.semaphore_wait(sem_b_fwd.at[db_idx], 1)
            pl.semaphore_wait(sem_a_fwd.at[db_idx], 1)
            def start_next(_):
                pltpu.make_async_copy(qkv_padded_ref.at[batch_idx, pl.ds((chunk_idx + 1) * chunk_size, chunk_size + prev_kernel_size), :], qkv_db.at[next_db_idx, ...], sem_qkv_fwd.at[next_db_idx]).start()
                pltpu.make_async_copy(b_ref.at[batch_idx, pl.ds((chunk_idx + 1) * chunk_size, chunk_size), :], b_db.at[next_db_idx, ...], sem_b_fwd.at[next_db_idx]).start()
                pltpu.make_async_copy(a_ref.at[batch_idx, pl.ds((chunk_idx + 1) * chunk_size, chunk_size), :], a_db.at[next_db_idx, ...], sem_a_fwd.at[next_db_idx]).start()
                return None
            jax.lax.cond(chunk_idx + 1 < num_chunks, start_next, lambda _: None, None)
            def wait_prev_write(_):
                pl.semaphore_wait(sem_state_fwd.at[db_idx], 1)
                return None
            jax.lax.cond(chunk_idx >= 2, wait_prev_write, lambda _: None, None)
            qkv_ext = qkv_db[db_idx, ...]
            b_ext = b_db[db_idx, ...]
            a_ext = a_db[db_idx, ...]
            conv_out_base = jnp.zeros((chunk_size, dim_size), dtype=qkv_ext.dtype)
            for k in range(kernel_size):
                conv_out_base += qkv_ext[k : k + chunk_size, :] * W[k, :]
            conv_out = conv_out_base + conv_bias
            k_start = n_kq * d_k
            k_end = k_start + n_kq * d_k
            v_start = k_end
            v_end = v_start + n_v * d_v
            k_raw = conv_out[:, k_start:k_end].reshape(chunk_size, n_kq, d_k).swapaxes(0, 1)
            v_raw = conv_out[:, v_start:v_end].reshape(chunk_size, n_v, d_v).swapaxes(0, 1)
            k_large = l2_norm(k_raw)
            v_large = v_raw
            k_repeat = jnp.repeat(k_large, v_per_kq_head, axis=0)
            beta = jax.nn.sigmoid(b_ext).reshape(1, chunk_size, n_v)
            gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_ext + dt_bias)
            gating_log = gating_log.reshape(1, chunk_size, n_v)
            iota_r_cumsum = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
            iota_c_cumsum = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
            lower_tri_mask = jnp.where(iota_r_cumsum >= iota_c_cumsum, 1.0, 0.0)
            gating_log_2d = gating_log[0]
            g_cum_sum_log_2d = _safe_dot(lower_tri_mask.astype(gating_log_2d.dtype), gating_log_2d)
            g_cum_sum_log = g_cum_sum_log_2d.reshape(1, chunk_size, n_v)
            g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)[:n_v]
            beta_large = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)[:n_v]
            g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
            g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
            gating_map = jnp.exp(g_cum_sum_diff_log)
            gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
            gating_forward = jnp.exp(g_cum_sum_log)
            gating_last = gating_forward[:, -1:]
            iota_r = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 1)
            iota_c = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 2)
            identity_mask = iota_r == iota_c
            strictly_lower_mask = iota_r > iota_c
            gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)
            k_beta_repeat = k_repeat * beta_large
            beta_k_k_t = _safe_dot(k_beta_repeat, k_repeat, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            t = jnp.where(identity_mask, 1, gating_map_masked * beta_k_k_t)
            t_inv = invert_triangular_matrix(t)
            v_beta_large = v_large * beta_large
            k_beta_gating = k_beta_repeat * gating_forward
            merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
            merged_uw = _safe_dot(t_inv, merged_v_k, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            u, w = jnp.split(merged_uw, [d_v], axis=-1)
            ws = _safe_dot(w, state_prev, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            u_ws = u - ws
            state_new = _safe_dot(k_repeat * gating_backward, u_ws, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
            state_curr = state_prev * gating_last + state_new
            state_vmem_db[db_idx, ...] = state_curr
            pltpu.make_async_copy(state_vmem_db.at[db_idx, ...], all_states_hbm_ref.at[batch_idx, chunk_idx + 1, ...], sem_state_fwd.at[db_idx]).start()
            return state_curr
        _ = jax.lax.fori_loop(0, num_chunks, fwd_body_fun, state_0)
        # --- FWD Pass Epilogue ---
        def wait_last_writes(_):
            pl.semaphore_wait(sem_state_fwd.at[(num_chunks - 1) % 2], 1)
        def wait_second_last(_):
            pl.semaphore_wait(sem_state_fwd.at[(num_chunks - 2) % 2], 1)
            return None
        jax.lax.cond(num_chunks >= 2, wait_second_last, lambda _: None, None)
        return None
        jax.lax.cond(num_chunks >= 1, wait_last_writes, lambda _: None, None)
        # --- BWD Pass Prologue ---
        def bwd_prologue(_):
            chunk_idx_0 = num_chunks - 1
            pltpu.make_async_copy(qkv_padded_ref.at[batch_idx, pl.ds(chunk_idx_0 * chunk_size, chunk_size + prev_kernel_size), :], qkv_db.at[0, ...], sem_qkv_bwd.at[0]).start()
            pltpu.make_async_copy(b_ref.at[batch_idx, pl.ds(chunk_idx_0 * chunk_size, chunk_size), :], b_db.at[0, ...], sem_b_bwd.at[0]).start()
            pltpu.make_async_copy(a_ref.at[batch_idx, pl.ds(chunk_idx_0 * chunk_size, chunk_size), :], a_db.at[0, ...], sem_a_bwd.at[0]).start()
            pltpu.make_async_copy(d_out_ref.at[batch_idx, pl.ds(chunk_idx_0 * chunk_size, chunk_size), :], d_out_db.at[0, ...], sem_dout_bwd.at[0]).start()
            pltpu.make_async_copy(all_states_hbm_ref.at[batch_idx, chunk_idx_0, ...], state_read_db.at[0, ...], sem_state_bwd.at[0]).start()
            return None
        jax.lax.cond(num_chunks > 0, bwd_prologue, lambda _: None, None)
        def bwd_body_fun(i, loop_state):
            chunk_idx = num_chunks - 1 - i
            db_idx = i % 2
            next_db_idx = (i + 1) % 2
            d_state_next, d_Y_next, d_W_acc, d_B_acc, d_a_log_acc, d_dt_bias_acc = loop_state
            pl.semaphore_wait(sem_qkv_bwd.at[db_idx], 1)
            pl.semaphore_wait(sem_b_bwd.at[db_idx], 1)
            pl.semaphore_wait(sem_a_bwd.at[db_idx], 1)
            pl.semaphore_wait(sem_dout_bwd.at[db_idx], 1)
            pl.semaphore_wait(sem_state_bwd.at[db_idx], 1)
            def start_next(_):
                next_chunk_idx = chunk_idx - 1
                pltpu.make_async_copy(qkv_padded_ref.at[batch_idx, pl.ds(next_chunk_idx * chunk_size, chunk_size + prev_kernel_size), :], qkv_db.at[next_db_idx, ...], sem_qkv_bwd.at[next_db_idx]).start()
                pltpu.make_async_copy(b_ref.at[batch_idx, pl.ds(next_chunk_idx * chunk_size, chunk_size), :], b_db.at[next_db_idx, ...], sem_b_bwd.at[next_db_idx]).start()
                pltpu.make_async_copy(a_ref.at[batch_idx, pl.ds(next_chunk_idx * chunk_size, chunk_size), :], a_db.at[next_db_idx, ...], sem_a_bwd.at[next_db_idx]).start()
                pltpu.make_async_copy(d_out_ref.at[batch_idx, pl.ds(next_chunk_idx * chunk_size, chunk_size), :], d_out_db.at[next_db_idx, ...], sem_dout_bwd.at[next_db_idx]).start()
                pltpu.make_async_copy(all_states_hbm_ref.at[batch_idx, next_chunk_idx, ...], state_read_db.at[next_db_idx, ...], sem_state_bwd.at[next_db_idx]).start()
                return None
            jax.lax.cond(i + 1 < num_chunks, start_next, lambda _: None, None)
            def wait_prev_write_bwd(_):
                pl.semaphore_wait(sem_dqkv_bwd.at[db_idx], 1)
                pl.semaphore_wait(sem_db_bwd.at[db_idx], 1)
                pl.semaphore_wait(sem_da_bwd.at[db_idx], 1)
                return None
            jax.lax.cond(i >= 2, wait_prev_write_bwd, lambda _: None, None)
            qkv_ext = qkv_db[db_idx, ...]
            b_ext = b_db[db_idx, ...]
            a_ext = a_db[db_idx, ...]
            d_out = d_out_db[db_idx, ...]
            state_prev = state_read_db[db_idx, ...]
            conv_out_base = jnp.zeros((chunk_size, dim_size), dtype=qkv_ext.dtype)
            for k in range(kernel_size):
                conv_out_base += qkv_ext[k : k + chunk_size, :] * W[k, :]
            conv_out = conv_out_base + conv_bias
            q_start = 0
            q_end = n_kq * d_k
            k_start = q_end
            k_end = k_start + n_kq * d_k
            v_start = k_end
            v_end = v_start + n_v * d_v
            q_raw = conv_out[:, q_start:q_end].reshape(chunk_size, n_kq, d_k).swapaxes(0, 1)
            k_raw = conv_out[:, k_start:k_end].reshape(chunk_size, n_kq, d_k).swapaxes(0, 1)
            v_raw = conv_out[:, v_start:v_end].reshape(chunk_size, n_v, d_v).swapaxes(0, 1)
            q_large = l2_norm(q_raw) * (d_k ** -0.5)
            k_large = l2_norm(k_raw)
            v_large = v_raw
            q_repeat = jnp.repeat(q_large, v_per_kq_head, axis=0)
            k_repeat = jnp.repeat(k_large, v_per_kq_head, axis=0)
            beta = jax.nn.sigmoid(b_ext).reshape(1, chunk_size, n_v)
            gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_ext + dt_bias)
            gating_log = gating_log.reshape(1, chunk_size, n_v)
            iota_r_cumsum = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
            iota_c_cumsum = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
            lower_tri_mask = jnp.where(iota_r_cumsum >= iota_c_cumsum, 1.0, 0.0)
            gating_log_2d = gating_log[0]
            g_cum_sum_log_2d = _safe_dot(
                lower_tri_mask.astype(gating_log_2d.dtype),
                gating_log_2d
            )
            g_cum_sum_log = g_cum_sum_log_2d.reshape(1, chunk_size, n_v)
            g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)[:n_v]
            beta_large = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)[:n_v]
            g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
            g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
            gating_map = jnp.exp(g_cum_sum_diff_log)
            gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
            gating_forward = jnp.exp(g_cum_sum_log)
            gating_last = gating_forward[:, -1:]
            iota_r = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 1)
            iota_c = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 2)
            identity_mask = iota_r == iota_c
            strictly_lower_mask = iota_r > iota_c
            lower_mask = iota_r >= iota_c
            gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)
            k_beta_repeat = k_repeat * beta_large
            beta_k_k_t = _safe_dot(k_beta_repeat, k_repeat, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            gating_beta_k_k_t = gating_map_masked * beta_k_k_t
            t = jnp.where(identity_mask, 1, gating_beta_k_k_t)
            t_inv = invert_triangular_matrix(t)
            v_beta_large = v_large * beta_large
            k_beta_gating = k_beta_repeat * gating_forward
            merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
            merged_uw = _safe_dot(t_inv, merged_v_k, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            u, w = jnp.split(merged_uw, [d_v], axis=-1)
            q_large_gating = q_repeat * gating_forward
            merged_w_q = jnp.concat([w, q_large_gating], axis=1)
            merged_ws_out_updated = _safe_dot(merged_w_q, state_prev, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
            u_ws = u - ws
            k_repeat_gating = k_repeat * gating_backward
            out_qk_raw = _safe_dot(q_large, k_large, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            out_qk_raw_repeated = jnp.repeat(out_qk_raw, v_per_kq_head, axis=0)
            out_qk = out_qk_raw_repeated * gating_map
            out_qk = jnp.where(lower_mask, out_qk, 0)
            d_out_new = d_out.reshape(chunk_size, n_v, d_v).swapaxes(0, 1)
            d_out_updated = d_out_new
            d_out_qk = _safe_dot(d_out_new, u_ws, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            d_out_qk = jnp.where(lower_mask, d_out_qk, 0)
            d_u_ws = _safe_dot(out_qk, d_out_new, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
            d_state_new = d_state_next
            d_k_repeat_gating = _safe_dot(d_state_new, u_ws, dimension_numbers=(((2,), (2,)), ((0,), (0,)))).swapaxes(1, 2)
            d_u_ws += _safe_dot(k_repeat_gating, d_state_new, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_u = d_u_ws
            d_ws = -d_u_ws
            d_merged_ws_out_updated = jnp.concat([d_ws, d_out_updated], axis=1)
            d_merged_w_q = _safe_dot(d_merged_ws_out_updated, state_prev, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            d_w, d_q_large_gating = jnp.split(d_merged_w_q, 2, axis=1)
            d_state_prev = _safe_dot(merged_w_q, d_merged_ws_out_updated, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
            d_state_prev += d_state_next * gating_last
            d_merged_uw = jnp.concat([d_u, d_w], axis=-1)
            d_t_inv = _safe_dot(d_merged_uw, merged_v_k, dimension_numbers=(((2,), (2,)), ((0,), (0,))))
            d_merged_v_k = _safe_dot(t_inv, d_merged_uw, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
            d_v_beta_large, d_k_beta_gating = jnp.split(d_merged_v_k, [d_v], axis=-1)
            t_inv_t = t_inv.swapaxes(1, 2)
            d_t = -_safe_dot(t_inv_t, _safe_dot(d_t_inv, t_inv_t, dimension_numbers=(((2,), (1,)), ((0,), (0,)))) , dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_t = jnp.where(strictly_lower_mask, d_t, 0)
            d_gating_beta_k_k_t = d_t
            d_beta_k_k_t = d_gating_beta_k_k_t * gating_map_masked
            d_k_beta_repeat = _safe_dot(d_beta_k_k_t, k_repeat, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_k_repeat = _safe_dot(d_beta_k_k_t.swapaxes(1, 2), k_beta_repeat, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_out_qk_unmasked = d_out_qk * gating_map
            d_out_qk_raw_repeated = d_out_qk_unmasked
            d_out_qk_raw = jnp.sum(d_out_qk_raw_repeated.reshape(n_kq, v_per_kq_head, chunk_size, chunk_size), axis=1)
            d_q_large = _safe_dot(d_out_qk_raw, k_large, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_k_large = _safe_dot(d_out_qk_raw.swapaxes(1, 2), q_large, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
            d_q_repeat = d_q_large_gating * gating_forward
            d_k_beta_repeat += d_k_beta_gating * gating_forward
            d_k_repeat += d_k_beta_repeat * beta_large
            d_k_repeat += d_k_repeat_gating * gating_backward
            d_q_large += jnp.sum(d_q_repeat.reshape(n_kq, v_per_kq_head, chunk_size, d_k), axis=1)
            d_k_large += jnp.sum(d_k_repeat.reshape(n_kq, v_per_kq_head, chunk_size, d_k), axis=1)
            d_v_large = d_v_beta_large * beta_large
            d_q_raw = l2_norm_bwd(d_q_large * (d_k ** -0.5), q_raw)
            d_k_raw = l2_norm_bwd(d_k_large, k_raw)
            d_v_raw = d_v_large
            d_q_raw = d_q_raw.swapaxes(0, 1).reshape(chunk_size, n_kq * d_k)
            d_k_raw = d_k_raw.swapaxes(0, 1).reshape(chunk_size, n_kq * d_k)
            d_v_raw = d_v_raw.swapaxes(0, 1).reshape(chunk_size, n_v * d_v)
            d_Y = jnp.concat([d_q_raw, d_k_raw, d_v_raw], axis=-1)
            d_gating_map = d_gating_beta_k_k_t * beta_k_k_t
            d_gating_map += d_out_qk * out_qk_raw_repeated
            d_g_cum_sum_diff_log = d_gating_map * gating_map
            d_gating_backward = jnp.sum(d_k_repeat_gating * k_repeat, axis=2, keepdims=True)
            d_g_cum_sum_diff_log_last = d_g_cum_sum_diff_log[..., -1:] - d_gating_backward * gating_backward
            d_g_cum_sum_diff_log = jnp.concatenate([d_g_cum_sum_diff_log[..., :-1], d_g_cum_sum_diff_log_last], axis=-1)
            d_gating_forward = jnp.sum(d_q_large_gating * q_repeat, axis=2, keepdims=True)
            d_gating_forward += jnp.sum(d_k_beta_gating * k_beta_repeat, axis=2, keepdims=True)
            d_gating_last = jnp.sum(d_state_next * state_prev, axis=(1, 2), keepdims=True)
            d_gating_forward_last = d_gating_forward[:, -1:] + d_gating_last
            d_gating_forward = jnp.concatenate([d_gating_forward[:, :-1], d_gating_forward_last], axis=1)
            d_g_cum_sum_log = d_gating_forward * gating_forward
            d_g_cum_sum_log += jnp.sum(d_g_cum_sum_diff_log, axis=2, keepdims=True)
            d_g_cum_sum_log -= jnp.sum(d_g_cum_sum_diff_log, axis=1, keepdims=True).swapaxes(1, 2)
            d_g_cum_sum_log_orig = d_g_cum_sum_log.swapaxes(0, 2)
            iota_r_rev = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
            iota_c_rev = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
            upper_tri_mask_rev = jnp.where(iota_r_rev <= iota_c_rev, 1.0, 0.0)
            d_g_cum_sum_log_2d = d_g_cum_sum_log_orig[0]
            d_gating_log_2d = _safe_dot(
                upper_tri_mask_rev.astype(d_g_cum_sum_log_2d.dtype),
                d_g_cum_sum_log_2d
            )
            d_gating_log = d_gating_log_2d.reshape(1, chunk_size, n_v)
            d_a = d_gating_log * (-jnp.exp(a_log)) * jax.nn.sigmoid(a_ext + dt_bias)
            d_a = d_a.reshape(chunk_size, n_v)
            d_beta_large = jnp.sum(d_k_beta_repeat * k_repeat, axis=2, keepdims=True)
            d_beta_large += jnp.sum(d_v_beta_large * v_large, axis=2, keepdims=True)
            d_beta = d_beta_large.swapaxes(0, 2)
            d_b = d_beta * beta * (1 - beta)
            d_b = d_b.reshape(chunk_size, n_v)
            d_a_log_chunk = jnp.sum(d_gating_log * gating_log, axis=(0, 1))
            d_dt_bias_chunk = jnp.sum(d_a, axis=0)
            d_Y_ext = jnp.concatenate([d_Y, d_Y_next], axis=0)
            d_X = jnp.zeros((chunk_size, dim_size), dtype=jnp.float32)
            for k in range(kernel_size):
                start_idx = kernel_size - 1 - k
                d_Y_slice = d_Y_ext[start_idx : start_idx + chunk_size, :]
                d_X += d_Y_slice * W[k]
            d_W_list = []
            for k in range(kernel_size):
                X_slice = qkv_ext[k : k + chunk_size, :]
                d_W_k = jnp.sum(d_Y * X_slice, axis=0)
                d_W_list.append(d_W_k)
            d_W_chunk = jnp.stack(d_W_list, axis=0)
            d_B_chunk = jnp.sum(d_Y, axis=0)
            d_qkv_db[db_idx, ...] = d_X
            d_b_db[db_idx, ...] = d_b
            d_a_db[db_idx, ...] = d_a
            pltpu.make_async_copy(d_qkv_db.at[db_idx, ...], d_qkv_ref.at[batch_idx, pl.ds(chunk_idx * chunk_size, chunk_size), :], sem_dqkv_bwd.at[db_idx]).start()
            pltpu.make_async_copy(d_b_db.at[db_idx, ...], d_b_ref.at[batch_idx, pl.ds(chunk_idx * chunk_size, chunk_size), :], sem_db_bwd.at[db_idx]).start()
            pltpu.make_async_copy(d_a_db.at[db_idx, ...], d_a_ref.at[batch_idx, pl.ds(chunk_idx * chunk_size, chunk_size), :], sem_da_bwd.at[db_idx]).start()
            return (
                d_state_prev,
                d_Y[:prev_kernel_size],
                d_W_acc + d_W_chunk,
                d_B_acc + d_B_chunk,
                d_a_log_acc + d_a_log_chunk,
                d_dt_bias_acc + d_dt_bias_chunk
            )
        final_state = jax.lax.fori_loop(0, num_chunks, bwd_body_fun, init_state)
        d_state_init, _, d_W_total, d_B_total, d_a_log_total, d_dt_bias_total = final_state
        # --- BWD Pass Epilogue ---
        def wait_last_writes_bwd(_):
            pl.semaphore_wait(sem_dqkv_bwd.at[(num_chunks - 1) % 2], 1)
            pl.semaphore_wait(sem_db_bwd.at[(num_chunks - 1) % 2], 1)
            pl.semaphore_wait(sem_da_bwd.at[(num_chunks - 1) % 2], 1)
        def wait_second_last_bwd(_):
            pl.semaphore_wait(sem_dqkv_bwd.at[(num_chunks - 2) % 2], 1)
            pl.semaphore_wait(sem_db_bwd.at[(num_chunks - 2) % 2], 1)
            pl.semaphore_wait(sem_da_bwd.at[(num_chunks - 2) % 2], 1)
            return None
        jax.lax.cond(num_chunks >= 2, wait_second_last_bwd, lambda _: None, None)
        return None
        jax.lax.cond(num_chunks >= 1, wait_last_writes_bwd, lambda _: None, None)
        d_conv_weight_ref[...] = d_W_total
        d_conv_bias_ref[...] = d_B_total
        d_a_log_ref[...] = d_a_log_total
        d_dt_bias_ref[...] = d_dt_bias_total
        d_state_out_scratch[...] = d_state_init
        pltpu.sync_copy(d_state_out_scratch, d_state_out_ref.at[batch_idx, ...])

    return kernel

def computation(
    qkv: jax.Array, b: jax.Array, a: jax.Array, d_out: jax.Array,
    conv_weight: jax.Array, conv_bias: jax.Array,
    a_log: jax.Array, dt_bias: jax.Array, state_init: jax.Array, d_state_final: jax.Array,
    chunk_size: int, n_kq: int, n_v: int, d_k: int, d_v: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    batch_size, num_chunks, _, dim_size = qkv.shape
    seq_len = num_chunks * chunk_size
    kernel_size = conv_weight.shape[0]
    prev_kernel_size = kernel_size - 1

    qkv_flat = qkv.reshape(batch_size, seq_len, dim_size)
    b_flat = b.reshape(batch_size, seq_len, n_v)
    a_flat = a.reshape(batch_size, seq_len, n_v)
    d_out_flat = d_out.reshape(batch_size, seq_len, n_v * d_v)

    qkv_padded = jnp.pad(qkv_flat, ((0, 0), (prev_kernel_size, 0), (0, 0)))

    d_qkv_shape = jax.ShapeDtypeStruct(qkv_flat.shape, qkv_flat.dtype)
    d_b_shape = jax.ShapeDtypeStruct(b_flat.shape, b_flat.dtype)
    d_a_shape = jax.ShapeDtypeStruct(a_flat.shape, a_flat.dtype)
    d_state_shape = jax.ShapeDtypeStruct(state_init.shape, state_init.dtype)
    d_conv_weight_shape = jax.ShapeDtypeStruct(conv_weight.shape, conv_weight.dtype)
    d_conv_bias_shape = jax.ShapeDtypeStruct(conv_bias.shape, conv_bias.dtype)
    d_a_log_shape = jax.ShapeDtypeStruct(a_log.shape, a_log.dtype)
    d_dt_bias_shape = jax.ShapeDtypeStruct(dt_bias.shape, dt_bias.dtype)
    all_states_shape = jax.ShapeDtypeStruct((batch_size, num_chunks + 1, n_v, d_k, d_v), state_init.dtype)

    grid = (batch_size,)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec()

    scratch_shapes = (
        pltpu.VMEM(shape=(2, chunk_size + prev_kernel_size, dim_size), dtype=qkv.dtype),
        pltpu.VMEM(shape=(2, chunk_size, n_v), dtype=b.dtype),
        pltpu.VMEM(shape=(2, chunk_size, n_v), dtype=a.dtype),
        pltpu.VMEM(shape=(2, chunk_size, n_v * d_v), dtype=d_out.dtype),
        pltpu.VMEM(shape=(2, chunk_size, dim_size), dtype=qkv.dtype),
        pltpu.VMEM(shape=(2, chunk_size, n_v), dtype=b.dtype),
        pltpu.VMEM(shape=(2, chunk_size, n_v), dtype=a.dtype),
        pltpu.VMEM(shape=(n_v, d_k, d_v), dtype=state_init.dtype),
        pltpu.VMEM(shape=(2, n_v, d_k, d_v), dtype=state_init.dtype),
        pltpu.VMEM(shape=(2, n_v, d_k, d_v), dtype=state_init.dtype),
        pltpu.VMEM(shape=(n_v, d_k, d_v), dtype=state_init.dtype),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SemaphoreType.REGULAR((2,)),
    )

    res = pl.pallas_call(
        get_kernel(chunk_size, n_kq, n_v, d_k, d_v),
        out_shape=(d_qkv_shape, d_b_shape, d_a_shape, d_state_shape, d_conv_weight_shape, d_conv_bias_shape, d_a_log_shape, d_dt_bias_shape, all_states_shape),
        grid=grid,
        in_specs=[hbm_spec] * 4 + [vmem_spec] * 4 + [hbm_spec, vmem_spec],
        out_specs=(hbm_spec, hbm_spec, hbm_spec, hbm_spec, vmem_spec, vmem_spec, vmem_spec, vmem_spec, hbm_spec),
        scratch_shapes=scratch_shapes,
    )(qkv_padded, b_flat, a_flat, d_out_flat, conv_weight, conv_bias, a_log, dt_bias, state_init, d_state_final)

    d_qkv_flat, d_b_flat, d_a_flat, d_state_init_out, d_conv_weight, d_conv_bias, d_a_log, d_dt_bias, _ = res
    d_qkv = d_qkv_flat.reshape(batch_size, seq_len, dim_size)
    d_b = d_b_flat.reshape(batch_size, seq_len, n_v)
    d_a = d_a_flat.reshape(batch_size, seq_len, n_v)

    return d_qkv, d_b, d_a, d_state_init_out, d_conv_weight, d_conv_bias, d_a_log, d_dt_bias

import functools
from typing import Optional, Tuple
from maxtext.models.hybrid_gdn import pure_jax_fused_conv1d_gdn

@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
def hybrid_bwd_fused_conv1d_gdn(
    qkv: jax.Array, b: jax.Array, a: jax.Array, conv_weight: jax.Array, conv_bias: Optional[jax.Array],
    a_log: jax.Array, dt_bias: jax.Array, conv_state: Optional[jax.Array], recurrent_state: Optional[jax.Array],
    num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int,
    conv_kernel_size: int, chunk_size: int, use_qk_norm_in_gdn: bool, compute_dtype: jnp.dtype,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], jax.Array]:
    return pure_jax_fused_conv1d_gdn(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
        num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size, chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm_in_gdn, compute_dtype=compute_dtype,
    )

def _hybrid_bwd_fwd(
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
    num_k_heads, num_v_heads, head_k_dim, head_v_dim,
    conv_kernel_size, chunk_size, use_qk_norm_in_gdn, compute_dtype
):
    out, states, tap_out = pure_jax_fused_conv1d_gdn(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
        num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size, chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm_in_gdn, compute_dtype=compute_dtype,
    )
    residuals = (qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state)
    return (out, states, tap_out), residuals

def _hybrid_bwd_bwd(
    num_k_heads, num_v_heads, head_k_dim, head_v_dim,
    conv_kernel_size, chunk_size, use_qk_norm_in_gdn, compute_dtype,
    residuals, cotangents
):
    (qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state) = residuals
    d_out, d_states, d_tap = cotangents
    d_conv_state, d_recurrent_state = d_states

    batch_size, seq_len, dim_size = qkv.shape
    num_chunks = seq_len // chunk_size

    if recurrent_state is None:
        recurrent_state = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=qkv.dtype)
    if d_recurrent_state is None:
        d_recurrent_state = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=qkv.dtype)

    # Convert to expected shapes
    # Pure JAX expects conv_weight shape: (kernel_size, 1, dim_size)
    # The kernel expects (kernel_size, dim_size)
    conv_weight_k = conv_weight.reshape(conv_kernel_size, dim_size)
    
    if conv_bias is None:
        conv_bias = jnp.zeros((dim_size,), dtype=qkv.dtype)
        
    qkv_reshaped = qkv.reshape(batch_size, num_chunks, chunk_size, dim_size)
    b_reshaped = b.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
    a_reshaped = a.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
    d_out_reshaped = d_out.reshape(batch_size, num_chunks, chunk_size, num_v_heads * head_v_dim)
    
    d_qkv, d_b, d_a, d_state_init_out, d_conv_weight, d_conv_bias, d_a_log, d_dt_bias = computation(
        qkv_reshaped, b_reshaped, a_reshaped, d_out_reshaped,
        conv_weight_k, conv_bias, a_log, dt_bias, recurrent_state, d_recurrent_state,
        chunk_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    
    d_conv_weight = d_conv_weight.reshape(conv_kernel_size, 1, dim_size)
    
    return d_qkv, d_b, d_a, d_conv_weight, d_conv_bias, d_a_log, d_dt_bias, jnp.zeros_like(conv_state) if conv_state is not None else None, d_state_init_out

hybrid_bwd_fused_conv1d_gdn.defvjp(_hybrid_bwd_fwd, _hybrid_bwd_bwd)
