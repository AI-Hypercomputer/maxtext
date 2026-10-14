# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gdn_recurrence_impl='fused' vs the fp32 jax reference (CPU interpret mode).

Bounds are relative to max|ref|. The bf16 jax path itself is at ~0.6% (out)
and ~0.7-1.1% (grads) on these inputs, so 1.5% / 2% allow bf16 rounding only.
B=1 because Pallas interpret mode cannot be differentiated under the
jax.checkpoint of the per-sequence lax.map (B>1).
"""

import jax
import jax.numpy as jnp
import pytest

from maxtext.models import qwen3


def _inputs(t, h, d=128, seed=1, h_k=None):
  h_k = h if h_k is None else h_k
  ks = jax.random.split(jax.random.PRNGKey(seed), 6)
  q = jax.random.normal(ks[0], (1, t, h_k, d))
  k = jax.random.normal(ks[1], (1, t, h_k, d))
  v = jax.random.normal(ks[2], (1, t, h, d))
  g = -jax.nn.softplus(jax.random.normal(ks[3], (1, t, h)) + 2.0) * 0.5
  beta = jax.nn.sigmoid(jax.random.normal(ks[4], (1, t, h)))
  do = jax.random.normal(ks[5], (1, t, h, d))
  return (q, k, v, g, beta), do


def _run(args, do, impl, cdt, hb, pallas_batch_map=False):
  """Runs forward and VJP of jax_chunk_gated_delta_rule_fast for testing."""

  def f(q, k, v, g, beta):
    o, _ = qwen3.jax_chunk_gated_delta_rule_fast(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        use_qk_norm_in_gdn=True,
        compute_dtype=cdt,
        recurrence_impl=impl,
        inverse_precision="highest",
        pallas_batch_map=pallas_batch_map,
        pallas_heads_per_block=hb,
    )
    return o.astype(jnp.float32)

  o, vjp = jax.vjp(f, *args)
  return o, vjp(do)


def _rel(a, b):
  return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.mark.parametrize("t,h,hb,h_k", [(256, 4, 4, 4), (320, 8, 4, 8), (256, 8, 8, 4)])
def test_fused_matches_reference(t, h, hb, h_k):
  args, do = _inputs(t, h, h_k=h_k)
  ref_o, ref_g = _run(args, do, "jax", jnp.float32, hb)
  o, grads = _run(args, do, "fused", jnp.bfloat16, hb)
  assert _rel(o, ref_o) < 0.015
  for name, a, b in zip("q k v g beta".split(), grads, ref_g):
    assert bool(jnp.all(jnp.isfinite(a))), name
    assert _rel(a, b) < 0.02, (name, _rel(a, b))

  o2, grads2 = _run(args, do, "fused", jnp.bfloat16, hb, pallas_batch_map=True)
  assert _rel(o2, ref_o) < 0.015
  for name, a, b in zip("q k v g beta".split(), grads2, ref_g):
    assert bool(jnp.all(jnp.isfinite(a))), name
    assert _rel(a, b) < 0.02, (name, _rel(a, b))


def test_grouped_conv1d_matches_ungrouped():
  ks = jax.random.split(jax.random.PRNGKey(7), 3)
  batch, seq_len, num_k_heads, head_k_dim, head_v_dim, v_per_k = 2, 32, 4, 16, 16, 2
  key_dim = num_k_heads * head_k_dim
  value_dim = num_k_heads * v_per_k * head_v_dim
  conv_kernel_size = 4
  qkv_dim_per_k = 2 * head_k_dim + v_per_k * head_v_dim

  mixed_qkvz = jax.random.normal(ks[0], (batch, seq_len, num_k_heads, qkv_dim_per_k + v_per_k * head_v_dim))
  conv_w = jax.random.normal(ks[1], (conv_kernel_size, 1, 2 * key_dim + value_dim))
  seg_ids = jnp.ones((batch, seq_len), dtype=jnp.int32).at[:, -4:].set(0)

  def ungrouped_fn(m_qkvz, w):
    q0, k0, v_raw0, _ = jnp.split(m_qkvz, [head_k_dim, 2 * head_k_dim, qkv_dim_per_k], axis=3)
    q = q0.reshape(batch, seq_len, -1)
    k = k0.reshape(batch, seq_len, -1)
    v = v_raw0.reshape(batch, seq_len, -1)
    qkv = jnp.concatenate([q, k, v], axis=-1)
    conv_in = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
    conv_out = jax.lax.conv_general_dilated(
        lhs=conv_in,
        rhs=w,
        window_strides=(1,),
        padding=((conv_kernel_size - 1, 0),),
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=qkv.shape[-1],
    )[:, -seq_len:, :]
    qkv_conv = jax.nn.silu(conv_out)
    q_c, k_c, v_c = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)
    query = q_c.reshape(batch, seq_len, num_k_heads, head_k_dim)
    key = k_c.reshape(batch, seq_len, num_k_heads, head_k_dim)
    value = v_c.reshape(batch, seq_len, num_k_heads * v_per_k, head_v_dim)
    mask = seg_ids != 0
    key = jnp.where(mask[..., None, None], key, 0.0)
    value = jnp.where(mask[..., None, None], value, 0.0)
    return jnp.sum(query * 1.3 + key * 0.7) + jnp.sum(value * 0.5)

  def grouped_fn(m_qkvz, w):
    qkv_grouped, _ = jnp.split(m_qkvz, [qkv_dim_per_k], axis=3)
    qkv = qkv_grouped.reshape(batch, seq_len, -1)
    w_q, w_k, w_v = jnp.split(w, [key_dim, 2 * key_dim], axis=-1)
    w_grouped = jnp.concatenate(
        [
            w_q.reshape(conv_kernel_size, 1, num_k_heads, head_k_dim),
            w_k.reshape(conv_kernel_size, 1, num_k_heads, head_k_dim),
            w_v.reshape(conv_kernel_size, 1, num_k_heads, v_per_k * head_v_dim),
        ],
        axis=-1,
    ).reshape(conv_kernel_size, 1, -1)
    conv_out = jax.lax.conv_general_dilated(
        lhs=qkv,
        rhs=w_grouped,
        window_strides=(1,),
        padding=((conv_kernel_size - 1, 0),),
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=qkv.shape[-1],
    )
    qkv_conv = jax.nn.silu(conv_out)
    qkv_conv_4d = qkv_conv.reshape(batch, seq_len, num_k_heads, qkv_dim_per_k)
    query, key, value_raw = jnp.split(qkv_conv_4d, [head_k_dim, 2 * head_k_dim], axis=-1)
    mask = seg_ids != 0
    key = jnp.where(mask[..., None, None], key, 0.0)
    value_raw = jnp.where(mask[..., None, None], value_raw, 0.0)
    value = value_raw.reshape(batch, seq_len, num_k_heads * v_per_k, head_v_dim)
    return jnp.sum(query * 1.3 + key * 0.7) + jnp.sum(value * 0.5)

  v1, g1 = jax.value_and_grad(ungrouped_fn, argnums=(0, 1))(mixed_qkvz, conv_w)
  v2, g2 = jax.value_and_grad(grouped_fn, argnums=(0, 1))(mixed_qkvz, conv_w)
  assert jnp.allclose(v1, v2, atol=1e-5, rtol=1e-5)
  assert jnp.allclose(g1[0], g2[0], atol=1e-5, rtol=1e-5)
  assert jnp.allclose(g1[1], g2[1], atol=1e-5, rtol=1e-5)
