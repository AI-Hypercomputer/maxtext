# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks the GatedDeltaNet cross-device state composition on a CPU mesh.

Runs as a subprocess so the forced device count takes effect before JAX
initializes; the parent pytest process has already initialized JAX with the
default device count.

The child checks three things, each on a 1-D context mesh and a 2-D fsdp x
context mesh, and the first also with a tuple axis name, which is what
`qwen3.py` passes when more than one context knob is set.

  - Forward. `gdn_cp.incoming_state` against a sequential single-device
    composition of the same affine maps.
  - Backward. `gdn_cp` defines no custom_vjp, so the backward pass is whatever
    JAX's transpose rules for `lax.scan`, `lax.ppermute` and `lax.psum` compose
    into. A `ppermute` transposes to the inverse permutation, so a cotangent
    has to travel back down the rank chain for an early shard to see any
    gradient from a late one.
  - End to end. `jax_chunk_gated_delta_rule` itself, sharded against unsharded,
    which catches a wrong chunk layout or a wrong replay.

Every one of these fails silently: the loss still falls while the state
entering each shard is wrong. So each check has a negative control that
confirms the comparison can fail.
"""

import os
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.kernels.attention import gdn_cp


@pytest.mark.cpu_only
def test_gdn_cp_prefix_scan_matches_sequential_reference_on_cpu_mesh():
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "GDN_CP_CHECKS_PASSED" in result.stdout


def _chunk_inputs(num_chunks, lead, chunk_len, k_dim, v_dim):
  """Random w, u, k and g in the layout compose_local scans over."""
  keys = jax.random.split(jax.random.PRNGKey(1), 4)
  w = jax.random.normal(keys[0], (num_chunks,) + lead + (chunk_len, k_dim), jnp.float32) * 0.1
  u = jax.random.normal(keys[1], (num_chunks,) + lead + (chunk_len, v_dim), jnp.float32) * 0.1
  k = jax.random.normal(keys[2], (num_chunks,) + lead + (chunk_len, k_dim), jnp.float32) * 0.1
  # Log-decay gates are negative and increase along the chunk, as in Qwen3-Next.
  ramp = jnp.arange(chunk_len, dtype=jnp.float32) / chunk_len
  g = -jax.nn.softplus(jax.random.normal(keys[3], (num_chunks,) + lead + (chunk_len,), jnp.float32)) * (1.0 - ramp)
  return w, u, k, g


def _direct_recurrence(w, u, k, g, h):
  """Applies the inter-chunk step chunk by chunk, in its original form.

  This reference never builds A or B and never calls compose, so it checks the
  affine reformulation the whole module rests on rather than restating it.
  """
  for i in range(w.shape[0]):
    w_c, u_c, k_c, g_c = w[i], u[i], k[i], g[i]
    g_last = g_c[..., -1]
    k_g_T = (k_c * jnp.exp(g_last[..., None] - g_c)[..., None]).swapaxes(-1, -2)
    h = jnp.exp(g_last)[..., None, None] * h + jnp.matmul(k_g_T, u_c - jnp.matmul(w_c, h))
  return h


def test_compose_local_folds_the_chunks_into_one_affine_map():
  """A_loc @ h + B_loc equals running the recurrence chunk by chunk."""
  lead, k_dim, v_dim = (1, 2), 8, 8
  w, u, k, g = _chunk_inputs(num_chunks=6, lead=lead, chunk_len=4, k_dim=k_dim, v_dim=v_dim)
  h_init = jax.random.normal(jax.random.PRNGKey(2), lead + (k_dim, v_dim), jnp.float32) * 0.1

  affine_a, affine_b = jax.jit(gdn_cp.compose_local)(w, u, k, g)
  folded = jnp.matmul(affine_a, h_init) + affine_b
  np.testing.assert_allclose(jax.device_get(folded), jax.device_get(_direct_recurrence(w, u, k, g, h_init)), atol=1e-5)

  # Negative control: reversing the chunks changes the answer, so the test sees order.
  reversed_chunks = _direct_recurrence(w[::-1], u[::-1], k[::-1], g[::-1], h_init)
  assert float(jnp.max(jnp.abs(folded - reversed_chunks))) > 1e-4


def test_incoming_state_is_the_identity_on_a_single_device_mesh():
  """With one shard there is nothing before it, so it enters with h_init."""
  mesh = Mesh(np.array(jax.devices()[:1]), ("context",))
  batch, heads, k_dim, v_dim = 1, 2, 8, 8
  keys = jax.random.split(jax.random.PRNGKey(3), 3)
  affine_a = jax.random.normal(keys[0], (1, batch, heads, k_dim, k_dim), jnp.float32) * (0.3 / k_dim**0.5)
  affine_b = jax.random.normal(keys[1], (1, batch, heads, k_dim, v_dim), jnp.float32) * 0.1
  h_init = jax.random.normal(keys[2], (batch, heads, k_dim, v_dim), jnp.float32) * 0.1

  mapped = partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(P("context", None, None, None, None), P("context", None, None, None, None), P(None, None, None, None)),
      out_specs=(P("context", None, None, None), P(None, None, None, None)),
      check_vma=False,
  )
  incoming, final = jax.jit(mapped(lambda a, b, h: gdn_cp.incoming_state(a[0], b[0], h, "context")))(
      affine_a, affine_b, h_init
  )
  np.testing.assert_allclose(jax.device_get(incoming.reshape(h_init.shape)), jax.device_get(h_init), atol=1e-6)
  expected_final = jnp.matmul(affine_a[0], h_init) + affine_b[0]
  np.testing.assert_allclose(jax.device_get(final), jax.device_get(expected_final), atol=1e-5)


def _sequential_reference(affine_a, affine_b, h_init):
  """Device i enters with every pair before it composed and applied to h_init."""
  incoming = []
  a_run = jnp.broadcast_to(jnp.eye(affine_a.shape[-1], dtype=affine_a.dtype), affine_a.shape[1:])
  b_run = jnp.zeros(affine_b.shape[1:], affine_b.dtype)
  for i in range(affine_a.shape[0]):
    incoming.append(jnp.matmul(a_run, h_init) + b_run)
    a_run, b_run = gdn_cp.compose((a_run, b_run), (affine_a[i], affine_b[i]))
  return jnp.stack(incoming), jnp.matmul(a_run, h_init) + b_run


def _run_composition_checks(mesh, cp_axis):
  """Runs the prefix scan and the negative control on one mesh."""
  axis_names = (cp_axis,) if isinstance(cp_axis, str) else cp_axis
  num_shards = int(np.prod([mesh.shape[name] for name in axis_names]))
  batch, heads, k_dim, v_dim = 1, 2, 8, 8
  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  # A contractive A keeps the composition over every shard finite.
  affine_a = jax.random.normal(keys[0], (num_shards, batch, heads, k_dim, k_dim), jnp.float32) * (0.3 / k_dim**0.5)
  affine_b = jax.random.normal(keys[1], (num_shards, batch, heads, k_dim, v_dim), jnp.float32) * 0.1
  h_init = jax.random.normal(keys[2], (batch, heads, k_dim, v_dim), jnp.float32) * 0.1

  def sharded(fn):
    mapped = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(cp_axis, None, None, None, None), P(cp_axis, None, None, None, None), P(None, None, None, None)),
        out_specs=(P(cp_axis, None, None, None), P(None, None, None, None)),
        check_vma=False,
    )
    return jax.jit(mapped(lambda a, b, h: fn(a[0], b[0], h, cp_axis)))(affine_a, affine_b, h_init)

  reference_incoming, reference_final = _sequential_reference(affine_a, affine_b, h_init)
  # shard_map concatenates each shard's (batch, heads, k_dim, v_dim) along the
  # batch axis, so at batch 1 the leading axis of the result is the shard index.
  incoming, final = sharded(gdn_cp.incoming_state)
  incoming = incoming.reshape(reference_incoming.shape)
  np.testing.assert_allclose(jax.device_get(incoming), jax.device_get(reference_incoming), atol=1e-5)
  np.testing.assert_allclose(jax.device_get(final), jax.device_get(reference_final), atol=1e-5)

  # Negative control: break the chain so every shard starts from h_init.
  broken, _ = sharded(lambda a, b, h, axis: (h, h))
  assert float(jnp.max(jnp.abs(broken.reshape(reference_incoming.shape) - reference_incoming))) > 1e-4


def _run_gradient_checks(mesh, cp_axis):
  """The backward pass, which the module never writes by hand.

  `gdn_cp` defines no custom_vjp, so the backward pass is whatever JAX's own
  transpose rules for `lax.scan`, `lax.ppermute` and `lax.psum` compose into.
  That is worth a test rather than an assumption: a `ppermute` transposes to the
  inverse permutation, so a cotangent has to travel back down the rank chain for
  an early shard to see any gradient from a late one at all. Getting it wrong is
  silent in the same way a wrong forward prefix is.
  """
  axis_names = (cp_axis,) if isinstance(cp_axis, str) else cp_axis
  num_shards = int(np.prod([mesh.shape[name] for name in axis_names]))
  batch, heads, k_dim, v_dim = 1, 2, 8, 8
  keys = jax.random.split(jax.random.PRNGKey(4), 3)
  affine_a = jax.random.normal(keys[0], (num_shards, batch, heads, k_dim, k_dim), jnp.float32) * (0.3 / k_dim**0.5)
  affine_b = jax.random.normal(keys[1], (num_shards, batch, heads, k_dim, v_dim), jnp.float32) * 0.1
  h_init = jax.random.normal(keys[2], (batch, heads, k_dim, v_dim), jnp.float32) * 0.1

  def cp_loss(a, b, h):
    mapped = jax.shard_map(
        lambda a_s, b_s, h_s: gdn_cp.incoming_state(a_s[0], b_s[0], h_s, cp_axis),
        mesh=mesh,
        in_specs=(P(cp_axis, None, None, None, None), P(cp_axis, None, None, None, None), P(None, None, None, None)),
        out_specs=(P(cp_axis, None, None, None), P(None, None, None, None)),
        check_vma=False,
    )
    incoming, final = mapped(a, b, h)
    return jnp.sum(incoming**2) + jnp.sum(final**2)

  def reference_loss(a, b, h):
    incoming, final = _sequential_reference(a, b, h)
    return jnp.sum(incoming**2) + jnp.sum(final**2)

  grad_cp = jax.jit(jax.grad(cp_loss, argnums=(0, 1, 2)))(affine_a, affine_b, h_init)
  grad_reference = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2)))(affine_a, affine_b, h_init)
  for actual, expected in zip(grad_cp, grad_reference):
    np.testing.assert_allclose(jax.device_get(actual), jax.device_get(expected), atol=1e-5, rtol=1e-4)

  # Every shard sees gradient. The last shard's pair reaches the first shard's
  # cotangent only through the reverse ppermute, so a zero here means the
  # cross-device backward is not connected.
  per_shard = jnp.linalg.norm(grad_cp[0].reshape(num_shards, -1), axis=1)
  assert float(jnp.min(per_shard)) > 1e-6, per_shard


def _run_end_to_end_checks(mesh, cp_axis):
  """`jax_chunk_gated_delta_rule` itself, sharded against unsharded.

  The composition tests above check gdn_cp in isolation. This one runs the real
  GatedDeltaNet function over a sharded sequence and compares every output
  token, and the final state, against the same function on one device. It is
  the check that catches a wrong chunk layout or a wrong replay, neither of
  which the isolated tests can see.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.models.qwen3 import jax_chunk_gated_delta_rule

  num_shards = int(np.prod([mesh.shape[name] for name in ((cp_axis,) if isinstance(cp_axis, str) else cp_axis)]))
  batch, heads, k_dim, v_dim, chunk = 1, 2, 16, 16, 8
  seq = num_shards * chunk * 2  # two local chunks per shard
  keys = jax.random.split(jax.random.PRNGKey(5), 6)
  query = jax.random.normal(keys[0], (batch, seq, heads, k_dim), jnp.float32) * 0.1
  key_t = jax.random.normal(keys[1], (batch, seq, heads, k_dim), jnp.float32) * 0.1
  value = jax.random.normal(keys[2], (batch, seq, heads, v_dim), jnp.float32) * 0.1
  g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq, heads), jnp.float32))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq, heads), jnp.float32))
  h_init = jax.random.normal(keys[5], (batch, heads, k_dim, v_dim), jnp.float32) * 0.1

  def call(cp):
    return jax_chunk_gated_delta_rule(
        query, key_t, value, g, beta, chunk_size=chunk, initial_state=h_init, cp_axis=cp, compute_dtype=jnp.float32
    )

  reference_out, reference_state = jax.jit(lambda: call(None))()

  qkv_spec = P(None, cp_axis, None, None)
  g_spec = P(None, cp_axis, None)
  state_spec = P(None, None, None, None)

  def sharded_call(q, k, v, gg, bb, hh):
    return jax_chunk_gated_delta_rule(
        q, k, v, gg, bb, chunk_size=chunk, initial_state=hh, cp_axis=cp_axis, compute_dtype=jnp.float32
    )

  cp_out, cp_state = jax.jit(
      jax.shard_map(
          sharded_call,
          mesh=mesh,
          in_specs=(qkv_spec, qkv_spec, qkv_spec, g_spec, g_spec, state_spec),
          out_specs=(qkv_spec, state_spec),
          check_vma=False,
      )
  )(query, key_t, value, g, beta, h_init)

  np.testing.assert_allclose(jax.device_get(cp_out), jax.device_get(reference_out), atol=2e-4, rtol=2e-3)
  np.testing.assert_allclose(jax.device_get(cp_state), jax.device_get(reference_state), atol=2e-4, rtol=2e-3)

  # Negative control: shift the sequence by one shard. If the comparison were
  # insensitive to which tokens land on which device, this would still pass.
  rolled = jnp.roll(reference_out, shift=chunk, axis=1)
  assert float(jnp.max(jnp.abs(rolled - reference_out))) > 1e-4


def test_block_doubling_unit_lower_inverse_fwd_and_bwd():
  """Verify block-doubling inverse + analytical VJP matches solve_triangular in fwd and bwd."""
  # pylint: disable=import-outside-toplevel,protected-access
  from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
  from maxtext.models import qwen3

  for chunk_size in (1, 12, 16, 64):
    k = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, chunk_size, 16), dtype=jnp.float32)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    shared = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 4, 1, 16), dtype=jnp.float32)
    shared = shared / (jnp.linalg.norm(shared, axis=-1, keepdims=True) + 1e-6)
    k = 0.7 * shared + 0.3 * k
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    s_mat = jnp.tril(jnp.matmul(k * 0.5, k.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST), k=-1)
    dout = jax.random.normal(jax.random.PRNGKey(2), s_mat.shape, dtype=jnp.float32)

    def ref_fn(s, cs=chunk_size):
      eye = jnp.eye(cs, dtype=jnp.float32)
      return jax.scipy.linalg.solve_triangular(eye + s, jnp.broadcast_to(eye, s.shape), lower=True, unit_diagonal=True)

    ref_val, ref_grad = jax.jit(jax.value_and_grad(lambda s, r=ref_fn, d=dout: jnp.sum(r(s) * d)))(s_mat)

    for fn in (
        qwen3._block_doubling_unit_lower_inverse,
        modeling_qwen3_8_flash_next._block_doubling_unit_lower_inverse,
    ):
      out = jax.jit(lambda s, f=fn, cs=chunk_size: f(s, cs))(s_mat)
      np.testing.assert_allclose(jax.device_get(out), jax.device_get(ref_fn(s_mat)), rtol=1e-5, atol=1e-5)
      val, grad = jax.jit(jax.value_and_grad(lambda s, f=fn, cs=chunk_size, d=dout: jnp.sum(f(s, cs) * d)))(s_mat)
      np.testing.assert_allclose(jax.device_get(val), jax.device_get(ref_val), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(jax.device_get(grad), jax.device_get(ref_grad), rtol=1e-5, atol=1e-5)

      # bfloat16 input: the cotangent must come back in the input dtype (custom_vjp contract) and the
      # float32-internal inverse must still track the reference up to bf16 input rounding.
      s_bf16 = s_mat.astype(jnp.bfloat16)
      out_bf16 = jax.jit(lambda s, f=fn, cs=chunk_size: f(s, cs))(s_bf16)
      assert out_bf16.dtype == jnp.float32, out_bf16.dtype
      grad_bf16 = jax.jit(jax.grad(lambda s, f=fn, cs=chunk_size, d=dout: jnp.sum(f(s, cs) * d)))(s_bf16)
      assert grad_bf16.dtype == jnp.bfloat16, grad_bf16.dtype
      ref_grad_bf16 = jax.jit(jax.grad(lambda s, r=ref_fn, d=dout: jnp.sum(r(s.astype(jnp.float32)) * d)))(s_bf16)
      np.testing.assert_allclose(
          jax.device_get(grad_bf16).astype(np.float32),
          jax.device_get(ref_grad_bf16).astype(np.float32),
          rtol=2e-2,
          atol=2e-2,
      )


def test_gdn_gqa_no_repeat_fwd_and_bwd():
  """Verify GDN chunk rule with H_k < H_v matches explicit jnp.repeat in forward and backward."""
  # pylint: disable=import-outside-toplevel
  from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
  from maxtext.models import qwen3

  b, s, h_k, h_v, d_k, d_v, chunk = 2, 32, 2, 4, 16, 16, 8
  keys = jax.random.split(jax.random.PRNGKey(42), 6)
  query = jax.random.normal(keys[0], (b, s, h_k, d_k), jnp.float32) * 0.1
  key_t = jax.random.normal(keys[1], (b, s, h_k, d_k), jnp.float32) * 0.1
  value = jax.random.normal(keys[2], (b, s, h_v, d_v), jnp.float32) * 0.1
  g = -jax.nn.softplus(jax.random.normal(keys[3], (b, s, h_v), jnp.float32))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, s, h_v), jnp.float32))
  dout = jax.random.normal(keys[5], (b, s, h_v, d_v), jnp.float32)

  def run_qwen3(qq, kk, vv, gg, bb):
    out, _ = qwen3.jax_chunk_gated_delta_rule(
        qq, kk, vv, gg, bb, chunk_size=chunk, use_qk_norm_in_gdn=True, compute_dtype=jnp.float32
    )
    return jnp.sum(out * dout)

  def run_m3(qq, kk, vv, gg, bb):
    out, _ = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
        qq, kk, vv, gg, bb, chunk_size=chunk, use_qk_l2norm_in_kernel=True
    )
    return jnp.sum(out * dout)

  for fn in (run_qwen3, run_m3):
    v_rep, g_rep = jax.jit(
        lambda qq, kk, vv, gg, bb, f=fn: jax.value_and_grad(f, argnums=(0, 1, 2, 3, 4))(
            jnp.repeat(qq, 2, axis=2), jnp.repeat(kk, 2, axis=2), vv, gg, bb
        )
    )(query, key_t, value, g, beta)
    v_norep, g_norep = jax.jit(
        lambda qq, kk, vv, gg, bb, f=fn: jax.value_and_grad(f, argnums=(0, 1, 2, 3, 4))(qq, kk, vv, gg, bb)
    )(query, key_t, value, g, beta)
    np.testing.assert_allclose(jax.device_get(v_norep), jax.device_get(v_rep), rtol=1e-5, atol=1e-5)
    # Sum repeated head gradients back to H_k for query and key
    g_rep_k = (
        g_rep[0].reshape(b, s, h_k, 2, d_k).sum(axis=3),
        g_rep[1].reshape(b, s, h_k, 2, d_k).sum(axis=3),
        g_rep[2],
        g_rep[3],
        g_rep[4],
    )
    for actual, expected in zip(g_norep, g_rep_k):
      np.testing.assert_allclose(jax.device_get(actual), jax.device_get(expected), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  _devices = np.array(jax.devices())
  assert len(_devices) == 8, jax.devices()
  _run_composition_checks(Mesh(_devices, ("context",)), "context")
  _run_composition_checks(Mesh(_devices.reshape(2, 4), ("fsdp", "context")), "context")
  # qwen3.py builds cp_axis as a tuple, so both context knobs can be live at once.
  _run_composition_checks(
      Mesh(_devices.reshape(2, 4), ("context", "context_usp_ulysses")),
      ("context", "context_usp_ulysses"),
  )
  _run_gradient_checks(Mesh(_devices, ("context",)), "context")
  _run_gradient_checks(Mesh(_devices.reshape(2, 4), ("fsdp", "context")), "context")
  _run_end_to_end_checks(Mesh(_devices, ("context",)), "context")
  test_block_doubling_unit_lower_inverse_fwd_and_bwd()
  test_gdn_gqa_no_repeat_fwd_and_bwd()
  print("GDN_CP_CHECKS_PASSED")
