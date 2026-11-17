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


def _run_conv_halo_checks(mesh, cp_axis):
  """Verifies unpacked `cp_causal_depthwise_conv1d` forward + backward against single-device conv."""
  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.attention import gdn_packing

  cp_axes = (cp_axis,) if isinstance(cp_axis, str) else cp_axis
  num_shards = int(np.prod([mesh.shape[name] for name in cp_axes]))
  batch, seq_per_rank, channels, kernel_size = 2, 6, 16, 4
  seq = num_shards * seq_per_rank
  keys = jax.random.split(jax.random.PRNGKey(11), 3)
  qkv = jax.random.normal(keys[0], (batch, seq, channels), jnp.float32) * 0.2
  conv_w = jax.random.normal(keys[1], (kernel_size, 1, channels), jnp.float32) * 0.2
  conv_b = jax.random.normal(keys[2], (channels,), jnp.float32) * 0.1

  def ref_conv(x, w, b):
    padded = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    out = sum(padded[:, k : k + seq, :] * w[k, 0, :] for k in range(kernel_size)) + b
    return out

  def cp_conv(x, w, b):
    return gdn_packing.cp_causal_depthwise_conv1d(
        qkv=x,
        conv_weight=w,
        conv_bias=b,
        kernel_size=kernel_size,
        dtype=jnp.float32,
        mesh=mesh,
        cp_axes=cp_axes,
    )

  ref_out = jax.jit(ref_conv)(qkv, conv_w, conv_b)
  _, ref_grads = jax.jit(jax.value_and_grad(lambda x, w, b: jnp.sum(jnp.sin(ref_conv(x, w, b))), argnums=(0, 1, 2)))(
      qkv, conv_w, conv_b
  )
  cp_out = jax.jit(cp_conv)(qkv, conv_w, conv_b)
  _, cp_grads = jax.jit(jax.value_and_grad(lambda x, w, b: jnp.sum(jnp.sin(cp_conv(x, w, b))), argnums=(0, 1, 2)))(
      qkv, conv_w, conv_b
  )
  np.testing.assert_allclose(jax.device_get(cp_out), jax.device_get(ref_out), atol=1e-5, rtol=1e-5)
  for actual_g, expected_g in zip(cp_grads, ref_grads):
    np.testing.assert_allclose(jax.device_get(actual_g), jax.device_get(expected_g), atol=1e-5, rtol=1e-5)


def _run_qwen3_gdn_layer_cp_checks(devices):
  """Verifies `Qwen3NextGatedDeltaNet` forward + backward across cp=1, seq-CP, and head-CP."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.models.qwen3 import Qwen3NextGatedDeltaNet
  from maxtext.utils import maxtext_utils

  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
  common_overrides = {
      "run_name": "gdn_cp_layer_test",
      "decoder_block": "qwen3_next",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": 32,
      "gdn_num_key_heads": 4,
      "gdn_num_value_heads": 8,
      "gdn_key_head_dim": 16,
      "gdn_value_head_dim": 16,
      "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 4,
      "use_qk_norm_in_gdn": True,
      "context_parallel_load_balance": False,
      "enable_checkpointing": False,
  }

  batch, seq = 2, 16
  x = jax.random.normal(jax.random.PRNGKey(21), (batch, seq, 32), jnp.float32) * 0.2
  # Row 0: left-padded single sequence (for unpacked) or multi-doc (for packed).
  unpacked_seg = jnp.array(
      [
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      ],
      dtype=jnp.int32,
  )
  packed_seg = jnp.array(
      [
          [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0],
      ],
      dtype=jnp.int32,
  )

  def _eval_layer(cfg, mesh, seg_ids, use_custom_remat=False):
    rules = cfg.logical_axis_rules
    with nn_partitioning.axis_rules(rules):
      layer = Qwen3NextGatedDeltaNet(
          config=cfg, mesh=mesh, dtype=jnp.float32, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(7)
      )
      graphdef, state = nnx.split(layer)
      policy = None
      if use_custom_remat:
        save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg)
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=save_names,
            names_which_can_be_offloaded=offload_names,
            offload_src="device",
            offload_dst="pinned_host",
        )

      def forward_only(st, inp):
        mod = nnx.merge(graphdef, st)
        out, _ = mod(inp, decoder_segment_ids=seg_ids, model_mode=MODEL_MODE_TRAIN)
        return out

      run_fn = jax.remat(forward_only, policy=policy) if policy is not None else forward_only

      def loss_fn(st, inp):
        out = run_fn(st, inp)
        return jnp.sum(out * jnp.cos(inp)), out

      (_, out), (grad_state, grad_x) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(state, x)
      return jax.device_get(out), jax.tree_util.tree_map(jax.device_get, grad_state), jax.device_get(grad_x)

  for packing, seg_ids in ((False, unpacked_seg), (True, packed_seg)):
    cfg_ref = pyconfig.initialize([None, base_yml], packing=packing, **common_overrides)
    ref_out, ref_g_state, ref_g_x = _eval_layer(cfg_ref, None, seg_ids)

    # 1D 4-device context mesh: both seq-CP and head-CP.
    mesh_cp4 = Mesh(devices[:4], ("context",))
    for cp_mode in ("seq", "head"):
      cfg_cp4 = pyconfig.initialize(
          [None, base_yml],
          packing=packing,
          ici_context_parallelism=4,
          gdn_cp_mode=cp_mode,
          remat_policy="custom",
          gdn_states="device",
          gdn_cp_state="device",
          **common_overrides,
      )
      out, g_state, g_x = _eval_layer(cfg_cp4, mesh_cp4, seg_ids, use_custom_remat=cp_mode == "seq")
      np.testing.assert_allclose(out, ref_out, atol=3e-4, rtol=2e-3)
      np.testing.assert_allclose(g_x, ref_g_x, atol=5e-4, rtol=3e-3)
      for g_actual, g_expected in zip(jax.tree_util.tree_leaves(g_state), jax.tree_util.tree_leaves(ref_g_state)):
        np.testing.assert_allclose(g_actual, g_expected, atol=5e-4, rtol=3e-3)

    # 2D (tensor=2, context=2) mesh: both seq-CP and head-CP.
    mesh_tp2_cp2 = Mesh(devices[:4].reshape(2, 2), ("tensor", "context"))
    for cp_mode in ("seq", "head"):
      cfg_tp_cp = pyconfig.initialize(
          [None, base_yml],
          packing=packing,
          ici_tensor_parallelism=2,
          ici_context_parallelism=2,
          gdn_cp_mode=cp_mode,
          **common_overrides,
      )
      out, g_state, g_x = _eval_layer(cfg_tp_cp, mesh_tp2_cp2, seg_ids)
      np.testing.assert_allclose(out, ref_out, atol=3e-4, rtol=2e-3)
      np.testing.assert_allclose(g_x, ref_g_x, atol=5e-4, rtol=3e-3)
      for g_actual, g_expected in zip(jax.tree_util.tree_leaves(g_state), jax.tree_util.tree_leaves(ref_g_state)):
        np.testing.assert_allclose(g_actual, g_expected, atol=5e-4, rtol=3e-3)


def test_gdn_cp_config_and_custom_remat_validation():
  """Validates `gdn_cp_mode`, DCN CP sizing, and disjoint GDN custom remat expansions."""
  # pylint: disable=import-outside-toplevel
  from types import SimpleNamespace
  from maxtext.configs import pyconfig
  from maxtext.configs import types as config_types
  from maxtext.layers.nnx_decoders import NNXDecoder
  from maxtext.utils import maxtext_utils

  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")

  # ICI + DCN + Ulysses context axes multiply into gdn_context_parallel_size and validate head divisibility.
  with pytest.raises(ValueError, match="divisible by the GatedDeltaNet context-parallel size"):
    pyconfig.initialize(
        [None, base_yml],
        run_name="bad_head_cp",
        decoder_block="qwen3_next",
        gdn_num_key_heads=6,
        gdn_num_value_heads=12,
        ici_context_parallelism=4,
        context_parallel_load_balance=False,
        gdn_cp_mode="head",
        enable_checkpointing=False,
    )

  with pytest.raises(ValueError, match="divisible by the GatedDeltaNet context-parallel size"):
    config_types.MaxTextConfig(
        run_name="bad_dcn_head_cp",
        decoder_block="qwen3_next",
        gdn_num_key_heads=6,
        gdn_num_value_heads=12,
        num_slices=2,
        ici_context_parallelism=2,
        dcn_context_parallelism=2,
        context_parallel_load_balance=False,
        gdn_cp_mode="head",
        enable_checkpointing=False,
    )

  with pytest.raises(ValueError, match="divisible by the GatedDeltaNet context-parallel size"):
    config_types.RLConfig(
        run_name="bad_rl_head_cp",
        decoder_block="qwen3_next",
        gdn_num_key_heads=6,
        gdn_num_value_heads=12,
        ici_context_usp_ulysses_parallelism=4,
        context_parallel_load_balance=False,
        gdn_cp_mode="head",
        enable_checkpointing=False,
    )

  # Conflicting `gdn` and `gdn_states` placements fail fast.
  with pytest.raises(ValueError, match="cannot place 'gdn' and 'gdn_states' in different storage tiers"):
    pyconfig.initialize(
        [None, base_yml],
        run_name="conflict_remat",
        decoder_block="qwen3_next",
        remat_policy="custom",
        gdn="device",
        gdn_states="offload",
        enable_checkpointing=False,
    )

  # Disjoint `gdn_states='offload'` + `gdn_cp_state='device'` expands cleanly and builds NNXDecoder remat policy.
  cfg_disjoint = pyconfig.initialize(
      [None, base_yml],
      run_name="disjoint_remat",
      decoder_block="qwen3_next",
      remat_policy="custom",
      gdn="remat",
      gdn_states="offload",
      gdn_cp_state="device",
      enable_checkpointing=False,
  )
  save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg_disjoint)
  assert set(save_names).isdisjoint(set(offload_names))
  assert {"gdn_cp_conv_halo", "gdn_cp_m_local", "gdn_cp_s_in"}.issubset(set(save_names))
  assert {"gdn_core_attn_out", "gdn_t_inv", "gdn_chunk_states"}.issubset(set(offload_names))
  assert "gdn_qkv" not in save_names and "gdn_qkv" not in offload_names

  policy = NNXDecoder.get_remat_policy(SimpleNamespace(config=cfg_disjoint))
  assert callable(policy)


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
  _run_conv_halo_checks(Mesh(_devices[:4], ("context",)), "context")
  _run_conv_halo_checks(Mesh(_devices.reshape(2, 4), ("tensor", "context")), "context")
  _run_qwen3_gdn_layer_cp_checks(_devices)
  print("GDN_CP_CHECKS_PASSED")
