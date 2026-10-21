# Copyright 2026 Google LLC
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
"""Tests for JAX ring attention."""

from __future__ import annotations

import functools
import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
from jax import random  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from maxtext.kernels.attention import jax_ring_attention  # pylint: disable=g-import-not-at-top
from maxtext.kernels.tokamax_splash_attention import base  # pylint: disable=g-import-not-at-top
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as mask_lib  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top

P = jax.sharding.PartitionSpec
partial = functools.partial


def _create_segment_ids(seq_len: int) -> base.SegmentIds:
  """Creates deterministic segment IDs with two packed segments."""
  half = seq_len // 2
  seg = jnp.concatenate(
      [
          jnp.zeros((half,), dtype=jnp.int32),
          jnp.ones((seq_len - half,), dtype=jnp.int32),
      ],
      axis=0,
  )
  return jax_ring_attention.SegmentIds(q=seg, kv=seg)


class JaxRingAttentionTest(parameterized.TestCase):
  """Tests for JAX ring attention forward, backward, and kernel wrapper."""

  def _assert_allclose(
      self,
      actual: jax.Array | np.ndarray,
      expected: jax.Array | np.ndarray,
      *,
      rtol: float,
      atol: float,
  ) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )

  def test_make_ring_attention_and_pytree_roundtrip(self) -> None:
    mask = mask_lib.make_causal_mask((64, 64))
    config = types.SimpleNamespace(
        block_q=32,
        block_kv=32,
        attn_logits_soft_cap=50.0,
    )
    kernel = jax_ring_attention.make_ring_attention(
        mask,
        config=config,
        ring_axis="ring",
        q_seq_shards=2,
        kv_seq_shards=2,
    )
    leaves, treedef = jax.tree_util.tree_flatten(kernel)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(restored, jax_ring_attention.RingJaxAttentionKernel)
    spec_kernel = restored.manual_sharding_spec()
    spec_leaves, _ = jax.tree_util.tree_flatten(spec_kernel)
    self.assertEqual(spec_leaves, [P()])

  def test_make_ring_attention_rejects_non_2d_mask(self) -> None:
    bad_mask = jnp.ones((2, 16, 16), dtype=jnp.bool_)
    with self.assertRaisesRegex(ValueError, "Unexpected mask shape"):
      jax_ring_attention.make_ring_attention(bad_mask, ring_axis="ring")

  @parameterized.product(
      ring_size=[2],
      num_heads=[2],
      head_dim=[32],
      dtype=[jnp.bfloat16],
      is_mqa=[False, True],
      is_segmented=[False, True],
      mask_type=["FULL", "CAUSAL"],
  )
  def test_ring_attention_kernel(
      self,
      ring_size: int,
      num_heads: int,
      head_dim: int,
      dtype: jax.typing.DTypeLike,
      is_mqa: bool,
      is_segmented: bool,
      mask_type: str,
  ) -> None:
    if len(jax.devices()) < ring_size:
      self.skipTest(f"This test requires {ring_size} devices, but has only" f" {len(jax.devices())} devices available.")

    ring_axis = "ring"
    devices = np.asarray(jax.devices()[:ring_size]).reshape(1, ring_size)
    mesh = jax.sharding.Mesh(devices, ("heads", ring_axis))
    seq_len = 64 * ring_size

    k1, k2, k3, k4 = random.split(random.key(0), 4)
    scale = head_dim**-0.5
    q = random.normal(k1, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    if is_mqa:
      k = random.normal(k2, (seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (seq_len, head_dim), dtype=dtype) * scale
    else:
      k = random.normal(k2, (num_heads, seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    do = random.normal(k4, q.shape, dtype=dtype) * scale

    if mask_type == "CAUSAL":
      mask = mask_lib.make_causal_mask((seq_len, seq_len))
    elif mask_type == "FULL":
      mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    else:
      raise ValueError(f"Unsupported mask type: {mask_type}")

    if is_segmented:
      segment_ids = _create_segment_ids(seq_len)
      segment_ids_spec = jax_ring_attention.SegmentIds(q=P(ring_axis), kv=P(ring_axis))
    else:
      segment_ids = segment_ids_spec = None

    q_spec = P(None, ring_axis, None)
    kv_spec = P(ring_axis, None) if is_mqa else q_spec

    ring_kernel = jax_ring_attention.make_ring_attention(
        mask,
        block_q=32,
        block_kv=32,
        is_mqa=is_mqa,
        ring_axis=ring_axis,
        q_seq_shards=ring_size,
        kv_seq_shards=ring_size,
    )
    kernel_spec = ring_kernel.manual_sharding_spec()

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
            segment_ids_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def ring_attn_fn(
        kernel: jax_ring_attention.RingJaxAttentionKernel,
        q_s: jax.Array,
        k_s: jax.Array,
        v_s: jax.Array,
        seg_s: jax_ring_attention.SegmentIds | None,
    ) -> jax.Array:
      return kernel(q_s, k_s, v_s, seg_s)

    ring_attn_ref = partial(base.attention_reference, is_mqa=is_mqa)

    with self.subTest("fwd"):
      out = ring_attn_fn(ring_kernel, q, k, v, segment_ids)
      out_ref = ring_attn_ref(q, k, v, mask[:, :], segment_ids)
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

    with self.subTest("bwd"):
      out, out_vjp = jax.vjp(ring_attn_fn, ring_kernel, q, k, v, segment_ids)
      out_ref, out_vjp_ref = jax.vjp(ring_attn_ref, q, k, v, mask[:, :], segment_ids)
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

      _, dq, dk, dv, _ = out_vjp(do)
      dq_ref, dk_ref, dv_ref, _, _ = out_vjp_ref(do.astype(jnp.float32))

      self._assert_allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dv, dv_ref, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      ring_size=[2],
      num_heads=[2],
      head_dim=[32],
      dtype=[jnp.bfloat16],
      is_mqa=[False, True],
      mask_type=["FULL", "CAUSAL"],
  )
  def test_ring_attention_functional(
      self,
      ring_size: int,
      num_heads: int,
      head_dim: int,
      dtype: jax.typing.DTypeLike,
      is_mqa: bool,
      mask_type: str,
  ) -> None:
    if len(jax.devices()) < ring_size:
      self.skipTest(f"This test requires {ring_size} devices, but has only" f" {len(jax.devices())} devices available.")

    ring_axis = "ring"
    devices = np.asarray(jax.devices()[:ring_size]).reshape(1, ring_size)
    mesh = jax.sharding.Mesh(devices, ("heads", ring_axis))
    seq_len = 64 * ring_size

    k1, k2, k3, k4 = random.split(random.key(0), 4)
    scale = head_dim**-0.5
    q = random.normal(k1, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    if is_mqa:
      k = random.normal(k2, (seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (seq_len, head_dim), dtype=dtype) * scale
    else:
      k = random.normal(k2, (num_heads, seq_len, head_dim), dtype=dtype) * scale
      v = random.normal(k3, (num_heads, seq_len, head_dim), dtype=dtype) * scale
    do = random.normal(k4, q.shape, dtype=dtype) * scale

    if mask_type == "CAUSAL":
      mask = mask_lib.make_causal_mask((seq_len, seq_len))
    elif mask_type == "FULL":
      mask = mask_lib.FullMask(_shape=(seq_len, seq_len))
    else:
      raise ValueError(f"Unsupported mask type: {mask_type}")

    q_spec = P(None, ring_axis, None)
    kv_spec = P(ring_axis, None) if is_mqa else q_spec

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            P(),
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def ring_attn_fn(
        q_s: jax.Array,
        k_s: jax.Array,
        v_s: jax.Array,
        mask_arr: jax.Array,
    ) -> jax.Array:
      return jax_ring_attention.ring_attention(
          q_s,
          k_s,
          v_s,
          mask=mask_arr,
          block_q=32,
          block_kv=32,
          axis_name=ring_axis,
      )

    ring_attn_ref = partial(base.attention_reference, is_mqa=is_mqa)

    with self.subTest("fwd"):
      out = ring_attn_fn(q, k, v, mask[:, :])
      out_ref = ring_attn_ref(q, k, v, mask[:, :])
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

    with self.subTest("bwd"):
      out, out_vjp = jax.vjp(ring_attn_fn, q, k, v, mask[:, :])
      out_ref, out_vjp_ref = jax.vjp(ring_attn_ref, q, k, v, mask[:, :])
      self._assert_allclose(out, out_ref, rtol=5e-3, atol=3e-3)

      dq, dk, dv, _ = out_vjp(do)
      dq_ref, dk_ref, dv_ref, _ = out_vjp_ref(do.astype(jnp.float32))

      self._assert_allclose(dq, dq_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dk, dk_ref, rtol=1e-2, atol=1e-2)
      self._assert_allclose(dv, dv_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  absltest.main()
