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

"""Unit tests for local SplashAttention FP8 and uniform-segment optimizations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import reference as base
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_kernel as splash
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_mask as mask_lib
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_mask_info as mask_info_lib


def _segment_ids(lengths: tuple[int, ...]) -> base.SegmentIds:
  ids = jnp.concatenate([jnp.full((n,), idx, jnp.int32) for idx, n in enumerate(lengths)])
  return base.SegmentIds(ids, ids)


class SplashAttentionOptTest(parameterized.TestCase):
  """Tests uniform-segment skipping, sparsity hints, and FP8 Q/K backward."""

  @parameterized.named_parameters(
      ("fwd", False),
      ("dkv", True),
  )
  def test_skip_uniform_segment_mask_promotes_only_uniform_tiles(self, is_dkv: bool):
    bq = bkv = 128
    lengths = (320, 192)
    seq_len = sum(lengths)
    ids = np.concatenate([np.full((n,), idx, np.int32) for idx, n in enumerate(lengths)])
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    process_fn = mask_info_lib.process_dynamic_mask_dkv if is_dkv else mask_info_lib.process_dynamic_mask
    mask_info = process_fn(jnp.asarray(mask[:, :]), (bq, bkv), downcast_smem_data=False)
    refined, _, _ = splash._refine_mask_info_with_segments(  # pylint: disable=protected-access
        mask_info,
        _segment_ids(lengths),
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
        is_dkv=is_dkv,
        skip_uniform_segment_mask=True,
    )
    num_active = int(np.asarray(refined.num_active_blocks)[0])
    rows = np.asarray(refined.active_rows)[:num_active]
    cols = np.asarray(refined.active_cols)[:num_active]
    bmask = np.asarray(refined.block_mask)[:num_active]
    tiles = {(int(r), int(c)): int(m) for r, c, m in zip(rows, cols, bmask, strict=True)}
    for (row_idx, col_idx), mask_val in tiles.items():
      if mask_val == 3:
        qi, kj = (col_idx, row_idx) if is_dkv else (row_idx, col_idx)
        qs = ids[qi * bq : (qi + 1) * bq]
        kvs = ids[kj * bkv : (kj + 1) * bkv]
        self.assertTrue(bool((qs[:, None] == kvs[None, :]).all()))

    interior_coord = (0, 1) if is_dkv else (1, 0)
    boundary_coord = (0, 2) if is_dkv else (2, 0)
    self.assertEqual(tiles[(0, 0)], 1)
    self.assertEqual(tiles[interior_coord], 3)
    self.assertEqual(tiles[boundary_coord], 2)

  def test_refine_mask_info_prunes_disjoint_segment_blocks(self):
    seq_len, bq, bkv = 1024, 128, 128
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
    mask_info = mask_info_lib.process_dynamic_mask(jnp.asarray(mask[:, :]), (bq, bkv), downcast_smem_data=False)
    self.assertEqual(int(np.asarray(mask_info.num_active_blocks)[0]), 36)
    refined, _, _ = splash._refine_mask_info_with_segments(  # pylint: disable=protected-access
        mask_info,
        _segment_ids((512, 512)),
        q_blocks=seq_len // bq,
        kv_blocks=seq_len // bkv,
        bq=bq,
        bkv=bkv,
        is_dkv=False,
    )
    self.assertEqual(int(np.asarray(refined.num_active_blocks)[0]), 20)

  @parameterized.named_parameters(
      ("default", False),
      ("qk_diag_skip", True),
  )
  def test_skip_uniform_segment_mask_is_bit_exact(self, qk_diag_skip: bool):
    if jax.default_backend() != "cpu":
      self.skipTest("Interpret mode only supported on CPU.")
    lengths = (300, 212, 390, 122)
    seq_len, head_dim, bq, bkv = sum(lengths), 128, 128, 128
    k1, k2, k3, k4 = jax.random.split(jax.random.key(11), 4)
    scale = jnp.bfloat16(0.25)
    q = jax.random.normal(k1, (4, seq_len, head_dim), jnp.bfloat16) * scale
    k = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16) * scale
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16) * scale
    do = jax.random.normal(k4, (4, seq_len, head_dim), jnp.bfloat16) * scale
    segment_ids = _segment_ids(lengths)
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))

    def _run(skip_uniform: bool):
      cfg = splash.SplashConfig(
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkv,
          block_q_dkv=bq,
          block_kv_dkv=bkv,
          block_kv_dkv_compute=bkv,
          dq_reduction_steps=3,
          qk_diag_skip=qk_diag_skip,
          skip_uniform_segment_mask=skip_uniform,
          interpret=True,
      )
      attn = splash.make_splash_mha_single_device(mask, config=cfg)
      out, vjp_fn = jax.vjp(attn, q, k, v, segment_ids)
      dq, dk, dv, _ = vjp_fn(do)
      return out, dq, dk, dv

    o_ref, dq_ref, dk_ref, dv_ref = _run(False)
    o_opt, dq_opt, dk_opt, dv_opt = _run(True)
    np.testing.assert_array_equal(o_opt, o_ref)
    np.testing.assert_array_equal(dq_opt, dq_ref)
    np.testing.assert_array_equal(dk_opt, dk_ref)
    np.testing.assert_array_equal(dv_opt, dv_ref)

  def test_fp8_qk_cast_s_to_v_and_bf16_dq_lhs_match_reference(self):
    if jax.default_backend() != "cpu":
      self.skipTest("Interpret mode only supported on CPU.")
    lengths = (300, 212, 512)
    seq_len, head_dim, bq, bkv = sum(lengths), 128, 128, 256
    k1, k2, k3, k4 = jax.random.split(jax.random.key(19), 4)
    q_bf16 = jax.random.normal(k1, (4, seq_len, head_dim), jnp.bfloat16) * jnp.bfloat16(0.25)
    half = jnp.bfloat16(0.5)
    k_bf16 = jax.random.normal(k2, (2, seq_len, head_dim), jnp.bfloat16) * half
    v = jax.random.normal(k3, (2, seq_len, head_dim), jnp.bfloat16) * half
    do = jax.random.normal(k4, (4, seq_len, head_dim), jnp.bfloat16) * half
    segment_ids = _segment_ids(lengths)
    mask = mask_lib.CausalMask(shape=(seq_len, seq_len))

    cfg = splash.SplashConfig(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=128,
        block_q_dkv=bq,
        block_kv_dkv=bkv,
        block_kv_dkv_compute=128,
        use_base2_exp=True,
        q_prescaled_log2e=True,
        skip_uniform_segment_mask=True,
        cast_s_to_v_dtype=True,
        dq_reduction_steps=3,
        interpret=True,
    )
    attn = splash.make_splash_mha_single_device(mask, config=cfg)

    def _loss(q_in, k_in, v_in):
      q_fp8 = (q_in * jnp.bfloat16(splash.LOG2E)).astype(jnp.float8_e4m3fn)
      k_fp8 = k_in.astype(jnp.float8_e4m3fn)
      out = attn(q_fp8, k_fp8, v_in, segment_ids)
      return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32)), out

    (_, out), (dq, dk, dv) = jax.value_and_grad(_loss, argnums=(0, 1, 2), has_aux=True)(q_bf16, k_bf16, v)

    dense_mask = jnp.asarray(mask[:, :])
    q_qref = (q_bf16 * jnp.bfloat16(splash.LOG2E)).astype(jnp.float8_e4m3fn).astype(jnp.float32) / splash.LOG2E
    k_qref = k_bf16.astype(jnp.float8_e4m3fn).astype(jnp.float32)
    v32, do32 = v.astype(jnp.float32), do.astype(jnp.float32)
    o_ref, stats_ref = base.attention_reference(
        q_qref,
        k_qref,
        v32,
        dense_mask,
        segment_ids,
        None,
        is_mqa=False,
        save_residuals=True,
    )
    dq_ref, dk_ref, dv_ref, _ = base.attention_reference_vjp(
        do32,
        q_qref,
        k_qref,
        v32,
        dense_mask,
        segment_ids,
        None,
        o_ref,
        stats_ref["logsumexp"],
        is_mqa=False,
    )
    np.testing.assert_allclose(out, o_ref, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dq, dq_ref, atol=3e-2, rtol=3e-2)
    np.testing.assert_allclose(dk, dk_ref, atol=3e-2, rtol=3e-2)
    np.testing.assert_allclose(dv, dv_ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
  absltest.main()
