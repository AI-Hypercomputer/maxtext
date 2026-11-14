# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for moe_combine_tc (Pallas interpret mode on CPU, compiled on TPU)."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import moe_combine_tc


def _interpret():
  """Interpret setting: compiled on TPU; otherwise the Mosaic interpreter with race detection."""
  if jax.default_backend() == "tpu":
    return False
  return pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")


def _sorted_problem(rng, experts, num_tokens, k, num_groups, hidden_dim, weight_dtype=jnp.bfloat16):
  """Expert-sorted rows, stable sort index, group sizes, weights and dy for a [T, K] routing into G experts."""
  flat_experts = jnp.asarray(experts.astype(np.int32).reshape(-1))
  sort_idx = jnp.argsort(flat_experts)  # stable
  group_sizes = jnp.bincount(flat_experts, length=num_groups).astype(jnp.int32)
  x = jnp.asarray(rng.standard_normal((num_tokens * k, hidden_dim)), jnp.float32).astype(jnp.bfloat16)
  weights = jnp.asarray(rng.random((num_tokens, k)), jnp.float32).astype(weight_dtype)
  dy = jnp.asarray(rng.standard_normal((num_tokens, hidden_dim)), jnp.float32).astype(jnp.bfloat16)
  return x, sort_idx, weights, group_sizes, dy


def _problem(seed, num_tokens, k, num_groups, hidden_dim, skew=0.0, weight_dtype=jnp.bfloat16):
  """Random routing (distinct experts per token, optionally skewed towards low expert ids)."""
  rng = np.random.default_rng(seed)
  if skew > 0:
    probs = np.exp(-skew * np.arange(num_groups))
    probs /= probs.sum()
  else:
    probs = np.full(num_groups, 1.0 / num_groups)
  experts = np.stack([rng.choice(num_groups, size=k, replace=False, p=probs) for _ in range(num_tokens)])
  return _sorted_problem(rng, experts, num_tokens, k, num_groups, hidden_dim, weight_dtype)


def _block_local_problem(seed, num_tokens, k, num_groups, hidden_dim, block_tokens, mix):
  """Routing where experts are mostly private to one token block of `block_tokens` tokens.

  Consecutive experts of a block then have adjacent row ranges (shared boundary
  granules); with `mix` > 0 a few tokens route anywhere, so some experts also
  span consecutive blocks.
  """
  rng = np.random.default_rng(seed)
  experts_per_block = num_groups // (num_tokens // block_tokens)
  rows = []
  for tok in range(num_tokens):
    if rng.random() < mix:
      rows.append(rng.choice(num_groups, size=k, replace=False))
    else:
      block_expert_base = (tok // block_tokens) * experts_per_block
      rows.append(block_expert_base + rng.choice(experts_per_block, size=k, replace=False))
  return _sorted_problem(rng, np.stack(rows), num_tokens, k, num_groups, hidden_dim)


def _dma_table_as_array(sort_idx, group_sizes, num_tokens, k, block_tokens):
  """Per-block DMA schedule table reshaped to `[num_blocks, _NUM_DMA_TABLE_FIELDS, num_groups]`."""
  num_groups = group_sizes.shape[0]
  num_blocks = num_tokens // block_tokens
  # pylint: disable=protected-access
  dma_table = moe_combine_tc._build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens)
  num_fields = moe_combine_tc._NUM_DMA_TABLE_FIELDS
  # pylint: enable=protected-access
  return (
      np.asarray(dma_table)
      .reshape(num_blocks, -1)[:, : num_fields * num_groups]
      .reshape(num_blocks, num_fields, num_groups)
  )


def _ref_grads(x, sort_idx, weights, dy):
  _, vjp = jax.vjp(lambda a, b: moe_combine_tc.combine_reference(a, sort_idx, b), x, weights)
  return vjp(dy)


def _f32(arr):
  return np.asarray(arr, np.float32)


class MoeCombineTest(parameterized.TestCase):

  def _check_combine(self, combine_fn, x, sort_idx, weights, group_sizes, dy, *, bf16_weights=True):
    """Forward and backward against the reference.

    With bf16 weights every product is exact, so y is within 1 bf16 ulp of the
    f32 reference and dx is bit-exact; non-bf16 weights go through the bf16
    hi + lo split and are checked with tolerances.
    """
    y, vjp = jax.vjp(lambda a, b: combine_fn(a, sort_idx, b, group_sizes), x, weights)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, weights)
    self.assertEqual(y.dtype, x.dtype)
    np.testing.assert_allclose(_f32(y), _f32(y_ref), rtol=1e-2, atol=1e-2)
    if bf16_weights:
      ulp = np.abs(_f32(y_ref)) * 2.0**-7 + 1e-6
      self.assertLessEqual(float(np.max(np.abs(_f32(y) - _f32(y_ref)) / ulp)), 1.01)

    dx, dw = vjp(dy)
    dx_ref, dw_ref = _ref_grads(x, sort_idx, weights, dy)
    self.assertEqual(dx.dtype, x.dtype)
    self.assertEqual(dw.dtype, weights.dtype)
    if bf16_weights:
      np.testing.assert_array_equal(_f32(dx), _f32(dx_ref))
    else:
      np.testing.assert_allclose(_f32(dx), _f32(dx_ref), rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_f32(dw), _f32(dw_ref), rtol=1e-2, atol=1e-2)
    return y, y_ref, dw

  @parameterized.named_parameters(
      ("base", 0, 256, 8, 16, 256, 128, 0.0),
      ("many_experts", 1, 256, 8, 128, 128, 128, 0.0),
      ("odd_groups_d384", 2, 384, 8, 9, 384, 128, 0.0),
      ("skewed_long_ranges", 3, 256, 8, 32, 128, 128, 1.0),
      ("k6_block256", 4, 512, 6, 12, 256, 256, 0.3),
      ("k2", 5, 256, 2, 5, 128, 128, 0.0),
      ("tiny_ranges_shared_granules", 8, 256, 2, 128, 128, 128, 0.0),
      ("tiny_ranges_align16", 9, 256, 2, 128, 128, 128, 0.0, 16),
      ("base_align16", 10, 256, 8, 16, 256, 128, 0.5, 16),
  )
  def test_fwd_bwd(self, seed, num_tokens, k, num_groups, hidden_dim, block_tokens, skew, align=8):
    x, sort_idx, weights, group_sizes, dy = _problem(seed, num_tokens, k, num_groups, hidden_dim, skew)
    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=block_tokens, interpret=_interpret(), align=align)
    self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy)

  @parameterized.named_parameters(("pure_local", 0.0), ("mixed", 0.02))
  def test_adjacent_ranges_in_block(self, mix):
    """Adjacent expert ranges within a block and (mixed) experts spanning consecutive blocks."""
    num_tokens, k, num_groups, hidden_dim, block_tokens = 512, 4, 32, 128, 128
    x, sort_idx, weights, group_sizes, dy = _block_local_problem(
        21, num_tokens, k, num_groups, hidden_dim, block_tokens, mix
    )
    dma_table = _dma_table_as_array(sort_idx, group_sizes, num_tokens, k, block_tokens)
    head_hbm = dma_table[:, moe_combine_tc._HEAD_STAGE_HBM]  # pylint: disable=protected-access
    tail_hbm = dma_table[:, moe_combine_tc._TAIL_STAGE_HBM]  # pylint: disable=protected-access
    head_carry = dma_table[:, moe_combine_tc._HEAD_CARRY_SRC]  # pylint: disable=protected-access
    tail_carry = dma_table[:, moe_combine_tc._TAIL_CARRY_SRC]  # pylint: disable=protected-access
    staged_any = int(np.sum((head_hbm >= 0) | (tail_hbm >= 0) | (head_carry >= 0) | (tail_carry >= 0)))
    if mix > 0:
      # Partial granules holding rows of earlier blocks are staged from HBM or,
      # for experts spanning consecutive blocks, carried from the previous
      # block's output buffer.
      self.assertGreater(staged_any, 0)
      self.assertGreater(int(np.sum(head_carry >= 0)), 0)
    else:
      # 64 rows per expert: every range boundary is aligned, nothing is staged.
      self.assertEqual(staged_any, 0)
    for block_idx in range(num_tokens // block_tokens):  # a granule is never staged twice within a block
      staged = [v for v in np.concatenate([head_hbm[block_idx], tail_hbm[block_idx]]) if v >= 0]
      self.assertEqual(len(staged), len(set(staged)))
    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=block_tokens, interpret=_interpret())
    self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy)

  def test_f32_weights_split(self):
    x, sort_idx, weights, group_sizes, dy = _problem(7, 256, 8, 16, 128, 0.0, weight_dtype=jnp.float32)
    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret())
    y, y_ref, dw = self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy, bf16_weights=False)
    _, dw_ref = _ref_grads(x, sort_idx, weights, dy)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(dw_ref), rtol=1e-4, atol=1e-4)
    # The lo part must be used: y matches the f32-weight reference far better
    # than the bf16-rounded-weight one (bf16 output rounding flips aside).
    y_bf16_weights = moe_combine_tc.combine_reference(x, sort_idx, weights.astype(jnp.bfloat16))
    self.assertLess(int(np.sum(_f32(y) != _f32(y_ref))) * 20, int(np.sum(_f32(y) != _f32(y_bf16_weights))))

  def test_check_grads_and_jit_scan_remat(self):
    # The Mosaic interpreter's callbacks are not allowed under remat, so
    # off-TPU this exercises jit + scan only.
    x, sort_idx, weights, group_sizes, _ = _problem(11, 256, 8, 16, 128, 0.5)
    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret())

    def loss(fn, act, w):
      def step(carry, _):
        def step_fn(aa, bb):
          return fn(aa, sort_idx, bb, group_sizes)

        if jax.default_backend() == "tpu":
          step_fn = jax.checkpoint(step_fn)
        y = step_fn(act, w)
        return carry + jnp.sum(y.astype(jnp.float32) ** 2), None

      out, _ = jax.lax.scan(step, 0.0, None, length=2)
      return out

    ref_fn = jax.jit(
        jax.value_and_grad(
            functools.partial(loss, lambda a, s, b, _: moe_combine_tc.combine_reference(a, s, b)), argnums=(0, 1)
        )
    )
    got_fn = jax.jit(jax.value_and_grad(functools.partial(loss, combine_fn), argnums=(0, 1)))
    (loss_ref, (gx_ref, gw_ref)), (loss_got, (gx_got, gw_got)) = ref_fn(x, weights), got_fn(x, weights)
    np.testing.assert_allclose(float(loss_got), float(loss_ref), rtol=1e-3)
    np.testing.assert_allclose(_f32(gx_got), _f32(gx_ref), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(_f32(gw_got), _f32(gw_ref), rtol=2e-2, atol=5e-1)

  def test_metadata_table_matches_argsort_ranges(self):
    """Every row of a token block lies in one of the block's windows, within the static buffer bound."""
    num_tokens, k, num_groups, block_tokens = 512, 8, 32, 128
    _, sort_idx, _, group_sizes, _ = _problem(13, num_tokens, k, num_groups, 128, 0.7)
    dma_table = _dma_table_as_array(sort_idx, group_sizes, num_tokens, k, block_tokens)
    row_block = np.asarray(sort_idx) // (k * block_tokens)
    max_packed_rows = 0
    for block_idx in range(num_tokens // block_tokens):
      rows_in_block = np.nonzero(row_block == block_idx)[0]
      covered_rows = set()
      for group_idx in range(num_groups):
        hbm_start = dma_table[block_idx, moe_combine_tc._WIN_HBM_START, group_idx]  # pylint: disable=protected-access
        num_rows = dma_table[block_idx, moe_combine_tc._WIN_NUM_ROWS, group_idx]  # pylint: disable=protected-access
        covered_rows |= set(range(hbm_start, hbm_start + num_rows))
      self.assertContainsSubset(rows_in_block.tolist(), covered_rows)
      max_packed_rows = max(max_packed_rows, len(covered_rows))
    max_buf_rows = moe_combine_tc._buffer_rows(block_tokens, k, num_groups)  # pylint: disable=protected-access
    self.assertLessEqual(max_packed_rows, max_buf_rows)

  def test_combine_unsupported_shape_raises(self):
    x, sort_idx, weights, group_sizes, _ = _problem(3, 128, 8, 16, 128)
    with self.assertRaisesRegex(ValueError, "require bfloat16 activations"):
      moe_combine_tc.combine(x.astype(jnp.float32), sort_idx, weights, group_sizes, block_tokens=128, interpret=True)
    with self.assertRaisesRegex(ValueError, "num_experts_per_tok must be a positive integer"):
      moe_combine_tc.combine(x, sort_idx, None, group_sizes, block_tokens=128, interpret=True)

  @parameterized.named_parameters(
      ("uniform", 0.0),
      ("skewed", 0.8),
  )
  def test_permute_fwd_bwd(self, skew):
    num_tokens, k, num_groups, hidden_dim, block_tokens = 256, 8, 16, 256, 128
    dy, sort_idx, _, group_sizes, x = _problem(7, num_tokens, k, num_groups, hidden_dim, skew)
    y_ref = x[sort_idx // k]
    unsorted_dy = dy.astype(jnp.float32)[jnp.argsort(sort_idx)]
    dx_ref = jnp.sum(unsorted_dy.reshape(num_tokens, k, hidden_dim), axis=1).astype(jnp.bfloat16)
    permute_fn = functools.partial(
        moe_combine_tc.permute, num_experts_per_tok=k, block_tokens=block_tokens, interpret=_interpret()
    )
    y, vjp = jax.vjp(lambda a: permute_fn(a, sort_idx, group_sizes), x)
    (dx,) = vjp(dy)
    np.testing.assert_array_equal(_f32(y), _f32(y_ref))
    np.testing.assert_allclose(_f32(dx), _f32(dx_ref), rtol=1e-2, atol=1e-2)

  def test_permute_unsupported_shape_raises(self):
    sort_idx = jnp.argsort(jnp.zeros((800,), jnp.int32))
    group_sizes = jnp.array([800] + [0] * 15, jnp.int32)
    x = jnp.zeros((100, 128), jnp.float32)  # only bf16 activations are supported
    with self.assertRaisesRegex(ValueError, "require bfloat16 activations"):
      moe_combine_tc.permute(x, sort_idx, group_sizes, 8, block_tokens=128, interpret=True)
    x = jnp.zeros((101, 128), jnp.bfloat16)  # 101 rows: not a multiple of the 8-row DMA alignment
    sort_idx = jnp.arange(101, dtype=jnp.int32)
    group_sizes = jnp.array([101] + [0] * 15, jnp.int32)
    with self.assertRaisesRegex(ValueError, "multiple of align"):
      moe_combine_tc.permute(x, sort_idx, group_sizes, 1, block_tokens=128, interpret=True)

  @parameterized.named_parameters(
      ("align8_384tok_k2_g16", 384, 2, 16, 128, 128, 8),
      ("align16_512tok_k1_g180", 512, 1, 180, 128, 128, 16),
  )
  def test_three_or_more_experts_sharing_single_granule(self, num_tokens, k, num_groups, hidden_dim, block_tokens, align):
    """When >= 3 experts in a token block start inside one 8/16-row granule, that granule is staged at most once."""
    x, sort_idx, weights, group_sizes, dy = _problem(42, num_tokens, k, num_groups, hidden_dim, skew=0.0)
    # pylint: disable=protected-access
    dma_table = moe_combine_tc._build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens, align=align)
    num_fields = moe_combine_tc._NUM_DMA_TABLE_FIELDS
    num_blocks = num_tokens // block_tokens
    table = (
        np.asarray(dma_table)
        .reshape(num_blocks, -1)[:, : num_fields * num_groups]
        .reshape(num_blocks, num_fields, num_groups)
    )
    head_hbm = table[:, moe_combine_tc._HEAD_STAGE_HBM]
    tail_hbm = table[:, moe_combine_tc._TAIL_STAGE_HBM]
    # pylint: enable=protected-access
    for block_idx in range(num_blocks):
      staged = [v for v in np.concatenate([head_hbm[block_idx], tail_hbm[block_idx]]) if v >= 0]
      self.assertEqual(len(staged), len(set(staged)))
    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=block_tokens, interpret=_interpret(), align=align)
    self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy)
    permute_fn = functools.partial(
        moe_combine_tc.permute, num_experts_per_tok=k, block_tokens=block_tokens, interpret=_interpret(), align=align
    )
    y_perm, vjp_perm = jax.vjp(lambda a: permute_fn(a, sort_idx, group_sizes), dy)
    (dx_perm,) = vjp_perm(x)
    np.testing.assert_array_equal(_f32(y_perm), _f32(dy[sort_idx // k]))
    unsorted_x = x.astype(jnp.float32)[jnp.argsort(sort_idx)]
    dx_perm_ref = jnp.sum(unsorted_x.reshape(num_tokens, k, hidden_dim), axis=1).astype(jnp.bfloat16)
    np.testing.assert_allclose(_f32(dx_perm), _f32(dx_perm_ref), rtol=1e-2, atol=1e-2)

  def test_max_token_65535_row_info_roundtrip(self):
    """Token 65535 (`num_tokens=65536`) has `token_lo=255, token_hi=255` and must not collide with `_INFO_FILL`."""
    k = 8
    block_tokens = 256
    block_start_token = 65536 - block_tokens  # 65280
    # Build sort_idx entries for the last token block (tokens 65280..65535), including token 65535 slots 0..7.
    tokens = np.arange(block_start_token, 65536, dtype=np.int32)
    sort_idx = jnp.asarray(tokens * k + (tokens % k), dtype=jnp.int32)
    # pylint: disable=protected-access
    row_info = moe_combine_tc._build_row_info(sort_idx, k)
    filler = jnp.full((block_tokens, moe_combine_tc._LANES), moe_combine_tc._INFO_FILL, dtype=jnp.bfloat16)
    window_info = jnp.concatenate([row_info, filler], axis=0)[None, :, :]  # [1, 512, 128]
    tok_in_block, topk_slot = moe_combine_tc._decode_row_info(
        window_info,
        0,
        0,
        2 * block_tokens,
        block_tokens,
        block_start_token,
        block_tokens,
        transpose=False,
    )
    tok_in_block_t, topk_slot_t = moe_combine_tc._decode_row_info(
        window_info,
        0,
        0,
        2 * block_tokens,
        block_tokens,
        block_start_token,
        block_tokens,
        transpose=True,
    )
    # pylint: enable=protected-access
    np.testing.assert_array_equal(np.asarray(tok_in_block[:block_tokens, 0]), np.arange(block_tokens, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(tok_in_block[block_tokens:, 0]), np.full(block_tokens, -1, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(topk_slot[:block_tokens, 0]), tokens % k)
    np.testing.assert_array_equal(np.asarray(tok_in_block_t[0]), np.asarray(tok_in_block[:, 0]))
    np.testing.assert_array_equal(np.asarray(topk_slot_t[0]), np.asarray(topk_slot[:, 0]))

  @parameterized.named_parameters(
      ("k1_top1", 256, 1, 8, 128, 128, "random"),
      ("single_expert_total", 256, 4, 1, 128, 128, "random"),
      ("all_tokens_one_expert", 256, 4, 16, 128, 128, "single_active"),
      ("trailing_empty_experts", 256, 4, 16, 128, 128, "first_half_only"),
      ("single_block_combine_256", 256, 4, 16, 128, 256, "random"),
      ("single_block_permute_512", 512, 4, 16, 128, None, "random"),
      ("many_experts_256_stride_gt_1024", 256, 4, 256, 128, 128, "random"),
      ("negative_and_zero_weights", 256, 4, 16, 128, 128, "neg_zero_weights"),
  )
  def test_routing_and_shape_corner_cases(self, num_tokens, k, num_groups, hidden_dim, block_tokens, mode):
    rng = np.random.default_rng(99)
    if mode == "single_active":
      experts = np.broadcast_to(np.arange(k, dtype=np.int32), (num_tokens, k)).copy()
      if k == 1:
        experts[:] = 5
      else:
        experts[:, 0] = 5
    elif mode == "first_half_only":
      experts = np.stack([rng.choice(num_groups // 2, size=k, replace=False) for _ in range(num_tokens)])
    elif num_groups == 1:
      experts = np.zeros((num_tokens, k), dtype=np.int32)
    else:
      experts = np.stack([rng.choice(num_groups, size=k, replace=False) for _ in range(num_tokens)])
    x, sort_idx, weights, group_sizes, dy = _sorted_problem(rng, experts, num_tokens, k, num_groups, hidden_dim)
    if mode == "neg_zero_weights":
      raw_w = rng.standard_normal((num_tokens, k)).astype(np.float32)
      raw_w[::3, 0] = 0.0
      weights = jnp.asarray(raw_w, dtype=jnp.bfloat16)

    combine_fn = functools.partial(moe_combine_tc.combine, block_tokens=block_tokens, interpret=_interpret())
    self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy)

    permute_fn = functools.partial(
        moe_combine_tc.permute, num_experts_per_tok=k, block_tokens=block_tokens, interpret=_interpret()
    )
    y_perm, vjp_perm = jax.vjp(lambda a: permute_fn(a, sort_idx, group_sizes), dy)
    (dx_perm,) = vjp_perm(x)
    np.testing.assert_array_equal(_f32(y_perm), _f32(dy[sort_idx // k]))
    unsorted_x = x.astype(jnp.float32)[jnp.argsort(sort_idx)]
    dx_perm_ref = jnp.sum(unsorted_x.reshape(num_tokens, k, hidden_dim), axis=1).astype(jnp.bfloat16)
    np.testing.assert_allclose(_f32(dx_perm), _f32(dx_perm_ref), rtol=1e-2, atol=1e-2)

  def test_vmem_limit_and_validation_fail_fast(self):
    """When VMEM budget or preconditions are violated, `_resolve_config` raises `ValueError`."""
    num_tokens, k, num_groups, hidden_dim = 512, 8, 64, 512
    # pylint: disable=protected-access
    vmem_128 = moe_combine_tc._required_vmem_bytes(128, k, num_groups, hidden_dim, 2, is_combine=True)
    vmem_256 = moe_combine_tc._required_vmem_bytes(256, k, num_groups, hidden_dim, 2, is_combine=True)
    vmem_512 = moe_combine_tc._required_vmem_bytes(512, k, num_groups, hidden_dim, 2, is_combine=True)
    self.assertLess(vmem_128, vmem_256)
    self.assertLess(vmem_256, vmem_512)

    group_sizes = jnp.full((num_groups,), (num_tokens * k) // num_groups, dtype=jnp.int32)
    cfg_256 = moe_combine_tc._resolve_config(
        (num_tokens * k, hidden_dim),
        jnp.bfloat16,
        num_tokens,
        k,
        group_sizes,
        block_tokens=256,
        interpret=False,
        align=8,
        split_weights=False,
        default_block_tokens=256,
        is_combine=True,
        vmem_limit_bytes=vmem_256,
    )
    self.assertEqual(cfg_256.block_tokens, 256)
    self.assertEqual((cfg_256.expert_chunks, cfg_256.col_chunks), (1, 1))

    with self.assertRaisesRegex(ValueError, "exceeding limit"):
      moe_combine_tc._resolve_config(
          (num_tokens * k, hidden_dim),
          jnp.bfloat16,
          num_tokens,
          k,
          group_sizes,
          block_tokens=512,
          interpret=False,
          align=8,
          split_weights=False,
          default_block_tokens=512,
          is_combine=True,
          vmem_limit_bytes=vmem_256,
      )

    # Increasing expert_chunks or col_chunks allows fitting within a tighter VMEM budget.
    cfg_chunked = moe_combine_tc._resolve_config(
        (num_tokens * k, hidden_dim),
        jnp.bfloat16,
        num_tokens,
        k,
        group_sizes,
        block_tokens=128,
        interpret=False,
        align=8,
        split_weights=False,
        default_block_tokens=128,
        is_combine=True,
        vmem_limit_bytes=vmem_128 - 1,
        expert_chunks=2,
        col_chunks=2,
    )
    self.assertEqual((cfg_chunked.block_tokens, cfg_chunked.expert_chunks, cfg_chunked.col_chunks), (128, 2, 2))
    # pylint: enable=protected-access

    # Verify kernel_supported bounds:
    self.assertFalse(moe_combine_tc.kernel_supported((2048, 0), jnp.bfloat16, 256, 8, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((0, 128), jnp.bfloat16, 256, 0, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((256 * 129, 128), jnp.bfloat16, 256, 129, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((512, 128), jnp.bfloat16, 64, 8, 64))
    self.assertFalse(moe_combine_tc.kernel_supported((2048, 128), jnp.bfloat16, 256, 8, 256, num_groups=0))
    self.assertTrue(moe_combine_tc.kernel_supported((65536 * 8, 128), jnp.bfloat16, 65536, 8, 256, num_groups=16))
    self.assertTrue(moe_combine_tc.kernel_supported(((65536 + 256) * 8, 128), jnp.bfloat16, 65536 + 256, 8, 256))
    self.assertFalse(moe_combine_tc.kernel_supported(((256**3 + 256) * 8, 128), jnp.bfloat16, 256**3 + 256, 8, 256))

  def _check_permute(self, permute_fn, x_tokens, sort_idx, group_sizes, dy_rows, k):
    """Permute forward (exact) and backward against a plain gather / gather-sum."""
    num_tokens, hidden_dim = x_tokens.shape
    y, vjp = jax.vjp(lambda a: permute_fn(a, sort_idx, group_sizes), x_tokens)
    (dx,) = vjp(dy_rows)
    np.testing.assert_array_equal(_f32(y), _f32(x_tokens[sort_idx // k]))
    unsorted_dy = dy_rows.astype(jnp.float32)[jnp.argsort(sort_idx)]
    dx_ref = jnp.sum(unsorted_dy.reshape(num_tokens, k, hidden_dim), axis=1).astype(jnp.bfloat16)
    self.assertEqual(dx.shape, x_tokens.shape)
    np.testing.assert_allclose(_f32(dx), _f32(dx_ref), rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      ("expert_chunks2", 256, 4, 16, 256, 2, 1, 0.0),
      ("col_chunks2", 256, 4, 16, 256, 1, 2, 0.0),
      ("expert_chunks3_ragged_col_chunks2", 256, 4, 16, 256, 3, 2, 0.5),
      ("expert_chunks4_tiny_ranges", 256, 2, 128, 128, 4, 1, 0.0),
      ("expert_chunks2_three_blocks_skewed", 384, 8, 9, 384, 2, 3, 1.0),
  )
  def test_expert_and_col_chunks(self, num_tokens, k, num_groups, hidden_dim, expert_chunks, col_chunks, skew):
    """Experts / hidden columns split over several grid passes give the same results as a single pass."""
    x, sort_idx, weights, group_sizes, dy = _problem(31, num_tokens, k, num_groups, hidden_dim, skew)
    kwargs = {"block_tokens": 128, "interpret": _interpret(), "expert_chunks": expert_chunks, "col_chunks": col_chunks}
    self._check_combine(functools.partial(moe_combine_tc.combine, **kwargs), x, sort_idx, weights, group_sizes, dy)
    self._check_permute(
        functools.partial(moe_combine_tc.permute, num_experts_per_tok=k, **kwargs), dy, sort_idx, group_sizes, x, k
    )

  def test_expert_chunks_f32_weights(self):
    x, sort_idx, weights, group_sizes, dy = _problem(33, 256, 8, 16, 128, 0.0, weight_dtype=jnp.float32)
    combine_fn = functools.partial(
        moe_combine_tc.combine, block_tokens=128, interpret=_interpret(), expert_chunks=2, col_chunks=1
    )
    self._check_combine(combine_fn, x, sort_idx, weights, group_sizes, dy, bf16_weights=False)

  @parameterized.named_parameters(
      ("tokens200_d200_b128", 200, 4, 16, 200, 128),
      ("tokens100_d128_b256", 100, 8, 16, 128, 256),
      ("tokens300_d320_b256", 300, 8, 12, 320, 256),
  )
  def test_padded_tokens_and_hidden_dim(self, num_tokens, k, num_groups, hidden_dim, block_tokens):
    """Token counts not divisible by `block_tokens` and hidden sizes not divisible by 128 are padded automatically."""
    x, sort_idx, weights, group_sizes, dy = _problem(35, num_tokens, k, num_groups, hidden_dim, 0.3)
    kwargs = {"block_tokens": block_tokens, "interpret": _interpret()}
    y, _, _ = self._check_combine(
        functools.partial(moe_combine_tc.combine, **kwargs), x, sort_idx, weights, group_sizes, dy
    )
    self.assertEqual(y.shape, (num_tokens, hidden_dim))
    self._check_permute(
        functools.partial(moe_combine_tc.permute, num_experts_per_tok=k, **kwargs), dy, sort_idx, group_sizes, x, k
    )

  @parameterized.named_parameters(("k1", 1, 1), ("k4_expert_chunks2", 4, 2))
  def test_unweighted_combine(self, k, expert_chunks):
    """Unweighted combine (`weights=None`, Llama4): forward is gather-sum, backward repeats each token's gradient."""
    num_tokens, num_groups, hidden_dim = 256, 16, 128
    x, sort_idx, _, group_sizes, dy = _problem(37, num_tokens, k, num_groups, hidden_dim)
    y, vjp = jax.vjp(
        lambda a: moe_combine_tc.combine(
            a,
            sort_idx,
            None,
            group_sizes,
            num_experts_per_tok=k,
            block_tokens=128,
            interpret=_interpret(),
            expert_chunks=expert_chunks,
        ),
        x,
    )
    (dx,) = vjp(dy)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, jnp.ones((num_tokens, k), jnp.bfloat16))
    np.testing.assert_allclose(_f32(y), _f32(y_ref), rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(_f32(dx), _f32(dy[sort_idx // k]))

  def test_wide_token_ids(self):
    """Token ids >= 65536 use a third base-256 lane of the row info."""
    k = 8
    tokens = np.array([65535, 65536, 70000, 256**3 - 1], dtype=np.int64)
    sort_idx = jnp.asarray(np.repeat(tokens * k, 64) + 3, dtype=jnp.int32)  # 256 rows, top-k slot 3
    # pylint: disable=protected-access
    row_info = moe_combine_tc._build_row_info(sort_idx, k)[None, :, :]
    for transpose in (False, True):
      tok, slot = moe_combine_tc._decode_row_info(
          row_info, 0, 0, 256, 256**3, 0, 256, transpose=transpose, wide_tokens=True
      )
      np.testing.assert_array_equal(np.asarray(tok).reshape(-1), np.repeat(tokens, 64))
      np.testing.assert_array_equal(np.asarray(slot).reshape(-1), np.full(256, 3))
    # End to end with the third lane forced on for a small problem.
    x, sort_idx, weights, group_sizes, dy = _problem(39, 256, 4, 16, 128)
    with mock.patch.object(moe_combine_tc, "_MAX_NARROW_TOKENS", 0):
      kwargs = {"block_tokens": 128, "interpret": _interpret()}
      self._check_combine(functools.partial(moe_combine_tc.combine, **kwargs), x, sort_idx, weights, group_sizes, dy)
      self._check_permute(
          functools.partial(moe_combine_tc.permute, num_experts_per_tok=4, **kwargs), dy, sort_idx, group_sizes, x, 4
      )
    # pylint: enable=protected-access

  def test_row_chunked_histogram_matches(self):
    """The (block, expert) histogram split into row chunks builds the same DMA table."""
    num_tokens, k, num_groups, block_tokens = 512, 8, 32, 128
    _, sort_idx, _, group_sizes, _ = _problem(41, num_tokens, k, num_groups, 128, 0.7)
    # pylint: disable=protected-access
    table = moe_combine_tc._build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens, expert_chunks=2)
    with mock.patch.object(moe_combine_tc, "_MAX_HISTOGRAM_ONEHOT_BYTES", 16 * 1024):
      chunked_table = moe_combine_tc._build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens, expert_chunks=2)
    # pylint: enable=protected-access
    np.testing.assert_array_equal(np.asarray(table), np.asarray(chunked_table))

  @parameterized.named_parameters(
      ("gemma4_26b_combine", 2816, 8, 128, True, 256, 1, 1),
      ("gemma4_26b_permute", 2816, 8, 128, False, 512, 1, 1),
      ("deepseek_v4_flash_combine", 4096, 6, 256, True, 128, 2, 1),
      ("deepseek_v3_combine", 7168, 8, 256, True, 128, 2, 2),
      ("qwen3_5_397b_permute", 4096, 10, 512, False, 128, 4, 1),
  )
  def test_model_configs_fit_vmem(self, hidden_dim, k, num_groups, is_combine, block_tokens, expert_chunks, col_chunks):
    """Verifies that Gemma-4 defaults (single pass) and chunked large-model configs fit the v6e VMEM budget."""
    num_tokens = 32768
    cfg = moe_combine_tc._resolve_config(  # pylint: disable=protected-access
        (num_tokens * k, hidden_dim),
        jnp.bfloat16,
        num_tokens,
        k,
        jnp.zeros((num_groups,), jnp.int32),
        block_tokens=block_tokens,
        interpret=False,
        align=8,
        split_weights=False,
        default_block_tokens=256 if is_combine else 512,
        is_combine=is_combine,
        expert_chunks=expert_chunks,
        col_chunks=col_chunks,
    )
    self.assertEqual((cfg.block_tokens, cfg.expert_chunks, cfg.col_chunks), (block_tokens, expert_chunks, col_chunks))


class PrivateFunctionsTest(parameterized.TestCase):
  """Direct unit tests for each major private helper in `moe_combine_tc`."""

  # pylint: disable=protected-access

  def test_dma_table_stride_and_buffer_rows(self):
    """`_dma_table_stride` rounds up to SMEM-legal sizes; `_buffer_rows` bounds worst-case aligned window rows."""
    # 10 fields * G experts: powers of two >= 128 up to 1024, then multiples of 1024.
    self.assertEqual(moe_combine_tc._dma_table_stride(1), 128)  # 10 -> 128
    self.assertEqual(moe_combine_tc._dma_table_stride(12), 128)  # 120 -> 128
    self.assertEqual(moe_combine_tc._dma_table_stride(16), 256)  # 160 -> 256
    self.assertEqual(moe_combine_tc._dma_table_stride(64), 1024)  # 640 -> 1024
    self.assertEqual(moe_combine_tc._dma_table_stride(128), 2048)  # 1280 -> 2048
    self.assertEqual(moe_combine_tc._dma_table_stride(256), 3072)  # 2560 -> 3072

    # Worst-case buffer rows: block_tokens*k + (2*align - 2)*min(G, block_tokens*k), rounded up to 256.
    # For block_tokens=256, k=8, G=16, align=8: 2048 + 14 * 16 = 2272 -> rounded up to 256 is 2304.
    self.assertEqual(moe_combine_tc._buffer_rows(256, 8, 16, align=8), 2304)
    # For align=16: 2048 + 30 * 16 = 2528 -> rounded up to 256 is 2560.
    self.assertEqual(moe_combine_tc._buffer_rows(256, 8, 16, align=16), 2560)

  def test_vmem_bytes_estimators_and_pad_trailing(self):
    """`_gather_vmem_bytes`, `_combine_bwd_vmem_bytes`, `_permute_fwd_vmem_bytes`, and `_pad_trailing`."""
    buf_rows, block_tokens, hidden_dim, itemsize = 1024, 128, 256, 2
    base_gather = moe_combine_tc._gather_vmem_bytes(
        buf_rows, block_tokens, hidden_dim, itemsize, weighted=True, split_weights=False, accumulate=False
    )
    split_gather = moe_combine_tc._gather_vmem_bytes(
        buf_rows, block_tokens, hidden_dim, itemsize, weighted=True, split_weights=True, accumulate=False
    )
    acc_gather = moe_combine_tc._gather_vmem_bytes(
        buf_rows, block_tokens, hidden_dim, itemsize, weighted=True, split_weights=False, accumulate=True
    )
    # `split_weights=True` adds a second `[block_tokens, buf_rows]` bf16 routing matrix in VMEM.
    self.assertEqual(split_gather - base_gather, block_tokens * buf_rows * itemsize)
    # `accumulate=True` adds a `[block_tokens, hidden_dim]` f32 accumulator in VMEM.
    self.assertEqual(acc_gather - base_gather, block_tokens * hidden_dim * 4)

    # `_required_vmem_bytes` returns `max(fwd_bytes, bwd_bytes)` for combine or permute.
    req_combine = moe_combine_tc._required_vmem_bytes(block_tokens, 4, 16, hidden_dim, itemsize, is_combine=True)
    bwd_combine = moe_combine_tc._combine_bwd_vmem_bytes(
        moe_combine_tc._buffer_rows(block_tokens, 4, 16), block_tokens, hidden_dim, itemsize
    )
    self.assertGreaterEqual(req_combine, bwd_combine)

    # `_pad_trailing` is a no-op when shapes already match, and zero-pads trailing rows/cols otherwise.
    arr = jnp.ones((3, 5), dtype=jnp.bfloat16)
    self.assertIs(moe_combine_tc._pad_trailing(arr, 3, 5), arr)
    padded = np.asarray(moe_combine_tc._pad_trailing(arr, 4, 8), dtype=np.float32)
    self.assertEqual(padded.shape, (4, 8))
    np.testing.assert_array_equal(padded[:3, :5], np.ones((3, 5), dtype=np.float32))
    np.testing.assert_array_equal(padded[3:, :], np.zeros((1, 8), dtype=np.float32))
    np.testing.assert_array_equal(padded[:, 5:], np.zeros((4, 3), dtype=np.float32))

  def test_block_expert_histogram(self):
    """`_block_expert_histogram` counts rows per `(token_block, expert)` on the MXU without scatters."""
    # 16 sorted rows across 3 experts: expert 0 owns [0, 6), expert 1 owns [6, 10), expert 2 owns [10, 16).
    group_starts = jnp.array([0, 6, 10], dtype=jnp.int32)
    group_ends = jnp.array([6, 10, 16], dtype=jnp.int32)
    # Assign each sorted row to one of 2 token blocks (0 or 1):
    #   expert 0 (rows 0..5): 4 rows in block 0, 2 rows in block 1
    #   expert 1 (rows 6..9): 0 rows in block 0, 4 rows in block 1
    #   expert 2 (rows 10..15): 3 rows in block 0, 3 rows in block 1
    row_block = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    expected = np.array([[4, 0, 3], [2, 4, 3]], dtype=np.int32)

    got = moe_combine_tc._block_expert_histogram(row_block, group_starts, group_ends, num_blocks=2)
    np.testing.assert_array_equal(np.asarray(got), expected)

    # Force multi-chunk execution (`lax.map`) by lowering `_MAX_HISTOGRAM_ONEHOT_BYTES`.
    with mock.patch.object(moe_combine_tc, "_MAX_HISTOGRAM_ONEHOT_BYTES", 8):
      got_chunked = moe_combine_tc._block_expert_histogram(row_block, group_starts, group_ends, num_blocks=2)
    np.testing.assert_array_equal(np.asarray(got_chunked), expected)

  def test_build_dma_table_all_10_fields_hand_traced(self):
    """Hand-traced test of all 10 fields of `_build_dma_table` and `_num_packed_rows`.

    Setup (`num_tokens=4`, `k=4`, `block_tokens=2`, `align=8`, `num_groups=3`, total rows `R = 16`):
      - Block 0 owns tokens {0, 1} (flat token-topk indices 0..7).
      - Block 1 owns tokens {2, 3} (flat token-topk indices 8..15).
      - Expert row ranges in `sorted_x`:
          Expert 0 (size 6, HBM rows [0, 6)):
            Block 0 owns [0, 3), Block 1 owns [3, 6).
            Aligned DMA window for both blocks is [0, 8).
          Expert 1 (size 4, HBM rows [6, 10)):
            Block 0 owns [6, 10), Block 1 owns [10, 10) (empty in Block 1!).
            Aligned DMA window in Block 0 is [0, 16) -> coalesced after Expert 0's [0, 8) to [8, 16).
          Expert 2 (size 6, HBM rows [10, 16)):
            Block 0 owns [10, 11), Block 1 owns [11, 16).
            In Block 0, aligned window [8, 16) is completely covered by Expert 1 -> coalesced to 0 rows!
            In Block 1, aligned window is [8, 16).
    """
    group_sizes = jnp.array([6, 4, 6], dtype=jnp.int32)
    # Construct `sort_idx` so that each expert's slice has the exact Block 0 (< 8) and Block 1 (>= 8) row counts above.
    sort_idx = jnp.array(
        [0, 1, 2, 8, 9, 10]  # Expert 0: 3 rows from Block 0, 3 rows from Block 1
        + [3, 4, 5, 6]  # Expert 1: 4 rows from Block 0, 0 rows from Block 1
        + [7, 11, 12, 13, 14, 15],  # Expert 2: 1 row from Block 0, 5 rows from Block 1
        dtype=jnp.int32,
    )
    raw_table = moe_combine_tc._build_dma_table(
        sort_idx, group_sizes, num_tokens=4, k=4, block_tokens=2, align=8, expert_chunks=1
    )
    stride = moe_combine_tc._dma_table_stride(3)
    table_2d = np.asarray(raw_table).reshape(2, stride)
    fields = table_2d[:, : 3 * moe_combine_tc._NUM_DMA_TABLE_FIELDS].reshape(2, moe_combine_tc._NUM_DMA_TABLE_FIELDS, 3)

    # --- Block 0 DMA windows & coalescing ---
    # Expert 0: range [0, 3) -> aligned [0, 8) -> DMA 8 rows from HBM 0 into VMEM 0 (delta = 0).
    # Expert 1: range [6, 10) -> aligned [0, 16), clamped by prev_covered_end=8 -> DMA 8 rows from HBM 8 into VMEM 8.
    # Expert 2: range [10, 11) -> aligned [8, 16), clamped by prev_covered_end=16 -> DMA 0 rows!
    np.testing.assert_array_equal(fields[0, moe_combine_tc._WIN_HBM_START], [0, 8, 16])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._WIN_NUM_ROWS], [8, 8, 0])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._WIN_BUF_OFFSET], [0, 8, 16])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._WIN_ROW_DELTA], [0, 0, 0])
    # Block 0 has no earlier blocks, so nothing is staged from HBM or carried from a previous slot.
    np.testing.assert_array_equal(fields[0, moe_combine_tc._HEAD_STAGE_HBM], [-1, -1, -1])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._TAIL_STAGE_HBM], [-1, -1, -1])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._HEAD_CARRY_SRC], [-1, -1, -1])
    np.testing.assert_array_equal(fields[0, moe_combine_tc._TAIL_CARRY_SRC], [-1, -1, -1])
    self.assertEqual(int(moe_combine_tc._num_packed_rows(table_2d[0], num_groups=3)), 16)

    # --- Block 1 DMA windows & scatter staging/carry ---
    # Expert 0: range [3, 6) -> aligned [0, 8) -> DMA 8 rows from HBM 0 into VMEM 0 (delta = 0).
    #   Boundary granule [0, 8) holds Block 0 rows [0, 3) and was written by Block 0 at VMEM row 0,
    #   so it is carried VMEM-to-VMEM (`_HEAD_CARRY_SRC = 0`, `_HEAD_CARRY_DST = 0`) instead of staged from HBM!
    # Expert 1: empty in Block 1 -> DMA 0 rows.
    # Expert 2: range [11, 16) -> aligned [8, 16) -> DMA 8 rows from HBM 8 into VMEM 8 (delta = 0).
    #   Head granule [8, 16) holds earlier rows (Expert 1's [6, 10) and Expert 2's [10, 11) from Block 0) and was
    #   written by Block 0 at VMEM row 8, so it is also carried VMEM-to-VMEM (`_HEAD_CARRY_SRC = 8`, `DST = 8`).
    np.testing.assert_array_equal(fields[1, moe_combine_tc._WIN_HBM_START], [0, 8, 8])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._WIN_NUM_ROWS], [8, 0, 8])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._WIN_BUF_OFFSET], [0, 8, 8])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._WIN_ROW_DELTA], [0, 0, 0])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._HEAD_STAGE_HBM], [-1, -1, -1])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._HEAD_CARRY_SRC], [0, -1, 8])
    np.testing.assert_array_equal(fields[1, moe_combine_tc._HEAD_CARRY_DST], [0, -1, 8])
    self.assertEqual(int(moe_combine_tc._num_packed_rows(table_2d[1], num_groups=3)), 16)

  def test_build_and_decode_row_info_all_cases(self):
    """`_build_row_info` encodes base-256 lanes and `_decode_row_info` filters out non-owned/unused rows."""
    k = 4
    # 4 sorted rows with known `(token_id, topk_slot)` across 2 expert chunks (2 rows per chunk):
    #   row 0: token 300 (lo=44, mid=1, hi=0), topk_slot=2, expert_chunk=0  (belongs to block [256, 384))
    #   row 1: token 10  (lo=10, mid=0, hi=0), topk_slot=1, expert_chunk=0  (earlier block -> masked to -1)
    #   row 2: token 310 (lo=54, mid=1, hi=0), topk_slot=3, expert_chunk=1  (wrong expert_chunk -> masked to -1)
    #   row 3: token 260 (lo=4,  mid=1, hi=0), topk_slot=0, expert_chunk=0  (at index >= packed_rows -> masked to -1)
    tokens = np.array([300, 10, 310, 260], dtype=np.int32)
    topk_slots = np.array([2, 1, 3, 0], dtype=np.int32)
    sort_idx = jnp.asarray(tokens * k + topk_slots, dtype=jnp.int32)
    group_sizes = jnp.array([2, 2], dtype=jnp.int32)  # 2 experts, expert_chunks=2 -> 1 expert (2 rows) per chunk

    row_info = moe_combine_tc._build_row_info(sort_idx, k, group_sizes=group_sizes, expert_chunks=2)
    self.assertEqual(row_info.shape, (4, 128))
    self.assertEqual(row_info.dtype, jnp.bfloat16)
    lanes = np.asarray(row_info[:, :5], dtype=np.int32)
    np.testing.assert_array_equal(lanes[:, moe_combine_tc._INFO_TOKEN_LO], [44, 10, 54, 4])
    np.testing.assert_array_equal(lanes[:, moe_combine_tc._INFO_TOKEN_MID], [1, 0, 1, 1])
    np.testing.assert_array_equal(lanes[:, moe_combine_tc._INFO_TOPK], [2, 1, 3, 0])
    np.testing.assert_array_equal(lanes[:, moe_combine_tc._INFO_TOKEN_HI], [0, 0, 0, 0])
    np.testing.assert_array_equal(lanes[:, moe_combine_tc._INFO_EXPERT_CHUNK], [0, 0, 1, 1])

    # Append 4 unused filler rows (`_INFO_FILL = -1.0`) to form an 8-row VMEM window buffer `[1, 8, 128]`.
    filler = jnp.full((4, 128), moe_combine_tc._INFO_FILL, dtype=jnp.bfloat16)
    window_info_vmem = jnp.concatenate([row_info, filler], axis=0)[None, :, :]

    # Decode for block_start_token=256, block_tokens=128, packed_rows=3, expert_chunk=0:
    # Only row 0 (token 300 -> token_in_block 44, topk_slot 2) is owned by (block 2, expert_chunk 0) with idx < 3!
    tok_in_block, slot = moe_combine_tc._decode_row_info(
        window_info_vmem,
        slot=0,
        row_offset=0,
        num_rows=8,
        block_tokens=128,
        block_start_token=256,
        packed_rows=3,
        transpose=False,
        expert_chunk=0,
    )
    np.testing.assert_array_equal(np.asarray(tok_in_block[:, 0]), [44, -1, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(np.asarray(slot[:, 0]), [2, 1, 3, 0, -1, -1, -1, -1])

  def test_split_into_bf16_parts_exactness(self):
    """`_split_into_bf16_parts` decomposes f32 values into bf16 parts whose f32 sum/dot recovers f32 precision."""
    rng = np.random.default_rng(123)
    val_f32 = jnp.asarray(rng.standard_normal((64, 128), dtype=np.float32))

    # 2 parts (`hi + lo`, ~16 mantissa bits): error is < 2^-15 relative to f32.
    parts_2 = moe_combine_tc._split_into_bf16_parts(val_f32, jnp.bfloat16, num_parts=2)
    self.assertLen(parts_2, 2)
    recon_2 = sum(p.astype(jnp.float32) for p in parts_2)
    np.testing.assert_allclose(np.asarray(recon_2), np.asarray(val_f32), rtol=2e-5, atol=1e-6)

    # 3 parts (`hi + mid + lo`, 3 * 8 = 24 mantissa bits): exact reconstruction of f32 normal numbers.
    parts_3 = moe_combine_tc._split_into_bf16_parts(val_f32, jnp.bfloat16, num_parts=3)
    self.assertLen(parts_3, 3)
    recon_3 = sum(p.astype(jnp.float32) for p in parts_3)
    np.testing.assert_allclose(np.asarray(recon_3), np.asarray(val_f32), rtol=1e-7, atol=1e-7)

    # Multiplying a binary `{0, 1}` bf16 indicator matrix by the 3 bf16 parts matches the f32 dot product.
    indicator_bf16 = jnp.asarray(rng.integers(0, 2, size=(32, 64)), dtype=jnp.bfloat16)
    dot_from_parts = sum(jnp.dot(indicator_bf16, part, preferred_element_type=jnp.float32) for part in parts_3)
    dot_ref = jnp.dot(indicator_bf16.astype(jnp.float32), val_f32, precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(dot_from_parts), np.asarray(dot_ref), rtol=1e-6, atol=1e-5)

  def test_dma_table_block_specs_traversal_order(self):
    """`_dma_table_block_specs` steps `cur_index` and `next_index` in the exact grid order of gather vs scatter."""
    num_blocks, experts_per_chunk, expert_chunks = 3, 8, 2

    # 1. Gather order (`expert_chunk_innermost=True`, grid `(col_chunk, block, expert_chunk)`):
    #    For each token block `b`, visits all `expert_chunks` `0..E-1` consecutively.
    cur_spec, next_spec = moe_combine_tc._dma_table_block_specs(
        num_blocks, experts_per_chunk, expert_chunks, expert_chunk_innermost=True
    )
    gather_steps = [(0, b, e) for b in range(num_blocks) for e in range(expert_chunks)]
    gather_cur = [int(cur_spec.index_map(*idx)[0]) for idx in gather_steps]
    gather_next = [int(next_spec.index_map(*idx)[0]) for idx in gather_steps]
    # Table layout is `[expert_chunk, block]`, so `(b=0,e=0)->0, (b=0,e=1)->3, (b=1,e=0)->1, (b=1,e=1)->4, ...`
    self.assertEqual(gather_cur, [0, 3, 1, 4, 2, 5])
    # Every step's `next_index` equals the next step's `cur_index` (clamped at the final step).
    self.assertEqual(gather_next[:-1], gather_cur[1:])

    # 2. Scatter order (`expert_chunk_innermost=False`, grid `(col_chunk, expert_chunk, block)`):
    #    For each `expert_chunk`, visits token blocks `0..num_blocks-1` consecutively.
    cur_spec_s, next_spec_s = moe_combine_tc._dma_table_block_specs(
        num_blocks, experts_per_chunk, expert_chunks, expert_chunk_innermost=False
    )
    scatter_steps = [(0, e, b) for e in range(expert_chunks) for b in range(num_blocks)]
    scatter_cur = [int(cur_spec_s.index_map(*idx)[0]) for idx in scatter_steps]
    scatter_next = [int(next_spec_s.index_map(*idx)[0]) for idx in scatter_steps]
    self.assertEqual(scatter_cur, [0, 1, 2, 3, 4, 5])
    # Within each expert_chunk pass `0..num_blocks-1`, `next_index` advances `b -> min(b + 1, num_blocks - 1)`.
    self.assertEqual(scatter_next, [1, 2, 2, 4, 5, 5])

  # pylint: enable=protected-access


if __name__ == "__main__":
  absltest.main()
