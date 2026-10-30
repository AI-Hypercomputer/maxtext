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

  def test_fallback_unsupported_shape(self):
    x, sort_idx, weights, group_sizes, _ = _problem(3, 100, 8, 16, 128)
    y = moe_combine_tc.combine(x, sort_idx, weights, group_sizes, block_tokens=128, interpret=True)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(moe_combine_tc.combine_reference(x, sort_idx, weights)))
    self.assertIsNone(
        moe_combine_tc.combine(x, sort_idx, weights, group_sizes, block_tokens=128, interpret=True, fallback=False)
    )

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

  def test_permute_unsupported_shape_returns_none(self):
    x = jnp.zeros((100, 128), jnp.bfloat16)
    sort_idx = jnp.argsort(jnp.zeros((800,), jnp.int32))
    group_sizes = jnp.array([800] + [0] * 15, jnp.int32)
    self.assertIsNone(moe_combine_tc.permute(x, sort_idx, group_sizes, 8, block_tokens=128, interpret=True))

  @parameterized.named_parameters(
      ("align8_384tok_k2_g16", 384, 2, 16, 128, 128, 8),
      ("align16_512tok_k1_g180", 512, 1, 180, 128, 128, 16),
  )
  def test_three_or_more_experts_sharing_single_granule(self, num_tokens, k, num_groups, hidden_dim, block_tokens, align):
    """When >= 3 experts in the same token block start inside one 8/16-row granule, that granule is staged at most once."""
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

  def test_vmem_limit_step_down_and_fallback(self):
    """When `block_tokens` exceeds `vmem_limit_bytes`, `_resolve_config` steps down to 256/128 or returns None."""
    num_tokens, k, num_groups, hidden_dim = 512, 8, 64, 512
    # pylint: disable=protected-access
    vmem_128 = moe_combine_tc._required_vmem_bytes(128, k, num_groups, hidden_dim, 2, is_combine=True)
    vmem_256 = moe_combine_tc._required_vmem_bytes(256, k, num_groups, hidden_dim, 2, is_combine=True)
    vmem_512 = moe_combine_tc._required_vmem_bytes(512, k, num_groups, hidden_dim, 2, is_combine=True)
    self.assertLess(vmem_128, vmem_256)
    self.assertLess(vmem_256, vmem_512)

    group_sizes = jnp.full((num_groups,), (num_tokens * k) // num_groups, dtype=jnp.int32)
    cfg_step_to_256 = moe_combine_tc._resolve_config(
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
    self.assertIsNotNone(cfg_step_to_256)
    self.assertEqual(cfg_step_to_256.block_tokens, 256)

    cfg_step_to_128 = moe_combine_tc._resolve_config(
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
        vmem_limit_bytes=vmem_128,
    )
    self.assertIsNotNone(cfg_step_to_128)
    self.assertEqual(cfg_step_to_128.block_tokens, 128)

    cfg_too_small = moe_combine_tc._resolve_config(
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
        vmem_limit_bytes=vmem_128 - 1,
    )
    self.assertIsNone(cfg_too_small)
    # pylint: enable=protected-access

    # Verify kernel_supported bounds:
    self.assertFalse(moe_combine_tc.kernel_supported((2048, 0), jnp.bfloat16, 256, 8, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((0, 128), jnp.bfloat16, 256, 0, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((256 * 129, 128), jnp.bfloat16, 256, 129, 256))
    self.assertFalse(moe_combine_tc.kernel_supported((512, 128), jnp.bfloat16, 64, 8, 64))
    self.assertFalse(moe_combine_tc.kernel_supported((2048, 128), jnp.bfloat16, 256, 8, 256, num_groups=0))
    self.assertTrue(moe_combine_tc.kernel_supported((65536 * 8, 128), jnp.bfloat16, 65536, 8, 256, num_groups=16))
    self.assertFalse(moe_combine_tc.kernel_supported(((65536 + 256) * 8, 128), jnp.bfloat16, 65536 + 256, 8, 256))


if __name__ == "__main__":
  absltest.main()
