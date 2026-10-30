# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gdn_conv1d_natural_layout and gdn_fwd_kernel_head_group_size: the GDN forward kernels give the same bits.

gdn_conv1d_natural_layout runs the prefill-only forward kernels' Conv1D + SiLU on the bf16 [chunk, dim] DMA buffer with
tokens on sublanes (compute_conv1d.natural_causal_conv1d) instead of through the per-token compact float32 scratch;
gdn_fwd_kernel_head_group_size sets the kernels' kq-head group size (compute_gdn.chunked_gdn_per_seq).

  - Conv stage, eager on CPU (every float32 op rounded on its own): natural_causal_conv1d + split_qkv_natural +
    select_natural_conv_halo vs the compact path's causal_conv1d + load_and_select_states on the same inputs, packed
    and unpacked, with and without an initial state, first and carried tiles. A tap order that only reorders the sum
    is caught, so the comparison can fail.
  - Head groups: chunked_gdn with groups of 4 / 2 vs all heads, bit for bit.
  - Whole kernels, in TPU interpret mode on CPU (the harness of gdn_pass1_states_and_out_proj_test): every output of
    the forward and states-only kernels, key on vs off. A q / k swap in the natural path is caught.
  - Validator: the keys default off and are refused where they would do nothing.
"""

# pylint: disable=protected-access

import contextlib
import dataclasses
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src.pallas.mosaic import primitives as mosaic_primitives
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from maxtext.configs import types as config_types
from maxtext.kernels.gdn import compute_conv1d
from maxtext.kernels.gdn import compute_gdn
from maxtext.kernels.gdn import config as gdn_config
from maxtext.kernels.gdn import memory_ref
from maxtext.kernels.gdn import vmem_ldst
from maxtext.kernels.gdn import wrapper
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api

_CHUNK = 64
_KERNEL = 4
_HEAD_DIM = 128


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


@contextlib.contextmanager
def _eager_roll():
  """pltpu.roll has a lowering (jnp.roll) but no eager rule; give it one with the same definition."""

  def impl(x, shift, *, axis, stride, stride_axis):
    assert stride is None and stride_axis is None
    return jnp.roll(x, shift, axis)

  old = mosaic_primitives.roll_p.impl
  mosaic_primitives.roll_p.def_impl(impl)
  try:
    yield
  finally:
    mosaic_primitives.roll_p.impl = old


def _cfg(nk, nv, seg, mode=gdn_config.GDNMode.PER_SEQ, group=None, batch=4 * _CHUNK):
  return gdn_config.GDNConfig(
      mode=mode,
      dtypes=gdn_config.Dtypes(jnp.bfloat16, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32),
      batch_size=batch,
      dim_size=(2 * nk + nv) * _HEAD_DIM,
      kernel_size=_KERNEL,
      tile_size=_CHUNK,
      num_kq_heads=nk,
      num_v_heads=nv,
      kq_head_dim=_HEAD_DIM,
      v_head_dim=_HEAD_DIM,
      has_seg_ids=seg,
      head_group_size=group,
      conv_dense_token_stack=True,
      is_prefill_only=True,
  )


def _b_lanes(seg_row, has_init, nv, rng):
  """b lanes for one sequence as _fused_conv1d_gdn packs them: betas, then the segment metadata lanes."""
  t = len(seg_row)
  b = rng.standard_normal((t, 128)).astype(np.float32)
  if seg_row is None:
    return b
  seg = compute_conv1d.canonicalize_segment_ids(jnp.asarray(seg_row, jnp.int32)[None])
  fill = 1.0 if has_init else 0.0
  s_enc, aux = wrapper._pack_fwd_segment_metadata(
      seg, jnp.full((1, _KERNEL - 1), fill), jnp.full((1,), fill), 1, _CHUNK, _KERNEL
  )
  b[:, nv] = np.asarray(s_enc)
  b[:, nv + 1] = np.asarray(aux)
  return b


def _conv_stage(nk, nv, seg_row, has_init, bias, seed, natural_conv):
  """(compact, natural) per-tile values [q, k, v, conv state, halo rows] for every tile of one 4-chunk sequence."""
  rng = np.random.default_rng(seed)
  seg = seg_row is not None
  cfg = _cfg(nk, nv, seg)
  dim, t = cfg.dim_size, 4 * _CHUNK
  prev, rows = _KERNEL - 1, gdn_config.NATURAL_CONV_CARRY_ROWS
  xs = jnp.asarray(rng.standard_normal((t, dim)).astype(np.float32)).astype(jnp.bfloat16)
  w = jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.5).astype(np.float32))
  cb = jnp.asarray(rng.standard_normal((dim,)).astype(np.float32)) if bias else None
  hbm = jnp.asarray(rng.standard_normal((prev, dim)).astype(np.float32))
  rec = jnp.zeros((1, nv, _HEAD_DIM, _HEAD_DIM), jnp.float32)
  b_all = (
      _b_lanes(seg_row if seg else [0] * t, has_init, nv, rng)
      if seg
      else rng.standard_normal((t, 128)).astype(np.float32)
  )
  n = t // _CHUNK
  md = memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=jnp.int32(n),
      p_id_to_s_idx=jnp.zeros((n,), jnp.int32),
      p_id_to_r_base=jnp.arange(n, dtype=jnp.int32) * _CHUNK,
      p_id_to_r_size=jnp.full((n,), _CHUNK, jnp.int32),
      p_id_is_first_tile=jnp.arange(n) == 0,
      p_id_is_last_tile=jnp.arange(n) == n - 1,
      s_idx_has_initial_state=jnp.array([has_init]),
      s_idx_to_state_indices=jnp.array([1], jnp.int32),
  )
  carry_c, carry_n = jnp.zeros((1, prev, 1, dim), jnp.float32), jnp.zeros((rows, dim), jnp.float32)
  compact, natural = [], []
  for j in range(n):
    x, b = xs[j * _CHUNK : (j + 1) * _CHUNK], jnp.asarray(b_all[j * _CHUNK : (j + 1) * _CHUNK])
    real, prev_conv, _ = vmem_ldst.load_and_select_states(
        metadata_ref=md,
        p_id=jnp.int32(j),
        conv_state_slot_ref=hbm.reshape(1, prev, 1, dim),
        recurrent_slot_ref=rec,
        carry_conv_scratch_ref=carry_c,
        carry_recurrent_scratch_ref=rec,
        cfg=cfg,
    )
    lhs = jnp.concatenate([prev_conv, x.astype(jnp.float32).reshape(1, _CHUNK, 1, dim)], axis=1)
    out, new_cs = compute_conv1d.causal_conv1d(real, lhs, w, cb, cfg, b.reshape(1, _CHUNK, 1, 128) if seg else None)
    y = jax.nn.silu(out)[0, :, 0, :]
    kd = nk * _HEAD_DIM
    compact.append(
        [
            y[:, :kd].reshape(_CHUNK, nk, _HEAD_DIM).transpose(1, 0, 2),
            y[:, kd : 2 * kd].reshape(_CHUNK, nk, _HEAD_DIM).transpose(1, 0, 2),
            y[:, 2 * kd :].reshape(_CHUNK, nv, _HEAD_DIM).transpose(1, 0, 2),
            new_cs[0, :, 0, :],
            prev_conv[0, :, 0, :],
        ]
    )
    record = md.get_record(jnp.int32(j), 0)
    halo = vmem_ldst.select_natural_conv_halo(hbm, carry_n, record.is_first_tile, md.s_idx_has_initial_state[0])
    x32 = x.astype(jnp.float32)
    q, k, v = vmem_ldst.split_qkv_natural(jax.nn.silu(natural_conv(x32, halo, w, cb, cfg, b if seg else None)), cfg)
    natural.append([q, k, v, x32[_CHUNK - prev :], halo[rows - prev :]])
    carry_c, carry_n = new_cs, x32[_CHUNK - rows :]
  return compact, natural


_SEG_ROWS = {
    "none": None,
    # Documents start at chunk rows 0, 1, 3 and mid-chunk.
    "starts": list(np.repeat([1, 2, 3, 4, 5, 6], [1, 63, 65, 2, 61, 64])),
    "padding": list(np.concatenate([np.repeat([1, 2], [100, 120]), np.zeros(36, np.int64)])),
}


class ConvStageTest(parameterized.TestCase):
  """The natural-layout conv stage gives the compact path's bits, eager on CPU."""

  @parameterized.product(seg=("none", "starts", "padding"), has_init=(False, True), bias=(False, True))
  def test_bit_identical(self, seg, has_init, bias):
    with _eager_roll():
      compact, natural = _conv_stage(2, 4, _SEG_ROWS[seg], has_init, bias, 7, compute_conv1d.natural_causal_conv1d)
    for tile, (c, n) in enumerate(zip(compact, natural)):
      for name, x, y in zip(("q", "k", "v", "conv_state", "halo"), c, n):
        self.assertEqual(x.shape, y.shape, f"tile {tile} {name}")
        self.assertEqual(_bits(x), _bits(y), f"tile {tile} {name}")

  def test_production_width(self):
    with _eager_roll():
      compact, natural = _conv_stage(16, 64, _SEG_ROWS["starts"], True, False, 9, compute_conv1d.natural_causal_conv1d)
    for c, n in zip(compact, natural):
      for x, y in zip(c, n):
        self.assertEqual(_bits(x), _bits(y))

  def test_reordered_sum_is_caught(self):
    """Positive control: adding the same taps in reverse order changes bits, so the comparison can fail."""

    def reversed_taps(x, halo, w, cb, cfg, b_large):
      parts = []
      for k in range(cfg.kernel_size):
        wk = jnp.zeros_like(w).at[k].set(w[k])
        parts.append(compute_conv1d.natural_causal_conv1d(x, halo, wk, None, cfg, b_large))
      out = jnp.zeros_like(parts[0])
      for p in parts[::-1]:
        out = out + p
      return out if cb is None else out + cb.reshape(1, -1)

    with _eager_roll():
      compact, natural = _conv_stage(2, 4, None, True, False, 7, reversed_taps)
    self.assertNotEqual(_bits(compact[1][2]), _bits(natural[1][2]))


class HeadGroupTest(parameterized.TestCase):
  """chunked_gdn with kq-head groups gives the all-heads bits."""

  @parameterized.product(
      seg=(False, True), mode=(gdn_config.GDNMode.PER_SEQ, gdn_config.GDNMode.PER_SEQ_STATES), group=(4, 2)
  )
  def test_bit_identical(self, seg, mode, group):
    rng = np.random.default_rng(group + 2 * seg)
    nk, nv = 16, 64
    cfg = _cfg(nk, nv, seg, mode=mode, batch=_CHUNK)
    q = jnp.asarray(rng.standard_normal((1, nk, _CHUNK, _HEAD_DIM)).astype(np.float32))
    k = jnp.asarray(rng.standard_normal((1, nk, _CHUNK, _HEAD_DIM)).astype(np.float32))
    v = jnp.asarray(rng.standard_normal((1, nv, _CHUNK, _HEAD_DIM)).astype(np.float32))
    seg_row = list(np.repeat([1, 2, 3], [10, 40, 14])) if seg else None
    b = jnp.asarray(_b_lanes(seg_row, True, nv, rng) if seg else rng.standard_normal((_CHUNK, 128)).astype(np.float32))
    a = jnp.asarray(rng.standard_normal((_CHUNK, 128)).astype(np.float32))
    state = jnp.asarray(rng.standard_normal((1, nv, _HEAD_DIM, _HEAD_DIM)).astype(np.float32) * 0.1)
    a_log = jnp.pad(jnp.asarray(rng.standard_normal((nv,)).astype(np.float32) * 0.1), (0, 128 - nv))
    dtb = jnp.pad(jnp.asarray(rng.standard_normal((nv,)).astype(np.float32) * 0.1), (0, 128 - nv))

    def run(c):
      return compute_gdn.chunked_gdn(
          real_sizes=jnp.array([_CHUNK], jnp.int32),
          q_large=q,
          k_large=k,
          v_large=v,
          b_large=b[None, None],
          a_large=a[None, None],
          state_prev=state,
          a_log=a_log,
          dt_bias=dtb,
          cfg=c,
      )

    ref = jax.jit(lambda: run(cfg))()
    got = jax.jit(lambda: run(dataclasses.replace(cfg, head_group_size=group)))()
    for x, y in zip(ref, got):
      self.assertEqual(x is None, y is None)
      if x is not None:
        self.assertEqual(_bits(x), _bits(y))


def _pltpu_roll_cpu(x, shift, axis, *, stride=None, stride_axis=None):
  assert stride is None and stride_axis is None
  return jnp.roll(x, shift, axis=axis)


def _pltpu_repeat_cpu(x, repeats, axis):
  """CPU stand-in for pltpu.repeat."""
  reps = [1] * x.ndim
  reps[axis] = repeats
  return jnp.tile(x, reps)


class _FlatView:
  """Flat slice view over a boxed ref slice."""

  def __init__(self, box, idx, shape):
    self.box, self.idx, self.shape = box, idx, shape

  def __getitem__(self, slc):
    return self.box.arr[self.idx].reshape(self.shape)[slc]

  def __setitem__(self, slc, val):
    sub = self.box.arr[self.idx]
    flat = sub.reshape(self.shape).at[slc].set(val)
    self.box.arr = self.box.arr.at[self.idx].set(flat.reshape(sub.shape))


class _BoxRef:
  """Array-backed stand-in for a VMEM ref whose helpers reshape it (the CPU interpreters cannot)."""

  def __init__(self, arr):
    self.arr = arr

  dtype = property(lambda self: self.arr.dtype)
  shape = property(lambda self: self.arr.shape)

  def reshape(self, *shape):
    return self.arr.reshape(*shape)

  def __getitem__(self, idx):
    return self.arr[idx]

  def __setitem__(self, idx, val):
    self.arr = val if idx is Ellipsis else self.arr.at[idx].set(val)


class _At:
  """Index helper for _BoxRef."""

  def __init__(self, box):
    self.box = box

  def __getitem__(self, idx):
    return types.SimpleNamespace(reshape=lambda *s: _FlatView(self.box, idx, s))


_BoxRef.at = property(_At)


@contextlib.contextmanager
def _interpret_context():
  """Real Pallas GDN forward kernels on CPU (gdn_pass1_states_and_out_proj_test's harness + the natural-path helpers)."""
  o_qkv, o_c2l, o_up = vmem_ldst.load_as_qkv_large, vmem_ldst.load_compact_to_large, vmem_ldst.upcast_large_to_compact
  o_lr, o_sr = vmem_ldst.load_compact_rows_as_large, vmem_ldst.store_large_rows_to_compact

  def up(in_ref, out_ref):
    box = _BoxRef(out_ref[...])
    o_up(_BoxRef(in_ref[...]), box)
    out_ref[...] = box.arr

  def store_rows(rows, ref):
    box = _BoxRef(ref[...])
    o_sr(rows, box)
    ref[...] = box.arr

  info = types.SimpleNamespace(num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024 * 1024)
  with (
      mock.patch.object(gdn_api, "target_platform", return_value="tpu"),
      mock.patch.object(pltpu, "get_tpu_info", return_value=info),
      mock.patch.object(pltpu, "repeat", side_effect=_pltpu_repeat_cpu, create=True),
      mock.patch.object(pltpu, "roll", side_effect=_pltpu_roll_cpu, create=True),
      mock.patch.object(vmem_ldst, "load_as_qkv_large", side_effect=lambda r, c: o_qkv(_BoxRef(r[...]), c)),
      mock.patch.object(vmem_ldst, "load_compact_to_large", side_effect=lambda r: o_c2l(_BoxRef(r[...]))),
      mock.patch.object(vmem_ldst, "upcast_large_to_compact", side_effect=up),
      mock.patch.object(vmem_ldst, "load_compact_rows_as_large", side_effect=lambda r: o_lr(_BoxRef(r[...]))),
      mock.patch.object(vmem_ldst, "store_large_rows_to_compact", side_effect=store_rows),
      pltpu.force_tpu_interpret_mode(),
  ):
    yield


def _kernel_inputs(seg, init, seed, nk=2, nv=4, batch=2, nchunks=3):
  """Construct synthetic inputs for local GDN kernel interpret checks."""
  rng = np.random.default_rng(seed)
  dim, t = (2 * nk + nv) * _HEAD_DIM, nchunks * _CHUNK
  segment_ids = None
  if seg:
    r0 = np.repeat([1, 2, 3, 4, 5], [65, 1, 2, 60, t - 128])
    r1 = np.concatenate([np.repeat([7, 8, 9], [3, 61, 66]), np.zeros(t - 130, np.int64)])
    segment_ids = jnp.asarray(np.stack([r0, r1][:batch]), jnp.int32)
  return {
      "qkv": jnp.asarray((rng.standard_normal((batch, t, dim)) * 0.5).astype(np.float32)).astype(jnp.bfloat16),
      "b": jnp.asarray(rng.standard_normal((batch, t, nv)).astype(np.float32)).astype(jnp.bfloat16),
      "a": jnp.asarray(rng.standard_normal((batch, t, nv)).astype(np.float32)).astype(jnp.bfloat16),
      "conv_weight": jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.3).astype(np.float32)),
      "conv_bias": jnp.asarray((rng.standard_normal((dim,)) * 0.1).astype(np.float32)),
      "a_log": jnp.asarray((rng.standard_normal((nv,)) * 0.1).astype(np.float32)),
      "dt_bias": jnp.asarray((rng.standard_normal((nv,)) * 0.1 - 4.0).astype(np.float32)),
      "conv_state": (
          jnp.asarray((rng.standard_normal((batch, _KERNEL - 1, dim)) * 0.3).astype(np.float32)).astype(jnp.bfloat16)
          if init
          else None
      ),
      "recurrent_state": (
          jnp.asarray((rng.standard_normal((batch, nv, _HEAD_DIM, _HEAD_DIM)) * 0.3).astype(np.float32)) if init else None
      ),
      "num_k_heads": nk,
      "num_v_heads": nv,
      "head_k_dim": _HEAD_DIM,
      "head_v_dim": _HEAD_DIM,
      "conv_kernel_size": _KERNEL,
      "chunk_size": _CHUNK,
      "use_qk_norm_in_gdn": True,
      "compute_dtype": jnp.float32,
      "segment_ids": segment_ids,
      "conv_dense_token_stack": True,
      "bf16_qkv_dma": True,
  }


class KernelInterpretTest(parameterized.TestCase):
  """Every output of the real forward and states-only kernels, key on vs off, in TPU interpret mode on CPU."""

  @parameterized.product(seg=(False, True), init=(False, True))
  def test_kernels_bit_identical(self, seg, init):
    common = _kernel_inputs(seg, init, 3 + 2 * seg + init)
    with _interpret_context():
      ref_f = gdn_api._run_local_gdn_decoupled_fwd(**common)
      ref_s = gdn_api._run_local_gdn_states_fwd(**common)
      for nat, hg in ((True, 0), (False, 2), (True, 2)):
        got_f = gdn_api._run_local_gdn_decoupled_fwd(**common, conv_natural_layout=nat, fwd_head_group_size=hg)
        got_s = gdn_api._run_local_gdn_states_fwd(**common, conv_natural_layout=nat, fwd_head_group_size=hg)
        for x, y in zip(jax.tree.leaves((ref_f, ref_s)), jax.tree.leaves((got_f, got_s))):
          self.assertEqual(_bits(x), _bits(y), f"natural={nat} group={hg}")

  def test_swapped_heads_are_caught(self):
    """Positive control: a natural path that swaps q and k changes the kernel outputs."""
    common = _kernel_inputs(True, True, 5)
    split = vmem_ldst.split_qkv_natural

    def swapped_split(y, c):
      q, k, v = split(y, c)
      return k, q, v

    jax.clear_caches()  # the kernels are jitted: a cached trace of the natural path would hide the patch
    try:
      with _interpret_context():
        ref = gdn_api._run_local_gdn_decoupled_fwd(**common)
        with mock.patch.object(vmem_ldst, "split_qkv_natural", side_effect=swapped_split):
          got = gdn_api._run_local_gdn_decoupled_fwd(**common, conv_natural_layout=True)
    finally:
      jax.clear_caches()
    self.assertNotEqual(_bits(ref[0][0]), _bits(got[0][0]))

  def test_natural_layout_needs_the_bf16_dma_buffer(self):
    common = {**_kernel_inputs(False, False, 1), "bf16_qkv_dma": False}
    with _interpret_context(), self.assertRaisesRegex(ValueError, "bf16 qkv DMA"):
      gdn_api._run_local_gdn_decoupled_fwd(**common, conv_natural_layout=True)

  def test_group_must_divide_heads(self):
    with self.assertRaisesRegex(ValueError, "gdn_fwd_kernel_head_group_size=3"):
      gdn_api._fwd_kernel_head_group_size(3, _CHUNK, 4, 8, jnp.bfloat16)


def _validator_cfg(**overrides):
  """Minimal namespace satisfying validate_gdn_config."""
  fields = {
      "decoder_block": config_types.DecoderBlockType.QWEN3_5,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_context_usp_ulysses_parallelism": 1,
      "dcn_context_usp_ulysses_parallelism": 1,
      "use_gdn_kernel": True,
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "gdn_chunk_size": 64,
      "gdn_key_head_dim": 128,
      "gdn_value_head_dim": 128,
      "gdn_num_key_heads": 16,
      "gdn_num_value_heads": 64,
      "use_qk_norm_in_gdn": True,
      "max_target_length": 4096,
      "gdn_conv1d_input_in_qkv_dtype": False,
      "gdn_conv1d_dense_token_stack": False,
      "gdn_gated_norm_layout": "auto",
      "gdn_cp_mode": "auto",
      "gdn_cp_pass1_states_and_bf16_qkv": True,
      "dtype": "bfloat16",
      "packing": True,
      "gdn_in_proj_kernel_order": False,
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


class ValidatorTest(parameterized.TestCase):

  def test_defaults_off(self):
    fields = config_types.Qwen3Next.model_fields
    self.assertIs(fields["gdn_conv1d_natural_layout"].default, False)
    self.assertEqual(fields["gdn_fwd_kernel_head_group_size"].default, 0)
    self.assertIs(gdn_config.GDNConfig.__dataclass_fields__["conv_natural_layout"].default, False)

  def test_accepted(self):
    config_types.MaxTextConfig.validate_gdn_config(
        _validator_cfg(gdn_conv1d_natural_layout=True, gdn_fwd_kernel_head_group_size=4), is_rl_config=False
    )

  @parameterized.named_parameters(
      (
          "natural_without_bf16_dma",
          {"gdn_conv1d_natural_layout": True, "gdn_cp_pass1_states_and_bf16_qkv": False},
          "gdn_conv1d_natural_layout",
      ),
      ("natural_without_kernel", {"gdn_conv1d_natural_layout": True, "use_gdn_kernel": False}, "use_gdn_kernel"),
      ("group_not_dividing", {"gdn_fwd_kernel_head_group_size": 3}, "gdn_fwd_kernel_head_group_size=3"),
      ("group_negative", {"gdn_fwd_kernel_head_group_size": -1}, "gdn_fwd_kernel_head_group_size"),
      (
          "group_without_gdn",
          {"gdn_fwd_kernel_head_group_size": 4, "decoder_block": config_types.DecoderBlockType.LLAMA2},
          "Gated DeltaNet",
      ),
  )
  def test_rejected(self, overrides, pattern):
    with self.assertRaisesRegex(ValueError, pattern):
      config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(**overrides), is_rl_config=False)

  def test_layout_only_field(self):
    """The natural layout is left out of the kernel name and metadata (the program outside the kernel is unchanged)."""
    off = _cfg(2, 4, True)
    on = dataclasses.replace(off, conv_natural_layout=True)
    self.assertEqual(off.get_kernel_name(), on.get_kernel_name())
    self.assertEqual(off.get_metadata(), on.get_metadata())


if __name__ == "__main__":
  absltest.main()
