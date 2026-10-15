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
"""The Pallas CP composition scans (`dh_scan`) against the XLA `cp_gdn` trees they replace on TPU.

Runs the kernel in Pallas interpret mode. XLA:CPU computes the tree's f32 matmuls in plain f32, so the comparison
bounds the scans' 3-pass bf16 matmul error and the reduction order, not TPU numerics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.gdn.gdn_bwd import cp_gdn
from maxtext.kernels.gdn.gdn_bwd import dh_scan

_NK, _NV, _D, _C, _NC = 2, 8, 128, 64, 4


def _inputs(seed=0):
  """One rank's K-slice conv output, gates and per-chunk T^{-1}, with weak decay."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  s = _NC * _C
  k_conv = jax.random.normal(keys[0], (1, s, _NK * _D), jnp.float32)
  b = jax.random.normal(keys[1], (1, s, _NV), jnp.float32).astype(jnp.bfloat16)
  # Weak decay keeps the product of the 4 chunk transitions far from 0.
  a = (jax.random.normal(keys[2], (1, s, _NV), jnp.float32) - 6.0).astype(jnp.bfloat16)
  a_log = jax.random.normal(keys[3], (_NV,), jnp.float32) * 0.3
  dt_bias = jax.random.normal(keys[4], (_NV,), jnp.float32) * 0.3
  t_inv = jnp.tril(jax.random.normal(keys[5], (1, _NC, _NV, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C)
  return k_conv, b, a, a_log, dt_bias, t_inv


def _tree_m(k_conv, b, a, a_log, dt_bias, t_inv, segment_ids, init_seg):
  """m_local of the XLA tree (f32 on CPU)."""
  m, _ = cp_gdn.compose_local_from_t_inv(
      k_conv,
      b,
      a,
      a_log,
      dt_bias,
      t_inv,
      num_k_heads=_NK,
      num_v_heads=_NV,
      head_k_dim=_D,
      head_v_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      s_ext_pass1=jnp.zeros((1, _NV, _D, _D), jnp.float32),
      segment_ids=segment_ids,
      init_seg=init_seg,
  )
  return m


def _scan_m(k_conv, b, a, a_log, dt_bias, t_inv, segment_ids, init_seg):
  """m_local of the Pallas scan, interpreted."""
  return dh_scan.gdn_cp_m_scan(
      k_conv,
      b,
      a,
      a_log,
      dt_bias,
      t_inv,
      num_v_heads=_NV,
      kq_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )


def _rel_err(got, ref):
  got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
  return np.abs(got - ref).max() / np.abs(ref).max()


# A document continuing from the previous rank and spanning the whole rank (the long single-document case), and a
# packed row: that document ends inside chunk 1, two more follow with boundaries inside chunks, then padding.
_SINGLE = np.ones((1, _NC * _C), np.int32)
_PACKED = np.zeros((1, _NC * _C), np.int32)
_PACKED[0, :70], _PACKED[0, 70:200], _PACKED[0, 200:240] = 1, 2, 3


@pytest.mark.parametrize("head_tile", [64, 4])  # one head group, two head groups
@pytest.mark.parametrize("layout", ["none", "single", "packed"])
def test_m_scan_matches_compose_local_tree(layout, head_tile, monkeypatch):
  monkeypatch.setattr(dh_scan, "_M_SCAN_HEAD_TILE", head_tile)
  segment_ids = {"none": None, "single": jnp.asarray(_SINGLE), "packed": jnp.asarray(_PACKED)}[layout]
  init_seg = None if segment_ids is None else jnp.ones((1,), jnp.float32)
  args = _inputs()
  ref = _tree_m(*args, segment_ids, init_seg)
  got = _scan_m(*args, segment_ids, init_seg)
  assert float(jnp.abs(ref).max()) > 1e-4, "degenerate reference: the transitions vanish"
  assert _rel_err(got, ref) < 5e-5


def test_m_scan_sees_chunk_order_and_segment_starts():
  """Negative controls: the comparison above can fail."""
  args = _inputs()
  segment_ids = jnp.asarray(_SINGLE)
  ref = _tree_m(*args, segment_ids, jnp.ones((1,), jnp.float32))
  k_conv, b, a, a_log, dt_bias, t_inv = args
  reversed_chunks = _scan_m(k_conv, b, a, a_log, dt_bias, t_inv[:, ::-1], segment_ids, jnp.ones((1,), jnp.float32))
  assert _rel_err(reversed_chunks, ref) > 1e-3
  # Without the incoming document, the rank's first token starts a new one, and nothing from the incoming state
  # reaches the rank's end state.
  fresh = _scan_m(*args, segment_ids, jnp.zeros((1,), jnp.float32))
  assert float(jnp.abs(fresh).max()) == 0.0


def _bwd_inputs(seed=1):
  """One rank's full conv output (q, k, v), gates, output cotangent and per-chunk T^{-1}."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 7)
  s = _NC * _C
  qkv_conv = jax.random.normal(keys[0], (1, s, 2 * _NK * _D + _NV * _D), jnp.float32).astype(jnp.bfloat16)
  b = jax.random.normal(keys[1], (1, s, _NV), jnp.float32).astype(jnp.bfloat16)
  a = (jax.random.normal(keys[2], (1, s, _NV), jnp.float32) - 2.0).astype(jnp.bfloat16)
  a_log = jax.random.normal(keys[3], (_NV,), jnp.float32) * 0.3
  dt_bias = jax.random.normal(keys[4], (_NV,), jnp.float32) * 0.3
  do = jax.random.normal(keys[5], (1, s, _NV, _D), jnp.float32).astype(jnp.bfloat16)
  t_inv = jnp.tril(jax.random.normal(keys[6], (1, _NC, _NV, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C)
  return qkv_conv, b, a, a_log, dt_bias, do, t_inv


@pytest.mark.parametrize("head_tile", [32, 4])  # one head group, two head groups
@pytest.mark.parametrize("layout", ["none", "single", "packed"])
def test_dh_scan_matches_compose_bwd_local_tree(layout, head_tile, monkeypatch):
  monkeypatch.setattr(dh_scan, "_DH_SCAN_HEAD_TILE", head_tile)
  segment_ids = {"none": None, "single": jnp.asarray(_SINGLE), "packed": jnp.asarray(_PACKED)}[layout]
  init_seg = None if segment_ids is None else jnp.ones((1,), jnp.float32)
  args = _bwd_inputs()
  _, ref = cp_gdn.compose_bwd_local_from_t_inv(
      *args,
      num_k_heads=_NK,
      num_v_heads=_NV,
      head_k_dim=_D,
      head_v_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
  )
  got = dh_scan.gdn_cp_dh_scan(
      *args,
      num_v_heads=_NV,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )
  assert float(jnp.abs(ref).max()) > 1e-4, "degenerate reference"
  assert _rel_err(got, ref) < 5e-5
  # Negative control: a perturbed t_inv must show.
  qkv_conv, b, a, a_log, dt_bias, do, t_inv = args
  bad = dh_scan.gdn_cp_dh_scan(
      qkv_conv,
      b,
      a,
      a_log,
      dt_bias,
      do,
      t_inv * 1.01,
      num_v_heads=_NV,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )
  assert _rel_err(bad, ref) > 1e-3
