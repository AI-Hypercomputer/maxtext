# Copyright 2023-2026 Google LLC
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

"""Pallas TPU kernels for the KDA chunk-state recurrence, with a hand-written backward.

The sub-block KDA core (`models/olmoe3.py`) hoists everything that does not depend on the carried state out of
the chunk loop, leaving per chunk c (state S_c entering the chunk, `[dk, dv]` per batch and head):

    delta_c = u_c - w_c S_c                       (bf16 output)
    S_{c+1} = dl_c * S_c + kc_c^T delta_c         (dl_c per key channel, 0 after a reset; S kept in bf16)

As an XLA while loop this costs ~27 us per chunk on v4 for ~5 us of work, so the loop is latency-bound. Here
each (batch, head) runs its chunks in one kernel with S resident in VMEM, and the grid's head axis is
`parallel` so v4's two megacore TensorCores split the heads.

The backward carries R_c = dL/dS_{c+1} in reverse:

    dd_c    = g_delta_c + kc_c R_c                (= du_c)
    R_{c-1} = g_states_c + dl_c * R_c - w_c^T dd_c

and the remaining gradients need no recurrence, so they run batched over all chunks outside the kernel:
dw_c = -dd_c S_c^T, dkc_c = delta_c R_c^T, ddl_c = sum_v S_c * R_c.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _fwd_kernel(uw_ref, kc_ref, dl_ref, states_ref, delta_ref, s_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  state_dtype = states_ref.dtype
  dv = s_ref.shape[1]
  s = s_ref[...]  # f32 holding state_dtype values
  s_cast = s.astype(state_dtype)
  states_ref[0, 0] = s_cast
  uw = uw_ref[0, 0]
  w = uw[:, dv:].astype(state_dtype)
  delta = uw[:, :dv] - jnp.dot(w, s_cast, preferred_element_type=jnp.float32)
  delta_cast = delta.astype(state_dtype)
  delta_ref[0, 0] = delta_cast
  new = s * dl_ref[0, 0] + jax.lax.dot_general(
      kc_ref[0, 0], delta_cast, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
  )
  s_ref[...] = new.astype(state_dtype).astype(jnp.float32)


def _bwd_kernel(
    gs_ref, gd_ref, kc_ref, uw_ref, dl_ref, states_ref, delta_ref, duw_ref, dkc_ref, ddl_ref, r_ref
):
  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  state_dtype = kc_ref.dtype
  dv = r_ref.shape[1]
  r = r_ref[...]
  r_c = r.astype(state_dtype)
  s_c = states_ref[0, 0]
  delta_c = delta_ref[0, 0]

  dd = gd_ref[0, 0].astype(jnp.float32) + jnp.dot(
      kc_ref[0, 0], r_c, preferred_element_type=jnp.float32
  )
  dd_c = dd.astype(state_dtype)
  dw = -jax.lax.dot_general(
      dd_c, s_c, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
  )
  dkc = jax.lax.dot_general(
      delta_c, r_c, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
  )
  ddl = jnp.sum(s_c.astype(jnp.float32) * r, axis=-1, keepdims=True)

  duw_ref[0, 0] = jnp.concatenate([dd, dw], axis=-1)
  dkc_ref[0, 0] = dkc.astype(state_dtype)
  ddl_ref[0, 0] = ddl

  w_c = uw_ref[0, 0][:, dv:].astype(state_dtype)
  r_ref[...] = (
      gs_ref[0, 0].astype(jnp.float32)
      + r * dl_ref[0, 0]
      - jax.lax.dot_general(w_c, dd_c, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
  )


def _spec(shape, reverse: bool = False, num_chunks: int = 0):
  if reverse:
    return pl.BlockSpec(shape, lambda bh, c: (bh, num_chunks - 1 - c, 0, 0))
  return pl.BlockSpec(shape, lambda bh, c: (bh, c, 0, 0))


def _scan_fwd_call(uw, kc, dl, interpret):
  bh, n, c, dk = kc.shape
  dv = uw.shape[-1] - dk
  state_dtype = kc.dtype
  return pl.pallas_call(
      _fwd_kernel,
      grid=(bh, n),
      in_specs=[_spec((1, 1, c, dv + dk)), _spec((1, 1, c, dk)), _spec((1, 1, dk, 1))],
      out_specs=[_spec((1, 1, dk, dv)), _spec((1, 1, c, dv))],
      out_shape=[
          jax.ShapeDtypeStruct((bh, n, dk, dv), state_dtype),
          jax.ShapeDtypeStruct((bh, n, c, dv), state_dtype),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      interpret=interpret,
  )(uw, kc, dl)


def _scan_bwd_call(g_states, g_delta, uw, kc, dl, states, delta, interpret):
  bh, n, c, dv = g_delta.shape
  dk = kc.shape[-1]
  state_dtype = kc.dtype
  rev = functools.partial(_spec, reverse=True, num_chunks=n)
  return pl.pallas_call(
      _bwd_kernel,
      grid=(bh, n),
      in_specs=[
          rev((1, 1, dk, dv)),
          rev((1, 1, c, dv)),
          rev((1, 1, c, dk)),
          rev((1, 1, c, dv + dk)),
          rev((1, 1, dk, 1)),
          rev((1, 1, dk, dv)),
          rev((1, 1, c, dv)),
      ],
      out_specs=[rev((1, 1, c, dv + dk)), rev((1, 1, c, dk)), rev((1, 1, dk, 1))],
      out_shape=[
          jax.ShapeDtypeStruct((bh, n, c, dv + dk), jnp.float32),
          jax.ShapeDtypeStruct((bh, n, c, dk), state_dtype),
          jax.ShapeDtypeStruct((bh, n, dk, 1), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      interpret=interpret,
  )(g_states.astype(state_dtype), g_delta.astype(state_dtype), kc, uw, dl, states, delta)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def kda_state_scan(uw, kc, dl, interpret=False):
  """Chunk states and deltas of the KDA recurrence.

  Args:
    uw: `[BH, N, C, dv + dk]` f32, the WY products u (values, first dv lanes) and w (multiplies the state).
      Taken whole, as the einsum produces it, so XLA need not slice and relayout u and w for the kernel.
    kc: `[BH, N, C, dk]` bf16 or f32, keys decayed to the chunk end (zero outside the chunk's last segment).
    dl: `[BH, N, dk, 1]` f32, per-channel decay over the chunk, 0 where the chunk holds a reset.
    interpret: run the kernels in interpret mode (CPU tests).

  Returns:
    states `[BH, N, dk, dv]` and delta `[BH, N, C, dv]` in `kc.dtype`.
  """
  states, delta = _scan_fwd_call(uw, kc, dl, interpret)
  return states, delta


def _kda_state_scan_fwd(uw, kc, dl, interpret):
  states, delta = _scan_fwd_call(uw, kc, dl, interpret)
  return (states, delta), (states, delta, uw, kc, dl)


def _kda_state_scan_bwd(interpret, res, g):
  states, delta, uw, kc, dl = res
  g_states, g_delta = g
  return _scan_bwd_call(g_states, g_delta, uw, kc, dl, states, delta, interpret)


kda_state_scan.defvjp(_kda_state_scan_fwd, _kda_state_scan_bwd)

_PARAMS = pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))


def _tn(a, b):  # a^T b, contracting the leading (chunk) axis of both
  return jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _nt(a, b):  # a b^T, contracting the trailing axis of both
  return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _view(ref):
  """The 2-D tile of a block, whatever its leading unit dims."""
  return ref.at[(0,) * (len(ref.shape) - 2)]


def _chunk_fwd_kernel(x_ref, vb_ref, kb_ref, kc_ref, dl_ref, qg_ref, sc_ref, out_ref, states_ref, delta_ref, s_ref):
  x_ref, vb_ref, kb_ref, kc_ref, dl_ref, qg_ref, sc_ref, out_ref, states_ref, delta_ref = (
      _view(r) for r in (x_ref, vb_ref, kb_ref, kc_ref, dl_ref, qg_ref, sc_ref, out_ref, states_ref, delta_ref)
  )

  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  f32, bf16 = jnp.float32, jnp.bfloat16
  s = s_ref[...]  # f32 holding bf16 values
  s_bf = s.astype(bf16)
  states_ref[...] = s_bf
  x = x_ref[...]
  u = jnp.dot(x, vb_ref[...], preferred_element_type=f32)
  w = jnp.dot(x, kb_ref[...], preferred_element_type=f32)
  delta_bf = (u - jnp.dot(w.astype(bf16), s_bf, preferred_element_type=f32)).astype(bf16)
  delta_ref[...] = delta_bf
  out_ref[...] = jnp.dot(qg_ref[...], s_bf, preferred_element_type=f32) + jnp.dot(
      sc_ref[...], delta_bf, preferred_element_type=f32
  )
  s_ref[...] = (s * dl_ref[...] + _tn(kc_ref[...], delta_bf)).astype(bf16).astype(f32)


def _chunk_bwd_kernel(
    x_ref,
    vb_ref,
    kb_ref,
    kc_ref,
    dl_ref,
    qg_ref,
    sc_ref,
    st_ref,
    de_ref,
    do_ref,
    dx_ref,
    dvb_ref,
    dkb_ref,
    dkc_ref,
    ddl_ref,
    dqg_ref,
    dsc_ref,
    r_ref,
):
  refs = (x_ref, vb_ref, kb_ref, kc_ref, dl_ref, qg_ref, sc_ref, st_ref, de_ref, do_ref)
  x_ref, vb_ref, kb_ref, kc_ref, dl_ref, qg_ref, sc_ref, st_ref, de_ref, do_ref = (_view(r) for r in refs)
  outs = (dx_ref, dvb_ref, dkb_ref, dkc_ref, ddl_ref, dqg_ref, dsc_ref)
  dx_ref, dvb_ref, dkb_ref, dkc_ref, ddl_ref, dqg_ref, dsc_ref = (_view(r) for r in outs)

  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  f32, bf16 = jnp.float32, jnp.bfloat16
  states, delta = st_ref[...], de_ref[...]
  x, kb, vb, kc = x_ref[...], kb_ref[...], vb_ref[...], kc_ref[...]
  d_out = do_ref[...].astype(bf16)
  # Readout o = qg S + scores delta.
  dqg_ref[...] = _nt(d_out, states).astype(dqg_ref.dtype)
  dsc_ref[...] = _nt(d_out, delta).astype(dsc_ref.dtype)
  g_states = _tn(qg_ref[...], d_out)
  g_delta = _tn(sc_ref[...], d_out)
  # Recurrence, carrying r = dL/dS_{c+1} backwards.
  r = r_ref[...]
  r_bf = r.astype(bf16)
  dd = g_delta + jnp.dot(kc, r_bf, preferred_element_type=f32)
  dd_bf = dd.astype(bf16)
  w_bf = jnp.dot(x, kb, preferred_element_type=f32).astype(bf16)
  dw_bf = (-_nt(dd_bf, states)).astype(bf16)
  dkc_ref[...] = _nt(delta, r_bf).astype(dkc_ref.dtype)
  ddl_ref[...] = jnp.sum(states.astype(f32) * r, axis=-1, keepdims=True)
  r_ref[...] = g_states + r * dl_ref[...] - _tn(w_bf, dd_bf)
  # u = X vb, w = X kb.
  dx_ref[...] = (_nt(dd_bf, vb) + _nt(dw_bf, kb)).astype(dx_ref.dtype)
  dvb_ref[...] = _tn(x, dd_bf).astype(dvb_ref.dtype)
  dkb_ref[...] = _tn(x, dw_bf).astype(dkb_ref.dtype)


def _chunk_fwd_call(x, vb, kb, kc, dl, qg, sc, interpret):
  bh, n, c = x.shape[:3]
  dk, dv = dl.shape[2], vb.shape[-1]
  sp = lambda *s: _spec((1, 1) + s)
  tok = lambda d: sp(c, d)
  tok_shape = (bh, n, c, dv)
  return pl.pallas_call(
      _chunk_fwd_kernel,
      grid=(bh, n),
      in_specs=[sp(c, c), tok(dv), tok(dk), tok(dk), sp(dk, 1), tok(dk), sp(c, c)],
      out_specs=[tok(dv), sp(dk, dv), sp(c, dv)],
      out_shape=[
          jax.ShapeDtypeStruct(tok_shape, jnp.float32),
          jax.ShapeDtypeStruct((bh, n, dk, dv), jnp.bfloat16),
          jax.ShapeDtypeStruct((bh, n, c, dv), jnp.bfloat16),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=_PARAMS,
      interpret=interpret,
  )(x, vb, kb, kc, dl, qg, sc)


def _chunk_bwd_call(x, vb, kb, kc, dl, qg, sc, states, delta, d_out, interpret):
  bh, n, c = x.shape[:3]
  dk, dv = dl.shape[2], delta.shape[-1]
  sp = lambda *s: _spec((1, 1) + s, reverse=True, num_chunks=n)
  tok = lambda d: sp(c, d)
  shp = lambda t: jax.ShapeDtypeStruct(t.shape, t.dtype)
  return pl.pallas_call(
      _chunk_bwd_kernel,
      grid=(bh, n),
      in_specs=[sp(c, c), tok(dv), tok(dk), tok(dk), sp(dk, 1), tok(dk), sp(c, c), sp(dk, dv), sp(c, dv), tok(dv)],
      out_specs=[sp(c, c), tok(dv), tok(dk), tok(dk), sp(dk, 1), tok(dk), sp(c, c)],
      out_shape=[shp(x), shp(vb), shp(kb), shp(kc), shp(dl), shp(qg), shp(sc)],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=_PARAMS,
      interpret=interpret,
  )(x, vb, kb, kc, dl, qg, sc, states, delta, d_out)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def kda_chunk_scan(x, vb, kb, kc, dl, qg, scores, interpret=False):
  """KDA output per chunk from the WY inverse, the recurrence and the readout, in one pass over the chunks."""
  out, _, _ = _chunk_fwd_call(x, vb, kb, kc, dl, qg, scores, interpret)
  return out


def _kda_chunk_scan_fwd(x, vb, kb, kc, dl, qg, scores, interpret):
  out, states, delta = _chunk_fwd_call(x, vb, kb, kc, dl, qg, scores, interpret)
  return out, (x, vb, kb, kc, dl, qg, scores, states, delta)


def _kda_chunk_scan_bwd(interpret, res, d_out):
  x, vb, kb, kc, dl, qg, scores, states, delta = res
  return _chunk_bwd_call(x, vb, kb, kc, dl, qg, scores, states, delta, d_out, interpret)


kda_chunk_scan.defvjp(_kda_chunk_scan_fwd, _kda_chunk_scan_bwd)

