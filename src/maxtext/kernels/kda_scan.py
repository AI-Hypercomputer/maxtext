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
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fwd_kernel(uw_ref, kct_ref, dl_ref, states_ref, delta_ref, s_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  dv = s_ref.shape[1]
  s = s_ref[...]  # f32 holding bf16 values
  s_bf = s.astype(jnp.bfloat16)
  states_ref[0, 0] = s_bf
  uw = uw_ref[0, 0]
  delta = uw[:, :dv] - jnp.dot(uw[:, dv:].astype(jnp.bfloat16), s_bf, preferred_element_type=jnp.float32)
  delta_bf = delta.astype(jnp.bfloat16)
  delta_ref[0, 0] = delta_bf
  new = s * dl_ref[0, 0] + jnp.dot(kct_ref[0, 0], delta_bf, preferred_element_type=jnp.float32)
  s_ref[...] = new.astype(jnp.bfloat16).astype(jnp.float32)


def _bwd_kernel(gs_ref, gd_ref, kc_ref, wt_ref, dl_ref, r_out_ref, dd_ref, r_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  r = r_ref[...]
  r_out_ref[0, 0] = r
  dd = gd_ref[0, 0].astype(jnp.float32) + jnp.dot(
      kc_ref[0, 0], r.astype(jnp.bfloat16), preferred_element_type=jnp.float32
  )
  dd_ref[0, 0] = dd
  r_ref[...] = (
      gs_ref[0, 0].astype(jnp.float32)
      + r * dl_ref[0, 0]
      - jnp.dot(wt_ref[0, 0], dd.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
  )


def _spec(shape, reverse=False, num_chunks=None):
  if reverse:
    return pl.BlockSpec(shape, lambda bh, c: (bh, num_chunks - 1 - c, 0, 0))
  return pl.BlockSpec(shape, lambda bh, c: (bh, c, 0, 0))


def _scan_fwd_call(uw, kc, dl, interpret):
  bh, n, c, dk = kc.shape
  dv = uw.shape[-1] - dk
  kct = jnp.swapaxes(kc, -1, -2)
  return pl.pallas_call(
      _fwd_kernel,
      grid=(bh, n),
      in_specs=[_spec((1, 1, c, dv + dk)), _spec((1, 1, dk, c)), _spec((1, 1, dk, 1))],
      out_specs=[_spec((1, 1, dk, dv)), _spec((1, 1, c, dv))],
      out_shape=[
          jax.ShapeDtypeStruct((bh, n, dk, dv), jnp.bfloat16),
          jax.ShapeDtypeStruct((bh, n, c, dv), jnp.bfloat16),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      interpret=interpret,
  )(uw, kct, dl)


def _scan_bwd_call(g_states, g_delta, w, kc, dl, interpret):
  bh, n, c, dv = g_delta.shape
  dk = w.shape[-1]
  wt = jnp.swapaxes(w, -1, -2)
  rev = functools.partial(_spec, reverse=True, num_chunks=n)
  return pl.pallas_call(
      _bwd_kernel,
      grid=(bh, n),
      in_specs=[rev((1, 1, dk, dv)), rev((1, 1, c, dv)), rev((1, 1, c, dk)), rev((1, 1, dk, c)), rev((1, 1, dk, 1))],
      out_specs=[rev((1, 1, dk, dv)), rev((1, 1, c, dv))],
      out_shape=[
          jax.ShapeDtypeStruct((bh, n, dk, dv), jnp.float32),
          jax.ShapeDtypeStruct((bh, n, c, dv), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      interpret=interpret,
  )(g_states, g_delta, kc, wt, dl)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def kda_state_scan(uw, kc, dl, interpret=False):
  """Chunk states and deltas of the KDA recurrence.

  Args:
    uw: `[BH, N, C, dv + dk]` f32, the WY products u (values, first dv lanes) and w (multiplies the state).
      Taken whole, as the einsum produces it, so XLA need not slice and relayout u and w for the kernel.
    kc: `[BH, N, C, dk]` bf16, keys decayed to the chunk end (zero outside the chunk's last segment).
    dl: `[BH, N, dk, 1]` f32, per-channel decay over the chunk, 0 where the chunk holds a reset.
    interpret: run the kernels in interpret mode (CPU tests).

  Returns:
    states `[BH, N, dk, dv]` bf16 (the state entering each chunk) and delta `[BH, N, C, dv]` bf16.
  """
  states, delta = _scan_fwd_call(uw, kc, dl, interpret)
  return states, delta


def _kda_state_scan_fwd(uw, kc, dl, interpret):
  states, delta = _scan_fwd_call(uw, kc, dl, interpret)
  return (states, delta), (states, delta, uw, kc, dl)


def _kda_state_scan_bwd(interpret, res, g):
  states, delta, uw, kc, dl = res
  g_states, g_delta = g
  bf16 = jnp.bfloat16
  dv = delta.shape[-1]
  w = uw[..., dv:].astype(bf16)
  r, dd = _scan_bwd_call(g_states.astype(bf16), g_delta.astype(bf16), w, kc, dl, interpret)
  dw = -jnp.einsum("...cv,...dv->...cd", dd.astype(bf16), states, preferred_element_type=jnp.float32)
  dkc = jnp.einsum("...cv,...dv->...cd", delta, r.astype(bf16), preferred_element_type=jnp.float32)
  ddl = jnp.sum(states.astype(jnp.float32) * r, axis=-1, keepdims=True)
  return jnp.concatenate([dd, dw], axis=-1), dkc.astype(kc.dtype), ddl


kda_state_scan.defvjp(_kda_state_scan_fwd, _kda_state_scan_bwd)
