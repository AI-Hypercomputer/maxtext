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

"""The Gated DeltaNet (linear-attention) layer of Qwen3.5.

Mirrors `Qwen3NextGatedDeltaNet.__call__` (MaxText's models/qwen3.py,
training path), its chunked gated delta rule (`jax_chunk_gated_delta_rule`) and
its packed-document helpers (kernels/attention/gdn_packing.py), op for op and
dtype for dtype; the config's `gdn_packing` and `gdn_chunk_inverse` follow
MaxText's options `packing` and `gdn_chunk_inverse`.  The details that are
easy to get wrong: the qkvz / ba
projections are interleaved per KEY head; the conv runs over the flat
[q | k | v] channel order, causal, no bias, silu in fp32; key heads are
`jnp.repeat`ed (not tiled) to the value heads, inside the delta rule, after
the l2 norm, the q scale and the q k^T / k k^T products; the l2 norm adds its
eps inside the sqrt, in the input dtype; the gates `g` in fp32 (`beta` in
`dtype`); masks are applied with -1e30 BEFORE `exp`; q, k, v, beta enter in
bf16, so the k k^T and q k^T products come out of bf16 matmuls (rounded, then
cast to fp32), beta and the decay multiply k k^T in fp32, and the WY factors
u, w are rounded back to bf16; the decay, the masks, the inverse and the state
recurrence are fp32 at HIGHEST.

Segment ids (`gated_delta_net`): padding (segment 0) is zeroed before the
causal conv and its k, v and g after it, so a padded token neither decays nor
updates the state.  Under `gdn_packing` (MaxText's `packing`, its default)
every segment id is a document (canonical: an id that comes back after another
starts a new one): the conv sums only its own document's taps
(`segmented_causal_conv`, in fp32 with the fp32 kernel, one rounding) and
writes zeros on padding, and the delta rule restarts the state at each
document (`_segment_masks`: a per-document cumulative decay as a masked
matmul, and masks on the state's entry, exit and carry).  Without it the state
and the conv window run on across documents.

The one deliberate difference from MaxText is the backward of the WY inverse
`A = (I + S)^-1`: `unit_lower_inv` computes the same forward (block doubling,
or `solve_triangular`, by `gdn_chunk_inverse`), but carries a custom VJP (two
matmuls and a mask) instead of differentiating the forward.  JAX's own
transpose rule for the batched triangular solve is not explicit-sharding clean
(jax 0.10.2, 0.11 and 0.12), the block doubling's autodiff costs four matmuls
a doubling level, and the custom rule costs the same as the solve's on TPU
(v5p, the 397B-shape core: 8.5 / 16.8 ms fwd / fwd+bwd either way), so it is
used in EVERY mode: an auto-vs-explicit comparison then differs in the
sharding mode only.  Its gradient matches the solve's to ~4e-7 in fp32.

Under explicit sharding the layer traces as written once four sites are
typed, all in `_ops`: the inverse's identity (E1), the state scan's carry
(E2), the per-key-head -> per-value-head regroups (E3), and the elementwise
weights (A_log, dt_bias, the conv kernel: replicated); the packed path adds
two contractions (`_ops.segment_cumsum`, `_ops.conv_taps`).
`config.gdn_core='manual'` runs the chunked rule per device instead, in a
`shard_map` over the batch, in both modes (`_ops.gdn_core`).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import layers

_HI = jax.lax.Precision.HIGHEST
_NEG = -1e30  # mask value applied BEFORE exp, so masked entries are exp(-inf) = 0


# ------------------------------------------------------------- primitives
def l2norm(x, eps=1e-6):
  """`normalizations.l2norm`: `x * rsqrt(sum(x^2) + eps)`, in `x.dtype`."""
  return x * jax.lax.rsqrt((x * x).sum(axis=-1, keepdims=True) + jnp.array(eps, dtype=x.dtype))


def causal_depthwise_conv(x, kernel, config):
  """`nnx.Conv(feature_group_count=C, padding='CAUSAL', use_bias=False)` as MaxText
  calls it (`qwen3.Qwen3NextGatedDeltaNet`'s `conv1d`), in `config.dtype`; kernel `[K, 1, C]`.

  Output t is `sum_{i<K} w[i] * x[t - (K-1) + i]` with x[<0] = 0.  MaxText pads
  K-1 zeros itself AND asks for 'CAUSAL' padding, then keeps the last S outputs;
  a single (K-1, 0) pad is the same conv, bitwise, in bf16 and fp32
  (`qwen35_model_test` pins it), without the extra `[B, S+K-1, C]` copy.
  """
  w = _ops.small_weight(kernel, config.dtype, config)
  x = jnp.asarray(x, config.dtype)
  k = w.shape[0]
  if config.gdn_conv_impl == "shift":
    return causal_conv_shift(x, w)
  return _ops.conv_general_dilated(
      x,
      w,
      window_strides=(1,),
      padding=((k - 1, 0),),
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=x.shape[-1],
      precision=layers.precision_of(config),
  )


def causal_conv_shift(x, w):
  """`causal_depthwise_conv`'s conv as its K shifted multiply-adds -- output t is `sum_{i<K} w[i] * x[t-(K-1)+i]`,
  x[<0] = 0 -- summed in fp32 in i's order and rounded once to x's dtype (config.gdn_conv_impl 'shift').  x [B, S, C],
  w [K, 1, C].  XLA:GPU fuses it (and its gradients: shifted sums and K reductions) into memory-bound loops: 0.35 ms
  forward, 0.81 ms forward + backward on a GB200 for a Qwen3.8 microbatch's [2, 4096, 20480] (~0.34 GB each way),
  where cuDNN's grouped conv took 5.1 / 9.8 ms (2026-11).  The same sum as the conv's to fp32 rounding, not
  bitwise."""
  k, s = w.shape[0], x.shape[1]
  xp = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0))).astype(jnp.float32)
  wf = w[:, 0, :].astype(jnp.float32)
  y = xp[:, :s, :] * wf[0]
  for i in range(1, k):
    y = y + xp[:, i : i + s, :] * wf[i]
  return y.astype(x.dtype)


# ------------------------------------------------------ packed documents
def canonical_segment_ids(segment_ids):
  """MaxText's gdn_packing.canonicalize_segment_ids: `[B, S]` raw ids -> int32 documents 1, 2, ... a row, 0 on padding
  (ids <= 0).  A document starts wherever a valid token's id differs from the last valid token's before it, so an id
  that comes back after another ([1, 2, 1]) is a new document ([1, 2, 3]), and padding inside one ([1, 0, 1]) does not
  split it."""
  seg = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  active = jax.lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg, axis=-1)
  prev_active = jnp.concatenate([active[..., :1] * 0, active[..., :-1]], axis=-1)
  valid = seg > 0
  doc = jnp.cumsum((valid & (seg != prev_active)).astype(jnp.int32), axis=-1)
  return jnp.where(valid, doc, 0).astype(jnp.int32)


def segmented_causal_conv(x, kernel, segment_ids, config):
  """`causal_depthwise_conv` within documents (MaxText's gdn_packing.segmented_causal_depthwise_conv1d, its
  single-device path): output t is `sum_{i<K} w[i] * x[t-(K-1)+i]` over the taps in t's own document
  (`canonical_segment_ids`), zero on padding.  x `[B, S, C]`, kernel `[K, 1, C]`, segment ids `[B, S]`.

  In fp32, with the fp32 kernel (not rounded to `dtype` as the plain conv's is), at HIGHEST, rounded once to `dtype`:
  the conv of the input with padding and the K-1 leading positions zeroed, and for the rows whose window reaches
  another document (`conv_taps`) the sum over their own taps instead -- MaxText's two expressions, so the bits are its.
  `gdn_conv_impl` 'shift': each row the sum over its own taps as K shifted multiply-adds in fp32, the same sum to fp32
  rounding."""
  w = _ops.small_weight(kernel, jnp.float32, config)
  k = w.shape[0]
  _, s, c = x.shape
  seg = canonical_segment_ids(segment_ids)
  halo = k - 1
  xp = jnp.pad(x, ((0, 0), (halo, 0), (0, 0)))
  segp = jnp.pad(seg, ((0, 0), (halo, 0)))
  xm = jnp.where((segp > 0)[..., None], xp, 0)
  tap_seg = [segp[:, i : i + s] for i in range(k)]  # the segment of each row's tap i
  if config.gdn_conv_impl == "shift":
    out = jnp.where((tap_seg[0] == seg)[..., None], xm[:, :s], 0).astype(jnp.float32) * w[0, 0]
    for i in range(1, k):
      out = out + jnp.where((tap_seg[i] == seg)[..., None], xm[:, i : i + s], 0).astype(jnp.float32) * w[i, 0]
  else:
    out = _ops.conv_general_dilated(
        xm.astype(jnp.float32),
        w,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NHC", "HIO", "NHC"),
        feature_group_count=c,
        precision=_HI,
    )
    if halo > 0:
      cross = None  # rows with a tap in another (valid) document
      for i in range(halo):
        hit = (seg > 0) & (tap_seg[i] > 0) & (tap_seg[i] != seg)
        cross = hit if cross is None else cross | hit
      taps = jnp.stack(
          [jnp.where((cross & (tap_seg[i] == seg))[..., None], xm[:, i : i + s], 0) for i in range(k)], axis=2
      )
      out = jnp.where(cross[..., None], _ops.conv_taps(taps, w[:, 0, :]), out)
  return jnp.where((seg > 0)[..., None], out, 0.0).astype(config.dtype)


def _segment_masks(segment_ids, n, c):
  """MaxText's gdn_packing.prepare_jax_gdn_segment_masks, its single-device path with no initial state, on raw
  segment ids `[B, S]` and the delta rule's N chunks of C: the per-chunk masks `[B, N, 1, ...]` of the packed rule.

  -> (valid [B,N,1,C] bool: a document's token, not padding (the S..N*C tail is padding of the row's last document);
      cum [B,N,1,C,C] fp32: lower triangle within one document, the per-document cumulative decay's matmul;
      strict, causal [B,N,1,C,C] bool: the strictly lower / lower triangle between valid tokens of one document;
      m_in [B,N,1,C] fp32: 1 where a token continues the document the chunk's incoming state belongs to;
      m_out [B,N,1,C] fp32: 1 where a token belongs to the document at the chunk's end, the one its state carries on;
      m_keep [B,N,1,1] fp32: 1 where that document is the incoming state's, so the state is kept, decayed)."""
  b, s = segment_ids.shape
  seg = jnp.maximum(canonical_segment_ids(segment_ids), 0)
  # signed: +d on document d's tokens, -d on padding after document d (0 before the first)
  active = jax.lax.associative_scan(lambda x, y: jnp.where(y > 0, y, x), seg, axis=-1)
  enc = jnp.where(seg > 0, seg, -active).astype(jnp.float32)
  if n * c > s:
    enc = jnp.concatenate([enc, jnp.repeat(-jnp.abs(enc[:, -1:]), n * c - s, axis=1)], axis=1)
  seg_c = enc.reshape(b, n, 1, c)
  valid = seg_c > 0.5
  act = jnp.abs(seg_c)
  act_end = act[..., -1]  # [B, N, 1]
  seg_prev = jnp.concatenate([act_end[:, :1] * 0.0, act_end[:, :-1, :]], axis=1)[..., None]  # [B, N, 1, 1]
  # The triangles traced, as MaxText writes them: numpy constants change how XLA fuses what they mask, and the fp32
  # results by a rounding (the delta rule's, from the strict triangle on).
  same_act = (jnp.abs(act[..., :, None] - act[..., None, :]) < 0.5) & (act[..., :, None] > 0.5)
  cum = jnp.tril(jnp.ones((c, c), dtype=jnp.float32)) * same_act.astype(jnp.float32)
  same_valid = (jnp.abs(seg_c[..., :, None] - seg_c[..., None, :]) < 0.5) & valid[..., :, None]
  strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1) & same_valid
  causal = jnp.tril(jnp.ones((c, c), dtype=bool), k=0) & same_valid
  act_last = act[..., -1:]
  m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
  m_out = ((jnp.abs(seg_c - act_last) < 0.5) & (act_last > 0.5)).astype(jnp.float32)
  m_keep = ((jnp.abs(act_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
  return valid, cum, strict, causal, m_in, m_out, m_keep


# ------------------------------------------------------------ the inverse
def _block_doubling(s):
  """MaxText's gdn_packing.block_doubling_unit_lower_inverse: `(I + s)^-1` for strictly-lower-triangular `s
  [..., C, C]` (fp32), bottom up -- from 1 x 1 diagonal blocks (inverse 1), each level merges two neighbouring b x b
  blocks with [[A11, 0], [A21, A22]]^-1 = [[A11^-1, 0], [-A22^-1 A21 A11^-1, A22^-1]], two matmuls at HIGHEST
  (ceil(log2 C) levels, any C)."""
  c = s.shape[-1]
  ar = jnp.arange(c)
  inv = _ops.broadcast_like(jnp.eye(c, dtype=jnp.float32), s)
  blk = 1
  while blk < c:
    # the lower-left blk x blk block of each 2 blk x 2 blk diagonal block: what inv has not folded in yet
    cross = ((ar[:, None] // (2 * blk)) == (ar[None, :] // (2 * blk))) & (
        (ar[:, None] // blk) == (ar[None, :] // blk) + 1
    )
    off = jnp.matmul(jnp.matmul(inv, jnp.where(cross, s, 0.0), precision=_HI), inv, precision=_HI)
    inv = inv - jnp.where(cross, off, 0.0)
    blk *= 2
  return inv


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def unit_lower_inv(s, method="block_doubling"):
  """`(I + s)^-1` for strictly-lower-triangular `s [..., C, C]`, as MaxText computes it under `gdn_chunk_inverse`
  `method`: 'block_doubling' (`_block_doubling`; the default, as gdn_chunk_inverse's and MaxText's), or 'solve'
  (MaxText's reference path, and what the Triton kernels' triangular solve stands for in their tests),
  `solve_triangular(I + S, I, lower=True, unit_diagonal=True)` -- with the backward written out either way:
  `dS = -tril(A^T dA A^T, -1)`."""
  if method == "block_doubling":
    return _block_doubling(s)
  if method != "solve":
    raise ValueError(f"unit_lower_inv: method must be 'block_doubling' or 'solve', got {method!r}")
  eye = jnp.eye(s.shape[-1], dtype=s.dtype)
  return jax.scipy.linalg.solve_triangular(eye + s, _ops.broadcast_like(eye, s), lower=True, unit_diagonal=True)


def _unit_lower_inv_fwd(s, method):
  a = unit_lower_inv(s, method)
  return a, a


def _unit_lower_inv_bwd(method, a, g):
  del method
  at = a.swapaxes(-1, -2)
  ds = -jnp.matmul(jnp.matmul(at, g, precision=_HI), at, precision=_HI)
  c = a.shape[-1]
  ds = jnp.where(jnp.tril(jnp.ones((c, c), dtype=bool), k=-1), ds, 0.0)
  return (_ops.retype_like(ds, a),)


unit_lower_inv.defvjp(_unit_lower_inv_fwd, _unit_lower_inv_bwd)


# ------------------------------------------------------- the delta rule
def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size,
    use_qk_norm,
    compute_dtype,
    scan_impl="scan",
    *,
    segment_ids=None,
    chunk_inverse="block_doubling",
):
  """`jax_chunk_gated_delta_rule` (models/qwen3.py), same op order and dtypes.

  query/key `[B, S, Hk, Dk]`, value `[B, S, H, Dv]`, g / beta `[B, S, H]`, value
  head j reading key head j // (H / Hk).  Returns `o [B, S, H, Dv]` in query's
  dtype.  The cumulative decay, the masks, the inverse and the state
  recurrence are fp32 at HIGHEST; q/k/v/beta are rounded to `compute_dtype`
  first and the WY factors u, w are rounded back to it.  S is padded up to a
  multiple of `chunk_size` with zeros (g = 0 there: no decay, no update).
  The initial state is zero, as in training.

  `segment_ids` `[B, S]` (packed documents, `Qwen35Config.gdn_packing`): the
  state restarts at every document (`_segment_masks`), padding neither reads
  nor writes it, and its output is zero.  None: one sequence a row.

  `chunk_inverse` (`gdn_chunk_inverse`): how `unit_lower_inv` inverts the WY
  matrix, which is built from k k^T at the key heads times fp32 beta and decay,
  the key heads repeated in here, as MaxText's GDN builds it.

  `scan_impl` (config.gdn_scan_impl): 'scan', the inter-chunk recurrence as a
  `lax.scan` (MaxText's); 'triton', Triton kernels (`gdn_triton`): with a
  chunk of 16 / 32 / 64 the whole rule (`gdn_triton.chunk`, FUSED), else the
  recurrence as one kernel each way and the chunk outputs batched over all
  chunks in XLA: GPU only, per device (the MPMD pipeline's stage programs,
  gdn_core='manual', or no mesh at all), the same math to fp32 rounding, and
  with no packed documents.
  """
  if scan_impl == "triton" and segment_ids is not None:
    raise ValueError(
        "gdn_scan_impl 'triton' has no packed documents: call the GDN without segment ids (as the MPMD pipeline "
        "does) or set gdn_packing=False"
    )
  initial_dtype = query.dtype
  if use_qk_norm:
    query = l2norm(query)
    key = l2norm(key)
  g = g.astype(jnp.float32)
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  query = query * scale

  b, s, hk, dk = key.shape
  h, dv = value.shape[2], value.shape[-1]
  if h % hk:
    raise ValueError(f"value heads {h} vs key heads {hk}: the key heads, repeated here, must divide the value heads")
  rep = h // hk
  c = chunk_size
  pad = (c - s % c) % c
  if pad:

    def pad_fn(x):
      return jnp.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2))

    query, key, value, g, beta = map(pad_fn, (query, key, value, g, beta))
  n = query.shape[1] // c

  def to_chunk(x):  # [B, S, Hx, D] -> [B, N, Hx, C, D]
    return x.reshape(b, n, c, x.shape[2], -1).transpose(0, 1, 3, 2, 4)

  def to_chunk_s(x):  # [B, S, H] -> [B, N, H, C]
    return x.reshape(b, n, c, h).transpose(0, 1, 3, 2)

  def heads(x, axis):  # key heads -> value heads (value head j <- key head j // rep)
    return jnp.repeat(x, rep, axis=axis) if rep > 1 else x

  q_c, k_c, v_c = to_chunk(query), to_chunk(key), to_chunk(value)
  g_c, beta_c = to_chunk_s(g), to_chunk_s(beta)
  # The op order is MaxText's: any order gives the same values, but an fp32 program's last bits can depend on how XLA
  # fuses it.
  if segment_ids is None:
    g_cum = jnp.cumsum(g_c, axis=-1)  # fp32
    strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
  else:
    valid, cum, strict, causal, m_in, m_out, m_keep = _segment_masks(segment_ids, n, c)
    q_c = jnp.where(valid[..., None], q_c, 0.0)
    k_c = jnp.where(valid[..., None], k_c, 0.0)
    v_c = jnp.where(valid[..., None], v_c, 0.0)
    g_c = jnp.where(valid, g_c, 0.0)
    beta_c = jnp.where(valid, beta_c, 0.0)
    g_cum = _ops.segment_cumsum(cum, g_c)  # fp32, restarting at each document
  if scan_impl == "triton":
    _triton_context()
    from maxtext.experimental.mpmd.jax.qwen35 import gdn_triton

    if gdn_triton.FUSED and gdn_triton.fused_supported(c, dk, dv):  # the whole rule on kernels
      o = gdn_triton.chunk(heads(q_c, 2), heads(k_c, 2), v_c, beta_c, g_cum, out_dtype=initial_dtype)
      return o.transpose(0, 1, 3, 2, 4).reshape(b, -1, h, dv)[:, :s]

  # Intra-chunk WY factors (all chunks in parallel): the WY matrix from k k^T at the key heads (rounded to
  # `compute_dtype` as it comes out of the matmul), repeated to the value heads, times fp32 beta and decay.
  kk = heads(jnp.matmul(k_c, k_c.swapaxes(-1, -2), precision=_HI).astype(jnp.float32), 2)
  g_diff = jnp.where(strict, g_cum[..., :, None] - g_cum[..., None, :], _NEG)
  smat = jnp.where(strict, kk * (beta_c.astype(jnp.float32)[..., :, None] * jnp.exp(g_diff)), 0.0)
  a = unit_lower_inv(smat, chunk_inverse)
  v_beta = v_c * beta_c[..., None]
  u = jnp.matmul(a, v_beta.astype(jnp.float32), precision=_HI).astype(compute_dtype)
  k_beta = heads(k_c, 2) * beta_c[..., None]
  decay_in = jnp.exp(g_cum) if segment_ids is None else jnp.exp(g_cum) * m_in
  k_beta_g = k_beta.astype(jnp.float32) * decay_in[..., None]
  w = jnp.matmul(a, k_beta_g, precision=_HI).astype(compute_dtype)

  if scan_impl == "triton":  # the recurrence on kernels
    from maxtext.experimental.mpmd.jax.qwen35 import gdn_triton

    o = gdn_triton.inter_chunk(heads(q_c, 2), heads(k_c, 2), u, w, g_cum)  # [B, N, H, C, Dv] fp32
    return o.transpose(0, 1, 3, 2, 4).reshape(b, -1, h, dv)[:, :s].astype(initial_dtype)
  if scan_impl != "scan":
    raise ValueError(f"scan_impl must be 'scan' or 'triton', got {scan_impl!r}")

  # Inter-chunk recurrence over the state H [B, H, Dk, Dv].
  perm_v, perm_s = (1, 0, 2, 3, 4), (1, 0, 2, 3)
  xs = (w.transpose(perm_v), u.transpose(perm_v), q_c.transpose(perm_v), k_c.transpose(perm_v), g_cum.transpose(perm_s))
  if segment_ids is not None:  # each chunk's masks
    xs += tuple(x.transpose(perm_v if x.ndim == 5 else perm_s) for x in (m_in, m_out, m_keep, causal, valid))

  def step(state, xs_i):
    w_i, u_i, q_i, k_i, g_i = xs_i[:5]
    if segment_ids is None:
      causal_i = jnp.tril(jnp.ones((c, c), dtype=bool))
    else:
      m_in_i, m_out_i, m_keep_i, causal_i, valid_i = xs_i[5:]
    # q k^T at the key heads, then the value heads
    attn = heads(jnp.matmul(q_i, k_i.swapaxes(-1, -2), precision=_HI).astype(jnp.float32), 1)
    q_i, k_i = heads(q_i, 1), heads(k_i, 1)
    decay_q = jnp.exp(g_i) if segment_ids is None else jnp.exp(g_i) * m_in_i
    q_g = q_i.astype(jnp.float32) * decay_q[..., None]
    attn_inter = jnp.matmul(q_g, state, precision=_HI)
    v_new = u_i.astype(jnp.float32) - jnp.matmul(w_i.astype(jnp.float32), state, precision=_HI)
    gd = jnp.where(causal_i, g_i[..., :, None] - g_i[..., None, :], _NEG)
    attn = jnp.where(causal_i, attn * jnp.exp(gd), 0.0)
    o = attn_inter + jnp.matmul(attn, v_new, precision=_HI)
    decay_state = jnp.exp(g_i[..., -1, None, None])
    if segment_ids is None:
      decay_k = jnp.exp(g_i[..., -1, None] - g_i)
    else:
      o = jnp.where(valid_i[..., None], o, 0.0)
      decay_state = decay_state * m_keep_i[..., None]
      decay_k = jnp.exp(jnp.where(m_out_i > 0.5, g_i[..., -1, None] - g_i, _NEG)) * m_out_i
    new_state = state * decay_state
    k_dec = k_i.astype(jnp.float32) * decay_k[..., None]
    new_state = new_state + jnp.matmul(k_dec.swapaxes(-1, -2), v_new, precision=_HI)
    return new_state, o

  state0 = _ops.zeros((b, h, dk, dv), jnp.float32, batch_dim=0)
  _, o = jax.lax.scan(step, state0, xs)
  o = o.transpose(1, 0, 3, 2, 4).reshape(b, -1, h, dv)[:, :s]
  return o.astype(initial_dtype)


def _triton_context():
  """gdn_scan_impl 'triton' runs its kernels per device on a GPU: refuse anywhere else, before a custom call
  reaches a partitioner that would gather its operands (auto) or a backend without Triton."""
  from maxtext.experimental.mpmd.jax import sharding_mode as mode

  if jax.default_backend() != "gpu":
    raise ValueError(
        f"gdn_scan_impl 'triton' runs Triton kernels: it needs a GPU backend, not "
        f"{jax.default_backend()!r} (gdn_scan_impl 'scan' runs anywhere)"
    )
  if mode.context_mode() not in ("none", "manual"):
    raise ValueError(
        f"gdn_scan_impl 'triton' runs per device: under the {mode.context_mode()!r} sharding mode set "
        "gdn_core='manual' (the rule mapped over the batch's devices), as the MPMD pipeline's stage "
        "programs run it"
    )


def recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_norm=True):
  """The definitional per-token recurrence, in numpy float64 (a test oracle).

  HF's `torch_recurrent_gated_delta_rule`: with q, k l2-normalised and q scaled
  by `Dk^-1/2`,  `S <- S exp(g_t);  d = beta_t (v_t - S^T k_t);  S <- S + k_t d^T;
  o_t = S^T q_t`.  The decay applies before the delta update.
  """
  q, k, v, g, beta = (np.asarray(t, np.float64) for t in (query, key, value, g, beta))
  if use_qk_norm:
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
  q = q / np.sqrt(q.shape[-1])
  b, s, h, dk = k.shape
  state = np.zeros((b, h, dk, v.shape[-1]))
  out = np.zeros(v.shape)
  for t in range(s):
    state = state * np.exp(g[:, t])[:, :, None, None]
    delta = (v[:, t] - np.einsum("bhkv,bhk->bhv", state, k[:, t])) * beta[:, t][..., None]
    state = state + np.einsum("bhk,bhv->bhkv", k[:, t], delta)
    out[:, t] = np.einsum("bhkv,bhk->bhv", state, q[:, t])
  return out


# ------------------------------------------------------------------ layer
def gated_rms_norm(params, x, gate, config):
  """`Qwen3NextRMSNormGated`: plain RMSNorm over the last axis (ones init, no
  offset), then `* silu(fp32(gate))`, rounded to `dtype`.

  `norm_numerics='fp32'`: the scaled value stays fp32 until that one rounding
  (MaxText since 2026-09-10, whose inner RMSNorm has `dtype=float32`); 'bf16':
  the normalised value is rounded to `dtype` and scaled there first (MaxText
  before then; HF with bf16 weights).
  """
  if config.norm_numerics == "fp32":
    y = _ops.norm_fp32(params["rms_norm"], x, config)
  else:
    y = layers.rms_norm(params["rms_norm"], x, config, zero_centered=False)
  return (y * jax.nn.silu(gate.astype(jnp.float32))).astype(config.dtype)


def gated_delta_net(params, x, segment_ids, config):
  """`Qwen3NextGatedDeltaNet.__call__`, training path.  `x [B, S, E]` -> `[B, S, E]`.

  `segment_ids` `[B, S]` (None: one sequence a row, nothing masked).  Padding
  (segment 0) is zeroed before the causal conv and its k, v and g after it, so
  a padded token neither decays nor updates the state.  Under `gdn_packing`
  (MaxText's `packing`) every segment id is a document: the conv sums only the
  document's own taps (`segmented_causal_conv`) and the delta rule restarts
  the state at each one (`chunk_gated_delta_rule`'s `segment_ids`).  Without it
  the state and the conv window run on across documents (MaxText's
  packing=false), while attention masks across them.
  """
  b, s, _ = x.shape
  hk, hv = config.gdn_num_key_heads, config.gdn_num_value_heads
  dk, dv = config.gdn_key_head_dim, config.gdn_value_head_dim
  vpk = config.gdn_v_per_k
  qkvz = _ops.dense(params["in_proj_qkvz"], x, config)  # [B, S, 2Hk*Dk + 2Hv*Dv]
  ba = _ops.dense(params["in_proj_ba"], x, config)  # [B, S, 2Hv]
  # Per-KEY-HEAD interleaved columns: [q_h | k_h | v_{h,0..vpk-1} | z_{h,0..vpk-1}],
  # and [b_{h,.} | a_{h,.}] -- value head j belongs to key head j // vpk.
  mixed = qkvz.reshape(b, s, hk, 2 * dk + 2 * vpk * dv)
  q, k, v_raw, z_raw = jnp.split(mixed, [dk, 2 * dk, 2 * dk + vpk * dv], axis=3)
  v = _ops.regroup(v_raw, (b, s, hv, dv))
  z = _ops.regroup(z_raw, (b, s, hv, dv))
  b_raw, a_raw = jnp.split(ba.reshape(b, s, hk, 2 * vpk), [vpk], axis=3)
  b_in = _ops.regroup(b_raw, (b, s, hv))
  a_in = _ops.regroup(a_raw, (b, s, hv))

  # The conv runs over the flat channel order [q_all | k_all | v_all].
  qkv = jnp.concatenate([q.reshape(b, s, -1), k.reshape(b, s, -1), v.reshape(b, s, -1)], axis=-1)
  packed = segment_ids is not None and config.gdn_packing
  if segment_ids is not None:
    qkv = jnp.where((segment_ids != 0)[..., None], qkv, 0.0)
  if packed:
    conv = segmented_causal_conv(qkv, params["conv1d"]["kernel"], segment_ids, config)
  else:
    conv = causal_depthwise_conv(qkv, params["conv1d"]["kernel"], config)
  qkv = jax.nn.silu(conv.astype(jnp.float32)).astype(config.dtype)
  qc, kc, vc = jnp.split(qkv, [config.gdn_key_dim, 2 * config.gdn_key_dim], axis=-1)
  query = qc.reshape(b, s, hk, dk)
  key = kc.reshape(b, s, hk, dk)
  value = vc.reshape(b, s, hv, dv)

  # Gates: g in fp32 (HF's bf16 checkpoints: in `dtype`), beta in `dtype`.
  gate_dtype = jnp.float32
  a_log = _ops.small_weight(params["A_log"], gate_dtype, config)
  dt_bias = _ops.small_weight(params["dt_bias"], gate_dtype, config)
  beta = jax.nn.sigmoid(b_in)
  g = -jnp.exp(a_log) * jax.nn.softplus(a_in.astype(gate_dtype) + dt_bias)
  if segment_ids is not None:
    keep = segment_ids != 0
    key = jnp.where(keep[..., None, None], key, 0.0)
    value = jnp.where(keep[..., None, None], value, 0.0)
    g = jnp.where(keep[..., None], g, 0.0)
  rule = functools.partial(
      chunk_gated_delta_rule,
      chunk_size=config.gdn_chunk_size,
      use_qk_norm=config.use_qk_norm_in_gdn,
      compute_dtype=config.dtype,
      scan_impl=config.gdn_scan_impl,
      chunk_inverse=config.gdn_chunk_inverse,
  )
  core = _ops.gdn_core(rule, query, key, value, g, beta, config, segment_ids=segment_ids if packed else None)
  y = gated_rms_norm(params["norm"], core, z, config).reshape(b, s, -1)
  return _ops.dense(params["out_proj"], y, config)
