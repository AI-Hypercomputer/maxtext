# Copyright 2023–2026 Google LLC
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

"""Synthetic MoE routing for performance experiments.

Generates `forced_routed_experts` batches ([batch, seq, num_layers, top_k]
int32) that mimic a steady-state router, so MoE load-balancing experiments
run on a fixed, reproducible expert distribution independent of data and
weights. Fitted to a real Qwen3.5-35B-A3B C4 routing dump (see
routing_sim/REPORT.md).

Modes (config.synthetic_routing):
  * "lognormal": per layer l, a persistent expert log-popularity w_l (fixed
    across steps):
      - default: w_l ~ N(0, sigma_l^2) iid over experts
        (synthetic_routing_sigma / synthetic_routing_sigmas);
      - if synthetic_routing_profile is set: w_l = a random permutation of the
        per-layer empirical log-popularity profile stored in that .npz
        (key "log_profile", [P, E]; captures the real heavy tail).
    Rows are packed with "documents" of exponential length (mean
    synthetic_routing_row_len tokens, random boundaries; <= 0 means one
    document per row). Each document d gets its own perturbation
    u_d ~ N(0, tau_l^2) per expert (synthetic_routing_row_tau /
    synthetic_routing_row_taus), fresh every pool step: this reproduces the
    per-sequence clustering and the step-to-step decorrelation of real
    routing. Each token draws top_k distinct experts from softmax(w_l + u_d)
    without replacement (Plackett-Luce, same law as Gumbel-top-k).
  * "balanced": every expert, and therefore every expert-parallel shard,
    receives exactly the mean number of (token, expert) assignments.

Per-layer lists (sigmas, row_taus, profile rows) whose length differs from
num_layers (e.g. a 60-layer fit used for a 4- or 8-layer run) are mapped by
relative depth (linear interpolation; profiles blend sorted quantiles).

Config fields (getattr with defaults, so older configs keep working):
  synthetic_routing            "lognormal" | "balanced"
  synthetic_routing_sigma      float, used if synthetic_routing_sigmas empty
  synthetic_routing_sigmas     comma list (1 or any number of layers)
  synthetic_routing_pool_size  int, distinct batches cycled through
  synthetic_routing_seed       int
  synthetic_routing_row_tau    float (NEW, default 0.0 = no per-doc noise)
  synthetic_routing_row_taus   comma list (NEW, optional; overrides row_tau)
  synthetic_routing_row_len    int mean document length (NEW, default 0 = row)
  synthetic_routing_profile    path to .npz with "log_profile" (NEW, optional)

A small pool of batches is generated once on the host (vectorized numpy,
layers in a thread pool) and cycled through.
"""

from concurrent import futures
import os

import numpy as np

_TAG_W, _TAG_SEG, _TAG_U, _TAG_TOK = 11, 12, 13, 14


def _parse_list(val):
  if val is None:
    return []
  if isinstance(val, (list, tuple, np.ndarray)):
    return [float(x) for x in val]
  s = str(val).strip()
  if not s:
    return []
  return [float(x) for x in s.split(",") if x.strip()]


def depth_map(vals, num_layers):
  """Map a per-layer list onto num_layers by relative depth (linear)."""
  vals = np.asarray(vals, dtype=np.float64)
  if vals.size == 1:
    return np.full(num_layers, float(vals[0]))
  if vals.size == num_layers:
    return vals
  return np.interp(np.linspace(0, 1, num_layers), np.linspace(0, 1, vals.size),
                   vals)


def depth_map_profiles(profiles, num_layers, num_experts):
  """[P, E'] log-profiles -> [num_layers, num_experts] sorted descending."""
  p = -np.sort(-np.asarray(profiles, dtype=np.float64), axis=1)
  if p.shape[1] != num_experts:
    q_in = (np.arange(p.shape[1]) + 0.5) / p.shape[1]
    q_out = (np.arange(num_experts) + 0.5) / num_experts
    p = np.stack([np.interp(q_out, q_in, r) for r in p])
  if p.shape[0] != num_layers:
    x = np.linspace(0, p.shape[0] - 1, num_layers)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, p.shape[0] - 1)
    f = (x - lo)[:, None]
    p = (1 - f) * p[lo] + f * p[hi]
  return p - p.mean(1, keepdims=True)


def layer_log_popularity(num_layers, num_experts, seed, sigmas=None,
                         profile=None):
  """Persistent per-layer log-popularity w, float64 [num_layers, E]."""
  w = np.empty((num_layers, num_experts))
  prof = (None if profile is None else
          depth_map_profiles(profile, num_layers, num_experts))
  sig = None if prof is not None else depth_map(sigmas, num_layers)
  for l in range(num_layers):
    rng = np.random.default_rng([seed, _TAG_W, l])
    if prof is not None:
      w[l] = prof[l][rng.permutation(num_experts)]
    else:
      w[l] = sig[l] * rng.standard_normal(num_experts)
  return w


def document_ids(batch, seq, row_len, rng):
  """Global document id of each token [batch, seq] and number of documents.

  Documents have exponential lengths with mean row_len and a random phase;
  row_len <= 0 or >= seq*8 means one document per row.
  """
  if row_len <= 0 or row_len >= 8 * seq:
    return np.repeat(np.arange(batch)[:, None], seq, 1), batch
  nmax = int(np.ceil(seq / row_len * 3)) + 8
  lens = rng.standard_exponential((batch, nmax)) * row_len
  ends = np.cumsum(lens, 1) - rng.random((batch, 1)) * lens[:, :1]
  ends = np.minimum(ends, seq)
  ends[:, -1] = seq
  off = (np.arange(batch) * seq)[:, None]
  pos = off + np.arange(seq)[None, :]
  ids = np.searchsorted((ends + off).ravel(), pos.ravel(), side="right")
  ids = np.minimum(ids, (np.arange(batch) * nmax + nmax - 1).repeat(seq))
  return ids.reshape(batch, seq), batch * nmax


def sample_topk(logits, doc, top_k, rng, draws=None):
  """Plackett-Luce top_k (= Gumbel-top-k law) for tokens sharing doc logits.

  logits [D, E] per document, doc [N] document id of each token. Draws
  experts with replacement by inverse CDF and keeps the first top_k distinct
  ones (exactly sequential sampling without replacement); rows that did not
  reach top_k distinct experts are redrawn with more draws, then by exact
  Gumbel-top-k. Returns int32 [N, top_k] in selection order.
  """
  n = doc.shape[0]
  num_experts = logits.shape[1]
  lg = logits - logits.max(1, keepdims=True)
  p = np.exp(lg)
  p /= p.sum(1, keepdims=True)
  cdf = np.cumsum(p, 1)
  cdf[:, -1] = 1.0
  flat = (cdf + np.arange(cdf.shape[0])[:, None]).ravel()
  out = np.empty((n, top_k), dtype=np.int32)
  todo = np.arange(n)
  idx = np.empty((0, 0), dtype=np.int64)
  for m in (draws or top_k + 6, 2 * top_k + 2, 6 * top_k):
    if todo.size == 0:
      break
    d = doc[todo]
    u = rng.random((todo.size, m)) + d[:, None]
    new = np.searchsorted(flat, u.ravel(), side="right").reshape(todo.size, m)
    new = np.minimum(new - d[:, None] * num_experts, num_experts - 1)
    # Unfinished rows keep their earlier draws (extending the same sequence
    # keeps the law exact; redrawing them from scratch would bias it).
    idx = new if idx.size == 0 else np.concatenate([idx, new], 1)
    mm = idx.shape[1]
    first = np.ones(idx.shape, dtype=bool)
    for j in range(1, mm):
      first[:, j] = ~(idx[:, :j] == idx[:, j:j + 1]).any(1)
    ok = first.sum(1) >= top_k
    sel = first & (np.cumsum(first, 1) <= top_k)
    rows = np.flatnonzero(ok)
    out[todo[rows]] = idx[rows][sel[rows]].reshape(-1, top_k)
    todo, idx = todo[~ok], idx[~ok]
  if todo.size:  # finish exactly: masked Gumbel-top-k after the chosen prefix
    first = np.ones(idx.shape, dtype=bool)
    for j in range(1, idx.shape[1]):
      first[:, j] = ~(idx[:, :j] == idx[:, j:j + 1]).any(1)
    cnt = first.sum(1)
    t = todo.size
    rows = np.arange(t)
    chosen = np.zeros((t, top_k), dtype=np.int64)
    rr, cc = np.nonzero(first)
    chosen[rr, (np.cumsum(first, 1) - 1)[rr, cc]] = idx[rr, cc]
    g = lg[doc[todo]] - np.log(rng.standard_exponential(
        (t, num_experts)) + 1e-30)
    mask = np.arange(top_k)[None, :] < cnt[:, None]
    g[rows[:, None].repeat(top_k, 1)[mask], chosen[mask]] = -np.inf
    gtop = np.argsort(-g, 1)[:, :top_k]
    pos = np.arange(top_k)[None, :]
    fill = np.take_along_axis(gtop, np.maximum(pos - cnt[:, None], 0), 1)
    out[todo] = np.where(mask, chosen, fill)
  return out


def lognormal_layer(w, tau, batch, seq, top_k, row_len, seed, step, layer):
  """[batch, seq, top_k] int32 routing of one layer."""
  rng_seg = np.random.default_rng([seed, _TAG_SEG, step, layer])
  doc, ndoc = document_ids(batch, seq, row_len, rng_seg)
  logits = np.broadcast_to(w, (ndoc, w.shape[0]))
  if tau > 0:
    u = np.random.default_rng([seed, _TAG_U, step, layer]).standard_normal(
        (ndoc, w.shape[0]))
    logits = logits + tau * u
  rng = np.random.default_rng([seed, _TAG_TOK, step, layer])
  return sample_topk(logits, doc.ravel(), top_k, rng).reshape(batch, seq, top_k)


def lognormal_routing(batch, seq, num_layers, num_experts, top_k, sigmas, seed,
                      step, *, row_taus=0.0, row_len=0, profile=None,
                      threads=None, w=None):
  """[batch, seq, num_layers, top_k] int32 synthetic routing."""
  if w is None:
    w = layer_log_popularity(num_layers, num_experts, seed, sigmas, profile)
  taus = depth_map(_parse_list(row_taus) or [0.0], num_layers)
  out = np.empty((batch, seq, num_layers, top_k), dtype=np.int32)

  def one(l):
    out[:, :, l, :] = lognormal_layer(w[l], taus[l], batch, seq, top_k,
                                      row_len, seed, step, l)

  threads = threads or min(num_layers, os.cpu_count() or 1, 32)
  if threads <= 1:
    for l in range(num_layers):
      one(l)
  else:
    with futures.ThreadPoolExecutor(threads) as ex:
      list(ex.map(one, range(num_layers)))
  return out


def balanced_routing(batch, seq, num_layers, num_experts, top_k, step):
  """Round-robin routing: every expert gets exactly the same number of assignments."""
  t = np.arange(batch * seq, dtype=np.int64)[:, None]
  k = np.arange(top_k, dtype=np.int64)[None, :]
  base = (t * top_k + k + step) % num_experts  # top_k consecutive experts: distinct per token
  out = np.broadcast_to(base.reshape(batch, seq, 1, top_k), (batch, seq, num_layers, top_k))
  return np.ascontiguousarray(out, dtype=np.int32)


def _layer_sigmas(config, num_layers):
  vals = _parse_list(getattr(config, "synthetic_routing_sigmas", ""))
  if not vals:
    vals = [float(getattr(config, "synthetic_routing_sigma", 0.0))]
  return depth_map(vals, num_layers)


def _row_taus(config):
  vals = _parse_list(getattr(config, "synthetic_routing_row_taus", ""))
  return vals or [float(getattr(config, "synthetic_routing_row_tau", 0.0) or 0.0)]


class SyntheticRouting:
  """Host-generated pool of forced routing batches, placed on device once."""

  def __init__(self, config, sharding):
    import jax  # pylint: disable=import-outside-toplevel

    self.mode = config.synthetic_routing
    batch = int(config.global_batch_size_to_load)
    seq = int(config.max_target_length)
    num_layers = int(config.num_decoder_layers)
    num_experts = int(config.num_experts)
    top_k = int(config.num_experts_per_tok)
    pool = int(config.synthetic_routing_pool_size)
    seed = int(config.synthetic_routing_seed)
    row_len = int(getattr(config, "synthetic_routing_row_len", 0) or 0)
    prof_path = getattr(config, "synthetic_routing_profile", "") or ""
    profile = None
    if prof_path and self.mode == "lognormal":
      if "://" in prof_path or prof_path.startswith("/cns/"):  # gs:// or CNS (etils ships with MaxText)
        from etils import epath  # pylint: disable=import-outside-toplevel
        opener = epath.Path(prof_path).open
      else:
        opener = lambda mode: open(prof_path, mode)  # pylint: disable=unnecessary-lambda-assignment
      with opener("rb") as f:
        profile = np.load(f)["log_profile"]
    w = None
    if self.mode == "lognormal":
      w = layer_log_popularity(num_layers, num_experts, seed,
                               _layer_sigmas(config, num_layers), profile)
    self.pool = []
    for step in range(pool):
      if self.mode == "lognormal":
        arr = lognormal_routing(batch, seq, num_layers, num_experts, top_k,
                                None, seed, step, row_taus=_row_taus(config),
                                row_len=row_len, w=w)
      elif self.mode == "balanced":
        arr = balanced_routing(batch, seq, num_layers, num_experts, top_k, step)
      else:
        raise ValueError(f"Unknown synthetic_routing mode {self.mode!r}")
      self.pool.append(jax.device_put(arr, sharding))
    self.i = 0

  def next(self):
    arr = self.pool[self.i % len(self.pool)]
    self.i += 1
    return arr
