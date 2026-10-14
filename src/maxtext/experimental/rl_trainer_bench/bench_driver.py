#!/usr/bin/env python3
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

"""Standalone driver for the e2e RL trainer program (no orchestrator, no rollout engines).

Runs as the Pathways client in the trainer pod and drives the same objects the e2e run drives, through the
Tunix code paths themselves:

  * trainer: `run_trainer_node._parse_args` + `_create_maxtext_trainer_factory` + `TrainerWorker`, i.e. the
    MaxText training engine wrapped in the Tunix adapter, built from the e2e trainer's own argv (and the
    MAXTEXT_EXTRA_FLAGS env it reads);
  * algorithm and inputs: `run_deepswe_dist._parse_args` + `_build_algo` (GRPOAdapter/GRPOConfig) from the e2e
    orchestrator's own argv, the program's `BatchConfig` and `create_batch_assembler` as `StandardRLProgram`
    builds them, and the TRAINER_WARMUP_COMPILE payload (`trainer_warmup.build_warmup_payload`), which runs
    synthetic trajectories through `GRPOAdapter.create_trainer_payloads` and the assembler;
  * configure_worker: the body of `DistributedRLEngine.configure_worker` for the ACTOR role (loss fn with
    has_aux=True, gen_model_input_fn with the assembler's pad/eos ids), then `TrainerWorker.compile` as
    `RLProgram.run_async` does for the warm-up;
  * steps: STEPS optimizer steps of MICRO_STEPS `fwd_bwd` calls on the payload, then `update`, as
    `DistributedRLEngine.train_step` drives them.

Payload (PAYLOAD=realistic, the default): the warm-up payload only fixes shapes, dtypes and the per-row segment
count; the micro-batches the steps run are rebuilt from it by `realistic_payload` below, with seeded varied token
ids, replayed routing whose per-layer max expert-shard load ratio follows a calibration table
(PROBE_REGULAR_MEAN_RATIO, measured on a 397B production run; ROUTING_TARGETS overrides it), and tail padding
(identical pad ids, UNSET routing, as the packer writes it). The step's last micro-batch additionally
has FINAL_PAD_FRACTION of its rows fully padded, which reproduces the final micro-batch's load spike, and is called
with apply_optimizer=True as the orchestrator does. PAYLOAD=warmup restores the old behaviour (every token the same
id, every routed expert UNSET, so the router sends every token to the same experts: a degenerate MoE load).

Numerics knobs (moe_exclude_padding_from_routing on/off A/B, see README "Numerics check"): ADVANTAGES=random gives
every segment a seeded N(0, 1) advantage (token ids and routing unchanged); LOGP_GATES=off strips the orchestrator's
seq_logprob_error_threshold / truncated_importance_sampling_* flags. With them on, the payload's synthetic rollout
log-probs (-0.7) are far from a random-init trainer's, both gates drop every sequence and the gradient is exactly 0.
After each step the driver logs `STEP_LOSS step=N loss= grad_norm= micro_loss=[..] micro_denominator=[..]` and
`STEP_METRICS step=N ...` (every trainer metric; per-micro-batch ones as lists).

The intended difference from e2e: no checkpoint is loaded (random init; the trainer argv's --maxtext_ckpt_path is
cleared by the launcher), so weights and routing values differ while shapes, shardings and the program match.

STAGE_BEFORE_UPDATE=on (default off; anything else raises): the step boundary of a trainer that stages the next step's
first micro-batch before the previous step's update (Tunix's `TrainerWorker.stage_fwd_bwd` before
`apply_deferred_update`). After step k's last `fwd_bwd`, and before its `update`, step k+1's first micro-batch is mapped
to loss inputs and placed on the mesh (`prepare_fwd_bwd_input`, i.e. its host-to-device copy runs while the device
still computes step k's last micro-batch), the driver waits for that micro-batch (`wait_for_device`, which reports how
long it still ran), then runs the update, the drain and the metrics read exactly as with the switch off, and step k+1's
first `fwd_bwd` takes the prepared input. The trainer sees the same micro-batches, in the same order, with the same
`apply_optimizer` flags and one `update` per step. It needs a training engine with `prepare_fwd_bwd_input` and
`wait_for_device` and fails at start without them. Off, the loop is the original one: every added line is behind the
switch. Each staged boundary logs `STAGE_BEFORE_UPDATE step=k ...` after the step's `completed step` line.
"""

import logging
import os
import shlex
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TrainerBench] %(message)s", force=True)

import dataclasses

import numpy as np

UNSET_ROUTED_EXPERT = -1  # tunix.experimental.common.datatypes.UNSET_ROUTED_EXPERT

# Default routing calibration. Per-layer max expert-shard load ratio (EP=8) of the regular (non-final) micro-batches
# of a Qwen3.5-397B-A17B GRPO production run, mean over 65 regular micro-batches, read from the engine's per-layer
# load-ratio log lines (per micro-batch the max over layers ranged 1.71-2.45, median 2.00). Probe order:
# position-major over the scanned 4-layer blocks, i.e. index k is model layer 4 * (k % 15) + k // 15 (see
# `probe_index_to_layer`). For another model or a later training phase, pass your own table in the same order with
# ROUTING_TARGETS="r0,r1,..." (one ratio per layer), or stretch this one with ROUTING_SCALE.
PROBE_REGULAR_MEAN_RATIO = (
    1.721, 1.499, 1.276, 1.515, 1.541, 1.797, 1.466, 1.607, 1.743, 1.341, 1.504, 1.917, 1.570, 1.506, 1.450,
    1.269, 1.823, 1.241, 1.496, 1.417, 1.388, 1.316, 1.665, 1.820, 1.353, 1.576, 1.336, 1.926, 1.355, 1.532,
    1.773, 1.518, 1.224, 1.733, 1.507, 1.250, 1.579, 1.466, 1.376, 1.507, 1.558, 1.309, 1.534, 1.733, 1.543,
    1.280, 1.519, 1.475, 1.497, 1.262, 1.511, 1.463, 1.698, 1.694, 1.746, 1.664, 1.407, 1.524, 1.721, 1.995,
)


def probe_index_to_layer(k, num_layers=60, cycle=4):
  """Model layer of index k of the probe's per-layer vector (position-major over `cycle`-layer scanned blocks)."""
  blocks = num_layers // cycle
  return cycle * (k % blocks) + k // blocks


def parse_routing_targets(text, num_layers):
  """ROUTING_TARGETS="r0,r1,...": one max shard-load ratio per layer in probe order; validated against the depth."""
  vals = [float(x) for x in text.replace(" ", "").split(",") if x]
  if len(vals) != num_layers:
    raise ValueError(f"ROUTING_TARGETS has {len(vals)} values, the model has {num_layers} MoE layers")
  if min(vals) < 1.0:
    raise ValueError("ROUTING_TARGETS values are max/balanced load ratios and must be >= 1.0")
  return tuple(vals)


def layer_targets(num_layers, scale=1.0, cycle=4, probe=None):
  """Per model layer target max shard-load ratio: the probe table (default PROBE_REGULAR_MEAN_RATIO, in probe
  order), stretched by `scale` around 1.0."""
  probe = np.asarray(PROBE_REGULAR_MEAN_RATIO if probe is None else probe, dtype=np.float64)
  if num_layers != probe.size:
    # Another depth: cycle the probe's model-ordered profile.
    by_layer = np.empty_like(probe)
    by_layer[[probe_index_to_layer(k, probe.size, cycle) for k in range(probe.size)]] = probe
    by_layer = np.resize(by_layer, num_layers)
  else:
    by_layer = np.empty_like(probe)
    by_layer[[probe_index_to_layer(k, num_layers, cycle) for k in range(num_layers)]] = probe
  return 1.0 + scale * (by_layer - 1.0)


def _gumbel_topk(log_w, n, k, rng):
  """n draws of k distinct experts, without replacement, with probability proportional to exp(log_w)."""
  g = rng.gumbel(size=(n, log_w.size)) + log_w[None, :]
  return np.argpartition(-g, k - 1, axis=1)[:, :k]


def _pool_ratio(pool, num_experts, ep):
  """Max over expert shards of the pool's assignments to the shard, over the balanced share."""
  counts = np.bincount((pool // (num_experts // ep)).ravel(), minlength=ep)
  return counts.max() * ep / pool.size


def routing_pool(target, num_experts, ep, top_k, pool_size, rng, hot_shard):
  """`pool_size` distinct-expert top-k rows whose max shard-load ratio is ~`target`: `hot_shard`'s experts get a
  weight multiplier found by bisection, the other experts are uniform."""
  per_shard = num_experts // ep
  hot = np.zeros(num_experts, dtype=bool)
  hot[hot_shard * per_shard:(hot_shard + 1) * per_shard] = True
  def draw(log_m, n):
    sub = np.random.default_rng(rng.integers(1 << 62))
    return _gumbel_topk(np.where(hot, log_m, 0.0), n, top_k, sub)
  lo, hi = 0.0, 12.0
  if target <= 1.0:
    return draw(0.0, pool_size)
  calib = min(pool_size, 4096)
  for _ in range(20):
    mid = 0.5 * (lo + hi)
    if _pool_ratio(draw(mid, calib), num_experts, ep) < target:
      lo = mid
    else:
      hi = mid
  return draw(0.5 * (lo + hi), pool_size)


def realistic_payload(base, *, num_experts, ep=8, routing="probe", scale=1.0, pad_fraction=0.07,
                      pad_rows_fraction=0.0, pad_id=0, vocab_size=151_000, seed=0, pool_size=8192, cycle=4,
                      layer_targets_override=None, advantages="inherit", probe_targets=None):
  """`base` (the warm-up payload) with realistic tokens, routing and padding; same shapes, dtypes, fields.

  Per row: the real tokens fill the first (1 - pad_fraction) of the row, split into the row's own number of
  segments in `base` (each segment a prompt part, completion_mask 0, then a completion part, mask 1, in the
  prompt/completion proportion of `base`); the tail is padding: segment id 0, pad_id, every mask 0 and UNSET
  routing, as the packer writes it. `pad_rows_fraction` of the rows (the last ones) are entirely padding.
  Real tokens get seeded ids in [256, vocab_size) and replayed routing: routing="probe" draws each (token, layer)
  from a per-layer pool calibrated to the probe's per-layer max shard-load ratio (scaled by `scale`), with a
  seeded hot shard per layer (`probe_targets` replaces the default table, same order); routing="uniform" draws
  top-k uniformly without replacement.
  Advantages are one value per segment, constant over its completion tokens (as `create_trainer_payloads` writes a
  trajectory's advantage): advantages="inherit" draws it from the warm-up payload's non-zero values (+-1 if it has
  none); advantages="random" draws it standard normal from a separate generator seeded [seed, 1], so token ids,
  segment cuts and routing stay byte-identical to the "inherit" payload of the same seed.
  """
  if advantages not in ("inherit", "random"):
    raise ValueError(f"advantages must be inherit or random, got {advantages!r}")
  adv_rng = np.random.default_rng([seed, 1])
  rng = np.random.default_rng(seed)
  ids0 = np.asarray(base.completion_ids)
  seg0 = np.asarray(base.segment_ids)
  cm0 = np.asarray(base.completion_mask)
  adv0 = np.asarray(base.advantages)
  routed0 = np.asarray(base.routed_experts) if base.routed_experts is not None else None
  rows, budget = ids0.shape
  real0 = seg0 > 0
  prompt_frac = 1.0 - (cm0[real0].mean() if real0.any() else 0.5)
  adv_vals = np.unique(adv0[adv0 != 0]) if (adv0 != 0).any() else np.array([1.0, -1.0])

  ids = np.full_like(ids0, pad_id)
  seg = np.zeros_like(seg0)
  pos = np.zeros_like(np.asarray(base.segment_positions))
  cm = np.zeros_like(cm0)
  adv = np.zeros_like(adv0)
  n_pad_rows = int(round(pad_rows_fraction * rows))
  real_len = int(round((1.0 - pad_fraction) * budget))
  real_mask = np.zeros((rows, budget), dtype=bool)
  for r in range(rows - n_pad_rows):
    n_seg = max(1, len(np.unique(seg0[r][seg0[r] > 0])))
    cuts = np.sort(rng.choice(np.arange(64, real_len, 64), size=n_seg - 1, replace=False)) if n_seg > 1 else []
    bounds = [0, *cuts, real_len]
    for s in range(n_seg):
      a, b = bounds[s], bounds[s + 1]
      seg[r, a:b] = s + 1
      pos[r, a:b] = np.arange(b - a)
      p_end = a + int(prompt_frac * (b - a))
      cm[r, p_end:b] = 1
      a_val = adv_vals[rng.integers(adv_vals.size)]  # always drawn, so `rng` advances the same in both modes
      adv[r, p_end:b] = adv_rng.standard_normal() if advantages == "random" else a_val
    pos[r, real_len:] = np.arange(budget - real_len)
    real_mask[r, :real_len] = True
  for r in range(rows - n_pad_rows, rows):
    pos[r] = np.arange(budget)
  ids[real_mask] = rng.integers(256, vocab_size, size=int(real_mask.sum()), dtype=ids.dtype)

  out = dict(completion_ids=ids, segment_ids=seg, segment_positions=pos, completion_mask=cm, advantages=adv)
  for name in ("old_per_token_logps", "rollout_per_token_logps"):
    v = getattr(base, name, None)
    if v is not None:
      out[name] = np.where(cm > 0, np.float32(-0.7), np.float32(0.0)).astype(np.asarray(v).dtype)
  if routed0 is not None:
    num_layers, top_k = routed0.shape[2], routed0.shape[3]
    routed = np.full(routed0.shape, UNSET_ROUTED_EXPERT, dtype=routed0.dtype)
    targets = (layer_targets_override if layer_targets_override is not None
               else layer_targets(num_layers, scale, cycle, probe_targets))
    flat_real = np.flatnonzero(real_mask.ravel())
    routed_flat = routed.reshape(rows * budget, num_layers, top_k)
    for l in range(num_layers):
      if routing == "uniform":
        pool = _gumbel_topk(np.zeros(num_experts), pool_size, top_k, rng)
      else:
        pool = routing_pool(float(targets[l]), num_experts, ep, top_k, pool_size, rng, int(rng.integers(ep)))
      routed_flat[flat_real, l, :] = pool[rng.integers(pool_size, size=flat_real.size)].astype(routed.dtype)
    out["routed_experts"] = routed
  return dataclasses.replace(base, **out)


def payload_shard_load_ratio(payload, *, num_experts, ep=8, ep_rows=2, context=4, chunks=4):
  """Per layer, the max over EP groups, token chunks and shards of the replayed (real-token) load over the
  balanced load, laid out as the ring-of-experts layer sees it under cp-as-ep: an EP group is `ep_rows` rows
  (expert axis) x `context` sequence slices, each slice split into `chunks` token chunks, and a chunk all-gathers
  its slice-chunk from every member. The balanced load counts every token, padding included, as the layer's
  buffer sizing does; padding's own (router-chosen) experts are unknown here and count for nothing, so this is the
  replayed tokens' share. Returns (per_layer_ratio[L], pad_token_fraction).
  """
  routed = np.asarray(payload.routed_experts)
  rows, budget, num_layers, top_k = routed.shape
  per_shard = num_experts // ep
  slice_len = budget // context
  chunk_len = slice_len // chunks
  best = np.zeros(num_layers)
  for g in range(rows // ep_rows):
    rsel = slice(g * ep_rows, (g + 1) * ep_rows)
    for j in range(chunks):
      cols = np.concatenate([np.arange(c * slice_len + j * chunk_len, c * slice_len + (j + 1) * chunk_len)
                             for c in range(context)])
      block = routed[rsel][:, cols]  # [ep_rows, tokens, L, K]
      n_tok = block.shape[0] * block.shape[1]
      balanced = n_tok * top_k / ep
      for l in range(num_layers):
        e = block[:, :, l, :].ravel()
        e = e[e >= 0]
        counts = np.bincount(e // per_shard, minlength=ep)
        best[l] = max(best[l], counts.max() / balanced)
  pad = float((np.asarray(payload.segment_ids) == 0).mean())
  return best, pad


# Orchestrator flags of the sampler-vs-trainer log-prob gates (GRPO loss: `sequence_loss_mask` and
# `truncated_importance_weights`). LOGP_GATES=off strips them, so they take their argparse default None (gate off).
LOGP_GATE_FLAGS = ("--seq_logprob_error_threshold", "--truncated_importance_sampling_type",
                   "--truncated_importance_sampling_ratio_min", "--truncated_importance_sampling_ratio")


def strip_logp_gates(argv):
  """`argv` without the log-prob gate flags (both `--flag=v` and `--flag v` forms)."""
  out, skip = [], False
  for t in argv:
    if skip:
      skip = False
      continue
    name = t.split("=", 1)[0]
    if name in LOGP_GATE_FLAGS:
      skip = "=" not in t
      continue
    out.append(t)
  return out


def summarize_metrics(buf):
  """Host-side summary of one step's MetricsBuffer (after `jax.device_get`).

  The engine appends one entry per micro-batch to every metric recorded in `fwd_bwd` (`loss` and the loss's aux, as
  WeightedMetric sums/denominators or scalar arrays) and one entry per step to those recorded in `update`
  (gradient_norm, learning_rate, step_skipped, moe_* counters). Returns a dict with:
    loss: the step's token-weighted loss, sum of the micro-batch sums over the sum of their denominators (the
      denominator the engine normalizes the accumulated gradients by), NaN when that denominator is 0;
    micro_loss / micro_denominator: per micro-batch sum/denominator (NaN where 0) and denominator;
    weighted: name -> pooled value (sum over sums / max(sum over denominators, min_denom), NaN when 0);
    scalar: name -> float for one entry, else the list of per-micro-batch values.
  """
  out = {"loss": float("nan"), "micro_loss": [], "micro_denominator": [], "weighted": {}, "scalar": {}}
  for name, wm in sorted((getattr(buf, "weighted_metrics", None) or {}).items()):
    sums = np.asarray(wm.unreduced_sum, dtype=np.float64).ravel()
    dens = np.asarray(wm.denominator, dtype=np.float64).ravel()
    tot = dens.sum() + (wm.eps or 0.0)
    if wm.min_denom is not None:
      tot = max(tot, wm.min_denom)
    pooled = float(sums.sum() / tot) if tot != 0 else float("nan")
    out["weighted"][name] = pooled
    if name == "loss":
      out["loss"] = pooled
      with np.errstate(divide="ignore", invalid="ignore"):
        out["micro_loss"] = [float(x) for x in np.where(dens != 0, sums / np.where(dens != 0, dens, 1), np.nan)]
      out["micro_denominator"] = [float(x) for x in dens]
  for name, v in sorted((getattr(buf, "scalar_metrics", None) or {}).items()):
    arr = np.asarray(v, dtype=np.float64).ravel()
    out["scalar"][name] = float(arr[0]) if arr.size == 1 else [float(x) for x in arr]
  return out


def metric_log_lines(step, summary):
  """`STEP_LOSS` (loss, per-micro-batch loss and denominators, grad norm) and `STEP_METRICS` (every metric; a
  per-micro-batch metric as its list) lines for one step."""
  fmt = lambda x: f"{x:.9g}"
  gn = summary["scalar"].get("gradient_norm", float("nan"))
  loss_line = (f"STEP_LOSS step={step} loss={fmt(summary['loss'])} grad_norm={fmt(gn) if isinstance(gn, float) else gn} "
               f"micro_loss=[{','.join(fmt(x) for x in summary['micro_loss'])}] "
               f"micro_denominator=[{','.join(fmt(x) for x in summary['micro_denominator'])}]")
  parts = [f"{k}={fmt(v)}" for k, v in summary["weighted"].items()]
  for k, v in summary["scalar"].items():
    parts.append(f"{k}={fmt(v)}" if isinstance(v, float) else f"{k}=[{','.join(fmt(x) for x in v)}]")
  return loss_line, f"STEP_METRICS step={step} " + " ".join(parts)


def main() -> int:
  trainer_argv = shlex.split(os.environ["TRAINER_ARGV"])
  orch_argv = shlex.split(os.environ["ORCH_ARGV"])
  # LOGP_GATES=off (numerics A/B): the payload's rollout log-probs are synthetic (-0.7 per completion token), so
  # against a random-init trainer the seq_logprob_error_threshold and seq-mask-tis gates drop every sequence, the
  # loss mask is empty and the gradient is exactly 0. Stripping the gate flags keeps every sequence in the loss.
  logp_gates = os.environ.get("LOGP_GATES", "on")
  if logp_gates == "off":
    orch_argv = strip_logp_gates(orch_argv)
  elif logp_gates != "on":
    raise ValueError(f"LOGP_GATES must be on or off, got {logp_gates!r}")
  steps = int(os.environ.get("STEPS", "3"))
  micro_steps = int(os.environ.get("MICRO_STEPS", "4"))
  stage_before_update = os.environ.get("STAGE_BEFORE_UPDATE", "off")
  if stage_before_update not in ("on", "off"):
    raise ValueError(f"STAGE_BEFORE_UPDATE must be on or off, got {stage_before_update!r}")
  stage_before_update = stage_before_update == "on"

  # Same JAX init as K8sJaxContext.initialize() under JAX_PLATFORMS=proxy.
  import pathwaysutils  # pylint: disable=g-import-not-at-top
  pathwaysutils.initialize()
  import jax  # pylint: disable=g-import-not-at-top

  from tunix.experimental.examples.common import run_trainer_node  # pylint: disable=g-import-not-at-top
  from tunix.experimental.examples.deepswe_dist import run_deepswe_dist  # pylint: disable=g-import-not-at-top
  from tunix.experimental.examples.deepswe_dist import trainer_warmup  # pylint: disable=g-import-not-at-top
  from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
  from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
  from tunix.experimental.worker import trainer_worker  # pylint: disable=g-import-not-at-top
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  t_args = run_trainer_node._parse_args(trainer_argv)  # pylint: disable=protected-access
  o_args = run_deepswe_dist._parse_args(orch_argv)  # pylint: disable=protected-access
  logging.info("Trainer args: %s", t_args)
  logging.info("Orchestrator args: %s", o_args)

  # Trainer worker, as run_trainer_node.main builds it.
  trainer_factory, mesh = run_trainer_node._create_maxtext_trainer_factory(t_args)  # pylint: disable=protected-access
  worker = trainer_worker.TrainerWorker(
      trainer_factory=trainer_factory,
      worker_id=t_args.worker_id,
      logps_chunk_size=t_args.compute_logps_chunk_size,
      logps_micro_batch_size=t_args.compute_logps_micro_batch_size,
      execution_context=mesh,
  )
  t0 = time.monotonic()
  worker.initialize()
  logging.info("Trainer worker initialized in %.1fs on %d devices.", time.monotonic() - t0, jax.device_count())
  if stage_before_update:
    engine = worker._trainer  # pylint: disable=protected-access
    missing = [n for n in ("prepare_fwd_bwd_input", "wait_for_device") if not callable(getattr(engine, n, None))]
    if missing:
      raise RuntimeError(f"STAGE_BEFORE_UPDATE=on needs a training engine with {missing}; {type(engine).__name__} "
                         "has none (install the engine overlay first)")
    logging.info("STAGE_BEFORE_UPDATE on: step k+1's first micro-batch is prepared on the mesh before step k's update "
                 "(%s.prepare_fwd_bwd_input, then wait_for_device).", type(engine).__name__)

  # Tokenizer ids, as run_deepswe_dist.main derives them.
  tokenizer_path = o_args.tokenizer_path or os.getenv("MODEL_DIR") or o_args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

  # Algorithm, BatchConfig and assembler, as run_deepswe_dist.main / StandardRLProgram build them.
  algo = run_deepswe_dist._build_algo(o_args)  # pylint: disable=protected-access
  batch_config = batch_assembly.BatchConfig(
      pad_id=pad_id,
      max_prompt_length=o_args.max_prompt_length,
      max_response_length=o_args.max_response_length,
      max_seq_token_per_tpu=o_args.max_seq_token_per_tpu,
      max_segments_per_packed_row=o_args.max_segments_per_packed_row,
      trainer_fsdp=o_args.trainer_fsdp,
      trainer_dp=o_args.trainer_dp,
      trainer_expert=o_args.trainer_expert,
  )
  assembler = batch_assembly.create_batch_assembler(
      num_generations=o_args.num_generations,
      mini_batch_size=o_args.mini_batch_size,
      train_micro_batch_size=getattr(algo, "train_micro_batch_size", 1),
      batch_config=batch_config,
  )

  # configure_worker (DistributedRLEngine.configure_worker, ACTOR role).
  a_pad = getattr(assembler, "pad_id", pad_id)
  a_eos = getattr(assembler, "eos_id", a_pad)
  logging.info("configure_worker: loss fn + gen_model_input_fn (pad_id=%s, eos_id=%s).", a_pad, a_eos)
  worker.with_loss_fn(algo.loss_fn(), has_aux=True)
  worker.with_gen_model_input_fn(algo.build_gen_model_input_fn(pad_id=a_pad, eos_id=a_eos))

  # TRAINER_WARMUP_COMPILE payload, as run_deepswe_dist.main builds it.
  routed_shape = None
  if o_args.trainer_warmup_routed_experts:
    routed_shape = trainer_warmup.routed_experts_shape_from_config(tokenizer_path)
    if routed_shape is None and o_args.trainer_warmup_routed_layers and o_args.trainer_warmup_routed_top_k:
      routed_shape = (o_args.trainer_warmup_routed_layers, o_args.trainer_warmup_routed_top_k)
  logging.info("routed_experts [num_layers, top_k] = %s", routed_shape)
  payload = trainer_warmup.build_warmup_payload(
      algo=algo,
      batch_config=batch_config,
      num_generations=o_args.num_generations,
      mini_batch_size=o_args.mini_batch_size,
      train_micro_batch_size=algo.train_micro_batch_size,
      routed_experts_shape=routed_shape,
  )
  logging.info("Bench micro-batch: %s", batch_assembly.describe_payload(payload))

  t0 = time.monotonic()
  worker.compile(dummy_data=payload)
  logging.info("Trainer compile finished in %.1fs.", time.monotonic() - t0)

  # The micro-batches the steps run (see the module docstring).
  mode = os.environ.get("PAYLOAD", "realistic")
  regular = final = payload
  if mode == "realistic":
    cfg = worker._trainer._config  # pylint: disable=protected-access
    knobs = dict(
        num_experts=int(cfg.num_experts),
        ep=int(os.environ.get("EP_SIZE", "8")),
        routing=os.environ.get("ROUTING", "probe"),
        scale=float(os.environ.get("ROUTING_SCALE", "1.0")),
        pad_id=a_pad,
        vocab_size=int(getattr(tokenizer, "vocab_size", 151_000)),
        cycle=int(getattr(cfg, "inhomogeneous_layer_cycle_interval", 4) or 4),
        advantages=os.environ.get("ADVANTAGES", "inherit"),
    )
    if os.environ.get("ROUTING_TARGETS"):
      n_layers = int(np.asarray(payload.routed_experts).shape[2]) if payload.routed_experts is not None else 0
      knobs["probe_targets"] = parse_routing_targets(os.environ["ROUTING_TARGETS"], n_layers)
    seed = int(os.environ.get("PAYLOAD_SEED", "0"))
    tg = time.monotonic()
    regular = realistic_payload(payload, pad_fraction=float(os.environ.get("PAD_FRACTION", "0.07")), seed=seed, **knobs)
    final = realistic_payload(payload, pad_fraction=float(os.environ.get("PAD_FRACTION", "0.07")),
                              pad_rows_fraction=float(os.environ.get("FINAL_PAD_FRACTION", "0.25")), seed=seed + 1,
                              **knobs)
    logging.info("Realistic payloads built in %.1fs (%s).", time.monotonic() - tg, knobs)
    for name, p in (("regular", regular), ("final", final)):
      a = np.asarray(p.advantages)[np.asarray(p.completion_mask) > 0]
      logging.info("PAYLOAD_ADVANTAGES %s rows=%d scored_tokens=%d nonzero_frac=%.4f mean=%.6g std=%.6g min=%.6g "
                   "max=%.6g", name, np.asarray(p.completion_ids).shape[0], a.size,
                   float((a != 0).mean()) if a.size else 0.0, a.mean() if a.size else 0.0,
                   a.std() if a.size else 0.0, a.min() if a.size else 0.0, a.max() if a.size else 0.0)
    for name, p in (("regular", regular), ("final", final)):
      if p.routed_experts is None:
        continue
      ratios, pad = payload_shard_load_ratio(p, num_experts=knobs["num_experts"], ep=knobs["ep"])
      top = np.argsort(-ratios)[:5]
      logging.info(
          "PAYLOAD_ROUTING %s pad_fraction=%.4f replayed_max_ratio=%.4f hottest_layers=%s ratios=[%s]",
          name, pad, ratios.max(), [(int(l), round(float(ratios[l]), 3)) for l in top],
          ",".join(f"{v:.3f}" for v in ratios),
      )
  elif mode != "warmup":
    raise ValueError(f"PAYLOAD must be realistic or warmup, got {mode!r}")

  staged = None  # STAGE_BEFORE_UPDATE: step k+1's first micro-batch as prepare_fwd_bwd_input returned it
  for step in range(steps):
    t0 = time.monotonic()
    times = []
    for m in range(micro_steps):
      tm = time.monotonic()
      last = m == micro_steps - 1
      req = datatypes.TrainRequest(request_id=f"bench_{step}_{m}", payload=final if last else regular, metadata={})
      if m == 0 and staged is not None:
        # The same micro-batch (step k+1's first, `regular`, or `final` when MICRO_STEPS is 1), already prepared.
        req = datatypes.TrainRequest(request_id=req.request_id, payload=staged, metadata={})
        staged = None
      # As DistributedRLEngine.train_step: the step's last micro-batch is the one with apply_optimizer=True.
      # Per-micro-step runtime load ratios: MOE_DROP_AUDIT lines with the MaxText flag log_moe_drops=true.
      worker.fwd_bwd(request=req, apply_optimizer=last)
      times.append(time.monotonic() - tm)
    if stage_before_update and step + 1 < steps:
      # Step k+1's first micro-batch (`regular`, or `final` when MICRO_STEPS is 1), mapped and placed on the mesh
      # while the device still computes this step's last micro-batch; then how long that micro-batch still ran.
      ts = time.monotonic()
      staged = engine.prepare_fwd_bwd_input(final if micro_steps == 1 else regular)
      stage_s = time.monotonic() - ts
      device_wait_s = engine.wait_for_device()
    worker.update()
    # Drain the step's device work before reading the clock, as maxtext_engine/train.py does.
    throttler = getattr(worker._trainer, "_throttler", None)  # pylint: disable=protected-access
    if throttler is not None:
      throttler.wait_for_all()
    else:
      jax.block_until_ready(worker.get_metrics())
    dt = time.monotonic() - t0
    logging.info(
        "completed step: %d, seconds: %.3f, micro_steps: %d, seconds/micro_step: %.3f, fwd_bwd dispatch s: %s",
        step, dt, micro_steps, dt / micro_steps, [round(x, 3) for x in times],
    )
    if stage_before_update and step + 1 < steps:
      logging.info("STAGE_BEFORE_UPDATE step=%d staged=bench_%d_0 stage_s=%.3f device_wait_s=%.3f", step, step + 1,
                   stage_s, device_wait_s)
    # Numerics comparison (e.g. moe_exclude_padding_from_routing on vs off): the step's loss (per micro-batch and
    # pooled), gradient norm and every other trainer metric, read after the step's timing so they do not affect it.
    try:
      loss_line, metrics_line = metric_log_lines(step, summarize_metrics(jax.device_get(worker.get_metrics())))
      logging.info("%s", loss_line)
      logging.info("%s", metrics_line)
    except Exception as e:  # pylint: disable=broad-except
      logging.info("STEP_METRICS step=%d unavailable: %r", step, e)
  worker.stop()
  return 0


if __name__ == "__main__":
  sys.exit(main())
