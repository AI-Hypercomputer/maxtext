"""Measure trainer-vs-sampler MaxText divergence on a small MoE, locally.

Why this exists
---------------
In the DeepSWE RL runs the sampler and the trainer are *both* MaxText, but they
are built from two different configs (see `train_maxtext_nb.py`, the
`pyconfig.initialize` calls for the trainer and for `sampler_config`). For an
MoE model those configs select structurally different code:

    trainer :  attention=flash      prefuse_moe_weights=False
               -> moe.py: w0_kernel = wi_0, w1_kernel = wi_1  (unfused)
               -> score_func = routed_score_func, applied in GateLogit

    sampler :  attention=vllm_rpa   prefuse_moe_weights=True
               -> moe.py:3469: fused_kernel = wi              (fused)
               -> score_func = "", applied inside fused_moe_gmm instead

A dense model has none of this, which is why the dense 4B runs showed a tiny
sampler/trainer offset and the 35B MoE run showed `seq_geomean` at 0.856 with
`tis/is_oob_ratio` = 1.0 and `grad_norm` = 0.

This script runs one forward pass down each path over identical tokens with
identical weights, and reports the same statistics the production loss reports,
so the numbers are directly comparable to wandb.

It needs no vLLM, no rollouts, no sandbox and no checkpoint. Random weights are
fine: we are measuring a numerical path difference, not model quality.

Two modes
---------
*Default* — two local forward passes, trainer config vs sampler config, over
random tokens. Measures the MaxText-vs-MaxText path difference. This mode has
already run and returned 0.000531 nats, inside the TIS band: **the MoE code
path is not the cause.**

*`--trajectory_csv`* — one trainer forward pass over the tokens a production
rollout actually produced, diffed against the per-token log-probabilities vLLM
recorded at the time. This compares the trainer against the *real sampler*,
which is what the production loss gates on, and it is the only thing here that
can see multi-turn context reassembly, paged-KV decode and prefix caching. It
reports where the disagreements land relative to assistant-turn boundaries,
which is the part that discriminates between the remaining hypotheses.

Usage
-----
    python3 moe_path_parity.py --layers 8 --seq_len 512
    python3 moe_path_parity.py --layers 8 --model qwen3-4b      # dense control
    python3 moe_path_parity.py --layers 8 --isolate_moe         # see below

    gcloud storage cp gs://niting-storage-europe-west4/deepswe-logs/\
qwen-35b-deepswe-v5p-256-maxtext-v6/trajectory_log_1789165722.csv .
    python3 moe_path_parity.py --layers 0 \
        --load_parameters_path gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items \
        --trajectory_csv trajectory_log_1789165722.csv --trajectory_rows 0,5

What the trajectory mode is predicted to show, so it can fail
-------------------------------------------------------------
From the production metrics (`token_weight_mean` 1.000235 with
`seq_geomean` 0.856 and `token_logdiff_absmean` 0.229), the disagreement
decomposes into a bulk of sigma ~0.10 nats plus a one-sided negative tail of
roughly 0.4-0.6% of scored tokens at 26+ nats -- about 32 per sequence against
a 30-turn agent budget. **If those outliers come back clustered at offset 0-2
of each assistant turn, the defect is in conversation reassembly. If they are
spread uniformly within turns, it is decode-time drift. If there are none, the
trainer scores the real tokens correctly and the defect is on the sampler
side.** All three are informative; the mode is only useless if it will not run.

`--isolate_moe` keeps attention identical on both sides and varies only the MoE
flags. That does NOT reproduce the production sampler path (the fused branch
requires vllm_rpa attention), but it runs without any vLLM machinery, so it is
the fallback if the vllm_rpa path will not execute standalone.

NOTE: this was written against the maxtext tree without being executed. Expect
to iterate on the two `pyconfig.initialize` argument lists in particular --
they mirror `train_maxtext_nb.py` lines 894-970, so diff against that file if
construction fails.
"""

import argparse
import gc
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

_RANDOM_WEIGHTS = False

os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")

from flax import nnx  # noqa: E402
from maxtext.configs import pyconfig, types  # noqa: E402
from maxtext.utils import model_creation_utils  # noqa: E402
from maxtext.utils import maxtext_utils_nnx  # noqa: E402
from maxtext.utils import maxtext_utils  # noqa: E402
from maxtext.common.common_types import MODEL_MODE_TRAIN  # noqa: E402
from flax import linen as nn  # noqa: E402


def _common_argv(args, layers=None):
  """Config keys shared by both paths."""
  return [
      f"model_name={args.model}",
      # The model yml pins base_num_decoder_layers (and use_mrope on qwen3.5),
      # and pyconfig refuses a CLI override of a model-config key unless this
      # is set.
      "override_model_config=True",
      *([f"base_num_decoder_layers={layers}"] if layers else []),
      f"max_target_length={args.seq_len}",
      f"max_prefill_predict_length={args.seq_len}",
      f"dtype={args.dtype}",
      f"float32_gate_logits={args.float32_gate_logits}",
      "skip_jax_distributed_system=True",
      "use_standalone_converter=False",
      "log_config=False",
      "allow_split_physical_axes=True",
      "scan_layers=True",
      "enable_dropout=False",
  ]


def shim_tpu_inference_envs(onehot_threshold=0):
  """Backfill MoE env flags the installed tpu_inference predates.

  MaxText's fused MoE path reads several tpu_inference.envs attributes
  (moe.py:3390-3393). tpu_inference.envs defines __getattr__ to raise on
  unknown names, so an older install fails the forward pass outright with
  AttributeError rather than falling back.

  The values below are upstream's own defaults. Nitin's JobSet does not set any
  of these either, so defaults are also what production runs with -- this
  restores the production behaviour rather than inventing one. It does mean the
  local tpu_inference is not the same build as production's, which is a caveat
  on any number this harness produces.
  """
  try:
    from tpu_inference import envs as tie
  except ImportError:
    return
  if onehot_threshold:
    setattr(tie, "ONEHOT_MOE_PERMUTE_THRESHOLD", onehot_threshold)
    print(f"  ONEHOT_MOE_PERMUTE_THRESHOLD = {onehot_threshold} -- taking the onehot permute "
          f"branch, NOT the sc_ragged_gather kernel production uses")
  for name, default in (("ENABLE_RS_KERNEL", False),
                        ("USE_GMM_FUSED_RS_KERNEL", False),
                        ("ONEHOT_MOE_PERMUTE_THRESHOLD", 0),
                        ("VLLM_MOE_CHUNK_SIZE", 0)):
    try:
      getattr(tie, name)
    except AttributeError:
      setattr(tie, name, default)
      print(f"  shimmed tpu_inference.envs.{name} = {default!r} (missing in this install)")

  # MaxText also passes kwargs the installed fused_moe_func may not accept
  # (moe.py:3367 sends moe_chunk_size, for instance). moe.py imports it inside
  # the function, so patch it at the source module. Every such kwarg is a
  # performance knob whose default is off, so dropping one leaves the installed
  # build at its own default -- but it does confirm the local tpu_inference is
  # older than production's image, which is a caveat on the result.
  try:
    import inspect
    from tpu_inference.layers.common import fused_moe_gmm as fmg
    real = fmg.fused_moe_func
    accepted = set(inspect.signature(real).parameters)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(real).parameters.values()):
      reported = set()

      def _compat(*a, **kw):
        drop = set(kw) - accepted
        if drop and drop != reported:
          reported.update(drop)
          print(f"  dropped kwargs this tpu_inference does not accept: {sorted(drop)}")
        return real(*a, **{k: v for k, v in kw.items() if k in accepted})

      fmg.fused_moe_func = _compat
  except ImportError:
    pass


def build_trainer_config(args, layers=None):
  """The trainer half of train_maxtext_nb.py:914-943."""
  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "post_train", "rl.yml")
  return pyconfig.initialize([
      "", base_yml, *_common_argv(args, layers),
      "attention=flash",
      f"prefuse_moe_weights={args.trainer_prefuse}",
      # train_maxtext_nb.py:901-908 maps --remat_policy=decoder onto MaxText
      # "full"; "decoder" itself trips the assert in get_remat_policy. Remat
      # only governs backward recompute and this harness is forward-only, so
      # it cannot move the numbers -- matched to production for fidelity only.
      f"remat_policy={args.remat_policy}",
      *([f"load_parameters_path={args.load_parameters_path}"] if args.load_parameters_path else []),
      # base.yml:550 declares an fsdp axis, so the trainer can shard freely.
      f"ici_fsdp_parallelism={args.devices}",
      "ici_tensor_parallelism=1",
  ], config_class=types.RLConfig)


def build_sampler_config(args, shard, layers=None):
  """The sampler half of train_maxtext_nb.py:946-970.

  `shard` is how many ways the weights may be split. On the vLLM mesh that is
  bounded by num_kv_heads -- see below -- so the caller derives it from the
  resolved trainer config.
  """
  cfg_dir = os.path.dirname(pyconfig.__file__)
  if args.isolate_moe:
    # Identical attention on both sides; only the MoE flags differ. Misses the
    # fused branch (which also needs vllm_rpa) but exercises the trainer mesh,
    # so it always runs.
    return pyconfig.initialize([
        "", os.path.join(cfg_dir, "post_train", "rl.yml"), *_common_argv(args, layers),
        "attention=flash",
        "prefuse_moe_weights=True",
        "model_call_mode=inference",
        "remat_policy=none",
        f"ici_fsdp_parallelism={args.devices}",
        "ici_tensor_parallelism=1",
    ], config_class=types.RLConfig)

  return pyconfig.initialize([
      "", os.path.join(cfg_dir, "inference", "vllm.yml"), *_common_argv(args, layers),
      "attention=vllm_rpa",
      f"prefuse_moe_weights={args.sampler_prefuse}",
      "model_call_mode=inference",
      "remat_policy=none",
      "use_mrope=False",
      # Sharding on the vLLM mesh, which is NOT the trainer's. vllm.yml:36:
      #   mesh_axes: ['data','attn_dp','model','expert','attn_dp_expert','dcp','pcp']
      # There is no fsdp axis, so ici_fsdp_parallelism is silently ignored
      # here. And vllm.yml:63 maps ['kv_heads', ['model','expert']], so both
      # shardable weight axes carry KV heads and their combined size must
      # divide num_kv_heads. Whatever is left over has to sit on 'data', which
      # replicates.
      f"ici_expert_parallelism={shard}",
      f"ici_data_parallelism={args.devices // shard}",
  ], config_class=types.RLConfig)


def copy_weights(src_state, dst_state):
  """Copy trainer params onto another model, conforming shape and layout.

  Two transformations are needed, and both are derived from the destination
  shape rather than assumed:

  fusing
    moe.py stores the MoE up-projections as wi_0/wi_1 when prefuse is off and
    as one concatenated `wi` when it is on; moe.py:2148 gives the relation as
    wi = concatenate([wi_0, wi_1], axis=-1).

  depth slicing
    The checkpoint is loaded at full depth and the working models are shallower.
    Layers are NOT stacked on a leading axis: Qwen3.5 sets
    inhomogeneous_layer_cycle_interval=4, so four layer types are scanned
    separately and the scan axis sits wherever param_scan_axis puts it -- e.g.
    A_log is (32, 10) at 40 layers and (32, 2) at 8. So rather than guess where
    the layer axis is, slice whichever axis differs from the destination and
    require every other axis to match exactly.
  """
  src = dict(jax.tree_util.tree_flatten_with_path(src_state)[0])
  dst_leaves, dst_def = jax.tree_util.tree_flatten_with_path(dst_state)
  by_name = {jax.tree_util.keystr(pth): v for pth, v in src.items()}

  def conform(v, want, key):
    v = np.asarray(v)
    if v.shape == want:
      return v, False
    if v.ndim != len(want):
      raise AssertionError(f"rank mismatch at {key}: {v.shape} vs {want}")
    bad = [i for i, (a_, b_) in enumerate(zip(v.shape, want)) if a_ != b_]
    for i in bad:
      if v.shape[i] < want[i]:
        raise AssertionError(
            f"{key}: source axis {i} is {v.shape[i]}, smaller than the required "
            f"{want[i]} -- this is not a depth slice, something else is wrong")
    for i in bad:
      v = np.take(v, range(want[i]), axis=i)
    return v, True

  out, fused, sliced, matched, missing = [], 0, 0, 0, []
  for pth, dst_val in dst_leaves:
    key = jax.tree_util.keystr(pth)
    want = tuple(np.asarray(dst_val).shape)
    if key in by_name:
      v, did = conform(by_name[key], want, key)
      matched += 1
      sliced += did
    elif key.endswith("['wi'].value") or key.endswith("['wi']"):
      # jax.tree_util.keystr renders a dict key as "['wi']", so matching on a
      # bare "wi" suffix silently never fires -- which is what left every MoE
      # weight unmatched. Swap only the final segment.
      suffix = "['wi'].value" if key.endswith("['wi'].value") else "['wi']"
      stem = key[: -len(suffix)]
      tail = ".value" if suffix.endswith(".value") else ""
      w0 = by_name.get(f"{stem}['wi_0']{tail}")
      w1 = by_name.get(f"{stem}['wi_1']{tail}")
      if w0 is None or w1 is None:
        missing.append(key)
        out.append(dst_val)
        continue
      v, did = conform(np.concatenate([np.asarray(w0), np.asarray(w1)], axis=-1), want, key)
      fused += 1
      sliced += did
    else:
      missing.append(key)
      out.append(dst_val)
      continue
    out.append(jnp.asarray(v))

  print(f"  weight copy: {matched} matched, {fused} fused (wi_0+wi_1 -> wi), "
        f"{sliced} depth-sliced, {len(missing)} unmatched")
  if missing:
    print("  UNMATCHED (left at this model's own init -- do not trust the result):")
    for k in missing[:20]:
      print("    ", k)
  return jax.tree_util.tree_unflatten(dst_def, out)


def per_token_logprobs(logits, ids):
  """log p(ids[t+1] | ids[:t+1]) -- the same quantity the RL loss compares."""
  lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  tgt = ids[:, 1:]
  return jnp.take_along_axis(lp[:, :-1], tgt[..., None], axis=-1)[..., 0]


def report(log_is, label=""):
  """Same statistics the production loss emits, so numbers are comparable."""
  a = np.asarray(log_is).ravel()
  finite = np.isfinite(a)
  n_bad = int((~finite).sum())
  if n_bad:
    pos = np.unique(np.argwhere(~np.isfinite(np.asarray(log_is)))[:, 1])
    print(f"\n  !! {n_bad} of {a.size} positions non-finite ({100*n_bad/a.size:.3f}%), "
          f"at {len(pos)} distinct sequence positions: {pos[:20].tolist()}")
    print("     Statistics below are over the FINITE subset only. Understand these")
    print("     before trusting the headline number -- a path that emits inf on some")
    print("     positions is not merely 'noisier', it is doing something different.")
    a = a[finite]
  geo = float(np.exp(a.mean()))
  print(f"\n===== {label} =====")
  print(f"  scored positions used                     : {a.size}")
  print(f"  mean signed log-ratio (trainer - sampler) : {a.mean():+.6f} nats")
  print(f"  seq_geomean equivalent  exp(mean)         : {geo:.6f}")
  print(f"  token_logdiff_absmean                     : {np.abs(a).mean():.6f}")
  print(f"  token_logdiff_absmax                      : {np.abs(a).max():.4f}")
  print(f"  mult_prob_error  mean(exp|d|)             : {np.exp(np.abs(a)).mean():.6g}")
  for thr in (2, 5, 10):
    n = int((np.abs(a) > thr).sum())
    print(f"  tokens with |log_is| > {thr:2d} nats            : {n} ({100*n/a.size:.4f}%)")
  lo, hi = 0.999, 1.002
  print(f"  in TIS band [{lo}, {hi}]?                 : {'YES' if lo <= geo <= hi else 'NO'}")
  print("\n  production 35B reference: seq_geomean 0.856, absmean 0.229, absmax ~25.9 nats")
  if _RANDOM_WEIGHTS:
    print("\n  " + "!" * 68)
    print("  RANDOM WEIGHTS: this number is NOT comparable to production.")
    print("  Untrained logits reach ~1300, where one bfloat16 step is 8 units, so")
    print("  log-softmax differences of ~1 nat are pure quantisation. A trained")
    print("  model sits at logits of 15-30 where the step is ~0.06. Pass")
    print("  --load_parameters_path to get a meaningful measurement.")
    print("  " + "!" * 68)


PAD_ID = 248044  # <|endoftext|>, what the rollout left-pads prompt_tokens with


def load_trajectory(csv_path, row_idx, max_tokens):
  """One row of a production `trajectory_log_*.csv`, ready to score.

  The two-forward-pass mode above compares MaxText against MaxText. This mode
  compares MaxText against *the sampler that actually generated the tokens*,
  using the per-token log-probabilities vLLM recorded at rollout time. That is
  the quantity the production loss gates on, and nothing else in this repo
  measures it.

  Returns `(ids, mask, sampler_lp, n_prompt, turn_id)` where `ids` is the full
  prompt+conversation stream, and `mask`/`sampler_lp`/`turn_id` are aligned to
  the *conversation* portion only.

  Three alignment details, each easy to get wrong and silently fatal:

  * `prompt_tokens` is left-padded with `<|endoftext|>` to `max_prompt_length`
    (4096 in every row observed). The padding is stripped here; leaving it in
    would score the conversation in a context of thousands of pad tokens.
  * The trainer's logprob at index `t` predicts token `t+1`, so conversation
    token `j` is scored by `lp[n_prompt + j - 1]`. The prompt must therefore be
    in the forward pass even though none of it is compared.
  * `conversation_masks` is the loss mask: 1 on assistant-generated tokens, 0
    on environment/tool output. Only masked-in positions are compared, exactly
    as `grpo_loss_fn` does.
  """
  import ast
  import csv
  import sys

  csv.field_size_limit(sys.maxsize)
  with open(csv_path, newline="") as fh:
    rows = list(csv.DictReader(fh))
  if row_idx >= len(rows):
    raise IndexError(f"row {row_idx} requested, CSV has {len(rows)}")
  r = rows[row_idx]

  def parse(field):
    s = r[field].strip()
    if not s or s == "None":
      raise ValueError(f"row {row_idx} has no {field}; pick another row")
    return np.asarray(ast.literal_eval(s if s.startswith("[") else "[" + s + "]"))

  prompt = parse("prompt_tokens")
  conv = parse("conversation_tokens")
  mask = parse("conversation_masks")
  lp = parse("old_logprobs")
  if not (len(conv) == len(mask) == len(lp)):
    raise ValueError(
        f"row {row_idx} is internally inconsistent: conversation_tokens={len(conv)} "
        f"conversation_masks={len(mask)} old_logprobs={len(lp)}. These must match; "
        "if they do not, the misalignment IS the bug and no forward pass is needed.")

  keep = int(np.argmax(prompt != PAD_ID)) if (prompt == PAD_ID).any() else 0
  prompt = prompt[keep:]
  if keep:
    print(f"  stripped {keep} leading pad tokens from prompt_tokens")

  if max_tokens and len(prompt) + len(conv) > max_tokens:
    room = max_tokens - len(prompt)
    if room < 64:
      raise ValueError(
          f"--trajectory_max_tokens {max_tokens} leaves only {room} tokens after a "
          f"{len(prompt)}-token prompt; raise it or pick a shorter row")
    conv, mask, lp = conv[:room], mask[:room], lp[:room]
    print(f"  truncated conversation to {room} tokens to fit --trajectory_max_tokens")

  # Turn index: assistant runs are maximal runs of mask==1. -1 on env tokens.
  turn_id = np.full(len(mask), -1, dtype=np.int32)
  on = mask > 0
  edges = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
  for t, (s, e) in enumerate(zip(np.where(edges == 1)[0], np.where(edges == -1)[0])):
    turn_id[s:e] = t

  ids = np.concatenate([prompt, conv]).astype(np.int32)
  print(f"  row {row_idx}: status={r.get('status')} reward={r.get('trajectory_reward')} "
        f"prompt={len(prompt)} conversation={len(conv)} "
        f"scored(masked-in)={int(on.sum())} assistant_turns={turn_id.max() + 1}")
  return ids, mask, lp.astype(np.float64), len(prompt), turn_id


def report_trajectory(log_is, mask, turn_id, ids, label=""):
  """Per-token attribution of the production sampler-vs-trainer disagreement.

  `report()` above answers "how big". This answers "where", which is the part
  that discriminates between the remaining hypotheses: a defect in multi-turn
  context reassembly fires at turn boundaries, paged-KV decode drift grows
  within a turn, and a pure numerics floor is uniform.
  """
  on = np.asarray(mask) > 0
  a = np.asarray(log_is)[on]
  tid = np.asarray(turn_id)[on]
  toks = np.asarray(ids)[len(ids) - len(mask):][on]
  finite = np.isfinite(a)
  if (~finite).sum():
    print(f"\n  !! {int((~finite).sum())} non-finite positions excluded from the statistics")
    a, tid, toks = a[finite], tid[finite], toks[finite]

  print(f"\n===== {label} =====")
  print(f"  scored positions (masked-in)              : {a.size}")
  print(f"  mean signed log-ratio (trainer - sampler) : {a.mean():+.6f} nats")
  print(f"  seq_geomean equivalent  exp(mean)         : {float(np.exp(a.mean())):.6f}")
  print(f"  token_logdiff_absmean                     : {np.abs(a).mean():.6f}")
  print(f"  token_logdiff_absmax                      : {np.abs(a).max():.4f}")
  print(f"  mult_prob_error  mean(exp|d|)             : {np.exp(np.abs(a)).mean():.6g}")
  for thr in (2, 5, 10):
    n = int((np.abs(a) > thr).sum())
    print(f"  tokens with |log_is| > {thr:2d} nats            : {n} "
          f"({100 * n / a.size:.4f}%, {n / max(tid.max() + 1, 1):.2f} per turn)")
  lo, hi = 0.999, 1.002
  geo = float(np.exp(a.mean()))
  print(f"  in TIS band [{lo}, {hi}]?                 : {'YES' if lo <= geo <= hi else 'NO'}")

  # The discriminating view. `token_outliers_per_seq` (tunix 8cf1035e) reports
  # the count; this reports where in the turn they land, which the metric cannot.
  out = np.abs(a) > 10.0
  print(f"\n  --- outlier attribution ({int(out.sum())} tokens beyond 10 nats) ---")
  if not out.any():
    print("  none. If production shows outliers and this pass does not, the defect is")
    print("  in generation or in conversation reassembly, not in the trainer's scoring.")
    return
  first_in_turn = np.concatenate([[True], tid[1:] != tid[:-1]])
  pos_in_turn = np.zeros(a.size, dtype=np.int64)
  run = 0
  for i in range(a.size):
    run = 0 if first_in_turn[i] else run + 1
    pos_in_turn[i] = run
  print(f"  sign: {int((a[out] < 0).sum())} negative, {int((a[out] > 0).sum())} positive")
  print(f"  distinct turns containing an outlier: {len(np.unique(tid[out]))} of {tid.max() + 1}")
  hist = np.bincount(np.minimum(pos_in_turn[out], 8), minlength=9)
  print(f"  offset from start of its assistant turn (0..7, 8+): {hist.tolist()}")
  print("  worst 15 (turn, offset, token_id, trainer-sampler nats):")
  for i in np.argsort(np.abs(a))[::-1][:15]:
    print(f"    turn {int(tid[i]):3d}  offset {int(pos_in_turn[i]):5d}  "
          f"id {int(toks[i]):6d}  {a[i]:+9.3f}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="qwen3.5-35b-a3b")
  p.add_argument("--load_parameters_path", default=None,
                 help="REQUIRED for a trustworthy number. Random weights give logits ~100x too "
                      "large, and bf16 resolution is relative to magnitude, so quantisation "
                      "noise swamps the path difference.")
  p.add_argument("--layers", type=int, default=8,
                 help="working depth. The checkpoint is loaded at full depth and its stacked "
                      "per-layer arrays are sliced to this many, so weights stay real while the "
                      "model fits. Use a multiple of 4: GDN layers interleave every 4 "
                      "(inhomogeneous_layer_cycle_interval). 0 = full depth.")
  p.add_argument("--seq_len", type=int, default=512)
  p.add_argument("--batch", type=int, default=4,
                 help="must be divisible by the data/fsdp axis (4 on a v5p-8)")
  p.add_argument("--dtype", default="bfloat16")
  p.add_argument("--float32_gate_logits", default="True")
  p.add_argument("--trainer_prefuse", default="False", help="production trainer default is False")
  p.add_argument("--remat_policy", default="full",
                 help="'full' is what production's --remat_policy=decoder maps to; forward-only, so inert")
  p.add_argument("--isolate_moe", action="store_true")
  p.add_argument("--devices", type=int, default=None, help="defaults to jax.device_count()")
  p.add_argument("--sampler_prefuse", default="True",
                 help="False runs the sampler with vllm_rpa attention but MaxText's own unfused "
                      "MoE GMM. moe.py:3469 needs BOTH prefuse and vllm_rpa to reach "
                      "tpu_inference's fused kernel, so this isolates the attention change and "
                      "sidesteps that kernel entirely.")
  p.add_argument("--onehot_moe_permute", type=int, default=0,
                 help="ONEHOT_MOE_PERMUTE_THRESHOLD. 0 = production default, which uses the "
                      "sc_ragged_gather SparseCore kernel. Set above batch*topk (e.g. 1000000) "
                      "to take the onehot permute branch instead (fused_moe_gmm.py:287) if that "
                      "kernel fails on this install -- but note it is then NOT the kernel "
                      "production runs.")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--trajectory_csv", default=None,
                 help="Path to a production trajectory_log_*.csv. Switches the script from "
                      "trainer-vs-sampler-config (two local forward passes) to "
                      "trainer-vs-PRODUCTION-SAMPLER: one trainer forward pass over the real "
                      "tokens, diffed against the per-token logprobs vLLM recorded at rollout "
                      "time. This is the only mode that can see the defect the 35B run is "
                      "actually failing on. Pull one with: gcloud storage cp "
                      "gs://niting-storage-europe-west4/deepswe-logs/"
                      "qwen-35b-deepswe-v5p-256-maxtext-v6/trajectory_log_*.csv .")
  p.add_argument("--trajectory_rows", default="0",
                 help="comma-separated row indices to score, one forward pass each")
  p.add_argument("--trajectory_max_tokens", type=int, default=16384,
                 help="truncate prompt+conversation to this length. 0 = no truncation. "
                      "Rows run 9k-35k tokens; the shortest is usually row 0.")
  p.add_argument("--allow_partial_depth", action="store_true",
                 help="permit --trajectory_csv at reduced depth. Off by default because the "
                      "comparison is against logprobs from the real 40-layer model: a sliced "
                      "model is a DIFFERENT model, so every token disagrees and the result is "
                      "meaningless. Only useful for shaking out plumbing.")
  args = p.parse_args()
  if args.devices is None:
    args.devices = jax.device_count()

  if args.trajectory_csv:
    if args.layers != 0 and not args.allow_partial_depth:
      raise SystemExit(
          f"--trajectory_csv needs --layers 0 (full depth), got {args.layers}.\n"
          "The comparison is against log-probabilities produced by the real 40-layer\n"
          "model. A depth-sliced model is a different model: every token disagrees,\n"
          "the offset is enormous, and the result says nothing about the defect.\n"
          "Pass --allow_partial_depth only to shake out plumbing.")
    if not args.load_parameters_path:
      raise SystemExit(
          "--trajectory_csv needs --load_parameters_path. Random weights cannot be\n"
          "compared against logprobs from a trained checkpoint.")

  print(f"devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
  shim_tpu_inference_envs(args.onehot_moe_permute)
  trainer_cfg = build_trainer_config(args, layers=args.layers)
  kv = getattr(trainer_cfg, "num_kv_heads", None) or trainer_cfg.base_num_kv_heads
  shard = math.gcd(args.devices, kv)
  print(f"num_kv_heads={kv} -> sampler shards {shard}-way on 'expert' and "
        f"replicates {args.devices // shard}-way on 'data' "
        f"(vllm.yml:63 puts kv_heads on both 'model' and 'expert')")
  sampler_cfg = build_sampler_config(args, shard, layers=args.layers)
  for name, c in (("trainer", trainer_cfg), ("sampler", sampler_cfg)):
    print(f"{name:8s} attention={c.attention:12s} prefuse_moe_weights={c.prefuse_moe_weights} "
          f"float32_gate_logits={c.float32_gate_logits} model_call_mode={getattr(c,'model_call_mode','train')}")

  devices = jax.devices()[:args.devices]

  def hbm(tag):
    try:
      st = devices[0].memory_stats()
      used, lim = st.get("bytes_in_use", 0) / 2**30, st.get("bytes_limit", 0) / 2**30
      print(f"    [hbm] {tag:34s} {used:6.1f} / {lim:.1f} GiB in use on device 0")
    except Exception:
      pass

  def make_batch(token_ids_1d):
    """Tile one token stream across the batch axis.

    The batch dimension here is a sharding artifact, not data: `--batch` must
    stay divisible by the data/fsdp axis (4 on a v5p-8), so a single trajectory
    is replicated and only row 0 is read back. Wasteful in activation memory,
    but it keeps the mesh identical to the two-forward-pass mode rather than
    introducing a second sharding path to get wrong.
    """
    t = jnp.asarray(token_ids_1d, dtype=jnp.int32)
    n = t.shape[0]
    return (jnp.broadcast_to(t[None, :], (args.batch, n)),
            jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (args.batch, n)),
            jnp.ones((args.batch, n), dtype=jnp.int32))

  key = jax.random.PRNGKey(args.seed + 1)
  ids = jax.random.randint(key, (args.batch, args.seq_len), 0, trainer_cfg.vocab_size, dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], (args.batch, args.seq_len))
  seg = jnp.ones((args.batch, args.seq_len), dtype=jnp.int32)
  print(f"tokens: batch={args.batch} seq_len={args.seq_len} -> {args.batch * (args.seq_len - 1)} scored positions")

  def fwd(model, cfg, mesh, ids, pos, seg):
    out = model(decoder_input_tokens=ids, decoder_positions=pos,
                decoder_segment_ids=seg, enable_dropout=False)
    # models.py:645 -- with attention in (vllm_rpa, vllm_batched_rpa) the model
    # returns (hidden_state, kv_caches) and leaves the unembedding to vLLM. The
    # trainer path returns logits directly.
    y = out[0] if isinstance(out, tuple) else out
    if y.shape[-1] == cfg.vocab_size:
      return y
    # Apply MaxText's own head (final norm + projection,
    # nnx_decoders.NNXDecoder.apply_output_head, :1536) so both sides are in
    # logit space. Head weights are identical on the two models, so this adds
    # no divergence. The embedding attribute is `token_embedder` on the nnx
    # Transformer (models.py:564); `shared_embedding` is the Linen name.
    assert hasattr(model.decoder, "apply_output_head"), (
        "decoder has no apply_output_head -- pure_nnx_decoder is probably False.")
    with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
      return model.decoder.apply_output_head(
          shared_embedding=model.token_embedder, y=y,
          deterministic=True, model_mode=MODEL_MODE_TRAIN)

  def score(model, cfg, mesh, name, ids=ids, pos=pos, seg=seg):
    """Forward pass, reduced to host-sized results before anything else runs.

    Only the per-token logprobs and a few scalars come back. The full logit
    tensor is batch x seq x vocab -- 2GB at 4x512x248320 -- and holding two of
    them alongside two models is what makes this not fit.
    """
    lg = fwd(model, cfg, mesh, ids, pos, seg)
    finite = jnp.isfinite(lg)
    stats = (jnp.min(jnp.where(finite, lg, jnp.inf)),
             jnp.max(jnp.where(finite, lg, -jnp.inf)),
             jnp.sum(~finite))
    lp = per_token_logprobs(lg, ids)
    lp, (lo, hi, nbad) = jax.device_get((lp, stats))
    print(f"  {name:8s} logits shape={lg.shape} min={float(lo):8.3f} max={float(hi):8.3f} "
          f"non-finite={int(nbad)}")
    if abs(float(hi)) > 100:
      print(f"  {'':8s} !! max |logit| is {float(hi):.0f}. A trained model sits at 15-30. The")
      print(f"  {'':8s}    checkpoint probably did not load, and bf16 quantisation at this")
      print(f"  {'':8s}    magnitude will swamp the measurement (see the note at the end).")
    del lg, finite, stats
    return np.asarray(lp)

  # ---- stage 1: load the checkpoint at full depth, purely to get weights ---
  src = f"checkpoint {args.load_parameters_path}" if args.load_parameters_path else "RANDOM WEIGHTS"
  global _RANDOM_WEIGHTS
  _RANDOM_WEIGHTS = not args.load_parameters_path
  full_cfg = build_trainer_config(args, layers=0)
  print(f"\nloading {src} at full depth ({full_cfg.num_decoder_layers} layers) ...")
  mesh_full = maxtext_utils.get_mesh_from_config(full_cfg, devices)
  if args.load_parameters_path:
    m_full = model_creation_utils.from_pretrained(
        full_cfg, mesh=mesh_full, rng_key=jax.random.PRNGKey(args.seed))
  else:
    print("  !! NO CHECKPOINT -- see the warning at the end; the result will not be usable")
    m_full = model_creation_utils.from_config(
        full_cfg, mesh=mesh_full,
        rngs=maxtext_utils_nnx.create_nnx_rngs(full_cfg, rng_key=jax.random.PRNGKey(args.seed)))
  hbm("after full-depth load")

  host_params = jax.device_get(nnx.state(m_full, nnx.Param))
  del m_full
  gc.collect()
  hbm("after staging to host")

  # ---- stage 2: trainer path at the working depth --------------------------
  print("\nbuilding trainer-path model...")
  mesh_train = maxtext_utils.get_mesh_from_config(trainer_cfg, devices)
  m_train = model_creation_utils.from_config(
      trainer_cfg, mesh=mesh_train,
      rngs=maxtext_utils_nnx.create_nnx_rngs(trainer_cfg, rng_key=jax.random.PRNGKey(args.seed)))
  nnx.update(m_train, copy_weights(host_params, nnx.state(m_train, nnx.Param)))
  hbm("after trainer build")
  if args.trajectory_csv:
    # Trainer vs the production sampler. Only the trainer path runs; the other
    # side of the comparison is `old_logprobs`, recorded by vLLM at rollout.
    for row_idx in [int(x) for x in args.trajectory_rows.split(",") if x.strip()]:
      print(f"\nloading trajectory row {row_idx} from {args.trajectory_csv} ...")
      t_ids, t_mask, t_lp, n_prompt, turn_id = load_trajectory(
          args.trajectory_csv, row_idx, args.trajectory_max_tokens)
      b_ids, b_pos, b_seg = make_batch(t_ids)
      print("forward pass: trainer path over real trajectory tokens...")
      lp = score(m_train, trainer_cfg, mesh_train, "trainer", b_ids, b_pos, b_seg)[0]
      # lp[t] scores token t+1, so conversation token j is lp[n_prompt + j - 1].
      trainer_lp = lp[n_prompt - 1:n_prompt - 1 + len(t_mask)]
      if len(trainer_lp) != len(t_mask):
        raise AssertionError(
            f"alignment slice produced {len(trainer_lp)} scores for {len(t_mask)} "
            f"conversation tokens (prompt={n_prompt}, lp={len(lp)})")
      report_trajectory(trainer_lp - t_lp, t_mask, turn_id, t_ids,
                        f"production sampler vs trainer -- row {row_idx}")
    del m_train
    gc.collect()
    return

  print("forward pass: trainer path...")
  lp_train = score(m_train, trainer_cfg, mesh_train, "trainer")
  del m_train
  gc.collect()
  hbm("after releasing trainer")

  # ---- stage 3: sampler path ----------------------------------------------
  print("\nbuilding sampler-path model...")
  mesh_samp = maxtext_utils.get_mesh_from_config(sampler_cfg, devices)
  m_samp = model_creation_utils.from_config(
      sampler_cfg, mesh=mesh_samp,
      rngs=maxtext_utils_nnx.create_nnx_rngs(sampler_cfg, rng_key=jax.random.PRNGKey(args.seed)))
  nnx.update(m_samp, copy_weights(host_params, nnx.state(m_samp, nnx.Param)))
  del host_params
  gc.collect()
  hbm("after sampler build")

  print("\nforward pass: sampler path...")
  lp_samp = score(m_samp, sampler_cfg, mesh_samp, "sampler")

  assert lp_train.shape == lp_samp.shape, (
      f"logprob shapes differ: {lp_train.shape} vs {lp_samp.shape}")

  report(lp_train - lp_samp,
         "MoE path divergence" + (" (isolate_moe)" if args.isolate_moe else " (production configs)"))


if __name__ == "__main__":
  main()
