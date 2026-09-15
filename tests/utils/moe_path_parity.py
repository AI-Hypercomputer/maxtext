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

  READ THIS BEFORE CITING 0.000531. This mode is far blinder than it looks, and
  the number covers the MoE weight layout and the config plumbing and very
  little else.

  * It calls `model(...)` with no `model_mode`, so `Transformer.__call__`
    defaults to MODEL_MODE_TRAIN (models.py:143). `attentions.py:1332` gates the
    vLLM branch on `model_mode != MODEL_MODE_TRAIN`, so on the "sampler" side
    `attention=vllm_rpa` silently falls through to `apply_attention_dot`
    (attention_op.py:1523). **The ragged-paged-attention kernel never runs**, so
    nothing here sees RPA v3's bf16 online-softmax accumulators.
  * It passes no `kv_cache` and no `attention_metadata`, so `use_paged_state` is
    False (qwen3.py:661-668) and **all 30 GatedDeltaNet layers run MaxText's own
    bf16 kernel on both sides**. tpu-inference's f32 GDN kernel never runs
    either.

  So this mode cannot see attention, GDN, paged KV, prefix caching or decode.
  Use `--trajectory_csv` for anything that touches those.

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

What the trajectory mode predicted, and what it measured
--------------------------------------------------------
The prediction, from the production metrics (`token_weight_mean` 1.000235,
`seq_geomean` 0.856, `token_logdiff_absmean` 0.229), was a bulk of sigma ~0.10
nats plus a one-sided negative tail of 0.4-0.6% of scored tokens at 26+ nats --
about 32 per sequence against a 30-turn agent budget -- and that the defect was
in the trainer's scoring of the real token stream.

**Measured (Sep 15, rows 0-2 of trajectory_log_1789165722.csv, full depth,
real checkpoint):**

    row 0  mean signed -0.052252  seq_geomean 0.949090  absmean 0.115  absmax 6.01
    row 1  mean signed -0.037470  seq_geomean 0.963223  absmean 0.091  absmax 5.30
    row 2  mean signed -0.016897  seq_geomean 0.983245  absmean 0.069  absmax 4.13
    production, same run       :  seq_geomean 0.856     absmean 0.229  absmax >=32.5

**Zero tokens beyond 10 nats in any row.** So on identical inputs -- the same
token stream, the same recorded `old_logprobs` -- a clean single-host trainer
reproduces only ~23% of production's centre offset and none of its outlier
tail. The bulk of the production defect is in the production trainer's own
forward pass, not in the data and not in the sampler's recorded numbers.

The per-turn table this mode prints is flat in turn index (row 0: first half
-0.067 -> second half -0.029), which rules out mechanisms that accumulate with
conversation length.

A/Bs run against that baseline, all single-variable, all on the same three rows:

    --trajectory_pad_mode production   -0.0523 -> -0.0653   (row 0; 25% worse)
    --trajectory_repair_eot            -0.0523 -> -0.0487   (7%, ~0% on rows 1-2)
    MAXTEXT_GDN_F32=1 (gate+core f32)  -0.0523 -> -0.0520   (~0%)
    logits_dot_in_fp32=true (trainer)  -0.0523 -> -0.0512   (~2%)

For reference, the rest of the ladder on the same v5p-8: a real vLLM prefill
through the adapter scores -0.00018 (7/8 sequences in band) and a real vLLM
decode scores -0.00206 (2/8 in band). The step that costs 18x is going from a
synthetic single-turn generation to a real 30-turn agentic trajectory.

`--isolate_moe` keeps attention identical on both sides and varies only the MoE
flags. That does NOT reproduce the production sampler path (the fused branch
requires vllm_rpa attention), but it runs without any vLLM machinery, so it is
the fallback if the vllm_rpa path will not execute standalone.

Running it on a v5p-8
---------------------
Use a jax >= 0.11 environment: MaxText imports
`jax.experimental.xla_metadata.must_fuse_call`, which does not exist on 0.10.
Keep `--trajectory_max_tokens` at or below 13312 -- the `[batch, seq, 248320]`
float32 logits are what bound this, and they already sit at 90 of 95.7 GiB per
chip at 13312 with the default `--batch 4`. 16384 OOMs.

Two shape constraints that are easy to trip:
  * the padded length must be a multiple of 512, not 128, or splash attention
    raises `q_block_size=512 should divide q_seq_len=...`;
  * `--batch` must stay divisible by the data/fsdp axis (4 on a v5p-8).

`--trainer_tp 2 --trainer_fsdp 2` mirrors production's mesh
(`fsdp: 64, tensor: 2`, zf0hn8pe_output.log:3775) but currently fails in
`qwen3.py:1364` with
`add got incompatible shapes: (4, S, 1024), (4, S, 2048)` -- the routed-expert
output comes back at emb_dim/tensor while the shared-expert output is full
width. Production does not hit this on its 128-device mesh; the combine is not
tensor-parallel-clean here.
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
      # Production's trainer mesh is NOT fsdp-only. zf0hn8pe_output.log:3775:
      #   Mesh('data': 1, 'fsdp': 64, 'tensor': 2, 'expert': 1, ...)
      # and post_train/rl.yml maps 'norm', 'activation_embed' and 'vocab' onto
      # the 'tensor' axis, so with tensor=2 the decoder norm reduction, the
      # unembedding contraction and the MoE MLP are all split across devices
      # and reduced through a different (bf16-typed) tree. A single-host
      # fsdp-only run cannot see that.
      *([kv for kv in args.extra_trainer_argv.split(",") if kv.strip()]
        if getattr(args, "extra_trainer_argv", "") else []),
      f"ici_fsdp_parallelism={args.trainer_fsdp or args.devices}",
      f"ici_tensor_parallelism={args.trainer_tp}",
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

  # The production sampler does NOT run the same MoE shape as the trainer.
  # `maxtext_vllm_adapter/adapter.py:184` injects `padded_base_moe_mlp_dim` on
  # the vLLM path only, and `moe.py:627` then uses it in place of
  # `intermediate_dim` when sizing the expert weights. On the 35B this is a 2x
  # inflation, logged by production at WARNING on every run:
  #
  #   Padding moe_intermediate_size from 512 to 1024 to match MLP MoE
  #   requirements (moe_mlp_tp_size=4, 2*num_lanes=256).
  #
  # This harness builds its configs through `pyconfig.initialize` directly and
  # never goes through the adapter, so before this flag both sides were 512 wide
  # and the padding was silently excluded from the comparison. That is why the
  # harness reported 0.000531 nats against production's 0.155: it was measuring
  # two unpadded models, not the pair production actually runs.
  #
  # Read the value out of the container log's warning line rather than trusting
  # the default -- it depends on `moe_mlp_tp_size`, which is a function of the
  # deployment's sharding, not of the model.
  pad = []
  if args.sampler_padded_moe_mlp_dim:
    pad = [f"padded_base_moe_mlp_dim={args.sampler_padded_moe_mlp_dim}"]

  return pyconfig.initialize([
      "", os.path.join(cfg_dir, "inference", "vllm.yml"), *_common_argv(args, layers),
      "attention=vllm_rpa",
      f"prefuse_moe_weights={args.sampler_prefuse}",
      "model_call_mode=inference",
      "remat_policy=none",
      "use_mrope=False",
      *pad,
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
      return v, False, False
    if v.ndim != len(want):
      raise AssertionError(f"rank mismatch at {key}: {v.shape} vs {want}")
    bad = [i for i, (a_, b_) in enumerate(zip(v.shape, want)) if a_ != b_]
    padded = False
    for i in bad:
      if v.shape[i] < want[i]:
        # The sampler's MoE is wider than the checkpoint when
        # padded_base_moe_mlp_dim is set: production inflates the expert
        # up-projections from 512 to 1024 on the vLLM path only. Zero is the
        # only fill that leaves the padding numerically inert -- SiLU(0) = 0 and
        # the gated branch multiplies, so zero-filled channels contribute
        # nothing downstream, and `wo` is unpadded (MaxText 450a65581) so it
        # never reads them.
        #
        # ASSUMPTION, and the main caveat on any result from this path: that
        # production also writes zeros there. If it instead leaves the padded
        # region at its init values, production's sampler is computing with
        # garbage in those channels and this harness will NOT reproduce it. A
        # near-zero divergence here therefore means "padding is inert when
        # zero-filled", not "padding is innocent" -- the follow-up is to dump
        # what the production weight loader actually leaves in [512:1024].
        pad = [(0, 0)] * v.ndim
        pad[i] = (0, want[i] - v.shape[i])
        v = np.pad(v, pad)
        padded = True
    for i in bad:
      if v.shape[i] > want[i]:
        v = np.take(v, range(want[i]), axis=i)
    return v, True, padded

  out, fused, sliced, matched, missing, padded_n = [], 0, 0, 0, [], 0
  for pth, dst_val in dst_leaves:
    key = jax.tree_util.keystr(pth)
    want = tuple(np.asarray(dst_val).shape)
    if key in by_name:
      v, did, pad = conform(by_name[key], want, key)
      matched += 1
      sliced += did
      padded_n += pad
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
      v, did, pad = conform(np.concatenate([np.asarray(w0), np.asarray(w1)], axis=-1), want, key)
      fused += 1
      sliced += did
      padded_n += pad
    else:
      missing.append(key)
      out.append(dst_val)
      continue
    out.append(jnp.asarray(v))

  print(f"  weight copy: {matched} matched, {fused} fused (wi_0+wi_1 -> wi), "
        f"{sliced} reshaped, {padded_n} zero-padded (MoE width), "
        f"{len(missing)} unmatched")
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


def per_token_logprobs_chunked(logits, ids, chunk=512):
  """`per_token_logprobs` without the full-sequence float32 copy.

  `log_softmax(logits.astype(float32))` materialises `[B, T, V]` in fp32. At the
  trajectory-mode shapes -- T ~7k against a 248,320-token vocabulary -- that is
  6.8 GB per device on top of the bf16 logits that already exist, and it is pure
  transient: only one gathered value per position survives. Slicing the sequence
  axis keeps the fp32 working set at `chunk` rows and changes no arithmetic,
  since log_softmax reduces over the vocabulary axis only.
  """
  out = []
  t_total = logits.shape[1] - 1
  for s in range(0, t_total, chunk):
    e = min(s + chunk, t_total)
    lp = jax.nn.log_softmax(logits[:, s:e].astype(jnp.float32), axis=-1)
    tgt = ids[:, s + 1:e + 1]
    out.append(np.asarray(jax.device_get(
        jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0])))
    del lp
  return np.concatenate(out, axis=1)


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


def load_trajectory(csv_path, row_idx, max_tokens, keep_prompt_padding=False,
                    repair_eot=False, eot_id=248046):
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

  if keep_prompt_padding:
    # Production does NOT strip this. `agentic_grpo_learner.py:719-726` left-pads
    # every prompt to `max_prompt_length` (4096) with `pad_id` and right-pads
    # every completion to `max_response_length`, then `common.process_ids`
    # (`tunix/rl/common.py:291-307`) rebuilds positions and segment ids from a
    # `tokens != pad_id` mask. Reproducing that is the only way to tell whether
    # the padded forward pass scores the same as the clean one.
    print(f"  keeping {int((prompt == PAD_ID).sum())} leading pad tokens "
          f"(production padding emulation)")
  else:
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

  if repair_eot:
    # Production sets rollout_vllm_sampling_kwargs["stop"] = ["</function>", ...]
    # (train_maxtext_nb.py:1229-1235). A stop *string* ends the generation before
    # the model ever emits <|im_end|>, so `rollout_output.tokens[0]` -- which is
    # what the trainer's conversation stream is built from
    # (trajectory_collect_engine.py:664-668, and Qwen has no
    # `update_assistant_end_tokens` override) -- has no terminator.
    #
    # The SAMPLER's prompt for the next turn does have one:
    # `QwenChatTemplateParser._parse_assistant` returns
    # `assistant_token + content + eot_token` unconditionally (parser.py:140-141),
    # and every turn's prompt is a full re-render of the message list
    # (agentic_rl_learner.py:437-442).
    #
    # So the two engines condition on streams that differ by one <|im_end|> per
    # assistant turn. This reinserts it -- mask 0, logprob 0, so the set of
    # scored tokens is unchanged -- which makes the trainer's context match what
    # the sampler actually saw. vllm_sampler.py:447-453 records that this exact
    # omission "produced 30+ nat sampler-trainer logp diffs" once before.
    on_ = mask > 0
    ed = np.diff(np.concatenate([[0], on_.astype(np.int8), [0]]))
    ends = np.where(ed == -1)[0]
    ins = [int(e) for e in ends if conv[e - 1] != eot_id]
    if ins:
      conv = np.insert(conv, ins, eot_id)
      mask = np.insert(mask, ins, 0)
      lp = np.insert(lp, ins, 0.0)
    print(f"  repaired {len(ins)} of {len(ends)} assistant turns with a missing "
          f"<|im_end|> ({eot_id})")

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

  # Per-turn trend. If the sampler and the trainer condition on different
  # contexts -- tunix rebuilds the sampler's prompt from the whole message list
  # every turn (`agentic_rl_learner.py:437-441`) while the trainer's stream is
  # the incremental concatenation -- the gap between them GROWS with turn index,
  # because each turn adds more re-rendered history. A per-token numerics floor
  # is flat in turn index instead. This is the cheapest discriminator available.
  nt = int(tid.max()) + 1
  print(f"\n  --- mean signed log-ratio by assistant turn ({nt} turns) ---")
  rows = []
  for t in range(nt):
    sel = tid == t
    if not sel.any():
      continue
    rows.append((t, int(sel.sum()), float(a[sel].mean()), float(np.abs(a[sel]).mean())))
  print(f"  {'turn':>4} {'n':>6} {'mean_signed':>12} {'absmean':>10}")
  for t, n, m, am in rows:
    if t < 3 or t % 5 == 0 or t == nt - 1:
      print(f"  {t:>4} {n:>6} {m:>+12.5f} {am:>10.5f}")
  if len(rows) >= 6:
    h = len(rows) // 2
    first = np.average([r[2] for r in rows[:h]], weights=[r[1] for r in rows[:h]])
    last = np.average([r[2] for r in rows[h:]], weights=[r[1] for r in rows[h:]])
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    ms = np.array([r[2] for r in rows], dtype=np.float64)
    slope = np.polyfit(ts, ms, 1)[0]
    print(f"  first half {first:+.5f} -> second half {last:+.5f}   "
          f"(ratio {last / first if first else float('nan'):.2f}x)")
    print(f"  OLS slope over turn index: {slope:+.6f} nats/turn")

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
  p.add_argument("--sampler_padded_moe_mlp_dim", type=int, default=1024,
                 help="padded_base_moe_mlp_dim for the SAMPLER config only, mirroring what "
                      "maxtext_vllm_adapter injects in production. 1024 is what the 35B logs "
                      "on a v5p-256 at moe_mlp_tp_size=4 ('Padding moe_intermediate_size from "
                      "512 to 1024'); confirm against the container log before trusting it, "
                      "since it depends on the deployment's sharding. 0 disables the padding "
                      "and reproduces the pre-Sep-14 harness behaviour, which measured two "
                      "unpadded models and so could not see this asymmetry at all.")
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
  p.add_argument("--extra_trainer_argv", default="",
                 help="Comma-separated extra pyconfig overrides for the TRAINER config only, "
                      "e.g. 'logits_dot_in_fp32=true'. Applied before the sharding flags.")
  p.add_argument("--trainer_tp", type=int, default=1,
                 help="ici_tensor_parallelism for the trainer config. Production used 2 "
                      "(zf0hn8pe_output.log:3775).")
  p.add_argument("--trainer_fsdp", type=int, default=0,
                 help="ici_fsdp_parallelism for the trainer; 0 = all devices. Set "
                      "--trainer_fsdp 2 --trainer_tp 2 on a v5p-8 to mirror production's split.")
  p.add_argument("--trajectory_repair_eot", action="store_true",
                 help="Reinsert the <|im_end|> that the `stop=[\"</function>\"]` sampling "
                      "config prevents vLLM from emitting, so the trainer's context matches "
                      "the one the sampler re-rendered for the next turn. Inserted with "
                      "mask 0 / logprob 0, so the scored token set is unchanged.")
  p.add_argument("--eot_id", type=int, default=248046, help="<|im_end|> for Qwen3.5")
  p.add_argument("--trajectory_pad_mode", default="stripped",
                 choices=["stripped", "production"],
                 help="stripped: drop the prompt's left padding and use arange positions "
                      "(the clean measurement). production: keep the 4096-wide left-padded "
                      "prompt and derive positions/segment_ids from the non-pad mask, which "
                      "is exactly what the production trainer feeds the model.")
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
          f"float32_gate_logits={c.float32_gate_logits} model_call_mode={getattr(c,'model_call_mode','train')} "
          f"padded_base_moe_mlp_dim={getattr(c, 'padded_base_moe_mlp_dim', None)}")
  if getattr(sampler_cfg, "padded_base_moe_mlp_dim", None) == getattr(
      trainer_cfg, "padded_base_moe_mlp_dim", None):
    print("  NOTE: both sides have the same MoE width, so this run does NOT reproduce\n"
          "        production's padded-sampler asymmetry. Pass --sampler_padded_moe_mlp_dim\n"
          "        with the value from the container log to test it.")

  devices = jax.devices()[:args.devices]

  def hbm(tag):
    try:
      st = devices[0].memory_stats()
      used, lim = st.get("bytes_in_use", 0) / 2**30, st.get("bytes_limit", 0) / 2**30
      print(f"    [hbm] {tag:34s} {used:6.1f} / {lim:.1f} GiB in use on device 0")
    except Exception:
      pass

  def make_batch(token_ids_1d, total_len):
    """Tile one token stream across the batch axis, padded to `total_len`.

    The batch dimension is a sharding artifact, not data: `--batch` must stay
    divisible by the data/fsdp axis (4 on a v5p-8), so a single trajectory is
    replicated and only row 0 is read back. Under `ici_fsdp_parallelism=4` the
    batch axis is what shards, so each device holds one sequence and the tiling
    costs no extra per-device memory.

    Padding carries `decoder_segment_ids = 0`, which is MaxText's "not part of
    any sequence" marker, so attention cannot read across the boundary.
    `max_target_length` is a single config value, so scoring several rows of
    different lengths in one model requires padding them to a common length
    rather than rebuilding the model per row.
    """
    t = np.asarray(token_ids_1d, dtype=np.int32)
    n = t.shape[0]
    assert n <= total_len, f"{n} tokens exceeds max_target_length {total_len}"
    padded = np.full(total_len, PAD_ID, dtype=np.int32)
    padded[:n] = t
    tile = lambda x: jnp.asarray(np.broadcast_to(x[None, :], (args.batch, total_len)))
    if args.trajectory_pad_mode == "production":
      # tunix/rl/common.py:291-307 verbatim: the mask is `tokens != pad_id` over
      # the whole [left-padded prompt | right-padded completion] buffer,
      # positions are cumsum(mask)-1, and segment_ids IS that 0/1 mask.
      mask = (padded != PAD_ID)
      cum = np.cumsum(mask)
      pos = (cum - (cum >= 1)).astype(np.int32)
      seg = mask.astype(np.int32)
      return tile(padded), tile(pos), tile(seg)
    seg = np.zeros(total_len, dtype=np.int32)
    seg[:n] = 1
    return (tile(padded),
            tile(np.arange(total_len, dtype=np.int32)),
            tile(seg))

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

  if args.trajectory_csv:
    # Trainer vs the production sampler. Only one model exists here -- the other
    # side of the comparison is `old_logprobs`, recorded by vLLM at rollout --
    # so this path skips the whole stage-1/stage-2 dance below.
    #
    # It must, for memory. The two-model flow builds the second model with
    # `from_config` (random init, then `copy_weights`), because the comparison
    # there demands bit-identical weights in two differently-configured models
    # and there is no other way to guarantee that. At full depth `from_config`
    # OOMs on a v5p-8: `_create_scanned_layers` materialises 40 layers and
    # stacks them before any sharding constraint applies, and the moveaxis of
    # one stacked MoE tensor asks for 5.00 GB against 4.80 GB free. Observed,
    # not predicted. `from_pretrained` loads the *same* config sharded straight
    # off the checkpoint and peaks at 16.1 GiB of 95.7 GiB. With one model we
    # can just use it.
    # Load the data first. `max_target_length` is baked into the config at
    # build time from --seq_len (default 512); a 7k-token trajectory against a
    # 512-token config is a shape failure, not a truncation, so the real length
    # has to be known before the model is constructed.
    rows_wanted = [int(x) for x in args.trajectory_rows.split(",") if x.strip()]
    loaded = []
    for row_idx in rows_wanted:
      print(f"\nloading trajectory row {row_idx} from {args.trajectory_csv} ...")
      loaded.append((row_idx, load_trajectory(
          args.trajectory_csv, row_idx, args.trajectory_max_tokens,
          keep_prompt_padding=(args.trajectory_pad_mode == "production"),
          repair_eot=args.trajectory_repair_eot, eot_id=args.eot_id)))
    total_len = max(len(t[1][0]) for t in loaded)
    # splash attention (attention=flash) requires q_block_size=512 to divide the
    # padded length, not merely 128; a 13,184-token row raised
    # "q_block_size=512 should divide q_seq_len=13184".
    total_len = int(np.ceil(total_len / 512) * 512)
    args.seq_len = total_len
    print(f"\nmax_target_length set to {total_len} from the data "
          f"(longest requested row is {max(len(t[1][0]) for t in loaded)} tokens)")

    print(f"loading {args.load_parameters_path} at full depth ...")
    traj_cfg = build_trainer_config(args, layers=0)
    traj_mesh = maxtext_utils.get_mesh_from_config(traj_cfg, devices)
    m_traj = model_creation_utils.from_pretrained(
        traj_cfg, mesh=traj_mesh, rng_key=jax.random.PRNGKey(args.seed))
    hbm("after trajectory-mode load")

    for row_idx, (t_ids, t_mask, t_lp, n_prompt, turn_id) in loaded:
      b_ids, b_pos, b_seg = make_batch(t_ids, total_len)
      print(f"\nforward pass over {len(t_ids)} real tokens padded to {total_len} "
            f"(batch {args.batch}, tiled; fsdp shards it one sequence per device)...")
      lg = fwd(m_traj, traj_cfg, traj_mesh, b_ids, b_pos, b_seg)
      hbm("after forward, logits live")
      lp_all = per_token_logprobs_chunked(lg, b_ids)
      del lg
      gc.collect()
      # lp[t] scores token t+1, so conversation token j is lp[n_prompt + j - 1].
      lp = lp_all[0]
      trainer_lp = lp[n_prompt - 1:n_prompt - 1 + len(t_mask)]
      if len(trainer_lp) != len(t_mask):
        raise AssertionError(
            f"alignment slice produced {len(trainer_lp)} scores for {len(t_mask)} "
            f"conversation tokens (prompt={n_prompt}, lp={len(lp)})")
      report_trajectory(trainer_lp - t_lp, t_mask, turn_id, t_ids,
                        f"production sampler vs trainer -- row {row_idx}")
    del m_traj
    gc.collect()
    return

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
