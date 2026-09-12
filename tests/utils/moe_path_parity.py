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

Usage
-----
    python3 moe_path_parity.py --layers 8 --seq_len 512
    python3 moe_path_parity.py --layers 8 --model qwen3-4b      # dense control
    python3 moe_path_parity.py --layers 8 --isolate_moe         # see below

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


def build_configs(args):
  """Mirror the two pyconfig.initialize calls in train_maxtext_nb.py:894-970."""
  cfg_dir = os.path.dirname(pyconfig.__file__)
  base_yml = os.path.join(cfg_dir, "post_train", "rl.yml")
  vllm_yml = os.path.join(cfg_dir, "inference", "vllm.yml")

  # Shrink the model so it fits a single host, but keep everything that shapes
  # routing: num_experts, num_experts_per_tok and norm_topk_prob stay at the
  # real values, because expert selection is the thing under test.
  common = [
      "",
      f"model_name={args.model}",
      # The model yml pins base_num_decoder_layers (and use_mrope on qwen3.5),
      # and pyconfig refuses a CLI override of a model-config key unless this is
      # set. Shrinking depth is the whole point here, so allow it.
      "override_model_config=True",
      *([f"base_num_decoder_layers={args.layers}"] if args.layers else []),
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

  trainer_argv = [
      common[0],
      base_yml,
      *common[1:],
      "attention=flash",
      f"prefuse_moe_weights={args.trainer_prefuse}",
      # train_maxtext_nb.py:901-908 maps its --remat_policy=decoder onto MaxText
      # "full"; passing "decoder" through trips the assert in
      # nnx_decoders.get_remat_policy. Remat only affects what the backward pass
      # recomputes, and this harness is forward-only, so it cannot move the
      # numbers either way -- it is set to match production, not because it
      # matters.
      f"remat_policy={args.remat_policy}",
      *([f"load_parameters_path={args.load_parameters_path}"] if args.load_parameters_path else []),
      f"ici_fsdp_parallelism={args.devices}",
      "ici_tensor_parallelism=1",
  ]

  if args.isolate_moe:
    # Same attention on both sides; only the MoE flags differ. Does not hit the
    # fused branch, but always runs.
    sampler_argv = [
        common[0],
        base_yml,
        *common[1:],
        "attention=flash",
        "prefuse_moe_weights=True",
        "model_call_mode=inference",
        "remat_policy=none",
        f"ici_fsdp_parallelism={args.devices}",
        "ici_tensor_parallelism=1",
    ]
  else:
    # The production sampler config.
    sampler_argv = [
        common[0],
        vllm_yml,
        *common[1:],
        "attention=vllm_rpa",
        "prefuse_moe_weights=True",
        "model_call_mode=inference",
        "remat_policy=none",
        "use_mrope=False",
        # Production maps rollout_fsdp onto ici_data_parallelism and relies on
        # tpu-inference sharding within each replica (TP=4 of 128 chips). On a
        # 4-chip host that degenerates to full replication and OOMs, so shard
        # by tensor parallelism instead. Trainer and sampler already differ in
        # sharding in production, so this does not introduce a difference in
        # kind -- but it is a knob, and worth flipping if a result looks odd.
        # Sharding, and why it is fsdp by default:
        #  - data parallelism REPLICATES weights; on 4 chips a 35B needs ~66GB
        #    per chip and OOMs at construction.
        #  - tensor parallelism shards KV heads, and this model has
        #    base_num_kv_heads=2, so TP>2 fails outright
        #    (attentions.py:638 -- heads are atomic under TP).
        #  - fsdp shards parameters without touching KV heads.
        # Matching the trainer's sharding also removes a confound: the only
        # remaining difference is the MoE/attention code path, which is the
        # thing under test. --sampler_tp mirrors production's TP instead, but
        # must divide num_kv_heads.
        *([f"ici_tensor_parallelism={args.sampler_tp}",
           f"ici_data_parallelism={args.devices // args.sampler_tp}",
           f"rollout_tensor_parallelism={args.sampler_tp}",
           f"rollout_data_parallelism={args.devices // args.sampler_tp}"]
          if args.sampler_tp > 1 else
          [f"ici_fsdp_parallelism={args.devices}",
           "ici_tensor_parallelism=1",
           "rollout_tensor_parallelism=1",
           f"rollout_data_parallelism={args.devices}"]),
    ]

  trainer_cfg = pyconfig.initialize(trainer_argv, config_class=types.RLConfig)
  sampler_cfg = pyconfig.initialize(sampler_argv, config_class=types.RLConfig)
  return trainer_cfg, sampler_cfg


def copy_weights(src_state, dst_state):
  """Copy trainer params onto the sampler tree, fusing wi_0/wi_1 -> wi.

  moe.py stores the MoE up-projections as two arrays when prefuse is off and as
  one concatenated array when it is on; moe.py:2148 shows the relation is
  `wi = concatenate([wi_0, wi_1], axis=-1)`. Everything else must match by path
  and shape, and we assert that rather than trusting it.
  """
  src = dict(jax.tree_util.tree_flatten_with_path(src_state)[0])
  dst_leaves, dst_def = jax.tree_util.tree_flatten_with_path(dst_state)

  by_name = {}
  for path, val in src.items():
    by_name.setdefault(jax.tree_util.keystr(path), val)

  out, fused, matched, missing = [], 0, 0, []
  for path, dst_val in dst_leaves:
    key = jax.tree_util.keystr(path)
    if key in by_name:
      v = by_name[key]
      assert v.shape == dst_val.shape, f"shape mismatch at {key}: {v.shape} vs {dst_val.shape}"
      out.append(v)
      matched += 1
    elif key.endswith("wi.value") or key.endswith("wi"):
      w0 = by_name.get(key.replace("wi", "wi_0"))
      w1 = by_name.get(key.replace("wi", "wi_1"))
      assert w0 is not None and w1 is not None, f"cannot fuse {key}: wi_0/wi_1 not found"
      v = jnp.concatenate([w0, w1], axis=-1)
      assert v.shape == dst_val.shape, f"fused shape mismatch at {key}: {v.shape} vs {dst_val.shape}"
      out.append(v)
      fused += 1
    else:
      missing.append(key)
      out.append(dst_val)

  print(f"  weight copy: {matched} matched, {fused} fused (wi_0+wi_1 -> wi), {len(missing)} unmatched")
  if missing:
    print("  UNMATCHED (left at the sampler's own init -- investigate before trusting the result):")
    for k in missing[:20]:
      print("    ", k)
  return jax.tree_util.tree_unflatten(dst_def, out)


def per_token_logprobs(logits, ids):
  """log p(ids[t+1] | ids[:t+1]) -- the same quantity the RL loss compares."""
  lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  tgt = ids[:, 1:]
  return jnp.take_along_axis(lp[:, :-1], tgt[..., None], axis=-1)[..., 0]


def diagnose_logits(name, logits, ids):
  """Non-finite logits poison every downstream statistic, so surface them."""
  lg = np.asarray(logits)
  bad = ~np.isfinite(lg)
  print(f"  {name:8s} logits shape={lg.shape} dtype={lg.dtype} "
        f"non-finite={int(bad.sum())} ({100*bad.mean():.4f}%) "
        f"min={np.nanmin(lg[np.isfinite(lg)]) if np.isfinite(lg).any() else float('nan'):.3f} "
        f"max={np.nanmax(lg[np.isfinite(lg)]) if np.isfinite(lg).any() else float('nan'):.3f}")
  if bad.any():
    rows = np.unique(np.argwhere(bad)[:, 1])
    print(f"           non-finite at {len(rows)} distinct positions, first 20: {rows[:20].tolist()}")


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


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="qwen3.5-35b-a3b")
  p.add_argument("--load_parameters_path", default=None,
                 help="REQUIRED for a trustworthy number. Random weights give logits ~100x too "
                      "large, and bf16 resolution is relative to magnitude, so quantisation "
                      "noise swamps the path difference.")
  p.add_argument("--layers", type=int, default=0, help="0 = use the model's real depth (required when loading a checkpoint); otherwise a multiple of 4, since GDN layers interleave every 4")
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
  p.add_argument("--sampler_tp", type=int, default=None,
                 help="1 (default) shards the sampler with fsdp, matching the trainer. Set >1 to "
                      "mirror production TP, but it must divide num_kv_heads (2 on Qwen3.5).")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  if args.devices is None:
    args.devices = jax.device_count()
  if args.sampler_tp is None:
    args.sampler_tp = 1  # fsdp; see build_configs

  print(f"devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
  trainer_cfg, sampler_cfg = build_configs(args)
  for name, c in (("trainer", trainer_cfg), ("sampler", sampler_cfg)):
    print(f"{name:8s} attention={c.attention:12s} prefuse_moe_weights={c.prefuse_moe_weights} "
          f"float32_gate_logits={c.float32_gate_logits} model_call_mode={getattr(c,'model_call_mode','train')}")

  devices = jax.devices()
  print("\nbuilding trainer-path model (random weights)...")
  mesh_train = maxtext_utils.get_mesh_from_config(trainer_cfg, devices)
  global _RANDOM_WEIGHTS
  _RANDOM_WEIGHTS = not args.load_parameters_path
  if args.load_parameters_path:
    m_train = model_creation_utils.from_pretrained(
        trainer_cfg, mesh=mesh_train, rng_key=jax.random.PRNGKey(args.seed))
  else:
    print("  !! NO CHECKPOINT -- see the warning printed at the end; the result will not be usable")
    rngs = maxtext_utils_nnx.create_nnx_rngs(trainer_cfg, rng_key=jax.random.PRNGKey(args.seed))
    m_train = model_creation_utils.from_config(trainer_cfg, mesh=mesh_train, rngs=rngs)

  print("building sampler-path model...")
  mesh_samp = maxtext_utils.get_mesh_from_config(sampler_cfg, devices)
  rngs2 = maxtext_utils_nnx.create_nnx_rngs(sampler_cfg, rng_key=jax.random.PRNGKey(args.seed))
  m_samp = model_creation_utils.from_config(sampler_cfg, mesh=mesh_samp, rngs=rngs2)

  print("copying weights trainer -> sampler so the ONLY difference is the code path...")
  st_train = nnx.state(m_train, nnx.Param)
  st_samp = nnx.state(m_samp, nnx.Param)
  nnx.update(m_samp, copy_weights(st_train, st_samp))

  # Deterministic pseudo-text. Real tokens would be better but are not needed:
  # the divergence is a property of the arithmetic, not the content.
  # Batch must be divisible by the data/fsdp axis: tpu_flash_attention asserts
  # query.shape[0] % devices_in_data_fsdp == 0 (attention_op.py:1849). With a
  # 4-device mesh that means batch >= 4.
  key = jax.random.PRNGKey(args.seed + 1)
  ids = jax.random.randint(key, (args.batch, args.seq_len), 0, trainer_cfg.vocab_size, dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], (args.batch, args.seq_len))
  seg = jnp.ones((args.batch, args.seq_len), dtype=jnp.int32)
  print(f"\ntokens: batch={args.batch} seq_len={args.seq_len} -> {args.batch * (args.seq_len - 1)} scored positions")

  def fwd(model, cfg, mesh):
    out = model(decoder_input_tokens=ids, decoder_positions=pos,
                decoder_segment_ids=seg, enable_dropout=False)
    # model_call_mode=inference returns (hidden_states, kv_cache); the trainer
    # path returns logits directly.
    y = out[0] if isinstance(out, tuple) else out
    if y.shape[-1] == cfg.vocab_size:
      return y
    # models.py:645 -- with attention in (vllm_rpa, vllm_batched_rpa) the model
    # returns (hidden_state, kv_caches) and leaves the unembedding to vLLM.
    # Apply MaxText's own head (final norm + projection,
    # nnx_decoders.NNXDecoder.apply_output_head, :1536) so both sides are
    # compared in logit space. Head weights are identical on the two models --
    # they came through the weight copy -- so this adds no divergence.
    # The embedding attribute is `token_embedder` on the nnx Transformer;
    # `shared_embedding` is the Linen variant's name (models.py:564 passes
    # shared_embedding=self.token_embedder).
    assert hasattr(model.decoder, "apply_output_head"), (
        "decoder has no apply_output_head -- pure_nnx_decoder is probably False, "
        "so decoder is a ToNNX wrapper. Set pure_nnx_decoder=True.")
    with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
      return model.decoder.apply_output_head(
          shared_embedding=model.token_embedder,
          y=y,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

  print("\nforward pass: trainer path...")
  lg_train = fwd(m_train, trainer_cfg, mesh_train)
  print("forward pass: sampler path...")
  lg_samp = fwd(m_samp, sampler_cfg, mesh_samp)

  print("\nlogit diagnostics:")
  assert lg_train.shape == lg_samp.shape, (
      f"logit shapes differ: trainer {lg_train.shape} vs sampler {lg_samp.shape}. "
      "The two paths are not returning comparable tensors; fix that before reading any statistic.")
  diagnose_logits("trainer", lg_train, ids)
  diagnose_logits("sampler", lg_samp, ids)

  lp_train = per_token_logprobs(lg_train, ids)
  lp_samp = per_token_logprobs(lg_samp, ids)

  report(lp_train - lp_samp,
         "MoE path divergence" + (" (isolate_moe)" if args.isolate_moe else " (production configs)"))


if __name__ == "__main__":
  main()
