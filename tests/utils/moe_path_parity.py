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

os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")

from flax import nnx  # noqa: E402
from maxtext.configs import pyconfig, types  # noqa: E402
from maxtext.utils import model_creation_utils  # noqa: E402
from maxtext.utils import maxtext_utils_nnx  # noqa: E402


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
      f"base_num_decoder_layers={args.layers}",
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
      "remat_policy=decoder",
      "ici_fsdp_parallelism=4",
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
        "ici_fsdp_parallelism=4",
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
        "ici_data_parallelism=4",
        "ici_tensor_parallelism=1",
        "rollout_data_parallelism=4",
        "rollout_tensor_parallelism=1",
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


def report(log_is, label=""):
  """Same statistics the production loss emits, so numbers are comparable."""
  a = np.asarray(log_is).ravel()
  geo = float(np.exp(a.mean()))
  print(f"\n===== {label} =====")
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


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="qwen3.5-35b-a3b")
  p.add_argument("--layers", type=int, default=8, help="multiple of 4: the model interleaves GDN layers every 4")
  p.add_argument("--seq_len", type=int, default=512)
  p.add_argument("--dtype", default="bfloat16")
  p.add_argument("--float32_gate_logits", default="True")
  p.add_argument("--trainer_prefuse", default="False", help="production trainer default is False")
  p.add_argument("--isolate_moe", action="store_true")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  print(f"devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
  trainer_cfg, sampler_cfg = build_configs(args)
  for name, c in (("trainer", trainer_cfg), ("sampler", sampler_cfg)):
    print(f"{name:8s} attention={c.attention:12s} prefuse_moe_weights={c.prefuse_moe_weights} "
          f"float32_gate_logits={c.float32_gate_logits} model_call_mode={getattr(c,'model_call_mode','train')}")

  devices = jax.devices()
  print("\nbuilding trainer-path model (random weights)...")
  rngs = maxtext_utils_nnx.create_nnx_rngs(trainer_cfg, rng_key=jax.random.PRNGKey(args.seed))
  m_train = model_creation_utils.from_config(trainer_cfg, devices=devices, rngs=rngs)

  print("building sampler-path model...")
  rngs2 = maxtext_utils_nnx.create_nnx_rngs(sampler_cfg, rng_key=jax.random.PRNGKey(args.seed))
  m_samp = model_creation_utils.from_config(sampler_cfg, devices=devices, rngs=rngs2)

  print("copying weights trainer -> sampler so the ONLY difference is the code path...")
  st_train = nnx.state(m_train, nnx.Param)
  st_samp = nnx.state(m_samp, nnx.Param)
  nnx.update(m_samp, copy_weights(st_train, st_samp))

  # Deterministic pseudo-text. Real tokens would be better but are not needed:
  # the divergence is a property of the arithmetic, not the content.
  key = jax.random.PRNGKey(args.seed + 1)
  ids = jax.random.randint(key, (1, args.seq_len), 0, trainer_cfg.vocab_size, dtype=jnp.int32)
  pos = jnp.arange(args.seq_len, dtype=jnp.int32)[None, :]
  seg = jnp.ones((1, args.seq_len), dtype=jnp.int32)

  def fwd(model):
    return model(decoder_input_tokens=ids, decoder_positions=pos,
                 decoder_segment_ids=seg, enable_dropout=False)

  print("\nforward pass: trainer path...")
  lp_train = per_token_logprobs(fwd(m_train), ids)
  print("forward pass: sampler path...")
  lp_samp = per_token_logprobs(fwd(m_samp), ids)

  report(lp_train - lp_samp,
         "MoE path divergence" + (" (isolate_moe)" if args.isolate_moe else " (production configs)"))


if __name__ == "__main__":
  main()
