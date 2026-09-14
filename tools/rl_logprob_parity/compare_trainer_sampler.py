#!/usr/bin/env python3
"""Qwen3.5-35B-A3B bf16: MaxText trainer vs vLLM sampler logprob parity, end to end in one script.

Four stages, in order:
  1. tokenize  (CPU)  real text -> HF tokenizer -> [B, S] token ids, saved once and reused by both sides
  2. sampler   (TPU)  real vLLM engine prompt_logprobs; --sampler adapter (MaxText-in-vLLM) or native (tpu-inference)
  3. trainer   (TPU)  MaxText nnx model, MODEL_MODE_TRAIN, full-model teacher-forced logprobs
  4. compare   (CPU)  per-token band stats + per-sequence seq-mask-tis is_oob_ratio

Stages 2 and 3 each need the whole TPU and conflicting process-level env (MODEL_IMPL_TYPE, LIBTPU_INIT_ARGS),
so `--stage all` re-execs this file as a subprocess per TPU stage and then compares in-process. Run a single
stage with `--stage sampler|trainer|compare` to drive them by hand.

Unlike the per-row scripts in this directory, both sides read their tokens from the same tokens.npz: the
trainer must never re-tokenize independently, or the two sides score different text.

Usage:
    python compare_trainer_sampler.py --stage all --sampler adapter
    python compare_trainer_sampler.py --stage compare --out-dir /path/to/existing/npz   # re-score saved runs

Paths default to autodetection (see resolve_paths) and can be overridden with --out-dir / --hf-home /
--maxtext-root or the env vars OUT_DIR / HF_HOME / MAXTEXT_ROOT.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

MODEL_HF = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT = "qwen3.5-35b-a3b"
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
B, SEQ_LEN = 8, 512
# TIS acceptance band; matches truncated_importance_sampling_ratio_min / _ratio in the RL trainer.
RATIO_MIN, RATIO_MAX = 0.999, 1.002

t0 = time.time()


def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- paths / env


def resolve_paths(args):
  """Locate the persist disk, the HF cache and the MaxText tree. The disk has moved between hosts before,
  so prefer an explicit flag/env var and fall back to whichever candidate actually holds the HF hub."""
  maxtext_root = args.maxtext_root or os.environ.get("MAXTEXT_ROOT") or os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  )
  hf_home = args.hf_home or os.environ.get("HF_HOME")
  if not hf_home:
    for cand in ("/wenxindong/mnt/disks/persist", "/mnt/disks/persist", os.path.expanduser("~/.cache/huggingface")):
      if os.path.isdir(os.path.join(cand, "hub")):
        hf_home = cand
        break
    else:
      raise SystemExit("could not locate an HF cache; pass --hf-home")
  out_dir = args.out_dir or os.environ.get("OUT_DIR") or os.path.join(hf_home, "rl_logprob_parity")
  os.makedirs(out_dir, exist_ok=True)
  return maxtext_root, hf_home, out_dir


def set_tpu_env(hf_home, maxtext_root, sampler):
  """Must run before jax/libtpu/vllm are imported anywhere in the process."""
  os.environ.setdefault("HF_HOME", hf_home)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
  os.environ["NEW_MODEL_DESIGN"] = "1"
  # Production serving env, copied from run_vllm.sh.
  for k, v in {
      "USE_MOE_EP_KERNEL": "0",
      "ATTN_BUCKETIZED_NUM_REQS": "true",
      "ATTN_CUSTOM_NUM_REQS_BUCKETS": "4",
      "ONEHOT_MOE_PERMUTE_THRESHOLD": "32768",
      "VLLM_MOE_CHUNK_SIZE": "256",
      "SLICE_ROPE_CACHE": "1",
      "DP_SCHED_BATCH_PREFILL": "false",
      "NUM_PRECOMPILE_WORKERS": "8",
      "SKIP_JAX_PRECOMPILE": "1",
      "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
      "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
  }.items():
    os.environ.setdefault(k, v)
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
      " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
      " --xla_tpu_ars_combiner_threshold_in_bytes=0"
      " --xla_tpu_enable_async_collective_merger=false"
      " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false",
  )
  os.environ["MODEL_IMPL_TYPE"] = "flax_nnx" if sampler == "adapter" else "vllm"
  src = os.path.join(maxtext_root, "src")
  if src not in sys.path:
    sys.path.insert(0, src)


# ---------------------------------------------------------------- 1. tokenize


def stage_tokenize(args, hf_home, maxtext_root, out_dir):
  """Tokenize the corpus once into [B, S]. Both TPU stages read this file, never the raw text."""
  path = os.path.join(out_dir, "tokens.npz")
  if os.path.exists(path) and not args.retokenize:
    z = np.load(path)
    log(f"tokens: reusing {path} {z['tokens'].shape}")
    return path
  os.environ.setdefault("HF_HOME", hf_home)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  from transformers import AutoTokenizer

  tok = AutoTokenizer.from_pretrained(MODEL_HF)
  if args.text_file:
    files = [args.text_file]
  else:
    files = sorted(glob.glob(os.path.join(maxtext_root, args.text_glob), recursive=True))
  if not files:
    raise SystemExit(f"no text matched {args.text_glob!r} under {maxtext_root}")
  text = "\n\n".join(open(f, encoding="utf-8").read() for f in files)
  ids = tok(text)["input_ids"]
  if len(ids) < B * SEQ_LEN:
    raise SystemExit(f"need {B * SEQ_LEN} tokens, corpus has {len(ids)}")
  tokens = np.array(ids[: B * SEQ_LEN], dtype=np.int32).reshape(B, SEQ_LEN)
  np.savez(path, tokens=tokens, n_files=np.int32(len(files)), n_corpus_tokens=np.int32(len(ids)))
  log(f"tokens: {tokens.shape} from {len(files)} files / {len(ids)} corpus tokens -> {path}")
  log(f"tokens: first = {tok.decode(tokens[0, :12])!r}")
  return path


# ---------------------------------------------------------------- 2. sampler


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  """Real vLLM engine, prompt_logprobs on the shared tokens. logp[b, i] = logprob of token i given tokens[:i]."""
  set_tpu_env(hf_home, maxtext_root, args.sampler)
  import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
  import tpu_inference  # noqa: F401  pylint: disable=unused-import
  from vllm import LLM, SamplingParams, TokensPrompt

  tokens = np.load(os.path.join(out_dir, "tokens.npz"))["tokens"]
  kw = dict(
      model=MODEL_HF,
      dtype="bfloat16",
      tensor_parallel_size=8,
      enable_expert_parallel=True,
      max_model_len=4096,
      max_num_seqs=16,
      max_num_batched_tokens=2048,
      block_size=256,
      enable_chunked_prefill=True,
      enable_prefix_caching=False,
      gpu_memory_utilization=args.gpu_memory_utilization,
      language_model_only=True,
      limit_mm_per_prompt={"image": 0, "video": 0},
      disable_log_stats=True,
      kv_cache_dtype="bfloat16",
      seed=0,
  )
  sharding = (
      {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": args.attn_dp}}
      if args.attn_dp > 1
      else None
  )
  if args.sampler == "adapter":
    # MaxText-in-vLLM: MoE runs TP-8 on this path (EP is not reachable through the adapter).
    sys.path.insert(0, os.path.join(maxtext_root, "src", "maxtext", "integration", "vllm"))
    import maxtext_vllm_adapter

    maxtext_vllm_adapter.register()
    mt_cfg = {
        "model_name": MODEL_MAXTEXT,
        "load_parameters_path": args.ckpt,
        "weight_dtype": "bfloat16",
        "dtype": "bfloat16",
        "attention": "vllm_rpa",
        "allow_split_physical_axes": True,
        "scan_layers": False,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "prefuse_moe_weights": True,
        "enable_dp_attention": args.attn_dp > 1,
        "log_config": False,
        "enable_checkpointing": True,
        "async_checkpointing": False,
        "checkpoint_storage_use_ocdbt": True,
        "checkpoint_storage_use_zarr3": True,
        "convert_checkpoint_if_possible": False,
        "float32_logits": True,
        "float32_gate_logits": True,
        "float32_weight_sum": True,
    }
    kw["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
    kw["additional_config"] = {"maxtext_config": mt_cfg, **({"sharding": sharding} if sharding else {})}
  elif sharding:
    kw["additional_config"] = {"sharding": sharding}

  llm = LLM(**kw)
  log(f"{args.sampler} engine up")

  if args.sampler == "adapter":
    # Text-only run: the MaxText adapter exposes no M-RoPE hook, so force plain 1-D positions.
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
      if obj is not None and hasattr(obj, "uses_mrope"):
        obj.uses_mrope = False

    def _text_mrope(prompt_token_ids, mm_features):
      pos = np.arange(len(prompt_token_ids), dtype=np.int64)
      return np.stack([pos, pos, pos]), 0

    runner.get_mrope_input_positions_fn = _text_mrope

  sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
  outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)

  logp = np.full(tokens.shape, np.nan, np.float32)
  top1 = np.full(tokens.shape, -1, np.int32)
  for b, o in enumerate(outs):
    for i, d in enumerate(o.prompt_logprobs):
      if d is None:  # position 0 has no conditioning context
        continue
      tid = int(tokens[b, i])
      logp[b, i] = d[tid].logprob if tid in d else np.nan
      top1[b, i] = max(d.items(), key=lambda kv: kv[1].logprob)[0]
  path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  np.savez(path, logp=logp, top1=top1, tokens=tokens)
  log(f"sampler: saved {path}; mean logp={np.nanmean(logp[:, 1:]):.4f} "
      f"top-1 acc={np.mean(top1[:, 1:] == tokens[:, 1:]):.3f}")
  return path


# ---------------------------------------------------------------- 3. trainer


def stage_trainer(args, hf_home, maxtext_root, out_dir):
  """MaxText nnx model in MODEL_MODE_TRAIN, teacher-forced over the same tokens.
  logp[b, i] = logprob of token i+1 given tokens[:i+1], i.e. shifted one left of the sampler's indexing."""
  set_tpu_env(hf_home, maxtext_root, args.sampler)
  import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
  import tpu_inference  # noqa: F401  pylint: disable=unused-import
  import jax
  import jax.numpy as jnp
  from flax import linen as nn
  from flax import nnx
  from jax.sharding import NamedSharding, PartitionSpec as P

  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.utils import model_creation_utils
  import maxtext.configs as maxtext_configs

  base_yml = os.path.join(os.path.dirname(maxtext_configs.__file__), "base.yml")
  tokens_np = np.load(os.path.join(out_dir, "tokens.npz"))["tokens"]

  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml, "attention=flash"],
      model_name=MODEL_MAXTEXT,
      load_parameters_path=args.ckpt,
      tokenizer_path=MODEL_HF,
      run_name="rl_logprob_parity",
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      scan_layers=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      dtype="bfloat16",
      weight_dtype="bfloat16",
      max_target_length=SEQ_LEN,
      max_prefill_predict_length=SEQ_LEN,
      per_device_batch_size=1.0,
      log_config=False,
      enable_nnx=True,
      pure_nnx=True,
      pure_nnx_decoder=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      float32_logits=True,
      float32_gate_logits=True,
      float32_weight_sum=True,
      use_tokamax_splash=True,
      sa_use_base2_exp=False,
      sa_fuse_reciprocal=True,
      sparse_matmul=True,
      megablox=True,
      use_tokamax_gmm=True,
      use_gmm_v2=True,
      wi_tile_fwd_batch_seq=256,
      wi_tile_fwd_embed_dim=128,
      wi_tile_fwd_mlp_dim=128,
  )
  log(f"trainer cfg: emb={cfg.emb_dim} q={cfg.num_query_heads} kv={cfg.num_kv_heads} "
      f"E={cfg.num_experts} k={cfg.num_experts_per_tok} layers={cfg.num_decoder_layers}")

  model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
  log("trainer model loaded")
  gd, st = nnx.split(model)

  @jax.jit
  def fwd(st, tokens, pos, seg):
    m = nnx.merge(gd, st)
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      out = m(tokens, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    return out[0] if isinstance(out, tuple) else out

  with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))
    tokens = jax.device_put(tokens_np, sh)
    pos = jax.device_put(np.broadcast_to(np.arange(SEQ_LEN, dtype=np.int32), (B, SEQ_LEN)), sh)
    seg = jax.device_put(np.ones((B, SEQ_LEN), np.int32), sh)
    logits = fwd(st, tokens, pos, seg)
    jax.block_until_ready(logits)

  lp = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), -1))
  nxt = np.roll(tokens_np, -1, axis=1)
  logp = np.take_along_axis(lp, nxt[..., None], -1)[..., 0]
  top1 = lp.argmax(-1)
  acc = np.mean(top1[:, :-1] == tokens_np[:, 1:])
  path = os.path.join(out_dir, "trainer_logprobs.npz")
  np.savez(path, logp=logp, top1=top1, tokens=tokens_np)
  log(f"trainer: saved {path}; mean logp={logp[:, :-1].mean():.4f} next-token top-1 acc={acc:.3f} (ckpt sanity)")
  return path


# ---------------------------------------------------------------- 4. compare


def masked_mean(x, mask, axis=None, global_normalization_factor=None):
  num = (x * mask).sum(axis=axis)
  if global_normalization_factor is not None:
    return num / global_normalization_factor
  return num / mask.sum(axis=axis)


def compare(prev_logprobs, generation_logprobs, token_mask=None, sample_mask=None):
  """prev_logprobs = trainer (pi_prev), generation_logprobs = sampler (pi_gen), already index-aligned.

  Per-sequence half mirrors the RL trainer's "seq-mask-tis" branch: nan_to_num the log ratio to 0.0
  (so a missing logprob counts as ratio 1.0 and still occupies a slot in the mean), take the masked
  mean per sequence, exponentiate, keep sequences whose geomean lands inside the band, and report
  is_oob_ratio as the *rejected* fraction.
  """
  raw = prev_logprobs - generation_logprobs
  log_is_ratio = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
  if token_mask is None:
    token_mask = np.isfinite(raw).astype(np.float64)
  n_seq = log_is_ratio.shape[0]
  if sample_mask is None:
    sample_mask = np.ones(n_seq, dtype=np.float64)
  global_valid_seqs = sample_mask.sum()

  # per-token
  r_tok = np.exp(log_is_ratio)
  in_tok = ((r_tok >= RATIO_MIN) & (r_tok <= RATIO_MAX)).astype(np.float64)
  tok_oob = masked_mean(1.0 - in_tok, token_mask)
  d = np.abs(log_is_ratio)[token_mask > 0]

  # per-sequence (seq-mask-tis)
  seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, axis=-1)
  seq_geomean_is_ratio = np.exp(seq_log_is_ratio_mean)
  seq_kept_mask = ((seq_geomean_is_ratio >= RATIO_MIN) & (seq_geomean_is_ratio <= RATIO_MAX)).astype(np.float64)
  seq_oob = masked_mean(1.0 - seq_kept_mask, sample_mask, global_normalization_factor=global_valid_seqs)

  return dict(
      n_tokens=int(token_mask.sum()),
      n_seqs=n_seq,
      n_nonfinite=int((~np.isfinite(raw)).sum()),
      token_oob_ratio=float(tok_oob),
      token_in_band=float(1.0 - tok_oob),
      abs_d_median=float(np.median(d)),
      abs_d_p99=float(np.percentile(d, 99)),
      abs_d_max=float(d.max()),
      seq_log_is_ratio_mean=seq_log_is_ratio_mean,
      seq_geomean_is_ratio=seq_geomean_is_ratio,
      seq_kept_mask=seq_kept_mask,
      is_oob_ratio=float(seq_oob),
  )


def print_report(name, m):
  print(f"\n### {name}")
  print(f"  tokens={m['n_tokens']}  seqs={m['n_seqs']}  nonfinite(zeroed)={m['n_nonfinite']}  "
        f"band=[{RATIO_MIN}, {RATIO_MAX}]")
  print(f"  per-token : oob {m['token_oob_ratio']:.2%}  in-band {m['token_in_band']:.2%}   "
        f"|dlogp| med {m['abs_d_median']:.4f} / p99 {m['abs_d_p99']:.3f} / max {m['abs_d_max']:.2f}")
  print(f"  {'seq':>4} {'seq_log_is_ratio_mean':>22} {'seq_geomean_is_ratio':>21} {'kept':>5}")
  for b in range(m["n_seqs"]):
    print(f"  {b:>4} {m['seq_log_is_ratio_mean'][b]:>22.6e} "
          f"{m['seq_geomean_is_ratio'][b]:>21.6f} {int(m['seq_kept_mask'][b]):>5}")
  print(f"  kept {int(m['seq_kept_mask'].sum())}/{m['n_seqs']}")
  print(f"  >>> is_oob_ratio (seq-mask-tis) = {m['is_oob_ratio']:.4f} ({m['is_oob_ratio']:.2%})")
  print(f"      oob_ratio    (per-token)   = {m['token_oob_ratio']:.4f} ({m['token_oob_ratio']:.2%})")


def stage_compare(args, out_dir):
  """Align and score. The trainer's logp[:, i] scores token i+1, the sampler's logp[:, i] scores token i,
  so the trainer is trimmed on the right and the sampler on the left."""
  tr_path = os.path.join(out_dir, "trainer_logprobs.npz")
  sa_path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  for p in (tr_path, sa_path):
    if not os.path.exists(p):
      raise SystemExit(f"missing {p}; run the corresponding stage first")
  tr = np.load(tr_path)
  sa = np.load(sa_path)
  if not np.array_equal(tr["tokens"], sa["tokens"]):
    raise SystemExit("trainer and sampler ran on different tokens; delete the npz files and rerun")
  m = compare(tr["logp"][:, :-1], sa["logp"][:, 1:])
  label = "MaxText-in-vLLM (adapter)" if args.sampler == "adapter" else "native tpu-inference"
  print_report(f"MaxText trainer vs {label} sampler — Qwen3.5-35B-A3B bf16/bf16, full model", m)
  path = os.path.join(out_dir, f"parity_{args.sampler}.npz")
  np.savez(path, **{k: v for k, v in m.items()})
  print(f"\nsaved {path}")
  return m


# ---------------------------------------------------------------- driver


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"])
  ap.add_argument("--sampler", default="adapter", choices=["adapter", "native"],
                  help="adapter = MaxText-in-vLLM (MODEL_IMPL_TYPE=flax_nnx); native = tpu-inference torchax path")
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--hf-home", default=None)
  ap.add_argument("--maxtext-root", default=None)
  ap.add_argument("--ckpt", default=CKPT, help="MaxText-format checkpoint for the trainer and the adapter")
  ap.add_argument("--attn-dp", type=int, default=4, help="attention DP degree inside the tp=8 mesh")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
  ap.add_argument("--text-glob", default="docs/**/*.md", help="corpus glob, relative to the MaxText tree")
  ap.add_argument("--text-file", default=None, help="single text file, overrides --text-glob")
  ap.add_argument("--retokenize", action="store_true", help="rebuild tokens.npz even if it exists")
  args = ap.parse_args()

  maxtext_root, hf_home, out_dir = resolve_paths(args)
  log(f"maxtext_root={maxtext_root}")
  log(f"hf_home={hf_home}")
  log(f"out_dir={out_dir}")

  if args.stage == "compare":
    stage_compare(args, out_dir)
    return
  if args.stage == "tokenize":
    stage_tokenize(args, hf_home, maxtext_root, out_dir)
    return
  if args.stage == "sampler":
    stage_tokenize(args, hf_home, maxtext_root, out_dir)
    stage_sampler(args, hf_home, maxtext_root, out_dir)
    return
  if args.stage == "trainer":
    stage_tokenize(args, hf_home, maxtext_root, out_dir)
    stage_trainer(args, hf_home, maxtext_root, out_dir)
    return

  # stage == all: tokenize here, then one fresh process per TPU stage (each needs the whole TPU and a
  # different MODEL_IMPL_TYPE), then compare in this process.
  stage_tokenize(args, hf_home, maxtext_root, out_dir)
  common = [
      sys.executable, os.path.abspath(__file__),
      "--sampler", args.sampler,
      "--out-dir", out_dir,
      "--hf-home", hf_home,
      "--maxtext-root", maxtext_root,
      "--ckpt", args.ckpt,
      "--attn-dp", str(args.attn_dp),
      "--gpu-memory-utilization", str(args.gpu_memory_utilization),
  ]
  for st in ("sampler", "trainer"):
    log(f"=== subprocess: --stage {st} ===")
    rc = subprocess.call(common + ["--stage", st])
    if rc != 0:
      raise SystemExit(f"stage {st} failed with exit code {rc}")
  stage_compare(args, out_dir)


if __name__ == "__main__":
  main()
