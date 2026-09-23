#!/usr/bin/env python3
"""Microbenchmark: Log-prob parity between MaxText Trainer and vLLM Sampler with Prefix Caching.

This script isolates and measures whether vLLM prefix caching (on Mamba/GDN layers)
causes log-probability divergence against the MaxText trainer (which runs from scratch).

Workflow:
  1. tokenize:
     Creates prompts that share an identical prefix P (of length prefix_len):
       - Prompt 0 (Cold): P + S_0
       - Prompt 1 (Warm / Cache Hit): P + S_1
       - Prompt 2 (Warm / Cache Hit): P + S_2
       ...
  2. sampler:
     Runs vLLM with `enable_prefix_caching=True`:
       - First evaluates Prompt 0 (Cold prefill -> populates prefix cache).
       - Then evaluates Prompt 1..N (Warm prefill -> hits prefix cache on P).
       - Also generates `gen_tokens` rollouts for each prompt.
  3. trainer:
     Evaluates teacher-forced log-probabilities on the full sequences using MaxText.
  4. compare:
     Computes:
       - Per-token log-prob differences: Δ log p = log p_trainer - log p_sampler
       - Tunix multiplicative error: mean(exp(|Δ log p|))
       - Parity metrics on Cold vs Warm prompts (Suffix & Output tokens).
"""
import argparse
import gc
import glob
import json
import os
import sys
import time
import numpy as np

MODEL_HF = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT = "qwen3.5-35b-a3b"
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
RATIO_MIN, RATIO_MAX = 0.999, 1.002

t0 = time.time()

def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def resolve_paths(args):
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
  out_dir = args.out_dir or os.path.join(hf_home, "prefix_cache_microbench")
  os.makedirs(out_dir, exist_ok=True)
  return maxtext_root, hf_home, out_dir


def set_tpu_env(hf_home, maxtext_root, sampler):
  os.environ.setdefault("HF_HOME", hf_home)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
  os.environ["NEW_MODEL_DESIGN"] = "1"
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


def stage_tokenize(args, hf_home, maxtext_root, out_dir):
  path = os.path.join(out_dir, "tokens.npz")
  if os.path.exists(path) and not args.retokenize:
    log(f"tokens: reusing existing {path}")
    return path

  import transformers
  tok = transformers.AutoTokenizer.from_pretrained(
      MODEL_HF, cache_dir=os.path.join(hf_home, "hub"), local_files_only=True
  )

  files = sorted(glob.glob(os.path.join(maxtext_root, args.text_glob), recursive=True))
  if not files:
    raise SystemExit(f"no text matched {args.text_glob!r} under {maxtext_root}")
  texts = []
  for f in files:
    with open(f, encoding="utf-8") as fh:
      texts.append(fh.read())
  text = "\n\n".join(texts)
  all_ids = tok(text)["input_ids"]

  prefix_len = args.prefix_len
  suffix_len = args.suffix_len
  total_needed = prefix_len + (args.num_prompts * suffix_len)
  if len(all_ids) < total_needed:
    raise SystemExit(f"Corpus has {len(all_ids)} tokens, but need {total_needed}")

  shared_prefix = all_ids[:prefix_len]
  rows = []
  cur = prefix_len
  for b in range(args.num_prompts):
    suffix = all_ids[cur : cur + suffix_len]
    rows.append(shared_prefix + suffix)
    cur += suffix_len

  tokens = np.array(rows, dtype=np.int32)
  np.savez(
      path,
      tokens=tokens,
      prefix_len=np.int32(prefix_len),
      suffix_len=np.int32(suffix_len),
      num_prompts=np.int32(args.num_prompts),
  )
  log(f"tokens: created {tokens.shape} (prefix_len={prefix_len}, suffix_len={suffix_len}) -> {path}")
  return path


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
  import tpu_inference  # noqa: F401
  from vllm import LLM, SamplingParams, TokensPrompt

  token_data = np.load(os.path.join(out_dir, "tokens.npz"))
  tokens = token_data["tokens"]
  prefix_len = int(token_data["prefix_len"])
  suffix_len = int(token_data["suffix_len"])
  num_prompts = int(token_data["num_prompts"])
  prompt_len = tokens.shape[1]

  kw = dict(
      model=MODEL_HF,
      dtype="bfloat16",
      tensor_parallel_size=8,
      enable_expert_parallel=args.ep > 1,
      max_model_len=max(4096, prompt_len + args.gen_tokens + 64),
      max_num_seqs=16,
      max_num_batched_tokens=2048,
      block_size=256,
      enable_chunked_prefill=True,
      enable_prefix_caching=args.enable_prefix_caching,
      mamba_cache_mode=args.mamba_cache_mode,
      prefix_cache_retention_interval=0,
      gpu_memory_utilization=args.gpu_memory_utilization,
      language_model_only=True,
      limit_mm_per_prompt={"image": 0, "video": 0},
      disable_log_stats=True,
      kv_cache_dtype="bfloat16",
      reasoning_parser="qwen3",
      seed=0,
  )

  sharding = None
  if args.ep > 1:
    sharding = {
        "sharding_strategy": {
            "expert_parallelism": args.ep,
            "tensor_parallelism": args.sharding_tp,
            "enable_dp_attention": True,
        }
    }
  elif args.attn_dp > 1:
    sharding = {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": args.attn_dp}}

  if args.sampler == "adapter":
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
        "prefuse_moe_weights": True,
        "enable_dp_attention": args.ep > 1 or args.attn_dp > 1,
        "log_config": False,
        "enable_checkpointing": True,
        "async_checkpointing": False,
        "checkpoint_storage_use_ocdbt": True,
        "checkpoint_storage_use_zarr3": True,
        "convert_checkpoint_if_possible": False,
        "float32_logits": True,
        "float32_gate_logits": True,
        "float32_weight_sum": True,
        "logits_dot_in_fp32": args.logits_dot_fp32,
    }
    kw["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
    kw["additional_config"] = {
        "maxtext_config": mt_cfg,
        "custom_mamba_cache_multiplier": args.custom_mamba_cache_multiplier,
        **({"sharding": sharding} if sharding else {}),
    }
  elif sharding:
    kw["additional_config"] = {
        "sharding": sharding,
        "custom_mamba_cache_multiplier": args.custom_mamba_cache_multiplier,
    }

  log(f"sampler: initializing LLM (enable_prefix_caching={args.enable_prefix_caching})...")
  llm = LLM(**kw)
  log("sampler: LLM engine ready")

  if args.sampler == "adapter":
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
      if obj is not None and hasattr(obj, "uses_mrope"):
        obj.uses_mrope = False

    def _text_mrope(prompt_token_ids, mm_features):
      pos = np.arange(len(prompt_token_ids), dtype=np.int64)
      return np.stack([pos, pos, pos]), 0

    runner.get_mrope_input_positions_fn = _text_mrope

  sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=-1, top_p=1.0, prompt_logprobs=1)
  gsp = SamplingParams(max_tokens=args.gen_tokens, min_tokens=args.gen_tokens,
                       temperature=args.gen_temperature, top_p=1.0, top_k=-1, logprobs=1, ignore_eos=True) if args.gen_tokens > 0 else None

  logp = np.full(tokens.shape, np.nan, np.float32)
  gen_ids = np.full((num_prompts, args.gen_tokens), -1, np.int32) if args.gen_tokens > 0 else None
  gen_logp = np.full((num_prompts, args.gen_tokens), np.nan, np.float32) if args.gen_tokens > 0 else None

  # Sequential execution to cleanly exercise cold-then-warm prefix caching
  for b in range(num_prompts):
    tag = "COLD (populates prefix cache)" if b == 0 else f"WARM (cache hit on {prefix_len} tokens)"
    log(f"sampler: processing prompt {b} [{tag}]...")
    p_req = [TokensPrompt(prompt_token_ids=[int(t) for t in tokens[b]])]
    
    # 1. Prompt logprobs
    outs = llm.generate(p_req, sp)
    for i, d in enumerate(outs[0].prompt_logprobs):
      if d is not None:
        tid = int(tokens[b, i])
        logp[b, i] = d[tid].logprob if tid in d else np.nan

    # 2. Decode rollouts
    if args.gen_tokens > 0:
      gouts = llm.generate(p_req, gsp)
      c = gouts[0].outputs[0]
      ids = list(c.token_ids)[:args.gen_tokens]
      gen_ids[b, :len(ids)] = ids
      for i in range(len(ids)):
        d = c.logprobs[i]
        gen_logp[b, i] = d[ids[i]].logprob if ids[i] in d else np.nan

    scored_prompt = np.isfinite(logp[b]).sum()
    log(f"  prompt {b} done: {scored_prompt} prompt tokens scored, mean logp={np.nanmean(logp[b]):.4f}")
    if gen_logp is not None:
      log(f"  decode {b} done: {args.gen_tokens} tokens generated, mean gen logp={np.nanmean(gen_logp[b]):.4f}")

  saved = dict(logp=logp, tokens=tokens, prefix_len=prefix_len, suffix_len=suffix_len)
  if gen_ids is not None:
    saved.update(gen_ids=gen_ids, gen_logp=gen_logp)

  path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  np.savez(path, **saved)
  log(f"sampler: saved {path}")

  del outs, llm
  gc.collect()
  log("sampler: engine released")
  return path


def stage_trainer(args, hf_home, maxtext_root, out_dir):
  import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
  import tpu_inference  # noqa: F401
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
  token_data = np.load(os.path.join(out_dir, "tokens.npz"))
  prompt_np = token_data["tokens"]
  prefix_len = int(token_data["prefix_len"])

  sa_path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  gen_ids = None
  if os.path.exists(sa_path):
    sa = np.load(sa_path)
    if "gen_ids" in sa.files:
      gen_ids = sa["gen_ids"]

  tokens_np = prompt_np if gen_ids is None else np.concatenate([prompt_np, gen_ids], axis=1).astype(np.int32)
  orig_seq_len = tokens_np.shape[1]
  # TokaMax Splash Attention requires sequence length to be a multiple of q_block_size (512).
  padded_seq_len = ((orig_seq_len + 511) // 512) * 512
  n_rows = tokens_np.shape[0]
  ep_size = args.ep or 1
  micro = max(ep_size, args.trainer_micro_batch or ep_size)
  if micro % ep_size != 0:
    micro = ((micro + ep_size - 1) // ep_size) * ep_size

  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml, "attention=flash"],
      model_name=MODEL_MAXTEXT,
      load_parameters_path=args.ckpt,
      tokenizer_path=MODEL_HF,
      run_name="prefix_cache_trainer",
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      scan_layers=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      dtype="bfloat16",
      weight_dtype="bfloat16",
      max_target_length=padded_seq_len,
      max_prefill_predict_length=padded_seq_len,
      per_device_batch_size=micro / len(jax.devices()),
      ici_tensor_parallelism=args.trainer_tp,
      ici_expert_parallelism=args.ep,
      allow_split_physical_axes=True,
      log_config=False,
      skip_jax_distributed_system=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      float32_logits=True,
      float32_gate_logits=True,
      float32_weight_sum=True,
      logits_dot_in_fp32=args.logits_dot_fp32,
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

  model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
  log("trainer: model loaded")
  gd, st = nnx.split(model)

  @jax.jit
  def fwd_logp(st, tokens, pos, seg, nxt):
    m = nnx.merge(gd, st)
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      out = m(tokens, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    logits = out[0] if isinstance(out, tuple) else out
    tgt = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0].astype(jnp.float32)
    lse = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
    return tgt - lse, jnp.argmax(logits, -1).astype(jnp.int32)

  if n_rows % micro != 0:
    n_pad = micro - (n_rows % micro)
    padded_tokens = np.concatenate([tokens_np, np.repeat(tokens_np[-1:], n_pad, axis=0)], axis=0)
  else:
    padded_tokens = tokens_np

  if padded_seq_len != orig_seq_len:
    pad_len = padded_seq_len - orig_seq_len
    padded_tokens = np.pad(padded_tokens, ((0, 0), (0, pad_len)), mode="constant")

  nxt_np = np.roll(padded_tokens, -1, axis=1)
  total_padded_rows = padded_tokens.shape[0]
  log(f"trainer: {n_rows} actual rows ({total_padded_rows} padded) in chunks of {micro} (seq_len={orig_seq_len} -> {padded_seq_len})")

  logp_parts, top1_parts = [], []
  with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))
    pos_np = np.broadcast_to(np.arange(padded_seq_len, dtype=np.int32), (micro, padded_seq_len))
    seg_np = np.ones((micro, padded_seq_len), np.int32)
    for i in range(0, total_padded_rows, micro):
      tokens = jax.device_put(padded_tokens[i : i + micro], sh)
      nxt = jax.device_put(nxt_np[i : i + micro], sh)
      pos = jax.device_put(pos_np, sh)
      seg = jax.device_put(seg_np, sh)
      lp_d, t1_d = fwd_logp(st, tokens, pos, seg, nxt)
      jax.block_until_ready(lp_d)
      logp_parts.append(np.asarray(lp_d))
      top1_parts.append(np.asarray(t1_d))
      log(f"trainer: chunk {i//micro + 1}/{(total_padded_rows + micro - 1)//micro} done")

  logp = np.concatenate(logp_parts, axis=0)[:n_rows, :orig_seq_len]
  top1 = np.concatenate(top1_parts, axis=0)[:n_rows, :orig_seq_len]
  acc = np.mean(top1[:, :orig_seq_len - 1] == tokens_np[:, 1:orig_seq_len])
  log(f"trainer: sanity check mean logp={logp[:, :orig_seq_len - 1].mean():.4f} next-token top-1 acc={acc:.3f}")

  path = os.path.join(out_dir, "trainer_logprobs.npz")
  saved = dict(logp=logp, top1=top1, tokens=prompt_np, full_tokens=tokens_np)
  if gen_ids is not None:
    n_prompt = prompt_np.shape[1]
    gen_logp_trainer = logp[:, n_prompt - 1 : orig_seq_len - 1]
    saved["gen_logp"] = gen_logp_trainer
    log(f"trainer: mean gen logp={gen_logp_trainer.mean():.4f}")
  np.savez(path, **saved)
  log(f"trainer: saved {path}")
  return path


def stage_compare(args, out_dir):
  tr = np.load(os.path.join(out_dir, "trainer_logprobs.npz"))
  sa = np.load(os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz"))
  token_data = np.load(os.path.join(out_dir, "tokens.npz"))

  prefix_len = int(token_data["prefix_len"])
  suffix_len = int(token_data["suffix_len"])
  num_prompts = int(token_data["num_prompts"])

  print("\n" + "=" * 80)
  print(f"=== PREFIX CACHING PARITY REPORT (prefix={prefix_len}, suffix={suffix_len}) ===")
  print("=" * 80)

  # Prompt suffix comparison
  # Trainer logp[:, i] is token i+1.
  # Sampler logp[:, i] is token i.
  # Suffix spans tokens prefix_len to prefix_len + suffix_len.
  # Suffix indices in sampler: [prefix_len : prefix_len + suffix_len]
  # Corresponding indices in trainer: [prefix_len - 1 : prefix_len + suffix_len - 1]
  tr_suffix = tr["logp"][:, prefix_len - 1 : prefix_len + suffix_len - 1]
  sa_suffix = sa["logp"][:, prefix_len : prefix_len + suffix_len]

  raw_suffix_diff = tr_suffix - sa_suffix
  abs_suffix_diff = np.abs(raw_suffix_diff)

  print("\n--- 1. PROMPT SUFFIX (Evaluated by Sampler after Prefix) ---")
  print(f"{'Row':>4} | {'Type':<25} | {'|Δlogp| med':>11} | {'|Δlogp| p99':>11} | {'|Δlogp| max':>11} | {'mult_prob_err':>14} | {'Gate Kept (<2.0)':>16}")
  print("-" * 105)

  for b in range(num_prompts):
    row_type = "Cold (Cache Populate)" if b == 0 else f"Warm (Cache Hit on {prefix_len})"
    row_abs_d = abs_suffix_diff[b][np.isfinite(abs_suffix_diff[b])]
    if len(row_abs_d) == 0:
      print(f"{b:>4} | {row_type:<25} | {'N/A':>11} | {'N/A':>11} | {'N/A':>11} | {'N/A':>14} | {'N/A':>16}")
      continue
    med = np.median(row_abs_d)
    p99 = np.percentile(row_abs_d, 99)
    mx = np.max(row_abs_d)
    # Tunix multiplicative error: mean(exp(|Δlogp|))
    mpe = np.mean(np.exp(row_abs_d))
    kept = "YES" if mpe <= 2.0 else "NO (DROPPED!)"
    print(f"{b:>4} | {row_type:<25} | {med:>11.4f} | {p99:>11.4f} | {mx:>11.4f} | {mpe:>14.2f} | {kept:>16}")

  # Decode rollouts comparison (if present)
  if "gen_logp" in sa.files and "gen_logp" in tr.files:
    tr_gen = tr["gen_logp"]
    sa_gen = sa["gen_logp"]
    raw_gen_diff = tr_gen - sa_gen
    abs_gen_diff = np.abs(raw_gen_diff)

    print("\n--- 2. OUTPUT TOKENS (Decoded Rollouts starting after Prefix+Suffix) ---")
    print(f"{'Row':>4} | {'Type':<25} | {'|Δlogp| med':>11} | {'|Δlogp| p99':>11} | {'|Δlogp| max':>11} | {'mult_prob_err':>14} | {'Gate Kept (<2.0)':>16}")
    print("-" * 105)
    for b in range(num_prompts):
      row_type = "Cold Rollout" if b == 0 else f"Warm Rollout (Post-Hit)"
      row_abs_d = abs_gen_diff[b][np.isfinite(abs_gen_diff[b])]
      if len(row_abs_d) == 0:
        continue
      med = np.median(row_abs_d)
      p99 = np.percentile(row_abs_d, 99)
      mx = np.max(row_abs_d)
      mpe = np.mean(np.exp(row_abs_d))
      kept = "YES" if mpe <= 2.0 else "NO (DROPPED!)"
      print(f"{b:>4} | {row_type:<25} | {med:>11.4f} | {p99:>11.4f} | {mx:>11.4f} | {mpe:>14.2f} | {kept:>16}")

  print("\n" + "=" * 80)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"])
  ap.add_argument("--sampler", default="adapter", choices=["adapter", "native"])
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--hf-home", default=None)
  ap.add_argument("--maxtext-root", default=None)
  ap.add_argument("--ckpt", default=CKPT)
  ap.add_argument("--prefix-len", type=int, default=1024, help="length of shared prefix (e.g. 1024, 2048, or 1850)")
  ap.add_argument("--suffix-len", type=int, default=256, help="length of distinct prompt suffix")
  ap.add_argument("--num-prompts", type=int, default=4, help="number of prompts (1 cold + N-1 warm)")
  ap.add_argument("--gen-tokens", type=int, default=64, help="decode rollout tokens per prompt")
  ap.add_argument("--gen-temperature", type=float, default=0.0)
  ap.add_argument("--logits-dot-fp32", action="store_true")
  ap.add_argument("--attn-dp", type=int, default=4)
  ap.add_argument("--ep", type=int, default=8)
  ap.add_argument("--sharding-tp", type=int, default=1)
  ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
  ap.add_argument("--disable-prefix-caching", dest="enable_prefix_caching", action="store_false")
  ap.add_argument("--mamba-cache-mode", default="align", choices=["align", "none"])
  ap.add_argument("--custom-mamba-cache-multiplier", type=int, default=16)
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
  ap.add_argument("--trainer-micro-batch", type=int, default=2)
  ap.add_argument("--trainer-tp", type=int, default=1)
  ap.add_argument("--text-glob", default="docs/**/*.md")
  ap.add_argument("--retokenize", action="store_true")
  args = ap.parse_args()

  maxtext_root, hf_home, out_dir = resolve_paths(args)
  log(f"maxtext_root={maxtext_root}")
  log(f"hf_home={hf_home}")
  log(f"out_dir={out_dir}")

  if args.stage == "compare":
    stage_compare(args, out_dir)
    return

  if args.stage != "tokenize":
    set_tpu_env(hf_home, maxtext_root, args.sampler)

  stage_tokenize(args, hf_home, maxtext_root, out_dir)
  if args.stage == "tokenize":
    return

  if args.stage in ("all", "sampler"):
    log("=== stage: sampler ===")
    stage_sampler(args, hf_home, maxtext_root, out_dir)
  if args.stage in ("all", "trainer"):
    log("=== stage: trainer ===")
    stage_trainer(args, hf_home, maxtext_root, out_dir)
  if args.stage == "all":
    stage_compare(args, out_dir)


if __name__ == "__main__":
  main()
