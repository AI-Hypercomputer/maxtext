#!/usr/bin/env python3
"""Qwen3.5-35B-A3B bf16: MaxText trainer vs vLLM sampler logprob parity, end to end in one script.

Four stages, in order:
  1. tokenize  (CPU)  real text -> HF tokenizer -> [B, S] token ids, saved once and reused by both sides
  2. sampler   (TPU)  real vLLM engine: prompt_logprobs, then --gen-tokens rollouts recording each sampled
                      token's logprob; --sampler adapter (MaxText-in-vLLM) or native (tpu-inference)
  3. trainer   (TPU)  MaxText nnx model, MODEL_MODE_TRAIN, teacher-forced over prompt+generated in one pass
  4. compare   (CPU)  per-token band stats + per-sequence seq-mask-tis is_oob_ratio, reported separately for
                      PROMPT (prefill) and OUTPUT (decode) tokens

All four stages run in one process. The TPU env (MODEL_IMPL_TYPE, LIBTPU_INIT_ARGS, the serving flags) is set
once up front, before jax/vllm are imported, and is the same for both TPU stages; the sampler's engine is
released before the trainer loads so only one set of 35B weights is resident at a time. If HBM is still tight,
run `--stage sampler` and `--stage trainer` as separate invocations — they hand off through the npz files.

Unlike the per-row scripts in this directory, both sides read their tokens from the same tokens.npz: the
trainer must never re-tokenize independently, or the two sides score different text.

Usage:
    # Real r2e-gym prompts from an RL rollout trace (32 x 32768) plus 8192 sampled tokens. ~8.5 min on
    # 8 x v7x. --trainer-micro-batch/--trainer-tp are required at this length: one 40960-token forward
    # over all 32 rows does not fit, and TP shards the vocab dimension of the logits.
    python compare_trainer_sampler.py \
        --prompts-file tools/rl_logprob_parity/r2e_prompts_32.jsonl \
        --prompt-len 32768 --gen-tokens 8192 \
        --trainer-micro-batch 4 --trainer-tp 2 --gpu-memory-utilization 0.7 \
        --out-dir /path/to/out

    python compare_trainer_sampler.py                                                   # synthetic corpus
    python compare_trainer_sampler.py --gen-tokens 0                                    # prompt tokens only
    python compare_trainer_sampler.py --stage compare --out-dir /path/to/existing/npz   # re-score saved runs

With --gen-tokens > 0 the sampler also writes the rollouts as text to generations_<sampler>.jsonl.

Paths default to autodetection (see resolve_paths) and can be overridden with --out-dir / --hf-home /
--maxtext-root or the env vars OUT_DIR / HF_HOME / MAXTEXT_ROOT.
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
# SEQ_LEN / --gen-tokens mirror the RL job's max_prompt_length and max_response_length.
B, SEQ_LEN = 8, 4096
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
  """Set the process-level TPU env. Call once, before jax/libtpu/vllm are imported anywhere in the process.

  The sampler and trainer stages share this env unchanged -- MODEL_IMPL_TYPE selects which vLLM model
  implementation the engine builds, and the trainer neither reads it nor is affected by it -- which is why
  both can run in a single process.
  """
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


def apply_gdn_conv_padding_fix(maxtext_root):
  """Patch Qwen3NextGatedDeltaNet.__call__ so conv1d does not smear left-padding into real tokens."""
  needle = "    qkv = jnp.concatenate([q, k, v], axis=-1)\n    batch, seq_len, _ = qkv.shape"
  replacement = (
      "    qkv = jnp.concatenate([q, k, v], axis=-1)\n"
      "    if decoder_segment_ids is not None:\n"
      "      qkv = jnp.where((decoder_segment_ids != 0)[..., None], qkv, 0.0)\n"
      "    batch, seq_len, _ = qkv.shape"
  )
  candidates = [
      os.path.join(maxtext_root, "src", "maxtext", "models", "qwen3.py"),
      "/opt/venv/lib/python3.12/site-packages/maxtext/models/qwen3.py",
  ]
  patched = 0
  for p in candidates:
    if not os.path.exists(p):
      continue
    txt = open(p, encoding="utf-8").read()
    if "qkv = jnp.where((decoder_segment_ids != 0)[..., None], qkv, 0.0)" in txt:
      log(f"[gdn-fix] already present in {p}")
      patched += 1
    elif needle in txt:
      open(p, "w", encoding="utf-8").write(txt.replace(needle, replacement, 1))
      log(f"[gdn-fix] PATCHED conv1d padding mask into {p}")
      patched += 1
    else:
      log(f"[gdn-fix] WARNING: needle not found in {p}")
  if not patched:
    raise SystemExit("--fix-gdn-conv-padding failed: no qwen3.py copy was patched")


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

  if args.prompts_file:
    # One real prompt per JSONL line ({"text": ...}); each becomes one row.
    #
    # Two layouts, and the difference between them is the entire point of
    # --left-pad:
    #
    #   default    every row is exactly SEQ_LEN dense real tokens. Simple, and
    #              wrong: NO PRODUCTION BATCH EVER LOOKS LIKE THIS, so an arm
    #              run this way is structurally incapable of saying anything
    #              about padding. Every "clean" result this harness produced
    #              before --left-pad existed was uninformative on that axis
    #              rather than exculpatory.
    #   --left-pad rows keep their natural length and are padded on the LEFT to
    #              SEQ_LEN, which is what tunix does to the trainer's batch
    #              (batch_assembly.py:812-826). The sampler is then handed the
    #              unpadded slice, exactly as production hands vLLM unpadded ids
    #              (vllm_sampler.py:650-661). This is the only mode in which the
    #              two engines see the layout they actually see in the RL job.
    #
    # prompt_starts[b] is the number of leading pad columns in row b, and is the
    # single source of truth every later stage uses to tell pad from real. It is
    # written even in the dense case (as zeros) so nothing downstream has to
    # branch on which layout produced the file.
    pad_id = tok.pad_token_id
    if pad_id is None:
      pad_id = tok.eos_token_id
    if pad_id is None:
      raise SystemExit("tokenizer exposes neither pad_token_id nor eos_token_id; cannot left-pad")
    rows, starts, short = [], [], []
    for ln in open(args.prompts_file, encoding="utf-8"):
      ln = ln.strip()
      if not ln:
        continue
      rec = json.loads(ln)
      ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
      if args.left_pad:
        # Keep the TAIL when over-long. tunix's _left_pad
        # (batch_assembly.py:137-148) keeps the tail, and for a VTC prompt the
        # tail is the part that must survive: it ends on the open <reasoning>
        # tag the model is supposed to close.
        ids = list(ids[-SEQ_LEN:])
        start = SEQ_LEN - len(ids)
        rows.append([pad_id] * start + ids)
        starts.append(start)
      else:
        if len(ids) < SEQ_LEN:
          short.append(len(ids))
        rows.append(ids[:SEQ_LEN])
        starts.append(0)
    if short:
      raise SystemExit(f"{len(short)} prompt(s) shorter than SEQ_LEN={SEQ_LEN}: {short[:5]}; "
                       "pass --left-pad to keep their natural length instead of padding the corpus")
    tokens = np.array(rows, dtype=np.int32)
    prompt_starts = np.array(starts, dtype=np.int32)
    np.savez(path, tokens=tokens, prompt_starts=prompt_starts, pad_id=np.int32(pad_id),
             n_files=np.int32(len(rows)), n_corpus_tokens=np.int32(tokens.size))
    log(f"tokens: {tokens.shape} from {len(rows)} prompts in {args.prompts_file} -> {path}")
    if args.left_pad:
      real = SEQ_LEN - prompt_starts
      log(f"tokens: LEFT-PADDED to {SEQ_LEN} with pad_id={pad_id}; real prompt length "
          f"min {real.min()} med {int(np.median(real))} max {real.max()}; "
          f"pad fraction {prompt_starts.sum() / tokens.size:.1%}")
      log(f"tokens: row 0 tail = {tok.decode(tokens[0, -24:])!r}")
    else:
      log(f"tokens: first = {tok.decode(tokens[0, :12])!r}")
    return path

  if args.text_file:
    files = [args.text_file]
  else:
    # The v10 container is built from a third-party base image that overlays only
    # four MaxText files, so docs/ is not guaranteed to be present. Fall back to
    # MaxText's own sources, which certainly are, and say which one won -- the
    # corpus identity has to be recoverable from the log alone.
    files = []
    for pattern in (args.text_glob, "src/maxtext/**/*.py", "src/maxtext/configs/*.yml"):
      files = sorted(glob.glob(os.path.join(maxtext_root, pattern), recursive=True))
      if files:
        log(f"tokens: corpus glob {pattern!r} matched {len(files)} files under {maxtext_root}")
        break
  if not files:
    raise SystemExit(f"no text matched {args.text_glob!r} (or the fallbacks) under {maxtext_root}")
  text = "\n\n".join(open(f, encoding="utf-8").read() for f in files)
  ids = tok(text)["input_ids"]
  # is_oob_ratio is a fraction of SEQUENCES, so at the default 8 rows its
  # resolution is 12.5 percentage points and a one-sequence flip looks like a
  # large move. Raise --rows when the per-sequence number is the thing under
  # test rather than the per-token one.
  rows = args.rows or B
  if len(ids) < rows * SEQ_LEN:
    raise SystemExit(f"need {rows * SEQ_LEN} tokens for {rows} rows, corpus has {len(ids)}")
  tokens = np.array(ids[: rows * SEQ_LEN], dtype=np.int32).reshape(rows, SEQ_LEN)
  # Zeros: the synthetic corpus is dense by construction. Written anyway so every
  # downstream stage can read prompt_starts unconditionally.
  np.savez(path, tokens=tokens, prompt_starts=np.zeros(rows, np.int32),
           n_files=np.int32(len(files)), n_corpus_tokens=np.int32(len(ids)))
  log(f"tokens: {tokens.shape} from {len(files)} files / {len(ids)} corpus tokens -> {path}")
  log(f"tokens: first = {tok.decode(tokens[0, :12])!r}")
  return path


# ---------------------------------------------------------------- 2. sampler


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
  except ImportError:
    try:
      import tpu_sync.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
    except ImportError:
      pass
  import tpu_inference  # noqa: F401  pylint: disable=unused-import
  from vllm import LLM, SamplingParams, TokensPrompt

  _tz = np.load(os.path.join(out_dir, "tokens.npz"))
  tokens = _tz["tokens"]
  # Zeros fallback keeps a tokens.npz written before --left-pad existed readable.
  starts = _tz["prompt_starts"] if "prompt_starts" in _tz.files else np.zeros(len(tokens), np.int32)
  kw = dict(
      model=MODEL_HF,
      dtype="bfloat16",
      # One host is the whole world for this script: it is a single-controller
      # program, so it can only see jax.local_devices(). On v5p that is 4 chips
      # (ct5p-hightpu-4t), not the 8 of the original 8xv7x run.
      tensor_parallel_size=args.sampler_tp,
      # EP off: the RL job's rollout mesh (fsdp=32, tp=4) sets no expert parallelism.
      enable_expert_parallel=False,
      max_model_len=max(4096, SEQ_LEN + args.gen_tokens + 64),
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
      reasoning_parser="qwen3",
      seed=0,
  )
  # The rest of the serving config has no offline equivalent: --enable-auto-tool-choice,
  # --tool-call-parser=qwen3_coder and --default-chat-template-kwargs are OpenAI-server frontend options
  # (vllm/entrypoints/openai/cli_args.py), not EngineArgs, so LLM() rejects them. They only shape how
  # generated text is parsed into API responses; this path feeds pre-tokenized ids via TokensPrompt and
  # reads prompt_logprobs, so no chat template, tool parser or reasoning parser ever runs.
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
        "logits_dot_in_fp32": getattr(args, "logits_dot_in_fp32", False),
        "cast_logits_to_fp32": True,
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

  # top_k=-1 disables top-k truncation (vLLM normalizes -1 to 0, "consider all tokens"). None of these
  # reach the measurement: prompt_logprobs is a plain log_softmax over the full vocab, and the single
  # generated token is discarded. They are set so the engine matches the serving config.
  sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=-1, top_p=1.0, prompt_logprobs=1)
  # Production hands vLLM UNPADDED ids (vllm_sampler.py:650-661); only the trainer's
  # batch is left-padded. Strip the pad here so the sampler sees what it sees in the
  # real job, then write each logprob back at its PADDED column so the sampler and
  # trainer arrays stay in one coordinate system and compare() needs no realignment.
  prompts = [TokensPrompt(prompt_token_ids=[int(t) for t in row[s:]]) for row, s in zip(tokens, starts)]
  outs = llm.generate(prompts, sp)

  logp = np.full(tokens.shape, np.nan, np.float32)
  top1 = np.full(tokens.shape, -1, np.int32)
  for b, o in enumerate(outs):
    s = int(starts[b])
    for i, d in enumerate(o.prompt_logprobs):
      if d is None:  # the first real token has no conditioning context
        continue
      col = s + i
      tid = int(tokens[b, col])
      logp[b, col] = d[tid].logprob if tid in d else np.nan
      top1[b, col] = max(d.items(), key=lambda kv: kv[1].logprob)[0]
  saved = dict(logp=logp, top1=top1, tokens=tokens, prompt_starts=starts)
  # Score only the positions the sampler actually returned. Under --left-pad the pad
  # columns keep top1 = -1, and averaging those in would silently deflate accuracy --
  # the ckpt-sanity check would then look like a bad checkpoint rather than padding.
  scored = np.isfinite(logp)
  log(f"sampler: prompt pass done; mean logp={np.nanmean(logp):.4f} "
      f"top-1 acc={np.mean(top1[scored] == tokens[scored]):.3f} over {int(scored.sum())} scored positions")
  path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  # Checkpoint before the rollouts: generation is by far the longest phase, and losing a completed prompt
  # pass to a crash partway through decode costs the whole prefill again.
  np.savez(path, **saved)

  if args.gen_tokens > 0:
    # Decode path: real rollouts from the same prompts. gen_logp[b, i] is the sampler's own logprob of
    # the token it sampled at step i -- the quantity an RL trainer would store as pi_gen for that token.
    # No per-request seed: the JAX/TPU path rejects it ("JAX does not support per-request seed").
    # Determinism comes from the engine-level seed=0 above.
    if args.stop_at_eos:
      # Let each rollout end at EOS, as it would in the RL loop. Rows then have different lengths; the
      # tail of each row is padded and its logprob left NaN, so compare()'s token mask drops it. Padding
      # sits after the real tokens and attention is causal, so it cannot affect the scored positions.
      gsp = SamplingParams(max_tokens=args.gen_tokens, temperature=args.gen_temperature,
                           top_p=1.0, top_k=-1, logprobs=1)
    else:
      gsp = SamplingParams(max_tokens=args.gen_tokens, min_tokens=args.gen_tokens,
                           temperature=args.gen_temperature, top_p=1.0, top_k=-1, logprobs=1,
                           ignore_eos=True)
    log(f"sampler: generating {args.gen_tokens} tokens x {len(tokens)} prompts (temperature={args.gen_temperature})")
    gouts = llm.generate(prompts, gsp)
    # Pad short rows with EOS so the trainer gets a rectangular batch; their logprobs stay NaN and are
    # masked out of every statistic.
    pad_id = llm.get_tokenizer().eos_token_id or 0
    gen_ids = np.full((len(tokens), args.gen_tokens), pad_id, np.int32)
    gen_logp = np.full((len(tokens), args.gen_tokens), np.nan, np.float32)
    gen_lens = np.zeros(len(tokens), np.int32)
    for b, o in enumerate(gouts):
      c = o.outputs[0]
      ids = list(c.token_ids)
      n = min(len(ids), args.gen_tokens)
      gen_ids[b, :n] = ids[:n]
      gen_lens[b] = n
      for i in range(n):
        d = c.logprobs[i]
        gen_logp[b, i] = d[ids[i]].logprob if ids[i] in d else np.nan
      if n < args.gen_tokens and not args.stop_at_eos:
        log(f"sampler: WARNING prompt {b} returned {n} < {args.gen_tokens} tokens")
    saved.update(gen_ids=gen_ids, gen_logp=gen_logp, gen_lens=gen_lens)
    log(f"sampler: generation done; mean gen logp={np.nanmean(gen_logp):.4f}; "
        f"lengths min {gen_lens.min()} med {int(np.median(gen_lens))} max {gen_lens.max()} "
        f"(hit cap: {(gen_lens >= args.gen_tokens).sum()}/{len(gen_lens)})")

    # Dump the rollouts as text too -- the npz only has ids, and the text is what you actually read when
    # judging whether the model is producing sane agent output.
    gen_tok = llm.get_tokenizer()
    gen_path = os.path.join(out_dir, f"generations_{args.sampler}.jsonl")
    with open(gen_path, "w", encoding="utf-8") as fh:
      for b in range(len(tokens)):
        ids = [int(t) for t in gen_ids[b, : gen_lens[b]]]
        fh.write(json.dumps({
            "row": b,
            "n_tokens": len(ids),
            "mean_logp": float(np.nanmean(gen_logp[b])),
            "prompt_tail": gen_tok.decode([int(t) for t in tokens[b, -200:]]),
            "text": gen_tok.decode(ids),
        }, ensure_ascii=False) + "\n")
    log(f"sampler: wrote rollout text to {gen_path}")
    del gouts

  path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  np.savez(path, **saved)
  log(f"sampler: saved {path}")

  # Release the engine's weights and KV cache before the trainer loads its own copy of the 35B.
  del outs, llm
  gc.collect()
  log("sampler: engine released")
  return path


# ---------------------------------------------------------------- 3. trainer


def stage_trainer(args, hf_home, maxtext_root, out_dir):
  """MaxText nnx model in MODEL_MODE_TRAIN, teacher-forced over the same tokens.
  logp[b, i] = logprob of token i+1 given tokens[:i+1], i.e. shifted one left of the sampler's indexing."""
  try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
  except ImportError:
    try:
      import tpu_sync.frameworks.jax._tpu_raiden_jax  # noqa: F401  pylint: disable=unused-import
    except ImportError:
      pass
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
  _tz = np.load(os.path.join(out_dir, "tokens.npz"))
  prompt_np = _tz["tokens"]
  # Zeros fallback keeps a tokens.npz written before --left-pad existed readable.
  starts = _tz["prompt_starts"] if "prompt_starts" in _tz.files else np.zeros(prompt_np.shape[0], np.int32)

  # If the sampler produced rollouts, teacher-force over prompt+generated so the same forward pass yields
  # the trainer's logprob for every sampled token.
  sa_path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  gen_ids = None
  if os.path.exists(sa_path):
    sa = np.load(sa_path)
    if "gen_ids" in sa.files:
      gen_ids = sa["gen_ids"]
  tokens_np = prompt_np if gen_ids is None else np.concatenate([prompt_np, gen_ids], axis=1).astype(np.int32)
  seq_len = tokens_np.shape[1]
  log(f"trainer: scoring {tokens_np.shape} ({'prompt only' if gen_ids is None else f'prompt {SEQ_LEN} + gen {gen_ids.shape[1]}'})")

  # These five are the settings where the production RL trainer
  # (tunix/utils/maxtext_utils.py build_maxtext_config) differs from this
  # harness's reference config. They are exposed individually so a regression
  # can be attributed to one of them rather than to the bundle.
  #
  # Note on dtype: parity is agreement with the SAMPLER, not accuracy. The
  # sampler runs bfloat16 activations. Widening only the trainer to float32
  # makes the trainer more accurate and therefore further from the sampler.
  # The logits head is the exception -- logits_dot_in_fp32 / cast_logits_to_fp32
  # are set on BOTH sides, so widening there is symmetric.
  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml, f"attention={args.trainer_attention}"],
      model_name=MODEL_MAXTEXT,
      load_parameters_path=args.ckpt,
      tokenizer_path=MODEL_HF,
      run_name="rl_logprob_parity",
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      scan_layers=args.trainer_scan_layers,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      dtype=args.trainer_dtype,
      weight_dtype="bfloat16",
      matmul_precision=args.trainer_matmul_precision,
      max_target_length=seq_len,
      max_prefill_predict_length=seq_len,
      # The forward pass runs one micro-batch at a time, so the global batch MaxText sizes activations for
      # is the micro-batch, not the full row count.
      per_device_batch_size=(args.trainer_micro_batch or tokens_np.shape[0]) / len(jax.devices()),
      # TP shards the vocab dimension, which is what makes the [micro, seq_len, 248320] logits fit.
      ici_tensor_parallelism=args.trainer_tp,
      allow_split_physical_axes=True,
      log_config=False,
      # Single host, and when the sampler stage ran first the vLLM engine has already brought up the JAX
      # backend in this process -- jax.distributed.initialize() would then fail outright.
      skip_jax_distributed_system=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      float32_logits=True,
      float32_gate_logits=True,
      float32_weight_sum=True,
      logits_dot_in_fp32=getattr(args, "logits_dot_in_fp32", False),
      cast_logits_to_fp32=True,
      use_tokamax_splash=True,
      sa_use_base2_exp=False,
      sa_fuse_reciprocal=True,
      sparse_matmul=True,
      megablox=True,
      use_tokamax_gmm=True,
      use_gmm_v2=True,
      # MXU tiling and accumulation order are keyed on matmul shape, so the MoE
      # GMM tile is a numerics knob, not just a performance one.
      wi_tile_fwd_batch_seq=args.wi_tile_batch_seq,
      wi_tile_fwd_embed_dim=args.wi_tile_embed_dim,
      wi_tile_fwd_mlp_dim=args.wi_tile_mlp_dim,
  )
  log(f"trainer cfg overrides: attention={args.trainer_attention} scan_layers={args.trainer_scan_layers} "
      f"dtype={args.trainer_dtype} matmul_precision={args.trainer_matmul_precision} "
      f"wi_tile={args.wi_tile_batch_seq}/{args.wi_tile_embed_dim}/{args.wi_tile_mlp_dim}")
  log(f"trainer cfg: emb={cfg.emb_dim} q={cfg.num_query_heads} kv={cfg.num_kv_heads} "
      f"E={cfg.num_experts} k={cfg.num_experts_per_tok} layers={cfg.num_decoder_layers}")

  model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
  log("trainer model loaded")
  gd, st = nnx.split(model)

  # Never materialize a [B, S, vocab] float32 array: at B=8, S=40960, vocab=248320 that is ~325 GB.
  # logprob(target) = logit(target) - logsumexp(logits), which reduces the vocab axis away and leaves
  # only [B, S] behind. argmax runs on the logits directly.
  @jax.jit
  def fwd_logp(st, tokens, pos, seg, nxt):
    m = nnx.merge(gd, st)
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      out = m(tokens, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    logits = out[0] if isinstance(out, tuple) else out
    tgt = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0].astype(jnp.float32)
    lse = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
    logp = jnp.minimum(tgt - lse, 0.0)
    return logp, jnp.argmax(logits, -1).astype(jnp.int32)

  nxt_np = np.roll(tokens_np, -1, axis=1)
  n_rows = tokens_np.shape[0]
  micro = args.trainer_micro_batch or n_rows
  if n_rows % micro:
    raise SystemExit(f"--trainer-micro-batch {micro} does not divide {n_rows} rows")

  # Positions and segment ids, reproducing tunix's production convention exactly
  # rather than approximating it (common.py:286-302):
  #
  #   segment_ids = (tokens != pad_id)                  -> pad 0, real 1
  #   positions   = cumsum(mask) - (cumsum >= 1)        -> 0 on every pad, and
  #                                                        the real tokens restart at 0
  #
  # MaxText compares segment ids for equality when it builds the attention mask
  # (attention_op.py:1105: `seg_q == seg_kv`), so pad and real end up in different
  # segments and a real token never attends to a pad -- wherever the padding lives.
  #
  # This replaces a hardcoded `arange` / `ones` pair. Those two lines silently
  # asserted that every row is dense and single-segment, which made this harness
  # structurally incapable of exhibiting a padding bug: pads would have been scored
  # as real tokens carrying wrong position ids. Getting the formula bit-exact
  # matters more than it looks -- "close enough" positions would be a THIRD layout,
  # belonging to neither the harness's old world nor production's.
  #
  # The generated block is never left-padded (every rollout starts at index 0 of
  # its own block), so only the prompt region contributes pads here.
  real_np = np.ones((n_rows, seq_len), np.int32)
  for b, s in enumerate(starts):
    real_np[b, : int(s)] = 0
  _cs = np.cumsum(real_np, axis=-1)
  pos_all = (_cs - (_cs >= 1)).astype(np.int32)
  seg_all = real_np
  if starts.max() > 0:
    log(f"trainer: LEFT-PADDED input; {int((real_np == 0).sum())} of {real_np.size} positions are pad "
        f"(segment 0). Row 0: start={int(starts[0])} pos[start-1:start+2]="
        f"{pos_all[0, max(int(starts[0]) - 1, 0):int(starts[0]) + 2].tolist()}")

  log(f"trainer: {n_rows} rows in chunks of {micro} (seq_len={seq_len})")
  logp_parts, top1_parts = [], []
  with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))
    for i in range(0, n_rows, micro):
      tokens = jax.device_put(tokens_np[i : i + micro], sh)
      nxt = jax.device_put(nxt_np[i : i + micro], sh)
      pos = jax.device_put(pos_all[i : i + micro], sh)
      seg = jax.device_put(seg_all[i : i + micro], sh)
      lp_d, t1_d = fwd_logp(st, tokens, pos, seg, nxt)
      jax.block_until_ready(lp_d)
      logp_parts.append(np.asarray(lp_d))
      top1_parts.append(np.asarray(t1_d))
      log(f"trainer: rows {i}-{i + micro - 1} done")
  logp = np.concatenate(logp_parts, axis=0)
  top1 = np.concatenate(top1_parts, axis=0)

  # Score only positions where BOTH the context token i and the target token i+1 are
  # real. Averaging pad columns in would make a correctly left-padded run look like a
  # broken checkpoint.
  score = (real_np[:, :-1] > 0) & (real_np[:, 1:] > 0)
  acc = np.mean(top1[:, :-1][score] == tokens_np[:, 1:][score])
  saved = dict(logp=logp, top1=top1, tokens=prompt_np, full_tokens=tokens_np, prompt_starts=starts)
  msg = (f"mean logp={logp[:, :-1][score].mean():.4f} next-token top-1 acc={acc:.3f} "
         f"(ckpt sanity, over {int(score.sum())} scored positions)")
  if gen_ids is not None:
    # logp[:, i] scores token i+1, so the sampled token at generated index j (absolute index n_prompt+j)
    # is scored by logp[:, n_prompt + j - 1]. Take n_prompt from the array, not the SEQ_LEN constant, so a
    # tokens.npz written at a different prompt length cannot silently mis-slice.
    n_prompt = prompt_np.shape[1]
    gen_logp_trainer = logp[:, n_prompt - 1 : seq_len - 1]
    saved["gen_logp"] = gen_logp_trainer
    msg += f"; mean gen logp={gen_logp_trainer.mean():.4f}"
  path = os.path.join(out_dir, "trainer_logprobs.npz")
  np.savez(path, **saved)
  log(f"trainer: saved {path}; {msg}")
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
  in_tok_1pct = ((r_tok >= 0.99) & (r_tok <= 1.01)).astype(np.float64)
  in_tok_5pct = ((r_tok >= 0.95) & (r_tok <= 1.05)).astype(np.float64)
  tok_oob = masked_mean(1.0 - in_tok, token_mask)
  tok_1pct = masked_mean(in_tok_1pct, token_mask)
  tok_5pct = masked_mean(in_tok_5pct, token_mask)
  d = np.abs(log_is_ratio)[token_mask > 0]

  # per-sequence (seq-mask-tis)
  seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, axis=-1)
  seq_geomean_is_ratio = np.exp(seq_log_is_ratio_mean)
  seq_kept_mask = ((seq_geomean_is_ratio >= RATIO_MIN) & (seq_geomean_is_ratio <= RATIO_MAX)).astype(np.float64)
  seq_in_1pct_mask = ((seq_geomean_is_ratio >= 0.99) & (seq_geomean_is_ratio <= 1.01)).astype(np.float64)
  seq_in_5pct_mask = ((seq_geomean_is_ratio >= 0.95) & (seq_geomean_is_ratio <= 1.05)).astype(np.float64)
  seq_oob = masked_mean(1.0 - seq_kept_mask, sample_mask, global_normalization_factor=global_valid_seqs)
  seq_1pct = masked_mean(seq_in_1pct_mask, sample_mask, global_normalization_factor=global_valid_seqs)
  seq_5pct = masked_mean(seq_in_5pct_mask, sample_mask, global_normalization_factor=global_valid_seqs)

  return dict(
      n_tokens=int(token_mask.sum()),
      n_seqs=n_seq,
      n_nonfinite=int((~np.isfinite(raw)).sum()),
      token_oob_ratio=float(tok_oob),
      token_in_band=float(1.0 - tok_oob),
      token_in_1pct=float(tok_1pct),
      token_in_5pct=float(tok_5pct),
      abs_d_median=float(np.median(d)),
      abs_d_p99=float(np.percentile(d, 99)),
      abs_d_max=float(d.max()),
      seq_log_is_ratio_mean=seq_log_is_ratio_mean,
      seq_geomean_is_ratio=seq_geomean_is_ratio,
      seq_kept_mask=seq_kept_mask,
      seq_in_1pct_mask=seq_in_1pct_mask,
      seq_in_5pct_mask=seq_in_5pct_mask,
      is_oob_ratio=float(seq_oob),
      seq_in_band=float(1.0 - seq_oob),
      seq_in_1pct=float(seq_1pct),
      seq_in_5pct=float(seq_5pct),
  )


def print_report(name, m):
  print(f"\n### {name}")
  print(f"  tokens={m['n_tokens']}  seqs={m['n_seqs']}  nonfinite(zeroed)={m['n_nonfinite']}  "
        f"band=[{RATIO_MIN}, {RATIO_MAX}]")
  print(f"  per-token : oob {m['token_oob_ratio']:.2%}  in-band [{RATIO_MIN}, {RATIO_MAX}] {m['token_in_band']:.2%}  "
        f"in ±1% {m.get('token_in_1pct', 0.0):.2%}  in ±5% {m.get('token_in_5pct', 0.0):.2%}   "
        f"|dlogp| med {m['abs_d_median']:.4f} / p99 {m['abs_d_p99']:.3f} / max {m['abs_d_max']:.2f}")
  print(f"  {'seq':>4} {'seq_log_is_ratio_mean':>22} {'seq_geomean_is_ratio':>21} {'kept':>5}")
  for b in range(m["n_seqs"]):
    print(f"  {b:>4} {m['seq_log_is_ratio_mean'][b]:>22.6e} "
          f"{m['seq_geomean_is_ratio'][b]:>21.6f} {int(m['seq_kept_mask'][b]):>5}")
  print(f"  kept {int(m['seq_kept_mask'].sum())}/{m['n_seqs']}")
  print(f"  >>> is_oob_ratio (seq-mask-tis) = {m['is_oob_ratio']:.4f} ({m['is_oob_ratio']:.2%})  "
        f"seq in-band={m.get('seq_in_band', 1.0 - m['is_oob_ratio']):.2%}  "
        f"seq in ±1%={m.get('seq_in_1pct', 0.0):.2%}  seq in ±5%={m.get('seq_in_5pct', 0.0):.2%}")
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
  label = "MaxText-in-vLLM (adapter)" if args.sampler == "adapter" else "native tpu-inference"
  head = f"MaxText trainer vs {label} sampler — Qwen3.5-35B-A3B bf16/bf16, full model"

  # The trainer's logp spans prompt+generated when rollouts were run, so bound the prompt comparison by the
  # sampler's prompt width rather than using the whole row.
  n_prompt = sa["logp"].shape[1]
  # Element i of these slices scores ABSOLUTE token i+1. Under --left-pad the first
  # `start` columns are padding and column `start` itself is the first real token,
  # which has no conditioning context and therefore no sampler logprob. So element i
  # is scoreable iff i+1 > start, i.e. i >= start.
  #
  # NaN propagation would drop the same columns today, because the sampler never
  # writes them. Building the mask explicitly anyway: it is auditable, it lets the
  # report state how many positions padding removed, and it does not quietly become
  # wrong the day the trainer returns NaN for a token that IS real.
  starts = sa["prompt_starts"] if "prompt_starts" in sa.files else np.zeros(sa["logp"].shape[0], np.int32)
  tr_prompt = tr["logp"][:, : n_prompt - 1]
  sa_prompt = sa["logp"][:, 1:n_prompt]
  cols = np.arange(n_prompt - 1)[None, :]
  prompt_mask = (cols >= starts[:, None]) & np.isfinite(tr_prompt - sa_prompt)
  if starts.max() > 0:
    print(f"\n[left-pad] prompt comparison keeps {int(prompt_mask.sum())} of {prompt_mask.size} columns; "
          f"real prompt length min {n_prompt - int(starts.max())} max {n_prompt - int(starts.min())}")
  m = compare(tr_prompt, sa_prompt, token_mask=prompt_mask.astype(np.float64))
  print_report(f"{head} — PROMPT tokens (teacher-forced, prefill)", m)
  out = {f"prompt_{k}": v for k, v in m.items()}

  if "gen_logp" in sa.files and "gen_logp" in tr.files:
    # Decode path. Both arrays are already indexed by generated position j, so no shift is needed:
    # the sampler reports its logprob for the token it sampled at step j, and the trainer's teacher-forced
    # logprob for that same token was sliced to match in stage_trainer.
    mg = compare(tr["gen_logp"], sa["gen_logp"])
    print_report(f"{head} — OUTPUT tokens (sampled, decode)", mg)
    out.update({f"output_{k}": v for k, v in mg.items()})
  else:
    print("\n(no rollouts in these npz files; rerun with --gen-tokens N for the output-token row)")

  path = os.path.join(out_dir, f"parity_{args.sampler}.npz")
  np.savez(path, **out)
  print(f"\nsaved {path}")
  return out


# ---------------------------------------------------------------- driver


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
  """Keep the module docstring's line breaks and still show each flag's default."""


def main():
  global SEQ_LEN  # pylint: disable=global-statement  (--prompt-len rebinds it for every stage)
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"],
                  help="all = every stage in this process; the rest hand off through npz files in --out-dir")
  ap.add_argument("--sampler", default="adapter", choices=["adapter", "native"],
                  help="adapter = MaxText-in-vLLM (MODEL_IMPL_TYPE=flax_nnx); native = tpu-inference torchax path")
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--hf-home", default=None)
  ap.add_argument("--maxtext-root", default=None)
  ap.add_argument("--ckpt", default=CKPT, help="MaxText-format checkpoint for the trainer and the adapter")
  ap.add_argument("--gen-tokens", type=int, default=61440,
                  help="output tokens to roll out per prompt for the decode-path comparison; 0 = prompt only")
  ap.add_argument("--gen-temperature", type=float, default=1.0, help="sampling temperature for the rollouts")
  ap.add_argument("--stop-at-eos", action="store_true",
                  help="let rollouts end at EOS (--gen-tokens becomes a cap) instead of forcing the full "
                       "length with ignore_eos; short rows are padded and their logprobs masked out")
  ap.add_argument("--attn-dp", type=int, default=4, help="attention DP degree inside the tp=8 mesh")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
  ap.add_argument("--prompt-len", type=int, default=SEQ_LEN,
                  help="prompt tokens per row; overrides the SEQ_LEN default")
  ap.add_argument("--rows", type=int, default=0,
                  help="rows (sequences) to build from the synthetic corpus; 0 uses the built-in default "
                       "of 8. Ignored when --prompts-file is given, which sets the count itself")
  ap.add_argument("--prompts-file", default=None,
                  help="JSONL of real prompts, one {\"text\": ...} per line; each becomes one row truncated "
                       "to SEQ_LEN. Overrides --text-glob/--text-file and sets the batch size.")
  ap.add_argument("--left-pad", action="store_true",
                  help="give each --prompts-file row its NATURAL length and LEFT-pad it to --prompt-len for "
                       "the trainer, while handing the sampler the unpadded slice -- the layout production "
                       "actually uses. Without this every row is dense, which no production batch ever is, "
                       "so the run cannot say anything about padding")
  ap.add_argument("--trainer-micro-batch", type=int, default=0,
                  help="rows per trainer forward pass; 0 = all at once. Must divide the row count")
  ap.add_argument("--sampler-tp", type=int, default=8,
                  help="tensor parallelism for the vLLM sampler; must equal the host's visible chip count "
                       "(4 on a single v5p host, 8 on the original v7x host)")
  ap.add_argument("--trainer-tp", type=int, default=1,
                  help="tensor parallelism for the trainer mesh; the rest of the devices go to FSDP")
  ap.add_argument("--logits-dot-in-fp32", action="store_true",
                  help="compute LM head vocabulary projection in FP32 to eliminate BF16 quantization noise")
  # Defaults below are the reference harness's values. Passing the production
  # RL trainer's values instead (--trainer-attention=dot_product
  # --trainer-scan-layers --trainer-dtype=float32 --trainer-matmul-precision=high
  # --wi-tile-batch-seq=512 --wi-tile-embed-dim=1024 --wi-tile-mlp-dim=1024)
  # reproduces tunix/utils/maxtext_utils.py build_maxtext_config offline.
  ap.add_argument("--trainer-attention", default="flash",
                  help="MaxText attention kernel for the trainer; the production RL trainer uses dot_product")
  ap.add_argument("--trainer-scan-layers", action="store_true",
                  help="scan the decoder layers, as the production RL trainer does; changes the lowering "
                       "and therefore the fusion and accumulation order")
  ap.add_argument("--trainer-dtype", default="bfloat16",
                  help="trainer ACTIVATION dtype. The sampler runs bfloat16; widening only the trainer "
                       "makes it more accurate and therefore further from the sampler")
  ap.add_argument("--trainer-matmul-precision", default="default",
                  help="jax.lax.Precision for the trainer; a no-op on bf16 operands, so it only bites "
                       "together with --trainer-dtype=float32")
  ap.add_argument("--wi-tile-batch-seq", type=int, default=256, help="MoE GMM forward tile: batch*seq")
  ap.add_argument("--wi-tile-embed-dim", type=int, default=128, help="MoE GMM forward tile: embed dim")
  ap.add_argument("--wi-tile-mlp-dim", type=int, default=128, help="MoE GMM forward tile: mlp dim")
  ap.add_argument("--text-glob", default="docs/**/*.md", help="corpus glob, relative to the MaxText tree")
  ap.add_argument("--text-file", default=None, help="single text file, overrides --text-glob")
  ap.add_argument("--retokenize", action="store_true", help="rebuild tokens.npz even if it exists")
  ap.add_argument("--fix-gdn-conv-padding", action="store_true",
                  help="patch Qwen3NextGatedDeltaNet.__call__ so conv1d zeros qkv where decoder_segment_ids == 0")
  args = ap.parse_args()
  SEQ_LEN = args.prompt_len
  # Fail closed. The synthetic corpus is dense by construction, so --left-pad there
  # would be a silent no-op -- and a silent no-op on the one axis this flag exists to
  # test is exactly the failure that made the old harness useless.
  if args.left_pad and not args.prompts_file:
    raise SystemExit("--left-pad requires --prompts-file: the synthetic corpus is dense by construction, "
                     "so there is nothing to pad and the flag would silently do nothing")

  maxtext_root, hf_home, out_dir = resolve_paths(args)
  # Sentinel. Injection into a container is a whole-file overwrite through a
  # non-fatal shell redirect: if the path is wrong the image's own copy runs and
  # nothing says so. Grep the log for this exact string before believing any
  # number below it.
  log("PARITY_HARNESS trellis-r3 (pr5231 + single-host sampler-tp + corpus fallback + left-pad + gdn-conv-fix)")
  log(f"arm: logits_dot_in_fp32={args.logits_dot_in_fp32} stop_at_eos={args.stop_at_eos} "
      f"sampler={args.sampler} sampler_tp={args.sampler_tp} attn_dp={args.attn_dp} "
      f"trainer_tp={args.trainer_tp} trainer_micro_batch={args.trainer_micro_batch} "
      f"fix_gdn_conv_padding={args.fix_gdn_conv_padding}")
  log(f"layout: left_pad={args.left_pad} "
      f"({'natural-length rows left-padded for the trainer, unpadded to the sampler (production layout)' if args.left_pad else 'dense rows, no padding anywhere (NOT the production layout)'})")
  log(f"shape: prompt_len={SEQ_LEN} gen_tokens={args.gen_tokens} gen_temperature={args.gen_temperature}")
  log(f"ckpt={args.ckpt}")
  log(f"maxtext_root={maxtext_root}")
  log(f"hf_home={hf_home}")
  log(f"out_dir={out_dir}")
  if args.fix_gdn_conv_padding:
    apply_gdn_conv_padding_fix(maxtext_root)

  if args.stage == "compare":
    stage_compare(args, out_dir)
    return

  # Everything below touches the TPU. Set the shared env once, before jax/vllm are imported.
  if args.stage != "tokenize":
    set_tpu_env(hf_home, maxtext_root, args.sampler)

  stage_tokenize(args, hf_home, maxtext_root, out_dir)
  if args.stage == "tokenize":
    return

  # The sampler runs first and frees its engine, so the trainer's weights are the only 35B resident
  # when it loads. Both stages share one process and one TPU init.
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
