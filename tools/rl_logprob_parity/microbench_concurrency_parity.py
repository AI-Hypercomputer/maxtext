#!/usr/bin/env python3
"""Multi-turn RL Logprob Parity & Concurrency Mamba Eviction Microbenchmark.

Tests logprob parity between vLLM rollout sampler and MaxText trainer under
continuous batching concurrency and aggressive Mamba cache eviction.

Usage:
    # Run end-to-end (sampler -> trainer -> compare):
    python microbench_concurrency_parity.py --stage all --out-dir /mnt/disks/persist/concurrency_bench

    # Run individual stages:
    python microbench_concurrency_parity.py --stage sampler --out-dir /mnt/disks/persist/concurrency_bench
    python microbench_concurrency_parity.py --stage trainer --out-dir /mnt/disks/persist/concurrency_bench --eval-convs 0,1,6,7
    python microbench_concurrency_parity.py --stage compare --out-dir /mnt/disks/persist/concurrency_bench --eval-convs 0,1,6,7
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
MODEL_HF = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT = "qwen3.5-35b-a3b"

TOPICS = [
    "Write a Python script to sort files by modification date and size.",
    "Explain how the Raft consensus algorithm handles network partitions and election timeouts.",
    "Implement a balanced binary search tree with insert, delete, and rebalancing operations.",
    "How does the Linux virtual filesystem (VFS) handle dentry and inode caching under heavy I/O?",
    "Describe the difference between process memory segments: text, data, bss, heap, and stack.",
    "Write an SQL query with window functions to find employees with salaries above department median.",
    "Explain the scaled dot-product attention mechanism in Transformer models with complete equations.",
    "How do TCP sliding window protocols manage congestion control with Reno and Cubic algorithms?",
]

t0 = time.time()
def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def sequence_mult_prob_error(trainer_logp, sampler_logp, mask):
  """Computes Tunix's sequence_mult_prob_error formula."""
  diff = np.abs(trainer_logp - sampler_logp)
  active = mask == 1
  if not np.any(active):
    return 1.0
  active_diff = diff[active]
  # Clamp large diffs to avoid float overflow
  clipped = np.clip(active_diff, 0.0, 70.0)
  return float(np.mean(np.exp(clipped)))


# ==============================================================================
# STAGE: SAMPLER
# ==============================================================================
def stage_sampler(args, out_dir):
  log("=== STAGE: SAMPLER (Concurrent Multi-Turn Rollouts under Mamba Eviction) ===")

  os.environ.setdefault("HF_HOME", "/mnt/disks/persist")
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
  os.environ["NEW_MODEL_DESIGN"] = "1"
  for k, v in {
      "USE_MOE_EP_KERNEL": "0",
      "ATTN_BUCKETIZED_NUM_REQS": "true",
      "ATTN_CUSTOM_NUM_REQS_BUCKETS": str(args.max_num_seqs),
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
      " --xla_tpu_enable_async_collective_merger=false"
      " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false",
  )
  os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"

  from transformers import AutoTokenizer
  import transformers
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  import maxtext_vllm_adapter
  maxtext_vllm_adapter.register()

  from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
  from tunix.rl.agentic import utils as agentic_utils
  from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
  from tunix.rl.agentic.agents import agent_types

  hf_tok = AutoTokenizer.from_pretrained(MODEL_HF)
  tok = tokenizer_adapter_lib.TokenizerAdapter(hf_tok)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(tok)

  mt_cfg = {
      "model_name": MODEL_MAXTEXT,
      "load_parameters_path": args.ckpt,
      "weight_dtype": "bfloat16",
      "dtype": "bfloat16",
      "attention": "vllm_rpa",
      "allow_split_physical_axes": True,
      "scan_layers": False,
      "prefuse_moe_weights": True,
      "enable_dp_attention": True,
      "log_config": False,
      "enable_checkpointing": True,
      "async_checkpointing": False,
      "checkpoint_storage_use_ocdbt": True,
      "checkpoint_storage_use_zarr3": True,
      "convert_checkpoint_if_possible": False,
      "float32_logits": True,
      "float32_gate_logits": True,
      "float32_weight_sum": True,
      "logits_dot_in_fp32": True,
  }

  kw = dict(
      model=MODEL_HF,
      dtype="bfloat16",
      tensor_parallel_size=8,
      enable_expert_parallel=True,
      max_model_len=4096,
      max_num_seqs=args.max_num_seqs,
      max_num_batched_tokens=2048,
      block_size=256,
      enable_chunked_prefill=True,
      enable_prefix_caching=True,
      mamba_cache_mode=args.mamba_cache_mode,
      prefix_cache_retention_interval=0,
      gpu_memory_utilization=args.gpu_memory_utilization,
      language_model_only=True,
      limit_mm_per_prompt={"image": 0, "video": 0},
      disable_log_stats=False,
      kv_cache_dtype="bfloat16",
      reasoning_parser="qwen3",
      seed=0,
      async_scheduling=True,
  )
  sharding = {
      "sharding_strategy": {
          "expert_parallelism": args.ep,
          "tensor_parallelism": 1,
          "enable_dp_attention": True,
      }
  }
  kw["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
  kw["additional_config"] = {
      "maxtext_config": mt_cfg,
      "sharding": sharding,
      "custom_mamba_cache_multiplier": args.custom_mamba_cache_multiplier,
  }

  log("sampler: initializing LLM engine...")
  llm = LLM(**kw)
  log("sampler: LLM engine ready.")

  runner = llm.llm_engine.model_executor.driver_worker.model_runner
  def _text_mrope(input_tokens, input_positions=None):
    pos = np.arange(len(input_tokens), dtype=np.int32)
    return np.stack([pos, pos, pos]), 0
  runner.get_mrope_input_positions_fn = _text_mrope

  # Build 8 concurrent conversation requests that share a common cold prefix
  shared_prefix = (
      "You are an expert systems engineer working on distributed fault-tolerant infrastructure. "
      "Background context: In high-performance distributed systems, state management, "
      "concurrency control, cache coherence, and fault tolerance are foundational requirements. "
      "When designing distributed algorithms, one must carefully account for partial failures, "
      "asynchronous network delays, message reordering, and split-brain scenarios. "
      "Reliable consensus protocols like Paxos and Raft ensure state machine replication even when nodes crash. "
      "In modern AI inference engines, multi-turn state caching and linear attention mechanisms present "
      "similar distributed coordination challenges across tensor-parallel and data-parallel ranks. "
  ) * args.context_repeat

  convs = []
  for i in range(args.num_requests):
    topic = TOPICS[i % len(TOPICS)]
    prompt_text = f"{shared_prefix}\nTask #{i}: Please provide a clear, step-by-step analysis and implementation for: {topic}"
    msgs = [
        {"role": "system", "content": "You are a helpful software engineering assistant."},
        {"role": "user", "content": prompt_text},
    ]
    p_tokens, _ = agentic_utils.tokenize_and_generate_masks(
        msgs, tokenizer=tok, parser=chat_parser, contains_first_msg=True, contains_generation_msg=True
    )
    convs.append({
        "id": i,
        "topic": topic,
        "history_messages": msgs,
        "initial_prompt_tokens": np.array(p_tokens, dtype=np.int32),
        "current_prompt_tokens": p_tokens,
        "trajectory": agent_types.Trajectory(prompt_tokens=np.array(p_tokens, dtype=np.int32)),
        "turns": [],
    })

  sp = SamplingParams(
      max_tokens=args.gen_tokens,
      min_tokens=args.gen_tokens,
      temperature=0.0,
      top_k=-1,
      top_p=1.0,
      logprobs=1,
      ignore_eos=True,
  )

  # ----------------------------------------------------------------------------
  # ROUND 1: Concurrent Cold Prefill
  # ----------------------------------------------------------------------------
  log("\n" + "=" * 80)
  log(f"=== ROUND 1: SUBMITTING ALL {args.num_requests} REQUESTS CONCURRENTLY IN ONE BATCH ===")
  log("=" * 80)

  prompts_r1 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  t_start1 = time.time()
  results_r1 = llm.generate(prompts_r1, sp)
  log(f"Round 1 batch finished in {time.time() - t_start1:.2f}s")

  for i, res in enumerate(results_r1):
    c = convs[i]
    out = res.outputs[0]
    gen_tokens = list(out.token_ids)
    gen_logp = [lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out.logprobs]
    text = hf_tok.decode(gen_tokens)
    log(f"Conv {i} Turn 1 preview: {repr(text[:90])}")

    # Build Turn 1 step
    asst_tokens_1, asst_masks_1 = agentic_utils.tokenize_and_generate_masks(
        [{"role": "assistant", "content": text}],
        tokenizer=tok, parser=chat_parser, contains_first_msg=False, contains_generation_msg=False,
    )
    n_append_1 = len(asst_tokens_1) - len(gen_tokens)
    step1_logp = np.concatenate([np.array(gen_logp, dtype=np.float32), np.zeros(n_append_1, dtype=np.float32)])

    env_msg = {"role": "user", "content": "Now run `pytest -v tests/` and show the output."}
    env_tokens_1, env_masks_1 = agentic_utils.tokenize_and_generate_masks(
        [env_msg], tokenizer=tok, parser=chat_parser, contains_first_msg=False, contains_generation_msg=True,
    )

    step0 = agent_types.Step()
    step0.assistant_tokens = np.array(asst_tokens_1, dtype=np.int32)
    step0.assistant_masks = asst_masks_1
    step0.logprobs = step1_logp
    step0.env_tokens = np.array(env_tokens_1, dtype=np.int32)
    step0.env_masks = env_masks_1
    c["trajectory"].steps.append(step0)

    c["history_messages"].append({"role": "assistant", "content": text})
    c["history_messages"].append(env_msg)
    p2_tokens, _ = agentic_utils.tokenize_and_generate_masks(
        c["history_messages"], tokenizer=tok, parser=chat_parser, contains_first_msg=True, contains_generation_msg=True,
    )
    c["current_prompt_tokens"] = p2_tokens
    c["turns"].append({"turn": 1, "text": text, "gen_tokens": gen_tokens, "logprobs": gen_logp})

  # ----------------------------------------------------------------------------
  # ROUND 2: Concurrent Multi-Turn Continuation
  # ----------------------------------------------------------------------------
  log("\n" + "=" * 80)
  log(f"=== ROUND 2: SUBMITTING ALL {args.num_requests} TURN 2 REQUESTS CONCURRENTLY ===")
  log("=" * 80)

  prompts_r2 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  t_start2 = time.time()
  results_r2 = llm.generate(prompts_r2, sp)
  log(f"Round 2 batch finished in {time.time() - t_start2:.2f}s")

  for i, res in enumerate(results_r2):
    c = convs[i]
    out = res.outputs[0]
    gen_tokens = list(out.token_ids)
    gen_logp = [lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out.logprobs]
    text = hf_tok.decode(gen_tokens)
    log(f"Conv {i} Turn 2 preview: {repr(text[:90])}")

    asst_tokens_2, asst_masks_2 = agentic_utils.tokenize_and_generate_masks(
        [{"role": "assistant", "content": text}],
        tokenizer=tok, parser=chat_parser, contains_first_msg=False, contains_generation_msg=False,
    )
    n_append_2 = len(asst_tokens_2) - len(gen_tokens)
    step2_logp = np.concatenate([np.array(gen_logp, dtype=np.float32), np.zeros(n_append_2, dtype=np.float32)])

    env_msg2 = {"role": "user", "content": "Now run `git diff` and show the changes you made to pass the tests."}
    env_tokens_2, env_masks_2 = agentic_utils.tokenize_and_generate_masks(
        [env_msg2], tokenizer=tok, parser=chat_parser, contains_first_msg=False, contains_generation_msg=True,
    )

    step1 = agent_types.Step()
    step1.assistant_tokens = np.array(asst_tokens_2, dtype=np.int32)
    step1.assistant_masks = asst_masks_2
    step1.logprobs = step2_logp
    step1.env_tokens = np.array(env_tokens_2, dtype=np.int32)
    step1.env_masks = env_masks_2
    c["trajectory"].steps.append(step1)

    c["history_messages"].append({"role": "assistant", "content": text})
    c["history_messages"].append(env_msg2)
    p3_tokens, _ = agentic_utils.tokenize_and_generate_masks(
        c["history_messages"], tokenizer=tok, parser=chat_parser, contains_first_msg=True, contains_generation_msg=True,
    )
    c["current_prompt_tokens"] = p3_tokens
    c["turns"].append({"turn": 2, "text": text, "gen_tokens": gen_tokens, "logprobs": gen_logp})

  # ----------------------------------------------------------------------------
  # ROUND 3: Concurrent Multi-Turn Continuation (Forces Eviction & Contamination)
  # ----------------------------------------------------------------------------
  log("\n" + "=" * 80)
  log(f"=== ROUND 3: SUBMITTING ALL {args.num_requests} TURN 3 REQUESTS CONCURRENTLY ===")
  log("=" * 80)

  prompts_r3 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  t_start3 = time.time()
  results_r3 = llm.generate(prompts_r3, sp)
  log(f"Round 3 batch finished in {time.time() - t_start3:.2f}s")

  for i, res in enumerate(results_r3):
    c = convs[i]
    out = res.outputs[0]
    gen_tokens = list(out.token_ids)
    gen_logp = [lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out.logprobs]
    text = hf_tok.decode(gen_tokens)
    log(f"Conv {i} Turn 3 preview: {repr(text[:90])}")

    asst_tokens_3, asst_masks_3 = agentic_utils.tokenize_and_generate_masks(
        [{"role": "assistant", "content": text}],
        tokenizer=tok, parser=chat_parser, contains_first_msg=False, contains_generation_msg=False,
    )
    n_append_3 = len(asst_tokens_3) - len(gen_tokens)
    step3_logp = np.concatenate([np.array(gen_logp, dtype=np.float32), np.zeros(n_append_3, dtype=np.float32)])

    step2 = agent_types.Step()
    step2.assistant_tokens = np.array(asst_tokens_3, dtype=np.int32)
    step2.assistant_masks = asst_masks_3
    step2.logprobs = step3_logp
    step2.env_tokens = None
    step2.env_masks = None
    c["trajectory"].steps.append(step2)
    c["turns"].append({"turn": 3, "text": text, "gen_tokens": gen_tokens, "logprobs": gen_logp})

    # Assemble and save full Tunix trajectory for each conversation
    conv_tokens = []
    conv_masks = []
    conv_logprobs = []
    turn_slices = []
    curr_idx = 0

    for s_idx, step in enumerate(c["trajectory"].steps):
      t_start = curr_idx
      if step.assistant_tokens is not None:
        conv_tokens.append(step.assistant_tokens)
        conv_masks.append(step.assistant_masks)
        conv_logprobs.append(step.logprobs)
        curr_idx += len(step.assistant_tokens)
      t_asst_end = curr_idx

      t_env_start = curr_idx
      if step.env_tokens is not None:
        conv_tokens.append(step.env_tokens)
        conv_masks.append(step.env_masks)
        conv_logprobs.append(np.zeros(len(step.env_tokens), dtype=np.float32))
        curr_idx += len(step.env_tokens)
      t_env_end = curr_idx

      turn_slices.append({
          "turn": s_idx + 1,
          "asst_slice": (t_start, t_asst_end),
          "env_slice": (t_env_start, t_env_end),
      })

    prompt_toks = c["initial_prompt_tokens"]
    comp_toks = np.concatenate(conv_tokens).astype(np.int32)
    comp_msks = np.concatenate(conv_masks).astype(np.int32)
    comp_lps = np.concatenate(conv_logprobs).astype(np.float32)

    rollout_path = os.path.join(out_dir, f"rollout_conv_{i}.npz")
    np.savez(
        rollout_path,
        prompt_tokens=prompt_toks,
        conversation_tokens=comp_toks,
        conversation_masks=comp_msks,
        sampler_logprobs=comp_lps,
        turn_slices_json=json.dumps(turn_slices),
    )
    log(f"Conv {i}: saved rollout to {rollout_path} (prompt={len(prompt_toks)}, comp={len(comp_toks)})")

  summary_path = os.path.join(out_dir, "sampler_summary.json")
  with open(summary_path, "w") as f:
    json.dump([{"id": c["id"], "topic": c["topic"], "turns": [t["text"] for t in c["turns"]]} for c in convs], f, indent=2)
  log(f"Saved sampler summary to {summary_path}")


# ==============================================================================
# STAGE: TRAINER
# ==============================================================================
def stage_trainer(args, out_dir):
  log("=== STAGE: TRAINER (MaxText Teacher-Forcing Forward Pass on Rollouts) ===")

  eval_convs = [int(x) for x in args.eval_convs.split(",") if x.strip()]
  log(f"Evaluating conversations: {eval_convs}")

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

  # Find max sequence length across evaluated rollouts to determine compilation bucket
  max_seq_len = 0
  conv_data = {}
  for cid in eval_convs:
    rpath = os.path.join(out_dir, f"rollout_conv_{cid}.npz")
    assert os.path.exists(rpath), f"Missing {rpath}"
    data = np.load(rpath)
    total_len = len(data["prompt_tokens"]) + len(data["conversation_tokens"])
    max_seq_len = max(max_seq_len, total_len)
    conv_data[cid] = data

  padded_seq_len = ((max_seq_len + 511) // 512) * 512
  ep_size = args.ep or 1
  micro = max(ep_size, args.trainer_micro_batch or ep_size)
  if micro % ep_size != 0:
    micro = ((micro + ep_size - 1) // ep_size) * ep_size

  log(f"trainer: max sequence length {max_seq_len}, compiling at bucket {padded_seq_len}, micro_batch={micro}")

  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml, "attention=flash"],
      model_name=MODEL_MAXTEXT,
      load_parameters_path=args.ckpt,
      tokenizer_path=MODEL_HF,
      run_name="concurrency_trainer",
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

  log("trainer: loading Flax NNX model weights...")
  model, mesh = model_creation_utils.from_pretrained(
      cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN
  )
  log("trainer: model loaded.")
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

  with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))

    for cid in eval_convs:
      log(f"\n--- Trainer forward pass on Conv {cid} ---")
      rdata = conv_data[cid]
      p_tokens = rdata["prompt_tokens"]
      c_tokens = rdata["conversation_tokens"]
      orig_len = len(p_tokens) + len(c_tokens)

      full_tokens = np.concatenate([p_tokens, c_tokens])[None, :].astype(np.int32)
      padded_tokens = np.pad(full_tokens, ((0, 0), (0, padded_seq_len - orig_len)), mode="constant")
      if micro > 1:
        padded_tokens = np.repeat(padded_tokens, micro, axis=0)

      nxt_np = np.roll(padded_tokens, -1, axis=1)
      pos_np = np.broadcast_to(np.arange(padded_seq_len, dtype=np.int32), (micro, padded_seq_len))
      seg_np = np.ones((micro, padded_seq_len), np.int32)

      tokens = jax.device_put(padded_tokens, sh)
      nxt = jax.device_put(nxt_np, sh)
      pos = jax.device_put(pos_np, sh)
      seg = jax.device_put(seg_np, sh)

      t_fwd_start = time.time()
      lp_d, _ = fwd_logp(st, tokens, pos, seg, nxt)
      jax.block_until_ready(lp_d)
      log(f"Forward pass completed in {time.time() - t_fwd_start:.2f}s")

      full_logp = np.asarray(lp_d)[0, :orig_len]
      n_p = len(p_tokens)
      n_c = len(c_tokens)
      trainer_comp_logp = full_logp[n_p - 1 : n_p + n_c - 1]

      tpath = os.path.join(out_dir, f"trainer_conv_{cid}.npz")
      np.savez(
          tpath,
          trainer_comp_logp=trainer_comp_logp,
          full_logp=full_logp,
      )
      log(f"Saved trainer logprobs to {tpath} (comp tokens={len(trainer_comp_logp)})")


# ==============================================================================
# STAGE: COMPARE
# ==============================================================================
def stage_compare(args, out_dir):
  log("=== STAGE: COMPARE (Tunix Sequence Mult Prob Error Comparison) ===")

  eval_convs = [int(x) for x in args.eval_convs.split(",") if x.strip()]
  results = []

  for cid in eval_convs:
    rpath = os.path.join(out_dir, f"rollout_conv_{cid}.npz")
    tpath = os.path.join(out_dir, f"trainer_conv_{cid}.npz")
    if not (os.path.exists(rpath) and os.path.exists(tpath)):
      log(f"Skipping Conv {cid}: missing files")
      continue

    rdata = np.load(rpath)
    tdata = np.load(tpath)

    comp_tokens = rdata["conversation_tokens"]
    comp_masks = rdata["conversation_masks"]
    sampler_logp = rdata["sampler_logprobs"]
    trainer_logp = tdata["trainer_comp_logp"]
    turn_slices = json.loads(str(rdata["turn_slices_json"]))

    diff = np.abs(trainer_logp - sampler_logp)
    active_mask = (comp_masks == 1)

    overall_err = sequence_mult_prob_error(trainer_logp, sampler_logp, comp_masks)

    log("\n" + "=" * 80)
    log(f"=== CONVERSATION {cid} PARITY REPORT ===")
    log("=" * 80)
    log(f"Total completion tokens: {len(comp_tokens)}, Active tokens (mask==1): {active_mask.sum()}")
    log(f"OVERALL sequence_mult_prob_error: {overall_err:,.2f}")

    turn_metrics = []
    for tinfo in turn_slices:
      tnum = tinfo["turn"]
      astart, aend = tinfo["asst_slice"]
      sub_mask = comp_masks[astart:aend]
      sub_trainer_lp = trainer_logp[astart:aend]
      sub_sampler_lp = sampler_logp[astart:aend]
      sub_err = sequence_mult_prob_error(sub_trainer_lp, sub_sampler_lp, sub_mask)
      sub_diff = diff[astart:aend][sub_mask == 1]
      max_d = float(sub_diff.max()) if len(sub_diff) > 0 else 0.0
      mean_d = float(sub_diff.mean()) if len(sub_diff) > 0 else 0.0

      # Exact generated tokens slice (excluding chat template formatting wrapper)
      gen_len = max(0, (aend - astart) - 5)
      gen_s_lp = sub_sampler_lp[:gen_len]
      gen_t_lp = sub_trainer_lp[4 : 4 + gen_len]
      gen_diff = np.abs(gen_t_lp - gen_s_lp)
      gen_err = float(np.mean(np.exp(np.clip(gen_diff, 0.0, 70.0)))) if len(gen_diff) > 0 else 1.0
      gen_max_d = float(gen_diff.max()) if len(gen_diff) > 0 else 0.0

      log(f"  Turn {tnum}:")
      log(f"    - Full Assistant Block (with template tokens): mult_prob_err = {sub_err:,.2f}, max |delta| = {max_d:.3f}")
      log(f"    - Exact Generated Tokens (sampler output tokens): mult_prob_err = {gen_err:,.2f}, max |delta| = {gen_max_d:.3f}")

      turn_metrics.append({
          "turn": tnum,
          "full_block_mult_prob_err": sub_err,
          "exact_gen_mult_prob_err": gen_err,
          "exact_gen_max_diff": gen_max_d,
          "mean_diff": mean_d,
          "max_diff": max_d,
      })

    results.append({
        "conv_id": cid,
        "overall_mult_prob_error": overall_err,
        "turns": turn_metrics,
    })

  log("\n" + "=" * 80)
  log("=== SUMMARY COMPARISON: CLEAN VS CONTAMINATED CONVERSATIONS ===")
  log("=" * 80)
  for r in results:
    cid = r["conv_id"]
    t1_err = r["turns"][0]["exact_gen_mult_prob_err"] if len(r["turns"]) >= 1 else 0.0
    t2_err = r["turns"][1]["exact_gen_mult_prob_err"] if len(r["turns"]) >= 2 else 0.0
    t3_err = r["turns"][2]["exact_gen_mult_prob_err"] if len(r["turns"]) >= 3 else 0.0
    status = "CONTAMINATED / EXPLODED" if max(t1_err, t2_err, t3_err) > 10.0 else "CLEAN (PARITY OK)"
    log(f"Conv {cid}: Turn 1 err = {t1_err:,.2f} | Turn 2 err = {t2_err:,.2f} | Turn 3 err = {t3_err:,.2f} [{status}]")

  out_summary = os.path.join(out_dir, "parity_comparison_summary.json")
  with open(out_summary, "w") as f:
    json.dump(results, f, indent=2)
  log(f"\nSaved comparison summary to {out_summary}")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
  ap = argparse.ArgumentParser(description="Multi-turn Concurrency & Eviction Logprob Parity Microbenchmark")
  ap.add_argument("--stage", default="all", choices=["all", "sampler", "trainer", "compare"])
  ap.add_argument("--out-dir", default="/mnt/disks/persist/concurrency_bench")
  ap.add_argument("--max-num-seqs", type=int, default=2)
  ap.add_argument("--num-requests", type=int, default=8)
  ap.add_argument("--custom-mamba-cache-multiplier", type=int, default=1)
  ap.add_argument("--ep", type=int, default=8)
  ap.add_argument("--trainer-tp", type=int, default=1)
  ap.add_argument("--trainer-micro-batch", type=int, default=8)
  ap.add_argument("--mamba-cache-mode", default="align", choices=["align", "none"])
  ap.add_argument("--gen-tokens", type=int, default=64)
  ap.add_argument("--context-repeat", type=int, default=5)
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
  ap.add_argument("--ckpt", default=CKPT)
  ap.add_argument("--logits-dot-fp32", action="store_true", default=True)
  ap.add_argument("--eval-convs", default="0,1,6,7", help="comma-separated conv IDs to evaluate in trainer")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  py = sys.executable

  if args.stage == "all":
    log("=== Running Stage 1: SAMPLER in dedicated subprocess ===")
    cmd_s = [
        py, sys.argv[0], "--stage", "sampler",
        "--out-dir", args.out_dir,
        "--max-num-seqs", str(args.max_num_seqs),
        "--num-requests", str(args.num_requests),
        "--custom-mamba-cache-multiplier", str(args.custom_mamba_cache_multiplier),
        "--ep", str(args.ep),
        "--mamba-cache-mode", args.mamba_cache_mode,
        "--gen-tokens", str(args.gen_tokens),
        "--context-repeat", str(args.context_repeat),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--ckpt", args.ckpt,
    ]
    subprocess.check_call(cmd_s)

    log("\n=== Running Stage 2: TRAINER in dedicated subprocess ===")
    cmd_t = [
        py, sys.argv[0], "--stage", "trainer",
        "--out-dir", args.out_dir,
        "--eval-convs", args.eval_convs,
        "--ep", str(args.ep),
        "--trainer-tp", str(args.trainer_tp),
        "--trainer-micro-batch", str(args.trainer_micro_batch),
        "--ckpt", args.ckpt,
    ]
    subprocess.check_call(cmd_t)

    log("\n=== Running Stage 3: COMPARE ===")
    stage_compare(args, args.out_dir)

  elif args.stage == "sampler":
    stage_sampler(args, args.out_dir)
  elif args.stage == "trainer":
    stage_trainer(args, args.out_dir)
  elif args.stage == "compare":
    stage_compare(args, args.out_dir)


if __name__ == "__main__":
  main()
