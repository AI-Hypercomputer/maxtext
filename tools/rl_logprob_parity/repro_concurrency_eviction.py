#!/usr/bin/env python3
"""Reproduction benchmark: Concurrency + Mamba Cache Eviction in Multi-turn Rollouts.

Tests whether concurrent multi-turn requests (max_num_seqs=2 per rank, attn_dp=4,
8 concurrent requests total) with small custom_mamba_cache_multiplier=1 cause
Mamba cache exhaustion/eviction races that produce non-coherent/corrupted responses.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from transformers import AutoTokenizer

os.environ.setdefault("HF_HOME", "/mnt/disks/persist")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["NEW_MODEL_DESIGN"] = "1"
for k, v in {
    "USE_MOE_EP_KERNEL": "0",
    "ATTN_BUCKETIZED_NUM_REQS": "true",
    "ATTN_CUSTOM_NUM_REQS_BUCKETS": "2",
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
os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"

t0 = time.time()
def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

MODEL_HF = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT = "qwen3.5-35b-a3b"
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"

TOPICS = [
    "Write a Python script to sort files by modification date and size.",
    "Explain how the Raft consensus algorithm handles network partitions and election timeouts.",
    "Implement a balanced binary search tree with insert, delete, and rebalancing operations.",
    "How does the Linux virtual filesystem (VFS) handle dentry and inode caching under heavy I/O?",
    "Describe the difference between process memory segments: text, data, bss, heap, and stack.",
    "Write an SQL query with window functions to find employees with salaries above department median.",
    "Explain the scaled dot-product attention mechanism in Transformer models with complete equations.",
    "How do TCP sliding window protocols manage congestion control with Reno and Cubic algorithms?",
    "Implement a thread-safe LRU cache in C++ with mutex locks and condition variables.",
    "What are the architecture trade-offs between B-trees and LSM-trees in modern storage engines?",
    "Write a bash script to monitor disk usage, parse df output, and send email alerts on thresholds.",
    "Explain the mathematical principles of zero-knowledge proofs and zk-SNARKs in cryptography.",
]

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--max-num-seqs", type=int, default=2, help="max_num_seqs per DP rank")
  ap.add_argument("--num-requests", type=int, default=8, help="number of concurrent requests")
  ap.add_argument("--custom-mamba-cache-multiplier", type=int, default=1)
  ap.add_argument("--ep", type=int, default=8, help="expert parallelism (8 gives attn_dp_expert=4 on 8 devices)")
  ap.add_argument("--gen-tokens", type=int, default=64)
  ap.add_argument("--context-repeat", type=int, default=5, help="context repetition for prompt length")
  ap.add_argument("--out-dir", default="/mnt/disks/persist/concurrency_repro")
  args = ap.parse_args()
  os.environ["ATTN_CUSTOM_NUM_REQS_BUCKETS"] = str(args.max_num_seqs)

  os.makedirs(args.out_dir, exist_ok=True)

  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt
  import maxtext_vllm_adapter
  maxtext_vllm_adapter.register()

  from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
  from tunix.rl.agentic import utils as agentic_utils
  from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

  hf_tok = AutoTokenizer.from_pretrained(MODEL_HF)
  tok = tokenizer_adapter_lib.TokenizerAdapter(hf_tok)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(tok)

  log(f"Configuration: max_num_seqs={args.max_num_seqs}, num_requests={args.num_requests}, "
      f"custom_mamba_cache_multiplier={args.custom_mamba_cache_multiplier}, ep={args.ep}")

  mt_cfg = {
      "model_name": MODEL_MAXTEXT,
      "load_parameters_path": CKPT,
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
      mamba_cache_mode="align",
      prefix_cache_retention_interval=0,
      gpu_memory_utilization=0.5,
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

  log("Initializing LLM engine with concurrent batching and align mamba mode...")
  llm = LLM(**kw)
  log("LLM engine ready.")

  runner = llm.llm_engine.model_executor.driver_worker.model_runner
  for obj in (
      runner,
      getattr(runner, "persistent_batch_manager", None),
      getattr(runner, "input_batch", None),
  ):
    if obj is not None and hasattr(obj, "uses_mrope"):
      obj.uses_mrope = False

  def _text_mrope(prompt_token_ids, mm_features):
    pos = np.arange(len(prompt_token_ids), dtype=np.int64)
    return np.stack([pos, pos, pos]), 0

  runner.get_mrope_input_positions_fn = _text_mrope

  # Build 8 concurrent conversation requests that share a long prefix (like GRPO rollouts)
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
        "current_prompt_tokens": p_tokens,
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

  # ROUND 1: Submit ALL 8 requests CONCURRENTLY in a single batch
  log("\n" + "=" * 80)
  log(f"=== ROUND 1: SUBMITTING ALL {args.num_requests} REQUESTS CONCURRENTLY IN ONE BATCH ===")
  log("=" * 80)

  prompts_r1 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  log(f"Batching {len(prompts_r1)} prompts (lengths: {[len(c['current_prompt_tokens']) for c in convs]})...")
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
    c["turns"].append({"turn": 1, "gen_tokens": gen_tokens, "logprobs": gen_logp, "text": text})

    # Prepare Turn 2: User observation
    env_msg = {"role": "user", "content": "Now run `pytest -v tests/` and show the output."}
    c["history_messages"].append({"role": "assistant", "content": text})
    c["history_messages"].append(env_msg)
    p2_tokens, _ = agentic_utils.tokenize_and_generate_masks(
        c["history_messages"], tokenizer=tok, parser=chat_parser, contains_first_msg=True, contains_generation_msg=True
    )
    c["current_prompt_tokens"] = p2_tokens

  # ROUND 2: Submit ALL 8 Turn-2 requests CONCURRENTLY in one batch
  log("\n" + "=" * 80)
  log(f"=== ROUND 2: SUBMITTING ALL {args.num_requests} TURN 2 REQUESTS CONCURRENTLY ===")
  log("=== (Forces concurrent decodes & chunked prefills against tiny Mamba pool) ===")
  log("=" * 80)

  prompts_r2 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  log(f"Batching {len(prompts_r2)} Turn 2 prompts (lengths: {[len(c['current_prompt_tokens']) for c in convs]})...")
  t_start2 = time.time()
  results_r2 = llm.generate(prompts_r2, sp)
  log(f"Round 2 batch finished in {time.time() - t_start2:.2f}s")

  gibberish_r2 = 0
  for i, res in enumerate(results_r2):
    c = convs[i]
    out = res.outputs[0]
    gen_tokens = list(out.token_ids)
    gen_logp = [lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out.logprobs]
    text = hf_tok.decode(gen_tokens)
    log(f"Conv {i} Turn 2 preview: {repr(text[:120])}")
    c["turns"].append({"turn": 2, "gen_tokens": gen_tokens, "logprobs": gen_logp, "text": text})

    suspicious = any(pat in text for pat in [
        "mesara", "str_replace>", "<function=", "!✍", "</a--parameter>", "<parameter.tar.gz}",
        "\\n\\n!✍", "BEGIN FUNCTION", "<count", "<bash activate_editor>", "! !"
    ])
    if suspicious or len(set(gen_tokens)) < 15 or text.count("!") > 5:
      log(f"*** GIBBERISH / CORRUPTION DETECTED in Conv {i} Turn 2! ***")
      log(f"Full text: {repr(text)}")
      gibberish_r2 += 1

    # Prepare Turn 3
    env_msg3 = {"role": "user", "content": "Observation: All tests passed. Now show git diff."}
    c["history_messages"].append({"role": "assistant", "content": text})
    c["history_messages"].append(env_msg3)
    p3_tokens, _ = agentic_utils.tokenize_and_generate_masks(
        c["history_messages"], tokenizer=tok, parser=chat_parser, contains_first_msg=True, contains_generation_msg=True
    )
    c["current_prompt_tokens"] = p3_tokens

  # ROUND 3: Submit ALL 8 Turn-3 requests CONCURRENTLY in one batch
  log("\n" + "=" * 80)
  log(f"=== ROUND 3: SUBMITTING ALL {args.num_requests} TURN 3 REQUESTS CONCURRENTLY ===")
  log("=" * 80)

  prompts_r3 = [TokensPrompt(prompt_token_ids=c["current_prompt_tokens"]) for c in convs]
  t_start3 = time.time()
  results_r3 = llm.generate(prompts_r3, sp)
  log(f"Round 3 batch finished in {time.time() - t_start3:.2f}s")

  gibberish_r3 = 0
  for i, res in enumerate(results_r3):
    c = convs[i]
    out = res.outputs[0]
    gen_tokens = list(out.token_ids)
    gen_logp = [lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out.logprobs]
    text = hf_tok.decode(gen_tokens)
    log(f"Conv {i} Turn 3 preview: {repr(text[:120])}")
    c["turns"].append({"turn": 3, "gen_tokens": gen_tokens, "logprobs": gen_logp, "text": text})

    suspicious = any(pat in text for pat in [
        "mesara", "str_replace>", "<function=", "!✍", "</a--parameter>", "<parameter.tar.gz}",
        "\\n\\n!✍", "BEGIN FUNCTION", "<count", "<bash activate_editor>", "! !"
    ])
    if suspicious or len(set(gen_tokens)) < 15 or text.count("!") > 5:
      log(f"*** GIBBERISH / CORRUPTION DETECTED in Conv {i} Turn 3! ***")
      log(f"Full text: {repr(text)}")
      gibberish_r3 += 1

  log("\n" + "=" * 80)
  log(f"=== FINAL RESULTS ===")
  log(f"Round 2 corrupted requests: {gibberish_r2}/{args.num_requests}")
  log(f"Round 3 corrupted requests: {gibberish_r3}/{args.num_requests}")
  log("=" * 80)

  out_file = os.path.join(args.out_dir, f"concurrency_repro_results.json")
  with open(out_file, "w") as f:
    json.dump([{"id": c["id"], "turns": [t["text"] for t in c["turns"]]} for c in convs], f, indent=2)
  log(f"Saved results to {out_file}")

if __name__ == "__main__":
  main()
