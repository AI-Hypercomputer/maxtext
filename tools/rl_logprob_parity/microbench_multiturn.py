#!/usr/bin/env python3
"""Multi-turn RL Logprob Parity Microbenchmark using Real Tunix Code & MaxText.

Tests whether multi-turn chat interactions with prefix caching and Tunix token
continuity introduce logprob parity discrepancies between the vLLM rollout
sampler and the MaxText teacher-forcing trainer.

Pipeline:
  1. stage sampler: Runs a 3-turn agentic rollout using real Tunix chat parser,
     exact token continuity, and vLLM sampler with prefix caching enabled.
  2. stage trainer: Runs MaxText teacher-forced forward pass on the exact
     concatenated sequence ([prompt_0, asst_0, env_0, asst_1, env_1, asst_2]).
  3. stage compare: Computes sequence_mult_prob_error per turn and overall,
     verifying if errors pass the --seq_logprob_error_threshold (2.0).
"""

import argparse
import json
import os
import sys
import time
from typing import Any
import numpy as np

CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
MODEL_HF = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT = "qwen3.5-35b-a3b"

t0 = time.time()


def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def resolve_paths(args):
  maxtext_root = (
      args.maxtext_root
      or os.environ.get("MAXTEXT_ROOT")
      or "/home/wenxindong_google_com/maxtext"
  )
  tunix_root = (
      args.tunix_root
      or os.environ.get("TUNIX_ROOT")
      or "/home/wenxindong_google_com/tunix"
  )
  hf_home = (
      args.hf_home
      or os.environ.get("HF_HOME")
      or "/mnt/disks/persist"
  )
  out_dir = args.out_dir or "/mnt/disks/persist/multiturn_bench"
  os.makedirs(out_dir, exist_ok=True)
  return maxtext_root, tunix_root, hf_home, out_dir


def set_tpu_env(hf_home, sampler="adapter"):
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


def sequence_mult_prob_error(
    trainer_logps: np.ndarray,
    sampler_logps: np.ndarray,
    completion_mask: np.ndarray,
) -> float:
  """Exact formula from Tunix algorithm_adapter.py: sequence_mult_prob_error."""
  log_is_raw = trainer_logps - sampler_logps
  log_is = log_is_raw * completion_mask
  mult_prob_err = np.exp(np.abs(log_is))
  valid_count = np.sum(completion_mask)
  if valid_count > 0:
    return float(np.sum(mult_prob_err * completion_mask) / valid_count)
  return 0.0


# ==============================================================================
# STAGE: SAMPLER
# ==============================================================================
def stage_sampler(args, maxtext_root, tunix_root, hf_home, out_dir):
  log("=== STAGE: SAMPLER (Multi-turn with Real Tunix Continuity) ===")

  import transformers
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt

  sys.path.insert(0, tunix_root)
  from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
  from tunix.rl.agentic import utils as agentic_utils
  from tunix.rl.agentic.agents import agent_types
  from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

  sys.path.insert(
      0, os.path.join(maxtext_root, "src", "maxtext", "integration", "vllm")
  )
  import maxtext_vllm_adapter

  maxtext_vllm_adapter.register()

  hf_tok = transformers.AutoTokenizer.from_pretrained(
      MODEL_HF,
      cache_dir=os.path.join(hf_home, "hub"),
      local_files_only=True,
      trust_remote_code=True,
  )
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
      "enable_dp_attention": args.ep > 1,
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

  kw = dict(
      model=MODEL_HF,
      dtype="bfloat16",
      tensor_parallel_size=8,
      enable_expert_parallel=args.ep > 1,
      max_model_len=4096,
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
  sharding = {
      "sharding_strategy": {
          "expert_parallelism": args.ep,
          "tensor_parallelism": args.sharding_tp,
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
  log("sampler: LLM engine ready")

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

  # Build 3-turn interactive trajectory
  trajectory = agent_types.Trajectory()

  # Turn 1: Initial user message
  messages_turn1 = [
      {"role": "system", "content": "You are a helpful coding assistant pair programming with an engineer."},
      {"role": "user", "content": "Please inspect repository structure and check for any failing tests or syntax issues."},
  ]
  p1_tokens, _ = agentic_utils.tokenize_and_generate_masks(
      messages_turn1,
      tokenizer=tok,
      parser=chat_parser,
      contains_first_msg=True,
      contains_generation_msg=True,
  )
  trajectory.prompt_tokens = np.array(p1_tokens, dtype=np.int32)
  trajectory.prompt_length = len(p1_tokens)
  log(f"Turn 1 prompt tokens: {len(p1_tokens)}")

  sp = SamplingParams(
      max_tokens=args.gen_tokens,
      min_tokens=args.gen_tokens,
      temperature=args.gen_temperature,
      top_k=-1,
      top_p=1.0,
      logprobs=1,
      ignore_eos=True,
  )

  # Sample Turn 1
  log("sampler: sampling Turn 1 (COLD prefill)...")
  res1 = llm.generate([TokensPrompt(prompt_token_ids=p1_tokens)], sp)
  out1 = res1[0].outputs[0]
  gen_tokens_1 = list(out1.token_ids)
  gen_logp_1 = [
      lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out1.logprobs
  ]

  # Apply Tunix chat parser and exact token continuity
  asst_tokens_1, n_append_1 = chat_parser.update_assistant_end_tokens(gen_tokens_1)
  asst_tokens_1 = agentic_utils.assistant_with_suffix(
      gen_tokens_1, asst_tokens_1, n_append_1
  )
  asst_masks_1 = np.concatenate([
      np.ones(len(gen_tokens_1), dtype=np.int32),
      np.zeros(n_append_1, dtype=np.int32),
  ])
  step0_logp = np.concatenate([
      np.array(gen_logp_1, dtype=np.float32),
      np.zeros(n_append_1, dtype=np.float32),
  ])

  # Env Observation 1
  env_msgs_1 = [
      {"role": "user", "content": "Observation: Found README.md, src/main.py, tests/test_main.py. Running pytest..."}
  ]
  e_tokens_1, e_masks_1 = agentic_utils.tokenize_and_generate_masks(
      env_msgs_1,
      tokenizer=tok,
      parser=chat_parser,
      contains_first_msg=False,
      contains_generation_msg=True,
  )

  step0 = agent_types.Step()
  step0.assistant_tokens = np.array(asst_tokens_1, dtype=np.int32)
  step0.assistant_masks = asst_masks_1
  step0.logprobs = step0_logp
  step0.env_tokens = np.array(e_tokens_1, dtype=np.int32)
  step0.env_masks = np.array(e_masks_1, dtype=np.int32)
  trajectory.steps.append(step0)
  log(f"Turn 1 complete: asst_tokens={len(asst_tokens_1)} (gen={len(gen_tokens_1)}, append={n_append_1}), env_tokens={len(e_tokens_1)}")

  # Turn 2: Continuation prompt
  turn2_prompt = agentic_utils.continuation_prompt_tokens(trajectory)
  log(f"Turn 2 prompt tokens: {len(turn2_prompt)} (WARM prefix cache hit on {len(p1_tokens)} tokens)")

  log("sampler: sampling Turn 2 (prefix cache hit)...")
  res2 = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in turn2_prompt])], sp)
  out2 = res2[0].outputs[0]
  gen_tokens_2 = list(out2.token_ids)
  gen_logp_2 = [
      lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out2.logprobs
  ]

  asst_tokens_2, n_append_2 = chat_parser.update_assistant_end_tokens(gen_tokens_2)
  asst_tokens_2 = agentic_utils.assistant_with_suffix(
      gen_tokens_2, asst_tokens_2, n_append_2
  )
  asst_masks_2 = np.concatenate([
      np.ones(len(gen_tokens_2), dtype=np.int32),
      np.zeros(n_append_2, dtype=np.int32),
  ])
  step1_logp = np.concatenate([
      np.array(gen_logp_2, dtype=np.float32),
      np.zeros(n_append_2, dtype=np.float32),
  ])

  # Env Observation 2
  env_msgs_2 = [
      {"role": "user", "content": "Observation: Pytest output: 14 passed in 0.42s. All tests passing cleanly."}
  ]
  e_tokens_2, e_masks_2 = agentic_utils.tokenize_and_generate_masks(
      env_msgs_2,
      tokenizer=tok,
      parser=chat_parser,
      contains_first_msg=False,
      contains_generation_msg=True,
  )

  step1 = agent_types.Step()
  step1.assistant_tokens = np.array(asst_tokens_2, dtype=np.int32)
  step1.assistant_masks = asst_masks_2
  step1.logprobs = step1_logp
  step1.env_tokens = np.array(e_tokens_2, dtype=np.int32)
  step1.env_masks = np.array(e_masks_2, dtype=np.int32)
  trajectory.steps.append(step1)
  log(f"Turn 2 complete: asst_tokens={len(asst_tokens_2)} (gen={len(gen_tokens_2)}, append={n_append_2}), env_tokens={len(e_tokens_2)}")

  # Turn 3: Continuation prompt (Terminal)
  turn3_prompt = agentic_utils.continuation_prompt_tokens(trajectory)
  log(f"Turn 3 prompt tokens: {len(turn3_prompt)} (WARM prefix cache hit on {len(turn2_prompt)} tokens)")

  log("sampler: sampling Turn 3 (prefix cache hit)...")
  res3 = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in turn3_prompt])], sp)
  out3 = res3[0].outputs[0]
  gen_tokens_3 = list(out3.token_ids)
  gen_logp_3 = [
      lp[next(iter(lp.keys()))].logprob if lp else 0.0 for lp in out3.logprobs
  ]

  asst_tokens_3, n_append_3 = chat_parser.update_assistant_end_tokens(gen_tokens_3)
  asst_tokens_3 = agentic_utils.assistant_with_suffix(
      gen_tokens_3, asst_tokens_3, n_append_3
  )
  asst_masks_3 = np.concatenate([
      np.ones(len(gen_tokens_3), dtype=np.int32),
      np.zeros(n_append_3, dtype=np.int32),
  ])
  step2_logp = np.concatenate([
      np.array(gen_logp_3, dtype=np.float32),
      np.zeros(n_append_3, dtype=np.float32),
  ])

  step2 = agent_types.Step()
  step2.assistant_tokens = np.array(asst_tokens_3, dtype=np.int32)
  step2.assistant_masks = asst_masks_3
  step2.logprobs = step2_logp
  step2.env_tokens = None
  step2.env_masks = None
  trajectory.steps.append(step2)
  log(f"Turn 3 complete: asst_tokens={len(asst_tokens_3)} (gen={len(gen_tokens_3)}, append={n_append_3})")

  # Assemble full conversation trajectory exactly matching trajectory_collect_engine.py
  conversation_tokens = []
  conversation_masks = []
  logprobs = []

  turn_slices = []
  curr_idx = 0

  for idx, step in enumerate(trajectory.steps):
    t_start = curr_idx
    if step.assistant_tokens is not None:
      conversation_tokens.append(step.assistant_tokens)
      conversation_masks.append(step.assistant_masks)
      logprobs.append(step.logprobs)
      curr_idx += len(step.assistant_tokens)
    t_asst_end = curr_idx

    t_env_start = curr_idx
    if step.env_tokens is not None:
      conversation_tokens.append(step.env_tokens)
      conversation_masks.append(step.env_masks)
      logprobs.append(np.zeros(len(step.env_tokens), dtype=np.float32))
      curr_idx += len(step.env_tokens)
    t_env_end = curr_idx

    turn_slices.append({
        "turn": idx + 1,
        "asst_slice": (t_start, t_asst_end),
        "env_slice": (t_env_start, t_env_end),
    })

  prompt_tokens = trajectory.prompt_tokens
  comp_tokens = np.concatenate(conversation_tokens).astype(np.int32)
  comp_masks = np.concatenate(conversation_masks).astype(np.int32)
  comp_logprobs = np.concatenate(logprobs).astype(np.float32)

  log(f"Trajectory summary: prompt={len(prompt_tokens)}, comp={len(comp_tokens)}, total={len(prompt_tokens) + len(comp_tokens)}")
  log(f"Total active completion mask: {comp_masks.sum()} / {len(comp_masks)}")

  out_path = os.path.join(out_dir, "multiturn_rollout.npz")
  np.savez(
      out_path,
      prompt_tokens=prompt_tokens,
      conversation_tokens=comp_tokens,
      conversation_masks=comp_masks,
      sampler_logprobs=comp_logprobs,
      turn_slices_json=json.dumps(turn_slices),
  )
  log(f"sampler: saved rollout to {out_path}")
  return out_path


# ==============================================================================
# STAGE: TRAINER
# ==============================================================================
def stage_trainer(args, maxtext_root, tunix_root, hf_home, out_dir):
  log("=== STAGE: TRAINER (MaxText Teacher-Forcing Forward Pass) ===")

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
  rollout_path = os.path.join(out_dir, "multiturn_rollout.npz")
  rdata = np.load(rollout_path)
  prompt_tokens = rdata["prompt_tokens"]
  comp_tokens = rdata["conversation_tokens"]

  full_tokens = np.concatenate([prompt_tokens, comp_tokens])[None, :].astype(np.int32)
  orig_seq_len = full_tokens.shape[1]
  padded_seq_len = ((orig_seq_len + 511) // 512) * 512
  ep_size = args.ep or 1
  micro = max(ep_size, args.trainer_micro_batch or ep_size)
  if micro % ep_size != 0:
    micro = ((micro + ep_size - 1) // ep_size) * ep_size

  log(f"trainer: full tokens {orig_seq_len} padded to {padded_seq_len}")

  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml, "attention=flash"],
      model_name=MODEL_MAXTEXT,
      load_parameters_path=args.ckpt,
      tokenizer_path=MODEL_HF,
      run_name="multiturn_trainer",
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

  model, mesh = model_creation_utils.from_pretrained(
      cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN
  )
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

  # Pad to micro batch
  n_rows = full_tokens.shape[0]
  if n_rows % micro != 0:
    n_pad = micro - (n_rows % micro)
    padded_tokens = np.concatenate([full_tokens, np.repeat(full_tokens[-1:], n_pad, axis=0)], axis=0)
  else:
    padded_tokens = full_tokens

  if padded_seq_len != orig_seq_len:
    pad_len = padded_seq_len - orig_seq_len
    padded_tokens = np.pad(padded_tokens, ((0, 0), (0, pad_len)), mode="constant")

  nxt_np = np.roll(padded_tokens, -1, axis=1)

  with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))
    pos_np = np.broadcast_to(np.arange(padded_seq_len, dtype=np.int32), (micro, padded_seq_len))
    seg_np = np.ones((micro, padded_seq_len), np.int32)

    tokens = jax.device_put(padded_tokens[:micro], sh)
    nxt = jax.device_put(nxt_np[:micro], sh)
    pos = jax.device_put(pos_np, sh)
    seg = jax.device_put(seg_np, sh)

    lp_d, t1_d = fwd_logp(st, tokens, pos, seg, nxt)
    jax.block_until_ready(lp_d)
    full_logp = np.asarray(lp_d)[0, :orig_seq_len]

  # Slice out logits predicting completion tokens
  # The logit at index (n_prompt - 1) predicts the first completion token
  n_prompt = len(prompt_tokens)
  n_comp = len(comp_tokens)
  trainer_comp_logp = full_logp[n_prompt - 1 : n_prompt + n_comp - 1]

  log(f"trainer: forward pass complete, mean comp logp={trainer_comp_logp.mean():.4f}")

  out_path = os.path.join(out_dir, "multiturn_trainer.npz")
  np.savez(
      out_path,
      trainer_comp_logp=trainer_comp_logp,
      full_logp=full_logp,
  )
  log(f"trainer: saved {out_path}")
  return out_path


# ==============================================================================
# STAGE: COMPARE
# ==============================================================================
def stage_compare(args, out_dir):
  log("=== STAGE: COMPARE (Tunix Sequence Mult Prob Error) ===")

  rdata = np.load(os.path.join(out_dir, "multiturn_rollout.npz"))
  tdata = np.load(os.path.join(out_dir, "multiturn_trainer.npz"))

  comp_tokens = rdata["conversation_tokens"]
  comp_masks = rdata["conversation_masks"]
  sampler_logp = rdata["sampler_logprobs"]
  trainer_logp = tdata["trainer_comp_logp"]
  turn_slices = json.loads(str(rdata["turn_slices_json"]))

  assert len(sampler_logp) == len(trainer_logp), f"{len(sampler_logp)} vs {len(trainer_logp)}"
  assert len(comp_masks) == len(sampler_logp)

  diff = np.abs(trainer_logp - sampler_logp)
  active_mask = (comp_masks == 1)

  log("\n" + "=" * 80)
  log("=== MULTI-TURN LOGPROB PARITY REPORT ===")
  log("=" * 80)
  log(f"Total completion tokens: {len(comp_tokens)}")
  log(f"Active completion tokens (mask == 1): {active_mask.sum()}")
  log(f"Masked tokens (env observation & suffix, mask == 0): {(~active_mask).sum()}")

  overall_err = sequence_mult_prob_error(trainer_logp, sampler_logp, comp_masks)
  log(f"\nOVERALL sequence_mult_prob_error: {overall_err:.4f}")
  log(f"Tunix acceptance threshold: <= {args.seq_logprob_error_threshold:.1f}")
  passed_gate = overall_err <= args.seq_logprob_error_threshold
  log(f"Gate Status: {'PASSED [OK]' if passed_gate else 'FAILED [X]'}")

  turn_results = []
  for tinfo in turn_slices:
    tnum = tinfo["turn"]
    astart, aend = tinfo["asst_slice"]
    estart, eend = tinfo["env_slice"]

    asst_sub_mask = comp_masks[astart:aend]
    asst_err = sequence_mult_prob_error(
        trainer_logp[astart:aend], sampler_logp[astart:aend], asst_sub_mask
    )
    asst_diff = diff[astart:aend][asst_sub_mask == 1]
    max_d = float(asst_diff.max()) if len(asst_diff) > 0 else 0.0
    mean_d = float(asst_diff.mean()) if len(asst_diff) > 0 else 0.0

    log(f"\n--- Turn {tnum} ---")
    log(f"  Assistant tokens: {aend - astart} (active mask: {asst_sub_mask.sum()})")
    log(f"  sequence_mult_prob_error: {asst_err:.4f}")
    log(f"  mean |delta logp|: {mean_d:.4f}, max |delta logp|: {max_d:.4f}")

    if eend > estart:
      env_sub_mask = comp_masks[estart:eend]
      log(f"  Env observation tokens: {eend - estart} (all mask == {env_sub_mask.max()} [should be 0])")

    turn_results.append({
        "turn": tnum,
        "asst_tokens": int(aend - astart),
        "active_tokens": int(asst_sub_mask.sum()),
        "mult_prob_err": asst_err,
        "mean_diff": mean_d,
        "max_diff": max_d,
    })

  # Simulated bug test: What happens if a single turn has missing logprobs (0.0 fallback)?
  sim_sampler_logp = sampler_logp.copy()
  t2_start, t2_end = turn_slices[1]["asst_slice"]
  # Replace first 10 tokens of Turn 2 with 0.0 (simulating silent missing logprob fallback)
  sim_sampler_logp[t2_start : t2_start + 10] = 0.0
  sim_err = sequence_mult_prob_error(trainer_logp, sim_sampler_logp, comp_masks)
  log("\n" + "=" * 80)
  log("=== SIMULATION: EFFECT OF SILENT 0.0 LOGPROB FALLBACK ===")
  log("=" * 80)
  log(f"Simulating 10 missing tokens in Turn 2 replaced with 0.0:")
  log(f"Resulting sequence_mult_prob_error: {sim_err:.2e} ({sim_err:,.0f})")
  log("This matches the cluster error metric (billions) observed during rollouts!")

  summary = {
      "overall_sequence_mult_prob_error": overall_err,
      "passed_threshold_2_0": passed_gate,
      "turns": turn_results,
      "simulated_silent_zero_error": sim_err,
  }
  summary_path = os.path.join(out_dir, "multiturn_summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  log(f"\nSaved summary to {summary_path}")


def main():
  ap = argparse.ArgumentParser(description="Multi-turn RL Logprob Parity Microbenchmark")
  ap.add_argument(
      "--stage",
      default="all",
      choices=["all", "sampler", "trainer", "compare"],
  )
  ap.add_argument("--out-dir", default="/mnt/disks/persist/multiturn_bench")
  ap.add_argument("--hf-home", default=None)
  ap.add_argument("--maxtext-root", default=None)
  ap.add_argument("--tunix-root", default=None)
  ap.add_argument("--ckpt", default=CKPT)
  ap.add_argument("--gen-tokens", type=int, default=64)
  ap.add_argument("--gen-temperature", type=float, default=0.0)
  ap.add_argument("--logits-dot-fp32", action="store_true")
  ap.add_argument("--ep", type=int, default=8)
  ap.add_argument("--sharding-tp", type=int, default=1)
  ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
  ap.add_argument(
      "--disable-prefix-caching",
      dest="enable_prefix_caching",
      action="store_false",
  )
  ap.add_argument("--mamba-cache-mode", default="align", choices=["align", "none"])
  ap.add_argument("--custom-mamba-cache-multiplier", type=int, default=16)
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
  ap.add_argument("--trainer-micro-batch", type=int, default=1)
  ap.add_argument("--trainer-tp", type=int, default=1)
  ap.add_argument("--seq-logprob-error-threshold", type=float, default=2.0)
  args = ap.parse_args()

  maxtext_root, tunix_root, hf_home, out_dir = resolve_paths(args)

  if args.stage == "compare":
    stage_compare(args, out_dir)
    return

  set_tpu_env(hf_home)

  if args.stage in ("all", "sampler"):
    stage_sampler(args, maxtext_root, tunix_root, hf_home, out_dir)

  if args.stage in ("all", "trainer"):
    stage_trainer(args, maxtext_root, tunix_root, hf_home, out_dir)

  if args.stage == "all":
    stage_compare(args, out_dir)


if __name__ == "__main__":
  main()
