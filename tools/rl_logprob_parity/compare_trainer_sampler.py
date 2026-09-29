#!/usr/bin/env python3
"""Qwen3.5-35B-A3B: MaxText trainer vs vLLM sampler logprob parity microbenchmark.

Supports:
  * BF16 baseline (MLPerf DeepSWE production aligned configuration)
  * FP8 baseline (qwen3.5-35b-a3b-fp8 with fp8_moe=True, weight_dtype="float8_e4m3fn")
  * Real r2e gym dataset prompts (r2e_prompts_32.jsonl)
  * Native router replay (forced_routed_experts captured in sampler and replayed in trainer)
  * Sequence-level is_oob_ratio (band [0.999, 1.002]) and sampler_is/token_logdiff_absmean

Four stages, in order:
  1. tokenize  (CPU)  real text / r2e prompts -> HF tokenizer -> [B, S] token ids, saved once
  2. sampler   (TPU)  real vLLM engine: prompt_logprobs, optional rollouts, optional router capture
  3. trainer   (TPU)  MaxText nnx model, MODEL_MODE_TRAIN, teacher-forced over prompt+generated in one pass
  4. compare   (CPU)  per-token band stats + per-sequence seq-mask-tis is_oob_ratio
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np

MODEL_HF_BF16 = "Qwen/Qwen3.5-35B-A3B"
MODEL_MAXTEXT_BF16 = "qwen3.5-35b-a3b"
CKPT_BF16 = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"

MODEL_HF_FP8 = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_MAXTEXT_FP8 = "qwen3.5-35b-a3b-fp8"
CKPT_FP8 = "gs://cloud-devkit/users/wenxindong/ckpt/qwen3.5-35b-a3b-fp8/unscanned/0/items"

# SEQ_LEN / --gen-tokens mirror the RL job's max_prompt_length and max_response_length.
B, SEQ_LEN = 8, 4096
# TIS acceptance band; matches truncated_importance_sampling_ratio_min / _ratio in the RL trainer.
RATIO_MIN, RATIO_MAX = 0.999, 1.002

t0 = time.time()


def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- paths / env


def resolve_paths(args):
  """Locate the persist disk, the HF cache and the MaxText tree."""
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
      # If running in environment without pre-existing hub directory, default to ~/.cache/huggingface
      hf_home = os.path.expanduser("~/.cache/huggingface")
      os.makedirs(hf_home, exist_ok=True)
  if args.tag and not args.out_dir and not os.environ.get("OUT_DIR"):
    out_dir = os.path.join(hf_home, "rl_logprob_parity", args.tag)
  else:
    out_dir = args.out_dir or os.environ.get("OUT_DIR") or os.path.join(hf_home, "rl_logprob_parity")
  os.makedirs(out_dir, exist_ok=True)

  # Auto-detect real r2e gym prompts file if not explicitly passed
  if not args.prompts_file:
    for cand_prompt in (
        os.path.join(maxtext_root, "tools", "rl_logprob_parity", "r2e_prompts_32.jsonl"),
        os.path.join(os.path.dirname(maxtext_root), "data", "r2e_prompts_32.jsonl"),
        os.path.join(maxtext_root, "..", "data", "r2e_prompts_32.jsonl"),
        "/workspace/rl_parity_ws/data/r2e_prompts_32.jsonl",
    ):
      if os.path.isfile(cand_prompt):
        args.prompts_file = cand_prompt
        log(f"prompts: auto-detected real r2e gym prompts: {cand_prompt}")
        break

  return maxtext_root, hf_home, out_dir


def set_tpu_env(hf_home, maxtext_root):
  """Set the process-level TPU env. Call once, before jax/libtpu/vllm are imported anywhere in the process."""
  os.environ.setdefault("HF_HOME", hf_home)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
  os.environ["NEW_MODEL_DESIGN"] = "1"
  # Production serving env, copied from run_vllm.sh & mlperf_base.sh
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
      "MAMBA_CACHE_MODE": "align",
      "VLLM_MAMBA_CACHE_MODE": "align",
  }.items():
    os.environ.setdefault(k, v)
  os.environ.setdefault(
      "LIBTPU_INIT_ARGS",
      " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
      " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
      " --xla_tpu_ars_combiner_threshold_in_bytes=0"
      " --xla_tpu_enable_async_collective_merger=false"
      " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
      " --xla_tpu_dvfs_p_state=7",
  )
  os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
  src = os.path.join(maxtext_root, "src")
  if src not in sys.path:
    sys.path.insert(0, src)


# ---------------------------------------------------------------- 1. tokenize


def get_safe_tokenizer(model_hf, hf_home=None):
  """Robust tokenizer loader that checks model_hf, falls back to BF16 or local snapshot."""
  from transformers import AutoTokenizer
  for name in (model_hf, MODEL_HF_BF16):
    try:
      tok = AutoTokenizer.from_pretrained(name)
      return tok, name
    except Exception as e:
      log(f"warning: failed to load tokenizer from {name}: {e}")
  if hf_home:
    import glob
    for pat in [
        os.path.join(hf_home, "hub", "models--Qwen--Qwen3.5-35B-A3B-FP8", "snapshots", "*"),
        os.path.join(hf_home, "hub", "models--Qwen--Qwen3.5-35B-A3B", "snapshots", "*"),
    ]:
      matches = glob.glob(pat)
      if matches:
        try:
          tok = AutoTokenizer.from_pretrained(matches[0])
          return tok, matches[0]
        except Exception:
          pass
  raise RuntimeError(f"Could not load tokenizer for {model_hf} or {MODEL_HF_BF16}")


def stage_tokenize(args, hf_home, maxtext_root, out_dir):
  """Tokenize the corpus once into [B, S]. Both TPU stages read this file, never the raw text."""
  path = os.path.join(out_dir, "tokens.npz")
  if os.path.exists(path) and not args.retokenize:
    z = np.load(path)
    log(f"tokens: reusing {path} {z['tokens'].shape}")
    return path
  os.environ.setdefault("HF_HOME", hf_home)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")

  model_hf = args.model_hf or (MODEL_HF_FP8 if args.model_type == "fp8" else MODEL_HF_BF16)
  tok, _ = get_safe_tokenizer(model_hf, hf_home)

  if args.prompts_file:
    rows, short = [], []
    with open(args.prompts_file, encoding="utf-8") as f:
      for ln in f:
        ln = ln.strip()
        if not ln:
          continue
        rec = json.loads(ln)
        ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < SEQ_LEN:
          short.append(len(ids))
        rows.append(ids[:SEQ_LEN])
    if short:
      raise SystemExit(f"{len(short)} prompt(s) shorter than SEQ_LEN={SEQ_LEN}: {short[:5]}")
    tokens = np.array(rows, dtype=np.int32)
    np.savez(path, tokens=tokens, n_files=np.int32(len(rows)), n_corpus_tokens=np.int32(tokens.size))
    log(f"tokens: {tokens.shape} from {len(rows)} prompts in {args.prompts_file} -> {path}")
    log(f"tokens: first = {tok.decode(tokens[0, :12])!r}")
    return path

  if args.text_file:
    files = [args.text_file]
  else:
    files = sorted(glob.glob(os.path.join(maxtext_root, args.text_glob), recursive=True))
  if not files:
    raise SystemExit(f"no text matched {args.text_glob!r} under {maxtext_root}")
  texts = []
  for f in files:
    with open(f, encoding="utf-8") as fh:
      texts.append(fh.read())
  text = "\n\n".join(texts)
  ids = tok(text)["input_ids"]
  if len(ids) < B * SEQ_LEN:
    raise SystemExit(f"need {B * SEQ_LEN} tokens, corpus has {len(ids)}")
  tokens = np.array(ids[: B * SEQ_LEN], dtype=np.int32).reshape(B, SEQ_LEN)
  np.savez(path, tokens=tokens, n_files=np.int32(len(files)), n_corpus_tokens=np.int32(len(ids)))
  log(f"tokens: {tokens.shape} from {len(files)} files / {len(ids)} corpus tokens -> {path}")
  log(f"tokens: first = {tok.decode(tokens[0, :12])!r}")
  return path


# ---------------------------------------------------------------- 2. sampler


def quantize_model_moe_fp8(model, per_expert: bool = True, legacy_round: bool = False):
  """Quantize only MoE expert weights in MaxText model to float8_e4m3fn with per-channel scaling (Peano lab post method).

  w / scale is cast straight to e4m3 (round-to-nearest-even in the FP8 grid). legacy_round=True reproduces the
  original jnp.round() before the cast, which snaps every value to an integer first and zeroes |w/scale| < 0.5.
  """
  import jax.numpy as jnp
  from flax import nnx

  FP8_MAX = 448.0

  def _q_expert_weight(w, channel_axis=2):
    if per_expert:
      reduce_axes = tuple(d for d in range(w.ndim) if d != 0 and d != channel_axis)
    else:
      reduce_axes = tuple(d for d in range(w.ndim) if d != channel_axis)
    max_val = jnp.max(jnp.abs(w), axis=reduce_axes, keepdims=True)
    scale = jnp.maximum(max_val / FP8_MAX, 1e-12).astype(jnp.float32)
    x = w.astype(jnp.float32) / scale
    if legacy_round:
      x = jnp.round(x)
    q_w = jnp.clip(x, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn)
    return q_w, scale

  count = 0
  decoder = getattr(model, "decoder", model)
  num_layers = getattr(getattr(decoder, "config", None), "num_decoder_layers", 48)

  moe_modules = []

  # 1. Unscanned layers: decoder.layers_0 .. layers_{N-1}
  for i in range(num_layers):
    layer = getattr(decoder, f"layers_{i}", None)
    if layer is not None and hasattr(layer, "mlp"):
      routed = getattr(layer.mlp, "routed_experts", None)
      if routed is not None:
        moe_modules.append(routed)

  # 2. Scanned blocks: decoder.layers, decoder.layers_remainder, decoder.scanned_blocks
  if not moe_modules:
    blocks = []
    if hasattr(decoder, "layers") and decoder.layers is not None:
      if isinstance(decoder.layers, (list, tuple)):
        blocks.extend(decoder.layers)
      else:
        blocks.append(decoder.layers)
    if hasattr(decoder, "layers_remainder") and decoder.layers_remainder is not None:
      blocks.append(decoder.layers_remainder)
    if hasattr(decoder, "scanned_blocks") and decoder.scanned_blocks is not None:
      blocks.append(decoder.scanned_blocks)

    for block in blocks:
      for j in range(16):
        sublayer = getattr(block, f"layer_{j}", None)
        if sublayer is not None and hasattr(sublayer, "mlp"):
          routed = getattr(sublayer.mlp, "routed_experts", None)
          if routed is not None and routed not in moe_modules:
            moe_modules.append(routed)
      if hasattr(block, "mlp"):
        routed = getattr(block.mlp, "routed_experts", None)
        if routed is not None and routed not in moe_modules:
          moe_modules.append(routed)

  log(f"moe-only-fp8: found {len(moe_modules)} MoE modules to quantize")

  for moe_mod in moe_modules:
    wi_axes = getattr(moe_mod, "wi_kernel_axes", None)
    wo_axes = getattr(moe_mod, "wo_kernel_axes", None)
    wi_scale_axes = (wi_axes[0], None, wi_axes[2]) if wi_axes is not None and len(wi_axes) >= 3 else None
    wo_scale_axes = (wo_axes[0], None, wo_axes[2]) if wo_axes is not None and len(wo_axes) >= 3 else None

    if hasattr(moe_mod, "wo") and moe_mod.wo is not None:
      wo_q, wo_s = _q_expert_weight(moe_mod.wo[...], channel_axis=2)
      moe_mod.wo = nnx.Param(wo_q, out_sharding=wo_axes)
      moe_mod.wo_scale = nnx.data(nnx.Param(wo_s, out_sharding=wo_scale_axes))

    if hasattr(moe_mod, "wi") and moe_mod.wi is not None:
      wi_q, wi_s = _q_expert_weight(moe_mod.wi[...], channel_axis=2)
      moe_mod.wi = nnx.Param(wi_q, out_sharding=wi_axes)
      moe_mod.wi_scale = nnx.data(nnx.Param(wi_s, out_sharding=wi_scale_axes))

    if hasattr(moe_mod, "wi_0") and moe_mod.wi_0 is not None:
      w0_q, w0_s = _q_expert_weight(moe_mod.wi_0[...], channel_axis=2)
      moe_mod.wi_0 = nnx.Param(w0_q, out_sharding=wi_axes)
      moe_mod.wi_0_scale = nnx.data(nnx.Param(w0_s, out_sharding=wi_scale_axes))

    if hasattr(moe_mod, "wi_1") and moe_mod.wi_1 is not None:
      w1_q, w1_s = _q_expert_weight(moe_mod.wi_1[...], channel_axis=2)
      moe_mod.wi_1 = nnx.Param(w1_q, out_sharding=wi_axes)
      moe_mod.wi_1_scale = nnx.data(nnx.Param(w1_s, out_sharding=wi_scale_axes))

    moe_mod.weight_dtype = jnp.float8_e4m3fn
    count += 1

  log(f"moe-only-fp8: successfully quantized {count} MoE layers to float8_e4m3fn per-channel")


def sync_weights_from_trainer(args, out_dir, runner):
  """Push trainer parameters into the live engine the way a Tunix RL step does."""
  import gc

  import jax
  from flax import nnx
  from tunix.generate import utils as gen_utils
  from tunix.rl import reshard

  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.utils import model_creation_utils
  import maxtext.configs as maxtext_configs

  dst_model = getattr(runner.model, "model", None)
  if dst_model is None:
    raise SystemExit(f"--weight-sync: no MaxText model on {type(runner.model).__name__}")

  def leaf_ids(model):
    _, st = nnx.split(model)
    return [id(getattr(leaf, "value", leaf)) for _, leaf in st.flat_state()]

  before = leaf_ids(dst_model)

  is_fp8 = args.model_type == "fp8"
  model_maxtext = args.model_maxtext or (MODEL_MAXTEXT_FP8 if is_fp8 else MODEL_MAXTEXT_BF16)
  model_hf = args.model_hf or (MODEL_HF_FP8 if is_fp8 else MODEL_HF_BF16)
  weight_dtype = "float8_e4m3fn" if is_fp8 else "bfloat16"
  ckpt_path = args.ckpt or (CKPT_FP8 if is_fp8 else CKPT_BF16)

  base_yml = os.path.join(os.path.dirname(maxtext_configs.__file__), "base.yml")
  _, tok_name = get_safe_tokenizer(model_hf, os.environ.get("HF_HOME"))
  cfg_kwargs = dict(
      model_name=model_maxtext,
      load_parameters_path=ckpt_path,
      tokenizer_path=tok_name,
      run_name="rl_logprob_parity_sync",
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      scan_layers=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      dtype="bfloat16",
      weight_dtype=weight_dtype,
      max_target_length=4096,
      max_prefill_predict_length=4096,
      per_device_batch_size=1 / len(jax.devices()),
      ici_tensor_parallelism=args.trainer_tp,
      ici_expert_parallelism=args.sync_src_ep if args.sync_src_ep is not None else args.trainer_ep,
      allow_split_physical_axes=True,
      log_config=False,
      skip_jax_distributed_system=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      prefuse_moe_weights=False if is_fp8 else True,
      float32_logits=True,
      float32_gate_logits=True,
      float32_weight_sum=True,
      logits_dot_in_fp32=args.logits_dot_fp32,
      sparse_matmul=True,
      megablox=True,
  )
  if is_fp8:
    cfg_kwargs["fp8_moe"] = True
    cfg_kwargs["weight_block_size"] = 128
  if args.trainer_kv_heads:
    cfg_kwargs["base_num_kv_heads"] = args.trainer_kv_heads
    cfg_kwargs["override_model_config"] = True

  cfg = pyconfig.initialize([sys.argv[0], base_yml, "attention=flash"], **cfg_kwargs)
  log("weight-sync: loading trainer params")
  src_model, _ = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
  _, src_state = nnx.split(src_model)
  _, dst_state = nnx.split(dst_model)

  log("weight-sync: transfer_state_directly(trainer -> sampler)")
  gen_utils.transfer_state_directly(
      src_state=src_state,
      dst_state=dst_state,
      reshard_fn=reshard.reshard_pytree,
      delete_dst_buffers=False,
  )
  nnx.update(dst_model, dst_state)
  del src_model, src_state, dst_state
  gc.collect()

  after = leaf_ids(dst_model)
  replaced = sum(1 for a, b in zip(before, after) if a != b) / max(len(before), 1)
  log(f"weight-sync: {replaced:.1%} of destination buffers replaced")
  if replaced == 0.0:
    raise SystemExit("--weight-sync: transfer did not modify the engine's weights; aborting")
  if hasattr(runner, "state"):
    runner.state_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
  return replaced


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  """Real vLLM engine, prompt_logprobs on the shared tokens. logp[b, i] = logprob of token i given tokens[:i]."""
  try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
  except ImportError:
    pass
  import vllm
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt

  model_hf = args.model_hf or (MODEL_HF_FP8 if args.model_type == "fp8" else MODEL_HF_BF16)
  model_maxtext = args.model_maxtext or (MODEL_MAXTEXT_FP8 if args.model_type == "fp8" else MODEL_MAXTEXT_BF16)
  ckpt_path = args.ckpt or (CKPT_FP8 if args.model_type == "fp8" else CKPT_BF16)
  is_fp8 = args.model_type == "fp8"

  z = np.load(os.path.join(out_dir, "tokens.npz"))
  tokens = z["tokens"]
  log(f"sampler: loaded tokens {tokens.shape} (model_type={args.model_type}, model_hf={model_hf})")

  _, tok_name = get_safe_tokenizer(model_hf, hf_home)
  kw = dict(
      tokenizer=tok_name,
      trust_remote_code=True,
      dtype="bfloat16",
      max_model_len=max(4096, SEQ_LEN + args.gen_tokens + 64),
      max_num_seqs=16,
      max_num_batched_tokens=2048,
      block_size=256,
      enable_chunked_prefill=True,
      enable_prefix_caching=args.enable_prefix_caching,
      prefix_cache_retention_interval=0,
      mamba_cache_mode=args.mamba_cache_mode,
      gpu_memory_utilization=0.6 if args.fp8_moe else args.gpu_memory_utilization,
      language_model_only=True,
      limit_mm_per_prompt={"image": 0, "video": 0},
      disable_log_stats=True,
      kv_cache_dtype="bfloat16",
      reasoning_parser="qwen3",
      seed=0,
  )

  if args.sampler_ep > 1:
    sharding = {"sharding_strategy": {
        "expert_parallelism": args.sampler_ep,
        "tensor_parallelism": args.sharding_tp,
        "enable_dp_attention": True,
    }}
  elif args.attn_dp > 1:
    sharding = {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": args.attn_dp}}
  else:
    sharding = None

  sys.path.insert(0, os.path.join(maxtext_root, "src", "maxtext", "integration", "vllm"))
  import maxtext_vllm_adapter

  maxtext_vllm_adapter.register()
  attn_kernel = "vllm_rpa"
  mt_cfg = {
      "model_name": model_maxtext,
      "load_parameters_path": ckpt_path,
      "weight_dtype": "float8_e4m3fn" if is_fp8 else "bfloat16",
      "dtype": "bfloat16",
      "attention": attn_kernel,
      "allow_split_physical_axes": True,
      "scan_layers": False,
      "prefuse_moe_weights": False if is_fp8 else True,
      "enable_dp_attention": args.sampler_ep > 1 or args.attn_dp > 1,
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
      "return_routed_experts": args.router_replay,
  }
  if is_fp8:
    mt_cfg["fp8_moe"] = True
    mt_cfg["weight_block_size"] = 128

  kw["model"] = tok_name
  kw["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
  kw["additional_config"] = {
      "maxtext_config": mt_cfg,
      "custom_mamba_cache_multiplier": 16,
      **({"sharding": sharding} if sharding else {}),
  }

  if args.router_replay:
    kw["enable_return_routed_experts"] = True

  if args.fp8_moe:
    # tpu-inference's model_loader captures graphdef + state treedef right after _get_nnx_model returns
    # (nnx.split(jit_model)), and every step runs nnx.merge on that capture. Quantizing runner.model after LLM()
    # only mutates the Python module (and adds wo_scale/wi_scale leaves the captured treedef does not have), so
    # dispatch kept running the bf16 weights. Quantize inside the loader, before the split.
    from tpu_inference.models.common import model_loader as _tpu_model_loader

    _orig_get_nnx_model = _tpu_model_loader._get_nnx_model

    def _get_nnx_model_fp8(*a, **k):
      jit_model = _orig_get_nnx_model(*a, **k)
      quantize_model_moe_fp8(jit_model.model, per_expert=True, legacy_round=args.fp8_legacy_round)
      return jit_model

    _tpu_model_loader._get_nnx_model = _get_nnx_model_fp8

  llm = LLM(**kw)
  log(f"adapter engine up (model_type={args.model_type})")

  runner = llm.llm_engine.model_executor.driver_worker.model_runner

  for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
    if obj is not None and hasattr(obj, "uses_mrope"):
      obj.uses_mrope = False

  def _text_mrope(prompt_token_ids, mm_features):
    pos = np.arange(len(prompt_token_ids), dtype=np.int64)
    return np.stack([pos, pos, pos]), 0

  runner.get_mrope_input_positions_fn = _text_mrope

  if args.fp8_moe:
    _tpu_model_loader._get_nnx_model = _orig_get_nnx_model
    # Check the dispatch view, not the module: these are the arrays model_fn actually receives.
    import jax.numpy as jnp

    n_fp8 = sum(1 for x in runner.state_leaves if getattr(x, "dtype", None) == jnp.float8_e4m3fn)
    log(f"sampler: {n_fp8} float8_e4m3fn leaves in runner.state_leaves")
    if n_fp8 == 0:
      raise RuntimeError("--fp8-moe: sampler dispatch state has no FP8 leaves; MoE quantization did not take effect")

  if args.weight_sync:
    sync_weights_from_trainer(args, out_dir, runner)

  # Standard evaluation pass
  sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=-1, top_p=1.0, prompt_logprobs=1)
  if args.router_replay:
    try:
      setattr(sp, "routed_experts_prompt_start", 0)
    except Exception:
      pass
  outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)

  logp = np.full(tokens.shape, np.nan, np.float32)
  top1 = np.full(tokens.shape, -1, np.int32)
  for b, o in enumerate(outs):
    for i, d in enumerate(o.prompt_logprobs):
      if d is None:
        continue
      tid = int(tokens[b, i])
      logp[b, i] = d[tid].logprob if tid in d else np.nan
      top1[b, i] = max(d.items(), key=lambda kv: kv[1].logprob)[0]
  saved = dict(logp=logp, top1=top1, tokens=tokens)
  log(f"sampler: prompt pass done; mean logp={np.nanmean(logp[:, 1:]):.4f} "
      f"top-1 acc={np.mean(top1[:, 1:] == tokens[:, 1:]):.3f}")
  path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  np.savez(path, **saved)

  if args.gen_tokens > 0:
    if args.stop_at_eos:
      gsp = SamplingParams(max_tokens=args.gen_tokens, temperature=args.gen_temperature,
                           top_p=1.0, top_k=-1, logprobs=1)
    else:
      gsp = SamplingParams(max_tokens=args.gen_tokens, min_tokens=args.gen_tokens,
                           temperature=args.gen_temperature, top_p=1.0, top_k=-1, logprobs=1,
                           ignore_eos=True)
    if args.router_replay:
      try:
        setattr(gsp, "routed_experts_prompt_start", 0)
      except Exception:
        pass
    log(f"sampler: generating {args.gen_tokens} tokens x {len(tokens)} prompts (temperature={args.gen_temperature})")
    gouts = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], gsp)
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

  # Router replay extraction across all sequences (prompt + decode if present) from vLLM output
  if args.router_replay:
    log(f"sampler: extracting router decisions from vLLM output for router-replay across {tokens.shape[0]} sequences...")
    try:
      extracted_fre = []
      total_expected_len = tokens.shape[1] + (args.gen_tokens if args.gen_tokens > 0 else 0)
      for b in range(len(tokens)):
        o = outs[b]
        re_prompt = None
        if hasattr(o, "prompt_routed_experts") and o.prompt_routed_experts is not None:
          re_prompt = o.prompt_routed_experts
        elif len(o.outputs) > 0 and hasattr(o.outputs[0], "routed_experts") and o.outputs[0].routed_experts is not None:
          re_prompt = o.outputs[0].routed_experts
        elif hasattr(o, "routed_experts") and o.routed_experts is not None:
          re_prompt = o.routed_experts

        if re_prompt is not None:
          re_p_arr = np.asarray(re_prompt, dtype=np.int16)
          if re_p_arr.shape[0] < tokens.shape[1]:
            re_p_arr = np.pad(re_p_arr, ((0, tokens.shape[1] - re_p_arr.shape[0]), (0, 0), (0, 0)), mode="edge")
          elif re_p_arr.shape[0] > tokens.shape[1]:
            re_p_arr = re_p_arr[: tokens.shape[1]]

          if args.gen_tokens > 0 and 'gouts' in locals():
            go = gouts[b]
            re_gen = None
            if len(go.outputs) > 0 and hasattr(go.outputs[0], "routed_experts") and go.outputs[0].routed_experts is not None:
              re_gen = go.outputs[0].routed_experts
            elif hasattr(go, "routed_experts") and go.routed_experts is not None:
              re_gen = go.routed_experts

            if re_gen is not None:
              re_g_arr = np.asarray(re_gen, dtype=np.int16)
              # If vLLM returned prompt + gen tokens in output routed_experts, slice the gen part
              if re_g_arr.shape[0] >= tokens.shape[1]:
                re_gen_only = re_g_arr[tokens.shape[1] :]
              else:
                re_gen_only = re_g_arr

              if re_gen_only.shape[0] < args.gen_tokens:
                pad_gen = np.full((args.gen_tokens - re_gen_only.shape[0], re_p_arr.shape[1], re_p_arr.shape[2]), -1, dtype=np.int16)
                re_gen_only = np.concatenate([re_gen_only, pad_gen], axis=0)
              elif re_gen_only.shape[0] > args.gen_tokens:
                re_gen_only = re_gen_only[: args.gen_tokens]
              seq_re = np.concatenate([re_p_arr, re_gen_only], axis=0)
            else:
              pad_gen = np.full((args.gen_tokens, re_p_arr.shape[1], re_p_arr.shape[2]), -1, dtype=np.int16)
              seq_re = np.concatenate([re_p_arr, pad_gen], axis=0)
          else:
            seq_re = re_p_arr

          log(f"sampler: sequence {b + 1}/{len(tokens)} router decisions shape: {seq_re.shape}")
          extracted_fre.append(seq_re)
        else:
          log(f"sampler: warning - no prompt router decisions for sequence {b + 1}")
          break

      if len(extracted_fre) == len(tokens):
        all_fre = np.stack(extracted_fre, axis=0)
        router_file = os.path.join(out_dir, f"router_indices_{args.sampler}.npz")
        np.savez_compressed(router_file, experts=all_fre)
        log(f"sampler: saved router replay -> {router_file} (shape={all_fre.shape}, size={all_fre.nbytes / 1e6:.1f}MB)")
      else:
        log("sampler: warning - could not extract router replay for all sequences from vLLM outputs")
    except Exception as e:
      log(f"sampler: warning - router replay extraction failed: {e}")

  if 'gouts' in locals():
    del gouts

  np.savez(path, **saved)
  log(f"sampler: saved {path}")
  del outs, llm
  gc.collect()
  log("sampler: engine released")
  return path


# ---------------------------------------------------------------- 3. trainer


def stage_trainer(args, hf_home, maxtext_root, out_dir):
  """MaxText nnx model in MODEL_MODE_TRAIN, teacher-forced over the same tokens."""
  try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
  except ImportError:
    pass
  import jax
  import jax.numpy as jnp
  from jax.sharding import NamedSharding, PartitionSpec as P
  from flax import nnx
  import flax.linen as nn

  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.utils import model_creation_utils
  import maxtext.configs as maxtext_configs

  model_hf = args.model_hf or (MODEL_HF_FP8 if args.model_type == "fp8" else MODEL_HF_BF16)
  model_maxtext = args.model_maxtext or (MODEL_MAXTEXT_FP8 if args.model_type == "fp8" else MODEL_MAXTEXT_BF16)
  ckpt_path = args.ckpt or (CKPT_FP8 if args.model_type == "fp8" else CKPT_BF16)
  is_fp8 = args.model_type == "fp8"

  base_yml = os.path.join(os.path.dirname(maxtext_configs.__file__), "base.yml")
  z = np.load(os.path.join(out_dir, "tokens.npz"))
  prompt_np = z["tokens"]

  sa_path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  gen_ids = None
  if os.path.exists(sa_path):
    sa = np.load(sa_path)
    if "gen_ids" in sa.files:
      gen_ids = sa["gen_ids"]
  tokens_np = prompt_np if gen_ids is None else np.concatenate([prompt_np, gen_ids], axis=1).astype(np.int32)
  seq_len = tokens_np.shape[1]
  log(f"trainer: scoring {tokens_np.shape} ({'prompt only' if gen_ids is None else f'prompt {SEQ_LEN} + gen {gen_ids.shape[1]}'})")

  _, tok_name = get_safe_tokenizer(model_hf, hf_home)
  cfg_kwargs = dict(
      model_name=model_maxtext,
      load_parameters_path=ckpt_path,
      tokenizer_path=tok_name,
      run_name="rl_logprob_parity",
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      scan_layers=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      dtype="bfloat16",
      weight_dtype="float8_e4m3fn" if is_fp8 else "bfloat16",
      max_target_length=seq_len,
      max_prefill_predict_length=seq_len,
      per_device_batch_size=(args.trainer_micro_batch or tokens_np.shape[0]) / len(jax.devices()),
      ici_tensor_parallelism=args.trainer_tp,
      ici_expert_parallelism=args.trainer_ep,
      allow_split_physical_axes=True,
      log_config=False,
      skip_jax_distributed_system=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      prefuse_moe_weights=False if is_fp8 else True,
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
  if is_fp8:
    cfg_kwargs["fp8_moe"] = True
    cfg_kwargs["weight_block_size"] = 128
  if args.trainer_kv_heads:
    cfg_kwargs["base_num_kv_heads"] = args.trainer_kv_heads
    cfg_kwargs["override_model_config"] = True

  cfg = pyconfig.initialize([sys.argv[0], base_yml, "attention=flash"], **cfg_kwargs)
  log(f"trainer cfg: emb={cfg.emb_dim} q={cfg.num_query_heads} kv={cfg.num_kv_heads} "
      f"E={cfg.num_experts} k={cfg.num_experts_per_tok} layers={cfg.num_decoder_layers} "
      f"quantization={cfg.quantization} weight_dtype={cfg.weight_dtype}")

  model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)
  log("trainer model loaded")

  quantize_trainer_moe = args.fp8_moe if getattr(args, "trainer_fp8_moe", None) is None else args.trainer_fp8_moe
  if quantize_trainer_moe:
    quantize_model_moe_fp8(model, per_expert=True, legacy_round=args.fp8_legacy_round)

  gd, st = nnx.split(model)

  @jax.jit
  def fwd_logp(st, tokens, pos, seg, nxt, fre=None):
    m = nnx.merge(gd, st)
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      call_kwargs = dict(enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
      if fre is not None:
        call_kwargs["forced_routed_experts"] = fre
      out = m(tokens, pos, seg, **call_kwargs)
    logits = out[0] if isinstance(out, tuple) else out
    tgt = jnp.take_along_axis(logits, nxt[..., None], -1)[..., 0].astype(jnp.float32)
    lse = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
    return tgt - lse, jnp.argmax(logits, -1).astype(jnp.int32)

  # Check for router replay
  fre_all = None
  if args.router_replay:
    router_file = os.path.join(out_dir, f"router_indices_{args.sampler}.npz")
    if os.path.exists(router_file):
      fre_all = np.load(router_file)["experts"]
      log(f"trainer: using router replay from {router_file} (shape={fre_all.shape})")
      if fre_all.shape[1] < seq_len:
        pad_len = seq_len - fre_all.shape[1]
        log(f"trainer: padding router replay with -1 from {fre_all.shape[1]} to {seq_len} tokens")
        pad_fre = np.full((fre_all.shape[0], pad_len, fre_all.shape[2], fre_all.shape[3]), -1, dtype=fre_all.dtype)
        fre_all = np.concatenate([fre_all, pad_fre], axis=1)
      elif fre_all.shape[1] > seq_len:
        fre_all = fre_all[:, :seq_len]
    else:
      log(f"trainer: WARNING --router-replay requested but {router_file} not found; falling back to dynamic routing")

  nxt_np = np.roll(tokens_np, -1, axis=1)
  n_rows = tokens_np.shape[0]
  micro = args.trainer_micro_batch or n_rows
  if n_rows % micro:
    raise SystemExit(f"--trainer-micro-batch {micro} does not divide {n_rows} rows")
  log(f"trainer: {n_rows} rows in chunks of {micro} (seq_len={seq_len})")
  logp_parts, top1_parts = [], []
  with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
    sh = NamedSharding(mesh, P(("data", "fsdp"), None))
    sh_fre = NamedSharding(mesh, P(("data", "fsdp"), None, None, None))
    pos_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (micro, seq_len))
    seg_np = np.ones((micro, seq_len), np.int32)
    for i in range(0, n_rows, micro):
      tokens = jax.device_put(tokens_np[i : i + micro], sh)
      nxt = jax.device_put(nxt_np[i : i + micro], sh)
      pos = jax.device_put(pos_np, sh)
      seg = jax.device_put(seg_np, sh)
      fre = jax.device_put(fre_all[i : i + micro], sh_fre) if fre_all is not None else None
      lp_d, t1_d = fwd_logp(st, tokens, pos, seg, nxt, fre)
      jax.block_until_ready(lp_d)
      logp_parts.append(np.asarray(lp_d))
      top1_parts.append(np.asarray(t1_d))
      log(f"trainer: rows {i}-{i + micro - 1} done")
  logp = np.concatenate(logp_parts, axis=0)
  top1 = np.concatenate(top1_parts, axis=0)

  acc = np.mean(top1[:, :-1] == tokens_np[:, 1:])
  saved = dict(logp=logp, top1=top1, tokens=prompt_np, full_tokens=tokens_np)
  msg = f"mean logp={logp[:, :-1].mean():.4f} next-token top-1 acc={acc:.3f} (ckpt sanity)"
  if gen_ids is not None:
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


def compare(prev_logprobs, generation_logprobs, token_mask=None, sample_mask=None, error_threshold=2.0):
  """prev_logprobs = trainer (pi_prev), generation_logprobs = sampler (pi_gen), already index-aligned."""
  raw = prev_logprobs - generation_logprobs
  log_is_ratio = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
  if token_mask is None:
    token_mask = np.isfinite(raw).astype(np.float64)
  n_seq = log_is_ratio.shape[0]

  # Per-sequence multiplicative error: mean_t exp(|log_is_t|) (Tunix algo_core.py)
  mult_prob_err_raw = np.exp(np.abs(log_is_ratio))
  seq_mult_prob_err = masked_mean(mult_prob_err_raw, token_mask, axis=-1)
  seq_valid = (token_mask.sum(axis=-1) > 0).astype(np.float64)

  if sample_mask is None:
    # Filter sequences where mult_prob_error > error_threshold (Tunix sample_mask / kept_frac)
    sample_mask = ((seq_mult_prob_err <= error_threshold) & (seq_valid > 0)).astype(np.float64)
  global_valid_seqs = sample_mask.sum()

  # per-token stats
  r_tok = np.exp(log_is_ratio)
  in_tok = ((r_tok >= RATIO_MIN) & (r_tok <= RATIO_MAX)).astype(np.float64)
  tok_oob = masked_mean(1.0 - in_tok, token_mask)
  d = np.abs(log_is_ratio)[token_mask > 0]
  token_logdiff_absmean = float(np.mean(d)) if len(d) > 0 else 0.0

  # per-sequence (seq-mask-tis)
  seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, axis=-1)
  seq_geomean_is_ratio = np.exp(seq_log_is_ratio_mean)
  seq_in_band = ((seq_geomean_is_ratio >= RATIO_MIN) & (seq_geomean_is_ratio <= RATIO_MAX)).astype(np.float64)
  seq_kept_mask = seq_in_band * seq_valid
  seq_oob = masked_mean(1.0 - seq_kept_mask, sample_mask, global_normalization_factor=global_valid_seqs)

  return dict(
      n_tokens=int(token_mask.sum()),
      n_seqs=n_seq,
      n_nonfinite=int((~np.isfinite(raw)).sum()),
      token_oob_ratio=float(tok_oob),
      token_in_band=float(1.0 - tok_oob),
      token_logdiff_absmean=token_logdiff_absmean,
      abs_d_median=float(np.median(d)) if len(d) > 0 else 0.0,
      abs_d_p99=float(np.percentile(d, 99)) if len(d) > 0 else 0.0,
      abs_d_max=float(d.max()) if len(d) > 0 else 0.0,
      seq_log_is_ratio_mean=seq_log_is_ratio_mean,
      seq_geomean_is_ratio=seq_geomean_is_ratio,
      seq_mult_prob_err=seq_mult_prob_err,
      seq_kept_mask=seq_kept_mask,
      sample_mask=sample_mask,
      kept_frac=float(sample_mask.sum() / max(seq_valid.sum(), 1.0)),
      mult_prob_error_mean=float(np.mean(seq_mult_prob_err[sample_mask > 0])) if sample_mask.sum() > 0 else 1.0,
      is_oob_ratio=float(seq_oob),
  )


def print_report(name, m):
  print(f"\n### {name}")
  print(f"  tokens={m['n_tokens']}  seqs={m['n_seqs']}  nonfinite(zeroed)={m['n_nonfinite']}  "
        f"band=[{RATIO_MIN}, {RATIO_MAX}]")
  print(f"  per-token : oob {m['token_oob_ratio']:.2%}  in-band {m['token_in_band']:.2%}   "
        f"|dlogp| med {m['abs_d_median']:.4f} / p99 {m['abs_d_p99']:.3f} / max {m['abs_d_max']:.2f}")
  print(f"  {'seq':>4} {'seq_log_is_ratio_mean':>22} {'seq_geomean_is_ratio':>21} {'mult_err':>10} {'kept':>5} {'active':>6}")
  for b in range(m["n_seqs"]):
    mult_str = f"{m['seq_mult_prob_err'][b]:.4f}" if "seq_mult_prob_err" in m else "N/A"
    act_str = str(int(m["sample_mask"][b])) if "sample_mask" in m else "1"
    print(f"  {b:>4} {m['seq_log_is_ratio_mean'][b]:>22.6e} "
          f"{m['seq_geomean_is_ratio'][b]:>21.6f} {mult_str:>10} {int(m['seq_kept_mask'][b]):>5} {act_str:>6}")
  active_cnt = int(m['sample_mask'].sum()) if 'sample_mask' in m else m['n_seqs']
  kept_frac_str = f"{m.get('kept_frac', 1.0):.2%}"
  print(f"  kept {int(m['seq_kept_mask'].sum())}/{m['n_seqs']} (active sequences: {active_cnt}/{m['n_seqs']}, kept_frac: {kept_frac_str})")
  print(f"  >>> is_oob_ratio (seq-mask-tis)       = {m['is_oob_ratio']:.4f} ({m['is_oob_ratio']:.2%})")
  print(f"      oob_ratio    (per-token)         = {m['token_oob_ratio']:.4f} ({m['token_oob_ratio']:.2%})")
  print(f"      sampler_is/token_logdiff_absmean = {m['token_logdiff_absmean']:.6f}")
  if "mult_prob_error_mean" in m:
    print(f"      sample_mask/mult_prob_error_mean = {m['mult_prob_error_mean']:.6f}")


def stage_compare(args, out_dir):
  """Align and score trainer vs sampler logprobs, and report module-level probing alignment."""
  tr_path = os.path.join(out_dir, "trainer_logprobs.npz")
  sa_path = os.path.join(out_dir, f"sampler_{args.sampler}_logprobs.npz")
  for p in (tr_path, sa_path):
    if not os.path.exists(p):
      raise SystemExit(f"missing {p}; run the corresponding stage first")
  tr = np.load(tr_path)
  sa = np.load(sa_path)
  if not np.array_equal(tr["tokens"], sa["tokens"]):
    raise SystemExit("trainer and sampler ran on different tokens; delete the npz files and rerun")
  label = "MaxText-in-vLLM (adapter)"
  quantize_trainer_moe = args.fp8_moe if getattr(args, "trainer_fp8_moe", None) is None else args.trainer_fp8_moe
  if getattr(args, "fp8_moe", False) and not quantize_trainer_moe:
    precision_tag = "BF16/BF16+MoE-FP8"
  elif getattr(args, "fp8_moe", False):
    precision_tag = "BF16+MoE-FP8/BF16+MoE-FP8"
  else:
    precision_tag = f"{args.model_type.upper()}/{args.model_type.upper()}"
  head = f"MaxText trainer vs {label} sampler — Qwen3.5-35B-A3B {precision_tag}, full model"

  n_prompt = sa["logp"].shape[1]
  err_thresh = getattr(args, "seq_logprob_error_threshold", 2.0)
  m = compare(tr["logp"][:, : n_prompt - 1], sa["logp"][:, 1:n_prompt], error_threshold=err_thresh)
  print_report(f"{head} — PROMPT tokens (teacher-forced, prefill)", m)
  out = {f"prompt_{k}": v for k, v in m.items()}

  if "gen_logp" in sa.files and "gen_logp" in tr.files:
    tr_gen = tr["gen_logp"]
    sa_gen = sa["gen_logp"]
    token_mask = np.zeros_like(sa_gen, dtype=np.float64)
    if "gen_lens" in sa.files:
      gen_lens = sa["gen_lens"]
      for b in range(len(gen_lens)):
        token_mask[b, : gen_lens[b]] = 1.0
    else:
      token_mask = np.isfinite(sa_gen).astype(np.float64)

    mg = compare(tr_gen, sa_gen, token_mask=token_mask, error_threshold=err_thresh)
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
  global SEQ_LEN
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"],
                  help="all = every stage in this process; the rest hand off through npz files in --out-dir")
  ap.add_argument("--sampler", default="adapter", choices=["adapter"],
                  help="vLLM rollout engine with MaxText adapter (MODEL_IMPL_TYPE=flax_nnx)")
  ap.add_argument("--model-type", default="bf16", choices=["bf16", "fp8"],
                  help="precision configuration for trainer and sampler (bf16 or fp8, default bf16)")
  ap.add_argument("--fp8", action="store_true", help="convenience alias to force --model-type fp8")
  ap.add_argument("--model-hf", default=None,
                  help="HuggingFace model repository or local path (defaults to Qwen/Qwen3.5-35B-A3B for bf16)")
  ap.add_argument("--model-maxtext", default=None,
                  help="MaxText model config identifier (defaults to qwen3.5-35b-a3b for bf16)")
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--hf-home", default=None)
  ap.add_argument("--maxtext-root", default=None)
  ap.add_argument("--ckpt", default=None,
                  help="MaxText checkpoint path (defaults to GCS BF16/FP8 checkpoints)")
  ap.add_argument("--gen-tokens", type=int, default=1024,
                  help="output tokens to roll out per prompt for the decode-path comparison; 0 = prompt only (default 1024)")
  ap.add_argument("--gen-temperature", type=float, default=1.0, help="sampling temperature for the rollouts")
  ap.add_argument("--logits-dot-fp32", action=argparse.BooleanOptionalAction, default=True,
                  help="run the lm_head projection in fp32 (logits_dot_in_fp32) on BOTH trainer and sampler (default True)")
  ap.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True,
                  help="let rollouts end at EOS; short rows are padded and masked out (default True)")
  ap.add_argument("--attn-dp", type=int, default=4, help="attention DP degree inside the tp=8 mesh")
  ap.add_argument("--ep", type=int, default=None,
                  help="override expert parallelism on both sampler and trainer simultaneously")
  ap.add_argument("--sampler-ep", type=int, default=8,
                  help="expert parallelism for the sampler (default 8, matching production rollout config)")
  ap.add_argument("--trainer-ep", type=int, default=1,
                  help="expert parallelism for the trainer (default 1, matching production trainer)")
  ap.add_argument("--sharding-tp", type=int, default=1,
                  help="sharding_strategy.tensor_parallelism for the sampler when sampler_ep > 1 (default 1)")
  ap.add_argument("--sync-src-ep", type=int, default=None,
                  help="expert parallelism of the --weight-sync source model; defaults to --trainer-ep")
  ap.add_argument("--weight-sync", action="store_true",
                  help="before sampling, load trainer model and push params into engine with tunix transfer_state_directly")
  ap.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                  help="fraction of TPU HBM for vLLM KV cache and model (default 0.7)")
  ap.add_argument("--prompt-len", type=int, default=SEQ_LEN,
                  help="prompt tokens per row; overrides the SEQ_LEN default (default 4096)")
  ap.add_argument("--prompts-file", default=None,
                  help="JSONL of real prompts, one {\"text\": ...} per line (auto-detects r2e_prompts_32.jsonl)")
  ap.add_argument("--trainer-micro-batch", type=int, default=2,
                  help="rows per trainer forward pass; must divide row count (default 2)")
  ap.add_argument("--trainer-kv-heads", type=int, default=4,
                  help="replicate the trainer's KV heads up to this count (default 4)")
  ap.add_argument("--trainer-tp", type=int, default=4,
                  help="tensor parallelism for the trainer mesh (default 4)")
  ap.add_argument("--text-glob", default="docs/**/*.md", help="corpus glob, relative to the MaxText tree")
  ap.add_argument("--text-file", default=None, help="single text file, overrides --text-glob")
  ap.add_argument("--retokenize", action="store_true", help="rebuild tokens.npz even if it exists")

  ap.add_argument("--router-replay", action=argparse.BooleanOptionalAction, default=True,
                  help="record sampler routing decisions and force them in trainer via forced_routed_experts (default True)")
  ap.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                  help="enable prefix caching in vLLM sampler (default True, matching MLPerf recipe)")
  ap.add_argument("--mamba-cache-mode", type=str, default="align",
                  help="mamba cache mode: 'align' or 'none' (default 'align', matching MLPerf recipe)")
  ap.add_argument("--seq-logprob-error-threshold", type=float, default=2.0,
                  help="Tunix seq_logprob_error_threshold for gating out-of-distribution trajectories (default 2.0)")
  ap.add_argument("--fp8-moe", action="store_true",
                  help="load BF16 model and quantize only MoE layers to FP8 per-channel (Peano lab post method)")
  ap.add_argument("--moe-only-fp8", dest="fp8_moe", action="store_true",
                  help="alias for --fp8-moe")
  ap.add_argument("--trainer-fp8-moe", action=argparse.BooleanOptionalAction, default=None,
                  help="quantize MoE layers in trainer; defaults to value of --fp8-moe")
  ap.add_argument("--fp8-legacy-round", action="store_true",
                  help="reproduce the original quantizer: jnp.round(w/scale) to an integer before the e4m3 cast")
  ap.add_argument("--fp8-moe-sampler-only", action="store_true",
                  help="quantize only MoE layers to FP8 in sampler, leaving trainer in pure BF16")
  ap.add_argument("--tag", default=None, help="run tag / experiment label")

  args = ap.parse_args()
  if args.fp8_moe_sampler_only:
    args.fp8_moe = True
    args.trainer_fp8_moe = False
  if args.fp8:
    args.model_type = "fp8"
  if args.ep is not None:
    args.sampler_ep = args.ep
    args.trainer_ep = args.ep
  if args.trainer_ep == 8:
    args.trainer_tp = 1
    args.trainer_kv_heads = None
    args.trainer_micro_batch = 1
  SEQ_LEN = args.prompt_len

  maxtext_root, hf_home, out_dir = resolve_paths(args)
  log(f"maxtext_root={maxtext_root}")
  log(f"hf_home={hf_home}")
  log(f"out_dir={out_dir}")
  log(f"model_type={args.model_type} (router_replay={args.router_replay})")

  if args.stage == "compare":
    stage_compare(args, out_dir)
    return

  if args.stage != "tokenize":
    set_tpu_env(hf_home, maxtext_root)

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
