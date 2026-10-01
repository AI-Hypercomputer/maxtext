#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MaxText Trainer vs. vLLM Sampler logprob parity, OOB convergence, and quantization audit script."""

import argparse
import gc
import glob
import json
import os
import sys
import time
import numpy as np

_MAXTEXT_ROOT = os.environ.get("MAXTEXT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _cand in (
    os.path.join(_MAXTEXT_ROOT, "src"),
    os.environ.get("TUNIX_ROOT"),
    os.path.join(os.path.dirname(_MAXTEXT_ROOT), "tunix"),
    "/workspace/tunix",
    "/tpu/tunix",
):
  if _cand and os.path.isdir(_cand) and _cand not in sys.path:
    sys.path.insert(0, _cand)

MODEL_HF_BF16 = os.environ.get("MAXTEXT_HF_BF16", "Qwen/Qwen3.5-35B-A3B")
MODEL_MAXTEXT_BF16 = os.environ.get("MAXTEXT_MODEL_BF16", "qwen3.5-35b-a3b")
CKPT_BF16 = os.environ.get("MAXTEXT_CKPT_BF16", "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items")

MODEL_HF_FP8 = os.environ.get("MAXTEXT_HF_FP8", "Qwen/Qwen3.5-35B-A3B-FP8")
MODEL_MAXTEXT_FP8 = os.environ.get("MAXTEXT_MODEL_FP8", "qwen3.5-35b-a3b-fp8")
CKPT_FP8 = os.environ.get(
    "MAXTEXT_CKPT_FP8", "gs://cloud-devkit/users/wenxindong/ckpt/qwen3.5-35b-a3b-fp8/unscanned/0/items"
)

VALID_MODES = ("bf16", "fp8_ckpt", "fp8_serve", "fp8_moe", "fp8_moe_native", "int8_moe")
VALID_SCALE_MODES = ("per_channel", "subchannel128", "block128")
RATIO_MIN, RATIO_MAX = 0.999, 1.002
t0 = time.time()


def log(*a):
  print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


def set_tpu_env(hf_home):
  os.environ["NEW_MODEL_DESIGN"] = "1"
  os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
  for k, v in {
      "HF_HOME": hf_home,
      "HF_HUB_OFFLINE": "1",
      "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
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
      "LIBTPU_INIT_ARGS": (
          " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
          " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
          " --xla_tpu_ars_combiner_threshold_in_bytes=0"
          " --xla_tpu_enable_async_collective_merger=false"
          " --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
          " --xla_tpu_dvfs_p_state=7"
      ),
  }.items():
    os.environ.setdefault(k, v)


def get_tokenizer(hf_home):
  from transformers import AutoTokenizer

  for name in ("models--Qwen--Qwen3.5-35B-A3B", "models--Qwen--Qwen3.5-35B-A3B-FP8"):
    matches = glob.glob(os.path.join(hf_home, "hub", name, "snapshots", "*"))
    if matches:
      return AutoTokenizer.from_pretrained(matches[0]), matches[0]
  return AutoTokenizer.from_pretrained(MODEL_HF_BF16), MODEL_HF_BF16


# --------------------------------------------------------- Audit & Quantization


def quantize_moe_fp8(model, scale_mode="per_channel", weight_qtype=None, set_serve_quant=False, block_size=128):
  """Quantize RoutedMoE weights in-place via `quantizations.quantize_weight_for_fused_moe` or `qwix.pallas.quantize`."""
  import jax.numpy as jnp
  from flax import nnx
  import qwix
  import qwix.pallas as qpl
  from maxtext.layers import quantizations

  if scale_mode not in VALID_SCALE_MODES:
    raise ValueError(f"Unsupported scale_mode={scale_mode!r}; expected one of {VALID_SCALE_MODES}")
  qtype = jnp.dtype(weight_qtype) if weight_qtype is not None else jnp.dtype(jnp.float8_e4m3fn)
  decoder = getattr(model, "decoder", model)
  num_layers = getattr(getattr(decoder, "config", None), "num_decoder_layers", 40)
  count = 0
  for i in range(num_layers):
    layer = getattr(decoder, f"layers_{i}", None)
    routed = getattr(getattr(layer, "mlp", None), "routed_experts", None) if layer is not None else None
    if routed is None:
      continue
    wi_axes, wo_axes = getattr(routed, "wi_kernel_axes", None), getattr(routed, "wo_kernel_axes", None)
    for w_name, s_name, w_ax in [
        ("wo", "wo_scale", wo_axes),
        ("wi", "wi_scale", wi_axes),
        ("wi_0", "wi_0_scale", wi_axes),
        ("wi_1", "wi_1_scale", wi_axes),
    ]:
      p = getattr(routed, w_name, None)
      if p is None:
        continue
      w = p[...]
      _, K, N = w.shape
      if scale_mode == "block128" and K % block_size == 0 and N % block_size == 0:
        q = qpl.quantize(w, qtype, channelwise_axes=(0,), tiled_axes={1: block_size, 2: block_size}, scale_dtype=jnp.float32)
        qw, qs = q.qvalue, q.scale
      else:
        tile_size = block_size if (scale_mode == "subchannel128" and K % block_size == 0) else None
        qw, scale_4d = quantizations.quantize_weight_for_fused_moe(w, qwix.QtRule(weight_qtype=qtype, tile_size=tile_size))
        qs = jnp.squeeze(scale_4d, axis=2)
      s_ax = ((w_ax[0], None, w_ax[2]) if qs.shape[1] == 1 else w_ax) if (w_ax and len(w_ax) >= 3) else None
      setattr(routed, w_name, nnx.Param(qw, out_sharding=w_ax))
      setattr(routed, s_name, nnx.Param(qs, out_sharding=s_ax))
    routed.weight_dtype = qtype
    if set_serve_quant:
      routed.quant = quantizations.ServeFp8WeightQuantization()
    count += 1
  log(f"quantize_moe_fp8: quantized {count} RoutedMoE layers (dtype={qtype}, scale={scale_mode}, serve={set_serve_quant})")
  return count


def audit_model(model, role: str, expected_mode: str, out_dir: str):
  """Inspect live NNX parameters and determine exact execution kernel paths."""
  from maxtext.common import common_types as ctypes
  from maxtext.layers import quantizations

  def _info(p):
    if p is None:
      return None
    arr = p[...] if hasattr(p, "__getitem__") else getattr(p, "value", p)
    s = np.asarray(arr[tuple(slice(0, min(d, 8)) for d in arr.shape)], dtype=np.float32)
    return {
        "dtype": str(arr.dtype),
        "shape": [int(d) for d in arr.shape],
        "sample_absmax": float(np.max(np.abs(s))),
        "sample_min": float(np.min(s)),
        "sample_max": float(np.max(s)),
    }

  decoder = getattr(model, "decoder", model)
  cfg, l0, l3 = getattr(decoder, "config", None), getattr(decoder, "layers_0", None), getattr(decoder, "layers_3", None)
  routed0 = getattr(getattr(l0, "mlp", None), "routed_experts", None)
  shared0 = getattr(getattr(l0, "mlp", None), "shared_expert", None)
  qkvz_mod = getattr(getattr(l0, "attention", None), "in_proj_qkvz", None)
  attn3_wrap = getattr(l3, "attention", None) if l3 is not None else None
  attn3 = getattr(attn3_wrap, "attention", attn3_wrap)

  wi_attr = "wi" if getattr(routed0, "wi", None) is not None else "wi_0"
  wi_s_attr = "wi_scale" if getattr(routed0, "wi_scale", None) is not None else "wi_0_scale"
  modules = {
      name: {"weight": _info(getattr(mod, w_a, None)), "scale": _info(getattr(mod, s_a, None))}
      for name, mod, w_a, s_a in [
          ("layers_0.gdn.in_proj_qkvz", qkvz_mod, "kernel", "kernel_scale"),
          ("layers_0.mlp.shared_expert.wi_0", getattr(shared0, "wi_0", None), "kernel", "kernel_scale"),
          ("layers_0.mlp.routed_experts.wi", routed0, wi_attr, wi_s_attr),
          ("layers_0.mlp.routed_experts.wo", routed0, "wo", "wo_scale"),
          ("layers_3.attn.query", getattr(attn3, "query", None), "kernel", "kernel_scale"),
      ]
  }
  wi_info, wi_scale_info = modules["layers_0.mlp.routed_experts.wi"]["weight"], modules["layers_0.mlp.routed_experts.wi"]["scale"]
  qkvz_info, qkvz_scale_info = modules["layers_0.gdn.in_proj_qkvz"]["weight"], modules["layers_0.gdn.in_proj_qkvz"]["scale"]

  is_fused_moe = getattr(cfg, "attention", "") in ("vllm_rpa", "vllm_batched_rpa") and not getattr(routed0, "is_hash_routing", False)
  is_fp8_moe = (
      getattr(cfg, "fp8_moe", False)
      or ctypes.is_fp8_dtype(getattr(routed0, "weight_dtype", None))
      or isinstance(getattr(routed0, "quant", None), quantizations.ServeFp8WeightQuantization)
  )
  wi_is_fp8 = wi_info is not None and "float8" in wi_info["dtype"]
  wi_is_int8 = wi_info is not None and "int8" in wi_info["dtype"]

  if is_fused_moe:
    if is_fp8_moe and wi_is_fp8 and wi_scale_info is not None:
      moe_path = f"FUSED_MOE_NATIVE_FP8 (tpu_inference, scale={wi_scale_info['shape']})"
    elif wi_is_int8 and wi_scale_info is not None:
      moe_path = f"FUSED_MOE_DEQUANT_TO_BF16 (dequantize_weight -> tpu_inference, scale={wi_scale_info['shape']})"
    else:
      moe_path = "FUSED_MOE_BF16 (tpu_inference)"
  else:
    scheme, axis = quantizations.infer_scale_granularity(tuple(wi_scale_info["shape"][1:])) if wi_scale_info else (None, None)
    native_gmm = (
        isinstance(getattr(routed0, "quant", None), quantizations.ServeFp8WeightQuantization)
        and getattr(cfg, "sparse_matmul", False)
        and getattr(cfg, "use_gmm_v2", False)
        and not getattr(cfg, "prefuse_moe_weights", False)
        and (scheme == "per_tensor" or (scheme == "per_channel" and axis == 1))
    )
    if native_gmm and wi_is_fp8 and wi_scale_info is not None:
      moe_path = f"GMM_V2_NATIVE_FP8 (qpl.QArray, scheme={scheme}, scale={wi_scale_info['shape']})"
    elif wi_is_fp8 or wi_is_int8 or wi_scale_info is not None:
      moe_path = f"GMM_V2_DEQUANT_TO_BF16 (dequantize_weight -> BF16 gmm_v2, scheme={scheme}, scale={wi_scale_info['shape'] if wi_scale_info else None})"
    else:
      moe_path = "GMM_V2_PURE_BF16 (BF16 gmm_v2)"

  if isinstance(getattr(qkvz_mod, "quant", None), quantizations.ServeFp8WeightQuantization) and qkvz_info and "float8" in qkvz_info["dtype"] and qkvz_scale_info:
    dense_path = "DENSE_NATIVE_FP8 (qwix W8A8 on 1D-contracted linears; multi-axis dequantizes to BF16)"
  elif qkvz_scale_info or (qkvz_info and "float8" in qkvz_info["dtype"]):
    dense_path = f"DENSE_DEQUANT_TO_BF16 (dequantize_weight -> BF16 dot_general, scale={qkvz_scale_info['shape'] if qkvz_scale_info else None})"
  else:
    dense_path = "DENSE_PURE_BF16 (BF16 dot_general)"

  report = {
      "role": role,
      "expected_mode": expected_mode,
      "config": {k: getattr(cfg, k, None) if k in ("model_name", "attention") else str(getattr(cfg, k, None)) for k in ("model_name", "attention", "weight_dtype", "dtype", "quantization")},
      "execution_paths": {"routed_moe": moe_path, "dense_linear": dense_path},
      "modules": modules,
  }
  print(f"\n================ MODEL AUDIT: {role.upper()} (mode={expected_mode}) ================")
  print(f"  MoE Exec Path  : {moe_path}\n  Dense Exec Path: {dense_path}")
  for mod_name, mdata in modules.items():
    w, s = mdata["weight"], mdata["scale"]
    w_str = f"{w['dtype']} {tuple(w['shape'])} (absmax={w['sample_absmax']:.4g})" if w else "None"
    s_str = f"{s['dtype']} {tuple(s['shape'])} (range=[{s['sample_min']:.4g}, {s['sample_max']:.4g}])" if s else "None"
    print(f"  {mod_name:<34}: weight={w_str:<42} | scale={s_str}")
  print("=" * 84 + "\n", flush=True)

  if expected_mode in ("fp8_ckpt", "fp8_serve", "fp8_moe", "fp8_moe_native"):
    assert wi_is_fp8 and wi_scale_info is not None and wi_info["sample_absmax"] > 1.0, f"[{role}] Invalid FP8 MoE: {wi_info}"
    if expected_mode == "fp8_moe_native" and not is_fused_moe:
      assert moe_path.startswith("GMM_V2_NATIVE_FP8"), f"[{role}] Expected GMM_V2_NATIVE_FP8, got {moe_path}"
    if expected_mode == "fp8_serve":
      assert dense_path.startswith("DENSE_NATIVE_FP8"), f"[{role}] Expected DENSE_NATIVE_FP8, got {dense_path}"
  elif expected_mode == "int8_moe":
    assert wi_is_int8 and wi_scale_info is not None and wi_info["sample_absmax"] > 1.0, f"[{role}] Invalid INT8 MoE: {wi_info}"
  elif expected_mode == "bf16":
    assert wi_info is not None and "bfloat16" in wi_info["dtype"], f"[{role}] Expected bfloat16 MoE, got {wi_info}"

  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"audit_{role}.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
  return report


# --------------------------------------------------------- Stage 1: Tokenize


def stage_tokenize(args, hf_home, out_dir):
  path = os.path.join(out_dir, "tokens.npz")
  if os.path.exists(path) and not args.retokenize:
    return np.load(path)["tokens"]
  tok, _ = get_tokenizer(hf_home)
  rows = []
  with open(args.prompts_file, encoding="utf-8") as f:
    for ln in f:
      if ln.strip():
        ids = tok(json.loads(ln)["text"], add_special_tokens=False)["input_ids"]
        assert len(ids) >= args.prompt_len, f"Prompt shorter than {args.prompt_len}: {len(ids)}"
        rows.append(ids[: args.prompt_len])
        if args.num_prompts and len(rows) >= args.num_prompts:
          break
  tokens = np.array(rows, dtype=np.int32)
  np.savez(path, tokens=tokens)
  log(f"tokenize: saved {tokens.shape} -> {path}")
  return tokens


# --------------------------------------------------------- Stage 2: Sampler


def stage_sampler(args, hf_home, maxtext_root, out_dir):
  try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
  except ImportError:
    pass
  import jax.numpy as jnp
  from tunix.rl import common as rl_common
  from tunix.utils import maxtext_utils as tunix_maxtext_utils
  from vllm import LLM, SamplingParams
  from vllm.inputs import TokensPrompt

  mode = args.sampler_mode
  is_fp8_ckpt = mode in ("fp8_ckpt", "fp8_serve")
  ckpt_path = args.sampler_ckpt or (CKPT_FP8 if is_fp8_ckpt else CKPT_BF16)
  tokens = np.load(os.path.join(out_dir, "tokens.npz"))["tokens"]
  n_seqs, prompt_len = tokens.shape
  _, tok_name = get_tokenizer(hf_home)

  sys.path.insert(0, os.path.join(maxtext_root, "src", "maxtext", "integration", "vllm"))
  import maxtext_vllm_adapter

  maxtext_vllm_adapter.register()

  additional_config = tunix_maxtext_utils.build_vllm_maxtext_additional_config(
      model_name=MODEL_MAXTEXT_FP8 if is_fp8_ckpt else MODEL_MAXTEXT_BF16,
      attention="vllm_rpa",
      prefuse_moe_weights=not is_fp8_ckpt,
      return_routed_experts=args.router_replay,
      float32_gate_logits=True,
      float32_logits=True,
  )
  mt_cfg = additional_config["maxtext_config"]
  mt_cfg.update(
      load_parameters_path=ckpt_path,
      dtype="bfloat16",
      enable_dp_attention=True,
      enable_checkpointing=True,
      async_checkpointing=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      convert_checkpoint_if_possible=False,
      float32_weight_sum=True,
      logits_dot_in_fp32=True,
  )
  if is_fp8_ckpt:
    mt_cfg.update(weight_dtype="float8_e4m3fn", fp8_moe=True, weight_block_size=128)
  if mode == "fp8_serve":
    mt_cfg["quantization"] = "serve_fp8_weight"
  additional_config["custom_mamba_cache_multiplier"] = 16
  additional_config["sharding"] = {
      "sharding_strategy": {"expert_parallelism": args.sampler_ep, "tensor_parallelism": 1, "enable_dp_attention": True}
  }

  in_place_moe = mode in ("fp8_moe", "fp8_moe_native", "int8_moe")
  llm = LLM(
      model=tok_name,
      tokenizer=tok_name,
      trust_remote_code=True,
      dtype="bfloat16",
      max_model_len=max(4096, prompt_len + args.gen_tokens + 64),
      max_num_seqs=16,
      max_num_batched_tokens=max(4096, prompt_len),
      block_size=256,
      enable_chunked_prefill=True,
      enable_prefix_caching=True,
      mamba_cache_mode="align",
      gpu_memory_utilization=0.6 if in_place_moe else 0.7,
      language_model_only=True,
      limit_mm_per_prompt={"image": 0, "video": 0},
      disable_log_stats=True,
      kv_cache_dtype="bfloat16",
      reasoning_parser="qwen3",
      seed=0,
      hf_overrides=dict(tunix_maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES),
      additional_config=additional_config,
      **({"enable_return_routed_experts": True} if args.router_replay else {}),
  )
  runner = llm.llm_engine.model_executor.driver_worker.model_runner
  for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
    if obj is not None and hasattr(obj, "uses_mrope"):
      obj.uses_mrope = False
  runner.get_mrope_input_positions_fn = runner.model.get_mrope_input_positions

  if in_place_moe:
    quantize_moe_fp8(
        runner.model.model,
        scale_mode=args.moe_scale_mode,
        weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn,
        set_serve_quant=(mode == "fp8_moe_native"),
    )
  audit_model(runner.model.model, role="sampler", expected_mode=mode, out_dir=out_dir)
  if args.audit_only:
    del runner, llm
    gc.collect()
    return

  max_toks = max(1, args.gen_tokens)
  sp = SamplingParams(
      max_tokens=max_toks,
      min_tokens=max_toks if args.gen_tokens > 0 else 1,
      temperature=args.gen_temperature if args.gen_tokens > 0 else 0.0,
      top_p=1.0,
      top_k=-1,
      prompt_logprobs=1,
      logprobs=1 if args.gen_tokens > 0 else None,
      ignore_eos=(args.gen_tokens > 0),
  )
  if args.router_replay:
    setattr(sp, "routed_experts_prompt_start", 0)

  outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)

  def _unpack_lp(lp_list, tids):
    lp = np.array([np.nan if d is None or int(t) not in d else d[int(t)].logprob for d, t in zip(lp_list, tids)], np.float32)
    t1 = np.array([-1 if d is None else max(d.items(), key=lambda kv: kv[1].logprob)[0] for d in lp_list], np.int32)
    return lp, t1

  prompt_logp, prompt_top1 = np.full((n_seqs, prompt_len), np.nan, np.float32), np.full((n_seqs, prompt_len), -1, np.int32)
  gen_ids = np.zeros((n_seqs, args.gen_tokens), np.int32) if args.gen_tokens > 0 else None
  gen_logp = np.full((n_seqs, args.gen_tokens), np.nan, np.float32) if args.gen_tokens > 0 else None
  gen_top1 = np.full((n_seqs, args.gen_tokens), -1, np.int32) if args.gen_tokens > 0 else None
  all_routed = []

  for b, o in enumerate(outs):
    prompt_logp[b], prompt_top1[b] = _unpack_lp(o.prompt_logprobs, tokens[b])
    c = o.outputs[0]
    if args.gen_tokens > 0:
      gen_ids[b] = list(c.token_ids)
      gen_logp[b], gen_top1[b] = _unpack_lp(c.logprobs, gen_ids[b])
    if args.router_replay:
      re_arr = np.asarray(c.routed_experts, dtype=np.int16)
      if args.gen_tokens > 0:
        pad_last = np.full((1, re_arr.shape[1], re_arr.shape[2]), rl_common.UNSET_ROUTED_EXPERT, dtype=np.int16)
        re_arr = np.concatenate([re_arr, pad_last], axis=0)
      all_routed.append(re_arr)

  saved = dict(logp=prompt_logp, top1=prompt_top1, tokens=tokens)
  if args.gen_tokens > 0:
    saved.update(gen_ids=gen_ids, gen_logp=gen_logp, gen_top1=gen_top1, gen_lens=np.full(n_seqs, args.gen_tokens, np.int32))
  np.savez(os.path.join(out_dir, "sampler_logprobs.npz"), **saved)
  if args.router_replay:
    np.savez_compressed(os.path.join(out_dir, "router_indices.npz"), experts=np.stack(all_routed, axis=0))

  del runner, outs, llm
  gc.collect()


# --------------------------------------------------------- Stage 3: Trainer


def stage_trainer(args, hf_home, maxtext_root, out_dir):
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
  from maxtext.utils import model_creation_utils
  from tunix.rl import common as rl_common
  from tunix.utils import maxtext_utils as tunix_maxtext_utils

  mode = args.trainer_mode
  is_fp8_ckpt = mode in ("fp8_ckpt", "fp8_serve")
  ckpt_path = args.trainer_ckpt or (CKPT_FP8 if is_fp8_ckpt else CKPT_BF16)

  prompt_np = np.load(os.path.join(out_dir, "tokens.npz"))["tokens"]
  sa_path = os.path.join(out_dir, "sampler_logprobs.npz")
  gen_ids = np.load(sa_path)["gen_ids"] if (not args.audit_only and os.path.exists(sa_path) and "gen_ids" in np.load(sa_path).files) else None
  tokens_np = prompt_np if gen_ids is None else np.concatenate([prompt_np, gen_ids], axis=1).astype(np.int32)
  n_prompt, orig_seq_len = prompt_np.shape[1], tokens_np.shape[1]
  pad_len = (512 - (orig_seq_len % 512)) % 512
  seq_len = orig_seq_len + pad_len

  dp_fsdp = max(1, len(jax.devices()) // max(1, args.trainer_tp * args.trainer_ep))
  micro = ((max(args.trainer_micro_batch, dp_fsdp) + dp_fsdp - 1) // dp_fsdp) * dp_fsdp
  _, tok_name = get_tokenizer(hf_home)

  extra_flags = [
      f"tokenizer_path={tok_name}",
      "scan_layers=False",
      "enable_checkpointing=True",
      "async_checkpointing=False",
      "checkpoint_storage_use_ocdbt=True",
      "checkpoint_storage_use_zarr3=True",
      "log_config=False",
      "float32_weight_sum=True",
      "logits_dot_in_fp32=True",
      f"use_gdn_kernel={args.trainer_tp == 1}",
      "gdn_chunk_size=64",
      "use_tokamax_splash=True",
      "sa_use_base2_exp=True",
      "sa_fuse_reciprocal=False",
      "sparse_matmul=True",
      "megablox=False",
      "wi_tile_fwd_batch_seq=512",
      "wi_tile_fwd_embed_dim=1024",
      "wi_tile_fwd_mlp_dim=1024",
      "wo_tile_fwd_batch_seq=512",
      "wo_tile_fwd_embed_dim=1024",
      "wo_tile_fwd_mlp_dim=1024",
      f"max_prefill_predict_length={seq_len}",
  ]
  if is_fp8_ckpt:
    extra_flags += ["weight_dtype=float8_e4m3fn", "fp8_moe=True", "weight_block_size=128"]
  if mode == "fp8_serve":
    extra_flags.append("quantization=serve_fp8_weight")
  os.environ["MAXTEXT_EXTRA_FLAGS"] = " ".join(extra_flags)

  cfg = tunix_maxtext_utils.build_maxtext_config(
      model_name=MODEL_MAXTEXT_FP8 if is_fp8_ckpt else MODEL_MAXTEXT_BF16,
      worker_id="rl_parity_audit",
      train_micro_batch_size=micro,
      mesh_fsdp=dp_fsdp,
      mesh_tp=args.trainer_tp,
      mesh_expert=args.trainer_ep,
      num_devices=len(jax.devices()),
      max_prompt_length=n_prompt,
      max_response_length=seq_len - n_prompt,
      load_parameters_path=ckpt_path,
      base_output_directory=os.path.join(out_dir, "maxtext_run"),
      base_num_kv_heads=max(2, args.trainer_tp),
      prefuse_moe_weights=not (is_fp8_ckpt or mode == "fp8_moe_native"),
      attention="flash",
      float32_gate_logits=True,
      float32_logits=True,
  )
  model, mesh = model_creation_utils.from_pretrained(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN)

  if mode in ("fp8_moe", "fp8_moe_native", "int8_moe"):
    quantize_moe_fp8(
        model,
        scale_mode=args.moe_scale_mode,
        weight_qtype=jnp.int8 if mode == "int8_moe" else jnp.float8_e4m3fn,
        set_serve_quant=(mode == "fp8_moe_native"),
    )
  audit_model(model, role="trainer", expected_mode=mode, out_dir=out_dir)
  if args.audit_only:
    return

  gd, st = nnx.split(model)
  gen_temp = float(args.gen_temperature)

  @jax.jit
  def fwd_logp(st, tokens, pos, seg, nxt, fre=None):
    m = nnx.merge(gd, st)
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      out = m(tokens, pos, seg, enable_dropout=False, model_mode=MODEL_MODE_TRAIN, **({"forced_routed_experts": fre} if fre is not None else {}))
    logits = out[0] if isinstance(out, tuple) else out
    if gen_temp not in (0.0, 1.0):
      logits = logits.astype(jnp.float32) / jnp.where(pos >= (n_prompt - 1), jnp.float32(gen_temp), jnp.float32(1.0))[..., None]
    return rl_common.selective_log_softmax(logits, nxt), jnp.argmax(logits, -1).astype(jnp.int32)

  n_rows = tokens_np.shape[0]
  fre_all = None
  if args.router_replay:
    raw_fre = np.load(os.path.join(out_dir, "router_indices.npz"))["experts"]
    fre_all = rl_common.align_routed_experts(
        list(raw_fre), completion_lengths=[orig_seq_len - n_prompt] * n_rows, prompt_width=n_prompt, completion_width=seq_len - n_prompt
    )

  tokens_pad = np.pad(tokens_np, ((0, 0), (0, pad_len)))
  nxt_pad = np.pad(np.roll(tokens_np, -1, axis=1), ((0, 0), (0, pad_len)))
  seg_pad = np.pad(np.ones((micro, orig_seq_len), np.int32), ((0, 0), (0, pad_len)))

  logp_parts, top1_parts = [], []
  with jax.set_mesh(mesh), nn.logical_axis_rules(cfg.logical_axis_rules):
    sh, sh_fre = NamedSharding(mesh, P(("data", "fsdp"), None)), NamedSharding(mesh, P(("data", "fsdp"), None, None, None))
    pos, seg = jax.device_put(np.broadcast_to(np.arange(seq_len, dtype=np.int32), (micro, seq_len)), sh), jax.device_put(seg_pad, sh)
    for i in range(0, n_rows, micro):
      lp_d, t1_d = fwd_logp(
          st,
          jax.device_put(tokens_pad[i : i + micro], sh),
          pos,
          seg,
          jax.device_put(nxt_pad[i : i + micro], sh),
          jax.device_put(fre_all[i : i + micro], sh_fre) if fre_all is not None else None,
      )
      jax.block_until_ready(lp_d)
      logp_parts.append(np.asarray(lp_d)[:, :orig_seq_len])
      top1_parts.append(np.asarray(t1_d)[:, :orig_seq_len])

  logp, top1 = np.concatenate(logp_parts, axis=0), np.concatenate(top1_parts, axis=0)
  saved = dict(logp=logp, top1=top1, tokens=prompt_np, full_tokens=tokens_np)
  if gen_ids is not None:
    saved.update(gen_logp=logp[:, n_prompt - 1 : orig_seq_len - 1], gen_top1=top1[:, n_prompt - 1 : orig_seq_len - 1])
  np.savez(os.path.join(out_dir, "trainer_logprobs.npz"), **saved)


# --------------------------------------------------------- Stage 4: Compare


def stage_compare(args, out_dir):
  import jax.numpy as jnp
  from tunix.rl import algo_core, common as rl_common

  for role in ("sampler", "trainer"):
    ap = os.path.join(out_dir, f"audit_{role}.json")
    if os.path.exists(ap):
      with open(ap, encoding="utf-8") as f:
        r = json.load(f)
      print(f"[{role.upper()} AUDIT] mode={r['expected_mode']} | MoE={r['execution_paths']['routed_moe']} | Dense={r['execution_paths']['dense_linear']}")

  tr, sa = np.load(os.path.join(out_dir, "trainer_logprobs.npz")), np.load(os.path.join(out_dir, "sampler_logprobs.npz"))
  assert np.array_equal(tr["tokens"], sa["tokens"]), "Token mismatch between sampler and trainer!"

  def _eval_band(title, tr_lp, sa_lp, tr_top1=None, sa_top1=None, token_mask=None):
    tr_lp, sa_lp = np.asarray(tr_lp, dtype=np.float32), np.asarray(sa_lp, dtype=np.float32)
    raw = tr_lp - sa_lp
    token_mask = np.ones_like(raw, dtype=np.float32) if token_mask is None else np.asarray(token_mask, dtype=np.float32)
    raw_jnp, mask_jnp = jnp.asarray(raw), jnp.asarray(token_mask)
    log_is_jnp = jnp.nan_to_num(raw_jnp, nan=0.0, posinf=0.0, neginf=0.0)

    seq_mask_res = algo_core.sequence_loss_mask(mask_jnp, log_is_raw=raw_jnp, mult_prob_error_threshold=2.0)
    seq_geomean_jnp, seq_valid_jnp = algo_core.sequence_geomean_ratio(log_is_jnp, mask_jnp)
    _, is_oob_jnp = algo_core.truncated_importance_weights(
        raw_jnp, seq_geomean_jnp, seq_valid_jnp, seq_mask_res.sample_mask, RATIO_MIN, RATIO_MAX
    )
    attr = algo_core.log_is_attribution(log_is_jnp, mask_jnp, jnp.asarray(sa_lp))
    finite_mask = (token_mask > 0) & np.isfinite(raw)
    agree = rl_common.sampler_trainer_agreement(
        np.where(finite_mask, sa_lp, 0.0), np.where(finite_mask, tr_lp, 0.0), finite_mask.astype(np.float32)
    )[0]

    seq_mult_err = np.asarray(seq_mask_res.mult_prob_error, dtype=np.float64)
    scored_seq = (seq_mult_err > 0) & np.isfinite(seq_mult_err)
    mult_err_mean = float(algo_core.masked_mean(jnp.where(scored_seq, seq_mult_err, 0.0), scored_seq.astype(np.float32))) if np.any(scored_seq) else 1.0
    in_tok = ((jnp.exp(log_is_jnp) >= RATIO_MIN) & (jnp.exp(log_is_jnp) <= RATIO_MAX) & jnp.isfinite(raw_jnp)).astype(jnp.float32)
    tok_oob = float(algo_core.masked_mean(1.0 - in_tok, mask_jnp))
    is_oob = float(is_oob_jnp)

    top1_agree = None
    if tr_top1 is not None and sa_top1 is not None:
      valid_t1 = (token_mask > 0) & (np.asarray(sa_top1) >= 0) & (np.asarray(tr_top1) >= 0)
      if np.any(valid_t1):
        top1_agree = float(np.mean(np.asarray(tr_top1)[valid_t1] == np.asarray(sa_top1)[valid_t1]))

    abs_mat = np.where(token_mask > 0, np.where(np.isfinite(raw), np.abs(raw), np.inf), -1.0)
    print(f"\n### {title}")
    print(f"  tokens={int(token_mask.sum())}  seqs={raw.shape[0]}  nonfinite={int((~np.isfinite(raw) & (token_mask > 0)).sum())}  band=[{RATIO_MIN}, {RATIO_MAX}]"
          + (f"  top1_agree={top1_agree:.2%}" if top1_agree is not None else ""))
    print(f"  >>> is_oob_ratio (seq-mask-tis)       = {is_oob:.4f} ({is_oob:.2%})")
    print(f"      oob_ratio    (per-token)         = {tok_oob:.4f} ({tok_oob:.2%})")
    print(f"      sampler_is/token_logdiff_mean    = {float(attr['sampler_is/token_logdiff_mean']):+.6f}")
    print(f"      sampler_is/token_logdiff_absmean = {float(agree['sampler_trainer/logp_diff_mean'][0]):.6f}")
    print(f"      sample_mask/mult_prob_error_mean = {mult_err_mean:.6f}")
    print("  Top-5 Worst Token Divergences:")
    for idx in np.argsort(abs_mat.ravel())[::-1][:5]:
      b, t = divmod(int(idx), abs_mat.shape[1])
      nxt5 = [round(x, 4) if np.isfinite(x) else float("inf") for x in abs_mat[b, t + 1 : t + 6].tolist()]
      print(f"    seq={b:2d} pos={t:4d} | |dlogp|={abs_mat[b, t]:7.4f} (sampler={sa_lp[b, t]:8.4f}, trainer={tr_lp[b, t]:8.4f}) | next_5={nxt5}")

    return {
        "is_oob_ratio": is_oob,
        "token_oob_ratio": tok_oob,
        "top1_agree": top1_agree,
        "kept_frac": float(algo_core.masked_mean(seq_mask_res.sample_mask, seq_valid_jnp)),
        "mult_prob_error_mean": mult_err_mean,
    }

  tag = f"Sampler={args.sampler_mode.upper()} vs Trainer={args.trainer_mode.upper()}"
  n_p = sa["logp"].shape[1]
  sa_p_lp = sa["logp"][:, 1:n_p]
  mp = _eval_band(
      f"{tag} — PROMPT tokens (prefill)",
      tr["logp"][:, : n_p - 1],
      sa_p_lp,
      tr_top1=tr["top1"][:, : n_p - 1] if "top1" in tr.files else None,
      sa_top1=sa["top1"][:, 1:n_p] if "top1" in sa.files else None,
      token_mask=np.isfinite(sa_p_lp).astype(np.float32),
  )
  mg = None
  if "gen_logp" in sa.files and "gen_logp" in tr.files:
    mg = _eval_band(
        f"{tag} — OUTPUT tokens (decode)",
        tr["gen_logp"],
        sa["gen_logp"],
        tr_top1=tr["gen_top1"] if "gen_top1" in tr.files else None,
        sa_top1=sa["gen_top1"] if "gen_top1" in sa.files else None,
    )
  return {"prompt": mp, "decode": mg}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--stage", default="all", choices=["all", "tokenize", "sampler", "trainer", "compare"])
  ap.add_argument("--mode", default=None, choices=VALID_MODES)
  ap.add_argument("--sampler-mode", default="bf16", choices=VALID_MODES)
  ap.add_argument("--trainer-mode", default="bf16", choices=VALID_MODES)
  ap.add_argument("--moe-scale-mode", default="per_channel", choices=VALID_SCALE_MODES)
  ap.add_argument("--audit-only", action="store_true")
  for flag in ("--out-dir", "--hf-home", "--maxtext-root", "--sampler-ckpt", "--trainer-ckpt", "--prompts-file"):
    ap.add_argument(flag, default=None)
  for flag, default in (("--num-prompts", 32), ("--prompt-len", 4096), ("--gen-tokens", 4096), ("--sampler-ep", 8), ("--trainer-ep", 1), ("--trainer-tp", 2), ("--trainer-micro-batch", 4)):
    ap.add_argument(flag, type=int, default=default)
  ap.add_argument("--gen-temperature", type=float, default=1.0)
  ap.add_argument("--router-replay", action=argparse.BooleanOptionalAction, default=True)
  ap.add_argument("--retokenize", action="store_true")
  args = ap.parse_args(argv)

  if args.mode:
    args.sampler_mode = args.trainer_mode = args.mode
  maxtext_root = args.maxtext_root or _MAXTEXT_ROOT
  hf_home = args.hf_home or os.environ.get("HF_HOME") or next(
      (c for c in ("/mnt/disks/persist", "/workspace/persist", os.path.expanduser("~/.cache/huggingface")) if os.path.isdir(os.path.join(c, "hub"))),
      os.path.expanduser("~/.cache/huggingface"),
  )
  out_dir = args.out_dir or os.path.join(hf_home, "rl_logprob_parity_audit")
  os.makedirs(out_dir, exist_ok=True)
  args.prompts_file = args.prompts_file or os.path.join(maxtext_root, "tools", "rl_logprob_parity", "r2e_prompts_32.jsonl")

  if args.stage == "compare":
    return stage_compare(args, out_dir)
  if args.stage != "tokenize":
    set_tpu_env(hf_home)
  stage_tokenize(args, hf_home, out_dir)
  if args.stage in ("all", "sampler"):
    stage_sampler(args, hf_home, maxtext_root, out_dir)
  if args.stage in ("all", "trainer"):
    stage_trainer(args, hf_home, maxtext_root, out_dir)
  if args.stage == "all" and not args.audit_only:
    return stage_compare(args, out_dir)


if __name__ == "__main__":
  main()
