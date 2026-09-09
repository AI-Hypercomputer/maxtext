"""MaxText-in-vLLM (OOT adapter) 35B sampler: attn_dp=4 x tp=2 (MoE runs TP-8: EP not reachable on this path), bf16,
real engine prompt_logprobs on the real token sequences."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, sys, time, numpy as np
sys.path.insert(0, "/home/wenxindong_google_com/work/maxtext/.claude/worktrees/pr4925/src/maxtext/integration/vllm")
import maxtext_vllm_adapter; maxtext_vllm_adapter.register()
from vllm import LLM, SamplingParams, TokensPrompt
OUT = "/mnt/disks/persist/pr4925_repro"; MODEL = "Qwen/Qwen3.5-35B-A3B"; ATTN_DP = int(os.environ.get("ATTN_DP", "4"))
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
tokens = np.load(f"{OUT}/real35b_L3.npz")["tokens"]; B, S = tokens.shape
mt_cfg = {"model_name": "qwen3.5-35b-a3b", "load_parameters_path": CKPT, "weight_dtype": "bfloat16", "dtype": "bfloat16",
          "attention": "vllm_rpa", "allow_split_physical_axes": True, "scan_layers": False, "enable_nnx": True, "pure_nnx": True,
          "pure_nnx_decoder": True, "prefuse_moe_weights": True, "enable_dp_attention": ATTN_DP > 1, "log_config": False,
          "enable_checkpointing": True, "async_checkpointing": False, "checkpoint_storage_use_ocdbt": True,
          "checkpoint_storage_use_zarr3": True, "convert_checkpoint_if_possible": False,
          "float32_logits": True, "float32_gate_logits": True, "float32_weight_sum": True}
sharding = {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": ATTN_DP}} if ATTN_DP > 1 else None
llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=8, enable_expert_parallel=True,
          hf_overrides={"architectures": ["MaxTextForCausalLM"]},
          additional_config={"maxtext_config": mt_cfg, **({"sharding": sharding} if sharding else {})},
          max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=2048, block_size=256, enable_chunked_prefill=True,
          enable_prefix_caching=False, gpu_memory_utilization=0.5, language_model_only=True,
          limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=True, kv_cache_dtype="bfloat16")
log("adapter engine up")
# text-only run: the MaxText adapter has no M-RoPE hook (get_mrope_input_positions_fn is None) -> use plain 1-D positions
runner = llm.llm_engine.model_executor.driver_worker.model_runner
log(f"uses_mrope={runner.uses_mrope}; mrope fn={runner.get_mrope_input_positions_fn}")
for obj in (runner, getattr(runner, "persistent_batch_manager", None), getattr(runner, "input_batch", None)):
  if obj is not None and hasattr(obj, "uses_mrope"):
    obj.uses_mrope = False
def _text_mrope(prompt_token_ids, mm_features):
  pos = np.arange(len(prompt_token_ids), dtype=np.int64); return np.stack([pos, pos, pos]), 0
runner.get_mrope_input_positions_fn = _text_mrope
sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)
lp = np.full((B, S), np.nan, np.float32); top1 = np.full((B, S), -1, np.int32)
for b, o in enumerate(outs):
  for i, d in enumerate(o.prompt_logprobs):
    if d is None: continue
    tid = int(tokens[b, i]); lp[b, i] = d[tid].logprob if tid in d else np.nan
    top1[b, i] = max(d.items(), key=lambda kv: kv[1].logprob)[0]
np.savez(f"{OUT}/adapter35b_prompt_logprobs_dp{ATTN_DP}.npz", logp=lp, top1=top1)
log(f"saved; mean logp={np.nanmean(lp[:, 1:]):.3f}; top-1 acc vs actual={np.mean(top1[:, 1:] == tokens[:, 1:]):.3f}")
