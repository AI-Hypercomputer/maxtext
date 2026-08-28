"""Production 35B engine (attn_dp=4 tp=2 EP, bf16): per-token prompt logprobs of the real token sequences."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, time, numpy as np
from vllm import LLM, SamplingParams, TokensPrompt
OUT = "/mnt/disks/persist/pr4925_repro"; MODEL = "Qwen/Qwen3.5-35B-A3B"; ATTN_DP = int(os.environ.get("ATTN_DP", "4"))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
tokens = np.load(f"{OUT}/real35b_L3.npz")["tokens"]; B, S = tokens.shape
sharding = {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": ATTN_DP}} if ATTN_DP > 1 else None
llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=8, enable_expert_parallel=True,
          additional_config={"sharding": sharding} if sharding else {}, max_model_len=4096, max_num_seqs=16,
          max_num_batched_tokens=2048, block_size=256, enable_chunked_prefill=True, enable_prefix_caching=False,
          gpu_memory_utilization=0.5, language_model_only=True, limit_mm_per_prompt={"image": 0, "video": 0},
          disable_log_stats=True, kv_cache_dtype="bfloat16")
log("engine up")
sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)
lp = np.full((B, S), np.nan, np.float32); top1 = np.full((B, S), -1, np.int32)
for b, o in enumerate(outs):
  pl = o.prompt_logprobs  # list (len S) of {token_id: Logprob} or None for the first token
  for i, d in enumerate(pl):
    if d is None: continue
    tid = int(tokens[b, i]); lp[b, i] = d[tid].logprob if tid in d else np.nan
    top1[b, i] = max(d.items(), key=lambda kv: kv[1].logprob)[0]
np.savez(f"{OUT}/vllm35b_prompt_logprobs_dp{ATTN_DP}.npz", logp=lp, top1=top1)
log(f"saved; mean logp (positions>=1) = {np.nanmean(lp[:, 1:]):.3f}; top-1 acc vs actual = {np.mean(top1[:, 1:] == tokens[:, 1:]):.3f}")
