"""Production native engine (attn_dp=4 tp=2 EP-8, bf16): RL-style rollouts from the 8 real prompts; record the sampler's
decode-path logprob of each sampled token, the sampled ids, prompt logprobs, and per-token routed experts."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, time, numpy as np
from vllm import LLM, SamplingParams, TokensPrompt
OUT = "/mnt/disks/persist/pr4925_repro"; MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B"); ATTN_DP = 4; GEN = int(os.environ.get("GEN", "256"))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
tokens = np.load(f"{OUT}/real35b_L3.npz")["tokens"]; B, S = tokens.shape
llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=8, enable_expert_parallel=True,
          additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": ATTN_DP}}},
          max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=2048, block_size=256, enable_chunked_prefill=True,
          enable_prefix_caching=False, gpu_memory_utilization=0.5, language_model_only=True,
          limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=True, kv_cache_dtype="bfloat16",
          enable_return_routed_experts=True, seed=0)
log("engine up")
sp = SamplingParams(max_tokens=GEN, min_tokens=GEN, temperature=1.0, top_p=1.0, top_k=-1, logprobs=1, prompt_logprobs=1, ignore_eos=True)
outs = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in row]) for row in tokens], sp)
gen_ids = np.full((B, GEN), -1, np.int32); gen_lp = np.full((B, GEN), np.nan, np.float32); plp = np.full((B, S), np.nan, np.float32)
routed = [None] * B
for b, o in enumerate(outs):
  c = o.outputs[0]; ids = list(c.token_ids); n = min(len(ids), GEN); gen_ids[b, :n] = ids[:n]
  for i in range(n):
    d = c.logprobs[i]; gen_lp[b, i] = d[ids[i]].logprob if ids[i] in d else np.nan
  for i, d in enumerate(o.prompt_logprobs):
    if d is not None and int(tokens[b, i]) in d: plp[b, i] = d[int(tokens[b, i])].logprob
  routed[b] = c.routed_experts
  log(f"prompt {b}: {n} tokens, routed_experts shape={None if routed[b] is None else routed[b].shape}, first text: {llm.get_tokenizer().decode(ids[:12])!r}")
tag = "fp8" if "FP8" in MODEL else "bf16"
np.savez(f"{OUT}/rollout35b_{tag}.npz", gen_ids=gen_ids, gen_logp=gen_lp, prompt_logp=plp,
         routed=np.stack([r for r in routed]) if all(r is not None and r.shape == routed[0].shape for r in routed) else np.array([], np.int32),
         routed_shapes=np.array([r.shape if r is not None else (0, 0, 0) for r in routed]))
log(f"saved rollout ({tag}); mean gen logp={np.nanmean(gen_lp):.3f}")
