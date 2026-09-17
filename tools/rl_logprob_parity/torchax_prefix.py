"""Native tpu-inference (torchax) Qwen3.5-35B-A3B: run layers 0..N-1 from real tokens through the production path,
record hidden states after 4 and 8 layers, logit-lens logprobs (final norm + lm_head) at both depths, per-layer MoE
expert indices (for router replay into the trainer), and layer 3 on the MaxText h_in with its routing captured."""
import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
import tpu_inference  # noqa: F401
import os, sys, time, types
from functools import partial
import jax, jax.numpy as jnp, numpy as np
import torch, torchax
from jax.sharding import NamedSharding, PartitionSpec as P
from torchax.interop import torch_view, jax_view
from vllm import LLM
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather
from tpu_inference.layers.common.attention_metadata import AttentionMetadata, GroupedAttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.vllm.vllm_model_wrapper_context import (get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)

OUT = "/mnt/disks/persist/pr4925_repro"; MODEL = "Qwen/Qwen3.5-35B-A3B"; NL = 8; L3 = 3
ATTN_DP = int(os.environ.get("ATTN_DP", "4")); FP32GATE = os.environ.get("FP32GATE") == "1"
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
f32 = lambda v: np.asarray(jax.device_get(v)).astype(np.float32)
z = np.load(f"{OUT}/real35b_L3.npz"); h_np, tokens = z["h_in"], z["tokens"]; B, S, D = h_np.shape; T = B * S

sharding = {"sharding_strategy": {"enable_dp_attention": True, "attn_dp_size": ATTN_DP}} if ATTN_DP > 1 else None
llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=8, enable_expert_parallel=True,
          additional_config={"sharding": sharding} if sharding else {}, max_model_len=4096, max_num_seqs=16,
          max_num_batched_tokens=2048, block_size=256, enable_chunked_prefill=True, enable_prefix_caching=False,
          gpu_memory_utilization=0.25, language_model_only=True, limit_mm_per_prompt={"image": 0, "video": 0},
          disable_log_stats=True, kv_cache_dtype="bfloat16", enable_return_routed_experts=True)
runner = llm.llm_engine.model_executor.driver_worker.model_runner
mesh = runner.mesh; state = runner.state; kv_caches = runner.kv_caches; vllm_config = llm.llm_engine.vllm_config
sc = vllm_config.sharding_config
_m = runner.model
for _ in range(4):
  if hasattr(_m, "vllm_model"): break
  _m = getattr(_m, "model")
torch_runner = _m; vllm_model = _m.vllm_model; lm = vllm_model.language_model
log(f"engine up; mesh {dict(zip(mesh.axis_names, mesh.devices.shape))}; tp={sc.tp_size} attn_dp={sc.attn_dp_size}; return_routed_experts={vllm_config.model_config.enable_return_routed_experts}")
layers = lm.model.layers; log("layer types:", [layers[i].layer_type[:4] for i in range(NL)], "| use_sequence_parallel=", lm.model.use_sequence_parallel, "| layer3 use_attn_reduce_scatter_for_moe=", layers[3].use_attn_reduce_scatter_for_moe)
if FP32GATE:
  for i in range(NL):
    g = getattr(layers[i].mlp.experts, "gate", None) or layers[i].mlp.gate
    g.forward = (lambda x, _g=g: (torch.nn.functional.linear(x.float(), _g.weight.float()), None))
  log("FP32GATE: patched router gates for layers 0..%d" % (NL - 1))

def _prefix(self, input_ids, positions, n):
  m = self.language_model.model; full = positions.shape[-1]
  hs = m.embed_tokens(input_ids); res = None; outs = {}
  if m.use_sequence_parallel:            # mirror Qwen3NextModel.forward
    hs = sequence_parallel_chunk(hs)
  for i in range(n):
    if os.environ.get("NO_FUSED_RESIDUAL") == "1" and res is not None:   # diagnostic: avoid the fused add-norm path
      hs = hs + res; res = None
    hs, res = m.layers[i](hidden_states=hs, residual=res, positions=positions)
    h = hs + res
    if os.environ.get("RESHARD") == "1":   # diagnostic: pin tokens to ATTN_DATA between layers like the runner's jit does
      h = torch_view(jax.lax.with_sharding_constraint(jax_view(h), tok_sh)); hs = h; res = None
    if m.use_sequence_parallel:
      h = tensor_model_parallel_all_gather(h, 0)[:full]
    outs[i] = h
  return outs
def _lens(self, h):  # logit lens: final norm + lm_head (+ vLLM logits processor)
  return self.language_model.compute_logits(self.language_model.model.norm(h))
def _layer(self, h, positions, idx):
  hs, res = self.language_model.model.layers[idx](hidden_states=h, residual=None, positions=positions); return hs + res
vllm_model.run_prefix = types.MethodType(_prefix, vllm_model); vllm_model.run_lens = types.MethodType(_lens, vllm_model); vllm_model.run_layer = types.MethodType(_layer, vllm_model)

# ---- attention metadata per DP rank; GDN layers additionally need padded_num_reqs + mamba_state_indices ----
DP = sc.attn_dp_size * sc.attn_dp_expert_size; block = vllm_config.cache_config.block_size; pps = (S + block - 1) // block
rpd = B // DP; local_pages = rpd * pps
md = AttentionMetadata(input_positions=jnp.asarray(np.tile(np.arange(S, dtype=np.int32), B)),
    block_tables=jnp.asarray(np.tile(np.arange(local_pages, dtype=np.int32), DP)),
    seq_lens=jnp.asarray(np.array([S] * B, np.int32)),
    query_start_loc=jnp.asarray(np.tile(np.arange(0, (rpd + 1) * S, S, dtype=np.int32), DP)),
    request_distribution=jnp.asarray(np.tile(np.array([0, 0, rpd], np.int32), DP)),
    mamba_state_indices=jnp.asarray(np.tile(np.arange(rpd, dtype=np.int32), DP)), padded_num_reqs=B)
groups = runner.kv_cache_config.kv_cache_groups
grouped = GroupedAttentionMetadata(groups=tuple(md for _ in groups), layer_names_per_group=tuple(tuple(g.layer_names) for g in groups))
name_to_idx = tuple(runner.layer_name_to_kvcache_index.items())
all_moe = list(vllm_config.compilation_config.static_all_moe_layers); moe_idx3 = next(i for i, n in enumerate(all_moe) if f".layers.{L3}." in n)
tok_sh = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)); vec_sh = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA))
log(f"DP={DP} block={block} local_pages={local_pages} rpd={rpd}")

def ctx(kv, attn_md):
  return (torchax.default_env(), set_vllm_model_wrapper_context(kv_caches=kv, mesh=mesh, layer_name_to_kvcache_index=dict(name_to_idx),
          vllm_config=vllm_config, shared_attn_metadata=None), set_forward_context(attn_metadata=attn_md, vllm_config=vllm_config, num_tokens=T))
def call(st, kwargs):
  return torch.func.functional_call(torch_runner, torch_view(st), kwargs=kwargs, tie_weights=False)
def lens_stats(logits, ids):  # logits [T,V] -> logp of actual next token, top-1 id, full logprobs (first 2 seqs) in fp32
  lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
  nxt = jnp.concatenate([ids[1:], ids[:1]])
  return lp[jnp.arange(T), nxt], jnp.argmax(lp, -1).astype(jnp.int32), lp[: 2 * S]

@jax.jit
def prefix_fn(st, kv, ids, positions, attn_md):
  c1, c2, c3 = ctx(kv, attn_md)
  with c1, c2, c3:
    get_forward_context().moe_layer_index = 0
    outs = jax_view(call(st, {"call_method": "run_prefix", "call_kwargs": {"input_ids": torch_view(ids), "positions": torch_view(positions), "n": NL}}))
    experts = list(get_vllm_model_wrapper_context().expert_indices_list)
    lens = {}
    for k in (4, 8):
      logits = jax_view(call(st, {"call_method": "run_lens", "call_kwargs": {"h": torch_view(outs[k - 1])}}))
      lens[k] = lens_stats(logits, ids)
  return {k: outs[k].astype(jnp.float32) for k in (2, 3, 7)}, lens, experts

@jax.jit
def layer3_fn(st, kv, h, positions, attn_md):
  c1, c2, c3 = ctx(kv, attn_md)
  with c1, c2, c3:
    get_forward_context().moe_layer_index = moe_idx3
    y = jax_view(call(st, {"call_method": "run_layer", "call_kwargs": {"h": torch_view(h), "positions": torch_view(positions), "idx": L3}}))
    experts = list(get_vllm_model_wrapper_context().expert_indices_list)
  return y.astype(jnp.float32), experts

with jax.set_mesh(mesh):
  ids = jax.device_put(jnp.asarray(tokens.reshape(-1).astype(np.int32)), vec_sh)
  pos = jax.device_put(jnp.asarray(np.tile(np.arange(S, dtype=np.int32), B)), vec_sh)
  outs, lens, experts = prefix_fn(state, kv_caches, ids, pos, grouped); jax.block_until_ready(outs[7])
  log(f"prefix 0..{NL-1} done; experts captured: {len(experts)} x {experts[0].shape} {experts[0].dtype}")
  h = jax.device_put(jnp.asarray(h_np).astype(jnp.bfloat16).reshape(T, D), tok_sh)
  y3, experts3 = layer3_fn(state, kv_caches, h, pos, grouped); jax.block_until_ready(y3)
  log(f"layer3 on MaxText h_in done; experts3: {len(experts3)} x {experts3[0].shape}")

tag = f"dp{sc.attn_dp_size}tp{sc.tp_size}_ep1" + ("_fp32gate" if FP32GATE else "") + ("_nofr" if os.environ.get("NO_FUSED_RESIDUAL") == "1" else "") + ("_reshard" if os.environ.get("RESHARD") == "1" else "")
res = {"h_after3": f32(outs[2]).reshape(B, S, D), "h_after4": f32(outs[3]).reshape(B, S, D), "h_after8": f32(outs[7]).reshape(B, S, D),
       "experts": np.stack([np.asarray(jax.device_get(e)).astype(np.int32) for e in experts], 0),
       "y3_on_maxtext_h": f32(y3).reshape(B, S, D), "experts3_on_maxtext_h": np.asarray(jax.device_get(experts3[0])).astype(np.int32)}
for k in (4, 8):
  lp_act, top1, lp_full = lens[k]
  res[f"lens{k}_logp_actual"] = f32(lp_act); res[f"lens{k}_top1"] = np.asarray(jax.device_get(top1)).astype(np.int32); res[f"lens{k}_logprobs_2seq"] = f32(lp_full)
np.savez(f"{OUT}/torchax_prefix_{tag}.npz", **res)
log(f"saved ({tag}); lens4 mean logp(actual)={res['lens4_logp_actual'].mean():.3f} top1-acc={np.mean(res['lens4_top1']==np.roll(tokens.reshape(-1),-1)):.3f}; "
    f"lens8 mean logp={res['lens8_logp_actual'].mean():.3f} top1-acc={np.mean(res['lens8_top1']==np.roll(tokens.reshape(-1),-1)):.3f}")
log("done")
