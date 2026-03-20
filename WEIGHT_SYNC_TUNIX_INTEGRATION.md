# Weight Transfer Integration: MaxText → vLLM via Tunix

## Background

`bench_weight_sync.py` implements `MaxTextToVLLMConverter` which takes a MaxText NNX
`model_state`, performs tensor transformations (QKV fusion, MoE expert fusion, transposes),
and pushes the converted weights directly into a live vLLM model state dict.

The goal is to integrate this logic properly into the Tunix RL training loop so that
after each training step, the updated actor weights are automatically synced to the
vLLM rollout engine.

---

## The Full Call Chain (Tunix RL Weight Sync)

```
RLCluster.sync_weights()                         # tunix/rl/rl_cluster.py:1000
  └─ self.rollout.update_params(nnx.state(actor_model), filter_types)
       └─ VllmRollout.update_params()            # tunix/rl/rollout/vllm_rollout.py:121
            └─ self._sampler.update_params(params, filter_types)
                 └─ VllmSampler.update_params()  # tunix/generate/vllm_sampler.py:177
                      ├─ utils.transfer_state_with_mappings(
                      │     src_state=updated_weights,
                      │     dst_state=self.transformer_state,   # vLLM model runner state
                      │     key_mappings=self.to_hf_key_mappings,
                      │     key_mapping_hook_fns=self.to_hf_hook_fns,
                      │     transpose_keys=self.to_hf_transpose_keys,
                      │  )
                      └─ (or) utils.transfer_state_directly(...)  # fallback for MaxText→MaxText
```

The mappings come from `MappingConfig.from_model(model)` at `VllmRollout.__init__` time,
which reads `.to_hf_mappings()`, `.to_hf_hook_fns()`, `.to_hf_transpose_keys()` off the
model passed in — i.e., **`TunixMaxTextAdapter`**.

---

## Key Files to Modify

### 1. Primary target: `QWEN3_VLLM_MAPPING`
**File:** `maxtext/src/maxtext/integration/tunix/weight_mapping/qwen3.py`

This is **the owned file where conversion logic belongs**. Currently:
- `to_hf_hook_fns()` returns `{}` — no QKV fusion, no MoE expert fusion
- `to_hf_transpose_keys()` returns `{}` — no transposes
- `to_hf_mapping()` only covers **non-MoE** Qwen3 (separate q/k/v proj, no `moe_block` entries)

### 2. `TunixMaxTextAdapter`
**File:** `maxtext/src/maxtext/integration/tunix/tunix_adapter.py`

Proxies `to_hf_mappings()`, `to_hf_hook_fns()`, `to_hf_transpose_keys()` to `VllmWeightMapping`,
which dispatches to the model-specific classes above. No changes needed unless adding new
dispatch logic.

### 3. `VllmSampler.update_params`
**File:** `venv1/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py:177`

The execution point. Calls `transfer_state_with_mappings` with the hook fns.
Read-only reference — part of the installed Tunix package.

---

## Mapping from `bench_weight_sync.py` → Tunix Hook System

| `bench_weight_sync.py` operation | Tunix mechanism | Notes |
|---|---|---|
| `_to_attn()` — fuse Q+K+V → `qkv_proj.weight` (GQA interleave) | `to_hf_hook_fns()` + key rename in `to_hf_mapping()` | Complex: needs all 3 tensors at once |
| Attention output projection transpose | `to_hf_transpose_keys()` | Simple axes spec |
| `_to_mlp_gate()` — transpose gate kernel `[d_model, l, e] → [l, e, d_model]` | `to_hf_hook_fns()` or `to_hf_transpose_keys()` | |
| `_to_mlp_expert_down()` — transpose down proj `[e, l, hidden, inter] → [l, e, inter, hidden]` | `to_hf_hook_fns()` | |
| `_to_mlp_expert_gate_up()` — fuse `wi_0` + `wi_1` → `w13_weight` (chunk interleave) | **Cannot be expressed as single-key hook** | See design gap below |
| Add `moe_block` key mappings entirely | `to_hf_mapping()` — currently **missing** | MoE Qwen3 variants only |

---

## Critical Design Gap: Multi-Key Fusion

`transfer_state_with_mappings` in `tunix/generate/utils.py` applies `hook_fns`
**per single source key, one at a time**. The `_to_mlp_expert_gate_up()` fusion requires
**both `wi_0` and `wi_1` simultaneously** to produce the fused `w13_weight`.

### Options

#### Option A: Stateful Hook (Accumulator Pattern)
Use a stateful callable for `wi_0`'s hook that caches the value, and `wi_1`'s hook that
reads the cache and emits the fused result. Fragile — depends on call ordering.

#### Option B: Custom `VllmRollout` Subclass ← **Recommended**
Subclass `VllmRollout`, override `update_params()`:

```python
class MaxTextQwen3VllmRollout(VllmRollout):
    def __init__(self, model, tokenizer, cache_config_or_size, mesh, rollout_config):
        super().__init__(...)
        self._converter = MaxTextToVLLMConverter(config=..., mesh=mesh)

    def update_params(self, params, filter_types=None):
        # Run your custom conversion first
        vllm_state = self._converter.convert(params)
        # Push converted state directly into vLLM model runner
        model_runner_state = self._sampler.transformer_state
        for key, weight in vllm_state.items():
            target_sharding = model_runner_state[key].sharding
            model_runner_state[key] = jax.device_put(np.asarray(weight), target_sharding)
```

Register this in `rl_cluster.py` / `RolloutConfig` as the rollout class.
This short-circuits the key-mapping path entirely and reuses your battle-tested converter.

#### Option C: Extend `transfer_state_with_mappings` to Support Multi-Key Hooks
Add a `multi_key_hook_fns` argument that receives a dict of `{src_key: fn}` where
the fn receives a batch of tensors. Requires modifying the installed Tunix package.

---

## Recommended Next Steps

1. **For non-MoE Qwen3**: Fill in `to_hf_hook_fns()` and `to_hf_transpose_keys()` in
   `weight_mapping/qwen3.py` with the attention projection transforms from `_to_attn()`.

2. **For MoE Qwen3 (30B-A3B, 235B-A22B)**: Add `moe_block` key entries to
   `to_hf_mapping()` and implement **Option B** for the gate+up fusion.

3. **Wire the custom rollout** into `train_rl.py` by passing it through
   `RolloutConfig` or `ClusterConfig` as the rollout class override.

---

## Reference: `bench_weight_sync.py` Tensor Shape Transforms

```python
# Attention: QKV fusion with GQA interleaving
# q: (d_model, l, num_q_heads, head_dim) -> per layer: (num_kv_heads*(hpg+2)*head_dim, d_model)
# Final: [gate_chunk0, up_chunk0, gate_chunk1, up_chunk1, ...] layout for TP

# MoE gate:     [d_model, l, num_experts] -> per layer: [num_experts, d_model]
# MoE down:     [num_experts, l, hidden, inter] -> per layer: [num_experts, inter, hidden]
# MoE gate+up:  wi_0 + wi_1 -> per layer: [num_experts, 2*d_inner, d_model]
#               chunk-interleaved for vLLM TP: [gate_c0, up_c0, gate_c1, up_c1, ...]
#               num_chunks = tensor_parallel_size (4 for Qwen3-30B)
```
