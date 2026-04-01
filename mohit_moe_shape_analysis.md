# MoE Shape Analysis — Qwen3-30B-A3B RL Training

## Setup

**Cluster:** `xfgu-v5p-64-pw`, v5p-32, `europe-west4-b`, project=`cloud-tpu-multipod-dev`

**Key config params from launch command:**
```
model_name=qwen3-30b-a3b
batch_size=32
train_micro_batch_size=16
rollout_micro_batch_size=32
rl.num_generations=4
max_target_length=288
max_prefill_predict_length=256
rollout_data_parallelism=4
rollout_tensor_parallelism=1
rollout_expert_parallelism=4
max_num_seqs=128
max_num_batched_tokens=544
kv_cache_buffer=512
scan_layers=True
enable_checkpointing=false
```

---

## Debug Prints Added

All prints use plain `print()` (not `jax.debug.print`) — safe for Pathways which doesn't support `RemoteXlaHostCallbackRegistry`.

### `src/maxtext/layers/moe.py`
- **Before `@functools.partial(jax.shard_map, ...)`** — fires at Python execution time per forward call:
  ```
  [moe] mesh.shape=...
  [moe] ep_name=... num_expert_parallelism=...
  [moe] is_batch_sharded_by_expert=... batch_logical_axis=...
  [moe] weight_gather=... w0_pspec=...
  [moe] input_partition_pspec=...
  ```
- **Inside `_permute_and_compute`** — fires at shard_map trace time:
  ```
  [permute] inputs_2d shape=...
  [permute] weights (top_k_weights) shape=...
  [permute] selected_experts shape=...
  [permute] sorted_inputs shape=...
  [permute] group_size shape=...
  ```
- **Inside `wrapper` (shard_map body)** — fires at shard_map trace time:
  ```
  [wrapper] shard x shape=...
  [wrapper] shard logits shape=...
  [wrapper] shard w0 shape=...
  [wrapper] shard wo shape=...
  [wrapper] post-permute x shape=...
  [wrapper] group_sizes shape=...
  [wrapper] num_expert_parallelism=...
  [wrapper] wi_gather_axes=... wo_gather_axes=...
  [wrapper] layer_w0/w1/intermediate_layer/intermediate_output shape=...
  [wrapper] output (post-unpermute) shape=...
  ```

### `src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py`
- **In `__call__` before `self.model(...)`** — fires at trace time:
  ```
  [adapter] model_mode=... input_ids.shape=... input_positions.shape=...
  ```

---

## Key Finding: When Prints Fire

- `[moe]` prints (outside shard_map): fire at **Python execution time per forward call**
- `[wrapper]` / `[permute]` prints (inside shard_map): fire at **shard_map trace time**
- `[adapter]` prints: fire at **JAX trace time** (model is inside JIT)

Confirmed by: `[wrapper]` count = `[adapter]` count × 48 MoE layers (perfectly consistent throughout logs).

Each `[adapter]` print = 1 new shape bucket compilation.

**Important:** `[wrapper]` traces only fire for the **inference mesh** (EP=4). Training `[wrapper]` traces are not visible on the Pathways head pod — training shard_map runs are dispatched to Pathways workers. Training is only observable via `[MoE.__call__]` Python-level prints.

---

## Training Shapes (Actor Mesh)

**Mesh:** `{fsdp: 16, data: 1, expert: 1, tensor: 1, ...}` (16 devices, all FSDP)

**Global training shape: `(64, 288, 2048)`** — fires immediately after rollout completes.

| Tensor | Shape | Notes |
|---|---|---|
| `inputs` (global) | `(64, 288, 2048)` | global batch=64, seq=288, embed=2048 |
| `gate_logits` (global) | `(64, 288, 128)` | 128 experts total |
| `w0_kernel` (global) | `(128, 2048, 768)` | all 128 experts (EP=1, FSDP all-gathers) |
| `w1_kernel` (global) | `(128, 2048, 768)` | gate proj |
| `wo_kernel` (global) | `(128, 768, 2048)` | down proj |
| per-shard `x` (inside shard_map) | `(4, 288, 2048)` | 64 ÷ 16 FSDP shards = 4/shard |
| per-shard `w0` (inside shard_map) | `(128, 2048, 768)` | all 128 experts local (EP=1, no expert sharding) |
| `inputs_2d` | `(1152, 2048)` | 4×288 tokens flattened |
| `sorted_inputs` | `(9216, 2048)` | 1152×8 (top-8 routing) |
| `group_sizes` | `(128,)` | token count per expert, local = global (EP=1) |

**Sharding config:**
- `input_partition_pspec = PartitionSpec('fsdp', None, None)` → batch sharded over FSDP
- `w0_pspec = PartitionSpec(None, None, None)` → weights NOT sharded in shard_map (FSDP all-gathers weights before shard_map)
- `batch_logical_axis = activation_batch_moe`
- `is_batch_sharded_by_expert = True` (but EP=1, so resolves to FSDP axis)
- EP=1 → `wi_gather_axes=[] wo_gather_axes=[]` (no extra gather)

**Global batch = 64 derivation (open question):**
- `batch_size=32` × `rl.num_generations=4` = 128 total RL sequences per step
- 128 ÷ 2 gradient accumulation steps = **64 per microbatch forward pass**
- This implies `train_micro_batch_size=16` controls something else (per-device chunk size?), or grad_accum=2 is inferred elsewhere
- Two thread IDs appear for training prints → likely forward + backward passes

**Two threads observed in training prints:**
- Thread `132942487615232` (2 layer calls)
- Thread `132947694655296` (2 layer calls)
- Only 4 of the expected 48 `[MoE.__call__]` prints visible — Pathways head logs a subset; remaining layer calls run on workers

---

## Inference Shapes (vLLM Rollout Mesh)

**Mesh:** `{data: 4, expert: 4, attn_dp: 1, attn_dp_expert: 1, model: 1}`

**Always `model_mode=autoregressive`** — MaxText never sees prefill mode. vLLM handles chunked prefill internally and presents all calls to MaxText as decode.

**Sharding config:**
- `input_partition_pspec = PartitionSpec(None, None, None)` → batch REPLICATED on all shards
- `w0_pspec = PartitionSpec('expert', None, None)` → expert dim sharded (32 experts/shard = 128/4)
- `batch_logical_axis = activation_batch_no_exp_moe`
- `is_batch_sharded_by_expert = False`
- EP=4 → `wi_gather_axes=[] wo_gather_axes=[]`

**Why batch is replicated (not sharded):**
From `vllm.yml` line 35: `['activation_batch_no_exp_moe', []]` — maps to no mesh axes = replicated. This is intentional for EP-style MoE: all tokens go to every expert shard, each shard runs GMM for its 32 local experts, then all-to-all combines results.

### Decode Buckets

| `[adapter]` | `[wrapper]` shard x | `sorted_inputs` | `intermediate_output` | Notes |
|---|---|---|---|---|
| `(1, 16)` compiled 1× | 47 calls | `(128, 2048)` = 16×8 | `(128, 2048)` | 16 active seqs |
| `(1, 32)` compiled 1× | 48 calls | `(256, 2048)` = 32×8 | `(256, 2048)` | 32 active seqs |
| `(1, 64)` compiled 1× | 48 calls | `(512, 2048)` = 64×8 | `(512, 2048)` | 64 active seqs |
| no adapter print | 12 calls | `(1024, 2048)` = 128×8 | `(1024, 2048)` | 128=max_num_seqs, cached from prior iter |

Formula: `sorted_inputs[0] = seq_len × top_k_8`. Per EP shard: each shard processes all tokens but only for its 32 experts.

### Prefill Buckets

| Shape | Notes |
|---|---|
| `(1, 4096, 2048)` | Fires at start of rollout iter 2 (cached — no `[adapter]` print). Larger than `max_num_batched_tokens=544` — vLLM compiles bucket from `max_position_embeddings`, not config limits. |

**The `(1, 4096)` open question:** vLLM compiles its default power-of-2 bucket set regardless of `max_num_batched_tokens`. Need to check vLLM's `max_position_embeddings` or internal max context length setting.

### EP interaction: `group_sizes` shape change

| | Training | Inference |
|---|---|---|
| `group_sizes` | `(128,)` | `(32,)` |
| `w0` (per shard) | `(128, 2048, 768)` | `(32, 2048, 768)` |

In inference, each EP shard only tracks token counts for its 32 local experts.

---

## KV Cache (Observed in Logs)

Revealed by `kv_cache_manager.py` logs at rollout start (iter 2):
```
shape=(num_blocks, (64, 4, 2, 128))
sharding=PartitionSpec(('data', 'attn_dp', 'attn_dp_expert'), None, ('model', 'expert'))
num_blocks=151776 per device (all 16 inference devices identical)
```

| Metric | Value |
|---|---|
| Blocks per device | 151,776 |
| Block shape | `(64, 4, 2, 128)` → `(tokens_per_block, heads, kv, head_dim)` |
| HBM before KV alloc | 15.67 GB / 95.0 GB |
| HBM after KV alloc | 71.25 GB / 95.0 GB |
| KV cache HBM consumption | **~55.6 GB per device** |

---

## Reshard: Training ↔ Inference

**Cost: 0.79s** — measured from `reshard.py:483` log.  
Happens between end of training step and start of each new rollout.

---

## Training vs Inference Comparison

| | Training | Inference |
|---|---|---|
| Mesh | fsdp=16, EP=1 | data=4, EP=4 |
| Global batch shape | `(64, 288, 2048)` | varies by decode bucket |
| Per-shard batch | `(4, 288, 2048)` | `(1, seq, 2048)` replicated |
| Expert sharding | None (weights replicated via FSDP all-gather) | `w0` sharded on `expert` axis |
| Local experts | 128 | 32 |
| `group_sizes` dim | 128 | 32 |
| Top-k | 8 | 8 |
| `[wrapper]` visible on head | No (Pathways workers) | Yes |

---

## Recompilation Pattern

- Prints confirmed to fire at **trace time** (not runtime) for inference
- Each RL iteration triggers recompilation of vLLM shape buckets — except buckets already compiled in prior iterations reuse the cached trace (no `[adapter]` print, just `[MoE.__call__]`)
- Root cause of per-iter recompile: weight resharding between training mesh and inference mesh invalidates JAX's in-memory compilation cache
- `jax_cache_dir` defaults to `~/jax_cache` — persistent across iterations within a single run

---

## Open Questions

1. **`(1, 4096)` prefill bucket:** Why does it appear given `max_num_batched_tokens=544`? Check vLLM's `max_position_embeddings` or internal max context length setting.
2. **Training global batch = 64:** Likely `batch_size=32 × rl.num_generations=4 / 2_grad_accum = 64`. Confirm by logging batch config at training step start, or check what controls grad_accum.
3. **Training `[wrapper]` not visible on head pod:** Pathways dispatches training shard_map to workers — to see per-shard training shapes, look at a Pathways worker pod instead.
4. **Two threads in training prints (`132942487615232`, `132947694655296`):** Forward + backward passes, or 2 concurrent microbatches?
