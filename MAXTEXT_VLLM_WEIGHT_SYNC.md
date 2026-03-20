# MaxText → vLLM Weight Sync via Custom Converter

## Problem

The Tunix RL training loop uses `VllmSampler.update_params` to sync actor weights to the
vLLM rollout engine after each training step. Internally it calls
`utils.transfer_state_with_mappings`, which applies per-key transforms driven by
`MappingConfig`. This path cannot handle the MoE gate+up fusion (`w13_weight`), which
requires two source tensors (`wi_0` + `wi_1`) simultaneously.

`bench_weight_sync.py` already contains a battle-tested `MaxTextToVLLMConverter` that
handles all required transforms correctly:

- QKV fusion with GQA interleaving (attention)
- MoE expert gate+up fusion (`w13_weight`, chunk-interleaved for TP)
- MoE gate / down transpose
- Layer-norm and LM-head transposes

## Solution

Replace the `transfer_state_with_mappings` path with a direct call to
`MaxTextToVLLMConverter.convert()` by subclassing `VllmSampler` and `VllmRollout`.

---

## Files Changed

### 1. NEW — `maxtext/src/maxtext/integration/tunix/maxtext_vllm_rollout.py`

Contains two new classes:

#### `MaxTextVllmSampler(VllmSampler)`

- Accepts an optional `converter` argument.
- Overrides `update_params`: runs `converter.convert(updated_weights)` then does a
  direct `jax.device_put` into the vLLM model-runner state dict, matching target
  shardings.
- Falls back to the base-class `transfer_state_with_mappings` path when no converter
  is supplied (safe drop-in replacement).

#### `MaxTextVllmRollout(VllmRollout)`

- Accepts a `maxtext_config` argument (pyconfig object) in addition to the standard
  `VllmRollout` arguments.
- Constructs `MaxTextToVLLMConverter(config=maxtext_config, mesh=mesh)` at init time.
- Creates `MaxTextVllmSampler` with the converter instead of the stock `VllmSampler`.
- `cache_config_or_size` is optional; when omitted (as `RLCluster` does for custom
  engines), it falls back to `rollout_config.kv_cache_size`.
- Param name is `rollout_actor` (not `model`) to match the kwarg that `RLCluster`
  passes to custom rollout engines.
- Initial `load_checkpoint` at the end of `__init__` runs through the converter so
  vLLM starts with real weights before any training step.

### 2. MODIFIED — `maxtext/src/maxtext/trainers/post_train/rl/train_rl.py`

Two changes:

**Added imports** (near existing tunix imports):
```python
import functools
from maxtext.integration.tunix.maxtext_vllm_rollout import MaxTextVllmRollout
```

**Replaced `rollout_engine="vllm"`** in `ClusterConfig`:
```python
# Before
rollout_engine="vllm",

# After
rollout_engine=functools.partial(MaxTextVllmRollout, maxtext_config=trainer_config),
```

`RLCluster` supports a custom rollout class/partial as `rollout_engine`. It calls it as:
```python
rollout_engine(rollout_actor=..., tokenizer=..., mesh=..., rollout_config=...)
```
The `partial` pre-fills `maxtext_config=trainer_config` so the signature matches.

---

## Call Chain After This Change

```
RLCluster.sync_weights()
  └─ rollout.update_params(nnx.state(actor_model))
       └─ MaxTextVllmRollout.update_params()   [inherited from VllmRollout]
            └─ MaxTextVllmSampler.update_params()
                 ├─ MaxTextToVLLMConverter.convert(updated_weights)
                 │     ├─ _convert_global()
                 │     ├─ _convert_attn()     [QKV fusion, o_proj, norms]
                 │     └─ _convert_moe()      [gate, w2, w13 fusion]
                 └─ jax.device_put(weight, target_sharding)  for each key
```

---

## Fixes Applied During Session

| Issue | Fix |
|---|---|
| `cannot import name 'mappings' from 'tunix.rl'` | Changed import to `from tunix.generate import mappings` |
| `RLCluster` passes `rollout_actor` not `model` | Renamed param to `rollout_actor` |
| `RLCluster` doesn't pass `cache_config_or_size` to custom engine | Made it optional, falling back to `rollout_config.kv_cache_size` |
