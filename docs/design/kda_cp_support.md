# Design Doc: KDA CP (Context Parallelism) Support

## Summary

This PR integrates KDA (Kimi Delta Attention) into MaxText with tokamax backend and CP (context parallelism) support. It adds the `KimiDeltaAttention` layer, `ShortConvolution`, QKV/beta/gate projections, and CP-aware causal convolution boundary handling. The `CPContext` mechanism passes context information to the `chunk_kda` kernel for coordinated recurrent state across CP ranks.

## Design

### CP Data Flow Overview

```
No CP:
  [B, T, E] → QKV proj → ShortConv → SiLU → L2Norm → chunk_kda → output

CP (cp_size > 1):
  [B, T/cp, E] → QKV proj → SHARD_MAP(ShortConv w/ halo)  ← independent conv shard_map
                           → SiLU + L2Norm
                           → CPContext(mesh, "context")      ← constructed outside shard_map
                           → _inject_context_on_T + _wsc     ← partition spec fixup
                           → SHARD_MAP(chunk_kda)            ← cp_context passed in
                           → [B, T/cp, E]
```

Key difference from MLA CP: MLA relies on splash attention kernel internally doing implicit all_gather K/V → local attention; KDA does not rely on all_gather. Instead, `CPContext` lets the kernel coordinate recurrent state across ranks during forward/backward.

### Plan 1: `halo_exchange_for_conv` (`utils/cp_utils.py`, new file)

ShortConvolution is a causal 1D depthwise convolution. Under CP sharding, each rank lacks the preceding `kernel_size-1` historical tokens at its left boundary.

```
rank 0: [t0  t1  t2  t3]    pad: [0  0  t0  t1  t2  t3]   ← zeros (sequence start)
rank 1: [t4  t5  t6  t7]    pad: [t2 t3 t4  t5  t6  t7]   ← pull t2, t3 from rank 0
```

**Algorithm**:
1. `jnp.pad(x, (halo_size, 0))` — left zero-pad
2. Outside CP scope or cp_size==1 → return padded directly (degenerate causal padding)
3. Inside CP scope: `ppermute` forward ring — rank i sends its last `halo_size` tokens to rank i+1, rank 0's halo is set to zero
4. `return jnp.concatenate([halo, x], axis=seq_axis)`

`ppermute` is a collective op and must be called inside a scope that exposes the `"context"` axis. See Plan 2.

### Plan 2: ShortConvolution CP Wrapper (`layers/attention_kda.py`)

`ShortConvolution.__call__` internally calls `halo_exchange_for_conv`, which requires the `"context"` axis scope. Inside `KimiDeltaAttention.__call__`, when CP is enabled, wrap the q/k/v conv calls in an independent `jax.shard_map`.

Change location: the conv call segment after QKV projection in `KimiDeltaAttention.__call__` (see dev branch implementation `attention_kda.py:407-429`).

Key design decisions:
- **conv shard_map and chunk_kda shard_map are independent**: two separate `jax.shard_map` invocations, freeing conv's ppermute buffer in between
- `check_vma=False`: FlashAttention custom rules may falsely report VMA errors
- Zero-overhead fallback when no CP: follows the original path exactly

### Plan 3: chunk_kda CPContext + Partition Spec (`attention_kda.py`)

#### 3a. CPContext Construction (outside shard_map)

```python
try:
    from cp_utils import CPContext
except ImportError:
    CPContext = None

cp_ctx = CPContext(mesh=self.mesh, axis_name="context")
```

`CPContext` is a frozen dataclass. `mesh` and `axis_name` are set at construction time; chain metadata fields are populated internally by `chunk_kda`.

#### 3b. Partition Spec Injection

`nnx.logical_to_mesh_axes` may map the T axis to `None` due to Flax rule priority + size-1 axis stripping, but shard_map requires the T axis to have `"context"` sharding:

```python
def _inject_context_on_T(pspec, t_axis=1):
    spec = list(pspec)
    if spec[t_axis] is None:
        spec[t_axis] = "context"
    return jax.sharding.PartitionSpec(*spec)
```

Applied to `qkv_pspec`, `beta_pspec`, `seg_pspec` when CP is enabled, followed by `with_sharding_constraint` to ensure tensor physical layout matches.

#### 3c. chunk_kda shard_map

Under CP, pass through `cp_context=cp_ctx` and `segment_ids` to the `chunk_kda` kernel.

segment_ids handling:
- **varlen**: pass through as-is
- **non-varlen + CP**: construct dummy `jnp.ones(q.shape[:2], dtype=jnp.int32)` (used internally by the kernel to derive per-rank cu_seqlens)

### Plan 4: CP and load_balance Mutual Exclusion

The Delta Rule's recurrent state `S_t = f(S_{t-1}, k_t, v_t, beta_t)` depends on strict token ordering. load_balance's DUAL_CHUNK_SWAP reorder scrambles token order, breaking the sequential dependency.

Runtime check (added at the `__call__` entry of `attention_kda.py`):

```python
if (getattr(cfg, "context_parallel_size", 1) > 1
        and getattr(cfg, "context_parallel_load_balance", False)):
    raise ValueError(
        "KDA CP does not support context_parallel_load_balance. "
        "Recurrent state S depends on exact token order; DUAL_CHUNK_SWAP "
        "reorder breaks the sequential dependency. Set "
        "context_parallel_load_balance=false when using KDA with CP."
    )
```

## segment_ids Data Flow

```
batch["inputs_segmentation"]   ← [B, T], seg=0 = padding
    │
    ▼
KimiDeltaAttention.__call__(decoder_segment_ids)
    │
    ├── chunk_size padding: pad to chunk size (64) multiple
    │
    ├── ShortConvolution: halo_exchange_for_conv(segment_ids)
    │     cross-segment boundary masking inside conv
    │
    ├── _inject_context_on_T + _wsc: inject "context" sharding
    │
    └── shard_map(chunk_kda):
            - real seg → pass chunk_kda(segment_ids=seg)
            - no seg + CP → pass dummy jnp.ones
```

## Files Changed

| File | Change | Lines |
|------|--------|:----:|
| `layers/attention_kda.py` | **New**: `KimiDeltaAttention`, `ShortConvolution`, CP support | ~586 |
| `kernels/kda/__init__.py` | **New**: `chunk_kda()` entry point | ~84 |
| `kernels/kda/tokamax.py` | **New**: tokamax backend adapter | ~99 |
| `utils/cp_utils.py` | **New**: `halo_exchange_for_conv` | ~66 |
| `configs/types.py` | **Modified**: `KdaAttention` config class | +48 |
| `tests/unit/kda_attention_test.py` | **New**: KDA layer + conv halo + CP equivalence tests | ~914 |
| `docs/design/kda_cp_support.md` | **New**: design doc | — |
| **Total** | | **~1979** |

## Key Constraints

1. **CPContext availability**: Assert with a clear error message when CPContext is unavailable; do not silently fall back.

2. **ShortConvolution halo shard_map is required**: Under CP, conv needs to read historical tokens across ranks. Without shard_map → each rank independently left-zero-pads → causal sequence is split into independent segments → **correctness bug**. Without CP, falls back to `jnp.pad`, zero overhead.

3. **conv and chunk_kda are two independent shard_maps**: Non-nested. conv only needs `ppermute`; chunk_kda needs `CPContext`. Separate shard_maps give independent XLA boundaries with resource release in between.

4. **KDA does not use the `apply_attention` dispatcher**: KDA has its own QKV projection + SiLU + L2Norm + beta/gate projections and does not share the interface with `AttentionOp`.

5. **CP + load_balance are mutually exclusive**: Recurrent state sequential dependency is irreversible. Runtime `ValueError`.

## Backward Compatibility

- `halo_exchange_for_conv`: degrades to `jnp.pad` when no CP, zero overhead
- ShortConv shard_map: only activated when `context_parallel_size > 1`
- CPContext import: `try/except`, assert with clear error if unavailable
- segment_ids dummy: auto-construct `jnp.ones` when no varlen + CP

## Test Plan

| Test | Coverage |
|------|----------|
| `test_short_conv_no_cp` | halo degrades to causal pad without CP |
| `test_short_conv_cp_halo` | conv under CP>1 equals single-rank reference |
| `test_kda_cp_equivalence` | CP multi-rank forward equals single-rank |
| `test_kda_cp_rejects_load_balance` | CP+load_balance raises ValueError |