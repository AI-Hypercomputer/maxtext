# Design Doc: KDA CP (Context Parallelism) Support

## Summary

This PR integrates KDA (Kimi Delta Attention) into MaxText with tokamax backend and CP (context parallelism) support. It adds the `KimiDeltaAttention` layer, `ShortConvolution`, QKV/beta/gate projections, and CP-aware causal convolution boundary handling. The `ContextParallelMetadata` mechanism passes context information to the `chunk_kda` kernel for coordinated recurrent state across CP ranks.

## Design

### CP Data Flow Overview

```
No CP:
  [B, T, E] → QKV proj → ShortConv → SiLU → L2Norm → chunk_kda → output

CP (cp_size > 1):
  [B, T/cp, E] → QKV proj → SHARD_MAP(ShortConv w/ halo)  ← independent conv shard_map
                           → SiLU + L2Norm
                           → ContextParallelMetadata(mesh, cfg.context_sharding)  ← constructed outside shard_map
                           → _inject_cp_axis_on_T + _wsc     ← partition spec fixup
                           → SHARD_MAP(chunk_kda)            ← context_parallel_metadata passed in
                           → [B, T/cp, E]
```

Key difference from MLA CP: MLA relies on splash attention kernel internally doing implicit all_gather K/V → local attention; KDA does not rely on all_gather. Instead, `ContextParallelMetadata` lets the kernel coordinate recurrent state across ranks during forward/backward.

### Plan 1: `halo_exchange_for_conv` (in `layers/attention_kda.py`, KDA-specific)

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

`ppermute` is a collective op and must be called inside a scope that exposes the CP axis (the `cfg.context_sharding` mesh axis, default `"context"`). See Plan 2.

**Constraint**: the exchange only reads from the immediately preceding rank, so it requires `halo_size <= T_local` (i.e. `linear_conv_kernel_dim - 1` must not exceed the per-rank sequence length). Larger receptive fields would span multiple ranks and are not implemented; `halo_exchange_for_conv` raises a `ValueError` in that case.

### Plan 2: ShortConvolution CP Wrapper (`layers/attention_kda.py`)

`ShortConvolution.__call__` internally calls `halo_exchange_for_conv`, which requires the CP axis scope (`cfg.context_sharding`). Inside `KimiDeltaAttention.__call__`, when CP is enabled, wrap the q/k/v conv calls in an independent `jax.shard_map`.

Change location: the conv call segment after QKV projection in `KimiDeltaAttention.__call__`.

Key design decisions:

- **conv shard_map and chunk_kda shard_map are independent**: two separate `jax.shard_map` invocations, freeing conv's ppermute buffer in between
- `check_vma=False`: FlashAttention custom rules may falsely report VMA errors
- Zero-overhead fallback when no CP: follows the original path exactly

### Plan 3: chunk_kda ContextParallelMetadata + Partition Spec (`attention_kda.py`)

#### 3a. ContextParallelMetadata Construction (outside shard_map)

```python
try:
    from tokamax._src.ops.experimental.kda.cp_utils import (
        ContextParallelMetadata as TokamaxContextParallelMetadata,
    )
except ImportError:
    TokamaxContextParallelMetadata = None

cp_axis_name = cfg.context_sharding  # default "context"; "expert" for expert-as-context
if cp_size > 1:
    if TokamaxContextParallelMetadata is None:
        raise ImportError(...)  # refuse to run: CP would silently break state
    cp_ctx = TokamaxContextParallelMetadata(mesh=self.mesh, axis_name=cp_axis_name)
```

`ContextParallelMetadata` is a frozen dataclass. `mesh` and `axis_name` are set at construction time; chain metadata fields are populated internally by `chunk_kda`. The `axis_name` comes from `cfg.context_sharding`, so expert-as-context meshes bind the metadata to the `"expert"` axis.

#### 3b. Partition Spec Injection

`nnx.logical_to_mesh_axes` may map the T axis to `None` (or to a mesh axis that does not carry the sequence shard) due to Flax rule priority + size-1 axis stripping, but shard_map requires the T axis to carry the CP axis:

```python
def _inject_cp_axis_on_T(pspec, t_axis=1):
    spec = list(pspec)
    spec[t_axis] = cp_axis_name  # overwritten unconditionally
    return jax.sharding.PartitionSpec(*spec)
```

Applied to `qkv_pspec`, `beta_pspec`, `seg_pspec` when CP is enabled, followed by `with_sharding_constraint` to ensure tensor physical layout matches.

`cp_axis_name` is `cfg.context_sharding` (default `"context"`; may be `"expert"` for expert-as-context). The T axis is overwritten **unconditionally** rather than only when it maps to `None`: the `activation_norm_length` logical-axis rules do not cover every CP strategy (notably expert-as-context), so an unconditional overwrite guarantees the shard_map always sees the per-rank sequence shards on the axis the collectives (halo exchange, cross-rank state merge) actually use.

#### 3c. chunk_kda shard_map

Under CP, pass through `context_parallel_metadata=cp_ctx` and `segment_ids` to the `chunk_kda` kernel.

segment_ids handling:

- **varlen**: pass through as-is
- **non-varlen + CP**: construct dummy `jnp.ones(q.shape[:2], dtype=jnp.int32)` (used internally by the kernel to derive per-rank cu_seqlens)

### Plan 4: CP and load_balance Mutual Exclusion

The Delta Rule's recurrent state `S_t = f(S_{t-1}, k_t, v_t, beta_t)` depends on strict token ordering. load_balance's DUAL_CHUNK_SWAP reorder scrambles token order, breaking the sequential dependency.

Runtime check (added at the `__call__` entry of `attention_kda.py`):

```python
cp_size = self.mesh.shape.get(cfg.context_sharding, 1)
if cp_size > 1 and getattr(cfg, "context_parallel_load_balance", False):
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
    ├── T-padding: pad sequence to a multiple of the chunk alignment (64)
    │
    ├── ShortConvolution: halo_exchange_for_conv(segment_ids)
    │     cross-segment boundary masking inside conv
    │
    ├── _inject_cp_axis_on_T + _wsc: inject cfg.context_sharding axis onto the T axis
    │
    └── shard_map(chunk_kda):
            - real seg → pass chunk_kda(segment_ids=seg)
            - no seg + CP → pass dummy jnp.ones
```

## Files Changed

| File                                         | Change                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `layers/attention_kda.py`                    | **New**: `KimiDeltaAttention`, `ShortConvolution`, and the conv halo exchange for CP        |
| `kernels/kda/__init__.py`                    | **New**: `chunk_kda()` entry point                                                          |
| `kernels/kda/tokamax.py`                     | **New**: tokamax backend adapter (lazy import; version-aware error when the API is missing) |
| `layers/nnx_decoders.py`                     | **Modified**: `attention_type='kda'` dispatches to `KimiDeltaAttention`                     |
| `common/common_types.py`                     | **Modified**: `AttentionType.KDA`                                                           |
| `configs/types.py`                           | **Modified**: `KdaAttention` config class + validators                                      |
| `configs/base.yml`                           | **Modified**: `attention_type` supported list + the four KDA flags                          |
| `tests/unit/kda_attention_test.py`           | **New**: layer + conv halo + CP fwd/bwd + packed-seg CP + parity + e2e smoke test           |
| `tests/unit/kda_decoder_integration_test.py` | **New**: decoder dispatch, config guards, and a training run through the real decoder       |
| `docs/reference/kda_cp_support.md`           | **New**: this design doc                                                                    |
| `docs/reference.md`                          | **Modified**: toctree and card entry for this doc                                           |

Line counts are deliberately not tracked here: they rot on every rebase, and the PR diff is the
authoritative list.

## Key Constraints

1. **ContextParallelMetadata availability**: raise `kda_api_unavailable` (an `ImportError` naming the first tokamax release that ships the KDA API, the version installed, and the fix) when it is unavailable; never fall back silently. See "tokamax KDA API availability" below.

2. **ShortConvolution halo shard_map is required**: Under CP, conv needs to read historical tokens across ranks. Without shard_map → each rank independently left-zero-pads → causal sequence is split into independent segments → **correctness bug**. Without CP, falls back to `jnp.pad`, zero overhead.

3. **conv and chunk_kda are two independent shard_maps**: Non-nested. conv only needs `ppermute`; chunk_kda needs `ContextParallelMetadata`. Separate shard_maps give independent XLA boundaries with resource release in between.

4. **KDA does not use the `apply_attention` dispatcher**: KDA has its own QKV projection + SiLU + L2Norm + beta/gate projections and does not share the interface with `AttentionOp`.

5. **CP + load_balance are mutually exclusive**: Recurrent state sequential dependency is irreversible. Runtime `ValueError`.

## Backward Compatibility

- `halo_exchange_for_conv`: degrades to `jnp.pad` when no CP, zero overhead
- ShortConv shard_map: only activated when `cp_size > 1` (derived from the mesh's `context_sharding` axis)
- ContextParallelMetadata import: `try/except` keeps the module importable; using KDA then raises `kda_api_unavailable` with the required version
- segment_ids dummy: auto-construct `jnp.ones` when no varlen + CP

## tokamax KDA API availability

`tokamax._src.ops.experimental.kda` first shipped in **tokamax 0.0.14** (0.0.12 and 0.0.13 do not contain it, and the 0.1.0 upload was yanked). MaxText deliberately does not express that as a `tokamax>=0.0.14` requirement:

- the floor would live in `base_requirements/`, which feeds `generate_requirements.sh`; raising it re-pins the whole lock set, JAX included,
- the decoupled-mode environment pins tokamax to an older release on purpose (`generate_decoupled_requirements.py`: newer tokamax imports `xprof` at module scope), and `layers/nnx_decoders.py` imports `attention_kda` unconditionally — so a hard failure at import time would break every NNX model there, not just KDA.

The module-level import therefore stays soft and the failure is deferred to use time: `kda_api_unavailable` names the release required, the version installed, and the fix command. The installed version is diagnostic only — a source install of a KDA branch can report an older number while providing the API — so "predates it" is asserted only when the version string parses and compares below 0.0.14. Until the generated locks are raised, CI resolves the lowest allowed tokamax and skips the kernel-invoking tests.

## Test Plan

Only tests that invoke the Mosaic Pallas kernel or multi-device CP carry the `tpu_only` marker; pure config / pure-op / non-CP tests run in regular CPU CI as well.

| Test                                        | Coverage                                                                                                                                                                                                                                                                    |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_short_conv_no_cp`                     | halo degrades to causal pad without CP                                                                                                                                                                                                                                      |
| `test_short_conv_cp_halo`                   | conv under CP>1 equals single-rank reference; parametrized over segment layouts: uniform, boundary on the rank split, and a segment spanning both ranks (halo + segment-mask interaction)                                                                                   |
| `test_short_conv_cp_rejects_oversized_halo` | `halo_size > T_local` under CP raises a clear ValueError (multi-rank receptive field not implemented)                                                                                                                                                                       |
| `test_kda_cp_equivalence`                   | kernel-level CP multi-rank forward equals single-rank, parametrized CP=2 and CP=4                                                                                                                                                                                           |
| `test_kda_cp_backward`                      | CP gradients (dq/dk/dv/dg/dbeta) equal the non-CP reference                                                                                                                                                                                                                 |
| `test_kda_cp_full_layer_dummy_segments`     | full layer under CP with no user segment_ids: covers the internal dummy-segment synthesis path, forward equivalence and backward finiteness                                                                                                                                 |
| `test_kda_cp_full_layer_packed_segments`    | full layer under CP with multiple real packed segments — one spanning the rank boundary, one boundary exactly at the split; forward + input/weight gradients equal the non-CP reference                                                                                     |
| `test_full_layer_mosaic_vs_xla_parity`      | full layer with identical weights, Mosaic kernel vs tokamax XLA reference implementation: forward + gradients match                                                                                                                                                         |
| `test_kda_cp_rejects_load_balance`          | CP+load_balance raises ValueError                                                                                                                                                                                                                                           |
| `test_packed_segment_no_leak_within_row`    | packed segments inside one row are structurally isolated in both directions                                                                                                                                                                                                 |
| `test_l2_normalize_produces_unit_norm`      | `_l2_normalize` yields unit L2 norm and preserves direction                                                                                                                                                                                                                 |
| `TestKdaConfigGuards`                       | config-time guards: safe-gate/lower_bound range, `use_kda_lora=True` rejection, packing without `max_segments_per_seq`                                                                                                                                                      |
| `TestKdaKernelGuards`                       | guards that fire before any kernel dispatch: `initial_state` / `output_final_state` rejected by both `chunk_kda` and the adapter; the KDA-API-unavailable error names the required tokamax release and an actionable fix, keeps the caller's detail, and chains `__cause__` |
