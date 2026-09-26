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

The difference from MLA CP is what the collective carries, not whether one happens at all. MLA all-gathers K/V for the sharded sequence, so its payload grows with sequence length. KDA all-gathers a fixed-size summary of the recurrent state instead. The shapes below are each rank's local shape, and `jax.lax.all_gather` adds a leading `cp_size` axis to the result:

| Path           | What is gathered                                                                                                                                                    | Shapes                            | Source                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------- |
| Forward        | `S_ext`, the affine contribution of each rank's last segment, and `M`, its transition matrix. `_merge_initial_state` composes them into each rank's incoming state. | `[H, B, K, V]` and `[H, B, K, K]` | `pallas_mosaic_tpu_fwd_kernel.py:411-414`   |
| Backward       | `dS_ext` and `dM`, packed along the last axis into one tensor so a single gather covers both                                                                        | `[B, H, K, V + K]`                | `pallas_mosaic_tpu_bwd_kernel.py:1752-1754` |
| Chain metadata | each rank's first and last segment id, used to derive `cu_seqlens`, `pre_num_ranks` and the related per-rank fields                                                 | `2 x cp_size` int32               | `cp_utils.py:260-261`                       |

None of the three carries T, since `chunk_gated_delta_rule_fwd_h_pre_process` returns `[H, B, K, V]` and `[H, B, K, K]` whatever `T_local` is. The CP collective cost is therefore `O(cp_size * B * H * K * (V + K))` per step rather than something proportional to sequence length. That is what `ContextParallelMetadata` coordinates: the merge of recurrent state across ranks in forward and backward, with the sequence itself staying sharded.

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
- **both pass `check_vma=False`**, see the `check_vma` section below
- Zero-overhead fallback when no CP: follows the original path exactly

#### Why `check_vma=False`

`check_vma=True` cannot be used while KDA runs through tokamax. Measured on 4xTPU v6e over the CP selection, `pytest tests/unit/kda_attention_test.py -k "Cp or cp or short_conv or halo"`, which collects 36 tests and skips 24 of them as `cpu_only`. With `check_vma=True` on the kernel-side shard_map, 9 passed and 3 failed. Enabling it on the conv side as well changes nothing: the same 9 pass and the same 3 fail. So the conv-side shard_map tolerates it and the failure is entirely kernel side, for two reasons that are both outside this repo:

1. `tokamax/_src/ops/experimental/kda/cp_utils.py:297`. The `fori_loop` inside `_derive_cp_metadata_from_segment_ids` enters with a replicated carry (`bool[]`, `int32[]`) and returns one that varies on the CP axis (`bool[]{V:context}`, `int32[]{V:context}`). JAX's VMA scan check rejects the type mismatch, and its own error message suggests `jax.lax.pcast(..., ('context',), to='varying')` on the initial carry.
2. `jax/_src/pallas/core.py:1888`. With `check_vma=True` on a shard_map, every `jax.ShapeDtypeStruct` must set `manual_axis_type`, which tokamax's KDA Pallas launcher does not.

The three failures are `test_kda_cp_full_layer_dummy_segments`, `test_kda_cp_full_layer_packed_segments` and `test_kda_no_cp_without_load_balance_ok`. The last one runs at `cp_size=1`, which shows that reason 2 applies to any KDA forward pass and not only under CP.

Until both are fixed upstream, `False` is what every other attention shard_map in MaxText uses. `layers/attention_op.py` lines 1778, 1843 and 2298 and `kernels/tokamax_splash_attention/splash_attention_kernel.py:2155` all hardcode it, and `config.check_vma` (default `False`, `configs/base.yml:719`, documented as covering "EP / FSDP ICI parallelisms") is consumed only by `layers/moe.py:2625`. It is an MoE knob today rather than an attention one, so adopting it across the attention path is a separate cleanup that needs the two tokamax fixes first.

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

MaxText's own `logical_to_mesh_axes` (imported from `maxtext.utils.sharding` at `attention_kda.py:57`) already resolves the T axis correctly when the CP axis is named `"context"`. Measured on 4xTPU v6e using the config's own `logical_axis_rules`:

| Config                                                  | Resolved pspec for `(activation_batch, activation_norm_length, None)` | Effect of `_inject_cp_axis_on_T` |
| ------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------- |
| `ici_context_parallelism=2`                             | `P('fsdp', 'context', None)`                                          | none, already correct            |
| `ici_context_parallelism=4`                             | `P(None, 'context', None)`                                            | none, already correct            |
| `context_sharding='expert'`, `ici_expert_parallelism=2` | `P(('fsdp', 'expert'), None, None)`                                   | overwrites T with `'expert'`     |

The third row is why the injection exists. `activation_norm_length` maps to `["tensor_sequence", "context", "context_usp_ulysses"]` (`configs/types.py:1383`), and that list has no `expert` entry, so expert-as-context resolves T to `None`. Without the overwrite the shard_map would run replicated over the very axis its collectives use.

```python
def _inject_cp_axis_on_T(pspec, t_axis=1):
    spec = list(pspec)
    spec[t_axis] = cp_axis_name
    return jax.sharding.PartitionSpec(*spec)
```

Applied to `qkv_pspec`, `beta_pspec` and `seg_pspec` when CP is enabled, followed by `with_sharding_constraint` to ensure tensor physical layout matches. `cp_axis_name` is `cfg.context_sharding`, which defaults to `"context"` and is `"expert"` for expert-as-context. Overwriting on every strategy rather than only when T resolves to `None` keeps both cases on one code path, and on `"context"` it is a no-op by construction, as the table above shows.

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
