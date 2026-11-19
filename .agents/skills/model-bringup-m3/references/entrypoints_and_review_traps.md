# `m3` Entrypoints, Registries, and PR Review Traps

Load this reference when wiring `m3/core/` or `m3/infra/` modules into MaxText
drivers, registering models in `src/maxtext/m3/models/__init__.py`, or reviewing
`m3` GitHub PRs.

## 1. Dual-Driver Transform Wiring (`model_creation_utils.py` vs. `loop.py`)

`m3` has **two** model-construction call paths that must stay synchronized:

1.  **Production & AOT Path (`src/maxtext/utils/model_creation_utils.py`)**:
    -   Invoked whenever `config.use_m3_model` is `True` by
        `MaxTextTrainingEngine` (RL/Tunix/SFT), `train.py` (pretrain), and
        `train_compile.py` (TPU AOT HBM/HLO verification).
    -   Every `m3/infra/` post-init transform (`scan`, `remat`, `quantization`,
        `named_scopes`, `offload`) must be applied here in deterministic order
        so AOT compilation and `MaxTextTrainingEngine` exercise the exact same
        transformed graph.
2.  **Synthetic Loop Path (`src/maxtext/m3/train/loop.py`)**:
    -   Standalone synthetic-token benchmark loop using `m3/configs/config.py`.
    -   **Recurring PR defect**: PRs frequently wire new `m3/infra/` transforms
        (`quantization.py`, `named_scopes.py`) only into `m3/train/loop.py` and
        omit `src/maxtext/utils/model_creation_utils.py`. Always check both call
        sites in every `m3/infra/` or `m3/core/` PR.

## 2. Reconciling `MaxTextTrainingEngine` vs. Classic `train.py` Contracts

| Check Aspect | `MaxTextTrainingEngine` Contract | Classic `train.py` / `train_compile.py` Contract |
| :--- | :--- | :--- |
| **Forward return value** | Custom loss adapter registered via `MODEL_ENGINE_CONFIG_REGISTRY` (`engine.with_loss_fn(..., has_aux=True)`) | Expects raw `logits` tensor unless `train.py` `loss_fn` explicitly handles the `m3` output container |
| **Auxiliary & MTP losses** | Folded into primary scalar loss inside the model's `engine_loss_fn` (carry `(hidden, aux_loss)` through `ScannedLayers`) | Classic `train.py` harvests legacy `nnx.Intermediate` keys via `nnx.pop(...)`; an `m3` model that neither sows nor hooks `train.py` silently trains with `0.0` aux/MTP loss |
| **Frozen parameters (router bias)** | Masked via `optax.multi_transform` inside `configure_{model}_engine` | Controlled by optimizer param-mask rules; verify non-trainable buffers either use frozen optimizer labels or non-`nnx.Param` `nnx.Variable` subclasses if supported by the engine |
| **Vocabulary tiling** | Optional depending on loss adapter | When `config.num_vocab_tiling > 1`, `vocab_tiling_nnx_loss` calls `model.logits_from_hidden_states_for_vocab_tiling(hidden_states, ...)` |
| **FLOPs accounting** | Optional | Mandatory: `maxtext_utils.py` dispatches to `m3.models.calculate_tflops_per_device` (`MODEL_FLOPS_REGISTRY`); missing entries crash `train_compile` immediately |
| **`ShardMode.EXPLICIT` & ZeRO-1** | Engine ZeRO-1 relies on `ShardMode.EXPLICIT`; verify `nnx.Embed` and custom `shard_map` blocks do not fail under explicit | Defaults to `ShardMode.AUTO` unless configured; test both single-axis and multi-axis (`TP > 1` and `EP > 1`) meshes |

## 3. `m3/core/` Kernel & Sharding Composition Rules

1.  **Static kernel/executor descriptors**:
    -   `shard_map`-wrapped Pallas/Mosaic kernels (Splash attention, Tokamax
        `gmm_v2`/`tgmm_v2`, Megablox) cannot be auto-partitioned by XLA and
        cannot infer `in_specs` reliably from `jax.typeof(x).sharding` under
        `ShardMode.AUTO` (which reports `P(None, ...)` inside `jit`).
    -   Resolve kernel/MoE executor specs from the `Sharding` object at module
        construction into a frozen dataclass (static graph metadata that
        survives `nnx.split` / `nnx.merge`, `jit`, and `ScannedLayers`).
2.  **Fail closed on unsupported mesh axes or remat tags**:
    -   Never silently fall back from Splash/Tokamax to dense $O(S^2)$ XLA
        attention, from `expert > 1` EP dispatch to replicated weights, or from
        a named `remat` policy whose `checkpoint_name` tags are missing in the
        model to silent full-layer remat. Raise a descriptive `ValueError` or
        `NotImplementedError` naming the unsupported config value.
3.  **Multi-axis sharding & Sequence Parallelism invariants**:
    -   Dense weights must include `"expert"` alongside `"fsdp"` (and
        `"fsdp_transpose"` when 2D FSDP is active) on the `embed` axis so
        `expert > 1` does not replicate dense layers across expert shards.
    -   Fused gated projections (`[gate | up]`) sharded on `"tensor"` must use a
        4D `(..., 2, intermediate)` layout (or separate `gate_proj` / `up_proj`)
        so splitting `gate` and `up` at `TP > 1` never triggers a full
        `all-gather`.
    -   Padding (`mask == 0`) must be zeroed out and packed-sequence
        `decoder_segment_ids` binarized (`segment_ids != 0`) in all MoE load
        balancing and auxiliary loss reductions.

## 4. Eight-Point `m3` PR Review Checklist

1.  **Dual-driver reachability**: Is any new `m3/infra/` or `m3/core/` feature
    wired into `src/maxtext/utils/model_creation_utils.py` (covering
    `MaxTextTrainingEngine`, `train.py`, and `train_compile.py`) in addition to
    `m3/train/loop.py`?
2.  **Path & state-treedef transparency**: Are `nnx.state(model, nnx.Param)`
    keys identical across `scan_layers=False/True`, `remat` policies, and
    quantization? Are there zero `.sow(...)` calls or step-mutated attributes on
    the engine path?
3.  **Eager sharding & multi-axis HLO verification**: Do all `nnx.Param`s use
    `eager_sharding=True` metadata (`param_metadata`), construct cleanly with
    `mesh=None`, and compile with zero unintended weight `all-gather` on meshes
    where **both** `tensor > 1` and `expert > 1`?
4.  **Registry completeness**: If a model is added to `MODEL_REGISTRY`, is it
    also registered in `MODEL_FLOPS_REGISTRY` and (when using aux/MTP losses or
    frozen params) `MODEL_ENGINE_CONFIG_REGISTRY`?
5.  **No silent kernel/sharding/remat fallbacks**: Does every accepted config
    mode reach its real kernel/mesh branch and raise loudly on unsupported axes
    or missing `checkpoint_name` tags?
6.  **TPU AOT pre-flight (`train_compile` / `aot_preflight.py`)**: For
    performance/kernel/mesh changes, was `train_compile` or `aot_preflight.py`
    (`compile_topology=tpu7x-128` or target slice) run to verify HBM headroom
    and `tpu_custom_call` / async collective lowering (since 4-CPU `pytest`
    skips Mosaic kernels)?
7.  **Torch-free 4-CPU `pytest` + negative controls + packing parity**: Do unit
    tests under `tests/unit/m3/` collect and pass without `torch` installed on
    4 CPU devices, pass `CheckId.PACKING_AND_PADDING_ISOLATION` (Check 9b) and
    `packing_train_check.py`, and include a perturbation negative control that
    fires?
8.  **Centralized checkpoint/FLOPs/train reuse & `**kwargs` driver contract**:
    Does the PR wire standard HF safetensors models into centralized
    `PARAM_MAPPING` / `HOOK_FNS` / `HF_SHAPE` (`to_maxtext` / `to_huggingface`),
    `MODEL_FLOPS_REGISTRY`, and `m3/train/loop.py` instead of adding bespoke
    `<model>_{ckpt,flops,train}.py` files, and does `Model.__call__` preserve
    `**kwargs` so generic `model_creation_utils` / engine callers do not raise
    `TypeError`?
