---
name: model-bringup-m3
description: Brings up a new LLM or MoE model family end-to-end in MaxText m3 (`src/maxtext/m3/`) using pure Flax NNX. Use when onboarding a HuggingFace model to `m3` (`use_m3_model=True`), implementing `Sharding` on `(data, fsdp, tensor, expert)`, writing HF/Orbax checkpoint converters, auditing `m3` structural contracts (`audit_m3_contracts.py`), running the 12-check M3 conformance suite (including Check 9b packing/padding isolation) + `packing_train_check.py` + `PerturbSpec` negative controls, or verifying real-weight `FrozenTopK` TPU parity. Do not use for legacy Flax Linen models under `src/maxtext/layers/`.
---

# MaxText M3 Model Bring-Up (`src/maxtext/m3/`)

Onboards a HuggingFace model family as a self-contained pure-NNX vertical slice in `src/maxtext/m3/models/<family>/`, verified end-to-end on CPU and TPU via `pytest`.

For the motivation and design rationale behind m3 (model garden layout, sharding as annotation, post-construction transforms, MoE runtime split), read `src/maxtext/m3/DESIGN.md` and [references/entrypoints_and_review_traps.md](references/entrypoints_and_review_traps.md). Where `DESIGN.md` disagrees with the code or `src/maxtext/m3/README.md`, the code and README are the source of truth.

## Scripts, Templates & References in This Skill

| Path | Purpose |
| :--- | :--- |
| [scripts/audit_m3_contracts.py](scripts/audit_m3_contracts.py) | Deterministic AST and wiring auditor for `src/maxtext/m3/` (`INFRA_NOT_IN_MODEL_CREATION_UTILS`, `MISSING_FLOPS_REGISTRY_ENTRY`, `FORBIDDEN_LINEN_OR_LEGACY_DECODER_IMPORT`, `MODEL_NAME_BRANCH_IN_SHARED_M3`, `ENGINE_UNSAFE_SOW_IN_M3`, `OUT_SHARDING_WITHOUT_EAGER_SHARDING`, `BESPOKE_PER_MODEL_BOILERPLATE_FILE`, and opt-in `--check-blaze-build`) |
| [references/entrypoints_and_review_traps.md](references/entrypoints_and_review_traps.md) | Dual-driver (`model_creation_utils.py` vs. `m3/train/loop.py`) transform wiring, `MaxTextTrainingEngine` vs. `train.py` contracts, static kernel descriptors, and 8-point `m3` PR review checklist |
| [templates/conformance_spec.py](templates/conformance_spec.py) | Ready-to-copy `ConformanceSpec` registration template aligned with `tests/unit/m3/conformance/spec.py` |
| [templates/parity_test.py](templates/parity_test.py), [templates/provenance_test.py](templates/provenance_test.py) | `unittest`/`pytest` component parity and SHA-256 provenance test templates using method-level `@unittest.skipUnless(HAS_TORCH, ...)` for torch-free CPU CI |
| [templates/design_handoff.md](templates/design_handoff.md), [templates/mini_twin.md](templates/mini_twin.md), [templates/golden_manifest.json](templates/golden_manifest.json) | Design handoff, mini-twin geometry specification, and SHA-256 golden manifest templates |
| [../model-bringup/scripts/](../model-bringup/scripts/) | Shared executable helpers: `inspect_hf_artifacts.py`, `family_diff.py`, `slice_hf_mini.py`, `capture_golden.py`, `negctl.py`, `layer_parity.py`, `e2e_check.py`, `ckpt_roundtrip_check.py`, `packing_train_check.py`, `check_jax_numerics_and_scale.py`, `aot_preflight.py`, `check_hlo_perf.py`, `extract_run_metrics.py` |

## 1. Vertical-Slice Isolation & M3 Architecture Rules

**Allowed touch list for a per-model vertical slice:**
- `src/maxtext/m3/models/<family>/`:
  - `__init__.py` — public exports (`<Family>Config`, `<Family>Sharding`, `<Family>Model`, `create_<family>_model`, helper and mapping exports)
  - `modeling_<family>.py` — config, sharding rules, norms, attention, MLP/MoE, stacked decoder, model, loss
  - `checkpoint_mappings_<family>.py` — declarative MaxText <-> HuggingFace parameter mapping (`<FAMILY>_MAXTEXT_TO_HF_PARAM_MAPPING`), tensor transform hooks (`<FAMILY>_MAXTEXT_TO_HF_PARAM_HOOK_FN`), and HF tensor shapes (`<FAMILY>_HF_WEIGHTS_TO_SHAPE`), plus lightweight table-driven in-memory weight helpers for unit tests (reusing the centralized conversion planner rather than a bespoke per-model CLI converter)
  - `utils_<family>.py` — `calculate_<family>_tflops_per_device(config, log=True)` and optional model-related helpers such as `configure_<family>_engine(engine)` (`MaxTextTrainingEngine` `aux_loss` adapter)
  - `conformance_spec.py` — `ConformanceSpec` registered via `register_conformance_spec(build_spec)` (from [templates/conformance_spec.py](templates/conformance_spec.py))
- `src/maxtext/m3/models/__init__.py` — lazy wrappers in `MODEL_REGISTRY`, `MODEL_FLOPS_REGISTRY`, `MODEL_ENGINE_CONFIG_REGISTRY`
- **Training Entrypoint Reuse (`src/maxtext/m3/train/loop.py`):** Text-only LLMs and MoEs reuse M3's dedicated training entrypoint `src/maxtext/m3/train/loop.py` (`train_loop` / `train_step`) directly — do **not** add a specialized `<family>_train.py` trainer. Only introduce a thin shim over `loop.py` when non-text/multimodal input or output modalities require model-specific data/step handling.
- **Centralized Checkpoint Conversion & Config Registration:**
  - `src/maxtext/configs/models/<model-name>.yml` — set `use_m3_model: true` and `param_scan_axis: 0` (omitting `"model_type"` so `m3.configs.config` selects `BaseConfig`, and keeping fields compatible with `MaxTextConfig` so both `pyconfig.initialize` and `m3.configs.config.initialize` load cleanly)
  - `src/maxtext/configs/types.py` & `src/maxtext/utils/globals.py` — register `<model-name>` in `ModelName` (enforcing `use_m3_model=True`) and `HF_IDS`
  - `src/maxtext/checkpoint_conversion/utils/{param_mapping.py,hf_shape.py,hf_model_configs.py}` — wire `checkpoint_mappings_<family>` into `PARAM_MAPPING`, `HOOK_FNS`, `HF_SHAPE`, and `HF_MODEL_CONFIGS` so `to_maxtext` and `to_huggingface` convert checkpoints end-to-end
- `tests/unit/m3/<family>*_test.py` — conformance, `PerturbSpec`, sequence-packing training parity, centralized checkpoint mapping/roundtrip, PyTorch parity, and TPU verification tests

**Never modify** legacy `src/maxtext/layers/` or `src/maxtext/models/` in a model vertical-slice PR; only shared `m3/infra/` transform PRs update `src/maxtext/utils/model_creation_utils.py` (see [references/entrypoints_and_review_traps.md](references/entrypoints_and_review_traps.md)). (Note: if `.agents/` is matched by `.gitignore` rules, check `git status` when staging skill updates.)

**Seven Load-Bearing NNX & `maxtext_engine` Invariants (`src/maxtext/m3/README.md`):**
1. **Pure Flax NNX built-ins:** Compose `nnx.Linear`, `nnx.Embed`, `nnx.RMSNorm`, `nnx.Param`, and `m3.core.rope.apply_rope` directly. No Linen bridges, no shared `Decoder`, no model-name branching in `m3/{core,infra,configs,train}/`.
2. **`nnx.Param` is the sole trainable variable type:** Other `nnx.Variable` subclasses silently receive zero gradient in `maxtext_engine`.
3. **Never `sow` or mutate non-`Param` state in `__call__`:** Mutating non-`Param` treedef permanently drops the engine onto its slow path. Pass inference `KVCache` explicitly as an external PyTree argument; default `return_aux: bool = False` in `__call__` (returning `logits` for `m3/train/loop.py` and `(logits, aux_dict)` when `return_aux=True`).
4. **Path-transparent `scan` and `remat`:** Create decoder layers as a single stacked module via `nnx.vmap` (`(num_layers, ...)` leading axis on every `nnx.Param`). Unrolled (`scan_layers=False`) and scanned (`scan_layers=True`) execution MUST share the exact same `nnx.GraphDef` and stacked parameter paths. **Never store Python `int` `self.layer_idx = i` on scanned submodules** (breaks `nnx.split` graphdef equality across layers).
5. **Fold auxiliary losses into primary loss:** Return `aux["moe_lb_loss"]` when `return_aux=True`, and wire `utils_<family>.configure_<family>_engine` into `MODEL_ENGINE_CONFIG_REGISTRY` via `engine.with_loss_fn(..., has_aux=True)` (keep `targets_segmentation` strictly on CE loss — never feed it into self-attention `attention_mask` or `moe_lb_loss`).
6. **Single shared `Mesh` + `nnx.eval_shape`-safe eager sharding:** Accept/validate `quant_mode_str: str | None = None` in `create_<family>_model` without forwarding it to `<Family>Config`. Construct parameters with `param_metadata` (`eager_sharding=True` alongside `out_sharding`); when tracing (`nnx.eval_shape` or `jax.jit` inside `create_nnx_sharded_model` / `from_pretrained`), guard `nnx.use_eager_sharding` consistently so abstract and jitted `nnx.Param` states share identical PyTree metadata. Set `self.mesh = mesh` (`model.mesh is driver_mesh`) and never store private `Mesh` objects on child modules.
7. **Zero post-init parameter constraints:** Shard weights once at init via `kernel_metadata=param_metadata(...)`. Inside `__call__`, call `sharding.constrain(x, name, axes)` on **activations only** — never call `with_sharding_constraint`, `reshard`, or `device_put` on parameter-derived weight arrays.

**Stacked Layer Construction & `NamedSharding` Placement Pattern:**
```python
forked_rngs = rngs.split(config.num_hidden_layers)
stacked = nnx.vmap(lambda r: FamilyDecoderLayer(config, sharding=sharding, rngs=r), in_axes=0, out_axes=0)(forked_rngs)
graphdef, state = nnx.split(stacked)

def _place_stacked(path, var):
  if not isinstance(var, nnx.Variable):
    return var
  orig = var.get_metadata("out_sharding", None)
  orig_spec = orig.spec if isinstance(orig, jax.sharding.NamedSharding) else orig
  new_spec = jax.sharding.PartitionSpec(None, *orig_spec)
  raw_val = var.get_raw_value()
  if mesh is not None and not mesh.empty and not isinstance(raw_val, jax.core.Tracer):
    return var.replace(
        jax.device_put(raw_val, jax.sharding.NamedSharding(mesh, new_spec)),
        out_sharding=new_spec,
        eager_sharding=True,
    )
  return var.replace(raw_val, out_sharding=new_spec, eager_sharding=True)

layers = nnx.merge(graphdef, jax.tree_util.tree_map_with_path(_place_stacked, state, is_leaf=lambda x: isinstance(x, nnx.Variable)))
```
At forward time, split `graphdef, params, non_params = nnx.split(self.layers, nnx.Param, ...)`, optionally wrap the per-layer body in `jax.checkpoint(..., policy=...)`, and run `jax.lax.scan` when `scan_layers=True` or a Python loop slicing axis `0` when `scan_layers=False`.

## 2. 4-Axis Logical Sharding, MoE Dispatch & Lazy Registries

Subclass `maxtext.m3.core.sharding.Sharding` over `MESH_AXIS_NAMES = ("data", "fsdp", "tensor", "expert")`:

```python
class FamilySharding(Sharding):
  def __init__(self, vocab_size: int = 0, num_experts: int = 1, tp_size: int = 1, fsdp_size: int = 1, ep_size: int = 1):
    self.vocab_tp = (vocab_size > 0 and vocab_size % tp_size == 0)
    self.compound_ep = (num_experts % (fsdp_size * ep_size) == 0)

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("data", "fsdp", "expert")
      case "embed", TensorType.WEIGHT:
        return ("fsdp", "expert")
      case "experts", TensorType.WEIGHT:
        return ("fsdp", "expert") if self.compound_ep else "expert"
      case "embed_exp", TensorType.WEIGHT:
        return None if self.compound_ep else "fsdp"
      case ("heads" | "kv_heads" | "mlp"), _:
        return "tensor"
      case "vocab", _:
        return "tensor" if self.vocab_tp else None
      case ("length" | "embed" | "head_dim" | "norm"), _:
        return None
      case _:
        raise KeyError(f"Unknown logical axis {axis!r} on {tensor_name!r}")
```

- **Non-divisible `vocab_size` guard:** Odd vocabularies (e.g., `49155`) leave `"vocab"` replicated (`None`) when `vocab_size % tp_size != 0` while still sharding `embed_tokens` `(V, H)` along `"embed"` (`("fsdp", "expert")`).
- **2-D+ Weight Coverage:** Every 2-D+ weight (`>= 256` unstacked elements) must shard across at least one active axis on `(data=1, fsdp=2, tensor=2, expert=1)` (store router kernel as `(hidden_size, num_experts)` on `("embed", None)`).
- **MoE Layout & Zero-All-Gather Dispatch (`FSDP >= 1, TP >= 1, EP >= 1`):**
  - Store `gate_up_proj` as 4-D `(E, H, 2, I)` (`("experts", "embed_exp", None, "mlp")`) so gate `[:, :, 0, :]` and up `[:, :, 1, :]` slice the replicated size-2 axis; store `down_proj` as 3-D `(E, I, H)` (`("experts", "mlp", "embed_exp")`).
  - When `compound_ep` holds and `fsdp * expert > 1` (or `tensor > 1`), dispatch inside `jax.shard_map(..., check_vma=False)` with exact parameter `PartitionSpec`s: keep axis tuple order `("fsdp", "expert")` identical across `P(("fsdp", "expert"), ...)` weights, `lax.axis_index(("fsdp", "expert"))` (`f_idx * ep_size + e_idx`), `lax.all_gather(..., ("fsdp", "expert"), axis=0, tiled=True)`, and `lax.psum_scatter(..., ("fsdp", "expert"), scatter_dimension=0, tiled=True)` (plus `lax.psum` over `"tensor"`). When token count `T` is not divisible by `fsdp * expert` (e.g., `B=1` KV-cache decode), fall back activation `in_specs`/`out_specs` to `P(None, None)` + `lax.psum(..., ("tensor", "fsdp", "expert"))` while keeping expert weights sharded.
  - **TPU `ragged_dot` VJP Tail Masking:** When `sum(group_sizes) < M` (partial local expert slots under EP/FSDP), explicitly mask non-local tail rows `[sum(group_sizes):M]` to `0` on `sorted_x`, SwiGLU `act`, and `sorted_out` via `jnp.where(sorted_valid, ..., 0)` — TPU XLA's `ragged-dot` lhs VJP does not zero rows past `sum(group_sizes)`.
- **Lazy Registries & Consolidated Model Helpers (`src/maxtext/m3/models/__init__.py`, `utils_<family>.py`):**
  - In `MODEL_REGISTRY`, validate `if rngs is None: raise ValueError(...)` and import `<family>` modules lazily inside wrapper functions.
  - In `utils_<family>.py`, consolidate `calculate_<family>_tflops_per_device(config, log=True) -> (total, weight, attn)` (`B = per_device_batch_size`, `3 * [2*B*S*(L*attn_params + L*H*E + L*k*3*H*I + H*V)] / 1e12` weight + `3 * [L * 2*B*S^2*H_q*D] / 1e12` causal attention, times `gradient_accumulation_steps`, verified within `< 0.5%` of analytical FLOPs) alongside `configure_<family>_engine(engine)`, which calls `engine.with_loss_fn(<family>_engine_loss_fn, has_aux=True)` and `engine.with_gen_model_input_fn(None)` (never mutating `engine._loss_fn` directly) and returns `(WeightedMetric(unreduced_sum=(total_loss * total_weights).astype(jnp.float32), denominator=total_weights), aux)`.

## 3. Source Audit, 12-Check Conformance, Packing Parity & Negative Controls

Read upstream HuggingFace `configuration_<family>.py` and `modeling_<family>.py` plus live `config.json` and `model.safetensors` headers (`../model-bringup/scripts/inspect_hf_artifacts.py` and `family_diff.py`) before coding:

| Common Model Pitfall | Verification Rule |
| :--- | :--- |
| Non-Llama scalar multipliers | Audit `embedding_multiplier`, `attention_multiplier` (replaces `1/sqrt(head_dim)`), `residual_multiplier` (both sublayers), and output `logits / logits_scaling`. |
| Tied `lm_head` with scaled embeddings | Project final norm output with **raw unscaled** `embed_tokens.embedding[...]` (`h @ W.T`), never pre-multiplied by `embedding_multiplier`. |
| Router & top-$k$ precision | Compute `router_logits` in `fp32` (`Precision.HIGHEST`); check whether `softmax` applies over top-$k$ logits only or full expert logits. |
| Switch `aux_loss` normalization | Check whether `tokens_per_expert` divides by `num_tokens` ($\sum_e f_e = k$, uniform loss $= k$) or `num_tokens * k`, and binarize `(attention_mask != 0)` / `(segment_ids != 0)`. |
| Square-matrix transpose trap | In synthetic tests, enforce `2 * intermediate_size != hidden_size` (e.g., `H=64, I=48`) and override `router_aux_loss_coef > 0` when testing router `aux_loss`. |

**Static Test Hygiene (`templates/parity_test.py`, `templates/provenance_test.py`):**
- Never put an unguarded top-level `import torch` or `import transformers` in `tests/unit/*_test.py`. Compute `HAS_TORCH = importlib.util.find_spec("torch") is not None` and put `@unittest.skipUnless(HAS_TORCH, ...)` strictly at the **method level** only on tests requiring PyTorch, so 100% of pure-JAX/golden/conformance tests collect and pass in torch-free CPU venvs.
- Mark TPU-hardware test modules (`*_tpu_parity_test.py`) with `pytestmark = pytest.mark.tpu_only` (guarded by `try: import pytest`) — `tests/conftest.py` auto-marks unmarked tests as `cpu_only` and skips them on active TPU/GPU runners.
- Pass `is_leaf=lambda x: isinstance(x, nnx.Variable)` to `jax.tree_util` calls inspecting `nnx.Variable` metadata (`nnx.Variable` is a registered PyTree node in Flax >= 0.12).
- Run `scripts/audit_m3_contracts.py`, `../model-bringup/scripts/check_jax_numerics_and_scale.py`, `pyink --pyink --pyink-indentation=2 --line-length=122`, and `pylint` before uploading.

**Mandatory M3 Conformance Suite (`tests/unit/m3/conformance/{spec.py,checks.py,conformance_test.py}`) + Bring-Up Gates:**
Register `src/maxtext/m3/models/<family>/conformance_spec.py` (from [templates/conformance_spec.py](templates/conformance_spec.py)) so `tests/unit/m3/conformance/conformance_test.py` runs Checks 1–12 automatically:
1. **Check 1 (`SCAN_REMAT_EQUIVALENCE`):** Identical parameter paths/shapes and forward/gradient outputs across `scan_remat_variants` (`(False/True, "none"/"full")`); `rngs=None` rejected.
2. **Check 2 (`PARAM_AND_GRAD_HEALTH`):** Every trainable leaf is `nnx.Param`, and every non-frozen leaf receives a finite, non-zero gradient.
3. **Check 3 (`NON_PARAM_STATE_STABILITY`):** Non-`Param` state treedef/leaves (outside declared `mutable_non_param_patterns`) stay identical and finite before and after `forward_fn` (zero `.sow(...)`).
4. **Check 4 (`AUX_LOSS_GRAD_COUPLING`):** Every declared `aux_losses` key alters gradients of its target parameter regexes when weighted `1.0` vs `0.0`.
5. **Check 5 (`DRIVER_MESH_AND_AXIS_NAMES`):** Single shared `driver_mesh` and canonical `MESH_AXIS_NAMES = ("data", "fsdp", "tensor", "expert")`.
6. **Check 6 (`NO_POST_INIT_PARAM_CONSTRAINTS`):** Zero `sharding_constraint`, `reshard`, or `device_put` ops on parameter-derived arrays in `forward_fn`.
7. **Check 7 (`WEIGHT_SHARDING_COVERAGE`):** Every 2-D+ weight (`>= min_sharded_param_elements`) shards across `(data=1, fsdp=2, tensor=2, expert=1)`.
8. **Check 8 (`NO_EXPERT_WEIGHT_ALL_GATHER`):** Zero `all-gather` ops materializing expert weight shapes on multi-axis meshes (`FSDP`, `TP`, `EP`).
9. **Check 9a/9b/9c (`BF16_FORWARD_PARITY`, `PACKING_AND_PADDING_ISOLATION`, `EXPERT_PARALLEL_PARITY`):**
   - **Check 9b (`PACKING_AND_PADDING_ISOLATION`) + `packing_train_check.py` (Mandatory):** Check 9b verifies multi-document packed `segment_ids` (`{1, 0, 2, 0}`) bit-exact forward isolation, padding invariance on primary outputs and `aux_losses`, and zero cross-document/padding input-embedding gradients (`d(loss_doc2)/d(doc1) == 0`, `d(loss_doc2)/d(pad) == 0`). Pair this with `../model-bringup/scripts/packing_train_check.py` (`verify_packed_training_parity` or `--model-name=<model-name> use_m3_model=true`) to verify per-leaf parameter gradient superposition, multi-step `train_step` + Optax trajectory parity, and the `segment_ids=1` mutant-kill negative control.
10. **Check 10 (`OPTIMIZER_STEP_AND_FROZEN_WEIGHT_DECAY`):** Multi-step optimizer update health and exact invariance of `frozen_param_patterns` under AdamW weight decay.
11. **Check 11 (`SHARDED_CHECKPOINT_SLICE_LOAD`):** Sharded checkpoint save/slice-load roundtrip consistency, paired with `../model-bringup/scripts/ckpt_roundtrip_check.py` (`to_maxtext` <-> `to_huggingface`).
12. **Check 12 (`FULL_SCALE_TOPOLOGY_AOT_AND_REMAT_HBM`):** Full-scale topology AOT compilation and rematerialization HBM reduction across `full_scale_topologies` (also testable standalone via `../model-bringup/scripts/aot_preflight.py` and `check_hlo_perf.py`).
13. **Extra Bring-Up Gates — Autoregressive KV-Cache & Torch-Free Mini-Twin Golden Parity:** Prefill + step-by-step `KVCache` decode parity (`max_abs_diff <= 1e-5` in `fp32`), plus component parity (`templates/parity_test.py` + `../model-bringup/scripts/layer_parity.py`) and SHA-256 golden provenance (`templates/provenance_test.py` + `capture_golden.py`).

**`PerturbSpec` & `negctl` Negative Controls (`REL_NORM_BAND = (5e-3, 2e-2)`):**
Use `../model-bringup/scripts/negctl.py` (`perturb_one_leaf`), `layer_parity.negative_control_fires`, and `e2e_check.negative_run_status` to perturb custom scalars (`embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, `logits_scaling`) and weight leaves within `REL_NORM_BAND = (5e-3, 2e-2)`, asserting every parity check **fails (`FIRED`)** under perturbation.

## 4. Real-Weight `/dev/shm` Conversion, `FrozenTopK` Parity & Sharded Training

Execute real-weight TPU verification under `/dev/shm/<user>_m3_poc/` (passing `cache_dir` explicitly without mutating global `HF_HOME` or `jax_default_matmul_precision` at import time; pin `revision=` in `snapshot_download` and scope `jax_default_matmul_precision="highest"` inside `setUpClass`/`tearDownClass`):
1. **Full Checkpoint Audit (`to_maxtext` / `from_pretrained` / `to_huggingface`):** Convert full HF safetensors to Orbax via centralized `maxtext.checkpoint_conversion.to_maxtext`, validate with `../model-bringup/scripts/{verify_checkpoint.py,audit_ckpt_vs_hf.py,ckpt_roundtrip_check.py}`, restore onto sharded M3 meshes via `model_creation_utils.from_pretrained`, and roundtrip back through `process_maxtext_param` (`to_huggingface`).
2. **Zero-Gen `FrozenTopK` Routed-MoE Parity (`e2e_check.py`):** Tiny `fp32` reduction differences can flip boundary-ULP router ties in deep MoEs. Verify:
   ```python
   # Inside router __call__(x, frozen_topk_indices=None):
   top_k_indices = frozen_topk_indices if frozen_topk_indices is not None else jax.lax.top_k(router_logits, self.top_k)[1]
   top_k_logits = jnp.take_along_axis(router_logits, top_k_indices, axis=-1)
   top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)
   ```
   - *Teacher-forced (`FrozenTopK`) + negative control:* Inject HF's per-layer `top_k_indices` (`(num_layers, B, S, k)`) into M3 while computing `top_k_logits` from M3's own `router_logits`, assert E2E `LogitsTol(kl_mean_max=1e-5, top1_min=...)` via `e2e_check.check_outputs`, and assert `e2e_check.negative_run_status(...) == ("FIRED", [])`.
   - *Teacher-forced router agreement + greedy decode:* Verify pre-override `router_logits` via `e2e_check.check_router_agreement` (`RouterAgreementTol(top_k=..., min_agreement=0.99, tie_margin_ulp=8.0)`), and verify multi-token greedy decode exact token-ID match vs `hf_model.generate(..., do_sample=False)`.
3. **Multi-Device Sharded Training Verification (`src/maxtext/m3/train/loop.py`):** Run `>= 5` sharded training steps on TPU (`FSDP` + `EP`/`TP`) via `maxtext.m3.train.loop` (`train_step` / `train_loop`), extract metrics via `../model-bringup/scripts/extract_run_metrics.py`, and verify `utils_<family>` engine loss (`router_aux_loss_coef > 0`), confirming finite non-zero grad norms, live non-zero `aux_loss`, strictly decreasing loss, and zero OOM/NaN.

## 5. Standard Verification Commands

- **M3 structural & registry contract audit:** `python3 .agents/skills/model-bringup-m3/scripts/audit_m3_contracts.py --maxtext-root . --check-registries --base origin/main src/maxtext/m3/models/<family>/modeling_<family>.py`
- **Sequence-packing multi-step training parity + mutant kill:** `PYTHONPATH=$PWD/src python3 .agents/skills/model-bringup/scripts/packing_train_check.py --model-name=<model-name> use_m3_model=true`
- **Torch-free collection smoke check:** `pytest --collect-only tests/unit/m3/<family>_test.py`
- **CPU 8-device unit + 12-check conformance + HLO suite:** `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=8" pytest -v tests/unit/m3/conformance/conformance_test.py tests/unit/m3/<family>_test.py`
- **TPU VM real-weight parity + unit suite:** `pytest -v tests/unit/m3/<family>_tpu_parity_test.py && pytest --noconftest -v tests/unit/m3/<family>_test.py`
