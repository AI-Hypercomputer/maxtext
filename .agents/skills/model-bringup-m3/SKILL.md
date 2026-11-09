---
name: model-bringup-m3
description: Brings up a new LLM or MoE model family end-to-end in MaxText m3 (`src/maxtext/m3/`) using pure Flax NNX. Use when onboarding a HuggingFace model to `m3` (`use_m3_model=True`), implementing `Sharding` on `(data, fsdp, tensor, expert)`, writing HF/Orbax checkpoint converters, running the 8-check M3 conformance + `PerturbSpec` suite, or verifying real-weight `FrozenTopK` TPU parity. Do not use for legacy Flax Linen models under `src/maxtext/layers/`.
---

# MaxText M3 Model Bring-Up (`src/maxtext/m3/`)

Onboards a HuggingFace model family as a self-contained pure-NNX vertical slice in `src/maxtext/m3/models/<family>/`, verified end-to-end on CPU and TPU via `pytest`.

## 1. Vertical-Slice Isolation & M3 Architecture Rules

**Allowed touch list:**
- `src/maxtext/m3/models/<family>/`:
  - `__init__.py` — public exports (`<Family>Config`, `<Family>Sharding`, `<Family>Model`, `create_<family>_model`, helper and mapping exports)
  - `modeling_<family>.py` — config, sharding rules, norms, attention, MLP/MoE, stacked decoder, model, loss
  - `checkpoint_mappings_<family>.py` — declarative MaxText <-> HuggingFace parameter mapping (`<FAMILY>_MAXTEXT_TO_HF_PARAM_MAPPING`), tensor transform hooks (`<FAMILY>_MAXTEXT_TO_HF_PARAM_HOOK_FN`), and HF tensor shapes (`<FAMILY>_HF_WEIGHTS_TO_SHAPE`), plus lightweight table-driven in-memory weight helpers for unit tests (reusing the centralized conversion planner rather than a bespoke per-model CLI converter)
  - `utils_<family>.py` — `calculate_<family>_tflops_per_device(config, log=True)` and optional model-related helpers such as `configure_<family>_engine(engine)` (`MaxTextTrainingEngine` `aux_loss` adapter)
- `src/maxtext/m3/models/__init__.py` — lazy wrappers in `MODEL_REGISTRY`, `MODEL_FLOPS_REGISTRY`, `MODEL_ENGINE_CONFIG_REGISTRY`
- **Training Entrypoint Reuse (`src/maxtext/m3/train/loop.py`):** Text-only LLMs and MoEs reuse M3's dedicated training entrypoint `src/maxtext/m3/train/loop.py` (`train_loop` / `train_step`) directly — do **not** add a specialized `<family>_train.py` trainer. Only introduce a thin shim over `loop.py` when non-text/multimodal input or output modalities require model-specific data/step handling.
- **Centralized Checkpoint Conversion & Config Registration:**
  - `src/maxtext/configs/models/<model-name>.yml` — set `use_m3_model: true` and `param_scan_axis: 0` (omitting `"model_type"` so `m3.configs.config` selects `BaseConfig`, and keeping fields compatible with `MaxTextConfig` so both `pyconfig.initialize` and `m3.configs.config.initialize` load cleanly)
  - `src/maxtext/configs/types.py` & `src/maxtext/utils/globals.py` — register `<model-name>` in `ModelName` (enforcing `use_m3_model=True`) and `HF_IDS`
  - `src/maxtext/checkpoint_conversion/utils/{param_mapping.py,hf_shape.py,hf_model_configs.py}` — wire `checkpoint_mappings_<family>` into `PARAM_MAPPING`, `HOOK_FNS`, `HF_SHAPE`, and `HF_MODEL_CONFIGS` so `to_maxtext` and `to_huggingface` convert checkpoints end-to-end
- `tests/unit/m3/<family>*_test.py` — conformance, `PerturbSpec`, centralized checkpoint mapping/roundtrip, PyTorch parity, and TPU verification tests

**Never modify** legacy `src/maxtext/layers/`, `src/maxtext/models/`, or `model_creation_utils.py`. (Note: `.agents/` is listed in `.gitignore`; stage new skill files with `git add -f` in OSS.)

**Seven Load-Bearing NNX & `maxtext_engine` Invariants (`src/maxtext/m3/README.md`):**
1. **Pure Flax NNX built-ins:** Compose `nnx.Linear`, `nnx.Embed`, `nnx.RMSNorm`, `nnx.Param`, and `m3.core.rope.apply_rope` directly. No Linen bridges, no shared `Decoder`, no model-name branching.
2. **`nnx.Param` is the sole trainable variable type:** Other `nnx.Variable` subclasses silently receive zero gradient in `maxtext_engine`.
3. **Never `sow` or mutate non-`Param` state in `__call__`:** Mutating non-`Param` treedef permanently drops the engine onto its slow path. Pass inference `KVCache` explicitly as an external PyTree argument; default `return_aux: bool = False` in `__call__` (returning `logits` for `m3/train/loop.py` and `(logits, aux_dict)` when `return_aux=True`).
4. **Path-transparent `scan` and `remat`:** Create decoder layers as a single stacked module via `nnx.vmap` (`(num_layers, ...)` leading axis on every `nnx.Param`). Unrolled (`scan_layers=False`) and scanned (`scan_layers=True`) execution MUST share the exact same `nnx.GraphDef` and stacked parameter paths. **Never store Python `int` `self.layer_idx = i` on scanned submodules** (breaks `nnx.split` graphdef equality across layers).
5. **Fold auxiliary losses into primary loss:** Return `aux["moe_lb_loss"]` when `return_aux=True`, and wire `utils_<family>.configure_<family>_engine` into `MODEL_ENGINE_CONFIG_REGISTRY` via `engine.with_loss_fn(..., has_aux=True)` (keep `targets_segmentation` strictly on CE loss — never feed it into self-attention `attention_mask` or `moe_lb_loss`).
6. **Single shared `Mesh` + `nnx.eval_shape`-safe eager sharding:** Accept/validate `quant_mode_str: str | None = None` in `create_<family>_model` without forwarding it to `<Family>Config`. When not tracing (`not isinstance(jnp.zeros(()), jax.core.Tracer)`), construct inside `with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):`; whenever tracing (`nnx.eval_shape` or `jax.jit` inside `create_nnx_sharded_model` / `from_pretrained`), construct inside `with nnx.use_eager_sharding(False):` so abstract and jitted `nnx.Param` states share identical `eager_sharding` PyTree metadata. Set `self.mesh = mesh` (`model.mesh is driver_mesh`) and never store private `Mesh` objects on child modules.
7. **Zero post-init parameter constraints:** Shard weights once at init via `kernel_metadata={"out_sharding": sharding(...)}`. Inside `__call__`, call `sharding.constrain(x, name, axes)` on **activations only** — never call `with_sharding_constraint`, `reshard`, or `device_put` on parameter-derived weight arrays.

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
    return var.replace(jax.device_put(raw_val, jax.sharding.NamedSharding(mesh, new_spec)), out_sharding=new_spec)
  return var.replace(raw_val, out_sharding=new_spec)

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

## 3. Source Audit, 8-Check Conformance & `PerturbSpec` Negative Controls

Read upstream HuggingFace `configuration_<family>.py` and `modeling_<family>.py` plus live `config.json` and `model.safetensors` headers before coding:

| Common Model Pitfall | Verification Rule |
| :--- | :--- |
| Non-Llama scalar multipliers | Audit `embedding_multiplier`, `attention_multiplier` (replaces `1/sqrt(head_dim)`), `residual_multiplier` (both sublayers), and output `logits / logits_scaling`. |
| Tied `lm_head` with scaled embeddings | Project final norm output with **raw unscaled** `embed_tokens.embedding[...]` (`h @ W.T`), never pre-multiplied by `embedding_multiplier`. |
| Router & top-$k$ precision | Compute `router_logits` in `fp32` (`Precision.HIGHEST`); check whether `softmax` applies over top-$k$ logits only or full expert logits. |
| Switch `aux_loss` normalization | Check whether `tokens_per_expert` divides by `num_tokens` ($\sum_e f_e = k$, uniform loss $= k$) or `num_tokens * k`, and binarize `(attention_mask != 0)`. |
| Square-matrix transpose trap | In synthetic tests, enforce `2 * intermediate_size != hidden_size` (e.g., `H=64, I=48`) and override `router_aux_loss_coef > 0` when testing router `aux_loss`. |

**Static Test Hygiene:**
- Guard `import torch` / `import transformers` with `HAS_TORCH` and `@unittest.skipUnless(HAS_TORCH, ...)` so `pytest --collect-only` works without `torch`.
- Mark TPU-hardware test modules (`*_tpu_parity_test.py`) with `pytestmark = pytest.mark.tpu_only` (guarded by `try: import pytest`) — `tests/conftest.py` auto-marks unmarked tests as `cpu_only` and skips them on active TPU/GPU runners.
- Pass `is_leaf=lambda x: isinstance(x, nnx.Variable)` to all `jax.tree_util` calls over NNX state dicts (`nnx.Variable` is a registered PyTree node in Flax >= 0.12).
- Keep a single `if __name__ == "__main__": absltest.main()` entry point at EOF, and run `pyink --pyink --pyink-indentation=2 --line-length=122` + `pylint --rcfile=pylintrc` (`10.00/10`) before uploading.

**Mandatory 8 M3 Conformance Checks + 2 Bring-Up Gates (`tests/unit/m3/<family>_test.py` / `conformance_spec.py`):**
1. **`scan` Equivalence & Path Identity:** Identical parameter paths/shapes and forward outputs (`rtol=1e-5`) across `scan_layers=True` and `scan_layers=False`; `rngs=None` raises `(TypeError, ValueError)`.
2. **All-Leaf & Per-Expert Finite Non-Zero Gradients:** Every trainable leaf is `nnx.Param`, and every leaf (including router and every individual routed expert slice) receives a finite, non-zero gradient under `nnx.value_and_grad`.
3. **`remat` Equivalence & State Stability:** All `(scan_layers, remat_policy)` combinations match forward loss and per-leaf gradients; non-`Param` state treedef/leaves remain identical and finite before and after `__call__` (zero `sow` / `nnx.Intermediate`).
4. **`aux_loss` Live, Masked & Gradient-Connected:** `aux["moe_lb_loss"] > 0`, masks `attention_mask == 0` / `segment_ids == 0` padding tokens (keeping SFT `targets_segmentation` strictly on CE loss), and alters router gradients when `router_aux_loss_coef > 0` via `MODEL_ENGINE_CONFIG_REGISTRY` `with_loss_fn(..., has_aux=True)`.
5. **Driver `Mesh` Identity & Canonical Axis Names:** `model.mesh is driver_mesh`, zero private `Mesh` objects on child modules, and every `NamedSharding` / `PartitionSpec` on `nnx.Variable` metadata and `jax.Array` leaves references `driver_mesh` and `MESH_AXIS_NAMES = ("data", "fsdp", "tensor", "expert")`.
6. **Zero Post-Init Parameter Sharding Constraints:** Jaxpr parameter-taint walk across unary transforms, `scan`, `remat`, and `shard_map` confirming zero `sharding_constraint`, `reshard`, or `device_put` ops on parameter-derived arrays in `__call__`.
7. **Weight Sharding Coverage:** Every 2-D+ weight (`>= 256` unstacked elements, including non-divisible `vocab_size` guard) shards across `(data=1, fsdp=2, tensor=2, expert=1)` with addressable shard smaller than global shape, exercising both `"fsdp"` and `"tensor"`.
8. **Compiled StableHLO Zero Expert `all-gather` & Compound Mesh Parity:** Lower and compile a sharded train step across 4-device `(1, 2, 2, 1)`, `(1, 2, 1, 2)`, `(1, 1, 2, 2)` AND 8-device `(1, 2, 2, 2)` (`FSDP=2, TP=2, EP=2` all `> 1`), verifying `fp32` forward/grad parity vs `(1, 1, 1, 1)` and zero `all-gather(?:-start|-done)?` ops producing full, unstacked, scanned-singleton, or TP-sliced expert weight shapes in `compiled.as_text()`.
9. **Extra Gate — Autoregressive KV-Cache Equivalence:** Prefill + step-by-step `KVCache` decode matches full causal forward logits (`max_abs_diff <= 1e-5` in `fp32`), respecting `segment_ids` (`!= 0` padding guard) and `attention_mask`.
10. **Extra Gate — Centralized HF <-> Orbax Checkpoint Roundtrip & Torch-Free Mini-Twin Golden:** 100% bijection between `PARAM_MAPPING` / `HOOK_FNS` / `HF_SHAPE` (`checkpoint_mappings_<family>.py`) and `nnx.eval_shape` abstract model leaves across `scan_layers=True` and `scan_layers=False`, lossless `to_maxtext` <-> `to_huggingface` (`process_maxtext_param`) tensor roundtrip (`max_abs_diff == 0`), plus a committed torch-free mini-twin golden so forward/backward parity and `PerturbSpec` negative controls run in CPU PR CI without PyTorch.

**`PerturbSpec` Negative Controls (`REL_NORM_BAND = (5e-3, 2e-2)`):**
```python
@dataclass(frozen=True)
class PerturbSpec:
  name: str
  apply: Callable[[FamilyModel], None]  # mutates cloned model weight or config scalar
```
Perturb every custom scalar (`embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, `logits_scaling`) and critical weight leaf (router kernel, `gate_up_proj` scaled into relative L2 norm band `5e-3..2e-2`), asserting that parity assertions **raise `AssertionError`** on every perturbation. Align MaxText pre-shifted `(inputs=x[:, :-1], targets=x[:, 1:])` against HF internal `(logits[:, :-1], labels[:, 1:])` in loss parity checks.

## 4. Real-Weight `/dev/shm` Conversion, `FrozenTopK` Parity & Sharded Training

Execute real-weight TPU verification under `/dev/shm/<user>_m3_poc/` (passing `cache_dir` explicitly without mutating global `HF_HOME` or `jax_default_matmul_precision` at import time; pin `revision=` in `snapshot_download` and scope `jax_default_matmul_precision="highest"` inside `setUpClass`/`tearDownClass`):
1. **Full Checkpoint Audit (`to_maxtext` / `from_pretrained` / `to_huggingface`):** Convert full HF safetensors to Orbax via centralized `maxtext.checkpoint_conversion.to_maxtext`, restore onto sharded M3 meshes via `model_creation_utils.from_pretrained`, and roundtrip back through `process_maxtext_param` (`to_huggingface`); assert `100%` of HF keys consumed, `100%` of NNX `Param` leaves populated with exact shapes, and `max_abs_diff == 0.0`.
2. **Zero-Gen `FrozenTopK` Routed-MoE Parity:** Tiny `fp32` reduction differences can flip boundary-ULP router ties in deep MoEs. Verify:
   ```python
   # Inside router __call__(x, frozen_topk_indices=None):
   top_k_indices = frozen_topk_indices if frozen_topk_indices is not None else jax.lax.top_k(router_logits, self.top_k)[1]
   top_k_logits = jnp.take_along_axis(router_logits, top_k_indices, axis=-1)
   top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)
   ```
   - *Teacher-forced (`FrozenTopK`) + negative control:* Inject HF's per-layer `top_k_indices` (`(num_layers, B, S, k)`) into M3 while computing `top_k_logits` from M3's own `router_logits`, assert E2E `KL(P_hf || P_m3) <= 1e-5`, and verify a negative control perturbing `frozen_topk_indices` fails the KL gate.
   - *Teacher-forced router agreement + greedy decode:* Measure router agreement on pre-override `frozen_out.router_logits`, requiring **zero flips** on tokens whose reference margin `sorted_logits[..., k-1] - sorted_logits[..., k] >= 8 * max(spacing(|logit|), finfo(float32).eps)` (keep unforced rollout KL as a diagnostic), and verify multi-token greedy decode exact token-ID match vs `hf_model.generate(..., do_sample=False)`.
3. **Multi-Device Sharded Training Verification (`src/maxtext/m3/train/loop.py`):** Run `>= 5` sharded training steps on TPU (`FSDP` + `EP`/`TP`) via `maxtext.m3.train.loop` (`train_step` / `train_loop`) and verify `utils_<family>` engine loss (`router_aux_loss_coef > 0`), confirming finite non-zero grad norms, live non-zero `aux_loss`, strictly decreasing loss, and zero OOM/NaN.

## 5. Standard Verification Commands

- **Torch-free collection smoke check:** `pytest --collect-only tests/unit/m3/<family>_test.py`
- **CPU 8-device unit + conformance + HLO suite:** `JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=8" pytest -v tests/unit/m3/<family>_test.py`
- **TPU VM real-weight parity + unit suite:** `pytest -v tests/unit/m3/<family>_tpu_parity_test.py && pytest --noconftest -v tests/unit/m3/<family>_test.py`
