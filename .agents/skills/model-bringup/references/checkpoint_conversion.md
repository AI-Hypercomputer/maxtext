# Checkpoint Conversion (to_maxtext / to_huggingface)

This guide covers Phase 3 of the MaxText model bring-up workflow: extending the central checkpoint conversion utilities to transform HuggingFace Safetensors into MaxText Orbax checkpoints, and vice versa.

> [!IMPORTANT]
> **Standardize Checkpoint Tools**:
> Never create standalone custom conversion scripts. Always extend and reuse the central `to_maxtext.py` and `to_huggingface.py` conversion frameworks.

---

## 1. The Central Mapping System (`param_mapping.py` & `hf_shape.py`)
All weight parameter translation maps, value transformation functions, and target HF shapes are configured centrally in `src/maxtext/checkpoint_conversion/utils/param_mapping.py` and `src/maxtext/checkpoint_conversion/utils/hf_shape.py`.

To support a new model, you must extend these files:
1. **`PARAM_MAPPING` Registration (`param_mapping.py`)**:
   - Define a map builder function for your new model (e.g. `NEWMODEL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)`).
   - Register the map in the central `PARAM_MAPPING` dictionary.
   - Keys in the mapping must match JAX/Flax parameter path formats (e.g. `params-decoder-layers_0-self_attention-query-kernel`, or a tuple of keys for composite parameters).
   - Values must map to their corresponding HuggingFace safetensors naming formats (e.g. `model.layers.0.self_attn.q_proj.weight`).
   - **Mini-Model Override Support**: Read layer and expert counts from `maxtext_config` first before falling back to `config` (e.g. `getattr(maxtext_config, "num_decoder_layers", None) or config["num_hidden_layers"]`), because `to_maxtext.py` passes the full model's `HF_MODEL_CONFIGS[model_name]` as `config` even when CLI overrides like `base_num_decoder_layers={N}` are set.
2. **`HOOK_FNS` (Tensor Transformations in `param_mapping.py`)**:
   - Dictionaries that map a MaxText parameter name to a specific transformation function (a "hook").
   - Hooks handle transpositions, matrix splits/fusions, scaling factors, or paddings in both directions (`saving_to_hf=False` and `saving_to_hf=True`).
   - Define hook functions (e.g. `NEWMODEL_MAXTEXT_TO_HF_PARAM_HOOK_FN`) and register them in the central `HOOK_FNS` dictionary.
3. **`HF_SHAPE` (Target HF Shapes in `hf_shape.py`)**:
   - Define a shape builder function (e.g. `NEWMODEL_HF_WEIGHTS_TO_SHAPE(config)`) and register it in `HF_SHAPE` so `to_huggingface.py` can validate and reshape exported tensors.

### Matrix Transposition Hook Example:
Since PyTorch linear layers weight matrices are `(out, in)` while Flax JAX expects `(in, out)`, a transposition hook is crucial:
```python
# Register transposition hook
def transpose_linear_hook(value, *args, **kwargs):
  # Reverts PyTorch weight shape (out, in) to JAX (in, out)
  return value.transpose()
```

---

## 2. Registering the Model ID
Ensure that the new HuggingFace model ID is registered in:
- `src/maxtext/utils/globals.py` inside the central `HF_IDS` repository dictionary.
- `src/maxtext/checkpoint_conversion/utils/hf_model_configs.py` inside the `HF_MODEL_CONFIGS` dictionary mapping to set target dimension keys.

---

## 3. Execution of Checkpoint Conversion on TPU VM
Always execute checkpoint conversions directly on your TPU VM, running commands from the MaxText root directory:

- **Avoid filling up VM local space**: Always export `HF_HOME=/dev/shm/$USER` before running the tool.
- **Fast Iteration on Mini Subset First**: Before converting full multi-gigabyte models, ALWAYS test conversion against your local mini model in `/dev/shm/hf_mini/{model_name}_mini` with `$MINI_OVERRIDES` (e.g. `base_num_decoder_layers={N}`). This saves massive amounts of time and memory.
- **Dump checkpoint to shared memory**: Always set `base_output_directory=/dev/shm/$USER/checkpoints/...` for saving converted checkpoints. `to_maxtext.py` writes the Orbax parameter checkpoint at `<base_output_directory>/0/items`, which is the path to pass as `load_parameters_path` in downstream commands. If subsequent checks fail, immediately delete the converted checkpoint using `rm -rf` to free up shared memory space.
- **Dtype & multimodal flags**: `to_maxtext.py` defaults to `--save_dtype=bfloat16`; pass `--save_dtype=float32` when converting `float32` weights for Gate 0 exact weight-parity checks. For multimodal models (`use_multimodal=true`), also pass `--lazy_load_tensors=false`.

### Convert Mini HuggingFace Model to MaxText (to_maxtext)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name={model_name} \
  --hf_model_path=/dev/shm/hf_mini/{model_name}_mini \
  base_num_decoder_layers={N} \
  base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax \
  scan_layers=true \
  hardware=cpu \
  skip_jax_distributed_system=True
```

### Convert Full Model HuggingFace to MaxText (to_maxtext)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name={model_name} \
  base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_orbax \
  scan_layers=true \
  hardware=cpu \
  skip_jax_distributed_system=True
```

### Convert MaxText to HuggingFace (to_huggingface)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
  model_name={model_name} \
  load_parameters_path=/dev/shm/$USER/checkpoints/{model_name}_orbax/0/items \
  base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_hf_converted \
  scan_layers=true \
  hardware=cpu \
  skip_jax_distributed_system=True
```

---

## 4. Handover Document Update
Update `{model_name}_handover.md` in the MaxText root directory:
- Record the exact conversion CLI commands (mini and full).
- Document any weight transposition, tensor reshaping, or key mapping bugs discovered.
- Set Phase 3 status to `DONE` and Phase 4 to `IN PROGRESS`.
