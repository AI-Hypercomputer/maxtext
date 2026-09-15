# Checkpoint Conversion (to_maxtext / to_huggingface)

This guide covers Phase 3 of the MaxText model bring-up workflow: extending the central checkpoint conversion utilities to transform HuggingFace Safetensors into MaxText Orbax checkpoints, and vice versa.

> [!IMPORTANT]
> **Standardize Checkpoint Tools**:
> Never create standalone custom conversion scripts. Always extend and reuse the central `to_maxtext.py` and `to_huggingface.py` conversion frameworks.

---

## 1. The Central Mapping System (`param_mapping.py`)
All weight parameter translation maps and value transformation functions are configured centrally in `src/maxtext/checkpoint_conversion/utils/param_mapping.py`.

To support a new model, you must extend this file:
1. **`PARAM_MAPPING` Registration**:
   - Define a map builder function for your new model (e.g. `NEWMODEL_MAXTEXT_TO_HF_PARAM_MAPPING`).
   - Register the map in the central `PARAM_MAPPING` dictionary.
   - Keys in the mapping must match JAX/Flax parameter path formats (e.g. `params-decoder-layers-layers_0-self_attention-query-kernel`).
   - Values must map to their corresponding HuggingFace safetensors naming formats (e.g. `model.layers.0.self_attn.q_proj.weight`).
2. **`HOOK_FNS` (Tensors Transformations)**:
   - Dictionaries that map a MaxText parameter name to a specific transformation function (a "hook").
   - Hooks handle transpositions, matrix splits, scaling factors, or paddings.
   - Define hook functions (e.g. `NEWMODEL_MAXTEXT_TO_HF_PARAM_HOOK_FN`) and register them in the central `HOOK_FNS` dictionary.

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
- **Fast Iteration on Mini Subset First**: Before converting full multi-gigabyte models, ALWAYS test conversion against your local mini safetensors subset created in `/dev/shm/hf_mini/{model_name}_xlayers`. Set `base_num_decoder_layers=N` in the command. This saves massive amounts of time and memory.
- **Dump checkpoint to shared memory**: Always set `base_output_directory=/dev/shm/$USER/checkpoints` (or a subdirectory within it) for saving converted checkpoints. If subsequent logits checks fail, immediately delete the converted checkpoint using `rm -rf` to free up shared memory space for the next iteration.

### Convert Mini HuggingFace Subset to MaxText (to_maxtext)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_maxtext   src/maxtext/configs/base.yml   model_name={model_name}   hf_model_path=/dev/shm/hf_mini/{model_name}_xlayers   base_num_decoder_layers={N}   base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax   scan_layers=true   hardware=cpu   skip_jax_distributed_system=True
```

### Convert Full Model HuggingFace to MaxText (to_maxtext)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_maxtext   src/maxtext/configs/base.yml   model_name={model_name}   base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_orbax   scan_layers=true   hardware=cpu   skip_jax_distributed_system=True
```

### Convert MaxText to HuggingFace (to_huggingface)
```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.checkpoint_conversion.to_huggingface   src/maxtext/configs/base.yml   model_name={model_name}   load_parameters_path=/dev/shm/$USER/checkpoints/{model_name}_orbax   base_output_directory=/dev/shm/$USER/checkpoints/{model_name}_hf_converted   scan_layers=true   hardware=cpu   skip_jax_distributed_system=True
```

---

## 4. Handover Document Update
Update `{model_name}_handover.md` in the MaxText root directory:
- Record the exact conversion CLI commands (mini and full).
- Document any weight transposition, tensor reshaping, or key mapping bugs discovered.
- Set Phase 3 status to `DONE` and Phase 4 to `IN PROGRESS`.
