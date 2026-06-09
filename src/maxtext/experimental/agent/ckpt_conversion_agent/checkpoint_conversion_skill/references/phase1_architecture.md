# Phase 1: Architecture Investigation for HF and MaxText

## Goal
Conduct a detailed analysis of model specifics for both Hugging Face and MaxText utilizing [`graph_tracing.py`](../../utils/graph_tracing.py) to establish a project roadmap.

## Expected Outcome / Outputs Generated
- **`<model_family>_tracing.json`**: A JSON file documenting the checkpoint architectures of the models. This file maps out the dimensions, names, and hierarchical structure of parameters, ensuring a thorough understanding before any mapping begins. This JSON file will be used as the primary input for Phase 2 and Phase 3.

- **Config Alignment** Read the target model's HuggingFace config.json and ensure every structural parameter perfectly matches the MaxText {MODEL}.yml file (or base.yml defaults). Pay explicit attention to:
rms_norm_eps
mlp_activations
logits_via_embedding (Weight tying)
attention_kwargs (softcapping, query scaling, sliding windows)
max_target_positions and RoPE base frequencies.


## Command to run scripts
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
src/maxtext/experimental/agent/ckpt_conversion_agent/utils/graph_tracing.py \
 src/maxtext/configs/base.yml model_name=phi4 scan_layers=False \
--hf_path google/gemma-3-4b-it \
--mt_path <path-to-maxtext-checkpoint> \
--output_file src/maxtext/experimental/agent/ckpt_conversion_agent/utils/gemma3_tracing.json \


Note that you should set `scan_layers=False`. Also ensure the output file is named `<model_family>_tracing.json` (e.g., `gemma3_tracing.json`) and placed in the `utils` directory so that subsequent scripts can find it.