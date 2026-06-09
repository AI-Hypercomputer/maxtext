# Phase 5: Bidirectional Conversion (Optional)

## GOAL
Confirm the operational integrity of the bidirectional conversion process.

## Expected Outcome
A newly generated Hugging Face checkpoint and an associated end-to-end verification summary, verifying that checkpoints can be seamlessly translated back from MaxText to the Hugging Face format without any degradation in model outputs.


## Command to run scripts

**1. Convert from Hugging Face to MaxText:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu <path_to_venv>/bin/python \
-m maxtext.checkpoint_conversion.to_maxtext \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
base_output_directory=<maxtext_checkpoint_output_dir> \
hf_access_token=${HF_TOKEN} \
hardware=cpu skip_jax_distributed_system=True scan_layers=False`

**2. Convert from MaxText to Hugging Face:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu <path_to_venv>/bin/python \
-m maxtext.checkpoint_conversion.to_huggingface \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
load_parameters_path=<maxtext_checkpoint_output_dir> \
hf_output_directory=<hf_checkpoint_output_dir> \
hardware=cpu skip_jax_distributed_system=True scan_layers=False`