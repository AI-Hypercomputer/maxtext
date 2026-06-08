# Phase 4: Execution of Conversion

## Objective
Conduct the conversion by applying the results from Phases 2 and 3 within the consolidated checkpoint conversion framework. This stage includes performing a comprehensive end-to-end checkpoint conversion assessment via `forward_pass_logit_checker.py`.

## Result
An initialized MaxText checkpoint alongside a detailed end-to-end verification report. The report confirms that given an identical input sequence, the max difference in logits between the HF reference and the MaxText implementation is within the acceptable threshold.

## Command to run scripts

**1. Convert from Hugging Face to MaxText:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
-m maxtext.checkpoint_conversion.to_maxtext \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
base_output_directory=<maxtext_checkpoint_output_dir> \
hf_access_token=${HF_TOKEN} \
hardware=cpu skip_jax_distributed_system=True scan_layers=True`

**2. Forward Pass Logit Checker:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
tests/utils/forward_pass_logit_checker.py \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
load_parameters_path=<maxtext_checkpoint_output_dir> \
--hf_model_path google/gemma-3-4b-it \
--run_hf_model True`

**3. Convert from MaxText to Hugging Face:**
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
-m maxtext.checkpoint_conversion.to_huggingface \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
load_parameters_path=<maxtext_checkpoint_output_dir> \
hf_output_directory=<hf_checkpoint_output_dir> \
hardware=cpu skip_jax_distributed_system=True scan_layers=True`
