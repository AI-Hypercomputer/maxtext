# Phase 4: Execution of Conversion

## GOAL
Conduct the conversion by applying the results from Phases 2 and 3 within the consolidated checkpoint conversion framework. This stage includes performing a comprehensive end-to-end checkpoint conversion assessment via `forward_pass_logit_checker.py`.

## Inputs from Previous Phases
- **`<model_family>_tracing.json`** (from phase 1)
- **`{MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING`** (from Phase 2)
- **`{MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN`** (from Phase 3)
- **MaxText checkpoint** (from Phase 3, if already successfully generated)

## Expected Outcome / Outputs Generated
- **An initialized MaxText checkpoint.**
- **A detailed end-to-end verification report**: This report confirms that given an identical input sequence, the max difference in logits between the HF reference and the MaxText implementation is within the acceptable threshold. **If this phase fails (e.g., logits diverge significantly), the agent must roll back to Phase 3 to debug and fix the layer transformation logic.**

## Command to run scripts

**1. Convert from Hugging Face to MaxText:**
*(Skip this step if a MaxText checkpoint was already generated in Phase 3 and successfully passed `layerwise_verify.py`)*
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
-m maxtext.checkpoint_conversion.to_maxtext \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
base_output_directory=<maxtext_checkpoint_output_dir> \
hf_access_token=${HF_TOKEN} \
hardware=cpu skip_jax_distributed_system=True scan_layers=False`

*(Note: Use `scan_layers=False` unless the model architecture explicitly defines a `ScannableBlockToLinen` wrapper in `decoders.py:get_decoder_layers()`.)*

**2. Forward Pass Logit Checker:**
*(If this check fails, record the errors and return to Phase 3 to debug and correct the layer-wise mappings.)*
`TMPDIR=/dev/shm JAX_PLATFORMS=cpu ${VIRTUAL_ENV}/bin/python \
tests/utils/forward_pass_logit_checker.py \
src/maxtext/configs/base.yml \
model_name=gemma3-4b \
load_parameters_path=<maxtext_checkpoint_output_dir> \
scan_layers=False \
--hf_model_path google/gemma-3-4b-it \
--run_hf_model True \
--trust_remote_code "" \
--atol 0.01 --max_kl_div 0.01`

*(Note: For modern transformers, bypass the `trust_remote_code` bug by passing `--trust_remote_code ""` rather than `False` due to `argparse` boolean parsing behavior. Ensure at least one test criteria like `--atol` or `--max_kl_div` is supplied.)*
