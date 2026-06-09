# Phase 2: Parameter Mapping

## GOAL
Align layer designations between Hugging Face and MaxText to confirm total parameter coverage using the information from phase 1 (`<model_family>_tracing.json` file).

## Inputs from Previous Phases
- **`<model_family>_tracing.json`** (from Phase 1): Used to understand the hierarchical structure and names of parameters to confirm total parameter coverage.

## Expected Outcome / Outputs Generated
- **`{MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING` function**: Implemented within [`param_mapping.py`](../../../../../checkpoint_conversion/utils/param_mapping.py). This function provides a 1:1 mapping structure linking Hugging Face parameter string paths to their MaxText equivalents. This mapping dictionary will be used in Phase 3 and Phase 4.
