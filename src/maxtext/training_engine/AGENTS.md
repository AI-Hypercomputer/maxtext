This directory is undergoing significant refactoring.
Our current goal is to port code from the `atwigg/mlperf` branch into `main`.
However, we want to fix some hacks that were added during the MLPerf rush.

Here are the refactoring rules:

## APIs and code structure
* We want to preserve the `AbstractTrainer` APIs. The MLPerf branch added several new APIs. Any API change must be clearly justified in PR descriptions.

## Configs
The execution scripts for this code currently live in the Tunix `atwigg/mlperf` branch under the `experimental` directory. Check if the features that we are porting were actually used in the final MLPerf submission.

* MaxText config settings (in `base.yml` and `types.py`) that are RL-specific should go into the `rl` subcategory (see `class RL(BaseModel):` in `types.py`). For example, the `abandon_failed_checkpoint_saves` flag should become `rl.abandon_failed_checkpoint_saves`.
* Similarly, rollout-specific config settings should move to the `vllm` subcategory (e.g., `rollout_fp8_moe` should become `vllm.rollout_fp8_moe`).
* MaxText should be configured using Pydantic config and not via environment variables. Replace environment variables with config flags.
* AI agents tend to use defensive `getattr(config, "flag_name", None)`. We should directly use `config.flag_name` instead. Some unit tests might need to be updated to use Pydantic to avoid test failures.

If config flags are changed according to the rules above, clearly document this in the PR description so that the execution scripts can be updated accordingly.
