# MaxText Review Focus Area: Config and Checkpoint Compatibility

<!--
Shared focus area for MaxText code review. Used by the GitHub PR review
(.gemini/commands/maxtext-review.toml) and by an internal Code Review Agent
skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `src/maxtext/configs/`,
`src/maxtext/common/checkpointing.py`,
`src/maxtext/common/emergency_checkpointing.py`,
`src/maxtext/checkpoint_conversion/`,
`src/maxtext/utils/standalone_checkpointer.py`, or
`src/maxtext/utils/generate_param_only_checkpoint.py`.

Users launch MaxText with YAML configs and command-line overrides, and resume
or convert long-running training checkpoints. Silent changes here break
existing launch scripts, change training behavior without anyone noticing, or
make saved checkpoints unrestorable. Keep a finding only if you can name a
concrete config, override, or checkpoint that breaks or changes behavior.

## What to check

1.  **Removed or renamed config keys.** A key removed or renamed in
    `configs/types.py` or `configs/base.yml` while still referenced by model
    YAMLs under `configs/`, tests, docs, scripts, or launch examples, or
    without a deprecation path for external users (MaxText tracks deprecated
    settings in `types.py`). Search the tree for other uses of the old name.
2.  **Changed defaults.** A changed default in `base.yml` or `types.py` alters
    behavior for every existing config that does not set the key (e.g.,
    precision, remat policy, sharding rules, optimizer, data pipeline). Flag it
    unless the description states it and justifies it; recommend keeping the
    old default behind the new option when the change is not clearly a fix.
3.  **Schema consistency.** New keys must be declared consistently in the
    pydantic models in `types.py` and given defaults in `base.yml` (or the
    relevant model/hardware YAML), with types and allowed values that match
    how the code reads them.
4.  **Validation placement.** Invalid combinations of options should be
    rejected in the config system (pydantic `field_validator` /
    `model_validator` in `types.py`) with a clear message, not discovered
    later deep in model or training code, and not silently "fixed up".
5.  **Checkpoint compatibility.** Changes to parameter names, PyTree structure,
    module nesting, dtypes, or optimizer state layout that make checkpoints
    saved before the change fail to restore, or restore silently wrong. Check
    whether the change needs a conversion or migration step, and whether
    checkpoint-conversion code (`checkpoint_conversion/`, HF import/export)
    was updated to match.
6.  **Checkpointing behavior.** Changes to `common/checkpointing.py`,
    `common/emergency_checkpointing.py`, or standalone checkpointers that could
    lose data or corrupt state: saving non-atomically, multiple hosts writing
    the same path, not waiting for async saves before exit, changing step
    numbering or directory layout, or breaking resume from the latest step.
7.  **Tests.** A compatibility-relevant change should come with a test that
    loads an existing config (or restores a checkpoint saved in the old format)
    and asserts the expected behavior, not just a test of the new path.
