# Phase 0: Orchestration & State Control

## Objective
Entry point for eng: One SKILL to rule them all.

## Expected Outcome
All codes written and passed full model logits checker.

As the primary entry point, the orchestration stage utilizes this skill to launch the sequence.

### Development Environment
The orchestration SKILL establishes the automated cycle by specifying:
- **Connectivity**: Details for the TPU-VM, such as the IP address and necessary SSH instructions, to be used for debugging (optional as Jetski is cloudtop-exclusive).
- **Workspace**: The local MaxText repository path and its virtual environment.
- **Commands**: Standard tool execution commands (see Phase 4 for examples).
- **Storage**: GCS bucket to store the checkpoints (optional).

### Staged Execution
Organize the workflow into distinct phases, ensuring each is successfully completed before proceeding to the next.

### Routing & Iteration
Oversee phase transitions and manage feedback loops. For example, if Phase 4 (exuection) fails due to a high max difference in `forward_pass_logit_checker.py`, revert back to Phase 3 (Transformation) and run `layer_shape_verify.py` and `layerwise_verify.py` to identify which specific layer is causing the divergence.

### Milestone Reporting
Record findings and implementation nuances after each milestone. This real-time documentation allows engineers to track progress effectively without navigating through raw log outputs.
