# Phase 0: Orchestration & State Control

## Objective
Entry point for eng: One SKILL to rule them all.

## Successful Criteria
All codes written and passed full model logits checker.

As the primary entry point, the orchestration stage utilizes `@ckpt-conversion-agent` to launch the sequence.

### Development Environment
The orchestration SKILL establishes the automated cycle by specifying:
- **Connectivity**: Details for the TPU-VM, such as the IP address and necessary SSH instructions, to be used for debugging (optional as Jetski is cloudtop-exclusive).
- **Workspace**: The local MaxText repository path and its virtual environment.
- **Commands**: Standard tool execution commands (e.g., `to_maxtext.py`).
- **Storage**: GCS bucket to store the checkpoints (optional).

### Staged Execution
Organize the workflow into four distinct phases, ensuring each is successfully completed before proceeding to the next.

### Routing & Iteration
Oversee phase transitions and manage feedback loops, such as reverting to Phase 3 (Mapping) if Phase 4 (Validation) fails.

### Milestone Reporting
Record findings and implementation nuances after each milestone. This real-time documentation allows engineers to track progress effectively without navigating through raw log outputs.
