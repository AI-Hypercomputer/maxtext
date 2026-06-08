---
name: ckpt-conversion-agent
description: >-
  Automates the end-to-end checkpoint conversion workflow between Hugging Face and MaxText. Use this skill when orchestrating the checkpoint conversion process, which includes architecture investigation, parameter mapping, layer-by-layer transformation, full execution of conversion, and optional bidirectional conversion.
---

# Checkpoint Conversion Workflow

The automatic checkpoint conversion agent flow is a 6-phase (0-5) structured process to convert model weights between Hugging Face and MaxText, ensuring accuracy and correctness at every step.

As the primary entry point, this orchestration SKILL establishes the automated cycle and delegates execution to the appropriate phases.

## Workflow Phases

### [Phase 0: Orchestration & State Control](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase0_orchestration.md)
**Objective**: Entry point for eng: One SKILL to rule them all. Establish the automated cycle, connectivity, environment, execution staging, and routing.

### [Phase 1: Architecture Investigation for HF and MaxText](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase1_architecture.md)
**Objective**: Conduct a detailed analysis of model specifics for both Hugging Face and MaxText utilizing `graph_tracing.py` to establish a project roadmap.

### [Phase 2: Parameter Mapping](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase2_parameter_mapping.md)
**Objective**: Align layer designations between Hugging Face and MaxText to confirm total parameter coverage.

### [Phase 3: Layer-by-Layer Transformation](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase3_layer_transformation.md)
**Objective**: Specify the necessary transformation logic for every individual layer. Pass `layerwise_verify.py` for layer-by-layer check.

### [Phase 4: Execution of Conversion](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase4_execution.md)
**Objective**: Conduct the conversion by applying the results from Phases 2 and 3 within the consolidated checkpoint conversion framework, and run E2E verification via `forward_pass_logit_checker.py`.

### [Phase 5: Bidirectional Conversion (Optional)](file:///usr/local/google/home/yixuannwang/projects/maxtext/src/maxtext/experimental/agent/ckpt_conversion_agent/checkpoint_convetsion_skill/references/phase5_bidirectional.md)
**Objective**: Confirm the operational integrity of the bidirectional conversion process (from MaxText back to Hugging Face).
