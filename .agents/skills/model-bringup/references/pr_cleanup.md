# Large PR Cleanup & Atomic Split

**Opt-in only:** `pr_cleanup=false` unless explicitly enabled by the user. Finish Phase 4 and record its evidence in `{model_name}_handover.md` before cleanup.

## 1. Clean the validated bring-up diff

Inspect the cumulative diff against the intended PR base, including earlier commits and uncommitted bring-up work. Preserve unrelated user changes; record the validated revision and reproduction commands before editing.

- Remove obsolete comparison/inspection scripts, activation dumps, one-off conversion helpers, and local fixtures after checking callers. Keep maintained tests, reusable validation tools, and fixtures needed to reproduce correctness.
- Before removing tools, preserve evidence/reproduction instructions in the handover or project docs; update commands referencing removed scripts.
- Remove unused debug prints/logging, breakpoints, commented-out experiments, stale comments/docstrings, pseudocode, and temporary shortcuts. Keep architectural explanations, public API docs, and maintained diagnostics.
- Replace hardcoded mini-model dimensions, bypassed checks, and temporary branches with validated configuration or supported implementations. Revalidate changed numerical behavior before proceeding.
- Format every added/modified file across the cumulative diff using current repository configuration and neighboring conventions. Avoid unrelated files; inspect formatter changes for behavior changes.

## 2. Resolve lint and revalidate

Read `.pre-commit-config.yaml`, `pylintrc`, and formatter configuration. Current Python style is Pyink, two-space indentation, 125-character lines; repository configuration is authoritative.

Run configured hooks in the intended dependency environment over all added/modified files. For Python, use `pre-commit run pyink --files ...` and `pre-commit run pylint --files ...`; run applicable non-Python hooks too.

Fix pylint findings and rerun until clean. Prefer code fixes; justify narrow suppressions and never weaken shared lint configuration. Report environment blockers and unrelated baseline findings separately with exact commands/results; blocked runs are not passes.

Rerun affected layer/unit tests, mapping tests, and config/shape checks; rerun logits and autoregressive decode checks if execution or conversion behavior changed. Record rerun results and which prior evidence still applies.

## 3. Choose atomic model-bringup layers

Split by dependency and review purpose, not file count/equal size. Each layer needs one coherent purpose, its own tests, and passing checks atop its predecessors. Adapt this example to the architecture:

| Order | Candidate PR | Contents and validation |
| :--- | :--- | :--- |
| 1 | Model scaffolding | Minimal architecture identifiers, registration plumbing, and shared interfaces; import/registry checks. Avoid advertising a usable model before its implementation exists. |
| 2 | Configuration addition | Model YAML, HF configuration metadata, and config validation. Place here only if it loads and passes checks without later layers; otherwise move after the blocks or combine with the implementation that makes it usable. |
| 3 | Basic blocks + unit tests | New attention, norms, RoPE, MLP/MoE, decoder, or multimodal blocks and integration, with weight-copy/allclose tests and cache/decode coverage. Split independent block families further when useful. |
| 4 | Full conversion parameter mappings | Central `PARAM_MAPPING`/`HOOK_FNS`, required conversion-framework extensions, and tests for transpose/reshape, scanned-layer indexing, tied weights, and supported conversion directions. Keep each mapping with the hook implementation and tests it needs. |
| 5 | End-to-end integration and documentation, if distinct | Remaining runnable integration, maintained logits/decode regression coverage, model support documentation, and reproduction instructions. Put documentation that belongs to an earlier layer there instead. |

Do not force five PRs: combine small scaffolding/config changes and reorder config after blocks if needed. Keep blocks and tests together. Shared framework changes may form a prerequisite with regression tests. Every intermediate state must avoid dangling imports, missing hooks, and configs selecting unimplemented decoders.

Before rearranging commits, write a stack manifest: each layer's title, purpose, files/hunks, predecessor, and validation commands/results. Assign shared-file hunks by dependency; cover every intended change exactly once.

## 4. Prepare the stack and hand off upload mechanics

Organize local atomic commits/branches from the manifest, preserving the validated source revision and unrelated work. Verify the cumulative stack diff matches the cleaned implementation without dropped mappings/tests/configs or extra changes. Check every layer and run final integration checks at the tip; tip-only success is insufficient.

Follow [maxtext-pr-upload](../../maxtext-pr-upload/SKILL.md) and its [stacked PR workflow](../../maxtext-pr-upload/references/stacked_prs.md), passing the manifest and validation evidence. Cleanup opt-in alone does not authorize PR publication or remote history rewrites.

Update the handover with removed artifacts, cleanup/lint/test results, layer order/dependencies, and blockers. Completion requires passing lint/relevant validation and preservation of the intended cumulative diff.
