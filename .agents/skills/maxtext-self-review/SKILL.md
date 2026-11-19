---
name: maxtext-self-review
description: >-
  Runs MaxText's adversarial, spec-driven 7-pillar senior-engineer review locally
  against an uncommitted or branch diff before submitting for review. Shares the
  exact criteria in .gemini/commands/maxtext-review-criteria.md and
  maxtext-review-*.md used by /maxtext-review. Use when asking for a local
  self-review or checking review readiness.
---

# MaxText Local Self-Review

Perform a read-only, adversarial senior-engineer review of the target change
using MaxText's canonical review criteria.

## Step 1: Collect Context & Diff (Read-Only)

Gather the diff and change description for the work under review (e.g. local
uncommitted diff or `git diff origin/main...HEAD` and commit message / PR
description).

Read full modified files (around changed hunks and callers) before forming
conclusions.

## Step 2: Load Canonical Review Criteria & Domain Focus Guides

Always read `.gemini/commands/maxtext-review-criteria.md` in full first and
execute its 7-pillar analysis + strict output budget.

In addition, check every path-specific focus guide matching the changed files
(relative to the MaxText repo root):
- `.gemini/commands/maxtext-review-workflows.md` when touching `.github/`
- `.gemini/commands/maxtext-review-config-checkpoint.md` when touching
  `src/maxtext/configs/`, `pyconfig.py`, `configs/types.py`,
  `checkpointing.py`, `checkpoints/`, or checkpoint conversion/export code
- `.gemini/commands/maxtext-review-kernels.md` when touching
  `src/maxtext/kernels/`
- `.gemini/commands/maxtext-review-performance.md` when touching
  `src/maxtext/layers/`, `models/`, `m3/`, `trainers/`, `training_engine/`,
  `optimizers/`, `inference/`, or `multimodal/`

Also verify compliance with root `AGENTS.md` ("One Well-Lit Path", strict typed
config access without `getattr` fallback defaults, no mock-driven production
shims, fail-fast error handling).

## Step 3: Output Format

Present the review concisely in chat following the Step 2 summary and comment
structure in `.gemini/commands/maxtext-review-criteria.md`:
1. **Verdict** (one line)
2. **Top issues** (at most 5, tagged `[Description]`, `[Bug]`, `[Spec]`,
   `[Tests]`, `[Alignment]`, `[Complexity]`, `[Size]`, `[Perf]`, `[Security]`,
   each with `->` concrete action)
3. **Inline-style Findings** (at most 6 non-critical findings, `🔴`/`🟠`/`🟡`
   prefixed, citing exact `file:line`; unresolved `🔴` are blocking)
4. **Inferred Spec & Test Validation** (concise status for R1..Rn, C1..Cn)

Do NOT modify any files during a self-review unless explicitly asked to fix
findings afterward.
