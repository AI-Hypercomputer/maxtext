# MaxText Wiki Authoring & Distillation Protocol

Read this file **only** when distilling findings from a completed bring-up or editing pages under `.agents/wiki/`. Routine readers should load [`index.md`](index.md) and matching leaf pages only.

---

## 1. Three-Tier Knowledge Lifecycle

| Tier | Location | Scope & Lifespan | Promotion Rule |
| :--- | :--- | :--- | :--- |
| **1. Episodic** | `{model_name}_handover.md` (Section 4) | Single model bring-up working log; captures *all* bugs, metrics, and repro commands (including one-off typos). | At Phase 4 pass, evaluate every row against the **4-Gate Promotion Rubric** below. |
| **2. Semantic Wiki** | `.agents/wiki/<category>/<slug>.md` | Verified, reusable cross-model playbooks, parity diagnostics, and implementation/conversion recipes (25–140 lines). | Merge into an existing page if the problem class exists; create a new page only for a distinct topic. |
| **3. Core Workflow / Code** | `.agents/skills/*/SKILL.md` or `src/maxtext/` | Universal invariants needed on *every* run, or repeated workarounds promoted to first-class MaxText flags/helpers. | When a wiki workaround repeats across $\ge 3$ bring-ups, upstream a flag/helper or add a 1-line rule in `SKILL.md`. |

---

## 2. The 4-Gate Promotion Rubric (Write-Back Gate)

Promote an item from `{model_name}_handover.md` into `.agents/wiki/` **only if all four gates pass**:

1. **Cross-Model Reusability**: Applies to $\ge 2$ models, a broader architectural pattern (e.g., GQA, MLA, MoE, hybrid linear attention, VLM), or shared TPU/Orbax infrastructure—never a single-model config typo.
2. **Verified Empirical Proof & Negative-Feedback Fixes**: Backed by concrete before/after evidence (passing test command, `KL` / `allclose` improvement, or compiled/runtime HBM delta). Never write unverified hypotheses mid-debugging. Conversely, if following an existing wiki page *failed* or required hardware/stack scoping (e.g., Linen vs M3, v5p vs v6e/tpu7x), updating or scoping that page is mandatory at wrap-up.
3. **CI & Default-Path Safety**: Explicitly documents whether code changes touch shared paths (`train_step`, `loss_fn`, `eval_step`) and how [`tests/integration/hlo_diff_test.py`](../../tests/integration/hlo_diff_test.py), `_Cfg` in [`tests/unit/train_nnx_test.py`](../../tests/unit/train_nnx_test.py), and default MFU are protected (opt-in flag defaulting `False` or revert during PR cleanup).
4. **Accumulate, Compact & Sanitize**:
   - Run [`scripts/harvest_candidates.py`](scripts/harvest_candidates.py) (or follow [`maxtext-wiki-distill`](../skills/maxtext-wiki-distill/SKILL.md)) to inspect overlapping wiki pages, shared-infra diffs, and pending handover rows before writing prose.
   - Check [`index.md`](index.md) first: if a page covers the topic, **merge** the new stage/caveat into it and append your model to `verified_on`. When merging pushes a page past **120 lines**, compact verbose prose into Markdown table rows rather than growing unboundedly.
   - Use `snake_case.md` filenames, keep leaf pages within **25–140 lines** (`index.md` $\le$ **150 lines**), and scrub all machine-specific or personal paths (use `$USER` and `{model_name}` placeholders).
   - **Lifecycle & Pruning (`status`)**: Optional `status: active | graduated | superseded` (with `superseded_by: "PR #..."` when non-active) marks manual workarounds that have been upstreamed into `src/maxtext/` so they can be shrunk or deleted.

---

## 3. Repo-Grounded YAML Frontmatter Schema

Every leaf page under `.agents/wiki/{troubleshooting,parity,patterns,conversion}/` must begin with YAML frontmatter validated by [`scripts/validate_wiki.py`](scripts/validate_wiki.py):

```yaml
---
title: <Concise Descriptive Title>
category: troubleshooting                  # troubleshooting | parity | patterns | conversion
status: active                             # active (default) | graduated | superseded
phases: [phase4, benchmark]                # subset of: phase1, phase2, phase3, phase4, benchmark, pr_cleanup
applies_to: [linen, m3]                    # subset of: linen, m3
tags: [hbm, oom, moe, remat]               # lowercase search tags
code_refs:                                 # repo-relative file paths (must exist on disk)
  - src/maxtext/configs/base.yml
config_keys:                               # MaxText config keys (must exist in base.yml or types.py; [] if none)
  - param_scan_axis
verified_on:
  - "<Model Name> (<Hardware>, PR #<N>)"
last_verified: YYYY-MM-DD
---
```

---

## 4. Category Page Templates (Required `##` Sections)

Choose the template matching `category` (`validate_wiki.py` checks required `##` headings):

### A. `category: troubleshooting` (Resource, HBM, Compilation & Infra Playbooks)
- `## Symptom / When to Use` — Exact error strings, topology constraints, and bring-up phase.
- `## Action Ladder` — Ordered stages from *Stage 1: Zero-Code Config/CLI Flags* to *Stage 2/3: Code-Level Mitigations*, plus a quantitative `Reference Impact` table where applicable.
- `## Guardrails & Cleanup` — Host DRAM/disk limits and what must be flag-gated or reverted before PR upload.
- `## Related` — Annotated relative `.md` links to sibling wiki pages or skills.

### B. `category: parity` (Numerical Divergence & Logit/Gradient Diagnostics)
- `## Symptom & Divergence Signature` — Metric signature (e.g., `fp32` layer-0 mismatch vs long-context `KL` spike vs Router Replay pass / Unforced fail).
- `## Fast Isolation Probe` — Minimal CLI toggle or $\le 15$-line tensor/mask check confirming the root cause in minutes.
- `## Root Cause & Fix Pattern` — PyTorch/HF vs MaxText (`linen` / `m3`) discrepancy and minimal fix.
- `## Verification & Negative Control` — Exact assertion or `PerturbSpec` proving the fix is live.
- `## Related` — Annotated relative `.md` links.

### C. `category: patterns` or `conversion` (Architecture & Checkpoint Recipes)
- `## When You Need This` — HuggingFace `config.json` keys or `modeling_*.py` / safetensors structures triggering this pattern.
- `## Canonical MaxText Implementation` — Closest reference files/tests plus minimal Linen and/or NNX M3 idiom.
- `## Common Traps & Invariants` — `scan_layers=True/False` indexing, sharding/HLO constraints, and strict `fp32` / round-trip invariants.
- `## Related` — Annotated relative `.md` links.

---

## 5. Validation, Drift Audit & PR Workflow

1. **Harvest candidates, validate schema/links, audit git drift, and spellcheck** from the MaxText repository root:
   ```bash
   python3 .agents/wiki/scripts/harvest_candidates.py --base origin/main
   python3 .agents/wiki/scripts/validate_wiki.py --check-drift
   codespell -L ND,nd,sems,TE,te,als,ROUGE,rouge,astroid,ags,dout .agents/wiki
   ```
2. **Stage explicitly (`git add -f`)**: `.agents/` is listed in `.gitignore`, so pass `-f` when staging new files:
   ```bash
   git add -f .agents/wiki/index.md .agents/wiki/<category>/<page>.md
   ```
3. **Ship decoupled or stacked**: Reusable wiki playbooks can open as a standalone 1-commit PR against `origin/main` immediately (so concurrent bring-ups benefit without waiting on a large model stack), or ride in the final documentation PR of a stacked bring-up series ([`maxtext-pr-upload`](../skills/maxtext-pr-upload/SKILL.md)).
