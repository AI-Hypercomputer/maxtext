---
name: maxtext-wiki-distill
description: >-
  Automatically harvests, triages, distills, and prunes reusable engineering
  lessons from a MaxText model bring-up branch, `{model_name}_handover.md`, or
  pull request into `.agents/wiki/`. Use after Phase 4 bring-up verification
  passes, during PR cleanup, when a user asks to distill a branch/session/PR
  into the wiki, or when auditing `.agents/wiki/` for stale or graduated pages.
  Don't use mid-debugging before a workaround has verified empirical evidence.
---

# MaxText Wiki Distillation & Self-Improvement Workflow

Turns raw, episodic model bring-up artifacts (`git diff`, fix commits, `{model_name}_handover.md`, and reviewer comments) into clean, deduplicated, repo-grounded knowledge in [`.agents/wiki/`](../../wiki/index.md)—while actively compacting and pruning stale workarounds.

> [!IMPORTANT]
> **Run in a Clean Post-Verification Pass**:
> During active debugging (Phases 1–4), log raw bugs and before/after metrics inline into `{model_name}_handover.md` Section 4 only. Execute this skill **after** verification passes (or in a fresh agent session) so unverified hypotheses never pollute `.agents/wiki/`.

---

## Step 1: Harvest Objective Candidate Signals

Run the zero-dependency harvester from the MaxText checkout root to inspect `merge-base` shared-infrastructure edits, added `base.yml` keys, fix commits, pending handover rows, and overlapping `.agents/wiki/` pages:

```bash
python3 .agents/wiki/scripts/harvest_candidates.py --base origin/main
```

- Inspecting another worktree/branch: pass `--repo-root /path/to/checkout` (and `--handover <file.md>` if named customly).
- Distilling an open PR with review threads: also run `python3 .agents/skills/maxtext-pr-upload/scripts/pr_comments.py <pr_number> --full` to capture reviewer-flagged CI or sharding traps.

---

## Step 2: Inspect Targeted Evidence & Protocol

Do **not** read the entire repository diff. Read only:
1. [`.agents/wiki/PROTOCOL.md`](../../wiki/PROTOCOL.md) and [`.agents/wiki/index.md`](../../wiki/index.md).
2. Every page listed under **`## 1. Overlapping Existing Wiki Pages`** in the harvester brief.
3. Focused diffs on **`## 3. Shared MaxText Infrastructure Files Modified`** (`git diff $(git merge-base origin/main HEAD) -- <shared_file>`) and any root verification reports listed in **`## 6`**.

---

## Step 3: Build the 4-Gate Triage Table First

Before editing any file in `.agents/wiki/`, classify every harvested candidate into one of five dispositions using the [4-Gate Promotion Rubric](../../wiki/PROTOCOL.md#2-the-4-gate-promotion-rubric-write-back-gate) (*Reusability*, *Verified Evidence*, *CI/Default Safety*, *Accumulate & Sanitize*):

| Candidate | Gate Check | Disposition | Action |
| :--- | :--- | :--- | :--- |
| Model-specific typo or dimension mismatch | Fails Gate 1 (single-model) | `local-only` (**DROP**) | Leave in `{model_name}_handover.md` only |
| Workaround without measured pass/metric | Fails Gate 2 (unverified) | `local-only` (**DROP**) | Leave out until verified |
| Rule already in `SKILL.md` / `ci_mechanics.md` | Duplicate of core skill | `local-only` (**DROP**) | Do not duplicate in wiki |
| New flag/caveat/hardware data on existing topic | Passes Gates 1–4 + overlaps wiki page | `merged:<category>/<page>.md` (**MERGE**) | Update existing page + append `verified_on` |
| Distinct cross-model failure class or recipe | Passes Gates 1–4 + zero topic overlap | `new:<category>/<slug>.md` (**NEW**) | Create 1 atomic page (40–110 lines) + register in `index.md` |
| Existing wiki hack now upstreamed into `src/maxtext/` | Upstreamed / obsolete | `graduated` or **DELETE** | Shrink page to document the new flag, mark `status: graduated`, or `git rm` |

Also check **Negative Feedback**: if following an existing wiki page during this bring-up *failed* or required hardware/stack scoping (e.g., Linen vs M3, TPU v5p vs v6e/tpu7x), schedule a **MERGE/FIX** edit on that page immediately.

---

## Step 4: Author Compaction-Aware Wiki Updates

1. **When Merging (`merged:<page>`)**:
   - Integrate the new flag, stage, or caveat cleanly into the existing structure.
   - Append `"<Model Name> (<Hardware>, PR #<N>)"` to `verified_on` and update `last_verified: YYYY-MM-DD`.
   - **Compaction Rule ($\ge 120$ lines)**: Never let a page grow into an append-only changelog. If adding content pushes a page past `120` lines (hard cap `140`), generalize prose and collapse per-model anecdotes into a compact Markdown table row.
2. **When Creating (`new:<category>/<slug>.md`)**:
   - Place under `troubleshooting/`, `parity/`, `patterns/`, or `conversion/` using the exact frontmatter and required `##` headings in [`PROTOCOL.md`](../../wiki/PROTOCOL.md#4-category-page-templates-required--sections).
   - Register a 1-line row in [`.agents/wiki/index.md`](../../wiki/index.md) under `## Topic Index` (and link in `## Model Archetype & Reference Precedents` if archetype-specific).
3. **Update `{model_name}_handover.md`**:
   - Fill the `Wiki Disposition` column in Section 4 (`local-only`, `merged:...`, or `new:...`) so re-running `harvest_candidates.py` reports `0` pending handover items.

---

## Step 5: Validate, Audit Drift & Stage Clean PR

Run structural validation, git code-drift audit, and spellcheck from the repository root:

```bash
python3 .agents/wiki/scripts/validate_wiki.py --check-drift
codespell -L ND,nd,sems,TE,te,als,ROUGE,rouge,astroid,ags,dout .agents/wiki
```

- Fix all errors (broken relative links, missing headings, line-budget violations, or stale `code_refs` / `config_keys`) and review any `ADVISORY:` drift notices.
- Stage explicitly (`git add -f .agents/wiki/...`) and ship as a standalone 1-commit PR against `origin/main` per [`maxtext-pr-upload`](../maxtext-pr-upload/SKILL.md), pasting the **Step 3 Triage Table** into `# Description` so reviewers see what was merged, created, and dropped at a glance.
