# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. XPK Execution & Docker Image Knowledge

- **`--base-docker-image` vs `--docker-image` in XPK**:
  - When creating XPK workloads via scripts like `run_qwen3_80b_xpk.sh`, ensure that `--base-docker-image` or a fresh workspace-built runner image is passed so that our active local code modifications run instead of just the pre-baked baseline Docker image.
  - If a workload is launched using a static baseline image without bundling local code changes, the resulting profile/trace reflects the baseline behavior instead of the current branch's code. Always verify image builds or use `--base-docker-image` when testing local changes on GKE TPU clusters.

- **Cluster & Workload Deletion Safety**:
  - NEVER delete or terminate workloads, JobSets, pods, or jobs that were not created by you or this session.
  - NEVER run blanket deletion commands such as `kubectl delete jobset --all`, `kubectl delete workloads --all`, or `kubectl delete pods --all`.
  - ALWAYS target ONLY the specific workload/JobSet name created for your active run (e.g. `kubectl delete jobset <my-specific-workload-name>`), leaving other shared workloads and pods untouched.

- **Configuration & Script Integrity**:
  - NEVER change, override, or modify any configuration parameters, model flags, or command-line arguments in scripts or YAML config files (e.g., `run_qwen3_80b_xpk.sh`, `run_qwen3_80b_aot.sh`, model YAMLs) without explicit approval from the user.
