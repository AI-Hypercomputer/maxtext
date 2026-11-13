# MaxText AI Agent Guidelines

These rules apply across all MaxText code, scripts, configs, and tests.

---

## 1. Core Engineering Philosophy: "One Well-Lit Path"

MaxText prioritizes high-performance, maintainable JAX code with zero hidden
fallback paths:

*   **Single Source of Truth (SSOT) & No Duplicate Paths**: There must be ONE
    canonical implementation for any behavior. Do not duplicate existing logic
    across models, trainers, or utilities, and do not leave legacy wrapper
    functions or pass-through shims when refactoring — update callers directly.
*   **Strict Config Access (Fail-Fast)**: Config fields belong in
    `src/maxtext/configs/types.py` and `src/maxtext/configs/base.yml`. Read
    config attributes directly (`config.foo`, not `getattr(config, "foo",
    default)` or `dict.get(...)` fallback defaults). Silent fallback defaults
    mask typos, hide unmigrated configs, and invalidate tests.
*   **Fail-Fast Over Defensive Masking**: Never add blanket `try/except
    Exception` blocks, silent fallbacks, or counter clamps (`max(0, ...)` when
    negatives indicate a bug). Raise explicit errors with context.
*   **Clean Downward Dependencies & Specific Typing**: Avoid `Any`, untyped
    `dict` bags, or lazy-import hacks to disguise upward/circular dependencies.
    Place shared types and utilities cleanly in lower layers (`configs/`,
    `common/`, `utils/`).
*   **Production Code > Tests**: Never introduce production shims, `getattr`
    compatibility hooks, or wrapper helpers solely to appease test mocks. Tests
    must adapt to the real production contract (prefer realistic fakes or typed
    config fixtures).

---

## 2. Change Scope, Clarity, and Naming Semantics

Keep every change focused, flat, and easy to review:

*   **Flat Control Flow**: Keep functions concise and readable; avoid gratuitous
    single-use wrapper helpers, deep pass-through call chains, and unnecessary
    nested closures.
*   **Naming Semantics**:
    *   `load_` / `read_`: I/O or checkpoint loading (explicit failure when
        missing).
    *   `get_`: cheap in-memory or dict lookup without silent side-effects.
    *   `resolve_`: precedence/override cascade (flags -> config -> defaults).
    *   `derive_` / `compute_`: pure calculation from inputs.
    *   `validate_`: checks invariants and raises loudly on violation.
    *   `ensure_`: idempotent resource/state preparation.
*   **No Redundant Tests**: Test the observable contract and edge/failure cases
    meaningfully rather than duplicating trivial wrapper tests or mock-only
    assertions ("test theater").

---

## 3. Project Skills & Self-Review

MaxText ships repo-local skills under `.agents/skills/`. Use them proactively:

*   **Local Self-Review**: Before uploading or requesting review, invoke
    `/maxtext-self-review` (`.agents/skills/maxtext-self-review/SKILL.md`). It
    applies the canonical spec-driven 7-pillar review criteria in
    `.gemini/commands/maxtext-review-criteria.md` and the path-specific focus
    guides (`.gemini/commands/maxtext-review-*.md`).
*   **Model Onboarding**:
    *   Use `model-bringup` (`.agents/skills/model-bringup/SKILL.md`) for Flax
        Linen models.
    *   Use `model-bringup-m3` (`.agents/skills/model-bringup-m3/SKILL.md`) for
        pure Flax NNX `m3` models (`src/maxtext/m3/`).
*   **Goodput / Monitoring**: Use `maxtext-goodput`
    (`.agents/skills/maxtext-goodput/SKILL.md`) when configuring or debugging ML
    Goodput measurement.
*   **PR Upload & CI**: Use `maxtext-pr-upload`
    (`.agents/skills/maxtext-pr-upload/SKILL.md`) for formatting/linting
    recipes, single-commit branch hygiene, and CI status triage.

---

## 4. Repo-Specific Conventions

*   **License Header**: Every `.py` file must include the Apache 2.0 header.
*   **Formatting**: 2-space indentation and max line length 122 (`pyink` and
    `pylintrc`).
*   **Test Discovery & Serial Execution**: Only `*_test.py` and `*_tests.py`
    files are collected by `pytest.ini`; run JAX/CPU unit tests serially (no
    `-n` flags) with `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
*   **Note on `.agents/` in Git**: `.agents/` is listed in `.gitignore`, so use
    `git add -f .agents/...` when adding new files under `.agents/`.
