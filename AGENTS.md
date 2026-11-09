# MaxText AI Agent Guidelines

Follow `.gemini/commands/maxtext-review-criteria.md` (and any matching
`.gemini/commands/maxtext-review-*.md` focus guide) as the single source of
truth for architecture, JAX/sharding safety, simplification, and test quality.

* **Typed Configs & Fail-Fast Code:** Declare config keys in
  `src/maxtext/configs/` or `src/maxtext/m3/configs/` and read `config.<field>`
  directly — no `getattr`/`dict.get` fallback defaults, silent `try/except`
  masking, or production shims for test mocks.
* **Formatting & Test Discovery:** Include the Apache 2.0 license header on
  `.py` files, use 2-space indentation with max line length 122 (`pyink`,
  `pylintrc`), and name test files `*_test.py` or `*_tests.py` (`pytest.ini`
  ignores `test_*.py`).
* **Repo Skills:** Check `.agents/skills/` for task-specific MaxText workflows
  (see `.agents/README.md`).
* **Self-Review:** Run a read-only self-review of your diff and description
  against the review criteria above before requesting review.
