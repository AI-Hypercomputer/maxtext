# MaxText AI Agent Guidelines

Shared engineering, JAX/sharding, and testing invariants for authoring and
reviewing MaxText changes.

## 1. Architecture, Configs, and Error Handling

* **Single Canonical Path:** Reuse established MaxText layers, sharding helpers
  (`maybe_shard_with_logical`, `sharding.constrain`), remat tags
  (`checkpoint_name`), and logging (`max_logging`) instead of adding duplicate
  utilities or hardcoded tunables. Fix issues at the owning layer (config
  validation, input pipeline, sharding rules, or shared layer) rather than
  patching callers, and update callers directly when refactoring instead of
  leaving single-use wrappers or pass-through shims. *(Exception: under
  `src/maxtext/m3/models/`, each model family intentionally defines
  self-contained attention, MLP, and decoder blocks rather than sharing
  cross-model layers; see `src/maxtext/m3/README.md`.)*
* **Typed Pydantic Configs (Fail-Fast):** Declare config keys in
  `src/maxtext/configs/` or `src/maxtext/m3/configs/` and read `config.<field>`
  directly — never use `getattr(config, "flag", default)` or `dict.get(...)`
  fallback defaults (unless indexing dynamic attribute names), redundant
  `isinstance`/`None` checks on guaranteed types, blanket `try/except Exception`
  blocks, or silent fallbacks. Release files, GCS handles, profiler sessions,
  and background threads via context managers or `try/finally`.
* **Clean Dependencies & Cross-File Contracts:** Use concrete types rather than
  `Any`, untyped `dict` bags, or lazy-import hacks that disguise upward or
  circular dependencies. When changing signatures, config keys, or parameter
  shapes/ranks/names, update all consumers: checkpoint conversion, LoRA and
  quantization rules, and optimizer metadata such as Muon dimension numbers
  (`muon_utils.get_muon_weight_dimension_numbers` and
  `maxtext_muon_utils.get_maxtext_muon_weight_dimension_numbers`). Declare new
  third-party imports in requirements files (`requirements_sync_test.py`).
* **Focused Scope & High-Signal Comments:** Keep changes small and
  single-concern (~400-500 net logic lines max; split preparatory refactors from
  features) without unused speculative parameters. Comment non-obvious *why*
  (`stop_gradient`/custom VJPs, `with_sharding_constraint`, donated buffers,
  `remat` policies, compiler workarounds), not line-by-line code narration.

## 2. JAX, Distributed Execution, and Performance

* **JIT Purity & Multi-Host Safety:** Avoid Python branching on traced arrays,
  side effects inside `jit`, unintended recompilation from missing/wrong static
  arguments, host-device syncs in training or decode loops, materialized or
  replicated large intermediates instead of sharded ones, extra collectives,
  Python loops where `jax.lax.scan` applies, multi-host races (e.g., every host
  writing the same file), or unsynchronized state shared with async
  checkpointing or data-loading threads.

## 3. Testing Standards

* **Spec-Level, Collectible Tests:** Name test files `*_test.py` or `*_tests.py`
  (`pytest.ini` ignores `test_*.py`) with `test*` methods. Assert observable
  end-to-end outcomes and regression cases that fail if the requirement is unmet
  — never add production shims for `SimpleNamespace`/`MagicMock`, assert only
  `assert_called` / `is not None` / shapes / "runs without crashing", or test
  only disabled/single-device/prefill-only paths when sharding or decode is in
  scope.
* **Execution Environment & JAX Pitfalls:** CPU CI runs with 4 fake devices
  (`--xla_force_host_platform_device_count=4`); tests skipping below >4 devices
  or marked `tpu_only`/`gpu_only` are skipped on CPU, and CPU op emulators
  (e.g., for `jax.lax.ragged_all_to_all`) do not replace a `tpu_only` test of
  the real op. Use `assertRaisesRegex` rather than bare
  `assertRaises(ValueError)` around sharded code, compare gradients as well as
  outputs in layout/sharding/kernel parity tests, and match tuple-shaped
  collectives in HLO regexes (e.g.,
  `r"\b(all-gather|all-reduce)(?:-start)?\("`).

* **CPU Test Efficiency:** Keep CPU unit tests fast by using minimal model
  dimensions/topologies, jitting model init and execution (`nnx.jit`/`jax.jit`),
  using `eval_shape` for HLO/jaxpr-only checks, setting `hardware="cpu"` (or
  `megablox=False`) in non-kernel MoE tests, and splitting `>4`-device CPU-mesh
  subprocesses via `tests/utils/cpu_mesh_child.py` (see `.agents/review/testing.md`).

## 4. Formatting, Domain Guides, and Workflows

* **Style & License:** Include the Apache 2.0 license header on `.py` files and
  use 2-space indentation with max line length 122 (`pyink`, `pylintrc`).
* **Domain Guides, Skills & Self-Review:** Follow matching checklists in
  `.agents/review/` (`workflows.md`, `config-checkpoint.md`, `kernels.md`,
  `performance.md`, `testing.md`), check `.agents/skills/` for task workflows (see
  `.agents/README.md`), and run a read-only self-review against
  `.agents/review/protocol.md` before requesting review.
