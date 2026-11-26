# MaxText Review Focus Area: Test Quality and Runtime Efficiency

<!--
Shared domain checklist for MaxText authoring and code review. Referenced from
AGENTS.md, the GitHub PR review (.gemini/commands/maxtext-review.toml), and an
internal Code Review Agent skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `tests/` (unit, integration,
post-training, and test utilities).

MaxText runs thousands of CPU unit tests on every PR (`cpu-unit` with 4 virtual
CPU devices under `pytest-xdist`). Expensive tests quickly dominate PR latency
and runner capacity when they compile full-scale models, execute NNX layers
eagerly on CPU, run Pallas kernels in CPU interpret mode, or serialize checks
in subprocesses. Keep a finding only if you can name the specific test, config
or execution pattern causing unnecessary runtime or missing coverage, and
provide a concrete fix that preserves tolerances and positive/negative controls.

## What to check

1.  **Oversized model configs and topologies on CPU.** Flag CPU unit tests (and
    TPU AOT compile tests in `tests/unit/train_compile_test.py`) that inherit
    production or large default dimensions (`emb_dim`, `mlp_dim`, `num_heads`,
    `vocab_size`, `max_target_length`, `num_experts`, `base_num_decoder_layers`,
    or multi-hundred-chip topologies) when a minimal override exercises the exact
    same code path. Keep only the minimum dimensions and topology needed for
    every mesh axis under test to be `> 1` and every structural case (e.g.,
    1-2 scanned blocks plus remainder, or minimal head/expert divisibility) to
    be covered. Reserve full-size configs for tests that genuinely validate
    full-scale HBM fit or offload placement, and mark those
    `@pytest.mark.scheduled_only` when a small PR-time test covers the same
    architecture.
2.  **Eager op-by-op execution on CPU.** Flag unjitted model construction
    (`model_creation_utils.from_config`, `create_model`, `models.Transformer`,
    `NNXDecoder`), forward passes, loss/grad calls, or autoregressive decode
    loops running eagerly on CPU. Recommend wrapping model init and execution in
    `nnx.jit` / `jax.jit` (and `jax.value_and_grad` instead of separate forward
    and backward calls).
3.  **Materializing weights or executing when only HLO/jaxpr/shapes are
    checked.** Flag tests that initialize real weights or run forward/backward
    passes only to inspect `jax.make_jaxpr`, `.lower().as_text()`, sharding
    specs, or config/shape validation errors. Recommend building models
    abstractly via `nnx.eval_shape` / `jax.eval_shape` and lowering with
    `jax.ShapeDtypeStruct` inputs.
4.  **Duplicate traces, lowers, and compiles.** Flag tests that call
    `jax.make_jaxpr`, `.lower().as_text()`, `.compile()`, and a separate
    `jit(...)` execution on the same function, or run redundant identical
    baselines for positive controls. Recommend tracing once (`.trace()`),
    lowering once (`.lower()`), compiling once (`.compile()`), and reusing the
    compiled executable and baseline results across assertions.
5.  **Accidental Pallas / Megablox CPU interpret mode.** Model YAMLs often set
    `hardware: tpu` and `megablox: true`, causing `RoutedMoE` on CPU to execute
    the Pallas GMM kernel in interpret mode (10-30x slower than
    `jax.lax.ragged_dot`). Flag CPU MoE tests whose purpose is not testing the
    Megablox/Tokamax kernel itself when they omit `hardware="cpu"` (or
    `megablox=False`). Do **not** flag tests that intentionally test Megablox
    or Tokamax kernel logic.
6.  **Serial multi-device subprocesses blocking xdist parallelism.** Flag new
    tests that pack multiple independent multi-device (`>4` CPU devices) checks
    into a single serial `subprocess.run` child, or spawn a subprocess when 4
    CPU devices already suffice (which `cpu-unit` provides in-process).
    Recommend `tests/utils/cpu_mesh_child.py` to split checks into parameterized
    phases that run in parallel across `pytest-xdist` workers and run
    in-process when the worker already matches the requested device count.
7.  **Shared fixture caching, xdist grouping, and process-global state.**
    `tests/conftest.py` automatically assigns `xdist_group` to test classes with
    `setUpClass` or class-scoped fixtures, and CPU CI keeps `xdist_group`
    members on one shard and worker. Flag classes that share expensive
    `@functools.lru_cache` or module-level fixtures across methods without
    `@pytest.mark.xdist_group("<name>")` (or without `setUpClass`), since CI
    sharding will otherwise rebuild the fixture on multiple shards. Flag tests
    that enable a process-global persistent JAX compilation cache
    (`jax_compilation_cache_dir`) or mutate global `jax.config` / env vars
    without restoring them, as they leak into subsequent tests on the same
    xdist worker.
8.  **AOT compile fast-path overrides (`train_compile_test.py`).**
    `TrainCompile._aot()` applies fast-compile XLA flags by default and skips
    writing compiled executable pickles to disk unless `save=True`. Flag new
    `TrainCompile` tests that pass `full_opt=True` without testing full-scale
    HBM fit or offloading, pass `save=True` outside serialization tests, or
    bypass `_aot()`.
9.  **Redundant sweeps and `scheduled_only` discipline.** Flag slow Cartesian
    sweeps or duplicate tests where an existing PR-time test already covers the
    same code path. Keep at least one fast PR-time test per model family,
    feature, and failure mode; never loosen numerical tolerances or drop
    positive/negative controls to save time. When marking a slow duplicate or
    full-size variant `@pytest.mark.scheduled_only`, require a short comment
    naming the PR-time test that covers the same path, and place the decorator
    directly above `def` (below `@parameterized.*` decorators).
10. **Spec-level correctness and CI collectibility.** Consistent with
    `AGENTS.md` Section 3, flag tests that cannot fail (vacuous assertions,
    empty loops, self-comparisons), layout/sharding/kernel parity tests that
    omit gradient comparisons, bare `assertRaises(ValueError)` around sharded
    code (use `assertRaisesRegex`), CPU emulator tests masquerading as
    `tpu_only` real-kernel coverage, and files ignored by `pytest.ini` or
    misnamed outside `*_test.py` / `*_tests.py`.
