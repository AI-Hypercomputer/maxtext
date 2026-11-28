# MaxText Review Focus Area: Test Quality and Runtime Efficiency

<!--
Shared domain checklist for MaxText authoring and code review. Referenced from
AGENTS.md, the GitHub PR review (.gemini/commands/maxtext-review.toml), and an
internal Code Review Agent skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `tests/` (unit, integration,
post-training, and test utilities).

MaxText runs thousands of CPU unit tests (`cpu-unit`, 4 virtual CPU devices
under `pytest-xdist`) and hundreds of TPU unit tests (`tpu-unit` / `tpu7x-unit`,
class-sharded via `tests/utils/class_split.py`) on every PR. Expensive tests
quickly dominate PR latency and runner capacity when they compile full-scale or
128-expert models, execute NNX layers or discarded `ToLinen.init` forwards
eagerly, recompute shared parity references, run Pallas kernels in CPU interpret
mode, mark pure-CPU tests `tpu_only`, pack dozens of slow TPU tests into one
class, or serialize checks in subprocesses. Keep a finding only if you can name
the specific test, config, or execution pattern causing unnecessary runtime or
missing coverage, and provide a concrete fix that preserves tolerances and
positive/negative controls.

## What to check

1.  **Oversized model configs and topologies on CPU and TPU.** Flag unit tests
    (CPU, TPU, and AOT compile tests in `tests/unit/train_compile_test.py`) that
    inherit production or large default dimensions (`emb_dim`, `mlp_dim`,
    `num_heads`, `vocab_size`, `max_target_length`, `num_experts`,
    `num_experts_per_tok`, `base_num_decoder_layers`, or multi-hundred-chip
    topologies) when a minimal override exercises the exact same code path (for
    example, inheriting `model_name="qwen3-30b-a3b"` without overriding
    `num_experts=128` compiles 128-expert GMM kernels instead of 8, rounded up
    to a multiple of `jax.device_count()` when sharded on `fsdp`). Keep only the
    minimum dimensions and topology needed for every mesh axis under test to be
    `> 1` and every structural case (e.g., 1-2 scanned blocks plus remainder, or
    minimal head/expert/tile divisibility) to be covered. Reserve full-size
    configs for tests that genuinely validate full-scale HBM fit, shape parity,
    or offload placement, and mark slow full-size variants
    `@pytest.mark.scheduled_only` when a small PR-time test covers the same
    architecture.
2.  **Eager op-by-op execution and discarded `ToLinen.init` forwards.** Flag
    unjitted model construction (`model_creation_utils.from_config`,
    `create_model`, `models.Transformer`, `NNXDecoder`), forward passes,
    loss/grad calls (`jax.grad` / `nnx.value_and_grad`), or autoregressive
    decode loops running eagerly on CPU or TPU. Recommend wrapping execution in
    `nnx.jit` / `jax.jit` (or `nnx.split` + `jax.jit` + `nnx.merge` when calling
    cached modules without mutating them) and merging separate forward and
    backward passes via `value_and_grad(..., has_aux=True)`. For
    `ToLinen`-wrapped NNX modules (`linen_wrappers.to_linen`), flag
    `model.init(rngs, inputs)` calls whose forward output is discarded:
    `nnx_wrappers.ToLinen.__call__` saves all NNX variables before invoking
    `nnx_method`, so eager `model.init` compiles and runs the entire Pallas/MoE
    forward pass on TPU for nothing; recommend
    `linen_wrappers.init_variables(model, rngs)` instead (except when the
    forward pass lazily creates collections such as static-scale `quant_stats`).
    Do **not** force whole-module JIT where bfloat16 XLA fusion perturbs
    discrete boundary ops (such as `use_indexer=True` `top_k` ties) or tight
    stateful cache parity checks.
3.  **Materializing weights or executing when only HLO/jaxpr/shapes are
    checked.** Flag tests that initialize real weights or run forward/backward
    passes only to inspect `jax.make_jaxpr`, `.lower().as_text()`, sharding
    specs, dtypes, or config/shape validation errors. Recommend building models
    abstractly via `nnx.eval_shape` / `jax.eval_shape` and lowering with
    `jax.ShapeDtypeStruct` inputs.
4.  **Duplicate traces, lowers, compiles, and recomputed references.** Flag
    tests that call `jax.make_jaxpr`, `.lower().as_text()`, `.compile()`, and a
    separate `jit(...)` execution on the same function, run redundant identical
    baselines for positive controls, or recompute identical single-device
    `dot_product` reference outputs/gradients across every parameterized TPU
    context-parallel or sharding case. Recommend tracing once (`.trace()`),
    lowering once (`.lower()`), compiling once (`.compile()`), and caching
    shared reference outputs/gradients per class (cleared in `tearDownClass`,
    with complete cache keys covering all varied config fields).
5.  **Accidental Pallas / Megablox CPU interpret mode.** Model YAMLs often set
    `hardware: tpu` and `megablox: true`, causing `RoutedMoE` on CPU to execute
    the Pallas GMM kernel in interpret mode (10-30x slower than
    `jax.lax.ragged_dot`). Flag CPU MoE tests (including tests moved off
    `tpu_only`) whose purpose is not testing the Megablox/Tokamax kernel itself
    when they omit `megablox=False` (or `hardware="cpu"`). Do **not** flag tests
    that intentionally test Megablox or Tokamax kernel logic.
6.  **Serial multi-device subprocesses blocking xdist parallelism.** Flag new
    tests that pack multiple independent multi-device (`>4` CPU devices) checks
    into a single serial `subprocess.run` child, or spawn a subprocess when 4
    CPU devices already suffice (which `cpu-unit` provides in-process).
    Recommend `tests/utils/cpu_mesh_child.py` to split checks into parameterized
    phases that run in parallel across `pytest-xdist` workers and run
    in-process when the worker already matches the requested device count.
7.  **Shared fixture caching, xdist grouping, TPU shard balance, and
    process-global state.** `tests/conftest.py` automatically assigns
    `xdist_group` to test classes with custom `setUpClass` / `tearDownClass`
    (`setup_class` / `teardown_class`) or test-defined class-scoped fixtures,
    and CPU CI keeps `xdist_group` members on one shard and worker. Flag
    classes that share expensive `@functools.lru_cache` or module-level fixtures
    across methods without `setUpClass`/`tearDownClass` or
    `@pytest.mark.xdist_group("<name>")`, since CI sharding will otherwise
    rebuild the fixture on multiple shards. On TPU CI, `class_split.py` keeps
    each class on one shard, so flag monolithic `tpu_only` classes that bundle
    dozens of unrelated slow tests into a single shard bottleneck (recommend
    splitting independent feature groups into separate classes). Also flag tests
    that enable a process-global persistent JAX compilation cache
    (`jax_compilation_cache_dir`), override `conftest.py`'s fast TPU effort
    defaults (`jax_optimization_level="O0"`, `jax_memory_fitting_level="O1"`)
    back to `"O2"` without testing async scheduling/MSA/remat, or mutate global
    `jax.config` / env vars without restoring them (note that `conftest.py`
    saves/restores `jax_default_prng_impl`, `jax_use_shardy_partitioner`,
    `jax_optimization_level`, `jax_memory_fitting_level`, and
    `LIBTPU_INIT_ARGS` around each test and class in pytest).
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
10. **Unnecessary `tpu_only` markers, CPU-covered TPU duplicates, and legacy
    Linen duplicates.** Flag `@pytest.mark.tpu_only` or `tpu_backend` on tests
    that only run `attention="dot_product"`, pure JAX/Flax math, or `<=4`-device
    host sharding without exercising real TPU Pallas/Mosaic/Tokamax kernels or
    TPU-only collectives — they belong on `cpu-unit` (set `megablox=False` for
    non-kernel MoE tests and `skip_jax_distributed_system=True` if removing
    `import pytest` from tests that also run outside pytest). Flag TPU test
    subclasses that re-run pure-dense or router cases already covered on CPU
    (restrict TPU subclasses to compiled kernel cases). Flag new tests using
    deprecated `models.transformer_as_linen` or adding duplicate `*_nnx` test
    methods when the base test already exercises native NNX
    (`models.Transformer(..., rngs=...)`).
11. **Spec-level correctness and CI collectibility.** Consistent with
    `AGENTS.md` Section 3, flag tests that cannot fail (vacuous assertions,
    empty loops, self-comparisons), layout/sharding/kernel parity tests that
    omit gradient comparisons, bare `assertRaises(ValueError)` around sharded
    code (use `assertRaisesRegex`), CPU emulator tests masquerading as
    `tpu_only` real-kernel coverage, mocks on shared cached fixtures that leak
    instance attributes across test orderings (prefer `patch.object` or
    `delattr` cleanup), and files ignored by `pytest.ini` or misnamed outside
    `*_test.py` / `*_tests.py`.
