# Review: `shuningjin-emb` — `quantize_logits_proj`

Branch: https://github.com/AI-Hypercomputer/maxtext/compare/shuningjin-emb
Base: `main` @ `2f4d42bca`
Reviewed: 2026-09-21

Diff: 8 files, +294/-4. Of those, 214 added lines are `sandbox/` scratch.

| File | Lines | Keep in PR? |
| --- | --- | --- |
| `docs/reference/core_concepts/quantization.md` | +14/-1 | yes |
| `src/maxtext/configs/base.yml` | +1 | yes |
| `src/maxtext/configs/types.py` | +23 | yes |
| `src/maxtext/layers/quantizations.py` | +9/-3 | yes |
| `tests/unit/quantizations_test.py` | +37 | yes |
| `sandbox/check_qwix_interception.py` | +62 | **no** |
| `sandbox/check_qwix_interception_log.txt` | +108 | **no** |
| `sandbox/plan.md` | +44 | **no** |

---

## Blocking

### B1. `sandbox/` is committed

Three scratch files are tracked and not gitignored (`git check-ignore` returns nothing).
`sandbox/plan.md` itself states "do not commit change". Drop them from the branch, or add
`sandbox/` to `.gitignore`.

### B2. Commit history

Messages are `init`, `update`, `.`, `update`. Squash into a single descriptive commit
before opening the PR.

---

## Correctness and semantics

### C1. MTP logits are silently quantized by this flag

MTP has no output head of its own: `multi_token_prediction.py:532` calls
`decoder.apply_output_head`, so the regex `decoder/logits_dense.*` intercepts both the
main-model and the MTP invocation of the same GEMM.

Consequence: with `quantize_mtp=False, quantize_logits_proj=True`, part of the MTP
forward is quantized despite the flag name saying otherwise.

This is defensible and arguably desirable (see `plan.md` §8.2), but the rationale
currently lives only in `sandbox/plan.md`. None of `base.yml:157`,
`types.py:558-565`, or `docs/.../quantization.md` mention it. Document it in the field
description at minimum.

### C2. Sparsity applies to `logits_dense` — DOCUMENTED, coupling kept

Original framing ("if unintended, emit a separate QtRule") was too quick. Checked the
qwix implementation; the substantive issue is narrower and different.

**Mechanism.** `dot_general_qt.py:386-387` calls `qarray.sparsify(rhs, rule)`, and
`qarray.py:255-277` is stateless: a magnitude mask is recomputed from the weight on every
step, then `jnp.where(mask, array, 0)`. Sparsity hits the `rhs` of the dot_general — for
`logits_dense`, that is the `[emb, vocab]` kernel.

**Correction to the original finding:** no checkpoint or state impact. Masks are not
variables on this path.

**Second finding, larger than this PR:** `weight_sparsity_start_step` and
`weight_sparsity_update_step` are honored only by
`qwix/contrib/sparsity/sparsity_module.py:84-88`, which MaxText instantiates **only** for
MoE expert weights (`moe.py:664-682`). Everything matched by the `QtRule` — the dense
layers, and now `logits_dense` — takes the stateless path, so those two config knobs are
inert there. Consequence: the vocab kernel is pruned from step 0 with no warmup, while
the MoE experts follow a schedule. Not introduced by this PR; worth a separate issue.

**Is it wrong for the head to be sparsified?** No strong first-principles reason. The
case for including it is symmetric with the case for quantizing it: 2:4 is a throughput
optimization, the head is the largest remaining dense GEMM, and excluding it caps the
gain. Hardware treats it like any other matmul.

The valid objections are narrower:

1. **Flag orthogonality.** A flag named `quantize_*` silently widens the scope of
   `weight_sparsity_n`/`weight_sparsity_m`. An fp8-head ablation on a sparse run would
   attribute part of the loss delta to fp8 when it is new pruning.
2. **No warmup**, per the finding above.
3. **Compounding** fp8 and 50% magnitude pruning on the tensor that directly produces
   logits. Untested composition. This is a "nobody measured it" claim, not evidence that
   it breaks.

**Resolution:** keep the coupling, do not add a separate rule or a `sparsify_logits_proj`
flag — no one runs sparsity + fp8 head today, and a third quantization-adjacent boolean
costs more than a sentence. Documented in the `types.py` field description. Revisit if a
sparse fp8 run actually needs the head quantized-but-dense.

### C3. No validation against `unquantized_modules`

`base.yml:191-195` accepts literal module names to keep unquantized, explicitly listing
`'logits_dense'`, and both shipped `qwen3.5-*-fp8.yml` configs include it.

`quantize_logits_proj=True` combined with `logits_dense` in `unquantized_modules` is a
contradiction (weights instantiated in `dtype`, matmul quantized by Qwix) that is
currently accepted silently. Either reject at config init or document which mechanism
wins.

### C4. `use_qwix_quantization` check — RESOLVED, original finding was wrong

Original claim: "the validator gates on `quantization == 'fp8_full'` but not on
`use_qwix_quantization`; with AQT the flag is a silent no-op." Verification showed this
is incorrect, and that adding the obvious check would have been a regression.

Three facts from source:

1. **AQT + `fp8_full` is already fatal, not silent.** `_get_quant_config`
   (`quantizations.py:715-737`) has no `fp8_full` branch and falls through to
   `raise ValueError(f"Invalid value configured for quantization {config.quantization}.")`.
   `configure_quantization:772` only short-circuits when `use_qwix_quantization` is
   True, so the AQT path reaches that raise. The existing `quantization == "fp8_full"`
   gate therefore already implies a Qwix path transitively.
2. **Requiring `use_qwix_quantization=True` would break batch-split.**
   `configure_quantization:761-768` returns `QwixQuantization(...)` for `fp8_full`
   under `use_batch_split_schedule=True` **regardless** of `use_qwix_quantization`. A
   naive `if not self.use_qwix_quantization: raise` would reject a working
   configuration.
3. **The real silent no-op is batch-split, not AQT.** `maybe_quantize_model:1061` gates
   on `... and not config.use_batch_split_schedule`, so batch-split never runs Qwix
   model interception. Its two consumers,
   `deepseek_batchsplit.py:2138` and `deepseek_batchsplit_fp8.py:963`, call
   `get_fp8_full_qwix_rule_w_sparsity(config)[-1]` and hand that single rule straight to
   a GMM kernel, where `module_path` is not consulted. `quantize_logits_proj=True` under
   batch-split would be accepted and do nothing.

**Fix applied:**

- `types.py` validator: reject `quantize_logits_proj` with
  `use_batch_split_schedule=True`, with the mechanism stated in the message.
- `types.py` field description, `base.yml:157`, `docs/.../quantization.md`: replaced the
  inaccurate "Applicable when `use_qwix_quantization=True`" claim with the actual
  conditions (`fp8_full` is Qwix-only because AQT rejects it; batch-split unsupported).
  This edit also fixes M1 and documents C1.
- Did **not** add a `use_qwix_quantization` check, for reason 2 above.

**Side observations, not addressed (pre-existing, out of scope for this PR):**

- `quantize_mtp` has the identical batch-split no-op and is not guarded.
- `quantize_router_proj=False` is also inert under batch-split: it prepends the
  gate-exclusion rule at index 0, and the batch-split consumers take `[-1]`, dropping it.
  Worth a follow-up issue; the `[-1]` indexing is fragile against any future change that
  appends rules.

**Verification status:** UNVERIFIED by execution. The local `.venv` lacks `maxtext` and
`pytest`, and `tests/__init__.py` imports `pathwaysutils`. Run on the TPU VM per §5 of
`plan.md`:

```
JAX_PLATFORMS=cpu pytest tests/unit/quantizations_test.py -k "LogitsProjQwixInterceptionTest"
```

plus a manual check that `quantize_logits_proj=true use_batch_split_schedule=true` now
raises at config init, and that `model_name=deepseek3-671b-batchsplit` with
`quantization=fp8_full` (flag left False) is unaffected.

### C5. `num_vocab_tiling > 1` — guarded, pending a decisive test

Upgraded from "untested" to a named mechanism after tracing the code path.

**Mechanism.** `maybe_quantize_model:1073` calls `qwix.quantize_model` with a dummy
forward and no `model_mode`, so it traces in train mode. Under `num_vocab_tiling > 1` in
train mode, `nnx_decoders.py:2285-2287` sets `logits = None`, sows the hidden states, and
never calls `apply_output_head`. `logits_dense` is therefore **absent from the graph Qwix
traces**. The projection instead executes later in `vocabulary_tiling.py:145-148`, via
`models.py:161 logits_from_hidden_states_for_vocab_tiling`, on an `nnx.merge`'d copy
outside that trace.

Same failure shape as C4: config accepted, rule potentially inert. Whether it actually
goes inert depends on whether Qwix interception is lazy at call time (works) or
materialized during tracing (silent no-op). UNRESOLVED.

**Twist.** With `mtp_num_layers > 0`, MTP calls `apply_output_head` inside the model call
(`multi_token_prediction.py:532`), so the head *is* traced. Behavior would therefore
differ by `mtp_num_layers` — worse than a uniform rejection.

**Fix applied:** reject `quantize_logits_proj` with `num_vocab_tiling > 1` at config
init, unconditionally on `mtp_num_layers`. Documented in the field description,
`base.yml:157`, and `docs/.../quantization.md`.

**Cost of the guard.** Vocab tiling pairs naturally with FP8 heads — both target
large-vocab logits memory/compute. `configs/tpu/v4/22b.sh:58` and `52b.sh:59` both run
`num_vocab_tiling=8`. Do not treat this guard as permanent.

**Decisive experiment to remove it** (harness already exists in
`tests/unit/tiling_test.py:132-150`):

1. Build a config via `_build_cfg_and_model` with `quantization=fp8_full`,
   `use_qwix_quantization=true`, `quantize_logits_proj=true`, `num_vocab_tiling=2`,
   `mtp_num_layers=0` (temporarily bypassing the new guard).
2. Call `vocab_tiling_nnx_loss` under `assertLogs(absl_logging.get_absl_logger(),
   level="DEBUG")`.
3. Grep the captured logs for `module='decoder/logits_dense'` with `op=dot_general`.
   - Intercepted with `rule=0` → drop the guard, keep the test.
   - Absent or `rule=None` → keep the guard, and the error message is already accurate.
4. Repeat with `mtp_num_layers=1` to confirm the twist above.

---

## Minor

### M1. Docstring regex does not match the code

`types.py:562` says the target is `r'.*decoder/logits_dense.*'`; the emitted path is
`decoder/logits_dense.*` (`quantizations.py:978`). A leading `.*` changes matching
semantics under fullmatch. Fix the string.

### M2. `base.yml` comment inconsistency

The `quantize_router_proj` line ends with "applicable when `use_qwix_quantization=true`
and quantization is set; ignored otherwise." The new `quantize_logits_proj` line omits
it. Mirror the wording.

### M3. Test asserts a hardcoded rule index

`quantizations_test.py:822-828` passes `expected_rule="rule=0"`, which holds only while
`quantize_router_proj` keeps its default `True` (no rule prepended). The sibling router
test derives the index instead.

Fix: add an explicit `quantize_router_proj=true` to the config list, or assert against
the returned rule list the way `RouterProjQwixInterceptionTest` does.

### M4. No test for the three new `ValueError` branches

`logits_via_embedding`, non-`fp8_full`, and `logits_dot_in_fp32` rejections are
untested. The repo has no precedent for config-validation tests
(`grep "can only be enabled when" tests/` is empty), so this is a suggestion rather than
a requirement — but the `logits_dot_in_fp32` branch encodes a non-obvious claim and is
worth pinning.

### M5. Doc diff mixes reflow with content

The `use_qwix_quantization` and `quantization_calibration_method` bullets are reflowed
into blank-line-separated blocks. Unrelated whitespace churn adjacent to the substantive
addition; makes the diff harder to read.

---

## Naming

Current name `quantize_logits_proj` is acceptable. `quantize_logits_dense` is marginally
better. `quantize_lm_head` should be avoided.

| Candidate | Assessment |
| --- | --- |
| `quantize_lm_head`, `quantize_output_head` | **Reject.** MaxText already uses "output head" for a *group*: `vocabulary_tiling.py:37` defines `_OUTPUT_HEAD_PATH_KEYS = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")`, and `apply_output_head` spans norm → dropout → projection → soft-cap → fp32 cast. The flag touches one of five things; these names overclaim. |
| `quantize_output_proj` (Megatron's `fp8_output_proj`) | **Reject.** Collides with `attentions.py:835 out_projection`, the attention output matmul. Unambiguous in Megatron, ambiguous in MaxText. |
| `quantize_logits_proj` (current) | Parallel to the sibling `quantize_router_proj`. But "logits_proj" appears nowhere else in the codebase. |
| `quantize_logits_dense` | **Preferred.** Matches the regex target, the `nnx_decoders.self.logits_dense` attribute, and the `unquantized_modules` accepted-name vocabulary. Makes the tied-embedding rejection self-explanatory (the module does not exist in that path) and removes the need to spell the regex out in the docstring — the docstring that is currently wrong (M1). Shared vocabulary with `unquantized_modules` also makes the C3 contradiction visible to users rather than hidden behind two different names for one module. |

Counter-consideration: `quantize_router_proj` is the weaker precedent, not a standard —
it is named after a concept while targeting module `gate`, which is exactly why its
docstring has to spell out `.*/gate$`.

Neither candidate name can convey that the flag also covers the MTP invocation (C1);
that belongs in the docstring regardless.

---

## Prior art check: does Megatron quantize the LM head for DeepSeek-V3 671B?

Short answer: **not by default, opt-in only under MXFP8, and NVIDIA does enable it for
DSv3 on Blackwell.** Full detail with source citations in `plan.md` §7.

Relevant to this review:

- Megatron's validator *forbids* head quantization under per-tensor FP8 recipes
  (tensorwise / delayed / blockwise); only `fp8_recipe='mxfp8'` is permitted.
- MaxText's `fp8_full` rule uses `absmax` with no tile size — plausibly the regime
  Megatron refuses for this module.
- **Action:** confirm Qwix's default granularity when no tile size is supplied. If it is
  per-tensor, this flag enables a configuration NVIDIA deliberately blocks. That does not
  make it wrong, but it does mean a loss-curve check should precede anyone turning it on,
  and the docstring should say so.

---

## Good

- The `paths` list + join refactor in `get_fp8_full_qwix_rule_w_sparsity` is cleaner than
  the nested if/else and scales to the next flag.
- Rejecting `logits_dot_in_fp32` is correct and mirrors the `float32_gate_logits`
  precedent. `nnx_decoders.py:472` confirms the flag only sets the compute dtype, which
  Qwix would requantize.
- Rejecting `logits_via_embedding=True` is correct: the tied path goes through
  `attend_on_embedding`, not `decoder/logits_dense`, so the rule would silently no-op.
  See `plan.md` §9 for why supporting it is non-trivial.
- The test mirrors `multi_token_prediction_test.py`'s structure, and asserting on Qwix
  interception logs is the right verification level for a rule-plumbing change.

---

## Checklist before opening the PR

- [ ] Remove `sandbox/` from the branch (B1)
- [ ] Squash commits with a real message (B2)
- [x] Document MTP coverage in the field description (C1) — done in the C4 pass
- [x] C2 resolved: sparsity coupling kept and documented in the field description
- [ ] File a separate issue: `weight_sparsity_start_step`/`update_step` are inert on the QtRule path (MoE-only schedule)
- [ ] Decide: validate or document `unquantized_modules` overlap (C3)
- [x] C4 resolved: batch-split rejection added; docs corrected; no `use_qwix_quantization` check
- [x] Fix the regex in the docstring (M1) — done in the C4 pass
- [ ] Align the `base.yml` comment wording (M2) — partially done; `quantize_logits_proj` line rewritten, `quantize_router_proj` line still claims `use_qwix_quantization` applicability and is now inconsistent with its neighbour
- [ ] De-brittle the rule-index assertion (M3)
- [ ] Decide on final flag name (`quantize_logits_proj` vs `quantize_logits_dense`)
- [ ] Confirm Qwix default granularity; note the accuracy caveat in the docstring
- [ ] Revert the unrelated doc reflow, or split it into its own commit (M5)
- [ ] Run the unit tests + batch-split rejection check on the TPU VM (see C4)
- [x] C5 guarded: `num_vocab_tiling > 1` rejected at config init
- [ ] Run the C5 decisive experiment; drop the guard if interception survives tiling
