# Design handoff: <model> on m3

<!-- Fill every <...>. At most 2 printed pages: decisions first, plain English,
no unexplained shorthand codes. Detail lives in the design proposal; link it. -->

**Decisions for you** (each: options, facts, strongest counter-argument, one
recommendation):

1.  <decision> - options: <A> / <B>. Recommend <A> because <reason>. Against
    it: <strongest counter-argument>.

**Model and reference**

-   Model: <name and size>; output kind: <LOGITS | DIFFUSION | EMBEDDINGS>.
-   Pinned reference: <reference module or in-test oracle>, <revision>.
-   Deviations from the reference: <CPU kernel replacements and how each was
    checked; numerics-changing overrides> or "none".

**Closest family model and reuse** (family diff:
<path of report.md> or "no family model")

| Component | Reference file:line | New / changed / reused | Evidence | m3 target |
| :-------- | :------------------ | :--------------------- | :------- | :-------- |
| <name>    | <file:line>         | <status>               | <family-diff row or "new"> | <file> |

**Mini-twin and goldens**

-   Mini-twin: <layers kept and why they cover every layer type>; geometry
    <seq vs top-k, scan blocks> (details: <mini_twin.md>).
-   Goldens: <what is captured>, stored under the project golden root, sha256
    pinned by the provenance test.

**Tolerances** (one row per check; every test cites a row)

| Row | Check | Metric | Tolerance | Measured noise floor | Justification |
| :-- | :---- | :----- | :-------- | :------------------- | :------------ |
| T1  | <component forward fp32> | <elementwise + frob_rel> | <value> | <value> | <why> |

**Conformance spec** (`tests/unit/m3/conformance/spec.py`):
`frozen_param_patterns` <...>; `aux_losses` <...>;
`stacked_param_path_patterns` <...>; `expert_weight_param_patterns` <...>;
`skips` <declare `SkipDeclaration(SkipStatus.NOT_APPLICABLE, reason)` where a
check does not apply; Check 9b `PACKING_AND_PADDING_ISOLATION` and
`packing_train_check.py` required>.

**PR plan** (one focus per PR; each PR carries its tests and negative control)

| PR | Focus | Tests | Lines (code / tests) |
| :- | :---- | :---- | :------------------- |
| 1  | <reference oracle, goldens, provenance test, conformance spec> | <provenance> | <n / n> |

**Risks and open questions**: <each with what would settle it>.
