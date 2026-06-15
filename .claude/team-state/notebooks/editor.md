# Editor Notebook

## Run 20260611-114500, Round 1

### Decision: REVISE

### Patterns observed:
- **Quantitative consistency is hard for technical deep-dives.** The writer provided per-optimization estimates that summed to 2x the actual measured total. This is a recurring risk when individual buffer sizes are estimated without accounting for XLA's global buffer sharing. Need to watch for this in future rounds -- the writer should either present non-additive estimates with strong caveats or deflate to match.
- **Fabricated acronyms.** VMA was expanded as "Verified Memory Annotation" instead of "Varying Manual Axes." This is an LLM confabulation pattern -- plausible-sounding but wrong. Always flag acronym expansions for facts verification.
- **Phantom code changes.** The draft claimed `meta.remove_axis` was removed, but the codebase still has it. The writer may have been describing intended changes from the brief rather than actual changes in the code. Need to ensure the writer verifies claims against actual code.
- **Compilation phase confusion.** SPMD partitioner placed at lowering instead of compilation. This is a subtle but important distinction that the target audience would catch immediately.
- **Unfinished calculations left in draft.** The microbatch size calculation trailed off with "actually..." -- this is a sign the writer hit a snag and moved on without resolving it.

### Calls I found hard:
- Whether to combine the SPMD partitioner timing issue (facts) with the "replicated vs sharded" claim (logic) into one revision item. They are related (both about misunderstanding the sharding model) but in different sections. Combined them since the fix requires consistent understanding of the same mechanism.
- Whether the savings table issue should be one item or two (table consistency + Section 9.2 inconsistency). Combined them since fixing one requires fixing the other.

## Run 20260611-114500, Round 2

### Decision: PASS

### Patterns observed:
- **Qualitative labels as a fix for inconsistent quantitative claims worked well.** The writer replaced specific GB estimates (which were internally inconsistent) with qualitative rankings ("largest single contributor," "moderate," "small") plus a measured combined total. All three verifiers accepted this as factually responsible. Good pattern for when standalone measurements are not available and buffer-sharing makes individual estimates non-additive.
- **All five v1 critical issues were successfully resolved.** The writer corrected factual errors (VMA, DeepSeek layers, MLA name, SPMD partitioner timing), removed the phantom code change section entirely, fixed the microbatch calculation, and restructured the savings table. Clean execution.
- **Closure variables in code snippets.** The craft verifier flagged `stream_buf_idx` appearing in a function body without being in the function signature. Verified against source: it is a closure variable captured from the enclosing scope. The snippet is technically accurate but reads oddly in isolation. Not critical, but a pattern to watch -- code excerpts should either show enough context for closure variables to be obvious, or add a comment noting the closure.

### Calls I found hard:
- None. Unanimous PASS across all three verifiers with only stylistic suggestions. Straightforward decision.
