## Decision
REVISE

## Reasoning
Two of three verifiers flagged critical issues: verifier-logic found 4, verifier-facts found 3, and verifier-craft passed with suggestions only. The critical issues are real factual and logical errors (fabricated acronym, wrong layer counts, misattributed compilation phases, inconsistent quantitative claims, and a phantom code change). No verifier conflicts to resolve -- the issues are complementary. Round 1 of 3, so REVISE is appropriate.

## Revision brief

1. **Fix fabricated acronym and wrong DeepSeek facts.**
   - Where: Section 6.1 ("check_vma (Verified Memory Annotation)") and Section 1 callout box ("160+ layers", "Multi-Latent Attention").
   - Why it matters: VMA stands for "Varying Manual Axes," not "Verified Memory Annotation." DeepSeek V2 has 60 layers, V3 has 61 -- not "160+." MLA is "Multi-head Latent Attention," not "Multi-Latent Attention." These are factual errors that undermine credibility with the target audience.
   - Suggested approach: Correct the VMA expansion. Fix DeepSeek layer counts and MLA full name. If unsure, remove the specific numbers rather than guessing.

2. **Fix SPMD partitioner timing and the "replicated along stage axis" claim.**
   - Where: Section 3.3 says "The SPMD partitioner runs during this phase [lowering]." Section 6.3 diagram caption says "The full [num_stages, ...] array lives on each device (SPMD replicated along stage axis)."
   - Why it matters: The SPMD partitioner runs during compilation (`.compile()`), not lowering (`.lower()`). The pipeline arrays are sharded along the stage axis, not replicated. These errors misstate the core mechanism the optimizations rely on.
   - Suggested approach: Move the SPMD partitioner mention to Section 3.4 (compilation). In the Section 6 diagram and text, explain that the local slice/concat operates on the logically full array which XLA's SPMD partitioner transforms into per-device operations -- the array is sharded, and XLA handles the communication if needed.

3. **Fix the savings breakdown table so the numbers tell a consistent story.**
   - Where: Section 12 table (individual estimates sum to ~16-21 GB, actual reduction is 9.5 GB), Section 9.2 (claims 7-9 GB from one optimization out of 9.5 GB total, leaving almost nothing for the other five optimizations listed in the table).
   - Why it matters: The table's estimates overshoot the measured total by 2x. The disclaimer ("buffer assignment is a global optimization") is too weak for a 2x discrepancy. And Section 9.2's 7-9 GB claim contradicts the table's allocation to other optimizations.
   - Suggested approach: Either (a) deflate the per-optimization estimates to ranges that sum closer to 9.5 GB, noting which buffers overlap in liveness and therefore don't yield additive savings, or (b) present them as "standalone impact if only this optimization were applied" and make the non-additive nature the headline, not a footnote. Section 9.2 should be consistent with whichever framing is chosen.

4. **Remove or correct the phantom "Removed meta.remove_axis" optimization (Section 7.3).**
   - Where: Section 7.3 claims `meta.remove_axis` was removed from the Pipeline class's circular path and attributes ~0.5 GB savings to it.
   - Why it matters: The current codebase still has `meta.remove_axis` calls in both `Pipeline.get_current_repeat_from_stages` (line 622) and `Pipeline.run_one_iteration` (line 684). If this optimization does not exist in the branch, claiming it does is a factual error, and the savings table is further inflated.
   - Suggested approach: Check whether the change actually exists in the branch diff. If it does not, remove Section 7.3 entirely and adjust the savings table. If it was a partial removal, describe accurately what was and was not removed.

5. **Fix the broken microbatch size calculation in Section 1.**
   - Where: The bullet "pipeline_microbatch_size = 24 / 64 = 0.375... actually computed as micro_batch_size_to_train_on / num_pipeline_microbatches" trails off with unfinished reasoning. A fractional microbatch size of 0.375 is nonsensical.
   - Why it matters: This is the running example the entire article builds on. A broken core parameter undermines all subsequent size calculations.
   - Suggested approach: Trace how `micro_batch_size_to_train_on` is actually computed for this config (it depends on `per_device_batch_size`, the number of devices outside the pipeline axis, etc.). Either show the correct derivation or remove the microbatch size bullet and note that it depends on the full mesh configuration.
