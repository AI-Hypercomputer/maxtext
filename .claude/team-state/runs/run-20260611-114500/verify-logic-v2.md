## Verdict
PASS

## Critical issues (must fix)

None.

## Suggestions (nice to fix)

- **Section 4.2, checkpoint liveness claim**: "These must remain live across the full forward-then-backward lifetime of the scan, making them the dominant tmem contributor." This implies all 32 checkpoint buffers (8 stages x 4 layers) are simultaneously live for the entire scan duration. In reality, the scan's remat policy governs per-iteration checkpoint saving; it is the remat wrapping the scan body that keeps checkpoints alive across iterations. The claim is directionally correct but slightly overstates the mechanism. Consider rephrasing to clarify that the scan carries these as part of its state/remat checkpoint, rather than suggesting they are all live for the "full forward-then-backward lifetime" as a single monolithic block.

- **Section 6.1, VMA claim**: "the VMA tracking can also cause the compiler to insert explicit copies to enforce the declared output sharding, adding buffer allocation to tmem." This is a bare assertion about compiler behavior. The paragraph explains what VMA does (tracks varying vs. invariant status) but does not explain *why* VMA would cause extra copies that would not otherwise occur. Since this is part of the argument for why ppermute costs more tmem, a sentence explaining the mechanism (e.g., "VMA validation may force the compiler to insert a copy when the output's varying status doesn't trivially match the declared spec") would close the gap.

- **Section 6.3, zero-copy claim**: "XLA can often implement the per-device portion as buffer views (pointer arithmetic) with zero copies, or at most a single local memcpy." The qualifier "often" hedges this, but the surrounding argument treats this as the definitive explanation for why local ops save tmem. The draft could strengthen this by noting that even when a copy is required, a single local copy is still cheaper in tmem terms than the send+recv buffer pair required by a collective.

- **Section 9.2, activation size arithmetic**: The calculation `3 * 4096 * 4096 * 2 = ~96 MB` evaluates to exactly 96 MB (100,663,296 bytes = 96 MiB). This is fine, but the draft then says "per_stage = num_layers_per_pipeline_stage * activation_size = 4 * 96 MB = ~384 MB" and immediately follows with a disclaimer that the exact savings cannot be isolated. Since this arithmetic is not connected to any measured quantity, it serves only as an intuition pump -- consider explicitly labeling it as such (e.g., "as a rough estimate of the scale").

## Notes

- The v2 draft's savings table (Section 12) avoids the per-optimization GB estimates that caused the arithmetic-overshoot problem in v1 (from notebook). The qualitative labels ("largest single contributor," "moderate," "small") with a clear non-additivity disclaimer are logically sound. Good fix.

- The draft's thesis is clear and can be stated in one sentence: "Six optimizations to MaxText's pipeline parallelism implementation reduce XLA temp compile memory from 29.9 GB to 20.4 GB by eliminating collective buffers, unnecessary sharding materializations, and remat checkpoint saves." Sections build logically from background (Sections 1-4) through problem statement (Section 5) to individual optimizations (Sections 6-11) and summary (Section 12-13).

- Factual observation outside my dimension: the draft shows a diff in Section 10 claiming `float32_weight_sum` default changed from `true` to `false` in both `base.yml` and `types.py`. The current codebase has `false` in both places, consistent with the "after" state. I cannot verify the "before" state from the current branch alone, but this is a facts-dimension concern.
