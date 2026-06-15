## Verdict
NEEDS_REVISION

## Critical issues (must fix)

- **Section 12 (savings breakdown table)**: The individual per-optimization estimates sum to ~16.1--21.1 GB (low-end: 5+1+0.5+0.5+7+2+0.1; high-end: 7+2+0.5+0.5+9+2+0.1), but the actual measured reduction is only 9.5 GB (29.9 - 20.4). The disclaimer that "per-optimization savings do not add up exactly because XLA's buffer assignment is a global optimization" is far too weak for a 2x overshoot. Buffer sharing explains why individual removals save *less* than the buffer size, not why the sum is *double* the real savings. Either the individual estimates need to be deflated (with explanation of which buffers overlap in liveness), or the disclaimer needs to be substantially strengthened to explain why a 2x overshoot is expected. As written, the table undermines the credibility of the entire quantitative argument.

- **Section 6.3 / diagram caption ("SPMD replicated along stage axis")**: The diagram states "The full [num_stages, ...] array lives on each device (SPMD replicated along stage axis)." But `stages_in_logical` uses `"activation_stage"` as the first axis, which maps to the `"stage"` mesh axis -- meaning the array is *sharded* across stages, not replicated. The local slice+concat works because XLA's SPMD partitioner transforms the logical operation into per-device communication or local operations as needed, not because each device holds the full array. This is a material misstatement of the mechanism that the optimization relies on.

- **Section 7.3 (Removed meta.remove_axis on Weights)**: The draft claims `meta.remove_axis` was removed from the Pipeline class's circular path. But the current codebase shows `meta.remove_axis` is still called in both `Pipeline.run_one_iteration` (line 684 of pipeline.py) and `Pipeline.get_current_repeat_from_stages` (line 622). The draft presents this removal as contributing ~0.5 GB of savings, building an argument on a change that does not appear to exist in the code.

- **Section 1 (pipeline_microbatch_size calculation)**: The draft writes `pipeline_microbatch_size = 24 / 64 = 0.375... actually computed as micro_batch_size_to_train_on / num_pipeline_microbatches`. This is unfinished reasoning left in the draft. A microbatch size of 0.375 is nonsensical (you cannot have a fractional batch size), which suggests the author realized the arithmetic was wrong but did not resolve it. The calculation needs to be completed correctly or removed -- this is a running example that the entire article builds on, so a broken core parameter undermines subsequent reasoning.

## Suggestions (nice to fix)

- **Section 4.2 ("dominant contributor") vs. Section 6 ("single largest tmem reduction after the remat checkpoint change")**: These claims are technically compatible, but the argument would be clearer if Section 4.2 stated the estimated size (7-9 GB) alongside the qualitative "dominant" label, and if Section 6 similarly stated its estimate (5-7 GB). The reader currently has to reach Section 12 to compare them.

- **Section 9.2 (Quantifying the Savings)**: The formula `total_decoder_layer_checkpoints = per_stage * total_iterations_active` is introduced but never evaluated for the running example. The section then jumps to "empirically this accounts for approximately 7-9 GB." For an article that promises to use the ds-proxy config as a concrete running example, this is a missed opportunity to show the math.

- **Section 10 (float32_weight_sum)**: The claim "adding approximately 2 GB of tmem per device for emb_dim=4096" is a bare assertion with no derivation. For consistency with the rest of the article's analytical depth, showing the buffer size calculation would strengthen this (e.g., `num_experts_per_tok * batch * seq * emb * sizeof(float32)` minus the bf16 equivalent).

- **Appendix (Numerical Correctness)**: The claim that recomputed activations are "bit-identical to the saved ones (assuming deterministic execution)" is stated parenthetically but the assumption is doing a lot of work. If the training uses dropout (the config has `enable_dropout`-related settings), recomputed activations will *not* be bit-identical unless the PRNG state is also saved. This deserves a clearer statement of when the assumption holds.

## Notes

- (Outside logic dimension, flagging for verifier-facts) The draft states `pipeline_microbatch_size = 24 / 64 = 0.375`. The codebase computes `self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches` (line 54 of pipeline.py). The `micro_batch_size_to_train_on` for a single node with `per_device_batch_size=24` and 8 pipeline-parallel devices may not be 24 -- it depends on how many devices are outside the pipeline axis. This needs factual verification.

- (Outside logic dimension) The draft does not include all the Excalidraw-style diagrams requested by the brief. Specifically, the brief asks for a diagram of "the remat checkpoint boundary: which activations are saved vs. recomputed" -- the draft has this. But the "memory layout" diagram (Section 13) is more of a conceptual block diagram than the HBM layout diagram the brief envisions. This is a craft/structure concern.
