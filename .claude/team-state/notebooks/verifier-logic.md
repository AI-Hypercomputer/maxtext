# Verifier-Logic Notebook

## Failure patterns observed

- **Savings arithmetic overshoot**: When listing per-optimization savings that should sum to a total, the individual estimates summed to 2x the actual measured total. Disclaimer about non-additivity was present but far too weak for the magnitude of the discrepancy. Watch for this in any article that lists per-item contributions to a known total.

- **Code claim vs. codebase mismatch**: Draft claimed a code change (removing meta.remove_axis) that does not exist in the actual codebase. Built a savings argument on a phantom change. Always spot-check at least one claimed code diff against the actual source.

- **Unfinished arithmetic left in draft**: A running-example parameter calculation was started, found to be nonsensical (fractional batch size), and trailed off with "actually computed as..." without resolution. The running example is foundational -- a broken core parameter undermines all downstream reasoning.

- **Sharded-vs-replicated confusion**: Draft said an array was "SPMD replicated along stage axis" when it was actually sharded along that axis. The mechanism explanation was built on this wrong premise. Watch for shard/replicate confusion in SPMD/JAX articles.

- **Bare assertions for numerical estimates**: Multiple GB-scale savings claims given without derivation in an article that otherwise goes deep on buffer-size math. Inconsistent rigor.

- **Savings arithmetic overshoot (resolved in v2)**: The v2 draft replaced per-optimization GB estimates with qualitative labels and a clear non-additivity disclaimer. This pattern was successfully caught and fixed. The qualitative approach is a good model for future articles with non-additive optimizations.

- **VMA/compiler-behavior bare assertions**: Draft asserted VMA tracking causes extra copies without explaining the mechanism. This is a recurring pattern: when explaining why a JAX/XLA feature has performance implications, the causal chain from feature to compiler behavior to buffer allocation needs to be explicit, not just asserted.
