# Verifier-Facts Notebook

## Failure Patterns Observed

### Run 20260611-114500 (tmem deep-dive)

1. **Fabricated acronym expansions**: Writer expanded "VMA" as "Verified Memory Annotation" when it actually stands for "Varying Manual Axes." Pattern: when a technical acronym isn't immediately obvious, the writer may invent a plausible-sounding expansion rather than looking it up.

2. **Confusing model parameters with layer counts**: Writer claimed DeepSeek-V2/V3 has "160+ layers" when V2 has 60 and V3 has 61 layers. The number 160 is the count of routed experts per MoE layer in V2. Pattern: numbers from one dimension of a model architecture being attributed to a different dimension.

3. **Misattributing compiler phases**: Writer placed the SPMD partitioner in the lowering phase (`.lower()`) when it actually runs during compilation (`.compile()`). Pattern: conflating the JAX lowering step with XLA compilation passes.

4. **"Multi-Latent Attention" vs "Multi-head Latent Attention"**: Dropped "head" from MLA's full name. Pattern: shortening proper technical names in ways that change the meaning.

### Run 20260611-114500 v2 (corrections applied)

All four issues from v1 were corrected in v2. No new failure patterns observed. The draft passed fact-check cleanly.

## Reliable Sources

- The actual source code files in the repo are reliable for verifying code claims.
- The config YAML file comments corroborate empirical measurements.
- JAX documentation for `CompiledMemoryStats` fields is consistent and reliable.
- DeepSeek-V2 paper (arXiv:2405.04434) confirms 60 layers, 160 routed experts per layer.
- DeepSeek-V3 technical report (arXiv:2412.19437) confirms 61 layers, 256 routed experts per layer.
- Edward Yang's blog post on JAX sharding type system confirms VMA = "Varying Manual Axes."
