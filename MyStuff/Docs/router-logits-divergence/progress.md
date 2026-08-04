# Progress Log: Router Logits Replay and Divergence Analysis

## Task Summary
1. Understand how router logits/indices are returned in TPU inference today.
2. Review MaxText PR #3881 (`origin/xfgu-router-replay`) for router replay (`forced_routed_experts`).
3. Build an end-to-end script that runs TPU inference to capture router logits/indices, feeds them into MaxText training forward pass (`forced_routed_experts`), and analyzes layer-by-layer divergence.

## Milestones & Status
- [x] Audited TPU Inference codebase (`tpu-inference`):
  - Identified `enable_return_routed_experts` mechanism.
  - Returns `routed_experts` array of shape `(num_tokens, num_moe_layers, top_k)`.
- [x] Audited MaxText PR #3881 (`origin/xfgu-router-replay`):
  - Located commit `13280fd97` on branch `origin/xfgu-router-replay`.
  - Created local branch `mohit/router-replay-analysis`.
  - Verified `forced_routed_experts` parameter threading across `Model`, `Decoder`, `DecoderLayer`, `MoEBlock`, and `RoutedMoE`.
- [ ] Implement end-to-end demonstration & divergence measurement script.
- [ ] Execute script and generate comprehensive report.
