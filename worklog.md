# MaxText Router Replay Work Log

## Project Activities & Experiment Runs

### Activity: Real vLLM Inference vs Training Logits Comparison (WITH & WITHOUT Router Replay)
- **Branch**: `mohit/router-replay-analysis` (tracking `origin/xfgu-router-replay`)
- **Status**: Completed & Verified on Physical TPU Hardware (v5p-8)
- **Files Created/Updated**:
  - Script: [`MyStuff/scratch/e2e_real_vllm_to_maxtext_training.py`](file:///usr/local/google/home/mohitkhatwani/maxtext/MyStuff/scratch/e2e_real_vllm_to_maxtext_training.py)
  - Report: [`MyStuff/Docs/router-logits-divergence/report.md`](file:///usr/local/google/home/mohitkhatwani/maxtext/MyStuff/Docs/router-logits-divergence/report.md)

### Key Results Summary
- Initialized real `vllm.LLM` engine with `enable_return_routed_experts=True` on TPU v5p-8 and extracted real `routed_experts` tensor `(20, 24, 4)`.
- Verified that MaxText Inference Logits vs Natural Training Logits (WITHOUT Router Replay) match **PERFECTLY (0.000000 MAE, 100.00% Token Agreement)** across all 24 layers.
- Plumbed real `vllm_routed_experts` into MaxText training (`forced_routed_experts`) and achieved **100.00% Top-1 Token Prediction Agreement** and Cosine Similarity $> 0.99998$ across all 24 layers.
