# Copyright 2026 Google LLC
"""Test real vLLM routed experts extraction."""

from vllm import LLM, SamplingParams


def test_vllm_routed_experts():
  """Tests vLLM routed experts extraction functionality."""
  print("=" * 80)
  print("1. INITIALIZING VLLM ENGINE WITH ROUTED EXPERTS ENABLED")
  print("=" * 80)

  engine = LLM(
      model="Qwen/Qwen1.5-MoE-A2.7B",
      load_format="dummy",
      trust_remote_code=True,
      max_model_len=128,
      max_num_batched_tokens=128,
      max_num_seqs=16,
      tensor_parallel_size=1,
      pipeline_parallel_size=1,
      enable_expert_parallel=False,
      enable_return_routed_experts=True,
  )

  prompt = "The capital of France is"
  sampling_params = SamplingParams(temperature=0, max_tokens=10)

  print("\nExecuting vLLM generate()...")
  outputs = engine.generate([prompt], sampling_params)
  output = outputs[0].outputs[0]

  print("\n" + "=" * 80)
  print("2. REAL VLLM ROUTED EXPERTS OUTPUT")
  print("=" * 80)
  print("Routed Experts Array Present :", output.routed_experts is not None)
  if output.routed_experts is not None:
    print("Routed Experts Shape         :", output.routed_experts.shape)
    print("Routed Experts Data Sample   :\n", output.routed_experts[:2])
  print("=" * 80)

  engine.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
  test_vllm_routed_experts()
