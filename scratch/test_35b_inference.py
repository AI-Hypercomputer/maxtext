"""Offline timing benchmark for Qwen 35B model."""

import time
import sys
import jax

# Set custom cache dir to force cold cache
cold_cache_dir = f"gs://cloud-pathways-staging/tmp/compilation_cache/igorts-cold-{int(time.time())}"
print(f"Setting JAX compilation cache to cold directory: {cold_cache_dir}", file=sys.stderr, flush=True)
jax.config.update("jax_compilation_cache_dir", cold_cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

import pathwaysutils

# Initialize Pathways
pathwaysutils.initialize()
print("Pathways initialized successfully", file=sys.stderr, flush=True)

from vllm import LLM, SamplingParams

model_id = "Qwen/Qwen3.5-35B-A3B"
print(f"Initializing LLM with model {model_id}...", file=sys.stderr, flush=True)

t0 = time.perf_counter()
llm = LLM(
    model=model_id,
    load_format="dummy",
    tensor_parallel_size=4,
)
t1 = time.perf_counter()
print(f"LLM initialization took {t1 - t0:.2f} seconds", file=sys.stderr, flush=True)

prompts = ["Hello, my name is"]
print("Generating...", file=sys.stderr, flush=True)
outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_k=50, max_tokens=16))
for output in outputs:
  print(f"Prompt: {output.prompt!r} -> Generated: {output.outputs[0].text!r}")
