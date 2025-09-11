## Goal

Work on writing a inference engine in @MaxText/inference/continuous_batching_engine.py which reduces python threads by not using loops or Threads and has the feature of continous batching. It should be implemented so it avoids devices to host transfer as much as possible. 
Only work on `@MaxText/inference/continuous_batching_engine.py` for making the optimizations

implement continous batching function which has a feature for delayed EOS checking and does not block next generate invokations.

One of the other goal is the there shouldn't be any transfer to host. Try to
keep everything in JAX arrays inside the jitted functions. Converting to numpy
or doing operations like .get might bring that data to host.

## Gemini Added Memories
* In the `_run_continuous_batching` function in `MaxText/inference/continuous_batching_engine.py`, after `prefill_vmap` is called, the continuous batching logic should be implemented. This logic must include a delayed or non-blocking check for EOS tokens to prevent stalling subsequent `jit_generate` invocations.
* Remember to not use python threads in your logic to avoid any python level overhead


## Test changes
Test your changes using following command: `JAX_PLATFORMS=proxy
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 python -m
MaxText.tests.inference.benchmark_offline_engine`


## Context

To know more about continous batching, see this https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/#continuous-batching


