# DeepSeek-V3 Benchmarking & Weight Sync Plan

This document outlines the plan for executing, validating, and optimizing the DeepSeek-V3 (671B) weight synchronization and vLLM inference benchmarking on the Pathways/XPK cluster.

## 1. Infrastructure
You are working on cluster `bodaborg-super-alpha-cluster` and you can connect to it using `gcloud container clusters get-credentials bodaborg-super-alpha-cluster --region us-central1 --project cloud-tpu-multipod-dev --dns-endpoint`

Don't exec into the pods to check into ongoing workloads

## 2. Workload Configuration (`deepseek_bench.yaml`)
* **Goal:** Verify and maintain the Kubernetes JobSet configurations for distributed execution.
* **Action Items:**
  * **Head Node (`pathways-head`):**
    * Ensure `nodeSelector` strictly targets `cloud.google.com/gke-nodepool: mega-cpu-np`.
    * Verify `MODEL_IMPL_TYPE=vllm` and `NEW_MODEL_DESIGN=True` are passed to force the vLLM PyTorch backend.
    * Ensure `load_format=runai_streamer` is used to load weights efficiently from GCS (`gs://maxtext-model-checkpoints/deepseek-v3/hf-weights`).
  * **TPU Workers (`worker`):**
    * Confirm correct TPU topology request: `cloud.google.com/gke-tpu-topology: 4x4x4` (16 pods x 4 chips = 64 chips? *Wait, 4x4x4 is 64 chips, but completions is 16. Needs continuous validation based on scaling needs*).
    * Ensure `LIBTPU_INIT_ARGS` optimizations (e.g., sparse core offloading, async all-gather) remain correctly tuned for the DeepSeek MoE architecture.

## 3. Weight Synchronization (`bench_weight_sync.py`)
* **Goal:** Validate the in-memory conversion of MaxText/Tunix weights to the vLLM state dictionary for DeepSeek-V3.
* **Action Items:**
  * **Conversion Logic (`DeepSeekV3ToVLLMConverter`):**
    * Validate the `_convert_mla` (Multi-Head Latent Attention) packing logic: ensures `wq_a`, `wkv_a`, `wq_b` are concatenated and reshaped correctly for vLLM.
    * Validate the `_convert_moe` logic: Ensure the routed experts' gate and up projections are correctly interleaved for Tensor Parallelism (TP) and the shared experts are packed correctly.
  * **FP8 Quantization:**
    * Monitor the logic where FP8 weights (`jnp.float8_e4m3fn`) are passed directly or quantized via `tpu_quantize_tensor`. 
    * Ensure MoE block-quantized weights (e.g., `w13_weight`, `w2_weight`) correctly skip standard linear FP8 scaling and retain their checkpoint format until vLLM's TPU backend fully supports block quantization processing. We might change this in the future and come up with ways to convert MaxText's bfloat16 weights to be converted to fp8 and put in vllm state
  * **Mesh Transfer (`_get_reshard_fn`):**
    * Monitor the JAX `device_put` / `jax.jit` out-sharding operations transferring weights from the MaxText mesh to the vLLM mesh. Ensure this doesn't cause OOM spikes on the head node during the `jax.effects_barrier()` step.

## 4. Execution, Profiling & Validation
* **Goal:** Run the benchmark and measure throughput, latency, and correctness.
* **Action Items:**
  1. **Deployment:** Apply the workload via `kubectl delete -f deepseek_bench.yaml; kubectl apply -f deepseek_bench.yaml`.
  2. **Monitoring:** 
     * Watch `pathways-head` logs: `kubectl logs -f <head-pod-name> -c jax-tpu`.
     * Verify all TPU devices (128) are visible to Pathways.
  3. **Performance Metrics:**
     * Record the execution times outputted by the `@timer` context managers:
       * *Convert Global Weights*
       * *Convert Layer Weights*
       * *Assigning weights to vLLM model*
       * *Generation* (End-to-end token generation time).
  4. **Correctness:** 
     * Verify the generation test (`llm.generate("Paris is", ...)`) outputs coherent text, confirming that the `runai_streamer` weight loading, MaxText-to-vLLM conversion, and FP8 quantization were all successful.

## 5. Repositories you have to go through

You are working with MaxText and tpu-inference repositories which can be found in ~/workspace folders. I will try to keep same copies of these repo in the docker image as well. 
Whenever I ask you to go through the repos understand them and go through the files in the local copies. I will rebuild their commits in the docker file `Dockerfile` in MaxText repo.


## Current Plan & Implementation Status

  * TPU-Inference Fast-Path: The process_fp8_moe_weights function is now JIT-compiled and executes directly on the TPU mesh. This eliminates the CPU-bound "FP8 -> FP32 -> FP8" round-trip that was crashing the proxy.
  * Scale Padding Fix: Patched moe_weights.py to correctly handle native DeepSeek block-quantized scale shapes (256, 56, 1, 32), preventing the TypeError reshape crash.
  * Windowed Pipelining: Modified bench_weight_sync.py to keep 4 layers "in-flight" at a time, overlapping network transfers with TPU processing while staying within the 850G proxy memory limit.
  * Unified Connection Poll: Moved the "Wait for 128 devices" logic inside the script to ensure a single, stable gRPC connection to the Pathways proxy.
  * Network Stability: Added missing MegaScale environment variables (MEGASCALE_COORDINATOR_ADDRESS, etc.) to worker pods to fix the 10-second connection timeout loop.

  1. Core Bug Fixes in tpu-inference
   * MoE Scale Reshape Fix: Discovered that tpu-inference's process_moe_weights (the Fast Path) was hardcoded to assume MoE scales matched the kernel's inner dimensions. DeepSeek-V3 uses native block-quantized scales of shape (256, 56, 1, 32), which caused a TypeError during reshape.
       * Fix: Patched moe_weights.py to use jnp.swapaxes for native scales, bypassing the incorrect padding logic.
   * Proxy OOM Prevention: Discovered that JAX's asynchronous device_put in the weight-loading loop would queue all 61 MoE layers (671B parameters) into the Pathways Proxy memory at once.
       * Fix: Added a block_until_ready() call in tpu_inference/layers/vllm/quantization/fp8.py to force sequential, layer-by-layer processing, keeping proxy memory usage stable at ~850G.

  2. Performance Optimizations
   * Fast Path Activation: tpu-inference defaults MOE_REQUANTIZE_WEIGHT_DTYPE to float8_e4m3fn, which triggers a "Slow Path" (CPU-bound FP32 upcasting and re-quantization).
       * Optimization: Setting MOE_REQUANTIZE_WEIGHT_DTYPE: "" (empty string) in the YAML forces the JIT-compiled fast path, executing weight reordering directly on the TPU mesh.
   * Windowed Pipelining (MaxText Converter): For the custom conversion loop in bench_weight_sync.py, we implemented a 4-layer sliding window.
       * Optimization: Instead of blocking after every layer, we keep 4 layers "in-flight." This overlaps the head-to-worker network transfer with the TPU sharding operations without exceeding memory limits.

  3. Infrastructure & Stability (YAML)
   * Startup Policy Circular Dependency: Discovered that startupPolicyOrder: InOrder can cause a deadlock. The head pod's Resource Manager refuses to be "Ready" until workers connect, but Kubernetes won't start workers until the head pod is "Ready."
       * Learning: Use startupPolicyOrder: AnyOrder for Pathways workloads on this cluster.
   * MegaScale Networking: Workers require specific environment variables to reliably connect to the head node leader for low-level coordination.
       * Required Vars: MEGASCALE_COORDINATOR_ADDRESS, MEGASCALE_NUM_SLICES, and MEGASCALE_SLICE_ID must point to the coordinator.
   * Node Selection Logic: Hardcoding cloud.google.com/gke-tpu-topology: 4x4x4 in the nodeSelector failed because the physical nodes are labeled with the full cluster topology (8x16x16).
       * Learning: Remove topology from nodeSelector. Instead, use kueue.x-k8s.io/podset-slice-required-topology: cloud.google.com/gke-tpu-partition-4x4x4-id to carve out the 64-chip slice.

  4. Identified Long-term Bottlenecks
   * The "Single Funnel" Problem: Even with the Fast Path, loading 641GB of weights through the head node Funnel (GCS -> Head CPU -> Proxy -> 16 Workers) is physically limited by the head node's network interface, taking ~15–20 minutes.
       * Future Recommendation: Implement Distributed Loading in tpu-inference, where each TPU worker uses runai_streamer to pull its specific shards directly from GCS, parallelizing the transfer 16x.

  5. Current Workspace State
   * Repository Branch: All tpu-inference fixes are committed and pushed to the mohit/patch_tiles branch.
   * Docker Image: The image gcr.io/tpu-prod-env-multipod/mohit-dsv3-build:20260410 is now a "golden" build containing all code fixes described above.