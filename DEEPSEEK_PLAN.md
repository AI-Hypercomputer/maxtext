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

