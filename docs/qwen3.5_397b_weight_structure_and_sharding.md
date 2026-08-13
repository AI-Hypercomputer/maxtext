# Qwen3.5-397B Weight Structure & Sharding (Training vs. Inference)

This document details the **Qwen3.5-397B** (`qwen3.5-397b-a17b`) model architecture in MaxText, specifying tensor shapes, dtypes, and sharding layouts under **Training (FSDP + EP via `rl.yml`)** versus **Inference (2-way Attention-DP + 2-way TP via `tpu-inference` / `vllm.yml`)**.

______________________________________________________________________

## 1. Architectural Dimensions & Hyperparameters

Qwen3.5-397B is a 60-layer hybrid Mixture-of-Experts (MoE) architecture combining **Linear Attention (Gated Delta Networks / GDN)** and **Full Gated Multi-Head Self-Attention (GQA)** with routed and shared experts.

### Core Model Dimensions

| Parameter                  | Symbol / Key                           | Value        | Description                                            |
| :------------------------- | :------------------------------------- | :----------- | :----------------------------------------------------- |
| **Embedding Dimension**    | $D$ (`emb_dim`)                        | **4,096**    | Hidden size / token feature dimension                  |
| **Number of Layers**       | $L$ (`num_decoder_layers`)             | **60**       | Total transformer decoder layers                       |
| **Vocabulary Size**        | $V$ (`vocab_size`)                     | **248,320**  | Vocabulary token count                                 |
| **Cycle Interval**         | (`inhomogeneous_layer_cycle_interval`) | **4**        | Repeating cycle: 3 GDN layers + 1 Full Attention layer |
| **Total MoE Experts**      | $E$ (`num_experts`)                    | **512**      | Routed experts per MoE block                           |
| **Active Experts / Token** | $K$ (`num_experts_per_tok`)            | **10**       | Top-K routing per token                                |
| **MoE Intermediate Dim**   | $I_{moe}$ (`moe_mlp_dim`)              | **1,024**    | Intermediate hidden dimension per routed expert        |
| **Shared Experts**         | (`shared_experts`)                     | **1**        | Always-active shared expert per layer                  |
| **Shared Expert Dim**      | (`shared_expert_intermediate_size`)    | **1,024**    | Intermediate hidden dimension for shared expert        |
| **Full Attn Query Heads**  | $H_q$ (`num_query_heads`)              | **32**       | Attention query heads ($32 \times 256 = 8,192$)        |
| **Full Attn KV Heads**     | $H_{kv}$ (`num_kv_heads`)              | **2**        | Key/Value heads ($2 \times 256 = 512$)                 |
| **Full Attn Head Dim**     | $d_h$ (`head_dim`)                     | **256**      | Dimension per attention head                           |
| **GDN Key Heads**          | $H_k$ (`gdn_num_key_heads`)            | **16**       | Linear attention key heads ($16 \times 128 = 2,048$)   |
| **GDN Value Heads**        | $H_v$ (`gdn_num_value_heads`)          | **64**       | Linear attention value heads ($64 \times 128 = 8,192$) |
| **GDN Head Dims**          | $d_k, d_v$ (`gdn_key/value_head_dim`)  | **128, 128** | Head dimensions for Key and Value                      |
| **GDN Conv Kernel Dim**    | (`gdn_conv_kernel_dim`)                | **4**        | 1D depthwise causal convolution kernel size            |

### Layer Interleaving (60 Layers)

The 60 layers repeat a 4-layer cycle 15 times ($15 \times 4 = 60$):

- **Layers $i \in \{0, 1, 2, 4, 5, 6, \dots, 58\}$ (45 layers total)**: **Linear Attention (GDN)** + **Sparse MoE Block**
- **Layers $i \in \{3, 7, 11, \dots, 59\}$ (15 layers total)**: **Full Self-Attention (GQA)** + **Sparse MoE Block**

______________________________________________________________________

## 2. Sharding Configurations

- **Training (`rl.yml`: FSDP + EP)**:
  - Active mesh axes: `fsdp` and `expert`. Tensor parallelism (`tensor`) is $1$.
  - Dense parameter matrices and embedding projections are sharded on `fsdp` via `embed` / `embed_vocab` / `embed_moe`.
  - Routed expert weights are sharded on `expert` (EP) via `exp`.
- **Inference (`tpu-inference` / `vllm.yml`: 2-way Attention-DP + 2-way TP)**:
  - Active mesh axes: `attn_dp = 2` and `model = 2` (TP = 2). `data = 1`, `expert = 1`, `attn_dp_expert = 1`.
  - **Embeddings & LM Head (`VocabParallelEmbedding` / `ParallelLMHead`)**: Sharded along the vocabulary dimension ($248,320$) across `('attn_dp', 'model')` via `ShardingAxisName.MLP_TENSOR`.
  - **Attention Layers**: Because Full Attention has only **2 KV heads**, TP is constrained to `model = 2` ($2 / 2 = 1\text{ KV head/rank}$). Attention layers are replicated across the 2 `attn_dp` replicas.
  - **MoE Blocks**: The intermediate dimensions (`mlp_moe` and `mlp`) are sharded across both `attn_dp` and `model` as `('attn_dp', 'model')`, fully utilizing all $2 \times 2 = 4$ chips for MLP computation.
  - **Layer Norms**: Replicated across all inference devices.

______________________________________________________________________

## 3. Tensor Keys, Shapes & Sharding Table

> **Sharding Notation**:
>
> - `P('fsdp', None)`: Dimension 0 sharded on `fsdp`, Dimension 1 replicated.
> - `P(None, 'model')`: Dimension 1 sharded on TP axis `model`, Dimension 0 replicated.
> - `P(('attn_dp', 'model'), None)`: Dimension 0 2D-sharded across `attn_dp` and `model`.
> - `P(None, ('attn_dp', 'model'))`: Dimension 1 2D-sharded across `attn_dp` and `model`.
> - `P('expert', 'fsdp', None)`: Dimension 0 sharded on `expert` (EP), Dimension 1 on `fsdp` (FSDP), Dimension 2 replicated.
> - `P(None, None)`: Fully replicated across all devices.

| Component                                | MaxText Parameter Key                                            | Shape               | Dtype      | Training Sharding (`rl.yml`: FSDP+EP) | Inference Sharding (`tpu-inference`: 2 Attn-DP + 2 TP) |
| :--------------------------------------- | :--------------------------------------------------------------- | :------------------ | :--------- | :------------------------------------ | :----------------------------------------------------- |
| **Embeddings**                           | `params.token_embedder.embedding`                                | `[248320, 4096]`    | `bfloat16` | `P(None, 'fsdp')`                     | `P(('attn_dp', 'model'), None)`                        |
| **LM Head**                              | `params.decoder.logits_dense.kernel`                             | `[4096, 248320]`    | `bfloat16` | `P('fsdp', None)`                     | `P(None, ('attn_dp', 'model'))`                        |
| **Final Norm**                           | `params.decoder.decoder_norm.scale`                              | `[4096]`            | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **Layer Norm 1**                         | `params.decoder.layers_{i}.input_layernorm.scale`                | `[4096]`            | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **Layer Norm 2**                         | `params.decoder.layers_{i}.post_attention_layernorm.scale`       | `[4096]`            | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **GDN Input QKVZ** *(45 layers)*         | `params.decoder.layers_{i}.attention.in_proj_qkvz.kernel`        | `[4096, 20480]`     | `bfloat16` | `P('fsdp', None)`                     | `P(None, 'model')`                                     |
| **GDN Input BA** *(45 layers)*           | `params.decoder.layers_{i}.attention.in_proj_ba.kernel`          | `[4096, 128]`       | `bfloat16` | `P('fsdp', None)`                     | `P(None, 'model')`                                     |
| **GDN Conv1D** *(45 layers)*             | `params.decoder.layers_{i}.attention.conv1d.kernel`              | `[4, 1, 12288]`     | `bfloat16` | `P(None, None, None)`                 | `P(None, None, 'model')`                               |
| **GDN A_log** *(45 layers)*              | `params.decoder.layers_{i}.attention.A_log`                      | `[64]`              | `bfloat16` | `P(None)`                             | `P('model')`                                           |
| **GDN dt_bias** *(45 layers)*            | `params.decoder.layers_{i}.attention.dt_bias`                    | `[64]`              | `bfloat16` | `P(None)`                             | `P('model')`                                           |
| **GDN RMSNorm** *(45 layers)*            | `params.decoder.layers_{i}.attention.norm.rms_norm.scale`        | `[128]`             | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **GDN Out Proj** *(45 layers)*           | `params.decoder.layers_{i}.attention.out_proj.kernel`            | `[8192, 4096]`      | `bfloat16` | `P(None, 'fsdp')`                     | `P('model', None)`                                     |
| **Full Attn Query** *(15 layers)*        | `params.decoder.layers_{i}.attention.attention.query.kernel`     | `[4096, 32, 512]`   | `bfloat16` | `P('fsdp', None, None)`               | `P(None, 'model', None)`                               |
| **Full Attn Key** *(15 layers)*          | `params.decoder.layers_{i}.attention.attention.key.kernel`       | `[4096, 2, 256]`    | `bfloat16` | `P('fsdp', None, None)`               | `P(None, 'model', None)`                               |
| **Full Attn Value** *(15 layers)*        | `params.decoder.layers_{i}.attention.attention.value.kernel`     | `[4096, 2, 256]`    | `bfloat16` | `P('fsdp', None, None)`               | `P(None, 'model', None)`                               |
| **Full Attn Out** *(15 layers)*          | `params.decoder.layers_{i}.attention.attention.out.kernel`       | `[32, 256, 4096]`   | `bfloat16` | `P(None, None, 'fsdp')`               | `P('model', None, None)`                               |
| **Full Attn Q-Norm** *(15 layers)*       | `params.decoder.layers_{i}.attention.attention.query_norm.scale` | `[256]`             | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **Full Attn K-Norm** *(15 layers)*       | `params.decoder.layers_{i}.attention.attention.key_norm.scale`   | `[256]`             | `bfloat16` | `P(None)`                             | `P(None)`                                              |
| **MoE Router Gate** *(All 60 layers)*    | `params.decoder.layers_{i}.mlp.routed_experts.gate.kernel`       | `[4096, 512]`       | `bfloat16` | `P('fsdp', None)`                     | `P(None, None)`                                        |
| **MoE Routed Gate** *(All 60 layers)*    | `params.decoder.layers_{i}.mlp.routed_experts.wi_0`              | `[512, 4096, 1024]` | `bfloat16` | `P('expert', 'fsdp', None)`           | `P(None, None, ('attn_dp', 'model'))`                  |
| **MoE Routed Up** *(All 60 layers)*      | `params.decoder.layers_{i}.mlp.routed_experts.wi_1`              | `[512, 4096, 1024]` | `bfloat16` | `P('expert', 'fsdp', None)`           | `P(None, None, ('attn_dp', 'model'))`                  |
| **MoE Routed Down** *(All 60 layers)*    | `params.decoder.layers_{i}.mlp.routed_experts.wo`                | `[512, 1024, 4096]` | `bfloat16` | `P('expert', None, 'fsdp')`           | `P(None, ('attn_dp', 'model'), None)`                  |
| **MoE Shared Gate** *(All 60 layers)*    | `params.decoder.layers_{i}.mlp.shared_expert.wi_0.kernel`        | `[4096, 1024]`      | `bfloat16` | `P('fsdp', None)`                     | `P(None, ('attn_dp', 'model'))`                        |
| **MoE Shared Up** *(All 60 layers)*      | `params.decoder.layers_{i}.mlp.shared_expert.wi_1.kernel`        | `[4096, 1024]`      | `bfloat16` | `P('fsdp', None)`                     | `P(None, ('attn_dp', 'model'))`                        |
| **MoE Shared Down** *(All 60 layers)*    | `params.decoder.layers_{i}.mlp.shared_expert.wo.kernel`          | `[1024, 4096]`      | `bfloat16` | `P(None, 'fsdp')`                     | `P(('attn_dp', 'model'), None)`                        |
| **Shared Expert Gate** *(All 60 layers)* | `params.decoder.layers_{i}.mlp.shared_expert_gate.kernel`        | `[4096, 1]`         | `bfloat16` | `P('fsdp', None)`                     | `P(None, None)`                                        |
