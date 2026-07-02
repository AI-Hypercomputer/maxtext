# DeepSeek-V4 Checkpoint Parameter Mapping

This document maps the parameters of the MaxText model to the corresponding parameters in the Hugging Face (PyTorch) model for our tiny variant (7 layers, 8 experts).

## Mappings Table

| MaxText Parameter Key | MaxText Shape | Hugging Face Parameter Key | HF Shape | Conversion / Transformation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Global Embeddings & Heads** | | | | |
| `params.token_embedder.embedding` | `N/A` | `model.embed_tokens.weight` | `[129280, 4096]` | Direct copy. |
| `params.decoder.decoder_norm.scale` | `N/A` | `model.norm.weight` | `[4096]` | Direct copy. |
| `params.decoder.logits_dense.kernel` | `N/A` | `head.weight` | `[129280, 4096]` | Transpose: `logits_dense.kernel` = `head.weight.T`. |
| `params.decoder.hc_head.hc_base` | `(4,)` | `model.hc_head.hc_base` | `[4]` | Direct copy. |
| `params.decoder.hc_head.hc_fn` | `(4, 16384)` | `model.hc_head.hc_fn` | `[4, 16384]` | Direct copy. |
| `params.decoder.hc_head.hc_scale` | `(1,)` | `model.hc_head.hc_scale` | `[1]` | Direct copy. |
| **Layer 0 (Type: 0)** | | | | |
| `params.params.decoder.layers_0.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.0.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_0.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.0.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 0 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_0.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_0.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_0.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.0.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_0.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.0.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_0.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.0.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_0.mhc_attention.pre_beta` | `(4,)` | `model.layers.0.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_0.mhc_attention.post_beta` | `(4,)` | `model.layers.0.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_0.mhc_attention.res_beta` | `(4, 4)` | `model.layers.0.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_0.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.0.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_0.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.0.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_0.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.0.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_0.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.0.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_0.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.0.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_0.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.0.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_0.mhc_mlp.pre_beta` | `(4,)` | `model.layers.0.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_0.mhc_mlp.post_beta` | `(4,)` | `model.layers.0.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_0.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.0.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_0.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.0.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_0.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.0.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_0.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.0.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 0 MoE MLP Block* | | | | |
| `params.Tid2EidVar.decoder.layers_0.mlp.MoeBlock_0.tid2eid` | `(129280, 3)` | `model.layers.0.mlp.gate.tid2eid` | `[129280, 3]` | Direct copy (Hash Routing lookup table). |
| `params.params.decoder.layers_0.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.0.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_0.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.0.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_0.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.0.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_0.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.0.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_0.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.0.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_0.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.0.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_0.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.0.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 0 Attention Block* | | | | |
| `params.params.decoder.layers_0.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.0.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_0.self_attention.q_norm.scale` | `(1024,)` | `model.layers.0.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_0.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.0.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_0.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.0.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_0.self_attention.kv_norm.scale` | `(512,)` | `model.layers.0.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_0.self_attention.sinks` | `(64,)` | `model.layers.0.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_0.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.0.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_0.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.0.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| **Layer 1 (Type: 1)** | | | | |
| `params.params.decoder.layers_1.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.1.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_1.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.1.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 1 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_1.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_1.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_1.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.1.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_1.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.1.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_1.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.1.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_1.mhc_attention.pre_beta` | `(4,)` | `model.layers.1.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_1.mhc_attention.post_beta` | `(4,)` | `model.layers.1.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_1.mhc_attention.res_beta` | `(4, 4)` | `model.layers.1.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_1.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.1.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_1.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.1.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_1.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.1.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_1.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.1.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_1.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.1.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_1.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.1.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_1.mhc_mlp.pre_beta` | `(4,)` | `model.layers.1.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_1.mhc_mlp.post_beta` | `(4,)` | `model.layers.1.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_1.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.1.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_1.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.1.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_1.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.1.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_1.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.1.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 1 MoE MLP Block* | | | | |
| `params.Tid2EidVar.decoder.layers_1.mlp.MoeBlock_0.tid2eid` | `(129280, 3)` | `model.layers.1.mlp.gate.tid2eid` | `[129280, 3]` | Direct copy (Hash Routing lookup table). |
| `params.params.decoder.layers_1.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.1.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_1.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.1.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_1.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.1.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_1.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.1.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_1.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.1.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_1.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.1.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_1.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.1.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 1 Attention Block* | | | | |
| `params.params.decoder.layers_1.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.1.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_1.self_attention.q_norm.scale` | `(1024,)` | `model.layers.1.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_1.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.1.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_1.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.1.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_1.self_attention.kv_norm.scale` | `(512,)` | `model.layers.1.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_1.self_attention.sinks` | `(64,)` | `model.layers.1.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_1.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.1.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_1.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.1.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| **Layer 2 (Type: CSA)** | | | | |
| `params.params.decoder.layers_2.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.2.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_2.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.2.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 2 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_2.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_2.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_2.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.2.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_2.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.2.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_2.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.2.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_2.mhc_attention.pre_beta` | `(4,)` | `model.layers.2.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_2.mhc_attention.post_beta` | `(4,)` | `model.layers.2.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_2.mhc_attention.res_beta` | `(4, 4)` | `model.layers.2.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_2.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.2.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_2.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.2.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_2.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.2.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_2.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.2.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_2.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.2.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_2.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.2.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_2.mhc_mlp.pre_beta` | `(4,)` | `model.layers.2.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_2.mhc_mlp.post_beta` | `(4,)` | `model.layers.2.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_2.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.2.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_2.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.2.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_2.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.2.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_2.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.2.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 2 MoE MLP Block* | | | | |
| `params.Tid2EidVar.decoder.layers_2.mlp.MoeBlock_0.tid2eid` | `(129280, 3)` | `model.layers.2.mlp.gate.tid2eid` | `[129280, 3]` | Direct copy (Hash Routing lookup table). |
| `params.params.decoder.layers_2.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.2.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_2.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.2.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_2.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.2.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_2.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.2.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_2.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.2.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_2.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.2.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_2.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.2.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 2 Attention Block* | | | | |
| `params.params.decoder.layers_2.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.2.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.q_norm.scale` | `(1024,)` | `model.layers.2.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_2.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.2.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_2.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.2.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_2.self_attention.kv_norm.scale` | `(512,)` | `model.layers.2.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_2.self_attention.sinks` | `(64,)` | `model.layers.2.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_2.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.2.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_2.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.2.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| *Layer 2 CSA Compressor* | | | | |
| `params.params.decoder.layers_2.self_attention.csa_compressor.kv_proj.kernel` | `(4096, 1024)` | `model.layers.2.self_attn.compressor.kv_proj.weight` | `[1024, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.gate_proj.kernel` | `(4096, 1024)` | `model.layers.2.self_attn.compressor.gate_proj.weight` | `[1024, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.position_bias` | `(4, 1024)` | `model.layers.2.self_attn.compressor.position_bias` | `[4, 1024]` | Direct copy. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.kv_norm.scale` | `(512,)` | `model.layers.2.self_attn.compressor.kv_norm.weight` | `[512]` | Direct copy. |
| *Layer 2 CSA Indexer* | | | | |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.gate_proj.kernel` | `(4096, 256)` | `model.layers.2.self_attn.compressor.indexer.gate_proj.weight` | `[256, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.kv_proj.kernel` | `(4096, 256)` | `model.layers.2.self_attn.compressor.indexer.kv_proj.weight` | `[256, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.q_proj.kernel` | `(1024, 8192)` | `model.layers.2.self_attn.compressor.indexer.q_b_proj.weight` | `[8192, 1024]` | Transpose: `q_proj.kernel` = `q_b_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.weights_proj.kernel` | `(4096, 64)` | `model.layers.2.self_attn.compressor.indexer.scorer.weights_proj.weight` | `N/A` | Transpose: `weights_proj.kernel` = `weights_proj.weight.T`. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.position_bias` | `(4, 256)` | `model.layers.2.self_attn.compressor.indexer.position_bias` | `[4, 256]` | Direct copy. |
| `params.params.decoder.layers_2.self_attention.csa_compressor.indexer.kv_norm.scale` | `(128,)` | `model.layers.2.self_attn.compressor.indexer.kv_norm.weight` | `[128]` | Direct copy. |
| **Layer 3 (Type: HCA)** | | | | |
| `params.params.decoder.layers_3.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.3.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_3.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.3.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 3 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_3.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_3.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_3.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.3.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_3.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.3.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_3.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.3.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_3.mhc_attention.pre_beta` | `(4,)` | `model.layers.3.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_3.mhc_attention.post_beta` | `(4,)` | `model.layers.3.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_3.mhc_attention.res_beta` | `(4, 4)` | `model.layers.3.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_3.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.3.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_3.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.3.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_3.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.3.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_3.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.3.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_3.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.3.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_3.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.3.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_3.mhc_mlp.pre_beta` | `(4,)` | `model.layers.3.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_3.mhc_mlp.post_beta` | `(4,)` | `model.layers.3.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_3.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.3.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_3.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.3.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_3.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.3.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_3.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.3.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 3 MoE MLP Block* | | | | |
| `params.params.decoder.layers_3.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.3.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_3.mlp.MoeBlock_0.gate.bias` | `N/A` | `model.layers.3.mlp.gate.e_score_correction_bias` | `[8]` | Direct copy. *Note: requires setting `routed_bias: true` in MaxText.* |
| `params.params.decoder.layers_3.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.3.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_3.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.3.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_3.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.3.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_3.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.3.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_3.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.3.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_3.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.3.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 3 Attention Block* | | | | |
| `params.params.decoder.layers_3.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.3.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_3.self_attention.q_norm.scale` | `(1024,)` | `model.layers.3.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_3.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.3.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_3.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.3.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_3.self_attention.kv_norm.scale` | `(512,)` | `model.layers.3.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_3.self_attention.sinks` | `(64,)` | `model.layers.3.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_3.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.3.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_3.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.3.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| *Layer 3 HCA Compressor* | | | | |
| `params.params.decoder.layers_3.self_attention.hca_compressor.kv_proj.kernel` | `(4096, 512)` | `model.layers.3.self_attn.compressor.kv_proj.weight` | `[512, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_3.self_attention.hca_compressor.gate_proj.kernel` | `(4096, 512)` | `model.layers.3.self_attn.compressor.gate_proj.weight` | `[512, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_3.self_attention.hca_compressor.position_bias` | `(128, 512)` | `model.layers.3.self_attn.compressor.position_bias` | `[128, 512]` | Direct copy. |
| `params.params.decoder.layers_3.self_attention.hca_compressor.kv_norm.scale` | `(512,)` | `model.layers.3.self_attn.compressor.kv_norm.weight` | `[512]` | Direct copy. |
| **Layer 4 (Type: CSA)** | | | | |
| `params.params.decoder.layers_4.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.4.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_4.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.4.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 4 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_4.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_4.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_4.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.4.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_4.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.4.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_4.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.4.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_4.mhc_attention.pre_beta` | `(4,)` | `model.layers.4.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_4.mhc_attention.post_beta` | `(4,)` | `model.layers.4.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_4.mhc_attention.res_beta` | `(4, 4)` | `model.layers.4.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_4.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.4.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_4.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.4.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_4.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.4.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_4.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.4.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_4.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.4.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_4.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.4.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_4.mhc_mlp.pre_beta` | `(4,)` | `model.layers.4.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_4.mhc_mlp.post_beta` | `(4,)` | `model.layers.4.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_4.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.4.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_4.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.4.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_4.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.4.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_4.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.4.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 4 MoE MLP Block* | | | | |
| `params.params.decoder.layers_4.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.4.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_4.mlp.MoeBlock_0.gate.bias` | `N/A` | `model.layers.4.mlp.gate.e_score_correction_bias` | `[8]` | Direct copy. *Note: requires setting `routed_bias: true` in MaxText.* |
| `params.params.decoder.layers_4.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.4.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_4.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.4.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_4.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.4.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_4.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.4.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_4.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.4.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_4.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.4.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 4 Attention Block* | | | | |
| `params.params.decoder.layers_4.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.4.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.q_norm.scale` | `(1024,)` | `model.layers.4.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_4.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.4.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_4.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.4.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_4.self_attention.kv_norm.scale` | `(512,)` | `model.layers.4.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_4.self_attention.sinks` | `(64,)` | `model.layers.4.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_4.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.4.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_4.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.4.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| *Layer 4 CSA Compressor* | | | | |
| `params.params.decoder.layers_4.self_attention.csa_compressor.kv_proj.kernel` | `(4096, 1024)` | `model.layers.4.self_attn.compressor.kv_proj.weight` | `[1024, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.gate_proj.kernel` | `(4096, 1024)` | `model.layers.4.self_attn.compressor.gate_proj.weight` | `[1024, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.position_bias` | `(4, 1024)` | `model.layers.4.self_attn.compressor.position_bias` | `[4, 1024]` | Direct copy. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.kv_norm.scale` | `(512,)` | `model.layers.4.self_attn.compressor.kv_norm.weight` | `[512]` | Direct copy. |
| *Layer 4 CSA Indexer* | | | | |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.gate_proj.kernel` | `(4096, 256)` | `model.layers.4.self_attn.compressor.indexer.gate_proj.weight` | `[256, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.kv_proj.kernel` | `(4096, 256)` | `model.layers.4.self_attn.compressor.indexer.kv_proj.weight` | `[256, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.q_proj.kernel` | `(1024, 8192)` | `model.layers.4.self_attn.compressor.indexer.q_b_proj.weight` | `[8192, 1024]` | Transpose: `q_proj.kernel` = `q_b_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.weights_proj.kernel` | `(4096, 64)` | `model.layers.4.self_attn.compressor.indexer.scorer.weights_proj.weight` | `N/A` | Transpose: `weights_proj.kernel` = `weights_proj.weight.T`. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.position_bias` | `(4, 256)` | `model.layers.4.self_attn.compressor.indexer.position_bias` | `[4, 256]` | Direct copy. |
| `params.params.decoder.layers_4.self_attention.csa_compressor.indexer.kv_norm.scale` | `(128,)` | `model.layers.4.self_attn.compressor.indexer.kv_norm.weight` | `[128]` | Direct copy. |
| **Layer 5 (Type: HCA)** | | | | |
| `params.params.decoder.layers_5.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.5.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_5.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.5.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 5 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_5.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_5.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_5.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.5.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_5.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.5.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_5.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.5.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_5.mhc_attention.pre_beta` | `(4,)` | `model.layers.5.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_5.mhc_attention.post_beta` | `(4,)` | `model.layers.5.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_5.mhc_attention.res_beta` | `(4, 4)` | `model.layers.5.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_5.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.5.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_5.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.5.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_5.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.5.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_5.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.5.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_5.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.5.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_5.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.5.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_5.mhc_mlp.pre_beta` | `(4,)` | `model.layers.5.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_5.mhc_mlp.post_beta` | `(4,)` | `model.layers.5.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_5.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.5.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_5.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.5.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_5.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.5.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_5.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.5.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 5 MoE MLP Block* | | | | |
| `params.params.decoder.layers_5.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.5.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_5.mlp.MoeBlock_0.gate.bias` | `N/A` | `model.layers.5.mlp.gate.e_score_correction_bias` | `[8]` | Direct copy. *Note: requires setting `routed_bias: true` in MaxText.* |
| `params.params.decoder.layers_5.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.5.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_5.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.5.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_5.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.5.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_5.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.5.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_5.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.5.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_5.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.5.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 5 Attention Block* | | | | |
| `params.params.decoder.layers_5.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.5.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_5.self_attention.q_norm.scale` | `(1024,)` | `model.layers.5.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_5.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.5.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_5.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.5.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_5.self_attention.kv_norm.scale` | `(512,)` | `model.layers.5.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_5.self_attention.sinks` | `(64,)` | `model.layers.5.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_5.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.5.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_5.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.5.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| *Layer 5 HCA Compressor* | | | | |
| `params.params.decoder.layers_5.self_attention.hca_compressor.kv_proj.kernel` | `(4096, 512)` | `model.layers.5.self_attn.compressor.kv_proj.weight` | `[512, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_5.self_attention.hca_compressor.gate_proj.kernel` | `(4096, 512)` | `model.layers.5.self_attn.compressor.gate_proj.weight` | `[512, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_5.self_attention.hca_compressor.position_bias` | `(128, 512)` | `model.layers.5.self_attn.compressor.position_bias` | `[128, 512]` | Direct copy. |
| `params.params.decoder.layers_5.self_attention.hca_compressor.kv_norm.scale` | `(512,)` | `model.layers.5.self_attn.compressor.kv_norm.weight` | `[512]` | Direct copy. |
| **Layer 6 (Type: CSA)** | | | | |
| `params.params.decoder.layers_6.pre_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.6.input_layernorm.weight` | `[4096]` | Direct copy. |
| `params.params.decoder.layers_6.post_self_attention_layer_norm.scale` | `(4096,)` | `model.layers.6.post_attention_layernorm.weight` | `[4096]` | Direct copy. |
| *Layer 6 mHC (Manifold Constrained Hyper Connections)* | | | | |
| `params.params.decoder.layers_6.mhc_attention.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_6.mhc_mlp.mhc_norm.scale` | `(16384,)` | `N/A` | `N/A` | Unweighted in HF. MaxText parameter is initialized to all `1.0`s. |
| `params.params.decoder.layers_6.mhc_attention.pre_alpha` | `(16384, 4)` | `model.layers.6.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_6.mhc_attention.post_alpha` | `(16384, 4)` | `model.layers.6.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_6.mhc_attention.res_alpha` | `(16384, 16)` | `model.layers.6.attn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_6.mhc_attention.pre_beta` | `(4,)` | `model.layers.6.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_6.mhc_attention.post_beta` | `(4,)` | `model.layers.6.attn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_6.mhc_attention.res_beta` | `(4, 4)` | `model.layers.6.attn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_6.mhc_attention.pre_alpha_scale` | `(1,)` | `model.layers.6.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_6.mhc_attention.post_alpha_scale` | `(1,)` | `model.layers.6.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_6.mhc_attention.res_alpha_scale` | `(1,)` | `model.layers.6.attn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| `params.params.decoder.layers_6.mhc_mlp.pre_alpha` | `(16384, 4)` | `model.layers.6.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[:4]` and transpose. |
| `params.params.decoder.layers_6.mhc_mlp.post_alpha` | `(16384, 4)` | `model.layers.6.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[4:8]` and transpose. |
| `params.params.decoder.layers_6.mhc_mlp.res_alpha` | `(16384, 16)` | `model.layers.6.ffn_hc.fn` | `[24, 16384]` | Split & Transpose: Slice HF `fn` at `[8:]` and transpose. |
| `params.params.decoder.layers_6.mhc_mlp.pre_beta` | `(4,)` | `model.layers.6.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[:4]`. |
| `params.params.decoder.layers_6.mhc_mlp.post_beta` | `(4,)` | `model.layers.6.ffn_hc.base` | `[24]` | Split: Slice HF `base` at `[4:8]`. |
| `params.params.decoder.layers_6.mhc_mlp.res_beta` | `(4, 4)` | `model.layers.6.ffn_hc.base` | `[24]` | Split & Reshape: Slice HF `base` at `[8:]` and reshape from `(16,)` to `(4, 4)`. |
| `params.params.decoder.layers_6.mhc_mlp.pre_alpha_scale` | `(1,)` | `model.layers.6.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[0]`. |
| `params.params.decoder.layers_6.mhc_mlp.post_alpha_scale` | `(1,)` | `model.layers.6.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[1]`. |
| `params.params.decoder.layers_6.mhc_mlp.res_alpha_scale` | `(1,)` | `model.layers.6.ffn_hc.scale` | `[3]` | Split: Slice HF `scale` at `[2]`. |
| *Layer 6 MoE MLP Block* | | | | |
| `params.params.decoder.layers_6.mlp.MoeBlock_0.gate.kernel` | `(4096, 8)` | `model.layers.6.mlp.gate.weight` | `[8, 4096]` | Transpose: `gate.kernel` = `gate.weight.T`. |
| `params.params.decoder.layers_6.mlp.MoeBlock_0.gate.bias` | `N/A` | `model.layers.6.mlp.gate.e_score_correction_bias` | `[8]` | Direct copy. *Note: requires setting `routed_bias: true` in MaxText.* |
| `params.params.decoder.layers_6.mlp.shared_experts.wi_0.kernel` | `(4096, 2048)` | `model.layers.6.mlp.shared_experts.gate_proj.weight` | `[2048, 4096]` | Transpose: `wi_0.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_6.mlp.shared_experts.wi_1.kernel` | `(4096, 2048)` | `model.layers.6.mlp.shared_experts.up_proj.weight` | `[2048, 4096]` | Transpose: `wi_1.kernel` = `up_proj.weight.T`. |
| `params.params.decoder.layers_6.mlp.shared_experts.wo.kernel` | `(2048, 4096)` | `model.layers.6.mlp.shared_experts.down_proj.weight` | `[4096, 2048]` | Transpose: `wo.kernel` = `down_proj.weight.T`. |
| `params.params.decoder.layers_6.mlp.MoeBlock_0.wi_0` | `(8, 4096, 2048)` | `model.layers.6.mlp.experts..{0..7}.w1.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_0[E] = experts..{E}.w1.weight.T`. |
| `params.params.decoder.layers_6.mlp.MoeBlock_0.wi_1` | `(8, 4096, 2048)` | `model.layers.6.mlp.experts..{0..7}.w3.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wi_1[E] = experts..{E}.w3.weight.T`. |
| `params.params.decoder.layers_6.mlp.MoeBlock_0.wo` | `(8, 2048, 4096)` | `model.layers.6.mlp.experts..{0..7}.w2.weight` | `N/A` | Stack & Transpose: Stack weights of the 8 experts along dimension 0 and transpose: `MoeBlock_0.wo[E] = experts..{E}.w2.weight.T`. |
| *Layer 6 Attention Block* | | | | |
| `params.params.decoder.layers_6.self_attention.wq_a.kernel` | `(4096, 1024)` | `model.layers.6.self_attn.q_a_proj.weight` | `[1024, 4096]` | Transpose: `wq_a.kernel` = `q_a_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.q_norm.scale` | `(1024,)` | `model.layers.6.self_attn.q_a_norm.weight` | `[1024]` | Direct copy. |
| `params.params.decoder.layers_6.self_attention.wq_b.kernel` | `(1024, 64, 512)` | `model.layers.6.self_attn.q_b_proj.weight` | `[32768, 1024]` | Transpose & Reshape: Transpose and reshape `q_b_proj.weight` from `[32768, 1024]` to `(1024, 64, 512)`. |
| `params.params.decoder.layers_6.self_attention.wkv.kernel` | `(4096, 1, 512)` | `model.layers.6.self_attn.kv_proj.weight` | `[512, 4096]` | Transpose & Reshape: Transpose and reshape `kv_proj.weight` from `[512, 4096]` to `(4096, 1, 512)`. |
| `params.params.decoder.layers_6.self_attention.kv_norm.scale` | `(512,)` | `model.layers.6.self_attn.kv_norm.weight` | `[512]` | Direct copy. |
| `params.params.decoder.layers_6.self_attention.sinks` | `(64,)` | `model.layers.6.self_attn.sinks` | `[64]` | Direct copy (reshaped from `[64]` to `(64,)`). |
| `params.params.decoder.layers_6.self_attention.o_a_proj.kernel` | `(8, 4096, 1024)` | `model.layers.6.self_attn.o_a_proj.weight` | `[8192, 4096]` | Reshape & Transpose: Reshape `o_a_proj.weight` from `[8192, 4096]` to `(8, 1024, 4096)` and transpose `(0, 2, 1)` to `(8, 4096, 1024)`. |
| `params.params.decoder.layers_6.self_attention.o_b_proj.kernel` | `(8192, 4096)` | `model.layers.6.self_attn.o_b_proj.weight` | `[4096, 8192]` | Transpose: `o_b_proj.kernel` = `o_b_proj.weight.T`. |
| *Layer 6 CSA Compressor* | | | | |
| `params.params.decoder.layers_6.self_attention.csa_compressor.kv_proj.kernel` | `(4096, 1024)` | `model.layers.6.self_attn.compressor.kv_proj.weight` | `[1024, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.gate_proj.kernel` | `(4096, 1024)` | `model.layers.6.self_attn.compressor.gate_proj.weight` | `[1024, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.position_bias` | `(4, 1024)` | `model.layers.6.self_attn.compressor.position_bias` | `[4, 1024]` | Direct copy. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.kv_norm.scale` | `(512,)` | `model.layers.6.self_attn.compressor.kv_norm.weight` | `[512]` | Direct copy. |
| *Layer 6 CSA Indexer* | | | | |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.gate_proj.kernel` | `(4096, 256)` | `model.layers.6.self_attn.compressor.indexer.gate_proj.weight` | `[256, 4096]` | Transpose: `gate_proj.kernel` = `gate_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.kv_proj.kernel` | `(4096, 256)` | `model.layers.6.self_attn.compressor.indexer.kv_proj.weight` | `[256, 4096]` | Transpose: `kv_proj.kernel` = `kv_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.q_proj.kernel` | `(1024, 8192)` | `model.layers.6.self_attn.compressor.indexer.q_b_proj.weight` | `[8192, 1024]` | Transpose: `q_proj.kernel` = `q_b_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.weights_proj.kernel` | `(4096, 64)` | `model.layers.6.self_attn.compressor.indexer.scorer.weights_proj.weight` | `N/A` | Transpose: `weights_proj.kernel` = `weights_proj.weight.T`. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.position_bias` | `(4, 256)` | `model.layers.6.self_attn.compressor.indexer.position_bias` | `[4, 256]` | Direct copy. |
| `params.params.decoder.layers_6.self_attention.csa_compressor.indexer.kv_norm.scale` | `(128,)` | `model.layers.6.self_attn.compressor.indexer.kv_norm.weight` | `[128]` | Direct copy. |
