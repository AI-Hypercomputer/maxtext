# DeepSeek-V4 Parameter Mapping Configuration
# Generated programmatically for tiny variant (7 layers, 8 experts)
#
# Use this dictionary to map MaxText parameters to HuggingFace parameters.
# Each entry is: MaxText Key -> (HuggingFace Key, Transformation Rule)

# Conversion Rules:
# - "direct": Direct shape-matched copy.
# - "transpose": MaxText parameter = HF parameter transposed (.T).
# - "stack_transpose": MaxText parameter = Stack of E experts transposed (e.g. MoeBlock).
# - "mhc_fn_pre", "mhc_fn_post", "mhc_fn_res": Split attn_hc.fn/ffn_hc.fn and transpose.
# - "mhc_base_pre", "mhc_base_post", "mhc_base_res": Split attn_hc.base/ffn_hc.base.
# - "mhc_scale_pre", "mhc_scale_post", "mhc_scale_res": Split scale parameters.
# - "ones": Constant initialization (HF does not have parameter).

PARAM_MAPPING = {
    # === Global / Embeddings ===
    "params.token_embedder.embedding": ("model.embed_tokens.weight", "direct"),
    "params.decoder.decoder_norm.scale": ("model.norm.weight", "direct"),
    "params.decoder.logits_dense.kernel": ("head.weight", "transpose"),
}

# Add layer-specific mappings programmatically for layers 0 to 6
for l in range(7):
    # Common norms
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.pre_self_attention_layer_norm.scale"] = (f"model.layers.{l}.input_layernorm.weight", "direct")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.post_self_attention_layer_norm.scale"] = (f"model.layers.{l}.post_attention_layernorm.weight", "direct")

    # mHC Attention
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.mhc_norm.scale"] = (None, "ones")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.pre_alpha"] = (f"model.layers.{l}.attn_hc.fn", "mhc_fn_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.post_alpha"] = (f"model.layers.{l}.attn_hc.fn", "mhc_fn_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.res_alpha"] = (f"model.layers.{l}.attn_hc.fn", "mhc_fn_res")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.pre_beta"] = (f"model.layers.{l}.attn_hc.base", "mhc_base_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.post_beta"] = (f"model.layers.{l}.attn_hc.base", "mhc_base_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.res_beta"] = (f"model.layers.{l}.attn_hc.base", "mhc_base_res")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.pre_alpha_scale"] = (f"model.layers.{l}.attn_hc.scale", "mhc_scale_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.post_alpha_scale"] = (f"model.layers.{l}.attn_hc.scale", "mhc_scale_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_attention.res_alpha_scale"] = (f"model.layers.{l}.attn_hc.scale", "mhc_scale_res")

    # mHC MLP
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.mhc_norm.scale"] = (None, "ones")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.pre_alpha"] = (f"model.layers.{l}.ffn_hc.fn", "mhc_fn_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.post_alpha"] = (f"model.layers.{l}.ffn_hc.fn", "mhc_fn_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.res_alpha"] = (f"model.layers.{l}.ffn_hc.fn", "mhc_fn_res")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.pre_beta"] = (f"model.layers.{l}.ffn_hc.base", "mhc_base_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.post_beta"] = (f"model.layers.{l}.ffn_hc.base", "mhc_base_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.res_beta"] = (f"model.layers.{l}.ffn_hc.base", "mhc_base_res")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.pre_alpha_scale"] = (f"model.layers.{l}.ffn_hc.scale", "mhc_scale_pre")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.post_alpha_scale"] = (f"model.layers.{l}.ffn_hc.scale", "mhc_scale_post")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mhc_mlp.res_alpha_scale"] = (f"model.layers.{l}.ffn_hc.scale", "mhc_scale_res")

    # MoE Block
    if l < 3:
        PARAM_MAPPING[f"params.Tid2EidVar.decoder.layers_{l}.mlp.MoeBlock_0.tid2eid"] = (f"model.layers.{l}.mlp.gate.tid2eid", "direct")
    
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.MoeBlock_0.gate.kernel"] = (f"model.layers.{l}.mlp.gate.weight", "transpose")
    if l >= 3:
        PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.MoeBlock_0.gate.bias"] = (f"model.layers.{l}.mlp.gate.e_score_correction_bias", "direct")
        
    # Shared Experts
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.shared_experts.wi_0.kernel"] = (f"model.layers.{l}.mlp.shared_experts.gate_proj.weight", "transpose")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.shared_experts.wi_1.kernel"] = (f"model.layers.{l}.mlp.shared_experts.up_proj.weight", "transpose")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.shared_experts.wo.kernel"] = (f"model.layers.{l}.mlp.shared_experts.down_proj.weight", "transpose")

    # Stacked Experts
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.MoeBlock_0.wi_0"] = ([f"model.layers.{l}.mlp.experts..{e}.w1.weight" for e in range(8)], "stack_transpose")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.MoeBlock_0.wi_1"] = ([f"model.layers.{l}.mlp.experts..{e}.w3.weight" for e in range(8)], "stack_transpose")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.mlp.MoeBlock_0.wo"] = ([f"model.layers.{l}.mlp.experts..{e}.w2.weight" for e in range(8)], "stack_transpose")

    # Attention
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.wq_a.kernel"] = (f"model.layers.{l}.self_attn.q_a_proj.weight", "transpose")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.q_norm.scale"] = (f"model.layers.{l}.self_attn.q_a_norm.weight", "direct")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.wq_b.kernel"] = (f"model.layers.{l}.self_attn.q_b_proj.weight", "transpose_reshape_q")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.wkv.kernel"] = (f"model.layers.{l}.self_attn.kv_proj.weight", "transpose_reshape_kv")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.kv_norm.scale"] = (f"model.layers.{l}.self_attn.kv_norm.weight", "direct")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.sinks"] = (f"model.layers.{l}.self_attn.sinks", "direct")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.o_a_proj.kernel"] = (f"model.layers.{l}.self_attn.o_a_proj.weight", "reshape_transpose_oa")
    PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.o_b_proj.kernel"] = (f"model.layers.{l}.self_attn.o_b_proj.weight", "transpose")

    # Attention Compressor
    if l >= 2:
        if l % 2 == 0 or l == 2:
            # CSA
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.kv_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.kv_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.gate_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.gate_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.position_bias"] = (f"model.layers.{l}.self_attn.compressor.position_bias", "direct")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.kv_norm.scale"] = (f"model.layers.{l}.self_attn.compressor.kv_norm.weight", "direct")
            
            # CSA Indexer
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.gate_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.indexer.gate_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.kv_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.indexer.kv_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.q_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.indexer.q_b_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.weights_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.indexer.scorer.weights_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.position_bias"] = (f"model.layers.{l}.self_attn.compressor.indexer.position_bias", "direct")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.csa_compressor.indexer.kv_norm.scale"] = (f"model.layers.{l}.self_attn.compressor.indexer.kv_norm.weight", "direct")
        else:
            # HCA
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.hca_compressor.kv_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.kv_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.hca_compressor.gate_proj.kernel"] = (f"model.layers.{l}.self_attn.compressor.gate_proj.weight", "transpose")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.hca_compressor.position_bias"] = (f"model.layers.{l}.self_attn.compressor.position_bias", "direct")
            PARAM_MAPPING[f"params.params.decoder.layers_{l}.self_attention.hca_compressor.kv_norm.scale"] = (f"model.layers.{l}.self_attn.compressor.kv_norm.weight", "direct")

