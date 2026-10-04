# Model Onboarding Handover: `convaiinnovations/laya`

## Status Summary
- **Phase 1 (HF Architecture Analysis & Mini Safetensors Subset)**: `DONE`
- **Phase 2 (JAX/NNX Layer Implementation, Configs, & Unit Tests)**: `DONE`
- **Phase 3 (Bidirectional Checkpoint Conversion `to_maxtext.py` & `to_huggingface.py`)**: `DONE`
- **Phase 4 (E2E Logits Verification & Full TPU Golden `laya.py` Verification)**: `DONE`

---

## 1. Architecture & Checkpoint Overview
- **HuggingFace Model ID**: `convaiinnovations/laya` (bundles `english` at root, `multilingual/` subfolder, and `typed-decisions/` subfolder)
- **Golden Example**: `/home/hengtaoguo_google_com/projects/laya.py`
  - Outputs:
    - `result["answers"]["department"]["choice"] == "billing"`
    - `result["answers"]["churn_risk"]["noul"] == 0.879`
    - `result["routing"]["model"] == "english"`
- **Architecture (`laya.common.DecisionModel`)**:
  1. **Encoder (`ModernBertModel`)**:
     - `english` (`laya`) & `typed-decisions` (`laya-typed-decisions`):
       - `hidden_size = 1024`, `num_hidden_layers = 28`, `num_attention_heads = 16`, `head_dim = 64`
       - `intermediate_size = 2624` (GeGLU: `Wi` projects `1024 -> 5248`, split in half into `gate` (`gelu_exact`) and `val`, `Wo` projects `2624 -> 1024`)
       - `vocab_size = 50368`, `norm_eps = 1e-5`, `norm_bias = False`, `attention_bias = False`, `mlp_bias = False`
       - **Layer 0 special case**: `encoder.layers.0` has NO `attn_norm` (`Identity`), only ` embeddings.norm` before layer 0 and `mlp_norm` inside layer 0. Layers `1..27` have both `attn_norm` and `mlp_norm`.
       - **Alternating Full / Sliding Bidirectional Attention**:
         - `global_attn_every_n_layers = 3`: layers `0, 3, 6, ..., 27` use full bidirectional attention with RoPE `theta = 160000.0`.
         - Other layers use bidirectional sliding-window attention (`local_attention = 128`, attending within `|i - j| <= 64`) with RoPE `theta = 10000.0`.
     - `multilingual` (`laya-multilingual`):
       - `hidden_size = 768`, `num_hidden_layers = 22`, `num_attention_heads = 12`, `head_dim = 64`, `intermediate_size = 1152` (`Wi` projects `768 -> 2304`), `vocab_size = 256000`.
  2. **Decision Head (`LayaDecisionHead`)**:
     - `type_emb`: `Embed(3, D)` added to encoder `last_hidden_state` by question type (`choice=0, score=1, noul=2`).
     - `head.layers.{0,1}` (`LayaHeadTransformerLayer`): 2 pre-norm bidirectional TransformerEncoderLayers (`d_model=D, nhead=D//64, dim_feedforward=4*D, activation=relu, norm_first=True, norm_eps=1e-5`, with biases on norms, `in_proj`, `out_proj`, `linear1`, `linear2`).
     - `scorer`: `LayerNorm(D, eps=1e-5, bias=True) -> Linear(D, D, bias=True) -> GELU -> Linear(D, 1, bias=True)` applied at gathered `[MASK]` option marker positions (`marker_pos`), masked with `-1e4` where `~marker_mask`.
     - `act_head`: `Linear(D + 4, 256, bias=True) -> GELU -> Linear(256, 2, bias=True)` taking `[h[:, 0], top1_prob, top1_minus_top2_prob, normalized_entropy, num_options / 255.0]`.
     - `temperature`: shape `(3,)` buffer.

---

## 2. Files Added / Modified in MaxText (`/home/hengtaoguo_google_com/projects/maxtext`)
1. `src/maxtext/models/laya.py`:
   - `LayaLayerNorm`, `LayaMLP`, `LayaAttention`, `LayaDecoderLayer`, `LayaHeadTransformerLayer`, `LayaDecisionHead`.
   - Uses explicit `precision=lax.Precision(self.config.matmul_precision)` in `jnp.einsum` for full float32 accuracy on TPU.
2. `src/maxtext/common/common_types.py` & `src/maxtext/configs/types.py`:
   - Added `DecoderBlockType.LAYA = "laya"` and registered `"laya"`, `"laya-multilingual"`, `"laya-typed-decisions"` in `ModelName`.
3. `src/maxtext/layers/nnx_decoders.py`:
   - Wired `DecoderBlockType.LAYA` embedding `LayerNorm` (`encoder_embed_norm`), decoder layers (`LayaDecoderLayer` with `layer_idx`), `decoder_norm` (`LayaLayerNorm`), and `decision_head` (`LayaDecisionHead`), plus sowing `hidden_states` in `NNXDecoder.__call__`.
4. `src/maxtext/configs/models/laya.yml`, `laya-multilingual.yml`, `laya-typed-decisions.yml`:
   - Model YAML configs for all three Laya checkpoints.
5. `src/maxtext/utils/globals.py`, `src/maxtext/checkpoint_conversion/utils/hf_model_configs.py`, `src/maxtext/checkpoint_conversion/utils/hf_shape.py`, `src/maxtext/checkpoint_conversion/utils/param_mapping.py`, `src/maxtext/checkpoint_conversion/utils/utils.py`, `src/maxtext/checkpoint_conversion/to_huggingface.py`:
   - Registered `HF_IDS`, `HF_MODEL_CONFIGS`, `LAYA_HF_WEIGHTS_TO_SHAPE`, `LAYA_MAXTEXT_TO_HF_PARAM_MAPPING`, and `LAYA_MAXTEXT_TO_HF_PARAM_HOOK_FN` for bidirectional checkpoint conversion (`to_maxtext.py` & `to_huggingface.py`).
6. `tests/unit/laya_layers_test.py`:
   - Unit and 4-layer E2E tests (`test_layer_norm`, `test_mlp_block`, `test_decoder_layers_full_and_sliding`, `test_4layer_encoder_and_decision_head_e2e`) — all 4 tests pass (`OK`).
7. `tests/utils/forward_pass_logit_checker.py`:
   - Added `_LayaHFLogitsWrapper` and tokenizer subfolder fallback for on-the-fly E2E logit comparison on short and ~493-token long prompts.
8. `src/maxtext/inference/laya.py`:
   - MaxText TPU runtime (`Agent`, `RLAgent`, `Router`, `load`) and `main()` running the golden `/home/hengtaoguo_google_com/projects/laya.py` example on TPU.

---

## 3. Verification Results on TPU VM (`v5p2`)
- **Layer & 4-Layer E2E Unit Tests (`tests/unit/laya_layers_test.py`)**:
  - `Ran 4 tests in 33.495s — OK` (`rtol=1e-4, atol=1e-4`).
- **Bidirectional Checkpoint Conversion (`to_maxtext.py` & `to_huggingface.py`)**:
  - Mini 4-layer checkpoint (`/dev/shm/hf_mini/laya_4layers` -> `/dev/shm/hengtaoguo/checkpoints/laya_mini_orbax`): `OK`.
  - Full 28-layer checkpoint (`/dev/shm/hengtaoguo/laya_full_hf` -> `/dev/shm/hengtaoguo/checkpoints/laya_orbax` with `--save_dtype=float32`): `OK`.
  - Roundtrip `to_huggingface.py` (`/dev/shm/hengtaoguo/checkpoints/laya_orbax/0/items` -> `/dev/shm/hengtaoguo/checkpoints/laya_hf_roundtrip`): `206 keys, max_abs_diff = 0.00e+00`.
- **Forward Pass Logit Checker (`tests/utils/forward_pass_logit_checker.py`)**:
  - 4-layer mini checkpoint (`--max_kl_div=1e-3`): `PASS` (short prompts max KL `1.33e-06`, ~493-token long prompt max KL `4.09e-04`, top-10 overlap `10/10`, rank agreement `100.0%`).
  - Full 28-layer checkpoint (`--max_kl_div=1e-3`): `PASS` (max KL across all prompts including ~493-token long prompt `2.86e-04 < 1e-3`, top-10 overlap `10/10`, rank agreement `100.0%`).
- **Golden `laya.py` Verification on TPU (`python -m maxtext.inference.laya`)**:
  - `department.choice == "billing"`
  - `churn_risk.noul == 0.879`
  - `routing.model == "english"`
