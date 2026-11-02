# OpenVLA-7B (`openvla/openvla-7b`) Bring-up Handover Document

Standard MaxText format (no M3): `model_name=openvla-7b`, `decoder_block: llama2`, `vision_encoder_block: openvla`.

## 1. HF Model Architectural Findings

- **HuggingFace Model ID**: `openvla/openvla-7b` (remote code `modeling_prismatic.py`, written for
  `transformers==4.40.1`, `timm==0.9.10`; does NOT run on the transformers 5.x in `venv1`).
- **Official example (golden reference)** — HF model card:
  `prompt = "In: What action should the robot take to {instruction}?\nOut:"`,
  `inputs = processor(prompt, image)`, `vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)`.
  `predict_action` appends token `29871` ("") after `Out:`, greedily generates `action_dim=7` tokens, maps
  token `t` -> bin `32000 - t - 1` of 256 uniform bins in [-1, 1], and unnormalizes with `norm_stats[unnorm_key]` (q01/q99).
- **Vision backbone** (`dinosiglip-vit-so-224px`, fused): two timm ViTs on the same 224x224 image
  (preprocessed twice, channel-stacked into 6 channels):
  - DINOv2 ViT-L/14 reg4 (`vit_large_patch14_reg4_dinov2.lvd142m`): 1024 dim, 16 heads, mlp 4096, 24 blocks,
    cls + 4 register tokens (pos-embed added to patches *before* prefix tokens are prepended: `no_embed_class=True`),
    LayerScale (`ls1/ls2.scale_factor` — HF renames timm `gamma`), LayerNorm eps 1e-6, exact GELU.
  - SigLIP So400m/14 (`vit_so400m_patch14_siglip_224`): 1152 dim, 16 heads, mlp 4304, 27 blocks, no prefix tokens,
    exact GELU (timm default, NOT gelu_tanh), LayerNorm eps 1e-6.
  - Prismatic uses `get_intermediate_layers(n={len(blocks)-2})`: patch tokens of the **second-to-last** block,
    no final norm, no SigLIP attention-pool head. => only 23 / 26 blocks are used.
  - Output: concat [256, 1024] + [256, 1152] -> [256, 2176].
- **Projector** (`no-align+fused-gelu-mlp`): Linear(2176, 8704) -> GELU -> Linear(8704, 4096) -> GELU -> Linear(4096, 4096).
- **LLM**: Llama-2-7B, homogeneous 32 layers, MHA 32 heads, head_dim 128, mlp 11008 SiLU-GLU, RoPE theta 10000,
  **rms_norm_eps = 1e-6** (OpenVLA's `text_config` keeps `LlamaConfig` defaults; Llama-2-7b-hf uses 1e-5),
  vocab 32000 + `<PAD>`(32000) padded to **32064**, untied `lm_head`.
- **Multimodal fusion**: 256 projected patch embeddings inserted right after BOS (no placeholder tokens in HF);
  attention is plain causal over the whole sequence (no bidirectional image block).
- **Image preprocessing**: `resize-naive` bicubic resize to 224x224 (PIL), `/255`, normalize with the
  `tvf_normalize_params` from `preprocessor_config.json` (bf16-rounded: DINO mean `[0.484375, 0.455078125, 0.40625]`,
  std `[0.228515625, 0.2236328125, 0.224609375]`; SigLIP 0.5/0.5). These JSON values override the timm defaults.
- **Decoder pattern**: homogeneous -> minimal subset N = 2 LM layers (full vision towers kept):
  `/dev/shm/hf_mini/openvla-7b_2layers`.

## 2. Phase-by-Phase Status

- [x] Phase 1: HF Discovery & Mini Safetensors Creation (DONE)
- [x] Phase 2: JAX Layer Implementation & Layer Tests (DONE) — 8/8 tests pass (`tests/unit/openvla_layers_test.py`)
- [x] Phase 3: Checkpoint Mapping & Conversion, scanned + unscanned, mini + full (DONE)
- [x] Phase 4: Logits (scanned + unscanned) & Autoregressive action decode (DONE)

### Results (TPU v5p-2, image `tests/assets/test_image.jpg`, instruction "pick up the space needle")

| Check | Unscanned | Scanned |
| :--- | :--- | :--- |
| Mini (2-layer) logits fp32 vs HF fp32, 277 tokens | max KL 1.7e-7 (2.5e-4 before the precision fix) | max KL 2.5e-4 (before the fix; same mapping path) |
| Mini greedy decode fp32 | `[31892, 31925, 31780, 31995, 31906, 31822, 16113]` == HF | — |
| **Full logits fp32 vs HF fp32** (`--max_kl_div=1e-4`) | **PASS, max KL 3.3e-7**, max abs 8.2e-4, top-10 rank 100% | **PASS, max KL 4.1e-7**, max abs 8.2e-4, top-10 rank 100% |
| Full logits bf16 vs HF bf16 | max KL 0.26 (image positions) — bf16 noise, see below | max KL 0.25 |
| **Full greedy decode fp32 (7 action tokens)** | **`[31912, 31956, 31841, 31884, 31900, 31957, 31744]` == HF** | **== HF** |
| Full greedy decode bf16 | == HF | `[.., 31893, ..]` (1 of 7 tokens flips, adjacent bin) |

| MaxText bf16 vs HF **fp32** golden (text positions) | max KL 2.5e-2, last-token top-5 identical | max KL 2.5e-2, last-token top-5 identical |
| Actions from MaxText ids (`action_tokens_to_actions`, `bridge_orig`) | identical to HF `predict_action` | identical (fp32) |

HF reference (`predict_action`, fp32 and bf16 identical): ids `[31912, 31956, 31841, 31884, 31900, 31957, 31744]`,
actions (`bridge_orig`) `[-0.00916, -0.02762, 0.01509, -0.00716, -0.02624, -0.13868, 0.99608]`.

bf16 calibration: HF bf16 vs HF fp32 itself has max KL 0.47 on image positions and 0.019 on text positions, (MaxText bf16 vs HF fp32: 0.17 / 0.025 — same magnitude), so bf16
cross-framework logits comparisons are dominated by bf16 rounding; fp32 is the parity criterion.

## 3. Bugs Encountered & Solutions

- **HF reference (Phase 1)**: `attn_implementation="eager"` crashes in `predict_action` (`size of tensor a (277) must
  match b (276)`): `predict_action` appends 29871 to `input_ids` but not to `attention_mask`; the official example uses
  flash-attn, which ignores the mismatch. Fix: golden generation uses `sdpa` (all-ones mask -> pure causal).
- **HF env (Phase 1)**: separate venv `~/projects/venv_openvla` (python3.10, torch 2.2.0+cpu, torchvision 0.17.0,
  transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, numpy<2). Installing timm pulled a CUDA torch; reinstalled CPU wheels.
- **Phase 4 (full model)**: full fp32 logits initially failed (max KL 3e-3; vision features off by ~1% rel). Root cause:
  the ViT attention `jnp.einsum`s did not pass `precision=config.matmul_precision`, so on TPU they ran at default
  (bf16-pass) precision even with `matmul_precision=highest`. CPU unit tests could not catch this. Fix: pass
  `precision` to both einsums -> vision features rel err ~5e-6, logits KL 3e-7.
- **Placeholder collision (design)**: the inference tokenizer pads with `<PAD>`=32000, so the image soft-token
  placeholder uses the never-emitted padded-vocab id **32001**.

## 4. Pitfalls & Model-Specific Gotchas

- The LLM query kernel is pre-scaled by `1/sqrt(128)` in the checkpoint (MaxText `llama2` folds the attention scale
  into the weights). No RoPE permutation: MaxText default `RotaryEmbedding` is rotate-half like HF.
- `get_image_offsets` for OpenVLA returns `256*num_images + 1`: the extra 1 is the trailing empty token 29871 appended by
  `add_extra_tokens_for_images_openvla` (mirrors `predict_action`). `reformat_prompt_openvla` wraps a bare instruction
  in the template and right-strips it so the tokenized prompt never already ends in 29871.
- Decode: to get exactly the 7 action tokens set `max_target_length = max_prefill_predict_length + 6`
  (1 token from prefill + 6 generate steps). Raw ids are printed with `--verbosity=1`.
  `processor_openvla.action_tokens_to_actions(ids, norm_stats["bridge_orig"])` reproduces `predict_action`'s actions.
- Unused HF weights (not converted, never used by OpenVLA): DINOv2 block 23 + final norm, SigLIP block 26 + final norm +
  `attn_pool.*`. A MaxText->HF export would need to re-add them.
- Conversion requires `use_multimodal=true` and `--lazy_load_tensors=false` (multimodal). fp32 checkpoints are ~14 GB
  on disk each (orbax compression of bf16-origin fp32).
- Vision towers are unscanned (like gemma3); `scan_layers` only affects the 32 Llama layers.

## 5. Full Executable Reproduction Commands

Common env (TPU VM): `cd ~/projects/maxtext && export PYTHONPATH=src:. HF_HOME=/dev/shm/hengtaoguo HF_HUB_OFFLINE=1 TMPDIR=/dev/shm`

- **HF download**: `snapshot_download('openvla/openvla-7b', local_dir='/dev/shm/hengtaoguo/openvla/openvla-7b')`
- **Mini safetensors slicing**: `~/projects/venv_openvla/bin/python ~/projects/slice_openvla_7b.py --n_layers 2`
- **HF golden logits + decode** (official example flow, writes `.pkl`, `_decode.json`, `_feats.npz`):
  ```
  ~/projects/venv_openvla/bin/python ~/projects/gen_openvla_golden.py --tag full --dtype float32   # or bfloat16
  ~/projects/venv_openvla/bin/python ~/projects/gen_openvla_golden.py --model_path /dev/shm/hf_mini/openvla-7b_2layers --tag mini2
  ```
  -> `/dev/shm/hengtaoguo/openvla/golden/golden_openvla_{full,mini2}_{float32,bfloat16}*`
- **Checkpoint conversion** (`scan_layers=false` / `true`; mini: add `override_model_config=true base_num_decoder_layers=2 --hf_model_path=/dev/shm/hf_mini/openvla-7b_2layers`):
  ```
  JAX_PLATFORMS=cpu ../venv1/bin/python -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=openvla-7b use_multimodal=true scan_layers=false weight_dtype=float32 hardware=cpu skip_jax_distributed_system=true \
    base_output_directory=/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_unscanned \
    --hf_model_path=/dev/shm/hengtaoguo/openvla/openvla-7b --lazy_load_tensors=false --save_dtype=float32 --simulated_cpu_devices_count=1
  ```
  Outputs: `/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_unscanned/0/items`, `/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_scanned/0/items`.
- **Forward pass logits check**:
  ```
  ../venv1/bin/python -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml model_name=openvla-7b use_multimodal=true \
    load_parameters_path=/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_unscanned/0/items scan_layers=false \
    tokenizer_type=huggingface tokenizer_path=/dev/shm/hengtaoguo/openvla/openvla-7b per_device_batch_size=1 \
    max_prefill_predict_length=320 max_target_length=320 attention=dot_product skip_jax_distributed_system=true \
    dtype=float32 weight_dtype=float32 dtype_mm=float32 matmul_precision=highest float32_logits=true float32_qk_product=true \
    --golden_logits_path=/dev/shm/hengtaoguo/openvla/golden/golden_openvla_full_float32.pkl --max_kl_div=1e-4
  ```
- **Autoregressive action decode** (7 tokens):
  ```
  ../venv1/bin/python -m maxtext.inference.decode src/maxtext/configs/base.yml model_name=openvla-7b use_multimodal=true \
    load_parameters_path=/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_unscanned/0/items scan_layers=false \
    tokenizer_type=huggingface tokenizer_path=/dev/shm/hengtaoguo/openvla/openvla-7b \
    image_path=tests/assets/test_image.jpg "prompt=pick up the space needle" per_device_batch_size=1 \
    max_prefill_predict_length=320 max_target_length=326 attention=dot_product decode_sampling_strategy=greedy \
    dtype=float32 weight_dtype=float32 dtype_mm=float32 matmul_precision=highest float32_logits=true float32_qk_product=true \
    skip_jax_distributed_system=true --verbosity=1
  ```
- **Full verification suite** (logits fp32/bf16 + decode fp32/bf16, scanned & unscanned): `bash ~/projects/verify_openvla_full.sh`
- **Layer unit tests** (CPU; timm 1.0.15 installed to a side dir): 
  `PYTHONPATH=src:.:/dev/shm/hengtaoguo/openvla/pyextra JAX_PLATFORMS=cpu OPENVLA_GOLDEN_PKL=/dev/shm/hengtaoguo/openvla/golden/golden_openvla_full_float32.pkl ../venv1/bin/python -m pytest tests/unit/openvla_layers_test.py -o addopts=''`
- **Intermediate-feature debugging**: `../venv1/bin/python ../debug_openvla_parity.py <maxtext overrides> -- /dev/shm/hengtaoguo/openvla/golden/golden_openvla_full_float32`

## 6. Files Changed / Added (MaxText repo)

- New: `src/maxtext/models/openvla_vision.py`, `src/maxtext/multimodal/processor_openvla.py`,
  `src/maxtext/configs/models/openvla-7b.yml`, `tests/unit/openvla_layers_test.py`
- Modified: `common/common_types.py` (`VisionEncoderBlockType.OPENVLA`), `layers/encoders.py`, `layers/nnx_decoders.py`,
  `multimodal/processor.py`, `configs/types.py` (`ModelName`, multimodal allow-list), `utils/globals.py` (`HF_IDS`),
  `checkpoint_conversion/utils/param_mapping.py` (`OPENVLA_MAXTEXT_TO_HF_PARAM_MAPPING` / `_HOOK_FN`),
  `checkpoint_conversion/utils/hf_model_configs.py`, `tests/utils/forward_pass_logit_checker.py` (pixel layout).

## 7. Action Decoding & Visualization (follow-up)

- `decode.py` now prints the decoded 7-DoF action for VLA models after the (meaningless) text decode:
  `mm_processor.format_generated_actions` -> `processor_openvla.format_actions_openvla`. Normalization stats are
  read from `action_norm_stats_path` (default `<tokenizer_path>/config.json`) with key `action_unnorm_key`
  (default `bridge_orig`); both are new config fields in `base.yml` / `types.py`.
- **Open-loop Bridge rollout video**: `~/projects/visualize_openvla_bridge.py` runs MaxText OpenVLA on every frame of a
  real BridgeData episode (`gs://gresearch/robotics/bridge/0.1.0`, shard copied to `/dev/shm/hengtaoguo/openvla/bridge/`)
  and renders camera frame + 3D EE path (recorded vs integrated prediction, per-step delta arrows) + per-dim
  pred-vs-GT time series. Episode 2 ("Put the cheese onto the cloth.", 40 steps, ~1 s/step fp32 on v5p-2):
  gripper close predicted at step 14 (GT 14), reopen at step 29 (GT 28); xyz/rotation deltas track GT.
  Output: `~/projects/samples/openvla/openvla_bridge_ep2{.mp4,_summary.png,.npz}`.
  ```
  ../venv1/bin/python ../visualize_openvla_bridge.py src/maxtext/configs/base.yml model_name=openvla-7b use_multimodal=true \
    load_parameters_path=/dev/shm/hengtaoguo/openvla/ckpt_full_fp32_unscanned/0/items scan_layers=false \
    tokenizer_type=huggingface tokenizer_path=/dev/shm/hengtaoguo/openvla/openvla-7b per_device_batch_size=1 \
    max_prefill_predict_length=320 max_target_length=320 attention=dot_product skip_jax_distributed_system=true \
    dtype=float32 weight_dtype=float32 dtype_mm=float32 matmul_precision=highest float32_logits=true float32_qk_product=true \
    -- --tfrecord /dev/shm/hengtaoguo/openvla/bridge/bridge-test.tfrecord-00000-of-00512 --episode 2 --out_dir ~/projects/samples/openvla
  ```
- Closed-loop rendering (model actions actually move a robot) needs a simulator, e.g. SimplerEnv WidowX (SAPIEN,
  Vulkan rendering — not available on the CPU-only TPU host) or LIBERO (MuJoCo; requires the LIBERO-finetuned OpenVLA).
