# pi0.5 base (lerobot/pi05_base) Bring-up Handover Document

Model: `lerobot/pi05_base` (openpi pi0.5 base, LeRobot PyTorch port, lerobot 0.7.0 / transformers 5.22.2).
MaxText name: `pi05-base` (`decoder_block: gemma`, `vision_encoder_block: gemma3` (SigLIP), `action_expert: pi05`).
Branch: `hengtaoguo-pi` (from upstream main `d3ef4c84e`). Machine: v5p-2 `hengtaoguo-dev-v5p2`. Traditional (non-M3) format, **unscanned only**.
Reference assets: `/dev/shm/hengtaoguo/pi05/` (HF checkpoint, goldens, converted checkpoints, DROID data, logs),
`~/projects/pi05_ref/` (scripts), `~/projects/venv_lerobot` (CPU torch 2.15 + lerobot 0.7.0, reference env).

## 1. HF Model Architectural Findings

- **Layout** (812 tensors, fp32, 14.5 GB, openpi key names `paligemma_with_expert.*`):
  - **VLM** = PaliGemma: SigLIP So400m/14 at 224 px (27 layers, width 1152, MLP 4304, 16 heads, 256 patches, learned
    pos-emb [256,1152], `post_layernorm`, no pooling head) + `multi_modal_projector.linear` (1152 -> 2048, **with bias**,
    **no** 1/sqrt(d) scaling in transformers 5.x) + Gemma-2B (18 layers, width 2048, 8 q heads / 1 kv head (MQA),
    head_dim 256, GeGLU 16384 `gelu_pytorch_tanh`, RMSNorm `x*(1+w)` eps 1e-6, RoPE theta 1e4, vocab 257152).
  - **No `embed_tokens` key**: the language embedding is tied to `paligemma.lm_head.weight` [257152, 2048]
    (LeRobot copies it). `GemmaTextScaledWordEmbedding` multiplies by sqrt(2048). Image embeddings are **not** scaled.
  - **Action expert** = Gemma-300M (`gemma_expert.model.layers.*`): 18 layers, width 1024, MLP 4096, same 8x256 heads and
    1 kv head as the VLM, **adaRMS** norms (`input_layernorm.dense`, `post_attention_layernorm.dense`, `norm.dense`:
    cond 1024 -> 3x1024 = scale/shift/gate, no learnable weight), gated residuals `x + gate * f(norm(x))`.
    `gemma_expert.lm_head` [257152,1024] is unused (not converted).
  - **Action head**: `action_in_proj` (32 -> 1024), `time_mlp_in/out` (1024, SiLU), `action_out_proj` (1024 -> 32), all with bias.
    pi0.5 has **no state projection**: the state is discretized into the prompt.
- **Joint layer** (`compute_layer_complete`): each stream computes its own norm + q/k/v, q/k/v are concatenated along the
  sequence, one softmax (eager, fp32 in fp32 mode, mask value -2.3819763e38), outputs split, separate o_proj / MLP.
- **Masks / positions**: prefix = [3 x 256 image tokens, 200 text tokens (right-padded)], att_mask 0 everywhere ->
  all valid prefix tokens attend bidirectionally. Suffix = 50 action tokens with att_mask [1,0,...] -> they attend all
  valid prefix tokens and each other; prefix never attends suffix (so the prefix can be KV-cached).
  Positions = cumsum(pad_mask) - 1; suffix positions continue from the number of valid prefix tokens.
- **Prompt**: `Task: {task}, State: {s_0 ... s_k};\nAction: ` with the quantile-normalized state digitized into 256 bins
  (`np.digitize(state, linspace(-1,1,257)[:-1]) - 1`), PaliGemma tokenizer (`google/paligemma-3b-pt-224`, BOS added),
  padded to 200.
- **Flow matching**: noise at t = 1, `x <- x + dt * v`, dt = -1/10, t = 1.0, 0.9, ..., 0.1. Time embedding:
  sinusoidal with periods log-spaced in [0.004, 4.0], computed in **float64** by the reference.
- **Images**: [-1, 1], `resize_with_pad` (bilinear, centered black padding) to 224; missing cameras are -1 images with mask False.
- **Decoder pattern**: homogeneous -> mini subset = 2 decoder layers (both experts) + 2 SigLIP layers.

## 2. Phase-by-Phase Status
- [x] Phase 1: HF discovery; mini subset `/dev/shm/hf_mini/pi05-base_2layers_2vit` (`~/projects/pi05_ref/slice_mini.py 2 2`) (DONE)
- [x] Phase 2: `src/maxtext/models/pi05.py` (Pi05DecoderLayer(GemmaDecoderLayer) + action expert, Pi05ActionHead,
      prefix/suffix forward, sampler, loss), `configs/models/pi05-base.yml`, config fields (`Pi05ActionExpert` in
      `types.py`, `base.yml`), decoder-layer selection, `Transformer.action_head`, multimodal router entry (DONE)
- [x] Phase 3: `PI05_MAXTEXT_TO_HF_PARAM_MAPPING` / `PI05_MAXTEXT_TO_HF_PARAM_HOOK_FN` in `param_mapping.py`,
      `pi05_base_config` in `hf_model_configs.py`, HF id in `globals.py`; conversion mini 0.5 min, full 0.8 min (33 GB peak) (DONE)
- [x] Phase 4: parity vs LeRobot PyTorch fp32 CPU goldens (`pi05_golden.py`), MaxText fp32 on TPU with `matmul_precision=highest` (DONE)

  | Check | mini (2 layers, 2 ViT) | full model |
  | :-- | :-- | :-- |
  | image embeddings (rel to max) | 4.5e-6 | 1.3e-5 |
  | layer-0 / layer-1 prefix hidden (rel) | 4.8e-6 / 5.0e-6 | 6.3e-6 / 7.2e-6 |
  | final prefix hidden (rel) | 7.7e-6 | 1.1e-4 (max abs 1.1e-2, cos 1.0) |
  | text logits (133 positions) KL mean / max | ~0 / 1.3e-7 | ~0 / 2.5e-7 |
  | top-1 match / top-10 overlap | 100% / 100% | 100% / 99.9% |
  | first-step velocity v(t=1) max abs | 4.8e-6 | 8.7e-6 |
  | 10-step action chunk max abs (rel) | 1.0e-5 (1.6e-5) | 4.9e-6 (6.1e-6) |

## 3. Design (MaxText)
- **No new decoder block**: `action_expert: pi05` makes `NNXDecoder.get_decoder_layers` return `Pi05DecoderLayer`,
  a `GemmaDecoderLayer` subclass with extra modules `pre_self_attention_norm_action`, `self_attention_action`,
  `pre_ffw_norm_action`, `mlp_action`. Called with the standard arguments it is exactly the Gemma layer.
- The pi0.5 forward drives the layers explicitly (`pi05.prefix_forward` -> per-layer rotated K/V, `pi05.denoise_step`
  -> action stream attending to [cached prefix K/V, suffix K/V]); this equals the reference joint layer because the
  prefix never attends the suffix, and the 10 denoise steps reuse the prefix pass (like LeRobot's KV cache).
- Vision reuses `Gemma3VisionEncoderLayer` (224 px -> 256 tokens, `VisionExit` is a no-op) and the existing
  `MultimodalMLPProjector` (`vision_projector_type: customized_vision_projector`, 1 layer, bias).
- Attention core is a small fp32 einsum softmax with the openpi mask constant (keeps padding-row behaviour identical).
- `Transformer.action_head` = `Pi05ActionHead` (in/out projections, time MLP, expert final adaRMS).

## 4. Bugs Encountered & Solutions
- **RoPE angle arithmetic**: MaxText `RotaryEmbedding` computes `position / timescale`, HF computes
  `position * inv_freq` (fp32). The ~1-ulp angle difference (~6e-5 rad at position ~900) gave 7e-5 relative hidden
  error per layer on the 968-token prefix. `pi05.apply_rope` uses the HF arithmetic -> 5e-6.
- **Tied-embedding logits in bf16**: MaxText `attend_on_embedding` always casts the embedding table to bfloat16
  (1.4e-3 relative logit error, KL 1e-4 even in fp32 runs). `pi05.prefix_logits(full_precision=True)` does the
  product in the table dtype -> KL 1e-7.
- Golden script: `PI05Config.dtype` must be a `torch.dtype` (not the string "float32"); transformers >= 5.8 removed the
  `vision_tower.vision_model` level (use `vision_tower.encoder.layers` for truncation; checkpoint keys keep `vision_model`).

## 5. Pitfalls & Gotchas
- Query scale 1/sqrt(256) is **folded into the q kernels** at conversion (MaxText Gemma attention does not scale);
  Gemma RMSNorm scales are stored +1; the token embedding is stored x sqrt(2048) (MaxText Gemma convention).
- The adaRMS `dense` layers have **no** `(1 + w)` weight; `normed * (1 + scale) + shift`, residual gated by `gate`.
- Keep the time embedding in float64 on the host for parity (`sinusoidal_time_embedding` does this for concrete t).
- The default MaxText decode / forward-pass logit checker do not apply: the pi0.5 prefix is bidirectional (prefix-LM)
  with images first; parity is checked with `~/projects/pi05_ref/mt_parity_pi05.py`.
- `pi05_base`'s bundled LeRobot normalizer stats are empty; use openpi's per-robot stats
  (`gs://openpi-assets/checkpoints/pi05_base/assets/<robot>/norm_stats.json`, copied to `/dev/shm/hengtaoguo/pi05/openpi_assets/`).

## 6. Reproduction Commands (VM, from `~/projects/maxtext`)
```bash
export PYTHONPATH=src:. HF_HOME=/dev/shm/hengtaoguo
# HF checkpoint + reference env: ~/projects/pi05_ref/setup.sh (snapshot_download lerobot/pi05_base; uv venv venv_lerobot; lerobot[pi])
# Mini subset
~/projects/venv1/bin/python ~/projects/pi05_ref/slice_mini.py 2 2
# Goldens (LeRobot PyTorch fp32 CPU)
~/projects/venv_lerobot/bin/python ~/projects/pi05_ref/pi05_golden.py --out /dev/shm/hengtaoguo/pi05/golden/mini_l2_v2.npz --num_layers 2 --num_vit_layers 2
~/projects/venv_lerobot/bin/python ~/projects/pi05_ref/pi05_golden.py --out /dev/shm/hengtaoguo/pi05/golden/full.npz
# Conversion (CPU); mini: add override_model_config=true base_num_decoder_layers=2 num_hidden_layers_for_vit=2 and the mini hf path
JAX_PLATFORMS=cpu ~/projects/venv1/bin/python -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=pi05-base use_multimodal=true scan_layers=false base_output_directory=/dev/shm/hengtaoguo/pi05/ckpt_full_fp32_unscanned \
  --hf_model_path=/dev/shm/hengtaoguo/pi05/pi05_base weight_dtype=float32 hardware=cpu skip_jax_distributed_system=true \
  --lazy_load_tensors=false --save_dtype=float32 --simulated_cpu_devices_count=1
# Parity (TPU)
~/projects/venv1/bin/python ~/projects/pi05_ref/mt_parity_pi05.py src/maxtext/configs/base.yml model_name=pi05-base \
  use_multimodal=true scan_layers=false load_parameters_path=/dev/shm/hengtaoguo/pi05/ckpt_full_fp32_unscanned/0/items \
  dtype=float32 weight_dtype=float32 dtype_mm=float32 matmul_precision=highest attention=dot_product \
  skip_jax_distributed_system=true -- --golden=/dev/shm/hengtaoguo/pi05/golden/full.npz
```

## 7. DROID rollout + MuJoCo rendering (DONE)
- **Data**: `lerobot/droid_1.0.1` chunk-000 file-000 (parquet + `exterior_1_left` / `wrist_left` mp4, 1.2 GB) in
  `/dev/shm/hengtaoguo/pi05/droid_101/`; episodes extracted with `~/projects/pi05_ref/droid_extract.py --episode N`
  (venv_lerobot, needs `av`, `pandas`) -> `/dev/shm/hengtaoguo/pi05/droid_eps/epN.npz`.
  (`lerobot/droid_100` is *not* usable: its state/action are cartesian, no joint positions.)
- **Policy I/O** (openpi `DroidInputs` / `DroidOutputs` for PI05, verified against openpi main):
  base_0_rgb = exterior_1_left, left_wrist_0_rgb = wrist_left, right_wrist_0_rgb = zeros with mask False;
  state = [joint_position(7), gripper_position(1)] quantile-normalized with `pi05_base/assets/droid/norm_stats.json`
  and digitized into the prompt (8 values, tokenized before padding); output = first 8 dims un-normalized ->
  [joint_velocity(7), gripper_position(1)]. Images: `resize_with_pad` 180x320 -> 224 (bilinear, black pad).
- **Rollout** (`~/projects/pi05_ref/pi05_droid_rollout.py`, MaxText fp32 on v5p-2): one 50-step chunk every 15 frames
  (1 s), first 15 actions executed open-loop from the recorded state. **82 ms per chunk** (prefix + 10 denoise steps,
  fp32, no TP, jitted; first call 84 s incl. compile).
- **Render** (`~/projects/pi05_ref/pi05_droid_render.py`, venv_mujoco, `MUJOCO_GL=egl`): Menagerie
  `franka_emika_panda` driven kinematically; predicted joints `q <- q + gain * v / 15` re-anchored to the recorded
  state at every chunk; `gain` (per joint, 0.48-0.83) maps DROID's commanded joint velocity to realized motion and is
  fitted on the recorded episode only. Baselines: "hold" (zero velocity) and "GT velocity replay" (same integrator with
  the recorded commands = the floor of this protocol).
- **Results** (end-effector deviation at the end of each 15-step chunk, cm; 15 episodes of file-000):

  | ep | task | pi0.5 | hold | GT-vel replay |
  | :-- | :-- | :-- | :-- | :-- |
  | 28 | Put the marker inside the white bowl | **3.9** | 8.1 | 2.9 |
  | 8 | Remove the yellow toy from the green bowl ... | **5.7** | 10.4 | 3.7 |
  | 3 | Pick up the longer upright white container ... | **4.9** | 7.4 | 2.9 |
  | 23 | Remove the blue block from the top of the stack | **5.0** | 7.2 | 2.2 |
  | 20 | Put the white bottle on the orange plate | **5.3** | 7.5 | 3.4 |
  | 1 | Put the blue block in the green bowl | **5.0** | 5.4 | 3.3 |
  | 7 | Move the tile letter s to the left | 4.1 | **3.6** | 2.9 |
  | 9 | Pick the marker and put it in the plastic box | 6.7 | 6.7 | 4.2 |
  | all 15 | | better than hold on 13/15 | | |

  Per-step velocity correlation with the recorded commands is ~0.4 on average (0.58 on ep28); the gripper open/close
  timing is reproduced. This is the **zero-shot base model** (openpi serves DROID with the fine-tuned `pi05_droid`), so
  per-step actions are noisy, but chunks integrate to trajectories close to the recorded ones.
- **Artifacts** (`~/projects/openvla_artifacts/`): `20261114-001712_pi05_maxtext_droid_ep28.{gif,mp4,json,_mid.png}`
  (headline), `20261114-001712_pi05_maxtext_droid_ep8.*`; rollouts `/dev/shm/hengtaoguo/pi05/rollouts/ep*_fp32.npz`,
  per-episode metrics `ep*_metrics.json`.

```bash
# extract, rollout (one model load for several episodes), render
~/projects/venv_lerobot/bin/python ~/projects/pi05_ref/droid_extract.py --episode 28 --out /dev/shm/hengtaoguo/pi05/droid_eps/ep28.npz
PYTHONPATH=src:. ~/projects/venv1/bin/python ~/projects/pi05_ref/pi05_droid_rollout.py src/maxtext/configs/base.yml \
  model_name=pi05-base use_multimodal=true scan_layers=false load_parameters_path=/dev/shm/hengtaoguo/pi05/ckpt_full_fp32_unscanned/0/items \
  dtype=float32 weight_dtype=float32 dtype_mm=float32 matmul_precision=highest attention=dot_product skip_jax_distributed_system=true \
  -- --episode_npz /dev/shm/hengtaoguo/pi05/droid_eps/ep28.npz --out /dev/shm/hengtaoguo/pi05/rollouts/ep28_fp32.npz
MUJOCO_GL=egl ~/projects/venv_mujoco/bin/python ~/projects/pi05_ref/pi05_droid_render.py \
  --episode_npz /dev/shm/hengtaoguo/pi05/droid_eps/ep28.npz --rollout_npz /dev/shm/hengtaoguo/pi05/rollouts/ep28_fp32.npz \
  --out ~/projects/openvla_artifacts/<ts>_pi05_maxtext_droid_ep28
```

## 8. Open items / next steps
- Scanned layers (`scan_layers=true`) and bf16 / tensor-parallel inference are not yet exercised (unscanned fp32 only).
- Training: `pi05.flow_matching_loss` is written but not wired into a trainer (needs a robot SFT shim like OpenVLA's
  `train_sft_robot.py`, Beta(1.5, 1) time sampling, and knowledge insulation if desired).
- Closed-loop evaluation needs a DROID/LIBERO scene with objects; the MuJoCo scene here is kinematic replay only.
- Optional: compare against the fine-tuned `pi05_droid` (openpi JAX checkpoint) for a stronger DROID demo.
