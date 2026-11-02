# LingBot-World-Fast bring-up handover (MaxText m3)

Branch: `hengtaoguo-exp12` (on top of main `647d4531b` + cherry-picked PR #5420 `wan_dit_pr4` = commit "feat(inference): add
checkpoint conversion, diffusion pipelines and sample.py inference entrypoint").
HF model: `robbyant/lingbot-world-fast` (DiT) + `robbyant/lingbot-world-base-cam` (Wan2.1 VAE `.pth`, Wan umt5-xxl
`.pth`, tokenizer `google/umt5-xxl/`). Reference code: https://github.com/Robbyant/lingbot-world (copy at
`~/projects/lingbot-world`, rev a43bec7).

## Phase status

| Phase | Status |
| :--- | :--- |
| 1. HF analysis | DONE |
| 2. JAX implementation (`src/maxtext/m3/models/lingbot/`) + layer tests | DONE |
| 3. Checkpoint conversion (`to_maxtext_diffusers.py`, `LINGBOT_WORLD_FAST_HF_TO_MAXTEXT`) | DONE |
| 4. E2E generation + parity vs reference (GPU) | DONE |

## 1. Original model learnings

- WanModelFast = Wan 2.2 I2V-A14B DiT (dim 5120, 40 heads x 128, 40 layers, ffn 13824, text_dim 4096, freq_dim 256,
  in_dim 36 = 16 noise + 4 mask + 16 image latent, out_dim 16, patch (1,2,2), eps 1e-6, qk RMSNorm over full dim,
  cross-attn affine LayerNorm, no CLIP branch).
- Additions: camera control (`control_type='cam'`, chosen by 'cam' in the ckpt path): Plücker rays (6 ch) at pixel
  resolution folded 8x8 into 384 ch, patchified to 1536 -> `patch_embedding_wancamctrl` -> `+ c2ws_hidden_states_layer2(silu(layer1(.)))`.
  Each block after self-attn: `h = cam_injector_layer2(silu(layer1(c))) + c; x = (1 + cam_scale_layer(h)) * x + cam_shift_layer(h)`.
- Block-causal AR: chunks of 3 latent frames; self-attn over KV cache prefix (no mask inside the chunk); RoPE frame axis
  offset by chunk start frame (`causal_rope_apply`). Cross-attn KV cached once (512 tokens, no padding mask).
- Sampler (`WanI2VFast.generate`): `FlowUniPCMultistepScheduler(shift=1).set_timesteps(1000, shift=10)`, timesteps
  index [0,179,358,679] -> int64 [999, 978, 947, 825]. Per step: x0 = xt - sigma*flow (sigma by argmin), then
  add_noise to next t (sigma by `index_for_timestep`, 2nd match on duplicates) with fresh randn. After a chunk: forward
  x0 at t=0 to write the clean KV cache. No CFG. Decode all latents with Wan2.1 VAE.
- Numerics: DiT weights bf16, autocast bf16 matmuls, residual stream fp32 (adds under fp32 autocast), time embedding /
  projection / modulation / head in fp32. T5 bf16. VAE fp32 (TF32 convs on A100).
- Geometry: lat_h = round(sqrt(area*h/w)//8//2*2): 480*832 gives 464x832 for 16:9 and even for a 480x832 image (float
  rounding). 81 frames -> 21 latent frames -> 7 chunks, 1508 tokens/frame, KV cache 31668 tokens.
- Image resize: torch bicubic (A=-0.75, align_corners=False, no antialias) from the full-res 5145x2894 example image.

## 2. Implementation (reuse)

- `transformer_lingbot.py`: `LingBotWorldModel` composes Wan `FP32LayerNorm`, `WanAttention` (cross, cached KV),
  `LingBotSelfAttention(WanAttention)` (same params; KV cache + RoPE offset; dot_product or custom Splash flash with KV
  segment ids), `WanFeedForward`, `WanTimeTextImageEmbedding`. Same param names as Wan 2.1 -> mapping reuse.
- `pipeline_lingbot.py`: `LingBotWorldPipeline(WanPipeline)` reuses mesh, UMT5 encoder + `_get_t5_prompt_embeds`, VAE
  encode/decode passes, `put_params_into_state` loading; adds the causal sampler (`transformer_chunk_pass`,
  `commit_kv_cache` donated in-place, host-side `LingBotFastScheduler`).
- `camera_lingbot.py`: numpy port of `cam_utils` (slerp/linear resampling, relative poses, Plücker).
- Registered in `sample.py` (`MODEL_NAME_TO_PIPELINE` now accepts fully qualified paths) and `DIFFUSERS_PARAM_MAPPING`.

## 3. Conversion

`to_maxtext_diffusers.py` gained `COMPONENT_SOURCES` support (per-component repo/subfolder, raw `.pth` reader, key
converter, default config) for non-diffusers layouts. LingBot mapping = Wan 2.1 mapping with diffusers->original key
renaming + camera linears; VAE renamed by diffusers `convert_wan_vae_to_diffusers`; T5 by
`checkpoint_lingbot.convert_wan_t5_to_hf`. Output: transformer 27G (bf16, fp32 norms/embedders), T5 8.3G (bf16), VAE 450M.

## 4. Results (v5p-8, 4 chips TP)

- Generation: 81 frames 464x832: conditioning ~25 s (incl. T5/VAE compile), denoise ~19 s (flash) / ~26 s
  (dot_product), VAE decode ~20 s. 153-frame forward-dolly trajectory also renders correctly.
- Unit tests (`tests/unit/m3/lingbot_test.py`, CPU fp32 vs reference modules): 12/12 pass (DiT 2 chunks with KV cache
  scanned/unscanned, Wan T5 `.pth`, Wan VAE `.pth`, camera, schedule, bicubic) at 1e-4 to 1e-5.
- E2E vs reference (same noise, 4xA100, `compare_reference.py`):
  - T5 context rel 2.9e-2 (bf16 vs bf16; fp32 MaxText T5 is *further*: 9.6e-2), I2V condition rel 4.7e-3, camera 1.4e-5.
  - Teacher-forced DiT (reference inputs for every call): rel L2 0.6-2.9% per call, cosine >= 0.9996, no growth across
    the 7 chunks -> KV cache / RoPE offsets / camera injection match.
  - Free-running video PSNR 25.2 dB (dot_product) / 23.8 dB (flash), first frame ~46 dB, decaying over chunks. Baseline:
    MaxText flash vs MaxText dot_product (same everything else) = 23.8 dB, i.e. the divergence is the AR model's
    amplification of bf16 rounding, not an implementation error.

## Bugs / pitfalls

1. PR's `WanPipeline.load_vae` casts VAE weights to `weights_dtype` (bf16); LingBot overrides to keep fp32.
2. Wan T5 `T5EncoderModel` evaluates `torch.cuda.current_device()` at import (patched in the test).
3. Reference attention kernels are CUDA/bf16-only; CPU tests swap in fp32 SDPA.
4. Wan `_flash_attention` treats Q != K as cross attention (one KV block = full KV) -> custom Splash call for the cache.
5. absl INFO logs are hidden outside `app.run`; `compare_reference.py` sets verbosity.

## Reproduction

```bash
cd ~/projects/maxtext
# weights (bf16 Fast shards, VAE, T5, tokenizer): ~/projects/lingbot_prepare.py /dev/shm/hengtaoguo
python src/maxtext/checkpoint_conversion/to_maxtext_diffusers.py src/maxtext/m3/models/lingbot/configs/lingbot-world-fast.yml \
  pretrained_model_name_or_path=/dev/shm/hengtaoguo/lingbot-world-base-cam/lingbot_world_fast \
  base_model_name_or_path=/dev/shm/hengtaoguo/lingbot-world-base-cam base_output_directory=/dev/shm/hengtaoguo/lingbot_ckpt scan_layers=true
python src/maxtext/inference/sample.py src/maxtext/m3/models/lingbot/configs/lingbot-world-fast.yml \
  checkpoint_directory=/dev/shm/hengtaoguo/lingbot_ckpt base_model_name_or_path=/dev/shm/hengtaoguo/lingbot-world-base-cam \
  image_path=~/projects/lingbot-world/examples/03/image.jpg action_path=~/projects/lingbot-world/examples/03 \
  num_frames=81 output_dir=/dev/shm/hengtaoguo/lingbot_sample
# reference dump (GPU runner, ~/projects): torchrun --nproc_per_node=4 lingbot_dump_reference.py ... --out lingbot_ref_81.npz
~/projects/lingbot_parity.sh dot_product /dev/shm/hengtaoguo/lingbot_out_dp     # compare_reference.py wrapper
LINGBOT_WORLD_REPO=~/projects/lingbot-world JAX_PLATFORMS=cpu python -m pytest tests/unit/m3/lingbot_test.py
```

## Not done / next

- Rolling-window cache (`local_attn_size`, `sink_size`) and the `act` (WASD) checkpoint variant: not supported (raise).
- Batch size 1 only; no performance tuning (flash block sizes, context parallel).
