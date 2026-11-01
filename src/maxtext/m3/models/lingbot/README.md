<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# LingBot-World-Fast (m3)

[robbyant/lingbot-world-fast](https://huggingface.co/robbyant/lingbot-world-fast) is a real-time world model: a
Wan 2.2 I2V-A14B DiT distilled into a 4-step, block-causal, autoregressive video generator conditioned on a first
frame, a prompt and an **action sequence** (a camera trajectory). Reference implementation:
[Robbyant/lingbot-world](https://github.com/Robbyant/lingbot-world) (`generate_fast.py`, `wan/image2video_fast.py`,
`wan/modules/model_fast.py`).

| File | Contents |
| :--- | :--- |
| `transformer_lingbot.py` | `LingBotWorldModel`: Wan blocks + block-causal self-attention with a KV cache, RoPE offset per chunk, camera (Plücker) FiLM injection, fp32 residual stream. Reuses `WanAttention` / `WanFeedForward` / `WanTimeTextImageEmbedding` / `FP32LayerNorm` and their parameter names. |
| `camera_lingbot.py` | Action sequence (`poses.npy` + `intrinsics.npy`) -> Plücker conditioning, numpy port of `cam_utils.py`. |
| `pipeline_lingbot.py` | `LingBotWorldPipeline(WanPipeline)`: reuses the Wan pipeline's mesh, UMT5 text encoder, VAE passes and checkpoint loading; adds the causal 4-step sampler. |
| `checkpoint_lingbot.py` | Configs / T5 key renaming of the original-layout `.pth` files for `to_maxtext_diffusers.py`. |
| `compare_reference.py` | End-to-end parity against a reference dump (same noise). |
| `configs/lingbot-world-fast.yml` | Model YAML. |

Conversion reuses `checkpoint_conversion/to_maxtext_diffusers.py` (`model_name: lingbot-world-fast`,
`LINGBOT_WORLD_FAST_HF_TO_MAXTEXT` in `param_mapping.py`, built on the Wan 2.1 mapping). Sampling reuses
`inference/sample.py`.

## Sampling algorithm

1. Latent geometry from the image aspect ratio and `height * width` (480x832 area -> 464x832 for a 16:9 image);
   `num_frames` is clipped to the trajectory and to a multiple of `chunk_size` (3) latent frames.
2. Conditioning: UMT5 prompt embedding (bf16, 512 tokens); first frame + zero frames through the VAE, plus a 4-channel
   first-frame mask (20 channels, concatenated to the 16 noise channels); per-latent-frame Plücker rays.
3. For each chunk of 3 latent frames: 4 steps at timesteps `[999, 978, 947, 825]` (shift-10 flow schedule, indices
   `[0, 179, 358, 679]`): predict flow, `x0 = x_t - sigma_t * flow`, re-noise to the next timestep; then run the clean x0
   once at t=0 to commit the chunk's keys / values to the KV cache. No CFG.
4. Decode all latents with the Wan 2.1 VAE.

## Commands

```bash
# 1. Convert (DiT from lingbot-world-fast, VAE + T5 + tokenizer from lingbot-world-base-cam; HF ids or local dirs).
python src/maxtext/checkpoint_conversion/to_maxtext_diffusers.py src/maxtext/m3/models/lingbot/configs/lingbot-world-fast.yml \
  pretrained_model_name_or_path=robbyant/lingbot-world-fast base_model_name_or_path=robbyant/lingbot-world-base-cam \
  base_output_directory=$CKPT scan_layers=true

# 2. Generate (v5p-8: tensor parallel over 4 chips, ~19 s denoise + ~20 s VAE decode for 81 frames at 464x832).
python src/maxtext/inference/sample.py src/maxtext/m3/models/lingbot/configs/lingbot-world-fast.yml \
  checkpoint_directory=$CKPT base_model_name_or_path=robbyant/lingbot-world-base-cam \
  image_path=$LINGBOT_WORLD/examples/03/image.jpg action_path=$LINGBOT_WORLD/examples/03 \
  num_frames=81 output_dir=$OUT prompt="A serene lakeside scene ..."
```

## Parity

`tests/unit/m3/lingbot_test.py` checks, against the reference modules on tiny random models in float32 (CPU, needs
torch and `LINGBOT_WORLD_REPO`): the converted DiT over two causal chunks (KV cache, RoPE offset, camera injection;
scanned and unscanned), the converted Wan T5 and Wan VAE `.pth` files, the camera conditioning, the distilled schedule
and the bicubic resize.

End to end, `compare_reference.py` replays the reference's Gaussian draws through the MaxText pipeline and reports
per-stage errors (`lingbot_dump_reference.py`, run on GPU with the reference repo, produces the `.npz`).

Results on the bundled lakeside example (81 frames, 464x832, v5p-8 vs the reference on 4x A100):

| Stage | MaxText vs reference |
| :--- | :--- |
| T5 context (bf16) | rel L2 2.9e-2 |
| I2V condition (VAE latents + mask) | rel L2 4.7e-3 |
| Camera conditioning | rel L2 1.4e-5 |
| DiT, teacher-forced (reference inputs for each of the 28 calls) | rel L2 0.6-2.9%, cosine >= 0.9996, flat across chunks |
| Video, free-running with the same noise | PSNR 25.2 dB (`attention=dot_product`), 23.8 dB (`flash`); first frame ~46 dB |

The free-running gap is the autoregressive amplification of bf16 rounding: two MaxText runs that differ only in the
attention kernel are 23.8 dB apart as well.
