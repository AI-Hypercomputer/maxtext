# Ming-3.0-audio-tiny Talker (text -> speech) Bring-up Handover

Branch: `hengtaoguo-ming-talker` (standard MaxText route: `models.Transformer` + `qwen2` NNX decoder,
central checkpoint conversion). Scope: **text input -> 44.1 kHz speech output** with the publisher's default
"IP" task (named voice `灵小柔`, no reference audio). The thinker (audio+text -> text) is unchanged.

## 1. HF model architectural findings

Thinker and talker are **independent models**: the talker consumes plain text (not thinker hidden states),
so "thinker: audio+text -> text" and "talker: text -> audio" are two separate flows (confirmed from
`talker/inference.py`: `MingTalker.synthesize(text)`). Talker weights live in the HF repo subfolder `talker/`
(`model.safetensors` 1.4 GB, `vae/model.safetensors` 1.56 GB).

- **LLM**: `Qwen2Model` (`talker/llm/config.json`): 24 layers, hidden 896, 14 heads / 2 KV heads, head_dim 64,
  SwiGLU 4864, q/k/v bias, RoPE theta 1e6, RMSNorm 1e-6, vocab 151936 (tied, no `lm_head`). Fed with
  **input embeddings**; the final-normed last hidden state is the condition for the CFM head.
- **Prompt** (each piece tokenized separately and concatenated):
  `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n` + `Please generate speech based
  on the following description.\n` + ` Text input:\n` + text + `<|im_end|>\n<|im_start|>assistant\n` +
  instruction JSON (`{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", ..., "IP": "灵小柔"}]}`) +
  `<|im_end|>` + `<audio>` (151706). ~140 tokens for a short sentence.
- **CFM** (`talker_module/{cfm,dit}.py`): DiT velocity net, 8 blocks, hidden 1024, 16 heads (head_dim 64),
  pre-RMSNorm attention (bias, **interleaved x-transformers RoPE, base 1e4**, bidirectional) + GELU(tanh) MLP x4,
  final RMSNorm + Linear(1024->64). Sequence: `[t_emb(sinusoidal 256, scale 1000 -> MLP) + Linear(896->1024)(h),
  x_emb(prev latent), x_emb(noisy latent)]`; output = last token. CFG: batch [cond, zeros], `pred + 2*(pred-null)`.
  Sampler: 10 Euler steps on EPSS timesteps `[0,2,4,6,8,12,16,20,24,28,32]/32` with sway (`t -> 1 - cos(pi t / 2)`);
  SDE term `sigma*sqrt(temperature)` = 0 by default (sigma 0.25, temperature 0) -> deterministic ODE given noise.
- **Aggregator**: same DiT blocks (8 x 1024) on `[learned cls, Linear(64->1024)(latent)]`, output cls ->
  RMSNorm + Linear(1024->896) = next LLM input embedding.
- **Stop head**: Linear(896->2) on the LLM hidden; stop when `softmax[1] > 0.5` and step > 3 (latent of the
  stopping step is kept).
- **AudioVAE decoder** (`AudioVAE/*`): Linear(64->896) -> `nn.Upsample(x4, linear, align_corners=False)` ->
  Qwen2 backbone (24 layers, same dims as the LLM, **causal sliding window 64**, no embeddings) -> RMSNorm ->
  Linear(896->3530) -> ISTFT head (`exp(mag)` clipped at 100, phase -> complex, `irfft(n_fft=3528)`, periodic
  Hann window, overlap-add hop 882, "same" trim 1323, window-envelope normalization). 3528 samples / latent
  (0.08 s at 44.1 kHz, 12.5 latents/s).
- **Frontend**: fragmenting (~50 chars, publisher's `text_fragments`) + text normalization (`TalkerTN`: pynini
  FSTs + `inflect` number verbalization, e.g. "3 pm" -> "three pm") + `silence_holder` trailing-silence trim
  (keep 0.3 s after the last frame with mean |x| > 1e-3). Each fragment is synthesized independently.
- Not ported (voice cloning only): AudioVAE encoder, CAM++ speaker encoder (`campplus.onnx`), `spk_head` is
  converted but unused.

## 2. Phase status

- [x] Phase 1: HF analysis, CPU HF reference + goldens (`tests/utils/ming_reference/talker_golden.py`) (DONE)
- [x] Phase 2: JAX modules (`ming_talker.py`) + layer tests (10/10 pass) + config + text frontend (DONE)
- [x] Phase 3: central checkpoint conversion (`to_maxtext`, 828 params, fp32 + bf16) (DONE)
- [x] Phase 4: new inference script `ming_talker_decode.py`; end-to-end parity vs HF (fp32) + ASR round trip (DONE)

## 3. Files

| File | Purpose |
| --- | --- |
| `src/maxtext/models/ming_talker.py` | `MingTalkerDiT` + `cfm_sample`, `MingTalkerAggregator`, `MingAudioVAEDecoder` (sliding-window Qwen2 + ISTFT), `MingTalkerHeads` container |
| `src/maxtext/configs/models/ming-3.0-audio-tiny-talker.yml` | talker config: `decoder_block: qwen2` LLM + `ming_talker: true` heads |
| `src/maxtext/configs/types.py` | `ming_talker*` / `ming_vae*` fields in the `Ming` group, model name |
| `src/maxtext/models/models.py` | `Transformer.ming_talker` built when `config.ming_talker` |
| `src/maxtext/inference/ming_talker_decode.py` | **new inference CLI** + `MingTalkerGenerator` (prefill/AR KV cache, CFM, stop, VAE, WAV) |
| `src/maxtext/multimodal/processor_ming_talker.py` | prompt/instruction builder, token ids, light fragmenter, optional HF TN frontend, silence trim |
| `src/maxtext/checkpoint_conversion/utils/{param_mapping,hf_model_configs}.py`, `src/maxtext/utils/globals.py` | `MING_TALKER_*` mapping/hooks (Qwen mapping + heads), HF config, HF id |
| `tests/unit/ming_talker_layers_test.py` | DiT+CFG, CFM sampler (ODE + SDE), aggregator, VAE decoder (+ padded bucketing), bf16 buffers, frontend |
| `tests/utils/ming_reference/talker_golden.py` | HF CPU reference goldens (replayable numpy noise) |
| `tests/utils/ming_reference/talker_parity.py` | end-to-end MaxText vs HF parity (free-running + teacher-forced) |

## 4. Bugs & resolutions

- **Checkpoint buffers are bf16-rounded**: `cfm/aggregator.rotary_embed.inv_freq` (stored F32) and
  `decoder.head.istft.window` (BF16) are exactly `bf16(standard value)`; HF loads them, so MaxText rounds the RoPE
  frequencies and Hann window to bf16 too (`bf16_round`). The Hann window must be computed in float32 like
  `torch.hann_window` (`np.hanning` in float64 flips 8 bf16 roundings).
- `bf16_round` via `jnp` inside `jit` raised `TracerArrayConversionError`; now pure numpy (`ml_dtypes`).
- AudioVAE compile (~12 s, dominated by the non-power-of-2 `irfft`) happened per fragment length -> latents are
  edge-padded to multiples of 64 and the ISTFT masks padded frames (exact; covered by the unit test).
- HF reference deps are not in venv1: installed with `--no-deps --target /dev/shm/hengtaoguo/pyextra`
  (x-transformers 2.31.10, torch_einops_utils, einx, frozendict, loguru, pynini 2.1.7, onnxruntime, fasttext-wheel,
  torchaudio 2.11.0+cpu, inflect, typeguard, more-itertools).

## 5. Pitfalls & gotchas

- The talker is not a token model; `decode.py`/MaxEngine cannot drive it. `ming_talker_decode.py` calls
  `model.decoder(..., decoder_input_embeddings=...)` directly and applies `decoder.apply_decoder_norm` to the
  returned (pre-norm) hidden state.
- One AUTOREGRESSIVE-mode model serves PREFILL (prompt padded to `max_prefill_predict_length`, segment ids mask
  the padding) and AR steps (positions continue from the true length); no insert step. Max latents per fragment =
  `max_target_length - max_prefill_predict_length` (e.g. 1280 - 256 = 1024 -> ~82 s).
- The stream is broadcast to all rows of the global batch (4 on v5p-8 / this VM); row 0 is used.
- Free-running fp32 latents drift slowly from HF (chaotic autoregression: rel err 1e-5 at step 0 -> ~1e-2 after
  ~30 steps) while waveforms stay ~0.9995-1.0 correlated; the strict check is **teacher forcing** (per-step error
  <= 3e-5). bf16 runs produce different but equally intelligible audio.
- Without `MING_TALKER_FRONTEND_PATH`, numbers are not verbalized (no pynini); the light splitter also fragments
  differently from the publisher. Set `MING_TALKER_FRONTEND_PATH=<snapshot>/talker` (+ pynini/inflect on
  `PYTHONPATH`) for the exact HF text frontend.
- `load_hf_dict_from_safetensors` rglobs the given dir: pass `--hf_model_path=<snapshot>/talker` (picks
  `model.safetensors` + `vae/model.safetensors`; key sets do not collide). The VAE encoder keys are ignored.

## 6. Results

| Check | Result |
| --- | --- |
| Layer tests (CPU fp32, `tests/unit/ming_talker_layers_test.py`) | 10/10 pass |
| Prompt token ids vs HF (4 fragments) | identical |
| HF text frontend through `HFTextFrontend` (3 fragments incl. TN "3 pm" -> "three pm") | identical |
| Short golden (1 fragment, 34 latents), TPU fp32 | same 34 latents / stop step; teacher-forced hidden 1.6e-5, latent 3.2e-5; VAE on HF latents 1.7e-6; waveform corr 0.99999 |
| Long golden (3 fragments, 32/40/48 latents), TPU fp32 | same latent counts; teacher-forced latent <= 3.0e-5; VAE <= 3.8e-6; waveform corr 1.000000 / 0.999999 / 0.999537 |
| ASR round trip (thinker transcribes the talker WAV) | HF and MaxText (fp32) long golden -> identical transcript; MaxText bf16 demo -> "Hello, I am the Ming talker running on TPU with max text. Today we are testing text to speech generation and it seems to be working quite well." |
| Speed (v5p, bf16, after compile) | ~35 ms / latent (0.08 s audio) -> ~2.3x faster than real time; AudioVAE < 1 s per fragment after bucket compile |

## 7. Reproduction commands

```bash
cd ~/projects/maxtext; PY=../venv1/bin/python; export HF_HOME=/dev/shm/hengtaoguo
T=/dev/shm/hengtaoguo/ming/hf_full/talker   # HF snapshot talker/ dir
# Download talker weights + code
$PY -c "from huggingface_hub import snapshot_download as s; s('inclusionAI/Ming-3.0-audio-tiny', allow_patterns=['talker/*','talker/**','voice_presets/**','*.py','*.json','*.jinja'], local_dir='/dev/shm/hengtaoguo/ming/hf_full')"

# Layer tests (CPU)
MING_TALKER_HF_PATH=$T MING_TALKER_EXTRA_PATH=/dev/shm/hengtaoguo/pyextra JAX_PLATFORMS=cpu $PY -m pytest tests/unit/ming_talker_layers_test.py -q

# Checkpoint conversion (CPU, ~15 s); use weight_dtype/save_dtype=bfloat16 for the default inference ckpt
JAX_PLATFORMS=cpu $PY -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=ming-3.0-audio-tiny-talker scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  weight_dtype=float32 base_output_directory=/dev/shm/hengtaoguo/ming/talker_ckpt_fp32 \
  --hf_model_path=$T --lazy_load_tensors=false --save_dtype=float32 --simulated_cpu_devices_count=1

# Text -> speech (TPU, bf16)
$PY -m maxtext.inference.ming_talker_decode src/maxtext/configs/base.yml model_name=ming-3.0-audio-tiny-talker \
  tokenizer_path=$T/llm load_parameters_path=/dev/shm/hengtaoguo/ming/talker_ckpt_bf16/0/items scan_layers=false \
  per_device_batch_size=1 max_prefill_predict_length=256 max_target_length=1280 attention=dot_product \
  skip_jax_distributed_system=true "prompt=Hello! I am the Ming talker, running on TPU with MaxText." \
  ming_talker_audio_output_path=/dev/shm/hengtaoguo/ming/talker_demo_bf16.wav
# optional: exact HF text normalization
#   MING_TALKER_FRONTEND_PATH=$T PYTHONPATH=/dev/shm/hengtaoguo/pyextra $PY -m maxtext.inference.ming_talker_decode ...
# knobs: ming_talker_ip_name, ming_talker_language, ming_talker_seed, ming_talker_cfg_strength,
#        ming_talker_sigma/temperature, ming_talker_max_steps, ming_talker_max_fragment_length

# HF goldens (CPU fp32) + parity (TPU fp32)
$PY -m tests.utils.ming_reference.talker_golden --talker_path=$T --extra_path=/dev/shm/hengtaoguo/pyextra \
  --text="Hello, this is a test of the Ming talker." --output=/dev/shm/hengtaoguo/ming/talker_golden.pkl
MING_TALKER_FRONTEND_PATH=$T PYTHONPATH=/dev/shm/hengtaoguo/pyextra $PY -m tests.utils.ming_reference.talker_parity \
  src/maxtext/configs/base.yml model_name=ming-3.0-audio-tiny-talker tokenizer_path=$T/llm \
  load_parameters_path=/dev/shm/hengtaoguo/ming/talker_ckpt_fp32/0/items scan_layers=false per_device_batch_size=1 \
  max_prefill_predict_length=256 max_target_length=768 attention=dot_product weight_dtype=float32 dtype=float32 \
  matmul_precision=highest float32_qk_product=true float32_logits=true skip_jax_distributed_system=true \
  --golden=/dev/shm/hengtaoguo/ming/talker_golden_long.pkl

# ASR round trip with the thinker
$PY -m maxtext.inference.decode src/maxtext/configs/base.yml model_name=ming-3.0-audio-tiny scan_layers=false \
  tokenizer_type=huggingface tokenizer_path=/dev/shm/hengtaoguo/ming/tokenizer \
  load_parameters_path=/dev/shm/hengtaoguo/ming/ckpt_full_fp32_scan_false/0/items \
  audio_path=/dev/shm/hengtaoguo/ming/talker_demo_bf16.wav "prompt=Please transcribe the audio." per_device_batch_size=1 \
  max_prefill_predict_length=160 max_target_length=224 attention=dot_product decode_sampling_strategy=greedy \
  sparse_matmul=false skip_jax_distributed_system=true hardware=tpu
```

## 8. Follow-ups

- Voice cloning ("TTS" task): port the AudioVAE encoder + CAM++ speaker embedding (or accept a precomputed
  192-d embedding), inject `spk_head(emb)` after `<|vision_start|>` and prompt latents after `<audio>`.
- Thinker -> talker chaining in one command (feed the thinker's generated text to the talker).
- `scan_layers=true` for the talker LLM (needs the scanned Qwen mapping path to be exercised).
- Streaming AudioVAE decode (the publisher supports chunked decode with KV/overlap buffers).
- Batch several fragments/requests per step instead of broadcasting one stream.
