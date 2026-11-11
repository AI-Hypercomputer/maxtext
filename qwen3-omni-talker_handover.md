# Qwen3-Omni-30B-A3B Talker (Audio Output) Bring-up — Handover

Goal: enable `decode.py` to emit **speech audio** for `qwen3-omni-30b-a3b`
(text/image/video/audio input → text **and** 24 kHz waveform output).

The Thinker (text output) was already supported in MaxText. This document
tracks the bring-up of the **Talker + Code Predictor + Code2Wav** stack.

## 1. HF model learnings (Qwen/Qwen3-Omni-30B-A3B-Instruct, config.json)

Weights for talker/code2wav live only in `model-000{13,14,15}-of-00015.safetensors`.

### Talker (`talker.model.*`, Qwen3OmniMoeTalkerModel) — MoE decoder
| field | value |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 20 (all MoE, `decoder_sparse_step=1`) |
| num_attention_heads / num_key_value_heads / head_dim | 16 / 2 / 128 |
| qk-norm | yes (RMSNorm on head_dim, eps 1e-6) |
| num_experts / top-k / moe_intermediate_size | 128 / **6** / 384, `norm_topk_prob=true` (softmax routing) |
| shared expert | single MLP, intermediate 768, gated by `sigmoid(shared_expert_gate(x))` (Linear 1024→1, no bias) |
| rope | theta 1e6, **interleaved MRoPE**, `mrope_section=[24,20,20]` (same as thinker) |
| vocab (codec) | 3072; `codec_embedding` (3072×1024); `codec_head` Linear 1024→3072 (separate weights) |
| `text_projection` / `hidden_projection` | ResizeMLP: Linear(2048→2048,+bias) → SiLU → Linear(2048→1024,+bias) |
| special codec ids | pad 2148, bos 2149, eos 2150, nothing 2155, think_bos 2156, think_eos 2157; speakers chelsie 2301 / ethan 2302 / aiden 2303 |
| `accept_hidden_layer` | 24 → HF `hidden_states[24]` = output of thinker layer index 23 (pre final-norm, deepstack already applied) |
| `num_code_groups` | 16 (1 from talker + 15 from code predictor) |
| thinker special ids | tts_pad 151671, tts_bos 151672, tts_eos 151673, im_start 151644, user 872, assistant 77091 |

### Code predictor (`talker.code_predictor.*`) — small dense decoder
hidden 1024, 5 layers, heads 16/8, head_dim 128, qk-norm, MLP 3072 (silu),
rope theta 1e6 (standard 1-D), rms eps 1e-6; 15 `codec_embedding[i]` (2048×1024)
and 15 `lm_head[i]` (1024→2048). Step i (0..14): input position i+1, logits = `lm_head[i]`,
next input = `codec_embedding[i](code_{i+1})`. Prefill input = `[talker_normed_hidden, talker.codec_embedding(code0)]`.

### Code2Wav (`code2wav.*`)
`code_embedding` (16·2048 × 1024, code + group·2048, mean over 16 groups) →
`pre_transformer` (8 layers, hidden 1024, 16 heads (head_dim 64), **no qk norm**, rope theta 1e4,
sliding window 72, RMSNorm eps 1e-5, residual layer-scale) → 2× [CausalTransConv(1024,1024,k=2,s=2) + ConvNeXt block]
→ CausalConv(1024→1536,k7) → 4 decoder blocks (SnakeBeta + CausalTransConv(k=2r,s=r), r∈(8,5,4,3), 3 residual units dilation 1/3/9)
→ SnakeBeta → CausalConv(→1, k7) → clamp[-1,1]. Total upsample = 8·5·4·3·2·2 = **1920 samples / code** (24 kHz; 12.5 codes/s).

### HF generation flow (what we replicate)
1. Thinker generates text (greedy here). Capture for the **prompt**: `thinker_embed` = merged input embeddings
   (HF `hidden_states[0]`), `thinker_hidden` = `hidden_states[24]`.
2. User part (tokens whose role is `user`): mm-token positions → `hidden_projection(thinker_hidden)`,
   text positions → `text_projection(thinker_embed)`.
3. Assistant part: `a = text_projection(embed([im_start, assistant, \n, gen_0 ...]))`;
   prefill tail = `concat(a[:3], tts_pad×4, tts_bos, a[3:4]) + concat(0×3, codec_embed([nothing, think_bos, think_eos, speaker, pad, bos]))`;
   `trailing_text_hidden = concat(a[4:], tts_eos)` (one entry per subsequent generated thinker token).
4. Each talker step: `code0 = sample(codec_head(h))` (ids 2048..3071 suppressed except eos 2150, repetition penalty 1.05),
   code predictor yields codes 1..15; next input = `sum(codec_embed(code0), cp.codec_embed[i-1](code_i)) + trailing_text_hidden[step] (or tts_pad_embed)`.
5. Codes `[16, T]` → Code2Wav (chunked 300 codes, 25 left context) → wav.

## 2. Phase status
| Phase | Status |
|---|---|
| 1. HF analysis | DONE |
| 2. JAX layers (talker / code predictor / code2wav) + unit tests | DONE — `tests/unit/qwen3_omni_talker_layers_test.py` (8 tests, CPU, fp32) all pass: talker prefill logits/hidden vs HF, prefill→AR KV-cache consistency, code predictor hidden + heads, Code2Wav waveform + chunked decode, `Transformer` param layout with/without `enable_audio_output` |
| 3. Checkpoint conversion (thinker + speech stack → one orbax ckpt) | DONE — `to_maxtext.py model_name=qwen3-omni-30b-a3b use_multimodal=true enable_audio_output=true` (eager, 18.6 min CPU, peak 245 GB RAM, 2210 params) → `/dev/shm/hengtaoguo/checkpoints/qwen3-omni-30b-a3b-audio/unscanned/0/items`, uploaded to `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-omni-30b-a3b-audio/2026-10-16/unscanned/0/items` (51.3 GiB, bf16). |
| 4. E2E decode.py with audio output + HF parity check | DONE (merged checkpoint) — `decode.py enable_audio_output=true`: text prompt → thinker *"Hello, I'm Qwen-Omni, a multimodal large-scale language model developed by Alibaba's Tongyi Lab."* → 92 frames / 7.34 s WAV (talker stops at codec EOS by itself); video+audio-in-video prompt (`mini.mp4`, 1675-token talker prefill with 3-D MRoPE) → thinker *"A basketball player dribbles on a court in front of a large crowd, while a commentator speaks excitedly over the sound of the bouncing ball."* → 98 frames / 7.82 s WAV. **Parity (real weights, merged ckpt)**: replaying the identical talker inputs with the MaxText talker in fp32 vs the HF talker (fp32, CPU): same 89 frames, all 16 code groups identical on every frame (full-frame agreement 1.000), waveform correlation 0.9999998. The default bf16 run diverges in greedy code choice after a few frames (bf16 autoregressive drift) but yields a comparable utterance. (v1 with a separate talker checkpoint had the same parity numbers for text and video.) |

## 3. Design (MaxText side) — single model / single checkpoint (v2)
* **One config**: all speech-stack fields live in `qwen3-omni-30b-a3b.yml` with HF names
  (`enable_audio_output`, `talker_*` ← `talker_config.text_config`, `num_code_groups`/`accept_hidden_layer`/`codec_*`/`speaker_id`
  ← `talker_config`, `code_predictor_*`, `code2wav_*`, `tts_*_token_id`/`im_start_token_id`/`user_token_id`/`assistant_token_id`
  ← top level; generation knobs = HF `generate` kwargs `speaker`, `talker_max_new_tokens`, `talker_do_sample`, `talker_top_k/p`,
  `talker_temperature`, `talker_repetition_penalty`). The only switch is `enable_audio_output`.
* **One model**: `models.Transformer` builds `self.talker` (`Qwen3OmniTalker`, which owns `code_predictor`) and `self.code2wav`
  when `enable_audio_output=true` — same pattern as `vision_encoder`/`audio_encoder`. Param paths mirror HF
  (`talker/...`, `talker/code_predictor/...`, `code2wav/...`). `Qwen3OmniTalker` derives its MaxText sub-config from the thinker
  config (`qwen3_omni_talker.talker_config`) so it reuses `AttentionWithNorm`/`RoutedMoE`/`MlpBlock`; its KV cache budget is
  `max_prefill_predict_length + 16` prefill + `talker_max_new_tokens` AR, at the thinker's decode batch.
* **One checkpoint**: `to_maxtext.py model_name=qwen3-omni-30b-a3b use_multimodal=true enable_audio_output=true` writes thinker +
  speech stack (`QWEN3_OMNI_MOE_*` mapping gains the talker entries when the flag is set). Thinker-only checkpoints keep working
  with the default `enable_audio_output=false` (no talker params are created, so nothing is missing at load).
* **Decode**: `NNXDecoder` sows the merged input embeddings + hidden state after `accept_hidden_layer` layers (gated on
  `enable_audio_output`); `maxengine._prefill_jit` returns them as `prefix["talker_inputs"]`; `decode.py` pops them before `insert`
  and calls `qwen3_omni_talker_pipeline.generate_speech(config, engine, params, ...)`, which splits `engine.model_ar.talker` /
  `.code2wav` graphdefs + param sub-states out of the loaded model (no second mesh/checkpoint) and runs prefill → AR frames →
  Code2Wav → WAV.
* **Training-ready**: the speech stack is an ordinary sub-module of `Transformer` (params under `talker`/`code2wav`, train-mode
  construction supported, code predictor runs teacher-forced over a full `num_code_groups` window), so a future talker loss only
  needs to call `model.talker(...)`/`model.talker.code_predictor(...)` and mask params by prefix.

## 4. Bugs & resolutions
* HF `Qwen3OmniMoeTalkerConfig()` (transformers 5.16) has no `spatial_merge_size` default → tests pass `spatial_merge_size=2`.
* `jax.lax.conv_transpose(..., transpose_kernel=True)` must read the PyTorch `ConvTranspose1d` weight `[in, out, k]`
  with `dimension_numbers=("NCH", "OIH", "NCH")` (the weight *is* the forward-conv kernel); `"IOH"` fails shape checks.
* `maxengine._insert_jit` donates the prefill prefix → `decode.py` pops `prefix["talker_inputs"]` before `insert`.
* `MaxEngine.load_params` materializes the non-Param/non-Cache ("rest") state of the thinker only; the talker's
  `code_predictor` attention carries 2 RNG leaves that are missing from it, so `nnx.merge(talker_graphdef, params, cache,
  rest)` failed with "expected 579 leaves, got 577". `qwen3_omni_talker_pipeline._rest_state` materializes the sub-module's
  abstract rest state (RNG key 0 / zero counters) whenever the engine's does not cover it.
* `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-omni-30b-a3b/unscanned/2026-07-01-15-47` uses the old `decoder.layers.N`
  layout and no longer loads; use `gs://maxtext-model-checkpoints/qwen3-omni-30b-a3b/unscanned/0/items` (the e2e script's).
* JetStream's HF tokenizer sends `Bearer <empty>` when no token is configured → export `HF_TOKEN` (see `decode_omni_audio.sh`).
* fp32 talker + megablox GMM runs out of scoped VMEM on v5p (`CompileTimeScopedVmemOom`) → for fp32 parity runs pass
  `megablox=false sparse_matmul=false` (bf16 default path is fine).
* `decord` / `librosa` were missing in `venv1`; `librosa>=1.0` cannot read mp4 audio (no audioread/ffmpeg fallback) →
  `pip install decord "librosa<1.0" audioread` + `sudo apt-get install ffmpeg` (now done on the v5p2 VM).
* transformers 5.16 stores MoE experts fused (`experts.gate_up_proj`/`down_proj`); the HF reference script fuses the
  per-expert checkpoint tensors before `load_state_dict` (otherwise experts stay random and generation is garbage).

## 5b. Parity / replay workflow (no thinker reload)
```bash
# decode.py also writes <wav>.thinker_outputs.npz and <wav>.talker_debug.npz. Replay only the talker (either dump works;
# the replay writes its own <out>.talker_debug.npz so the HF reference can consume it):
python -m maxtext.inference.qwen3_omni_talker_pipeline src/maxtext/configs/base.yml model_name=qwen3-omni-30b-a3b \
  tokenizer_path=Qwen/Qwen3-Omni-30B-A3B-Instruct tokenizer_type=huggingface skip_jax_distributed_system=true enable_audio_output=true \
  load_parameters_path=/dev/shm/hengtaoguo/checkpoints/qwen3-omni-30b-a3b-audio/unscanned/0/items use_multimodal=true scan_layers=false \
  per_device_batch_size=1 attention=dot_product max_prefill_predict_length=64 max_target_length=192 \
  weight_dtype=float32 dtype=float32 matmul_precision=highest megablox=false sparse_matmul=false \
  --dump=/dev/shm/hengtaoguo/omni_text.wav.thinker_outputs.npz --output=/dev/shm/hengtaoguo/omni_text_v2_fp32.wav
# HF reference on the same inputs (CPU fp32, loads shards 13-15 only):
HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HOME=/dev/shm/hengtaoguo python tests/utils/qwen3_omni_talker_hf_reference.py \
  --dump=/dev/shm/hengtaoguo/omni_text_v2_fp32.wav.talker_debug.npz --output=/dev/shm/hengtaoguo/omni_text_v2_hf.wav
```

## 4b. Session wrap-up & pitfalls
* The thinker's prefill is the only thinker-side change: `NNXDecoder` sows `talker_input_embeddings` / `talker_accept_hidden_states`
  (gated on `enable_audio_output` so training/text decode are unaffected) and `maxengine._prefill_jit` returns them as `prefix["talker_inputs"]`.
* The talker consumes *projected* thinker states: `hidden_projection(hidden_states[24])` at multimodal token positions,
  `text_projection(hidden_states[0])` elsewhere; generated tokens use plain thinker embeddings. Only the first generated token enters the
  prefill; the rest arrive one per talker step via `trailing_text_hidden` (then `tts_pad`).
* Talker runs at the thinker's decode batch (stream 0 broadcast) with MaxText `Attention` KV caches: one AR-mode module serves both
  PREFILL and AUTOREGRESSIVE calls (cache shapes are identical, so no `insert` step is needed).
* Code predictor is cache-free over a fixed 16-position window (15 unrolled calls inside one jit) – exact and simple.
* Greedy decoding in bf16 diverges from the fp32 reference after a few frames (chaotic AR process) but still yields fluent,
  correctly-terminated speech; use `weight_dtype=float32 dtype=float32 matmul_precision=highest megablox=false sparse_matmul=false`
  for bit-level parity work. Sampling (`talker_do_sample=true`) matches HF's default generation settings.
* Outputs: `/dev/shm/hengtaoguo/omni_text.wav`, `/dev/shm/hengtaoguo/omni_video.wav` (+ `omni_text_v2_fp32.wav`, `omni_text_v2_hf.wav`,
  debug npz files).

## 5. Reproduction commands
```bash
# (VM) one-off: fetch only the talker/code2wav shards + configs
HF_HOME=/dev/shm/hengtaoguo python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Omni-30B-A3B-Instruct', allow_patterns=['*.json','model-0001[345]-of-00015.safetensors'])"

# full checkpoint conversion incl. speech stack (CPU, eager; needs the whole HF repo in HF_HOME)
cd ~/projects/maxtext && HF_HOME=/dev/shm/hengtaoguo PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml model_name=qwen3-omni-30b-a3b base_output_directory=/dev/shm/hengtaoguo/checkpoints/qwen3-omni-30b-a3b-audio/unscanned \
  scan_layers=false use_multimodal=true enable_audio_output=true hardware=cpu skip_jax_distributed_system=True weight_dtype=bfloat16 \
  --lazy_load_tensors=False --simulated_cpu_devices_count=16

# unit tests (CPU)
PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest tests/unit/qwen3_omni_talker_layers_test.py -q

# decode with audio (see decode_omni_audio.sh in the repo root): text or video (+audio-in-video) prompt
bash decode_omni_audio.sh text    # -> /dev/shm/hengtaoguo/omni_text.wav
bash decode_omni_audio.sh video   # -> /dev/shm/hengtaoguo/omni_video.wav (mini.mp4)
```
Key decode flags: `enable_audio_output=true audio_output_path=out.wav` with a checkpoint converted with `enable_audio_output=true`
(optional `speaker=ethan|chelsie|aiden`, `talker_do_sample`, `talker_top_k/top_p/temperature`, `talker_repetition_penalty`,
`talker_max_new_tokens`).
