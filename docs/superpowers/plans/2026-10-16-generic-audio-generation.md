# Generic audio generation implementation plan

**Goal:** Add model-independent audio construction and generation without changing existing checkpoint paths, shapes, or dtypes.

**Architecture / approved design:** Keep the current packages. `layers/audio_decoders.py` dispatches component construction and declares required text features. `inference/audio_generation.py` selects a generator with a common request/result contract. Qwen retains its conditioning, codec loop, and vocoder algorithm. Decoder feature capture and MaxEngine transport are independent of audio. Attach components directly under `Transformer.talker` and `Transformer.code2wav`, preserving their child attributes and initialization order.

**Execution:** Implement inline in the existing checkout, preserving the user's current branch and checkpoints. No commits or publication are required.

## Constraints

- Existing Qwen configurations and `enable_audio_output=true` continue to select Qwen automatically.
- Audio-disabled models create no speech parameters and need no Qwen imports.
- WAV output and both replay dump formats remain compatible.
- Feature capture is prefill-only initially; scanned hidden-layer capture is rejected explicitly.
- Existing `prefill` callers retain the two-result API; opt-in feature consumers receive features outside the donated prefix.

## Tasks

1. Add regression tests for backend selection, feature capture/extraction, waveform writing, and old parameter-tree compatibility. Run the current parity tests as baseline, and record the original complete parameter manifest before changing construction.
2. Introduce audio construction dispatch, generic decoder feature requests, and a separate Qwen Code2Wav module. Preserve the attachment names and constructor order. Add shared configuration controls with backward-compatible automatic selection.
3. Introduce a shared generation request/result/protocol and runtime component access. Adapt Qwen's generator while retaining its public replay helpers. Replace Qwen-specific capture/extraction in the decoder and MaxEngine; simplify `decode.py` to invoke the selected generator.
4. Run targeted CPU tests, full parameter manifest comparison, format/lint checks, and related engine tests. Review the final diff and report any environment limitations.

## Review focus

- Existing speech checkpoints restore with exactly the same leaf names, shapes, and dtypes.
- Existing text-only callers keep their prefill return structure.
- Captured features survive insertion donation and are not captured during autoregressive steps.
- Unsupported audio backends and invalid hidden-layer requests fail before generation.
- Qwen dump names/content and replay entry point remain usable.

## Verification

- Original baseline: 8 talker parity/schema tests passed.
- Final CPU run: 21 tests passed, plus 5 subtests, across `audio_generation_test.py`, `qwen3_omni_talker_layers_test.py`, and `maxengine_nnx_test.py`.
- Complete tiny-model checkpoint schema: all 184 parameter paths, shapes, and dtypes match the recorded pre-refactor manifest.
- Actual tiny-model integration: generated waveform and both replay dumps; features survived insertion donation and subsequent generation; legacy prefill returns two results.
- Independent review found no correctness regressions and verified moved Code2Wav bodies are unchanged.
- Formatting and diff whitespace checks passed. Pylint findings remain in existing decoder lambdas and unrelated configuration code; the test lint invocation excludes the previously observed Transformers constructor-inference findings (E1123).
- No full-size existing checkpoint restore or TPU run was performed.
