# Qwen3.8-Flash-Next bringup

## Scope and status

Branch: `hengtaoguo-exp10`. Implemented the unscanned text backbone through native
MaxText decoding and central checkpoint conversion. Vision and MTP are excluded.

- Phase 1, source discovery and representative slice: DONE.
- Phase 2, native layers and reference comparisons: DONE for text backbone.
- Phase 3, central conversion and mini Orbax checkpoint: DONE.
- Phase 4, real-weight logits and cached decode: DONE for four-layer fixture.
- Full 48-layer conversion, TPU validation, and performance benchmarking: NOT STARTED.
- Inverse mapping and shapes are tested; full HF export/roundtrip has not run.

Pinned source: `Qwen/Qwen3.8-Flash-Next`, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Reference implementation: `../transformers/src/transformers/models/qwen4_exp/`.

## Architecture findings

Hidden size 2560, 48 layers, vocabulary 248320. Repeating three Gated DeltaNet
layers followed by one Qwen Sparse Attention layer; layers 0–3 are the minimal
representative prefix. Query heads 24, KV heads 2, head dimension 256, rotary
dimension 64, theta 10000000, RMS epsilon 1e-6. GDN uses 16 key heads, 48 value
heads, dimensions 128, convolution width 4, and sigmoid output gates. MoE has
512 experts, top 10, expert/shared width 640. Gated residual hyper-connections
use four streams and rank 320; grouped RMSNorm is zero-centered `(1 + weight)`.

PLE config `ple_layer_ids=[2]` is ONE-BASED: checkpoint layer is `layers.1`.
Two n-gram orders, eight heads each, 160 features per head. The 128 independent
source table tensors each have shape `(2500012,160)`, totaling 95.37 GiB BF16.
Hashing requires exact signed int64 multiplication/overflow, source EOS resets,
and checkpoint multipliers, head offsets, and prime vocabulary sizes.

## PLE conversion and validation workaround

Mini validation fetches exact rows using strict HTTP byte ranges, retaining sorted
global row IDs and original BF16 values. The fixture has **1,336 rows** and includes
all three prompts, the HF 16-token rollout, and validation padding transitions.
Missing rows fail explicitly. PLE remains active; this fixture supports only the
recorded inputs and must never be used as a general serving checkpoint.

Full conversion preserves all 128 source shards as independent leaves. Remote
streaming caches headers only, reads one tensor through bounded ranges, and writes
one lazy leaf before materializing the next. It avoids concatenating the 95 GiB
table or downloading entire containing safetensors files. Full text-backbone
mapping covers all **1,294 source tensors**; logical BF16 output is **329.58 GiB**.
Use external output storage rather than `/dev/shm` for the full checkpoint.

PLE rows are replicated because 2,500,012 is indivisible by common tensor mesh
sizes. Its 160 columns use the existing `embed_vocab` logical rule. Choose an
FSDP/context/expert axis product dividing 160, or override the rule. This layout
avoids row-divisibility failures but is not a performance-tuned distributed lookup.

## Validation results

- Standard forward-pass checker passed all three real-weight prompts:
  `I love to`, `The capital of France is`, `Hello world!`.
- Maximum KL divergence: **7.6079e-8**, below the 1e-3 threshold.
- Native cached greedy decode matched all **16 HF token IDs** exactly.
- **29 focused tests passed**. They cover official HF primitives, PLE hashing/lookup/convolution,
  recurrent and QSA caching beyond selection budget, padding, sampler int32 dtype,
  range bounds, integer conversion, inverse mappings, and fresh-process Orbax restore.
- Mini source excluding PLE: 21.756 GiB BF16. Converted float32 logical weights:
  43.5 GiB. Conversion took 4.89 minutes; measured peak process memory 17.81 GB.
- Evidence: `/dev/shm/qwen38_bringup/{logits.log,decode_verified.log,decode_comparison.json,unit_final.log}`.

## Reproduction

Run from the MaxText repository. This environment uses `../venv1/bin/python`:
Python 3.12, JAX 0.11.1, Torch 2.11 CPU, local Transformers 5.16 dev. The TPU probe
failed on a libtpu lockfile; no lockfile was removed. Verification ran on CPU.
The shared-memory filesystem is 221 GiB and already contains other users' files;
check available space and RAM before preparing the fixture. Do not remove others'
artifacts. Configure all new caches and temporary files in shared memory:

```bash
export HF_HOME=/dev/shm/qwen38_bringup/hf
export TMPDIR=/dev/shm
export MPLCONFIGDIR=/dev/shm/qwen38_bringup/matplotlib
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export PYTHONPATH=src
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" /dev/shm/qwen38_bringup/jax_cache

../venv1/bin/python -m tests.utils.qwen38_slice
../venv1/bin/python -m tests.utils.qwen38_golden
../venv1/bin/python -m tests.utils.qwen38_golden --augment-padding-only
```

The golden utility runs the official HF text model with demand-fetched PLE rows;
it freezes the row set before JAX validation. The final row count is 1,336 for
this pinned revision and these inputs. Verify the count in `fixture.json`.

```bash
../venv1/bin/python -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml model_name=qwen3.8-flash-next \
  override_model_config=true base_num_decoder_layers=4 qwen38_ple_compact_rows=1336 \
  base_output_directory=/dev/shm/qwen38_bringup/mini_orbax run_name=qwen38_mini \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  per_device_batch_size=1 max_target_length=32 max_prefill_predict_length=16 \
  attention=dot_product weight_dtype=float32 dtype=float32 \
  jax_cache_dir=/dev/shm/qwen38_bringup/jax_cache \
  --save_dtype=float32 --simulated_cpu_devices_count=1 \
  --hf_model_path=/dev/shm/hf_mini/qwen3.8-flash-next_4layers

../venv1/bin/python -m tests.utils.forward_pass_logit_checker \
  src/maxtext/configs/base.yml model_name=qwen3.8-flash-next \
  override_model_config=true base_num_decoder_layers=4 qwen38_ple_compact_rows=1336 \
  load_parameters_path=/dev/shm/qwen38_bringup/mini_orbax/0/items \
  tokenizer_path=/dev/shm/hf_mini/qwen3.8-flash-next_4layers tokenizer_type=huggingface \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  per_device_batch_size=1 max_target_length=32 max_prefill_predict_length=16 \
  attention=dot_product weight_dtype=float32 dtype=float32 matmul_precision=highest \
  jax_cache_dir=/dev/shm/qwen38_bringup/jax_cache \
  --golden_logits_path=/dev/shm/qwen38_bringup/golden_mini_logits.jsonl --max_kl_div=0.001

../venv1/bin/python -m maxtext.inference.decode --verbosity=1 \
  src/maxtext/configs/base.yml model_name=qwen3.8-flash-next \
  override_model_config=true base_num_decoder_layers=4 qwen38_ple_compact_rows=1336 \
  load_parameters_path=/dev/shm/qwen38_bringup/mini_orbax/0/items \
  tokenizer_path=/dev/shm/hf_mini/qwen3.8-flash-next_4layers tokenizer_type=huggingface \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  per_device_batch_size=1 max_target_length=31 max_prefill_predict_length=16 \
  attention=dot_product weight_dtype=float32 dtype=float32 matmul_precision=highest \
  jax_cache_dir=/dev/shm/qwen38_bringup/jax_cache \
  prompt='I love to' decode_sampling_strategy=greedy

../venv1/bin/python -m pytest tests/unit/qwen38_ple_test.py \
  tests/unit/qwen38_conversion_test.py tests/unit/safetensors_range_test.py -q
```

Expected generated token IDs:
`[56493,9847,10172,170800,66141,4330,236451,55444,71343,159275,104353,203356,73616,18595,60758,4032]`.

## Full conversion command (documented, NOT RUN)

After the mini pass, replace `YOUR_BUCKET` with a writable destination. Keep the
source revision pinned. Defaults select all 48 text layers; compact rows must be
zero. One simulated CPU device keeps conversion from eagerly materializing a
multi-device sharded model. Set credentials for the output destination normally.

```bash
../venv1/bin/python -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml model_name=qwen3.8-flash-next qwen38_ple_compact_rows=0 \
  base_output_directory=gs://YOUR_BUCKET/qwen38-flash-next run_name=qwen38_full \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true \
  per_device_batch_size=1 max_target_length=32 max_prefill_predict_length=16 \
  attention=dot_product weight_dtype=bfloat16 dtype=bfloat16 \
  jax_cache_dir=/dev/shm/qwen38_bringup/jax_cache \
  --save_dtype=bfloat16 --lazy_load_tensors=true --stream_remote_tensors=true \
  --simulated_cpu_devices_count=1 \
  --revision=de4b8e4d43b917e7706784d8bb445c9af86a3540
```

## Bugs resolved and remaining limits

Integer checkpoint buffers now retain int64 instead of being cast to float.
JAX x64 is required for hashes, while sampling returns int32 to match engine state.
Host callbacks convert their input to NumPy before reduction; using JAX reduction
inside a CPU callback deadlocked. PLE/GDN/QSA have native fixed-shape caches, with
explicit batch-axis insertion into MaxEngine slots. Inactive QSA scatter writes
are dropped so repeated padding positions cannot overwrite valid keys.

Lazy Orbax writes are sequential unstarted coroutines. Leaf metadata uses the
standard `np.ndarray` type string for fresh-process restore. The first temporary
mini checkpoint's old leaf type metadata was repaired in place after discovering
this issue; tensor data was unchanged. New conversion writes correct metadata,
covered by a fresh-process regression. Standard readers may warn about falling
back from the custom checkpoint item handler.

Only contiguous, unpacked text with right padding is supported. Packed segments,
left padding, scanned layers, vision and MTP are excluded. Sparse mini parity does
not establish arbitrary-input or full-model parity. Full-table distributed serving,
BF16 full-model numerics, throughput, and long-context behavior need subsequent
validation. QSA gathers selected blocks instead of dense quadratic attention;
MoE gathers per token to bound temporary memory. Both prioritize correctness over
peak throughput. Conversion streaming improves onboarding memory and I/O; it
does not itself speed steady-state decoding. PLE decode looks up 16 rows per token,
so full-table device residency and communication are the next performance concerns.
