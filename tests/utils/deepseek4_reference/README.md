# DeepSeek-V4 official reference (CPU / autograd)

Vendored copy of the official DeepSeek-V4 PyTorch inference model, used as a
numerical reference by MaxText DeepSeek-V4 tests.

## Provenance

- Upstream: Hugging Face repo `deepseek-ai/DeepSeek-V4-Flash`, snapshot
  `60d8d70770c6776ff598c94bb586a859a38244f1`, file `inference/model.py`
  (sha256 `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`).
- License: MIT, Copyright (c) 2023 DeepSeek. Full text in [LICENSE](LICENSE).

## Files

- `model.py`: upstream `inference/model.py` with a prepended comment header
  (Apache + MIT notices, `# pylint: skip-file`, `# fmt: off`). Below it, every
  edit is on a line containing `# [CHANGE]` with its justification; all other
  lines are byte-identical.
  `tests/unit/deepseek_v4_reference_self_test.py` strips the header, reverts the
  edits and checks the result against the upstream sha256.
- `kernel.py`: pure-torch replacement for upstream `inference/kernel.py`
  (tilelang, CUDA-only). `set_mode("train")` makes FP8/FP4 activation quant an
  identity; `set_mode("qat")` emulates quant-dequant with a straight-through
  gradient. The module docstring maps each function to upstream line ranges.
- `fast_hadamard_transform.py`: pure-torch replacement for the CUDA
  `fast_hadamard_transform` package (natural-order Sylvester Hadamard).
- `__init__.py`: `configure(dtype)`, `init_params()`, `fp64_promotion()`,
  `tiny_args()`, and the upstream sha256 constants.
- `encoding_dsv4.py`: official DeepSeek-V4 chat encoder (the model ships no
  Jinja chat template), upstream `encoding/encoding_dsv4.py` at the same
  snapshot (sha256
  `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0`), MIT.
  Byte-identical below a prepended license/lint header. Not imported by
  `__init__.py`; `tests/unit/deepseek_v4_sft_masking_test.py` loads it by path.
- `testdata/`: upstream `encoding/tests/test_{input,output}_{1..4}.{json,txt}`
  golden encoder fixtures at the same snapshot, byte-identical.

## Compressor kv_cache `[CHANGE]`

On its first call, upstream `Attention.forward` gives the compressor a view of
the attention cache (`self.compressor.kv_cache = self.kv_cache[:, win:]`,
upstream `model.py:491`). In the compressor, grad-carrying kv is written in
place into that view (upstream `model.py:374`) after upstream `model.py:523` has
written the base, and backward raises an in-place-on-leaf error. It fails when
`bsz == max_batch_size`, on a second forward, and at some sequence lengths.

With `torch.is_grad_enabled()` the vendored line gives the compressor its own
zero buffer instead; with grad disabled it keeps the upstream view.

- Prefill output is the same: upstream `model.py:525-526` consume the
  compressor's return value, not the shared cache.
- Decode would differ, because the attention cache no longer sees compressed
  entries written by the compressor.
- The assignment runs once (guarded by `self.compressor.kv_cache is None`), so
  the branch is fixed by the grad mode of the first forward; toggling grad mode
  afterwards does not switch it.

## Regenerating the diff

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --revision 60d8d70770c6776ff598c94bb586a859a38244f1 --local-dir /tmp/ds4 \
  --include "inference/model.py" "encoding/encoding_dsv4.py" "encoding/tests/*"
diff -u /tmp/ds4/inference/model.py tests/utils/deepseek4_reference/model.py
diff -u /tmp/ds4/encoding/encoding_dsv4.py tests/utils/deepseek4_reference/encoding_dsv4.py
diff -r /tmp/ds4/encoding/tests tests/utils/deepseek4_reference/testdata
```
