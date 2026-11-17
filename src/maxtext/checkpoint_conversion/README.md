# Checkpoint conversion utilities

This directory contains tools for converting and loading checkpoints between
Hugging Face and MaxText formats:

- **Offline conversion (`to_maxtext.py` / `to_huggingface.py`):** See the
  [convert_checkpoint](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/convert_checkpoint.html)
  guide.
- **On-the-fly loading (`source_checkpoint_layout="safetensors_dynamic"`):**
  Load Hugging Face `.safetensors` directly into sharded device memory without
  offline conversion - see the
  [safetensors_dynamic](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/safetensors_dynamic.html)
  guide for architecture, parity results, and TPU benchmarks.
