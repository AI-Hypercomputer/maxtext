# Qwen2.5 Dense

Qwen2.5 is the latest series of large language models by Qwen, released in September 2024. The models use a dense
transformer architecture. You can find more information in the [blog](https://qwenlm.github.io/blog/qwen2.5/). The currently supported models
are [qwen2.5-7b](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
and [qwen2.5-14b](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).

## Running the End-to-End Test

The `test_qwen2.5-14b.sh` script automates the following steps:

1. **Checkpoint Conversion**: Converts the Hugging Face checkpoint to MaxText-compatible format (scanned and unscanned).
1. **Logit Check**: Verifies the forward pass logits against the Hugging Face model.
1. **SFT**: Runs a Supervised Fine-Tuning (SFT) job.

### Prerequisites

- Ensure you have write access to a GCS bucket for output logs and checkpoints.

### Usage

To run the test for Qwen 2.5 14B:

```bash
export BASE_OUTPUT_PATH=gs://your-gcs-bucket/
bash tests/end_to_end/tpu/qwen/dense/qwen2.5-14b/test_qwen2.5-14b.sh
```

This will:

- Download the `Qwen/Qwen2.5-14B-Instruct` model.
- Convert it to MaxText format.
- Run validation and training tests.
- Store artifacts in `${BASE_OUTPUT_PATH}/qwen2.5-14b/<timestamp>`.

### Qwen 2.5 7B

Similarly, for Qwen 2.5 7B:

```bash
export BASE_OUTPUT_PATH=gs://your-gcs-bucket/
bash tests/end_to_end/tpu/qwen/dense/qwen2.5-7b/test_qwen2.5-7b.sh
```
