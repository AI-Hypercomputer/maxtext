# MaxText Omni: Heterogeneous Multimodal Model Stitching

This directory contains code and documentation for **MaxText-Omni**, a framework designed to assemble ("stitch"), train, and evaluate heterogeneous multimodal models within MaxText.

Here, we pair the Vision Tower from **Gemma 3 (4B)** with the language generation backbone of **Qwen 3 (4B)** using a custom trainable **Multi-Layer Perceptron (MLP) Vision Projector**.


## Getting Started

### Prerequisites

1.  **Google Cloud TPU VM:** A TPU slice (single- or multi-host) with JAX and MaxText installed.
2.  **Google Cloud Storage (GCS):** A GCS bucket to store converted weights, stitched checkpoints, and training logs.
3.  **Hugging Face Access Token (`HF_TOKEN`):** Required to download checkpoints.

### Setup

Export the required environment variables:

```bash
# 1. GCS root directory for checkpoints and experiment runs
export BASE_OUTPUT_DIRECTORY="gs://YOUR_BUCKET/omni-gemma3-qwen3/multimodal"

# 2. Your Hugging Face access token
export HF_TOKEN="<YOUR_HF_TOKEN>"

# 3. Add repository src/ to PYTHONPATH
export PYTHONPATH=src:${PYTHONPATH:-}
```

Optional configuration overrides:
```bash
export TRAIN_STEPS=50           # Override steps for fast smoke testing
export SCAN_LAYERS=true         # Layer scan optimization (default: true)
export EVAL_NUM_EXAMPLES=100    # Evaluation sample count (-1 for full test split)
export EVAL_SPLIT="test"        # Evaluation dataset split (default: test)
```


## Running the End-to-End Pipeline

### Quick Start

```bash
bash src/maxtext/experimental/omni_poc/maxtext_omni_pipeline_e2e.sh
```

### How It Works (Step-by-Step)

#### Step 1: Convert Hugging Face Checkpoints to MaxText Format
Uses `to_maxtext` to convert raw huggingface checkpoints from two original models into MaxText format:
```bash
# 1a. Convert Gemma 3 4B Vision Tower
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=gemma3-4b \
  base_output_directory=${BASE_OUTPUT_DIRECTORY}/converted/gemma3-4b \
  hf_access_token=${HF_TOKEN} \
  use_multimodal=true \
  scan_layers=true

# 1b. Convert Qwen 3 4B Language Decoder
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=qwen3-4b \
  base_output_directory=${BASE_OUTPUT_DIRECTORY}/converted/qwen3-4b \
  hf_access_token=${HF_TOKEN} \
  use_multimodal=false \
  scan_layers=true
```

#### Step 2: Stitch Subtrees into Unified Omni Checkpoint
Runs [`utils/stitch_checkpoint.py`](utils/stitch_checkpoint.py) to extract the `vision_encoder` from Gemma 3, the `decoder` from Qwen 3, initialize the fresh projector, and save the stitched base checkpoint:
```bash
python3 -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
  src/maxtext/experimental/omni_poc/maxtext-omni-gemma3-qwen3.yml \
  hf_access_token=${HF_TOKEN} \
  vision_load_path=${BASE_OUTPUT_DIRECTORY}/converted/gemma3-4b/0/items \
  llm_load_path=${BASE_OUTPUT_DIRECTORY}/converted/qwen3-4b/0/items \
  stitched_output_path=${BASE_OUTPUT_DIRECTORY}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items
```

#### Step 3: Stage 1 Alignment Pretraining (ChartNet)
Pretrains the MLP connector on chart summaries from `ibm-granite/ChartNet`. This step aligns the vision tower with the language model through our customized MLP.
```bash
python3 -m maxtext.experimental.omni_poc.train_sft_omni \
  src/maxtext/experimental/omni_poc/configs/pretrain-maxtext-omni-gemma3-qwen3-chartnet.yml \
  load_parameters_path=${BASE_OUTPUT_DIRECTORY}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items \
  base_output_directory=${BASE_OUTPUT_DIRECTORY}/pretrain_chartnet \
  run_name=pretrain_chartnet \
  hf_access_token=${HF_TOKEN}
```

#### Step 4: Stage 2 Supervised Fine-Tuning (ChartQA)
Fine-tunes the projector for visual question answering using `HuggingFaceM4/ChartQA`:
```bash
python3 -m maxtext.experimental.omni_poc.train_sft_omni \
  src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
  load_parameters_path=${PRETRAIN_FINAL_CKPT} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft_after_chartnet \
  run_name=sft_chartqa \
  hf_access_token=${HF_TOKEN}
```

#### Step 5: Multimodal Quality Evaluation (ChartQA Test Split)
Runs benchmark evaluation on the ChartQA test split using [`eval_sft_omni.py`](eval_sft_omni.py):
```bash
python3 -m maxtext.experimental.omni_poc.eval_sft_omni \
  src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
  load_parameters_path=${SFT_FINAL_CKPT} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY}/sft_after_chartnet \
  run_name=eval \
  hf_access_token=${HF_TOKEN} \
  --ckpt_type=sft \
  --num_examples=-1 \
  --hf_eval_split=test
```

---

## Interactive Single-Image Decoding

For quick qualitative testing and visual validation without running a training or benchmark job, use [`utils/decode_omni.py`](utils/decode_omni.py):

```bash
python3 -m maxtext.experimental.omni_poc.utils.decode_omni \
  --config_path=src/maxtext/experimental/omni_poc/maxtext-omni-gemma3-qwen3.yml \
  --checkpoint_path=gs://YOUR_BUCKET/path/to/checkpoint/0/items \
  --image_path=/path/to/test_chart.png \
  --prompt="What is the highest value in this chart?" \
  --max_decode_steps=64
```


## Running Verification Tests

Unit and integration tests are organized under [`tests/`](tests/):

```bash
# Set base directory for checkpoint tests (pointing to your pipeline output root)
export OMNI_TEST_BASE_DIR="${BASE_OUTPUT_DIRECTORY}"

# 1. Verify checkpoint stitching, layer counts, and weight equality
python3 -m unittest src/maxtext/experimental/omni_poc/tests/stitch_checkpoint_test.py

# 2. Verify tokenizer placeholder expansion and multimodal offset calculations (no GPU/GCS needed)
python3 -m unittest src/maxtext/experimental/omni_poc/tests/processor_maxtext_omni_test.py

# 3. Checkpoint diff audit: verify only projector weights changed after SFT
python3 -m maxtext.experimental.omni_poc.tests.compare_sft_checkpoint_test \
  src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
  --stitched_checkpoint_path=${BASE_OUTPUT_DIRECTORY}/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items \
  --sft_checkpoint_path=${BASE_OUTPUT_DIRECTORY}/sft_after_chartnet/sft_chartqa/checkpoints/19/items

# 4. Verify custom projector forward pass and parameter freezing invariants on Qwen3-VL
python3 -m maxtext.experimental.omni_poc.tests.custom_vision_projector_test \
  --load_parameters_path=${BASE_OUTPUT_DIRECTORY}/converted/gemma3-4b/0/items
```

## Key Concepts

*   **Multimodality as Composition:** Combine an arbitrary pretrained vision encoder and text decoder by training only a lightweight connector between them while freezing both backbones.
*   **Checkpoint Stitching:** Extracts parameter subtrees from two separate checkpoints (`vision_encoder` from vision, `decoder` from LLM) into a unified checkpoint with a fresh projector.
*   **Dynamic MLP Connector:** Dynamically adapts any `(vision_dim, text_dim)` pair (e.g., Gemma 3's 1152d to Qwen 3's 2560d).
*   **Placeholder Token Masking:** Injects `<|image_pad|>` tokens into text prompts, which are replaced with projected visual embeddings during the forward pass.
*   **Frozen Backbone Training:** Keeps the vision tower and LLM decoder frozen, training only the connector (`trainable_parameters_mask: ["custom_linear"]`) for compute-efficient alignment.


## Architecture Overview

```
       [ Input Image (896x896) ]                     [ Text Prompt ]
                   |                                        |
                   v                                        v
      +------------------------+                +-----------------------+
      |  Gemma 3 Vision Tower  |                |   Qwen 3 Tokenizer    |
      |   (27 layers, ViT)     |                |  (<|vision_start|>... |
      +------------------------+                |   <|image_pad|> * 256 |
                   |                            |   <|vision_end|>)     |
         (256 tokens x 1152d)                   +-----------------------+
                   |                                        |
                   v                                        |
      +------------------------+                            |
      |  Custom MLP Projector  |                            |
      |   3-Layer GELU + Bias  |                            |
      |   (1152 -> 4096 ->     |                            |
      |         2560d)         |                            |
      +------------------------+                            |
                   |                                        |
         (256 tokens x 2560d)                               |
                   \                                        /
                    \                                      /
                     v                                    v
                  +------------------------------------------+
                  |           Interleaved Embeddings         |
                  +------------------------------------------+
                                       |
                                       v
                  +------------------------------------------+
                  |           Qwen 3 LLM Decoder             |
                  |   (36 layers, 2560 hidden dimension)     |
                  +------------------------------------------+
                                       |
                                       v
                             [ Text Completion ]
```

### Model Specifications

| Component | Donor Model | MaxText Config Setting | Details |
| :--- | :--- | :--- | :--- |
| **Vision Tower** | Gemma 3 4B | `vision_encoder_block: "gemma3"` | 27 ViT layers, 896x896 image size, 14x14 patch size, 1152 hidden dim $\to$ 256 visual tokens |
| **Vision Projector** | Random Init | `vision_projector_type: "customized_vision_projector"` | 3-layer MLP (`1152 -> 4096 -> 2560`) with GELU activations and bias (~12.5M params) |
| **Language Decoder** | Qwen 3 4B | `decoder_block: "qwen3"` | 36 layers, GQA (32 query / 8 KV heads), 2560 hidden dim, 151,936 vocab |


## Codebase Organization

```
src/maxtext/experimental/omni_poc/
├── README.md                                       # Documentation (this file)
├── maxtext-omni-gemma3-qwen3.yml                   # Base architecture config for the stitched model
├── maxtext_omni_pipeline_e2e.sh                    # Automated 5-stage end-to-end pipeline script
├── prepare_checkpoint.sh                           # Standalone conversion and stitching runner
├── train_sft_omni.py                               # Training entry point (Pretraining & SFT)
├── eval_sft_omni.py                                # Benchmark evaluation entry point
│
├── configs/
│   ├── pretrain-maxtext-omni-gemma3-qwen3-chartnet.yml # Stage 1 alignment configuration (ChartNet)
│   └── sft-maxtext-omni-gemma3-qwen3.yml           # Stage 2 SFT configuration (ChartQA)
│
├── utils/
│   ├── stitch_checkpoint.py                        # Checkpoint surgery: merges subtrees from Model A & B
│   ├── decode_omni.py                              # Interactive single-sample multimodal decoding
│   └── processor_maxtext_omni.py                   # Tokenizer offsets & special token placeholder expansion
│
└── tests/
    ├── stitch_checkpoint_test.py                   # Tests parameter shapes, subtrees, and layer matching
    ├── custom_vision_projector_test.py             # Tests projector shapes, forward pass, and frozen gradients
    ├── compare_sft_checkpoint_test.py              # Audits checkpoint weights before and after SFT
    └── processor_maxtext_omni_test.py              # Tests token expansions, masking, and chat formatting
```

---
