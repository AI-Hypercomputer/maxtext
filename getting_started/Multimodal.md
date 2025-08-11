

# Multimodal Support on MaxText

## Introduction

This document provides a guide to use MaxText for multimodal training and inference. Now MaxText has the capabalities with image+text as input and generate text output. Currently, the supported models are:

- Gemma3: 4B, 12B, 27B
- LLama4: Scout, Maverick

## Checkpoint Conversion

Recently we have onboarded a new centralized tool for bidirectional checkpoint conversion between MaxText and HuggingFace (README). And the Gemma3 family has been migrated towards it. Use this command to convert an unscanned checkpoint from HuggingFace to MaxText, and save it to `MAXTEXT_CKPT_GCS_PATH`:

```shell
python -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=gemma3-4b \
    hf_access_token=${HF_ACCESS_TOKEN} \
    base_output_directory=${MAXTEXT_CKPT_GCS_PATH} \
    use_multimodal=true \
    scan_layers=false
```

For the Llama4 model families, we are using a different conversion script:

```shell
python -m MaxText.llama4_ckpt_unscanned \
    --model-size=llama4-17b-16e \
    --huggingface-checkpoint=True \
    --base-model-path=${LOCAL_HF_MODEL_PATH}$ \
    --maxtext-model-path=${MAXTEXT_CKPT_GCS_PATH}$
```

## Multimodal Decode
Currently, MaxText supports multimodal decode with text + one image as input and text as output. Multiple-image support is coming soone. Each model may have their own chatting template during pretraining, so we implemented those templates in `multimodal_utils.py`. The user may need to specify where you want to put the image token in the field `prompt`. Of note, Gemma3 is using `<start_of_image>` as the image placeholder token and Llama4 with `<|image|>`. To run a forward pass and check the model's output, use the following command:

```shell
# Gemma3 decode
python -m MaxText.decode \
    MaxText/configs/base.yml \
    model_name=gemma3-4b \
    tokenizer_path=assets/tokenizer.gemma3 \
    load_parameters_path=${GCS_CKPT}$ \
    per_device_batch_size=1 \
    run_name=ht_test \
    max_prefill_predict_length=272 \
    max_target_length=300 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=false \
    use_multimodal=true \
    prompt='Describe image <start_of_image>' \
    image_path='MaxText/test_assets/test_image.jpg' \
    attention='dot_product'
```

You are expected to see this outcome:
```
Input `<start_of_turn>user
Describe image <start_of_image><end_of_turn>
<start_of_turn>model
` -> `Here's a description of the image:

**Overall Impression:** The image is a bright, expansive cityscape view of Seattle, Washington, with`
```

Since Llama4-Scout is a 108B model, we suggest run decoding on a TPU cluster such as v5p-16. For Llama4 decoding, using this command:

```shell
python -m MaxText.decode \
    MaxText/configs/base.yml \
    model_name=llama4-17b-16e \
    tokenizer_path=meta-llama/Llama-4-Scout-17B-16E \
    per_device_batch_size=1 
    run_name=ht_test 
    max_prefill_predict_length=744 
    max_target_length=754 
    steps=1 
    async_checkpointing=false 
    scan_layers=false 
    use_multimodal=true 
    load_parameters_path=${MAXTEXT_CKPT_GCS_PATH} 
    prompt=\'\<\|image\|\>Describe\ this\ image\ in\ two\ sentences.\' 
    image_path=\'MaxText/test_assets/test_image.jpg\' 
    attention=\'dot_product\' 
    hf_access_token=${HF_ACCESS_TOKEN} 
```

## Supervised Fine-Tuning (SFT)


```shell
python -m MaxText.sft_trainer MaxText/configs/sft-vision-chartqa.yml \
    run_name=$idx \
    model_name=$MODEL_NAME tokenizer_path="google/gemma-3-4b-pt" \
    per_device_batch_size=1 \
    max_prefill_predict_length=1024 max_target_length=2048 \
    steps=$SFT_STEPS \
    scan_layers=$SCAN_LAYERS async_checkpointing=False \
    attention=dot_product \
    dataset_type=hf hf_path=parquet hf_access_token=$HF_TOKEN \
    hf_train_files=gs://aireenmei-multipod/dataset/hf/chartqa/train-* \
    base_output_directory=$BASE_OUTPUT_DIRECTORY \
    load_parameters_path=$UNSCANNED_CKPT_PATH \
    dtype=bfloat16 weight_dtype=bfloat16 sharding_tolerance=0.05
```
