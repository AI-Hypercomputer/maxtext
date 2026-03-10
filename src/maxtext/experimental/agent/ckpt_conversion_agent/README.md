# Checkpoint conversion agent
The agent is used to automate the model-specific mappings of checkpoint conversion.  It is designed to cooperate with the new checkpoint conversion [framework](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/checkpoint_conversion).

## Quick starts
To begin, you'll need:

1. A Google atccount.
2. An API key (create one in [Google AI Studio](https://aistudio.google.com/app/apikey)).
3. Install the Google Generative AI Python library:
```
pip install -q -U "google-genai>=1.0.0"
```
4. The target/source models must be implemented in MaxText and Hugging Face and we can retrieve random weights to learn its parameter names and tensor shapes.

5. A full run of the agent typically takes about 30 minutes.

## 1. Prepare the context file

The agent requires context files about the target and source model's parameter names and tensor shapes. You can generate them using the [`save_param.py`](../ckpt_conversion_agent/utils/save_param.py) script. The output directory defined by `config.base_output_directory`. The default is `src/MaxText/experimental/agent/ckpt_conversion_agent/context/<model_name>` folder.
```bash
python3 -m maxtext.experimental.agent.ckpt_conversion_agent.utils.save_param src/maxtext/configs/base.yml \
  per_device_batch_size=1 run_name=param_<model_name> model_name=<model_name> scan_layers=false \
  --hf_model_config=<hf_model_id>
```
After it, you can get two `*.json` files in `config.base_output_directory` folder.

## 2. Run the agent

### 2.1 Step 1: Generate the conversion plan and check DSL

```bash
python3 -m maxtext.experimental.agent.ckpt_conversion_agent.step1 --target_model=<model_name> \
  --dir_path=src/MaxText/experimental/agent/ckpt_conversion_agent --api_key=<Your-API-KEY>
```

Our engineer should check the `src/MaxText/experimental/agent/ckpt_conversion_agent/outputs/proposed_dsl.txt` for potential new DSL and assess if it's needed. Then we need to add this ops into `src/MaxText/experimental/agent/ckpt_conversion_agent/context/dsl.txt`.

### 2.2 Step 2: Generate mappings

```bash
python3 -m maxtext.experimental.agent.ckpt_conversion_agent.step2 --target_model=<model_name> \
  --dir_path=src/MaxText/experimental/agent/ckpt_conversion_agent --api_key=<Your-API-KEY>
```

## Evaluation and Debugging
There are two primary ways to check the generated code.

### Automated Evaluation (with Ground-Truth Code)

You can automatically verify the output by comparing the generated code against a "ground-truth" version—a manually-written, correct implementation. This is the fastest way to confirm the conversion works as expected.

\*\**Note: This method is only possible if you have a ground-truth code implementation available for comparison.*

```bash
python3 -m maxtext.experimental.agent.ckpt_conversion_agent.evaluation --files ground_truth/<model>.py \
  outputs/hook_fn.py --api_key=<Your-API-KEY> --dir_path=src/MaxText/experimental/agent/ckpt_conversion_agent
```

### Manual Debugging (No Ground-Truth Code)
If a ground-truth version isn't available, you'll need to debug the conversion manually. The recommended process is to:
1. Add the model mappings into [checkpoint conversion framework](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/checkpoint_conversion/README.md#adding-support-for-new-models).

2. Execute the conversion process layer-by-layer, using [to_maxtext.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/checkpoint_conversion/README.md#hugging-face-to-maxtext) or [to_huggingface.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/checkpoint_conversion/README.md#maxtext-to-hugging-face).
  - If the tensor shape are not matched after conversion, error message will print out the parameter name that caused error.

3. After the conversion is done, run a decode to check the correctness of the generated code.
Example command:
```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml model_name=gemma3-4b tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3 \
  load_parameters_path=<Your-converted-ckpt-path> per_device_batch_size=1 run_name=ht_test \
  max_prefill_predict_length=8 max_target_length=16 steps=1 async_checkpointing=false scan_layers=true \
  prompt='I love to' attention='dot_product'
```
If outputs are wrong, you can use jax.debug.print() to print the layer-wise mean/max/min values for debugging.

4. To further validate the converted checkpoint, we recommend to use the [forward_pass_logit_checker.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/checkpoint_conversion/README.md#verifying-conversion-correctness) to compare the original ckpt with the converted ckpt:
```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    tokenizer_path=assets/tokenizers/<tokenizer> \
    load_parameters_path=<path-to-maxtext-checkpoint> \
    model_name=<MODEL_NAME> \
    scan_layers=false \
    use_multimodal=false \
    --run_hf_model=True \
    --hf_model_path=<path-to-HF-checkpoint> \
    --max_kl_div=0.015
```

**Key arguments:**

  * `load_parameters_path`: The path to the source MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
  * `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
  * `scan_layers`: Indicates if the output checkpoint is scanned (scan_layers=true) or unscanned (scan_layers=false).
  * `use_multimodal`: Indicates if multimodality is used.
  * `--run_hf_model`: Indicates if loading Hugging Face model from the hf_model_path. If not set, it will compare the maxtext logits with pre-saved golden logits.
  * `--hf_model_path`: The path to the Hugging Face checkpoint.
  * `--max_kl_div`: Max KL divergence tolerance during comparisons.


## Debugging tips

1. If a response from Gemini is `None`, wait for a moment and retry.


2. If a converted checkpoint loads without errors but produces incorrect output, consider these common issues:

  * **Symptom**: The model generates garbage or nonsensical tokens.

      * **Potential Cause**: The query/key/value (Q/K/V) or Out vectors weights were likely reshaped incorrectly during conversion.

  * **Symptom**: The model generates repetitive text sequences.

      * **Potential Cause**: The layer normalization parameters may have been converted incorrectly.

## Baselines
We have implemented two baselines of the checkpoint conversion agent.

### One-shot prompt
Run the [One-shot agent Jyputer notebook](./baselines/one-shot-agent.ipynb)

### Prompt-chain Agent:
```bash
python3 -m maxtext.experimental.agent.ckpt_conversion_agent.prompt_chain --target_model=<model_name> \
  --dir_path=src/MaxText/experimental/agent/ckpt_conversion_agent --api_key=<Your-API-KEY>
```