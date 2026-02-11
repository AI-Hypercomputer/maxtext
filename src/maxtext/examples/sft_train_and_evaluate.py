#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This script performs SFT training and evaluation workflow on OpenAI's GSM8K dataset.
The primary goal is to demonstrate the end-to-end process of:
1. Pre-SFT Evaluation: Calculating baseline accuracy for the model before training.
2. SFT Training: Fine-tune the model using MaxText & Tunix SFT trainer.
3. Post-SFT Evaluation: Re-running the evaluation loop after training to measure the performance gain achieved by SFT.

## Example command to run on single-host TPU:
```
# Install dependencies in virtual environment:
# https://maxtext.readthedocs.io/en/latest/install_maxtext.html#from-pypi-recommended

# Install post-training dependencies in virtual environment:
bash tools/setup/setup_post_training_requirements.sh

# Environment configurations
export RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)
export OUTPUT_PATH=<GCS Bucket Path for output/logs>
export MODEL_NAME=llama3.1-8b
export TOKENIZER_PATH=meta-llama/Llama-3.1-8B-Instruct
export MODEL_CHECKPOINT_PATH=<GCS path to model checkpoint>
export HF_ACCESS_TOKEN=<Hugging Face access token>

python3 -m maxtext.examples.sft_train_and_evaluate maxtext/configs/post_train/sft.yml \
  run_name=$RUN_NAME base_output_directory=$OUTPUT_PATH \
  model_name=$MODEL_NAME load_parameters_path=$MODEL_CHECKPOINT_PATH \
  hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH
```

## Example command to run on multi-host TPUs using McJAX:
```
# Build & upload docker image
export DOCKER_IMAGE_NAME=${USER}_runner
bash docker_build_dependency_image.sh MODE=post-training && bash docker_upload_runner.sh CLOUD_IMAGE_NAME=$DOCKER_IMAGE_NAME

# Environment configurations
export PROJECT=<Google Cloud Project ID>
export CLUSTER_NAME=<Mame of GKE cluster>
export ZONE=<CGKE cluster zone>
export TPU_TYPE=<TPU type>
export DOCKER_IMAGE="gcr.io/${PROJECT}/${DOCKER_IMAGE_NAME}
export RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)
export OUTPUT_PATH=<GCS Bucket Path for output/logs>
export MODEL_NAME=llama3.1-8b
export TOKENIZER_PATH=meta-llama/Llama-3.1-8B-Instruct
export MODEL_CHECKPOINT_PATH=<GCS path to model checkpoint>
export HF_ACCESS_TOKEN=<Hugging Face access token>

# Run workload via XPK
xpk workload create \
--cluster ${CLUSTER_NAME} \
--docker-image ${DOCKER_IMAGE} \
--workload=sft-${RUN_NAME} \
--tpu-type ${TPU_TYPE} --num-slices=1 --zone=${ZONE} \
--project=${PROJECT} \
--command "HF_TOKEN=$HF_ACCESS_TOKEN python3 -m maxtext.examples.sft_train_and_evaluate maxtext/configs/post_train/sft.yml \
  run_name=$RUN_NAME base_output_directory=$OUTPUT_PATH \
    model_name=$MODEL_NAME load_parameters_path=$MODEL_CHECKPOINT_PATH \
      hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH"
```
"""

from absl import app
from tqdm.auto import tqdm
from typing import Sequence

import datasets
import grain
import os
import re
import transformers

from flax import nnx

from MaxText.globals import MAXTEXT_REPO_ROOT
from MaxText import pyconfig
from MaxText.input_pipeline import instruction_data_processing
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.trainers.post_train.sft import train_sft
from maxtext.utils import max_logging
from maxtext.utils import max_utils

# Suppress vLLM logging with a severity level below ERROR
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout

# Skip JAX precompilation to make vLLM startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

DATASET_NAME = "openai/gsm8k"
DATASET_DATA_DIR = "main"
DATASET_TRAIN_SPLIT = "train"
DATASET_TEST_SPLIT = "test"
DATASET_DATA_COLUMN = ["question", "answer"]
TRAIN_STEPS = 10000
SEED = 42
BATCH_SIZE = 4
NUM_TEST_SAMPLES = 1320
MAX_TOKENS_TO_GENERATE = 768
MAX_PROMPT_LENGTH = 256
EVALUATION_CONFIG = {"temperature": 1e-4, "top_k": 1, "top_p": 1.0}
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
# Regex to check the full format (reasoning + answer markers)
MATCH_FORMAT = re.compile(
    rf"^.*?" rf"{REASONING_START}.+?{REASONING_END}.*?" rf"{ANSWER_START}(.+?){ANSWER_END}" rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
# Regex to extract the final numerical answer
MATCH_ANSWER = re.compile(rf"{ANSWER_START}.*?([\d\.\,\$]{{1,}})", flags=re.MULTILINE | re.DOTALL)
CHAT_TEMPLATE_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "examples", "chat_templates", "math_qa.json")


def get_test_dataset(config, tokenizer):
  """Loads and prepares the test dataset from Hugging Face.

  Args:
    config: The pyconfig object containing run configurations, including
      `hf_access_token`.
    tokenizer: The tokenizer for processing the text data.

  Returns:
    A grain.MapDataset instance for the test split, with prompts and target
    answers.
  """
  template_config = instruction_data_processing.load_template_from_file(config.chat_template_path)
  dataset = datasets.load_dataset(
      DATASET_NAME,
      data_dir=DATASET_DATA_DIR,
      split=DATASET_TEST_SPLIT,
      token=config.hf_access_token,
  )

  return (
      grain.MapDataset.source(dataset)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              "question": x["question"],
              "prompt": tokenizer.apply_chat_template(
                  [
                      {
                          "role": "user",
                          "content": template_config["PROMPT_TEMPLATE"].format(question=x["question"].strip()),
                      }
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              "target_answer": instruction_data_processing.extract_reasoning_and_answer(
                  x["answer"], template_config["REASONING_ANSWER_SEPARATOR"]
              )[1],
          }
      )
  )


def evaluate_model(dataset, vllm_rollout, debug=True):
  """Runs evaluation on the model using vLLM.

  Args:
    dataset: The dataset to evaluate on.
    vllm_rollout: The vLLM rollout object for generating responses.
    debug: If True, prints debug information for each sample.

  Returns:
    A dictionary containing evaluation scores: 'correct', 'partially_correct',
    and 'correct_format' percentages.
  """
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
      max_prompt_length=MAX_PROMPT_LENGTH,
      temperature=EVALUATION_CONFIG["temperature"],
      top_p=EVALUATION_CONFIG["top_p"],
      top_k=EVALUATION_CONFIG["top_k"],
      data_type="bfloat16",
  )

  total, total_correct, total_partially_correct, total_correct_format = 0, 0, 0, 0
  for batch in tqdm(dataset):
    batch_response = vllm_rollout.generate(batch["prompt"], rollout_config)
    for i, question in enumerate(batch["question"]):
      if debug:
        print("========================================")
        print(f"Question: {question}")
        print("----------------------------------------")
        print(f"Model Generated Response: {batch_response.text[i]}")
        print("----------------------------------------")
        print(f"Target Response: {batch["target_answer"][i]}")
        print("========================================")

      is_correct, is_partially_correct, has_correct_format = score_response(
          target=batch["target_answer"][i], prediction=batch_response.text[i], debug=debug
      )
      if is_correct:
        total_correct += 1
      if is_partially_correct:
        total_partially_correct += 1
      if has_correct_format:
        total_correct_format += 1
      total += 1

  return {
      "correct": (total_correct / total) * 100,
      "partially_correct": (total_partially_correct / total) * 100,
      "correct_format": (total_correct_format / total) * 100,
  }


def safe_string_to_float(text):
  """Cleans a string to make it safely convertible to a float.

  Removes commas, spaces, and dollar signs.

  Args:
    text: The input string.

  Returns:
    The cleaned string.
  """
  text = text.replace(",", "").replace(" ", "")  # converts "2,125" to "2125"
  text = text.replace("$", "")  # converts "$50" to "50"
  return text


def score_response(target, prediction, debug=True):
  """Scores the model's prediction against the target answer.

  It checks for exact correctness, partial correctness (within 10%), and
  whether the response follows the expected format.

  Args:
    target: The ground truth answer string.
    prediction: The model's generated response string.
    debug: If True, prints exceptions during scoring.

  Returns:
    A tuple of booleans: (is_correct, is_partially_correct, has_correct_format).
  """
  is_correct, is_partially_correct, has_correct_format = False, False, False
  extracted_response = guess.group(1) if (guess := MATCH_ANSWER.search(prediction)) is not None else ""
  extracted_response = safe_string_to_float(extracted_response)
  target = safe_string_to_float(target)
  try:
    # Check exact correctness
    if float(extracted_response.strip()) == float(target.strip()):
      is_correct = True

    # Check partial correctness (within 10%)
    ratio = float(extracted_response.strip()) / float(target.strip())
    if 0.9 <= ratio <= 1.1:
      is_partially_correct = True

    if MATCH_FORMAT.search(prediction) is not None:
      has_correct_format = True
  except (ValueError, TypeError, ZeroDivisionError) as e:
    if debug:
      print("Evaluation exception: ", e)

  return is_correct, is_partially_correct, has_correct_format


def create_vllm_rollout(config, model, mesh, tokenizer):
  """Creates a vLLM rollout engine for text generation.

  Args:
    config: The pyconfig object containing run configurations.
    model: The NNX model graph.
    mesh: The JAX device mesh.
    tokenizer: The tokenizer.

  Returns:
    A VllmRollout instance configured for the model and hardware.
  """
  tunix_model = TunixMaxTextAdapter(model)
  return VllmRollout(
      model=tunix_model,
      tokenizer=tokenizer,
      cache_config_or_size=MAX_PROMPT_LENGTH + MAX_TOKENS_TO_GENERATE + 256,
      mesh=mesh,
      rollout_config=base_rollout.RolloutConfig(
          rollout_vllm_model_version=config.tokenizer_path,
          rollout_vllm_hbm_utilization=0.2,
          rollout_vllm_init_with_random_weights=True,
          rollout_vllm_tpu_backend_type="jax",
      ),
  )


def get_tokenizer(config):
  """Initializes and returns the tokenizer.

  Args:
    config: The pyconfig object with `tokenizer_path` and `hf_access_token`.

  Returns:
    A Hugging Face tokenizer instance.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      token=config.hf_access_token,
  )
  return tokenizer


def train_and_evaluate(config):
  """Orchestrates the pre-train evaluation, SFT, and post-train evaluation.

  Args:
    config: The pyconfig object containing all run configurations.
  """
  tokenizer = get_tokenizer(config)
  test_dataset = get_test_dataset(config, tokenizer)
  test_dataset = test_dataset[:NUM_TEST_SAMPLES]
  test_dataset = test_dataset.to_iter_dataset().batch(BATCH_SIZE, drop_remainder=True)
  trainer, mesh = train_sft.setup_trainer_state(config)
  vllm_rollout = create_vllm_rollout(config, trainer.model, mesh, tokenizer)

  # 1. Pre-SFT Evaluation
  max_logging.log("Running Pre-SFT evaluation...")
  score = evaluate_model(test_dataset, vllm_rollout)
  print("Score for PRE-SFT EVALUATION: ", score)

  # 2. SFT Training
  max_logging.log("Starting SFT training...")
  trainer = train_sft.train_model(config, trainer, mesh)

  # 3. Post-SFT Evaluation
  max_logging.log("Running Post-SFT evaluation...")
  tunix_model = TunixMaxTextAdapter(trainer.model)
  state = nnx.state(tunix_model)
  vllm_rollout.update_params(state)
  score = evaluate_model(test_dataset, vllm_rollout)
  print("Score for POST-SFT EVALUATION: ", score)


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training and evaluation.

  Args:
    argv: Command-line arguments.
  """

  common_argv_dict = {
      "hf_path": DATASET_NAME,
      "train_split": DATASET_TRAIN_SPLIT,
      "hf_data_dir": DATASET_DATA_DIR,
      "train_data_columns": DATASET_DATA_COLUMN,
      "per_device_batch_size": 1,
      "steps": TRAIN_STEPS,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "learning_rate": 3e-6,
      "chat_template_path": CHAT_TEMPLATE_PATH,
  }
  for arg in argv:
    if arg.startswith("model_name="):
      model_name = arg.split("=")[1]
      if not model_name.lower().startswith("qwen"):
        common_argv_dict["tokenizer_type"] = "huggingface"
  for key, value in common_argv_dict.items():
    argv.append(f"{key}={value}")

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_and_evaluate(config)


if __name__ == "__main__":
  app.run(main)
