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
SFT training and evaluation script.

pip install rouge_score

Example command:
RUN_NAME=$(date +%Y-%m-%d-%H-%M-%S)
python3 -m MaxText.examples.sft_train_evaluate_demo MaxText/configs/sft.yml \
  run_name=$RUN_NAME base_output_directory=gs://sjsurbhi-multipod/maxtext-tunix-sft/llama3.1-8b \
  model_name=llama3.1-8b load_parameters_path=gs://maxtext-model-checkpoints/llama3.1_8b_instruct/2025-10-16/scanned/0/items \
  hf_access_token=$HF_TOKEN tokenizer_path=meta-llama/Llama-3.1-8B-Instruct 

Results:
### Llama3.1-8B
Running Pre-SFT evaluation...
Rouge score:  {'rouge1': np.float64(51.1975), 'rouge2': np.float64(33.2061), 'rougeL': np.float64(47.0017), 'rougeLsum': np.float64(46.95)}
Starting SFT training...

Running Post-SFT evaluation...
Rouge score:  {'rouge1': np.float64(30.4736), 'rouge2': np.float64(18.9841), 'rougeL': np.float64(27.7844), 'rougeLsum': np.float64(28.4555)}
"""

from absl import app
from tqdm.auto import tqdm
from typing import Sequence

import datasets
import evaluate
import grain
import nltk
import numpy as np
import os
import transformers

from MaxText import max_logging
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.input_pipeline import translation_data_processing
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from MaxText.sft import sft_trainer

# Suppress vLLM logging with a severity level below ERROR
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout

DATASET_NAME = "Helsinki-NLP/opus-100"
DATASET_DATA_DIR = "en-fr"
DATASET_TRAIN_SPLIT = "train"
DATASET_TEST_SPLIT = "test"
DATASET_DATA_COLUMN = "translation"
SEED = 42
BATCH_SIZE = 1
NUM_TEST_SAMPLES = 5
MAX_TOKENS_TO_GENERATE = 256
MAX_PROMPT_LENGTH = 128
DECODE_STRATEGY = "greedy"
DEBUG = True
EVALUATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 0.01, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# for vLLM we can skip JAX precompilation to make startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

def get_test_dataset(config, tokenizer):
    dataset = datasets.load_dataset(
      DATASET_NAME,
      data_dir=DATASET_DATA_DIR,
      split=DATASET_TEST_SPLIT,
      token=config.hf_access_token,
    )

    return (
      grain.MapDataset.source(dataset)
      .shuffle(seed=SEED)
      .map(lambda x: translation_data_processing.map_to_conversation(x, DATASET_DATA_COLUMN))
      .map(lambda x: {
         "prompts": x["messages"][0]["content"],
         "processed_prompts": tokenizer.apply_chat_template(
              [x["messages"][0]],
              tokenize=False,
              add_generation_prompt=True,
              add_special_tokens=False,
          ),
         "target_responses": x["messages"][1]["content"]
      })
      .batch(BATCH_SIZE)
    )


def evaluate_model(dataset, config, model, mesh, tokenizer):
    """Runs evaluation on the model using vLLM."""
    metric = evaluate.load("rouge")
    rollout_config, vllm_rollout = create_vllm_rollout(config, model, mesh, tokenizer)
    for batch in tqdm(dataset):
      batch_response = vllm_rollout.generate(batch["processed_prompts"], rollout_config)

      if DEBUG:
        for i, prompt in enumerate(batch["prompts"]):
            print("========================================")
            print(f"Prompt: {prompt}")
            print("----------------------------------------")
            print(f"Model Generated Response: {batch_response.text[i]}")
            print("----------------------------------------")
            print(f"Target Response: {batch["target_responses"][i]}")
            print("========================================")

      metric.add_batch(predictions=batch_response.text, references=batch["target_responses"])

    score = metric.compute()
    score = {k: round(np.mean(v) * 100, 4) for k, v in score.items()}
    print("Rouge score: ", score)


def create_vllm_rollout(config, model, mesh, tokenizer):
  tunix_model = TunixMaxTextAdapter(base_model=model)
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
      max_prompt_length=MAX_PROMPT_LENGTH,
      temperature=EVALUATION_CONFIGS[DECODE_STRATEGY]["temperature"],
      top_p=EVALUATION_CONFIGS[DECODE_STRATEGY]["top_p"],
      top_k=EVALUATION_CONFIGS[DECODE_STRATEGY]["top_k"],
  )
  vllm_rollout = VllmRollout(
      model=tunix_model,
      tokenizer=tokenizer,
      cache_config_or_size=MAX_PROMPT_LENGTH + MAX_TOKENS_TO_GENERATE,
      mesh=mesh,
      model_version=config.tokenizer_path,
      hbm_utilization=0.5,
      init_with_random_weights=True,
      tpu_backend_type="jax",
  )
  return rollout_config, vllm_rollout


def get_tokenizer(config):
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.tokenizer_path,
    token=config.hf_access_token,
  )
  tokenizer.bos_token = None
  return tokenizer 


def show_hbm_usage():
  import functools
  import humanize
  import jax
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def train_and_evaluate(config):
  nltk.download("punkt")

  tokenizer = get_tokenizer(config)
  test_dataset = get_test_dataset(config, tokenizer)[:NUM_TEST_SAMPLES]
  trainer, mesh = sft_trainer.setup_trainer_state(config)

  print("HBM usage before evaluating:")
  show_hbm_usage()

  max_logging.log(f"Running Pre-SFT evaluation...")
  evaluate_model(test_dataset, config, trainer.model, mesh, tokenizer)

  print("HBM usage after evaluating, and before training:")
  show_hbm_usage()

  max_logging.log(f"Starting SFT training...")
  sft_trainer.start_training(config, trainer, mesh)

  print("HBM usage after training, and before evaluating:")
  show_hbm_usage()

  max_logging.log(f"Running Post-SFT evaluation...")
  evaluate_model(test_dataset, config, trainer.model, mesh, tokenizer)


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training and evaluation.

  Args:
    argv: Command-line arguments.
  """

  common_argv_dict = {
    "tokenizer_type": "huggingface",
    "hf_path": DATASET_NAME,
    "train_split": DATASET_TRAIN_SPLIT,
    "hf_data_dir": DATASET_DATA_DIR,
    "train_data_columns": [DATASET_DATA_COLUMN],
    "per_device_batch_size": 1,
    "steps": 100,
    "dtype": "float32",
  }

  for key, value in common_argv_dict.items():
    argv.append(f"{key}={value}")
  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  train_and_evaluate(mt_config)


if __name__ == "__main__":
  app.run(main)
