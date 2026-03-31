# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a simple script for MMLU benchmark for a trained checkpoint using perplexity.
Dataset: https://huggingface.co/datasets/lighteval/mmlu

To run the MMLU perplexity benchmark:
python3 -m benchmarks.mmlu.mmlu_eval_perplexity src/maxtext/configs/base.yml \
  tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  load_parameters_path=check_point_path model_name=llama3.1-8b \
  max_target_length=2048 per_device_batch_size=1
"""

import collections
import queue
import re
import sys
import threading

from absl import flags

import datasets
import jax
import jax.numpy as jnp
import tqdm
import numpy as np

from benchmarks.mmlu.mmlu_categories import categories, subcategories

from maxtext.configs import pyconfig
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import train_utils
from maxtext.utils import sharding
from maxtext.input_pipeline.tokenizer import build_tokenizer


ASCII_UPPERCASE_A = ord("A")

DEFAULT_PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

{question}
{choices}
Answer:"""


_PROMPT_TEMPLATE = flags.DEFINE_string(
    "prompt_template",
    default=DEFAULT_PROMPT_TEMPLATE,
    help="prompt template",
)


def construct_prompt(subject, question, choices):
  subject = subject.replace("_", " ")
  choices_text = "\n".join(f"{chr(ASCII_UPPERCASE_A + idx)}. {choice}" for idx, choice in enumerate(choices))
  prompt = _PROMPT_TEMPLATE.value.format(subject=subject, question=question, choices=choices_text)
  return prompt


def get_tokenizer(config):
  """Builds and returns the tokenizer based on config."""
  return build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      add_bos=False,
      add_eos=False,
      hf_access_token=config.hf_access_token,
  )


def get_letter_indices(tokenizer):
  """Get token IDs for A, B, C, D."""
  indices = []
  for letter in ["A", "B", "C", "D"]:
    # Encode with standard tokenizer
    tokens = tokenizer.encode(letter)
    indices.append(tokens[-1])
  return indices


def _batch_generator(dataset_iterator, tokenizer, max_target_length, global_batch_size, max_eval_steps):
  """Generates batches of data for evaluation."""
  batch_inputs = []
  batch_segmentation = []
  batch_positions = []
  batch_labels = []
  batch_subjects = []
  batches_processed = 0

  for example in dataset_iterator:
    subject = example["subject"]  # pytype: disable=unsupported-operands
    question = example["question"]  # pytype: disable=unsupported-operands
    choices = example["choices"]  # pytype: disable=unsupported-operands
    label = example["answer"]  # pytype: disable=unsupported-operands
    prompt = construct_prompt(subject, question, choices)

    tokens = tokenizer.encode(prompt)
    if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id is not None and tokenizer.bos_id >= 0:
      tokens = [tokenizer.bos_id] + tokens
    true_length = len(tokens)
    if true_length > max_target_length:
      max_logging.log(f"Warning: Prompt length {true_length} exceeds {max_target_length}. Truncating.")
      tokens = tokens[:max_target_length]
      true_length = max_target_length
      
    pad_id = getattr(tokenizer, 'pad_id', 0)
    if pad_id is None or pad_id < 0:
      pad_id = 0
      
    inputs = tokens + [pad_id] * (max_target_length - len(tokens))
    inputs_segmentation = [1] * true_length + [0] * (max_target_length - true_length)
    inputs_position = list(range(max_target_length))

    batch_inputs.append(inputs)
    batch_segmentation.append(inputs_segmentation)
    batch_positions.append(inputs_position)
    batch_labels.append(label)
    batch_subjects.append(subject)

    if len(batch_inputs) == global_batch_size:
      yield (np.array(batch_inputs, dtype=np.int32), 
             np.array(batch_segmentation, dtype=np.int32), 
             np.array(batch_positions, dtype=np.int32), 
             batch_labels, batch_subjects, global_batch_size)
      batch_inputs, batch_segmentation, batch_positions, batch_labels, batch_subjects = [], [], [], [], []
      batches_processed += 1
      if max_eval_steps > 0 and batches_processed >= max_eval_steps:
        return

  if batch_inputs:
    valid_items = len(batch_inputs)
    pad_len = global_batch_size - valid_items
    if pad_len > 0:
      batch_inputs.extend([[0] * max_target_length for _ in range(pad_len)])
      batch_segmentation.extend([[0] * max_target_length for _ in range(pad_len)])
      batch_positions.extend([[0] * max_target_length for _ in range(pad_len)])
    yield (np.array(batch_inputs, dtype=np.int32), 
           np.array(batch_segmentation, dtype=np.int32), 
           np.array(batch_positions, dtype=np.int32), 
           batch_labels, batch_subjects, valid_items)


def _prefetch_iterator(iterator, buffer_size=4):
  """Allows background thread prefetching of dataset tokenization/batching."""
  q = queue.Queue(maxsize=buffer_size)
  
  def _worker():
    try:
      for item in iterator:
        q.put(("item", item))
      q.put(("done", None))
    except Exception as e:
      q.put(("error", e))
      
  t = threading.Thread(target=_worker, daemon=True)
  t.start()
  
  while True:
    status, item = q.get()
    if status == "done":
      break
    if status == "error":
      raise item
    yield item


def eval_step(model, config, state, data, dropout_rng):
  """Evaluates the model to get the logits of the last token."""
  params = state.params
  from flax import linen as nn

  if isinstance(model, nn.Module):
    logits = model.apply(
        params,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
        rngs={"params": dropout_rng},
        mutable=False,
    )
  else:
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
    )

  # Extract the logit for the last valid token of the prompt.
  batch_indices = jnp.arange(data["inputs"].shape[0])
  prompt_lens = jnp.sum(data["inputs_segmentation"], axis=-1)
  last_indices = jnp.maximum(0, prompt_lens - 1)

  last_logits = logits[batch_indices, last_indices, :]
  return last_logits


def main(config):
  # Set up the model and params
  init_rng, checkpoint_manager, state_mesh_shardings, model, mesh, learning_rate_schedule, data_iterator, data_loader, rampup_manager, eval_data_iterator, state = train_utils.setup_train_loop(
      config, recorder=None
  )

  tokenizer = get_tokenizer(config)
  letter_indices = get_letter_indices(tokenizer)

  max_target_length = config.max_target_length

  # Compile eval_step
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  p_eval_step = train_utils.jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)

  # Initialize counters
  correct_count = 0
  total_count = 0
  subject_correct = collections.defaultdict(int)
  subject_total = collections.defaultdict(int)
  subcat_correct = collections.defaultdict(int)
  subcat_total = collections.defaultdict(int)

  mmlu_test_ds = datasets.load_dataset("lighteval/mmlu", "all", split="test")

  # Global batch size implies the number of test sequences we'll evaluate at once
  # We'll batch until we hit global_batch_size to do the unrolled multi-device eval.
  global_batch_size = max(1, config.global_batch_size_to_load)

  # We need a random key for the dropout_rng argument even if dropout is disabled.
  eval_rng = jax.random.PRNGKey(0)

  dataset_iterator = iter(tqdm.tqdm(mmlu_test_ds, desc="Evaluating MMLU using Perplexity"))
  max_eval_steps = getattr(config, 'steps', -1)

  batch_gen = _batch_generator(dataset_iterator, tokenizer, max_target_length, global_batch_size, max_eval_steps)
  prefetch_iter = _prefetch_iterator(batch_gen)

  future_batches = []
  
  def _process_oldest_batch():
    nonlocal correct_count, total_count
    past_logits, past_valid_items, past_labels, past_subjects = future_batches.pop(0)
    past_logits = np.array(past_logits)
    
    for i in range(past_valid_items):
      logits_ABCD = past_logits[i, letter_indices]
      predicted_idx = int(np.argmax(logits_ABCD))
      correct_idx = past_labels[i]
      subject_i = past_subjects[i]
      
      is_correct = (predicted_idx == correct_idx)
      
      predicted_answer = chr(ASCII_UPPERCASE_A + predicted_idx)
      correct_answer = chr(ASCII_UPPERCASE_A + correct_idx)
      
      if total_count % 100 == 0:
        max_logging.log(f"{total_count} | predicted: {predicted_answer}, true: {correct_answer}, correct: {is_correct}")

      if is_correct:
        correct_count += 1
        subject_correct[subject_i] += 1
        
      total_count += 1
      subject_total[subject_i] += 1

  # Overlap TPU dispatch with host-side dataset processing
  for batch_inputs, batch_segmentation, batch_positions, batch_labels, batch_subjects, valid_items in prefetch_iter:
    data = {
        "inputs": jnp.array(batch_inputs),
        "inputs_position": jnp.array(batch_positions),
        "inputs_segmentation": jnp.array(batch_segmentation),
    }

    with mesh:
      last_logits = p_eval_step(state, data, eval_rng)
      
    future_batches.append((last_logits, valid_items, batch_labels, batch_subjects))
    
    # Process older batches to overlap CPU interpretation with TPU execution
    # Keeping 2 batches in flight ensures the TPU always stays busy!
    if len(future_batches) >= 2:
      _process_oldest_batch()
      
  # Drain the remainder
  while future_batches:
    _process_oldest_batch()

  if total_count > 0:
    accuracy = correct_count / total_count
    max_logging.log(f"\nFinal accuracy on MMLU dataset matching probabilities for A/B/C/D: {accuracy:.4f}")
  else:
    max_logging.log("No valid predictions made.")

  # Map subject accuracies to subcategories
  for subject_ in subject_total:
    if subject_ in subcategories:
      subcat_labels = subcategories[subject_]
      for subcat_label in subcat_labels:
        subcat_correct[subcat_label] += subject_correct[subject_]
        subcat_total[subcat_label] += subject_total[subject_]

  # Subcategory accuracies
  max_logging.log("\nSubcategory Accuracies:")
  for subcat_label in subcat_total:
    if subcat_total[subcat_label] > 0:
      acc = subcat_correct[subcat_label] / subcat_total[subcat_label]
      max_logging.log(f"Accuracy for subcategory '{subcat_label}': {acc:.4f}")

  # Category accuracies
  cat_correct = collections.defaultdict(int)
  cat_total = collections.defaultdict(int)

  for category_name, subcat_labels in categories.items():
    for subcat_label in subcat_labels:
      cat_correct[category_name] += subcat_correct[subcat_label]
      cat_total[category_name] += subcat_total[subcat_label]

  max_logging.log("\nCategory Accuracies:")
  for category_name in cat_total:
    if cat_total[category_name] > 0:
      acc = cat_correct[category_name] / cat_total[category_name]
      max_logging.log(f"Accuracy for category '{category_name}': {acc:.4f}")
    else:
      max_logging.log(f"Accuracy for category '{category_name}': No data available.")


def validate_config(config):
  assert not config.load_full_state_path, (
      "Decode doesn't operate on full states! Convert to parameter checkpoint"
      " first. Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  flags.FLAGS(sys.argv)
  cfg = pyconfig.initialize(sys.argv)
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg)