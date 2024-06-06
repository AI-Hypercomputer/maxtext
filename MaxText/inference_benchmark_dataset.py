# '''
# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example run command
'''
python MaxText/inference-benchmark-dataset.py  MaxText/configs/base.yml async_checkpointing=false \
  attention=autoselected model_name=llama2-7b weight_dtype=bfloat16 tokenizer_path=assets/tokenizer.llama2 \
  scan_layers=false run_name=test_mbdataset_path=gs://maxtext-dataset ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=-1 async_checkpointing=false max_prefill_predict_length=1024 \
  base_output_directory=gs://${USER}-maxtext-outputs  max_target_length=2048 per_device_batch_size=4 \
  load_parameters_path=gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items \
  inference_benchmark_dataset_file_path=/home/${USER}/datasets/openorca_json/output_100.json \
  quantization="" quantize_kvcache=False  \
  profiler=xplane run_name="test_mb"
'''

'''Inference microbenchmark for prefill and autoregressive steps.'''
import datetime
import evaluate
import jax
import jax.numpy as jnp
import json
import logging
import numpy as np
import nltk
import os
import statistics
import sys

import max_utils
import maxengine
import maxtext_utils
import profiler
import pyconfig

from jetstream.engine import token_utils

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

_PREFILL_LENGTH = 1024 # For padding input tokens

# Benchmark type
_WARMUP = 'warmup'
_PROFILE = 'profile'
_BENCHMARK = 'benchmark'
_WARMUP_BATCHES = 2 # Number of batches in dataset used for warmup
_PROFILE_BATCHES = 0 # Number of batches to profile

# Stages
_PREFILL = 'prefill'
_PREFILL_INSERT = 'prefill_insert'
_AUTOREGRESSIVE = 'autoregressive'
_FULL = 'full'

# Stats
_GLOBAL_BATCH_SIZE = 'global_batch_size'
_MSEC_PROFILED = "msec_profiled"
_MSEC_PER_SEQ = 'msec_per_seq'
_MSEC_PER_TOK = 'msec_per_token'
_TOKENS_PER_SEC = 'tokens_per_sec'

def get_array_stats(name, stats_array, n=10):
  if len(stats_array) < n:
    return f'\n{name} -  min: {min(stats_array)}, max: {max(stats_array)}, \
    \n\t\tvalues:{stats_array}'
  return f'\n{name} -  min: {min(stats_array)}, max: {max(stats_array)}, \
    \n\t\tdeciles:{[round(s, 2) for s in statistics.quantiles(stats_array, n=n)]}'


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]
  # rougeLSum expects newline after each sentence
  preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ['\n'.join(nltk.sent_tokenize(target)) for target in targets]
  return preds, targets


def eval_accuracy(predicted, target):
  assert len(predicted) == len(target), 'Mismatched predicted and target outputs.'
  metric = evaluate.load('rouge')
  nltk.download('punkt')
  preds = []
  targets = []
  preds, targets = postprocess_text(predicted, target)
  result = metric.compute(
      predictions=preds,
      references=targets,
      use_stemmer=True,
      use_aggregator=False,
  )
  result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  target_lens = [len(tgt) for tgt in targets]
  result['num_requests'] = len(preds)
  result['sum_len_preds'] = int(np.sum(prediction_lens))
  result['sum_len_targets'] = int(np.sum(target_lens))
  result['num_requests'] = len(preds)
  return result


def read_openorca_dataset(filepath, print_stats=True):
  assert os.path.exists(filepath), f'input dataset file: {filepath}  does not exist'
  # Read file
  with open(filepath) as f:
    dataset = json.load(f)
  return dataset


def benchmark_prefill(config, engine, params, tokens, true_lengths,
                      start_row, end_row, profile):
  '''Benchmarking prefill step.'''
  stage = _PREFILL
  batch_size = end_row - start_row
  # Benchmark
  logging.info('Benchmark prefill')
  if profile:
    profile_name = f'{stage}[{start_row}:{end_row}]'
    prof = profiler.Profiler(config, profile_name)
    prof.activate()
  start = datetime.datetime.now()
  for i in range(batch_size):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens[i], true_length=true_lengths[i])
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  if profile:
    prof.deactivate()
  max_utils.delete_pytree(prefill_result)
  # Stats
  time_seconds = (end - start).total_seconds()
  stats = {
    _MSEC_PROFILED: round(time_seconds*1000),
    _MSEC_PER_SEQ: round(time_seconds*1000/(batch_size), 2)
  }
  return stats

def benchmark_prefill_insert(config, engine, decode_state, params, tokens,
                             true_lengths, start_row, end_row, profile):
  '''Benchmarking prefill and insert step.'''
  stage = _PREFILL_INSERT
  batch_size = end_row - start_row
  total_slots = engine.max_concurrent_decodes
  # Benchmark
  logging.info(f'Benchmark prefill insert batch_size {batch_size}')
  if profile:
    profile_name = f'{stage}[{start_row}:{end_row}]'
    prof = profiler.Profiler(config, profile_name)
    prof.activate()
  start = datetime.datetime.now()
  decode_states = [decode_state]
  for i in range(batch_size):
    slot = int(i % total_slots)
    prefill_result, first_token = engine.prefill(
      params=params, padded_tokens=tokens[i], true_length=true_lengths[i]
    )
    decode_states.append(
      engine.insert(prefill_result, decode_states.pop(0), slot)
    )
    del prefill_result
  jax.block_until_ready(decode_states)
  end = datetime.datetime.now()
  if profile:
    prof.deactivate()
  # Stats
  time_seconds = (end - start).total_seconds()
  stats = {
    _MSEC_PROFILED: round(time_seconds*1000),
    _MSEC_PER_SEQ: round(time_seconds*1000/(batch_size), 2)
  }
  return stats, decode_states.pop(0)


def benchmark_autoregressive(config, engine, decode_state, params,
                             start_row, end_row, profile):
  '''Benchmarking autoregressive step.'''
  stage = _AUTOREGRESSIVE
  batch_size = end_row - start_row
  total_slots = engine.max_concurrent_decodes
  # Benchmark
  logging.info(f'Benchmark autoregressive batchsize {batch_size}')
  if profile:
    profile_name = f'{stage}[{start_row}:{end_row}]'
    prof = profiler.Profiler(config, profile_name)
    prof.activate()
  start = datetime.datetime.now()
  sampled_tokens_list = []
  steps = range(config.max_prefill_predict_length, config.max_target_length)
  for _ in steps:
    # Note we do not do an early stop here on eos - measuring worst case performance.
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(sampled_tokens)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  if profile:
    prof.deactivate()
  max_utils.delete_pytree(decode_state)
  # Stats
  time_seconds = (end - start).total_seconds()
  stats = {
    _GLOBAL_BATCH_SIZE: batch_size,
    _MSEC_PROFILED: round(time_seconds*1000),
    _MSEC_PER_SEQ: round(time_seconds*1000/(batch_size), 2),
    _MSEC_PER_TOK: round(time_seconds*1000/(len(steps)*batch_size), 2),
    _TOKENS_PER_SEC: round(batch_size*len(steps)/time_seconds, 2)
  }
  return stats, sampled_tokens_list


def benchmark(config, engine, decode_state, params, tokens, true_lengths,
              start_row, end_row, profile):
    batch_size = end_row - start_row
    batch_stats = {}
    batch_tokens = []
    batch_stats[_PREFILL] = benchmark_prefill(
      config, engine, params, tokens, true_lengths, start_row, end_row,
      profile)
    batch_stats[_PREFILL_INSERT], decode_state = benchmark_prefill_insert(
      config, engine, decode_state, params, tokens, true_lengths, start_row,
      end_row, profile)
    batch_stats[_AUTOREGRESSIVE], tokens_list = benchmark_autoregressive(
      config, engine, decode_state, params, start_row, end_row, profile)
    for slot in range(batch_size):
      batch_tokens.append(
        [tokens.get_result_at_slot(slot).tokens.item() for tokens in tokens_list]
      )
    return batch_stats, batch_tokens


def aggregate_stats(stats_list):
  output_stats = {}
  for stats in stats_list:
    for k_stage in stats.keys():
      for k_metric in stats[k_stage].keys():
        k_out = f'{k_stage}-{k_metric}'
        if not k_out in output_stats:
          output_stats[k_out] = []
        output_stats[k_out].append(stats[k_stage][k_metric])
  result = {}
  for k_out in output_stats.keys():
    if len(output_stats[k_out]) > 1:
      result[k_out] = get_array_stats(k_out, output_stats[k_out])
    else:
     result[k_out] = output_stats[k_out]
  return result


def compute_rouge_score(tokenizer, dataset, output_tokens_orig):
  assert len(dataset) == len(output_tokens_orig)
  n = len(dataset)
  prompt_text = [dataset[j]['prompt'] for j in range(n)]
  target_text = [dataset[j]['output'] for j in range(n)]
  len_target_tokens = [dataset[j]['len_output_tokens'] for j in range(n)]
  output_tokens = []
  output_text = []
  output_text_orig = []
  for j in range(len(dataset)):
    tokens = output_tokens_orig[j]
    text_orig = tokenizer.decode(output_tokens_orig[j])
    output_text_orig.append(text_orig)
    outlen = min(len_target_tokens[j], len(tokens))
    # print(f"outlen = {outlen} len_tokens :{len(tokens)} target {len_target_tokens[j]}")
    tokens_new = tokens[0:outlen]
    output_tokens.append(tokens_new)
    text = tokenizer.decode(output_tokens[j])
    output_text.append(text)


  rouge_scores = {
    'rouge_score': eval_accuracy(output_text, target_text),
    'rouge_score_orig': eval_accuracy(output_text_orig, target_text)
  }


  len_target_str = [len(s) for s in target_text]
  len_output_tokens = [len(token) for token in output_tokens]
  len_output_str = [len(s) for s in output_text]
  len_output_str_orig = [len(s) for s in output_text_orig]
  stats = {
    'num_requests': len(dataset),
    'len_target_str': get_array_stats('len_target_str',len_target_str),
    'len_output_str': get_array_stats('len_output_str', len_output_str),
    'len_output_str_original': get_array_stats('len_output_str_orig', len_output_str_orig),
    'len_target_tokens': get_array_stats('len_target_tokens', len_target_tokens),
    'len_output_tokens': get_array_stats('len_output_tokens', len_output_tokens),
  }
  return rouge_scores, stats


def benchmark_dataset(dataset, batch_size, config, engine, params, vocab, run_type):
  stats = []
  output_tokens = []
  profile = (run_type == _PROFILE)
  dataset_size =  len(dataset)
  for start_row in range(0, dataset_size, batch_size):
    end_row = min(start_row + batch_size, dataset_size)
    #logging.info(f"start_row: {start_row}, end_row: {end_row}, dataset_size: {dataset_size}")
    decode_state = engine.init_decode_state()
    prefill_tokens = {}
    prefill_true_lengths = {}
    for j in range(start_row, end_row):
      #logging.info(f" trying row - {j}")
      text = dataset[j]['prompt']
      if j == start_row:
        logging.info(f'\n{run_type} batchsize {batch_size} rows[{start_row}:{end_row}]')
      idx = j - start_row
      prefill_tokens[idx], prefill_true_lengths[idx] = token_utils.tokenize_and_pad(
        text, vocab, is_bos=True, prefill_lengths=[_PREFILL_LENGTH])
    batch_stats, batch_tokens = benchmark(
      config, engine, decode_state, params, prefill_tokens, prefill_true_lengths,
      start_row, end_row, profile)
    if run_type == _PROFILE or run_type == _WARMUP:
      return None, None
    # Print row level stats for debugging
    for k in batch_stats.keys():
      logging.info(f'\n{k}: {batch_stats[k]}')
    stats.append(batch_stats)
    output_tokens.extend(batch_tokens)
  return stats, output_tokens

def get_tokens_multipack_one_prefill(dataset, vocab, max_seq_length, start_idx):
  prefill_token_indices = []
  prefill_token_sizes = []
  prefill_data_indices = []
  data_idx = start_idx
  final_tokens, final_true_length = jax.numpy.zeros(max_seq_length), 0
  while True:
    print(f"Processing data_idx {data_idx}")
    cur_tokens, cur_true_length = token_utils.tokenize_and_pad(
      dataset[data_idx]['prompt'], vocab, is_bos=True, prefill_lengths=[max_seq_length]
    )
    if final_true_length + cur_true_length >= max_seq_length:
      break
    if not final_true_length:
      final_tokens = cur_tokens
    else:
      final_tokens = jax.numpy.concatenate(
        [final_tokens[:final_true_length], cur_tokens[:cur_true_length]])
      final_tokens = jnp.resize(final_tokens, max_seq_length)
    prefill_token_indices.append(final_true_length)
    prefill_token_sizes.append(cur_true_length)
    prefill_data_indices.append(data_idx)
    final_true_length += cur_true_length
    data_idx += 1

  print(f"Prefill token ready with {prefill_token_sizes} prompts packed")
  return final_tokens, final_true_length, data_idx, prefill_token_sizes, prefill_data_indices

def get_tokens_multipack(dataset, vocab, batch_size, max_seq_length):
  prefill_true_lengths = {}
  prefill_tokens = {}
  prefill_token_sizes = {}
  prefill_data_indices = {}
  data_idx = 0
  for i in range(batch_size):
    prefill_tokens[i], prefill_true_lengths[i], data_idx, prefill_token_sizes[i], prefill_data_indices[i] = get_tokens_multipack_one_prefill(dataset, vocab, max_seq_length, data_idx)

  return prefill_tokens, prefill_true_lengths, prefill_token_sizes, prefill_data_indices

def prefill_and_insert_multipack(config, engine, params, vocab, dataset, decode_state):
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  total_slots = engine.max_concurrent_decodes
  batch_size = total_slots
  max_seq_length = _PREFILL_LENGTH

  # Single Prefill
  data_idx = 0
  prefill_tokens, prefill_true_lengths, data_idx, prefill_token_sizes, prefill_data_indices = get_tokens_multipack_one_prefill(dataset, vocab, max_seq_length, data_idx)
  prefill_result_and_first_tokens = engine.prefill(
      params=params,
      padded_tokens=prefill_tokens,
      true_length=prefill_true_lengths,
      multiprompt_sizes=prefill_token_sizes,
  )

  logging.info(f"After single prefill with {data_idx} prompts")
  # Insert now
  results = {}
  completes = {}
  # prefill_result_and_first_tokens = [prefill_result_and_first_tokens]
  for i in range(len(prefill_result_and_first_tokens)):
    prefill_result, first_token = prefill_result_and_first_tokens[i]
    slot = int(i % total_slots)
    logging.info(f"Inserting dataset {i} in slot {slot}")
    completes[slot] = np.zeros((1,), np.bool_)
    results[slot] = []
    result, completes[slot] = token_utils.process_result_tokens(
        tokenizer_model,
        slot=0, # always 0 as prefill only run 1 sample
        slot_max_length=config.max_target_length,
        result_tokens=first_token.convert_to_numpy(),
        complete=completes[slot],
        is_client_side_tokenization=True,
        debug=False)
    results[slot].extend(result[0].token_ids)
    decode_state = engine.insert(prefill_result, decode_state, slot)
  jax.block_until_ready(decode_state)
  #decode_state = decode_states.pop(0)
  logging.info(f"After prefill and insert of {prefill_true_lengths} prompt lengths")
  return decode_state, results, completes

def prefill_and_insert_single(config, engine, params, vocab, dataset, decode_state):
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  total_slots = engine.max_concurrent_decodes
  batch_size = total_slots 
  prefill_tokens = {}
  prefill_true_lengths = {}
  results = {}
  completes = {}
  for i in range(batch_size):
    text = dataset[i]['prompt']
    prefill_tokens[i], prefill_true_lengths[i] = token_utils.tokenize_and_pad(
        text, vocab, is_bos=True, prefill_lengths=[_PREFILL_LENGTH])

    prefill_result, first_token = engine.prefill(
      params=params, padded_tokens=prefill_tokens[i], true_length=prefill_true_lengths[i]
    )
    slot = int(i % total_slots)
    logging.info(f"Inserting dataset {i} in slot {slot}")
    completes[slot] = np.zeros((1,), np.bool_)
    results[slot] = []
    result, completes[slot] = token_utils.process_result_tokens(
        tokenizer_model,
        slot=0, # always 0 as prefill only run 1 sample
        slot_max_length=config.max_target_length,
        result_tokens=first_token.convert_to_numpy(),
        complete=completes[slot],
        is_client_side_tokenization=True,
        debug=False)
    results[slot].extend(result[0].token_ids)
    decode_state = engine.insert(prefill_result, decode_state, slot)
  jax.block_until_ready(decode_state)
  #decode_state = decode_states.pop(0)
  logging.info(f"After prefill and insert of {prefill_true_lengths} prompt lengths")
  return decode_state, results, completes

def generate_output(config, engine, params, vocab, dataset):
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  decode_state = engine.init_decode_state()
  total_slots = engine.max_concurrent_decodes
  batch_size = total_slots 
  #decode_state, results, completes = prefill_and_insert_single(
  decode_state, results, completes = prefill_and_insert_multipack(
    config, engine, params, vocab, dataset, decode_state
  )

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(sampled_tokens)
  jax.block_until_ready(decode_state)

  logging.info(f"Printing generated texts for {batch_size} prompts")
  results_text = {}
  for slot in range(batch_size):
    text = dataset[slot]['prompt']
    for sampled_tokens in sampled_tokens_list:
      result, completes[slot] = token_utils.process_result_tokens(
          tokenizer_model,
          slot=slot,
          slot_max_length=config.max_target_length,
          result_tokens=sampled_tokens.convert_to_numpy(),
          complete=completes[slot],
          is_client_side_tokenization=True,
          debug=False)
      if completes[slot].all():
        break
      results[slot].extend(result[0].token_ids)
    results_text[slot] = tokenizer_model.decode(results[slot])
    expected_output = dataset[slot]['output']
    logging.info(f"Slot: {slot}, ############################## \n  Output `{results_text[slot]}` \n")
    # logging.info(f"Expected:`{expected_output}`")
  logging.info('#################End################')


def main(config):
  # Load model and setup initial state
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()
  batch_size = engine.max_concurrent_decodes
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  # Read the dataset
  #assert config.inference_benchmark_dataset_file_path, 'Dataset filepath not defined'
  #dataset = read_openorca_dataset(config.inference_benchmark_dataset_file_path)
  dataset = read_openorca_dataset("/home/vipannalla/local/openorca_output_100.json")
  
  # Generate output
  generate_output(config, engine, params, vocab, dataset)
  return 

  # Warmup
  dataset_size = min(len(dataset), batch_size*_WARMUP_BATCHES)
  _, _ = benchmark_dataset(dataset[0:dataset_size], batch_size, config, engine,
                           params, vocab, _WARMUP)
  # Profile
  dataset_size = min(len(dataset), batch_size*_PROFILE_BATCHES)
  _, _ = benchmark_dataset(dataset[0:dataset_size], batch_size, config, engine,
                           params, vocab, _PROFILE)
  # Benchmark
  stats, output_tokens = benchmark_dataset(dataset, batch_size, config, engine,
                                           params, vocab, _BENCHMARK)

  # Aggregate stats
  logging.info('\nAggregating stats')
  result = aggregate_stats(stats)

  # Output stats
  decode_state = engine.init_decode_state()
  _, cache_size, _ = max_utils.summarize_pytree_data(decode_state['cache'], name='Cache')
  num_params, model_size, _ = max_utils.summarize_pytree_data(params, name='Model')

  print('\nPerformance')
  for k in result.keys():
    print('\t' + f'{k}: {result[k]}')

  # Output overall rogue score
  print('\nAccuracy')
  scores, data_stats = compute_rouge_score(vocab.tokenizer, dataset, output_tokens)
  for k in scores.keys():
    print('\t' + f'{k}: {scores[k]}')

  print('\nData stats:')
  for k in data_stats.keys():
    print('\t' + f'{k}: {data_stats[k]}')

  logging.info('\nEnd')

if __name__ == '__main__':
  pyconfig.initialize(sys.argv)
  main(pyconfig.config)

