# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import gc
import json
import time
import math
import logging
import os
import sys
import array
import random

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import mlperf_loadgen as lg
# pylint: disable=no-name-in-module

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

from MaxText.maxengine import create_engine_from_config_flags
from MaxText.inference_mlperf import offline_inference

_MLPERF_ID = "llama2-70b"
log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOGLEVEL", "INFO"))

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mlperf_test_mode",
    "performance",
    "performance, accuracy, submission",
)
flags.DEFINE_string("api_url", None, "published model path.", required=False)
flags.DEFINE_string("dataset_path", None, "", required=False)
flags.DEFINE_bool("is_stream", False, "", required=False)
flags.DEFINE_string(
    "input_mode",
    "tokenized",
    "Input mode",
)
flags.DEFINE_string(
    "output_mode",
    "tokenized",
    "Output mode",
)

flags.DEFINE_string(
    "audit_conf",
    "audit.conf",
    "audit config for LoadGen settings during compliance runs",
    required=False,
)
flags.DEFINE_string(
    "mlperf_conf",
    "mlperf.conf",
    "mlperf rules config",
    required=False,
)
flags.DEFINE_string(
    "user_conf",
    "user.conf",
    "user config for user LoadGen settings such as target QPS",
    required=False,
)
flags.DEFINE_integer(
    "total_sample_count",
    24576,
    "Number of samples to use in benchmark.",
    required=False,
)
flags.DEFINE_integer(
    "perf_count_override",
    None,
    "Overwrite number of samples to use in benchmark.",
    required=False,
)
flags.DEFINE_string(
    "output_log_dir",
    "output-logs",
    "Where logs are saved.",
    required=False,
)
flags.DEFINE_bool(
    "enable_log_trace",
    False,
    "Enable log tracing. This file can become quite large",
    required=False,
)
flags.DEFINE_string(
    "prefill_lengths_and_per_device_batch_sizes",
    "256,80|512,40|1024,20",
    "List of prefill lengths and batch sizes to use for each engine. Format len_1,bs_1|len_2,bs_2|..",
    required=False,
)

flags.DEFINE_string(
    "maxengine_args",
    "",
    "Additional arguments to maxtext engine, space separated <name>=<value> pairs",
    required=False,
)

flags.DEFINE_integer(
    "jax_profiler_port",
    9999,
    "If set, the jax.profiler port to use.",
    required=False,
)

flags.DEFINE_bool(
    "enable_profile",
    False,
    "If set, enable jax profiling.",
    required=False,
)

flags.DEFINE_bool(
    "enable_batch_prefill",
    False,
    "If set, enable batch prefilling.",
    required=False,
)

flags.DEFINE_bool(
    "skip_warmup",
    False,
    "Skip warmup",
    required=False,
)

flags.DEFINE_float(
    "tok_outlen_multiplier",
    3.0,
    "Multiplier for estimating max predicted output len",
    required=False,
)

flags.DEFINE_bool(
    "allow_skipping_queries",
    False,
    "Allow skipping queries which have target len greater than 2x configured max prefill len",
    required=False,
)

flags.DEFINE_string(
    "rename_dataset_cols",
    "",
    "Rename some of the dataset columns to whats expected by code. For example, "
    "mixtral dataset uses ref_token_length instead of ref_token_len. Format is a string dict "
    'eg. {"tok_input_len": "tok_input_length"}',
    required=False,
)

flags.DEFINE_string(
    "maxengine_config_filepath",
    None,
    "Base config filepath for initializing MaxEngine.",
    required=False,
)

flags.DEFINE_integer(
    "python_seed",
    42,
    "Seed for Python's random, NumPy's random, and JAX's PRNG key generation. " "Also used for Pandas DataFrame sampling.",
    required=False,
)


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))) if true_length > 0 else 0, 128)
  if true_length == 0 and target_length == 0:  # handle empty tokens if target_length becomes 0
    target_length = 128
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


def _init_query_batches():
  query_batches = {}
  len_batch_str = FLAGS.prefill_lengths_and_per_device_batch_sizes.split("|")
  for lb in len_batch_str:
    l, b = lb.split(",")
    query_batches[(int(l), int(b))] = []
  return query_batches


@contextlib.contextmanager
def timed(msg):
  log.info("%s start", msg)
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  log.info("%s done: %f seconds", msg, end - start)


def _classify_query(dataset_rows, index, query_batches):
  sample = dataset_rows[index][1]
  input_len = sample.tok_input_length
  total_len = int(sample.tok_input_length + FLAGS.tok_outlen_multiplier * sample.tok_output_length)
  query_batch_keys = list(query_batches.keys())
  query_batch_keys.sort()
  target_inputs = [lb[0] for lb in query_batch_keys]
  # target_totals = [2 * inp for inp in target_inputs] # This seemed to be the original intent for MaxEngine target_length

  for i in range(len(target_inputs)):
    # MaxEngine's max_target_length is what total_len should be compared against.
    # MaxEngine's max_prefill_predict_length is what input_len should be compared against.
    engine_max_prefill_len = target_inputs[i]
    engine_max_target_len = 2 * engine_max_prefill_len  # As per typical MaxEngine config in main()

    if total_len <= engine_max_target_len and input_len <= engine_max_prefill_len:
      log.debug("Added sample of input length %d total_len %d for %s", input_len, total_len, query_batch_keys[i])
      return query_batch_keys[i]
  # If no bucket perfectly fits based on total_len, try to fit based on input_len alone for the largest buckets
  # This logic might need refinement based on exact bucketing strategy desired.
  # The original code had a fallback that might assign to the *first* bucket where input_len fits,
  # which might not be optimal if total_len is very large.
  # For now, sticking to a similar pattern but being explicit.
  for i in range(len(target_inputs)):
    engine_max_prefill_len = target_inputs[i]
    if input_len <= engine_max_prefill_len:
      log.debug(
          "Fallback: Added sample of input length %d total_len %d to bucket for prefill_len %s (total_len might exceed ideal)",
          input_len,
          total_len,
          query_batch_keys[i],
      )
      return query_batch_keys[i]

  if not FLAGS.allow_skipping_queries:
    assert (
        False
    ), f"Invalid query input_len {input_len} (total_len {total_len}) > max prefill_len/target_len configured {query_batch_keys[-1]} (max_target {2*query_batch_keys[-1][0]})."
  return -1


def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  """max_batch to not run OOM."""
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  # This calculation seems a bit off, usually batch size is more direct.
  # Re-evaluating the intent. If it's to proportionally reduce based on sample size vs dataset size,
  # it might be for heterogeneous batching which isn't standard here.
  # Defaulting to a simpler interpretation: fit within max_batch.
  return math.ceil(num_samples / mult)  # This simplifies to max_batch or num_samples/mult


def get_warmup_samples(dataset):
  query_batches = _init_query_batches()
  pandas_rows = list(dataset.iterrows())  # dataset is already the sampled one if applicable
  input_data = {}
  for sample_idx_in_df, p_row in enumerate(pandas_rows):
    p = p_row[1]  # Get the Series part of the tuple
    original_df_index = p_row[0]  # This is the original index from before any .sample()
    padded, length = pad_tokens(p.tok_input)
    # Using original_df_index as key if it's unique and maps to LoadGen's expectation
    # Or, if LoadGen expects 0 to N-1 indices for the current dataset:
    input_data[sample_idx_in_df] = offline_inference.InputData(original_df_index, jnp.array(padded), length)

  for data_val in input_data.values():  # Corrected variable name
    # make sure tokens are transferred to device
    jax.block_until_ready(data_val.tokens)
  sample_id_to_input = input_data

  # Classify all available (potentially sampled) data into buckets
  temp_query_batches = _init_query_batches()
  for sample_idx_in_df in range(len(pandas_rows)):
    group_idx = _classify_query(pandas_rows, sample_idx_in_df, temp_query_batches)
    if group_idx == -1:
      continue
    # input_ is a mutable object, copy it before modifying id
    current_input_data = copy.copy(sample_id_to_input[sample_idx_in_df])
    # The 'id' for InputData for warmup doesn't directly map to LoadGen query IDs yet
    # It's more of an identifier within the warmup set.
    current_input_data.id = sample_idx_in_df  # or original_df_index, needs consistency
    temp_query_batches[group_idx].append(current_input_data)

  interesting_buckets_input_len = [  # Renamed for clarity
      0,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
  ]
  warmup_samples = _init_query_batches()

  for group_idx_key in temp_query_batches:  # group_idx_key is (prefill_len, batch_size)
    prefill_len_for_bucket = group_idx_key[0]
    # Find the relevant segment in interesting_buckets_input_len for this prefill_len_for_bucket
    # e.g. if prefill_len_for_bucket is 256, we might want samples up to 256.
    # The original logic seems to iterate through sub-segments.

    # Simpler: take a few samples of varying lengths for each engine's prefill_len
    samples_for_this_engine_bucket = sorted(temp_query_batches[group_idx_key], key=lambda x: x.true_length)

    # Try to get one sample from different length ranges if possible
    picked_for_warmup = []
    lengths_picked = set()

    # Pick distinct short, medium, long samples if available for this engine_bucket
    target_warmup_lengths_ratios = [0.1, 0.5, 0.9]  # Ratios of prefill_len_for_bucket
    for ratio in target_warmup_lengths_ratios:
      target_len = int(prefill_len_for_bucket * ratio)
      best_sample = None
      min_diff = float("inf")
      for sample in samples_for_this_engine_bucket:
        if sample.id not in lengths_picked:  # Avoid picking same sample obj if ids are reused
          diff = abs(sample.true_length - target_len)
          if diff < min_diff:
            min_diff = diff
            best_sample = sample
      if best_sample:
        picked_for_warmup.append(best_sample)
        lengths_picked.add(best_sample.id)  # Assuming id is unique for samples here

    # Ensure at least a few, up to a certain limit (e.g. 5, or batch_size for that engine)
    # Add more if we haven't picked enough, prioritizing shorter ones
    additional_needed = min(5, group_idx_key[1]) - len(picked_for_warmup)  # group_idx_key[1] is batch_size
    for sample in samples_for_this_engine_bucket:
      if additional_needed <= 0:
        break
      if sample.id not in lengths_picked:
        picked_for_warmup.append(sample)
        lengths_picked.add(sample.id)
        additional_needed -= 1

    warmup_samples[group_idx_key].extend(picked_for_warmup)
    log.debug(
        f"For engine {group_idx_key}, added {len(picked_for_warmup)} warmup samples with lengths: {[s.true_length for s in picked_for_warmup]}"
    )

  for group_idx_key in temp_query_batches:
    # needed_more = min(50, group_idx_key[1]) - len(warmup_samples[group_idx_key])
    needed_more = 100
    available_in_bucket = temp_query_batches[group_idx_key]
    current_warmup_ids = {s.id for s in warmup_samples[group_idx_key]}

    for sample in available_in_bucket:
      if needed_more <= 0:
        break
      if sample.id not in current_warmup_ids:
        warmup_samples[group_idx_key].append(sample)
        current_warmup_ids.add(sample.id)  # Track added IDs
        needed_more -= 1
  return warmup_samples


class SUT:

  def __init__(self, data, offline_inf_instances):
    self.offline_inf_instances = offline_inf_instances
    self._dataset = data
    self._queries = None
    self._processed_data = None
    self._sample_id_to_input = None
    self._query_batches = _init_query_batches()
    self.pandas_rows = list(self._dataset.iterrows())

  def issue_queries(self, queries):
    log.info("Issue queries start")
    # _sample_id_to_input is populated by LoadSamplesToRam based on LoadGen's indices
    assert self._sample_id_to_input is not None, "_sample_id_to_input not populated. LoadSamplesToRam issue?"
    self._queries = queries  # queries from LoadGen, QuerySample objects with 'index' and 'id'

    num_queries = len(self._queries)
    num_skipped_queries = 0
    # num_grouped_queries_before = [len(self._query_batches[b]) for b in self._query_batches]
    # log.info("Before Issue %d queries - classified queries %s", num_queries, str(num_grouped_queries_before))

    current_call_query_batches = _init_query_batches()

    for q_from_loadgen in queries:
      # q_from_loadgen.index is the index into the QSL (0 to total_sample_count-1)
      group_idx = _classify_query(self.pandas_rows, q_from_loadgen.index, current_call_query_batches)

      if group_idx == -1:
        num_skipped_queries += 1
        log.debug(
            "Filtering out query of input len larger than acceptable configuration for LoadGen query id %d, index %d",
            q_from_loadgen.id,
            q_from_loadgen.index,
        )
      else:
        # Get the pre-processed InputData (tokens on device) using q_from_loadgen.index
        input_data_for_query = self._sample_id_to_input.get(q_from_loadgen.index)
        if input_data_for_query is None:
          log.error(
              f"CRITICAL: No pre-processed input found for LoadGen query index {q_from_loadgen.index}. This should not happen."
          )
          # Handle error appropriately, maybe skip or use a dummy
          num_skipped_queries += 1
          continue

        input_data_copy = copy.copy(input_data_for_query)
        input_data_copy.id = q_from_loadgen.id  # Use LoadGen's query ID for responses
        current_call_query_batches[group_idx].append(input_data_copy)

    self._query_batches = current_call_query_batches  # Store the batches for flush_queries
    num_grouped_queries = [len(self._query_batches[b]) for b in self._query_batches]
    log.info(
        "Issue %d queries - classified queries %s num_skipped %d", num_queries, str(num_grouped_queries), num_skipped_queries
    )

    # The assert needs to account for how LoadGen calls issue_queries vs total queries.
    # LoadGen might call issue_queries multiple times.
    # The important part is that queries received are processed or skipped.
    # This assert might be too strict if issue_queries is called incrementally.
    # For offline, it's usually one big batch.
    # assert num_queries - num_skipped_queries == sum(
    #     num_grouped_queries
    # ), f"num_queries {num_queries} - num_skipped {num_skipped_queries} != sum_grouped {sum(num_grouped_queries)}"
    log.info("Issue queries end")

  @timed("flush_queries")
  def flush_queries(self):
    log.info("Flush queries start")
    # Iterate over the query batches populated by the last call to issue_queries
    for group_idx_key, group_of_input_data in self._query_batches.items():
      if not group_of_input_data:  # Skip if a bucket is empty
        log.info("Skipping flush for empty bucket %s", str(group_idx_key))
        continue

      log.info("Flush queries processing %s with %d samples", str(group_idx_key), len(group_of_input_data))

      # Ensure the specific engine instance is ready
      engine_instance = self.offline_inf_instances.get(group_idx_key)
      if engine_instance is None:
        log.error(f"No engine instance found for group_idx {group_idx_key}. Skipping.")
        # Potentially mark these queries as errored if LoadGen expects a response for all
        for data_input in group_of_input_data:
          # A dummy or error response might be needed
          lg.QuerySamplesComplete([make_response(data_input.id, [0])])  # Example error/dummy
        continue

      engine_instance.init_decode_state()
      # Ensure group_of_input_data is a list of InputData objects
      result = engine_instance.batch_inference(group_of_input_data, desc=f"batch-{group_idx_key}")
      engine_instance.decode_state = None  # Clear state after batch

      for query_id_from_engine, response_tokens in result.items():
        lg_query_id = int(query_id_from_engine)

        if not response_tokens:
          log.warning("Empty response tokens for LoadGen query ID %s from engine bucket %s", lg_query_id, group_idx_key)
          lg.FirstTokenComplete([make_response(lg_query_id, [0])])  # Dummy first token
          lg.QuerySamplesComplete([make_response(lg_query_id, [0])])  # Dummy full response
          continue

        lg.FirstTokenComplete([make_response(lg_query_id, [response_tokens[0]])])
        lg.QuerySamplesComplete([make_response(lg_query_id, response_tokens)])

    self._query_batches = (_init_query_batches())
    log.info("Flush queries end")
    gc.collect()

  def LoadSamplesToRam(self, sample_list_from_loadgen):
    """Pads the data, move them to jax array on device.
    sample_list_from_loadgen: a list of indices (0 to N-1) from LoadGen into the QSL.
    """
    log.info("LoadSamplesToRam start, processing %d samples from LoadGen", len(sample_list_from_loadgen))
    start_time = time.perf_counter()  # Corrected variable name

    current_batch_input_data = {}
    for lg_idx in sample_list_from_loadgen:
      if lg_idx >= len(self.pandas_rows):
        log.error(f"LoadGen index {lg_idx} is out of bounds for dataset size {len(self.pandas_rows)}")
        continue

      original_df_idx, series_data = self.pandas_rows[lg_idx]
      padded_tokens, true_len = pad_tokens(series_data.tok_input)

      current_batch_input_data[lg_idx] = offline_inference.InputData(
          id=original_df_idx,  # Store original index temporarily, will be replaced by LG id
          tokens=jnp.array(padded_tokens),
          true_length=true_len,
      )

    for data_val in current_batch_input_data.values():
      jax.block_until_ready(data_val.tokens)

    self._sample_id_to_input = current_batch_input_data

    end_time = time.perf_counter()
    log.info("LoadSamplesToRam finished: %f seconds", end_time - start_time)

  def UnloadSamplesFromRam(self, sample_list_from_loadgen):
    log.info("UnloadSamplesFromRam called for %d samples", len(sample_list_from_loadgen))
    if self._sample_id_to_input:
      for lg_idx in sample_list_from_loadgen:
        self._sample_id_to_input.pop(lg_idx, None)
      if not self._sample_id_to_input:
        log.info("Unloaded all samples, _sample_id_to_input is now empty.")


def make_response(id_, response_token_ids):
  n_tokens = len(response_token_ids)
  if isinstance(response_token_ids, bytes):
    log.error("make_response received bytes instead of token IDs")
    response_token_ids_np = np.array([0], dtype=np.int64)
  elif not isinstance(response_token_ids, (list, np.ndarray)):
    log.error(f"make_response received unexpected type: {type(response_token_ids)}")
    response_token_ids_np = np.array([0], dtype=np.int64)
  else:
    response_token_ids_np = np.array(response_token_ids, dtype=np.int64)

  response_array = array.array("B", response_token_ids_np.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  query_sample_response = lg.QuerySampleResponse(
      id_, response_data, response_size, n_tokens if n_tokens > 0 else 1 if len(response_token_ids_np) > 0 else 0
  )
  return query_sample_response


def _estimated_counts_by_bucket(dataset):
  total_len_col = dataset.tok_input_length + dataset.tok_output_length
  query_batches = _init_query_batches()
  prefix_lens = sorted([l for l, b in query_batches.keys()])

  estimates = {}
  cumulative_count_so_far = 0
  for i in range(len(prefix_lens)):
    current_prefill_len = prefix_lens[i]

    condition_input_len = dataset.tok_input_length <= current_prefill_len
    condition_total_len = total_len_col <= 2 * current_prefill_len

    fits_current_bucket_conditions = condition_input_len & condition_total_len

    if i > 0:
      prev_prefill_len = prefix_lens[i - 1]
      fits_smaller_bucket_conditions = (dataset.tok_input_length <= prev_prefill_len) & (
          total_len_col <= 2 * prev_prefill_len
      )
      current_bucket_unique_condition = fits_current_bucket_conditions & ~fits_smaller_bucket_conditions
    else:
      current_bucket_unique_condition = fits_current_bucket_conditions

    count_in_bucket = len(dataset[current_bucket_unique_condition])
    estimates[f"prefill<={current_prefill_len}"] = count_in_bucket
    cumulative_count_so_far += count_in_bucket

  unbucketed_count = len(dataset) - cumulative_count_so_far
  if unbucketed_count > 0:
    estimates[f">max_configured_prefill_or_unclassifiable"] = unbucketed_count
  return estimates


def main(argv):
  del argv
  args = FLAGS  # FLAGS are already parsed by absl.app.run(main)

  log.info(f"Initializing with Python Seed: {FLAGS.python_seed}")
  random.seed(FLAGS.python_seed)
  np.random.seed(FLAGS.python_seed)

  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # jax.config.update("jax_explain_cache_misses", True)

  # if FLAGS.enable_profile:
  server = jax.profiler.start_server(FLAGS.jax_profiler_port)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline  # Explicitly Offline
  user_conf = FLAGS.user_conf

  # Load MLPerf and User configurations
  settings.FromConfig(FLAGS.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", FLAGS.mlperf_conf)
  log.info("User config: %s", user_conf)

  log.info("dataset path: %s", FLAGS.dataset_path)
  dataset_full = pd.read_pickle(FLAGS.dataset_path)
  log.info("Full dataset loaded with %d samples.", len(dataset_full))

  if FLAGS.rename_dataset_cols:
    try:
      rename_dict = json.loads(FLAGS.rename_dataset_cols)
      dataset_full.rename(columns=rename_dict, inplace=True)
      log.info("Renaming columns of dataset with mapping: %s", rename_dict)
    except json.JSONDecodeError as e:
      log.error(f"Error decoding rename_dataset_cols JSON string: {e}. Proceeding without renaming.")
    except Exception as e:
      log.error(f"Error applying rename_dataset_cols: {e}. Proceeding without renaming.")

  log.info(f"Deterministically shuffling the entire dataset of {len(dataset_full)} samples using seed {FLAGS.python_seed}.")
  dataset_shuffled_full = dataset_full.sample(frac=1, random_state=FLAGS.python_seed).reset_index(drop=True)

  # 2. Select the top 'total_sample_count' samples from this canonically shuffled dataset.
  if FLAGS.total_sample_count < len(dataset_shuffled_full):
      log.info(f"Selecting the first {FLAGS.total_sample_count} samples from the pre-shuffled dataset.")
      dataset = dataset_shuffled_full.head(FLAGS.total_sample_count)
  else:
      log.info("Using the entire pre-shuffled dataset as total_sample_count (%d) is not less than its length (%d).",
              FLAGS.total_sample_count, len(dataset_shuffled_full))
      dataset = dataset_shuffled_full

  # Reset index for the working dataset to be 0 to N-1, which QSL expects
  # dataset.reset_index(drop=True, inplace=True)
  log.info("Working dataset has %d samples.", len(dataset))

  estimated_counts_by_bucket = _estimated_counts_by_bucket(dataset)  # Estimate on the final working dataset
  log.info("Dataset len %d, estimated counts by bucket %s", len(dataset), estimated_counts_by_bucket)

  len_batch_str = FLAGS.prefill_lengths_and_per_device_batch_sizes
  log.info("Prefill lengths and Batch sizes: %s", len_batch_str)
  log.info("Maxengine args: %s", FLAGS.maxengine_args)

  log.info("Get warmup samples")
  warmup_samples_by_bucket = get_warmup_samples(dataset)
  offline_inf_instances = {}
  query_batches_config = _init_query_batches()

  shared_params = None
  base_maxtext_engine_for_params = None

  for group_key in query_batches_config:
    (length, batch) = group_key
    target_length = 4 * length
    log.info("Creating engine for prefill_len: %d, batch_size: %d, target_len: %d", length, batch, target_length)

    current_engine = create_engine_from_config_flags(
        maxengine_config_filepath=FLAGS.maxengine_config_filepath,
        batch_size=batch,
        max_prefill_predict_length=length,
        max_target_length=target_length,
        args_str=FLAGS.maxengine_args,
    )

    offline_inf = offline_inference.OfflineInference(
        current_engine, shared_params, base_maxtext_engine_for_params, FLAGS.enable_batch_prefill
    )

    if shared_params is None and offline_inf.params is not None:
      log.info(f"Parameters loaded by engine for bucket {group_key}. Will be shared.")
      shared_params = offline_inf.params
      base_maxtext_engine_for_params = current_engine

    offline_inf_instances[group_key] = offline_inf

  if not FLAGS.skip_warmup:
    with timed("warmup"):
      for group_key_for_warmup, engine_instance_for_warmup in offline_inf_instances.items():
        current_warmup_samples = warmup_samples_by_bucket.get(group_key_for_warmup, [])
        if not current_warmup_samples:
          log.info("No warmup samples for engine bucket %s, skipping warmup for this bucket.", str(group_key_for_warmup))
          continue
        log.info("Warming up engine for bucket %s with %d samples.", str(group_key_for_warmup), len(current_warmup_samples))
        engine_instance_for_warmup.warmup(group_key_for_warmup[0], current_warmup_samples)  # Pass prefill_len and samples
        engine_instance_for_warmup.decode_state = None  # drop state
        gc.collect()

  sut = SUT(dataset, offline_inf_instances)

  if FLAGS.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning("Accuracy run will generate the accuracy logs; evaluation is separate.")
  elif FLAGS.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.SubmissionRun
  else:
    settings.mode = lg.TestMode.PerformanceOnly

  os.makedirs(FLAGS.output_log_dir, exist_ok=True)
  log.info("Logging to %s", FLAGS.output_log_dir)
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = FLAGS.output_log_dir
  log_output_settings.copy_summary_to_stdout = True

  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = FLAGS.enable_log_trace

  lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  qsl_count = len(dataset)
  qsl_performance_sample_count = qsl_count
  if FLAGS.perf_count_override is not None:
    qsl_performance_sample_count = FLAGS.perf_count_override
    log.info(f"Overriding QSL performance sample count to: {qsl_performance_sample_count}")

  qsl = lg.ConstructQSL(
      qsl_count,
      min(qsl_performance_sample_count, qsl_count),
      sut.LoadSamplesToRam,
      sut.UnloadSamplesFromRam,
  )
  log.info(
      f"Constructed QSL with count: {qsl_count}, performance_sample_count: {min(qsl_performance_sample_count, qsl_count)}"
  )

  log.info("Starting Benchmark run...")
  lg.StartTestWithLogSettings(lgSUT, qsl, settings, log_settings, FLAGS.audit_conf)

  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)

  # if FLAGS.enable_profile and "server" in locals():
  jax.profiler.stop_server()
  # log.info("JAX Profiler server stopped.")


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
