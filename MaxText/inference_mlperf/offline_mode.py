# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
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
import collections

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import mlperf_loadgen as lg
# pylint: disable=no-name-in-module

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from maxengine import create_engine_from_config_flags
import offline_inference

_MLPERF_ID = "llama2-70b"
log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOGLEVEL", "INFO"))

sys.path.insert(0, os.getcwd())


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mlperf_test_mode",
    "performance",
    "performance, accuracy, submission",
)
flags.DEFINE_string("api_url", None, "SAX published model path.", required=False)
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
    "prefill_lengths_and_batch_sizes",
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

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))), 32)
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


def _init_query_batches():
  query_batches = {}
  len_batch_str = FLAGS.prefill_lengths_and_batch_sizes.split("|")
  len_batch = []
  for lb in len_batch_str:
    l, b = lb.split(",")
    query_batches[(int(l), int(b))] = []
  return query_batches


@contextlib.contextmanager
def timed(msg):
  log.info(msg + " start")
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  log.info(msg + " done: " + str(end - start))


def _classify_query(dataset_rows, index, query_batches):
  sample = dataset_rows[index][1]
  input_len = sample.tok_input_length
  total_len = int(sample.tok_input_length + FLAGS.tok_outlen_multiplier * sample.tok_output_length)
  query_batch_keys = list(query_batches.keys())
  query_batch_keys.sort()
  target_inputs = [lb[0] for lb in query_batch_keys]
  target_totals = [2 * inp for inp in target_inputs]

  for i in range(len(target_inputs)):
    if total_len <= target_totals[i] and input_len <= target_inputs[i]:
      log.debug(f"Added sample of input length {input_len} total_len {total_len} for {query_batch_keys[i]}")
      return query_batch_keys[i]
  if input_len <= target_inputs[i]:
    log.debug(f"Added sample of input length {input_len} total_len {total_len} for {query_batch_keys[i]}")
    return query_batch_keys[i]
  if not FLAGS.allow_skipping_queries:
    assert False, f"Invalid query input_len {input_len} > max prefill_len configured {query_batch_keys[-1]}."
  return -1


def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  """max_batch to not run OOM."""
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  return math.ceil(num_samples / mult * (sample_size / dataset_size))


def get_warmup_samples(dataset):
  query_batches = _init_query_batches()
  pandas_rows = list(dataset.iterrows())
  input_data = {}
  for sample_id in range(len(pandas_rows)):
    p = pandas_rows[sample_id][1]
    padded, length = pad_tokens(p.tok_input)
    input_data[sample_id] = offline_inference.InputData("", jnp.array(padded), length)  # to be filled later
  for data in input_data.values():
    # make sure tokens are transferred to device
    jax.block_until_ready(data.tokens)
  sample_id_to_input = input_data
  for sample_id in range(len(input_data)):
    group_idx = _classify_query(pandas_rows, sample_id, query_batches)
    if group_idx == -1:
      continue
    input_ = copy.copy(sample_id_to_input[sample_id])
    input_.id = sample_id
    query_batches[group_idx].append(input_)

  interesting_buckets = [
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

  for group_idx in query_batches:
    prefill_len = group_idx[0]
    idx = int(math.log2(prefill_len)) - 3
    for start, end in zip(interesting_buckets[:idx], interesting_buckets[1 : (idx + 1)]):
      log.debug(f"idx:{group_idx} start:{start} end:{end}")
      for sample in query_batches[group_idx]:
        if start < sample.true_length <= end:
          warmup_samples[group_idx].append(sample)
          log.debug(f"Added warmup sample of length {sample.true_length} for ({start}, {end}) bucket for group {group_idx}")
          break
    warmup_samples[group_idx].extend(query_batches[group_idx][:50])
  return warmup_samples


class SUT:

  def __init__(self, data, offline_inf_instances):
    # dict of int (cache length) -> offline_inf_instances
    self.offline_inf_instances = offline_inf_instances

    # pandas dataframe, it has tok
    self._dataset = data

    # List of things with .id and .index
    self._queries = None

    # index to loaded data
    self._processed_data = None

    self._sample_id_to_input = None
    self._query_batches = _init_query_batches()

  def issue_queries(self, queries):
    log.info("Issue queries start")
    assert self._sample_id_to_input is not None
    self._processed_data = []
    self._queries = queries

    num_queries = len(self._queries)
    num_skipped_queries = 0
    num_grouped_queries = [len(self._query_batches[b]) for b in self._query_batches]
    log.info(f"Before Issue {num_queries} queries - classified queries {num_grouped_queries}")
    self._query_batches = _init_query_batches()
    for q in queries:
      group_idx = _classify_query(self.pandas_rows, q.index, self._query_batches)
      if group_idx == -1:
        num_skipped_queries += 1
        log.debug("Filtering out query of input len larger than acceptable configuration")
      else:
        input_data = copy.copy(self._sample_id_to_input[q.index])
        input_data.id = q.id
        self._query_batches[group_idx].append(input_data)
    num_grouped_queries = [len(self._query_batches[b]) for b in self._query_batches]
    log.info(f"Issue {num_queries} queries - classified queries {num_grouped_queries} num_skipped {num_skipped_queries}")

    assert len(self._queries) - num_skipped_queries == sum(
        num_grouped_queries
    ), f"num_queries {num_queries} does not match num_grouped_queries {num_grouped_queries}"
    # At this point _processed_data is ready
    log.info("Issue queries end")

  @timed("flush_queries")
  def flush_queries(self):
    log.info("Flush queries start")
    start = time.perf_counter()
    for group_idx in self._query_batches:
      group = self._query_batches[group_idx]
      log.info(f"Flush queries processing {group_idx} with {len(group)} samples")
      self.offline_inf_instances[group_idx].init_decode_state()
      result = self.offline_inf_instances[group_idx].batch_inference(group, desc=f"batch-{group_idx}")
      self.offline_inf_instances[group_idx].decode_state = None
      gc.collect()
      for key, val in result.items():
        if not val:
          log.info(f"Value empty for key {key}")
          continue
        key = int(key)
        lg.FirstTokenComplete([make_response(key, [val[0]])])
        resp = make_response(key, val)
        lg.QuerySamplesComplete([resp])

    log.info("Flush queries end")
    end = time.perf_counter()

  def LoadSamplesToRam(self, sample_list):
    """Pads the data, move them to jax array on device"""
    log.info("LoadSamplesToRam start")
    start = time.perf_counter()
    input_data = {}
    self.pandas_rows = list(self._dataset.iterrows())

    for sample_id in sample_list:
      p = self.pandas_rows[sample_id][1]
      padded, length = pad_tokens(p.tok_input)
      input_data[sample_id] = offline_inference.InputData("", jnp.array(padded), length)  # to be filled later

    for data in input_data.values():
      # make sure tokens are transferred to device
      jax.block_until_ready(data.tokens)

    self._sample_id_to_input = input_data

    end = time.perf_counter()
    log.info(f"LoadSamplesToRam finished: {end - start}s")

  def UnloadSamplesFromRam(self, sample_list):
    log.info("UnloadSamplesFromRam called")
    pass


def make_response(id_, response_token_ids):
  n_tokens = len(response_token_ids)
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  query_sample_response = lg.QuerySampleResponse(id_, response_data, response_size, n_tokens)
  return query_sample_response


def _estimated_counts_by_bucket(dataset):
  total_len = dataset.tok_input_length + dataset.tok_output_length
  query_batches = _init_query_batches()
  prefix_lens = [l for l, b in list(query_batches.keys())]
  prefix_lens.sort()

  # with 5 percent extra
  mult = FLAGS.total_sample_count / len(dataset) * 1.05
  prev_len = 0
  total_count = 0
  estimates = {}
  for prefix_len in prefix_lens[:-1]:
    target_len = 2 * prefix_len
    condition = (total_len <= target_len) & (dataset.tok_input_length <= prefix_len)
    count = len(dataset[condition])
    estimates[f"{prev_len}-{prefix_len}"] = math.ceil((count - total_count) * mult)
    total_count = count
  estimates[f">{prefix_lens[-1]}"] = math.ceil((len(dataset) - total_count) * mult)
  return estimates


def main(argv):
  del argv
  args = FLAGS
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # jax.config.update("jax_explain_cache_misses", True)

  if FLAGS.enable_profile:
    server = jax.profiler.start_server(FLAGS.jax_profiler_port)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline
  user_conf = FLAGS.user_conf

  settings.FromConfig(FLAGS.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", FLAGS.mlperf_conf)
  log.info("User config: %s", user_conf)

  log.info("dataset path: %s", FLAGS.dataset_path)
  dataset = pd.read_pickle(FLAGS.dataset_path)
  if FLAGS.rename_dataset_cols:
    rename_dict = json.loads(FLAGS.rename_dataset_cols)
    dataset.rename(columns=rename_dict, inplace=True)
    log.info(f"Renaming columns of dataset with mapping: {rename_dict}")

  if FLAGS.total_sample_count < len(dataset):
    dataset = dataset.sample(n=FLAGS.total_sample_count)
  estimated_counts_by_bucket = _estimated_counts_by_bucket(dataset)
  log.info(f"Dataset len {len(dataset)}, estimated counts by bucket {estimated_counts_by_bucket}")

  rows = list(dataset.iterrows())
  len_batch_str = FLAGS.prefill_lengths_and_batch_sizes
  log.info(f"Prefill lengths and Batch sizes: {len_batch_str}")
  log.info(f"Maxengine args: {FLAGS.maxengine_args}")

  log.info("Get warmup samples")
  warmup_samples = get_warmup_samples(dataset)
  offline_inf_instances = {}
  query_batches = _init_query_batches()
  params = None
  base_engine = None
  # Create an engine and corresponding offline_inf_instance per batch of queries
  for group_idx in query_batches:
    (length, batch) = group_idx
    target_length = 2 * length
    log.info(f"Using batch size: {batch} and length: {length}")
    engine = create_engine_from_config_flags(
        batch_size=batch,
        max_prefill_predict_length=length,
        max_target_length=target_length,
        args_str=FLAGS.maxengine_args,
    )
    offline_inf = offline_inference.OfflineInference(engine, params, base_engine, FLAGS.enable_batch_prefill)
    if params is None and offline_inf.params is not None:
      base_engine = engine
    params = offline_inf.params
    offline_inf_instances[group_idx] = offline_inf

  if not FLAGS.skip_warmup:
    with timed("warmup"):
      for group_idx in offline_inf_instances:
        (length, batch) = group_idx
        log.info(f"warm up for {length}")
        offline_inf_instances[group_idx].init_decode_state()
        offline_inf_instances[group_idx].warmup(length, warmup_samples[group_idx])
        offline_inf_instances[group_idx].decode_state = None  # drop state
        gc.collect()

  sut = SUT(dataset, offline_inf_instances)

  if FLAGS.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
  elif FLAGS.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.Submission
    settings.print_timestamps = True
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  settings.use_token_latencies = True

  os.makedirs(FLAGS.output_log_dir, exist_ok=True)
  log.info(f"Logging to {FLAGS.output_log_dir}")
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = FLAGS.output_log_dir
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = FLAGS.enable_log_trace

  lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  qsl = lg.ConstructQSL(
      len(dataset),
      FLAGS.total_sample_count,
      sut.LoadSamplesToRam,
      sut.UnloadSamplesFromRam,
  )
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(lgSUT, qsl, settings, log_settings, FLAGS.audit_conf)
  log.info(f"query counts {[len(sut._query_batches[q]) for q in sut._query_batches]}")
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)

  if FLAGS.enable_profile:
    jax.profiler.stop_server()


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
