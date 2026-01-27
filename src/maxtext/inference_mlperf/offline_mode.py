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
""" inference mlperf offline_mode module """

import argparse
import array
import contextlib
import copy
import gc
import json
import logging
import math
import os
import time
import warnings

import numpy as np

import pandas as pd

import jax
import jax.numpy as jnp

import mlperf_loadgen as lg  # pytype: disable=import-error
# pylint: disable=no-name-in-module

from MaxText.maxengine import create_engine_from_config_flags
from maxtext.inference_mlperf import offline_inference


warnings.simplefilter("ignore", category=FutureWarning)


_MLPERF_ID = "llama2-70b"
log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOGLEVEL", "INFO"))

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def parse_args():
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(description="Run MLPerf offline inference.")
  parser.add_argument("--mlperf_test_mode", type=str, default="performance", help="performance, accuracy, submission")
  parser.add_argument("--api_url", type=str, default=None, help="published model path.")
  parser.add_argument("--dataset_path", type=str, default=None, help="")
  parser.add_argument("--is_stream", action="store_true", help="")
  parser.add_argument("--input_mode", type=str, default="tokenized", help="Input mode")
  parser.add_argument("--output_mode", type=str, default="tokenized", help="Output mode")
  parser.add_argument(
      "--audit_conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs"
  )
  parser.add_argument("--mlperf_conf", type=str, default="mlperf.conf", help="mlperf rules config")
  parser.add_argument(
      "--user_conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS"
  )
  parser.add_argument("--total_sample_count", type=int, default=24576, help="Number of samples to use in benchmark.")
  parser.add_argument(
      "--perf_count_override", type=int, default=None, help="Overwrite number of samples to use in benchmark."
  )
  parser.add_argument("--output_log_dir", type=str, default="output-logs", help="Where logs are saved.")
  parser.add_argument(
      "--enable_log_trace", action="store_true", help="Enable log tracing. This file can become quite large"
  )
  parser.add_argument(
      "--prefill_lengths_and_per_device_batch_sizes",
      type=str,
      default="256,80|512,40|1024,20",
      help="list of prefill lengths and batch sizes to use for each engine. Format len_1,bs_1|len_2,bs_2|..",
  )
  parser.add_argument(
      "--maxengine_args",
      type=str,
      default="",
      help="Additional arguments to maxtext engine, space separated <name>=<value> pairs",
  )
  parser.add_argument("--jax_profiler_port", type=int, default=9999, help="If set, the jax.profiler port to use.")
  parser.add_argument("--enable_profile", action="store_true", help="If set, enable jax profiling.")
  parser.add_argument("--enable_batch_prefill", action="store_true", help="If set, enable batch prefilling.")
  parser.add_argument("--skip_warmup", action="store_true", help="Skip warmup")
  parser.add_argument(
      "--tok_outlen_multiplier", type=float, default=3.0, help="Multiplier for estimating max predicted output len"
  )
  parser.add_argument(
      "--allow_skipping_queries",
      action="store_true",
      help="Allow skipping queries which have target len greater than 2x configured max prefill len",
  )
  parser.add_argument(
      "--rename_dataset_cols",
      type=str,
      default="",
      help="Rename some of the dataset columns to what is expected by code. For example, "
      "mixtral dataset uses ref_token_length instead of ref_token_len. Format is a string dict "
      'eg. \'{"tok_input_len": "tok_input_length"}\'',
  )
  parser.add_argument(
      "--maxengine_config_filepath", type=str, default=None, help="Base config filepath for initializing MaxEngine."
  )
  return parser.parse_args()


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))), 128)
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


def _init_query_batches(args):
  query_batches = {}
  len_batch_str = args.prefill_lengths_and_per_device_batch_sizes.split("|")
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
  log.info("%s done: %d", msg, end - start)


def _classify_query(dataset_rows, index, query_batches, args):
  """classify query"""
  sample = dataset_rows[index][1]
  input_len = sample.tok_input_length
  total_len = int(sample.tok_input_length + args.tok_outlen_multiplier * sample.tok_output_length)
  query_batch_keys = list(query_batches.keys())
  query_batch_keys.sort()
  target_inputs = [lb[0] for lb in query_batch_keys]
  target_totals = [2 * inp for inp in target_inputs]

  for i, target_input in enumerate(target_inputs):
    if total_len <= target_totals[i] and input_len <= target_input:
      log.debug("Added sample of input length %d total_len %d for %s", input_len, total_len, query_batch_keys[i])
      return query_batch_keys[i]
  if input_len <= target_inputs[-1]:
    log.debug("Added sample of input length %d total_len %d for %s", input_len, total_len, query_batch_keys[-1])
    return query_batch_keys[-1]
  if not args.allow_skipping_queries:
    assert False, f"Invalid query input_len {input_len} > max prefill_len configured {query_batch_keys[-1]}."
  return -1


def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  """max_batch to not run OOM."""
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  return math.ceil(num_samples / mult * (sample_size / dataset_size))


def get_warmup_samples(dataset, args):
  """get warmup samples"""
  query_batches = _init_query_batches(args)
  pandas_rows = tuple(dataset.iterrows())
  input_data = {}
  for sample_id, panda_row in enumerate(pandas_rows):
    p = panda_row[1]
    padded, length = pad_tokens(p.tok_input)
    input_data[sample_id] = offline_inference.InputData("", jnp.array(padded), length)  # to be filled later
  for data in input_data.values():
    # make sure tokens are transferred to device
    jax.block_until_ready(data.tokens)
  sample_id_to_input = input_data
  for sample_id in range(len(input_data)):
    group_idx = _classify_query(pandas_rows, sample_id, query_batches, args)
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
  warmup_samples = _init_query_batches(args)

  for group_idx, group_val in query_batches.items():
    prefill_len = group_idx[0]
    idx = int(math.log2(prefill_len)) - 3
    for start, end in zip(interesting_buckets[:idx], interesting_buckets[1 : (idx + 1)]):
      log.debug("idx:%d start:%d end:%d", group_idx, start, end)
      for sample in group_val:
        if start < sample.true_length <= end:
          warmup_samples[group_idx].append(sample)
          log.debug(
              "Added warmup sample of length %d for (%d, %d) bucket for group %d",
              sample.true_length,
              start,
              end,
              group_idx,
          )
          break
    warmup_samples[group_idx].extend(group_val[:50])
  return warmup_samples


class SUT:
  """System Under Test (SUT) class"""

  def __init__(self, data, offline_inf_instances, args):
    # dict of int (cache length) -> offline_inf_instances
    self.offline_inf_instances = offline_inf_instances

    # pandas dataframe, it has tok
    self._dataset = data

    # list of things with .id and .index
    self._queries = None

    # index to loaded data
    self._processed_data = None

    self._sample_id_to_input = None
    self._query_batches = _init_query_batches(args)
    self.args = args

  def issue_queries(self, queries):
    """issue queries"""
    log.info("Issue queries start")
    assert self._sample_id_to_input is not None
    self._processed_data = []
    self._queries = queries

    num_queries = len(self._queries)
    num_skipped_queries = 0
    num_grouped_queries = list(map(len, self._query_batches.values()))
    log.info("Before Issue %d queries - classified queries %s", num_queries, str(num_grouped_queries))
    self._query_batches = _init_query_batches(self.args)
    for q in queries:
      group_idx = _classify_query(self.pandas_rows, q.index, self._query_batches, self.args)
      if group_idx == -1:
        num_skipped_queries += 1
        log.debug("Filtering out query of input len larger than acceptable configuration")
      else:
        input_data = copy.copy(self._sample_id_to_input[q.index])
        input_data.id = q.id
        self._query_batches[group_idx].append(input_data)
    num_grouped_queries = list(map(len, self._query_batches.values()))
    log.info(
        "Issue %d queries - classified queries %s num_skipped %d",
        num_queries,
        str(num_grouped_queries),
        num_skipped_queries,
    )

    assert len(self._queries) - num_skipped_queries == sum(
        num_grouped_queries
    ), f"num_queries {num_queries} does not match num_grouped_queries {num_grouped_queries}"
    # At this point _processed_data is ready
    log.info("Issue queries end")

  @timed("flush_queries")
  def flush_queries(self):
    """flush queries"""
    log.info("Flush queries start")
    start = time.perf_counter()
    for group_idx, group in self._query_batches.items():
      log.info("Flush queries processing %s with %d samples", str(group_idx), len(group))
      self.offline_inf_instances[group_idx].init_decode_state()
      result = self.offline_inf_instances[group_idx].batch_inference(group, desc=f"batch-{group_idx}")
      self.offline_inf_instances[group_idx].decode_state = None
      for key, val in result.items():
        if not val:
          log.info("Value empty for key %s", key)
          continue
        key = int(key)
        lg.FirstTokenComplete([make_response(key, [val[0]])])
        resp = make_response(key, val)
        lg.QuerySamplesComplete([resp])

    end = time.perf_counter()
    log.info("Flush queries end-start: %d", end - start)
    gc.collect()

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
    log.info("LoadSamplesToRam finished: %ds", end - start)

  def UnloadSamplesFromRam(self, sample_list):
    log.info("UnloadSamplesFromRam called")


def make_response(id_, response_token_ids):
  n_tokens = len(response_token_ids)
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  query_sample_response = lg.QuerySampleResponse(id_, response_data, response_size, n_tokens)
  return query_sample_response


def _estimated_counts_by_bucket(dataset, args):
  """estimated counts by bucket"""
  total_len = dataset.tok_input_length + dataset.tok_output_length
  query_batches = _init_query_batches(args)
  prefix_lens = [l for l, b in list(query_batches.keys())]
  prefix_lens.sort()

  # with 5 percent extra
  mult = args.total_sample_count / len(dataset) * 1.05
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


def main():
  args = parse_args()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # jax.config.update("jax_explain_cache_misses", True)

  if args.enable_profile:
    jax.profiler.start_server(args.jax_profiler_port)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline
  user_conf = args.user_conf

  settings.FromConfig(args.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", args.mlperf_conf)
  log.info("User config: %s", user_conf)

  log.info("dataset path: %s", args.dataset_path)
  dataset = pd.read_pickle(args.dataset_path)
  if args.rename_dataset_cols:
    rename_dict = json.loads(args.rename_dataset_cols)
    dataset.rename(columns=rename_dict, inplace=True)
    log.info("Renaming columns of dataset with mapping: %s", rename_dict)

  if args.total_sample_count < len(dataset):
    dataset = dataset.sample(n=args.total_sample_count)
  estimated_counts_by_bucket = _estimated_counts_by_bucket(dataset, args)
  log.info("Dataset len %d, estimated counts by bucket %s", len(dataset), estimated_counts_by_bucket)

  len_batch_str = args.prefill_lengths_and_per_device_batch_sizes
  log.info("Prefill lengths and Batch sizes: %s", len_batch_str)
  log.info("Maxengine args: %s", args.maxengine_args)

  log.info("Get warmup samples")
  warmup_samples = get_warmup_samples(dataset, args)
  offline_inf_instances = {}
  query_batches = _init_query_batches(args)
  params = None
  base_engine = None
  # Create an engine and corresponding offline_inf_instance per batch of queries
  for group_idx in query_batches:
    (length, batch) = group_idx
    target_length = 2 * length
    log.info("Using batch size: %d and length: %d", batch, length)
    engine = create_engine_from_config_flags(
        maxengine_config_filepath=args.maxengine_config_filepath,
        batch_size=batch,
        max_prefill_predict_length=length,
        max_target_length=target_length,
        args_str=args.maxengine_args,
    )
    offline_inf = offline_inference.OfflineInference(engine, params, base_engine, args.enable_batch_prefill)
    if params is None and offline_inf.params is not None:
      base_engine = engine
    params = offline_inf.params
    offline_inf_instances[group_idx] = offline_inf

  if not args.skip_warmup:
    with timed("warmup"):
      for group_idx in offline_inf_instances:  # pylint: disable=consider-using-dict-items
        length, batch = group_idx
        log.info("warm up for %d", length)
        offline_inf_instances[group_idx].warmup(length, warmup_samples[group_idx])
        offline_inf_instances[group_idx].decode_state = None  # drop state
        gc.collect()

  sut = SUT(dataset, offline_inf_instances, args)

  if args.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
  elif args.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.Submission
    settings.print_timestamps = True
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  settings.use_token_latencies = True

  os.makedirs(args.output_log_dir, exist_ok=True)
  log.info("Logging to %s", args.output_log_dir)
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = args.output_log_dir
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = args.enable_log_trace

  lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  qsl = lg.ConstructQSL(
      len(dataset),
      args.total_sample_count,
      sut.LoadSamplesToRam,
      sut.UnloadSamplesFromRam,
  )
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(lgSUT, qsl, settings, log_settings, args.audit_conf)
  # pylint: disable=protected-access
  log.info("query counts %s", str(list(map(len, sut._query_batches.values()))))
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)

  if args.enable_profile:
    jax.profiler.stop_server()


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  main()
