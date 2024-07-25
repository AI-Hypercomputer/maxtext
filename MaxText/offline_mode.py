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
from maxengine import create_engine_from_config_flags
import offline_inference
import register_jax_proxy_backend

_MLPERF_ID="llama2-70b"

logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.getcwd())
log = logging.getLogger("main2.py")


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mlperf_test_mode",
    "performance",
    "performance, accuracy, submission",
)
flags.DEFINE_string(
    "api_url", None, "SAX published model path.", required=False
)
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


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))), 32)
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


@contextlib.contextmanager
def timed(msg):
  log.info(msg + " start")
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  log.info(msg + " done: " + str(end - start))


def _classify_query(dataset_rows, index):
  # return grouped indexes
  sample = dataset_rows[index][1]
  input_len = sample.tok_input_length
  total_len = sample.tok_input_length + 3*sample.tok_output_length
  if total_len <= 512 and input_len <= 256:
    return 0
  elif total_len <= 1024 and input_len <= 512:
    return 1
  else:
    return 2


def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  """max_batch to not run OOM."""
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  return math.ceil(num_samples / mult * (sample_size / dataset_size))

def get_warmup_samples(dataset):
  groupped_queries = [[], [], []] 
  pandas_rows = list(dataset.iterrows())
  input_data = {}
  for sample_id in range(len(pandas_rows)):
    p = pandas_rows[sample_id][1]
    padded, length = pad_tokens(p.tok_input)
    input_data[sample_id] = offline_inference.InputData(
        "", jnp.array(padded), length  # to be filled later
    )
  for data in input_data.values():
    # make sure tokens are transferred to device
    jax.block_until_ready(data.tokens)
  sample_id_to_input = input_data 
  for sample_id in range(len(input_data)):
    group = _classify_query(pandas_rows, sample_id)  
    input_ = copy.copy(sample_id_to_input[sample_id])
    input_.id = sample_id
    groupped_queries[group].append(input_)

  warmup_samples = [[], [], []]
  for group_idx, group in enumerate(groupped_queries):
    warmup_samples[group_idx] = group[:50]
  return warmup_samples

class SUT:

  def __init__(self, data, offline_inf):
    # dict of int (cache length) -> offline_inf
    self.offline_inf = offline_inf

    # pandas dataframe, it has tok
    self._dataset = data

    # List of things with .id and .index
    self._queries = None

    # index to loaded data
    self._processed_data = None

    # self.replicated = self.offline_inf.engine.env.sharding_by_axis(-1)
    self._sample_id_to_input = None
    self._groupped_queries = [[], [], []]

  def issue_queries(self, queries):
    log.info("Issue queries start")
    assert self._sample_id_to_input is not None
    self._processed_data = []
    self._queries = queries
    for q in queries:
      group = _classify_query(self.pandas_rows, q.index)
      input_data = copy.copy(self._sample_id_to_input[q.index])
      input_data.id = q.id
      self._groupped_queries[group].append(input_data)

    log.info("Issue queries - classified queries")
    assert len(self._queries) == sum(len(q) for q in self._groupped_queries)
    # At this point _processed_data is ready
    log.info("Issue queries end")

  @timed("flush_queries")
  def flush_queries(self):
    log.info("Flush queries start")
    start = time.perf_counter()
    for group_idx, group in enumerate(self._groupped_queries):
      log.info(f"Flush queries processing {group_idx} with {len(group)} samples")
      self.offline_inf[group_idx].init_decode_state()
      result = self.offline_inf[group_idx].batch_inference(group)
      self.offline_inf[group_idx].decode_state = None
      gc.collect()
      for key, val in result.items():
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
      input_data[sample_id] = offline_inference.InputData(
          "", jnp.array(padded), length  # to be filled later
      )

    for data in input_data.values():
      # make sure tokens are transferred to device
      jax.block_until_ready(data.tokens)

    self._sample_id_to_input = input_data

    end = time.perf_counter()
    log.info(f"LoadSamplesToRam finished: {end - start}s")

  def UnloadSamplesFromRam(self, sample_list):
    print("UnloadSamplesFromRam called")
    pass


def make_response(id_, response_token_ids):
  n_tokens = len(response_token_ids)
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  query_sample_response = lg.QuerySampleResponse(
      id_, response_data, response_size, n_tokens
  )
  return query_sample_response


def _count_by_bucket(dataset):

  total_len = dataset.tok_input_length + dataset.tok_output_length

  group1 = (total_len <= 512) & (dataset.tok_input_length <= 256)
  group2 = (total_len <= 1024) & (dataset.tok_input_length <= 512)

  # with 5 percent extra
  mult = FLAGS.total_sample_count / len(dataset) * 1.05

  counts = [
      math.ceil(len(dataset[group1]) * mult),
      math.ceil(len(dataset[~group1 & group2]) * mult),
      math.ceil(len(dataset[~group1 & ~group2]) * mult),
  ]
  return counts

def main(argv):
  del argv
  args = FLAGS
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # jax.config.update("jax_explain_cache_misses", True)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline
  user_conf = FLAGS.user_conf

  settings.FromConfig(FLAGS.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", FLAGS.mlperf_conf)
  log.info("User config: %s", user_conf)

  log.info("dataset path: %s", FLAGS.dataset_path)
  dataset = pd.read_pickle(FLAGS.dataset_path)
  rows = list(dataset.iterrows())
  counts_by_bucket = _count_by_bucket(dataset)
  log.info(f"Counts by bucket {counts_by_bucket}")

  length_and_batch = (
      (256, 48),
      (512, 24),
      (1024, 12),
  )
  engines = []
  params = None
  base_engine = None
  for i, (length, max_batch) in enumerate(length_and_batch):
    batch = counts_by_bucket[i]
    target_length = 2*length
    log.info(f"Using batch size: {max_batch} and length: {length}")
    engine = create_engine_from_config_flags(
      batch_size=max_batch,
      max_prefill_predict_length=length,
      max_target_length=target_length)
    offline_inf = offline_inference.OfflineInference(engine, params, base_engine)
    if params is None and offline_inf.params is not None:
      base_engine = engine
    # offline_inf.dummy = True
    params = offline_inf.params
    engines.append(offline_inf)

  warmup_samples = get_warmup_samples(dataset)
  with timed("warmup"):
    warmup_grp = 0
    for (length, _), engine in zip(length_and_batch, engines):
      log.info(f"warm up for {length}")
      engine.init_decode_state()
      engine.warmup(length, warmup_samples[warmup_grp])
      engine.decode_state = None  # drop state
      gc.collect()
      warmup_grp += 1

  sut = SUT(dataset, engines)

  if FLAGS.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning(
        "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
    )
  elif FLAGS.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.Submission
    settings.print_timestamps = True
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  settings.use_token_latencies = True

  os.makedirs(FLAGS.output_log_dir, exist_ok=True)
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
  lg.StartTestWithLogSettings(
      lgSUT, qsl, settings, log_settings, FLAGS.audit_conf
  )
  log.info(f"query counts {[len(q) for q in sut._groupped_queries]}")
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
