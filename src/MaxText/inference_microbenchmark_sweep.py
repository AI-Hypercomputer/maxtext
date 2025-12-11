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

"""Sweep across inference microbenchmarks."""

import os
import sys
import json

import jsonlines

import jax

from MaxText import inference_microbenchmark
from MaxText import pyconfig

try:
  JaxRuntimeError = jax.errors.JaxRuntimeError  # added in JAX 0.4.34
except AttributeError:
  from jax._src.lib import xla_extension

  JaxRuntimeError = xla_extension.XlaRuntimeError


def main():
  """
  User needs to set the config's inference_metadata_file, which is a path to a
  json file.

  This json should contain the following keys:
    - two_axis_order_product_id_list: comma separated string of two_axis_order_product_id
    - prefill_cache_axis_order_list: comma delimited string of prefill_cache_axis_order
    - ar_cache_axis_order_list: comma delimited string of ar_cache_axis_order
    - accelerator: name of the accelerator
    - flatten_microbenchmark_results: Whether or not to flatten results. Should
      be true
  """
  config = pyconfig.initialize(sys.argv)
  base_run_name = config.run_name

  with open(config.inference_metadata_file, "rt", encoding="utf-8") as json_file:
    inference_metadata = json.load(json_file)
    print(f"inference_metadata: {inference_metadata}")

  two_axis_order_product_id_list = inference_metadata["two_axis_order_product_id_list"].split(":")
  prefill_cache_axis_order_list = inference_metadata["prefill_cache_axis_order_list"].split(":")
  ar_cache_axis_order_list = inference_metadata["ar_cache_axis_order_list"].split(":")

  start_two_axis_order_product_id = two_axis_order_product_id_list[0]
  end_two_axis_order_product_id = two_axis_order_product_id_list[-1]

  results = []
  for (
    two_axis_order_product_id,
    prefill_cache_axis_order,
    ar_cache_axis_order,
  ) in zip(
    two_axis_order_product_id_list,
    prefill_cache_axis_order_list,
    ar_cache_axis_order_list,
  ):
    print(f"two_axis_order_product_id {two_axis_order_product_id}")
    print(f"prefill_cache_axis_order {prefill_cache_axis_order}")
    print(f"ar_cache_axis_order {ar_cache_axis_order}")

    run_tag = (
      f"{two_axis_order_product_id}-{prefill_cache_axis_order.replace(',', '')}-{ar_cache_axis_order.replace(',', '')}"
    )
    run_name = f"{base_run_name}/{run_tag}"

    tensorboard_dir = os.path.join(config.base_output_directory, run_name, "tensorboard", "")
    pyconfig._config.keys["prefill_cache_axis_order"] = prefill_cache_axis_order  # pylint: disable=protected-access
    pyconfig._config.keys["ar_cache_axis_order"] = ar_cache_axis_order  # pylint: disable=protected-access
    pyconfig._config.keys["tensorboard_dir"] = tensorboard_dir  # pylint: disable=protected-access
    pyconfig._config.keys["run_name"] = run_name  # pylint: disable=protected-access

    # Prepare metadata (dimensions) json for XLML
    dimensions_json = {
      "base_output_directory": config.base_output_directory,
      "model_name": config.model_name,
      "tokenizer": config.tokenizer_path,
      "weight_dtype": config.weight_dtype,
      "inference_microbenchmark_prefill_lengths": f"{config.inference_microbenchmark_prefill_lengths}",
      "inference_microbenchmark_stages": config.inference_microbenchmark_stages,
      "inference_microbenchmark_loop_iters": f"{config.inference_microbenchmark_loop_iters}",
      "max_prefill_predict_length": f"{config.max_prefill_predict_length}",
      "max_target_length": f"{config.max_target_length}",
      "per_device_batch_size": f"{config.per_device_batch_size}",
      "ici_fsdp_parallelism": f"{config.ici_fsdp_parallelism}",
      "ici_autoregressive_parallelism": f"{config.ici_autoregressive_parallelism}",
      "ici_tensor_parallelism": f"{config.ici_tensor_parallelism}",
      "profiler": f"{config.profiler}",
      "scan_layers": f"{config.scan_layers}",
      "quantization": config.quantization,
      "quantize_kvcache": f"{config.quantize_kvcache}",
      "attention": config.attention,
      "two_axis_order_product_id": f"{two_axis_order_product_id}",
      "prefill_cache_axis_order": f"{prefill_cache_axis_order}",
      "ar_cache_axis_order": f"{ar_cache_axis_order}",
      "compute_axis_order": f"{config.compute_axis_order}",
      "reshape_q": f"{config.reshape_q}",
      "kv_quant_axis": f"{config.kv_quant_axis}",
      "run_name": f"{run_name}",
      "run_tag": f"{run_tag}",
      "config_json_string": json.dumps(
        pyconfig._config.keys,  # pylint: disable=protected-access
        default=lambda x: f"<<non-serializable: {type(x).__qualname__}>>",
      ),
    }
    dimensions_json = {
      **dimensions_json,
      **inference_metadata,
    }
    try:
      microbenchmark_results = inference_microbenchmark.run_benchmarks_with_unsafe_rbg(
        config, inference_metadata=inference_metadata
      )
      if microbenchmark_results:
        metrics = microbenchmark_results["flattened_results"]
        metrics = {k.lower(): v for k, v in metrics.items()}
      else:
        metrics = {}
      dimensions_json["oom"] = "False"
      print(
        f"Completed run {two_axis_order_product_id} out of: "
        f"{start_two_axis_order_product_id} to {end_two_axis_order_product_id}"
      )
    except JaxRuntimeError:
      # OOM
      metrics = {}
      dimensions_json["oom"] = "True"
      print(
        f"Failed at run {two_axis_order_product_id} out of: "
        f"{start_two_axis_order_product_id} to {end_two_axis_order_product_id}"
      )

    final = {"metrics": metrics, "dimensions": dimensions_json}
    print(f"Result: {final}")
    results.append(final)

  print(f"All results {results}")
  path = "inference_microbenchmark_sweep_results.jsonl"
  with jsonlines.open(path, mode="w") as writer:
    writer.write_all(results)


if __name__ == "__main__":
  main()
