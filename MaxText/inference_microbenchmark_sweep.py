"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Sweep across inference microbenchmarks."""

import os
import sys
import json
import jsonlines
import inference_microbenchmark
import max_utils
import pyconfig
from jax._src.lib import xla_extension


def main():
  pyconfig.initialize(sys.argv)
  config = pyconfig.config

  print(f"pwd: {os.getcwd()}")
  with open(config.inference_metadata_file, encoding='utf-8') as json_file:
    inference_metadata = json.load(json_file)
    print(inference_metadata)

  key_value_axis_order_product_id_list = [
    item for item in inference_metadata['key_value_axis_order_product_id_list'].split(':')
  ]
  prefill_key_axis_order_list = [
    item for item in inference_metadata['prefill_key_axis_order_list'].split(':')
  ]
  prefill_value_axis_order_list = [
    item for item in inference_metadata['prefill_value_axis_order_list'].split(':')
  ]
  ar_key_axis_order_list = [
    item for item in inference_metadata['ar_key_axis_order_list'].split(':')
  ]
  ar_value_axis_order_list = [
    item for item in inference_metadata['ar_value_axis_order_list'].split(':')
  ]
  # inference_microbenchmark_sweep_key_value_axis_order_product_id_list = [
  #   item for item in config.inference_microbenchmark_sweep_key_value_axis_order_product_id_list.split(':')
  # ]
  # inference_microbenchmark_sweep_prefill_key_axis_order_list = [
  #   item for item in config.inference_microbenchmark_sweep_prefill_key_axis_order_list.split(':')
  # ]
  # inference_microbenchmark_sweep_prefill_value_axis_order_list = [
  #   item for item in config.inference_microbenchmark_sweep_prefill_value_axis_order_list.split(':')
  # ]
  # inference_microbenchmark_sweep_ar_key_axis_order_list = [
  #   item for item in config.inference_microbenchmark_sweep_ar_key_axis_order_list.split(':')
  # ]
  # inference_microbenchmark_sweep_ar_value_axis_order_list = [
  #   item for item in config.inference_microbenchmark_sweep_ar_value_axis_order_list.split(':')
  # ]

  results = []
  for (
    key_value_axis_order_product_id,
    prefill_key_axis_order,
    prefill_value_axis_order,
    ar_key_axis_order,
    ar_value_axis_order,
  ) in zip(
    # inference_microbenchmark_sweep_key_value_axis_order_product_id_list,
    # inference_microbenchmark_sweep_prefill_key_axis_order_list,
    # inference_microbenchmark_sweep_prefill_value_axis_order_list,
    # inference_microbenchmark_sweep_ar_key_axis_order_list,
    # inference_microbenchmark_sweep_ar_value_axis_order_list,
    key_value_axis_order_product_id_list,
    prefill_key_axis_order_list,
    prefill_value_axis_order_list,
    ar_key_axis_order_list,
    ar_value_axis_order_list,
  ):
    print(f"key_value_axis_order_product_id {key_value_axis_order_product_id}")
    print(f"prefill_key_axis_order {prefill_key_axis_order}")
    print(f"prefill_value_axis_order {prefill_value_axis_order}")
    print(f"ar_key_axis_order {ar_key_axis_order}")
    print(f"ar_value_axis_order {ar_value_axis_order}")

    # Manually update the config
    # Don't set key_value_axis_order_product_id; otherwise it will recompute
    # ar_key_axis_order and ar_value_axis_order

    # name = f"{key_value_axis_order_product_id}-{prefill_key_axis_order}-{prefill_value_axis_order}-{ar_key_axis_order}-{ar_value_axis_order}"
    # profile_name = f"{key_value_axis_order_product_id}-{prefill_key_axis_order}-{ar_key_axis_order}"
    quant = 'bf16' if not config.quantization else config.quantization
    run_name = f"{inference_metadata['accelerator']}-{config.model_name}-{quant}-{key_value_axis_order_product_id}-{prefill_key_axis_order}-{ar_key_axis_order}"
    tensorboard_dir = os.path.join(config.base_output_directory, run_name, "tensorboard", "")
    checkpoint_dir = os.path.join(config.base_output_directory, run_name, "checkpoint", "")
    metrics_dir = os.path.join(config.base_output_directory, run_name, "metrics", "")
    pyconfig._config.keys['prefill_key_axis_order'] = prefill_key_axis_order
    pyconfig._config.keys['prefill_value_axis_order'] = prefill_value_axis_order
    pyconfig._config.keys['ar_key_axis_order'] = ar_key_axis_order
    pyconfig._config.keys['ar_value_axis_order'] = ar_value_axis_order
    pyconfig._config.keys['tensorboard_dir'] = tensorboard_dir
    pyconfig._config.keys['checkpoint_dir'] = checkpoint_dir
    pyconfig._config.keys['metrics_dir'] = metrics_dir
    pyconfig._config.keys['run_name'] = run_name
    max_utils.write_config_raw_keys_for_gcs(pyconfig._config.keys)

    print(f"prefill_key_axis_order {config.prefill_key_axis_order}")
    print(f"prefill_value_axis_order {config.prefill_value_axis_order}")
    print(f"ar_key_axis_order {config.ar_key_axis_order}")
    print(f"ar_value_axis_order {config.ar_value_axis_order}")
    dimensions_json = {}
    # dimensions_json['key_value_axis_order_product_id'] = key_value_axis_order_product_id
    # dimensions_json['prefill_key_axis_order'] = prefill_key_axis_order
    # dimensions_json['prefill_value_axis_order'] = prefill_value_axis_order
    # dimensions_json['ar_key_axis_order'] = ar_key_axis_order
    # dimensions_json['ar_value_axis_order'] = ar_value_axis_order
    dimensions_json['overwrite_key_value_axis_order_product_id'] = key_value_axis_order_product_id
    dimensions_json['overwrite_prefill_key_axis_order'] = prefill_key_axis_order
    dimensions_json['overwrite_prefill_value_axis_order'] = prefill_value_axis_order
    dimensions_json['overwrite_ar_key_axis_order'] = ar_key_axis_order
    dimensions_json['overwrite_ar_value_axis_order'] = ar_value_axis_order
    dimensions_json['overwrite_tensorboard_dir'] = tensorboard_dir
    dimensions_json['overwrite_run_name'] = run_name
    dimensions_json = {
      **dimensions_json,
      **inference_metadata,
    }
    try:
      metrics = inference_microbenchmark.main(config)
      metrics = {k.lower(): v for k, v in metrics.items()}
      dimensions_json['oom'] = 'False'
    except xla_extension.XlaRuntimeError:
      # OOM
      metrics = {}
      dimensions_json['oom'] = 'True'

    final = {'metrics': metrics, 'dimensions': dimensions_json}
    print(f"Result: {final}")
    results.append(final)
  
  print(f"All results {results}")
  path = 'inference_microbenchmark_sweep_results.jsonl'
  with jsonlines.open(path, mode="w") as writer:
    writer.write_all(results)


if __name__ == "__main__":
  main()