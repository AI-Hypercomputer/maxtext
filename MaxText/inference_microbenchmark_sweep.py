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

  with open(config.inference_metadata_file, encoding='utf-8') as json_file:
    inference_metadata = json.load(json_file)
    print(f"inference_metadata: {inference_metadata}")

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

  results = []
  for (
    key_value_axis_order_product_id,
    prefill_key_axis_order,
    prefill_value_axis_order,
    ar_key_axis_order,
    ar_value_axis_order,
  ) in zip(
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
      "enable_profiler": config.enable_profiler,
      "scan_layers": config.scan_layers,
      "quantization": config.quantization,
      "quantize_kvcache": config.quantize_kvcache,
      "attention": config.attention,
    }
    dimensions_json = {
      **dimensions_json,
      **inference_metadata,
    }
    try:
      metrics = inference_microbenchmark.main(config, inference_metadata=inference_metadata)
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