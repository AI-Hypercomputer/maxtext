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

import sys
import json
import jsonlines
import inference_microbenchmark
import pyconfig
from jax._src.lib import xla_extension


def main():
  pyconfig.initialize(sys.argv)
  config = pyconfig.config

  inference_microbenchmark_sweep_ar_key_axis_order_list = [
    item for item in config.inference_microbenchmark_sweep_ar_key_axis_order_list.split(':')
  ]
  inference_microbenchmark_sweep_ar_value_axis_order_list = [
    item for item in config.inference_microbenchmark_sweep_ar_value_axis_order_list.split(':')
  ]

  results = []
  for (
    ar_key_axis_order,
    ar_value_axis_order,
  ) in zip(
    inference_microbenchmark_sweep_ar_key_axis_order_list,
    inference_microbenchmark_sweep_ar_value_axis_order_list,
  ):
    print(f"ar_key_axis_order {ar_key_axis_order}")
    print(f"ar_value_axis_order {ar_value_axis_order}")

    # Manually update
    pyconfig._config.keys['ar_key_axis_order'] = ar_key_axis_order
    pyconfig._config.keys['ar_value_axis_order'] = ar_value_axis_order

    print(f"@@config.ar_key_axis_order {config.ar_key_axis_order}")
    print(f"@@config.ar_value_axis_order {config.ar_value_axis_order}")
    dimensions_json = {}
    dimensions_json['ar_key_axis_order'] = ar_key_axis_order
    dimensions_json['ar_value_axis_order'] = ar_value_axis_order
    dimensions_json = {
      **dimensions_json,
      **json.loads(config.inference_microbenchmark_sweep_additional_metadata)
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
    print(f"final {final}")
    results.append(final)
  
  print(f"results {results}")
  path = 'inference_microbenchmark_sweep_results.jsonl'
  with jsonlines.open(path, mode="w") as writer:
    writer.write_all(results)


if __name__ == "__main__":
  main()