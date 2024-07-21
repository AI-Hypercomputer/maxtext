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
import pyconfig
from jax._src.lib import xla_extension

from typing import Any, Union
import max_utils


def _lists_to_tuples(l: list[Any]) -> Union[tuple[Any], list[Any]]:
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


def _lists_to_json(l: list[Any]) -> dict[str, Union[str, list[str]]]:
  return {array_axis_name: mesh_axis_names for array_axis_name, mesh_axis_names in l}


def sweep_logical_axis_rules(tpu, quant_mode):

  logical_axis_rules_empty_base = [
    ['embed', []],
    ['activation_embed_and_logits_batch', []],
    ['activation_length', []],
    ['cache_sequence', []],
    ['cache_kv', []],
    ['cache_scale_sequence', []],
    ['cache_scale_heads', []],
    ['cache_scale_kv', []],
  ]

  logical_axis_rules_base = [
    ['norm', ['tensor', 'autoregressive']],
    ['activation_embed', ['tensor', 'autoregressive']],
    ['vocab', ['tensor', 'autoregressive']],
    ['activation_vocab', ['tensor', 'autoregressive']],
    ['kv_heads', ['tensor']],
    ['activation_kv_heads', ['tensor']],
    ['mlp', ['tensor', 'autoregressive']],
    ['activation_mlp', ['tensor', 'autoregressive']],
    ['cache_batch', ['autoregressive']],
    ['cache_heads', ['tensor']],
    ['cache_scale_batch', ['autoregressive']],
  ]

  logical_axis_rule_options = {
    't_on_heads': [
      ['heads', ['tensor']],
      ['kv', []],
      ['activation_batch', []],
      ['activation_heads', ['tensor']],
      ['activation_kv', []],
    ],
    't_on_heads_a_on_batch': [
      ['heads', ['tensor']],
      ['kv', []],
      ['activation_batch', ['autoregressive']],
      ['activation_heads', ['tensor']],
      ['activation_kv', []],
    ],
    't_a_on_heads': [
      ['heads', ['tensor', 'autoregressive']],
      ['kv', []],
      ['activation_batch', []],
      ['activation_heads', ['tensor', 'autoregressive']],
      ['activation_kv', []],
    ],
    't_on_heads_a_on_kvs': [
      ['heads', ['tensor']],
      ['kv', ['autoregressive']],
      ['activation_batch', ['autoregressive']],
      ['activation_heads', ['tensor']],
      ['activation_kv', ['autoregressive']],
    ]
  }

  ici_tensor_n_autoregressive_parallelisms_v5e_16 = (
    (8, 2),
    (2, 8),
  )

  ici_tensor_n_autoregressive_parallelisms_v5e_8 = (
    (8, 1),
    (4, 2),
    (2, 4),
    (1, 8),
  )

  skips_on_v5e_8 = [
    't_on_heads_2_4', # OOM
    't_on_heads_1_8', # OOM
    't_on_heads_a_on_batch_2_4', # OOM
    't_on_heads_a_on_batch_1_8', # OOM
    't_on_heads_a_on_kvs_8_1', # Bad Perf
    't_on_heads_a_on_kvs_4_2', # Bad Perf
    't_on_heads_a_on_kvs_2_4', # Bad Perf
    't_on_heads_a_on_kvs_1_8', # Bad Perf
  ]

  skips_on_v5e_16_w_i8_kv_i8 = [
    't_on_heads_8_2', # OOM
    't_on_heads_2_8', # OOM
    't_on_heads_a_on_batch_8_2', # OOM
    't_on_heads_a_on_batch_2_8', # OOM
    't_a_on_heads_8_2', # done
    't_a_on_heads_2_8', # done
  ]

  skips_on_v5e_16_w_b16_kv_b16 = [
    't_on_heads_8_2', # OOM
    't_on_heads_2_8', # OOM
    't_on_heads_a_on_batch_8_2', # OOM
    't_on_heads_a_on_batch_2_8', # OOM
    't_a_on_heads_8_2', # OOM
    't_a_on_heads_2_8', # OOM
  ]

  ici_tensor_n_autoregressive_parallelisms = None
  skips = []
  if tpu == "v5e-8":
    ici_tensor_n_autoregressive_parallelisms = ici_tensor_n_autoregressive_parallelisms_v5e_8
    skips = skips_on_v5e_8
  elif tpu == "v5e-16":
    ici_tensor_n_autoregressive_parallelisms = ici_tensor_n_autoregressive_parallelisms_v5e_16
    if quant_mode == "w-i8-kv-i8":
      skips = skips_on_v5e_16_w_i8_kv_i8
    elif quant_mode == "w-b16-kv-b16":
      skips = skips_on_v5e_16_w_b16_kv_b16

  if ici_tensor_n_autoregressive_parallelisms is None:
    raise ValueError("ici_tensor_n_autoregressive_parallelisms is not set")

  all_logical_axis_rules_n_ici_parallelisms = dict()
  for logical_axis_rule_tag, logical_axis_rule_option in logical_axis_rule_options.items():
    logical_axis_rules = logical_axis_rules_empty_base + logical_axis_rules_base + logical_axis_rule_option
    for ici_tensor_n_autoregressive_parallelism in ici_tensor_n_autoregressive_parallelisms:
      ici_tensor_parallelisms, ici_autoregressive_parallelisms = ici_tensor_n_autoregressive_parallelism
      logical_axis_rule_tag_w_ici_parallelisms = f"{logical_axis_rule_tag}_{ici_tensor_parallelisms}_{ici_autoregressive_parallelisms}"
      if logical_axis_rule_tag_w_ici_parallelisms not in skips:
        all_logical_axis_rules_n_ici_parallelisms[logical_axis_rule_tag_w_ici_parallelisms] = (logical_axis_rules, ici_tensor_n_autoregressive_parallelism)
  return all_logical_axis_rules_n_ici_parallelisms


def main():
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  base_output_directory = config.base_output_directory
  base_run_name = config.run_name
  print(f"Existing logical axis rules: ", config.logical_axis_rules)
  print(f"{config.tpu=}")

  all_logical_axis_rules_n_ici_parallelisms = sweep_logical_axis_rules(config.tpu, config.quant_mode)
  for logical_axis_rule_tag_w_ici_parallelisms, (logical_axis_rules, (ici_tensor_parallelism, ici_autoregressive_parallelism)) in all_logical_axis_rules_n_ici_parallelisms.items():

    run_name = f"{base_run_name}/{logical_axis_rule_tag_w_ici_parallelisms}"
    pyconfig._config.keys['run_name'] = run_name # pylint: disable=protected-access
    tensorboard_dir = os.path.join(base_output_directory, run_name, "tensorboard", "")
    pyconfig._config.keys['tensorboard_dir'] = tensorboard_dir # pylint: disable=protected-access

    print(f"run_tag: {logical_axis_rule_tag_w_ici_parallelisms}")
    print(f"ici_tensor_parallelism: {ici_tensor_parallelism}")
    print(f"ici_autoregressive_parallelism: {ici_autoregressive_parallelism}")
    print(f"logical_axis_rules: {logical_axis_rules}")

    pyconfig._config.keys['logical_axis_rules'] = _lists_to_tuples(logical_axis_rules)
    pyconfig._config.keys['ici_tensor_parallelism'] = ici_tensor_parallelism
    pyconfig._config.keys['ici_autoregressive_parallelism'] = ici_autoregressive_parallelism

    results = {}
    results["run_tag"] = logical_axis_rule_tag_w_ici_parallelisms
    results["logical_axis_rules"] = _lists_to_json(logical_axis_rules)
    results["ici_tensor_parallelism"] = ici_tensor_parallelism
    results["ici_autoregressive_parallelism"] = ici_autoregressive_parallelism
    metrics = {}
    try:
      microbenchmark_results = inference_microbenchmark.main(config)
      metrics = microbenchmark_results['flattened_results']
      metrics = {k.lower(): v for k, v in metrics.items()}
      print(f"Completed run: {run_name}")
    except xla_extension.XlaRuntimeError:
      # OOM
      print(f"Failed run: {run_name}")

    results["metrics"] = metrics
    with open(f"results.json", "w") as f:
      json.dump(results, f)
    max_utils.upload_blob(f"{base_output_directory}/{run_name}/results.json", f"results.json")

if __name__ == "__main__":
  main()
