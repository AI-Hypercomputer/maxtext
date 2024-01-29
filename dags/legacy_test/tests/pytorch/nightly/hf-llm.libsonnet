// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local command_copy_metrics = |||
    gsutil -m cp -r /tmp/test-clm/*.json $(MODEL_DIR)
  |||,

  local gpt2_model = self.gpt2_model,
  gpt2_model:: common.PyTorchTest {
    local config = self,
    modelName: 'hf-gpt2',
    paramsOverride:: {
      scriptPath: 'examples/pytorch/xla_spawn.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--num_cores=4',
        'examples/pytorch/language-modeling/run_clm.py',
        '--num_train_epochs=3',
        '--dataset_name=wikitext',
        '--dataset_config_name=wikitext-2-raw-v1',
        '--per_device_train_batch_size=%d ' % config.paramsOverride.per_device_train_batch_size,
        '--per_device_eval_batch_size=%d ' % config.paramsOverride.per_device_eval_batch_size,
        '--do_train',
        '--do_eval',
        '--logging_dir=./tensorboard-metrics',
        '--cache_dir=./cache_dir',
        '--output_dir=/tmp/test-clm',
        '--overwrite_output_dir',
        '--cache_dir=/tmp',
        '--config_name=%s' % config.paramsOverride.config_name,
        '--tokenizer_name=gpt2',
        '--block_size=1024',
        '--optim=adafactor',
        '--adafactor=true',
        '--save_strategy=no',
        '--logging_strategy=no',
        '--fsdp=full_shard',
        '--fsdp_config=examples/pytorch/language-modeling/fsdp_config.json',
      ],
    },
    command: utils.scriptCommand(
      |||
        %s
        %s
      ||| % [
        utils.toCommandString(self.paramsOverride.trainCommand),
        command_copy_metrics,
      ]
    ),
  },


  local config_2B = self.config_2B,
  config_2B:: common.Convergence {
    modelName: 'hf-gpt2-2b',
    paramsOverride+:: {
      config_name: '/home/xl-ml-test/transformers/examples/pytorch/language-modeling/my_config_2.json',
      per_device_train_batch_size: 8,
      per_device_eval_batch_size: 8,
    },
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'eval/loss': {
              FINAL: {
                fixed_value: {
                  comparison: 'LESS',
                  value: 7,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },

  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName+: '-pjrt',
    tpuSettings+: {
      tpuVmExports+: |||
        cd transformers/
        export LD_LIBRARY_PATH=/usr/local/lib/
        export PT_XLA_DEBUG=0
        export USE_TORCH=ON
      |||,
      tpuVmExtraSetup: |||
        pip install --upgrade accelerate
        git clone https://github.com/huggingface/transformers.git
        cd transformers
        git checkout ebdb185befaa821304d461ed6aa20a17e4dc3aa2
        pip install .
        git log -1
        pip install datasets evaluate scikit-learn
        gsutil cp -r gs://cloud-tpu-tpuvm-artifacts/config/xl-ml-test/pytorch/gpt2/my_config_*.json examples/pytorch/language-modeling/
        gsutil cp gs://cloud-tpu-tpuvm-artifacts/config/xl-ml-test/pytorch/gpt2/fsdp_config.json examples/pytorch/language-modeling/
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    gpt2_model + v4_8 + config_2B + timeouts.Hours(1) + pjrt,
  ],
}
