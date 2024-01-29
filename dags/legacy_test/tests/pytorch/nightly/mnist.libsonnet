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
local gpus = import 'templates/gpus.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local mnist = self.mnist,
  mnist:: common.PyTorchTest {
    modelName: 'mnist',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    command: [
      'python3',
      'pytorch/xla/test/test_train_mp_mnist.py',
      '--datadir=/datasets/mnist-data',
    ] + if self.flags.modelDir != null then [
      '--logdir=%s' % self.flags.modelDir,
    ] else [],
    flags:: {
      modelDir: '$(MODEL_DIR)',
    },
  },

  local fake_data = self.fake_data,
  fake_data:: common.Functional {
    command+: ['--fake_data'],
  },

  local convergence = self.convergence,
  convergence:: common.Convergence {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 98.0,
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
  // DDP converges worse than MP.
  local convergence_ddp = self.convergence_ddp,
  convergence_ddp:: convergence {
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  value: 97.0,
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

  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  local gpu = self.gpu,
  gpu:: common.GpuMixin {
    // Disable XLA metrics report on GPU
    command+: [
      '--nometrics_debug',
    ],
    flags+: {
      modelDir: null,
    },
  },
  local v100x4 = self.v100x4,
  v100x4:: gpu {
    accelerator: gpus.teslaV100 { count: 4 },
  },

  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName+: '-pjrt',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage
      |||,
    },
  },
  local pjrt_ddp = self.pjrt_ddp,
  pjrt_ddp:: {
    modelName+: '-ddp',
    command+: [
      '--ddp',
      '--pjrt_distributed',
      // DDP converges worse than MP, override the accuracy target in Python script.
      '--target_accuracy=97.0',
    ],
  },

  configs: [
    mnist + convergence + v2_8 + timeouts.Hours(1) + pjrt,
    mnist + fake_data + v2_8 + timeouts.Hours(1) + pjrt + mixins.Experimental,
    mnist + convergence_ddp + v2_8 + timeouts.Hours(1) + pjrt + pjrt_ddp,
  ],
}
