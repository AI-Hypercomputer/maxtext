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

{
  local resnet50 = self.resnet50,
  resnet50:: common.PyTorchTest {
    modelName: 'resnet50-mp',
    trainScript: 'pytorch/xla/test/test_train_mp_imagenet.py',
    batch_size: null,
    command: [
      'python3',
      self.trainScript,
      '--model=resnet50',
      '--log_steps=200',
    ] + (
      if self.flags.modelDir != null then [
        '--logdir=%s' % self.flags.modelDir,
      ] else []
    ) + (
      if self.batch_size != null then [
        '--batch_size=%s' % self.batch_size,
      ] else []
    ),
    flags:: {
      modelDir: '$(MODEL_DIR)',
    },
    volumeMap+: {
      datasets: common.datasetsVolume,
    },

    cpu: '90.0',
    memory: '400Gi',
  },

  local fake_data = self.fake_data,
  fake_data:: common.Functional {
    mode: 'fake',
    command+: [
      '--fake_data',
    ],
  },
  local functional = self.functional,
  functional:: common.Functional {
    command+: [
      '--num_epochs=2',
      '--datadir=/datasets/imagenet-mini',
    ],
  },
  local convergence = self.convergence,
  convergence:: common.Convergence {
    local config = self,

    command+: [
      '--num_epochs=90',
      '--datadir=/datasets/imagenet',
    ],
    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          aggregateAssertionsMap+:: {
            'Accuracy/test': {
              FINAL: {
                fixed_value: {
                  comparison: 'GREATER',
                  // Larger global batch size gives lower final accuracy
                  value: if config.accelerator.replicas == 1 then 75 else 74,
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
                  value: 65,
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
  local v3_8 = self.v3_8,
  v3_8:: {
    accelerator: tpus.v3_8,
  },
  local v3_32 = self.v3_32,
  v3_32:: {
    accelerator: tpus.v3_32,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
    // Keep same global batch size as v3
    batch_size: 256,
  },
  local v4_32 = self.v4_32,
  v4_32:: {
    accelerator: tpus.v4_32,
    batch_size: 256,
  },

  local gpu = self.gpu,
  gpu:: common.GpuMixin {
    local config = self,

    cpu: '7.0',
    memory: '40Gi',

    command+: [
      '--pjrt_distributed',
      '--nometrics_debug',
      '--num_epochs=2',
    ],
    flags+: {
      modelDir: null,
    },

    jobTemplate+:: {
      spec+: {
        completionMode: 'Indexed',
        completions: config.accelerator.num_hosts,
        parallelism: config.accelerator.num_hosts,
      }
    },
  },
  local v100x2 = self.v100x2,
  v100x2:: gpu {
    accelerator: gpus.teslaV100 { count: 2 },
  },
  local v100x2x2 = self.v100x2x2,
  v100x2x2:: gpu {
    accelerator: gpus.teslaV100 { count: 2, num_hosts: 2 },
  },

  local plugin = self.plugin,
  plugin:: {
    modelName+: '-plugin',
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_REGISTER_INSTALLED_PLUGINS=1
      |||,
    },
  },

  local pjrt_ddp = self.pjrt_ddp,
  pjrt_ddp:: {
    modelName+: '-ddp',
    command+: [
      '--ddp',
      '--pjrt_distributed',
    ],
  },

  local tpuVm = {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage
      |||,
    },
  },
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin + tpuVm {
    modelName: 'resnet50-pjrt',
  },
  local spmd(sharding, replica_batch_size=128) = self.spmd(sharding, replica_batch_size),
  spmd(sharding, replica_batch_size):: {
    // Include sharding spec in the test name
    modelName: std.join('-', ['resnet50-spmd'] + sharding),
    trainScript: 'pytorch/xla/test/spmd/test_train_spmd_imagenet.py',
    command+: ['--sharding=' + std.join(',', sharding)],
    // Keep the same global batch size. In SPMD, the global batch size is
    // divided across all devices.
    batch_size: self.accelerator.size * replica_batch_size,
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_SPMD=1
      |||,
    },
    accelerator+: {
      processes: 1,
    },
  },

  configs: [
    resnet50 + fake_data + v100x2 + timeouts.Hours(3) + plugin,
    resnet50 + fake_data + v100x2x2 + timeouts.Hours(3) + plugin,
    resnet50 + fake_data + v100x2x2 + timeouts.Hours(3),
    resnet50 + fake_data + v100x2x2 + timeouts.Hours(4) + spmd(['batch'], 32),
    // PJRT
    resnet50 + fake_data + v2_8 + timeouts.Hours(3) + pjrt,
    resnet50 + fake_data + v3_8 + timeouts.Hours(2) + pjrt,
    resnet50 + convergence + v3_8 + timeouts.Hours(24) + pjrt,
    resnet50 + fake_data + v3_8 + timeouts.Hours(2) + pjrt + pjrt_ddp,
    resnet50 + fake_data + v3_32 + timeouts.Hours(1) + pjrt,
    resnet50 + fake_data + v4_8 + timeouts.Hours(2) + pjrt,
    resnet50 + convergence + v4_8 + timeouts.Hours(14) + pjrt,
    resnet50 + fake_data + v4_8 + timeouts.Hours(2) + pjrt + pjrt_ddp,
    resnet50 + convergence_ddp + v4_8 + timeouts.Hours(14) + pjrt + pjrt_ddp,
    resnet50 + fake_data + v4_32 + timeouts.Hours(2) + pjrt,
    resnet50 + convergence + v4_32 + timeouts.Hours(24) + pjrt,
    // SPMD
    resnet50 + functional + v4_8 + timeouts.Hours(2) + pjrt + spmd(['batch']),
    resnet50 + functional + v4_8 + timeouts.Hours(2) + pjrt + spmd(['spatial']),
  ],
}
