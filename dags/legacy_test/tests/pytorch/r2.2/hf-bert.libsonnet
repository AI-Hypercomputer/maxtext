// Copyright 2023 Google LLC
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

{
  local bert = self.bert,
  bert:: common.PyTorchTest {
    modelName: 'hf-bert',
    volumeMap+: {
      datasets: common.datasetsVolume,
    },
    command: [
      'python3',
      'pytorch/xla/test/pjrt/test_train_hf_transformer.py',
      '--logdir=$(MODEL_DIR)',
    ],
  },

  local functional = self.functional,
  functional:: common.Functional {
    command+: [
      '--short_data',
    ],
  },
  local convergence = self.convergence,
  convergence:: common.Convergence,

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName+: '-pjrt',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install tensorboardX google-cloud-storage transformers evaluate sacrebleu sacremoses
      |||,
    },
  },

  configs: [
    bert + functional + v4_8 + pjrt + timeouts.Hours(2),
    bert + convergence + v4_8 + pjrt + timeouts.Hours(12),
  ],
}
