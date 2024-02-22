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

local base = import 'base.libsonnet';

{
  GPUSpec:: base.BaseAccelerator {
    local gpu = self,

    name: '%(version)s-x%(count)dx%(num_hosts)d' % gpu,
    type: 'gpu',
    version: error 'Must specify GPUSpec `version`',
    count: 1,
    replicas: gpu.count,
    num_hosts: 1,
    // Label used in GCE API
    accelerator_type: error 'Must specify GPUSpec `accelerator_type',

    // Ignore TPU settings.
    PodTemplate(_):: {
      spec+: {
        containerMap+: {
          train+: {
            resources+: {
              limits+: {
                'nvidia.com/gpu': gpu.count,
              },
            },
          },
        },
        nodeSelector+: {
          'cloud.google.com/gke-accelerator': 'nvidia-tesla-%(version)s' % gpu,
        },
      },
    },
  },

  teslaV100: self.GPUSpec { version: 'v100', accelerator_type: 'nvidia-tesla-v100' },
  teslaA100: self.GPUSpec { version: 'a100', accelerator_type: 'nvidia-tesla-a100' },
}
