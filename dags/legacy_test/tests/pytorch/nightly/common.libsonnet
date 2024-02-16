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

local common = import '../common.libsonnet';
local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  local Nightly = {
    frameworkPrefix: 'pt-nightly',
    tpuSettings+: {
      softwareVersion: 'pytorch-nightly',
    },
    imageTag: 'nightly_3.7',
  },
  PyTorchTest:: common.PyTorchTest + Nightly {
    local config = self,

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': {
            image: config.podTemplate.spec.containerMap.train.image,
            env+: [
              {
                name: 'TPU_NAME',
                valueFrom: {
                  fieldRef: {
                    fieldPath: "metadata.annotations['name.cloud-tpus.google.com/train']",
                  },
                },
              },
            ],
            command: [
              'python3',
              '-c',
              |||
                import importlib_metadata
                import os
                import re

                import cloud_tpu_client

                requirements = importlib_metadata.requires('torch_xla')
                libtpu_pattern = r'libtpu-nightly ?@ https:\/\/storage.googleapis.com\/cloud-tpu-tpuvm-artifacts\/wheels\/libtpu-nightly\/libtpu_nightly-\d.\d.dev(\d{8})-\w+-\w+-\w+.whl'
                libtpu_matches = [
                  re.findall(libtpu_pattern, req)[0]
                  for req in requirements
                  if re.match(libtpu_pattern, req)
                ]
                assert len(libtpu_matches) == 1, f'{len(libtpu_matches)} matches in {requirements} (pattern: `{libtpu_pattern}`)'
                libtpu_date = libtpu_matches[0]
                print('libtpu date:', libtpu_date)

                ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                ctc.wait_for_healthy()
                ctc.configure_tpu_version(f'pytorch-nightly-dev{libtpu_date}', restart_type='always')
                ctc.wait_for_healthy()
              |||,
            ],
          },
        },
      },
    },
  },
  Functional:: mixins.Functional {
    schedule: '0 6 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence,
  PyTorchTpuVmMixin:: experimental.PyTorchTpuVmMixin + experimental.PjRt {
    local config = self,

    tpuSettings+: {
      softwareVersion: 'tpu-ubuntu2204-base',
      tpuVmPytorchSetup: |||
        pip3 install -U setuptools
        # `unattended-upgr` blocks us from installing apt dependencies
        sudo systemctl stop unattended-upgrades
        sudo apt-get -y update
        sudo apt install -y libopenblas-base
        # for huggingface tests
        sudo apt install -y libsndfile-dev
        pip3 install --user --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
        pip install --user \
          'torch_xla[tpuvm] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl'
        pip3 install pillow
        git clone --depth=1 https://github.com/pytorch/pytorch.git
        cd pytorch
        git clone https://github.com/pytorch/xla.git
      |||,
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
      },
    },
  },

  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
  GpuMixin:: {
    local config = self,
    imageTag+: '_cuda_11.8',

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
        containerMap+:: {
          train+: {
            envMap+: {
              GPU_NUM_DEVICES: '%d' % config.accelerator.count,
            },
          },
        },
      },
    },
  },


  Accelerate:: {
    local config = self,
    tpuSettings+: {
      tpuVmExports+: |||
        export PATH=~/.local/bin:$PATH
      |||,
      tpuVmExtraSetup: |||
        git clone https://github.com/huggingface/accelerate.git
        pip install --user ./accelerate

        mkdir -p ~/.cache/huggingface/accelerate/
        cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'HF_CONFIG_EOF'
        compute_environment: LOCAL_MACHINE
        distributed_type: XLA
        downcast_bf16: 'no'
        machine_rank: 0
        main_training_function: main
        mixed_precision: 'no'
        num_machines: 1
        num_processes: %d
        rdzv_backend: static
        same_network: true
        tpu_env: []
        tpu_use_cluster: false
        tpu_use_sudo: false
        use_cpu: false
        HF_CONFIG_EOF

        .local/bin/accelerate env
      ||| % [config.accelerator.numCores],
    },
  },

  // DEPRECATED: Use PyTorchTpuVmMixin instead
  tpu_vm_nightly_install: self.PyTorchTpuVmMixin.tpuSettings.tpuVmPytorchSetup,
}
