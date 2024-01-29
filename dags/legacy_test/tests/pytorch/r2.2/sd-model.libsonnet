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
  local sd_model = self.sd_model,
  sd_model:: common.PyTorchTest {
    local config = self,
    modelName: 'lightning-sd-model',
    paramsOverride:: {
      scriptPath: 'stable-diffusion/main_tpu.py',
      trainCommand: [
        'python3',
        self.scriptPath,
        '--train',
        '--no-test',
        '--base=stable-diffusion/configs/latent-diffusion/cin-ldm-vq-f8-ss.yaml',
        '--',
        'data.params.batch_size=32',
        'lightning.trainer.max_epochs=5',
        'model.params.first_stage_config.params.ckpt_path=stable-diffusion/models/first_stage_models/vq-f8/model.ckpt',
        'lightning.trainer.enable_checkpointing=False',
        'lightning.strategy.sync_module_states=False',
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName+: '-pjrt',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        git clone https://github.com/pytorch-tpu/stable-diffusion.git
        cd stable-diffusion
        pip install transformers==4.19.2 diffusers invisible-watermark
        pip install -e .
        pip install torchmetrics==0.7.0
        pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
        pip install lmdb einops omegaconf
        pip install taming-transformers clip kornia==0.6 albumentations==0.4.3
        pip install starlette==0.27.0 && pip install tensorboard
        sudo apt-get update -y && sudo apt-get install libgl1 -y
        # wget -nv https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
        pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
        echo w | pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
        # mv quantize.py ~/.local/lib/python3.8/site-packages/taming/modules/vqvae/

        # taming-transformers and CLIP override existing torch and torchvision so we need to reinstall
        pip uninstall -y torch torchvision
        pip3 install --user --pre torch torchvision --index-url https://download.pytorch.org/whl/test/cpu
        pip install --user \
          'torch_xla[tpuvm] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0rc8-cp310-cp310-manylinux_2_28_x86_64.whl'

        # Setup data
        wget -nv https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
        tar -xf  imagenette2.tgz
        mkdir -p ~/.cache/autoencoders/data/ILSVRC2012_train/data
        mkdir -p ~/.cache/autoencoders/data/ILSVRC2012_validation/data
        mv imagenette2/train/*  ~/.cache/autoencoders/data/ILSVRC2012_train/data
        mv imagenette2/val/* ~/.cache/autoencoders/data/ILSVRC2012_validation/data

        # Get first stage model
        wget -nv -O models/first_stage_models/vq-f8/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
        cd  models/first_stage_models/vq-f8/
        unzip -o model.zip
        cd ~/stable-diffusion/

        # Fix syntax error
        sed -i 's/from torch._six import string_classes/string_classes = (str, bytes)/g' src/taming-transformers/taming/data/utils.py

        # Remove Checkpointing
        sed -i 's/trainer_kwargs\["callbacks"\]/# trainer_kwargs\["callbacks"\]/g' main_tpu.py

        echo 'export PATH=~/.local/bin:$PATH' >> ~/.bash_profile
      |||,
    },
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },
  configs: [
    sd_model + v4_8 + common.Functional + timeouts.Hours(3) + pjrt,
  ],
}
