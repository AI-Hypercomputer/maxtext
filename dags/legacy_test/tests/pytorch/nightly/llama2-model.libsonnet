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
  local llama2_inference = self.llama2_inference,
  llama2_inference:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-i',
    paramsOverride:: {
      scriptPath: 'llama/7B/llama2inference.sh',
      trainCommand: [
        'bash',
        self.scriptPath,
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local llama2_training = self.llama2_training,
  llama2_training:: common.PyTorchTest {
    local config = self,
    modelName: 'llama2-t',
    paramsOverride:: {
      scriptPath: 'transformers/7B/llama2training.sh',
      trainCommand: [
        'bash',
        self.scriptPath,
      ],
    },
    command: self.paramsOverride.trainCommand,
  },
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    modelName: 'llama2-pjrt',
  },
  local infer = self.infer,
  infer:: common.PyTorchTpuVmMixin + pjrt {
    modelName+: '-infer',
    tpuSettings+: {
      tpuVmExtraSetup: |||
        # install tokenizer model
        wget https://storage.googleapis.com/tpu-pytorch/lsiyuan-experiment/llama/spiece.model

        # git clone and build llama
        git clone --branch llama2-google-next-inference https://github.com/pytorch-tpu/llama.git
        cd llama
        pip3 install -r requirements.txt
        pip3 install -e .

        # 7B config
        mkdir 7B
        cd 7B/
        echo -e '{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}' >> params.json

        # save llama2 test
        echo -e 'python3 llama/example_text_completion.py True "/home/xl-ml-test/llama/7B" /home/xl-ml-test/spiece.model --max_seq_len=2048 --max_gen_len=1000 --max_batch_size=2 --dynamo=openxla > output.txt' >> llama2inference.sh
        echo -e 'file = open("output.txt")' >> getvalue.py
        echo -e 'content = file.readlines()' >> getvalue.py
        echo -e 'warm_line = content[-6]' >> getvalue.py
        echo -e 'warm_value = float((warm_line.split())[5])' >> getvalue.py
        echo -e 'if warm_value > 7.948752 or warm_value < 7.191728:' >> getvalue.py
        echo -e '    raise ValueError("warm latency/token exceeded throuhold 7.57024 +- 5%")' >> getvalue.py
        echo -e 'else:' >> getvalue.py
        echo -e '    print("Finished llama2 test and warm latency/token within expected throuhold 7.57024 +- 5%")' >> getvalue.py
        echo -e 'cat output.txt' >> llama2inference.sh
        echo -e 'python3 llama/7B/getvalue.py' >> llama2inference.sh
        cat llama2inference.sh
      |||,
    },
  },
  local spmd = self.spmd,
  spmd:: common.PyTorchTpuVmMixin + pjrt {
    modelName+: '-train-spmd',
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=1
        export XLA_IR_DEBUG=1
        export XLA_HLO_DEBUG=1
        export BATCH_SIZE=32
        export NUM_EPOCH=5
        export PROFILE_EPOCH=2
        export PROFILE_STEP=0
        export PROFILE_DURATION_MS=20000
        export XLA_USE_SPMD=1
        export PJRT_DEVICE=TPU
        export TPU_MEGACORE=megacore_dense
      |||,
      tpuVmExtraSetup: |||
        # install tokenizer model
        wget https://storage.googleapis.com/tpu-pytorch/lsiyuan-experiment/llama/spiece.model

        # git clone and build transformers ### llama/transformers/
        git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git
        cd transformers
        sudo pip3 uninstall transformers
        sudo pip3 install -e .
        pip3 install datasets
        pip3 install evaluate
        pip3 install scikit-learn
        pip3 install accelerate
        pwd
        ls

        # 7B config
        mkdir 7B
        cd 7B/
        wget https://storage.googleapis.com/manfei_public_experimental/2B.json

        # save llama2 training
        echo -e 'python transformers/examples/pytorch/language-modeling/run_clm.py --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs 1 --do_train --output_dir /tmp/output --overwrite_output_dir --config_name transformers/7B/2B.json --save_strategy no --logging_strategy no --remove_unused_columns no --spmd_fsdp_sharding --torch_dtype bfloat16 --dataloader_drop_last yes --spmd_grad_chkpt --report_to none' >> llama2training.sh
        cat llama2training.sh
        pwd
        ls
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    llama2_inference + v4_8 + common.Functional + timeouts.Hours(3) + infer,
    llama2_training + v4_8 + common.Functional + timeouts.Hours(3) + spmd,
  ],
}
