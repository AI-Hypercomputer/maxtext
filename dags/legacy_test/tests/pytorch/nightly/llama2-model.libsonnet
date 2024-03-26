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
  local llama2 = self.llama2,
  llama2:: common.PyTorchTest {
    modelName: 'llama2',
  },

  local infer = self.infer,
  infer:: common.Functional + common.PyTorchTpuVmMixin {
    modelName+: '-infer',
    command: [
      'bash',
      '-c',
      |||
        python3 llama/example_text_completion.py True llama/7B ~/spiece.model --max_seq_len=2048 --max_gen_len=1000 --max_batch_size=2 --dynamo=openxla |& tee output.txt

        # Defined in setup script
        python3 getvalue.py
      |||,
    ],

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

        cd
        cat > getvalue.py << GETVALUE_EOF
        file = open("output.txt")
        content = file.readlines()
        warm_line = content[-6]
        warm_value = float((warm_line.split())[5])
        if warm_value > 7.948752:
            raise ValueError(f"warm latency/token {warm_value} exceeded threshold 7.57024 + 5%")
        else:
            print(f"Finished llama2 test and warm latency/token {warm_value} within expected threshold 7.57024 + 5%")
        GETVALUE_EOF
      |||,
    },
  },
  local spmd = self.spmd,
  spmd:: common.Functional + common.PyTorchTpuVmMixin {
    modelName+: '-train-spmd',
    command: [
      'python',
      'transformers/examples/pytorch/language-modeling/run_clm.py',
      '--tokenizer_name=gpt2',
      '--dataset_name=wikitext',
      '--dataset_config_name=wikitext-2-raw-v1',
      '--per_device_train_batch_size=32',
      '--per_device_eval_batch_size=8',
      '--num_train_epochs=1',
      '--do_train',
      '--output_dir=/tmp/output',
      '--overwrite_output_dir',
      '--config_name=7B/2B.json',
      '--save_strategy=no',
      '--logging_strategy=no',
      '--remove_unused_columns=no',
      '--spmd_fsdp_sharding',
      '--torch_dtype=bfloat16',
      '--dataloader_drop_last=yes',
      '--spmd_grad_chkpt',
      '--report_to=none',
    ],
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

        cd
        # 7B config
        mkdir 7B
        cd 7B/
        wget https://storage.googleapis.com/manfei_public_experimental/2B.json
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    llama2 + v4_8 + infer + timeouts.Hours(3),
    llama2 + v4_8 + spmd + timeouts.Hours(3),
  ],
}
