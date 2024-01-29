local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local diffusers = self.diffusers,
  diffusers:: common.PyTorchTest + common.Functional {
    modelName: 'hf-diffusers',
    command: [
      'accelerate',
      'launch',
      'train_text_to_image.py',
      '--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4',
      '--dataset_name=lambdalabs/pokemon-blip-captions',
      '--use_ema',
      '--resolution=512',
      '--center_crop',
      '--random_flip',
      '--train_batch_size=1',
      '--learning_rate=1e-05',
      '--max_grad_norm=1',
      '--lr_scheduler=constant',
      '--lr_warmup_steps=0',
      '--output_dir=/tmp/sd-pokemon-model',
      '--checkpoints_total_limit=1',
      '--checkpointing_steps=6000',
    ],
  },

  local functional = self.functional,
  functional:: common.Functional {
    command+: [
      '--max_train_steps=100',
    ],
  },
  local convergence = self.convergence,
  convergence:: common.Convergence {
    command+: [
      '--max_train_steps=5000',
    ],
  },

  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin + common.Accelerate {
    tpuSettings+: {
      tpuVmExports+: |||
        export XLA_USE_BF16=1
        cd diffusers/examples/text_to_image/
      |||,
      tpuVmExtraSetup+: |||
        git clone https://github.com/huggingface/diffusers
        cd diffusers
        pip install .

        cd examples/text_to_image
        sed '/accelerate/d' requirements.txt > clean_requirements.txt
        sed '/torchvision/d' requirements.txt > clean_requirements.txt
        sed -i 's/transformers>=.*/transformers>=4.36.2/g' clean_requirements.txt
        pip install -r clean_requirements.txt

        # Skip saving the pretrained model, which contains invalid tensor storage
        sed -i 's/pipeline.save_pretrained(args.output_dir)//g' train_text_to_image.py
      |||,
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

  configs: [
    diffusers + functional + v4_8 + pjrt,
    diffusers + convergence + v4_8 + pjrt,
  ],
}
