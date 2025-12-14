curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

laude, how do i finetune qwen3-30b-a3b.yml in this repo? 

⏺ Explore(Find qwen3-30b-a3b finetuning info)
  ⎿  Done (22 tool uses · 37.5k tokens · 49s)                                                                                                                                                                 

⏺ Finetuning Qwen3-30B-A3B in MaxText

  The config is at src/MaxText/configs/models/qwen3-30b-a3b.yml. Here's how to finetune:

  Prerequisites

  # Set up environment
  export VENV_NAME=maxtext_venv
  pip install uv
  uv venv --python 3.12 --seed $VENV_NAME
  source $VENV_NAME/bin/activate

  uv pip install -e .[tpu] --resolution=lowest
  bash tools/setup/setup_post_training_requirements.sh

  # HuggingFace token for model access
  export HF_TOKEN=<your_token>

  SFT Finetuning (Recommended)

  1. Convert HuggingFace checkpoint to MaxText format:
  python3 -m MaxText.utils.ckpt_conversion.to_maxtext \
      src/MaxText/configs/base.yml \
      model_name=qwen3-30b-a3b \
      hf_access_token=${HF_TOKEN} \
      base_output_directory=gs://your-bucket/maxtext-checkpoint \
      scan_layers=True skip_jax_distributed_system=True

  2. Run SFT training:
  python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml \
      run_name=my-finetune-run \
      base_output_directory=gs://your-bucket/output \
      model_name=qwen3-30b-a3b \
      load_parameters_path=gs://your-bucket/maxtext-checkpoint/0/items \
      tokenizer_path=Qwen/Qwen3-30B-A3B \
      hf_path=HuggingFaceH4/ultrachat_200k \
      train_split=train_sft \
      per_device_batch_size=1 \
      steps=1000

  Key Docs & Examples

  | Resource        | Path                                                          |
  |-----------------|---------------------------------------------------------------|
  | SFT Tutorial    | docs/tutorials/posttraining/sft.md                            |
  | Full Finetuning | docs/tutorials/posttraining/full_finetuning.md                |
  | Qwen3 SFT Demo  | src/MaxText/examples/sft_qwen3_demo.ipynb                     |
  | Test Script     | end_to_end/tpu/qwen/moe/qwen3-30b-a3b/1_test_qwen3_30b_a3b.sh |


               | 2.03G/4.00G [00:02<00:02, 702MB/s]
                                                                                                                                                                                 
(.venv) ohadr@t1v-n-5269661a-w-0:~/maxtext
(.venv) ohadr@t1v-n-5269661a-w-0:~/maxtext
(.venv) ohadr@t1v-n-5269661a-w-0:~/maxtext


pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install hf_transfer huggingface_hub[hf_transfer]

export HF_HOME=/dev/shm/huggingface_cache && export HF_HUB_CACHE=/dev/shm/huggingface_cache/hub && HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /dev/shm/huggingface_cache/qwen3-30b-a3b                                                                                       

cd ~/maxtext && python3 -m MaxText.utils.ckpt_scripts.convert_qwen3_moe --base_model_path /dev/shm/huggingface_cache/qwen3-30b-a3b --maxtext_model_path /dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext --model_size qwen3-30b-a3b
cd ~/maxtext && python3 -m MaxText.train src/MaxText/configs/base.yml load_parameters_path=/dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext/0/items model_name=qwen3-30b-a3b per_device_batch_size=1 max_target_length=128 ici_fsdp_parallelism=8 dataset_type=synthetic steps=5 base_output_directory=/tmp/maxtext_output run_name=qwen3_test
cd ~/maxtext && python3 -m MaxText.train src/MaxText/configs/base.yml load_parameters_path=/dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext/0/items model_name=qwen3-30b-a3b per_device_batch_size=1 max_target_length=128 ici_fsdp_parallelism=4 ici_tensor_parallelism=2 dataset_type=synthetic steps=5 base_output_directory=/tmp/maxtext_output run_name=qwen3_test2
cd ~/maxtext && python3 -m MaxText.train src/MaxText/configs/base.yml load_parameters_path=/dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext/0/items model_name=qwen3-30b-a3b per_device_batch_size=1 max_target_length=128 ici_tensor_parallelism=4 dataset_type=synthetic steps=5 base_output_directory=/tmp/maxtext_output run_name=qwen3_test3


cd ~/maxtext && python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml run_name=qwen3_sft base_output_directory=/tmp/maxtext_output model_name=qwen3-30b-a3b load_parameters_path=/dev/shm/huggingface_cache/qwen3-30b-a3b-maxtext/0/items tokenizer_path=Qwen/Qwen3-30B-A3B hf_path=HuggingFaceH4/ultrachat_200k train_split=train_sft per_device_batch_size=1 steps=5