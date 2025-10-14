import torch
from transformers import AutoConfig, GptOssForCausalLM

from transformers import AutoModelForCausalLM

# Download configuration from huggingface.co and cache.


# hf_model_path = "/home/anfals/ran_ckpt/hf-671b-bf16"
hf_model_path = "/home/shuningjin/deepseek3-671b/hf-671b-bf16"
hf_model_path = "/home/shuningjin/deepseek2-16b/hf-16b"
hf_model_path = "/home/shuningjin/deepseek3-671b/hf-v3-custom-small"
config = AutoConfig.from_pretrained(hf_model_path)
print(config)
sys.exit(1)
hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, local_files_only=True, low_cpu_mem_usage=True)
