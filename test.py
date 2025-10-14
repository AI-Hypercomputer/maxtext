import torch
from transformers import AutoConfig, GptOssForCausalLM
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM


hf_model_path = "/home/anfals/ran_ckpt/hf-671b-bf16"
hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
#tokenizer = AutoTokenizer.from_pretrained(model_id)
# hf_model = AutoModelForCausalLM.from_pretrained(
#     hf_model_path,
#     # torch_dtype=torch.bloat16,
#     # trust_remote_code=True,
# )