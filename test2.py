from transformers import AutoModelForCausalLM

model_name = "/home/shuningjin/deepseek3-671b/hf-v3-custom-small"
model_name = "/tmp/ds-hf"
model_name = "/tmp/ds-hf"
model_name = "Qwen/Qwen3-4B"
model_name = "/home/shuningjin/deepseek3-671b/hf-671b-bf16"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
for name, val in model.named_parameters():
  print(name, val.shape)
