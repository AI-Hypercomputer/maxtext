from transformers import AutoModelForCausalLM
model_name ="/home/shuningjin/deepseek3-671b/hf-v3-custom-small"
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype="auto",
)
for name, val in model.named_parameters():
  print(name, val.shape)