import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

huggingface_model = "/usr/local/google/home/lancewang/tempdisk/DeepSeek-R1-Distill-Llama-8B"
# huggingface_model = "/usr/local/google/home/lancewang/tempdisk/DeepSeek-R1-Distill-Llama-70B"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(huggingface_model)  # Enable trust_remote_code for safetensors
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

model.to(device)

text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000)


response = text_generation_pipeline("I love to")
print(response)
