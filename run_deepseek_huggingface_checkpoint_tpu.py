from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torch_xla.core.xla_model as xm


huggingface_model = "/usr/local/google/home/lancewang/tempdisk/DeepSeek-R1-Distill-Llama-8B"
# huggingface_model = "/usr/local/google/home/lancewang/tempdisk/DeepSeek-R1-Distill-Llama-70B"

device = xm.xla_device()
print(f"Running on {device}")

model = AutoModelForCausalLM.from_pretrained(huggingface_model)  # Enable trust_remote_code for safetensors
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

model.to(device)

from torch_xla.distributed.parallel_loader import ParallelLoader

train_loader = ParallelLoader(train_loader, [device])

text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000)


response = text_generation_pipeline("I love to")
print(response)
