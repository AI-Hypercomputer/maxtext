"""
run on cpu
python test-20b.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# "openai/gpt-oss-20b"
origin="~/gpt-oss-20b/gpt-oss-20b-origin"
dequantized="~/gpt-oss-20b/gpt-oss-20b-bf16-v2"

model = AutoModelForCausalLM.from_pretrained(origin)
tokenizer = AutoTokenizer.from_pretrained(origin)
model2 = AutoModelForCausalLM.from_pretrained(dequantized, dtype=torch.bfloat16)

# TEST 1: most important
for (name1, param1), (name2, param2), in zip(model.named_parameters(), model2.named_parameters()):
  assert name1 == name2
  #assert torch.allclose(param1.data, param2.data, atol=1e-8, rtol=1e-8)
  assert torch.allclose(param1.data, param2.data, atol=0, rtol=0)

# TEST 2
# 2.1
prompt = "The capital of France is Paris, and the capital of Germany is"
inputs = tokenizer(prompt, return_tensors="pt")#.to(device)
# Generate text with the first model
output1 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
text1 = tokenizer.decode(output1[0], skip_special_tokens=True)
# Generate text with the second model
output2 = model2.generate(**inputs, max_new_tokens=20, do_sample=False)
text2 = tokenizer.decode(output2[0], skip_special_tokens=True)
print("--- Original Model Output ---")
print(text1)
print("\n--- BF16 Model Output ---")
print(text2)

# 2.2
# Using the same tokenized prompt from before
with torch.no_grad():
    logits1 = model(**inputs).logits
    logits2 = model2(**inputs).logits

# Check if the logits are close enough
# torch.all(logits1 == logits2)
assert torch.allclose(logits1, logits2, atol=0, rtol=0)