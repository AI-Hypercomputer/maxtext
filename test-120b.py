"""
run on cpu

cp gpt-oss-120b-origin/bc75b44b8a2a116a0e4c6659bcd1b7969885f423/config.json gpt-oss-120b-bf16-v2/config.json 
python test-120b.py
https://paste.googleplex.com/4798567478460416
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# "openai/gpt-oss-120b"
origin="/home/shuningjin_google_com/gpt-oss-120b/gpt-oss-120b-origin/bc75b44b8a2a116a0e4c6659bcd1b7969885f423"
dequantized="/home/shuningjin_google_com/gpt-oss-120b/gpt-oss-120b-bf16-v2"

print("Loading")
model = AutoModelForCausalLM.from_pretrained(origin)
tokenizer = AutoTokenizer.from_pretrained(origin)
model2 = AutoModelForCausalLM.from_pretrained(dequantized, dtype=torch.bfloat16)

# TEST 1: most important
print("Test1")
for (name1, param1), (name2, param2), in zip(model.named_parameters(), model2.named_parameters()):
  assert name1 == name2
  #assert torch.allclose(param1.data, param2.data, atol=1e-8, rtol=1e-8)
  assert torch.allclose(param1.data, param2.data, atol=0, rtol=0)
print("\tAll param are close.")

# TEST 2
# 2.1
print("Test2.1")
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
print("Test2.2")
with torch.no_grad():
    logits1 = model(**inputs).logits
    logits2 = model2(**inputs).logits

# Check if the logits are close enough
# assert torch.allclose(logits1, logits2, atol=1e-8, rtol=1e-8)
assert torch.allclose(logits1, logits2, atol=0, rtol=0)
print("\tAll logits are close.")