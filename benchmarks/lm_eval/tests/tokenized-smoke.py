# quick_tokenized_smoke.py
import requests, transformers
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ids = tok("The capital of France is", add_special_tokens=False).input_ids

payload = {
  "model": "maxtext-test",
  "prompt": [ids],     # list[list[int]] — tokenized batch of size 1
  "max_tokens": 10,
  "logprobs": 1,
  "echo": True
}
r = requests.post("http://localhost:8000/v1/completions", json=payload)
print(r.status_code)
print(r.json())
