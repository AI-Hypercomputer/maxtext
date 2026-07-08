import os
import json
import re

os.environ["HF_HOME"] = "/dev/shm/hengtaoguo"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bharatgenai/Param2-17B-A2.4B-Thinking"
cache_dir = "/dev/shm/hengtaoguo"
device = torch.device("cpu")


def parse_model_output(text):
    original_text = text
    text = re.sub(r"<\|im_start\|>.*?(\n|$)", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<\|user\|>", "", text).strip()

    reasoning = []
    complete_think_pattern = r"<think>(.*?)</think>"
    for match in re.finditer(complete_think_pattern, text, re.DOTALL):
        reasoning.append(match.group(1).strip())
    if "<think>" in text and "</think>" not in text:
        match = re.search(r"<think>(.*)", text, re.DOTALL)
        if match:
            reasoning.append(match.group(1).strip())
    text = re.sub(complete_think_pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    tool_calls = []
    if "<tool_call>" in text and "</tool_call>" not in text:
        text += "</tool_call>"
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    for match in re.finditer(tool_call_pattern, text, re.DOTALL):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
            tool_calls.append(
                {"name": parsed.get("name"), "arguments": parsed.get("arguments", {}), "raw": raw}
            )
        except json.JSONDecodeError:
            tool_calls.append({"error": "Invalid JSON", "raw": raw})
    text = re.sub(tool_call_pattern, "", text, flags=re.DOTALL).strip()

    return {
        "raw_output": original_text,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "final_answer": text,
    }

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    device_map=None,
)
model.to(device)
model.eval()

conversation = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "Where is Paris?"},
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_tokens = output[0][inputs.shape[-1]:]

# 🔥 IMPORTANT: skip_special_tokens=False
generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

parsed = parse_model_output(generated_text)

print("\n========== RAW ==========\n", generated_text)
print("\n========== REASONING ==========\n", parsed["reasoning"])
print("\n========== TOOL CALLS ==========\n", parsed["tool_calls"])
print("\n========== FINAL ANSWER ==========\n", parsed["final_answer"])
