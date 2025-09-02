"""
python hf_forward.py
"""

import torch
from transformers import AutoConfig, GptOssForCausalLM
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM


model_id = "openai/gpt-oss-20b"
hf_model_path = "/home/shuningjin/gpt-oss-20b/gpt-oss-20b-bf16-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(
    hf_model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# config = AutoConfig.from_pretrained("gpt-oss-20b-test-bf16.json")
# model = GptOssForCausalLM(config=config)
# model.save_pretrained("/home/shuningjin/gpt-oss-20b/test-bf16")

# Dictionary to store the outputs from the HF model
hf_intermediates = {}


def get_hook(name):
  """Returns a hook function that saves the output of a module."""

  def hook(model, input, output):
    # The output might be a tuple, so we store the first element
    hf_intermediates[name] = output[0] if isinstance(output, tuple) else output

  return hook


def get_hook2(name):
  """Returns a hook function that saves the output of a module."""

  def hook(model, input, output):
    # The output might be a tuple, so we store the first element
    hf_intermediates[name] = input[0] if isinstance(input, tuple) else input

  return hook


# Assuming 'hf_layer_0' is your GptOssDecoderLayer instance
hf_layer_0 = hf_model.model.layers[0]
# Hook the initial token embedding layer
hf_model.model.embed_tokens.register_forward_hook(get_hook("token_embeddings"))
# Register hooks on the sub-modules you want to inspect
hf_layer_0.input_layernorm.register_forward_hook(get_hook("post_input_layernorm"))
# Hook the Q, K, and V projections INDIVIDUALLY to get their states before RoPE
hf_layer_0.self_attn.q_proj.register_forward_hook(get_hook("q_proj_pre_rope"))
hf_layer_0.self_attn.k_proj.register_forward_hook(get_hook("k_proj_pre_rope"))
hf_layer_0.self_attn.v_proj.register_forward_hook(get_hook("v_proj_pre_rope"))
hf_layer_0.self_attn.o_proj.register_forward_hook(get_hook2("attention_op_out"))
# TODO
hf_layer_0.self_attn.register_forward_hook(get_hook("attention_output"))
# TODO
hf_layer_0.post_attention_layernorm.register_forward_hook(get_hook("post_attention_layernorm"))
hf_layer_0.mlp.register_forward_hook(get_hook("decoder_layer_output"))  # Hook the MLP for the final output


model = hf_model
prompt_text = "I love to sleep"
assert not model.training

# # Now, when you run the forward pass, the hf_intermediates dict will be populated
# # Note: The residual connection is handled inside the forward pass, so we'll calculate it manually.
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
# # Get the logits for the prompt + completion
with torch.no_grad():
  outputs = model(input_ids)
  logits = outputs.logits.cpu().numpy().astype("float32")

# Prepare data to be saved
data_to_save = {
    "prompt": prompt_text,
    "tokens": input_ids.tolist()[0],
    "logits": logits.tolist()[0],  # Convert numpy array to list for JSON serialization
}
# # Run inference
with torch.no_grad():
  # Pass both input_ids and the attention_mask to the model
  outputs = model(input_ids=input_ids)
  logits = outputs.logits.cpu().numpy().astype("float32")

# tokenizer_output = tokenizer(
#     prompt_text,
#     padding='max_length',      # Pad to the specified max_length
#     max_length=4,              # The target length
#     truncation=True,           # Truncate if the input is ever longer than 4
#     return_tensors="pt"        # Return PyTorch tensors
# )

# # The tokenizer now returns a dictionary, so we extract the tensors
# input_ids = tokenizer_output['input_ids']
# attention_mask = tokenizer_output['attention_mask'] # Also get the attention mask

# print(f"Padded input_ids: {input_ids}")
# print(f"Shape: {input_ids.shape}")
# print(f"Attention mask: {attention_mask}")


# # Run inference
# with torch.no_grad():
#   # Pass both input_ids and the attention_mask to the model
#   outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#   logits = outputs.logits.cpu().numpy().astype("float32")


# Assuming you have already run the code from the previous example

# Access the stored tensor from the dictionary
# print(hf_intermediates)
# layer_0_output_tensor = hf_intermediates["decoder_layer_0_output"]

for k, v in hf_intermediates.items():
  # (batch_size, seq_len, num_heads, head_dim)
  if k.endswith("proj_pre_rope"):
    v = v.reshape(1, 4, -1, 64)
  if k == "attention_op_out":
    v = v.reshape(1, 4, -1, 64)
  print(k)
  print(f"mean={v.mean()}")
  print(f"shape={v.shape}")
  print(v)


# # Now you can inspect it
# print("Successfully accessed the tensor!")
# print("Shape of decoder_layer_0_output:", layer_0_output_tensor.shape)
# print("Type:", type(layer_0_output_tensor))
