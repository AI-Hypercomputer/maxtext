import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Load the tokenizer and model from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)

# prompt = "Who are you?"
prompt = "<start_of_turn>user\nDescribe image <start_of_image><end_of_turn>\n<start_of_turn>model\n"
image = Image.open("/home/hengtaoguo/projects/maxtext/MaxText/test_assets/test_image.jpg").convert("RGB")

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(device)

model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=300,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

# Decode the first generated sequence
generated = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)


# # Save to disk
# output_path = "golden_data_gemma3-4b.jsonl"


# # Your prompt text
# prompt_texts = ["I love to find"]
# all_data_to_save = []


# for prompt_text in prompt_texts:
#   # Encode the prompt text
#   input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
#   print(input_ids.shape)
#   # Get the logits for the prompt + completion
#   with torch.no_grad():
#     outputs = model(input_ids)
#     logits = outputs.logits
#     print(outputs)

#     # Convert logits to fp32
#     logits = logits.cpu().numpy().astype("float32")

#     print(logits.shape)

#     # Prepare data to be saved
#     data_to_save = {
#         "prompt": prompt_text,
#         "tokens": input_ids.tolist()[0],
#         "logits": logits.tolist()[0],  # Convert numpy array to list for JSON serialization
#     }
#     all_data_to_save.append(data_to_save)

# with jsonlines.open(output_path, "w") as f:
#   f.write_all(all_data_to_save)


# print(f"Data saved to {output_path}")