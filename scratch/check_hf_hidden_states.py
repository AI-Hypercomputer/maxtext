from diffusers import DiTPipeline
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Check if transformer supports output_hidden_states
print(pipe.transformer.__class__.__name__)
if hasattr(pipe.transformer, 'transformer_blocks'):
    print(f"Block type: {type(pipe.transformer.transformer_blocks[0])}")
    print(pipe.transformer.transformer_blocks[0])
    print("FF net components:")
    for i, layer in enumerate(pipe.transformer.transformer_blocks[0].ff.net):
        print(f"Layer {i}: {type(layer)}")
        print(layer)
elif hasattr(pipe.transformer, 'blocks'):
    print(f"Block type: {type(pipe.transformer.blocks[0])}")
    print(pipe.transformer.blocks[0])
else:
    print("Could not find blocks attribute.")

    # Try running with output_hidden_states=True
    latents = torch.randn(1, 4, 32, 32)
    timesteps = torch.tensor([0], dtype=torch.int64)
    class_labels = torch.tensor([0])
    
    # Print parameter shapes
    if hasattr(pipe.transformer, 'transformer_blocks'):
        block = pipe.transformer.transformer_blocks[0]
        print(f"ff.net.0.proj.weight shape: {block.ff.net[0].proj.weight.shape}")
        print(f"ff.net.0.proj.bias shape: {block.ff.net[0].proj.bias.shape}")
        print(f"ff.net.2.weight shape: {block.ff.net[2].weight.shape}")
        print(f"ff.net.2.bias shape: {block.ff.net[2].bias.shape}")

try:
    outputs = pipe.transformer(latents, timestep=timesteps, class_labels=class_labels, return_dict=True, output_hidden_states=True)
    if hasattr(outputs, 'hidden_states'):
        print("Success! Hidden states available.")
        print(f"Number of hidden states: {len(outputs.hidden_states)}")
        for i, hs in enumerate(outputs.hidden_states):
            print(f"Hidden state {i} shape: {hs.shape}")
    else:
        print("No hidden_states attribute in output.")
        print(f"Output type: {type(outputs)}")
except Exception as e:
    print(f"Error running with output_hidden_states=True: {e}")
