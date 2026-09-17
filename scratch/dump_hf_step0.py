from diffusers import DiTPipeline
import torch
import numpy as np
import os

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

def my_call(self, class_labels, num_inference_steps=50):
    batch_size = len(class_labels)
    latent_size = self.transformer.config.sample_size
    latent_channels = self.transformer.config.in_channels

    # Fix seed for reproducibility in case we need it, but we will dump anyway
    torch.manual_seed(42)
    latents = torch.randn(
        batch_size, latent_channels, latent_size, latent_size
    )
    
    # Dump initial latents
    np.save("scratch/hf_latents_step0.npy", latents.numpy())
    print("Dumped initial latents to scratch/hf_latents_step0.npy")

    latent_model_input = torch.cat([latents] * 2)

    class_labels = torch.tensor(class_labels).reshape(-1)
    class_null = torch.tensor([1000] * batch_size)
    class_labels_input = torch.cat([class_labels, class_null], 0)
    
    # Dump class labels
    np.save("scratch/hf_class_labels_input.npy", class_labels_input.numpy())
    print("Dumped class labels to scratch/hf_class_labels_input.npy")

    emb_module = self.transformer.transformer_blocks[0].norm1.emb
    np.save("scratch/hf_y_emb_weight.npy", emb_module.class_embedder.embedding_table.weight.detach().numpy())
    print("Dumped y_emb weights to scratch/hf_y_emb_weight.npy")

    # Dump timestep embedder weights
    np.save("scratch/hf_t_emb_l1_w.npy", emb_module.timestep_embedder.linear_1.weight.detach().numpy())
    np.save("scratch/hf_t_emb_l1_b.npy", emb_module.timestep_embedder.linear_1.bias.detach().numpy())
    np.save("scratch/hf_t_emb_l2_w.npy", emb_module.timestep_embedder.linear_2.weight.detach().numpy())
    np.save("scratch/hf_t_emb_l2_b.npy", emb_module.timestep_embedder.linear_2.bias.detach().numpy())
    print("Dumped timestep embedder weights.")

    self.scheduler.set_timesteps(num_inference_steps)
    t = self.scheduler.timesteps[0]
    
    # Dump timestep
    np.save("scratch/hf_timestep_step0.npy", np.array([t]))
    print(f"Dumped timestep {t} to scratch/hf_timestep_step0.npy")

    # broadcast to batch dimension
    timesteps = torch.tensor([t], dtype=torch.int64).expand(latent_model_input.shape[0])

    # Register hooks
    attn1_input = []
    attn1_output = []
    modulation_output = []
    def hook_attn1(module, input, output):
        # input is a tuple, hidden_states is likely the first element
        attn1_input.append(input[0])
        # output might be a tuple (output, attn_weights) or just tensor
        if isinstance(output, tuple):
            attn1_output.append(output[0])
        else:
            attn1_output.append(output)

    modulation_input = []
    def hook_modulation(module, input, output):
        modulation_input.append(input[0])
        modulation_output.append(output)

    t_emb_output = []
    def hook_t_emb(module, input, output):
        t_emb_output.append(output)

    y_emb_output = []
    def hook_y_emb(module, input, output):
        y_emb_output.append(output)

    time_proj_output = []
    def hook_time_proj(module, input, output):
        time_proj_output.append(output)

    handle_attn1 = self.transformer.transformer_blocks[0].attn1.register_forward_hook(hook_attn1)
    handle_modulation = self.transformer.transformer_blocks[0].norm1.linear.register_forward_hook(hook_modulation)
    
    emb_module = self.transformer.transformer_blocks[0].norm1.emb
    handle_t_emb = emb_module.timestep_embedder.register_forward_hook(hook_t_emb)
    handle_y_emb = emb_module.class_embedder.register_forward_hook(hook_y_emb)
    handle_time_proj = emb_module.time_proj.register_forward_hook(hook_time_proj)

    # Run transformer
    outputs = self.transformer(
        latent_model_input, timestep=timesteps, class_labels=class_labels_input,
        return_dict=True
    )
    noise_pred = outputs.sample
    
    # Dump noise_pred
    np.save("scratch/hf_noise_pred_step0.npy", noise_pred.detach().numpy())
    print("Dumped noise_pred to scratch/hf_noise_pred_step0.npy")

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states
        # Dump block 0 output (hidden_states[1])
        np.save("scratch/hf_block0_out_step0.npy", hidden_states[1].detach().numpy())
        print("Dumped block 0 output to scratch/hf_block0_out_step0.npy")
        
        # Dump embedding output (hidden_states[0]) just in case
        np.save("scratch/hf_emb_out_step0.npy", hidden_states[0].detach().numpy())
        print("Dumped embedding output to scratch/hf_emb_out_step0.npy")
    else:
        print("hidden_states not available in outputs")

    # Dump Attention intermediate activations
    if attn1_input:
        np.save("scratch/hf_attn1_in_step0.npy", attn1_input[0].detach().numpy())
        print("Dumped attn1 input to scratch/hf_attn1_in_step0.npy")
    if attn1_output:
        np.save("scratch/hf_attn1_out_step0.npy", attn1_output[0].detach().numpy())
        print("Dumped attn1 output to scratch/hf_attn1_out_step0.npy")

    # Remove hook
    handle_attn1.remove()
    handle_modulation.remove()
    handle_t_emb.remove()
    handle_y_emb.remove()
    handle_time_proj.remove()

    if modulation_input:
        np.save("scratch/hf_modulation_in_step0.npy", modulation_input[0].detach().numpy())
        print("Dumped modulation input to scratch/hf_modulation_in_step0.npy")
    if modulation_output:
        np.save("scratch/hf_modulation_out_step0.npy", modulation_output[0].detach().numpy())
        print("Dumped modulation output to scratch/hf_modulation_out_step0.npy")

    if t_emb_output:
        np.save("scratch/hf_t_emb_step0.npy", t_emb_output[0].detach().numpy())
        print("Dumped t_emb to scratch/hf_t_emb_step0.npy")
    if y_emb_output:
        np.save("scratch/hf_y_emb_step0.npy", y_emb_output[0].detach().numpy())
        print("Dumped y_emb to scratch/hf_y_emb_step0.npy")

    if time_proj_output:
        np.save("scratch/hf_time_proj_step0.npy", time_proj_output[0].detach().numpy())
        print("Dumped time_proj to scratch/hf_time_proj_step0.npy")

    return noise_pred

my_call(pipe, [2, 879]) # white shark, umbrella
