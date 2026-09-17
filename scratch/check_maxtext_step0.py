import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from maxtext.configs import pyconfig
from maxtext.models.dit import DiT
from maxtext.utils import maxtext_utils
from maxtext.utils import max_utils
from typing import Sequence
from absl import app

def main(argv: Sequence[str]) -> None:
    print("Initializing config...")
    config = pyconfig.initialize(argv)
    
    if len(jax.devices()) == 4:
        devices = jax.devices()
    elif len(jax.devices()) >= 8:
        devices = jax.devices()[4:8]
    else:
        devices = jax.devices()[:4]
    print(f"Restricting to devices: {devices}")
    devices_array = maxtext_utils.create_device_mesh(config, devices=devices)

    mesh = Mesh(devices_array, config.mesh_axes)
    replicated_sharding = NamedSharding(mesh, P())
    
    print("Instantiating DiT model...")
    rng = jax.random.PRNGKey(0)
    rng, rng_model = jax.random.split(rng)
    nnx_rngs = nnx.Rngs(rng_model)
    
    model = DiT(config, mesh, rngs=nnx_rngs)
    
    if config.load_parameters_path:
        from maxtext.common import checkpointing
        print(f"Loading parameters from {config.load_parameters_path}...")
        _, params, _ = nnx.split(model, nnx.Param, ...)
        checkpointing.load_params_from_path(
            config.load_parameters_path,
            params,
            config.checkpoint_storage_concurrent_gb,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
        print("Moving parameters to device...")
        params = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), params)
        print("Updating model...")
        nnx.update(model, params)
        print("Parameters loaded successfully.")
        
        # Compare Label Embedder Weights
        # Need to load hf_y_emb_weight here if we want to check it immediately
        try:
            hf_y_emb_weight = np.load("scratch/hf_y_emb_weight.npy")
            maxtext_y_emb_weight = model.label_embedder.embedding.embedding.get_value()
            print(f"hf_y_emb_weight shape: {hf_y_emb_weight.shape}")
            print(f"maxtext_y_emb_weight shape: {maxtext_y_emb_weight.shape}")
            
            if maxtext_y_emb_weight.shape == hf_y_emb_weight.shape:
                is_close_y_w = np.allclose(maxtext_y_emb_weight, hf_y_emb_weight, atol=1e-5, rtol=1e-5)
                print(f"Label Embedding Weights match? {is_close_y_w}")
                if not is_close_y_w:
                    diff_y_w = np.abs(maxtext_y_emb_weight - hf_y_emb_weight)
                    print(f"Label Embedding Weight Max diff: {diff_y_w.max()}")
                    print(f"Label Embedding Weight Mean diff: {diff_y_w.mean()}")
            else:
                print(f"Label Embedding Weights shapes mismatch: MaxText {maxtext_y_emb_weight.shape}, HF {hf_y_emb_weight.shape}")
        except FileNotFoundError:
            print("hf_y_emb_weight.npy not found, skipping weight check.")

        # Compare Timestep Embedder Weights
        try:
            hf_t_emb_l1_w = np.load("scratch/hf_t_emb_l1_w.npy")
            hf_t_emb_l1_b = np.load("scratch/hf_t_emb_l1_b.npy")
            hf_t_emb_l2_w = np.load("scratch/hf_t_emb_l2_w.npy")
            hf_t_emb_l2_b = np.load("scratch/hf_t_emb_l2_b.npy")
            
            maxtext_t_emb_l1_w = model.timestep_embedder.linear1.kernel.get_value()
            maxtext_t_emb_l1_b = model.timestep_embedder.linear1.bias.get_value()
            maxtext_t_emb_l2_w = model.timestep_embedder.linear2.kernel.get_value()
            maxtext_t_emb_l2_b = model.timestep_embedder.linear2.bias.get_value()
            
            maxtext_t_emb_l1_w_T = maxtext_t_emb_l1_w.T
            maxtext_t_emb_l2_w_T = maxtext_t_emb_l2_w.T
            
            if maxtext_t_emb_l1_w_T.shape == hf_t_emb_l1_w.shape:
                is_close_l1_w = np.allclose(maxtext_t_emb_l1_w_T, hf_t_emb_l1_w, atol=1e-5, rtol=1e-5)
                print(f"Timestep Embedder L1 Weights match? {is_close_l1_w}")
                if not is_close_l1_w:
                    diff_l1_w = np.abs(maxtext_t_emb_l1_w_T - hf_t_emb_l1_w)
                    print(f"Timestep Embedder L1 Weight Max diff: {diff_l1_w.max()}")
            
            if maxtext_t_emb_l1_b.shape == hf_t_emb_l1_b.shape:
                is_close_l1_b = np.allclose(maxtext_t_emb_l1_b, hf_t_emb_l1_b, atol=1e-5, rtol=1e-5)
                print(f"Timestep Embedder L1 Bias match? {is_close_l1_b}")
                if not is_close_l1_b:
                    diff_l1_b = np.abs(maxtext_t_emb_l1_b - hf_t_emb_l1_b)
                    print(f"Timestep Embedder L1 Bias Max diff: {diff_l1_b.max()}")

            if maxtext_t_emb_l2_w_T.shape == hf_t_emb_l2_w.shape:
                is_close_l2_w = np.allclose(maxtext_t_emb_l2_w_T, hf_t_emb_l2_w, atol=1e-5, rtol=1e-5)
                print(f"Timestep Embedder L2 Weights match? {is_close_l2_w}")
                if not is_close_l2_w:
                    diff_l2_w = np.abs(maxtext_t_emb_l2_w_T - hf_t_emb_l2_w)
                    print(f"Timestep Embedder L2 Weight Max diff: {diff_l2_w.max()}")
            
            if maxtext_t_emb_l2_b.shape == hf_t_emb_l2_b.shape:
                is_close_l2_b = np.allclose(maxtext_t_emb_l2_b, hf_t_emb_l2_b, atol=1e-5, rtol=1e-5)
                print(f"Timestep Embedder L2 Bias match? {is_close_l2_b}")
                if not is_close_l2_b:
                    diff_l2_b = np.abs(maxtext_t_emb_l2_b - hf_t_emb_l2_b)
                    print(f"Timestep Embedder L2 Bias Max diff: {diff_l2_b.max()}")

        except FileNotFoundError as e:
            print(f"Could not load timestep embedder debug files: {e}")

        print(f"DEBUG: block 0 attn query_pre_attn_scalar: {model.blocks[0].attn.query_pre_attn_scalar}")
        print(f"DEBUG: block 0 mlp wi kernel shape: {model.blocks[0].mlp.wi.kernel.shape}")
        print(f"DEBUG: block 0 mlp wo kernel shape: {model.blocks[0].mlp.wo.kernel.shape}")

    # Load dumped HF inputs
    try:
        hf_latents = np.load("scratch/hf_latents_step0.npy")
        hf_class_labels_input = np.load("scratch/hf_class_labels_input.npy")
        hf_timestep = np.load("scratch/hf_timestep_step0.npy")
        hf_noise_pred = np.load("scratch/hf_noise_pred_step0.npy")
        hf_modulation_in = np.load("scratch/hf_modulation_in_step0.npy")
        hf_modulation_out = np.load("scratch/hf_modulation_out_step0.npy")
        hf_t_emb = np.load("scratch/hf_t_emb_step0.npy")
        hf_y_emb = np.load("scratch/hf_y_emb_step0.npy")
        hf_time_proj = np.load("scratch/hf_time_proj_step0.npy")
        hf_y_emb_weight = np.load("scratch/hf_y_emb_weight.npy")
    except FileNotFoundError as e:
        print(f"Error loading dumped HF files: {e}")
        return

    print(f"HF Latents shape: {hf_latents.shape}")
    print(f"HF Class labels shape: {hf_class_labels_input.shape}")
    print(f"HF Timestep: {hf_timestep}")
    print(f"HF Noise pred shape: {hf_noise_pred.shape}")

    # Patchify HF latents to MaxText format
    # HF shape: (B, C, H, W) -> (2, 4, 32, 32)
    # Target shape: (B, seq_len, patch_dim) -> (2, 256, 16)
    B, C, H, W = hf_latents.shape
    p = config.patch_size_for_vit
    side = H // p # 16
    
    # Reshape to split H and W into side blocks of size p
    hf_reshaped = hf_latents.reshape(B, C, side, p, side, p)
    
    # Transpose to group (side_h, side_w) and (C, p_h, p_w)
    # axes: B(0), C(1), side_h(2), p_h(3), side_w(4), p_w(5)
    # target: B(0), side_h(2), side_w(4), C(1), p_h(3), p_w(5)
    hf_transposed = np.transpose(hf_reshaped, (0, 2, 4, 1, 3, 5))
    
    # Flatten
    maxtext_input = hf_transposed.reshape(B, side * side, C * p * p)
    
    print(f"MaxText input shape: {maxtext_input.shape}")
    
    # Concatenate inputs to match batch size 4 (B*2)
    maxtext_input = jnp.concatenate([maxtext_input] * 2, axis=0)
    t_batch = jnp.array([hf_timestep[0]] * (B * 2))
    y_batch = jnp.array(hf_class_labels_input)

    
    print(f"MaxText input shape after concat: {maxtext_input.shape}")
    print(f"t_batch shape: {t_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    
    # Move to device
    maxtext_input = jax.device_put(maxtext_input, replicated_sharding)
    t_batch = jax.device_put(t_batch, replicated_sharding)
    y_batch = jax.device_put(y_batch, replicated_sharding)
    
    # Run MaxText model
    noise_pred_maxtext = model(maxtext_input, t_batch, y_batch)

    
    print(f"MaxText Output shape: {noise_pred_maxtext.shape}")
    
    # Now we need to compare this output with hf_noise_pred
    
    # Let's also compare block 0 output if available
    # Let's compare intermediate attention activations
    try:
        hf_attn1_in = np.load("scratch/hf_attn1_in_step0.npy")
        hf_attn1_out = np.load("scratch/hf_attn1_out_step0.npy")
        print(f"HF Attn1 in shape: {hf_attn1_in.shape}")
        print(f"HF Attn1 out shape: {hf_attn1_out.shape}")
        
        print("Running MaxText step by step manually...")
        
        # 1. Embedding
        x = model.patch_embed(maxtext_input)
        x = x + model.pos_embed.get_value()
        print(f"MaxText Emb out shape: {x.shape}")
        
        # 2. Timestep and Label Embedders
        # Manual Time Projection Check
        t_reshaped = t_batch[:, jnp.newaxis]
        temb_raw = model.timestep_embedder.pos_emb(seq_len=1, position=t_reshaped)
        temb_raw = temb_raw[:, 0, :]
        
        # Flip sine and cosine embeddings to match HF flip_sin_to_cos=True
        half_dim = temb_raw.shape[-1] // 2
        temb_flipped = jnp.concatenate([temb_raw[:, half_dim:], temb_raw[:, :half_dim]], axis=-1)
        
        print(f"hf_time_proj shape: {hf_time_proj.shape}")
        print(f"temb_flipped shape: {temb_flipped.shape}")
        if temb_flipped.shape == hf_time_proj.shape:
            is_close_proj = np.allclose(temb_flipped, hf_time_proj, atol=1e-5, rtol=1e-5)
            print(f"Time Projection (sin/cos) matches? {is_close_proj}")
            if not is_close_proj:
                diff_proj = np.abs(temb_flipped - hf_time_proj)
                print(f"Time Projection Max diff: {diff_proj.max()}")
                print(f"Time Projection Mean diff: {diff_proj.mean()}")
        else:
            print(f"Time Projection shapes mismatch: MaxText {temb_flipped.shape}, HF {hf_time_proj.shape}")

        t_emb = model.timestep_embedder(t_batch)
        y_emb = model.label_embedder(y_batch)
        
        print(f"hf_t_emb shape: {hf_t_emb.shape}")
        print(f"t_emb shape: {t_emb.shape}")
        if t_emb.shape == hf_t_emb.shape:
            is_close_t = np.allclose(t_emb, hf_t_emb, atol=1e-5, rtol=1e-5)
            print(f"Timestep Embedding matches? {is_close_t}")
            if not is_close_t:
                diff_t = np.abs(t_emb - hf_t_emb)
                print(f"Timestep Embedding Max diff: {diff_t.max()}")
                print(f"Timestep Embedding Mean diff: {diff_t.mean()}")
        else:
            print(f"Timestep Embedding shapes mismatch: MaxText {t_emb.shape}, HF {hf_t_emb.shape}")

        print(f"hf_y_emb shape: {hf_y_emb.shape}")
        print(f"y_emb shape: {y_emb.shape}")
        if y_emb.shape == hf_y_emb.shape:
            is_close_y = np.allclose(y_emb, hf_y_emb, atol=1e-5, rtol=1e-5)
            print(f"Label Embedding matches? {is_close_y}")
            if not is_close_y:
                diff_y = np.abs(y_emb - hf_y_emb)
                print(f"Label Embedding Max diff: {diff_y.max()}")
                print(f"Label Embedding Mean diff: {diff_y.mean()}")
        else:
            print(f"Label Embedding shapes mismatch: MaxText {y_emb.shape}, HF {hf_y_emb.shape}")

        c = jax.nn.silu(t_emb + y_emb)
        
        # 3. Block 0 Manual Steps (AdaLN + Norm1)
        block = model.blocks[0]
        
        # Compare c with hf_modulation_in
        # hf_modulation_in shape: (B*2, hidden_size) or similar
        print(f"hf_modulation_in shape: {hf_modulation_in.shape}")
        print(f"c shape: {c.shape}")
        
        # Reshape or transpose if needed
        # In HF, it might be (4, 1152)
        # In MaxText, c might be (4, 1152)
        
        if c.shape == hf_modulation_in.shape:
            is_close_c = np.allclose(c, hf_modulation_in, atol=1e-5, rtol=1e-5)
            print(f"Modulation Input (c) matches? {is_close_c}")
            if not is_close_c:
                diff_c = np.abs(c - hf_modulation_in)
                print(f"Modulation In Max diff: {diff_c.max()}")
                print(f"Modulation In Mean diff: {diff_c.mean()}")
        else:
            print(f"Modulation Input shapes mismatch: MaxText {c.shape}, HF {hf_modulation_in.shape}")

        modulation = block.adaLN_modulation(c)
        
        print(f"hf_modulation_out shape: {hf_modulation_out.shape}")
        print(f"modulation shape: {modulation.shape}")
        
        if modulation.shape == hf_modulation_out.shape:
            is_close_mod = np.allclose(modulation, hf_modulation_out, atol=1e-5, rtol=1e-5)
            print(f"Modulation Output matches? {is_close_mod}")
            if not is_close_mod:
                diff_mod = np.abs(modulation - hf_modulation_out)
                print(f"Modulation Out Max diff: {diff_mod.max()}")
                print(f"Modulation Out Mean diff: {diff_mod.mean()}")
        else:
            print(f"Modulation Output shapes mismatch: MaxText {modulation.shape}, HF {hf_modulation_out.shape}")

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
        
        shift_msa = shift_msa[:, jnp.newaxis, :]
        scale_msa = scale_msa[:, jnp.newaxis, :]
        
        normed_x = block.norm1(x)
        modulated_x = normed_x * (1 + scale_msa) + shift_msa
        
        print(f"MaxText Modulated input shape: {modulated_x.shape}")
        
        # Compare with HF Attn1 in
        if modulated_x.shape == hf_attn1_in.shape:
            is_close_in = np.allclose(modulated_x, hf_attn1_in, atol=1e-5, rtol=1e-5)
            print(f"Attention Inputs match? {is_close_in}")
            if not is_close_in:
                diff_in = np.abs(modulated_x - hf_attn1_in)
                print(f"Attn In Max diff: {diff_in.max()}")
                print(f"Attn In Mean diff: {diff_in.mean()}")
        else:
            print(f"Attention Input shapes mismatch: MaxText {modulated_x.shape}, HF {hf_attn1_in.shape}")
            
        # 4. Attention
        attn_out, _ = block.attn(
            inputs_q=modulated_x,
            inputs_kv=modulated_x,
            deterministic=False,
        )
        
        print(f"MaxText Attention output shape: {attn_out.shape}")
        
        # Compare with HF Attn1 out
        if attn_out.shape == hf_attn1_out.shape:
            is_close_out = np.allclose(attn_out, hf_attn1_out, atol=1e-5, rtol=1e-5)
            print(f"Attention Outputs match? {is_close_out}")
            if not is_close_out:
                diff_out = np.abs(attn_out - hf_attn1_out)
                print(f"Attn Out Max diff: {diff_out.max()}")
                print(f"Attn Out Mean diff: {diff_out.mean()}")
        else:
            print(f"Attention Output shapes mismatch: MaxText {attn_out.shape}, HF {hf_attn1_out.shape}")

    except FileNotFoundError as e:
        print(f"Could not load attention debug files: {e}")

    # hf_noise_pred has shape (2, 8, 32, 32) -> Wait, it is 4D!
    # MaxText output is (2, 256, 32) -> Wait, C_out might be different!
    # Let's check config.out_channels_for_vit
    c_out = getattr(config, 'out_channels_for_vit', C)
    print(f"Expected C_out: {c_out}")
    
    # MaxText output shape is (B, seq_len, p * p * c_out)
    # If c_out is 8, then patch_dim is 2 * 2 * 8 = 32.
    
    # To compare, we should unpatchify MaxText output or patchify HF output.
    # Let's patchify HF noise pred to match MaxText output.
    # hf_noise_pred shape: (4, 8, 32, 32) -> (B*2, C_out, H, W)
    
    hf_noise_pred_reshaped = hf_noise_pred.reshape(B * 2, c_out, side, p, side, p)
    
    # Transpose to group (side_h, side_w) and (p_h, p_w, C_out)
    # axes: B(0), C_out(1), side_h(2), p_h(3), side_w(4), p_w(5)
    # target: B(0), side_h(2), side_w(4), p_h(3), p_w(5), C_out(1)
    hf_noise_pred_transposed = np.transpose(hf_noise_pred_reshaped, (0, 2, 4, 3, 5, 1))
    hf_noise_pred_flattened = hf_noise_pred_transposed.reshape(B * 2, side * side, p * p * c_out)



    
    print(f"HF Noise pred flattened shape: {hf_noise_pred_flattened.shape}")
    
    # Compare
    is_close = np.allclose(noise_pred_maxtext, hf_noise_pred_flattened, atol=1e-5, rtol=1e-5)
    print(f"Outputs match? {is_close}")
    
    if not is_close:
        diff = np.abs(noise_pred_maxtext - hf_noise_pred_flattened)
        print(f"Max diff: {diff.max()}")
        print(f"Mean diff: {diff.mean()}")
        
        # We can also compare in 4D if we unpatchify MaxText
        # But this is enough to verify if the core transformer is doing same thing.


if __name__ == "__main__":
    app.run(main)
