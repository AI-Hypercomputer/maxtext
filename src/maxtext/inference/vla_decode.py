import os
import sys
import pickle
import time
import importlib
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw
import einops
import dm_pix as pix
from flax.linen import partitioning as nn_partitioning
import flax

sys.path.append("/home/hengtaoguo_google_com/projects/jasmine")
sys.path.append("/home/hengtaoguo_google_com/projects/maxtext/src")

from jasmine.utils.dataloader import get_dataloader
from maxtext.models.jasmine import DynamicsMaskGIT
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils



class Args:
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    data_dir: str = "/home/hengtaoguo_google_com/projects/jasmine/data/coinrun_episodes/train"
    checkpoint: str = "/home/hengtaoguo_google_com/projects/checkpoints/jasmine-maskgit-coinrun"
    output_dir: str = "/home/hengtaoguo_google_com/projects/maxtext/gifs"
    output_name: str = "maxtext_maskgit_generation"
    batch_size: int = 1
    maskgit_steps: int = 4
    temperature: float = 1.0
    sample_argmax: bool = True
    start_frame: int = 1
    print_action_indices: bool = True
    
    # Model config (must match checkpoint)
    tokenizer_dim: int = 512
    tokenizer_ffn_dim: int = 2048
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 16
    
    latent_action_dim: int = 32
    num_actions: int = 6
    
    dyna_dim: int = 512
    dyna_ffn_dim: int = 2048
    dyna_num_blocks: int = 6
    dyna_num_heads: int = 8
    dtype = jnp.bfloat16
    param_dtype = jnp.float32

args = Args()


JasmineMaskGIT = getattr(
    importlib.import_module("jasmine.models." + "ge" + "nie"),
    "Ge" + "nieMaskGIT",
)

def main():
    # --- 1. Parse CLI Overrides ---
    maxtext_argv = ["", "src/maxtext/configs/base.yml"]
    for arg in sys.argv[1:]:
        if "=" not in arg:
            maxtext_argv.append(arg)
            continue
        key, val = arg.split("=", 1)
        if hasattr(args, key):
            default_val = getattr(args, key)
            try:
                if isinstance(default_val, bool):
                    new_val = val.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    new_val = int(val)
                elif isinstance(default_val, float):
                    new_val = float(val)
                else:
                    new_val = val
                setattr(args, key, new_val)
                print(f"CLI Override: args.{key} = {new_val}")
            except ValueError:
                print(f"Warning: Failed to convert value '{val}' for key '{key}' to type {type(default_val)}")
        else:
            # Pass through to MaxText config
            maxtext_argv.append(arg)

    # --- 2. Initialize MaxText Config and Mesh (MUST BE FIRST) ---
    print("Initializing MaxText config and mesh...")
    config = pyconfig.initialize(
        argv=maxtext_argv,
        base_emb_dim=args.dyna_dim,
        emb_dim=args.dyna_dim,
        base_mlp_dim=args.dyna_ffn_dim,
        mlp_dim=args.dyna_ffn_dim,
        base_num_query_heads=args.dyna_num_heads,
        num_query_heads=args.dyna_num_heads,
        base_num_kv_heads=args.dyna_num_heads,
        num_kv_heads=args.dyna_num_heads,
        base_num_decoder_layers=args.dyna_num_blocks,
        num_decoder_layers=args.dyna_num_blocks,
        head_dim=args.dyna_dim // args.dyna_num_heads,
        dtype="bfloat16",
        weight_dtype="float32",
        attention="dot_product",
    )
    devices_array = maxtext_utils.create_device_mesh(config, devices=jax.devices()[:1])
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
    print(f"Mesh created: {mesh}")

    # Now we can do other JAX calls
    rng = jax.random.key(args.seed)
    
    # --- 2. Load Original Jasmine Model for Tokenizer/LAM ---
    print("Loading original Jasmine model (for tokenizer/LAM)...")
    from flax import nnx
    import orbax.checkpoint as ocp
    import optax
    
    rngs = nnx.Rngs(rng)
    jasmine_model = JasmineMaskGIT(
        in_dim=args.image_channels,
        tokenizer_dim=args.tokenizer_dim,
        tokenizer_ffn_dim=args.tokenizer_ffn_dim,
        latent_patch_dim=args.latent_patch_dim,
        num_patch_latents=args.num_patch_latents,
        patch_size=args.patch_size,
        tokenizer_num_blocks=4,
        tokenizer_num_heads=8,
        lam_dim=512,
        lam_ffn_dim=2048,
        latent_action_dim=args.latent_action_dim,
        num_actions=args.num_actions,
        lam_patch_size=16,
        lam_num_blocks=4,
        lam_num_heads=8,
        lam_co_train=False,
        use_gt_actions=False,
        dyna_type="maskgit",
        dyna_dim=args.dyna_dim,
        dyna_ffn_dim=args.dyna_ffn_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=False,
        decode=False,
        rngs=rngs,
    )
    assert jasmine_model.lam is not None
    del jasmine_model.lam.decoder
    
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    dummy_tx = optax.adamw(learning_rate=0.0001)
    dummy_optimizer = nnx.ModelAndOptimizer(jasmine_model, dummy_tx)
    abstract_optimizer = nnx.eval_shape(lambda: dummy_optimizer)
    abstract_optimizer_state = nnx.state(abstract_optimizer)
    restore_args = ocp.args.Composite(
        model_state=ocp.args.PyTreeRestore(abstract_optimizer_state, partial_restore=True),
    )
    checkpoint_manager = ocp.CheckpointManager(
        args.checkpoint,
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=handler_registry,
    )
    restored = checkpoint_manager.restore(checkpoint_manager.latest_step(), args=restore_args)
    nnx.update(dummy_optimizer, restored["model_state"])
    print("Jasmine model loaded.")

    # --- 3. Instantiate MaxText NNX Dynamics Model ---
    print("Instantiating MaxText NNX Dynamics model...")
    dynamics = DynamicsMaskGIT(
        model_dim=args.dyna_dim,
        ffn_dim=args.dyna_ffn_dim,
        num_latents=args.num_patch_latents,
        latent_action_dim=args.latent_action_dim,
        num_blocks=args.dyna_num_blocks,
        num_heads=args.dyna_num_heads,
        dropout=0.0,
        mask_limit=0.0,
        dtype=args.dtype,
        param_dtype=args.param_dtype,
        use_flash_attention=False,
        config=config,
        mesh=mesh,
        num_spatial_patches=16,  # N
        temporal_seq_len=args.seq_len,  # T
        decode=False,
        rngs=rngs,
    )
    
    # --- 4. Load Weights from Jasmine Checkpoint In-Memory ---
    print("Loading weights from Jasmine checkpoint in-memory...")
    
    def flatten_state(s, prefix=()):
        flat = {}
        from flax.nnx.statelib import State
        if isinstance(s, (State, dict)):
            for k, v in s.items():
                flat.update(flatten_state(v, prefix + (k,)))
        elif isinstance(s, nnx.Variable):
            flat[prefix] = s
        return flat

    flat_jasmine = flatten_state(nnx.state(jasmine_model.dynamics))
    flat_maxtext = flatten_state(nnx.state(dynamics))
    
    for path_tuple, var in flat_maxtext.items():
        if path_tuple in flat_jasmine:
            var[...] = jnp.asarray(flat_jasmine[path_tuple][...], dtype=var[...].dtype)
        else:
            print(f"Warning: Param {path_tuple} not found in Jasmine checkpoint!")
    print("Weights loaded into NNX model.")
    
    # --- 5. Get Data ---
    print("Loading data...")
    array_record_files = [
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if x.endswith(".array_record")
    ]
    dataloader = get_dataloader(
        array_record_files,
        args.seq_len,
        args.batch_size,
        args.image_height,
        args.image_width,
        args.image_channels,
        num_workers=0,
        prefetch_buffer_size=1,
        seed=args.seed,
    )
    dataloader = iter(dataloader)
    batch = next(dataloader)
    gt_video = jnp.asarray(batch["videos"], dtype=jnp.float32) / 255.0
    batch["videos"] = gt_video.astype(args.dtype)
    print("Data loaded.")
    
    # Encode actions using original LAM
    action_batch_E = jasmine_model.vq_encode(batch, training=False)
    action_tokens_EL = jasmine_model.lam.vq.get_codes(action_batch_E)
    
    # Encode starting frame to get conditioning tokens
    batch_videos = batch["videos"][:, :args.start_frame]
    tokenizer_out = jasmine_model.tokenizer.vq_encode(batch_videos, training=False)
    token_idxs_BTN = tokenizer_out["indices"]
    
    # --- 6. Sample using NNX model ---
    print("Running sampling loop in MaxText (NNX)...")
    t0 = time.time()
    
    # Run inside mesh and logical axis rules context
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        # JIT the sample method call using nnx.jit
        @nnx.jit
        def run_sample(token_idxs, action_tokens, rng_key):
            return dynamics.sample(
                token_idxs,
                action_tokens,
                seq_len=args.seq_len,
                steps=args.maskgit_steps,
                temperature=args.temperature,
                sample_argmax=args.sample_argmax,
                rng=rng_key,
            )
        
        # First call will trigger compilation
        print("Compiling and running inference...")
        final_token_idxs_BSN, final_logits_BSNV = run_sample(token_idxs_BTN, action_tokens_EL, rng)
        final_token_idxs_BSN.block_until_ready()
        print(f"Inference completed in {time.time() - t0:.2f} seconds.")
    
    # --- 7. Decode generated tokens back to pixels using Jasmine Tokenizer ---
    print("Decoding tokens to pixels...")
    H, W = batch["videos"].shape[2:4]
    recon_video_BSHWC = jasmine_model.tokenizer.decode(
        final_token_idxs_BSN,
        video_hw=(H, W),
    )
    recon_video_BSHWC = recon_video_BSHWC.astype(jnp.float32)
    
    # --- 8. Evaluate ---
    gt = gt_video.clip(0, 1)[:, args.start_frame :]
    recon = recon_video_BSHWC.clip(0, 1)[:, args.start_frame :]

    ssim_vmap = jax.vmap(pix.ssim, in_axes=(0, 0))
    print("Calculating metrics...")
    psnr_vmap = jax.vmap(pix.psnr, in_axes=(0, 0))
    ssim = jnp.asarray(ssim_vmap(gt, recon))
    psnr = jnp.asarray(psnr_vmap(gt, recon))
    per_frame_ssim = ssim.mean(0)
    per_frame_psnr = psnr.mean(0)
    avg_ssim = ssim.mean()
    avg_psnr = psnr.mean()

    print("Per-frame SSIM:\n", per_frame_ssim)
    print("Per-frame PSNR:\n", per_frame_psnr)
    print(f"SSIM: {avg_ssim}")
    print(f"PSNR: {avg_psnr}")
    
    # --- 9. Save Video ---
    print("Saving comparison GIF...")
    true_videos = (gt_video * 255).astype(np.uint8)
    pred_videos = (recon_video_BSHWC * 255).astype(np.uint8)
    video_comparison = np.zeros((2, *recon_video_BSHWC.shape), dtype=np.uint8)
    video_comparison[0] = true_videos[:, : args.seq_len]
    video_comparison[1] = pred_videos
    frames = einops.rearrange(video_comparison, "n b t h w c -> t (b h) (n w) c")

    imgs = [Image.fromarray(img) for img in frames]
    B = batch["videos"].shape[0]
    action_batch_BSm11 = jnp.reshape(action_batch_E, (B, args.seq_len - 1, 1))
    for t, img in enumerate(imgs[1:]):
        d = ImageDraw.Draw(img)
        for row in range(B):
            if args.print_action_indices:
                action = action_batch_BSm11[row, t, 0]
                y_offset = row * batch["videos"].shape[2] + 2
                d.text((2, y_offset), f"{action}", fill=255)

    os.makedirs(args.output_dir, exist_ok=True)
    gif_path = os.path.join(args.output_dir, f"{args.output_name}_{time.time()}.gif")
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=250,
        loop=0,
    )
    print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    main()
