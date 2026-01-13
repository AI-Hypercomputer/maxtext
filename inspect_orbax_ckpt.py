import sys
import jax
import orbax.checkpoint

def inspect(ckpt_path):
    print(f"Inspecting {ckpt_path}...")
    # Use standard PyTreeCheckpointHandler which handles OCDBT
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    try:
        # Restore without target; Orbax should infer structure from metadata (OCDBT)
        restored = checkpointer.restore(ckpt_path)
        
        # Flatten and print keys
        flat, _ = jax.tree_util.tree_flatten_with_path(restored)
        print(f"Successfully restored. Found {len(flat)} leaves.")
        print("First 50 keys:")
        for i, (path, val) in enumerate(flat):
            if i >= 50: break
            key_str = "/".join([str(p.key) for p in path])
            # Check if it's a real array or abstract
            val_type = type(val)
            print(f"{key_str} -> {val_type}")
            
    except Exception as e:
        print(f"Failed to restore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_orbax_ckpt.py <path_to_checkpoint>")
        sys.exit(1)
    inspect(sys.argv[1])
